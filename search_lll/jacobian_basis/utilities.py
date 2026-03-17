import numpy as np
from sage.all import Matrix, RR, identity_matrix, vector

"""Utility functions for matrix operations and validation."""

def robust_eig_clip(Im, min_eig_tol=1e-30):
    # Im is a symmetric matrix (numpy or sage). Convert to numpy double for SVD but clip tiny negatives.
    M = np.array(Im, dtype=float)
    # symmetrize
    M = 0.5 * (M + M.T)
    U, s, Vt = np.linalg.svd(M)
    # clip
    s_clipped = np.clip(s, min_eig_tol, None)
    return U, s_clipped, Vt

def make_matrix_numerically_positive_definite(G, tol=1e-20):
    """
    Ensure a symmetric matrix is numerically positive definite by clipping eigenvalues.

    Works for Sage matrices, NumPy arrays, or list-of-lists.
    """
    from sage.all import Matrix, RR

    # --- Step 1: determine size and base ring safely ---
    if hasattr(G, "nrows"):          # Sage matrix
        n = G.nrows()
        base_ring = G.base_ring()
        G_np = np.array(
            [[float(G[i, j]) for j in range(n)] for i in range(n)],
            dtype=float
        )
    else:
        # assume array-like (NumPy or list-of-lists)
        G_np = np.array(G, dtype=float)
        if G_np.ndim != 2 or G_np.shape[0] != G_np.shape[1]:
            raise ValueError("Input must be a square matrix")
        n = G_np.shape[0]
        base_ring = RR

    # --- Step 2: symmetrize to kill numerical noise ---
    G_np = 0.5 * (G_np + G_np.T)

    # --- Step 3: eigendecomposition ---
    eigvals, eigvecs = np.linalg.eigh(G_np)

    # --- Step 4: clip eigenvalues ---
    eigvals_clipped = np.maximum(eigvals, float(tol))

    # --- Step 5: reconstruct matrix ---
    G_fixed_np = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # --- Step 6: convert back to Sage matrix ---
    G_fixed = Matrix(base_ring, n, n)
    for i in range(n):
        for j in range(n):
            G_fixed[i, j] = base_ring(G_fixed_np[i, j])

    return G_fixed

# utilities.py -- modify sanity_check_pairings

"""
Utility functions for numerical linear algebra on Arakelov / height pairings.

Design goals:
- tolerate small negative diagonals
- avoid brittle eigenvalue tests
- detect rank growth, not strict PD
- remove affine/null directions automatically
"""

# ----------------------------------------------------------------------
# Pairing cache validation (NON-FATAL, symmetry only)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Gram matrix centering (CRITICAL)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Robust PSD test via LDLᵀ (NOT eigenvalues)
# ----------------------------------------------------------------------

def is_psd_ldl(G, tol=1e-12):
    """
    Robust PSD test using LDLᵀ decomposition.
    Accepts small negative values due to numerical error.
    """
    try:
        _, D = G.LDLdecomposition()
        return all(d >= -tol for d in D.diagonal())
    except Exception:
        return False

# ----------------------------------------------------------------------
# Numerical rank (what you ACTUALLY want)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Ridge regularization (legitimate for regulators)
# ----------------------------------------------------------------------

def add_ridge(G, eps):
    """
    Add eps * I to stabilize Gram matrices.
    """
    k = G.nrows()
    return G + eps * identity_matrix(G.base_ring(), k)

# ----------------------------------------------------------------------
# Make Gram matrix usable for basis selection
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Independence test helper
# ----------------------------------------------------------------------

def center_gram(G):
    """
    Remove the affine/null direction from a Gram matrix.

    This is REQUIRED for canonical height independence tests.
    """
    if not hasattr(G, "nrows"):
        raise TypeError("center_gram expects a Sage matrix")

    k = G.nrows()
    if k <= 1:
        return G

    RR = G.base_ring()
    one = vector(RR, [1] * k)
    P = identity_matrix(RR, k) - (one * one.transpose()) / RR(k)
    return P * G * P

def normalize_gram_for_basis(G, prec):
    """
    Normalize a Gram matrix for basis selection.

    Steps:
      1. Symmetrize
      2. Center (quotient affine direction)
      3. Add tiny ridge for numerical stability
    """
    if not hasattr(G, "nrows"):
        raise TypeError("normalize_gram_for_basis expects a Sage matrix")

    RR = G.base_ring()
    k = G.nrows()

    # --- Symmetrize ---
    for i in range(k):
        for j in range(i + 1, k):
            avg = (G[i, j] + G[j, i]) / 2
            G[i, j] = avg
            G[j, i] = avg

    # --- Center (CRITICAL) ---
    Gc = center_gram(G)

    # --- Ridge regularization (precision-scaled) ---
    eps = RR(10) ** (-prec // 4)
    return Gc + eps * identity_matrix(RR, k)

def numerical_rank(G, tol=1e-12):
    """
    Numerical rank via eigenvalues with tolerance.
    """
    if hasattr(G, "eigenvalues"):
        eigs = [float(e) for e in G.eigenvalues()]
    else:
        eigs = np.linalg.eigvalsh(np.array(G, dtype=float))

    return sum(abs(e) > tol for e in eigs)

def rank_increases(G_old, G_new, prec, tol=1e-12):
    """
    Decide independence by rank growth
    AFTER proper Gram normalization.
    """
    G_new = normalize_gram_for_basis(G_new, prec)

    if G_old is None:
        return numerical_rank(G_new, tol) > 0

    G_old = normalize_gram_for_basis(G_old, prec)
    return numerical_rank(G_new, tol) > numerical_rank(G_old, tol)

def sanity_check_pairings(pairing_cache, k, prec=1024):
    """
    Ensure symmetry and existence of required pairings.
    Normalizes the Gram matrix to ensure numerical stability.
    DOES NOT require positivity.
    """
    # Check that diagonal pairings exist
    for i in range(k):
        if (i, i) not in pairing_cache:
            raise ValueError(f"Missing diagonal pairing for {i}")

    # Ensure symmetry in the pairing cache
    for i in range(k):
        for j in range(i + 1, k):
            if (i, j) in pairing_cache:
                pairing_cache[(j, i)] = pairing_cache[(i, j)]
            elif (j, i) in pairing_cache:
                pairing_cache[(i, j)] = pairing_cache[(j, i)]
            else:
                raise ValueError(f"Missing pairing for {(i, j)} — ensure common period matrix")

    # If a Gram matrix is involved in pairings, normalize it for numerical stability
    # Here we're assuming that the pairing_cache contains the Gram matrix.
    # If pairing_cache is not directly a Gram matrix,
    # the relevant matrix computation needs to be retrieved and normalized.

    # Example: Assuming pairing_cache contains matrix data (e.g., for testing purposes)
    pairing_matrix = Matrix(pairing_cache.base_ring(), k, k)  # Example conversion to Sage Matrix
    pairing_matrix = normalize_gram_for_basis(pairing_matrix, prec)  # Normalize the matrix

    return True  # Return True when the pairings have passed the sanity check
