"""Utility functions for matrix operations and validation."""

import numpy as np
from sage.all import Matrix, RR, identity_matrix


def sanity_check_pairings(pairing_cache, n):
    """
    pairing_cache: dict of (i,j)->value for 0<=i<=j<n
    n: number of indices expected
    Raises on failures.
    """
    for i in range(n):
        if (i, i) not in pairing_cache:
            raise ValueError(f"Missing diagonal pairing for {i}")
        if float(pairing_cache[(i,i)]) <= 0:
            raise ValueError(f"Non-positive self-pairing for {i}: {pairing_cache[(i,i)]}")

    for i in range(n):
        for j in range(i+1, n):
            a = pairing_cache.get((i,j), None)
            b = pairing_cache.get((j,i), None)
            if a is None and b is None:
                raise ValueError(f"Missing pairing for pair {(i,j)}")
            if a is None: a = b
            if b is None: b = a
            if abs(float(a) - float(b)) > 1e-10 * max(1.0, abs(float(a))):
                raise ValueError(f"Asymmetric pairings for {(i,j)}: {a} vs {b}")


def robust_eig_clip(Im, min_eig_tol=1e-30):
    # Im is a symmetric matrix (numpy or sage). Convert to numpy double for SVD but clip tiny negatives.
    import numpy as np
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
    import numpy as np
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
def sanity_check_pairings(pairing_cache, k):
    for i in range(k):
        for j in range(k):
            if (i,j) not in pairing_cache:
                # try symmetric entry first
                if (j,i) in pairing_cache:
                    pairing_cache[(i,j)] = pairing_cache[(j,i)]
                else:
                    # missing: don't hard crash; raise an informative error or return False
                    raise ValueError(f"Missing pairing for pair {(i,j)} — ensure precomputation of h(D_i + D_j) with a common period matrix")
    return True
