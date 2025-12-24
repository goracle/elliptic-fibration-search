"""Neron-Tate pairing computations."""

import numpy as np
import math
from sage.all import (
    QQ, ZZ, RealField, ComplexField,
    PolynomialRing, HyperellipticCurve, Matrix, vector
)
from multiprocessing import Pool

from .heights import arakelov_canonical_height
from .parallel import compute_pairing_worker
from .utilities import *

# Functions: get_pairing, neron_tate_height_pairing,
# precompute_pairings_parallel, gram_logdet_and_cond


"""Neron-Tate pairing computations."""


# Functions: get_pairing, neron_tate_height_pairing,
# precompute_pairings_parallel, gram_logdet_and_cond

def get_pairing(i, j, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    if i > j:
        i, j = j, i

    if (i, j) not in pairing_cache:
        # Compute on-demand if not in cache
        pairing_cache = precompute_pairings_parallel([i, j], jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs)

    return pairing_cache[(i, j)]

def neron_tate_height_pairing(z1, z2, Im_tau, prec=300, normalization_factor=1.0):
    """
    Compute Néron-Tate height pairing between two Abel-Jacobi images.
    
    Formula: <z1, z2> = normalization_factor * Re(z1^† * Im(τ)^(-1) * z2)
    
    where z1^† means transpose + complex conjugate,
    z1, z2 are vectors in C^2 and Im(τ) is the imaginary part of 
    the period matrix.
    
    The normalization_factor accounts for different conventions:
    - Some references use no factor (1.0)
    - Some use π
    - Some use 2π  
    - Some use 1/(2π) or 1/π
    
    The canonical height h(div) relates to the self-pairing by:
    h(div) = <div, div> / 2  (in some conventions)
    or
    h(div) = <div, div>      (in other conventions)
    
    Returns: real number (QQ approximation)
    """
    import math
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Invert Im(τ)
    try:
        Im_tau_inv = Im_tau.inverse()
    except:
        # Singular - shouldn't happen if period matrix is correct
        raise
        return QQ(0)
    
    # Convert Im_tau_inv to complex field for proper arithmetic
    Im_tau_inv_CC = Matrix(CC, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau_inv_CC[i,j] = CC(Im_tau_inv[i,j])
    
    # Convert z1, z2 to column vectors in CC
    z1_vec = vector(CC, [CC(z1[0]), CC(z1[1])])
    z2_vec = vector(CC, [CC(z2[0]), CC(z2[1])])
    
    # Compute z1^† * Im_tau_inv * z2
    # = conj(z1)^T * Im_tau_inv * z2
    
    z1_conj = vector(CC, [z1_vec[0].conjugate(), z1_vec[1].conjugate()])
    
    # Matrix-vector multiply: Im_tau_inv * z2
    temp = Im_tau_inv_CC * z2_vec
    
    # Dot product: z1_conj · temp
    result = z1_conj[0] * temp[0] + z1_conj[1] * temp[1]
    
    # Take real part and apply normalization
    real_part = RR(result.real()) * RR(normalization_factor)
    
    # Convert to QQ
    return real_part.nearby_rational(max_error=RR(2)**(-prec+10))


def gram_logdet_and_cond(basis_indices, get_pairing):
    """
    Build float Gram matrix for basis_indices (using get_pairing -> float),
    return dict with:
        n, svals (list), log10_abs_det, log10_cond, numeric_rank_est
    """
    import numpy as np, math
    n = len(basis_indices)
    if n == 0:
        return {"n":0, "svals": [], "log10_abs_det": None, "log10_cond": None, "numeric_rank": 0}
    G = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            v = float(get_pairing(basis_indices[i], basis_indices[j]))
            G[i,j] = v; G[j,i] = v
    # enforce symmetry
    G = 0.5*(G + G.T)
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        # Note: We do NOT raise here. We let the SVD logic handle near-singular matrices
        # unless it's strictly required by the caller.
        pass
        
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    svals = [float(x) for x in S]
    # log10_abs_det = sum(log10(svals)) (sign positive for Gram of PD matrix)
    log10_abs_det = sum(math.log10(max(s, 1e-300)) for s in svals)
    cond = svals[0] / (svals[-1] if svals[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)
    # numeric rank: count s > smax * tol_rel
    tol_rel = 1e-12
    smax = svals[0]
    numeric_rank = sum(1 for s in svals if s > smax * tol_rel)
    return {"n": n, "svals": svals, "log10_abs_det": log10_abs_det, "log10_cond": log10_cond, "numeric_rank": numeric_rank}


from .utilities import normalize_gram_for_basis


def precompute_pairings_parallel(indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    """Precompute all pairings for given indices in parallel, ensuring numerical stability with Gram matrix normalization"""
    
    pairs_to_compute = []
    for r in range(len(indices)):
        for c in range(r, len(indices)):
            i, j = indices[r], indices[c]
            if i > j:
                i, j = j, i
            if (i, j) not in pairing_cache:
                div_i = jac_elements[i][0]
                div_j = jac_elements[j][0]
                pairs_to_compute.append((
                    i, j, div_i, div_j, f_coeffs, prec,
                    height_cache[i], height_cache[j]
                ))

    if not pairs_to_compute:
        return pairing_cache

    print(f"[arakelov] Computing {len(pairs_to_compute)} pairings...")
    
    if len(pairs_to_compute) > 2 and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = []
            # Use compute_pairing_worker (no underscore)
            for i, res in enumerate(pool.imap_unordered(compute_pairing_worker, pairs_to_compute)):
                results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(pairs_to_compute)}")
            
        # After computing the pairings, normalize the Gram matrix
        for (i, j), val, error in results:
            if error:
                # A failure in both P+Q and P-Q is a critical geometric failure
                raise RuntimeError(f"Pairing computation failed for ({i},{j}): {error}")
            pairing_cache[(i, j)] = val

    else:
        for args in pairs_to_compute:
            (i, j), val, error = compute_pairing_worker(args)
            if error:
                raise RuntimeError(f"Pairing computation failed for ({i},{j}): {error}")
            pairing_cache[(i, j)] = val

    # Now, normalize the Gram matrix in the pairing cache (if relevant)
    if isinstance(pairing_cache, dict):
        # Iterate over pairing_cache and normalize matrices
        for key, matrix in pairing_cache.items():
            if isinstance(matrix, Matrix):
                # Normalize the matrix using the desired precision
                normalized_matrix = normalize_gram_for_basis(matrix, prec)
                pairing_cache[key] = normalized_matrix

    return pairing_cache
