"""Core basis building functionality."""

import numpy as np
import math
from sage.all import (
    QQ, ZZ, RR, RealField, ComplexField,
    PolynomialRing, HyperellipticCurve, Matrix, vector
)
from multiprocessing import Pool, cpu_count

from .pairings import precompute_pairings_parallel, gram_logdet_and_cond
from .heights import arakelov_canonical_height
from .utilities import make_matrix_numerically_positive_definite, sanity_check_pairings
from .parallel import compute_height_worker, compute_pairing_worker
import warnings
from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, QQ)
from multiprocessing import cpu_count
import math
from search_lll.homology import *

# Functions: dedupe_basis, gram_logdet_and_cond (moved to pairings),
# select_independent_indices_from_gram, arakelov_build_basis_with_heights,
# is_independent_by_projection_log

def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Incremental basis builder: only compute pairings as needed.
    """
    from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, QQ)
    from multiprocessing import cpu_count
    import sys

    PM = get_period_matrix_auto_B(f_coeffs, prec=prec)
    n = len(all_divisors)
    heights = [None] * n

    if not all_divisors:
        return [], 0, None

    if n_jobs == -1:
        try:
            n_jobs = cpu_count()
        except Exception:
            n_jobs = 1
            raise

    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
        print(f"[arakelov] Parallelization: {n_jobs} workers")

    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    jac_elements = []
    for div in all_divisors:
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        div2 = J([u_poly, v_poly])
        jac_elements.append((div, div2))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    # Compute individual heights (parallel)
    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_args = [(i, jac_elements[i][0], f_coeffs, prec) for i in range(n)]
    height_cache = {}

    if n > 1 and n_jobs > 1:
        from multiprocessing import Pool
        with Pool(processes=n_jobs) as pool:
            results = pool.map(compute_height_worker, height_args)
        for i, h, error in results:
            if error:
                raise RuntimeError(f"Height computation failed for divisor {i}: {error}")
            height_cache[i] = h
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")
    else:
        for i in range(n):
            _, div = jac_elements[i]
            h = arakelov_canonical_height(div, f_coeffs, prec=prec)
            height_cache[i] = h
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")

    # Pairing cache (compute on-demand)
    pairing_cache = {}

    # Helper to get pairing (with lazy computation)
    def get_pairing_lazy(i, j):
        if i > j:
            i, j = j, i
        if (i, j) in pairing_cache:
            return pairing_cache[(i, j)]
        
        # Compute on-demand
        if i == j:
            val = height_cache[i]
        else:
            div_i = jac_elements[i][0]
            div_j = jac_elements[j][0]
            args = (i, j, div_i, div_j, f_coeffs, prec, height_cache[i], height_cache[j])
            _, val, error = compute_pairing_worker(args)
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
        
        #sanity_check_pairings(pairing_cache, min(len(all_divisors), 1))
        pairing_cache[(i, j)] = val
        return val

    # Incremental basis selection
    if debug:
        print("[arakelov] Selecting basis incrementally...")
    
    basis_indices = []
    
    for cand_idx in range(n):
        if len(basis_indices) >= 2 * C.genus():  # genus 2 -> rank <= 4
            break
            
        if len(basis_indices) == 0:
            # First element: just check if nonzero height
            h = height_cache[cand_idx]
            if abs(float(h)) > 1e-6:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (first basis element)")
            continue
        
        # Test independence via projection residual
        is_indep, info = is_independent_by_projection_log(
            basis_indices,
            cand_idx,
            get_pairing_lazy,
            prec,
            debug=debug
        )
        
        if is_indep:
            k = len(basis_indices)
            from sage.all import RealField, Matrix
            RR = RealField(max(128, int(prec//4)))
            G_test = Matrix(RR, k, k)
            for r in range(k):
                for c in range(r, k):
                    pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                    G_test[r, c] = RR(pv)
                    G_test[c, r] = G_test[r, c]
            
            det_val = G_test.determinant()
            if det_val > 0:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (basis now size {len(basis_indices)})")
            else:
                print(f"  Rejected divisor {cand_idx}: dependent")
        elif debug:
            print(f"  Rejected divisor {cand_idx}: dependent")

    # ADD THE SANITY CHECK HERE (after the loop, before building final Gram matrix):
    if len(basis_indices) > 0:
        # Only check pairings that were actually computed during basis selection
        # The cache will have diagonal entries for selected basis elements
        sanity_check_pairings(pairing_cache, min(len(basis_indices), 4))


    basis = [jac_elements[i][0] for i in basis_indices]
    basis, basis_indices = dedupe_basis(basis, basis_indices, debug=debug)

    final_rank = len(basis)
    H_final = None
    
    if final_rank > 0:
        RR = RealField(max(128, int(prec//4)))
        H_final = Matrix(RR, final_rank, final_rank)
        
        if debug:
            print(f"[arakelov] Building final {final_rank}×{final_rank} Gram matrix...")
        
        for r in range(final_rank):
            for c in range(r, final_rank):
                pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                H_final[r, c] = RR(pv)
                H_final[c, r] = H_final[r, c]

        eigs = H_final.eigenvalues()
        print("eigenvalues:", eigs)
        f_eigs = [float(i) for i in eigs]
        print("eigenvalues (float):", f_eigs)
        print("determinant:", float(H_final.determinant()))
        print("condition number (approx):", float(max(eigs)/min(eigs)))
        # regulator is det(G) (for basis of free part of rank k)
        print("regulator:", float(H_final.determinant()))

    if debug and final_rank > 0:
        try:
            gd = gram_logdet_and_cond(basis_indices, get_pairing_lazy)
            print(f"[arakelov] Final numeric rank: {gd['numeric_rank']}/{gd['n']}")
            print(f"[arakelov] log10|det|={gd['log10_abs_det']:.3g}, log10(cond)={gd['log10_cond']:.3g}")
            if H_final is not None:
                try:
                    print(f"[arakelov] Final determinant: {float(H_final.determinant()):.6g}")
                except Exception:
                    raise
        except Exception as E:
            if debug:
                print(f"[arakelov] Diagnostic failed: {E}")
            raise

    return basis, final_rank, H_final


def dedupe_basis(basis, basis_indices, debug=False):
    """
    Remove duplicated divisors (preserve order). Returns (basis_u, basis_indices_u).
    """
    seen = set()
    basis_u = []
    basis_indices_u = []
    for div, idx in zip(basis, basis_indices):
        if idx in seen:
            if debug:
                print(f"[dedupe] removing duplicate basis index {idx}")
            continue
        seen.add(idx)
        basis_u.append(div)
        basis_indices_u.append(idx)
    return basis_u, basis_indices_u

def select_independent_indices_from_gram(
    G,
    prec_bits=2048,
    safety_digits=10,
    rel_sv_tol=1e-12,
    pivot_tol_factor=1e-9,
    debug=False,
):
    """
    Given a symmetric float Gram matrix G (n x n), return a deterministic list
    of indices that form a numerically independent set.

    Parameters
    ----------
    G : np.ndarray, shape (n,n), symmetric floats
    prec_bits : int, bit-precision used upstream (only for diagnostic scaling)
    safety_digits : int, digits of safety when building eigenvalue threshold
    rel_sv_tol : float, fallback relative threshold for eigenvalues (smax * rel_sv_tol)
    pivot_tol_factor : float, factor to multiply sqrt(min_positive_eig) to set pivot stop
    debug : bool

    Returns
    -------
    selected_indices : list of ints (length <= numeric_rank)
    info : dict with keys:
        'eigvals' (descending), 'numeric_rank', 'log10_abs_det', 'log10_cond'
    """
    # symmetrize (be safe)
    G = 0.5 * (G + G.T)
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        raise RuntimeError(f"Gram not PD after clipping: min_eig={min_ev:.3e}")
    n = G.shape[0]
    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(G)  # ascending eigenvalues
    eigvals = np.array(eigvals, dtype=float)
    # flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    smax = eigvals[0] if eigvals.size else 0.0
    # build eigenvalue threshold robustly:
    # convert bits -> decimal digits estimate, but cap to avoid under/overflow
    dec_digits = int(prec_bits * 0.30103) if prec_bits > 0 else 50
    dec_digits_cap = min(max(dec_digits, 0), 50)  # cap to [0,50] to avoid absurd exponents
    # threshold by safety_digits (but cap using rel_sv_tol)
    ev_thresh = max(smax * (10.0 ** (-(max(safety_digits, dec_digits_cap)))), smax * rel_sv_tol, 1e-300)

    # positive eigenvalues indices
    pos_mask = eigvals > ev_thresh
    pos_indices = np.nonzero(pos_mask)[0]
    r = len(pos_indices)
    if debug:
        print(f"[select_from_gram] smax={smax:.3g}, ev_thresh={ev_thresh:.3g}, num_pos={r}")

    if r == 0:
        return [], {
            "eigvals": eigvals.tolist(),
            "numeric_rank": 0,
            "log10_abs_det": None,
            "log10_cond": None,
        }

    # Build embedding E (n x r) such that G ≈ E E^T
    Spos = eigvals[pos_indices]                     # length r (descending)
    Upos = eigvecs[:, pos_indices]                  # n x r
    sqrtS = np.sqrt(np.maximum(Spos, 0.0))
    # E[i,:] is embedding vector for candidate i
    E = Upos * sqrtS[np.newaxis, :]                 # shape (n, r)

    # Determine pivot stop tolerance from smallest retained eigenvalue
    min_pos_eig = Spos[-1]
    pivot_tol = math.sqrt(max(min_pos_eig, 0.0)) * pivot_tol_factor
    # also ensure pivot_tol not ridiculously tiny:
    pivot_tol = max(pivot_tol, smax * 1e-16)

    # Deterministic pivoting: iteratively pick row with largest residual norm,
    # orthogonalize rows against chosen normalized vector, stop when residuals small
    rows = E.copy()  # will be modified (deflation)
    norms = np.linalg.norm(rows, axis=1)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    while True:
        # choose argmax among not selected
        cand = int(np.argmax(norms + (selected_mask * -1e300)))  # ensures selected masked
        maxnorm = norms[cand]
        if debug:
            print(f"[pivot] pick candidate {cand} maxnorm={maxnorm:.6g} selected={len(selected)}")
        if maxnorm <= pivot_tol:
            break
        # add candidate
        selected.append(cand)
        selected_mask[cand] = True
        # normalize
        v = rows[cand].copy()
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            # can't orthonormalize further
            break
        v = v / vnorm
        # deflate all rows by projection onto v
        proj = rows @ v   # projection coefficients (n,)
        rows = rows - np.outer(proj, v)
        # recompute norms only for not yet selected
        norms = np.linalg.norm(rows, axis=1)
        # keep selecting until we've chosen r rows
        if len(selected) >= r:
            break

    # if we picked fewer than r (rare), we can accept them as basis; numeric_rank = len(selected)
    numeric_rank = len(selected)

    # diagdet/log det from singular values (embedding singulars are sqrt of eigvals)
    # For Gram matrix, singular values = eigvals (nonnegative). So log10|det| = sum log10(eigvals_pos)
    log10_abs_det = sum(math.log10(max(x, 1e-300)) for x in Spos)
    cond = Spos[0] / (Spos[-1] if Spos[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)

    info = {
        "eigvals": eigvals.tolist(),
        "numeric_rank": numeric_rank,
        "log10_abs_det": log10_abs_det,
        "log10_cond": log10_cond,
    }
    return selected, info

def is_independent_by_projection_log(
    basis_indices,
    candidate_index,
    get_pairing,
    prec,
    debug=False,
):
    """Use high-precision Sage arithmetic throughout"""
    from sage.all import RealField, Matrix, vector
    import math
    
    k = len(basis_indices)
    if k == 0:
        return True, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
    
    # High-precision reals
    RR = RealField(max(128, int(prec // 4)))
    
    # Build Gram in Sage
    G = Matrix(RR, k, k)
    c = vector(RR, k)
    
    for r in range(k):
        for s in range(r, k):
            v = get_pairing(basis_indices[r], basis_indices[s])
            G[r, s] = RR(v)
            G[s, r] = G[r, s]
        c[r] = RR(get_pairing(basis_indices[r], candidate_index))
    
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        raise RuntimeError(f"Gram not PD after clipping: min_eig={min_ev:.3e}")

    vv = RR(get_pairing(candidate_index, candidate_index))
    
    # Special case: if k=1, just check if candidate has different height
    if k == 1:
        g00 = G[0, 0]
        c0 = c[0]
        
        if abs(g00) < 1e-10:
            # First basis element has ~zero height, reject candidate
            if debug:
                print(f"[proj-log] First basis element has near-zero height")
            return False, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
        
        # Projection: proj_sq = c0^2 / g00
        proj_sq = (c0 * c0) / g00
        res_sq = vv - proj_sq
        
        res_sq_f = float(res_sq)
        if res_sq_f <= 0:
            if debug:
                print(f"[proj-log] k=1: negative residual res_sq={res_sq_f}")
            return False, {"res_sq": res_sq_f, "log10_res": float("-inf"), "log10_tol": float("-inf"), "min_sv": None}
        
        dec_digits = int(prec * 0.30103)
        safety_digits = 12
        diag_max = max(abs(float(g00)), abs(float(vv)), 1.0)
        log10_tol = math.log10(diag_max) - max(0, (dec_digits - safety_digits))
        log10_res = math.log10(res_sq_f) if res_sq_f > 0 else float("-inf")
        
        is_independent = (log10_res > log10_tol)
        
        info = {"res_sq": res_sq_f, "log10_res": log10_res, "log10_tol": log10_tol, "min_sv": None}
        if debug:
            print(f"[proj-log] k=1 cand={candidate_index} res={res_sq_f:.3g} log10_res={log10_res:.3g} log10_tol={log10_tol:.3g}")
        
        return is_independent, info
    
    # For k >= 2: use Cholesky (requires positive definite)
    try:
        L = G.cholesky()
        y = L.solve_left(c)
        proj_sq = y.dot_product(y)
    except Exception as e:
        if debug:
            print(f"[proj-log] Cholesky failed for candidate {candidate_index}: {e}")
        raise
        return False, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
    
    res_sq = vv - proj_sq
    
    res_sq_f = float(res_sq)
    vv_f = float(vv)
    
    if res_sq_f <= 0:
        diag_max = max(float(G[r,r]) for r in range(k)) if k > 0 else 1.0
        diag_max = max(diag_max, vv_f, 1.0)
        if abs(res_sq_f) <= 1e-12 * diag_max:
            res_sq_f = 0.0
        else:
            if debug:
                print(f"[proj-log] negative residual (proj > self): res_sq={res_sq_f}")
            return False, {"res_sq": res_sq_f, "log10_res": float("-inf"), "log10_tol": float("-inf"), "min_sv": None}
    
    dec_digits = int(prec * 0.30103)
    safety_digits = 12
    diag_max = max(max(abs(float(G[r,r])) for r in range(k)), vv_f, 1.0)
    log10_tol = math.log10(diag_max) - max(0, (dec_digits - safety_digits))
    log10_res = math.log10(res_sq_f) if res_sq_f > 0 else float("-inf")
    
    is_independent = (log10_res > log10_tol)
    
    info = {"res_sq": res_sq_f, "log10_res": log10_res, "log10_tol": log10_tol, "min_sv": None}
    if debug:
        print(f"[proj-log] cand={candidate_index} res={res_sq_f:.3g} log10_res={log10_res:.3g} log10_tol={log10_tol:.3g}")
    
    return is_independent, info


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Incremental basis builder: compute period matrix once, precompute heights,
    precompute pairings for a small prefix, and compute pairings on-demand using
    the same period matrix. Robust to occasional theta/finite-place failures:
    missing/failed computations are recorded conservatively as 0.0 with a warning.
    """
    # local imports from your package - assumed present
    # compute_height_worker / compute_pairing_worker are not used here;
    # we call arakelov_canonical_height directly to ensure a shared PM.

    if not all_divisors:
        return [], 0, None

    # determine number of workers (kept for logs only; we do serial heights to keep PM stable)
    if n_jobs == -1:
        try:
            n_jobs = cpu_count()
        except Exception:
            n_jobs = 1

    # 1) Compute period matrix once at requested precision
    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
        print(f"[arakelov] Parallelization: {n_jobs} workers (heights computed serially for PM consistency)")

    try:
        PM = get_period_matrix_auto_B(f_coeffs, prec=prec)
    except Exception as e:
        # If period matrix itself fails, there's nothing much to do: propagate but give context.
        raise RuntimeError(f"[arakelov] get_period_matrix_auto_B failed at prec={prec}: {e}")

    # 2) Build Jacobian elements (Sage objects) from raw mumford dicts
    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    jac_elements = []
    for div in all_divisors:
        # build Mumford polynomials (assumes genus 2; adapt if variable degree)
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        div_j = J([u_poly, v_poly])
        jac_elements.append((div, div_j))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    # 3) Compute individual canonical heights using the same PM (serial for stability).
    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_cache = {}
    for i in range(n):
        try:
            _, jacP = jac_elements[i]
            h = arakelov_canonical_height(jacP, f_coeffs, prec=prec, debug=debug, period_matrix=PM)
            # store as float for subsequent linear algebra usage
            height_cache[i] = float(h)
            if debug:
                print(f"  Divisor {i}: h = {height_cache[i]:.6g}")
        except Exception as e:
            warnings.warn(f"[arakelov] height computation failed for index {i}: {e}. Using 0.0 as conservative fallback.", RuntimeWarning)
            height_cache[i] = 0.0

    # 4) Pairing cache & precompute a small prefix of pairings to avoid missing entries in sanity checks.
    pairing_cache = {}
    # Store diagonal entries consistently with existing code (they used height_cache[i] for diag)
    for i in range(n):
        pairing_cache[(i, i)] = height_cache[i]

    # Decide how many prefix indices to precompute pairings for.
    # This should cover the typical small basis sanity checks; make it conservative.
    try:
        genus = C.genus()
    except Exception:
        genus = 2
    prefix_precompute = min(n, max(8, 2 * genus + 4))  # e.g., for genus 2 => 8

    if debug:
        print(f"[arakelov] Precomputing pairings for first {prefix_precompute} divisors (prefix).")

    for i in range(prefix_precompute):
        for j in range(i + 1, prefix_precompute):
            if (i, j) in pairing_cache:
                continue
            try:
                # compute height of sum using same PM, then pairing = 0.5*(h_ij - h_i - h_j)
                _, Ji = jac_elements[i]
                _, Jj = jac_elements[j]
                hij = arakelov_canonical_height(Ji + Jj, f_coeffs, prec=prec, debug=debug, period_matrix=PM)
                pairing_val = 0.5 * (float(hij) - float(height_cache[i]) - float(height_cache[j]))
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = float(pairing_val)
                if debug:
                    print(f"  Pair ({i},{j}): {pairing_cache[(i,j)]:.6g}")
            except Exception as e:
                warnings.warn(f"[arakelov] pairing precompute failed for ({i},{j}): {e}. Using 0.0 fallback.", RuntimeWarning)
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = 0.0

    # Helper to compute or fetch pairing lazily, using the same PM so results are consistent.
    def get_pairing_lazy(i, j):
        # canonicalize order
        if i > j:
            i, j = j, i
        if (i, j) in pairing_cache:
            return pairing_cache[(i, j)]

        # if diagonal, return stored height (kept behavior consistent with original)
        if i == j:
            val = height_cache.get(i, 0.0)
            pairing_cache[(i, i)] = val
            return val

        # compute on-demand: compute h(D_i + D_j) with shared PM, then pairing = 0.5*(h_ij - hi - hj)
        try:
            _, Ji = jac_elements[i]
            _, Jj = jac_elements[j]
            hij = arakelov_canonical_height(Ji + Jj, f_coeffs, prec=prec, debug=debug, period_matrix=PM)
            val = 0.5 * (float(hij) - float(height_cache.get(i, 0.0)) - float(height_cache.get(j, 0.0)))
            pairing_cache[(i, j)] = pairing_cache[(j, i)] = float(val)
            return pairing_cache[(i, j)]
        except Exception as e:
            warnings.warn(f"[arakelov] on-demand pairing computation failed for ({i},{j}): {e}. Using 0.0 fallback.", RuntimeWarning)
            pairing_cache[(i, j)] = pairing_cache[(j, i)] = 0.0
            return 0.0

    # 5) Incremental basis selection (same algorithm as before, but using the shared pairing function)
    if debug:
        print("[arakelov] Selecting basis incrementally...")

    basis_indices = []
    max_basis_size = 2 * genus

    for cand_idx in range(n):
        # stop if we already have the expected maximal free-rank bound
        if len(basis_indices) >= max_basis_size:
            break

        # first element: accept non-zero height
        if len(basis_indices) == 0:
            h = height_cache.get(cand_idx, 0.0)
            if abs(float(h)) > 1e-8:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (first basis element)")
            else:
                if debug:
                    print(f"  Skipping divisor {cand_idx}: height ~ 0")
            continue

        # test independence using projection-based test
        is_indep, info = is_independent_by_projection_log(
            basis_indices,
            cand_idx,
            get_pairing_lazy,
            prec,
            debug=debug
        )

        if not is_indep:
            if debug:
                print(f"  Rejected divisor {cand_idx}: dependent (proj test)")
            continue

        # quick Gram-check using current basis_indices + candidate (use lower precision RealField)
        k = len(basis_indices)
        from sage.all import RealField as SFRealField, Matrix as SFMatrix
        RR = SFRealField(max(128, int(prec // 4)))
        G_test = SFMatrix(RR, k + 1, k + 1)

        # fill matrix for basis_indices U {cand_idx}
        indices_to_fill = basis_indices + [cand_idx]
        for r in range(k + 1):
            for c in range(r, k + 1):
                pv = get_pairing_lazy(indices_to_fill[r], indices_to_fill[c])
                G_test[r, c] = RR(pv)
                G_test[c, r] = G_test[r, c]

        try:
            det_val = float(G_test.determinant())
        except Exception:
            # if determinant fails numerically, conservatively reject candidate
            if debug:
                print(f"  Rejected divisor {cand_idx}: numeric determinant failed")
            continue

        if det_val > 0:
            basis_indices.append(cand_idx)
            if debug:
                print(f"  Added divisor {cand_idx} (basis now size {len(basis_indices)})")
        else:
            if debug:
                print(f"  Rejected divisor {cand_idx}: dependent (Gram det <= 0)")

    # 6) Ensure sanity_check_pairings has the entries it needs: compute any missing pairs among selected basis_indices
    if len(basis_indices) > 0:
        need_k = min(len(basis_indices), 4)
        # ensure pairings exist for pairs among the first need_k basis entries
        for ii in range(need_k):
            for jj in range(ii, need_k):
                i = basis_indices[ii]; j = basis_indices[jj]
                if (i, j) not in pairing_cache:
                    # compute and fill
                    _ = get_pairing_lazy(i, j)
        # run sanity check (will raise only if truly missing)
        sanity_check_pairings(pairing_cache, need_k)

    # 7) Deduplicate basis and build final Gram matrix
    basis = [jac_elements[i][0] for i in basis_indices]
    basis, basis_indices = dedupe_basis(basis, basis_indices, debug=debug)

    final_rank = len(basis)
    H_final = None

    if final_rank > 0:
        RR = RealField(max(128, int(prec // 4)))
        H_final = Matrix(RR, final_rank, final_rank)
        if debug:
            print(f"[arakelov] Building final {final_rank}×{final_rank} Gram matrix...")

        for r in range(final_rank):
            for c in range(r, final_rank):
                pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                H_final[r, c] = RR(pv)
                H_final[c, r] = H_final[r, c]

        try:
            eigs = H_final.eigenvalues()
            f_eigs = [float(e) for e in eigs]
            if debug:
                print("eigenvalues (float):", f_eigs)
                print("determinant:", float(H_final.determinant()))
                # avoid division by zero
                if min(f_eigs) > 0:
                    print("condition number (approx):", float(max(f_eigs) / min(f_eigs)))
                print("regulator:", float(H_final.determinant()))
        except Exception as e:
            warnings.warn(f"[arakelov] final Gram diagnostics failed: {e}", RuntimeWarning)

    # optional diagnostics
    if debug and final_rank > 0:
        try:
            gd = gram_logdet_and_cond(basis_indices, get_pairing_lazy)
            print(f"[arakelov] Final numeric rank: {gd['numeric_rank']}/{gd['n']}")
            print(f"[arakelov] log10|det|={gd['log10_abs_det']:.3g}, log10(cond)={gd['log10_cond']:.3g}")
            if H_final is not None:
                print(f"[arakelov] Final determinant: {float(H_final.determinant()):.6g}")
        except Exception as E:
            if debug:
                print(f"[arakelov] Diagnostic failed: {E}")

    return basis, final_rank, H_final
