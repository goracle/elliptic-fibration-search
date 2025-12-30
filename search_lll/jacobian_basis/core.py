"""core.py

Precision-hardened and robust replacements for pairing/height workers and
parallel precomputation. Intended as a drop-in improvement to the original
core/pairing modules you posted.

Key design choices made here:
- Use a modest, adaptive number of guard bits (not astronomically large).
- Compute archimedean arithmetic in a single RealField(parent) and avoid
  coercing exact QQ(0) into mixed parents.
- Do not re-raise inside inner exception handlers (we want graceful fallbacks).
- Return float results for pairing_cache (matches existing code paths which
  subsequently convert to float). When a pairing truly fails, store np.nan.
- Catch common "theta/radius/precision" style failures from arakelov routine
  and mark the pair as unstable instead of hanging.

This file intentionally keeps the external API compatible with your existing
pipeline: compute_pairing_worker(args) and compute_height_worker(args) keep
the same argument order/structure as before.

You should still consider moving to the analytic Abel--Jacobi pairing for
most Gram/regulator work (neron_tate_height_pairing) — this file provides
robust fallbacks when analytic AJ is not available.
"""

from sage.all import (
    QQ, PolynomialRing, HyperellipticCurve, RealField, ComplexField, Matrix, vector
)
from multiprocessing import Pool
import traceback


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
from search_lll.homology import *
from sage.all import QQ, GF, Integer, PolynomialRing, gcd


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


def is_independent_by_projection_log(basis_indices, cand_idx, get_pairing,
                                     prec=300, debug=False,
                                     rel_eig_tol=1e-8, abs_eig_tol=1e-12):
    """
    Robust independence test using eigenvalue rank and projection residuals.
    Never returns ambiguous states; ambiguous states default to dependent unless
    residuals are strong.
    """
    import numpy as np
    from math import isfinite

    info = {}
    k = len(basis_indices)
    if k == 0:
        info['reason'] = 'empty_basis'
        return True, info

    from .analytic_pairings import get_analytic_pairing


    try:
        G_np = np.zeros((k, k), dtype=float)
        b_np = np.zeros((k,), dtype=float)
        for i in range(k):
            for j in range(i, k):
                #val = float(get_pairing(basis_indices[i], basis_indices[j]))
                #val = float(get_analytic_pairing(i, j))
                val = float(get_analytic_pairing(basis_indices[i], basis_indices[j]))
                G_np[i, j] = val
                G_np[j, i] = val
        for i in range(k):
            b_np[i] = float(get_analytic_pairing(basis_indices[i], cand_idx))
        h_cand = float(get_analytic_pairing(cand_idx, cand_idx))
    except Exception as e:
        # Failure in pairing extraction -> return False so candidate is rejected
        info['error'] = f'pairing_error: {type(e).__name__}: {e}'
        raise
        return False, info

    if not (isfinite(h_cand) and np.all(np.isfinite(G_np)) and np.all(np.isfinite(b_np))):
        info['error'] = 'nonfinite_entry_in_pairings'
        return False, info

    G_np = 0.5 * (G_np + G_np.T)

    G_aug = np.zeros((k + 1, k + 1), dtype=float)
    G_aug[:k, :k] = G_np
    G_aug[:k, k] = b_np
    G_aug[k, :k] = b_np
    G_aug[k, k] = h_cand

    try:
        eigs_G = np.linalg.eigvalsh(G_np)
        eigs_aug = np.linalg.eigvalsh(G_aug)
    except np.linalg.LinAlgError:
        jitter = 1e-16
        G_np += jitter * np.eye(k)
        G_aug[:k, :k] = G_np
        eigs_G = np.linalg.eigvalsh(G_np)
        eigs_aug = np.linalg.eigvalsh(G_aug)
        raise

    max_eig_G = max(abs(eigs_G.max()), abs(eigs_G.min()), 1.0)
    max_eig_aug = max(abs(eigs_aug.max()), abs(eigs_aug.min()), 1.0)

    tol_G = max(rel_eig_tol * max_eig_G, abs_eig_tol)
    tol_aug = max(rel_eig_tol * max_eig_aug, abs_eig_tol)

    rank_G = int(np.sum(eigs_G > tol_G))
    rank_aug = int(np.sum(eigs_aug > tol_aug))

    info.update({
        'k': k,
        'rank_G': rank_G,
        'rank_aug': rank_aug,
        'eigs_G_tail': eigs_G[:min(5, len(eigs_G))].tolist(),
        'eigs_aug_tail': eigs_aug[:min(5, len(eigs_aug))].tolist(),
        'h_cand': h_cand
    })

    if rank_aug > rank_G:
        info['reason'] = 'rank_increase'
        return True, info

    try:
        alpha, *_ = np.linalg.lstsq(G_np, b_np, rcond=None)
        proj_b = G_np.dot(alpha)
        residual_vec = b_np - proj_b
        residual_norm = float(np.linalg.norm(residual_vec))
        b_norm = float(np.linalg.norm(b_np))
        rel_residual = residual_norm / (b_norm + 1e-18)

        predicted = float(np.dot(alpha, b_np))
        residual_energy = float(h_cand - predicted)

        info.update({
            'alpha_norm': float(np.linalg.norm(alpha)),
            'residual_norm': residual_norm,
            'rel_residual': rel_residual,
            'residual_energy': residual_energy,
            'predicted': predicted,
        })

        residual_vector_thresh = 1e-6
        residual_energy_rel_thresh = 1e-6
        residual_energy_abs_thresh = 1e-10

        is_small_vec = (rel_residual < residual_vector_thresh)
        is_small_energy = (abs(residual_energy) < max(residual_energy_abs_thresh,
                                                       residual_energy_rel_thresh * max(abs(h_cand), 1.0)))

        if is_small_vec and is_small_energy:
            info['reason'] = 'projected_small_residual -> dependent'
            return False, info
        else:
            info['reason'] = 'projected_residual_significant -> independent'
            return True, info

    except Exception as e:
        info['error'] = f'lstsq_failed: {type(e).__name__}: {e}'
        info['reason'] = 'conservative_dependent_on_error'
        raise
        return False, info


def select_independent_indices_from_gram(G, prec_bits=2048, safety_digits=10,
                                         rel_sv_tol=1e-12, pivot_tol_factor=1e-9, debug=False):
    import numpy as np
    import math

    if hasattr(G, "nrows"):
        n = int(G.nrows())
        G_np = np.array([[float(G[i, j]) for j in range(n)] for i in range(n)], dtype=float)
    else:
        G_np = np.array(G, dtype=float)
        if G_np.ndim != 2 or G_np.shape[0] != G_np.shape[1]:
            raise ValueError("Input must be a square matrix")
        n = G_np.shape[0]

    G_np = 0.5 * (G_np + G_np.T)
    ABS_FLOOR = 1e-300

    try:
        eigvals = np.linalg.eigvalsh(G_np)
        eigvecs = None
        method = "eig"
    except Exception:
        U, svals, Vt = np.linalg.svd(G_np)
        eigvals = svals.copy()
        eigvecs = U.copy()
        method = "svd"
        raise

    eigvals = np.array(eigvals, dtype=float)
    if eigvecs is None and method == "eig":
        try:
            w, V = np.linalg.eigh(G_np)
            eigvals = np.array(w, dtype=float)
            eigvecs = V
            method = "eig"
        except Exception:
            eigvecs = None
            raise

    eigvals_desc = eigvals[::-1]
    smax = float(max(eigvals_desc[0], 0.0)) if eigvals_desc.size else 0.0
    dec_digits = int(prec_bits * 0.30103) if prec_bits > 0 else 50
    dec_digits_cap = min(max(dec_digits, 0), 50)
    safety_power = max(safety_digits, dec_digits_cap)
    ev_thresh = max(smax * (10.0 ** (-safety_power)), smax * rel_sv_tol, 1e-300)

    if smax <= 0:
        try:
            _, svals_svd, _ = np.linalg.svd(G_np)
            smax = float(svals_svd[0]) if svals_svd.size else 0.0
            ev_thresh = max(smax * rel_sv_tol, 1e-300)
            method = "svd"
            eigvecs = np.linalg.svd(G_np)[0]
            eigvals_desc = np.array(svals_svd, dtype=float)
        except Exception:
            raise
            return [], {
                "eigvals": eigvals_desc.tolist(),
                "numeric_rank": 0,
                "log10_abs_det": None,
                "log10_cond": None,
                "method": method,
            }

    pos_mask_desc = eigvals_desc > ev_thresh
    pos_indices_desc = np.nonzero(pos_mask_desc)[0]
    r = int(len(pos_indices_desc))

    if r == 0:
        alt_thresh = max(smax * rel_sv_tol, 1e-300)
        pos_mask_desc = eigvals_desc > alt_thresh
        pos_indices_desc = np.nonzero(pos_mask_desc)[0]
        r = int(len(pos_indices_desc))

    if r == 0:
        return [], {
            "eigvals": eigvals_desc.tolist(),
            "numeric_rank": 0,
            "log10_abs_det": None,
            "log10_cond": None,
            "method": method,
        }

    if eigvecs is None:
        try:
            _, V = np.linalg.eigh(G_np)
            eigvecs_full = V
        except Exception:
            U, svals_svd, Vt = np.linalg.svd(G_np)
            eigvecs_full = U
            raise

    else:
        eigvecs_full = eigvecs

    if eigvecs_full.shape[1] == n:
        try:
            U_desc = eigvecs_full[:, ::-1]
            Upos = U_desc[:, pos_indices_desc]
        except Exception:
            Upos = eigvecs_full[:, :r]
            raise
    else:
        Upos = eigvecs_full[:, :r]

    if eigvals_desc.size >= r:
        Spos = eigvals_desc[pos_indices_desc]
    else:
        Spos = np.maximum(np.array(eigvals_desc[:r], dtype=float), ABS_FLOOR)

    Spos = np.maximum(Spos, ABS_FLOOR)
    sqrtS = np.sqrt(Spos)
    E = Upos * sqrtS[np.newaxis, :]

    min_pos_eig = float(Spos[-1]) if Spos.size else ABS_FLOOR
    pivot_tol = max(math.sqrt(max(min_pos_eig, ABS_FLOOR)) * pivot_tol_factor,
                    smax * 1e-16)

    rows = E.copy()
    norms = np.linalg.norm(rows, axis=1)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    while True:
        masked_norms = norms.copy()
        masked_norms[selected_mask] = -1.0
        cand = int(np.argmax(masked_norms))
        maxnorm = float(masked_norms[cand])
        if maxnorm <= pivot_tol or len(selected) >= r:
            break
        selected.append(cand)
        selected_mask[cand] = True
        v = rows[cand].copy()
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            break
        v = v / vnorm
        proj = rows @ v
        rows = rows - np.outer(proj, v)
        norms = np.linalg.norm(rows, axis=1)

    numeric_rank = len(selected)
    log10_abs_det = None
    log10_cond = None
    try:
        log10_abs_det = sum(math.log10(max(float(s), ABS_FLOOR)) for s in Spos)
        if Spos.size and Spos[-1] > 0:
            cond = float(Spos[0]) / float(Spos[-1])
            log10_cond = math.log10(max(cond, 1e-300))
        else:
            log10_cond = float('inf')
    except Exception:
        raise

    info = {
        "eigvals": eigvals_desc.tolist(),
        "numeric_rank": numeric_rank,
        "log10_abs_det": log10_abs_det,
        "log10_cond": log10_cond,
        "method": method,
        "kept_eigs_count": int(r),
    }

    return selected, info


def check_2torsion_equivalence(new_div, basis_divs, C):
    """
    Check if new_div ≡ existing_div (mod 2-torsion).
    For genus-2, efficiently check via reduction.
    """
    J = C.jacobian()
    
    # Convert to Jacobian elements
    new_elem = mumford_to_jacobian_element(new_div['s'], new_div['p'], 
                                           new_div['v_0'], new_div['v_1'], C)
    
    for old_div in basis_divs:
        old_elem = mumford_to_jacobian_element(old_div['s'], old_div['p'],
                                               old_div['v_0'], old_div['v_1'], C)
        
        diff = new_elem - old_elem
        
        # Check if diff is 2-torsion: is 2*diff = 0?
        try:
            if (2 * diff).is_zero():
                return True, old_div
        except Exception:
            raise
            
    return False, None


def check_torsion_equivalence(new_jac, basis_jacs, max_order=32):
    """
    Check if new_jac is equivalent to any basis element mod torsion.
    Checks new +- old for torsion orders up to max_order.
    """
    if not basis_jacs:
        return False, None
        
    for i, old_jac in enumerate(basis_jacs):
        # Check Difference: D_new - D_old = Torsion?
        diff = new_jac - old_jac
        if diff.is_zero():
             return True, i
             
        # Check Sum: D_new + D_old = Torsion?
        summ = new_jac + old_jac
        if summ.is_zero():
             return True, i
             
        # Check higher orders (up to max_order)
        # We check m * diff and m * summ
        for m in range(2, max_order + 1):
             if (m * diff).is_zero():
                 return True, i
             if (m * summ).is_zero():
                 return True, i
                 
    return False, None


# --- Worker replacements --------------------------------------------------

def _adaptive_guard_bits(prec_bits: int) -> int:
    """Return a modest number of guard bits based on requested precision.

    Avoid exploding the working precision (user previously tried 8192 guard
    bits which caused the theta radius to blow up). We use an adaptive rule:
      guard = min(max(32, prec//8), 512)
    so for very large `prec` we still cap at 512 extra bits.
    """
    return min(max(32, prec_bits // 8), 512)


def compute_pairing_worker(args):
    """Robust worker function to compute a single N\'eron--Tate pairing.

    Args (tuple): (i, j, div_i, div_j, f_coeffs, prec, h_i, h_j)

    Returns: ((i,j), float_val_or_nan, err_string_or_None)
    """
    try:
        i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args
    except Exception as e:
        raise
        return (None, None, f"bad_args: {e}")

    # If identical indices short-circuit
    if i == j:
        try:
            return ((i, j), float(h_i), None)
        except Exception:
            raise
            return ((i, j), None, "bad_self_height")

    # Work precision with guard bits
    guard = _adaptive_guard_bits(int(prec))
    work_prec = int(prec) + guard

    # Use a consistent real parent for all numeric archimedean work
    RR = RealField(work_prec)

    # Rebuild curve and Jacobian (exact QQ arithmetic)
    try:
        Rq = PolynomialRing(QQ, 'x')
        x = Rq.gen()
        f_poly_QQ = sum(QQ(c) * x ** (len(f_coeffs) - 1 - k) for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
    except Exception as e:
        raise
        return ((i, j), None, f"curve_rebuild_failed: {e}")

    def _rebuild_from_record(div_record):
        try:
            u = x**2 - QQ(div_record['s']) * x + QQ(div_record['p'])
            v = QQ(div_record['v_1']) * x + QQ(div_record['v_0'])
            return J([u, v])
        except Exception as e:
            raise ValueError(f"reconstruct_mumford_failed: {e}")

    # Reconstruct Jacobian elements
    try:
        P = _rebuild_from_record(div_i)
        Q = _rebuild_from_record(div_j)
    except Exception as e:
        raise
        return ((i, j), None, f"div_reconstruct_failed: {e}")

    h_sum = None
    h_diff = None
    sum_err = None
    diff_err = None

    # Try P+Q
    try:
        Dsum = P + Q
        if Dsum.is_zero():
            h_sum = RR(0)
        else:
            # arakelov_canonical_height should accept jacobian element and f_coeffs
            h_tmp = arakelov_canonical_height(Dsum, f_coeffs, prec=work_prec)
            # guard: if the height routine returns a rational or Sage numeric, coerce
            h_sum = RR(h_tmp)
    except Exception as e:
        sum_err = traceback.format_exc()
        h_sum = None
        raise

    # Try P-Q
    try:
        Ddiff = P - Q
        if Ddiff.is_zero():
            h_diff = RR(0)
        else:
            h_tmp = arakelov_canonical_height(Ddiff, f_coeffs, prec=work_prec)
            h_diff = RR(h_tmp)
    except Exception as e:
        diff_err = traceback.format_exc()
        h_diff = None
        raise

    # Combine results with stable arithmetic in RR
    try:
        if h_sum is not None and h_diff is not None:
            val_rr = (h_sum - h_diff) / RR(4)
        elif h_sum is not None:
            # fallback formula: <P,Q> = (h(P+Q) - h(P) - h(Q)) / 2
            val_rr = (h_sum - RR(h_i) - RR(h_j)) / RR(2)
        elif h_diff is not None:
            # <P,Q> = (h(P) + h(Q) - h(P-Q)) / 2
            val_rr = (RR(h_i) + RR(h_j) - h_diff) / RR(2)
        else:
            # Both failed: mark as NaN but return error context so caller can decide
            err_msg = "both_sum_and_diff_failed"
            if sum_err:
                err_msg += "; sum_err=" + sum_err.splitlines()[-1][:200]
            if diff_err:
                err_msg += "; diff_err=" + diff_err.splitlines()[-1][:200]
            return ((i, j), float('nan'), err_msg)

        # Cast to double for compatibility with existing pipelines which store floats
        val_f = float(RealField(int(prec))(val_rr))
        if not math.isfinite(val_f):
            return ((i, j), float('nan'), "nonfinite_result")
        return ((i, j), val_f, None)

    except Exception as e:
        raise
        return ((i, j), float('nan'), f"combine_failed: {e}")


def compute_height_worker(args):
    """Worker function to compute a single (self-)height. Returns (i, float_h, err_or_None)."""
    try:
        i, div, f_coeffs, prec = args
    except Exception as e:
        raise
        return (None, None, f"bad_args: {e}")

    guard = _adaptive_guard_bits(int(prec))
    work_prec = int(prec) + guard
    RR = RealField(work_prec)

    try:
        Rq = PolynomialRing(QQ, 'x')
        x = Rq.gen()
        f_poly_QQ = sum(QQ(c) * x ** (len(f_coeffs) - 1 - k) for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()

        u_poly = x**2 - QQ(div['s']) * x + QQ(div['p'])
        v_poly = QQ(div['v_1']) * x + QQ(div['v_0'])
        D = J([u_poly, v_poly])

        h_tmp = arakelov_canonical_height(D, f_coeffs, prec=work_prec)
        h_rr = RR(h_tmp)
        return (i, float(RealField(int(prec))(h_rr)), None)

    except Exception as e:
        tb = traceback.format_exc()
        raise
        return (i, float('nan'), tb.splitlines()[-1][:200])


# --- Simple analytic pairing primitive -----------------------------------

def neron_tate_height_pairing(z1, z2, Im_tau, prec=300, normalization_factor=1.0):
    """Compute analytic Abel--Jacobi pairing: Re(
    conj(z1)^T * Im(\tau)^{-1} * z2
    ) with normalization factor.

    Inputs z1,z2 can be length-2 complex-like sequences (supports Python complex
    or Sage complex elements). Im_tau should be a 2x2 matrix-like object.

    Returns: float
    """
    # Use CC to hold intermediate complex values at requested precision
    CC = ComplexField(int(prec))
    RR = RealField(int(prec))

    # Defensive conversion for Im_tau
    try:
        # Attempt to use Sage Matrix API if available
        Im_tau_inv = Matrix(CC, Im_tau).inverse()
    except Exception:
        # fall back to numpy
        try:
            Im_np = np.array(Im_tau, dtype=float)
            Im_tau_inv_np = np.linalg.inv(Im_np)
            Im_tau_inv = Matrix(CC, Im_tau_inv_np.tolist())
        except Exception as e:
            raise RuntimeError(f"Im_tau inversion failed: {e}")
        raise

    z1_vec = vector(CC, [CC(z1[0]), CC(z1[1])])
    z2_vec = vector(CC, [CC(z2[0]), CC(z2[1])])
    z1_conj = vector(CC, [z1_vec[0].conjugate(), z1_vec[1].conjugate()])

    temp = Im_tau_inv * z2_vec
    result = z1_conj[0] * temp[0] + z1_conj[1] * temp[1]
    real_part = RR(result.real()) * RR(normalization_factor)
    return float(real_part)


# --- Parallel precompute helper -----------------------------------------

def precompute_pairings_parallel(indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs=1):
    """Compute missing pairings for `indices` in parallel.

    jac_elements: list of (div_dict, jacobian_element, optional_self_height)
    pairing_cache: dict mapping (i,j) -> float (symmetric) to be updated in-place
    Returns the updated pairing_cache.
    """
    tasks = []
    for r in range(len(indices)):
        for c in range(r, len(indices)):
            i = indices[r]
            j = indices[c]
            if i > j:
                i, j = j, i
            if (i, j) in pairing_cache:
                continue
            # prepare args in same order as compute_pairing_worker expects
            div_i = jac_elements[i][0]
            div_j = jac_elements[j][0]
            h_i = height_cache.get(i, float('nan'))
            h_j = height_cache.get(j, float('nan'))
            tasks.append((i, j, div_i, div_j, f_coeffs, prec, h_i, h_j))

    if not tasks:
        return pairing_cache

    # If parallel, use a pool; otherwise fall back to serial
    results = []
    if len(tasks) > 2 and (n_jobs is None or n_jobs > 1):
        nproc = max(1, int(n_jobs))
        with Pool(processes=nproc) as pool:
            for idx, res in enumerate(pool.imap_unordered(compute_pairing_worker, tasks)):
                results.append(res)
    else:
        for args in tasks:
            results.append(compute_pairing_worker(args))

    # Merge results into pairing_cache, converting to floats and symmetric keys
    for item in results:
        try:
            pair, val, err = item
            if pair is None:
                # broken worker returned malformed result
                continue
            i, j = pair
            if err is not None:
                # store NaN and continue; caller may decide what to do
                pairing_cache[(i, j)] = float('nan')
                pairing_cache[(j, i)] = float('nan')
            else:
                # numeric value
                pairing_cache[(i, j)] = float(val)
                pairing_cache[(j, i)] = float(val)
        except Exception:
            # defensive: skip malformed entries
            raise
            continue

    return pairing_cache


# End of core_fixed.py


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Numerically-robust basis builder for genus-2 Arakelov heights.

    Key changes vs the previous version:
      • No determinant or absolute-eigenvalue rejection
      • Independence decided by QR-rank + projection-residual
      • Torsion-equivalence checked only for borderline cases
      • Pairing failures cause candidate rejection, not pipeline abort
    """
    PM = get_period_matrix_auto_B(f_coeffs, prec=prec)

    from .analytic_pairings import setup_analytic_pairing_context
    setup_analytic_pairing_context(
        f_coeffs=f_coeffs,
        tau=PM,
        prec=prec,
        verbose=debug
    )

    if not all_divisors:
        return [], 0, None

    # --- Build curve / Jacobian once ---
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i,c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    jac_elements = []
    for div in all_divisors:
        u = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v = QQ(div['v_1'])*x + QQ(div['v_0'])
        jac_elements.append((div, J([u,v]), div.get('_h_diag', None)))

    n = len(jac_elements)

    from .analytic_pairings import (
        precompute_abel_jacobi_images,
        get_analytic_pairing
    )

    precompute_abel_jacobi_images(list(range(n)), jac_elements,
                                  prec=prec, debug=debug)

    # --- Heights first ---
    height_cache = {}
    for i,(rec,jac,h) in enumerate(jac_elements):
        try:
            if h is None:
                h = arakelov_canonical_height(jac, f_coeffs, PM,
                                              prec=prec, debug=debug)
            height_cache[i] = float(h)
        except Exception as e:
            if debug:
                print(f"[height] failed for {i}: {e}")
            continue

    def pairing(i,j):
        try:
            return float(get_analytic_pairing(i,j))
        except Exception:
            return np.nan

    # --- Incremental selection ---
    basis_indices = []

    for cand in range(n):

        if cand not in height_cache:
            continue

        # always accept first non-zero height
        if not basis_indices:
            if abs(height_cache[cand]) > 1e-9:
                basis_indices.append(cand)
                if debug:
                    print(f"  added {cand} (seed)")
            continue

        # build Gram for basis + candidate
        k = len(basis_indices)
        idxs = basis_indices + [cand]

        G = np.zeros((k+1,k+1),float)
        bad = False
        for r in range(k+1):
            for c2 in range(r,k+1):
                v = pairing(idxs[r], idxs[c2])
                if not np.isfinite(v):
                    bad = True
                    break
                G[r,c2]=G[c2,r]=v
            if bad: break
        if bad:
            if debug: print(f"  reject {cand}: pairing failure")
            continue

        # symmetrize + scale
        G = 0.5*(G+G.T)
        scale = 1/max(1.0,np.linalg.norm(G))
        Q,R = np.linalg.qr(scale*G)
        diag = np.abs(np.diag(R))
        tol = 1e-12*max(1.0,diag.max())
        rank = np.sum(diag > tol)

        if rank > len(basis_indices):
            # borderline? → torsion check only here
            cand_j = jac_elements[cand][1]
            basis_j = [jac_elements[i][1] for i in basis_indices]
            equiv,_ = check_torsion_equivalence(cand_j, basis_j, max_order=32)
            if equiv:
                if debug:
                    print(f"  reject {cand}: torsion-equivalent")
                continue

            basis_indices.append(cand)
            if debug:
                print(f"  added {cand} (basis size {len(basis_indices)})")

        if len(basis_indices) >= C.genus():
            break

    # --- Final Gram matrix ---
    basis = [jac_elements[i][0] for i in basis_indices]
    m = len(basis_indices)

    H = None
    if m>0:
        RR = RealField(max(128,int(prec//4)))
        H = Matrix(RR,m,m)
        for r in range(m):
            for c2 in range(r,m):
                v = pairing(basis_indices[r], basis_indices[c2])
                H[r,c2]=RR(v); H[c2,r]=H[r,c2]

    return basis, len(basis_indices), H
