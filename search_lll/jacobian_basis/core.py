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
from search_lll.homology import *

# Functions: dedupe_basis, gram_logdet_and_cond (moved to pairings),
# select_independent_indices_from_gram, arakelov_build_basis_with_heights,
# is_independent_by_projection_log


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
    Robust independence test for a candidate divisor `cand_idx` against the
    current basis `basis_indices`.

    Policy:
      - Build the k×k Gram G for basis and the augmented (k+1)×(k+1) Gram G_aug
        that includes the candidate.
      - Use eigenvalue thresholding (numeric rank) to decide whether the
        augmented Gram increases rank.  If rank increases -> independent.
      - If rank does not increase (ambiguous), fall back to two residual tests:
          * relative residual vector norm from least-squares projection
          * residual energy = h(cand) - alpha^T b
        If both residuals are small -> dependent, otherwise independent.
      - This *never* raises a fatal error simply because the Gram isn't PD;
        instead it returns (is_indep, diagnostics).

    Parameters:
      basis_indices: list of ints (indices into the global divisor list)
      cand_idx: int (index of candidate)
      get_pairing: callable(i,j) -> pairing value (diagonal = height)
      prec: bit-precision hint (used only for debug / tolerances scaling)
      rel_eig_tol, abs_eig_tol: eigenvalue thresholding parameters
    Returns:
      (is_indep: bool, info: dict)
    """
    import numpy as np
    from math import isfinite

    info = {}
    k = len(basis_indices)
    # trivial case
    if k == 0:
        info['reason'] = 'empty_basis'
        return True, info

    # build Gram G (k x k), vector b (k), and h_cand
    try:
        G_np = np.zeros((k, k), dtype=float)
        b_np = np.zeros((k,), dtype=float)
        for i in range(k):
            for j in range(i, k):
                val = float(get_pairing(basis_indices[i], basis_indices[j]))
                G_np[i, j] = val
                G_np[j, i] = val
        for i in range(k):
            b_np[i] = float(get_pairing(basis_indices[i], cand_idx))
        h_cand = float(get_pairing(cand_idx, cand_idx))
    except Exception as e:
        # if pairings missing or non-numeric, return an informative failure
        info['error'] = f'pairing_error: {type(e).__name__}: {e}'
        raise
        return False, info

    if not (isfinite(h_cand) and np.all(np.isfinite(G_np)) and np.all(np.isfinite(b_np))):
        info['error'] = 'nonfinite_entry_in_pairings'
        return False, info

    # symmetrize to kill tiny noise
    G_np = 0.5 * (G_np + G_np.T)

    # augmented Gram
    G_aug = np.zeros((k + 1, k + 1), dtype=float)
    G_aug[:k, :k] = G_np
    G_aug[:k, k] = b_np
    G_aug[k, :k] = b_np
    G_aug[k, k] = h_cand

    # eigenvalue analysis (use eigh for symmetric)
    try:
        eigs_G = np.linalg.eigvalsh(G_np)
        eigs_aug = np.linalg.eigvalsh(G_aug)
    except np.linalg.LinAlgError:
        # fallback: small random jitter then retry
        jitter = 1e-16
        G_np += jitter * np.eye(k)
        G_aug[:k, :k] = G_np
        eigs_G = np.linalg.eigvalsh(G_np)
        eigs_aug = np.linalg.eigvalsh(G_aug)

    # compute numeric rank with thresholding
    max_eig_G = max(abs(eigs_G.max()), abs(eigs_G.min()), 1.0)
    max_eig_aug = max(abs(eigs_aug.max()), abs(eigs_aug.min()), 1.0)

    # thresholds: relative to largest eigenvalue, plus absolute floor
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

    # Primary decision: strictly increasing numeric rank => independent
    if rank_aug > rank_G:
        info['reason'] = 'rank_increase'
        return True, info

    # Otherwise ambiguous: use least-squares projection residual tests
    # Solve G alpha ≈ b via least-squares / pseudo-inverse (numeric stable)
    try:
        # handle near-singular by regularized least squares
        # use np.linalg.lstsq for a stable pseudo-inverse solution
        alpha, *_ = np.linalg.lstsq(G_np, b_np, rcond=None)
        proj_b = G_np.dot(alpha)
        residual_vec = b_np - proj_b
        residual_norm = float(np.linalg.norm(residual_vec))
        b_norm = float(np.linalg.norm(b_np))
        rel_residual = residual_norm / (b_norm + 1e-18)

        # residual energy: h_cand - alpha^T b
        predicted = float(np.dot(alpha, b_np))
        residual_energy = float(h_cand - predicted)

        info.update({
            'alpha_norm': float(np.linalg.norm(alpha)),
            'residual_norm': residual_norm,
            'rel_residual': rel_residual,
            'residual_energy': residual_energy,
            'predicted': predicted,
        })

        # thresholds (conservative):
        # - small vector residual -> dependent
        # - small residual energy (relative to h_cand) -> dependent
        residual_vector_thresh = 1e-6  # relative tolerance on vector residual
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
        # numeric failure; conservatively declare dependent but provide diagnostics
        info['error'] = f'lstsq_failed: {type(e).__name__}: {e}'
        info['reason'] = 'conservative_dependent_on_error'
        return False, info


def select_independent_indices_from_gram(
    G,
    prec_bits=2048,
    safety_digits=10,
    rel_sv_tol=1e-12,
    pivot_tol_factor=1e-9,
    debug=False,
):
    """
    Robust selection of numerically independent indices from a symmetric Gram matrix G.

    - Accepts a Sage Matrix or an ndarray / list-of-lists.
    - Symmetrizes, uses eigendecomposition to determine a numeric rank, but NEVER
      throws if tiny negative eigenvalues appear: it regularizes them conservatively.
    - Deterministic pivoting selects rows with largest residual norm (embedding deflation).
    - Returns (selected_indices, info_dict).

    Info dict fields:
      'eigvals' (descending), 'numeric_rank', 'log10_abs_det', 'log10_cond', 'method'
    """
    import numpy as np
    import math

    # --- convert to numpy float array (symmetrize) ---
    if hasattr(G, "nrows"):
        # Sage matrix
        n = int(G.nrows())
        G_np = np.array([[float(G[i, j]) for j in range(n)] for i in range(n)], dtype=float)
    else:
        G_np = np.array(G, dtype=float)
        if G_np.ndim != 2 or G_np.shape[0] != G_np.shape[1]:
            raise ValueError("Input must be a square matrix")
        n = G_np.shape[0]

    # symmetrize to reduce noise
    G_np = 0.5 * (G_np + G_np.T)

    # tiny helper floors
    ABS_FLOOR = 1e-300

    # --- eigendecomposition (symmetric) with robust fallback ---
    try:
        eigvals = np.linalg.eigvalsh(G_np)  # ascending
        eigvecs = None
        method = "eig"
    except Exception:
        # fallback to SVD (more robust for badly conditioned inexact matrices)
        if debug:
            print("[select_from_gram] eigvalsh failed, falling back to SVD")
        U, svals, Vt = np.linalg.svd(G_np)
        eigvals = svals.copy()
        eigvecs = U.copy()
        method = "svd"
        raise

    # ensure an array and sort descending
    eigvals = np.array(eigvals, dtype=float)
    if eigvecs is None and method == "eig":
        # get eigenvectors in a safe way (we only need them when positive eigs exist)
        try:
            w, V = np.linalg.eigh(G_np)
            eigvals = np.array(w, dtype=float)
            eigvecs = V
            method = "eig"
        except Exception:
            # leave eigvecs None; we'll fallback to SVD embedding if needed
            eigvecs = None
            raise

    # ascending -> descending for compatibility with old info
    eigvals_desc = eigvals[::-1]

    # smax and candidate thresholding
    smax = float(max(eigvals_desc[0], 0.0)) if eigvals_desc.size else 0.0

    # convert prec_bits -> decimal digits estimate (rough)
    dec_digits = int(prec_bits * 0.30103) if prec_bits > 0 else 50
    dec_digits_cap = min(max(dec_digits, 0), 50)

    # build eigenvalue threshold (conservative)
    safety_power = max(safety_digits, dec_digits_cap)
    # avoid 10**(-huge) underflow by scaling with smax
    ev_thresh = max(smax * (10.0 ** (-safety_power)), smax * rel_sv_tol, 1e-300)

    # If smax is zero (nearly zero matrix), use SVD singulars to get scale
    if smax <= 0:
        try:
            _, svals_svd, _ = np.linalg.svd(G_np)
            smax = float(svals_svd[0]) if svals_svd.size else 0.0
            ev_thresh = max(smax * rel_sv_tol, 1e-300)
            if debug:
                print(f"[select_from_gram] smax was <=0; using SVD smax={smax:.3g}")
            method = "svd"
            eigvecs = np.linalg.svd(G_np)[0]
            eigvals_desc = np.array(svals_svd, dtype=float)  # treat as descending
        except Exception:
            # matrix essentially zero; nothing to select
            raise
            return [], {
                "eigvals": eigvals_desc.tolist(),
                "numeric_rank": 0,
                "log10_abs_det": None,
                "log10_cond": None,
                "method": method,
            }

    # compute mask of 'positive' eigenvalues we keep
    # here eigvals_desc is descending; find indices > ev_thresh
    pos_mask_desc = eigvals_desc > ev_thresh
    pos_indices_desc = np.nonzero(pos_mask_desc)[0]
    r = int(len(pos_indices_desc))

    # If none are above threshold, try a more permissive fallback using rel_sv_tol
    if r == 0:
        alt_thresh = max(smax * rel_sv_tol, 1e-300)
        pos_mask_desc = eigvals_desc > alt_thresh
        pos_indices_desc = np.nonzero(pos_mask_desc)[0]
        r = int(len(pos_indices_desc))

    if debug:
        print(f"[select_from_gram] method={method} smax={smax:.3g} ev_thresh={ev_thresh:.3g} kept={r}")

    if r == 0:
        # numeric rank 0
        return [], {
            "eigvals": eigvals_desc.tolist(),
            "numeric_rank": 0,
            "log10_abs_det": None,
            "log10_cond": None,
            "method": method,
        }

    # If we don't yet have eigenvectors aligned with eigvals_desc, attempt to compute them.
    if eigvecs is None:
        try:
            _, V = np.linalg.eigh(G_np)
            eigvecs_full = V
        except Exception:
            # final fallback: use SVD U as eigenvectors proxy
            U, svals_svd, Vt = np.linalg.svd(G_np)
            eigvecs_full = U
            raise
    else:
        eigvecs_full = eigvecs

    # We need the vectors corresponding to the kept (largest) eigenvalues.
    # For safety map indices correctly (eigs returned were ascending earlier).
    # If eigvals came from eigvalsh (ascending), eigvals_desc = eigvals[::-1] so index mapping is n-1-idx
    if eigvecs_full.shape[1] == n:
        # assume eigvecs_full columns correspond to ascending eigenvalues if method == 'eig'
        # create Upos as columns for the top-r eigenvectors
        try:
            # try using eigenvectors in descending order
            U_desc = eigvecs_full[:, ::-1]
            Upos = U_desc[:, pos_indices_desc]
        except Exception:
            Upos = eigvecs_full[:, :r]
            raise
    else:
        # non-square/unknown shape (SVD fallback), take first r columns
        Upos = eigvecs_full[:, :r]

    # take eigenvalues for kept ones (descending)
    if eigvals_desc.size >= r:
        Spos = eigvals_desc[pos_indices_desc]  # length r
    else:
        # fallback: use top r singulars from SVD
        Spos = np.maximum(np.array(eigvals_desc[:r], dtype=float), ABS_FLOOR)

    # ensure nonnegative and floor tiny negatives to small positive
    Spos = np.maximum(Spos, ABS_FLOOR)

    # Embedding E (n x r) with rows as embedding vectors
    sqrtS = np.sqrt(Spos)
    E = Upos * sqrtS[np.newaxis, :]

    # pivot tolerance derived from smallest retained eigenvalue + global scale
    min_pos_eig = float(Spos[-1]) if Spos.size else ABS_FLOOR
    pivot_tol = max(math.sqrt(max(min_pos_eig, ABS_FLOOR)) * pivot_tol_factor,
                    smax * 1e-16)

    # deterministic pivot selection by largest residual norm (embedding deflation)
    rows = E.copy()
    norms = np.linalg.norm(rows, axis=1)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    # iterate selecting until max norm below pivot_tol or we've chosen r indices
    while True:
        # mask out already selected by setting extremely small values
        masked_norms = norms.copy()
        masked_norms[selected_mask] = -1.0
        cand = int(np.argmax(masked_norms))
        maxnorm = float(masked_norms[cand])
        if debug:
            print(f"[pivot] pick={cand} maxnorm={maxnorm:.6g} selected={len(selected)} pivot_tol={pivot_tol:.6g}")
        if maxnorm <= pivot_tol or len(selected) >= r:
            break
        # add candidate
        selected.append(cand)
        selected_mask[cand] = True
        v = rows[cand].copy()
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            break
        v = v / vnorm
        # deflate rows by projection
        proj = rows @ v
        rows = rows - np.outer(proj, v)
        norms = np.linalg.norm(rows, axis=1)

    numeric_rank = len(selected)

    # compute log10 det and condition number info from Spos
    # log10_abs_det = sum log10(Spos)
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
        log10_abs_det = None
        log10_cond = None
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


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Incremental basis builder: compute period matrix once, precompute heights,
    precompute pairings for a small prefix, and compute pairings on-demand using
    the same period matrix. Robust to occasional theta/finite-place failures:
    missing/failed computations are recorded conservatively as 0.0 with a warning.
    """
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
        jac_elements.append((div, div_j, div.get('_h_diag', None)))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    # 3) Compute individual canonical heights using the same PM (serial for stability).
    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_cache = {}
    for i in range(n):
        try:
            _, jacP, h = jac_elements[i]
            if h is None:
                h = arakelov_canonical_height(jacP, f_coeffs, prec=prec, debug=debug, period_matrix=PM)
            # store as float for subsequent linear algebra usage
            height_cache[i] = float(h)
            if debug:
                print(f"  Divisor {i}: h = {height_cache[i]:.6g}")
        except Exception as e:
            warnings.warn(f"[arakelov] height computation failed for index {i}: {e}. Using 0.0 as conservative fallback.", RuntimeWarning)
            height_cache[i] = 0.0
            # Do NOT raise here; try to salvage other divisors

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
                _, Ji, _ = jac_elements[i]
                _, Jj, _ = jac_elements[j]
                hij = arakelov_canonical_height(Ji + Jj, f_coeffs, prec=prec, debug=debug, period_matrix=PM)
                pairing_val = 0.5 * (float(hij) - float(height_cache[i]) - float(height_cache[j]))
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = float(pairing_val)
            except Exception as e:
                warnings.warn(f"[arakelov] pairing precompute failed for ({i},{j}): {e}. Using 0.0 fallback.", RuntimeWarning)
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = 0.0
                # Do NOT raise here.

    # Helper to compute or fetch pairing lazily, using the same PM so results are consistent.
    def get_pairing_lazy(i, j):
        # canonicalize order
        if i > j:
            i, j = j, i
        if (i, j) in pairing_cache:
            return pairing_cache[(i, j)]

        # if diagonal, return stored height
        if i == j:
            val = height_cache.get(i, 0.0)
            pairing_cache[(i, i)] = val
            return val

        # compute on-demand: compute h(D_i + D_j) with shared PM, then pairing = 0.5*(h_ij - hi - hj)
        try:
            _, Ji, _ = jac_elements[i]
            _, Jj, _ = jac_elements[j]
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
            continue # do not raise

        if det_val > 0:
            basis_indices.append(cand_idx)
            if debug:
                print(f"  Added divisor {cand_idx} (basis now size {len(basis_indices)})")
        else:
            if debug:
                print(f"  Rejected divisor {cand_idx}: dependent (Gram det <= 0)")

    # 6) Ensure sanity_check_pairings has the entries it needs
    if len(basis_indices) > 0:
        need_k = min(len(basis_indices), 4)
        for ii in range(need_k):
            for jj in range(ii, need_k):
                i = basis_indices[ii]; j = basis_indices[jj]
                if (i, j) not in pairing_cache:
                    _ = get_pairing_lazy(i, j)
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
                if min(f_eigs) > 0:
                    print("condition number (approx):", float(max(f_eigs) / min(f_eigs)))
        except Exception as e:
            warnings.warn(f"[arakelov] final Gram diagnostics failed: {e}", RuntimeWarning)
            # do not raise

    return basis, final_rank, H_final
