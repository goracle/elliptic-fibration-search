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


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Incremental basis builder: compute period matrix once, precompute heights,
    precompute pairings for a small prefix, and compute pairings on-demand.
    
    Robustness:
      - Uses a strictly raising height implementation.
      - If height/pairing computations fail, the candidate divisor is REJECTED 
        rather than poisoning the basis with 0.0.
      - Explicitly checks for 2-torsion equivalence to avoid rank inflation.
    """
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

    try:
        PM = get_period_matrix_auto_B(f_coeffs, prec=prec)
    except Exception as e:
        raise RuntimeError(f"[arakelov] get_period_matrix_auto_B failed at prec={prec}: {e}")

    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    jac_elements = []
    for div in all_divisors:
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        div_j = J([u_poly, v_poly])
        jac_elements.append((div, div_j, div.get('_h_diag', None)))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_cache = {}
    for i in range(n):
        try:
            _, jacP, h = jac_elements[i]
            if h is None:
                h = arakelov_canonical_height(jacP, f_coeffs, PM, prec=prec, debug=debug)
            height_cache[i] = float(h)
            if debug:
                print(f"  Divisor {i}: h = {height_cache[i]:.6g}")
        except Exception as e:
            if debug:
                warnings.warn(f"[arakelov] height computation failed for index {i}: {e}. Skipping divisor.", RuntimeWarning)
            raise

    pairing_cache = {}
    for i in height_cache:
        pairing_cache[(i, i)] = height_cache[i]

    try:
        genus = C.genus()
    except Exception:
        genus = 2
        raise
    prefix_precompute = min(n, max(8, 2 * genus + 4))

    if debug:
        print(f"[arakelov] Precomputing pairings for first {prefix_precompute} divisors (prefix).")

    for i in range(prefix_precompute):
        for j in range(i + 1, prefix_precompute):
            if i not in height_cache or j not in height_cache:
                continue
            if (i, j) in pairing_cache:
                continue
            try:
                _, Ji, _ = jac_elements[i]
                _, Jj, _ = jac_elements[j]
                hij = arakelov_canonical_height(Ji + Jj, f_coeffs, PM, prec=prec, debug=debug)
                pairing_val = 0.5 * (float(hij) - float(height_cache[i]) - float(height_cache[j]))
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = float(pairing_val)
                if debug:
                    print(f"  Pair ({i},{j}): {pairing_val:.6g}")
            except Exception as e:
                if debug:
                    warnings.warn(f"[arakelov] pairing precompute failed for ({i},{j}): {e}. Ignoring.", RuntimeWarning)
                import numpy as np
                pairing_cache[(i, j)] = pairing_cache[(j, i)] = np.nan

    def get_pairing_lazy(i, j):
        if i > j: i, j = j, i
        
        if i not in height_cache or j not in height_cache:
            raise ValueError(f"Cannot pair divisors {i},{j}: invalid self-heights")

        if (i, j) in pairing_cache:
            return pairing_cache[(i, j)]

        if i == j:
            return height_cache[i]

        try:
            _, Ji, _ = jac_elements[i]
            _, Jj, _ = jac_elements[j]
            hij = arakelov_canonical_height(Ji + Jj, f_coeffs, PM, prec=prec, debug=debug)
            val = 0.5 * (float(hij) - float(height_cache[i]) - float(height_cache[j]))
            pairing_cache[(i, j)] = pairing_cache[(j, i)] = float(val)
            return val
        except Exception as e:
            raise RuntimeError(f"Pairing computation failed for ({i},{j}): {e}")

    if debug:
        print("[arakelov] Selecting basis incrementally...")

    basis_indices = []

    for cand_idx in range(n):
            
        if cand_idx not in height_cache:
            if debug:
                print(f"  Skipping divisor {cand_idx}: invalid height")
            continue

        if len(basis_indices) == 0:
            h = height_cache[cand_idx]
            if abs(h) > 1e-8:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (first basis element)")
            else:
                if debug:
                    print(f"  Skipping divisor {cand_idx}: height ~ 0")
            continue

        cand_jac = jac_elements[cand_idx][1]
        is_torsion_equiv = False
        
        for b_idx in basis_indices:
            basis_jac = jac_elements[b_idx][1]
            try:
                diff = cand_jac - basis_jac
                if (2 * diff).is_zero():
                    is_torsion_equiv = True
                    if debug:
                        print(f"  Rejected divisor {cand_idx}: equivalent to basis {b_idx} mod 2-torsion (diff)")
                    break
                
                summ = cand_jac + basis_jac
                if (2 * summ).is_zero():
                    is_torsion_equiv = True
                    if debug:
                        print(f"  Rejected divisor {cand_idx}: equivalent to basis {b_idx} mod 2-torsion (sum)")
                    break
            except Exception:
                raise
        
        if is_torsion_equiv:
            continue

        try:
            is_indep, info = is_independent_by_projection_log(
                basis_indices,
                cand_idx,
                get_pairing_lazy,
                prec,
                debug=debug
            )
        except Exception as e:
            if debug:
                print(f"  Rejected divisor {cand_idx}: pairing computation failed ({e})")
            raise
            continue

        if not is_indep:
            if debug:
                print(f"  Rejected divisor {cand_idx}: dependent (proj test)")
            continue

        try:
            k = len(basis_indices)
            from sage.all import RealField as SFRealField, Matrix as SFMatrix
            RR_Gram = SFRealField(max(128, int(prec // 4)))
            G_test = SFMatrix(RR_Gram, k + 1, k + 1)

            indices_to_fill = basis_indices + [cand_idx]
            for r in range(k + 1):
                for c in range(r, k + 1):
                    pv = get_pairing_lazy(indices_to_fill[r], indices_to_fill[c])
                    G_test[r, c] = RR_Gram(pv)
                    G_test[c, r] = G_test[r, c]
            
            det_val = float(G_test.determinant())
            
            # Compute eigenvalues to check condition number
            eigs = [float(e) for e in G_test.eigenvalues()]
            eigs_sorted = sorted([abs(e) for e in eigs], reverse=True)
            min_eig = eigs_sorted[-1]
            max_eig = eigs_sorted[0]
            
            # Reject if any eigenvalue is too small (relative to max)
            if min_eig < 1e-6 * max_eig:
                if debug:
                    print(f"  Rejected divisor {cand_idx}: matrix too ill-conditioned (min_eig={min_eig:.3e}, max_eig={max_eig:.3e})")
                continue
                
            # Also reject if smallest eigenvalue is absolutely tiny
            if min_eig < 1e-9:
                if debug:
                    print(f"  Rejected divisor {cand_idx}: smallest eigenvalue too small ({min_eig:.3e})")
                continue
            
        except Exception as e:
            if debug:
                print(f"  Rejected divisor {cand_idx}: Gram check failed ({e})")
            raise
            continue

        if det_val > 1e-9:
            basis_indices.append(cand_idx)
            if debug:
                print(f"  Added divisor {cand_idx} (basis now size {len(basis_indices)})")
        else:
            if debug:
                print(f"  Rejected divisor {cand_idx}: dependent (Gram det <= 1e-9)")

    if len(basis_indices) > 0:
        need_k = min(len(basis_indices), 4)
        for ii in range(need_k):
            for jj in range(ii, need_k):
                i = basis_indices[ii]; j = basis_indices[jj]
                if (i, j) not in pairing_cache:
                    try:
                        _ = get_pairing_lazy(i, j)
                    except Exception:
                        raise

    basis = [jac_elements[i][0] for i in basis_indices]
    basis, basis_indices = dedupe_basis(basis, basis_indices, debug=debug)

    final_rank = len(basis)
    H_final = None

    if final_rank > 0:
        RR_Final = RealField(max(128, int(prec // 4)))
        H_final = Matrix(RR_Final, final_rank, final_rank)
        if debug:
            print(f"[arakelov] Building final {final_rank} Gram matrix...")

        for r in range(final_rank):
            for c in range(r, final_rank):
                try:
                    pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                    H_final[r, c] = RR_Final(pv)
                    H_final[c, r] = H_final[r, c]
                except Exception:
                    #H_final[r, c] = H_final[c, r] = 0 # ai thought it was ok to make this zero!  amazing!
                    raise

        try:
            eigs = [float(e) for e in H_final.eigenvalues()]
            det_final = float(H_final.determinant())
            if debug:
                print("eigenvalues (float):", eigs)
                print("determinant:", det_final)
            
            # Final sanity check: matrix must be positive definite
            assert det_final > 0, f"Gram matrix has negative determinant {det_final} - basis is dependent!"
            assert all(e > -1e-12 for e in eigs), f"Gram matrix has negative eigenvalue - not positive definite!"
            
        except Exception as e:
             warnings.warn(f"[arakelov] final Gram diagnostics failed: {e}", RuntimeWarning)
             raise

    return basis, final_rank, H_final
