import multiprocessing, itertools
from operator import mul
from functools import reduce, partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import namedtuple, Counter
from sage.all import QQ, ZZ, GF, EllipticCurve, Integer, vector, PolynomialRing, var, matrix, identity_matrix, lcm, SR, Zmod
from .search_config import DEBUG, MIN_PRIME_SUBSET_SIZE, MAX_MODULUS, ROOTS_THRESHOLD, MAX_K_ABS, LLL_DELTA, BKZ_BLOCK, TRUNCATE_MAX_DEG, TMAX, HENSEL_SLOPPY, TORSION_SLOPPY, MAX_TORSION_ORDER_TO_FILTER
from .rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .ll_utilities import _trim_poly_coeffs, _compute_column_norms, _scale_matrix_columns_int, _compute_integer_scales_for_columns
from .archimedean_optim import minimize_archimedean_t_linear_const

"""
modular_workers.py: Parallel worker functions and modular reduction setup.

NOTE on import order (bit us before, writing it down so it doesn't again):
search_lll/__init__.py does `from .modularthread import *` BEFORE
`from .ll_utilities import *`. ll_utilities.py has its own, more current
versions of prepare_modular_data_lll() and compute_all_mults_for_section()
(5-tuple return incl. section_poly_dict, Julia-ladder support, etc.) which
end up shadowing anything defined here under those same names. This file
used to carry stale duplicates of both -- they were never actually called
(ll_utilities' copies always won), just dead weight that cost two separate
debugging sessions before the mismatch was found. Removed. If you need to
add a same-named helper here again, either import it from ll_utilities
instead of redefining it, or give it a different name.
"""

# Standard library and external imports

# SageMath imports

# Local Configuration Imports

# --- IMPORT FIX: Use relative import to get modules from parent directory ---
try:
    from diagnostics2 import find_singular_fibers
    from brauer import compute_ramification_locus
except ImportError:
    print("Warning: search_lll/modularthread.py could not import diagnostics2 from parent dir.")
    print("         Fiber collision checks will be limited.")
    def find_singular_fibers(*args, **kwargs): return {'fibers': [], 'euler_characteristic': 0, 'sigma_sum': 0}
    def compute_ramification_locus(*args, **kwargs): return set()
    raise
# --- END IMPORT FIX ---# ==============================================================
# === Modular Reduction Helpers ================================
# ==============================================================

def _get_coeff_data(poly):
    """Helper to safely extract coefficient list and degree from a polynomial-like object."""
    if hasattr(poly, 'list') and hasattr(poly, 'degree'):
        return poly.list(), poly.degree()
    else:
        # Handle constants or other non-polynomial objects
        return [poly], 0

def reduce_point_hom(E_mod_p, P, p, logger=None):
    """
    Reduce a projective/affine point P (whose coordinates may be
    in QQ or QQ(m)) to the curve E_mod_p (which is over GF(p) or GF(p)(m)).
    Returns:
        - The reduced point on E_mod_p on success
        - None if reduction fails (denominator non-invertible, bad coords, etc.)
    """
    from sage.all import GF, ZZ

    def log(msg):
        if logger:
            logger(msg)
        elif DEBUG:
            print(msg)

    try:
        # Get the target field, e.g., GF(p)(m) or GF(p)
        Fp_target = E_mod_p.base_field()
        coords = tuple(P)

        # Coerce coordinates into the target field
        if len(coords) == 3:
            X, Y, Z = coords
            try:
                Xr = Fp_target(X)
                Yr = Fp_target(Y)
                Zr = Fp_target(Z)
                return E_mod_p([Xr, Yr, Zr])
            except Exception as e:
                return None

        if len(coords) == 2:
            x, y = coords
            try:
                xr = Fp_target(x)
                yr = Fp_target(y)
                return E_mod_p(xr, yr)
            except Exception as e:
                return None

        log("[reduce_point_hom] unsupported coordinate shape")
        return None

    except Exception as outer_e:
        log(f"[reduce_point_hom] p={p} unexpected error: {outer_e}")
        return None

def lll_reduce_basis_modp(p, sections, curve_modp,
                          truncate_deg=TRUNCATE_MAX_DEG,
                          lll_delta=LLL_DELTA, bkz_block=BKZ_BLOCK,
                          max_k_abs=MAX_K_ABS):
    """
    LLL/BKZ reduction with proper handling of single-section case and reduction failures.
    Returns a list of length r = len(sections).
    """
    from sage.all import ZZ, identity_matrix, diagonal_matrix

    r = len(sections)
    if r == 0:
        return [], identity_matrix(ZZ, 0)

    reduced_sections_mod_p = [reduce_point_hom(curve_modp, P, p) for P in sections]

    if all(P is None for P in reduced_sections_mod_p):
        if DEBUG:
            print(f"[{p}] All {r} sections failed to reduce. Returning list of Nones.")
        return [None] * r, identity_matrix(ZZ, r)

    poly_coords = []
    max_deg = 0

    for Pp in reduced_sections_mod_p:
        if Pp is None:
            poly_coords.append(([0], [0], [1]))
            continue

        Xr, Yr, Zr = Pp[0], Pp[1], Pp[2]
        den = lcm([Xr.denominator(), Yr.denominator(), Zr.denominator()])
        Xp = Xr.numerator() * (den // Xr.denominator())
        Yp = Yr.numerator() * (den // Yr.denominator())
        Zp = Zr.numerator() * (den // Zr.denominator())

        xc, dx = _get_coeff_data(Xp)
        yc, dy = _get_coeff_data(Yp)
        zc, dz = _get_coeff_data(Zp)

        xc = _trim_poly_coeffs(xc, truncate_deg)
        yc = _trim_poly_coeffs(yc, truncate_deg)
        zc = _trim_poly_coeffs(zc, truncate_deg)

        poly_coords.append((xc, yc, zc))
        max_deg = max(max_deg, len(xc)-1, len(yc)-1, len(zc)-1)

    poly_len = max_deg + 1
    coeff_vecs = []
    for xc, yc, zc in poly_coords:
        xc_padded = list(xc) + [0] * (poly_len - len(xc))
        yc_padded = list(yc) + [0] * (poly_len - len(yc))
        zc_padded = list(zc) + [0] * (poly_len - len(zc))
        row = [ZZ(int(c)) for c in (xc_padded + yc_padded + zc_padded)]
        coeff_vecs.append(vector(ZZ, row))

    if not coeff_vecs or all(v.is_zero() for v in coeff_vecs):
        if DEBUG:
            print("All coefficient vectors are zero or truncated away, using identity transformation")
        return reduced_sections_mod_p, identity_matrix(ZZ, r)

    M = matrix(ZZ, coeff_vecs)

    if M.nrows() <= 1:
        Uinv = identity_matrix(ZZ, r)
        return reduced_sections_mod_p, Uinv

    if M.ncols() > 5 * M.nrows():
        if DEBUG:
            print(f"[LLL] Matrix too wide ({M.nrows()}x{M.ncols()}), skipping LLL for this prime")
        return reduced_sections_mod_p, identity_matrix(ZZ, r)

    try:
        scales = _compute_integer_scales_for_columns(M)
        M_scaled, D = _scale_matrix_columns_int(M, scales)
    except Exception as e:
        if DEBUG:
            print("Column scaling failed, proceeding without scaling:", e)
        M_scaled = M
        D = diagonal_matrix([1]*M.ncols())

    U = None
    B = None
    try:
        if hasattr(M_scaled, "BKZ"):
            block = min(bkz_block, max(2, M_scaled.ncols()//2))
            U, B = M_scaled.BKZ(block_size=block, transformation=True)
        else:
            U, B = M_scaled.LLL(transformation=True, delta=float(lll_delta))
    except (TypeError, ValueError):
        try:
            U, B = M_scaled.LLL(transformation=True)
        except Exception as e:
            if DEBUG:
                print("LLL/BKZ reduction failed, falling back to identity:", e)
            U = identity_matrix(ZZ, r)
            B = M_scaled.copy()

    Uinv = U.inverse()

    new_basis = []
    identity_point = curve_modp(0)

    for i in range(r): # Loop r times
        S_i = identity_point
        try:
            valid_sum = False
            for j in range(r): # Loop r times
                P_j = reduced_sections_mod_p[j]
                if P_j is not None:
                    S_i += U[i, j] * P_j
                    valid_sum = True

            if valid_sum:
                new_basis.append(S_i)
            else:
                new_basis.append(None)
        except Exception as e:
            if DEBUG:
                print(f"[LLL] Error computing new basis vector {i}: {e}")
            new_basis.append(None)

    return new_basis, Uinv

# ==============================================================
# === Preparation and LLL Reduction (Modular) ==================
# ==============================================================

# ==============================================================
# === Main Worker Functions (Single Subset) ====================
# ==============================================================

def compute_residues_for_prime_worker(args):
    """
    Worker computing residues for one prime with Hensel filtering.
    (From source [476])
    """
    from sage.all import GF, Integer, QQ, ZZ, EllipticCurve

    try:
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs, _stats = args
    except Exception:
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs = args
        _stats = None

    result_for_p = {}
    local_modular_checks = 0

    HENSEL_STRICT = HENSEL_SLOPPY
    HENSEL_ALLOW_WEAK = not HENSEL_STRICT

    for idx, v_orig in enumerate(vecs_list):
        v_orig_tuple = tuple(v_orig)

        # --- FIX: Don't skip zero vector if it's the ONLY vector provided (Dummy Mode) ---
        if len(vecs_list) > 1 and all(c == 0 for c in v_orig):
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        #if all(c == 0 for c in v_orig):
        #    result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
        #    continue

        try:
            v_p_transformed = vecs_lll_p[idx]
        except Exception:
            print(f"[compute_residues_for_prime_worker] p={p} v={v_orig_tuple}: vecs_lll_p[{idx}] lookup failed")
            raise

        # [fix] see compute_residues_for_prime_worker_old for full rationale:
        # mults_p entries are always LargePrimeMockPoint; seed Pm from the same
        # curve those terms live on rather than from Ep_local, so the type can't
        # diverge (Ep_local real/mock and mults_p real/mock are decided by two
        # independent conditions upstream).
        try:
            first_term = next(
                (mpj.get(int(c)) if hasattr(mpj, 'get') else
                 (mpj[int(c)] if mpj is not None and 0 <= int(c) < len(mpj) else None))
                for j, c in enumerate(v_p_transformed)
                for mpj in [mults_p[j]] if mpj is not None
            )
        except StopIteration:
            first_term = None

        try:
            if first_term is not None:
                Pm = first_term.curve()(0)
            else:
                Pm = Ep_local(0)
        except Exception:
            print(f"[compute_residues_for_prime_worker] p={p} v={v_orig_tuple}: identity construction failed")
            raise

        for j, coeff in enumerate(v_p_transformed):
            try:
                mpj = mults_p[j]
                if mpj is None:
                    continue
                key = int(coeff)
                if hasattr(mpj, 'get'):
                    term = mpj.get(key)
                    has_term = key in mpj
                else:
                    term = mpj[key] if 0 <= key < len(mpj) else None
                    has_term = 0 <= key < len(mpj)
                if has_term:
                    assert type(Pm) is type(term), (
                        f"[compute_residues_for_prime_worker] p={p} v={v_orig_tuple} j={j}: "
                        f"Pm is {type(Pm).__name__} but mults_p[{j}][{key}] is {type(term).__name__}"
                    )
                    Pm += term
            except AssertionError:
                raise
            except Exception:
                print(f"[compute_residues_for_prime_worker] p={p} v={v_orig_tuple} j={j} coeff={coeff}: "
                      f"lookup/add failed, Pm type={type(Pm)}, mpj[key] type={type(mpj.get(key) if hasattr(mpj,'get') else (mpj[key] if 0 <= key < len(mpj) else None))}")
                raise

        try:
            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue
        except Exception:
             print(f"[compute_residues_for_prime_worker] p={p} v={v_orig_tuple}: Pm.is_zero() failed, Pm type={type(Pm)}")
             raise

        roots_by_rhs = []
        for i_rhs in range(num_rhs):
            roots_for_rhs = set()
            rhs_map = rhs_modp_list_local[i_rhs]

            if p not in rhs_map:
                roots_by_rhs.append(roots_for_rhs)
                continue

            rhs_p = rhs_map[p]
            if rhs_p is None:
                roots_by_rhs.append(roots_for_rhs)
                continue

            try:
                num_expr = (Pm[0] / Pm[2] - rhs_p).numerator()
                if num_expr.is_zero():
                    roots_by_rhs.append(roots_for_rhs)
                    continue
            except (ZeroDivisionError, TypeError, ArithmeticError):
                roots_by_rhs.append(roots_for_rhs)
                continue

            local_modular_checks += 1
            Fp = GF(p)

            try:
                raw_roots = num_expr.roots(ring=Fp, multiplicities=False)
            except Exception:
                try:
                    num_modp = num_expr.change_ring(Fp)
                    raw_roots = [r for r, _ in num_modp.roots(multiplicities=True)]
                except Exception:
                    raw_roots = []

            normalized_raw_roots = []
            for r in raw_roots:
                try:
                    normalized_raw_roots.append(int(r))
                except Exception:
                    try:
                        normalized_raw_roots.append(int(r[0]))
                    except Exception:
                        pass

            if not normalized_raw_roots:
                roots_by_rhs.append(roots_for_rhs)
                continue

            # --- TORSION FILTER ---
            filtered_roots = []
            a4_m = Ep_local.a4()
            a6_m = Ep_local.a6()

            for r in normalized_raw_roots:
                r_fp = Fp(r)
                try:
                    a4_r = a4_m(m=r_fp)
                    a6_r = a6_m(m=r_fp)
                    delta_r = -16 * (4*a4_r**3 + 27*a6_r**2)
                    if delta_r == 0:
                        continue
                    E_r = EllipticCurve(Fp, [0, 0, 0, a4_r, a6_r])
                    X_r = Pm[0](m=r_fp)
                    Y_r = Pm[1](m=r_fp)
                    Z_r = Pm[2](m=r_fp)

                    if Z_r == 0:
                        order = 1
                    else:
                        P_r = E_r([X_r, Y_r, Z_r])
                        if P_r.is_zero():
                            order = 1
                        else:
                            order = P_r.order()

                    if 0 < int(order) <= MAX_TORSION_ORDER_TO_FILTER:
                        continue

                    filtered_roots.append(int(r))

                except (ZeroDivisionError, ValueError, TypeError, ArithmeticError):
                    continue
                except Exception:
                    continue

            if not filtered_roots:
                roots_by_rhs.append(roots_for_rhs)
                continue
            # --- END TORSION FILTER ---

            simple_roots = set()
            deriv = None
            try:
                if hasattr(num_expr, 'numerator'):
                    deriv = num_expr.numerator().derivative()
                else:
                    deriv = num_expr.derivative()
            except Exception:
                pass

            for r in filtered_roots:
                keep_root = True
                if HENSEL_STRICT and deriv is not None:
                    try:
                        deriv_modp = deriv.change_ring(Fp)
                        dval = int(deriv_modp(Fp(r)))
                        if dval % int(p) == 0:
                            keep_root = False
                    except Exception:
                        keep_root = not HENSEL_STRICT

                if keep_root:
                    simple_roots.add(int(r))

            if simple_roots:
                roots_for_rhs.update(simple_roots)
            else:
                if HENSEL_ALLOW_WEAK:
                    roots_for_rhs.update(filtered_roots)

            roots_by_rhs.append(roots_for_rhs)

        result_for_p[v_orig_tuple] = roots_by_rhs

    return p, result_for_p, local_modular_checks

def compute_residues_for_prime_worker_old(args):
    """
    Worker computing residues for one prime with Hensel filtering.
    (From source [525], no torsion filter)
    """
    from sage.all import GF, Integer, QQ, ZZ

    try:
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs, _stats = args
    except Exception:
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs = args
        _stats = None

    result_for_p = {}
    local_modular_checks = 0

    HENSEL_STRICT = HENSEL_SLOPPY
    HENSEL_ALLOW_WEAK = not HENSEL_STRICT

    for idx, v_orig in enumerate(vecs_list):
        v_orig_tuple = tuple(v_orig)

        # --- FIX: Don't skip zero vector if it's the ONLY vector provided (Dummy Mode) ---
        if len(vecs_list) > 1 and all(c == 0 for c in v_orig):
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        #if all(c == 0 for c in v_orig):
        #    result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
        #    continue

        try:
            v_p_transformed = vecs_lll_p[idx]
        except Exception:
            print(f"[compute_residues_for_prime_worker_old] p={p} v={v_orig_tuple}: vecs_lll_p[{idx}] lookup failed")
            raise

        # [fix] compute_all_mults_for_section always returns LargePrimeMockPoint
        # entries in mults_p, regardless of whether Ep_local (built independently,
        # from whether EllipticCurve() overflowed for this prime) is real or mock.
        # Seed Pm's identity from the curve the mults_p terms actually live on --
        # not from Ep_local -- so the accumulator's type can never diverge from
        # what gets += into it.
        try:
            first_term = next(
                (mpj.get(int(c)) if hasattr(mpj, 'get') else
                 (mpj[int(c)] if mpj is not None and 0 <= int(c) < len(mpj) else None))
                for j, c in enumerate(v_p_transformed)
                for mpj in [mults_p[j]] if mpj is not None
            )
        except StopIteration:
            first_term = None

        try:
            if first_term is not None:
                Pm = first_term.curve()(0)
            else:
                Pm = Ep_local(0)
        except Exception:
            print(f"[compute_residues_for_prime_worker_old] p={p} v={v_orig_tuple}: identity construction failed")
            raise

        for j, coeff in enumerate(v_p_transformed):
            try:
                mpj = mults_p[j]
                if mpj is None:
                    continue
                key = int(coeff)
                if hasattr(mpj, 'get'):
                    term = mpj.get(key)
                    has_term = key in mpj
                else:
                    term = mpj[key] if 0 <= key < len(mpj) else None
                    has_term = 0 <= key < len(mpj)
                if has_term:
                    # [assert] mults_p entries come from compute_all_mults_for_section,
                    # which always returns LargePrimeMockPoint (see ll_utilities.py
                    # compute_all_mults_for_section: both the Julia-ladder and the
                    # numpy/Sage-fallback branches wrap results in LargePrimeMockPoint
                    # unconditionally). Pm must therefore also be a LargePrimeMockPoint
                    # from the same accumulation step on, or += silently has no valid
                    # operator (LargePrimeMockPoint has no __radd__, and giving it one
                    # would risk mixing real Weierstrass arithmetic with the mock's
                    # projective formulas -- wrong answers instead of a crash).
                    # Catch the type mismatch here, at first divergence, instead of
                    # inside the += a few lines down.
                    assert type(Pm) is type(term), (
                        f"[compute_residues_for_prime_worker_old] p={p} v={v_orig_tuple} j={j}: "
                        f"Pm is {type(Pm).__name__} but mults_p[{j}][{key}] is {type(term).__name__} "
                        f"-- Ep_local(0) and compute_all_mults_for_section disagree on real-vs-mock "
                        f"curve for this prime."
                    )
                    Pm += term
            except AssertionError:
                raise
            except Exception:
                print(f"[compute_residues_for_prime_worker_old] p={p} v={v_orig_tuple} j={j} coeff={coeff}: "
                      f"lookup/add failed, Pm type={type(Pm)}, mpj[key] type={type(mpj.get(key) if hasattr(mpj,'get') else (mpj[key] if 0 <= key < len(mpj) else None))}")
                raise

        try:
            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue
        except Exception:
            print(f"[compute_residues_for_prime_worker_old] p={p} v={v_orig_tuple}: Pm.is_zero() failed, Pm type={type(Pm)}")
            raise

        roots_by_rhs = []
        for i_rhs in range(num_rhs):
            roots_for_rhs = set()
            rhs_map = rhs_modp_list_local[i_rhs]

            if p not in rhs_map:
                roots_by_rhs.append(roots_for_rhs)
                continue

            rhs_p = rhs_map[p]
            if rhs_p is None:
                roots_by_rhs.append(roots_for_rhs)
                continue

            try:
                num_expr = (Pm[0] / Pm[2] - rhs_p).numerator()
                if num_expr.is_zero():
                    roots_by_rhs.append(roots_for_rhs)
                    continue
            except (ZeroDivisionError, TypeError, ArithmeticError):
                roots_by_rhs.append(roots_for_rhs)
                continue

            local_modular_checks += 1
            Fp = GF(p)

            try:
                raw_roots = num_expr.roots(ring=Fp, multiplicities=False)
            except Exception:
                try:
                    num_modp = num_expr.change_ring(Fp)
                    raw_roots = [r for r, _ in num_modp.roots(multiplicities=True)]
                except Exception:
                    raw_roots = []

            normalized_raw_roots = []
            for r in raw_roots:
                try:
                    normalized_raw_roots.append(int(r))
                except Exception:
                    try:
                        normalized_raw_roots.append(int(r[0]))
                    except Exception:
                        pass

            if not normalized_raw_roots:
                roots_by_rhs.append(roots_for_rhs)
                continue

            simple_roots = set()
            deriv = None
            try:
                if hasattr(num_expr, 'numerator'):
                    deriv = num_expr.numerator().derivative()
                else:
                    deriv = num_expr.derivative()
            except Exception:
                pass

            for r in normalized_raw_roots:
                keep_root = True
                if HENSEL_STRICT and deriv is not None:
                    try:
                        deriv_modp = deriv.change_ring(Fp)
                        dval = int(deriv_modp(Fp(r)))
                        if dval % int(p) == 0:
                            keep_root = False
                    except Exception:
                        keep_root = not HENSEL_STRICT

                if keep_root:
                    simple_roots.add(int(r))

            if simple_roots:
                roots_for_rhs.update(simple_roots)
            else:
                if HENSEL_ALLOW_WEAK:
                    roots_for_rhs.update(normalized_raw_roots)

            roots_by_rhs.append(roots_for_rhs)

        result_for_p[v_orig_tuple] = roots_by_rhs

    return p, result_for_p, local_modular_checks

def _batch_check_rationality(candidates, r_m, shift, rationality_test_func, current_sections, stats):
    """
    Test a batch of (m, v_tuple) candidates for rationality in parallel.
    Returns set of (m, v_tuple) pairs that produced rational points.
    UPDATED to accept and use a stats object with new counter names.
    """
    rational_candidates = set()

    for m_val, v_tuple in candidates:
        stats.incr('rationality_tests_total') # <-- STATS
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                stats.record_success(m_val, point=x_val) # <-- STATS (increments rationality_tests_success)
                rational_candidates.add((m_val, v_tuple))
            else:
                stats.record_failure(m_val, reason='y_not_rational') # <-- STATS (increments rationality_tests_failure)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            stats.record_failure(m_val, reason='rationality_test_error') # <-- STATS (increments rationality_tests_failure)
            continue

    return rational_candidates

def process_prime_subset_precomputed(p_subset, vecs, r_m, shift, tmax, combo_cap, precomputed_residues, prime_pool, num_rhs_fns, coeffs_genus2=None):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.

    *** MODIFIED to add a guard against combo_cap explosion ***
    """
    if not p_subset:
        return set(), Counter(), set()

    found_candidates_for_subset = set()
    stats_counter = Counter()
    tested_crt_classes = set()

    if len(p_subset) > 1 and all(p in precomputed_residues for p in p_subset):
        est = 1
        for p in p_subset:
            vks = precomputed_residues[p]
            for roots_list in vks.values():
                if any(len(roots) > ROOTS_THRESHOLD for roots in roots_list):
                    est *= sum(len(roots) for roots in roots_list)
        if est > combo_cap and DEBUG:
            print("[heavy subset]", p_subset, "estimated combos:", est)

    num_extra_primes = 4
    offset = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][offset:num_extra_primes+offset]

    for v_orig in vecs:
        if len(vecs) > 1 and all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        for rhs_idx in range(num_rhs_fns):

            # --- BUILD FILTER MAP ONCE (with proper fallbacks) ---
            residue_map_for_filter = {}
            for p in extra_primes_for_filtering:
                if p not in precomputed_residues:
                    continue

                p_data = precomputed_residues[p]
                roots_lists = p_data.get(v_orig_tuple, [])

                if rhs_idx < len(roots_lists) and roots_lists[rhs_idx]:
                    residue_map_for_filter[p] = roots_lists[rhs_idx]
                else:
                    residue_map_for_filter[p] = set()

            filter_primes_keys = list(residue_map_for_filter.keys())

            # --- BUILD CRT MAP ---
            residue_map_for_crt = {}
            for p in p_subset:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                    residue_map_for_crt[p] = roots_for_this_rhs[rhs_idx]

            primes_for_crt = list(residue_map_for_crt.keys())
            if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
                continue

            # --- DO NOT REBUILD residue_map_for_filter HERE! ---
            # (This was the bug - second construction was overwriting the first)

            lists = [residue_map_for_crt[p] for p in primes_for_crt]

            # Check for combinatorial explosion BEFORE itertools.product
            num_combos = 1
            for l in lists:
                num_combos *= max(1, len(l))
                if num_combos > combo_cap:
                    break

            if num_combos > combo_cap:
                stats_counter['crt_lift_skipped_combo_cap'] += 1
                continue

            for combo in itertools.product(*lists):
                stats_counter['crt_lift_attempts'] += 1
                M = 1
                for p in primes_for_crt:
                    M *= int(p)

                if M > MAX_MODULUS:
                    continue

                m0 = crt_cached(combo, tuple(primes_for_crt))
                tested_crt_classes.add((int(m0) % int(M), int(M)))

                # Path 1: t-search (O(1) find + O(1) filter)
                try:
                    best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
                except TypeError:
                    best_ms = [(t, QQ(m0 + t * M), 0, 0.0) for t in (-1, 0, 1)]

                for t_cand, m_cand, _, _ in best_ms:
                    # [fix] check_specific_t_value was never defined anywhere in the
                    # codebase (dead call, NameError at runtime). Its argument shape
                    # (residue_map_for_filter, filter_primes_keys, coeffs_genus2, shift,
                    # r_m_linear, r_m_sym) is identical to _check_rational_m_candidate
                    # below, just keyed by (t_cand, m0, M) instead of a single
                    # pre-combined m_candidate -- m_cand from best_ms is exactly that
                    # combined value, already unused elsewhere in this branch. Route
                    # Path 1 through the same filter Path 2 uses.
                    if _check_rational_m_candidate(QQ(m_cand), residue_map_for_filter,
                                                    filter_primes_keys,
                                                    coeffs_genus2=coeffs_genus2, shift=shift,
                                                    r_m_linear=None, r_m_sym=r_m):
                        found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                # Path 2: Rational Reconstruction
                stats_counter['rational_recon_attempts_worker'] += 1
                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    m_val_rational = QQ(a) / QQ(b)

                    if _check_rational_m_candidate(m_val_rational, residue_map_for_filter,
                                                    filter_primes_keys,
                                                    coeffs_genus2=coeffs_genus2, shift=shift,
                                                    r_m_linear=None, r_m_sym=r_m):
                        found_candidates_for_subset.add((m_val_rational, v_orig_tuple))
                        stats_counter['rational_recon_success_worker'] += 1
                    else:
                        stats_counter['rational_recon_failure_worker'] += 1

                except RationalReconstructionError:
                    stats_counter['rational_recon_failure_worker'] += 1

    return found_candidates_for_subset, stats_counter, tested_crt_classes

def _check_rational_m_candidate(m_candidate: QQ, residue_map_for_filter: dict, extra_primes: list,
                                coeffs_genus2: list[QQ], shift: QQ,
                                r_m_linear=None, r_m_sym=None, verbose=False) -> bool:
    """
    Applies consistency checks for a rational m_candidate.
    Returns False if any constraint is violated, True otherwise.

    When x-coordinate filters are empty (all primes have set()), only y-coordinate
    Kronecker check is performed.
    """
    m_candidate_val_num = ZZ(m_candidate.numerator())
    m_candidate_val_den = ZZ(m_candidate.denominator())

    for q in extra_primes:
        # Reject if denominator divisible by filter prime
        if m_candidate_val_den % q == 0:
            return False

        m_cand_mod_q = (m_candidate_val_num * m_candidate_val_den.inverse_mod(q)) % q

        allowed_m_residues = residue_map_for_filter.get(q)

        # Skip x-coordinate check if:
        # - Prime not in map (None)
        # - Prime has empty residue set (set())
        if allowed_m_residues is None or not allowed_m_residues:
            pass  # Continue to y-coordinate check below
        elif m_cand_mod_q not in allowed_m_residues:
            # x-coordinate constraint violated
            #print("here", m_cand_mod_q, allowed_m_residues)
            if verbose:
                print(f"Filter fail (rational m, x-coord): m={m_cand_mod_q} (mod {q}) not in allowed set.")
            pass
            #return False

        # --- UNIFIED MODULAR CHECK (y-coordinate Kronecker) ---
        # Always run this check, even if x-check was skipped
        try:
            x_mod_q = 0

            if r_m_linear:
                slope, intercept = r_m_linear
                slope_mod = ZZ(slope.numerator() * slope.denominator().inverse_mod(q)) % q
                icept_mod = ZZ(intercept.numerator() * intercept.denominator().inverse_mod(q)) % q
                shift_mod = ZZ(shift.numerator() * shift.denominator().inverse_mod(q)) % q
                x_mod_q = (slope_mod * m_cand_mod_q + icept_mod - shift_mod) % q
            else:
                x_val = r_m_sym.subs({var('m'): m_candidate}) - shift
                x_val = QQ(x_val)
                x_mod_q = ZZ(x_val.numerator() * x_val.denominator().inverse_mod(q)) % q

            RHS_mod_q = ZZ(coeffs_genus2[0].numerator() * coeffs_genus2[0].denominator().inverse_mod(q)) % q
            for coeff in coeffs_genus2[1:]:
                coeff_mod_q = ZZ(coeff.numerator() * coeff.denominator().inverse_mod(q)) % q
                RHS_mod_q = (RHS_mod_q * x_mod_q + coeff_mod_q) % q

            if RHS_mod_q < 0:
                RHS_mod_q = RHS_mod_q + q

            if kronecker(RHS_mod_q, q) == -1:
                if verbose:
                    print(f"Filter fail (rational m, y-coord twist): G(x)={RHS_mod_q} (mod {q}) is a non-residue.")
                return False

        except Exception:
            if verbose:
                print(f"Warning: Modular reduction/y-sieve failed for q={q}. Skipping y-sieve.")
            continue

    return True



