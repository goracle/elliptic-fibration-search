import multiprocessing, itertools
from operator import mul
from functools import reduce, partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import namedtuple, Counter
from sage.all import QQ, ZZ, GF, EllipticCurve, Integer, vector, PolynomialRing, var, matrix, identity_matrix, lcm, SR, Zmod, kronecker
from .search_config import DEBUG, MIN_PRIME_SUBSET_SIZE, MAX_MODULUS, ROOTS_THRESHOLD, MAX_K_ABS, LLL_DELTA, BKZ_BLOCK, TRUNCATE_MAX_DEG, TMAX, HENSEL_SLOPPY, TORSION_SLOPPY, MAX_TORSION_ORDER_TO_FILTER, MAX_COMBOS_PER_SUBSET
from .rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError, lattice_rational_lift_exists, modulus_is_informative
from .ll_utilities import _trim_poly_coeffs, _compute_column_norms, _scale_matrix_columns_int, _compute_integer_scales_for_columns
from .archimedean_optim import minimize_archimedean_t_linear_const

# --- y-coordinate ("rail_ok") filter safety ---------------------------------
# The Kronecker/QR filters below evaluate the ORIGINAL curve RHS at
# r_m(m) - shift.  That is only the right x when no Mobius transform T is
# active (rationality_test_func applies T_inv otherwise), so refuse to run
# them if MOBIUS_TRANS is on -- or if we cannot tell.
try:
    from .search_config import MOBIUS_TRANS as _MOBIUS_TRANS
except Exception:
    try:
        from search_common import MOBIUS_TRANS as _MOBIUS_TRANS
    except Exception:
        _MOBIUS_TRANS = None
_RAIL_Y_FILTER_OK = (_MOBIUS_TRANS is False)
RAIL_Y_FILTER = True   # master switch for the y-coordinate (rail_ok) filters
_RAIL_WARNED = set()


def _rail_warn_once(key, msg):
    """Print a filter problem once per process so it can never fail silently."""
    if key not in _RAIL_WARNED:
        _RAIL_WARNED.add(key)
        print(f"[rail_ok] WARNING: {msg}")


if not _RAIL_Y_FILTER_OK:
    _rail_warn_once(
        "disabled",
        f"y-coordinate Kronecker filters DISABLED (MOBIUS_TRANS={_MOBIUS_TRANS!r}; "
        "they assume untransformed x).")

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

def _kronecker_prefilter_domain(p, residues_p, coeffs_genus2, shift, r_m_linear, r_m_sym, stats_counter=None):
    """
    Single-prime y-coordinate (Kronecker/QR) prefilter, applied directly to a
    prime's own residue domain before any CRT happens at all.

    This is the cheapest possible check in the pipeline: for m = a (mod p),
    the induced x = r_m(a) - shift (mod p) and thus G(x) (mod p) -- where
    G is the curve's RHS polynomial -- depend only on `a` and `p`, not on
    any other prime or on a fully-reconstructed rational m. So instead of
    waiting until a candidate is fully CRT'd (across the whole subset) and
    rationally reconstructed before ever checking whether G(x) is even a
    quadratic residue mod p, we can kill non-residue `a` values right here,
    one prime at a time, before they ever reach itertools.product.

    This duplicates the y-coordinate half of _check_rational_m_candidate's
    per-extra-prime loop, but keyed by residue instead of by candidate: it's
    the same math (Kronecker symbol of G(x) mod q), just run mod p itself
    against p's own domain rather than mod each extra prime against a
    finished candidate. The x-coordinate half of that check needs a real
    filter-prime residue *set* to compare against and isn't meaningful here
    (there's nothing to compare `a` against except itself), so only the
    y-coordinate/Kronecker half applies at this stage.

    If stats_counter is given, every residue that hits the except-and-keep
    path increments 'arc_consistency_kronecker_exceptions' -- if that number
    equals len(residues_p) summed across a run, this prefilter is silently
    keeping everything (as opposed to genuinely finding every residue a
    quadratic residue, which the exception path is NOT testing for).

    Returns the subset of residues_p that pass (same type as residues_p).
    """
    if p == 2 or not _RAIL_Y_FILTER_OK or not RAIL_Y_FILTER:
        # Every class mod 2 is a square, and kronecker(a, 2) is NOT a QR test.
        return set(residues_p)
    survivors = set()
    for a in residues_p:
        try:
            a_mod_p = int(a) % p
            if r_m_linear:
                slope, intercept = r_m_linear
                slope_mod = ZZ(slope.numerator() * slope.denominator().inverse_mod(p)) % p
                icept_mod = ZZ(intercept.numerator() * intercept.denominator().inverse_mod(p)) % p
                shift_mod = ZZ(shift.numerator() * shift.denominator().inverse_mod(p)) % p
                x_mod_p = (slope_mod * a_mod_p + icept_mod - shift_mod) % p
            else:
                # NOTE: must substitute a genuine QQ value here, not a bare
                # Python int. _check_rational_m_candidate's identical branch
                # (the only other caller of this .subs(...) pattern) always
                # substitutes a QQ m_candidate and that's known to work --
                # substituting plain `a_mod_p` (a Python int) here silently
                # produced an SR expression that QQ(...) couldn't coerce
                # cleanly, so every residue fell through to `except: pass`
                # and this whole prefilter was a no-op (0 pruned every run).
                x_val = r_m_sym.subs({var('m'): QQ(a_mod_p)}) - shift
                x_val = QQ(x_val)
                x_mod_p = ZZ(x_val.numerator() * x_val.denominator().inverse_mod(p)) % p

            RHS_mod_p = ZZ(coeffs_genus2[0].numerator() * coeffs_genus2[0].denominator().inverse_mod(p)) % p
            for coeff in coeffs_genus2[1:]:
                coeff_mod_p = ZZ(coeff.numerator() * coeff.denominator().inverse_mod(p)) % p
                RHS_mod_p = (RHS_mod_p * x_mod_p + coeff_mod_p) % p

            if RHS_mod_p < 0:
                RHS_mod_p = RHS_mod_p + p

            if kronecker(RHS_mod_p, p) == -1:
                continue  # a is killed: G(x) is a non-residue mod p
        except (ZeroDivisionError, ArithmeticError, ValueError):
            # Expected: denominator divisible by p, so the residue can't be
            # reduced -- keep it and let downstream checks handle it.
            if stats_counter is not None:
                stats_counter['arc_consistency_kronecker_exceptions'] += 1
        except Exception as e:
            # UNEXPECTED (e.g. NameError, coercion TypeError).  Previously this
            # was swallowed, which made the whole prefilter a silent no-op
            # (kronecker was never imported).  Still fail open -- never drop a
            # residue on an error -- but make it visible.
            if stats_counter is not None:
                stats_counter['arc_consistency_kronecker_unexpected'] += 1
            _rail_warn_once(("prefilter", type(e).__name__),
                            f"Kronecker prefilter raised {type(e).__name__}: {e} "
                            f"(p={p}); keeping residue, filter not effective.")
        survivors.add(a)
    return survivors


def filter_residues_by_rail(precomputed_residues, coeffs_genus2, shift, r_m,
                            stats_counter=None, verbose=True):
    """
    Node-level "rail_ok" filter for the residue CRT graph.

    Drops every residue a (m = a mod p) for which the induced x = r_m(a) - shift
    gives G(x) a quadratic NON-residue mod p: a rational point needs
    y^2 = G(x) to be solvable mod every good p, so such a residue can never be
    the reduction of a genuine point.  Real points always pass; a random
    residue survives about half the time.

    The test depends only on (p, a), so it is computed once per prime over the
    union of that prime's residues, then applied to every vector's residue
    lists.  Structure and container types of precomputed_residues are kept.
    Returns a NEW dict; the input is not modified.  Fails open: any residue
    that can't be evaluated is kept (see _kronecker_prefilter_domain).
    """
    if not RAIL_Y_FILTER or not _RAIL_Y_FILTER_OK or coeffs_genus2 is None:
        return precomputed_residues
    out, before, after = {}, 0, 0
    for p, mapping in precomputed_residues.items():
        if not mapping:
            out[p] = mapping
            continue
        union = set()
        for rhs_lists in mapping.values():
            for rl in rhs_lists:
                union.update(rl)
        keep = _kronecker_prefilter_domain(p, union, coeffs_genus2, shift, None, r_m,
                                           stats_counter=stats_counter)
        new_map = {}
        for v, rhs_lists in mapping.items():
            new_lists = []
            for rl in rhs_lists:
                kept = [a for a in rl if a in keep]
                before += len(rl)
                after += len(kept)
                new_lists.append(set(kept) if isinstance(rl, (set, frozenset)) else kept)
            new_map[v] = new_lists
        out[p] = new_map
    if verbose:
        pct = (100.0 * after / before) if before else 100.0
        print(f"[rail_ok] graph residues: {before:,} -> {after:,} ({pct:.1f}% kept)")
        if before >= 200 and after == before:
            _rail_warn_once("noop", "filter removed nothing across a large residue set; "
                                    "expected ~50% (check for swallowed errors above).")
    return out


def _pairwise_crt_survivors(anchor_modulus, residues_anchor, q, residues_q, height_bound, stats_counter=None):
    """
    Arc-consistency-style pairwise prune: for each residue class `a` mod
    anchor_modulus, check whether ANY residue `b` in D_q CRTs (mod
    anchor_modulus*q) to a class that could contain a small-height rational
    -- i.e. whether the CRT class admits a lattice point (r, s) with
    |r| <= height_bound AND |s| <= height_bound (the CRT-then-lattice-
    reduction test: see lattice_rational_lift_exists). If no b works for a
    given a, that a cannot participate in any global small-height solution
    and is dropped.

    NOTE on anchor_modulus: this is no longer required to be a single
    prime -- the caller (process_prime_subset_precomputed) chains several
    partner primes into a running product, CRT-ing them into the anchor's
    modulus one at a time, precisely so the modulus can grow past the
    "uninformative" regime a single prime pair sits in (see
    modulus_is_informative below). anchor_modulus and q just need to be
    coprime, which holds automatically since primes_for_crt only ever
    contains distinct primes from the pool. residues_anchor is therefore a
    set of residues mod anchor_modulus (a genuine composite modulus after
    the first fold), not mod a single prime -- crt_cached and the `% `
    reductions below work identically either way.

    *** WHY THIS REPLACED THE OLD rational_reconstruct(max_den=height_bound)
    CALL ***

    That version only bounded the reconstructed DENOMINATOR against
    height_bound and let the numerator be anything under the modulus M.
    For a two-small-prime modulus (M = p*q, typically a few thousand, vs.
    height_bound in the tens of thousands), M < height_bound essentially
    always -- so a small-denominator representative exists for nearly
    every residue trivially (there just aren't very many classes mod M to
    begin with, and height_bound was bigger than M itself). Measured
    directly: for M = 97*89 = 8633 and height_bound = 37000, ALL 8633
    residues reconstructed "successfully", which is exactly why
    arc_consistency_pairwise_pruned sat at 0 -- the test had no power.

    The fix is three-fold:
      1. Bound BOTH r and s (the fraction's numerator and denominator), not
         just s -- lattice_rational_lift_exists walks every convergent of
         the continued-fraction chain and requires |r| <= H and |s| <= H
         simultaneously.
      2. Only trust the test once the modulus M is actually large enough
         relative to height_bound that "no small lift exists" is a real
         statement and not just pigeonhole overflow (see
         modulus_is_informative).
      3. Chain multiple partner primes into a growing anchor_modulus
         (handled by the caller) instead of testing pairs at a permanently
         small fixed modulus, so informativeness is actually reachable.

    Returns the subset of residues_anchor that have at least one compatible
    b in residues_q.
    """
    anchor_modulus = int(anchor_modulus)
    q = int(q)
    M = anchor_modulus * q
    informative = modulus_is_informative(M, height_bound)
    if stats_counter is not None and not informative:
        stats_counter['arc_consistency_pairwise_uninformative_modulus'] += 1

    survivors = set()
    for a in residues_anchor:
        a_int = int(a) % anchor_modulus
        found_partner = False
        for b in residues_q:
            b_int = int(b) % q
            c = crt_cached((a_int, b_int), (anchor_modulus, q))
            if lattice_rational_lift_exists(int(c) % M, M, int(height_bound)):
                found_partner = True
                break
        if found_partner:
            survivors.add(a)
    return survivors


def process_prime_subset_precomputed(p_subset, vecs, r_m, shift, tmax, combo_cap, precomputed_residues, prime_pool, num_rhs_fns, coeffs_genus2=None, height_bound=None, bad_primes=frozenset()):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.

    *** height_bound may now be EITHER a flat int/float (old behavior,
    same bound for every vector) OR a dict {v_orig_tuple: bound} produced
    by height_bound.build_vector_height_bounds -- a real, per-vector bound
    derived from the Mordell-Weil canonical height pairing rather than the
    old flat HEIGHT_BOUND=37000 constant. See search_lll/height_bound.py
    for the derivation and the validation step that MUST be run before
    trusting a dict here to reject anything (i.e. before this actually
    prunes in Stage 2 below). Passing the flat scalar still works
    unchanged for anyone not ready to switch over. ***

    *** MODIFIED to add a guard against combo_cap explosion ***

    NOTE on `combo_cap` (the parameter): kept for backward compatibility with
    callers (search_main.py passes its own locally-computed combo_cap here),
    but it is NOT what actually bounds combinatorial work inside this
    function anymore -- that's MAX_COMBOS_PER_SUBSET (search_config.py),
    which is sized against real iteration cost rather than against "won't
    overflow / won't spuriously reject a subset". See the comment at
    MAX_COMBOS_PER_SUBSET's definition and the guard below for why the
    caller's combo_cap alone wasn't catching this.

    *** MODIFIED to prune residue domains before the full CRT product ***

    Before building the n-prime itertools.product over all of p_subset,
    each prime's residue domain is pruned in two cheap passes:
      1. A per-prime Kronecker/QR prefilter (_kronecker_prefilter_domain) --
         kills residues that can never yield a rational point regardless of
         any other prime, using only that one prime.
      2. A pairwise CRT/height-compatibility prefilter
         (_pairwise_crt_survivors), against one anchor prime from the
         subset -- kills residues that have no partner residue anywhere in
         the anchor's domain compatible with a small-height rational lift.
    Only the survivors feed the existing n-prime CRT search below. This is
    the "arc consistency" sieve: cheap pairwise/single-prime tests remove
    the vast majority of dead residues before they reach the expensive
    full-subset combinatorial stage, rather than after (which is what the
    existing extra-primes / Kronecker filter in _check_rational_m_candidate
    was doing -- correct, but only after the expensive work was already
    done).
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

    # --- SELECT EXTRA (FILTERING) PRIMES ---
    #
    # These are primes NOT in p_subset itself, used purely as independent
    # cross-checks (y-coordinate Kronecker/QR test; x-coordinate check is
    # currently a no-op, see _check_rational_m_candidate below) that a
    # candidate m must pass before being accepted.
    #
    # *** TUNING: using more extra primes is cheap (each is just one more
    # Kronecker-symbol computation per candidate), so it's tempting to just
    # grab more/larger ones. The earlier attempt at this (sorting purely by
    # prime size, including primes up to and past 97) was reverted because
    # it cost real rational points: a prime that's simply LARGE is not
    # necessarily a GOOD prime for this curve -- if `q` divides the
    # discriminant, makes a4/a6's denominator vanish, or otherwise gives a
    # degenerate reduction, the Kronecker/QR test at q is not a valid
    # filter at all, and _check_rational_m_candidate's hard `return False`
    # on a failed y-coordinate check will silently reject true points along
    # with false ones.
    #
    # The actual fix: filter candidate_extra_primes down to primes already
    # known-good for this surface (bad_primes, computed once at curve-build
    # time via is_good_prime_for_surface -- see search_common.py) BEFORE
    # doing anything else, then take as many of the largest good primes as
    # we want (this is where "more filtering is cheap" is true and safe to
    # act on, since every one of them is a prime the reduction is actually
    # valid at). If bad_primes wasn't supplied (empty default), this
    # degrades to the previous behavior of not excluding anything by
    # goodness -- pass bad_primes from the caller to get the safety benefit.
    good_candidate_extra_primes = [p for p in prime_pool if p not in p_subset and p not in bad_primes]
    num_extra_primes = 8 if bad_primes else 4
    extra_primes_for_filtering = sorted(good_candidate_extra_primes, reverse=True)[:num_extra_primes]

    for v_orig in vecs:
        if len(vecs) > 1 and all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        # Resolve this vector's height bound. If height_bound is a dict
        # (the new per-vector bounds from height_bound.py), look up this
        # v_orig_tuple specifically -- a vector missing from the dict is
        # treated as "no bound available" (None) rather than silently
        # falling back to some other vector's bound, since bounds are not
        # interchangeable across vectors (that's the entire point of
        # making this per-vector in the first place).
        if isinstance(height_bound, dict):
            vector_height_bound = height_bound.get(v_orig_tuple)
        else:
            vector_height_bound = height_bound

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

            # --- DO NOT REBUILD residue_map_for_filter HERE! ---
            # (This was the bug - second construction was overwriting the first)

            # --- BUILD CRT MAP ---
            # primes_for_crt must stay a concrete tuple: it's re-iterated once
            # per combo below (crt_cached needs the moduli tuple every time)
            # and once up front for M and the combo-count guard, so there's
            # nothing to gain by deferring it -- but residue_map_for_crt's
            # values feed itertools.product as a generator (no `lists` copy),
            # and residue_map_for_filter's keys are passed as a plain dict
            # keys view (`.keys()`) instead of a list, since both are only
            # ever walked once per use, in order.
            residue_map_for_crt = {}
            for p in p_subset:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                    residue_map_for_crt[p] = roots_for_this_rhs[rhs_idx]

            primes_for_crt = tuple(residue_map_for_crt.keys())
            if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
                continue

            # --- ARC-CONSISTENCY SIEVE (before the full CRT product) ---
            # Stage 1: per-prime Kronecker/QR prefilter. Cheap (no CRT), and
            # applies independently to every prime's domain.
            r_m_linear = None  # matches _check_rational_m_candidate's own default; r_m below is passed as r_m_sym
            if coeffs_genus2 is not None:
                for p in primes_for_crt:
                    before = residue_map_for_crt[p]
                    if not before:
                        continue
                    after = _kronecker_prefilter_domain(
                        p, before, coeffs_genus2, shift, r_m_linear, r_m, stats_counter=stats_counter
                    )
                    stats_counter['arc_consistency_kronecker_pruned'] += (len(before) - len(after))
                    residue_map_for_crt[p] = after

            # Stage 2: CRT-then-lattice-reduction prefilter against an
            # anchor prime.
            #
            # *** RE-ENABLED, against vector_height_bound instead of the
            # old flat height_bound constant. WHY IT WAS DISABLED BEFORE:
            # this test enforces |r| <= H AND |s| <= H on the CRT-lifted
            # class, but the pipeline's downstream acceptance path
            # (rational_reconstruct(m0 % M, M) further below) previously
            # called with NO max_den argument -- defaulting to sqrt(M/2)
            # instead of the same H -- so a genuine point could reconstruct
            # successfully downstream at a size this sieve didn't allow,
            # and get thrown away here first. That is fixed by (a) using
            # vector_height_bound here, derived per-vector from the
            # Mordell-Weil canonical height pairing (see
            # search_lll/height_bound.py) rather than a flat guessed
            # constant, and (b) passing that SAME value as max_den to the
            # rational_reconstruct call in the acceptance path below, so
            # both stages test the identical bound instead of two
            # different ones.
            #
            # *** THIS MUST NOT BE TRUSTED UNTIL VALIDATED: before relying
            # on this to reject anything in a real search run, run
            # height_bound.validate_against_known_points against every
            # known rational point in your test curves and confirm none
            # of them exceed their vector's computed bound. A wrong
            # (too-tight) bound here silently drops real points exactly
            # like the old flat-constant bug did, just via a new route --
            # see height_bound.py's module docstring for the full caveat,
            # including the m_map_height_factor gap (the bound derived
            # there is proven for x([v]P)'s numerator/denominator, not yet
            # for m itself, unless r_m/shift/T's height distortion has
            # been folded in and validated). ***
            if vector_height_bound is not None and len(primes_for_crt) > 1:
                anchor = min(primes_for_crt, key=lambda p: len(residue_map_for_crt[p]))
                anchor_domain = residue_map_for_crt[anchor]
                partners_sorted = sorted(
                    (p for p in primes_for_crt if p != anchor and residue_map_for_crt[p]),
                    key=lambda p: len(residue_map_for_crt[p]),
                )

                if anchor_domain and partners_sorted:
                    # Grow the partner group until anchor*partner_modulus
                    # clears the informative threshold (or we run out of
                    # partners).
                    partner_group = []
                    partner_modulus = 1
                    for p in partners_sorted:
                        partner_group.append(p)
                        partner_modulus *= int(p)
                        if modulus_is_informative(int(anchor) * partner_modulus, vector_height_bound):
                            break

                    combined_modulus = int(anchor) * partner_modulus
                    partner_combo_count = 1
                    for p in partner_group:
                        partner_combo_count *= max(1, len(residue_map_for_crt[p]))

                    if modulus_is_informative(combined_modulus, vector_height_bound) and partner_combo_count <= MAX_COMBOS_PER_SUBSET:
                        # Build the joint CRT domain over the partner group:
                        # every tuple of partner residues, combined via CRT
                        # into a single residue mod partner_modulus. This is
                        # the same O(prod |D_q|) cost the final n-prime
                        # product would pay for just this sub-group, but
                        # sub-groups here are the 1-2 smallest-domain
                        # partners, so it stays cheap, and it only runs once
                        # per anchor instead of once per full subset combo.
                        # Guarded against MAX_COMBOS_PER_SUBSET the same way
                        # the main product loop is, in case the smallest
                        # partner domains still happen to be large.
                        partner_domains = [residue_map_for_crt[p] for p in partner_group]
                        partner_moduli = tuple(int(p) for p in partner_group)
                        joint_partner_residues = set()
                        for combo in itertools.product(*partner_domains):
                            combo_ints = tuple(int(b) % pm for b, pm in zip(combo, partner_moduli))
                            joint_partner_residues.add(crt_cached(combo_ints, partner_moduli))

                        before = anchor_domain
                        after = _pairwise_crt_survivors(
                            int(anchor), before, partner_modulus, joint_partner_residues,
                            vector_height_bound, stats_counter=stats_counter,
                        )
                        stats_counter['arc_consistency_pairwise_pruned'] += (len(before) - len(after))
                        stats_counter['arc_consistency_pairwise_informative_hits'] += 1
                        residue_map_for_crt[anchor] = after
                    else:
                        # Either the combined modulus never cleared the
                        # informative threshold even using every partner
                        # prime in the subset (Stage 2 has no reliable
                        # signal here), or the joint partner-residue combo
                        # count was too large to build cheaply -- either
                        # way, leave the anchor's domain untouched rather
                        # than act on a pigeonhole-driven false pass or pay
                        # for an expensive joint CRT build.
                        stats_counter['arc_consistency_pairwise_uninformative_skip'] += 1

            # Re-check subset size and combo count now that domains have
            # shrunk -- a subset that looked viable before pruning may not
            # be worth (or even eligible for) the full product anymore.
            if any(not residue_map_for_crt[p] for p in primes_for_crt):
                stats_counter['arc_consistency_subset_emptied'] += 1
                continue

            # --- Combinatorial explosion guard ---
            # This is a COUNT of root-combinations (roughly
            # avg_roots**len(primes_for_crt)). The `combo_cap` parameter this
            # function receives is an upstream estimate-vs-cap check tuned to
            # never spuriously reject a subset (at subset size 65+ it clamps
            # to 50000**40 ~= 10**188 -- see search_config.py's
            # MAX_COMBOS_PER_SUBSET comment for the full explanation) -- it
            # passes through real per-subset combo counts of ~10**3-10**5
            # without complaint, which is exactly the case that was making
            # every subset here slow. Compare against MAX_COMBOS_PER_SUBSET
            # instead, which is sized against actual iteration cost, not
            # against "won't overflow".
            #
            # No intermediate `lists = [residue_map_for_crt[p] for p in ...]`
            # here -- iterate primes_for_crt directly and look each set up on
            # the fly; the sets already live in residue_map_for_crt, so that
            # list was just a second, throwaway container of the same values.
            # (Domains here are the arc-consistency survivors, not the raw
            # precomputed residues -- this count is now post-pruning.)
            num_combos = 1
            for p in primes_for_crt:
                num_combos *= max(1, len(residue_map_for_crt[p]))
                if num_combos > MAX_COMBOS_PER_SUBSET:
                    break

            if num_combos > MAX_COMBOS_PER_SUBSET:
                stats_counter['crt_lift_skipped_combo_cap'] += 1
                continue

            # Precompute M once per (vector, rhs) -- it's the same for every
            # combo in this inner loop (primes_for_crt doesn't change), so
            # recomputing it inside the loop below was pure waste.
            M = 1
            for p in primes_for_crt:
                M *= int(p)
            if M > MAX_MODULUS:
                stats_counter['crt_lift_skipped_modulus_cap'] += 1
                continue

            combos_processed_this_group = 0
            for combo in itertools.product(*(residue_map_for_crt[p] for p in primes_for_crt)):
                # --- Hard runtime bailout ---
                # The up-front num_combos estimate above is exact for this
                # loop shape (it's the same product-of-lengths itertools.product
                # will emit), so in practice this should never trigger once the
                # up-front check passed. It's kept as a defense-in-depth cutoff
                # (not just an estimate) in case residue_map_for_crt is ever
                # mutated, contains duplicate-but-distinct entries, or this
                # function's call sites change -- we never again want "the
                # loop just keeps going" to be possible regardless of what fed it.
                if combos_processed_this_group >= MAX_COMBOS_PER_SUBSET:
                    stats_counter['crt_lift_truncated_mid_loop'] += 1
                    break
                combos_processed_this_group += 1

                stats_counter['crt_lift_attempts'] += 1

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
                    # (residue_map_for_filter, residue_map_for_filter.keys(), coeffs_genus2, shift,
                    # r_m_linear, r_m_sym) is identical to _check_rational_m_candidate
                    # below, just keyed by (t_cand, m0, M) instead of a single
                    # pre-combined m_candidate -- m_cand from best_ms is exactly that
                    # combined value, already unused elsewhere in this branch. Route
                    # Path 1 through the same filter Path 2 uses.
                    if _check_rational_m_candidate(QQ(m_cand), residue_map_for_filter,
                                                    residue_map_for_filter.keys(),
                                                    coeffs_genus2=coeffs_genus2, shift=shift,
                                                    r_m_linear=None, r_m_sym=r_m,
                                                    stats_counter=stats_counter):
                        found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                # Path 2: Rational Reconstruction
                #
                # *** max_den is now vector_height_bound when available,
                # instead of always falling back to rational_reconstruct's
                # default (floor(sqrt(M/2))). This is the other half of
                # the Stage-2 reconciliation above: both the prefilter and
                # this acceptance step now test the SAME bound for this
                # v_orig, so Stage 2 can only ever reject a candidate that
                # this step would also reject. When vector_height_bound is
                # None (old flat-height_bound callers, or a vector missing
                # from a per-vector dict), this keeps the previous
                # unbounded-default behavior exactly as before. ***
                stats_counter['rational_recon_attempts_worker'] += 1
                try:
                    if vector_height_bound is not None:
                        a, b = rational_reconstruct(m0 % M, M, max_den=int(vector_height_bound))
                    else:
                        a, b = rational_reconstruct(m0 % M, M)
                    m_val_rational = QQ(a) / QQ(b)

                    if _check_rational_m_candidate(m_val_rational, residue_map_for_filter,
                                                    residue_map_for_filter.keys(),
                                                    coeffs_genus2=coeffs_genus2, shift=shift,
                                                    r_m_linear=None, r_m_sym=r_m,
                                                    stats_counter=stats_counter):
                        found_candidates_for_subset.add((m_val_rational, v_orig_tuple))
                        stats_counter['rational_recon_success_worker'] += 1
                    else:
                        stats_counter['rational_recon_failure_worker'] += 1

                except RationalReconstructionError:
                    stats_counter['rational_recon_failure_worker'] += 1

    return found_candidates_for_subset, stats_counter, tested_crt_classes

def _check_rational_m_candidate(m_candidate: QQ, residue_map_for_filter: dict, extra_primes: list,
                                coeffs_genus2: list[QQ], shift: QQ,
                                r_m_linear=None, r_m_sym=None, verbose=False,
                                stats_counter=None) -> bool:
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
            # x-coordinate constraint violated.
            #
            # *** REVERTED: this was briefly changed to `return False` on
            # the theory that the commented-out reject was simply a bug.
            # It was reverted back to `pass` because doing so was found to
            # discard genuine rational points, not just bad candidates --
            # residue_map_for_filter[q] is built from ONE specific
            # (v_orig_tuple, rhs_idx) branch (see the "BUILD FILTER MAP
            # ONCE" block above in process_prime_subset_precomputed), so a
            # true point's m need only satisfy this residue constraint for
            # the branch it actually came from, not for every extra prime
            # q's precomputed roots under that same branch label -- q's
            # roots can be incomplete, keyed to a different RHS choice, or
            # otherwise not a valid constraint on m in general. Treating a
            # mismatch here as a hard rejection was cutting real points,
            # not just false ones (we were losing points, not residues).
            # Left as a no-op/diagnostic print, exactly as the original
            # code (with its "pass  #return False" comment) had it -- only
            # the y-coordinate Kronecker check below is trusted enough to
            # actually reject a candidate. ***
            if verbose:
                print(f"Filter fail (rational m, x-coord): m={m_cand_mod_q} (mod {q}) not in allowed set.")
            pass

        # --- UNIFIED MODULAR CHECK (y-coordinate Kronecker) ---
        # Always run this check, even if x-check was skipped
        if q == 2 or not _RAIL_Y_FILTER_OK or not RAIL_Y_FILTER:
            continue  # see _kronecker_prefilter_domain
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
                if stats_counter is not None:
                    stats_counter['extra_prime_y_coord_rejects'] += 1
                return False

        except (ZeroDivisionError, ArithmeticError, ValueError):
            if verbose:
                print(f"Warning: Modular reduction/y-sieve failed for q={q}. Skipping y-sieve.")
            continue
        except Exception as e:
            _rail_warn_once(("candidate", type(e).__name__),
                            f"y-sieve raised {type(e).__name__}: {e} (q={q}); skipping, filter not effective.")
            continue

    return True



