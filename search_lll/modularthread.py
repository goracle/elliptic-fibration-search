"""
modular_workers.py: Parallel worker functions and modular reduction setup.
"""

# Standard library and external imports
import multiprocessing
import itertools
from operator import mul
from functools import reduce, partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# SageMath imports
from sage.all import (
    QQ, ZZ, GF, EllipticCurve, Integer, vector, PolynomialRing, var,
    matrix, identity_matrix, lcm, SR, Zmod
)

# Local Configuration Imports
from .search_config import (
    DEBUG, MIN_PRIME_SUBSET_SIZE, MAX_MODULUS, ROOTS_THRESHOLD,
    MAX_K_ABS, LLL_DELTA, BKZ_BLOCK, TRUNCATE_MAX_DEG, TMAX,
    HENSEL_SLOPPY, TORSION_SLOPPY, MAX_TORSION_ORDER_TO_FILTER
)

from .rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .ll_utilities import _trim_poly_coeffs, _compute_column_norms, _scale_matrix_columns_int, _compute_integer_scales_for_columns
from .archimedean_optim import minimize_archimedean_t_linear_const

# Assuming diagnostics2.py is in the parent directory or installed
try:
    from diagnostics2 import find_singular_fibers, compute_ramification_locus
except ImportError:
    print("Warning: diagnostics2.py not found. Fiber collision checks will be limited.")
    def find_singular_fibers(*args, **kwargs): return {}
    def compute_ramification_locus(*args, **kwargs): return set()


# ==============================================================
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

def compute_all_mults_for_section(Pi, required_ks, stats,
                                  max_k=None, debug=False):
    """
    Compute specific multiples {k: k*Pi} for a reduced section Pi.
    Handles None input for Pi (returns None).
    """
    if Pi is None:
        return None

    from sage.all import ZZ
    
    if max_k is None:
        try:
            max_k = max(abs(k) for k in required_ks)
        except ValueError:
            max_k = MAX_K_ABS # fallback
    
    max_k = min(int(max_k), MAX_K_ABS)
    
    computed = {}
    try:
        identity = Pi.curve()(0)
        computed[0] = identity
    except Exception:
        computed[0] = None
         
    computed[1] = Pi
    
    for k_abs in range(2, max_k + 1):
        if k_abs in computed:
            continue
        try:
            computed[k_abs] = k_abs * Pi
            if stats: stats.incr('modular_mults')
        except Exception:
            if debug:
                print(f"    [mults] k*Pi failed at k={k_abs}")
            break 
         
    final_mults = {}
    for k in required_ks:
        k_abs = abs(int(k))
        if k_abs not in computed:
            continue 
        
        k_val = computed[k_abs]
        if k_val is None:
            continue

        if k < 0:
            final_mults[k] = -k_val
        else:
            final_mults[k] = k_val
            
    return final_mults


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

def detect_fiber_collision(Delta_poly, p, debug=DEBUG):
    """
    Detect if discriminant Delta(m) has repeated roots mod p.
    Returns (has_collision, gcd_poly).
    """
    from sage.all import GF, PolynomialRing, gcd
    
    try:
        Fp = GF(p)
        R = PolynomialRing(Fp, 'm')
        
        Delta_modp = R([int(c) % p for c in Delta_poly.list()])
        dDelta = Delta_modp.derivative()
        
        g = gcd(Delta_modp, dDelta)
        has_collision = (g.degree() > 1)
        
        if has_collision and debug:
            print(f"⚠️  Fiber collision detected at p={p}: gcd degree {g.degree()}")
        
        return has_collision, g
        
    except Exception as e:
        if debug:
            print(f"[detect_fiber_collision] p={p}: error {e}")
        return False, None


# ==============================================================
# === Preparation and LLL Reduction (Modular) ==================
# ==============================================================

def prepare_modular_data_lll(cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=None):
    """
    Prepare modular data for LLL-based search across multiple primes.
    NOW RECORDS REJECTED PRIMES IN STATS FOR POSTERIOR ADJUSTMENT.
    """
    if search_primes is None:
        search_primes = prime_pool

    r = len(current_sections)
    if r == 0:
        return {}, [], {}, {}

    Ep_dict, rhs_modp_list = {}, [{} for _ in rhs_list]
    multiplies_lll, vecs_lll = {}, {}
    rejected_primes = []  
    
    PR_m = PolynomialRing(QQ, 'm')
    var_sym = var('m')

    processed_rhs_list = [{'num': PR_m(rhs.numerator()), 'den': PR_m(rhs.denominator())} for rhs in rhs_list]
    a4_num, a4_den = PR_m(cd.a4.numerator()), PR_m(cd.a4.denominator())
    a6_num, a6_den = PR_m(cd.a6.numerator()), PR_m(cd.a6.denominator())

    for p in search_primes:
        try:
            if any(int(QQ(c).denominator()) % p == 0 for c in a4_num.coefficients(sparse=False)):
                rejected_primes.append((p, "a4_denom_divisible"))
                continue
            if any(int(QQ(c).denominator()) % p == 0 for c in a6_num.coefficients(sparse=False)):
                rejected_primes.append((p, "a6_denom_divisible"))
                continue

            Rp = PolynomialRing(GF(p), 'm')
            Fp_m = Rp.fraction_field()

            try:
                if a4_den.change_ring(GF(p)).is_zero() or a6_den.change_ring(GF(p)).is_zero():
                    rejected_primes.append((p, "a4_a6_denom_zero"))
                    continue
            except Exception:
                rejected_primes.append((p, "denom_coercion_failed"))
                continue

            a4_modp = Fp_m(a4_num) / Fp_m(a4_den)
            a6_modp = Fp_m(a6_num) / Fp_m(a6_den)

            if p == 3 and DEBUG:
                print("\n" + "="*70)
                print(f"--- RUNNING MOD-{p} GEOMETRIC ANALYSIS (from diagnostics2.py) ---")
                try:
                    mod_p_fiber_report = find_singular_fibers(a4=a4_modp, a6=a6_modp, verbose=True)
                    print(f"--- MOD-{p} ANALYSIS COMPLETE ---")
                except Exception as e_diag:
                    print(f"--- MOD-{p} ANALYSIS FAILED: {e_diag} ---")
                print("="*70 + "\n")

            try:
                disc_modp = -16 * (4 * a4_modp**3 + 27 * a6_modp**2)
                if disc_modp.is_zero():
                    rejected_primes.append((p, "singular_discriminant"))
                    continue
            except Exception:
                rejected_primes.append((p, "discriminant_check_failed"))
                continue

            try:
                Delta_poly = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
                if hasattr(Delta_poly, 'numerator'):
                    Delta_poly = Delta_poly.numerator()
                Delta_pr = PR_m(SR(Delta_poly))
                
                has_collision, gcd_poly = detect_fiber_collision(Delta_pr, p, debug=DEBUG)
                
                if has_collision:
                    deg = gcd_poly.degree() if gcd_poly is not None else "N/A"
                    rejected_primes.append((p, f"fiber_collision_deg_{deg}"))
                    continue
            except Exception as e:
                if DEBUG:
                    print(f"[fiber_collision_check] p={p}: error {e} -- continuing cautiously")

            try:
                Ep_local = EllipticCurve(Fp_m, [0, 0, 0, a4_modp, a6_modp])
            except ArithmeticError as e:
                rejected_primes.append((p, "elliptic_curve_construction_failed"))
                continue

            rhs_modp_for_p = {}
            for i, rhs_data in enumerate(processed_rhs_list):
                try:
                    if rhs_data['den'].change_ring(GF(p)).is_zero():
                        continue
                    rhs_modp_for_p[i] = Fp_m(rhs_data['num']) / Fp_m(rhs_data['den'])
                except Exception:
                    pass

            new_basis, Uinv = lll_reduce_basis_modp(p, current_sections, Ep_local)

            if Uinv is None:
                Uinv_mat = identity_matrix(ZZ, r)
            else:
                try:
                    Uinv_mat = matrix(ZZ, [[int(Uinv[i, j]) for j in range(Uinv.ncols())] for i in range(Uinv.nrows())])
                except Exception:
                    Uinv_mat = identity_matrix(ZZ, r)

            vecs_transformed_for_p = []
            for v in vecs:
                try:
                    vZ = vector(ZZ, [int(c) for c in v])
                    transformed = vZ * Uinv_mat
                    vecs_transformed_for_p.append(tuple(int(transformed[i]) for i in range(len(transformed))))
                except Exception:
                    try:
                        vecs_transformed_for_p.append(tuple(int(c) for c in v))
                    except Exception:
                        vecs_transformed_for_p.append(None)

            required_ks_per_section = [set() for _ in range(r)]
            for v_trans in vecs_transformed_for_p:
                if v_trans is None:
                    continue
                for j, coeff in enumerate(v_trans):
                    required_ks_per_section[j].add(int(coeff))

            mults = [{} for _ in range(r)]
            any_mult_error = False
            for i_sec in range(r):
                Pi = new_basis[i_sec]
                required_ks = required_ks_per_section[i_sec]
                if not required_ks:
                    required_ks = {-1, 0, 1}

                mults_i = compute_all_mults_for_section(
                    Pi, required_ks, stats,
                    max_k=max((abs(k) for k in required_ks), default=1),
                    debug=(r > 1)
                )

                if mults_i is None:
                    any_mult_error = True
                    rejected_primes.append((p, f"multiplier_computation_failed_sec_{i_sec}"))
                    break
                mults[i_sec] = mults_i

            if any_mult_error:
                continue

            Ep_dict[p] = Ep_local
            for i, rhs_p_val in rhs_modp_for_p.items():
                rhs_modp_list[i][p] = rhs_p_val
            multiplies_lll[p] = mults
            vecs_lll[p] = vecs_transformed_for_p

        except (ZeroDivisionError, TypeError, ValueError, ArithmeticError) as e:
            if DEBUG and (p not in (2, 5)):
                print(f"Skipping prime {p} due to error during preparation: {e}")
            rejected_primes.append((p, f"exception_{type(e).__name__}"))
            continue

    if stats is not None:
        if not hasattr(stats, 'rejected_primes'):
            stats.rejected_primes = []
        stats.rejected_primes.extend(rejected_primes)
        
        if DEBUG:
            print(f"\n[prepare_modular_data_lll] Rejected {len(rejected_primes)} primes:")
            for p, reason in rejected_primes:
                print(f"  p={p}: {reason}")

            try:
                ram_locus = compute_ramification_locus(cd)
                detected_collisions = set(p for p, reason in rejected_primes if 'collision' in str(reason))
                assert detected_collisions.issubset(ram_locus), \
                    f"Detected collisions {detected_collisions} not in ramification locus {ram_locus}"
            except Exception as e:
                print(f"Ramification locus check failed: {e}")

    return Ep_dict, rhs_modp_list, multiplies_lll, vecs_lll


# ==============================================================
# === Main Worker Functions (Single Subset) ====================
# ==============================================================

def _process_prime_subset(p_subset, cd, current_sections, prime_pool, r_m, shift, rhs_list, vecs, tmax):
    """
    Worker function to find m-candidates for a single subset of primes.
    (This is the OLDER version from source [22])
    """
    if not p_subset:
        return set()

    # Prepare modular data for this specific prime subset.
    Ep_dict, rhs_modp_list_full, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, stats=None, search_primes=p_subset
    )

    if not Ep_dict:
        return set()

    found_candidates_for_subset = set()
    r = len(current_sections)

    # Process each search vector for this subset
    for idx, v_orig in enumerate(vecs):
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig) 

        residue_map = {}
        for p in p_subset:
            if p not in Ep_dict:
                continue

            v_p_list = vecs_lll.get(p)
            if v_p_list is None:
                continue
            if idx >= len(v_p_list):
                continue

            v_p_transformed = v_p_list[idx]
            mults = mult_lll.get(p)
            if mults is None:
                continue

            Ep = Ep_dict[p]
            
            Pm = Ep(0)
            for j, coeff in enumerate(v_p_transformed):
                if int(coeff) in mults[j]:
                    Pm += mults[j][int(coeff)]

            if Pm.is_zero():
                continue

            # Find roots for each RHS function
            roots_for_p = set()
            for i, rhs_ff in enumerate(rhs_list):
                # Get the correct RHS list for this prime
                if i not in rhs_modp_list_full:
                    continue
                rhs_modp_list = rhs_modp_list_full[i]
                if p not in rhs_modp_list:
                    continue

                rhs_p = rhs_modp_list[p]
                try:
                    num_modp = (Pm[0]/Pm[2] - rhs_p).numerator()
                    if not num_modp.is_zero():
                        roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                        roots_for_p.update(roots)
                except (ZeroDivisionError, ArithmeticError):
                    continue

            if roots_for_p:
                residue_map[p] = roots_for_p

        # Apply CRT to find m-candidates from the collected roots
        primes_for_crt = [p for p in p_subset if p in residue_map]
        if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
            continue

        lists = [residue_map[p] for p in primes_for_crt]
        for combo in itertools.product(*lists):
            M = reduce(mul, primes_for_crt, 1)

            if M > MAX_MODULUS:
                continue

            m0 = crt_cached(combo, tuple(primes_for_crt))
            
            try:
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
            except TypeError:
                best_ms = [(t, QQ(m0 + t * M), 0, 0.0) for t in (-1, 0, 1)] # t, m, x, score

            for t_cand, m_cand, _, _ in best_ms:
                found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

            try:
                a, b = rational_reconstruct(m0 % M, M)
                found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            except RationalReconstructionError:
                pass

    return found_candidates_for_subset, Counter(), set() # Return empty stats


def check_specific_t_value(t_candidate, m0, M, residue_map_for_filter, extra_primes, verbose=False):
    """
    Checks if a *single* integer t is valid against the extra prime constraints.
    """
    m0 = int(m0)
    M = int(M)
    t_candidate = int(t_candidate)

    for q in extra_primes:
        allowed_m_residues = residue_map_for_filter.get(q)

        if not allowed_m_residues:
            if verbose:
                print(f"Filter fail: Prime {q} has no allowed m-residues.")
            return False

        m_cand_mod_q = (m0 + t_candidate * M) % q

        if m_cand_mod_q not in allowed_m_residues:
            if verbose: print(f"Filter fail: t={t_candidate} -> m={m_cand_mod_q} (mod {q}) not in allowed set.")
            return False

    if verbose: print(f"Filter pass: t={t_candidate} is valid.")
    return True


def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, tmax, combo_cap, precomputed_residues, prime_pool, num_rhs_fns):
    """
    Worker function to find m-candidates for a single subset of primes.
    (This is the NEWER version from source [319])
    """
    if not p_subset:
        return set(), Counter(), set()

    found_candidates_for_subset = set()
    stats_counter = Counter()
    tested_crt_classes = set()

    num_extra_primes = 4
    offset = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][offset:num_extra_primes+offset]

    for v_orig in vecs:
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        for rhs_idx in range(num_rhs_fns):
            
            residue_map_for_crt = {}
            for p in p_subset:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                    residue_map_for_crt[p] = roots_for_this_rhs[rhs_idx]

            primes_for_crt = list(residue_map_for_crt.keys())
            if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
                continue

            residue_map_for_filter = {}
            for p in extra_primes_for_filtering:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                     residue_map_for_filter[p] = roots_for_this_rhs[rhs_idx]

            lists = [residue_map_for_crt[p] for p in primes_for_crt]
            
            for combo in itertools.product(*lists):
                stats_counter['crt_lift_attempts'] += 1
                M = 1
                for p in primes_for_crt:
                    M *= int(p)

                if M > MAX_MODULUS:
                    continue

                m0 = crt_cached(combo, tuple(primes_for_crt))
                tested_crt_classes.add((int(m0) % int(M), int(M)))
                
                try:
                    best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
                except TypeError:
                    best_ms = [(0, QQ(m0 + t * M), 0, 0.0) for t in (-1, 0, 1)] # t, m, x, score

                for t_cand, m_cand, _, _ in best_ms:
                    if check_specific_t_value(t_cand, m0, M, residue_map_for_filter, extra_primes_for_filtering):
                        found_candidates_for_subset.add( (QQ(m_cand), v_orig_tuple) )

                stats_counter['rational_recon_attempts_worker'] += 1
                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    found_candidates_for_subset.add( (QQ(a) / QQ(b), v_orig_tuple) )
                    stats_counter['rational_recon_success_worker'] += 1
                except RationalReconstructionError:
                    stats_counter['rational_recon_failure_worker'] += 1
                    pass # Do not raise, just fail reconstruction

    return found_candidates_for_subset, stats_counter, tested_crt_classes


# ==============================================================
# === Parallel Execution Helpers ===============================
# ==============================================================

def _make_executor(max_workers=None):
    """
    Try to create a ProcessPoolExecutor with 'fork' on Linux, fall back to threads.
    """
    try:
        ctx = multiprocessing.get_context("fork")
        return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    except Exception as e:
        if DEBUG:
            print(f"Warning: couldn't start process pool with fork: {e}. Falling back to threads.")
        return ThreadPoolExecutor(max_workers=max_workers)

def process_candidate_numeric(m_val, v_tuple, r_m_callable, shift, rationality_test_func, current_sections):
    """Fast candidate processor (pickleable for multiprocessing)."""
    try:
        x_val = r_m_callable(m_val) - shift
        y_val = rationality_test_func(x_val)
        if y_val is not None:
            v = vector(QQ, v_tuple)
            new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections))) if current_sections else None
            return m_val, x_val, y_val, v, new_sec
    except (TypeError, ZeroDivisionError, ArithmeticError):
        if DEBUG: print("here we are.")
        return None
    return None

def r_m_numeric_top(m_val, r_m_expr):
    """
    Evaluate symbolic r_m_expr at numeric m_val.
    Returns QQ(x)
    """
    SR_m = var('m')
    val = r_m_expr.subs({SR_m: m_val})
    return QQ(val)

def _compute_residues_for_prime_worker(args):
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

        if all(c == 0 for c in v_orig):
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        try:
            v_p_transformed = vecs_lll_p[idx]
        except Exception:
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue # Skip vector if transform failed

        try:
            Pm = Ep_local(0)
        except Exception:
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue # Skip vector if identity fails

        for j, coeff in enumerate(v_p_transformed):
            try:
                mpj = mults_p[j]
                if mpj is None:
                    continue
                key = int(coeff)
                if hasattr(mpj, 'get'):
                    if key in mpj:
                        Pm += mpj[key]
                else:
                    if 0 <= key < len(mpj):
                        Pm += mpj[key]
            except Exception:
                continue # Skip coeff if lookup fails

        try:
            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue
        except Exception:
             result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
             continue

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


def _compute_residues_for_prime_worker_old(args):
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

        if all(c == 0 for c in v_orig):
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        try:
            v_p_transformed = vecs_lll_p[idx]
        except Exception:
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        try:
            Pm = Ep_local(0)
        except Exception:
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

        for j, coeff in enumerate(v_p_transformed):
            try:
                mpj = mults_p[j]
                if mpj is None:
                    continue
                key = int(coeff)
                if hasattr(mpj, 'get'):
                    if key in mpj:
                        Pm += mpj[key]
                else:
                    if 0 <= key < len(mpj):
                        Pm += mpj[key]
            except Exception:
                continue

        try:
            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue
        except Exception:
            result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
            continue

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
