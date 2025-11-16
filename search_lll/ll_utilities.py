"""
ll_utilities.py: Matrix and lattice reduction helpers.
"""
from sage.all import ZZ, diagonal_matrix
from .search_config import *
from search_common import NUM_PRIME_SUBSETS


from collections import defaultdict, Counter
from sage.all import QQ, ZZ, Integer, PolynomialRing, GF

from sage.all import gcd

from collections import Counter
import math
from sage.all import Zmod, Integer

# ----------------------------------------
# helpers for residue orders
# ----------------------------------------
from collections import Counter, defaultdict
import math
from sage.all import Zmod, Integer

"""
Complete residue analysis with proper diagnostics.
Add this to ll_utilities.py, replacing the incomplete versions.
"""

from collections import Counter, defaultdict
import math
from sage.all import Zmod, Integer, QQ, var

# ============================================================================
# HELPER FUNCTIONS (keep existing ones, add these)
# ============================================================================

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
ll_utilities.py: Matrix and lattice reduction helpers.
"""
from sage.all import ZZ, diagonal_matrix
from .search_config import *
from search_common import NUM_PRIME_SUBSETS


def _compute_column_norms(M):
    """
    Compute L2 norm per column of integer matrix M (sage matrix).
    Returns list of floats.
    """
    col_norms = []
    for j in range(M.ncols()):
        s = 0
        for i in range(M.nrows()):
            v = int(M[i, j])
            s += v * v
        col_norms.append(float(s**0.5))
    return col_norms

def _compute_integer_scales_for_columns(M, target_norm=TARGET_COLUMN_NORM, max_scale=MAX_COL_SCALE):
    """
    Return integer scale factors (one per column) so that after multiplying column j by scales[j],
    its norm is approximately target_norm (but not exceeding max_scale).
    """
    norms = _compute_column_norms(M)
    scales = []
    for norm in norms:
        if norm <= 1e-12:
            scales.append(1)
            continue
        scale = int(ceil(target_norm / max(1.0, norm)))
        if scale < 1:
            scale = 1
        if scale > max_scale:
            scale = max_scale
        scales.append(int(scale))
    return scales

def _scale_matrix_columns_int(M, scales):
    """
    Return M_scaled = M * D where D = diag(scales), scales are ints.
    """
    D = diagonal_matrix([ZZ(s) for s in scales])
    return M * D, D

def prepare_modular_data_lll(cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=None):
    """
    Prepare modular data for LLL-based search across multiple primes.
    NOW RECORDS REJECTED PRIMES IN STATS FOR POSTERIOR ADJUSTMENT.
    """
    from sage.all import GF, PolynomialRing, QQ, SR, var, EllipticCurve, Integer, identity_matrix, vector, matrix

    if search_primes is None:
        search_primes = prime_pool

    r = len(current_sections)
    if r == 0:
        return {}, [], {}, {}

    Ep_dict, rhs_modp_list = {}, [{} for _ in rhs_list]
    multiplies_lll, vecs_lll = {}, {}
    rejected_primes = []  # Track (prime, reason) tuples
    
    PR_m = PolynomialRing(QQ, 'm')
    var_sym = var('m')

    processed_rhs_list = [{'num': PR_m(rhs.numerator()), 'den': PR_m(rhs.denominator())} for rhs in rhs_list]
    a4_num, a4_den = PR_m(cd.a4.numerator()), PR_m(cd.a4.denominator())
    a6_num, a6_den = PR_m(cd.a6.numerator()), PR_m(cd.a6.denominator())

    for p in search_primes:
        try:
            # Skip primes dividing denominators of a4/a6 coefficients
            if any(int(QQ(c).denominator()) % p == 0 for c in a4_num.coefficients(sparse=False)):
                if DEBUG:
                    print(f"[prepare_modular_data_lll] skip p={p}: a4 numerator has coeff with denom divisible by p")
                rejected_primes.append((p, "a4_denom_divisible"))
                continue
            if any(int(QQ(c).denominator()) % p == 0 for c in a6_num.coefficients(sparse=False)):
                if DEBUG:
                    print(f"[prepare_modular_data_lll] skip p={p}: a6 numerator has coeff with denom divisible by p")
                rejected_primes.append((p, "a6_denom_divisible"))
                continue

            Rp = PolynomialRing(GF(p), 'm')
            Fp_m = Rp.fraction_field()

            try:
                if a4_den.change_ring(GF(p)).is_zero() or a6_den.change_ring(GF(p)).is_zero():
                    if DEBUG:
                        print(f"[prepare_modular_data_lll] skip p={p}: a4/a6 denominator zero mod p")
                    rejected_primes.append((p, "a4_a6_denom_zero"))
                    continue
            except Exception:
                if DEBUG:
                    print(f"[prepare_modular_data_lll] skip p={p}: denominator coercion error")
                rejected_primes.append((p, "denom_coercion_failed"))
                raise
                continue

            a4_modp = Fp_m(a4_num) / Fp_m(a4_den)
            a6_modp = Fp_m(a6_num) / Fp_m(a6_den)

            # Diagnostic block for p=3 (optional, keep if useful)
            if p == 3 and DEBUG:
                print("\n" + "="*70)
                print(f"--- RUNNING MOD-{p} GEOMETRIC ANALYSIS (from diagnostics2.py) ---")
                print(f"Analyzing surface: y^2 = x^3 + a4_mod{p}(m)*x + a6_mod{p}(m)")
                try:
                    from diagnostics2 import find_singular_fibers
                    mod_p_fiber_report = find_singular_fibers(a4=a4_modp, a6=a6_modp, verbose=True)
                    print(f"--- MOD-{p} ANALYSIS COMPLETE ---")
                except Exception as e_diag:
                    print(f"--- MOD-{p} ANALYSIS FAILED: {e_diag} ---")
                    raise
                print("="*70 + "\n")

            # Check discriminant (singular curve)
            try:
                disc_modp = -16 * (4 * a4_modp**3 + 27 * a6_modp**2)
                if disc_modp.is_zero():
                    if DEBUG:
                        print(f"Skipping prime {p}: resulting curve is singular (discriminant = 0 mod {p})")
                    rejected_primes.append((p, "singular_discriminant"))
                    continue
            except Exception:
                if DEBUG:
                    print(f"[prepare_modular_data_lll] discriminant check failed at p={p}; skipping")
                rejected_primes.append((p, "discriminant_check_failed"))
                raise
                continue

            # *** CRITICAL: Fiber collision check ***
            try:
                Delta_poly = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
                if hasattr(Delta_poly, 'numerator'):
                    Delta_poly = Delta_poly.numerator()
                Delta_pr = PR_m(SR(Delta_poly))
                
                from search_lll import detect_fiber_collision
                has_collision, gcd_poly = detect_fiber_collision(Delta_pr, p, debug=DEBUG)
                
                if has_collision:
                    deg = gcd_poly.degree() if gcd_poly is not None else "N/A"
                    if DEBUG:
                        print(f"Skipping prime {p}: fiber collision detected (gcd degree={deg})")
                    rejected_primes.append((p, f"fiber_collision_deg_{deg}"))
                    continue
            except Exception as e:
                if DEBUG:
                    print(f"[fiber_collision_check] p={p}: error {e} -- continuing cautiously")
                raise

            # Construct elliptic curve
            try:
                Ep_local = EllipticCurve(Fp_m, [0, 0, 0, a4_modp, a6_modp])
            except ArithmeticError as e:
                if DEBUG:
                    print(f"Skipping prime {p}: EllipticCurve construction failed: {e}")
                rejected_primes.append((p, "elliptic_curve_construction_failed"))
                continue

            # Build rhs_modp for this prime
            rhs_modp_for_p = {}
            for i, rhs_data in enumerate(processed_rhs_list):
                try:
                    if rhs_data['den'].change_ring(GF(p)).is_zero():
                        if DEBUG:
                            print(f"[prepare_modular_data_lll] skip RHS#{i} at p={p}: denominator zero mod p")
                        continue
                    rhs_modp_for_p[i] = Fp_m(rhs_data['num']) / Fp_m(rhs_data['den'])
                except Exception:
                    if DEBUG:
                        print(f"[prepare_modular_data_lll] RHS#{i} reduction failed at p={p}")
                    raise

            # Run LLL reduction
            new_basis, Uinv = lll_reduce_basis_modp(p, current_sections, Ep_local)

            # Fallback for Uinv
            if Uinv is None:
                Uinv_mat = identity_matrix(ZZ, r)
            else:
                try:
                    nonint = False
                    for i_row in range(Uinv.nrows()):
                        for j_col in range(Uinv.ncols()):
                            entry = Uinv[i_row, j_col]
                            if hasattr(entry, 'denominator'):
                                if int(entry.denominator()) != 1:
                                    nonint = True
                                    break
                            else:
                                if QQ(entry) != Integer(entry):
                                    nonint = True
                                    break
                        if nonint:
                            break
                    if nonint:
                        Uinv_mat = identity_matrix(ZZ, r)
                    else:
                        Uinv_mat = matrix(ZZ, [[int(Uinv[i, j]) for j in range(Uinv.ncols())] for i in range(Uinv.nrows())])
                except Exception:
                    Uinv_mat = identity_matrix(ZZ, r)
                    raise

            # Transform vecs
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
                        raise
                    raise

            # Build required multiplier indices
            required_ks_per_section = [set() for _ in range(r)]
            for v_trans in vecs_transformed_for_p:
                if v_trans is None:
                    continue
                for j, coeff in enumerate(v_trans):
                    required_ks_per_section[j].add(int(coeff))

            # Compute multipliers
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
                    if DEBUG:
                        print(f"[prepare_modular_data_lll] p={p}: Failed to compute multipliers for basis section {i_sec}")
                    rejected_primes.append((p, f"multiplier_computation_failed_sec_{i_sec}"))
                    break

                mults[i_sec] = mults_i

            if any_mult_error:
                continue

            # Success - publish data for this prime
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

    # *** CRITICAL: Record rejected primes in stats ***
    if stats is not None:
        if not hasattr(stats, 'rejected_primes'):
            stats.rejected_primes = []
        stats.rejected_primes.extend(rejected_primes)
        
        if DEBUG:
            print(f"\n[prepare_modular_data_lll] Rejected {len(rejected_primes)} primes:")
            for p, reason in rejected_primes:
                print(f"  p={p}: {reason}")

                ram_locus = compute_ramification_locus(cd)
                detected_collisions = set(p for p, reason in rejected_primes if 'collision' in str(reason))
                assert detected_collisions.issubset(ram_locus), \
                    f"Detected collisions {detected_collisions} not in ramification locus {ram_locus}"



    return Ep_dict, rhs_modp_list, multiplies_lll, vecs_lll



def lll_reduce_basis_modp(p, sections, curve_modp,
                          truncate_deg=TRUNCATE_MAX_DEG,
                          lll_delta=LLL_DELTA, bkz_block=BKZ_BLOCK,
                          max_k_abs=MAX_K_ABS):
    """
    LLL/BKZ reduction with proper handling of single-section case and reduction failures.
    
    --- FIX 2 ---
    This function now *always* returns a list of length r = len(sections).
    If a reduction fails or a basis vector can't be computed,
    it places 'None' in that slot.
    """
    from sage.all import ZZ, identity_matrix, diagonal_matrix
    
    r = len(sections)
    if r == 0:
        return [], identity_matrix(ZZ, 0)

    # --- Start Fix: Handle reduction failures robustly ---
    
    # First, reduce all sections, padding with None on failure.
    # This list will have length r.
    reduced_sections_mod_p = [reduce_point_hom(curve_modp, P, p) for P in sections]
    
    # Check if *all* reductions failed
    if all(P is None for P in reduced_sections_mod_p):
        if DEBUG:
            print(f"[{p}] All {r} sections failed to reduce. Returning list of Nones.")
        return [None] * r, identity_matrix(ZZ, r)

    poly_coords = []
    max_deg = 0

    for Pp in reduced_sections_mod_p:
        # If reduction failed, use a placeholder (0,0,1) for LLL
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
        # Return the original reduced sections, which has length r
        return reduced_sections_mod_p, identity_matrix(ZZ, r)

    M = matrix(ZZ, coeff_vecs)
    
    # Handle r=1 case
    if M.nrows() <= 1:
        Uinv = identity_matrix(ZZ, r)
        # Return the original reduced sections, which has length r
        return reduced_sections_mod_p, Uinv

    # Handle matrix too wide
    if M.ncols() > 5 * M.nrows():
        if DEBUG:
            print(f"[LLL] Matrix too wide ({M.nrows()}x{M.ncols()}), skipping LLL for this prime")
        # Return the original reduced sections, which has length r
        return reduced_sections_mod_p, identity_matrix(ZZ, r)

    # --- End Fix ---

    try:
        scales = _compute_integer_scales_for_columns(M)
        M_scaled, D = _scale_matrix_columns_int(M, scales)
    except Exception as e:
        if DEBUG: 
            print("Column scaling failed, proceeding without scaling:", e)
        M_scaled = M
        D = diagonal_matrix([1]*M.ncols())
        raise

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

    if DEBUG:
        print(f"[LLL_DEBUG] p={p}, r={r}, M_scaled shape: {M_scaled.nrows()}x{M_scaled.ncols()}")
        print(f"[LLL_DEBUG] U type: {type(U).__name__}, has nrows: {hasattr(U, 'nrows')}")
        if hasattr(U, 'nrows'):
            print(f"[LLL_DEBUG] U shape: {U.nrows()}x{U.ncols()}")
        else:
            print(f"[LLL_DEBUG] U is not a matrix. repr (first 100 chars): {repr(U)[:100]}")
        print(f"[LLL_DEBUG] B type: {type(B).__name__}")

    Uinv = None
    assert U.nrows() == U.ncols(), "LLL transform U must be square"
    detU = int(U.det())
    assert abs(detU) == 1, "LLL transform U not unimodular; det = " + str(detU)
    Uinv = U.inverse()
    
    # --- FIX 3 ---
    # Rebuild the basis, safely handling None in reduced_sections_mod_p
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
                # This basis vector is a combination of only failed reductions
                new_basis.append(None)
        except Exception as e:
            if DEBUG:
                print(f"[LLL] Error computing new basis vector {i}: {e}")
            new_basis.append(None) # Signal failure for this vector
            raise

    # new_basis now has length r
    return new_basis, Uinv

def reduce_point_hom(E_mod_p, P, p, logger=None):
    """
    Reduce a projective/affine point P (whose coordinates may be
    in QQ or QQ(m)) to the curve E_mod_p (which is over GF(p) or GF(p)(m)).
    
    Returns:
        - The reduced point on E_mod_p on success
        - None if reduction fails (denominator non-invertible, bad coords, etc.)
    
    Returning None (not Ep(0)) allows callers to distinguish "couldn't reduce"
    from "reduced to identity".
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
                # This catches:
                # 1. QQ denominators divisible by p
                # 2. QQ(m) denominators that are 0 mod p
                # 3. Other type/coercion errors
                # log(f"[reduce_point_hom] p={p} failed to coerce projective coords: {e}")
                return None
                
        if len(coords) == 2:
            x, y = coords
            try:
                xr = Fp_target(x)
                yr = Fp_target(y)
                return E_mod_p(xr, yr)
            except Exception as e:
                # log(f"[reduce_point_hom] p={p} failed to coerce affine coords: {e}")
                return None

        log("[reduce_point_hom] unsupported coordinate shape")
        return None
        
    except Exception as outer_e:
        log(f"[reduce_point_hom] p={p} unexpected error: {outer_e}")
        return None



def _get_coeff_data(poly):
    """Helper to safely extract coefficient list and degree from a polynomial-like object."""
    if hasattr(poly, 'list') and hasattr(poly, 'degree'):
        return poly.list(), poly.degree()
    else:
        # Handle constants or other non-polynomial objects
        return [poly], 0


def _trim_poly_coeffs(coeff_list, max_deg=TRUNCATE_MAX_DEG):
    """Truncate coefficient list (low->high) to length max_deg+1."""
    if len(coeff_list) <= max_deg + 1:
        return coeff_list
    # Keep low-degree coefficients (assumed stored as [c0, c1, ..., cN])
    return coeff_list[: max_deg + 1]



def compute_all_mults_for_section(Pi, required_ks, stats,
                                  max_k=None, debug=False):
    """
    Compute specific multiples {k: k*Pi} for a reduced section Pi.
    Handles None input for Pi (returns None).
    """
    # --- FIX 1 ---
    # Handle failed reduction input from LLL basis
    if Pi is None:
        return None
    # -------------

    # Original function logic continues...
    from sage.all import ZZ
    
    if max_k is None:
        try:
            max_k = max(abs(k) for k in required_ks)
        except ValueError:
            max_k = MAX_K_ABS # fallback
            raise
    
    max_k = min(int(max_k), MAX_K_ABS)
    
    computed = {}
    try:
        identity = Pi.curve()(0)
        computed[0] = identity
    except Exception:
        # Fallback if curve is weird
        computed[0] = None
        
    computed[1] = Pi
    
    # Store by absolute value to minimize computations
    # e.g., if we need -5, compute 5 and then negate
    for k_abs in range(2, max_k + 1):
        if (not k_abs in required_ks and not -1*k_abs in required_ks):
            continue
        if k_abs in computed:
            continue
        try:
            computed[k_abs] = k_abs * Pi
            if stats: stats.incr('modular_mults')
        except Exception:
            # If 2*Pi fails, we can't compute much
            if debug:
                print(f"    [mults] k*Pi failed at k={k_abs}")
            break # Stop computing
            
    # Now build the final map from the required_ks
    final_mults = {}
    for k in required_ks:
        k_abs = abs(int(k))
        if k_abs not in computed:
            continue # Couldn't compute this multiple
        
        k_val = computed[k_abs]
        if k_val is None:
            continue

        if k < 0:
            final_mults[k] = -k_val
        else:
            final_mults[k] = k_val
            
    return final_mults


def estimate_prime_stats(prime_pool, precomputed_residues, sample_vecs, num_rhs=1):
    """Estimate average residue survival ratio r_p for each prime."""
    stats = {}
    for p in prime_pool:
        mapping = precomputed_residues.get(p, {})
        if not mapping:
            continue
        total = count = 0
        for v in sample_vecs:
            v_t = tuple(v)
            roots_list = mapping.get(v_t, [])
            if not roots_list:
                continue
            # combine across RHSs
            if num_rhs > 1:
                roots_union = set().union(*roots_list)
            else:
                roots_union = roots_list[0] if roots_list else set()
            total += len(roots_union)
            count += p
        stats[p] = (total / count) if count else 0.0
    return stats


def choose_extra_primes(stats, target_density=1e-5, max_extra=6, skip_small={2,3,5}):
    """Select extra primes based on measured r_p values."""
    cand = [(p, r) for p, r in stats.items()
            if p not in skip_small and EXTRA_PRIME_MIN_R < r < EXTRA_PRIME_MAX_R]
    # sort by discriminatory power (entropy-like)
    cand.sort(key=lambda t: -(t[1] * (1 - t[1])))
    chosen, prod = [], 1.0
    for p, r in cand:
        if len(chosen) >= max_extra:
            break
        prod *= r
        chosen.append(p)
        if prod <= target_density:
            break
    if DEBUG:
        print(f"[auto-tune] selected extra primes {chosen} with expected density {prod:.2e}")
    return chosen



def generate_biased_prime_subsets_by_coverage(prime_pool, precomputed_residues, vecs,
                                              num_subsets, min_size, max_size, combo_cap,
                                              seed=SEED_INT, force_full_pool=False, debug=DEBUG,
                                              roots_threshold=ROOTS_THRESHOLD):
    """
    Generate diverse prime subsets biased toward high-coverage primes, but skip
    combinatorially-pathological subsets whose estimated Cartesian-product of
    roots would exceed combo_cap.
    """
    # basic assertions
    assert isinstance(prime_pool, (list, tuple))
    assert isinstance(vecs, (list, tuple))
    assert num_subsets >= 1
    assert 1 <= min_size <= max_size
    assert max_size <= len(prime_pool)

    # random must be imported at module scope; fail loudly if not
    if 'random' not in globals():
        raise ImportError("module 'random' must be imported at module scope (add 'import random' at file top)")

    # compute coverage per prime (fraction of vectors with at least one root)
    coverage = compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=debug)

    # build sampling weights from coverage (floor at 0.05)
    weights = []
    for p in prime_pool:
        w = coverage.get(p, 0.0)
        if w < 0.05:
            w = 0.05
        weights.append(w)

    if debug:
        total_weight = sum(weights)
        avg_weight = total_weight / len(weights) if weights else 0
        print("generate_biased_coverage: avg weight =", avg_weight, "min =", min(weights), "max =", max(weights))

    # Identify top coverage primes (e.g., top 30-40%)
    sorted_primes = sorted(prime_pool, key=lambda p: coverage.get(p, 0.0), reverse=True)
    top_k = max(3, len(sorted_primes) // 3)  # top third of primes
    top_primes = sorted_primes[:top_k]

    if debug:
        print(f"generate_biased_coverage: Top {len(top_primes)} primes by coverage: {top_primes[:10]}")

    subsets = []
    if force_full_pool:
        subsets.append(list(prime_pool))

    # Generate some subsets that include combinations of top primes
    forced_subsets = []
    num_forced = min(20, max(2, num_subsets // 10))  # Increased from 10 to 20
    if len(top_primes) >= 2:
        import random
        for i in range(num_forced):
            # Vary the size: alternate between small (min_size) and larger
            if i % 3 == 0:
                # Small subsets focusing on top primes (1/3 of forced subsets)
                size = min_size
                num_top = min(size, len(top_primes))
                subset = random.sample(top_primes, k=num_top)
            else:
                # Larger subsets mixing top primes with others (2/3 of forced subsets)
                size = random.randint(min_size, min(max_size, len(prime_pool)))
                num_top = min(2, size // 2, len(top_primes))
                subset = random.sample(top_primes, k=num_top)
                remaining_slots = size - len(subset)
                if remaining_slots > 0:
                    other_primes = [p for p in prime_pool if p not in subset]
                    if other_primes:
                        subset.extend(random.sample(other_primes, k=min(remaining_slots, len(other_primes))))
            forced_subsets.append(tuple(sorted(subset)))

    subsets.extend(forced_subsets)

    # Calculate how many more subsets we need
    remaining = num_subsets - len(subsets)
    if remaining <= 0:
        # dedupe and return early
        seen = set()
        unique = []
        for s in subsets:
            t = tuple(sorted(s))
            if t not in seen:
                seen.add(t)
                unique.append(list(t))
        if debug:
            print("generate_biased_coverage: Generated", len(unique), "unique subsets")
        return unique

    # helper to estimate root-product for a candidate subset
    def _estimate_subset_explosion(subset):
        est = 1
        for p in subset:
            mapping = precomputed_residues.get(p)
            if not mapping:
                return 0
            roots_total = 0
            for roots_lists in mapping.values():
                for rl in roots_lists:
                    roots_total += len(rl)
            if roots_total == 0:
                return 0
            if roots_total > roots_threshold:
                est *= roots_total
                if est > combo_cap:
                    return est
        return est

    # produce remaining subsets by weighted sampling, skipping heavy ones
    max_attempts_per_subset = 200
    import random
    for _ in range(remaining):
        attempts = 0
        chosen = None
        while attempts < max_attempts_per_subset:
            attempts += 1
            size = random.randint(min_size, min(max_size, len(prime_pool)))

            subset = None
            try:
                subset = random.sample(prime_pool, k=size)
            except TypeError:
                subset = []
                tries_inner = 0
                while len(subset) < size and tries_inner < size * 20:
                    p = random.choices(prime_pool, weights=weights, k=1)[0]
                    if p not in subset:
                        subset.append(p)
                    tries_inner += 1
                if len(subset) < size:
                    remaining_primes = [p for p in prime_pool if p not in subset]
                    need = min(size - len(subset), len(remaining_primes))
                    if need > 0:
                        subset.extend(random.sample(remaining_primes, k=need))
                raise

            subset = tuple(sorted(subset))

            est = _estimate_subset_explosion(subset)
            if est == 0:
                continue
            if est > combo_cap:
                continue

            chosen = list(subset)
            break

        if chosen is None:
            chosen = list(random.sample(prime_pool, k=min(min_size, len(prime_pool))))
            if debug:
                print("generate_biased_coverage: fallback subset used after attempts")

        subsets.append(tuple(sorted(chosen)))

    # deduplicate preserving order
    seen = set()
    unique_subsets = []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            unique_subsets.append(list(s))

    if debug:
        print("generate_biased_coverage: Generated", len(unique_subsets), "unique subsets")
        if unique_subsets:
            sample_show = unique_subsets[:3]
            print("generate_biased_coverage: Sample subsets:", sample_show)

    #print("subsets used:", unique_subsets)
    return unique_subsets


def compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=DEBUG):
    """
    For each prime, compute what fraction of search vectors have roots mod that prime.
    
    Args:
        prime_pool (list): Primes to analyze
        precomputed_residues (dict): {p: {v_tuple: [roots_per_rhs]}} from worker
        vecs (list): All search vectors
        debug (bool): Print diagnostics
    
    Returns:
        dict: {p: coverage_fraction} where coverage in [0, 1]
    """
    coverage = {}
    
    num_vecs = len(vecs)
    if num_vecs == 0:
        return {p: 0.5 for p in prime_pool}
    
    for p in prime_pool:
        p_data = precomputed_residues.get(p, {})
        if not p_data:
            coverage[p] = 0.0
            continue
        
        # Count vectors that have at least one root for this prime (across any RHS)
        vectors_with_roots = 0
        for v in vecs:
            v_tuple = tuple(v)
            roots_list = p_data.get(v_tuple, [])
            
            # roots_list is a list of sets (one per RHS function)
            # Check if any RHS has roots
            has_roots = any(rhs_roots for rhs_roots in roots_list)
            
            if has_roots:
                vectors_with_roots += 1
        
        coverage[p] = float(vectors_with_roots) / float(num_vecs)
    
    if debug:
        sorted_by_cov = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        print(f"[compute_prime_coverage] Prime coverage (top 20):")
        for p, cov in sorted_by_cov[:20]:
            print(f"  p={p}: coverage={cov:.1%}")
    
    return coverage

def print_subset_productivity_stats(productive, all_subsets):
    """Print quick stats on which prime subsets were productive"""
    from collections import Counter
    
    total = len(all_subsets)
    productive_count = len(productive)
    total_candidates = sum(p['candidates'] for p in productive)
    
    print(f"\n[subset stats] {productive_count}/{total} subsets produced candidates "
          f"({100*productive_count/total:.1f}%)")
    print(f"[subset stats] {total_candidates} total candidates from productive subsets")
    
    # By size
    by_size = Counter(p['size'] for p in productive)
    all_by_size = Counter(len(s) for s in all_subsets)
    
    print(f"[subset stats] Productivity by size:")
    for size in sorted(all_by_size.keys()):
        prod_count = by_size.get(size, 0)
        total_count = all_by_size[size]
        rate = 100 * prod_count / total_count if total_count > 0 else 0
        cands = sum(p['candidates'] for p in productive if p['size'] == size)
        print(f"  Size {size}: {prod_count}/{total_count} productive ({rate:.1f}%), "
              f"{cands} candidates")
    
    # Top productive subsets
    top = sorted(productive, key=lambda x: x['candidates'], reverse=True)[:5]
    print(f"[subset stats] Top 5 productive subsets:")
    for p in top:
        print(f"  {p['primes']}: {p['candidates']} candidates")




# Add near other helpers in search_lll.py (no leading underscores)
from collections import defaultdict, Counter
from sage.all import QQ, ZZ, Integer, PolynomialRing, GF

def compute_residues_by_prime_numeric(precomputed_residues):
    """
    Convert precomputed_residues -> residues_by_prime_numeric:
      input: {p: {v_tuple: [set(roots_for_rhs0), ...]}}
      output: {p: set(int_residues)}
    """
    residues_by_prime = {}
    for p, mapping in precomputed_residues.items():
        s = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        s.add(int(r % p))
        residues_by_prime[int(p)] = s
    return residues_by_prime


from sage.all import gcd

from collections import Counter
import math
from sage.all import Zmod, Integer

# ----------------------------------------
# helpers for residue orders
# ----------------------------------------
def residue_order_additive(residue, p):
    """Order of residue in additive group Z/pZ"""
    if residue % p == 0:
        return 1
    from math import gcd
    return p // gcd(residue, p)

from collections import Counter, defaultdict
import math
from sage.all import Zmod, Integer

def summarize_order_stats(order_list):
    """Return frequency, dominance, and entropy of a list of orders"""
    total = len(order_list)
    if total == 0:
        return {'freq': {}, 'dominance': None, 'entropy': None}
    count = Counter(order_list)
    most_common_freq = count.most_common(1)[0][1]
    dominance = most_common_freq / total
    entropy = -sum((v/total) * math.log2(v/total) for v in count.values())
    return {'freq': dict(count), 'dominance': dominance, 'entropy': entropy}


"""
Complete residue analysis with proper diagnostics.
Add this to ll_utilities.py, replacing the incomplete versions.
"""

from collections import Counter, defaultdict
import math
from sage.all import Zmod, Integer, QQ, var

# ============================================================================
# HELPER FUNCTIONS (keep existing ones, add these)
# ============================================================================

def quadratic_character(r, p):
    """Legendre symbol (r/p). RAISES on error."""
    if r % p == 0:
        return 0
    from sage.all import kronecker
    return kronecker(r, p)


def discriminant_valuation(r, p, Delta_pr):
    """v_p(Delta(r)) for discriminant polynomial. RAISES on error."""
    if Delta_pr is None:
        return None
    val_at_r = Delta_pr(r)
    if val_at_r == 0:
        return "inf"
    ret = QQ(val_at_r).valuation(p)
    return ret


def compute_trace_of_frobenius(r, p, Ep_m):
    """Trace of Frobenius a_p. RAISES on error."""
    if Ep_m is None:
        return None
    a4_r = Ep_m.a4()(r)
    a6_r = Ep_m.a6()(r)
    disc_r = -16 * (4 * a4_r**3 + 27 * a6_r**2)
    if disc_r == 0:
        return "sing"
    from sage.all import EllipticCurve, GF
    Ep_r = EllipticCurve(GF(p), [0, 0, 0, a4_r, a6_r])
    return Ep_r.trace_of_frobenius()


def local_height_contribution(r, p, rhs_fn):
    """Local height lambda_p(r). RAISES on error."""
    from sage.all import var, Integer, QQ
    m = var('m')
    f_r = rhs_fn.subs({m: r})
    f_r_qq = QQ(f_r)
    num_val = Integer(f_r_qq.numerator()).valuation(p)
    den_val = Integer(f_r_qq.denominator()).valuation(p)
    return max(0, -num_val) + max(0, den_val)


def residue_order_multiplicative(residue, p):
    """Order in (Z/pZ)*. RAISES on error."""
    from sage.all import Zmod
    if residue % p == 0:
        return 0
    return Zmod(p)(residue).multiplicative_order()


def liftability_order(r, p, f):
    """p-adic liftability v_p(f(r)). RAISES on error."""
    from sage.all import var, Integer, QQ
    m = var('m')
    f_r = f.subs({m: r})
    f_r = QQ(f_r)
    num_val = Integer(f_r.numerator()).valuation(p)
    den_val = Integer(f_r.denominator()).valuation(p)
    return num_val - den_val


def detect_residue_patterns(per_prime):
    """Detect patterns. RAISES on error."""
    from collections import Counter
    
    patterns = {
        'disc_pattern_used': Counter(),
        'disc_pattern_unused': Counter(),
        'qc_used': Counter(),
        'qc_unused': Counter(),
        'lift_range_used': Counter(),
        'lift_range_unused': Counter(),
        'a_p_sign_used': Counter(),
        'a_p_sign_unused': Counter(),
        'local_height_used': [],
        'local_height_unused': [],
        'composite_patterns': {
            'used_high_lift_high_disc': 0,
            'unused_high_lift_high_disc': 0,
            'used_qc_minus_lift_pos': 0,
            'unused_qc_minus_lift_pos': 0,
            'used_zero_local_height': 0,
            'unused_zero_local_height': 0,
        }
    }
    
    for p, info in per_prime.items():
        diag = info['diagnostics']
        used = info['used_residues']
        unused = info['unused_residues']
        
        # Process used
        for r in used:
            if r not in diag:
                continue
            d = diag[r]
            patterns['disc_pattern_used'][d.get('disc_pattern', 'unknown')] += 1
            patterns['qc_used'][d.get('qc', 'NA')] += 1
            
            lh = d.get('local_height')
            if lh is not None:
                patterns['local_height_used'].append(lh)
                if lh == 0:
                    patterns['composite_patterns']['used_zero_local_height'] += 1
            
            lift = d.get('liftability_order', 0)
            if lift < -1:
                patterns['lift_range_used']['highly_negative'] += 1
            elif lift == -1:
                patterns['lift_range_used']['negative'] += 1
            elif lift == 0:
                patterns['lift_range_used']['zero'] += 1
            elif lift == 1:
                patterns['lift_range_used']['positive'] += 1
            else:
                patterns['lift_range_used']['highly_positive'] += 1
            
            ap = d.get('a_p')
            if ap == 'sing':
                patterns['a_p_sign_used']['singular'] += 1
            elif isinstance(ap, (int, float)):
                if ap > 0:
                    patterns['a_p_sign_used']['positive'] += 1
                elif ap < 0:
                    patterns['a_p_sign_used']['negative'] += 1
                else:
                    patterns['a_p_sign_used']['zero'] += 1
            
            if lift >= 1 and isinstance(d.get('disc_val'), int) and d.get('disc_val') >= 1:
                patterns['composite_patterns']['used_high_lift_high_disc'] += 1
            if d.get('qc') == -1 and lift > 0:
                patterns['composite_patterns']['used_qc_minus_lift_pos'] += 1
        
        # Process unused
        for r in unused:
            if r not in diag:
                continue
            d = diag[r]
            patterns['disc_pattern_unused'][d.get('disc_pattern', 'unknown')] += 1
            patterns['qc_unused'][d.get('qc', 'NA')] += 1
            
            lh = d.get('local_height')
            if lh is not None:
                patterns['local_height_unused'].append(lh)
                if lh == 0:
                    patterns['composite_patterns']['unused_zero_local_height'] += 1
            
            lift = d.get('liftability_order', 0)
            if lift < -1:
                patterns['lift_range_unused']['highly_negative'] += 1
            elif lift == -1:
                patterns['lift_range_unused']['negative'] += 1
            elif lift == 0:
                patterns['lift_range_unused']['zero'] += 1
            elif lift == 1:
                patterns['lift_range_unused']['positive'] += 1
            else:
                patterns['lift_range_unused']['highly_positive'] += 1
            
            ap = d.get('a_p')
            if ap == 'sing':
                patterns['a_p_sign_unused']['singular'] += 1
            elif isinstance(ap, (int, float)):
                if ap > 0:
                    patterns['a_p_sign_unused']['positive'] += 1
                elif ap < 0:
                    patterns['a_p_sign_unused']['negative'] += 1
                else:
                    patterns['a_p_sign_unused']['zero'] += 1
            
            if lift >= 1 and isinstance(d.get('disc_val'), int) and d.get('disc_val') >= 1:
                patterns['composite_patterns']['unused_high_lift_high_disc'] += 1
            if d.get('qc') == -1 and lift > 0:
                patterns['composite_patterns']['unused_qc_minus_lift_pos'] += 1
    
    # Compute stats - RAISES on error
    if patterns['local_height_used']:
        import statistics
        patterns['local_height_stats_used'] = {
            'mean': statistics.mean(patterns['local_height_used']),
            'median': statistics.median(patterns['local_height_used']),
            'max': max(patterns['local_height_used'])
        }
    if patterns['local_height_unused']:
        import statistics
        patterns['local_height_stats_unused'] = {
            'mean': statistics.mean(patterns['local_height_unused']),
            'median': statistics.median(patterns['local_height_unused']),
            'max': max(patterns['local_height_unused'])
        }
    
    return patterns


def summarize_unused_residue_characteristics(analysis_ret, top_k=10):
    """Summary with patterns. RAISES on error."""
    per_prime = analysis_ret['per_prime']
    global_summary = analysis_ret['global']

    prime_counts = sorted(global_summary['per_prime_counts'].items(), key=lambda x: -x[1])
    top_primes = prime_counts[:top_k]

    heavy = []
    for p, info in per_prime.items():
        mult = info['multiplicity']
        for r, c in mult.items():
            if c > 1:
                heavy.append((p, r, c))
    heavy_sorted = sorted(heavy, key=lambda x: -x[2])[:top_k]

    patterns = detect_residue_patterns(per_prime)

    return {
        'num_primes': global_summary['num_primes'],
        'total_residues': global_summary['total_residues'],
        'total_unused': global_summary['total_unused_residues'],
        'top_primes': top_primes,
        'heavy_residues': heavy_sorted,
        'patterns': patterns
    }


def print_residue_analysis(analysis):
    """Print analysis. RAISES on error."""
    summary = summarize_unused_residue_characteristics(analysis)
    
    print("\n" + "="*70)
    print("RESIDUE PATTERN ANALYSIS")
    print("="*70)
    
    print(f"\nTotal primes analyzed: {summary['num_primes']}")
    print(f"Total residues: {summary['total_residues']}")
    print(f"Unused residues: {summary['total_unused']}")
    
    if 'patterns' in summary:
        patterns = summary['patterns']
        
        print("\n--- Discriminant Patterns ---")
        print(f"  Used: {dict(patterns['disc_pattern_used'])}")
        print(f"  Unused: {dict(patterns['disc_pattern_unused'])}")
        
        print("\n--- Quadratic Character ---")
        print(f"  Used: {dict(patterns['qc_used'])}")
        print(f"  Unused: {dict(patterns['qc_unused'])}")
        
        print("\n--- Liftability ---")
        print(f"  Used: {dict(patterns['lift_range_used'])}")
        print(f"  Unused: {dict(patterns['lift_range_unused'])}")
        
        if 'local_height_stats_used' in patterns and 'local_height_stats_unused' in patterns:
            print("\n--- Local Height ---")
            u = patterns['local_height_stats_used']
            un = patterns['local_height_stats_unused']
            print(f"  Used: mean={u['mean']:.2f}, median={u['median']:.2f}, max={u['max']}")
            print(f"  Unused: mean={un['mean']:.2f}, median={un['median']:.2f}, max={un['max']}")
        
        print("\n--- Composite Patterns ---")
        comp = patterns['composite_patterns']
        print(f"  High lift + high disc: used={comp['used_high_lift_high_disc']}, unused={comp['unused_high_lift_high_disc']}")
        print(f"  QC=-1 + lift>0: used={comp['used_qc_minus_lift_pos']}, unused={comp['unused_qc_minus_lift_pos']}")
        print(f"  Zero local height: used={comp['used_zero_local_height']}, unused={comp['unused_zero_local_height']}")
        
        # Discriminative power
        print("\n--- Discriminative Power ---")
        total_used = sum(patterns['qc_used'].values())
        total_unused = sum(patterns['qc_unused'].values())
        
        if total_used > 0 and total_unused > 0:
            for feature in ['qc', 'disc_pattern', 'lift_range', 'a_p_sign']:
                used_dist = patterns[f'{feature}_used']
                unused_dist = patterns[f'{feature}_unused']
                
                categories = set(used_dist.keys()) | set(unused_dist.keys())
                max_diff = 0
                best_cat = None
                for cat in categories:
                    used_freq = used_dist.get(cat, 0) / max(1, total_used)
                    unused_freq = unused_dist.get(cat, 0) / max(1, total_unused)
                    diff = abs(used_freq - unused_freq)
                    if diff > max_diff:
                        max_diff = diff
                        best_cat = cat
                
                if best_cat:
                    used_pct = 100 * used_dist.get(best_cat, 0) / max(1, total_used)
                    unused_pct = 100 * unused_dist.get(best_cat, 0) / max(1, total_unused)
                    print(f"  {feature}: diff={max_diff:.3f} at '{best_cat}' (used={used_pct:.1f}%, unused={unused_pct:.1f}%)")
    
    print("="*70)



def analyze_unused_residue_orders(precomputed_residues,
                                  rhs_list,
                                  found_m_set=None,
                                  found_xs=None,
                                  prime_pool=None,
                                  max_lift_k=3,
                                  debug=False,
                                  Delta_pr=None,
                                  Ep_dict=None):
    """
    Complete residue analysis with all diagnostics.
    RAISES ALL EXCEPTIONS - NO SILENT FAILURES.
    """
    from sage.all import ZZ, QQ, GF, Integer as SageInteger
    from collections import Counter, defaultdict
    
    # Build numeric residues
    residues_by_prime = {}
    for p, mapping in precomputed_residues.items():
        s = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        s.add(int(r % p))
        residues_by_prime[int(p)] = s

    if prime_pool is None:
        prime_list = sorted(residues_by_prime.keys())
    else:
        prime_list = [p for p in prime_pool if p in residues_by_prime]

    # Build found residues - CORRECT rational reduction
    found_residues_by_prime = defaultdict(set)
    if found_m_set:
        for m in found_m_set:
            m_qq = QQ(m)  # RAISE if can't convert
            for p in prime_list:
                denom = m_qq.denominator()
                if denom % p == 0:
                    # m has pole at p, skip
                    continue
                # Proper reduction: a/b mod p = a * b^(-1) mod p
                Fp = GF(p)
                m_mod_p = int(Fp(m_qq.numerator()) / Fp(denom))
                found_residues_by_prime[p].add(m_mod_p)

    # Analyze each prime
    per_prime_report = {}
    
    for p in prime_list:
        residues = sorted(residues_by_prime.get(p, []))
        used = found_residues_by_prime.get(p, set())
        unused = [r for r in residues if r not in used]

        # Multiplicity tracking
        multiplicity = Counter()
        origin_vectors = defaultdict(list)
        mapping = precomputed_residues.get(p, {})
        for vtuple, rhs_lists in mapping.items():
            for rhs_idx, rl in enumerate(rhs_lists):
                for r in rl:
                    if isinstance(r, int):
                        rmod = int(r % p)
                        multiplicity[rmod] += 1
                        origin_vectors[rmod].append((vtuple, rhs_idx))

        # Get curve for this prime
        Ep_m_p = Ep_dict.get(p) if Ep_dict else None

        # Compute diagnostics for all residues
        diagnostics = {}
        
        for r in residues:
            rhs_fn = rhs_list[0]
            
            # All these RAISE on error
            vp = liftability_order(r, p, rhs_fn)
            qc = quadratic_character(r, p)
            dv = discriminant_valuation(r, p, Delta_pr)
            ap = compute_trace_of_frobenius(r, p, Ep_m_p)
            lh = local_height_contribution(r, p, rhs_fn)
            
            # Composite diagnostics
            lift_disc = vp * (dv if isinstance(dv, (int, SageInteger)) else 0)
            qc_lift = qc * vp
            
            # Categorize - FIXED to handle Sage Integers
            if dv == "inf":
                disc_pattern = "singular"
            elif dv is None:
                disc_pattern = "unknown"
            elif dv > 2:
                disc_pattern = "high_ramification"
            elif dv == 1:
                disc_pattern = "simple_ramification"
            elif dv == 0:
                disc_pattern = "smooth"
            else:
                disc_pattern = "unknown"
            
            diagnostics[r] = {
                'multiplicity': multiplicity.get(r, 0),
                'origin_count': len(origin_vectors.get(r, [])),
                'order_mult': residue_order_multiplicative(r, p),
                'liftability_order': vp,
                'qc': qc,
                'disc_val': dv,
                'a_p': ap,
                'local_height': lh,
                'lift_disc_product': lift_disc,
                'qc_lift_product': qc_lift,
                'disc_pattern': disc_pattern
            }

        per_prime_report[p] = {
            'residues': residues,
            'used_residues': sorted(list(used)),
            'unused_residues': unused,
            'multiplicity': multiplicity,
            'origins': origin_vectors,
            'diagnostics': diagnostics
        }

    global_summary = {
        'num_primes': len(prime_list),
        'total_residues': sum(len(residues_by_prime.get(p, [])) for p in prime_list),
        'total_unused_residues': sum(len(per_prime_report[p]['unused_residues']) for p in prime_list),
        'per_prime_counts': {p: len(per_prime_report[p]['unused_residues']) for p in prime_list},
    }

    return {
        'per_prime': per_prime_report,
        'global': global_summary
    }



"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

"""
Enhanced prime subset generation with QC-aware biasing.
Add this to ll_utilities.py
"""

def compute_qc_bias_scores(prime_pool, precomputed_residues, rhs_list, 
                           target_qc_ratio=None, debug=False):
    """
    Compute QC distribution bias for each prime.
    
    Returns:
        dict: {p: {'qc_ratio': float, 'qc_entropy': float, 'coverage': float}}
        
    Strategy: Primes with QC ratios matching the target are more likely productive.
    If target_qc_ratio is None, don't use QC bias (fall back to coverage only).
    """
    from collections import Counter
    from sage.all import kronecker, QQ
    import math
    
    scores = {}
    
    for p in prime_pool:
        mapping = precomputed_residues.get(p, {})
        if not mapping:
            scores[p] = {'qc_ratio': 1.0, 'qc_entropy': 0.0, 'coverage': 0.0}
            continue
        
        # Collect all numeric residues for this prime
        all_residues = set()
        vectors_with_roots = 0
        total_vectors = len(mapping)
        
        for vtuple, rhs_lists in mapping.items():
            has_roots = any(rhs_roots for rhs_roots in rhs_lists)
            if has_roots:
                vectors_with_roots += 1
            
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        all_residues.add(int(r % p))
        
        if not all_residues:
            scores[p] = {'qc_ratio': 1.0, 'qc_entropy': 0.0, 'coverage': 0.0}
            continue
        
        # Compute QC distribution
        qc_counts = Counter()
        for r in all_residues:
            try:
                qc = kronecker(r, p)
                qc_counts[qc] += 1
            except Exception:
                continue
        
        # Compute ratio (with smoothing to avoid division by zero)
        qc_minus = qc_counts.get(-1, 0)
        qc_plus = qc_counts.get(1, 0)
        qc_zero = qc_counts.get(0, 0)
        
        # Add Laplace smoothing
        qc_ratio = (qc_minus + 1) / (qc_plus + 1)
        
        # Compute entropy (measure of QC uniformity)
        total = qc_minus + qc_plus + qc_zero
        if total > 0:
            probs = [c / total for c in [qc_minus, qc_plus, qc_zero] if c > 0]
            qc_entropy = -sum(p * math.log2(p) for p in probs)
        else:
            qc_entropy = 0.0
        
        coverage = vectors_with_roots / max(1, total_vectors)
        
        scores[p] = {
            'qc_ratio': qc_ratio,
            'qc_entropy': qc_entropy,
            'coverage': coverage,
            'qc_counts': dict(qc_counts),
            'total_residues': len(all_residues)
        }
    
    if debug and target_qc_ratio is not None:
        print("\n[QC Bias Analysis]")
        sorted_by_ratio = sorted(scores.items(), 
                                key=lambda x: abs(x[1]['qc_ratio'] - target_qc_ratio), 
                                reverse=False)
        print(f"Primes closest to target QC ratio ({target_qc_ratio:.3f}):")
        for p, data in sorted_by_ratio[:10]:
            print(f"  p={p}: qc_ratio={data['qc_ratio']:.3f}, "
                  f"entropy={data['qc_entropy']:.3f}, "
                  f"coverage={data['coverage']:.1%}, "
                  f"counts={data.get('qc_counts', {})}")
    
    return scores


def generate_qc_biased_prime_subsets(prime_pool, precomputed_residues, vecs,
                                     rhs_list, num_subsets, min_size, max_size,
                                     combo_cap, seed=None, debug=False,
                                     roots_threshold=12, target_qc_ratio=None):
    """
    Generate prime subsets biased toward primes with favorable QC distributions.
    
    Args:
        target_qc_ratio: Target QC=-1/QC=1 ratio. If None, use coverage only (no QC bias)
    
    Strategy:
        1. Score primes by how close their QC ratio is to target (if provided)
        2. Weight subsets to include primes with good QC ratios + high coverage
        3. Maintain diversity by including some "average" primes too
    """
    import random
    import math
    if seed is not None:
        random.seed(seed)
    
    # Compute coverage (always use this)
    coverage = compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=False)
    
    # Decide whether to use QC bias
    if target_qc_ratio is None:
        # Pure coverage-based (original behavior)
        if debug:
            print("[QC-Biased] target_qc_ratio=None, falling back to coverage-only")
        composite_scores = coverage
    else:
        # Compute QC scores
        qc_scores = compute_qc_bias_scores(prime_pool, precomputed_residues, 
                                           rhs_list, target_qc_ratio=target_qc_ratio, 
                                           debug=debug)
        
        # Build composite score: balance QC ratio + coverage
        composite_scores = {}
        for p in prime_pool:
            qc_data = qc_scores[p]
            cov = coverage.get(p, 0.0)
            
            # Distance from target QC ratio (smaller is better)
            qc_distance = abs(qc_data['qc_ratio'] - target_qc_ratio)
            
            # QC score: exponential penalty for distance from target
            # Use -3 instead of -2 to make it sharper
            qc_score = math.exp(-3 * qc_distance)  # Peaks sharply at ratio=target
            
            # Composite score: balance coverage and QC
            # Scale coverage to [0,1] range and weight equally
            composite = 0.5 * cov + 0.5 * qc_score
            
            composite_scores[p] = composite
        
        if debug:
            print(f"\n[QC-Biased Generation] Target QC ratio: {target_qc_ratio:.3f}")
            print("Top primes by composite score:")
            sorted_primes = sorted(composite_scores.items(), key=lambda x: -x[1])
            for p, score in sorted_primes[:10]:
                qc_r = qc_scores[p]['qc_ratio']
                cov = coverage.get(p, 0)
                print(f"  p={p}: score={score:.3f}, qc_ratio={qc_r:.3f}, coverage={cov:.1%}")
    
    # Normalize to weights
    total_weight = sum(composite_scores.values())
    if total_weight == 0:
        weights = [1.0] * len(prime_pool)
    else:
        weights = [composite_scores[p] / total_weight for p in prime_pool]
    
    # Identify top primes by composite score
    sorted_primes = sorted(composite_scores.items(), key=lambda x: -x[1])
    top_k = max(5, len(sorted_primes) // 3)
    top_primes = [p for p, _ in sorted_primes[:top_k]]
    
    if debug and target_qc_ratio is not None:
        print(f"[QC-Biased] Top {len(top_primes)} primes: {top_primes[:10]}")
    
    # Generate subsets (rest of function unchanged)
    subsets = []
    
    # Phase 1: Forced subsets featuring top primes
    num_forced = min(30, max(5, num_subsets // 8))
    for i in range(num_forced):
        if i % 2 == 0:
            size = min_size
            num_top = min(size, len(top_primes))
            subset = random.sample(top_primes, k=num_top)
        else:
            size = random.randint(min_size, min(max_size, len(prime_pool)))
            num_top = min(3, size // 2, len(top_primes))
            subset = random.sample(top_primes, k=num_top)
            
            remaining_slots = size - len(subset)
            if remaining_slots > 0:
                other_primes = [p for p in prime_pool if p not in subset]
                if other_primes:
                    other_weights = [composite_scores.get(p, 0.1) for p in other_primes]
                    chosen = random.choices(other_primes, 
                                          weights=other_weights,
                                          k=min(remaining_slots, len(other_primes)))
                    subset.extend(chosen)
        
        subsets.append(tuple(sorted(set(subset))))
    
    # Phase 2: Random weighted subsets
    remaining = num_subsets - len(subsets)
    for _ in range(remaining):
        size = random.randint(min_size, min(max_size, len(prime_pool)))
        
        subset = []
        attempts = 0
        while len(subset) < size and attempts < size * 20:
            p = random.choices(prime_pool, weights=weights, k=1)[0]
            if p not in subset:
                subset.append(p)
            attempts += 1
        
        subsets.append(tuple(sorted(subset)))
    
    # Deduplicate and filter by combo_cap
    unique_subsets = []
    seen = set()
    
    residues_by_prime_numeric = {}
    for p, mapping in precomputed_residues.items():
        residues_set = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        residues_set.add(r)
        residues_by_prime_numeric[p] = residues_set
    
    for subset in subsets:
        if subset in seen:
            continue
        
        est = 1
        viable = True
        for p in subset:
            count = len(residues_by_prime_numeric.get(p, set()))
            if count == 0:
                viable = False
                break
            if count > roots_threshold:
                est *= count
                if est > combo_cap:
                    viable = False
                    break
            else:
                est *= max(1, count)
                if est > combo_cap:
                    viable = False
                    break
        
        if viable:
            seen.add(subset)
            unique_subsets.append(list(subset))
    
    if debug:
        print(f"[QC-Biased] Generated {len(unique_subsets)} unique subsets")
        if unique_subsets:
            print("Sample subsets:", unique_subsets[:3])
    
    return unique_subsets


# Drop-in replacement wrapper
def generate_biased_prime_subsets_by_coverage_v2(prime_pool, precomputed_residues, 
                                                  vecs, num_subsets, min_size, 
                                                  max_size, combo_cap, seed=None,
                                                  force_full_pool=False, debug=False,
                                                  roots_threshold=12, rhs_list=None,
                                                  use_qc_bias=True, target_qc_ratio=1.2):
    """
    Enhanced version that can use QC bias or fall back to original coverage-only.
    
    Set use_qc_bias=True to enable QC-aware generation.
    """
    if use_qc_bias and rhs_list is not None:
        return generate_qc_biased_prime_subsets(
            prime_pool, precomputed_residues, vecs, rhs_list,
            num_subsets, min_size, max_size, combo_cap,
            seed=seed, debug=debug, roots_threshold=roots_threshold,
            target_qc_ratio=target_qc_ratio  # <-- Pass through
        )
    else:
        # Fall back to original coverage-only version
        return generate_biased_prime_subsets_by_coverage(
            prime_pool, precomputed_residues, vecs,
            num_subsets, min_size, max_size, combo_cap,
            seed=seed, force_full_pool=force_full_pool,
            debug=debug, roots_threshold=roots_threshold
        )


def compute_adaptive_num_subsets(fiber_collision_fraction, avg_density, 
                                 target_coverage=0.40, base_subsets=NUM_PRIME_SUBSETS):
    """
    Dynamically size NUM_SUBSETS based on surface geometry.
    
    Args:
        fiber_collision_fraction: Fraction of primes with fiber collisions (0.0 to 1.0)
        avg_density: Average residue density across primes (typically 0.02 to 0.15)
        target_coverage: Desired m-space coverage (default 40%)
        base_subsets: Baseline for "normal" surfaces (default 500)
    
    Returns:
        int: Recommended number of prime subsets
    
    Theory:
        - Higher density → easier to find points → need fewer subsets
        - More collisions → less reachable space → need more subsets
        - Formula balances these competing factors
    """
    # Reachable fraction of m-space (after accounting for collisions)
    reachable = 1.0 - fiber_collision_fraction
    if reachable < 0.5:
        reachable = 0.5  # Floor at 50% to avoid explosion
    
    # --- FIX: These factors should be independent ---
    # The old logic incorrectly mixed them, double-counting the 'reachable' term.

    # Density factor: higher density → fewer subsets needed
    # Reference: 0.08 is "normal" density
    density_to_use = max(0.02, avg_density) # Floor to avoid explosion
    density_factor = 0.8 / density_to_use
    
    # Reachability factor: lower reachability → more subsets needed
    # Reference: 0.80 is "normal" reachability (20% collision rate)
    reachability_factor = 0.95 / reachable
    
    # Combined adjustment
    adjusted = base_subsets * density_factor * reachability_factor
    
    print("density factor", density_factor)
    print("reachability factor", reachability_factor)
    
    # Clamp to reasonable range [100, 2000]
    adjusted = max(100, min(2000, adjusted))
    
    return int(adjusted)
