"""
ll_utilities.py: Matrix and lattice reduction helpers.
"""
from sage.all import ZZ, diagonal_matrix
from .search_config import *


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

def _trim_poly_coeffs(coeff_list, max_deg=TRUNCATE_MAX_DEG):
    """Truncate coefficient list (low->high) to length max_deg+1."""
    if len(coeff_list) <= max_deg + 1:
        return coeff_list
    # Keep low-degree coefficients (assumed stored as [c0, c1, ..., cN])
    return coeff_list[: max_deg + 1]



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

