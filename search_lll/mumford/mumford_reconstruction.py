from functools import lru_cache
from sage.all import QQ, ZZ
#from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from ..rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .mumford_verification import verify_mumford_pair, canonicalize_and_dedup, discriminant_has_nonqr_s_p
from itertools import product, islice
from .mumford_timing import *
from .mumford_basis import *
from search_lll.smoothness import *
from .selmer_local_pipeline import *
from .selmer_tools import *
import multiprocessing

import itertools
import time
import traceback


def reconstruct_mumford_combo_fast(sol_combo, primes, M, max_height):
    """
    Fast reconstruction of a single combination with early rejection.
    
    Returns (s, p, v0, v1) or raises RationalReconstructionError early.
    """
    rec_vals = []
    
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        
        # Reconstruct with BOTH numerator and denominator bounds
        #num, den = rational_reconstruct(crt_val, M, max_den=max_height)
        num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
        
        rec_vals.append(QQ(num) / QQ(den))
    
    return rec_vals


def rational_reconstruct_fast(c, N, max_den=None, max_num=None):
    """
    Fast rational reconstruction with early height rejection.
    
    Fails fast if height bounds will be violated, avoiding unnecessary computation.
    Returns (num, den) or raises RationalReconstructionError.
    
    Args:
        c: integer to reconstruct
        N: modulus
        max_den: maximum allowed denominator (default: floor(sqrt(N/2)))
        max_num: maximum allowed numerator (if None, no numerator check)
    """
    if max_den is None:
        max_den = floor(sqrt(N / QQ(2)))
    
    c = c % N
    if c == 0:
        return 0, 1
    if c == 1 and max_den >= 1:
        return 1, 1
    
    # Early rejection: if c is too large, it's likely to produce large numerators
    # This is a heuristic but catches many cases
    if max_num is not None and c > max_num and (N - c) > max_num:
        raise RationalReconstructionError(f"CRT value too large: c={c}")
    
    # Extended Euclidean Algorithm with early termination
    r0, r1 = N, c
    t0, t1 = 0, 1
    
    while r1 != 0:
        # Check denominator bound BEFORE next iteration
        if abs(t1) > max_den:
            a, b = r0, t0
            break
        
        # Early numerator check during iteration
        if max_num is not None and abs(r1) > max_num:
            raise RationalReconstructionError(f"Numerator exceeds bound: {abs(r1)} > {max_num}")
        
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    else:
        # Loop finished because r1 == 0
        a, b = r0, t0
    
    # Final validation
    if abs(b) > max_den or b == 0:
        raise RationalReconstructionError(f"No reconstruction for c={c}, N={N}, max_den={max_den}")
    
    if b < 0:
        a, b = -a, -b
    
    # Final numerator check
    if max_num is not None and abs(a) > max_num:
        raise RationalReconstructionError(f"Numerator exceeds bound: {abs(a)} > {max_num}")
    
    if (a - c * b) % N != 0:
        raise RationalReconstructionError(f"Validation failed for c={c}, N={N}: got a={a}, b={b}")
    
    g = gcd(abs(a), abs(b))
    return int(a // g), int(b // g)


@lru_cache
def rational_reconstruct_with_height_check(crt_val, M, max_height):
    """
    Rational reconstruction with immediate height rejection.
    Returns (num, den) or raises RationalReconstructionError.
    """
    # Use standard denominator bound for reconstruction
    max_den = floor(sqrt(M / QQ(2)))
    num, den = rational_reconstruct(crt_val, M, max_den=max_den)
    
    # Then check BOTH against the actual height limit
    if abs(num) > max_height or abs(den) > max_height:
        raise RationalReconstructionError("Height too large")
    
    return num, den

def setup_crt_constants(primes):
    """
    Precompute weights for fast CRT: result = sum(val_i * w_i) % M.
    Returns (M, weights).
    """
    M = 1
    for p in primes:
        M *= p
    
    weights = []
    for p in primes:
        m_i = M // p
        # inverse of m_i mod p
        # use python int pow to ensure ValueError on failure, though m_i is coprime to p by definition
        inv = pow(int(m_i), -1, int(p))
        w_i = (m_i * inv)
        weights.append(w_i)
        
    return M, weights

def fast_rational_reconstruct_check(val, M, max_height):
    """
    Pure integer rational reconstruction check. 
    Returns (True, num, den) or (False, 0, 0).
    Optimized for tight loops: avoids object creation and returns early.
    """
    r0, r1 = M, val
    t0, t1 = 0, 1
    
    # Unrolled Euclidean Algorithm
    while r1 > max_height:
        if t1 > max_height or t1 < -max_height:
            return False, 0, 0
            
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
        
    if abs(t1) > max_height:
        return False, 0, 0
        
    if t1 < 0:
        t1 = -t1
        r1 = -r1
        
    if abs(r1) > max_height:
        return False, 0, 0
        
    return True, r1, t1

def batch_crt_for_combo(sol_combo, primes):
    """
    Compute CRT for all 4 coordinates at once.
    Returns list of 4 CRT values.
    """
    crt_vals = []
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        crt_vals.append(crt_val)
    return crt_vals

def prefilter_solutions_by_discriminant(sol_lists, primes):
    """
    Filter solution combinations that can't possibly satisfy s^2 - 4p >= 0 mod all primes.
    This eliminates many impossible combinations early.
    
    Returns: filtered generator of solution combinations
    """
    for sol_combo in itertools.product(*sol_lists):
        # Quick discriminant check mod each prime
        all_good = True
        for i, p in enumerate(primes):
            s_mod = sol_combo[i][0] % p
            p_mod = sol_combo[i][1] % p
            disc_mod = (s_mod * s_mod - 4 * p_mod) % p
            
            # If discriminant is negative mod p and p > 2, skip
            # (This is a quick heuristic, not perfect)
            if p > 2 and disc_mod != 0:
                # Check if disc_mod is a quadratic residue
                if pow(disc_mod, (p - 1) // 2, p) == p - 1:
                    all_good = False
                    break
        
        if all_good:
            yield sol_combo


#


def prefilter_solutions_algebraic(sol_list, prime, f_coeffs):
    """
    Filter solutions by algebraic constraint mod p BEFORE CRT.
    This eliminates ~83% of invalid combinations early.
    
    Returns: list of solutions that pass verify_mumford_pair mod p
    """
    from sage.all import GF, PolynomialRing
    
    R = PolynomialRing(GF(prime), 'x')
    x = R.gen()
    
    # Build f(x) mod p
    f_poly_coeffs = [int(c) % prime for c in f_coeffs]
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    filtered = []
    for sol in sol_list:
        s_val, p_val, v0_val, v1_val = [int(v) % prime for v in sol]
        
        # Build u(x) = x² - s*x + p
        u_poly = x**2 - s_val*x + p_val
        
        # Build v(x) = v1*x + v0
        v_poly = v1_val*x + v0_val
        
        # Check: v(x)² ≡ f(x) (mod u(x))
        diff = v_poly**2 - f_poly
        remainder = diff % u_poly
        
        if remainder.is_zero():
            filtered.append(sol)
    
    return filtered

def _reconstruct_worker_parallel_v2(args):
    """
    Optimized worker with robust error handling for modular inverses.
    """
    # Unpack the new v_tuple argument at the end
    combo_batch, primes, M_in, f_coeffs, max_height, v_tuple = args
    
    # Setup fast CRT constants once per batch
    M, weights = setup_crt_constants(primes)
    
    results = []
    stats = {
        'attempted': len(combo_batch),
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }
    
    # Pre-calculate prime integers to avoid sage overhead in loop
    primes_int = [int(p) for p in primes]
    range_primes = range(len(primes))
    
    for sol_combo in combo_batch:
        # 1. Reconstruct 's' (index 0)
        crt_s = 0
        for i in range_primes:
            crt_s += sol_combo[i][0] * weights[i]
        crt_s %= M
        
        success_s, num_s, den_s = fast_rational_reconstruct_check(crt_s, M, max_height)
        if not success_s:
            stats['height_reject'] += 1
            continue
            
        # 2. Reconstruct 'p' (index 1)
        crt_p = 0
        for i in range_primes:
            crt_p += sol_combo[i][1] * weights[i]
        crt_p %= M
        
        success_p, num_p, den_p = fast_rational_reconstruct_check(crt_p, M, max_height)
        if not success_p:
            stats['height_reject'] += 1
            continue

        # 3. Reconstruct v0 (index 2)
        crt_v0 = 0
        for i in range_primes:
            crt_v0 += sol_combo[i][2] * weights[i]
        crt_v0 %= M
        
        success_v0, num_v0, den_v0 = fast_rational_reconstruct_check(crt_v0, M, max_height)
        if not success_v0:
            stats['height_reject'] += 1
            continue
            
        # 4. Reconstruct v1 (index 3)
        crt_v1 = 0
        for i in range_primes:
            crt_v1 += sol_combo[i][3] * weights[i]
        crt_v1 %= M
        
        success_v1, num_v1, den_v1 = fast_rational_reconstruct_check(crt_v1, M, max_height)
        if not success_v1:
            stats['height_reject'] += 1
            continue

        # 5. Consistency Check (Mod P)
        reconstruction_ok = True
        
        # We work with python ints to avoid Sage ZeroDivisionErrors on mod invert
        for i in range_primes:
            p_int = primes_int[i]
            expected = sol_combo[i]
            
            try:
                # Use pow(val, -1, mod) which raises ValueError on failure
                
                # Check s
                if (num_s * pow(den_s, -1, p_int)) % p_int != expected[0]:
                    reconstruction_ok = False; break
                
                # Check p
                if (num_p * pow(den_p, -1, p_int)) % p_int != expected[1]:
                    reconstruction_ok = False; break
                    
                # Check v0
                if (num_v0 * pow(den_v0, -1, p_int)) % p_int != expected[2]:
                    reconstruction_ok = False; break
                    
                # Check v1
                if (num_v1 * pow(den_v1, -1, p_int)) % p_int != expected[3]:
                    reconstruction_ok = False; break

            except (ValueError, ZeroDivisionError):
                # Denominator divisible by prime -> reconstruction failed
                reconstruction_ok = False
                break
        
        if not reconstruction_ok:
            stats['consistency_reject'] += 1
            continue

        # 6. Convert to Sage types for algebraic verification
        s_qq = QQ(num_s) / QQ(den_s)
        p_qq = QQ(num_p) / QQ(den_p)
        v0_qq = QQ(num_v0) / QQ(den_v0)
        v1_qq = QQ(num_v1) / QQ(den_v1)

        # 7. Algebraic Verification
        if not verify_mumford_pair(f_coeffs, s_qq, p_qq, v0_qq, v1_qq, modulus=None, debug_first_failure=False):
            stats['algebraic_reject'] += 1
            continue
        
        # Attach the vector here so it survives the return trip
        results.append({'s': s_qq, 'p': p_qq, 'v_0': v0_qq, 'v_1': v1_qq, 'vector': v_tuple})
        stats['success'] += 1
    
    return results, stats

def quick_dependence_check(div1, div2):
    """Check if two divisors with same u are dependent"""
    if (div1['s'], div1['p']) != (div2['s'], div2['p']):
        return False  # different u
    
    # Same u - check if v1 ≡ ±v2 (mod u)
    # For Mumford rep: v(x) = v_1*x + v_0
    # Check: (v1_1*x + v1_0) ≡ ±(v2_1*x + v2_0) (mod u(x))
    
    # Simplest: just check if coefficients are ± each other
    if (div1['v_0'] == div2['v_0'] and div1['v_1'] == div2['v_1']):
        return True  # identical
    if (div1['v_0'] == -div2['v_0'] and div1['v_1'] == -div2['v_1']):
        return True  # negatives
    
    return False  # might still be dependent, but not obviously


# Add to mumford_reconstruction.py, after the CRT reconstruction loop

def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Optimized reconstruction with batched parallel processing.

    Rewritten to:
      - run algebraic prefilter (per-prime) as before,
      - run a per-(v_tuple,x_res) fake-Selmer probe (no global mixing),
      - explore CRT lifts with rational reconstruction + canonicalization (bounded),
      - seed reconstruction with any canonicalized selmer seeds found,
      - then run the normal CRT reconstruction loop (parallel/serial) unchanged.
    """
    import time
    import math
    import itertools
    import multiprocessing
    import traceback
    from collections import defaultdict
    from itertools import islice, product

    # local helpers expected to exist in the module:
    # prefilter_solutions_algebraic, canonicalize_and_dedup,
    # build_per_prime_fp_map_from_prime_data, accumulate_and_stabilize_intersection,
    # crt_lift_fp_representatives_explore_with_recon,
    # crt_cached, rational_reconstruct, RationalReconstructionError, QQ,
    # crt_cached, fast_rational_reconstruct_check, verify_mumford_pair,
    # mumford_timers_reset, mumford_timer_add, mumford_timers_print,
    # _reconstruct_worker_parallel_v2, crt_cached, naive helpers...

    t_start_total = time.time()

    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE")
    print("="*70)

    # reset timers
    try:
        mumford_timers_reset()
    except Exception:
        pass

    found_xs = set()
    mumford_divisors_raw = []

    # Build grouping: by_vector_and_xres[v_tuple][x_res_key][prime] = list(solutions)
    t0 = time.time()
    by_vector_and_xres = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in residues:
        for v_tuple, x_res_dict in residues[p].items():
            if isinstance(x_res_dict, list):
                by_vector_and_xres[v_tuple]['unknown'][p] = x_res_dict
            elif isinstance(x_res_dict, dict):
                for x_res, sols in x_res_dict.items():
                    by_vector_and_xres[v_tuple][x_res][p] = sols
    mumford_timer_add("residue_grouping", time.time() - t0)

    # EARLY ALGEBRAIC FILTER (per-prime)
    t0 = time.time()
    total_before_filter = 0
    total_after_filter = 0
    if debug:
        print("\n=== EARLY ALGEBRAIC FILTERING (per prime) ===")

    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            for prime in list(prime_data.keys()):
                sol_list = prime_data[prime]
                total_before_filter += len(sol_list)
                try:
                    filtered = prefilter_solutions_algebraic(sol_list, prime, f_coeffs)
                except Exception:
                    # If prefilter helper fails, keep the original list (fail-safe)
                    filtered = sol_list
                total_after_filter += len(filtered)
                prime_data[prime] = filtered
                if debug and len(filtered) < len(sol_list):
                    pct = 100.0 * len(filtered) / len(sol_list) if len(sol_list) else 0.0
                    print(f"  Prime {prime}: {len(sol_list)} → {len(filtered)} sols ({pct:.1f}%)")

    filter_reduction = total_before_filter / max(1, total_after_filter)
    if debug:
        print(f"\nAlgebraic pre-filter: {total_before_filter:,} → {total_after_filter:,} ({filter_reduction:.1f}x reduction)")
    mumford_timer_add("algebraic_prefiltering", time.time() - t0)

    num_groups = sum(len(xres_groups) for xres_groups in by_vector_and_xres.values())
    print(f"Grouped into {len(by_vector_and_xres)} vectors, {num_groups} (vector,x-residue) pairs")

    total_attempted = 0
    total_stats = {
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0,
        'prefilter_reject': 0,
        'skipped_2prime': 0,
        'skipped_high_density': 0
    }

    # ----------------- per-group fake-Selmer probing (safe) -----------------
    # Collect canonicalized Selmer seeds (QQ dicts) per group to optionally seed reconstruction.
    selmer_seeds = []
    # Helpers that might exist in module space; fallback to no-op functions if missing.
    _has_build_fp = 'build_per_prime_fp_map_from_prime_data' in globals()
    _has_accumulate = 'accumulate_and_stabilize_intersection' in globals()
    _has_explorer = 'crt_lift_fp_representatives_explore_with_recon' in globals()

    # If the dedicated helpers are missing, we fall back to a conservative global probe (less preferred).
    use_per_group_probe = _has_build_fp and _has_accumulate and _has_explorer

    if not use_per_group_probe and debug:
        print("[selmer-probe] warning: per-group selmer helpers not found; falling back to cheaper global probe")

    # iterate groups
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            # build per-group prime list
            primes_grp = sorted(prime_data.keys())
            if not primes_grp:
                continue

            # per-group prime->solutions map
            prime_map_grp = {}
            for p in primes_grp:
                # dedupe and ensure int tuples
                prime_map_grp[p] = sorted(set(tuple(int(x) for x in s) for s in prime_data[p]))

            if use_per_group_probe:
                try:
                    # build per-prime fingerprint sets & maps
                    per_p_fpset, per_p_fpmap = build_per_prime_fp_map_from_prime_data(prime_map_grp, f_coeffs, None, None)
                    # stabilize the intersection across primes for this group (allow smaller min_primes)
                    intersection, history_grp, used_primes = accumulate_and_stabilize_intersection(primes_grp, per_p_fpset,
                                                                                                  min_primes=2, stable_rounds=1,
                                                                                                  show_progress=False)
                    if not intersection:
                        # nothing stable for this group
                        continue

                    # explore CRT lifts with rational reconstruction + canonicalize; bounded exploration
                    explore_res = crt_lift_fp_representatives_explore_with_recon(
                        intersection, per_p_fpmap, used_primes, f_coeffs, canonicalize_and_dedup,
                        rep_limit=4, max_combinations=2000, show_progress=False, debug_fp_failures=2,
                        rational_recon_factor=10
                    )

                    # collect accepted canonicalized candidates with provenance
                    for ent in explore_res:
                        if ent.get('accepted'):
                            for c in ent['accepted']:
                                # add provenance so we know where seed came from
                                c['_selmer_provenance'] = {'v_tuple': v_tuple, 'x_res_key': x_res_key, 'fp': ent['fp'], 'primes': used_primes}
                                selmer_seeds.append(c)
                except Exception as e:
                    if debug:
                        print(f"[selmer-group] exception for group v={v_tuple} xres={x_res_key}: {e}")
                        traceback.print_exc()
                    # continue without aborting the whole run
                    continue
            else:
                # Fallback (simple): collect per-prime fingerprints and attempt 1-shot CRT-lift
                # Keep this conservative: do not seed reconstruction unless canonicalization succeeds.
                try:
                    # Build fingerprints per prime
                    per_p_fpset = {}
                    per_p_fpmap = {}
                    for p, sols in prime_map_grp.items():
                        fps = set()
                        fp_map = {}
                        for sol in sols:
                            s_m, p_m, v0_m, v1_m = map(int, sol)
                            Delta = (s_m * s_m - 4 * p_m) % p
                            shape = 'double' if Delta == 0 else ('split' if pow(Delta, (p - 1)//2, p) == 1 else 'irr')
                            fp = (shape, int(s_m))
                            fps.add(fp)
                            fp_map.setdefault(fp, []).append((s_m, p_m, v0_m, v1_m))
                        per_p_fpset[p] = fps
                        per_p_fpmap[p] = {k: sorted(set(v)) for k, v in fp_map.items()}

                    # crude intersection across primes (one-shot)
                    all_fps = list(per_p_fpset.values())
                    if not all_fps:
                        continue
                    inter = set.intersection(*all_fps)
                    if not inter:
                        continue

                    # For each fp, attempt to pick first representatives and try rational reconstruct+canonicalize
                    for fp in sorted(inter):
                        reps_per_prime = []
                        for p in primes_grp:
                            reps = per_p_fpmap.get(p, {}).get(fp, [])
                            if not reps:
                                reps_per_prime = []
                                break
                            reps_per_prime.append(reps[:2])  # small try
                        if not reps_per_prime:
                            continue
                        # try small number combos
                        tried = 0
                        accepted = []
                        for combo in islice(product(*reps_per_prime), 200):
                            tried += 1
                            vals_s = [int(rep[0]) for rep in combo]
                            vals_p = [int(rep[1]) for rep in combo]
                            vals_v0 = [int(rep[2]) for rep in combo]
                            vals_v1 = [int(rep[3]) for rep in combo]
                            try:
                                S_int, M = crt_cached(tuple(vals_s), tuple(primes_grp)), None
                                # Try rational_reconstruct on each coordinate using your helper if available
                                if 'rational_reconstruct' in globals():
                                    try:
                                        num_s, den_s = rational_reconstruct(int(S_int), math.prod(primes_grp))
                                        # quick QQ cast and check
                                        s_q = QQ(num_s)/QQ(den_s)
                                    except Exception:
                                        continue
                                    # construct candidate and canonicalize
                                    try:
                                        canonical = canonicalize_and_dedup([{'s': s_q, 'p': 0, 'v_0': 0, 'v_1': 0}], f_coeffs, show_progress=False)
                                        if canonical:
                                            for c in canonical:
                                                c['_selmer_provenance'] = {'v_tuple': v_tuple, 'x_res_key': x_res_key, 'fp': fp, 'primes': primes_grp}
                                                selmer_seeds.append(c)
                                                accepted.append(c)
                                                break
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                        # end combos
                except Exception:
                    if debug:
                        print(f"[selmer-fallback] exception for group v={v_tuple} xres={x_res_key}")
                        traceback.print_exc()
                    continue
    # dedupe selmer seeds
    unique_seeds = {}
    for s in selmer_seeds:
        key = (str(s.get('u_poly','')), str(s.get('v_poly','')), str(s.get('s')), str(s.get('p')))
        unique_seeds[key] = s
    selmer_seeds = list(unique_seeds.values())

    if selmer_seeds:
        print(f"\n[selmer] collected {len(selmer_seeds)} canonicalized selmer-seed divisors from groups")
        if debug:
            for s in selmer_seeds[:10]:
                print("  seed:", {k: s[k] for k in ('s','p','v_0','v_1') if k in s}, "prov:", s.get('_selmer_provenance'))

    # Optionally seed the reconstruction with selmer seeds (this helps avoid missing true global classes)
    # We append them into mumford_divisors_raw now so they survive canonicalization/dedup later.
    # Seeds are canonicalized QQ dicts; convert to expected raw shape with a dummy vector (vector will be fixed later)
    for seed in selmer_seeds:
        try:
            s_q = seed['s']; p_q = seed['p']; v0_q = seed.get('v_0', QQ(0)); v1_q = seed.get('v_1', QQ(0))
            mumford_divisors_raw.append({'vector': seed.get('_selmer_provenance', {}).get('v_tuple', (0,)),
                                         's': s_q, 'p': p_q, 'v_0': v0_q, 'v_1': v1_q})
        except Exception:
            continue

    # Add this section right after the per-group fake-Selmer probing section
    # and before building work items for CRT reconstruction

    # ----------------- GLOBAL fake-Selmer rank upper bound -----------------
    print("\n=== COMPUTING GLOBAL FAKE-SELMER RANK UPPER BOUND ===")

    # Build a unified prime->solutions map across all groups for global analysis
    global_prime_data = {}
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            for prime, sols in prime_data.items():
                if prime not in global_prime_data:
                    global_prime_data[prime] = []
                global_prime_data[prime].extend(sols)

    # Dedupe solutions per prime
    for prime in global_prime_data:
        global_prime_data[prime] = sorted(set(tuple(int(x) for x in s) for s in global_prime_data[prime]))

    if global_prime_data:
        try:
            # Use the selmer tools to compute global upper bound
            per_prime_fpset, per_prime_fp_map = build_per_prime_fp_map_from_prime_data(
                global_prime_data, f_coeffs, None, None
            )

            prime_list_sorted = sorted(global_prime_data.keys())
            intersection, history, primes_used = accumulate_and_stabilize_intersection(
                prime_list_sorted, per_prime_fpset,
                min_primes=2, stable_rounds=1, show_progress=debug
            )

            selmer_size = len(intersection)
            print(f"\nGlobal fake-Selmer set size: {selmer_size}")

            if selmer_size > 0:
                if (selmer_size & (selmer_size - 1)) == 0:
                    # Power of 2
                    ub_rank = int(math.log2(selmer_size))
                    print(f"Global fake-Selmer rank upper bound: {ub_rank}")
                else:
                    # Not power of 2, give floor estimate
                    ub_rank_floor = int(math.floor(math.log2(selmer_size)))
                    print(f"Global fake-Selmer size {selmer_size} (not power of 2)")
                    print(f"Floor(log2(|S|)) = {ub_rank_floor}")
            else:
                print("Global fake-Selmer set is empty")

            if debug and history:
                print("\nStabilization history (prime, intersection_size):")
                for p, size in history:
                    print(f"  Prime {p}: {size}")

        except Exception as e:
            print(f"Global fake-Selmer computation failed: {e}")
            if debug:
                traceback.print_exc()
    else:
        print("No global prime data available for fake-Selmer computation")

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Now build work items and run CRT reconstruction loop (keeps most of your original logic)
    t0 = time.time()
    num_workers = max(8, multiprocessing.cpu_count())
    all_work_items = []

    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            # note: prime_data mutated earlier by algebraic prefilter
            primes = sorted(prime_data.keys())
            sol_lists = [prime_data[p] for p in primes]

            # local fingerprint prune (cheap): keep only solutions whose (shape,s_mod) survive intersection
            try:
                if len(primes) >= 2:
                    per_prime_fps = []
                    per_prime_fp_map = []
                    for p, sl in zip(primes, sol_lists):
                        fps = set()
                        fp_map = []
                        for sol in sl:
                            s_mod = int(sol[0]) % p
                            p_mod = int(sol[1]) % p
                            Delta = (s_mod * s_mod - 4 * p_mod) % p
                            if Delta == 0:
                                shape = 'double'
                            else:
                                leg = pow(Delta, (p - 1) // 2, p)
                                shape = 'split' if leg == 1 else 'irr'
                            fp = (shape, int(s_mod))
                            fps.add(fp)
                            fp_map.append(fp)
                        per_prime_fps.append(fps)
                        per_prime_fp_map.append(fp_map)
                    intersection = set.intersection(*per_prime_fps) if per_prime_fps else set()
                    if intersection and any(len(intersection) < len(fps) for fps in per_prime_fps):
                        new_sol_lists = []
                        for sl, fp_map in zip(sol_lists, per_prime_fp_map):
                            filtered = [sol for sol, fp in zip(sl, fp_map) if fp in intersection]
                            new_sol_lists.append(filtered)
                        sol_lists = new_sol_lists
                        if debug:
                            old_total = 1
                            new_total = 1
                            for sl_old, sl_new in zip(prime_data.values(), sol_lists):
                                old_total *= max(1, len(sl_old)); new_total *= max(1, len(sl_new))
                            print(f"  [selmer-prune] primes={primes}: combos {old_total:,} → {new_total:,} by fingerprint intersection (kept {len(intersection)} fps)")
            except Exception as _sel_err:
                if debug:
                    print(f"  [selmer-prune] exception, skipping local selmer prune: {_sel_err}")

            # Skip if empty
            if any(len(sl) == 0 for sl in sol_lists):
                if debug:
                    print(f"  Skipping group: algebraic filter eliminated all solutions")
                continue

            total_combos_raw = 1
            for sl in sol_lists:
                total_combos_raw *= len(sl)

            # heuristics for 2-prime groups / densities (copied from original)
            if len(primes) == 2:
                is_rare = (len(xres_groups) <= 5)
                if total_combos_raw > 100000 and not is_rare:
                    if debug:
                        print(f"  Skipping 2-prime group: {total_combos_raw:,} combos, not rare")
                    total_stats['skipped_2prime'] += 1
                    continue
                elif debug and total_combos_raw > 50000:
                    print(f"  Accepting 2-prime group: {total_combos_raw:,} combos (rare divisor)")

            if len(primes) < 2:
                continue

            M = 1
            for p in primes:
                M *= p

            avg_sols_per_prime = sum(len(sl) for sl in sol_lists) / len(sol_lists) if sol_lists else 0

            if debug and total_combos_raw > 10000:
                print(f"  Vector group: {len(primes)} primes, avg {avg_sols_per_prime:.1f} sols/prime")
                print(f"  Raw combinations: {total_combos_raw:,}")

            # aggressive prefilter if very dense (copied from original)
            if total_combos_raw > 500000 and len(primes) >= 3:
                if debug:
                    print(f"  High density detected - applying aggressive pre-filtering")
                t_prefilter = time.time()
                p0, p1 = primes[0], primes[1]
                M_small = p0 * p1
                candidate_divisors = []
                max_candidates = 50000
                tried = 0
                for sol0 in sol_lists[0]:
                    for sol1 in sol_lists[1]:
                        tried += 1
                        if tried > max_candidates:
                            break
                        rec_vals = []
                        for idx in range(4):
                            vals = (sol0[idx], sol1[idx])
                            crt_val = crt_cached(vals, (p0, p1))
                            try:
                                num, den = rational_reconstruct(crt_val, M_small)
                                if abs(num) > 10 * M_small or abs(den) > 10 * M_small:
                                    raise RationalReconstructionError("Height too large")
                                rec_vals.append(QQ(num)/QQ(den))
                            except RationalReconstructionError:
                                break
                        if len(rec_vals) == 4:
                            s, p_val, v0, v1 = rec_vals
                            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                                candidate_divisors.append((s, p_val, v0, v1))
                    if tried > max_candidates:
                        break

                if debug:
                    print(f"    Found {len(candidate_divisors)} candidates from first 2 primes (tried {tried:,})")

                if candidate_divisors:
                    filtered_sol_lists = [sol_lists[0], sol_lists[1]]
                    for p_idx in range(2, len(primes)):
                        p = primes[p_idx]
                        filtered = []
                        for sol in sol_lists[p_idx]:
                            for cand_s, cand_p, cand_v0, cand_v1 in candidate_divisors:
                                try:
                                    s_mod = (int(cand_s.numerator()) * pow(int(cand_s.denominator()), -1, p)) % p
                                    p_mod = (int(cand_p.numerator()) * pow(int(cand_p.denominator()), -1, p)) % p
                                    v0_mod = (int(cand_v0.numerator()) * pow(int(cand_v0.denominator()), -1, p)) % p
                                    v1_mod = (int(cand_v1.numerator()) * pow(int(cand_v1.denominator()), -1, p)) % p
                                    if (sol[0] == s_mod and sol[1] == p_mod and sol[2] == v0_mod and sol[3] == v1_mod):
                                        filtered.append(sol)
                                        break
                                except ZeroDivisionError:
                                    continue
                        if not filtered:
                            filtered = sol_lists[p_idx]
                        filtered_sol_lists.append(filtered)
                        if debug and len(filtered) != len(sol_lists[p_idx]):
                            pct = 100.0 * len(filtered) / len(sol_lists[p_idx])
                            print(f"    Prime {p}: {len(sol_lists[p_idx])} → {len(filtered)} sols ({pct:.1f}%)")
                    sol_lists = filtered_sol_lists

                total_combos_filtered = 1
                for sl in sol_lists:
                    total_combos_filtered *= len(sl)
                prefilter_reduction = total_combos_raw / max(1, total_combos_filtered)
                total_stats['prefilter_reject'] += (total_combos_raw - total_combos_filtered)
                if debug:
                    print(f"  Pre-filtering took {time.time() - t_prefilter:.2f}s")
                    print(f"  Filtered combinations: {total_combos_filtered:,} ({prefilter_reduction:.1f}x reduction)")
                mumford_timer_add("prefiltering", time.time() - t_prefilter)

            # build work item
            total_combos = 1
            for sl in sol_lists:
                total_combos *= len(sl)
            if total_combos > 5_000_000:
                if debug:
                    print(f"  Skipping group - too many combinations after filtering ({total_combos:,})")
                total_stats['skipped_high_density'] += 1
                continue

            disc_deg = len(f_coeffs) - 1
            expected_rank = min(disc_deg - 1, 4)
            base_limit = 1000000 * expected_rank
            adaptive_limit = min(base_limit, 500_000, total_combos)
            if debug and total_combos > 10000:
                print(f"  Adaptive limit: {adaptive_limit:,} (total combos: {total_combos:,})")

            if len(primes) == 2:
                max_height = int(M ** 0.6)
            else:
                max_height = int(M ** 0.5)

            all_work_items.append({
                'v_tuple': v_tuple,
                'primes': primes,
                'M': M,
                'sol_lists': sol_lists,
                'max_height': max_height,
                'adaptive_limit': adaptive_limit,
                'total_combos': total_combos
            })

    # If no work items and we already have seeds, short-circuit to canonicalization & basis build
    if not all_work_items:
        print("  No work items to process")
        mumford_timer_add("crt_reconstruction_loop", time.time() - t0)
        # canonicalize existing seeds/raw divisors and return
        if not mumford_divisors_raw:
            # nothing found
            mumford_timers_print()
            return found_xs, []
        t0 = time.time()
        try:
            mumford_divisors_raw = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)
        except Exception:
            # fallback: trust existing raw list
            pass
        mumford_timer_add("canonicalization", time.time() - t0)
        # proceed to follow-on logic (same as below)
        # (duplicate of later cleanup code)
        mumford_divisors = []
        for i, divi in enumerate(mumford_divisors_raw):
            is_dep = False
            for j, divj in enumerate(mumford_divisors_raw):
                if i <= j:
                    continue
                if quick_dependence_check(divi, divj):
                    is_dep = True
            if not is_dep:
                mumford_divisors.append(divi)
        t0 = time.time()
        for div in mumford_divisors:
            s, p_val = div['s'], div['p']
            try:
                disc = s*s - 4*p_val
            except Exception:
                div['has_rational_roots'] = False
                continue
            if disc >= 0 and disc.is_square():
                div['has_rational_roots'] = True
                r1 = (s + disc.sqrt())/2
                r2 = (s - disc.sqrt())/2
                for r in (r1, r2):
                    x_cand = r - shift
                    if rationality_test(x_cand) is not None:
                        found_xs.add(x_cand)
            else:
                div['has_rational_roots'] = False
        mumford_timer_add("rational_root_check", time.time() - t0)
        print(f"  Unique Rational Points: {len(found_xs)}")
        # finish with basis building if possible
        if mumford_divisors:
            unique = {frozenset(d.items()): d for d in mumford_divisors}
            mumford_divisors = list(unique.values())
            def naive_sort_key(d):
                return abs(QQ(d['s'])) + abs(QQ(d['p'])) + abs(QQ(d['v_0'])) + abs(QQ(d['v_1']))
            mumford_divisors.sort(key=naive_sort_key)
            try:
                t0 = time.time()
                basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(mumford_divisors, f_coeffs, debug=True)
                mumford_timer_add("basis_construction", time.time() - t0)
                print(f"\nBasis Construction Results:")
                print(f"  Found {basis_rank} independent divisors")
                if basis_H is not None:
                    print(f"  Determinant (float): {float(basis_H.determinant())}")
                mumford_timers_print()
                return found_xs, basis_divisors
            except Exception as e:
                print(f"Basis construction failed: {e}")
                traceback.print_exc()
                mumford_timers_print()
                raise
        mumford_timers_print()
        return found_xs, mumford_divisors

    # Prepare CRT batches and run reconstruction (parallel or serial)
    total_work = sum(item['total_combos'] for item in all_work_items)
    use_parallel = total_work > 100000

    if use_parallel:
        if debug:
            print(f"\n  Batched parallel processing: {len(all_work_items)} groups, {total_work:,} total combos")
            print(f"  Using {num_workers} workers")
        all_batches = []
        for item in all_work_items:
            limit = min(item['adaptive_limit'], item['total_combos'])
            all_combos = list(islice(itertools.product(*item['sol_lists']), limit))
            batch_size = max(5000, len(all_combos) // (num_workers * 2))
            for i in range(0, len(all_combos), batch_size):
                batch = all_combos[i:i+batch_size]
                all_batches.append((
                    batch,
                    item['primes'],
                    item['M'],
                    f_coeffs,
                    item['max_height'],
                    item['v_tuple']
                ))
        if debug:
            print(f"  Created {len(all_batches)} batches for parallel processing")
        try:
            ctx = multiprocessing.get_context("fork")
            pool = ctx.Pool(num_workers)
        except Exception:
            pool = multiprocessing.Pool(num_workers)
        try:
            for batch_results, batch_stats in pool.imap_unordered(_reconstruct_worker_parallel_v2, all_batches):
                for div in batch_results:
                    mumford_divisors_raw.append(div)
                total_stats['height_reject'] += batch_stats.get('height_reject', 0)
                total_stats['consistency_reject'] += batch_stats.get('consistency_reject', 0)
                total_stats['algebraic_reject'] += batch_stats.get('algebraic_reject', 0)
                total_stats['success'] += batch_stats.get('success', 0)
                total_attempted += batch_stats.get('attempted', 0)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
    else:
        if debug:
            print(f"\n  Serial processing: {len(all_work_items)} groups, {total_work:,} total combos")
        first_diagnostic_done = False
        for item in all_work_items:
            primes = item['primes']
            M = item['M']
            sol_lists = item['sol_lists']
            max_height = item['max_height']
            v_tuple = item['v_tuple']
            limit = min(item['adaptive_limit'], item['total_combos'])
            for sol_combo in islice(itertools.product(*sol_lists), limit):
                total_attempted += 1
                try:
                    rec_vals = []
                    for idx in range(4):
                        vals = [sol[idx] for sol in sol_combo]
                        crt_val = crt_cached(tuple(vals), tuple(primes))
                        num, den = rational_reconstruct(crt_val, M)
                        if abs(num) > max_height or abs(den) > max_height:
                            raise RationalReconstructionError("Height too large")
                        rec_vals.append(QQ(num)/QQ(den))
                    s, p_val, v0, v1 = rec_vals
                    if not first_diagnostic_done and debug:
                        print(f"\n=== FIRST RECONSTRUCTION DIAGNOSTIC ===")
                        print(f"Reconstructed: s={s}, p={p_val}, v0={v0}, v1={v1}")
                        print(f"Heights: |num(s)|={abs(s.numerator())}, |den(s)|={abs(s.denominator())}")
                        print(f"         |num(p)|={abs(p_val.numerator())}, |den(p)|={abs(p_val.denominator())}")
                        print(f"Max height allowed: {max_height}")
                        print(f"M = {M} ({len(primes)} primes)")
                        first_diagnostic_done = True
                except RationalReconstructionError:
                    total_stats['height_reject'] += 1
                    continue
                reconstruction_ok = True
                for i, prime in enumerate(primes):
                    expected_sol = sol_combo[i]
                    try:
                        s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, prime)) % prime
                        p_mod = (int(p_val.numerator()) * pow(int(p_val.denominator()), -1, prime)) % prime
                        v0_mod = (int(v0.numerator()) * pow(int(v0.denominator()), -1, prime)) % prime
                        v1_mod = (int(v1.numerator()) * pow(int(v1.denominator()), -1, prime)) % prime
                    except ZeroDivisionError:
                        reconstruction_ok = False
                        break
                    if (s_mod != expected_sol[0] % prime or
                        p_mod != expected_sol[1] % prime or
                        v0_mod != expected_sol[2] % prime or
                        v1_mod != expected_sol[3] % prime):
                        reconstruction_ok = False
                        break
                if not reconstruction_ok:
                    total_stats['consistency_reject'] += 1
                    continue
                if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                    if total_stats['algebraic_reject'] == 0 and debug:
                        print(f"\n=== FIRST ALGEBRAIC REJECTION ===")
                        print(f"s={s}, p={p_val}, v0={v0}, v1={v1}")
                        print(f"Re-running with debug...")
                        verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=True)
                    total_stats['algebraic_reject'] += 1
                    continue
                mumford_divisors_raw.append({
                    'vector': v_tuple, 's': s, 'p': p_val, 'v_0': v0, 'v_1': v1
                })
                total_stats['success'] += 1

    mumford_timer_add("crt_reconstruction_loop", time.time() - t0)

    # Summary
    print(f"\n=== RECONSTRUCTION SUMMARY ===")
    print(f"  Combinations tried: {total_attempted:,}")
    print(f"  Groups skipped (2-prime): {total_stats['skipped_2prime']}")
    print(f"  Groups skipped (high density): {total_stats['skipped_high_density']}")
    print(f"  Pre-filtered out: {total_stats['prefilter_reject']:,}")
    print(f"  Rejected by height: {total_stats['height_reject']:,}")
    print(f"  Rejected by consistency: {total_stats['consistency_reject']:,}")
    print(f"  Rejected by algebraic constraint: {total_stats['algebraic_reject']:,}")
    print(f"  Successful reconstructions: {total_stats['success']:,}")

    if not mumford_divisors_raw:
        print("  WARNING: No valid Mumford divisors reconstructed!")
        mumford_timers_print()
        return found_xs, []

    # Canonicalize & dedup
    t0 = time.time()
    mumford_divisors_raw = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)
    mumford_timer_add("canonicalization", time.time() - t0)

    # filter obvious dependencies
    mumford_divisors = []
    for i, divi in enumerate(mumford_divisors_raw):
        is_dep = False
        for j, divj in enumerate(mumford_divisors_raw):
            if i <= j:
                continue
            if quick_dependence_check(divi, divj):
                is_dep = True
        if not is_dep:
            mumford_divisors.append(divi)

    # rational root check & collect x's
    t0 = time.time()
    for div in mumford_divisors:
        s, p_val = div['s'], div['p']
        try:
            disc = s*s - 4*p_val
        except Exception:
            div['has_rational_roots'] = False
            continue
        if disc >= 0 and disc.is_square():
            div['has_rational_roots'] = True
            r1 = (s + disc.sqrt())/2
            r2 = (s - disc.sqrt())/2
            for r in (r1, r2):
                x_cand = r - shift
                try:
                    if rationality_test(x_cand) is not None:
                        found_xs.add(x_cand)
                except Exception:
                    # rationality_test may raise — ignore
                    pass
        else:
            div['has_rational_roots'] = False
    mumford_timer_add("rational_root_check", time.time() - t0)

    print(f"  Unique Rational Points: {len(found_xs)}")

    if mumford_divisors:
        unique = {frozenset(d.items()): d for d in mumford_divisors}
        mumford_divisors = list(unique.values())

        # sort by naive height for stability
        def naive_sort_key(d):
            return abs(QQ(d['s'])) + abs(QQ(d['p'])) + abs(QQ(d['v_0'])) + abs(QQ(d['v_1']))
        mumford_divisors.sort(key=naive_sort_key)

        rational_roots_count = sum(1 for div in mumford_divisors_raw if 'has_rational_roots' in div and div.get('has_rational_roots'))
        print(f"  {rational_roots_count} of {len(mumford_divisors_raw)} original divisors had rational roots in u(x)")
        print(f"\n--- Building Independent Mumford Basis ---")
        print("first 10 divisors:")
        for i in mumford_divisors[:10]:
            print(i)

        try:
            t0 = time.time()
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(mumford_divisors, f_coeffs, debug=True)
            mumford_timer_add("basis_construction", time.time() - t0)
            print(f"\nBasis Construction Results:")
            print(f"  Found {basis_rank} independent divisors")
            if basis_H is not None:
                try:
                    print(f"  Determinant (float): {float(basis_H.determinant())}")
                except Exception:
                    pass
            mumford_timers_print()
            return found_xs, basis_divisors
        except Exception as e:
            print(f"Basis construction failed: {e}")
            traceback.print_exc()
            mumford_timers_print()
            raise

    mumford_timers_print()
    return found_xs, mumford_divisors
