from functools import lru_cache
from sage.all import QQ, ZZ
#from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from ..rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .mumford_verification import *
from itertools import product, islice
from .mumford_timing import *
from .mumford_basis import *
from search_lll.smoothness import *
import multiprocessing

import itertools
import time
import traceback
from search_common import DATA_PTS_GENUS2


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


from search_common import DATA_PTS_GENUS2, FINITE_FIELD
from sage.all import GF


def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Optimized reconstruction with batched parallel processing.
    """
    # === FINITE FIELD MODE ===
    if FINITE_FIELD:
        return _reconstruct_mumford_finite_field(residues, f_coeffs, shift, rationality_test, debug)
    
    # === RATIONAL (QQ) MODE ===
    t_start_total = time.time()
    
    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE")
    print("="*70)

    mumford_timers_reset()
    
    found_xs = set()
    mumford_divisors_raw = []

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


    # NEW: Apply algebraic filtering to each prime's solutions
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
                
                # FILTER: Check algebraic constraint mod p
                filtered = prefilter_solutions_algebraic(sol_list, prime, f_coeffs)
                
                total_after_filter += len(filtered)
                prime_data[prime] = filtered
                
                if debug and len(filtered) < len(sol_list):
                    pct = 100.0 * len(filtered) / len(sol_list)
                    print(f"  Prime {prime}: {len(sol_list)} → {len(filtered)} sols ({pct:.1f}%)")
    
    filter_reduction = total_before_filter / max(1, total_after_filter)
    if debug:
        print(f"\nAlgebraic pre-filter: {total_before_filter:,} → {total_after_filter:,} "
              f"({filter_reduction:.1f}x reduction)")
    
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

    t0 = time.time()
    
    num_workers = max(8, multiprocessing.cpu_count())
    
    all_work_items = []
    
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            primes = sorted(prime_data.keys())
            sol_lists = [prime_data[p] for p in primes]

            # Skip if algebraic filtering eliminated everything
            if any(len(sl) == 0 for sl in sol_lists):
                if debug:
                    print(f"  Skipping group: algebraic filter eliminated all solutions")
                continue

            total_combos_raw = 1
            for sl in sol_lists:
                total_combos_raw *= len(sl)
            
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
            
            avg_sols_per_prime = sum(len(sl) for sl in sol_lists) / len(sol_lists)
            
            if debug and total_combos_raw > 10000:
                print(f"  Vector group: {len(primes)} primes, avg {avg_sols_per_prime:.1f} sols/prime")
                print(f"  Raw combinations: {total_combos_raw:,}")
            
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
                                    
                                    if (sol[0] == s_mod and sol[1] == p_mod and 
                                        sol[2] == v0_mod and sol[3] == v1_mod):
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
    
    if not all_work_items:
        print("  No work items to process")
        mumford_timer_add("crt_reconstruction_loop", time.time() - t0)
        mumford_timers_print()
        return found_xs, []
    
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
                    item['v_tuple']  # PASS THE VECTOR TUPLE HERE
                ))
        
        if debug:
            print(f"  Created {len(all_batches)} batches for parallel processing")
        
        try:
            ctx = multiprocessing.get_context("fork")
            pool = ctx.Pool(num_workers)
        except Exception:
            pool = multiprocessing.Pool(num_workers)
            raise
        
        try:
            for batch_results, batch_stats in pool.imap_unordered(_reconstruct_worker_parallel_v2, all_batches):
                for div in batch_results:
                    # Do not override div['vector'] here; use the one from the worker
                    mumford_divisors_raw.append(div)
                
                total_stats['height_reject'] += batch_stats['height_reject']
                total_stats['consistency_reject'] += batch_stats['consistency_reject']
                total_stats['algebraic_reject'] += batch_stats['algebraic_reject']
                total_stats['success'] += batch_stats['success']
                total_attempted += batch_stats['attempted']
            
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

    t0 = time.time()
    mumford_divisors_raw = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs, seed_x_coords=DATA_PTS_GENUS2)
    print("accepted:", len(mumford_divisors_raw))
    for d in mumford_divisors_raw:
        # check parity idempotence
        assert normalize_infinity_parity(d['v_poly'], build_f_poly(f_coeffs, PolynomialRing(QQ,'x'))) == d['v_poly']
    # ensure no u duplicates modulo ±v
    for i in range(len(mumford_divisors_raw)):
        for j in range(i):
            assert mumford_divisors_raw[i]['u_poly'] != mumford_divisors_raw[j]['u_poly'] or not same_v_up_to_sign_mod_u(mumford_divisors_raw[i]['v_poly'], mumford_divisors_raw[j]['v_poly'], mumford_divisors_raw[i]['u_poly'])
    print("canonicalization invariants OK")


    mumford_timer_add("canonicalization", time.time() - t0)


    # In reconstruct_and_verify_mumford, after the reconstruction loop:

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
        disc = s*s - 4*p_val

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


    if mumford_divisors:
        unique = {frozenset(d.items()): d for d in mumford_divisors}
        mumford_divisors = list(unique.values())

        # [Fix] Deterministic Sort
        # Sort by naive height sum first, then use coefficients as tie-breakers.
        # This prevents race conditions in parallel processing from altering the truncated list.
        def deterministic_sort_key(d):
            h = abs(QQ(d['s'])) + abs(QQ(d['p'])) + abs(QQ(d['v_0'])) + abs(QQ(d['v_1']))
            return (h, QQ(d['s']), QQ(d['p']), QQ(d['v_0']), QQ(d['v_1']))
        
        mumford_divisors.sort(key=deterministic_sort_key)
        #mumford_divisors.reverse() # psych!

        rational_roots_count = sum(1 for div in mumford_divisors_raw
                                   if 'has_rational_roots' in div and div.get('has_rational_roots'))

        print(f"  {rational_roots_count} of {len(mumford_divisors_raw)} original divisors had rational roots in u(x)")
        print(f"\n--- Building Independent Mumford Basis ---")
        print("first 10 divisors:")
        for i in mumford_divisors[:10]:
            print(i)
        
        try:
            t0 = time.time()
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(
                mumford_divisors, 
                f_coeffs, 
                debug=True
            )
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


# In mumford_reconstruction.py, modify _reconstruct_mumford_finite_field:


def _ff_worker_verify_batch(args):
    """
    Worker function for parallel verification of Mumford solutions in finite field mode.
    Returns: (verified_divisors, stats)
    """
    sol_batch, f_coeffs, p, shift, v_tuple = args
    
    from sage.all import GF
    
    F = GF(p)
    verified = []
    stats = {
        'processed': len(sol_batch),
        'alg_fail': 0,
        'no_roots': 0,
        'success': 0
    }
    
    def eval_f(x_val):
        val = F(0)
        for c in f_coeffs:
            val = val * F(x_val) + F(c)
        return val
    
    def get_u_roots(s_val, p_val):
        delta = (s_val * s_val - 4 * p_val) % p
        delta_ele = F(delta)
        if not delta_ele.is_square():
            return None
        sqrt_delta = delta_ele.sqrt()
        inv_2 = F(2).inverse()
        r1 = int((F(s_val) + sqrt_delta) * inv_2)
        r2 = int((F(s_val) - sqrt_delta) * inv_2)
        return tuple(sorted((r1, r2)))
    
    for sol in sol_batch:
        s_val, p_val, v0_val, v1_val = [int(x) for x in sol]
        
        if not verify_mumford_pair(f_coeffs, s_val, p_val, v0_val, v1_val, modulus=p):
            stats['alg_fail'] += 1
            continue
        
        roots = get_u_roots(s_val, p_val)
        if roots is None:
            stats['no_roots'] += 1
            continue
        
        div_entry = {
            's': s_val, 'p': p_val, 'v_0': v0_val, 'v_1': v1_val,
            'vector': v_tuple,
            'has_rational_roots': True,
            'roots': list(roots)
        }
        verified.append(div_entry)
        stats['success'] += 1
    
    return verified, stats


import math


from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
import random
from collections import defaultdict, Counter
from search_common import FINITE_FIELD


def get_u_roots_mod_p(s_val, p_val, F):
    """
    Finds roots of u(x) = x^2 - s*x + p in F.
    Returns sorted tuple of roots or None if irreducible.
    """
    delta = (s_val * s_val - 4 * p_val)
    if not delta.is_square():
        return None
    sqrt_delta = delta.sqrt()
    inv_2 = F(2).inverse()
    r1 = int((s_val + sqrt_delta) * inv_2)
    r2 = int((s_val - sqrt_delta) * inv_2)
    return tuple(sorted((r1, r2)))

def _reconstruct_mumford_finite_field(residues, f_coeffs, shift, rationality_test, debug):
    """
    Index calculus reconstruction for finite fields.
    Collects solutions and performs random walks (addition) to fill the factor base rank.
    """
    if FINITE_FIELD is None:
        raise RuntimeError("FINITE_FIELD is not set; cannot run finite-field reconstruction.")
    
    p = int(FINITE_FIELD)
    F = GF(p)
    R = PolynomialRing(F, 'x')
    x = R.gen()
    
    # Build the Jacobian for group operations
    f_poly = R(f_coeffs[::-1])
    
    try:
        C = HyperellipticCurve(f_poly)
        J = C.jacobian()
    except Exception as e:
        print(f"Failed to initialize Jacobian: {e}")
        return set(), []

    if p not in residues:
        if debug:
            print(f"  No residues found for field characteristic {p}")
        return set(), []

    res_p = residues[p]
    
    print(f"\n=== MUMFORD SEARCH (FINITE FIELD GF({p})) ===")

    found_xs = set()
    active_pool = []
    seen_supports = set()
    unique_roots_set = set() 
    
    # 1. Harvest Initial Solutions
    print("  Harvesting initial solutions...")
    count_raw = 0
    
    for v_tuple, val in res_p.items():
        if isinstance(val, list):
            sols = val
        elif isinstance(val, dict):
            sols = [s for sublist in val.values() for s in sublist]
        else:
            sols = [val]
            
        for sol in sols:
            count_raw += 1
            s_val, p_val, v0_val, v1_val = [int(x) for x in sol]
            
            # Fast algebraic check
            u_poly = x**2 - F(s_val)*x + F(p_val)
            v_poly = F(v1_val)*x + F(v0_val)
            if (v_poly**2 - f_poly) % u_poly != 0:
                continue

            div_J = J(u_poly, v_poly)
            
            # Check smoothness
            roots = get_u_roots_mod_p(F(s_val), F(p_val), F)
            if roots is not None:
                if roots not in seen_supports:
                    seen_supports.add(roots)
                    for r in roots:
                        unique_roots_set.add(r)
                        
                    div_data = {
                        's': s_val, 'p': p_val, 'v_0': v0_val, 'v_1': v1_val,
                        'vector': v_tuple,
                        'roots': list(roots),
                        'has_rational_roots': True
                    }
                    active_pool.append((div_J, div_data))
                    
                    for r in roots:
                        x_cand = int(r) - int(shift)
                        if x_cand not in found_xs:
                            res = rationality_test(x_cand)
                            if res is not None:
                                found_xs.add(x_cand)

    print(f"  Initial harvest: {len(active_pool)} unique smooth divisors from {count_raw} candidates.")
    print(f"  Factor Base (Unique x): {len(unique_roots_set)}")
    
    if not active_pool:
        print("  No smooth divisors found initially. Cannot proceed with mixing.")
        return found_xs, []

    # 2. Random Walk / Mixing to fill Rank
    # We require a significant buffer (excess relations) to guarantee full rank for sparse matrices.
    
    MAX_ROUNDS = 150000
    patience = 0
    max_patience = 30000
    
    print("  Starting structured random walk to improve Rank/FB ratio...")
    
    generated_count = 0
    
    while generated_count < MAX_ROUNDS:
        fb_size = len(unique_roots_set)
        num_rels = len(active_pool)
        
        # Dynamic buffer: Require at least 15% excess or 200 relations, whichever is larger
        target_buffer = max(200, int(fb_size * 0.15))
        current_excess = num_rels - fb_size
        
        if num_rels > fb_size + target_buffer:
            print(f"  Target reached: {num_rels} relations > {fb_size} FB + {target_buffer} buffer.")
            break
            
        if generated_count % 1000 == 0 and generated_count > 0:
            print(f"  [Iter {generated_count}] Rel: {num_rels}, FB: {fb_size}, Excess: {current_excess}/{target_buffer}")

        # Pick random pair
        idx1 = random.randrange(len(active_pool))
        idx2 = random.randrange(len(active_pool))
        D1, _ = active_pool[idx1]
        D2, _ = active_pool[idx2]
        
        # Try both Addition and Subtraction
        ops = []
        if random.random() < 0.5:
             ops.append(D1 + D2)
        else:
             ops.append(D1 - D2)
             
        for D3 in ops:
            u3 = D3[0].monic()
            v3 = D3[1]
            coeffs_u = u3.list()
            
            if len(coeffs_u) == 3:
                s_new = -coeffs_u[1]
                p_new = coeffs_u[0]
            else:
                continue 
            
            roots = get_u_roots_mod_p(s_new, p_new, F)
            
            if roots is not None:
                # Calculate how many NEW roots this adds
                new_roots_count = 0
                for r in roots:
                    if r not in unique_roots_set:
                        new_roots_count += 1
                
                # Filter Logic:
                # 0 new roots: Excellent (Pure collision). Always accept.
                # 1 new root:  Okay (Maintains Rank/Var ratio). Accept.
                # 2 new roots: Bad (Explodes FB). Reject.
                
                if new_roots_count <= 1:
                    if roots not in seen_supports:
                        seen_supports.add(roots)
                        for r in roots:
                            unique_roots_set.add(r)
                            
                        # Extract v
                        coeffs_v = v3.list()
                        v1_new = coeffs_v[1] if len(coeffs_v) >= 2 else F(0)
                        v0_new = coeffs_v[0] if len(coeffs_v) >= 1 else F(0)

                        new_data = {
                            's': int(s_new), 'p': int(p_new), 
                            'v_0': int(v0_new), 'v_1': int(v1_new),
                            'vector': 'mixed', 
                            'roots': list(roots),
                            'has_rational_roots': True
                        }
                        
                        active_pool.append((D3, new_data))
                        generated_count += 1
                        patience = 0
                        
                        # Rational point check (if cheap/needed)
                        for r in roots:
                            x_cand = int(r) - int(shift)
                            if x_cand not in found_xs:
                                res = rationality_test(x_cand)
                                if res is not None:
                                    found_xs.add(x_cand)
                    else:
                        patience += 1
                else:
                    patience += 1
            else:
                patience += 1
        
        if patience > max_patience:
            print("  Max patience reached (mining exhausted). Stopping.")
            break

    mumford_divisors = [d for _, d in active_pool]
    
    # Final diagnostic
    all_roots = set()
    for d in mumford_divisors:
        all_roots.update(d['roots'])
    print(f"  Final Status: {len(mumford_divisors)} relations, {len(all_roots)} factor base elements.")
    
    return found_xs, mumford_divisors
