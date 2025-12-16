import multiprocessing
import signal
from tqdm import tqdm
from .mumford_solver import solve_mumford_mod_p_optimized
from .mumford_verification import verify_mumford_pair, discriminant_has_nonqr_s_p
from .mumford_reconstruction import setup_crt_constants, fast_rational_reconstruct_check
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL
import time
import sys
import traceback
from sage.all import GF, QQ
from .mumford_timing import mumford_timer_add


def mumford_precompute_residues_sequential(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                           rhs_modp_list, vecs_list, debug=DEBUG):
    """
    Sequential fallback: runs the parallel routine with a single worker.
    """
    print("Sequential fallback is using the parallel routine with a single worker.")
    return mumford_precompute_residues_parallel(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                                rhs_modp_list, vecs_list, num_workers=1, debug=debug)

def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _solve_worker_wrapper(args):
    p, f_coeffs_ints, x_residues_map, const_val_int = args
    try:
        p_results = {}
        for v_tuple, x_res_list in x_residues_map.items():
            if isinstance(x_res_list, int):
                x_res_list = [x_res_list]
            
            x_res_to_sols = {}
            for x_res in x_res_list:
                sols = solve_mumford_mod_p_optimized(f_coeffs_ints, p, x_res, const_val_int)
                
                verified_sols = []
                for sol in sols:
                    s, p_val, v0, v1 = sol
                    if verify_mumford_pair(f_coeffs_ints, s, p_val, v0, v1, modulus=p):
                        verified_sols.append(sol)
                
                if verified_sols:
                    x_res_to_sols[x_res] = verified_sols
            
            if x_res_to_sols:
                p_results[v_tuple] = x_res_to_sols
                
        return p, p_results
    except Exception:
        sys.stderr.write(f"\nCRITICAL ERROR IN MUMFORD WORKER (p={p}):\n")
        traceback.print_exc(file=sys.stderr)
        raise
        return p, {}

def _mumford_worker_entry(args):
    """Legacy entry point (placeholder)."""
    # NOTE: The provided code only used _solve_worker_wrapper
    return args[0], {} 


def _reconstruct_worker_parallel(args):
    """
    Worker for parallel CRT reconstruction.
    Processes a batch of solution combinations.
    """
    combo_batch, primes, M, f_coeffs, max_height = args
    
    results = []
    stats = {
        'attempted': 0,
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }
    
    for sol_combo in combo_batch:
        stats['attempted'] += 1
        
        try:
            rec_vals = []
            for idx in range(4):
                vals = tuple(sol[idx] for sol in sol_combo)
                crt_val = crt_cached(vals, tuple(primes))
                num, den = rational_reconstruct(crt_val, M)
                
                if abs(num) > max_height or abs(den) > max_height:
                    raise RationalReconstructionError("Height too large")
                
                rec_vals.append(QQ(num)/QQ(den))
            
            s, p_val, v0, v1 = rec_vals
            
        except RationalReconstructionError:
            stats['height_reject'] += 1
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
            stats['consistency_reject'] += 1
            continue
        
        if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
            stats['algebraic_reject'] += 1
            continue
        
        results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
        stats['success'] += 1
    
    return results, stats

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

        if not discriminant_has_nonqr_s_p(s_qq, p_qq, PRIMES_NR):
            stats['algebraic_reject'] += 1
            continue

        # Attach the vector here so it survives the return trip
        results.append({'s': s_qq, 'p': p_qq, 'v_0': v0_qq, 'v_1': v1_qq, 'vector': v_tuple})
        stats['success'] += 1
    
    return results, stats

def reconstruct_parallel(sol_lists, primes, f_coeffs, adaptive_limit, num_workers=8, debug=False):
    """
    Parallel CRT reconstruction with batching.
    
    Returns: list of successfully reconstructed divisors
    """
    M = 1
    for p in primes:
        M *= p
    
    max_height = max(100000, int(M ** 0.35))
    
    # Generate all combinations (up to limit)
    all_combos = list(islice(product(*sol_lists), adaptive_limit))
    
    if debug:
        print(f"[parallel_crt] Processing {len(all_combos)} combinations with {num_workers} workers")
    
    # Batch combinations for workers
    batch_size = max(100, len(all_combos) // (num_workers * 4))
    batches = []
    for i in range(0, len(all_combos), batch_size):
        batch = all_combos[i:i+batch_size]
        batches.append((batch, primes, M, f_coeffs, max_height))
    
    # Process in parallel
    try:
        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(num_workers)
    except Exception:
        pool = multiprocessing.Pool(num_workers)
        raise
    
    all_results = []
    try:
        for batch_results in pool.imap_unordered(reconstruct_worker_wrapper, batches):
            all_results.extend(batch_results)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    
    return all_results

def adaptive_limit_with_early_stopping(sol_lists, primes, f_coeffs, base_limit, 
                                       check_interval=10000, target_divisors=10, debug=False):
    """
    Process combinations with early stopping if we're finding enough divisors.
    
    Strategy: Check success rate every `check_interval` combinations.
    If we've found `target_divisors` and success rate drops, stop early.
    """
    M = 1
    for p in primes:
        M *= p
    
    max_height = max(100000, int(M ** 0.35))
    
    results = []
    checked = 0
    last_check_count = 0
    
    for sol_combo in islice(product(*sol_lists), base_limit):
        checked += 1
        
        try:
            rec_vals = []
            for idx in range(4):
                vals = tuple(sol[idx] for sol in sol_combo)
                crt_val = crt_cached(vals, tuple(primes))
                num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
                rec_vals.append(QQ(num)/QQ(den))
            
            s, p_val, v0, v1 = rec_vals
            
            # Quick consistency check (just first prime)
            if len(primes) > 0:
                p0 = primes[0]
                expected = sol_combo[0]
                try:
                    s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, p0)) % p0
                    if s_mod != expected[0] % p0:
                        continue
                except ZeroDivisionError:
                    raise
                    continue
            
            # Full verification
            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
        
        except RationalReconstructionError:
            raise
            continue
        except Exception:
            raise
            continue
        
        # Early stopping check
        if checked % check_interval == 0:
            new_found = len(results) - last_check_count
            success_rate = new_found / check_interval
            
            if debug and checked % (check_interval * 5) == 0:
                print(f"[adaptive] Checked {checked}/{base_limit}, found {len(results)} total, recent rate: {success_rate:.6f}")
            
            # Stop if we have enough and success rate is very low
            if len(results) >= target_divisors and success_rate < 1e-5:
                if debug:
                    print(f"[adaptive] Early stop: found {len(results)} divisors, success rate dropped to {success_rate:.6f}")
                break
            
            last_check_count = len(results)
    
    return results, checked

def consistency_check_cached(s, p_val, v0, v1, sol_combo, primes, inv_cache):
    """
    Consistency check with cached modular inverses.
    Returns True if all primes match.
    """
    for i, prime in enumerate(primes):
        expected_sol = sol_combo[i]
        
        # Get modular inverses with caching
        s_inv = inv_cache.inv(s.denominator(), prime)
        p_inv = inv_cache.inv(p_val.denominator(), prime)
        v0_inv = inv_cache.inv(v0.denominator(), prime)
        v1_inv = inv_cache.inv(v1.denominator(), prime)
        
        if None in (s_inv, p_inv, v0_inv, v1_inv):
            return False
        
        s_mod = (int(s.numerator()) * s_inv) % prime
        p_mod = (int(p_val.numerator()) * p_inv) % prime
        v0_mod = (int(v0.numerator()) * v0_inv) % prime
        v1_mod = (int(v1.numerator()) * v1_inv) % prime
        
        if (s_mod != expected_sol[0] % prime or
            p_mod != expected_sol[1] % prime or
            v0_mod != expected_sol[2] % prime or
            v1_mod != expected_sol[3] % prime):
            return False
    
    return True

class ModInverseCache:
    """Cache for modular inverses."""
    def __init__(self):
        self.cache = {}
    
    def inv(self, a, p):
        key = (a % p, p)
        if key not in self.cache:
            try:
                self.cache[key] = pow(int(a), -1, p)
            except (ValueError, ZeroDivisionError):
                raise
                return None
        return self.cache[key]

def mumford_precompute_residues_parallel(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                                         rhs_modp_list, vecs_list, num_workers=8, debug=False):
    """
    Parallel residue computation with timing.
    Same function signature as original.
    """
    t_start = time.time()
    
    f_coeffs = eqs_dict['f_coeffs']
    f_coeffs_ints = [int(c) for c in f_coeffs]
    
    try:
        const_val_int = int(QQ(eqs_dict['const']))
    except Exception:
        const_val_int = 0
        raise
        
    if debug:
        print(f"[mumford] Generating tasks for {len(prime_list)} primes...")

    t0 = time.time()
    tasks = []
    
    for p in prime_list:
        if p not in Ep_dict:
            continue
        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        if not p_vecs:
            continue
        
        try:
            Fp = GF(p)
            R_m = Fp['m']
            m_var = R_m.gen()
            rhs_poly = -m_var + Fp(const_val_int)
        except Exception:
            raise
            continue

        x_residues_map = {}
        p_mults = mult_lll.get(p, {})
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue
            
            Pm = Ep(0)
            valid_vec = True
            v_coeffs = p_vecs[v_idx]

            for i, c in enumerate(v_coeffs):
                k = int(c)
                if k == 0:
                    continue
                
                try:
                    mults_for_sec = p_mults[i]
                    if k in mults_for_sec:
                        Pm += mults_for_sec[k]
                    else:
                        valid_vec = False
                        break
                except (IndexError, KeyError, TypeError):
                    valid_vec = False
                    raise
                    break
            
            if not valid_vec or Pm.is_zero() or Pm[2] == 0:
                continue
            
            try:
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator()
                
                if diff_num.is_zero():
                    continue
                    
                roots = diff_num.roots(multiplicities=False)
                
                if roots:
                    valid_residues = []
                    for m_root in roots:
                        m_val = int(m_root)
                        x_val = (-m_val + const_val_int) % p
                        valid_residues.append(x_val)
                    
                    if valid_residues:
                        x_residues_map[v_tuple] = valid_residues
            except Exception:
                raise
                continue
            
        if x_residues_map:
            tasks.append((p, f_coeffs_ints, x_residues_map, const_val_int))

    mumford_timer_add("task_generation", time.time() - t0)

    if not tasks:
        if debug:
            print("[mumford] No tasks generated!")
        return {}
    
    t0 = time.time()
    try:
        ctx = multiprocessing.get_context("fork")
        pool_obj = ctx.Pool(num_workers, initializer=_init_worker)
    except Exception:
        pool_obj = multiprocessing.Pool(num_workers, initializer=_init_worker)
        raise

    results_dict = {}
    with pool_obj as pool:
        for p, result_map in tqdm(pool.imap_unordered(_solve_worker_wrapper, tasks), 
                                  total=len(tasks), desc="Solving Mumford Mod P"):
            results_dict[p] = result_map
    
    mumford_timer_add("parallel_solving", time.time() - t0)
    mumford_timer_add("residue_computation_total", time.time() - t_start)
    
    if debug:
        print(f"[mumford] Residue computation took {time.time() - t_start:.2f}s")
            
    return results_dict
