import multiprocessing
import signal
from tqdm import tqdm
from .mumford_solver import solve_mumford_mod_p_optimized
from .mumford_verification import verify_mumford_pair, discriminant_has_nonqr_s_p
from .mumford_reconstruction import setup_crt_constants, fast_rational_reconstruct_check
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL, FINITE_FIELD
import time
import sys
import traceback
from sage.all import GF, QQ
from .mumford_timing import mumford_timer_add
import csv
from collections import defaultdict, OrderedDict
from collections import defaultdict
from itertools import islice, product
from .mumford_reconstruction import setup_crt_constants, fast_rational_reconstruct_check, RationalReconstructionError


MAX_TASK_DEGREE = 200


def _mumford_worker_entry(args):
    """Legacy entry point (placeholder)."""
    # NOTE: The provided code only used _solve_worker_wrapper
    return args[0], {} 


# Cap polynomial degree to prevent OOM. 
# Vectors with degree > 80 caused the previous crash.

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

def _reconstruct_worker_parallel(args):
    """Legacy entry point (placeholder)."""
    return args[0], {} 


def analyze_active_dead_vectors(results_dict, vecs_generated_list, vecs_list_for_p, prime):
    """
    Analyzes which vectors produced solutions.
    """
    # results for the prime (None if not present)
    pmap = results_dict.get(prime, {})
    
    # bookkeeping
    gen_set = [tuple(v) if isinstance(v, (list, tuple)) else (v,) for v in vecs_generated_list]
    available_vectors = set(pmap.keys())            # vectors we actually solved for
    all_generated = gen_set                         # canonical ordering of generated vectors
    
    # per-vector stats
    per_vec = {}
    seen_supports = set()
    cumulative_supports = []
    cum_count = 0
    
    # We'll iterate in the order of vecs_list_for_p (the order you attempted)
    order_list = [tuple(v) for v in vecs_list_for_p]
    
    for idx, v in enumerate(order_list):
        vkey = tuple(v)
        entry = {'index': idx,
                 'available': vkey in available_vectors,
                 'active': False,
                 'raw_solution_count': 0,
                 'verified_solution_count': 0,
                 'unique_supports_count': 0,
                 'new_supports_count': 0,
                 'hot_xs': [],
                 'new_supports': []}
        
        if vkey in pmap:
            xmap = pmap[vkey]   # x_res -> [sols]
            raw_count = 0
            verified_count = 0
            supports_this_vec = set()
            for x_res, sols in xmap.items():
                raw_count += len(sols)
                supports_this_vec.add(int(x_res))
                verified_count += len(sols)
            entry['raw_solution_count'] = raw_count
            entry['verified_solution_count'] = verified_count
            entry['unique_supports_count'] = len(supports_this_vec)
            # new supports
            new = [s for s in supports_this_vec if s not in seen_supports]
            entry['new_supports_count'] = len(new)
            entry['new_supports'] = new
            entry['hot_xs'] = sorted(list(supports_this_vec), key=lambda a: -list(supports_this_vec).count(a))[:5]
            if new:
                entry['active'] = True
            # update global seen_supports and cumulative
            for s in new:
                seen_supports.add(s)
                cum_count += 1
        else:
            # unavailable vector
            pass
        
        cumulative_supports.append(cum_count)
        per_vec[vkey] = entry
    
    # global stats
    generated_count = len(all_generated)
    available_count = sum(1 for v in order_list if tuple(v) in available_vectors)
    active_count = sum(1 for v in order_list if per_vec.get(tuple(v), {}).get('active', False))
    raw_solutions_total = sum(per_vec[v]['raw_solution_count'] for v in per_vec)
    unique_supports_total = len(seen_supports)
    duplicates = raw_solutions_total - unique_supports_total
    
    summary = {
        'prime': prime,
        'generated_count': generated_count,
        'available_count': available_count,
        'active_count': active_count,
        'raw_solutions_processed': raw_solutions_total,
        'unique_supports': unique_supports_total,
        'duplicates_rejected': duplicates,
        'availability_rate': available_count / float(generated_count) if generated_count>0 else 0.0,
        'activation_rate': active_count / float(available_count) if available_count>0 else 0.0,
    }
    
    print("PRIME:", prime)
    print("generated:", generated_count, "available:", available_count, "active:", active_count)
    print("raw_solutions:", raw_solutions_total, "unique_supports:", unique_supports_total, "duplicates:", duplicates)
    print("availability_rate:", summary['availability_rate'], "activation_rate:", summary['activation_rate'])
    
    return summary, per_vec, cumulative_supports


def find_poly_roots_fp_python(coeffs, p):
    """
    Find roots of polynomial over F_p using Sage's optimized backend.
    Replaces previous brute-force Python implementation.
    """
    try:
        # Use Sage's GF and polynomial ring (cached automatically by Sage)
        # This uses FLINT/NTL under the hood which is asymptotically fast
        R = GF(p)['x']
        f = R(coeffs)
        
        # .roots(multiplicities=False) returns a list of values
        # We convert them to standard Python ints
        roots = [int(r) for r in f.roots(multiplicities=False)]
        return roots
    except Exception:
        # Fallback only if Sage fails for some reason
        raise
        return []

def _brute_force_roots(coeffs, p):
    """Optimized brute force root finding."""
    # Dead code: kept to minimize file diffs
    roots = []
    if coeffs[0] % p == 0:
        roots.append(0)
    for x in range(1, p):
        val = 0
        for c in reversed(coeffs):
            val = (val * x + c) % p
        if val == 0:
            roots.append(x)
    return roots

def _quadratic_roots_fp(coeffs, p):
    """Solve ax^2 + bx + c = 0 mod p."""
    # Dead code: replaced by Sage
    c, b, a = coeffs[0], coeffs[1], coeffs[2]
    a, b, c = a % p, b % p, c % p
    if a == 0:
        if b == 0: return list(range(p)) if c == 0 else []
        b_inv = pow(int(b), -1, p)
        return [(-c * b_inv) % p]
    disc = (b * b - 4 * a * c) % p
    sqrt_disc = _tonelli_shanks(disc, p)
    if sqrt_disc is None: return []
    a_inv = pow(int(a), -1, p)
    two_a_inv = (2 * a_inv) % p
    root1 = ((-b + sqrt_disc) * two_a_inv) % p
    root2 = ((-b - sqrt_disc) * two_a_inv) % p
    return [root1, root2] if root1 != root2 else [root1]

def _cubic_roots_fp(coeffs, p):
    return _brute_force_roots(coeffs, p)

def _tonelli_shanks(n, p):
    """Tonelli-Shanks algorithm for modular square root."""
    # Dead code: replaced by Sage
    n = n % p
    if n == 0: return 0
    if pow(n, (p - 1) // 2, p) != 1: return None
    if p % 4 == 3: return pow(n, (p + 1) // 4, p)
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) == 1: z += 1
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    while t != 1:
        i = 1
        temp = (t * t) % p
        while temp != 1 and i < m:
            temp = (temp * temp) % p
            i += 1
        if i == m: return None
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    return r


def mumford_precompute_residues_parallel(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                                         rhs_modp_list, vecs_list, num_workers=16, debug=False):
    """
    Generates tasks with a safety clamp on worker count and polynomial degree.
    """
    t_start = time.time()
    
    # --- SAFETY CLAMP ---
    # Even if called with 20 workers, we force it down to 6 to prevent OOM
    if num_workers > 6:
        print(f"[mumford] NOTICE: Reducing workers from {num_workers} to 6 to prevent memory exhaustion.")
        num_workers = 16
    # --------------------

    f_coeffs = eqs_dict['f_coeffs']
    f_coeffs_ints = [int(c) for c in f_coeffs]
    const_val_int = int(QQ(eqs_dict['const']))
        
    if debug:
        print(f"[mumford] Generating tasks for {len(prime_list)} primes...")

    t0 = time.time()
    tasks_with_metadata = []
    skipped_count = 0
    
    # CRITICAL: Do ALL Sage operations here in main process
    for p in prime_list:
        if p not in Ep_dict:
            continue
        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        if not p_vecs:
            continue
        
        # Do Sage operations HERE - workers will only use Python ints
        Fp = GF(p)
        R_m = Fp['m']
        m_var = R_m.gen()
        rhs_poly = -m_var + Fp(const_val_int)
        p_mults = mult_lll.get(p, {})
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue
            
            # ALL SAGE OPERATIONS HERE
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
            
            if not valid_vec or Pm[2] == 0:
                continue
            if hasattr(Pm, 'is_zero') and Pm.is_zero():
                continue
            
            try:
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator()
                
                if diff_num.is_zero():
                    continue
                
                # Extract coefficients as Python ints - ONLY data passed to workers
                coeffs = diff_num.list()
                
                poly_degree = len(coeffs) - 1
                if poly_degree > MAX_TASK_DEGREE and False: # turn off degree cap since roots are fast
                    skipped_count += 1
                    continue
                
                coeffs_ints = [int(c) for c in coeffs]
                
                # Store task with metadata for sorting
                task = (int(p), f_coeffs_ints, [(v_tuple, coeffs_ints)], const_val_int)
                tasks_with_metadata.append((poly_degree, task))
                
            except Exception:
                raise
                continue
    
    # CRITICAL: Sort by polynomial degree (descending) for load balancing
    tasks_with_metadata.sort(key=lambda x: -x[0])
    tasks = [task for degree, task in tasks_with_metadata]
    
    if debug:
        if skipped_count > 0:
            print(f"[mumford] Skipped {skipped_count} tasks with degree > {MAX_TASK_DEGREE} to prevent OOM")
        
        if tasks:
            degrees = [deg for deg, _ in tasks_with_metadata]
            print(f"[mumford] Task degrees: min={min(degrees)}, max={max(degrees)}, "
                  f"median={sorted(degrees)[len(degrees)//2]}")

    mumford_timer_add("task_generation", time.time() - t0)

    if not tasks:
        if debug:
            print("[mumford] No tasks generated!")
        return {}
    
    if debug:
        print(f"[mumford] Generated {len(tasks)} tasks for {num_workers} workers ({len(tasks)/num_workers:.1f} tasks per worker)")
    
    # Use spawn context to avoid Sage pollution in workers
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    pool_obj = ctx.Pool(num_workers, initializer=_init_worker)

    results_dict = {}
    with pool_obj as pool:
        for p, result_map in tqdm(pool.imap_unordered(_solve_worker_wrapper, tasks), 
                                  total=len(tasks), desc="Solving Mumford Mod P"):
            if p not in results_dict:
                results_dict[p] = {}
            results_dict[p].update(result_map)
    
    mumford_timer_add("parallel_solving", time.time() - t0)
    mumford_timer_add("residue_computation_total", time.time() - t_start)
    
    if debug:
        print(f"[mumford] Residue computation took {time.time() - t_start:.2f}s")
            
    return results_dict

def _reconstruct_worker_parallel_v2(args):
    """
    Batch CRT reconstruct + algebraic verification worker.

    args = (combo_batch, primes, M_in, f_coeffs, max_height)
    Returns: (results_list, stats_dict)
    """
    combo_batch, primes, M_in, f_coeffs, max_height = args

    # Setup CRT constants (weights, M) if M_in is not provided
    if M_in is None:
        M, weights = setup_crt_constants(primes)
    else:
        M = M_in
        _, weights = setup_crt_constants(primes)  # This will compute weights matching M
    if M <= 0:
        raise ValueError("Invalid CRT modulus M")

    primes_int = [int(p) for p in primes]
    range_primes = range(len(primes_int))

    results = []
    stats = {
        'attempted': len(combo_batch),
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }

    for sol_combo in combo_batch:
        # Reconstruct each coordinate by CRT
        try:
            # s
            crt_s = 0
            for i in range_primes:
                crt_s += int(sol_combo[i][0]) * int(weights[i])
            crt_s %= M
            ok_s, num_s, den_s = fast_rational_reconstruct_check(crt_s, M, max_height)
            if not ok_s:
                stats['height_reject'] += 1
                continue

            # p
            crt_p = 0
            for i in range_primes:
                crt_p += int(sol_combo[i][1]) * int(weights[i])
            crt_p %= M
            ok_p, num_p, den_p = fast_rational_reconstruct_check(crt_p, M, max_height)
            if not ok_p:
                stats['height_reject'] += 1
                continue

            # v0
            crt_v0 = 0
            for i in range_primes:
                crt_v0 += int(sol_combo[i][2]) * int(weights[i])
            crt_v0 %= M
            ok_v0, num_v0, den_v0 = fast_rational_reconstruct_check(crt_v0, M, max_height)
            if not ok_v0:
                stats['height_reject'] += 1
                continue

            # v1
            crt_v1 = 0
            for i in range_primes:
                crt_v1 += int(sol_combo[i][3]) * int(weights[i])
            crt_v1 %= M
            ok_v1, num_v1, den_v1 = fast_rational_reconstruct_check(crt_v1, M, max_height)
            if not ok_v1:
                stats['height_reject'] += 1
                continue

        except Exception:
            # bubble up helpful info
            raise

        # Consistency check mod each prime (use python ints)
        reconstruction_ok = True
        for i, p_int in enumerate(primes_int):
            expected = sol_combo[i]
            try:
                if (num_s * pow(den_s, -1, p_int)) % p_int != expected[0]:
                    reconstruction_ok = False; break
                if (num_p * pow(den_p, -1, p_int)) % p_int != expected[1]:
                    reconstruction_ok = False; break
                if (num_v0 * pow(den_v0, -1, p_int)) % p_int != expected[2]:
                    reconstruction_ok = False; break
                if (num_v1 * pow(den_v1, -1, p_int)) % p_int != expected[3]:
                    reconstruction_ok = False; break
            except (ValueError, ZeroDivisionError) as e:
                # denominator divisible by prime -> fail consistency
                reconstruction_ok = False
                break

        if not reconstruction_ok:
            stats['consistency_reject'] += 1
            continue

        # Convert to QQ
        s_qq = QQ(num_s) / QQ(den_s)
        p_qq = QQ(num_p) / QQ(den_p)
        v0_qq = QQ(num_v0) / QQ(den_v0)
        v1_qq = QQ(num_v1) / QQ(den_v1)

        # Algebraic verification
        try:
            ok_alg = verify_mumford_pair(f_coeffs, s_qq, p_qq, v0_qq, v1_qq, modulus=None, debug_first_failure=False)
            if not ok_alg:
                stats['algebraic_reject'] += 1
                continue

            # Try to call discriminant_has_nonqr_s_p conservatively:
            try:
                ok_disc = discriminant_has_nonqr_s_p(s_qq, p_qq, primes)
            except TypeError:
                # fallback: if the function expected a number, pass the length
                ok_disc = discriminant_has_nonqr_s_p(s_qq, p_qq, len(primes))
            if not ok_disc:
                stats['algebraic_reject'] += 1
                continue

        except Exception:
            stats['algebraic_reject'] += 1
            continue

        results.append({'s': s_qq, 'p': p_qq, 'v_0': v0_qq, 'v_1': v1_qq})
        stats['success'] += 1

    return results, stats


def reconstruct_parallel(sol_lists, primes, f_coeffs, adaptive_limit, num_workers=20, debug=False):
    """
    Parallel CRT reconstruction with batching.
    Each worker returns (results_list, stats), we aggregate both.
    """
    # Precompute full modulus M
    M = 1
    for p in primes:
        M *= int(p)

    max_height = max(100000, int(M ** 0.35))

    # Generate combos up to adaptive_limit
    all_combos = list(islice(product(*sol_lists), adaptive_limit))

    if debug:
        print(f"[parallel_crt] Processing {len(all_combos)} combinations with {num_workers} workers")

    # Split into batches
    batch_size = max(100, max(1, len(all_combos) // (max(1, num_workers) * 4)))
    batches = []
    for i in range(0, len(all_combos), batch_size):
        batch = all_combos[i:i+batch_size]
        batches.append((batch, primes, M, f_coeffs, max_height))

    # Create pool (try spawn to be safe)
    try:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(num_workers)
    except Exception:
        # fallback to default context, but don't raise here
        pool = multiprocessing.Pool(num_workers)

    all_results = []
    aggregated_stats = defaultdict(int)

    try:
        # Use the corrected worker name
        for worker_out in pool.imap_unordered(_reconstruct_worker_parallel_v2, batches):
            if not worker_out:
                continue
            batch_results, batch_stats = worker_out
            # extend results
            all_results.extend(batch_results)
            # aggregate stats
            for k, v in batch_stats.items():
                aggregated_stats[k] += v
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise

    if debug:
        print(f"[parallel_crt] Done. totals: {dict(aggregated_stats)}")

    return all_results


def adaptive_limit_with_early_stopping(sol_lists, primes, f_coeffs, base_limit,
                                       check_interval=10000, target_divisors=10, debug=False):
    """
    Sequential adaptive reconstruction using CRT constants + fast rational reconstruct.
    """
    M, weights = setup_crt_constants(primes)
    max_height = max(100000, int(M ** 0.35))

    results = []
    checked = 0
    last_check_count = 0

    for sol_combo in islice(product(*sol_lists), base_limit):
        checked += 1
        try:
            # Reconstruct with weights
            rec_vals = []
            for idx_coord in range(4):
                crt_val = 0
                for i, p in enumerate(primes):
                    crt_val += int(sol_combo[i][idx_coord]) * int(weights[i])
                crt_val %= M
                ok, num, den = fast_rational_reconstruct_check(crt_val, M, max_height)
                if not ok:
                    raise RationalReconstructionError("height check failed")
                rec_vals.append(QQ(num) / QQ(den))

            s, p_val, v0, v1 = rec_vals

            # Quick consistency check modulo first prime
            if len(primes) > 0:
                p0 = int(primes[0])
                expected = sol_combo[0]
                try:
                    s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, p0)) % p0
                    if s_mod != expected[0] % p0:
                        continue
                except ZeroDivisionError:
                    raise

            # Full algebraic verification
            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})

        except RationalReconstructionError:
            # let it raise to blow up if you want; user wanted loud failures
            raise
        except Exception:
            # propagate everything (so you get a traceback)
            raise

        # early-stop check
        if checked % check_interval == 0:
            new_found = len(results) - last_check_count
            success_rate = new_found / float(check_interval)
            if debug and checked % (check_interval * 5) == 0:
                print(f"[adaptive] Checked {checked}/{base_limit}, found {len(results)} total, recent rate: {success_rate:.6e}")
            if len(results) >= target_divisors and success_rate < 1e-5:
                if debug:
                    print(f"[adaptive] Early stop: found {len(results)} divisors, recent rate {success_rate:.6e}")
                break
            last_check_count = len(results)

    return results, checked


def _solve_worker_wrapper(args):
    """
    Worker with detailed timing diagnostics.
    args = (p, f_coeffs_ints, chunk_items, const_val_int)
    """
    p, f_coeffs_ints, chunk_items, const_val_int = args

    roots_cache = {}
    p_results = {}

    chunk_start = time.time()

    for item_idx, (v_tuple, diff_coeffs_list) in enumerate(chunk_items):
        item_start = time.time()

        coeff_key = tuple(c % p for c in diff_coeffs_list)

        if all(c == 0 for c in coeff_key):
            continue

        t0 = time.time()
        if coeff_key not in roots_cache:
            roots = find_poly_roots_fp_python(coeff_key, p)
            roots_cache[coeff_key] = roots
        else:
            roots = roots_cache[coeff_key]
        root_time = time.time() - t0

        if not roots:
            continue

        t0 = time.time()
        x_res_to_sols = {}

        for m_root in roots:
            x_val = (-m_root + const_val_int) % p

            # Respect FINITE_FIELD flag properly
            if FINITE_FIELD:
                max_sols = 10000
            else:
                max_sols = 500

            sols = solve_mumford_mod_p_optimized(f_coeffs_ints, p, x_val, const_val_int, max_solutions=max_sols)

            verified_sols = []
            for sol in sols:
                s, p_val, v0, v1 = sol
                # verify with modulus p
                if verify_mumford_pair(f_coeffs_ints, s, p_val, v0, v1, modulus=p):
                    verified_sols.append(sol)

            if verified_sols:
                x_res_to_sols[x_val] = verified_sols

        mumford_time = time.time() - t0
        item_time = time.time() - item_start

        if item_time > 0.5:
            sys.stderr.write(f"[Worker p={p}] Vector {v_tuple}: deg={len(coeff_key)-1}, roots={len(roots)}, "
                             f"root_time={root_time:.3f}s, mumford_time={mumford_time:.3f}s, total={item_time:.3f}s\n")

        if x_res_to_sols:
            p_results[v_tuple] = x_res_to_sols

    chunk_time = time.time() - chunk_start
    if chunk_time > 1.0:
        sys.stderr.write(f"[Worker p={p}] Chunk of {len(chunk_items)} items took {chunk_time:.3f}s\n")

    return p, p_results


class ModInverseCache:
    """Cache for modular inverses."""
    def __init__(self):
        self.cache = {}

    def inv(self, a, p):
        key = (int(a) % int(p), int(p))
        if key not in self.cache:
            # pow(..., -1, p) will raise ValueError if not invertible
            self.cache[key] = pow(int(a), -1, int(p))
        return self.cache[key]


def consistency_check_cached(s, p_val, v0, v1, sol_combo, primes, inv_cache):
    """
    Consistency check; raises on modular inverse failure.
    Returns True if consistent, False otherwise.
    """
    for i, prime in enumerate(primes):
        expected_sol = sol_combo[i]

        # This will raise if denominator not invertible; let it propagate
        s_inv = inv_cache.inv(s.denominator(), prime)
        p_inv = inv_cache.inv(p_val.denominator(), prime)
        v0_inv = inv_cache.inv(v0.denominator(), prime)
        v1_inv = inv_cache.inv(v1.denominator(), prime)

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
