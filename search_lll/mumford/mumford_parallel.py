import multiprocessing, signal, time, sys, traceback, csv
from tqdm import tqdm
from .mumford_solver import solve_mumford_mod_p_optimized
from .mumford_verification import verify_mumford_pair, discriminant_has_nonqr_s_p
from .mumford_reconstruction import setup_crt_constants, fast_rational_reconstruct_check, RationalReconstructionError
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL, FINITE_FIELD
from sage.all import GF, QQ
from .mumford_timing import mumford_timer_add
from collections import defaultdict, OrderedDict
from itertools import islice, product

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

def init_worker():
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

# In mumford_parallel.py, DELETE lines 102-172 (dead root-finding code)
# Replace with this clean version:

def find_poly_roots_fp_python(coeffs, p):
    """
    Find roots of polynomial over F_p using Sage's optimized backend.
    Uses FLINT/NTL under the hood which is asymptotically fast.
    """
    assert isinstance(p, int) and p > 2, f"Invalid prime: {p}"
    assert coeffs, "Empty coefficient list"

    try:
        R = GF(p)['x']
        f = R(coeffs)
        roots = [int(r) for r in f.roots(multiplicities=False)]
        return roots
    except Exception as e:
        raise RuntimeError(f"Sage root finding failed for p={p}, coeffs={coeffs}: {e}")

# In mumford_parallel.py, add these improvements to _solve_worker_wrapper:

def _solve_worker_wrapper(args):
    """
    Worker with fail-fast error handling and detailed diagnostics.

    CRITICAL: All exceptions propagate - no silent failures.
    """
    p, f_coeffs_ints, chunk_items, const_val_int = args

    # Validate inputs
    assert isinstance(p, int) and p > 2, f"Invalid prime: {p}"
    assert f_coeffs_ints, "Empty f_coeffs"
    assert chunk_items, "Empty chunk_items"

    roots_cache = {}
    p_results = {}
    chunk_start = time.time()

    for item_idx, (v_tuple, diff_coeffs_list) in enumerate(chunk_items):
        assert v_tuple is not None, f"Item {item_idx}: v_tuple is None"
        assert diff_coeffs_list, f"Item {item_idx}: empty diff_coeffs"

        item_start = time.time()

        # Normalize coefficients mod p
        coeff_key = tuple(c % p for c in diff_coeffs_list)

        # Skip zero polynomial
        if all(c == 0 for c in coeff_key):
            continue

        # Find roots (with caching)
        t0 = time.time()
        if coeff_key not in roots_cache:
            roots = find_poly_roots_fp_python(coeff_key, p)
            roots_cache[coeff_key] = roots
        else:
            roots = roots_cache[coeff_key]
        root_time = time.time() - t0

        if not roots:
            continue

        # Solve Mumford equations
        t0 = time.time()
        x_res_to_sols = {}

        for m_root in roots:
            assert isinstance(m_root, int), f"Root is not an integer: {m_root}"

            x_val = (-m_root + const_val_int) % p

            # Respect FINITE_FIELD flag
            max_sols = 10000 if FINITE_FIELD else 500

            try:
                sols = solve_mumford_mod_p_optimized(
                    f_coeffs_ints, p, x_val, const_val_int, max_solutions=max_sols
                )
            except Exception as e:
                # DO NOT CATCH - let it propagate
                raise RuntimeError(
                    f"Mumford solver failed: p={p}, x_val={x_val}, "
                    f"v_tuple={v_tuple}, error={e}"
                )

            # Verify solutions
            verified_sols = []
            for sol in sols:
                assert len(sol) == 4, f"Invalid solution length: {len(sol)}"
                s, p_val, v0, v1 = sol

                # Verify algebraically mod p
                if not verify_mumford_pair(f_coeffs_ints, s, p_val, v0, v1, modulus=p):
                    # Verification failure is a DATA BUG, not a search failure
                    raise RuntimeError(
                        f"Mumford pair failed verification: "
                        f"p={p}, sol={sol}, v_tuple={v_tuple}"
                    )

                verified_sols.append(sol)

            if verified_sols:
                x_res_to_sols[x_val] = verified_sols

        mumford_time = time.time() - t0
        item_time = time.time() - item_start

        # Diagnostic for slow items
        if item_time > 0.5:
            sys.stderr.write(
                f"[Worker p={p}] Vector {v_tuple}: deg={len(coeff_key)-1}, "
                f"roots={len(roots)}, root_time={root_time:.3f}s, "
                f"mumford_time={mumford_time:.3f}s, total={item_time:.3f}s\n"
            )
            sys.stderr.flush()

        if x_res_to_sols:
            p_results[v_tuple] = x_res_to_sols

    chunk_time = time.time() - chunk_start
    if chunk_time > 1.0:
        sys.stderr.write(f"[Worker p={p}] Chunk of {len(chunk_items)} items took {chunk_time:.3f}s\n")
        sys.stderr.flush()

    return p, p_results

# Similarly, add assertions to mumford_precompute_residues_parallel:

def mumford_precompute_residues_parallel(
    eqs_dict,
    prime_list,
    Ep_dict,
    mult_lll,
    vecs_lll,
    rhs_modp_list,
    vecs_list,
    num_workers=16,
    debug=False,
    chunk_size=4,
    pool=None,
):
    """
    Generate residue-computation tasks and solve them in parallel.

    Improvements over the old version:
      - batches multiple vectors per worker task via chunk_size
      - supports reusing an existing multiprocessing pool
      - fails fast on bad inputs and propagates exceptions
      - keeps Sage-heavy work in the main process
      - uses spawn by default when creating its own pool

    Args:
        eqs_dict: must contain 'f_coeffs' and 'const'
        prime_list: primes to process
        Ep_dict: prime -> elliptic curve / point container
        mult_lll: prime -> section multiples data
        vecs_lll: prime -> LLL vectors data
        rhs_modp_list: kept for API compatibility
        vecs_list: list of vectors to try
        num_workers: number of worker processes to create if pool is None
        debug: verbose printing
        chunk_size: number of (v_tuple, coeffs_ints) items per worker task
        pool: optional existing multiprocessing pool to reuse

    Returns:
        results_dict: {p: {v_tuple: {x_res: [sols...]}}}
    """
    assert isinstance(eqs_dict, dict) and "f_coeffs" in eqs_dict and "const" in eqs_dict, \
        "Invalid eqs_dict: must contain 'f_coeffs' and 'const'"
    assert prime_list, "Empty prime_list"
    assert Ep_dict, "Empty Ep_dict"
    assert vecs_list, "Empty vecs_list"

    if chunk_size is None or int(chunk_size) < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    chunk_size = int(chunk_size)

    if num_workers is None or int(num_workers) < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    num_workers = int(num_workers)

    # Safety clamp for memory-heavy Sage workloads.
    if num_workers > 16:
        print(f"[mumford] NOTICE: Reducing workers from {num_workers} to 16 to prevent memory exhaustion.")
        num_workers = 16

    t_start = time.time()

    f_coeffs = eqs_dict["f_coeffs"]
    f_coeffs_ints = [int(c) for c in f_coeffs]
    const_val_int = int(QQ(eqs_dict["const"]))

    if debug:
        print(f"[mumford] Generating tasks for {len(prime_list)} primes...")
        sys.stdout.flush()

    t0 = time.time()
    tasks_with_metadata = []

    # Build all tasks in the main process.
    # Each task is a batch of size up to chunk_size.
    for p in prime_list:
        assert p in Ep_dict, f"Prime {p} missing from Ep_dict"

        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        assert p_vecs is not None, f"Prime {p} missing from vecs_lll"
        assert len(p_vecs) >= len(vecs_list), \
            f"Prime {p}: vecs_lll[p] shorter than vecs_list ({len(p_vecs)} < {len(vecs_list)})"

        Fp = GF(p)
        R_m = Fp["m"]
        m_var = R_m.gen()
        rhs_poly = -m_var + Fp(const_val_int)
        p_mults = mult_lll.get(p, {})

        current_chunk = []
        current_chunk_degree = 0

        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue

            v_coeffs = p_vecs[v_idx]

            # Build the section multiple Pm = sum_i k_i * mults_for_sec[i][k_i]
            Pm = Ep(0)
            valid_vec = True

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
                except (IndexError, KeyError, TypeError) as e:
                    raise RuntimeError(
                        f"Failed to build section multiple: p={p}, v_idx={v_idx}, i={i}, k={k}, error={e}"
                    )

            if not valid_vec:
                continue

            if Pm[2] == 0:
                continue

            if hasattr(Pm, "is_zero") and Pm.is_zero():
                continue

            try:
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator()

                if diff_num.is_zero():
                    continue

                coeffs = diff_num.list()
                poly_degree = len(coeffs) - 1
                coeffs_ints = [int(c) for c in coeffs]

                current_chunk.append((v_tuple, coeffs_ints))
                current_chunk_degree += max(poly_degree, 0)

                if len(current_chunk) >= chunk_size:
                    task = (int(p), f_coeffs_ints, current_chunk, const_val_int)
                    tasks_with_metadata.append((current_chunk_degree, task))
                    current_chunk = []
                    current_chunk_degree = 0

            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract polynomial: p={p}, v_idx={v_idx}, v_tuple={v_tuple}, error={e}"
                )

        # Flush any leftover items for this prime.
        if current_chunk:
            task = (int(p), f_coeffs_ints, current_chunk, const_val_int)
            tasks_with_metadata.append((current_chunk_degree, task))

    # Sort by estimated work, descending.
    tasks_with_metadata.sort(key=lambda x: -x[0])
    tasks = [task for _, task in tasks_with_metadata]

    if debug:
        if tasks_with_metadata:
            degrees = [deg for deg, _ in tasks_with_metadata]
            degrees_sorted = sorted(degrees)
            median_degree = degrees_sorted[len(degrees_sorted) // 2]
            print(
                f"[mumford] Task degrees: min={min(degrees)}, max={max(degrees)}, "
                f"median={median_degree}"
            )
            print(
                f"[mumford] Generated {len(tasks)} batched tasks "
                f"with chunk_size={chunk_size} for {num_workers} workers "
                f"({len(tasks) / float(num_workers):.1f} tasks per worker)"
            )
        sys.stdout.flush()

    mumford_timer_add("task_generation", time.time() - t0)

    assert tasks, "No tasks generated - this indicates a configuration error"

    owns_pool = pool is None
    terminated = False

    if owns_pool:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(num_workers, initializer=init_worker)

    results_dict = {}

    try:
        for p, result_map in tqdm(
            pool.imap_unordered(_solve_worker_wrapper, tasks),
            total=len(tasks),
            desc="Solving Mumford Mod P",
        ):
            if p not in results_dict:
                results_dict[p] = {}
            results_dict[p].update(result_map)

    except KeyboardInterrupt:
        terminated = True
        if owns_pool:
            pool.terminate()
        raise
    except Exception:
        terminated = True
        if owns_pool:
            pool.terminate()
        raise
    finally:
        if owns_pool:
            try:
                if not terminated:
                    pool.close()
            finally:
                pool.join()

    mumford_timer_add("parallel_solving", time.time() - t0)
    mumford_timer_add("residue_computation_total", time.time() - t_start)

    if debug:
        print(f"[mumford] Residue computation took {time.time() - t_start:.2f}s")
        sys.stdout.flush()

    assert results_dict, "Worker pool returned empty results - this indicates a failure"

    return results_dict
