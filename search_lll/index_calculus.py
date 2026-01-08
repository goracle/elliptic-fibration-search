from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors
from .smoothness import extract_factor_base, tonelli_shanks
from sage.all import Curve, Jacobian, PolynomialRing, GF
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve


# Optimized anchoring worker for genus-2 index calculus
# Replaces find_smooth_decomposition_worker and find_smooth_decomposition
# Key ideas:
#  - Use baby-step table (window) to amortize scalar multiplication cost.
#  - For each batch pick a random base multiple (a multiple of window); compute it once per batch.
#  - For each candidate, compute t_div = base_mul + baby[i] (single Jacobian add).
#  - Perform a cheap filter: evaluate u(x) at a small random sample of factor-base x-roots.
#    Only if the cheap filter indicates potential full smoothness do the expensive full factorization/row build.
#  - Tune parameters: WINDOW (baby table size), SAMPLE_K (how many FB roots to sample), BATCH_CANDIDATES.

from sage.all import Integer, Zmod, set_random_seed
import random


from sage.all import Zmod, set_random_seed
from multiprocessing import Pool, cpu_count


# Optimized anchoring and worker initializer for genus-2 index calculus
# - Uses Pool initializer to avoid per-task serialization of large Sage objects
# - Precomputes baby-step table once and installs it in each worker
# - Caches canonical y-values for factor-base roots (Tonelli-Shanks results)
# - Cheap discriminant (Legendre symbol) check to kill ~50% of candidates
# - Samples top-k frequent factor-base roots for a very cheap filter

from sage.all import (
    matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer,
    Zmod, prime_factors, set_random_seed
)


# ----------------------------- GLOBALS (set in worker initializer) -----------------------------
# These globals are NOT intended to be imported by other modules; they are internal to worker processes
_GLOBAL_GENERATOR = None
_GLOBAL_TARGET_POINT = None
_GLOBAL_ROOT_TO_IDX = None
_GLOBAL_SAMPLE_ROOTS_INT = None
_GLOBAL_BABY = None
_GLOBAL_P = None
_GLOBAL_ORDER = None
_GLOBAL_WINDOW_SIZE = None
_GLOBAL_FB_Y_CACHE = None
_GLOBAL_F_POLY = None


# ----------------------------- UTILS -----------------------------


# Modified get_canonical_y that first consults the cached table installed in the worker.
# If the cache is provided (in the parent) the worker will have the mapping available.
# Parent should prepare fb_y_cache as {int(x): int(y_canonical)} for factor-base roots.


# Updated get_relation_row that uses the cached canonical y-values. Works with Sage Jacobian divisor
# objects created in the worker process (so u_poly/v_poly are Sage objects), but canonical y lookup
# and final arithmetic is done with ints.


# ----------------------------- WORKER CORE (hot loop) -----------------------------


# ----------------------------- PARENTALLY-DRIVEN ENTRY POINT -----------------------------


# ======================================================================================
# End of optimized anchoring module
# ======================================================================================

import time
from multiprocessing import Pool, cpu_count, Manager

# Add to _worker_init to track worker state


# Replace the final section of find_smooth_decomposition (starting from "with Pool...")
# with this version that includes progress tracking:


# ----------------------------- WORKER CORE -----------------------------

_GLOBAL_GENERATOR = None
_GLOBAL_TARGET_POINT = None
_GLOBAL_ROOT_TO_IDX = None
_GLOBAL_SAMPLE_ROOTS_INT = None
_GLOBAL_BABY = None
_GLOBAL_P = None
_GLOBAL_ORDER = None
_GLOBAL_WINDOW_SIZE = None
_GLOBAL_FB_Y_CACHE = None
_GLOBAL_F_POLY = None


import sys

def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G) on the Jacobian of y^2 = f(x).
    Intended ONLY for testing the index calculus machinery.
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    # Generate a random point on the curve, then convert to Jacobian element
    x_coord = K.random_element()
    y_squared = f(x_coord)
    
    # Keep trying until we find a point
    max_attempts = 1000
    for _ in range(max_attempts):
        if y_squared.is_square():
            y_coord = y_squared.sqrt()
            # Create a divisor from the point
            P = C((x_coord, y_coord))
            G = J(P)
            break
        x_coord = K.random_element()
        y_squared = f(x_coord)
    else:
        raise RuntimeError(f"Failed to find a valid point after {max_attempts} attempts")
    
    if target_d is None:
        order = compute_jacobian_order(f_poly.list(), p)
        target_d = ZZ.random_element(1, order)
    
    Q = target_d * G
    return G, Q, Integer(target_d)


def solve_dlp_index_calculus(valid_rows, g_anchored, q_anchored, ell, verbose=True):
    """
    Solves the system mod ell. ell MUST be prime.
    Uses the G-anchor to define the logs and Q-anchor to find the target.
    """
    K = GF(ell)
    
    # Build the relation matrix (use sparse matrix for efficiency)
    M = matrix(K, valid_rows, sparse=True)
    
    # We include the G-anchored row to find absolute logs of FB elements
    rg, row_g = g_anchored
    rq, row_q = q_anchored
    
    # Augmented matrix: [Relations] = 0, [row_g] = rg
    aug_rows = [vector(K, r) for r in valid_rows] + [vector(K, row_g)]
    aug_targets = [K(0)] * len(valid_rows) + [K(rg)]
    
    M_sys = matrix(K, aug_rows, sparse=True)
    target_v = vector(K, aug_targets)
    
    if verbose:
        print(f"Matrix size: {M_sys.nrows()}x{M_sys.ncols()}, Rank: {M_sys.rank()}")

    # solve_right finds the vector 'logs' such that M_sys * logs = target_v
    try:
        fb_logs = M_sys.solve_right(target_v)
    except ValueError:
        if verbose:
            print("System inconsistent or no solution found.")
        raise
        return None

    # log(Q) = sum(row_q * fb_logs) - rq (mod ell)
    v_q = vector(K, row_q)
    log_q = (v_q.dot_product(fb_logs) - K(rq))
    
    return Integer(log_q)


def compute_jacobian_order(f_coeffs, p):
    """
    Computes approximate Jacobian order using Hasse-Weil bound for large primes.
    For p > 2^64, returns approximate value suitable for probabilistic algorithms.
    """
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    f = P_x(f_coeffs)
    
    # For genus 2: Hasse-Weil gives |#J - (p^2+1)| <= 4*sqrt(p^3)
    if p > 2**64:
        return Integer(p**2 + 1)
    
    C = HyperellipticCurve(f)
    return C.count_points(1)[0]


def get_relation_row(divisor, root_to_idx, f_poly, p):
    """
    Converts a Mumford divisor (u, v) into a relation row with signs.
    Ensures that u(x) splits completely over the factor base.
    """
    u_poly = divisor[0]
    v_poly = divisor[1]
    
    if u_poly.degree() != 2:
        return None
    
    try:
        roots_data = u_poly.roots(GF(p))
    except Exception:
        raise
        return None
    
    if sum(mult for _, mult in roots_data) != 2:
        return None
        
    row_data = [0] * len(root_to_idx)
    
    for x_val, mult in roots_data:
        x_val = Integer(x_val)
        if x_val not in root_to_idx:
            return None
            
        y_val = Integer(v_poly(x_val))
        y_can = get_canonical_y(x_val, f_poly, p)
        
        if y_can is None:
            return None
            
        idx = root_to_idx[x_val]
        if y_val == y_can:
            row_data[idx] += int(mult)
        elif (p - y_val) % p == y_can:
            row_data[idx] -= int(mult)
        else:
            return None
            
    return row_data


def get_canonical_y(x, f_poly, p):
    """
    Returns a canonical y-coordinate for a given x such that y^2 = f(x) mod p.
    """
    y2 = f_poly(x)
    if y2 == 0:
        return None
    
    if pow(int(y2), (p-1)//2, p) != 1:
        return None
        
    y = tonelli_shanks(int(y2), p)
    return min(int(y), int(p - y))


# ----------------------------- WORKER CORE -----------------------------

_GLOBAL_GENERATOR = None
_GLOBAL_TARGET_POINT = None
_GLOBAL_ROOT_TO_IDX = None
_GLOBAL_SAMPLE_ROOTS_INT = None
_GLOBAL_BABY = None
_GLOBAL_P = None
_GLOBAL_ORDER = None
_GLOBAL_WINDOW_SIZE = None
_GLOBAL_FB_Y_CACHE = None
_GLOBAL_F_POLY = None

def _is_quadratic_residue(a_int, p_int):
    return pow(a_int % p_int, (p_int - 1) // 2, p_int) == 1

def get_canonical_y_cached(x_int):
    global _GLOBAL_FB_Y_CACHE
    if _GLOBAL_FB_Y_CACHE is None:
        return None
    return _GLOBAL_FB_Y_CACHE.get(int(x_int), None)

def get_relation_row_cached(divisor):
    global _GLOBAL_ROOT_TO_IDX, _GLOBAL_P
    u_poly = divisor[0]
    v_poly = divisor[1]

    if u_poly.degree() != 2:
        return None

    try:
        roots_data = u_poly.roots(GF(_GLOBAL_P))
    except Exception:
        raise
        return None

    if sum(m for _, m in roots_data) != 2:
        return None

    row = [0] * len(_GLOBAL_ROOT_TO_IDX)

    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in _GLOBAL_ROOT_TO_IDX:
            return None

        y_val = int(v_poly(x_elem))
        y_can = get_canonical_y_cached(x_int)
        if y_can is None:
            return None

        idx = _GLOBAL_ROOT_TO_IDX[x_int]
        if y_val == y_can:
            row[idx] += int(mult)
        elif (_GLOBAL_P - y_val) % _GLOBAL_P == y_can:
            row[idx] -= int(mult)
        else:
            return None

    return row

def _worker_core_try_batch(batch_candidates):
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_BABY
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_P, _GLOBAL_ORDER, _GLOBAL_SAMPLE_ROOTS_INT

    Z_ord = Zmod(_GLOBAL_ORDER)
    max_blocks = max(1, (_GLOBAL_ORDER // _GLOBAL_WINDOW_SIZE) + 1)
    
    # Random block
    block_index = int(Z_ord.random_element()) % max_blocks
    base_r = Integer(block_index) * Integer(_GLOBAL_WINDOW_SIZE)

    try:
        base_mul = base_r * _GLOBAL_GENERATOR
    except Exception:
        raise
        return None

    baby = _GLOBAL_BABY
    sample_roots = _GLOBAL_SAMPLE_ROOTS_INT
    P_int = _GLOBAL_P

    for _ in range(batch_candidates):
        i = random.randrange(_GLOBAL_WINDOW_SIZE)
        r_val = int(block_index * _GLOBAL_WINDOW_SIZE + i)

        try:
            # We add target_point to the random walk point
            t_div = _GLOBAL_TARGET_POINT + (base_mul + baby[i])
        except Exception:
            raise
            continue

        if t_div[0].degree() != 2:
            continue

        # u = x^2 - s*x + p_coeff
        try:
            u = t_div[0]
            a0 = int(u[0])
            a1 = int(u[1])
        except Exception:
            raise
            continue

        s_coeff = (-a1) % P_int
        p_coeff = a0 % P_int

        disc = (s_coeff * s_coeff - 4 * p_coeff) % P_int
        if disc != 0:
            if not _is_quadratic_residue(disc, P_int):
                continue

        # Cheap filter
        hit = False
        for xr in sample_roots:
            if ((xr * xr - s_coeff * xr + p_coeff) % P_int) == 0:
                hit = True
                break
        if not hit:
            continue

        row = get_relation_row_cached(t_div)
        if row is not None:
            return (r_val, row)

    return None

def _worker_init(generator, target_point, root_to_idx, sample_roots_int, baby_table,
                 fb_y_cache, f_poly, p_int, order_int, window_size):
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_ROOT_TO_IDX
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_BABY, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY

    _GLOBAL_GENERATOR = generator
    _GLOBAL_TARGET_POINT = target_point
    _GLOBAL_ROOT_TO_IDX = root_to_idx
    _GLOBAL_SAMPLE_ROOTS_INT = sample_roots_int
    _GLOBAL_BABY = baby_table
    _GLOBAL_FB_Y_CACHE = fb_y_cache
    _GLOBAL_F_POLY = f_poly
    _GLOBAL_P = int(p_int)
    _GLOBAL_ORDER = int(order_int)
    _GLOBAL_WINDOW_SIZE = int(window_size)

def find_smooth_decomposition_worker(seed_and_batch):
    seed, batch_candidates = seed_and_batch
    random.seed(int(seed))
    return _worker_core_try_batch(batch_candidates)

def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order,
                              max_tries=None, num_workers=None,
                              window_size=2048, sample_k=32, batch_candidates=512,
                              factor_base_freq=None):
    """
    Parallel search for smooth divisor.
    target_point: Starting offset (e.g., Q)
    generator: Point to walk with (e.g., G)
    Result: r, row such that target_point + [r]*generator ~ row
    """
    from sage.all import set_random_seed
    
    num_workers = cpu_count() if num_workers is None else num_workers
    p_int = int(p)
    order_int = int(order)

    fb_roots = sorted(list(root_to_idx.keys()))
    fb_y_cache = {}
    for x in fb_roots:
        x_int = int(x)
        y2 = int(f_poly(x))
        if y2 == 0: continue
        if _is_quadratic_residue(y2, p_int):
            y_can = tonelli_shanks(y2, p_int)
            y_can = min(int(y_can), p_int - int(y_can))
            fb_y_cache[x_int] = int(y_can)

    fb_root_list = list(root_to_idx.keys())
    if factor_base_freq:
        fb_roots_sorted = sorted(fb_root_list, key=lambda r: factor_base_freq.get(r, 0), reverse=True)
        sample_roots = [int(r) for r in fb_roots_sorted[:sample_k]]
    else:
        sample_roots = [int(r) for r in random.sample(fb_root_list, min(sample_k, len(fb_root_list)))]

    # Precompute baby steps
    zero = generator.parent().zero()
    baby = [zero]
    curr = zero
    for _ in range(1, window_size):
        curr = curr + generator
        baby.append(curr)

    # Adaptive max_tries
    if max_tries is None:
        fb_size = len(root_to_idx)
        # Probability ~ (B/p)^2
        estimated_prob = (fb_size / float(p_int)) ** 2
        # Safety factor of 5.0, min 100k
        max_tries = max(100000, int(5.0 / max(estimated_prob, 1e-12)))

    total_batches = (max_tries + batch_candidates - 1) // batch_candidates

    print(f"  {total_batches} batches × {batch_candidates} = {max_tries} total trials planned")
    print(f"  FB={len(root_to_idx)}, p={p_int}")
    print(f"  Est. Prob: {estimated_prob:.2e} => Expected trials: {1.0/max(estimated_prob, 1e-12):.1e}")
    sys.stdout.flush()

    print("[DEBUG] Preparing worker arguments...", flush=True)
    initargs = (
        generator, target_point, root_to_idx, sample_roots, baby,
        fb_y_cache, f_poly, p_int, order_int, window_size
    )
    print(f"[DEBUG] Worker arguments prepared. Size of baby table: {len(baby)}", flush=True)

    print(f"[DEBUG] Generating {total_batches} tasks...", flush=True)
    tasks = [(random.randint(0, 2**31 - 1), batch_candidates) for _ in range(total_batches)]
    print(f"[DEBUG] Tasks generated.", flush=True)

    start_time = time.time()
    last_print = start_time
    batches_done = 0

    print(f"[DEBUG] Initializing Pool with {num_workers} workers...", flush=True)
    with Pool(processes=num_workers, initializer=_worker_init, initargs=initargs) as pool:
        print("[DEBUG] Pool initialized. Starting task execution...", flush=True)
        try:
            # Use chunksize to reduce IPC overhead for large number of tasks
            # With 475k tasks, default chunksize=1 is very slow
            chunk_size = min(1000, max(1, len(tasks) // (num_workers * 4)))
            print(f"[DEBUG] Using chunksize={chunk_size} for {len(tasks)} tasks.", flush=True)
            
            for result in pool.imap_unordered(find_smooth_decomposition_worker, tasks, chunksize=chunk_size):
                if batches_done == 0:
                    print("[DEBUG] First result received!", flush=True)

                batches_done += 1
                
                now = time.time()
                if now - last_print >= 5.0:
                    elapsed = now - start_time
                    rate = batches_done * batch_candidates / elapsed if elapsed > 0 else 0
                    pct = 100.0 * batches_done / total_batches
                    eta = (total_batches - batches_done) * batch_candidates / rate if rate > 0 else 0
                    print(f"  [{elapsed:.0f}s] {pct:.1f}% done. {rate:.0f} cand/s. ETA: {eta:.0f}s", flush=True)
                    last_print = now
                
                if result is not None:
                    pool.terminate()
                    elapsed = time.time() - start_time
                    print(f"  ✓ Found after {elapsed:.1f}s ({batches_done*batch_candidates} candidates)", flush=True)
                    return result
        except KeyboardInterrupt:
            pool.terminate()
            raise

    elapsed = time.time() - start_time
    print(f"  ✗ Failed after {elapsed:.1f}s. Tried {batches_done*batch_candidates} candidates.", flush=True)
    return None, None


def perform_dlp_attack(g_pt, q_pt, smooth_divs, p, f_coeffs, order, verbose=True):
    """
    Main entry point. Automatically factors order and solves mod the largest prime.
    """
    factors = prime_factors(order)
    ell = max(factors)
    
    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order: {ell}")
    
    K_p = GF(p)
    poly_ring = PolynomialRing(K_p, 'x')
    x = poly_ring.gen()
    f_p = poly_ring(f_coeffs[::-1])
    
    C = HyperellipticCurve(f_p)
    J = C.jacobian()
    
    fb_data = extract_factor_base(smooth_divs, p, verbose=False)
    roots_filtered = sorted([r for r in fb_data['roots'] if f_p(r) != 0])
    r_to_idx = {r: i for i, r in enumerate(roots_filtered)}
    
    if verbose:
        print(f"Factor base size: {len(roots_filtered)}")
    
    sage_divisors = []
    for d in fb_data['unique_divisors']:
        s_val = int(d['s']) % p
        p_val = int(d['p']) % p
        v0_val = int(d['v_0']) % p
        v1_val = int(d['v_1']) % p
        
        u_poly = x**2 - K_p(s_val)*x + K_p(p_val)
        v_poly = K_p(v1_val)*x + K_p(v0_val)
        
        D = J([u_poly, v_poly])
        sage_divisors.append(D)
    
    valid_rows = []
    for d in sage_divisors:
        row = get_relation_row(d, r_to_idx, f_p, p)
        if row:
            valid_rows.append(row)
            
    from collections import Counter
    root_freq = Counter()
    for d in sage_divisors:
        u_poly = d[0]
        if u_poly.degree() == 2:
            try:
                roots_data = u_poly.roots(K_p)
                for x_val, mult in roots_data:
                    x_int = Integer(x_val)
                    if x_int in r_to_idx:
                        root_freq[x_int] += int(mult)
            except Exception:
                raise
    
    if verbose:
        print(f"Anchoring G...")
        print(f"  G = {g_pt}")
        print(f"  Attempting to find r such that [r]G is FB-smooth...")
    
    # G Anchoring: Find r such that 0 + [r]G is smooth
    rg, row_g = find_smooth_decomposition(
        g_pt.parent().zero(), g_pt, r_to_idx, f_p, p, order,
        max_tries=None,  # Let the function calculate needed tries
        factor_base_freq=root_freq
    )
    
    if row_g is None:
        if verbose:
            print(f"  ✗ FAILED to anchor G")
        assert False, "Failed to anchor G into the factor base."
    
    if verbose:
        print(f"  ✓ G anchored: r={rg}")
        print(f"Anchoring Q...")
        print(f"  Q = {q_pt}")
        print(f"  Attempting to find r such that Q - [r]G is FB-smooth...")
        
    # Q Anchoring: Find r such that Q + [r](-G) is smooth
    # This gives: Q - rG = smooth_div => Q = smooth_div + rG
    neg_g = -g_pt
    rq, row_q = find_smooth_decomposition(
        q_pt, neg_g, r_to_idx, f_p, p, order,
        max_tries=None,  # Let the function calculate needed tries
        factor_base_freq=root_freq
    )
    
    if row_q is None:
        if verbose:
            print(f"  ✗ FAILED to anchor Q")
        assert False, "Failed to anchor Q into the factor base."
    
    if verbose:
        print(f"  ✓ Q anchored: r={rq}")

    d_mod_ell = solve_dlp_index_calculus(valid_rows, (rg, row_g), (rq, row_q), ell, verbose=verbose)
    
    if d_mod_ell is None:
        print("Linear algebra failed.")
        return None

    # Verification
    factor_cofactor = order // ell
    try:
        lhs = d_mod_ell * (factor_cofactor * g_pt)
        rhs = (factor_cofactor * q_pt)
        if lhs == rhs:
            if verbose:
                print(f"✓ Success mod {ell}. Log: {d_mod_ell}")
            return d_mod_ell
        else:
            print("Verification failed. Spurious relations?")
            return None
    except Exception as e:
        print(f"Verification error: {e}")
        raise
        return d_mod_ell
