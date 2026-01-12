from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors, set_random_seed
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from .smoothness import extract_factor_base, tonelli_shanks
from search_common import *
import random
import time
import sys
from multiprocessing import Pool, cpu_count
from collections import Counter

# ============================================================================
# WORKER GLOBALS & INITIALIZATION
# ============================================================================

_GLOBAL_GENERATOR = None
_GLOBAL_TARGET_POINT = None
_GLOBAL_ROOT_TO_IDX = None
_GLOBAL_SAMPLE_ROOTS_INT = None
_GLOBAL_BABY = None
_GLOBAL_P = None
_GLOBAL_ORDER = None
_GLOBAL_WINDOW_SIZE = None
_GLOBAL_FB_Y_CACHE = None
global _GLOBAL_F_POLY # can't create this under multiprocessing, segfault
_GLOBAL_F_POLY = None
_GLOBAL_OFFSET_CACHE = None
K = GF(FINITE_FIELD)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_quadratic_residue(a_int, p_int):
    return pow(a_int % p_int, (p_int - 1) // 2, p_int) == 1

def get_canonical_y_cached(x_int):
    global _GLOBAL_FB_Y_CACHE
    if _GLOBAL_FB_Y_CACHE is None:
        return None
    return _GLOBAL_FB_Y_CACHE.get(int(x_int), None)


def generate_random_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G).
    """
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    C = HyperellipticCurve(f)
    J = C.jacobian()

    # Generate random point
    while True:
        x_coord = K.random_element()
        y2 = f(x_coord)
        if y2.is_square():
            y_coord = y2.sqrt()
            G = J(C((x_coord, y_coord)))
            # Check not order 2 (unlikely for large p)
            if 2*G != J(0):
                break

    if target_d is None:
        target_d = ZZ.random_element(1, p**2) # Approximate order

    Q = target_d * G
    return G, Q, Integer(target_d)


def compute_jacobian_order(f_coeffs, p):
    """
    Computes approximate Jacobian order using Hasse-Weil bound for large primes.
    For p > 2^64, returns approximate value suitable for probabilistic algorithms.
    """
    P_x = PolynomialRing(K, 'x')
    f = sage_poly_from_coeffs(f_coeffs, P_x)

    # For genus 2: Hasse-Weil gives |#J - (p^2+1)| <= 4*sqrt(p^3)
    if p > 2**64:
        return Integer(p**2 + 1)
    
    C = HyperellipticCurve(f)
    return C.count_points(1)[0]

def _worker_init(gen_mumford, target_mumford, root_to_idx, sample_roots_int, 
                 fb_y_cache, p_int, order_int, window_size, offset_coeffs):
    """
    Initializes worker process. Reconstructs Sage objects from plain Python data.
    """
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_ROOT_TO_IDX
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_BABY, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY, _GLOBAL_OFFSET_CACHE

    _GLOBAL_ROOT_TO_IDX = root_to_idx
    _GLOBAL_SAMPLE_ROOTS_INT = sample_roots_int
    _GLOBAL_FB_Y_CACHE = fb_y_cache
    _GLOBAL_P = int(p_int)
    _GLOBAL_ORDER = int(order_int)
    _GLOBAL_WINDOW_SIZE = int(window_size)
    
    # Reconstruct f_poly
    R = PolynomialRing(K, 'x')

    # Reconstruct curve and Jacobian
    C = HyperellipticCurve(_GLOBAL_F_POLY)
    J = C.jacobian()
    
    # Reconstruct generator
    if gen_mumford is not None:
        gen_u_coeffs, gen_v_coeffs = gen_mumford
        u_poly = R(gen_u_coeffs)
        v_poly = R(gen_v_coeffs)
        _GLOBAL_GENERATOR = J([u_poly, v_poly])
    else:
        _GLOBAL_GENERATOR = None
    
    # Reconstruct target
    if target_mumford is not None:
        target_u_coeffs, target_v_coeffs = target_mumford
        u_poly = R(target_u_coeffs)
        v_poly = R(target_v_coeffs)
        _GLOBAL_TARGET_POINT = J([u_poly, v_poly])
    else:
        _GLOBAL_TARGET_POINT = None
    
    # Precompute baby steps in worker
    print(f"  [Worker] Precomputing {window_size} baby steps...")
    zero = J.zero()
    _GLOBAL_BABY = [zero]
    curr = zero
    for _ in range(1, window_size):
        curr = curr + _GLOBAL_GENERATOR
        _GLOBAL_BABY.append(curr)
    
    # Reconstruct offset cache
    _GLOBAL_OFFSET_CACHE = []
    if offset_coeffs:
        x = R.gen()
        for (s, p_val, v0, v1) in offset_coeffs:
            try:
                u_poly = x**2 - K(int(s))*x + K(int(p_val))
                v_poly = K(int(v1))*x + K(int(v0))
                _GLOBAL_OFFSET_CACHE.append(J([u_poly, v_poly])) # this line is broken, the u_poly and v_poly aren't valid for some reason
            except Exception:
                raise
                continue


def _worker_core_try_batch(r_val):
    """
    Worker core: attempts to find a relation for a specific scalar r.
    Returns plain Python types only (no Sage objects).
    """
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_ROOT_TO_IDX
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_BABY, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY, _GLOBAL_OFFSET_CACHE

    P_int = _GLOBAL_P
    sample_roots = _GLOBAL_SAMPLE_ROOTS_INT
    agg_stats = Counter()
    agg_stats['tried'] = 1

    try:
        if _GLOBAL_TARGET_POINT is None:
            D = r_val * _GLOBAL_GENERATOR
        else:
            D = r_val * _GLOBAL_GENERATOR + _GLOBAL_TARGET_POINT
    except Exception:
        raise
        return ("STATS", dict(agg_stats))
    
    for off_idx, offset_D in enumerate([None] + _GLOBAL_OFFSET_CACHE):
        try:
            cand_D = D + offset_D if offset_D else D
            # Fixed: Sage Jacobian points are already iterable/indexable as (u, v)
            cand_div = cand_D 
        except Exception:
            raise
            continue
        
        try:
            u = cand_div[0]
            if u.degree() != 2:
                continue
                
            u0 = int(u[0])
            u1 = int(u[1])

            hit = False
            for xr in sample_roots:
                if (xr * xr + u1 * xr + u0) % P_int == 0:
                    hit = True
                    break
            
            if not hit:
                agg_stats['sample_miss'] += 1
                continue

            row_vec = get_relation_row_cached(cand_div)
            if row_vec is not None:
                r_val_int = int(r_val)
                row_vec_plain = {int(k): int(v) for k, v in row_vec.items()}
                offset_idx = off_idx - 1
                return ("SUCCESS", (r_val_int, row_vec_plain, offset_idx))
        except Exception:
            raise
            continue
            
    return ("STATS", dict(agg_stats))


# ... (Previous imports and helper functions remain the same)

def find_smooth_decomposition_worker(seed_and_batch):
    seed, batch_candidates = seed_and_batch
    set_random_seed(seed)
    random.seed(int(seed))
    
    batch_stats = Counter()
    
    for _ in range(batch_candidates):
        r_val = random.randint(1, _GLOBAL_ORDER - 1)
        result = _worker_core_try_batch(r_val)
        
        if result[0] == "SUCCESS":
            return result # Return the full ("SUCCESS", data) tuple
        else:
            # Update local batch stats with the "STATS" dict returned
            batch_stats.update(result[1])
    
    return ("STATS", dict(batch_stats))


def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order,
                              max_tries=None, num_workers=None,
                              window_size=2048, sample_k=32, batch_candidates=512,
                              offset_coeffs=None):
    """
    Parallel search for smooth divisor with restored serialization and live progress.

    NOTE: This function now RAISES if no decomposition found (instead of returning (None, None)),
    to follow the "loud failure" policy you requested.
    """

    # (existing body left mostly intact) -- we only change end behavior to raise
    from multiprocessing import Pool, cpu_count
    import random
    import time
    import sys

    num_workers = cpu_count() if num_workers is None else num_workers
    p_int = int(p)
    order_int = int(order)
    R = PolynomialRing(K, 'x')

    # --- Setup Factor Base Cache ---
    fb_roots = sorted(list(root_to_idx.keys()))
    fb_y_cache = {}
    for x_val in fb_roots:
        y2 = int(f_poly(x_val))
        if y2 == 0:
            continue
        if pow(y2, (p_int - 1) // 2, p_int) == 1:
            y_can = tonelli_shanks(y2, p_int)
            fb_y_cache[int(x_val)] = int(min(y_can, p_int - y_can))

    fb_root_list = list(root_to_idx.keys())
    if len(fb_root_list) == 0:
        raise ValueError("Empty factor base provided to find_smooth_decomposition")

    sample_roots = [int(r) for r in random.sample(fb_root_list, min(sample_k, len(fb_root_list)))]
    coeffs_genus2 = [int(c) for c in f_poly.list()]
    coeffs_genus2.reverse()

    # --- FIX: Restore Serialization of Mumford Coordinates ---
    gen_mumford = None
    if generator is not None:
        gen_u = [int(c) for c in generator[0].list()]
        gen_v = [int(c) for c in generator[1].list()]
        gen_mumford = (gen_u, gen_v)

    target_mumford = None
    if target_point is not None:
        target_u = [int(c) for c in target_point[0].list()]
        target_v = [int(c) for c in target_point[1].list()]
        target_mumford = (target_u, target_v)

    initargs = (
        gen_mumford, target_mumford, root_to_idx, sample_roots,
        fb_y_cache, p_int, order_int, window_size, offset_coeffs
    )

    # --- Search Execution with Progress Tracking ---
    max_tries = max_tries or 10000000
    total_batches = (max_tries + batch_candidates - 1) // batch_candidates
    tasks = [(random.randint(0, 2**31 - 1), batch_candidates) for _ in range(total_batches)]

    start_time = time.time()
    total_tried = 0
    total_hits = 0

    print(f"  [Search] Targeting prime order {order_int}")
    print(f"  [Search] Workers: {num_workers} | Factor Base: {len(fb_roots)}")

    GLOBAL_FIELD = GF(p)
    GLOBAL_RING = PolynomialRing(GLOBAL_FIELD, 'x')
    global _GLOBAL_F_POLY
    _GLOBAL_F_POLY = f_poly_from_coeffs(coeffs_genus2, p)  # uses GLOBAL_FIELD internally
    print("_GLOBAL_F_POLY", _GLOBAL_F_POLY)

    with Pool(processes=num_workers, initializer=_worker_init, initargs=initargs) as pool:
        try:
            for result in pool.imap_unordered(find_smooth_decomposition_worker, tasks):
                if result is None:
                    continue

                status, data = result
                if status == "SUCCESS":
                    (r_val, row_vec, off_idx) = data
                    print(f"\n  [!] SUCCESS: Anchor found in {total_tried} tries.")

                    # Finalize row with offset if needed
                    if off_idx >= 0 and offset_coeffs:
                        (s, pp, v0, v1) = offset_coeffs[off_idx]
                        u_off = R.gen()**2 - K(int(s))*R.gen() + K(int(pp))
                        v_off = K(int(v1))*R.gen() + K(int(v0))
                        off_row = get_relation_row([u_off, v_off], root_to_idx, f_poly, p_int)
                        if off_row:
                            for idx, val in off_row.items():
                                row_vec[idx] = row_vec.get(idx, 0) - val
                                if row_vec[idx] == 0:
                                    del row_vec[idx]

                    pool.terminate()
                    # return anchor scalar and anchor row
                    return r_val, {int(k): int(v) for k, v in row_vec.items()}

                elif status == "STATS":
                    total_tried += data.get('tried', 0)
                    total_hits += (data.get('tried', 0) - data.get('sample_miss', 0))

                    # Live Diagnostic Update
                    elapsed = time.time() - start_time
                    rate = total_tried / elapsed if elapsed > 0 else 0
                    if not total_hits % 100000:
                        sys.stdout.write(
                            f"\r  Trying... Total: {total_tried} | Sample Hits: {total_hits} | Speed: {rate:.1f} t/s"
                        )
                        sys.stdout.flush()

        except KeyboardInterrupt:
            pool.terminate()
            raise

    # If we exit the loop without success, raise loudly (you asked code to explode)
    raise RuntimeError("find_smooth_decomposition: exhausted search without finding a smooth decomposition")


def dlp_bsgs(G, Q, order):
    """
    Solve Q = k*G in a cyclic group of given order using BSGS.
    Raises ValueError if no solution exists.
    """
    import math

    m = int(math.ceil(math.sqrt(order)))

    def divisor_to_key(D):
        """Convert Jacobian divisor to hashable tuple of Mumford coords."""
        if D.is_zero():
            return (0,)
        u_coeffs = tuple(int(c) for c in D[0].list())
        v_coeffs = tuple(int(c) for c in D[1].list())
        return (u_coeffs, v_coeffs)

    # baby steps
    table = {}
    R = G.parent()(0)  # identity in Jacobian
    for j in range(m):
        table[divisor_to_key(R)] = j
        R = R + G

    # giant steps: Q - i*m*G
    mG = m * G
    S = Q
    for i in range(m):
        key = divisor_to_key(S)
        if key in table:
            return (i * m + table[key]) % order
        S = S - mG

    raise ValueError("No discrete log found in subgroup")


# In index_calculus.py, replace the extract_factor_base function:


def solve_dlp_index_calculus(valid_rows, g_anchored, q_anchored, ell, verbose=True):
    """
    Solves the sparse linear system modulo ell to find log_G(Q).
    Optimized to use GF(ell) for fast LinBox/IML sparse solver dispatch.
    """
    if ell is None or int(ell) <= 1:
        raise ValueError("Invalid ell provided to solve_dlp_index_calculus")

    # CRITICAL FIX: Use GF(ell) instead of Zmod(ell). 
    # Sage's sparse solve_right for Zmod is often significantly slower because it
    # may not trigger optimized LinBox routines even if ell is prime.
    try:
        R = GF(int(ell))
    except (ValueError, TypeError):
        # Fallback if ell is not prime (though prime_factors ensures it is)
        R = Zmod(int(ell))

    rg, row_g = g_anchored
    rq, row_q = q_anchored

    if row_g is None or row_q is None:
        raise ValueError("Anchor rows (G or Q) cannot be None")

    # Determine number of variables by finding the maximum index across all relations
    num_rels = len(valid_rows)
    max_idx = -1
    for r in valid_rows:
        if r:
            local_max = max(r.keys())
            if local_max > max_idx:
                max_idx = local_max
    if row_g:
        max_idx = max(max_idx, max(row_g.keys()))
    if row_q:
        max_idx = max(max_idx, max(row_q.keys()))
    
    if max_idx < 0:
        raise ValueError("No variables found in relations or anchors")

    num_vars = max_idx + 1

    if verbose:
        print(f"  Building matrix {num_rels+1}x{num_vars} over {R}...")

    # Build matrix entries
    entries = {}
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            entries[(i, idx)] = R(int(count) % int(ell))

    # Add G anchor row at the end
    for idx, count in row_g.items():
        entries[(num_rels, idx)] = R(int(count) % int(ell))

    # Construct matrix
    M_sys = matrix(R, num_rels + 1, num_vars, entries, sparse=True)

    # Construct RHS vector: zeros for relation rows, rg for the G-anchor row
    targets = [R(0)] * num_rels + [R(int(rg) % int(ell))]
    V = vector(R, targets)

    if verbose:
        print(f"  Solving system via solve_right...")
        sys.stdout.flush()

    try:
        # If sparse still feels slow, one can try: logs_vec = M_sys.dense_matrix().solve_right(V)
        # But with GF(ell), sparse solve_right should take < 1 second.
        logs_vec = M_sys.solve_right(V)
    except Exception as e:
        raise RuntimeError(f"Linear solve failed over {R}: {e}")

    # Compute log(Q) = <row_q, logs> - rq  (mod ell)
    sum_logs = R(0)
    for idx, count in row_q.items():
        if idx >= len(logs_vec):
             raise IndexError(f"Q-anchor index {idx} exceeds log vector length {len(logs_vec)}")
        sum_logs += R(int(count) % int(ell)) * logs_vec[idx]

    log_q = R(sum_logs - R(int(rq) % int(ell)))
    
    if verbose:
        print(f"  Linear algebra complete. Log(Q) mod {ell} = {log_q}")
        
    return Integer(int(log_q))


# ============================================================================
# COMBINED FIX: Degree-1 Support + Protected G/Q Injection
# ============================================================================

def get_relation_row(div, root_to_idx, f_poly, p):
    """
    Main process relation builder (non-cached).
    Updated to handle degree-1 (weight 1) divisors for G/Q anchoring.
    """
    u = div[0]
    v = div[1]
    
    # Allow degree 1 or 2 (Weight 1 or 2 divisors)
    if u.degree() not in [1, 2]:
        return None

    try:
        roots = u.roots(K)
    except Exception:
        raise

    # Sum of multiplicities must match degree (full splitting)
    if sum(mult for r, mult in roots) != u.degree():
        return None

    row = {}
    for r_val, mult in roots:
        r_int = int(r_val)
        if r_int not in root_to_idx:
            return None
            
        # Canonical Y check
        y_val = int(v(r_val))
        y2 = int(f_poly(r_val))
        
        if pow(y2, (p-1)//2, p) != 1:
            return None
            
        y_can = tonelli_shanks(y2, p)
        y_can = min(y_can, p - y_can)
        
        idx = root_to_idx[r_int]
        
        if y_val == y_can:
            row[idx] = row.get(idx, 0) + mult
        elif (p - y_val) % p == y_can:
            row[idx] = row.get(idx, 0) - mult
        else:
            return None
        
    return row


def get_relation_row_cached(divisor):
    """
    Checks if a divisor is smooth over the factor base and returns its relation vector.
    Updated to handle degree-1 divisors.
    """
    global _GLOBAL_ROOT_TO_IDX, _GLOBAL_P, _GLOBAL_FB_Y_CACHE
    u_poly, v_poly = divisor[0], divisor[1]

    # Allow degree 1 or 2
    if u_poly.degree() not in [1, 2]:
        return None

    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != u_poly.degree():
        return None

    row = {}
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in _GLOBAL_ROOT_TO_IDX:
            return None

        y_val = int(v_poly(x_elem))
        y_can = _GLOBAL_FB_Y_CACHE.get(x_int)
        
        if y_can is None:
            return None

        idx = _GLOBAL_ROOT_TO_IDX[x_int]
        if y_val == y_can:
            row[idx] = row.get(idx, 0) + int(mult)
        elif (_GLOBAL_P - y_val) % _GLOBAL_P == y_can:
            row[idx] = row.get(idx, 0) - int(mult)
        else:
            return None
    return row


def check_if_in_factor_base(divisor, root_to_idx, f_poly, p):
    """
    Check if a divisor is already expressible over the factor base.
    Returns (scalar, row_dict) if smooth, else (None, None).
    Updated to handle degree-1 divisors.
    """
    try:
        u_poly, v_poly = divisor[0], divisor[1]
        if u_poly.degree() not in [1, 2]:
            return None, None
        
        K = GF(p)
        roots_data = u_poly.roots(K)
        if sum(m for _, m in roots_data) != u_poly.degree():
            return None, None
        
        row = {}
        for x_elem, mult in roots_data:
            x_int = int(x_elem)
            if x_int not in root_to_idx:
                return None, None
            
            y_val = int(v_poly(x_elem))
            y2 = int(f_poly(x_elem))
            
            if pow(y2, (p-1)//2, p) != 1:
                return None, None
            
            y_can = tonelli_shanks(y2, p)
            y_can = min(y_can, p - y_can)
            
            idx = root_to_idx[x_int]
            if y_val == y_can:
                row[idx] = row.get(idx, 0) + int(mult)
            elif (p - y_val) % p == y_can:
                row[idx] = row.get(idx, 0) - int(mult)
            else:
                return None, None
        
        return 1, row
    except Exception:
        raise


def extract_factor_base(divisors, p, verbose=True):
    """
    Extract the factor base without support-based deduplication.
    Updated to handle explicit 'roots' lists from forced G/Q injection.
    """
    all_roots = []
    unique_divisors = []
    seen_divs = set() 
    
    for d in divisors:
        # Create hashable key - support both formats
        if 'u_coeffs' in d:
            key = (tuple(d['u_coeffs']), tuple(d['v_coeffs']))
        else:
            key = (int(d['s']), int(d['p']), int(d['v_0']), int(d['v_1']))
        
        if key in seen_divs:
            continue
        seen_divs.add(key)
        
        roots = []
        if 'roots' in d:
            roots = d['roots']
        else:
            # Standard quadratic reconstruction
            s = int(d['s']) % p
            pp = int(d['p']) % p
            disc = (s*s - 4*pp) % p
            
            if disc == 0:
                r = (s * pow(2, -1, p)) % p
                roots = [r, r]
            elif pow(disc, (p-1)//2, p) == 1:
                sqrt_disc = tonelli_shanks(disc, p)
                r1 = (s + sqrt_disc) * pow(2, -1, p) % p
                r2 = (s - sqrt_disc) * pow(2, -1, p) % p
                roots = [r1, r2]
        
        if roots:
            all_roots.extend(roots)
            unique_divisors.append(d)
    
    unique_roots = sorted(list(set(all_roots)))
    
    if verbose:
        print(f"\n[Factor Base Extraction]")
        print(f"  Total unique divisors: {len(unique_divisors)}")
        print(f"  Distinct x-coordinates: {len(unique_roots)}")
    
    return {
        'roots': unique_roots,
        'size': len(unique_roots),
        'unique_divisors': unique_divisors,
        'root_to_idx': {r: i for i, r in enumerate(unique_roots)}
    }


def canonicalize_divisor_to_factor_base(divisor, r_to_idx, f_p, p):
    """
    Re-express a divisor using canonical y-coordinates matching the factor base.
    Updated to handle degree-1 divisors.
    """
    u_poly = divisor[0]
    v_poly = divisor[1]
    
    if u_poly.degree() not in [1, 2]:
        return None
    
    K = GF(p)
    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != u_poly.degree():
        return None
    
    row = {}
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in r_to_idx:
            return None
        
        y_val = int(v_poly(x_elem))
        y2 = int(f_p(x_elem))
        
        if pow(y2, (p-1)//2, p) != 1:
            return None
        
        y_can = tonelli_shanks(y2, p)
        y_can = min(y_can, p - y_can)
        
        idx = r_to_idx[x_int]
        
        if y_val == y_can:
            row[idx] = row.get(idx, 0) + int(mult)
        elif (p - y_val) % p == y_can:
            row[idx] = row.get(idx, 0) - int(mult)
        else:
            return None
    
    return row


def divisor_to_dict(div_J, p):
    """
    Convert Jacobian divisor to dict format.
    Supports degree 1 & 2 divisors.
    """
    u_poly = div_J[0]
    v_poly = div_J[1]
    deg = u_poly.degree()
    
    if deg not in [1, 2]:
        return None
        
    roots_data = u_poly.roots(GF(p))
    if sum(m for _, m in roots_data) != deg:
        return None  # Not smooth
        
    roots_list = []
    for r, m in roots_data:
        roots_list.extend([int(r)] * m)

    res = {
        'roots': roots_list,
        'u_coeffs': [int(c) for c in u_poly.list()],
        'v_coeffs': [int(c) for c in v_poly.list()],
        'origin': 'keypair'
    }

    # Add standard keys for compatibility
    if deg == 2:
        coeffs_u = u_poly.list()
        res['s'] = int(-coeffs_u[1]) if len(coeffs_u) > 1 else 0
        res['p'] = int(coeffs_u[0]) if len(coeffs_u) > 0 else 0
        coeffs_v = v_poly.list()
        res['v_1'] = int(coeffs_v[1]) if len(coeffs_v) > 1 else 0
        res['v_0'] = int(coeffs_v[0]) if len(coeffs_v) > 0 else 0
    else:
        # Degree 1: use sentinel values
        res['s'] = 0
        res['p'] = 0
        res['v_1'] = 0
        res['v_0'] = 0
         
    return res


# ============================================================================
# MAIN DLP ATTACK - COMBINED FIX
# ============================================================================


def f_poly_from_coeffs(coeffs, p):
    K = GF(p)
    R = PolynomialRing(K, 'x')
    return sage_poly_from_coeffs(coeffs, R)


def u_poly_roots_in_fp(div, p):
    """
    Return list of integer x-roots of the u-polynomial of a Mumford divisor,
    or None if it does not split completely over GF(p).
    """
    K = GF(p)
    u_poly = div[0]

    try:
        roots = u_poly.roots(K)
    except Exception:
        return None

    # Check complete split
    total_mult = sum(m for _, m in roots)
    if total_mult != u_poly.degree():
        return None

    return [int(r) for r, _ in roots]


def _u_poly_roots_in_fp(div, p):
    """
    Return list of integer x-roots of the u-polynomial of a Mumford divisor,
    or None if it does not split completely over GF(p).
    """
    K = GF(p)
    u_poly = div[0]
    try:
        roots = u_poly.roots(K)
    except Exception:
        return None
    total_mult = sum(m for _, m in roots)
    if total_mult != u_poly.degree():
        return None
    return [int(r) for r, _ in roots]


def _build_signed_row_from_divisor(div, r_to_idx, f_p, p):
    """
    Given a Mumford divisor (u,v), build the signed row dict {col_idx: count}
    using the same Min(y, p-y) convention used across the codebase.
    Returns None if any root isn't in r_to_idx or sign resolution fails.
    """
    u_poly, v_poly = div[0], div[1]
    K = GF(p)
    x = K(0)  # placeholder, we evaluate with elements
    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != u_poly.degree():
        return None

    row = {}
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in r_to_idx:
            return None

        # value of v at x_elem (in GF(p) then int)
        y_val = int(v_poly(x_elem))
        # compute canonical y (min of sqrt, p - sqrt)
        y2 = int(f_p(x_elem))
        if pow(y2, (p-1)//2, p) != 1:
            return None
        y_can = tonelli_shanks(y2, p)
        y_can = min(y_can, p - y_can)

        idx = r_to_idx[x_int]
        if y_val == y_can:
            row[idx] = row.get(idx, 0) + int(mult)
        elif (_GLOBAL_P if False else (p - y_val) % p) == y_can:
            # note: using pure ints avoids accidental GF element comparison issues
            row[idx] = row.get(idx, 0) - int(mult)
        else:
            # v(x) doesn't match expected square root canonicalization
            return None
    return row


def is_divisor_fb_smooth_by_u(div, fb_roots_set, p):
    """
    True iff all roots of div.u split into degree-1 factors and lie in fb_roots_set.
    """
    roots = _u_poly_roots_in_fp(div, p)
    if roots is None:
        return False
    return all(r in fb_roots_set for r in roots)


def perform_dlp_attack(G, Q, smooth_divs, p, f_coeffs, order, verbose=True, force_index_calculus=False):
    """
    Index-calculus + BSGS dispatcher with correct genus-2 Q-smooth predicate.

    Important behavior changes from previous versions:
      - Q is considered FB-smooth iff all roots of its u(x) split linearly
        and each x-root appears in the factor base. If Q is not FB-smooth
        we raise immediately (anchoring/random self-reduction is disabled).
      - G must also be FB-smooth (we refuse to attempt expensive anchoring).
    """
    # Input validation
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")
    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    factors = prime_factors(order)
    if not factors:
        raise ValueError("Failed to factor Jacobian order")
    ell = max(factors)
    if ell <= 1:
        raise ValueError("No non-trivial prime factor found")

    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order ell = {ell}")

    # Project to ell-torsion
    cofactor = Integer(order) // Integer(ell)
    if cofactor == 0:
        raise ValueError("Computed cofactor is zero")
    G_ell = cofactor * G
    Q_ell = cofactor * Q
    J0 = G.parent().zero()
    if G_ell == J0:
        raise ValueError("G projects to identity in ell-torsion")
    if verbose:
        print("Projected to ell-torsion")

    # BSGS fallback for small ell
    BSGS_THRESHOLD = 10**6
    if ell < BSGS_THRESHOLD and not force_index_calculus:
        if verbose:
            print(f"\n[Strategy] Subgroup size {ell} < {BSGS_THRESHOLD}")
            print(f"[Strategy] Using BSGS")
        d_log = dlp_bsgs(G_ell, Q_ell, ell)
        if Integer(d_log) * G_ell != Q_ell:
            raise RuntimeError("BSGS discrete log failed verification")
        if verbose:
            print(f"✓ Discrete log found via BSGS: {d_log}")
        return Integer(d_log)

    # Index Calculus path
    if verbose:
        if force_index_calculus:
            print(f"\n[Strategy] FORCING Index Calculus (testing factor base)")
        else:
            print(f"\n[Strategy] Subgroup size {ell} >= {BSGS_THRESHOLD}")
        print(f"[Strategy] Using Index Calculus")

    # Build polynomial f_p (main process)
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    if verbose:
        print("f_p =", f_p)

    # Extract protected roots from G and Q (their u-polynomial roots)
    protected_roots = set()
    for div, name in [(G, 'G'), (Q, 'Q')]:
        roots = _u_poly_roots_in_fp(div, p)
        if roots is None:
            if verbose:
                print(f"  WARNING: {name}.u(x) does not split completely over GF({p})")
            # We choose to fail hard here (no anchoring)
            raise RuntimeError(f"{name} is not split into linear factors over GF({p})")
        protected_roots.update(roots)
    if verbose:
        print(f"  Protected roots from G/Q: {sorted(list(protected_roots))}")

    # Prepare extended divisor list: prefer to include G and Q (as dicts) first
    extended_divs = []
    for div, name in [(G, 'G'), (Q, 'Q')]:
        div_dict = divisor_to_dict(div, p)
        if div_dict:
            extended_divs.append(div_dict)
            if verbose:
                print(f"  Added {name} to smooth divisor list (B-smooth!)")
        else:
            # divisor_to_dict failed (shouldn't happen because we already checked u splits)
            raise RuntimeError(f"Failed to convert {name} to dict representation")

    extended_divs.extend(smooth_divs)

    # Extract factor base and root->index mapping
    fb_data = extract_factor_base(extended_divs, p, verbose=False)
    roots = fb_data['roots']
    r_to_idx = fb_data['root_to_idx']
    unique_divs = fb_data.get('unique_divisors', [])
    if not roots:
        raise RuntimeError("Empty factor base")
    if verbose:
        print(f"  [Factor Base] size: {len(roots)}")

    # Ensure protected roots are in the factor base
    missing_protected = protected_roots - set(roots)
    if missing_protected:
        raise RuntimeError(f"Protected roots missing from FB: {sorted(list(missing_protected))}")
    if verbose:
        print(f"  ✓ Protected roots in factor base: {len(protected_roots & set(roots))}/{len(protected_roots)}")

    # Build relation rows from unique_divs
    valid_rows = []
    for d in unique_divs:
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
            v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p)
        if row:
            valid_rows.append({int(k): int(v) for k, v in row.items()})
    if not valid_rows:
        raise RuntimeError("No valid relations from smooth_divs")
    if verbose:
        print(f"Loaded {len(valid_rows)} factor base relations")

    # Prune factor base preserving protected indices (Phase 1 + Phase 2 same as before)
    protected_indices = {r_to_idx[r] for r in protected_roots if r in r_to_idx}
    if verbose:
        print(f"  Protected indices (locked): {sorted(list(protected_indices))}")

    # Phase 1: remove unused
    used_indices = set(protected_indices)
    for row in valid_rows:
        used_indices.update(row.keys())
    if len(used_indices) < len(roots):
        if verbose:
            print(f"  [Pruning Phase 1] {len(roots) - len(used_indices)} unused elements")
            print(f"    Keeping {len(protected_indices)} protected")
        used_roots = sorted([roots[i] for i in used_indices])
        r_to_idx = {r: i for i, r in enumerate(used_roots)}
        roots = used_roots
        protected_indices = {r_to_idx[r] for r in protected_roots if r in r_to_idx}
        # rebuild valid_rows
        new_valid_rows = []
        for d in unique_divs:
            if 'u_coeffs' in d:
                u_poly = R(d['u_coeffs'])
                v_poly = R(d['v_coeffs'])
            else:
                u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
                v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p)
            if row:
                new_valid_rows.append({int(k): int(v) for k, v in row.items()})
        valid_rows = new_valid_rows
        if verbose:
            print(f"  [Phase 1] Factor base: {len(roots)} elements")

    # Phase 2: rank-based pruning
    entries = {}
    for i, row in enumerate(valid_rows):
        for idx, _ in row.items():
            entries[(i, idx)] = 1
    M_test = matrix(GF(2), len(valid_rows), len(roots), entries, sparse=True)
    actual_rank = M_test.rank()
    if actual_rank < len(roots):
        if verbose:
            print(f"  [Pruning Phase 2] Rank {actual_rank} < {len(roots)} cols")
            print(f"    Protecting {len(protected_indices)} indices")
        M_rref = M_test.rref()
        pivot_cols = set()
        for i in range(M_rref.nrows()):
            for j in range(M_rref.ncols()):
                if M_rref[i, j] == 1:
                    pivot_cols.add(j)
                    break
        kept_indices_set = pivot_cols.union(protected_indices)
        kept_indices = sorted(list(kept_indices_set))
        if verbose:
            only_protected = protected_indices - pivot_cols
            if only_protected:
                print(f"    Kept {len(only_protected)} protected non-pivots")
        kept_roots = [roots[i] for i in kept_indices]
        r_to_idx = {r: i for i, r in enumerate(kept_roots)}
        roots = kept_roots
        # rebuild rows
        new_valid_rows = []
        for d in unique_divs:
            if 'u_coeffs' in d:
                u_poly = R(d['u_coeffs'])
                v_poly = R(d['v_coeffs'])
            else:
                u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
                v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p)
            if row:
                new_valid_rows.append({int(k): int(v) for k, v in row.items()})
        valid_rows = new_valid_rows
        if verbose:
            print(f"  [Phase 2] Factor base: {len(roots)} elements (Pivot + Protected)")

    # Final protected check
    final_protected = protected_roots & set(roots)
    if len(final_protected) != len(protected_roots):
        missing = protected_roots - final_protected
        raise RuntimeError(f"Pruning removed protected roots: {sorted(list(missing))}")
    if verbose:
        print(f"  ✓ All {len(protected_roots)} protected roots survived")

    # Diagnostics
    if verbose:
        print(f"\n[Matrix Diagnostics]")
        print(f"  Relations (rows): {len(valid_rows)}")
        print(f"  Factor base (cols): {len(roots)}")
        col_counts = [0] * len(roots)
        for row in valid_rows:
            for idx in row.keys():
                col_counts[idx] += 1
        empty_cols = [i for i, c in enumerate(col_counts) if c == 0]
        if empty_cols:
            print(f"  WARNING: {len(empty_cols)} empty columns!")
        else:
            print(f"  All columns have entries")
        total_entries = sum(len(row) for row in valid_rows)
        density = total_entries / (len(valid_rows) * len(roots))
        print(f"  Matrix density: {density:.4f}")
        print(f"  Avg entries per row: {total_entries / len(valid_rows):.2f}")

    # -----------------------------
    # PROPER Q-SMOOTH CHECK (GENUS 2)
    # -----------------------------
    fb_roots_set = set(roots)
    if not is_divisor_fb_smooth_by_u(Q, fb_roots_set, p):
        raise RuntimeError("Q is not FB-smooth: u(x) has factors outside the factor base")
    if not is_divisor_fb_smooth_by_u(G, fb_roots_set, p):
        raise RuntimeError("G is not FB-smooth: u(x) has factors outside the factor base")
    if verbose:
        print(f"  ✓ Q and G pass FB-smooth test (u(x) roots in factor base)")

    # -----------------------------
    # Build canonical/signed rows for G and Q
    # -----------------------------
    # G: canonicalize preferred
    row_g_canon = canonicalize_divisor_to_factor_base(G, r_to_idx, f_p, p)
    if row_g_canon is None:
        # try to build directly from u-roots (should succeed because we tested smoothness)
        row_g = _build_signed_row_from_divisor(G, r_to_idx, f_p, p)
        if row_g is None:
            raise RuntimeError("Failed to canonicalize or build signed row for G")
        rg = 1
    else:
        rg = 1
        row_g = {int(k): int(v) for k, v in row_g_canon.items()}

    # Q: prefer canonical, otherwise expand from u-roots
    row_q_canon = canonicalize_divisor_to_factor_base(Q, r_to_idx, f_p, p)
    if row_q_canon is None:
        row_q = _build_signed_row_from_divisor(Q, r_to_idx, f_p, p)
        if row_q is None:
            raise RuntimeError("Failed to canonicalize or build signed row for Q")
        rq = 1
    else:
        rq = 1
        row_q = {int(k): int(v) for k, v in row_q_canon.items()}

    if verbose:
        print(f"\n[Canonical Check] Built anchor rows:")
        print(f"  G row entries: {len(row_g)}")
        print(f"  Q row entries: {len(row_q)}")

    # Solve DLP via index-calculus linear algebra
    d_log = solve_dlp_index_calculus(
        valid_rows,
        (int(rg) % ell, {int(k): int(v) for k, v in row_g.items()}),
        (int(rq) % ell, {int(k): int(v) for k, v in row_q.items()}),
        ell,
        verbose=verbose
    )

    d_log = int(d_log) % int(ell)

    # Verify in ell-subgroup
    if Integer(d_log) * G_ell != Q_ell:
        raise RuntimeError("Discrete log failed verification in ell-subgroup")

    if verbose:
        print(f"✓ Discrete log found via Index Calculus: {d_log}")

    return Integer(d_log)


def is_divisor_fb_smooth(div, r_to_idx, f_p, p):
    """
    Check if a divisor is actually expressible over the factor base.
    
    For genus-2, this requires THREE conditions:
    1. u(x) splits completely into linear factors over GF(p)
    2. All x-roots lie in the factor base
    3. The y-coordinates are compatible (can build valid signed row)
    
    Returns True only if ALL three conditions hold.
    """
    K = GF(p)
    u_poly = div[0]
    v_poly = div[1]
    
    # Condition 1: u(x) must split completely
    if u_poly.degree() not in [1, 2]:
        return False
    
    try:
        roots_data = u_poly.roots(K)
    except Exception:
        return False
    
    if sum(m for _, m in roots_data) != u_poly.degree():
        return False
    
    # Condition 2: All x-roots must be in factor base
    for x_elem, _ in roots_data:
        x_int = int(x_elem)
        if x_int not in r_to_idx:
            return False
    
    # Condition 3: y-coordinates must be compatible
    # This is the critical check that your old version was missing
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        
        y_val = int(v_poly(x_elem))
        y2 = int(f_p(x_elem))
        
        if pow(y2, (p-1)//2, p) != 1:
            return False
        
        y_can = tonelli_shanks(y2, p)
        y_can = min(y_can, p - y_can)
        
        # Check if v(x) matches canonical square root (with sign)
        if y_val != y_can and (p - y_val) % p != y_can:
            return False
    
    return True
