from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors, set_random_seed, factor, crt
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from .smoothness import tonelli_shanks
from .smoothness import extract_factor_base  # replace local duplicate
from search_common import *
import random
import time
import sys
from multiprocessing import Pool, cpu_count
from collections import Counter
from prime_subgroup_projection import *

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


# ============================================================================
# COMBINED FIX: Degree-1 Support + Protected G/Q Injection
# ============================================================================


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


# --- helpers for converting between dict <-> Jacobian element and projecting ---
def _dict_to_jacobian(d, J, R, p):
    """
    Build a Jacobian point J_elem = J([u_poly, v_poly]) from a divisor dict `d`.
    Accepts dicts produced by the worker or by keypair serialization.
    Raises on malformed input.
    """
    K = GF(p)
    x = R.gen()

    # Prefer explicit coefficients if present
    if 'u_coeffs' in d and 'v_coeffs' in d:
        try:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        except Exception as e:
            raise RuntimeError(f"_dict_to_jacobian: failed to build polys from coeffs: {e}")
    else:
        # Fallback to (s,p,v0,v1) entries for degree-2
        try:
            s = int(d['s'])
            pp = int(d['p'])
            v0 = int(d.get('v_0', 0))
            v1 = int(d.get('v_1', 0))
        except Exception as e:
            raise RuntimeError(f"_dict_to_jacobian: missing keys in divisor dict: {e}")

        # Build u(x) = x^2 - s*x + p  (Mumford convention used elsewhere)
        u_poly = x**2 - K(int(s)) * x + K(int(pp))
        v_poly = K(int(v1)) * x + K(int(v0))

    try:
        J_elem = J([u_poly, v_poly])
    except Exception as e:
        raise RuntimeError(f"_dict_to_jacobian: failed to create Jacobian element: {e}")

    return J_elem


def _jacobian_to_dict(J_elem, p):
    """
    Convert a Jacobian point J_elem to the standard dict format used by the rest of the pipeline.
    This includes 'u_coeffs', 'v_coeffs' and canonical s,p,v_0,v_1 for degree-2.
    """
    u_poly = J_elem[0]
    v_poly = J_elem[1]

    u_coeffs = [int(c) for c in u_poly.list()]
    v_coeffs = [int(c) for c in v_poly.list()]

    deg = u_poly.degree()
    res = {
        'u_coeffs': u_coeffs,
        'v_coeffs': v_coeffs,
        'origin': 'projected'  # helpful diagnostics
    }

    if deg == 2:
        # Mumford u = x^2 - s*x + p  -> s = -coeff[1], p = coeff[0]
        coeffs_u = u_coeffs
        res['s'] = int(-coeffs_u[1]) if len(coeffs_u) > 1 else 0
        res['p'] = int(coeffs_u[0]) if len(coeffs_u) > 0 else 0
        coeffs_v = v_coeffs
        res['v_1'] = int(coeffs_v[1]) if len(coeffs_v) > 1 else 0
        res['v_0'] = int(coeffs_v[0]) if len(coeffs_v) > 0 else 0
    else:
        # degree 1 or 0: put sentinel values to keep compatibility
        res['s'] = 0
        res['p'] = 0
        res['v_1'] = 0
        res['v_0'] = 0

    # Optionally compute and store explicit roots if convenient (not required)
    try:
        K = GF(p)
        roots_data = u_poly.roots(K)
        if sum(m for _, m in roots_data) == u_poly.degree():
            res['roots'] = [int(r) for r, _ in roots_data]
    except Exception:
        # If root extraction fails, that's fine — caller can still use u_coeffs/v_coeffs
        pass

    return res


def get_relation_row(div, root_to_idx, f_poly, p, fb_y_cache=None):
    """
    Given a Mumford divisor (u,v) in projected form (already in J[ell]), return
    a signed row dict mapping factor base column -> multiplicity (signed).
    Returns None if divisor is not smooth over the provided factor base.

    Arguments:
      div         : tuple/list-like [u_poly, v_poly] (sage polynomials over GF(p))
      root_to_idx : mapping x_int -> column index (from extract_factor_base)
      f_poly      : polynomial f(x) over GF(p) used to compute canonical sqrt
      p           : prime modulus (int)
      fb_y_cache  : optional dict x_int -> canonical_y to speed sign checks
    """
    K = GF(p)
    R = f_poly.parent()
    u_poly, v_poly = div[0], div[1]

    # Only degree 1 or 2 allowed for relations
    if u_poly.degree() not in (1, 2):
        return None

    try:
        roots_data = u_poly.roots(K)
    except Exception:
        return None

    if sum(m for _, m in roots_data) != u_poly.degree():
        return None

    row = {}
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in root_to_idx:
            return None

        # evaluate v(x) at the root (gives an integer mod p)
        try:
            y_val = int(v_poly(x_elem))
        except Exception:
            return None

        # compute canonical sqrt of f(x)
        if fb_y_cache is not None and x_int in fb_y_cache:
            y_can = fb_y_cache[x_int]
            # fb_y_cache may contain 0 for y^2 == 0
            if y_can == 0:
                # special-case: if y2==0 then y_val must be 0
                if y_val != 0:
                    return None
                # multiplicity sign irrelevant: treat as +mult
                row[root_to_idx[x_int]] = row.get(root_to_idx[x_int], 0) + int(mult)
                continue
        else:
            y2 = int(f_poly(x_elem))
            if pow(y2, (p-1)//2, p) != 1:
                return None
            y_can = tonelli_shanks(y2, p)
            y_can = int(min(y_can, p - y_can))

        idx = root_to_idx[x_int]

        # compare v(x) against canonical root with sign
        if y_val == y_can:
            row[idx] = row.get(idx, 0) + int(mult)
        elif (p - y_val) % p == y_can:
            row[idx] = row.get(idx, 0) - int(mult)
        else:
            # y-value doesn't match expected square root (incompatible)
            return None

    return row


def is_divisor_fb_smooth(div, r_to_idx, f_p, p, fb_y_cache=None):
    """
    True iff `div` (Mumford (u,v) over GF(p)) splits into degree-1 factors
    and each root is present in r_to_idx and has compatible y sign.
    """
    K = GF(p)
    u_poly = div[0]
    v_poly = div[1]

    # degree check
    if u_poly.degree() not in (1, 2):
        return False

    try:
        roots_data = u_poly.roots(K)
    except Exception:
        return False

    if sum(m for _, m in roots_data) != u_poly.degree():
        return False

    # check presence in FB and sign compatibility
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in r_to_idx:
            return False

        # evaluate v(x)
        try:
            y_val = int(v_poly(x_elem))
        except Exception:
            return False

        # canonical sqrt
        if fb_y_cache is not None and x_int in fb_y_cache:
            y_can = fb_y_cache[x_int]
            if y_can == 0:
                if y_val != 0:
                    return False
                else:
                    continue
        else:
            y2 = int(f_p(x_elem))
            if pow(y2, (p-1)//2, p) != 1:
                return False
            y_can = tonelli_shanks(y2, p)
            y_can = int(min(y_can, p - y_can))

        if y_val != y_can and (p - y_val) % p != y_can:
            return False

    return True


# ... (Previous globals and helper functions remain unchanged) ...


# ... (Previous globals and helper functions remain unchanged) ...


# ... (perform_dlp_attack remains unchanged from previous correction) ...


def setup_prime_subgroup_cryptosystem(p, coeffs_genus2, base_pts_x, secret_key):
    """
    Setup the HECC cryptosystem for the prime-order subgroup.
    
    FORCE FIX: Ensures target Q is split (has rational roots) over F_p.
    Returns the ADJUSTED secret key used to generate the split Q.
    """
    F = GF(p)
    R = PolynomialRing(F, 'x')
    f_poly = R([F(c) for c in reversed(coeffs_genus2)])
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    order = compute_jacobian_order(coeffs_genus2, p)
    factorization = factor(order)
    ell = max([Integer(prime) for prime, _ in factorization])
    cofactor = order // ell
    
    print(f"Jacobian order: {order}")
    print(f"Factorization: {factorization}")
    print(f"Largest prime ℓ: {ell}")
    print(f"Cofactor h: {cofactor}")
    
    # Generate base G
    G_original, basex, basey = generate_random_curve_point(f_poly, p)
    print(f"Original base divisor G: {G_original}")
    
    G = Integer(cofactor) * G_original
    
    if G.is_zero():
        raise RuntimeError("Projection sent G to identity - base point was torsion!")
    
    print(f"Projected base divisor G: {G}")
    print(f"Projected G support: {[int(r) for r, _ in G[0].roots()]}")
    
    base_pts_x = [basex]
    print(f"Original base point: ({basex}, {basey})")
    
    # *** FORCE SPLIT Q ***
    print("\n--- Searching for SPLIT target Q ---")
    
    current_secret = Integer(secret_key) % ell
    Q = None
    final_secret = None
    
    for offset in range(1000):
        test_secret = (current_secret + offset) % ell
        if test_secret == 0:
            continue  # Skip identity
            
        Q_candidate = Integer(test_secret) * G
        
        if Q_candidate.is_zero():
            continue
            
        u_poly = Q_candidate[0]
        
        # Check if fully split
        if u_poly.degree() == 0:
            # Identity divisor
            continue
        elif u_poly.degree() == 1:
            # Weight-1, automatically split
            Q = Q_candidate
            final_secret = test_secret
            print(f"Found weight-1 target Q at secret={final_secret}")
            break
        else:
            # Check discriminant for genus-2
            disc = u_poly.discriminant()
            if disc.is_square() and disc != 0:
                # Fully splits
                Q = Q_candidate
                final_secret = test_secret
                print(f"Found SPLIT target Q at secret={final_secret} (offset +{offset})")
                break
    
    if Q is None:
        raise RuntimeError("Failed to find split Q after 1000 attempts")
    
    print(f"Target divisor Q: {Q}")
    print(f"Target Q support: {[int(r) for r, _ in Q[0].roots()]}")
    
    # Extract preferred coordinates
    preferred_x_coords = set()
    for D in [G, Q]:
        u = D[0]
        for root, _ in u.roots():
            preferred_x_coords.add(int(root))
    
    print(f"Preferred x-coordinates (from projected divisors): {sorted(preferred_x_coords)}")
    
    if not preferred_x_coords:
        raise RuntimeError("No preferred coordinates extracted")
    
    return ell, base_pts_x, G, Q, preferred_x_coords, final_secret  # ← RETURN NEW SECRET


def solve_linear_system_hensel_lift(valid_rows, rhs_values, row_q, rq, p, exponent, num_vars, verbose=True):
    """
    Solves Ax = b mod p^k using Hensel Lifting.
    A is built from valid_rows.
    b is built from rhs_values.
    
    After solving for the logs x, computes log(Q) = row_q . x - rq.
    """
    num_rels = len(valid_rows)
    K = GF(p)
    
    # 1. Build Sparse Matrix A and Vector b over GF(p)
    entries = {}
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            val = K(int(count))
            if val != 0:
                entries[(i, idx)] = val
            
    # Create Sage sparse matrix
    A_mod_p = matrix(K, num_rels, num_vars, entries, sparse=True)

    # 2. Prepare integer sparse multiplication for residual calculation
    def sparse_mat_vec_mult_int(vec_int):
        res = [0] * num_rels
        for i, rel in enumerate(valid_rows):
            acc = 0
            for idx, count in rel.items():
                if idx < len(vec_int):
                    acc += int(count) * vec_int[idx]
            res[i] = acc
        return res

    # Initial RHS (b) from the relation scalars
    b_int = [int(val) for val in rhs_values]
    
    # Initialize solution x = 0
    x_accum = [0] * num_vars
    p_pow = 1
    
    for k in range(exponent):
        if verbose:
            print(f"    [Hensel] Step {k+1}/{exponent} (mod {p}^{k+1})")
        
        # Calculate target residue: target = (b - A * x_accum) / p^k
        if k == 0:
            target_int = list(b_int)
        else:
            Ax = sparse_mat_vec_mult_int(x_accum)
            target_int = [(b_val - ax_val) // p_pow for b_val, ax_val in zip(b_int, Ax)]
        
        # Convert target to GF(p) vector
        target_vec_p = vector(K, [K(val) for val in target_int])
        sys.stdout.flush()
        
        # Solve A * sol = target over GF(p)
        try:
            # solve_right works for rectangular matrices (least squares or consistent system)
            # We assume the system is over-determined and consistent.
            sol_p = A_mod_p.solve_right(target_vec_p)
        except ValueError as e:
             raise RuntimeError(f"System inconsistent at Hensel step {k+1} (mod {p}). Matrix rank: {A_mod_p.rank()}. Error: {e}")
        
        # Update accumulator
        sol_int = [int(x) for x in sol_p]
        for i in range(len(sol_int)):
            x_accum[i] += sol_int[i] * p_pow
            
        p_pow *= p

    # Final result vector x_accum contains discrete logs of FB elements
    # Compute log(Q) = (sum(row_q[i] * log(P_i)) - beta)
    sum_logs = 0
    for idx, count in row_q.items():
        if idx < len(x_accum):
            sum_logs += int(count) * x_accum[idx]
            
    final_mod = p**exponent
    # Q_smooth = Q + beta*G => log(Q_smooth) = log(Q) + beta
    # log(Q) = log(Q_smooth) - beta
    # where log(Q_smooth) is the sum_logs computed from FB logs
    log_val = (sum_logs - int(rq)) % final_mod
    return Integer(log_val)


def solve_dlp_index_calculus(valid_rows, rhs_values, q_anchored, modulus, verbose=True):
    """
    Solves the sparse linear system modulo `modulus` where Ax = b.
    valid_rows: list of dicts (rows of A)
    rhs_values: list of ints (elements of b)
    q_anchored: (beta, row_q) for the target
    """
    if modulus is None or int(modulus) <= 1:
        raise ValueError("Invalid modulus provided")

    N = Integer(modulus)
    rq, row_q = q_anchored

    # Determine number of variables (max column index)
    num_rels = len(valid_rows)
    max_idx = -1
    for r in valid_rows:
        if r:
            max_idx = max(max_idx, max(r.keys()))
    if row_q:
        max_idx = max(max_idx, max(row_q.keys()))
    
    num_vars = max_idx + 1

    if verbose:
        print(f"  [Matrix] System size: {num_rels} rows x {num_vars} cols")
        print(f"  [Solver] Modulus: {N}")

    factors = list(factor(N))
    
    # Case 1 & 2: Prime or Prime Power (Hensel)
    if len(factors) == 1:
        p, exp = factors[0]
        if verbose:
            print(f"  [Solver] Using Hensel lifting for p^k = {p}^{exp}")
        
        return solve_linear_system_hensel_lift(
            valid_rows, rhs_values, row_q, rq,
            int(p), int(exp), num_vars, verbose=verbose
        )
    
    # Case 3: Composite (Direct Z/NZ)
    if verbose:
        print(f"  [Solver] Solving directly over Z/{N}Z")
    
    from sage.all import Zmod
    K = Zmod(N)
    
    entries = {}
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            val = K(int(count))
            if val != 0:
                entries[(i, idx)] = val
    
    A_mod_N = matrix(K, num_rels, num_vars, entries, sparse=True)
    b_vec = vector(K, [K(int(v)) for v in rhs_values])
    
    try:
        sol = A_mod_N.solve_right(b_vec)
    except ValueError as e:
        raise RuntimeError(f"Direct solve failed: {e}")
    
    sum_logs = K(0)
    for idx, count in row_q.items():
        if idx < len(sol):
            sum_logs += K(int(count)) * sol[idx]
    
    log_val = (sum_logs - K(int(rq)))
    return Integer(log_val)


def perform_dlp_attack(G, Q, smooth_divs, p, f_coeffs, order, verbose=True, force_index_calculus=False):
    """
    Solves DLP using relations found by the search.
    Treats smooth divisors D as D = r*G, creating equations log(D) = r.
    """
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")

    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    # ----------------------------
    # Setup
    # ----------------------------
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    C = HyperellipticCurve(f_p)
    J = C.jacobian()
    J0 = J.zero()

    factors = prime_factors(order)
    ell = max(factors)
    G_ell = G
    Q_ell = Q
    
    if G_ell == J0:
        raise RuntimeError("G is the identity element")

    # ----------------------------
    # Prepare Factor Base and Relations
    # ----------------------------
    extended_divs = []
    # Add G and Q to factor base pool
    extended_divs.append(_jacobian_to_dict(G_ell, p))
    extended_divs.append(_jacobian_to_dict(Q_ell, p))
    # Add all smooth divisors found
    for d in smooth_divs:
        extended_divs.append(d)

    # Extract Factor Base
    fb_data = extract_factor_base(extended_divs, p, f_p, verbose=False)
    roots = fb_data['roots']
    r_to_idx = fb_data['root_to_idx']
    fb_y_cache = fb_data['fb_y_cache']
    
    if verbose:
        print(f"  Factor base size: {len(roots)}")

    # Build Matrix Rows and RHS
    valid_rows = []
    rhs_values = []
    
    # 1. Process collected smooth divisors as relations D = r*G
    for d in smooth_divs:
        # Extract the scalar 'r' from the vector
        # The search stores the linear combination in 'vector'.
        # We assume vector[0] is the coefficient for G.
        vec = d.get('vector', None)
        if vec is None:
            continue
        
        # Assuming single-section search where S_0 = G, so D = vec[0]*G
        r_val = int(vec[0])
        
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            try:
                u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
                v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            except Exception:
                continue

        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p, fb_y_cache=fb_y_cache)
        if row:
            valid_rows.append({int(k): int(v) for k, v in row.items()})
            rhs_values.append(r_val)

    if not valid_rows:
        raise RuntimeError("No valid relations found in collected divisors")

    if verbose:
        print(f"Loaded {len(valid_rows)} relation rows with scalars.")

    # ----------------------------
    # Smoothing Strategy (Generates anchor relations)
    # ----------------------------
    
    # 1. Smooth Generator: G_smooth = (1 + alpha) * G
    #    Equation: log(G_smooth) = 1 + alpha
    row_g = None
    alpha_g = 0
    max_smoothing_tries = 2000

    if is_divisor_fb_smooth(G_ell, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
        row_g = canonicalize_divisor_to_factor_base(G_ell, r_to_idx, f_p, p) or \
                _build_signed_row_from_divisor(G_ell, r_to_idx, f_p, p)
        if verbose: print("  [Smoothing] Generator G_ell is already smooth.")
    
    if row_g is None:
        if verbose: print(f"  [Smoothing] Generator G_ell not smooth. Attempting random smoothing...")
        for i in range(1, max_smoothing_tries + 1):
            r = ZZ.random_element(1, int(ell))
            cand_G = (1 + r) * G_ell
            
            if is_divisor_fb_smooth(cand_G, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
                row_g = canonicalize_divisor_to_factor_base(cand_G, r_to_idx, f_p, p) or \
                        _build_signed_row_from_divisor(cand_G, r_to_idx, f_p, p)
                if row_g:
                    alpha_g = r
                    if verbose: print(f"  [Smoothing] Found smooth generator G' at iter {i}")
                    break
    
    if row_g is None:
        raise RuntimeError("Failed to smooth Generator G_ell")

    # Add G relation to the system: row_g * logs = 1 + alpha_g
    valid_rows.append({int(k): int(v) for k, v in row_g.items()})
    rhs_values.append(1 + alpha_g)


    # 2. Smooth Target: Q_smooth = Q + beta * G
    #    We solve for logs, then use this to find log(Q)
    row_q = None
    beta_q = 0
    
    if is_divisor_fb_smooth(Q_ell, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
        row_q = canonicalize_divisor_to_factor_base(Q_ell, r_to_idx, f_p, p) or \
                _build_signed_row_from_divisor(Q_ell, r_to_idx, f_p, p)
        if verbose: print("  [Smoothing] Target Q_ell is already smooth.")

    if row_q is None:
        if verbose: print(f"  [Smoothing] Target Q_ell not smooth. Attempting random smoothing...")
        for i in range(1, max_smoothing_tries + 1):
            r = ZZ.random_element(1, int(ell))
            cand_Q = Q_ell + r * G_ell
            
            if is_divisor_fb_smooth(cand_Q, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
                row_q = canonicalize_divisor_to_factor_base(cand_Q, r_to_idx, f_p, p) or \
                        _build_signed_row_from_divisor(cand_Q, r_to_idx, f_p, p)
                if row_q:
                    beta_q = r
                    if verbose: print(f"  [Smoothing] Found smooth target Q' at iter {i}")
                    break

    if row_q is None:
        raise RuntimeError("Failed to smooth Target Q_ell")

    # ----------------------------
    # Solve Linear System
    # ----------------------------
    # Solve M * y = r  (where y are FB logs, r are scalars)
    # Then log(Q) = (row_q * y) - beta_q
    
    d_log_val = solve_dlp_index_calculus(
        valid_rows,
        rhs_values,
        (beta_q, {int(k): int(v) for k, v in row_q.items()}),
        ell,
        verbose=verbose
    )

    # ----------------------------
    # Verify
    # ----------------------------
    if Integer(d_log_val) * G_ell == Q_ell:
        if verbose:
            print(f"✓ Discrete log verified in ell-torsion: {d_log_val}")
        return Integer(d_log_val)
    else:
        raise RuntimeError("Discrete log found did not verify in ell-torsion.")
