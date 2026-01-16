from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors, set_random_seed, factor, crt
from sage.all import Zmod, Integer, matrix, vector
from sage.all import GF, PolynomialRing
from sage.all import GF, PolynomialRing, ZZ, Integer
from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Integer, Zmod, factor
from sage.all import factor, Integer
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
from .sparse_linalg_modp import *
from .smoothness import tonelli_shanks, extract_factor_base
from sage.all import GF
    

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


def jacobian_to_dict(J_elem, p):
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


"""
Anchoring fix for inconsistent linear system in perform_dlp_attack.

Key insight: Relations built from divisors with different x-coordinate anchors
are not directly comparable. We need to rebase them all to a single canonical
anchor point.

This patch adds:
1. single_point_row() - builds degree-1 divisor row for a single (x,y) point
2. Rebasing logic in perform_dlp_attack to normalize all relations
"""


# ============================================================================
# PATCH FOR perform_dlp_attack
# ============================================================================
# Insert this logic where valid_rows are being built from smooth_divs


# ============================================================================
# USAGE IN perform_dlp_attack
# ============================================================================
# Replace the section that builds valid_rows from smooth_divs with:
#
# valid_rows, rhs_values = build_anchored_relations(
#     smooth_divs, r_to_idx, f_p, p, fb_y_cache, preferred_x_coords, verbose=verbose
# )
#
# Then proceed with smoothing G and Q as before.


"""
CORRECTED anchoring fix - handles the fact that smooth_divs are NOT scalar multiples of G.

The smooth divisors from Mumford search are just arbitrary smooth divisors over the FB.
They give us LINEAR DEPENDENCIES among FB elements, NOT scalar equations.

Only G and Q (after smoothing) give us actual scalar equations.
"""


def single_point_row(x_int, r_to_idx, f_p, p, fb_y_cache=None):
    """
    Build factor-base row for degree-1 divisor at point (x_int, y_canonical).
    
    Returns:
        dict {col_idx: 1} if x_int is in factor base and has canonical y
        None otherwise
    """
    x_int = int(x_int)
    
    if x_int not in r_to_idx:
        return None
    
    # Get canonical y-coordinate
    if fb_y_cache is not None and x_int in fb_y_cache:
        y_can = fb_y_cache[x_int]
    else:
        K = GF(p)
        y2 = int(f_p(K(x_int)))
        
        if y2 == 0:
            y_can = 0
        elif pow(y2, (p - 1) // 2, p) != 1:
            return None
        else:
            y_can = tonelli_shanks(y2, p)
            y_can = int(min(y_can, p - y_can))
    
    idx = r_to_idx[x_int]
    return {idx: 1}


def rebase_relation_row(row, current_anchor_x, base_anchor_x, r_to_idx, f_p, p, fb_y_cache=None):
    """
    Rebase a relation row from current_anchor to base_anchor.
    
    Mathematically: D' = D - (current_anchor) + (base_anchor)
    """
    if current_anchor_x == base_anchor_x:
        return dict(row)
    
    current_row = single_point_row(current_anchor_x, r_to_idx, f_p, p, fb_y_cache)
    base_row = single_point_row(base_anchor_x, r_to_idx, f_p, p, fb_y_cache)
    
    if current_row is None or base_row is None:
        return None
    
    new_row = dict(row)
    
    # Subtract current anchor
    for col, val in current_row.items():
        new_row[col] = new_row.get(col, 0) - val
        if new_row[col] == 0:
            del new_row[col]
    
    # Add base anchor
    for col, val in base_row.items():
        new_row[col] = new_row.get(col, 0) + val
    
    return new_row


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_quadratic_residue(a_int, p_int):
    return pow(a_int % p_int, (p_int - 1) // 2, p_int) == 1

def sage_poly_from_coeffs(coeffs, R):
    """Build polynomial from coefficient list (highest degree first)"""
    result = R(0)
    for c in coeffs:
        result = result * R.gen() + R(c)
    return result

# ============================================================================
# DIVISOR SMOOTHNESS CHECKING
# ============================================================================

def is_divisor_fb_smooth(div, r_to_idx, f_p, p, fb_y_cache=None):
    """
    True iff `div` (Mumford (u,v) over GF(p)) splits into degree-1 factors
    and each root is present in r_to_idx with compatible y sign.
    """
    K = GF(p)
    u_poly = div[0]
    v_poly = div[1]

    if u_poly.degree() not in (1, 2):
        return False

    try:
        roots_data = u_poly.roots(K)
    except Exception:
        return False

    if sum(m for _, m in roots_data) != u_poly.degree():
        return False

    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in r_to_idx:
            return False

        try:
            y_val = int(v_poly(x_elem))
        except Exception:
            return False

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


def canonicalize_divisor_to_factor_base(divisor, r_to_idx, f_p, p):
    """
    Re-express a divisor using canonical y-coordinates matching the factor base.
    Returns sparse row dict or None.
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


def _build_signed_row_from_divisor(div, r_to_idx, f_p, p):
    """
    Build signed row dict from Mumford divisor with sign resolution.
    Fallback when canonicalize_divisor_to_factor_base fails.
    """
    u_poly, v_poly = div[0], div[1]
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


# ============================================================================
# MAIN DLP ATTACK
# ============================================================================


# ============================================================================
# LINEAR ALGEBRA SOLVER
# ============================================================================

def solve_dlp_index_calculus(valid_rows, rhs_values, q_anchored, modulus, verbose=True):
    """
    Solves sparse linear system Ax = b (mod modulus).
    
    System structure:
      - Most rows: homogeneous relations (RHS=0)
      - Last row: G relation (RHS = 1 + alpha_g)
      - q_anchored: (beta_q, row_q) for computing final discrete log
    """
    assert modulus is not None and int(modulus) > 1, "Invalid modulus"
    
    N = Integer(modulus)
    rq, row_q = q_anchored

    num_rels = len(valid_rows)
    
    # Determine system dimensions
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
        sys.stdout.flush()

    # Check if we should use Hensel lifting
    factors = list(factor(N))
    
    if len(factors) == 1:
        p, exp = factors[0]
        if verbose:
            print(f"  [Solver] Using Hensel lifting for p^k = {p}^{exp}")
            sys.stdout.flush()
        
        return solve_linear_system_hensel_lift(
            valid_rows, rhs_values, row_q, rq,
            int(p), int(exp), num_vars, verbose=verbose
        )
    
    # Direct solve over Z/NZ
    if verbose:
        print(f"  [Solver] Solving directly over Z/{N}Z")
        sys.stdout.flush()
    
    K = Zmod(N)
    
    # Build sparse matrix
    entries = {}
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            val = K(int(count))
            if val != 0:
                entries[(i, idx)] = val
    
    if verbose:
        print(f"  [Matrix] Non-zero entries: {len(entries)}")
        print(f"  [Matrix] Density: {100.0 * len(entries) / (num_rels * num_vars):.3f}%")
        sys.stdout.flush()
    
    A_mod_N = matrix(K, num_rels, num_vars, entries, sparse=True)
    b_vec = vector(K, [K(int(v)) for v in rhs_values])
    
    if verbose:
        print(f"  [Solver] Computing solution...")
        sys.stdout.flush()
    
    try:
        sol = A_mod_N.solve_right(b_vec)
    except ValueError as e:
        raise RuntimeError(f"Linear system solve failed: {e}")
    
    # Compute discrete log from solution
    sum_logs = K(0)
    for idx, count in row_q.items():
        if idx < len(sol):
            sum_logs += K(int(count)) * sol[idx]
    
    log_val = (sum_logs - K(int(rq)))
    
    if verbose:
        print(f"  [Solver] ✓ Solution computed")
        sys.stdout.flush()
    
    return Integer(log_val)


def solve_linear_system_hensel_lift(valid_rows, rhs_values, row_q, rq, 
                                    p, exponent, num_vars, verbose=True):
    """
    Hensel lifting solver for systems mod p^k.
    Lifts solution iteratively from mod p to mod p^k.
    """
    num_rels = len(valid_rows)
    K = GF(p)
    
    # Build matrix mod p
    entries = {}
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            val = K(int(count))
            if val != 0:
                entries[(i, idx)] = val
            
    A_mod_p = matrix(K, num_rels, num_vars, entries, sparse=True)
    b_vec_p = vector(K, [K(val) for val in rhs_values])

    if verbose:
        print(f"  [Hensel] Starting lift from mod {p} to mod {p}^{exponent}")
        sys.stdout.flush()

    # Sparse matrix-vector multiply helper
    def sparse_mat_vec_mult_int(vec_int):
        res = [0] * num_rels
        for i, rel in enumerate(valid_rows):
            acc = 0
            for idx, count in rel.items():
                if idx < len(vec_int):
                    acc += int(count) * vec_int[idx]
            res[i] = acc
        return res

    b_int = [int(val) for val in rhs_values]
    x_accum = [0] * num_vars
    p_pow = 1
    
    for k in range(exponent):
        if verbose:
            print(f"    [Hensel] Step {k+1}/{exponent} (mod {p}^{k+1})")
            sys.stdout.flush()
        
        if k == 0:
            target_int = list(b_int)
        else:
            Ax = sparse_mat_vec_mult_int(x_accum)
            target_int = [(b_val - ax_val) // p_pow for b_val, ax_val in zip(b_int, Ax)]
        
        target_vec_p = vector(K, [K(val) for val in target_int])
        
        try:
            sol_p = A_mod_p.solve_right(target_vec_p)
        except ValueError as e:
            raise RuntimeError(f"Hensel lift failed at step {k+1}: {e}")
        
        sol_int = [int(x) for x in sol_p]
        for i in range(len(sol_int)):
            x_accum[i] += sol_int[i] * p_pow
        p_pow *= p

    # Compute final discrete log
    sum_logs = 0
    for idx, count in row_q.items():
        if idx < len(x_accum):
            sum_logs += int(count) * x_accum[idx]
            
    final_mod = p**exponent
    log_val = (sum_logs - int(rq)) % final_mod
    
    if verbose:
        print(f"  [Hensel] ✓ Lift complete")
        sys.stdout.flush()
    
    return Integer(log_val)


def _legacy_build_relations_from_mumford(smooth_divs, G, Q, p, f_coeffs, verbose=True):
    """
    Convert legacy 'smooth_divs' (list of mumford-divisor dicts) into:
      - valid_rows (homogeneous relations, already rebased)
      - rhs_values (zeros for homogeneous relations)
      - fb_roots, r_to_idx, fb_y_cache used by smoothing of G/Q

    This function:
      - constructs an extended_divs list containing G and Q (as jacobian dicts)
      - calls extract_factor_base(extended_divs, ...)
      - calls build_anchored_relations_homogeneous(...) to produce valid_rows
    Returns: (valid_rows, rhs_values, fb_roots, r_to_idx, fb_y_cache)
    Raises RuntimeError on failure.
    """
    # sanity
    if smooth_divs is None or not isinstance(smooth_divs, (list, tuple)):
        raise RuntimeError("_legacy_build_relations_from_mumford: expected list of divisors")

    # Build polynomial + ring for extract_factor_base call (extract_factor_base expects f_p as sage poly)
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)

    # Prepare extended divisors for factor-base extraction:
    extended_divs = []
    # include G/Q as jacobian dicts to ensure their x-coordinates are present in FB
    try:
        extended_divs.append(jacobian_to_dict(G, p))
        extended_divs.append(jacobian_to_dict(Q, p))
    except Exception:
        # If G/Q are not Jacobian objects (edge-case), just skip them:
        pass

    # append the provided smooth_divs (assumed dicts with u_coeffs/v_coeffs or s/p/v_*)
    extended_divs.extend(list(smooth_divs))

    if verbose:
        print(f"  [Legacy->Relations] Building factor base from {len(extended_divs)} sample divisors...")

    fb_data = extract_factor_base(extended_divs, p, f_p, verbose=False)
    fb_roots = fb_data.get('roots', [])
    r_to_idx = fb_data.get('root_to_idx', {})
    fb_y_cache = fb_data.get('fb_y_cache', {})

    if len(r_to_idx) == 0:
        raise RuntimeError("_legacy_build_relations_from_mumford: empty factor base extracted")

    if verbose:
        print(f"  [Legacy->Relations] Factor base size: {len(r_to_idx)}")

    # Build homogeneous relations from the supplied smooth_divs (rebased to canonical anchor)
    # [FIXED]: Unpack 3 values, though we might not use rhs_coeffs_a/b directly if they are 0
    valid_rows, rhs_a, rhs_b = build_anchored_relations_homogeneous(
        smooth_divs, r_to_idx, f_p, p, fb_y_cache,
        preferred_x_coords=None,  # allow function to choose fallback anchor (G/Q anchors already injected)
        verbose=verbose
    )

    if not valid_rows:
        raise RuntimeError("_legacy_build_relations_from_mumford: no valid homogeneous relations built")
    
    # Construct RHS values. If relations are a*G + b*Q = Sum(FB), then Sum(FB) - a*G - b*Q = 0
    # But usually this legacy function produces FB dependencies (RHS=0). 
    # If the inputs have 'scalar_a'/'scalar_b', we might need to handle them differently
    # For now, stick to the assumption that these are pure FB relations (rhs=0).
    # If scalars are present, valid_rows will represent Sum(FB), so we just set rhs to 0
    # and let the solver handle the extra G/Q variables if it supports them.
    # However, existing perform_dlp_attack handles G and Q separately.
    rhs_values = [0] * len(valid_rows)

    return valid_rows, rhs_values, fb_roots, r_to_idx, fb_y_cache


   

def get_largest_prime_factor(n):
    """
    Return the largest prime factor (as sage Integer) of n.
    Raises RuntimeError if factorization fails or n==1.
    """
    n = Integer(n)
    if n <= 1:
        raise RuntimeError("Invalid order for factorization")
    fac = factor(n)
    if not fac:
        raise RuntimeError("Factorization returned empty")
    # fac is sequence of (prime, exp)
    primes = [Integer(p) for p, e in fac]
    if not primes:
        raise RuntimeError("No prime factors found")
    return max(primes)


def project_relations_and_solve_mod_l(valid_rows, rhs_values, q_anchored, full_order, G, Q,
                                      verbose=True):
    """
    Project the assembled relations system to the largest-prime-factor modulus ell,
    solve A x = b (mod ell) and return x_log (Integer mod ell).

    Arguments:
      valid_rows   : list of sparse dicts {col_idx: coeff_int} representing homogeneous + G rows
      rhs_values   : list of ints (RHS) aligned with valid_rows
      q_anchored   : tuple (rq, row_q) where rq is int (beta_q) and row_q is sparse dict
      full_order   : full Jacobian order |J|
      G, Q         : Sage Jacobian elements (used only for final subgroup verification)
    Returns:
      (d_mod_ell : Integer)  discrete log modulo ell (i.e. in Z/ellZ)
    Raises:
      RuntimeError on failure; instructs to consider block Wiedemann if direct solve fails.
    """
    # --- 1. compute ell and cofactor h ---
    ell = int(get_largest_prime_factor(full_order))
    h = int(Integer(full_order) // Integer(ell))

    if verbose:
        print(f"  [Project] Full order: {full_order}")
        print(f"  [Project] Largest prime ℓ: {ell}, cofactor h: {h}")
        sys.stdout.flush()

    # --- 2. construct sparse matrix over Z/ellZ ---
    K = Zmod(ell)
    num_rels = len(valid_rows)
    max_idx = -1
    for r in valid_rows:
        if r:
            max_idx = max(max_idx, max(r.keys()))
    rq, row_q = q_anchored
    if row_q:
        max_idx = max(max_idx, max(row_q.keys()))
    num_vars = max_idx + 1

    if verbose:
        print(f"  [Matrix] Building A (mod {ell}): {num_rels} x {num_vars}")
        sys.stdout.flush()

    entries = {}
    nonzero = 0
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            if idx < 0:
                continue
            val = K(int(count) % ell)
            if val != 0:
                entries[(i, idx)] = val
                nonzero += 1

    # RHS vector mod ell
    b_vec = vector(K, [K(int(v) % ell) for v in rhs_values])

    # build sparse matrix
    A_mod_ell = matrix(K, num_rels, num_vars, entries, sparse=True)

    if verbose:
        print(f"  [Matrix] Non-zero entries: {nonzero}")
        sys.stdout.flush()

    # --- 3. Attempt direct solve over Z/ellZ ---
    try:
        if verbose:
            print(f"  [Solver] Attempting direct solve over Z/{ell}Z ...")
            sys.stdout.flush()
        sol = A_mod_ell.solve_right(b_vec)  # may raise ValueError if inconsistent or undetermined
    except ValueError as e:
        # Direct dense/sparse solve failed — recommend block Wiedemann (sparse iterative)
        raise RuntimeError(
            f"Direct solve over Z/{ell}Z failed: {e}. "
            "Consider running a Block-Wiedemann solver on the projected system (mod ℓ)."
        )

    # --- 4. compute discrete-log from solution (mod ell) ---
    sum_logs = K(0)
    for idx, count in (row_q or {}).items():
        if idx < len(sol):
            sum_logs += K(int(count) % ell) * sol[idx]

    log_val_mod_ell = (sum_logs - K(int(rq) % ell))  # this is element of Zmod(ell)

    # convert to Integer in [0, ell-1]
    d_mod_ell = Integer(int(log_val_mod_ell))

    if verbose:
        print(f"  [Solver] Candidate discrete-log (mod ℓ): {d_mod_ell}")
        sys.stdout.flush()

    # --- 5. Verify inside ℓ-subgroup: compare h*Q and (d_mod_ell) * h*G ---
    hG = Integer(h) * G
    hQ = Integer(h) * Q
    verification = Integer(d_mod_ell) * hG

    if verification == hQ:
        if verbose:
            print("  [Verify] ✓ Discrete log verified inside ℓ-subgroup: d (mod ℓ) is correct")
        return d_mod_ell
    else:
        # It's possible the solution solved a different representative (rare). Fail clearly.
        raise RuntimeError(
            "Projected-solution did not verify inside ℓ-subgroup: "
            f"{d_mod_ell} * (h*G) != (h*Q). "
            "Either the system is inconsistent mod ℓ or the factor base / rows are misaligned."
        )


def build_anchored_relations(smooth_divs, r_to_idx, f_p, p, fb_y_cache, 
                           preferred_x_coords=None, verbose=True):
    """
    Build relation rows with scalars.
    
    Simplified Version: Removes 'Anchoring' logic.
    Extracts scalar 'r' from the divisor metadata (d['vector'][0]).
    
    Returns:
        (valid_rows, rhs_values)
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    valid_rows = []
    rhs_values = []
    
    if verbose:
        print(f"  [Relations] Building relations (Direct/No-Anchoring)...")
    
    for d in smooth_divs:
        # Extract scalar r from the vector (assumes vector[0] is coefficient for G)
        vec = d.get('vector', None)
        if vec is None:
            continue
        # Allow crash if vector structure is unexpected, per user preference for loud failures
        try:
            r_val = int(vec[0])
        except (IndexError, TypeError, ValueError):
            continue

        # Build divisor polynomials
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            # Fallback for s/p/v dicts
            try:
                s = int(d['s'])
                pp = int(d['p'])
                v1 = int(d['v_1'])
                v0 = int(d['v_0'])
                u_poly = R.gen()**2 - K(s)*R.gen() + K(pp)
                v_poly = K(v1)*R.gen() + K(v0)
            except (KeyError, ValueError):
                continue
        
        # Get relation row
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p, fb_y_cache=fb_y_cache)
        
        if row:
            valid_rows.append({int(k): int(v) for k, v in row.items()})
            rhs_values.append(r_val)
    
    if verbose:
        print(f"  [Relations] Built {len(valid_rows)} relations")
    
    return valid_rows, rhs_values


# ============================================================================
# MODIFIED perform_dlp_attack TO USE NEW RELATIONS
# ============================================================================

    


def build_anchored_relations_homogeneous(smooth_divs, r_to_idx, f_p, p, fb_y_cache, 
                                        preferred_x_coords=None, verbose=True):
    """
    Build relations from divisors with known (a,b) coefficients.
    Each relation: a*G + b*Q = smooth_divisor
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    valid_rows = []
    rhs_coeffs_a = []  # Coefficients of G
    rhs_coeffs_b = []  # Coefficients of Q
    
    if verbose:
        print(f"  [Relations] Building relations from random combinations...")

    for d in smooth_divs:
        # Extract the (a, b) coefficients
        # Priority 1: scalar_a, scalar_b keys
        if 'scalar_a' in d and 'scalar_b' in d:
            a = int(d['scalar_a'])
            b = int(d['scalar_b'])
        # Priority 2: vector key (assuming dim 2)
        elif 'vector' in d:
            try:
                vec = d['vector']
                if len(vec) >= 2:
                    a = int(vec[0])
                    b = int(vec[1])
                elif len(vec) == 1:
                    a = int(vec[0])
                    b = 0
                else:
                    a = 0
                    b = 0
            except (ValueError, TypeError, IndexError):
                continue
        else:
            continue
        
        # Build divisor polynomials
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            try:
                s = int(d['s'])
                pp = int(d['p'])
                v1 = int(d['v_1'])
                v0 = int(d['v_0'])
                u_poly = R.gen()**2 - K(s)*R.gen() + K(pp)
                v_poly = K(v1)*R.gen() + K(v0)
            except (KeyError, ValueError):
                continue
        
        # Get factor base decomposition of the smooth divisor
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p, fb_y_cache=fb_y_cache)
        
        if row:
            valid_rows.append({int(k): int(v) for k, v in row.items()})
            rhs_coeffs_a.append(a)
            rhs_coeffs_b.append(b)
            
    if verbose:
        print(f"  [Relations] Built {len(valid_rows)} relations")
    
    return valid_rows, rhs_coeffs_a, rhs_coeffs_b


# Add this to index_calculus.py


# ============================================================================
# INTEGRATION POINT: search_main.py calls this
# ============================================================================


# ============================================================================
# MODIFIED perform_dlp_attack TO USE NEW RELATIONS
# ============================================================================

# Add this to index_calculus.py

def ingest_mumford_relations(mumford_divisors, f_coeffs, p, verbose=True):
    """
    Adapter: Convert Mumford divisors to index calculus relations.
    
    This is the ONLY place where Mumford structure crosses into index calculus.
    
    Args:
        mumford_divisors: List of dicts from mumford_reconstruction with keys:
            - 's', 'p', 'v_0', 'v_1' (Mumford coords)
            - 'scalar_a', 'scalar_b' (a*G + b*Q = divisor)
            - 'roots' (optional, x-coords where u splits)
        f_coeffs: Curve polynomial coefficients (highest degree first)
        p: Prime modulus
        
    Returns:
        dict with keys:
            - 'relations': list of sparse dicts {col_idx: signed_exp}
            - 'rhs_a': list of ints (coefficients of G)
            - 'rhs_b': list of ints (coefficients of Q)
            - 'fb_roots': sorted list of factor base x-coords
            - 'fb_map': dict {x_int: column_index}
            - 'fb_y_cache': dict {x_int: canonical_y}
    """
    from sage.all import GF, PolynomialRing
    from .smoothness import tonelli_shanks, extract_factor_base
    
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    
    if verbose:
        print(f"\n{'='*70}")
        print("MUMFORD → INDEX CALCULUS ADAPTER")
        print(f"{'='*70}")
        print(f"Input: {len(mumford_divisors)} Mumford divisors")
    
    # Step 1: Extract factor base from the divisors
    fb_data = extract_factor_base(mumford_divisors, p, f_p, verbose=verbose)
    fb_roots = fb_data['roots']
    fb_map = fb_data['root_to_idx']
    fb_y_cache = fb_data.get('fb_y_cache', {})
    
    if verbose:
        print(f"Factor base size: {len(fb_roots)}")
    
    # Step 2: Convert each divisor to a relation vector
    relations = []
    rhs_a = []  # Coefficients of G
    rhs_b = []  # Coefficients of Q
    
    skipped_reasons = {'not_smooth': 0, 'missing_scalars': 0, 'conversion_error': 0}
    
    for idx, div in enumerate(mumford_divisors):
        try:
            # Extract scalar coefficients
            if 'scalar_a' in div and 'scalar_b' in div:
                a = int(div['scalar_a'])
                b = int(div['scalar_b'])
            else:
                skipped_reasons['missing_scalars'] += 1
                continue
            
            # Build Mumford polynomials
            if 'u_coeffs' in div and 'v_coeffs' in div:
                u_poly = R(div['u_coeffs'])
                v_poly = R(div['v_coeffs'])
            else:
                # Fallback: construct from s,p,v_0,v_1
                s = int(div['s'])
                p_val = int(div['p'])
                v0 = int(div.get('v_0', 0))
                v1 = int(div.get('v_1', 0))
                
                x = R.gen()
                u_poly = x**2 - K(s)*x + K(p_val)
                v_poly = K(v1)*x + K(v0)
            
            # Convert to factor base decomposition
            row = _mumford_to_fb_row(u_poly, v_poly, fb_map, f_p, p, fb_y_cache)
            
            if row is None:
                skipped_reasons['not_smooth'] += 1
                continue
            
            # Normalize and store
            row_normalized = _normalize_relation_row(row, p)
            relations.append(row_normalized)
            rhs_a.append(a)
            rhs_b.append(b)
            
        except Exception as e:
            if verbose and skipped_reasons['conversion_error'] < 3:
                print(f"  [Warning] Conversion error for divisor {idx}: {e}")
            skipped_reasons['conversion_error'] += 1
            continue
    
    if verbose:
        print(f"\nConversion results:")
        print(f"  Valid relations: {len(relations)}")
        print(f"  Skipped (not smooth): {skipped_reasons['not_smooth']}")
        print(f"  Skipped (missing scalars): {skipped_reasons['missing_scalars']}")
        print(f"  Skipped (errors): {skipped_reasons['conversion_error']}")
    
    if not relations:
        raise RuntimeError("No valid relations extracted from Mumford divisors")
    
    return {
        'type': 'relations',  # Marker for precomputed relations
        'relations': relations,
        'rhs_a': rhs_a,
        'rhs_b': rhs_b,
        'fb_roots': fb_roots,
        'fb_map': fb_map,
        'fb_y_cache': fb_y_cache
    }


def _mumford_to_fb_row(u_poly, v_poly, fb_map, f_p, p, fb_y_cache):
    """
    Convert a single Mumford divisor (u, v) to factor base row.
    
    Returns:
        dict {col_idx: signed_multiplicity} or None if not smooth
    """
    from sage.all import GF
    from .smoothness import tonelli_shanks
    
    K = GF(p)
    
    # Check degree
    if u_poly.degree() not in [1, 2]:
        return None
    
    # Extract roots
    try:
        roots_data = u_poly.roots(K)
    except:
        return None
    
    # Check complete splitting
    if sum(mult for _, mult in roots_data) != u_poly.degree():
        return None
    
    row = {}
    
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        
        # Check if in factor base
        if x_int not in fb_map:
            return None
        
        # Get y-coordinate from v(x)
        y_val = int(v_poly(x_elem))
        
        # Get canonical y from cache or compute
        if x_int in fb_y_cache:
            y_can = fb_y_cache[x_int]
        else:
            y2 = int(f_p(x_elem))
            if y2 == 0:
                y_can = 0
            elif pow(y2, (p-1)//2, p) != 1:
                return None  # Not a quadratic residue
            else:
                y_can = tonelli_shanks(y2, p)
                y_can = int(min(y_can, p - y_can))
        
        # Determine sign
        col_idx = fb_map[x_int]
        
        if y_val == y_can:
            row[col_idx] = row.get(col_idx, 0) + int(mult)
        elif (p - y_val) % p == y_can:
            row[col_idx] = row.get(col_idx, 0) - int(mult)
        else:
            return None  # Sign mismatch
    
    return row


def _normalize_relation_row(row, modulus):
    """
    Normalize a relation row: reduce mod ℓ, drop zeros.
    
    This is the ONLY place where normalization happens.
    """
    normalized = {}
    
    for col_idx, exp in row.items():
        exp_mod = int(exp) % int(modulus)
        if exp_mod != 0:
            normalized[int(col_idx)] = exp_mod
    
    return normalized


# ============================================================================
# INTEGRATION POINT: search_main.py calls this
# ============================================================================

def prepare_relations_for_dlp(mumford_divisors, f_coeffs, p, verbose=True):
    """
    Single entry point: Mumford divisors → DLP-ready relations.
    
    Call this from search_main.py instead of passing raw divisors.
    
    Returns:
        Relations dict suitable for perform_dlp_attack(precomputed=True)
    """
    relations_data = ingest_mumford_relations(mumford_divisors, f_coeffs, p, verbose)
    
    # Wrap in list for perform_dlp_attack's precomputed format
    return [relations_data]


# ============================================================================
# REWRITTEN perform_dlp_attack TO USE ADAPTER
# ============================================================================

def perform_dlp_attack(G, Q, smooth_divs_or_rels, p, f_coeffs, order,
                       verbose=True, force_index_calculus=False):
    """
    REWRITTEN: Robust wrapper for Index-Calculus DLP using adapter pattern.
    
    Args:
        G: Generator (Jacobian element)
        Q: Target (Jacobian element)  
        smooth_divs_or_rels: Either:
            - List with single dict {'type': 'relations', ...} (from prepare_relations_for_dlp)
            - List of raw Mumford divisor dicts (legacy path, will auto-convert)
        p: Prime modulus
        f_coeffs: Curve coefficients
        order: Full Jacobian order
        verbose: Print diagnostics
        force_index_calculus: Unused (kept for compatibility)
        
    Returns:
        Integer: Discrete log d such that Q = d*G
    """
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")
    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    full_order = Integer(order)
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)

    # Detect if precomputed relations or raw divisors
    precomputed = False
    if (isinstance(smooth_divs_or_rels, (list, tuple)) and 
        len(smooth_divs_or_rels) >= 1 and
        isinstance(smooth_divs_or_rels[0], dict) and
        smooth_divs_or_rels[0].get('type') == 'relations'):
        precomputed = True

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEX CALCULUS DLP ATTACK")
        print(f"{'='*70}")
        print(f"Full Jacobian order |J|: {full_order}")
        sys.stdout.flush()

    # Extract or build relations
    if precomputed:
        data = smooth_divs_or_rels[0]
        valid_rows = data['relations']
        rhs_a = data['rhs_a']
        rhs_b = data['rhs_b']
        fb_map = data['fb_map']
        fb_y_cache = data['fb_y_cache']
        
        if verbose:
            print(f"  [Precomputed] Using {len(valid_rows)} relations")
            print(f"  [Factor Base] Size: {len(fb_map)}")
    else:
        # Legacy path: auto-convert raw Mumford divisors
        if verbose:
            print("  [Legacy] Auto-converting Mumford divisors to relations...")
        relations_bundle = prepare_relations_for_dlp(smooth_divs_or_rels, f_coeffs, p, verbose=False)
        data = relations_bundle[0]
        valid_rows = data['relations']
        rhs_a = data['rhs_a']
        rhs_b = data['rhs_b']
        fb_map = data['fb_map']
        fb_y_cache = data['fb_y_cache']

    if not valid_rows:
        raise RuntimeError("No valid relations available")

    if verbose:
        print(f"  [Relations] {len(valid_rows)} homogeneous relations loaded")
        sys.stdout.flush()

    # ========================================================================
    # SMOOTH GENERATOR G
    # ========================================================================
    
    if verbose:
        print("\n  [Smoothing] Attempting to smooth generator G...")
        sys.stdout.flush()
    
    row_g = None
    alpha_g = 0  # We want G itself, or (1+r)*G if G not smooth
    
    # Try G directly
    if is_divisor_fb_smooth(G, fb_map, f_p, p, fb_y_cache=fb_y_cache):
        row_g = canonicalize_divisor_to_factor_base(G, fb_map, f_p, p)
        if row_g is None:
            row_g = _build_signed_row_from_divisor(G, fb_map, f_p, p)
        if row_g:
            alpha_g = 0
            if verbose:
                print("  [Smoothing] G is smooth over factor base")
    
    # Random walk if G not smooth
    if row_g is None:
        if verbose:
            print("  [Smoothing] G not smooth, trying random multiples...")
            sys.stdout.flush()
        
        for attempt in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_G = (1 + r) * G
            
            if is_divisor_fb_smooth(cand_G, fb_map, f_p, p, fb_y_cache=fb_y_cache):
                row_g = canonicalize_divisor_to_factor_base(cand_G, fb_map, f_p, p)
                if row_g is None:
                    row_g = _build_signed_row_from_divisor(cand_G, fb_map, f_p, p)
                
                if row_g:
                    alpha_g = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth (1+{r})*G after {attempt} attempts")
                    break
        
        if row_g is None:
            raise RuntimeError("Failed to smooth generator G after 2000 attempts")

    # ========================================================================
    # SMOOTH TARGET Q  
    # ========================================================================
    
    if verbose:
        print("\n  [Smoothing] Attempting to smooth target Q...")
        sys.stdout.flush()
    
    row_q = None
    beta_q = 0  # We want Q + r*G for some r
    
    # Try Q directly
    if is_divisor_fb_smooth(Q, fb_map, f_p, p, fb_y_cache=fb_y_cache):
        row_q = canonicalize_divisor_to_factor_base(Q, fb_map, f_p, p)
        if row_q is None:
            row_q = _build_signed_row_from_divisor(Q, fb_map, f_p, p)
        if row_q:
            beta_q = 0
            if verbose:
                print("  [Smoothing] Q is smooth over factor base")
    
    # Random walk if Q not smooth
    if row_q is None:
        if verbose:
            print("  [Smoothing] Q not smooth, trying Q + r*G...")
            sys.stdout.flush()
        
        for attempt in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_Q = Q + r * G
            
            if is_divisor_fb_smooth(cand_Q, fb_map, f_p, p, fb_y_cache=fb_y_cache):
                row_q = canonicalize_divisor_to_factor_base(cand_Q, fb_map, f_p, p)
                if row_q is None:
                    row_q = _build_signed_row_from_divisor(cand_Q, fb_map, f_p, p)
                
                if row_q:
                    beta_q = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth Q+{r}*G after {attempt} attempts")
                    break
        
        if row_q is None:
            raise RuntimeError("Failed to smooth target Q after 2000 attempts")

    # ========================================================================
    # BUILD EXTENDED SYSTEM
    # ========================================================================
    
    # Our relations are: a_i*G + b_i*Q = Sum_j(e_{ij} * FB_j)
    # Rearranging: Sum_j(e_{ij} * log(FB_j)) - a_i*log(G) - b_i*log(Q) = 0
    #
    # We also have:
    #   (1 + alpha_g)*G = Sum_j(g_j * FB_j)  =>  Sum(g_j * log(FB_j)) = (1+alpha_g)
    #   Q + beta_q*G = Sum_j(q_j * FB_j)     =>  Sum(q_j * log(FB_j)) - beta_q = log(Q)
    
    # Extended system has variables: [log(FB_0), ..., log(FB_n), log(G), log(Q)]
    num_fb = len(fb_map)
    idx_log_G = num_fb      # Index for log(G)
    idx_log_Q = num_fb + 1  # Index for log(Q)
    num_vars = num_fb + 2
    
    extended_rows = []
    extended_rhs = []
    
    # Add homogeneous relations: Sum(e_{ij} * log(FB_j)) - a_i*log(G) - b_i*log(Q) = 0
    for i, row in enumerate(valid_rows):
        extended_row = dict(row)  # Copy FB part
        
        # Add -a_i for log(G)
        if rhs_a[i] != 0:
            extended_row[idx_log_G] = -int(rhs_a[i])
        
        # Add -b_i for log(Q)
        if rhs_b[i] != 0:
            extended_row[idx_log_Q] = -int(rhs_b[i])
        
        extended_rows.append(extended_row)
        extended_rhs.append(0)  # Homogeneous
    
    # Add G equation: Sum(g_j * log(FB_j)) + 1*log(G) = (1 + alpha_g)
    g_row = dict(row_g)
    g_row[idx_log_G] = 1
    extended_rows.append(g_row)
    extended_rhs.append(int(1 + alpha_g))
    
    # Add Q equation: Sum(q_j * log(FB_j)) + 0*log(G) + 1*log(Q) = beta_q
    q_row = dict(row_q)
    q_row[idx_log_Q] = 1
    extended_rows.append(q_row)
    extended_rhs.append(int(beta_q))
    
    if verbose:
        print(f"\n  [System] Extended system: {len(extended_rows)} equations, {num_vars} variables")
        print(f"           Variables: {num_fb} FB logs + log(G) + log(Q)")
        sys.stdout.flush()

    # ========================================================================
    # SOLVE USING BLOCK-WIEDEMANN
    # ========================================================================
    
    if verbose:
        print(f"\n  [Solver] Calling Block-Wiedemann solver...")
        sys.stdout.flush()
    
    # The BW solver expects:
    # - valid_rows: homogeneous relations (FB variables only)
    # - rhs_values: zeros for homogeneous system
    # - row_q_dict: FB decomposition of Q (or Q + beta*G)
    # - beta_q: scalar offset
    # It will internally:
    # 1. Project everything mod ℓ
    # 2. Build the augmented system with log(G) and log(Q) variables
    # 3. Solve and extract log(Q)
    
    # Prepare inputs for BW solver
    # Note: valid_rows already contain the homogeneous relations (a*G + b*Q = Sum(FB))
    # We need to pass them as-is, along with the (a, b) coefficients
    
    # The current BW solver signature doesn't handle (a, b) coefficients!
    # We need to build the extended system ourselves before calling BW.
    
    # Extended system has variables: [log(FB_0), ..., log(FB_n), log(G), log(Q)]
    num_fb = len(fb_map)
    idx_log_G = num_fb
    idx_log_Q = num_fb + 1
    
    extended_rows = []
    extended_rhs = []
    
    # Homogeneous relations: Sum(e_ij * log(FB_j)) - a_i*log(G) - b_i*log(Q) = 0
    for i, row in enumerate(valid_rows):
        extended_row = dict(row)  # Copy FB coefficients
        
        if rhs_a[i] != 0:
            extended_row[idx_log_G] = -int(rhs_a[i])
        if rhs_b[i] != 0:
            extended_row[idx_log_Q] = -int(rhs_b[i])
        
        extended_rows.append(extended_row)
        extended_rhs.append(0)
    
    # G smoothing equation: (1 + alpha_g)*G = Sum(g_j * FB_j)
    # In logs: Sum(g_j * log(FB_j)) + log(G) = (1 + alpha_g)
    g_row = dict(row_g)
    g_row[idx_log_G] = 1
    extended_rows.append(g_row)
    extended_rhs.append(int(1 + alpha_g))
    
    # Q smoothing equation: Q + beta_q*G = Sum(q_j * FB_j)
    # In logs: Sum(q_j * log(FB_j)) + log(Q) = beta_q
    q_row_extended = dict(row_q)
    q_row_extended[idx_log_Q] = 1
    extended_rows.append(q_row_extended)
    extended_rhs.append(int(beta_q))
    
    if verbose:
        print(f"  [System] Extended: {len(extended_rows)} eqs, {num_fb + 2} vars (FB + log(G) + log(Q))")
        sys.stdout.flush()
    
    # Now we need to extract log(Q) from the solution
    # The BW solver returns solution[idx_log_Q] directly
    
    # Call BW solver with the extended system
    from .sparse_linalg_modp import solve_dlp_mod_l_block_wiedemann
    
    # TRICK: We'll use row_q as the "target row" and beta_q as the "target scalar"
    # But since log(Q) is now a variable at index idx_log_Q, we need to tell BW
    # that we want to solve for variable idx_log_Q = beta_q - Sum(q_j * log(FB_j))
    
    # Actually, the BW solver computes: beta_q - Sum(q_j * solution[j])
    # So if we give it row_q and beta_q, it will compute:
    #   result = beta_q - Sum_over_FB(q_j * solution[j])
    # But we want solution[idx_log_Q], not that formula.
    
    # FIX: Create a "dummy" row_q that extracts variable idx_log_Q:
    #   We want: solution[idx_log_Q] = ?
    #   So we set row_q_dict = {idx_log_Q: 1} and beta_q = 0
    #   Then: result = 0 - 1*solution[idx_log_Q] = -solution[idx_log_Q]
    #   So: solution[idx_log_Q] = -result
    
    # Wait, that's backwards. Let me re-read the BW solver...
    # From sparse_linalg_modp.py line ~550:
    #   dlog = Integer(beta_q_l)
    #   for k, v in row_q_l.items():
    #       dlog = (dlog - Integer(v) * Integer(coeff)) % Integer(ell)
    #
    # So it computes: dlog = beta_q - Sum(v_k * solution[k])
    #
    # For us to extract solution[idx_log_Q], we need:
    #   dlog = 0 - (-1) * solution[idx_log_Q] = solution[idx_log_Q]
    #
    # So: row_q_dict = {idx_log_Q: -1}, beta_q = 0
    
    row_q_extract = {idx_log_Q: -1}
    beta_q_extract = 0
    
    try:
        log_Q_mod_ell = solve_dlp_mod_l_block_wiedemann(
            valid_rows=extended_rows,
            rhs_values=extended_rhs,
            row_q_dict=row_q_extract,
            beta_q=beta_q_extract,
            full_order=full_order,
            G=G,
            Q=Q,
            verbose=verbose,
            block_size=1,  # Standard Wiedemann
        )
    except Exception as e:
        raise RuntimeError(f"Block-Wiedemann solver failed: {e}") from e
    
    if verbose:
        print(f"\n  [Result] Discrete log (mod ℓ): {log_Q_mod_ell}")
        sys.stdout.flush()
    
    return Integer(log_Q_mod_ell)
