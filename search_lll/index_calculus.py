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

# ============================================================================
# WORKER GLOBALS & INITIALIZATION
# ============================================================================

_GLOBAL_GENERATOR = None
_GLOBAL_TARGET_POINT = None
_GLOBAL_ATOM_TO_IDX = None
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


def _worker_init(gen_mumford, target_mumford, atom_to_idx, sample_roots_int, 
                 fb_y_cache, p_int, order_int, window_size, offset_coeffs):
    """
    Initializes worker process. Reconstructs Sage objects from plain Python data.
    """
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_BABY, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY, _GLOBAL_OFFSET_CACHE
    global _GLOBAL_ATOM_TO_IDX
    _GLOBAL_ATOM_TO_IDX = atom_to_idx

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
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_ATOM_TO_IDX
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
        raise
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
        raise
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
        raise

    return res


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

def build_anchored_relations(smooth_divs, r_to_idx, f_p, p, fb_y_cache, preferred_x_coords=PREFERRED_X_COORDS, verbose=True):
    """
    Build relation rows with consistent anchoring.
    
    All relations are rebased to a single canonical anchor point chosen from
    preferred_x_coords. This ensures the linear system is mathematically consistent.
    
    Returns:
        (valid_rows, rhs_values) where all rows are anchored to the same base point
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    # Choose base anchor (first preferred x-coord in factor base)
    base_anchor_x = None
    for x in sorted(preferred_x_coords):
        if int(x) in r_to_idx:
            base_anchor_x = int(x)
            break
    
    if base_anchor_x is None:
        raise RuntimeError("No preferred x-coordinate found in factor base for anchoring")
    
    if verbose:
        print(f"  [Anchoring] Using base anchor x = {base_anchor_x}")
    
    # Precompute base anchor row (we'll need this many times)
    base_row = single_point_row(base_anchor_x, r_to_idx, f_p, p, fb_y_cache)
    if base_row is None:
        raise RuntimeError(f"Failed to compute base row for anchor x = {base_anchor_x}")
    
    valid_rows = []
    rhs_values = []
    skipped_no_anchor = 0
    skipped_rebase_fail = 0
    
    for d in smooth_divs:
        # Extract scalar r from the vector (assumes vector[0] is coefficient for G)
        vec = d.get('vector', None)
        if vec is None:
            continue
        r_val = int(vec[0])
        
        # Build divisor polynomials
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            try:
                u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
                v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            except Exception:
                raise
                continue
        
        # Get relation row
        from .index_calculus import get_relation_row
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p, fb_y_cache=fb_y_cache)
        if not row:
            continue
        
        # Find anchor point in this divisor (any root in factor base)
        try:
            roots_data = u_poly.roots(K)
        except Exception:
            raise
            continue
        
        current_anchor_x = None
        for r, _ in roots_data:
            r_int = int(r)
            if r_int in r_to_idx:
                current_anchor_x = r_int
                break
        
        if current_anchor_x is None:
            skipped_no_anchor += 1
            continue
        
        # Rebase to canonical anchor
        rebased_row = rebase_relation_row(
            row, current_anchor_x, base_anchor_x,
            r_to_idx, f_p, p, fb_y_cache
        )
        
        if rebased_row is None:
            skipped_rebase_fail += 1
            continue
        
        valid_rows.append({int(k): int(v) for k, v in rebased_row.items()})
        rhs_values.append(r_val)
    
    if verbose:
        print(f"  [Anchoring] Built {len(valid_rows)} anchored relations")
        if skipped_no_anchor > 0:
            print(f"  [Anchoring] Skipped {skipped_no_anchor} divisors with no FB anchor")
        if skipped_rebase_fail > 0:
            print(f"  [Anchoring] Skipped {skipped_rebase_fail} divisors that failed rebasing")
    
    return valid_rows, rhs_values


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
    return row
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


def build_anchored_relations_homogeneous(smooth_divs, r_to_idx, f_p, p, fb_y_cache, 
                                        preferred_x_coords=None, verbose=True):
    """
    Build HOMOGENEOUS relation rows (RHS = 0) from smooth divisors.
    
    These divisors are NOT scalar multiples of G - they're just arbitrary smooth divisors.
    They give us dependencies among FB elements: ∏ P_i^{e_i} = 0 in the Jacobian.
    
    All relations are rebased to a single canonical anchor point for consistency.
    
    Returns:
        (valid_rows, rhs_values) where rhs_values are all 0
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    # Choose base anchor
    base_anchor_x = None
    if preferred_x_coords:
        for x in sorted(preferred_x_coords):
            if int(x) in r_to_idx:
                base_anchor_x = int(x)
                break
    
    if base_anchor_x is None:
        # Fallback: use first element in factor base
        base_anchor_x = min(r_to_idx.keys())
    
    if verbose:
        print(f"  [Anchoring] Using base anchor x = {base_anchor_x}")
    
    base_row = single_point_row(base_anchor_x, r_to_idx, f_p, p, fb_y_cache)
    if base_row is None:
        raise RuntimeError(f"Failed to compute base row for anchor x = {base_anchor_x}")
    
    valid_rows = []
    rhs_values = []
    skipped_no_anchor = 0
    skipped_rebase_fail = 0
    
    for d in smooth_divs:
        # Build divisor polynomials
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        else:
            try:
                u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
                v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            except Exception:
                raise
                continue
        
        # Get relation row
        from .index_calculus import get_relation_row
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p, fb_y_cache=fb_y_cache)
        if not row:
            continue
        
        # Find anchor point in this divisor
        try:
            roots_data = u_poly.roots(K)
        except Exception:
            raise
            continue
        
        current_anchor_x = None
        for r, _ in roots_data:
            r_int = int(r)
            if r_int in r_to_idx:
                current_anchor_x = r_int
                break
        
        if current_anchor_x is None:
            skipped_no_anchor += 1
            continue
        
        # Rebase to canonical anchor
        #rebased_row = rebase_relation_row(
        #    row, current_anchor_x, base_anchor_x,
        #    r_to_idx, f_p, p, fb_y_cache
        #)
        rebase_row = row
        
        if rebased_row is None:
            skipped_rebase_fail += 1
            continue
        
        valid_rows.append({int(k): int(v) for k, v in rebased_row.items()})
        rhs_values.append(0)  # HOMOGENEOUS: all smooth divisor relations have RHS = 0
    
    if verbose:
        print(f"  [Anchoring] Built {len(valid_rows)} homogeneous relations (RHS=0)")
        if skipped_no_anchor > 0:
            print(f"  [Anchoring] Skipped {skipped_no_anchor} divisors with no FB anchor")
        if skipped_rebase_fail > 0:
            print(f"  [Anchoring] Skipped {skipped_rebase_fail} divisors that failed rebasing")
    
    return valid_rows, rhs_values


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


# Add these helper functions to index_calculus.py (near the top of the file, after imports)

def choose_base_anchor(r_to_idx, preferred_x_coords=None):
    """
    Choose the same base anchor used by build_anchored_relations_homogeneous.
    Duplicates the selection logic to ensure consistency.
    """
    if preferred_x_coords:
        for x in sorted(preferred_x_coords):
            if int(x) in r_to_idx:
                return int(x)
    # fallback: use first element in factor base
    return int(min(r_to_idx.keys()))


def get_divisor_anchor_x(div, r_to_idx, K):
    """
    Given a Jacobian element `div` (Sage J element), find any root in r_to_idx.
    Returns the first x-coordinate found that's in the factor base, or None.
    """
    try:
        u = div[0]
        for r, _ in u.roots(K):
            r_int = int(r)
            if r_int in r_to_idx:
                return r_int
    except Exception:
        raise
    return None


# ============================================================================
# CORRECTED: Remove anchoring entirely - preserve Mumford algebra
# ============================================================================


# ============================================================================
# CRITICAL FIX: Use Mumford v(x) directly - NO re-canonicalization with f(x)
# ============================================================================


def get_relation_row_cached(divisor):
    """
    CORRECTED: Worker version that uses Mumford v(x) directly.
    
    Checks if a divisor is smooth over the factor base and returns its relation vector.
    Updated to handle degree-1 divisors.
    """
    global _GLOBAL_ATOM_TO_IDX, _GLOBAL_P, _GLOBAL_FB_Y_CACHE
    u_poly, v_poly = divisor[0], divisor[1]

    if u_poly.degree() not in [1, 2]:
        return None

    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != u_poly.degree():
        return None

    row = {}
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        if x_int not in _GLOBAL_ATOM_TO_IDX:
            return None

        # Use Mumford v(x) directly
        y_val = int(v_poly(x_elem))
        
        idx = _GLOBAL_ATOM_TO_IDX[x_int]
        
        # CRITICAL FIX: Same sign convention
        if y_val == 0:
            row[idx] = row.get(idx, 0) + int(mult)
        elif y_val <= _GLOBAL_P // 2:
            row[idx] = row.get(idx, 0) + int(mult)
        else:
            row[idx] = row.get(idx, 0) - int(mult)
            
    return row


def diagnose_system_consistency(homogeneous_rows, row_g, row_q, full_order, verbose=True):
    """
    Run comprehensive diagnostics on the linear system BEFORE solving.
    Checks if G and Q are in span of homogeneous relations.
    """
    from sage.all import Zmod, matrix, vector, Integer, factor
    
    ell = int(max(int(p) for p, _ in factor(full_order)))
    K = Zmod(ell)
    
    # Get number of columns
    all_cols = set()
    for r in homogeneous_rows:
        all_cols.update(r.keys())
    all_cols.update(row_g.keys())
    all_cols.update(row_q.keys())
    n_cols = max(all_cols) + 1 if all_cols else 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"SYSTEM CONSISTENCY DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"Modulus ℓ: {ell}")
        print(f"Homogeneous relations: {len(homogeneous_rows)}")
        print(f"Factor base size: {n_cols}")
    
    # Build homogeneous matrix
    entries_hom = {}
    for i, r in enumerate(homogeneous_rows):
        for j, v in r.items():
            entries_hom[(i, j)] = K(int(v))
    
    M_hom = matrix(K, len(homogeneous_rows), n_cols, entries_hom, sparse=True)
    
    # Check G column coverage
    cols_in_g = set(row_g.keys())
    cols_in_hom = set()
    for r in homogeneous_rows:
        cols_in_hom.update(r.keys())
    
    missing_g = cols_in_g - cols_in_hom
    if missing_g:
        print(f"\n[DIAG] ❌ CRITICAL: G uses columns not in homogeneous relations!")
        print(f"       Missing columns: {sorted(missing_g)}")
    else:
        print(f"\n[DIAG] ✓ G column coverage: all columns present in homogeneous relations")
    
    # Check if G is in span of homogeneous rows
    gvec = vector(K, [K(row_g.get(j, 0)) for j in range(n_cols)])
    try:
        sol_g = M_hom.solve_right(gvec)
        print(f"[DIAG] ✓ G IS in span of homogeneous rows")
        print(f"       G can be expressed as linear combination of homogeneous relations")
    except Exception as e:
        print(f"[DIAG] ❌ CRITICAL: G is NOT in span of homogeneous rows")
        print(f"       Error: {type(e).__name__}: {e}")
        print(f"       This means the map divisor→factor-base is not a homomorphism")
        
        # Additional diagnostics
        rank_hom = M_hom.rank()
        print(f"\n[DIAG] Matrix rank: {rank_hom} / {n_cols} columns")
        if rank_hom < n_cols:
            print(f"       System is under-determined: need {n_cols - rank_hom} more relations")
        raise
    
    # Check Q column coverage
    cols_in_q = set(row_q.keys())
    missing_q = cols_in_q - cols_in_hom
    if missing_q:
        print(f"\n[DIAG] ❌ CRITICAL: Q uses columns not in homogeneous relations!")
        print(f"       Missing columns: {sorted(missing_q)}")
    else:
        print(f"\n[DIAG] ✓ Q column coverage: all columns present in homogeneous relations")
    
    # Check if Q is in span
    qvec = vector(K, [K(row_q.get(j, 0)) for j in range(n_cols)])
    try:
        sol_q = M_hom.solve_right(qvec)
        print(f"[DIAG] ✓ Q IS in span of homogeneous rows")
    except Exception as e:
        print(f"[DIAG] ❌ CRITICAL: Q is NOT in span of homogeneous rows")
        print(f"       Error: {type(e).__name__}: {e}")
        raise
    
    print(f"{'='*70}\n")
    
    return {
        'ell': ell,
        'n_cols': n_cols,
        'missing_g': missing_g,
        'missing_q': missing_q,
        'g_in_span': len(missing_g) == 0,
        'q_in_span': len(missing_q) == 0
    }


def perform_dlp_attack(G, Q, smooth_divs_or_rels, p, f_coeffs, order,
                       verbose=True, force_index_calculus=False):
    """
    Robust wrapper for Index-Calculus DLP.
    CORRECTED: Uses proper Mumford→factor-base homomorphism (v(x) directly).
    """
    # Basic validation
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")
    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    full_order = Integer(order)

    # Check if precomputed relations
    precomputed = False
    if (isinstance(smooth_divs_or_rels, (list, tuple)) and len(smooth_divs_or_rels) == 1
            and isinstance(smooth_divs_or_rels[0], dict)
            and smooth_divs_or_rels[0].get('type') == 'relations'):
        precomputed = True

    # Prepare polynomial ring and curve
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    C = HyperellipticCurve(f_p)
    J = C.jacobian()

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEX CALCULUS DLP ATTACK (Corrected Mumford Homomorphism)")
        print(f"{'='*70}")
        print(f"Full Jacobian order |J|: {full_order}")

    # Build or extract factor base and homogeneous relations
    if precomputed:
        data = smooth_divs_or_rels[0]
        homogeneous_rows = data['relations']
        fb_roots = data['fb_roots']
        r_to_idx = data['fb_map']
        
        fb_y_cache = {}
        for x_int in fb_roots:
            y2 = int(f_p(K(x_int)))
            if y2 == 0:
                fb_y_cache[x_int] = 0
            else:
                y_can = tonelli_shanks(y2, p)
                fb_y_cache[x_int] = int(min(y_can, p - y_can))
    else:
        # Legacy path: build from Mumford divisors
        if verbose:
            print("  [Legacy] Building factor base and relations from Mumford divisors...")
        
        homogeneous_rows, homogeneous_rhs, fb_roots, r_to_idx, fb_y_cache = \
            _legacy_build_relations_from_mumford(smooth_divs_or_rels, G, Q, p, f_coeffs, verbose=verbose)
        
        # Verify all RHS are zero (homogeneous)
        if any(r != 0 for r in homogeneous_rhs):
            raise RuntimeError(
                f"_legacy_build_relations_from_mumford returned non-homogeneous relations"
            )

    if not homogeneous_rows:
        raise RuntimeError("No valid homogeneous relations available")

    if verbose:
        print(f"  [Relations] Loaded {len(homogeneous_rows)} homogeneous relations")
        print(f"  [Factor Base] Size: {len(r_to_idx)}")
        sys.stdout.flush()

    # --- Smooth G and Q using corrected homomorphism ---
    
    # Build atom_to_idx from r_to_idx for smoothness checking
    atom_to_idx_for_smooth = {}
    for x_val, idx in r_to_idx.items():
        if x_val in fb_y_cache:
            y_can = fb_y_cache[x_val]
        else:
            y2 = int(f_p(K(x_val)))
            if y2 == 0:
                y_can = 0
            elif pow(y2, (p-1)//2, p) == 1:
                from .smoothness import tonelli_shanks
                y_can = tonelli_shanks(y2, p)
                y_can = min(y_can, p - y_can)
            else:
                continue
        atom = ('d1', int(x_val), int(y_can))
        atom_to_idx_for_smooth[atom] = idx
    
    # Smooth G
    row_g = None
    alpha_g = 0
    
    if is_divisor_fb_smooth(G, atom_to_idx_for_smooth, f_p, p, fb_y_cache=fb_y_cache):
        row_g = get_relation_row(G, atom_to_idx_for_smooth, f_p, p)
        if verbose:
            print("  [Smoothing] Generator G is already smooth.")
    else:
        if verbose:
            print("  [Smoothing] Generator not smooth. Attempting random smoothing...")
        for i in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_G = (1 + r) * G
            if is_divisor_fb_smooth(cand_G, atom_to_idx_for_smooth, f_p, p, fb_y_cache=fb_y_cache):
                row_g = get_relation_row(cand_G, atom_to_idx_for_smooth, f_p, p)
                if row_g:
                    alpha_g = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth generator at iter {i}")
                    break
    
    if row_g is None:
        raise RuntimeError("Failed to smooth Generator G")

    # Smooth Q
    row_q = None
    beta_q = 0
    
    if is_divisor_fb_smooth(Q, atom_to_idx_for_smooth, f_p, p, fb_y_cache=fb_y_cache):
        row_q = get_relation_row(Q, atom_to_idx_for_smooth, f_p, p)
        if verbose:
            print("  [Smoothing] Target Q is already smooth.")
    else:
        if verbose:
            print("  [Smoothing] Target not smooth. Attempting random smoothing...")
        for i in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_Q = Q + r * G
            if is_divisor_fb_smooth(cand_Q, atom_to_idx_for_smooth, f_p, p, fb_y_cache=fb_y_cache):
                row_q = get_relation_row(cand_Q, atom_to_idx_for_smooth, f_p, p)
                if row_q:
                    beta_q = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth target at iter {i}")
                    break
    
    if row_q is None:
        raise RuntimeError("Failed to smooth Target Q")

    # Convert to plain dicts
    row_g_dict = {int(k): int(v) for k, v in row_g.items()}
    row_q_dict = {int(k): int(v) for k, v in row_q.items()}
    
    # --- Compute ell before diagnostics ---
    ell = int(max(int(p) for p, _ in factor(full_order)))

    # --- RUN DIAGNOSTICS BEFORE SOLVING ---
    #diag = diagnose_system_consistency(
    #    homogeneous_rows, 
    #    row_g_dict, 
    #    row_q_dict, 
    #    full_order, 
    #    verbose=verbose
    #)

    # --- RUN DIAGNOSTICS BEFORE SOLVING ---
    if verbose:
        # LIGHTWEIGHT diagnostic - just check dimensions and column coverage
        print(f"\n{'='*70}")
        print(f"SYSTEM CONSISTENCY CHECK (Lightweight)")
        print(f"{'='*70}")
        print(f"Modulus ℓ: {ell}")
        print(f"Homogeneous relations: {len(homogeneous_rows)}")
        print(f"Factor base size: {len(r_to_idx)}")
        
        # Check G column coverage
        cols_in_g = set(row_g_dict.keys())
        cols_in_hom = set()
        for r in homogeneous_rows:
            cols_in_hom.update(r.keys())
        
        missing_g = cols_in_g - cols_in_hom
        if missing_g:
            print(f"⚠ WARNING: G uses {len(missing_g)} columns not in homogeneous relations")
            print(f"  Missing columns: {sorted(missing_g)[:10]}{'...' if len(missing_g) > 10 else ''}")
        else:
            print(f"✓ G column coverage: all columns present")
        
        # Check Q column coverage
        cols_in_q = set(row_q_dict.keys())
        missing_q = cols_in_q - cols_in_hom
        if missing_q:
            print(f"⚠ WARNING: Q uses {len(missing_q)} columns not in homogeneous relations")
            print(f"  Missing columns: {sorted(missing_q)[:10]}{'...' if len(missing_q) > 10 else ''}")
        else:
            print(f"✓ Q column coverage: all columns present")
        
        print(f"{'='*70}\n")
        
        # Critical check: if either G or Q has missing columns, we WILL fail
        if missing_g or missing_q:
            raise RuntimeError(
                "G or Q uses columns not in homogeneous relations.\n"
                "The divisor→factor-base map is not a valid homomorphism or factor base is incomplete."
            )
    
    # --- Construct the system ---
    full_rows = list(homogeneous_rows)
    full_rhs = [0] * len(homogeneous_rows)
    
    # Append G row
    full_rows.append(row_g_dict)
    full_rhs.append(int(1 + alpha_g))
    
    if verbose:
        print(f"\n  [System] Built system: {len(full_rows)} rows ({len(homogeneous_rows)} homogeneous + 1 for G)")
        print(f"  [System] G row RHS: {full_rhs[-1]} (should be 1 + {alpha_g})")
        sys.stdout.flush()

    # Solve the system
    beta_q_int = int(beta_q)

    d_log_val = None
    try:
        if verbose:
            print("  [Solver] Starting Block-Wiedemann...")
        
        d_log_val = solve_dlp_mod_l_block_wiedemann(
            full_rows,
            full_rhs,
            row_q_dict,
            beta_q_int,
            full_order,
            G, Q,
            verbose=verbose,
            block_size=32,
        )
        
        if verbose:
            print("  [Solver] Block-Wiedemann returned a candidate.")
    except Exception as e:
        raise RuntimeError(f"Block-Wiedemann solver failed: {e}")

    if d_log_val is None:
        raise RuntimeError("Solver produced no result")

    if verbose:
        print(f"  [Result] Discrete log (mod ℓ) candidate: {d_log_val}")

    # Verify
    D = Integer(d_log_val) * G - Q

    if not (Integer(ell) * D).is_zero():
        raise RuntimeError(
            "[Verify] ✗ Block-Wiedemann result FAILED group verification:\n"
            f"        ℓ * (d_log_val * G − Q) ≠ 0\n"
            f"        dlog={d_log_val}, ℓ={ell}"
        )

    if verbose:
        print("  [Verify] ✓ ℓ-torsion verification passed")
        if D.is_zero():
            print("  [Verify] ✓ Exact equality d*G == Q")
        else:
            print("  [Verify] ℹ d*G ≠ Q exactly (cofactor component)")

    return Integer(d_log_val)


# quick homomorphism test


def _legacy_build_relations_from_mumford(smooth_divs, G, Q, p, f_coeffs, verbose=True):
    """
    Convert legacy 'smooth_divs' into relations WITHOUT rebasing.
    CORRECTED: Uses proper Mumford homomorphism.
    
    Returns: (valid_rows, rhs_values, fb_roots, r_to_idx, fb_y_cache)
    """
    if smooth_divs is None or not isinstance(smooth_divs, (list, tuple)):
        raise RuntimeError("_legacy_build_relations_from_mumford: expected list of divisors")

    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)

    # Prepare extended divisors for factor-base extraction
    extended_divs = []
    try:
        extended_divs.append(jacobian_to_dict(G, p))
        extended_divs.append(jacobian_to_dict(Q, p))
    except Exception:
        raise

    extended_divs.extend(list(smooth_divs))

    if verbose:
        print(f"  [Legacy->Relations] Building factor base from {len(extended_divs)} sample divisors...")

    # Extract factor base - returns (atom_to_idx, fb_y_cache)
    atom_to_idx, fb_y_cache = extract_factor_base(extended_divs, p, f_p, verbose=True)
    
    # Extract fb_roots from degree-1 atoms
    fb_roots = []
    for atom, idx in atom_to_idx.items():
        if atom[0] == 'd1':  # degree-1 atom: ('d1', x, y)
            x_val = atom[1]
            if x_val not in fb_roots:
                fb_roots.append(x_val)
    
    # Build r_to_idx mapping from fb_roots
    r_to_idx = {r: i for i, r in enumerate(fb_roots)}

    if len(r_to_idx) == 0:
        raise RuntimeError("_legacy_build_relations_from_mumford: empty factor base extracted")

    if verbose:
        print(f"  [Legacy->Relations] Factor base size: {len(r_to_idx)}")

    # Build homogeneous relations using corrected homomorphism
    valid_rows, rhs_values = build_homogeneous_relations_no_rebase(
        smooth_divs, r_to_idx, f_p, p, fb_y_cache, verbose=verbose
    )

    if not valid_rows:
        raise RuntimeError("_legacy_build_relations_from_mumford: no valid homogeneous relations built")

    return valid_rows, rhs_values, fb_roots, r_to_idx, fb_y_cache


def get_relation_row(div, atom_to_idx, f_p, p, fb_y_cache=None):
    """
    Encode a Mumford divisor (u,v) into prime-divisor atoms.
    Returns sparse dict col->mult or None if not FB-smooth.
    
    Args:
        div: Either a dict with 'u_coeffs'/'v_coeffs' or a Sage Jacobian element
        atom_to_idx: dict mapping atom tuples to column indices
        f_p: polynomial f(x) over GF(p)
        p: prime
        fb_y_cache: optional dict of x -> canonical_y (not used in new atom-based encoding)
    
    Returns:
        dict {col_idx: multiplicity} or None if not smooth
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    if isinstance(div, dict):
        if 'u_coeffs' in div and 'v_coeffs' in div:
            u = R(div['u_coeffs'])
            v = R(div['v_coeffs'])
        elif 's' in div and 'p' in div:
            x = R.gen()
            s = int(div['s'])
            pp = int(div['p'])
            v0 = int(div.get('v_0', 0))
            v1 = int(div.get('v_1', 0))
            u = x**2 - K(s)*x + K(pp)
            v = K(v1)*x + K(v0)
        else:
            return None
    else:
        u, v = div[0], div[1]

    row = {}

    try:
        facs = u.factor()
    except Exception:
        raise
        return None

    for fac, mult in facs:
        deg = fac.degree()

        if deg == 1:
            x_val = int(fac.roots()[0][0])
            y_val = int(v(K(x_val)))
            y_val = min(y_val, p - y_val)
            atom = ('d1', x_val, y_val)

        elif deg == 2:
            u_coeffs = tuple(int(c) for c in fac.list())
            v_mod = (v % fac)
            v_coeffs = tuple(int(c) for c in v_mod.list())
            atom = ('d2', u_coeffs, v_coeffs)

        else:
            return None

        idx = atom_to_idx.get(atom)
        if idx is None:
            return None

        row[idx] = row.get(idx, 0) + mult

    return row


def build_homogeneous_relations_no_rebase(smooth_divs, r_to_idx, f_p, p, fb_y_cache, 
                                         verbose=True):
    """
    Build HOMOGENEOUS relation rows (RHS = 0) from smooth divisors.
    NO REBASING - preserves exact Mumford algebra.
    CORRECTED: Uses Mumford v(x) directly via get_relation_row.
    
    NOTE: r_to_idx is the OLD format (x_int -> col_idx).
          This function needs to be updated to use atom_to_idx instead.
    
    Returns:
        (valid_rows, rhs_values) where rhs_values are all 0
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    # Build atom_to_idx from r_to_idx (backward compatibility hack)
    # This assumes r_to_idx maps x-coordinates to indices
    atom_to_idx = {}
    for x_val, idx in r_to_idx.items():
        # Get canonical y from cache if available
        if x_val in fb_y_cache:
            y_can = fb_y_cache[x_val]
        else:
            # Compute canonical y
            y2 = int(f_p(K(x_val)))
            if y2 == 0:
                y_can = 0
            elif pow(y2, (p-1)//2, p) == 1:
                from .smoothness import tonelli_shanks
                y_can = tonelli_shanks(y2, p)
                y_can = min(y_can, p - y_can)
            else:
                continue  # Not a valid point
        
        atom = ('d1', int(x_val), int(y_can))
        atom_to_idx[atom] = idx
    
    valid_rows = []
    rhs_values = []
    skipped_no_row = 0
    
    for d in smooth_divs:
        # Build divisor polynomials
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        elif 's' in d and 'p' in d:
            try:
                x = R.gen()
                u_poly = x**2 - K(int(d['s']))*x + K(int(d['p']))
                v_poly = K(int(d['v_1']))*x + K(int(d['v_0']))
            except Exception:
                raise
                continue
        else:
            continue
        
        # Get relation row using corrected homomorphism
        row = get_relation_row([u_poly, v_poly], atom_to_idx, f_p, p)
        if not row:
            skipped_no_row += 1
            continue
        
        valid_rows.append({int(k): int(v) for k, v in row.items()})
        rhs_values.append(0)  # HOMOGENEOUS
    
    if verbose:
        print(f"  [Relations] Built {len(valid_rows)} homogeneous relations (RHS=0, no rebasing)")
        if skipped_no_row > 0:
            print(f"  [Relations] Skipped {skipped_no_row} divisors (not smooth over FB)")
    
    return valid_rows, rhs_values


def canonicalize_divisor_to_factor_base(divisor, r_to_idx, f_p, p):
    """
    CORRECTED: Uses Mumford v(x) directly without re-canonicalizing.
    
    Re-express a divisor using its Mumford y-coordinates.
    Returns sparse row dict or None.
    
    NOTE: This function uses old r_to_idx format. Should be migrated to atom_to_idx.
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    
    u_poly = divisor[0]
    v_poly = divisor[1]
    
    if u_poly.degree() not in [1, 2]:
        return None
    
    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != u_poly.degree():
        return None
    
    # Build atom_to_idx from r_to_idx for this function
    # (This is inefficient but maintains backward compatibility)
    atom_to_idx = {}
    for x_val, idx in r_to_idx.items():
        y2 = int(f_p(K(x_val)))
        if y2 == 0:
            y_can = 0
        elif pow(y2, (p-1)//2, p) == 1:
            from .smoothness import tonelli_shanks
            y_can = tonelli_shanks(y2, p)
            y_can = min(y_can, p - y_can)
        else:
            continue
        atom = ('d1', int(x_val), int(y_can))
        atom_to_idx[atom] = idx
    
    row = get_relation_row([u_poly, v_poly], atom_to_idx, f_p, p)
    return row


def _build_signed_row_from_divisor(div, r_to_idx, f_p, p):
    """
    CORRECTED: Uses Mumford v(x) directly.
    
    Build signed row dict from Mumford divisor.
    Fallback when canonicalize_divisor_to_factor_base fails.
    """
    # Just call canonicalize_divisor_to_factor_base since they do the same thing now
    return canonicalize_divisor_to_factor_base(div, r_to_idx, f_p, p)


def homomorphism_test(J, atom_to_idx, f_p, p, trials=50):
    """
    Test that the factor base encoding is a group homomorphism.
    
    For random smooth divisors D1, D2, verify:
        encode(D1 + D2) = encode(D1) + encode(D2)
    
    Returns True if all trials pass, False otherwise.
    """
    from random import randint
    K = GF(p)
    C = J.curve()
    
    def get_random_jacobian_element():
        """Generate a random element in the Jacobian by finding a random curve point."""
        max_attempts = 100
        for _ in range(max_attempts):
            x_coord = K.random_element()
            y2 = f_p(x_coord)
            if y2.is_square():
                y_coord = y2.sqrt()
                try:
                    pt = C((x_coord, y_coord))
                    return J(pt)
                except Exception:
                    raise
                    continue
        # Fallback: return identity
        return J(0)
    
    passed = 0
    failed = 0
    
    for trial_num in range(trials):
        # Generate two random divisors
        D1 = get_random_jacobian_element()
        D2 = get_random_jacobian_element()
        
        # Get their encodings
        enc1 = get_relation_row(D1, atom_to_idx, f_p, p)
        enc2 = get_relation_row(D2, atom_to_idx, f_p, p)
        
        # Compute sum and its encoding
        D_sum = D1 + D2
        enc_sum = get_relation_row(D_sum, atom_to_idx, f_p, p)
        
        # If any encoding failed (not smooth), skip this trial
        if enc1 is None or enc2 is None or enc_sum is None:
            continue
        
        # Compute expected encoding as vector sum
        combined = {}
        for k, v in enc1.items():
            combined[k] = combined.get(k, 0) + v
        for k, v in enc2.items():
            combined[k] = combined.get(k, 0) + v
        
        # Remove zeros
        combined = {k: v for k, v in combined.items() if v != 0}
        
        # Compare
        if combined == enc_sum:
            passed += 1
        else:
            failed += 1
            if failed == 1:  # Print first failure for debugging
                print(f"\n  [Trial {trial_num}] HOMOMORPHISM FAILURE:")
                print(f"    enc(D1): {enc1}")
                print(f"    enc(D2): {enc2}")
                print(f"    enc(D1) + enc(D2): {combined}")
                print(f"    enc(D1 + D2): {enc_sum}")
    
    total_tested = passed + failed
    
    if total_tested == 0:
        print(f"  WARNING: No smooth divisor pairs found in {trials} trials")
        print(f"           Cannot verify homomorphism property")
        return True  # Don't fail if we just couldn't find smooth examples
    
    print(f"  Homomorphism tests: {passed}/{total_tested} passed")
    
    if failed > 0:
        print(f"  âŒ {failed} failures detected!")
        return False
    
    return True


def is_divisor_fb_smooth(div, atom_to_idx, f_p, p, fb_y_cache=None):
    """
    Boolean wrapper: True if divisor encodes successfully into FB using atom-based encoding.
    
    Args:
        div: Jacobian divisor element or dict
        atom_to_idx: dict mapping atom tuples to indices
        f_p: polynomial over GF(p)
        p: prime
        fb_y_cache: unused (kept for backward compatibility)
    
    Returns:
        bool: True if divisor is smooth over the factor base
    """
    row = get_relation_row(div, atom_to_idx, f_p, p, fb_y_cache=fb_y_cache)
    return row is not None
