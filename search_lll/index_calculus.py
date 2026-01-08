from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors, set_random_seed
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from .smoothness import extract_factor_base, tonelli_shanks
from search_common import FINITE_FIELD
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


def get_relation_row(div, root_to_idx, f_poly, p):
    """
    Main process relation builder (non-cached).
    """
    u = div[0]
    v = div[1]
    
    if u.degree() != 2:
        return None

    try:
        roots = u.roots(K)
    except Exception:
        raise
        return None

    if sum(mult for r, mult in roots) != 2:
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

# ============================================================================
# WORKER LOGIC
# ============================================================================


# ============================================================================
# MAIN SEARCH FUNCTION
# ============================================================================


# ============================================================================
# LINEAR ALGEBRA & ATTACK DRIVER
# ============================================================================

def generate_test_keypair(f_poly, p, target_d=None):
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


def solve_dlp_index_calculus(valid_rows, g_anchored, q_anchored, ell, verbose=True):
    """
    Solves the sparse linear system mod ell to find log_G(Q).
    """
    rg, row_g = g_anchored
    rq, row_q = q_anchored
    
    # System: M * x = 0 (relations from factor base search)
    # Augment with G anchor: Sum(row_g_i * x_i) = rg
    # Augment with Q anchor: Sum(row_q_i * x_i) = rq + log(Q) 
    # Wait, anchoring G gives: [rg]G = Smooth => rg = Sum(row_g * x_i)
    # Anchoring Q gives: Q + [rq]G = Smooth => log(Q) + rq = Sum(row_q * x_i)
    
    # We construct a matrix to solve for x_i (logs of factor base elements)
    # Matrix rows: The collected smooth relations.
    # Target vector: Since collected relations are [0] = Smooth, 
    # if relations are just smooth divisors summing to 0 (principal divisors),
    # then M * x = 0.
    
    # NOTE: The relations collected in `valid_rows` (from `index_calculus_factor_base_analysis`) 
    # are linear dependencies among factor base elements that sum to PRINCIPAL divisors.
    # So M * x = 0 (mod ell).
    
    # We need to solve for x relative to one base. 
    # Usually we fix one x_k = 1 or use the G-anchor.
    # G-anchor equation: Sum(row_g * x) = rg (mod ell).
    
    
    # 1. Build Sparse Matrix M from relations
    # valid_rows is a list of dicts {idx: count}
    num_rels = len(valid_rows)
    # Find max index to determine FB size
    max_idx = 0
    for r in valid_rows:
        if r: max_idx = max(max_idx, max(r.keys()))
    if row_g: max_idx = max(max_idx, max(row_g.keys()))
    if row_q: max_idx = max(max_idx, max(row_q.keys()))
    
    num_vars = max_idx + 1
    
    if verbose:
        print(f"  Building sparse matrix {num_rels}x{num_vars}...")
        
    M = matrix(K, num_rels, num_vars, sparse=True)
    
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            M[i, idx] = count
            
    # 2. Add the G-anchor row: Sum(row_g * x) = rg
    # We add this as a row to the system with target value rg.
    # But sage's solve_right expects M*x = v.
    # The existing M rows correspond to principal divisors, so target is 0.
    
    # Append G row
    M_aug = matrix(K, num_rels + 1, num_vars, sparse=True)
    # Copy M (inefficient if dense, but M is sparse)
    # Better: Construct list of triples for sparse matrix
    triples = []
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            triples.append((i, idx, count))
            
    # Add G row at index num_rels
    for idx, count in row_g.items():
        triples.append((num_rels, idx, count))
        
    M_sys = matrix(K, num_rels + 1, num_vars, triples)
    
    # Target vector
    targets = [0] * num_rels + [rg]
    V = vector(K, targets)
    
    if verbose:
        print(f"  Solving system size {M_sys.nrows()}x{M_sys.ncols()}...")
        
    try:
        # Solve for logarithms of factor base
        logs = M_sys.solve_right(V)
    except ValueError:
        print("  System is inconsistent or under-determined.")
        raise
        return None
        
    # 3. Compute log(Q)
    # log(Q) = Sum(row_q * x) - rq
    
    sum_logs = 0
    for idx, count in row_q.items():
        if idx < len(logs):
            sum_logs += count * logs[idx]
            
    log_q = sum_logs - rq
    return Integer(log_q)


def compute_jacobian_order(f_coeffs, p):
    """
    Computes approximate Jacobian order using Hasse-Weil bound for large primes.
    For p > 2^64, returns approximate value suitable for probabilistic algorithms.
    """
    P_x = PolynomialRing(K, 'x')
    f = P_x(list(reversed(list(f_coeffs))))
    
    # For genus 2: Hasse-Weil gives |#J - (p^2+1)| <= 4*sqrt(p^3)
    if p > 2**64:
        return Integer(p**2 + 1)
    
    C = HyperellipticCurve(f)
    return C.count_points(1)[0]


def extract_factor_base(divisors, p, verbose=True):
    """
    Extract the factor base with support-based deduplication.
    Only keep ONE divisor per unique support to avoid linear dependence.
    """
    all_roots = []
    factored_count = 0
    
    # Track which supports we've seen
    seen_supports = set()
    unique_divisors = []
    
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        
        # Check if u(x) = x² - sx + p splits over GF(p)
        if disc == 0:
            # Double root
            r = (s * pow(2, -1, p)) % p
            roots = [r, r]
            factored_count += 1
        elif pow(disc, (p-1)//2, p) == 1:
            # Two distinct roots
            sqrt_disc = tonelli_shanks(disc, p)
            r1 = (s + sqrt_disc) * pow(2, -1, p) % p
            r2 = (s - sqrt_disc) * pow(2, -1, p) % p
            roots = [r1, r2]
            factored_count += 1
        else:
            # Not smooth
            continue
        
        # Get support key
        support = tuple(sorted(roots))
        
        # Only add to factor base if this is the FIRST time we see this support
        if support not in seen_supports:
            seen_supports.add(support)
            all_roots.extend(roots)
            unique_divisors.append(d)
    
    unique_roots = set(all_roots)
    
    if verbose:
        print(f"\n[Factor Base - Support Deduplicated]")
        print(f"  Unique supports: {len(seen_supports)}")
        print(f"  Distinct x-coordinates: {len(unique_roots)}")
        print(f"  Kept after support dedup: {len(unique_divisors)}")
    
    return {
        'roots': unique_roots,
        'size': len(unique_roots),
        'unique_divisors': unique_divisors,
        'root_to_idx': {r: i for i, r in enumerate(sorted(list(unique_roots)))}
    }


# ============================================================================
# WORKER LOGGING + BATCH STATS (REPLACEMENT FUNCTIONS)
# ============================================================================


# ============================================================================
# WORKER GLOBALS & INITIALIZATION
# ============================================================================

def get_relation_row_cached(divisor):
    """
    Checks if a divisor is smooth over the factor base and returns its relation vector.
    """
    global _GLOBAL_ROOT_TO_IDX, _GLOBAL_P, _GLOBAL_FB_Y_CACHE
    u_poly, v_poly = divisor[0], divisor[1]

    if u_poly.degree() != 2:
        return None

    # u(x) = x^2 + a*x + b
    a, b = int(u_poly[1]), int(u_poly[0])
    disc = (a*a - 4*b) % _GLOBAL_P
    
    # Quick filter for splitting
    if pow(disc, (_GLOBAL_P-1)//2, _GLOBAL_P) != 1 and disc != 0:
        return None

    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != 2:
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


def perform_dlp_attack(G, Q, smooth_divs, p, f_coeffs, order, verbose=True):
    """
    Main entry point for DLP. Ensures coefficients are reversed for Sage.
    """
    factors = prime_factors(order)
    ell = max(factors)
    
    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order: {ell}")
    
    R = PolynomialRing(K, 'x')
    # Sage expects coefficients in low-to-high order
    f_p = R(f_coeffs[::-1])
    print("f_p =", f_p)
    
    fb_data = extract_factor_base(smooth_divs, p, verbose=False)
    r_to_idx = {r: i for i, r in enumerate(sorted(list(fb_data['roots'])))}
    
    valid_rows = []
    for d in smooth_divs:
        try:
            # Reconstruction matching: u = x^2 - sx + p
            u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
            v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
            row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p)
            if row: valid_rows.append(row)
        except Exception:
            raise
            continue

    offset_coeffs = [(int(d['s']), int(d['p']), int(d['v_0']), int(d['v_1'])) for d in smooth_divs[:50]]

    print(f"Anchoring G...")
    rg, row_g = find_smooth_decomposition(None, G, r_to_idx, f_p, p, order, offset_coeffs=offset_coeffs)
    if row_g is None: raise ValueError("Failed to anchor G")
    
    print(f"Anchoring Q...")
    rq, row_q = find_smooth_decomposition(Q, G, r_to_idx, f_p, p, order, offset_coeffs=offset_coeffs)
    if row_q is None: raise ValueError("Failed to anchor Q")
    
    d_log = solve_dlp_index_calculus(valid_rows, (rg, row_g), (rq, row_q), ell, verbose=verbose)
    if d_log and (d_log * (order // ell) * G) == ((order // ell) * Q):
        print(f"✓ Key verified: {d_log}")
        return d_log
    return None


from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod


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
            cand_div = cand_D.mumford()
        except Exception:
            raise
            continue
        
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
            
    return ("STATS", dict(agg_stats))


def find_smooth_decomposition_worker(seed_and_batch):
    seed, batch_candidates = seed_and_batch
    set_random_seed(seed)
    random.seed(int(seed))
    
    for r_val in range(batch_candidates):
        result = _worker_core_try_batch(r_val)
        if result[0] == "SUCCESS":
            return result[1]
    
    return None


def _worker_init(gen_mumford, target_mumford, root_to_idx, sample_roots_int, 
                 fb_y_cache, f_coeffs_plain, p_int, order_int, window_size, offset_coeffs):
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
    _GLOBAL_F_POLY = R(f_coeffs_plain)
    
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
                _GLOBAL_OFFSET_CACHE.append(J([u_poly, v_poly]))
            except Exception:
                raise
                continue


def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order,
                              max_tries=None, num_workers=None,
                              window_size=2048, sample_k=32, batch_candidates=512,
                              factor_base_freq=None, offset_coeffs=None):
    """
    Parallel search for smooth divisor. Ensures f_poly is correctly oriented.
    """
    from multiprocessing import Pool, cpu_count
    import random
    import time

    num_workers = cpu_count() if num_workers is None else num_workers
    p_int = int(p)
    order_int = int(order)
    R = PolynomialRing(K, 'x')

    fb_roots = sorted(list(root_to_idx.keys()))
    fb_y_cache = {}
    for x_val in fb_roots:
        y2 = int(f_poly(x_val))
        if y2 == 0:
            continue
        if pow(y2, (p_int - 1) // 2, p_int) == 1:
            from .smoothness import tonelli_shanks
            y_can = tonelli_shanks(y2, p_int)
            fb_y_cache[int(x_val)] = int(min(y_can, p_int - y_can))

    fb_root_list = list(root_to_idx.keys())
    sample_roots = [int(r) for r in random.sample(fb_root_list, min(sample_k, len(fb_root_list)))]

    # Extract f_poly coefficients as plain Python ints
    f_coeffs_plain = [int(c) for c in f_poly.list()]
    
    # Serialize generator and target_point as Mumford coordinates
    gen_mumford = None
    if generator is not None:
        # Use bracket notation [0] and [1] to get u and v polynomials
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
        fb_y_cache, f_coeffs_plain, p_int, order_int, window_size, offset_coeffs
    )

    if max_tries is None:
        max_tries = 1000000

    total_batches = (max_tries + batch_candidates - 1) // batch_candidates
    tasks = [(random.randint(0, 2**31 - 1), batch_candidates) for _ in range(total_batches)]
    
    with Pool(processes=num_workers, initializer=_worker_init, initargs=initargs) as pool:
        try:
            for result in pool.imap_unordered(find_smooth_decomposition_worker, tasks):
                if result is not None:
                    pool.terminate()
                    (r_val, row_vec, off_idx) = result
                    
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
                    return r_val, row_vec
        except KeyboardInterrupt:
            pool.terminate()
            raise
    return None, None
