from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors, set_random_seed
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from .smoothness import extract_factor_base, tonelli_shanks
from search_common import FINITE_FIELD, SECRET_KEY
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


from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod


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


def solve_dlp_index_calculus(valid_rows, g_anchored, q_anchored, ell, verbose=True):
    """
    Solves the sparse linear system modulo ell to find log_G(Q).

    - valid_rows: list of dicts {idx: count} from collected relations
    - g_anchored: (rg, row_g) where row_g is dict {idx: count} and rg is integer (anchor scalar)
    - q_anchored: (rq, row_q) similarly for Q anchor
    - ell: prime modulus for logs (integer)

    This function constructs the linear system over Z/ellZ and solves it.
    It raises on inconsistency or if the solve fails.
    """
    if ell is None or int(ell) <= 1:
        raise ValueError("Invalid ell provided to solve_dlp_index_calculus")

    # Use ring Z/ellZ
    R = Zmod(int(ell))

    rg, row_g = g_anchored
    rq, row_q = q_anchored

    if row_g is None or row_q is None:
        raise ValueError("Anchor rows cannot be None")

    # Determine number of relations and variables
    num_rels = len(valid_rows)
    max_idx = -1
    for r in valid_rows:
        if r:
            local_max = max(r.keys())
            if local_max > max_idx:
                max_idx = local_max
    if row_g:
        local_max = max(row_g.keys())
        if local_max > max_idx:
            max_idx = local_max
    if row_q:
        local_max = max(row_q.keys())
        if local_max > max_idx:
            max_idx = local_max
    if max_idx < 0:
        raise ValueError("No variables found in relations/anchors")

    num_vars = max_idx + 1

    if verbose:
        print(f"  Building sparse matrix {num_rels}x{num_vars} over Z/{ell}Z...")

    # Build triples for sparse construction, reduce counts modulo ell
    triples = []
    for i, rel in enumerate(valid_rows):
        for idx, count in rel.items():
            if idx < 0 or idx >= num_vars:
                raise IndexError(f"Relation index out of bounds: {idx}")
            triples.append((i, idx, R(int(count) % int(ell))))

    # Add G anchor row at index num_rels
    for idx, count in row_g.items():
        if idx < 0 or idx >= num_vars:
            raise IndexError(f"G-anchor index out of bounds: {idx}")
        triples.append((num_rels, idx, R(int(count) % int(ell))))

    # Construct matrix over Z/ellZ
    M_sys = matrix(R, num_rels + 1, num_vars, triples)

    # Construct RHS vector: zeros for relation rows, rg for the anchor row
    targets = [R(0)] * num_rels + [R(int(rg) % int(ell))]
    V = vector(R, targets)

    if verbose:
        print(f"  Solving system size {M_sys.nrows()}x{M_sys.ncols()} over Z/{ell}Z...")

    # Solve
    try:
        logs_vec = M_sys.solve_right(V)
    except ValueError as e:
        # propagate with context
        raise ValueError(f"Linear solve failed (possibly inconsistent/under-determined) over Z/{ell}Z: {e}")

    # logs_vec is a vector over R of length num_vars (or raises earlier)
    # Compute log(Q) = <row_q, logs> - rq  (mod ell)
    sum_logs = R(0)
    for idx, count in row_q.items():
        if idx < 0 or idx >= num_vars:
            raise IndexError(f"Q-anchor index out of bounds: {idx}")
        # If logs_vec shorter (shouldn't be), treat missing as 0
        if idx < len(logs_vec):
            sum_logs += R(int(count) % int(ell)) * logs_vec[idx]
        else:
            # Shouldn't happen, but explicitly fail loud
            raise IndexError(f"Log vector shorter than expected: idx {idx}")

    log_q = R(sum_logs - R(int(rq) % int(ell)))
    # Normalize to Python int and return Sage Integer
    return Integer(int(log_q))


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
    f_coeffs_plain = [int(c) for c in f_poly.list()]

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
        fb_y_cache, f_coeffs_plain, p_int, order_int, window_size, offset_coeffs
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


def perform_dlp_attack(G, Q, smooth_divs, p, f_coeffs, order, verbose=True):
    """
    Main entry point for DLP.

    Steps:
      1. Factor |J| and choose largest prime ell
      2. Project G,Q into ell-torsion
      3. If ell is small (< 10^6), use BSGS directly
      4. Otherwise, build factor base + relations and use index calculus
      5. Verify d*G_ell = Q_ell

    Raises on any failure.
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")

    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    # ----------------------------
    # Pick ell
    # ----------------------------
    factors = prime_factors(order)
    if not factors:
        raise ValueError("Failed to factor Jacobian order")

    ell = max(factors)
    if ell <= 1:
        raise ValueError("No non-trivial prime factor found")

    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order ell = {ell}")

    # ----------------------------
    # Project to ell-torsion
    # ----------------------------
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

    # ----------------------------
    # CRITICAL DECISION POINT: BSGS vs Index Calculus
    # ----------------------------
    BSGS_THRESHOLD = 10**6  # Use BSGS for subgroups smaller than 1 million

    if ell < BSGS_THRESHOLD:
        if verbose:
            print(f"\n[Strategy] Subgroup size {ell} < {BSGS_THRESHOLD}")
            print(f"[Strategy] Using BSGS (expected ~{int(2*ell**0.5)} group ops)")
        
        d_log = dlp_bsgs(G_ell, Q_ell, ell)
        
        # Verify
        if Integer(d_log) * G_ell != Q_ell:
            raise RuntimeError("BSGS discrete log failed verification")
        
        if verbose:
            print(f"✓ Discrete log found via BSGS: {d_log}")
        
        return Integer(d_log)
    
    # ----------------------------
    # For large ell: Index Calculus path
    # ----------------------------
    if verbose:
        print(f"\n[Strategy] Subgroup size {ell} >= {BSGS_THRESHOLD}")
        print(f"[Strategy] Using Index Calculus")

    # ----------------------------
    # Polynomial
    # ----------------------------
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = R(f_coeffs[::-1])

    if verbose:
        print("f_p =", f_p)

    # ----------------------------
    # Factor base
    # ----------------------------
    fb_data = extract_factor_base(smooth_divs, p, verbose=False)
    roots = sorted(list(fb_data['roots']))
    if not roots:
        raise RuntimeError("Empty factor base")

    r_to_idx = {r: i for i, r in enumerate(roots)}

    # ----------------------------
    # Relation matrix
    # ----------------------------
    valid_rows = []
    for d in smooth_divs:
        u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
        v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
        row = get_relation_row([u_poly, v_poly], r_to_idx, f_p, p)
        if row:
            valid_rows.append({int(k): int(v) for k, v in row.items()})

    if not valid_rows:
        raise RuntimeError("No valid relations from smooth_divs")

    if verbose:
        print(f"Loaded {len(valid_rows)} factor base relations")

    # ----------------------------
    # Offsets for anchoring
    # ----------------------------
    offset_coeffs = [
        (int(d['s']), int(d['p']), int(d['v_0']), int(d['v_1']))
        for d in smooth_divs[:50]
    ]

    # ----------------------------
    # Anchor G_ell
    # ----------------------------
    if verbose:
        print("Anchoring G_ell...")
    rg, row_g = find_smooth_decomposition(
        None, G_ell, r_to_idx, f_p, p, ell, offset_coeffs=offset_coeffs
    )
    if row_g is None:
        raise RuntimeError("Failed to anchor G_ell")

    # ----------------------------
    # Anchor Q_ell
    # ----------------------------
    if verbose:
        print("Anchoring Q_ell...")
    rq, row_q = find_smooth_decomposition(
        Q_ell, G_ell, r_to_idx, f_p, p, ell, offset_coeffs=offset_coeffs
    )
    if row_q is None:
        raise RuntimeError("Failed to anchor Q_ell")

    # ----------------------------
    # Solve DLP
    # ----------------------------
    d_log = solve_dlp_index_calculus(
        valid_rows,
        (int(rg) % ell, {int(k): int(v) for k, v in row_g.items()}),
        (int(rq) % ell, {int(k): int(v) for k, v in row_q.items()}),
        ell,
        verbose=verbose
    )

    d_log = int(d_log) % int(ell)

    # ----------------------------
    # Verify in ell-torsion
    # ----------------------------
    if Integer(d_log) * G_ell != Q_ell:
        raise RuntimeError("Discrete log failed verification in ell-subgroup")

    if verbose:
        print(f"✓ Discrete log found via Index Calculus: {d_log}")

    return Integer(d_log)
