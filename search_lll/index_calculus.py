import sys, time, random, multiprocessing
from math import ceil, sqrt, gcd
from multiprocessing import Pool, cpu_count, Process, SimpleQueue, Event
from collections import Counter
from copy import deepcopy
from queue import Full
from sage.all import Integer, Zmod, GF, ZZ, matrix, vector, PolynomialRing, factor, crt, prime_factors, set_random_seed
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from sage.matrix.berlekamp_massey import berlekamp_massey
from search_common import SECRET_KEY, BLOCK_WIEDEMANN, FINITE_FIELD, PREFERRED_X_COORDS
from .smoothness import tonelli_shanks, extract_factor_base
from .sparse_linalg_modp import *
from .cofactor import *
from .walker import *
from .fiber_augment import build_fiber_augmented_relations

# Standard library

# Sage imports (consolidated)

# Local imports

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
# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints

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

# ============================================================================
# CORRECTED: Remove anchoring entirely - preserve Mumford algebra
# ============================================================================

# ============================================================================
# CRITICAL FIX: Use Mumford v(x) directly - NO re-canonicalization with f(x)
# ============================================================================

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
        raise

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

# quick homomorphism test

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

# --- Replace get_relation_row_cached and homomorphism_test with the following ---

def combine_vectors(vec1, vec2, sign=1):
    """Pad shorter vector with zeros and add: vec1 + sign*vec2"""
    # Convert to lists of ints
    if not hasattr(vec1, '__len__'):
        vec1 = (vec1,)
    if not hasattr(vec2, '__len__'):
        vec2 = (vec2,)
    n1, n2 = len(vec1), len(vec2)
    n = max(n1, n2)
    v1 = list(vec1) + [0] * (n - n1)
    v2 = list(vec2) + [0] * (n - n2)
    if sign == 1:
        return tuple(int(a) + int(b) for a, b in zip(v1, v2))
    else:
        return tuple(int(a) - int(b) for a, b in zip(v1, v2))

# [Rest of the file remains the same until diagnose_bw_failure]

# [Keep all other functions unchanged - berlekamp_massey, block_wiedemann_solve, etc.]

def canonicalize_divisor_to_factor_base(divisor, atom_to_idx, f_p, p):
    """
    CORRECTED: Uses Mumford v(x) directly without re-canonicalizing.

    Re-express a divisor using its Mumford y-coordinates.
    Returns sparse row dict or None.

    NOW accepts atom_to_idx (tuple-keyed dict), not r_to_idx (int-keyed dict).
    """
    row = get_relation_row(divisor, atom_to_idx, f_p, p)
    return row

def _build_signed_row_from_divisor(div, atom_to_idx, f_p, p):
    """
    CORRECTED: Uses Mumford v(x) directly.

    Build signed row dict from Mumford divisor.
    Fallback when canonicalize_divisor_to_factor_base fails.

    NOW accepts atom_to_idx (tuple-keyed dict), not r_to_idx (int-keyed dict).
    """
    # Just call canonicalize_divisor_to_factor_base since they do the same thing now
    return canonicalize_divisor_to_factor_base(div, atom_to_idx, f_p, p)

def _legacy_build_relations_from_mumford(smooth_divs, G, Q, p, f_coeffs, verbose=True):
    """
    Convert legacy 'smooth_divs' into relations WITHOUT rebasing.
    CORRECTED: Uses proper Mumford homomorphism.

    CRITICAL FIX: Now returns atom_to_idx instead of r_to_idx.

    Returns: (valid_rows, rhs_values, fb_roots, atom_to_idx, fb_y_cache)
             NOT (valid_rows, rhs_values, fb_roots, r_to_idx, fb_y_cache)
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
    initialize_global_factor_base(atom_to_idx)

    # Extract fb_roots from degree-1 atoms (for backward compatibility with diagnostics)
    fb_roots = []
    for atom, idx in atom_to_idx.items():
        if atom[0] == 'd1':  # degree-1 atom: ('d1', x, y)
            x_val = atom[1]
            if x_val not in fb_roots:
                fb_roots.append(x_val)

    if len(atom_to_idx) == 0:
        raise RuntimeError("_legacy_build_relations_from_mumford: empty factor base extracted")

    if verbose:
        print(f"  [Legacy->Relations] Factor base size: {len(atom_to_idx)}")

    # Build homogeneous relations using corrected homomorphism
    # CRITICAL: Pass atom_to_idx directly (NOT r_to_idx)
    valid_rows, rhs_values = build_homogeneous_relations_no_rebase(
        smooth_divs, atom_to_idx, f_p, p, fb_y_cache, f_coeffs, verbose=verbose,
        use_collision_walks=False) # turn off collision walks while testing fiber relations
    

    if not valid_rows:
        raise RuntimeError("_legacy_build_relations_from_mumford: no valid homogeneous relations built")

    # CRITICAL: Return atom_to_idx, NOT r_to_idx
    # The old version built r_to_idx = {x_int: col_idx} here, which is WRONG
    return valid_rows, rhs_values, fb_roots, atom_to_idx, fb_y_cache

def verify_relation_is_ell_torsion(row, atom_to_idx, ell, J, verbose=False):
    """
    Verify that a relation row represents a divisor in ℓ-torsion.

    Reconstructs D = Σ row[i] * FB[i] and checks ℓ * D == 0.

    Args:
        row: dict {col_idx: multiplicity}
        atom_to_idx: dict {atom_tuple: col_idx}
        ell: prime modulus
        J: Jacobian
        verbose: print diagnostics

    Returns:
        True if ℓ * D == 0, False otherwise

    Raises on reconstruction failure
    """
    from sage.all import GF, PolynomialRing

    # Build inverse map
    idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}

    # Reconstruct divisor from row
    D = J.zero()

    for idx, mult in row.items():
        atom = idx_to_atom.get(int(idx))
        if atom is None:
            raise RuntimeError(f"Relation references unknown atom index {idx}")

        # Convert atom to Jacobian element
        if atom[0] == 'd1':
            # Degree-1: ('d1', x, y)
            _, x_val, y_val = atom
            K = J.base_ring()
            R = PolynomialRing(K, 'x')
            x_poly = R.gen()

            u_poly = x_poly - K(int(x_val))
            v_poly = K(int(y_val))

            atom_div = J([u_poly, v_poly])
        elif atom[0] == 'd2':
            # Degree-2: ('d2', u_coeffs_tuple, v_coeffs_tuple)
            _, u_coeffs, v_coeffs = atom
            K = J.base_ring()
            R = PolynomialRing(K, 'x')

            u_poly = R(list(u_coeffs))
            v_poly = R(list(v_coeffs))

            atom_div = J([u_poly, v_poly])
        else:
            raise RuntimeError(f"Unknown atom type: {atom[0]}")

        # Add mult * atom to D
        D += int(mult) * atom_div

    # Check ℓ * D == 0
    ellD = Integer(ell) * D
    is_torsion = ellD.is_zero()

    if verbose:
        print(f"  [Relation Check] ℓ * D is zero? {is_torsion}")
        if not is_torsion:
            print(f"                   Row: {row}")
            print(f"                   Reconstructed D: {D}")

    return is_torsion

def verify_all_relations_are_ell_torsion(projected_rows, atom_to_idx, ell, J,
                                         sample_size=100, verbose=True):
    """
    Sample-check that projected homogeneous relations are ℓ-torsion.

    Args:
        projected_rows: list of relation dicts (after h-projection)
        atom_to_idx: factor base atom map
        ell: prime
        J: Jacobian
        sample_size: how many to check (or None for all)
        verbose: print progress

    Raises if any relation fails the check
    """

    n_check = min(sample_size, len(projected_rows)) if sample_size else len(projected_rows)

    if verbose:
        print(f"  [Sanity] Checking {n_check}/{len(projected_rows)} relations are ℓ-torsion...")
        sys.stdout.flush()

    indices = random.sample(range(len(projected_rows)), n_check) if sample_size else range(len(projected_rows))

    failures = []
    for i in indices:
        row = projected_rows[i]
        try:
            is_torsion = verify_relation_is_ell_torsion(row, atom_to_idx, ell, J, verbose=False)
            if not is_torsion:
                failures.append((i, row))
        except Exception as e:
            raise RuntimeError(f"Relation {i} failed reconstruction: {e}")

    if failures:
        print(f"  [Sanity] ✗ FAILED: {len(failures)} relations are NOT ℓ-torsion!")
        for i, row in failures[:5]:  # Show first 5
            print(f"           Row {i}: {row}")
        raise RuntimeError(
            f"{len(failures)}/{n_check} relations failed ℓ-torsion check!\n"
            f"Homogeneous relations MUST be in J[ℓ] after h-projection."
        )

    if verbose:
        print(f"  [Sanity] ✓ All {n_check} sampled relations are ℓ-torsion")
        sys.stdout.flush()

def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order,
                              max_tries=None, num_workers=None,
                              window_size=2048, sample_k=32, batch_candidates=512,
                              offset_coeffs=None):
    """
    FIXED: Now properly passes f_coeffs to worker initialization.

    Parallel search for smooth divisor with restored serialization and live progress.
    Raises if no decomposition found.
    """

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

    # FIXED: Extract f_poly coefficients to pass to workers
    coeffs_genus2 = [int(c) for c in f_poly.list()]

    # --- Restore Serialization of Mumford Coordinates ---
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

    # FIXED: Add f_coeffs to initargs
    initargs = (
        gen_mumford, target_mumford, root_to_idx, sample_roots,
        fb_y_cache, p_int, order_int, window_size, offset_coeffs,
        coeffs_genus2  # <- ADDED
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

    # REMOVED: Parent-process _GLOBAL_F_POLY setting (does nothing for workers)

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

    raise RuntimeError("find_smooth_decomposition: exhausted search without finding a smooth decomposition")

def _worker_core_try_batch(r_val):
    """
    Updated: Returns (r_val_int, row_vec_plain, offset_idx, target_coeff)
    where target_coeff is 1 if _GLOBAL_TARGET_POINT was added, else 0.
    """
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT, _GLOBAL_ATOM_TO_IDX
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_OFFSET_CACHE

    P_int = _GLOBAL_P
    sample_roots = _GLOBAL_SAMPLE_ROOTS_INT
    agg_stats = Counter()
    agg_stats['tried'] = 1

    try:
        # Track if we are solving for [r]G or [r]G + Q
        if _GLOBAL_TARGET_POINT is None:
            D = r_val * _GLOBAL_GENERATOR
            target_coeff = 0
        else:
            D = r_val * _GLOBAL_GENERATOR + _GLOBAL_TARGET_POINT
            target_coeff = 1
    except Exception:
        raise
        return ("STATS", dict(agg_stats))

    for off_idx, offset_D in enumerate([None] + _GLOBAL_OFFSET_CACHE):
        try:
            cand_D = D + offset_D if offset_D else D
            cand_div = cand_D

            u = cand_div[0]
            if u.degree() != 2:
                continue

            u0, u1 = int(u[0]), int(u[1])
            hit = False
            for xr in sample_roots:
                if (xr * xr + u1 * xr + u0) % P_int == 0:
                    hit = True
                    break

            if not hit:
                agg_stats['sample_miss'] += 1
                continue

            # This helper must return the FB exponent vector
            row_vec = get_relation_row_cached(cand_div)
            if row_vec is not None:
                r_val_int = int(r_val)
                row_vec_plain = {int(k): int(v) for k, v in row_vec.items()}
                offset_idx = off_idx - 1
                # SUCCESS: Return exponents AND the RHS scalars (r, target_coeff)
                return ("SUCCESS", (r_val_int, row_vec_plain, offset_idx, target_coeff))
        except Exception:
            raise
            continue

    return ("STATS", dict(agg_stats))

def _worker_init(gen_mumford, target_mumford, atom_to_idx, sample_roots_int,
                 fb_y_cache, p_int, order_int, window_size, offset_coeffs, f_coeffs):
    """
    Worker initialization with proper error handling.
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

    K = GF(int(p_int))
    R = PolynomialRing(K, 'x')
    _GLOBAL_F_POLY = sage_poly_from_coeffs(f_coeffs, R)

    C = HyperellipticCurve(_GLOBAL_F_POLY)
    J = C.jacobian()

    if gen_mumford is not None:
        gen_u_coeffs, gen_v_coeffs = gen_mumford
        u_poly = R(gen_u_coeffs)
        v_poly = R(gen_v_coeffs)
        _GLOBAL_GENERATOR = J([u_poly, v_poly])
    else:
        _GLOBAL_GENERATOR = None

    if target_mumford is not None:
        target_u_coeffs, target_v_coeffs = target_mumford
        u_poly = R(target_u_coeffs)
        v_poly = R(target_v_coeffs)
        _GLOBAL_TARGET_POINT = J([u_poly, v_poly])
    else:
        _GLOBAL_TARGET_POINT = None

    zero = J.zero()
    _GLOBAL_BABY = [zero]
    curr = zero
    for _ in range(1, window_size):
        curr = curr + _GLOBAL_GENERATOR
        _GLOBAL_BABY.append(curr)

    _GLOBAL_OFFSET_CACHE = []
    if offset_coeffs:
        x = R.gen()
        failed_offsets = []
        for idx, (s, p_val, v0, v1) in enumerate(offset_coeffs):
            try:
                u_poly = x**2 - K(int(s))*x + K(int(p_val))
                v_poly = K(int(v1))*x + K(int(v0))
                _GLOBAL_OFFSET_CACHE.append(J([u_poly, v_poly]))
            except Exception as e:
                failed_offsets.append((idx, s, p_val, e))
                raise

        if failed_offsets and len(failed_offsets) < len(offset_coeffs):
            print(f"  [Worker] Warning: {len(failed_offsets)}/{len(offset_coeffs)} offset divisors failed")
        elif failed_offsets:
            raise RuntimeError(f"_worker_init: ALL offsets failed: {failed_offsets[0]}")

# ---------------------------------------------------------------------
# REPLACE project_relations_and_solve_mod_l (NO anchoring; accept row_q mapping)
# ---------------------------------------------------------------------
def project_relations_and_solve_mod_l(valid_rows, rhs_values, row_q_map, full_order, G, Q,
                                      verbose=True):
    from sage.all import Integer, Zmod, matrix, vector

    # get ell and h
    fac = full_order.factor() if hasattr(full_order, "factor") else None
    # best-effort: factor may not be available; fall back to integer arithmetic
    if fac:
        ell = int(max(int(pr) for pr, _ in fac))
    else:
        # naive largest prime factor (fallback)
        n = int(full_order)
        ell = 2
        p_temp = ell
        while p_temp * p_temp <= n:
            if n % p_temp == 0:
                ell = p_temp
                while n % p_temp == 0:
                    n //= p_temp
            p_temp += 1 if p_temp == 2 else 2
        if n > 1:
            ell = int(n)
    h = int(Integer(full_order) // Integer(ell))

    if verbose:
        print(f"  [Project] |J|={full_order}  ℓ={ell}  h={h}")

    Zell = Zmod(ell)

    num_rels = len(valid_rows)
    if len(rhs_values) != num_rels:
        raise RuntimeError("Mismatch: len(valid_rows) != len(rhs_values)")

    # determine num_vars (max index + 1)
    max_idx = -1
    for r in valid_rows:
        if r:
            max_idx = max(max_idx, max(r.keys()))
    if row_q_map:
        max_idx = max(max_idx, max(row_q_map.keys()))
    num_vars = max_idx + 1

    if verbose:
        print(f"  [Matrix] Building projected system: {num_rels} rows x {num_vars} cols")

    hom_rows = []
    hom_rhs = []
    inhom_rows = []
    inhom_rhs = []

    for r, b in zip(valid_rows, rhs_values):
        b_mod = int(b) % ell
        if b_mod == 0:
            hom_rows.append(r)
            hom_rhs.append(0)
        else:
            inhom_rows.append(r)
            inhom_rhs.append(b_mod)

    if len(inhom_rows) != 1:
        raise RuntimeError(f"Expected exactly one inhomogeneous G-row after projection; found {len(inhom_rows)}")

    row_g = inhom_rows[0]
    rhs_g_val = int(h % ell)  # normalization: solving logs w.r.t. G -> RHS = h

    entries_hom = {}
    for i, row in enumerate(hom_rows):
        for j, mult in row.items():
            val = Zell(int(mult) % ell)
            if val != 0:
                entries_hom[(i, j)] = val

    A_hom = matrix(Zell, len(hom_rows), num_vars, entries_hom, sparse=True)
    rank_A = A_hom.rank()

    if verbose:
        print(f"  [Project] A_hom: {A_hom.nrows()} x {A_hom.ncols()}, rank = {rank_A}")

    g_row_entries = {}
    for j, mult in row_g.items():
        v = Zell(int(mult) % ell)
        if int(v) != 0:
            g_row_entries[j] = v

    entries_aug = dict(entries_hom)
    for j, val in g_row_entries.items():
        entries_aug[(len(hom_rows), j)] = val

    M_aug = matrix(Zell, len(hom_rows) + 1, num_vars, entries_aug, sparse=True)
    b_aug = vector(Zell, [Zell(0)] * len(hom_rows) + [Zell(rhs_g_val)])

    rank_A_aug = M_aug.rank()
    if verbose:
        print(f"  [Project] rank(A_hom) = {rank_A}  rank(A_hom + G-row) = {rank_A_aug}")

    # invariants: homogeneous must be n_cols-1, augmented full rank
    if rank_A != max(0, num_vars - 1):
        raise RuntimeError(f"Homogeneous projected matrix has unexpected rank: {rank_A} (expected {max(0, num_vars - 1)})")
    if rank_A_aug != num_vars:
        try:
            _ = M_aug.solve_right(b_aug)
        except Exception as e:
            raise RuntimeError("Augmented projected system is inconsistent or underdetermined: " + str(e))
        raise RuntimeError("Augmented projected system did not achieve full rank after adding G-row.")

    sol = M_aug.solve_right(b_aug)  # vector length = num_vars

    # compute d modulo ell from Q-decomposition (no anchoring scalar)
    if not row_q_map:
        raise RuntimeError("Cannot compute discrete log: Q is not expressed over the factor base (row_q missing).")

    sum_logs = Zell(0)
    for idx, coeff in row_q_map.items():
        if idx >= len(sol):
            raise RuntimeError("row_q index out of range for solved variable vector")
        sum_logs += Zell(int(coeff) % ell) * sol[idx]

    d_mod_ell = int(sum_logs)  # integer in 0..ell-1

    # verify in ℓ-subgroup
    hG = Integer(h) * G
    hQ = Integer(h) * Q
    if Integer(d_mod_ell) * hG != hQ:
        raise RuntimeError("Projected solution failed verification in the ℓ-subgroup: computed d does not satisfy d*h*G == h*Q")

    if verbose:
        print(f"  [Solver] Candidate d (mod ℓ): {d_mod_ell}  ✓ verified")

    return Integer(d_mod_ell)

def filter_g_q_from_list(div_list, G, Q, p, f_coeffs):
    """
    Explicitly remove G and Q (and -G, -Q) from the list of divisors.
    Used to sanitize the factor base / relation pool.
    """
    if not div_list:
        return []

    # If we aren't in a mode with G/Q, nothing to filter
    if G is None and Q is None:
        return div_list

    print(f"  [Filter] Checking {len(div_list)} divisors against G and Q...")

    # We need the Jacobian to perform comparisons
    K = GF(int(p))
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    C = HyperellipticCurve(f_p)
    J = C.jacobian()

    # Comparison targets (pre-compute negation)
    targets = []
    if G is not None:
        targets.append(G)
        targets.append(-G)
    if Q is not None:
        targets.append(Q)
        targets.append(-Q)

    clean_list = []
    incinerated_count = 0

    # Helper to convert dict to J element
    def _to_J(d):
        u_poly = R.gen()**2 - K(int(d['s']))*R.gen() + K(int(d['p']))
        v_poly = K(int(d['v_1']))*R.gen() + K(int(d['v_0']))
        return J([u_poly, v_poly])

    for d in div_list:
        try:
            val_J = _to_J(d)
        except Exception:
            # Per aimist.txt, allow exceptions to propagate if logic/data is fundamentally broken
            raise

        is_forbidden = False
        for T in targets:
            if val_J == T:
                is_forbidden = True
                break

        if is_forbidden:
            incinerated_count += 1
        else:
            clean_list.append(d)

    if incinerated_count > 0:
        print(f"  [Filter] 🔥 INCINERATED {incinerated_count} divisors matching BASE/TARGET or inverses.")
    else:
        print(f"  [Filter] ✓ No forbidden divisors found (G/Q clean).")

    return clean_list

def are_sparse_vectors_dependent(v1, v2):
    """
    Check if v1 = k * v2 for some scalar k (integer/rational).
    v1, v2 are dicts {idx: val}.
    """
    if len(v1) != len(v2):
        return False
    if len(v1) == 0:
        return True # 0 and 0 are dependent

    keys1 = set(v1.keys())
    keys2 = set(v2.keys())
    if keys1 != keys2:
        return False

    # Pick a pivot to calculate ratio v1[i] / v2[i]
    pivot = next(iter(keys1))
    val1 = v1[pivot]
    val2 = v2[pivot]

    # Check cross-product v1[i]*val2 == v2[i]*val1 for all i
    # This avoids floating point issues and handles integer scaling
    for k in keys1:
        if v1[k] * val2 != v2[k] * val1:
            return False

    return True

def filter_forbidden_relations(rows, atom_to_idx, f_p, p, G, Q, verbose=True):
    """
    Filters out relations that are linearly dependent on G or Q vectors.
    Uses vector space dependence check (scalar multiples).
    """
    if not rows:
        return rows

    targets = []

    # Generate factor base vectors for G and Q
    # We use get_relation_row to get the exact FB representation (including signs)
    for name, Div in [("G", G), ("Q", Q)]:
        if Div is not None:
            try:
                r = get_relation_row(Div, atom_to_idx, f_p, p)
                if r:
                    targets.append(r)
            except Exception:
                # If G/Q fail to encode (e.g. not smooth), they can't be in the relations anyway
                raise # but we anyway raise, just to be safe.

    if not targets:
        return rows

    clean_rows = []
    incinerated = 0

    for row in rows:
        hit = False
        for t in targets:
            # Check for linear dependence (row = k * target)
            if are_sparse_vectors_dependent(row, t):
                hit = True
                break

        if hit:
            incinerated += 1
        else:
            clean_rows.append(row)

    if verbose:
        if incinerated > 0:
            print(f"  [Filter] 🔥 INCINERATED {incinerated} relations dependent on G or Q.")
        else:
            print(f"  [Filter] ✓ Relation vectors are clean (no linear dependence on G/Q).")

    return clean_rows


def perform_dlp_attack(G, Q, smooth_divs_or_rels, p, f_coeffs, order,
                       verbose=True, force_index_calculus=False,
                       E_rhs_m=None, x_b=None, f_shifted_poly=None):
    """
    CORRECTED: Traditional Index Calculus with proper kernel solver.

    CRITICAL: G and Q are assumed to already be in J(F_p)[ℓ] (ℓ-torsion).
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
        print(f"INDEX CALCULUS DLP ATTACK (Full Jacobian Group)")
        print(f"{'='*70}")
        print(f"Full Jacobian order |J|: {full_order}")

    # Compute ℓ and h
    factors = factor(full_order)
    ell = int(max(int(p) for p, _ in factors))
    h = int(full_order // ell)

    if verbose:
        print(f"Largest prime ℓ: {ell}")
        print(f"Cofactor h: {h}")

    # Build or extract factor base and homogeneous relations
    if precomputed:
        data = smooth_divs_or_rels[0]
        homogeneous_rows = data['relations']
        fb_roots = data['fb_roots']
        atom_to_idx = data['fb_map']

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

        homogeneous_rows, homogeneous_rhs, fb_roots, atom_to_idx, fb_y_cache = \
            _legacy_build_relations_from_mumford(smooth_divs_or_rels, G, Q, p, f_coeffs, verbose=verbose)

        # snapshot for precheck (Mumford atoms only, known to satisfy f_p)
        atom_to_idx_mumford = dict(atom_to_idx)
        homogeneous_rows_mumford = list(homogeneous_rows)

        # now augment
        if E_rhs_m is not None and x_b is not None and f_shifted_poly is not None:
            fiber_rows = build_fiber_augmented_relations(
                E_rhs_m, f_shifted_poly, x_b, p, atom_to_idx, fb_y_cache,
                full_order=full_order, ell=ell,
                verbose=verbose
            )
            homogeneous_rows.extend(fiber_rows)

        # Verify all RHS are zero (homogeneous)
        if any(r != 0 for r in homogeneous_rhs):
            raise RuntimeError(
                f"_legacy_build_relations_from_mumford returned non-homogeneous relations:\n"
                f"  Found {sum(1 for r in homogeneous_rhs if r != 0)} nonzero RHS values"
            )

        if verbose:
            print("  [Fiber Augment] +" + str(len(fiber_rows)) + " relations, FB now " + str(len(atom_to_idx_mumford)) + " atoms")

    if not homogeneous_rows:
        raise RuntimeError("No valid homogeneous relations available")

    # Filter forbidden relations (G/Q dependent)
    homogeneous_rows = filter_forbidden_relations(
        homogeneous_rows, atom_to_idx, f_p, p, G, Q, verbose=verbose
    )

    if not homogeneous_rows:
        raise RuntimeError("All relations were filtered out (they were all G or Q dependent!)")

    if verbose:
        print(f"  [Relations] Loaded {len(homogeneous_rows)} homogeneous relations")
        print(f"  [Factor Base] Size: {len(atom_to_idx)}")
        sys.stdout.flush()

    # === SMOOTH G AND Q ===
    # CRITICAL FIX: Use get_relation_row with require_signed_d2=False to allow d1 fallback
    row_g = None
    alpha_g = 0

    # Try encoding G directly (already ℓ-torsion)
    row_g = get_relation_row(G, atom_to_idx, f_p, p, require_signed_d2=False)

    if row_g is not None:
        if verbose:
            print("  [Smoothing] Generator G is already smooth.")
    else:
        # G not smooth - try random multiples
        if verbose:
            print("  [Smoothing] Generator not smooth. Attempting random smoothing...")

        for i in range(1, 2001):
            r = ZZ.random_element(1, int(ell))
            cand_G = (1 + r) * G

            row_g = get_relation_row(cand_G, atom_to_idx, f_p, p, require_signed_d2=False)

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

    row_q = get_relation_row(Q, atom_to_idx, f_p, p, require_signed_d2=False)

    if row_q is not None:
        if verbose:
            print("  [Smoothing] Target Q is already smooth.")
    else:
        if verbose:
            print("  [Smoothing] Target not smooth. Attempting random smoothing...")

        for i in range(1, 2001):
            r = ZZ.random_element(1, int(ell))
            cand_Q = Q + r * G

            row_q = get_relation_row(cand_Q, atom_to_idx, f_p, p, require_signed_d2=False)

            if row_q:
                beta_q = r
                if verbose:
                    print(f"  [Smoothing] Found smooth target at iter {i}")
                break

    if row_q is None:
        raise RuntimeError("Failed to smooth Target Q")

    g_support, q_support = check_gq_connectivity(homogeneous_rows, row_g, row_q, verbose=True)

    # Union-Find version — faster for large supports
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for row in homogeneous_rows:
        support = [idx for idx, val in row.items() if val != 0]
        for i in range(1, len(support)):
            union(support[0], support[i])

    g_roots = {find(idx) for idx in g_support if idx in parent or True}
    q_roots = {find(idx) for idx in q_support if idx in parent or True}
    connected = bool(g_roots & q_roots)
    print("graph is connected t/f:", connected)


    # === COFACTOR PROJECTION PRE-CHECK ===
    precheck = precheck_cofactor_projection(
        atom_to_idx=atom_to_idx,
        homogeneous_rows=homogeneous_rows,
        row_g=row_g,
        row_q=row_q,
        full_order=full_order,
        J=J,
        f_coeffs=f_coeffs,
        p=p,
        verbose=verbose
    )

    char_res = detect_nontrivial_character_from_projection(
        precheck['filtered_rows'],
        precheck['alive_fb_indices'],
        precheck['ell'],
        verbose=True)
    if char_res['found']:
        verify_character_vectors(
            precheck['filtered_rows'],
            char_res['alive_idx_list'],
            char_res['basis'],
            precheck['ell'],
            verbose=True
        )
    else:
        print("No nontrivial character found by linear algebra on projected homogeneous matrix.")

    if precheck['safe_to_project']:
        # Apply filtering
        filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q = apply_cofactor_filter(
            precheck, atom_to_idx, homogeneous_rows, row_g, row_q, verbose=verbose
        )

        if verbose:
            print(f"\n[Strategy] Using FILTERED ℓ-torsion solve")
            print(f"  Filtered FB: {len(filtered_atom_to_idx)} elements")
            print(f"  Filtered relations: {len(filtered_rows)}")
            sys.stdout.flush()

        try:
            dlog = solve_dlp_mod_l_cofactor_projection(
                filtered_rows,
                filtered_row_g,
                0,
                filtered_row_q,
                0,
                full_order,
                G, Q,
                filtered_atom_to_idx,
                J,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  ! Filtered ℓ-torsion solve failed: {e}")
                print(f"  Falling back to full Jacobian solve...")
            raise
    else:
        if verbose:
            print(f"\n[Strategy] Pre-check FAILED: {precheck['reason']}")
            print(f"  Skipping ℓ-torsion solve")
            print(f"  Using full Jacobian solve instead")
            sys.stdout.flush()

        raise RuntimeError(f"Cofactor projection unsafe: {precheck['reason']}")

    if verbose:
        print(f"  [Result] Discrete log (mod ℓ) = {dlog}")

    # === FINAL VERIFICATION ===
    D = Integer(dlog) * G - Q

    if not D.is_zero():
        raise RuntimeError(
            f"[Verify] ✗ Final verification FAILED:\n"
            f"        dlog * G ≠ Q\n"
            f"        dlog={dlog}, ℓ={ell}"
        )

    if verbose:
        print("  [Verify] ✓ Exact equality dlog * G == Q")

    return Integer(dlog)

def check_gq_connectivity(homogeneous_rows, row_g, row_q, verbose=True):
    """
    Check if G and Q are in the same connected component of the relations graph.
    Nodes = FB atom indices. Each relation row is a hyperedge over its support.
    BFS from support(row_g), check if support(row_q) is reachable.
    Returns True if connected, False if disconnected.
    """
    from collections import defaultdict, deque

    # Build adjacency: atom -> set of atoms reachable via any shared relation
    adj = defaultdict(set)
    for row in homogeneous_rows:
        support = [idx for idx, val in row.items() if val != 0]
        for i in range(len(support)):
            for j in range(i + 1, len(support)):
                adj[support[i]].add(support[j])
                adj[support[j]].add(support[i])

    g_support = set(idx for idx, val in row_g.items() if val != 0)
    q_support = set(idx for idx, val in row_q.items() if val != 0)

    if not g_support or not q_support:
        if verbose:
            print("[connectivity] G or Q has empty support, cannot check")
        return False

    visited = set()
    queue = deque(g_support)
    visited.update(g_support)

    while queue:
        node = queue.popleft()
        if node in q_support:
            if verbose:
                print("[connectivity] G and Q are CONNECTED in the relations graph")
            return True
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    if verbose:
        print("[connectivity] G and Q are DISCONNECTED")
        print("[connectivity] G support: " + str(sorted(g_support)))
        print("[connectivity] Q support: " + str(sorted(q_support)))
        print("[connectivity] Atoms reachable from G: " + str(len(visited)))
        q_reachable = q_support & visited
        print("[connectivity] Q support atoms reachable from G: " + str(q_reachable))

    return g_support, q_support
