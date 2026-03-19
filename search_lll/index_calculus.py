import sys, time, random, multiprocessing, math
from math import ceil, sqrt, gcd
from multiprocessing import Pool, cpu_count, Process, SimpleQueue, Event
from collections import Counter, deque, defaultdict
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
from .fiber_augment import *
from .recursive_smoothing import *
from typing import Dict, List, Tuple, Any, Optional

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
    row = get_relation_row(div, atom_to_idx, f_p, p, fb_y_cache=fb_y_cache, require_signed_d2=False)
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
    row = get_relation_row(divisor, atom_to_idx, f_p, p, require_signed_d2=False)
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
                        off_row = get_relation_row([u_off, v_off], root_to_idx, f_poly, p_int, require_signed_d2=False)
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
            row_vec = get_relation_row_cached(cand_div, require_signed_d2=False)
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
        # If it's a relation object, extract the resulting divisor d3
        if 'type' in d and d['type'] == 'relation':
            d = d['d3']
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
                r = get_relation_row(Div, atom_to_idx, f_p, p, require_signed_d2=False)
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

# NOTE: This function assumes the following are available in the module namespace:
# - Integer, GF, PolynomialRing, HyperellipticCurve (sage imports you already had)
# - _legacy_build_relations_from_mumford, filter_forbidden_relations,
#   get_relation_row, build_fiber_augmented_relations, precheck_cofactor_projection,
#   detect_nontrivial_character_from_projection, verify_character_vectors,
#   apply_cofactor_filter, solve_dlp_mod_l_cofactor_projection
# If any of those are not present, keep the placeholders and supply them.

def _convert_divisor_relations_to_atom_relations(rs):
    """
    Convert a RecursiveSmoother instance 'rs' into homogeneous_rows, fb_roots, atom_to_idx
    with atom keys matching your pipeline: ('d1', x_int, y_can) or any other hashable atom.
    We assume rs.roots are atom keys already (so we use them directly).
    """
    atom_to_idx = {}
    next_idx = 0

    def get_atom_idx(atom):
        nonlocal next_idx
        if atom not in atom_to_idx:
            atom_to_idx[atom] = next_idx
            next_idx += 1
        return atom_to_idx[atom]

    homogeneous_rows = []

    # rs.relations is list of {div_idx: coeff}
    # rs.divisors is list of divisor pairs (root1, root2)
    for rel in rs.relations:
        atom_row = {}
        for div_idx, coeff in rel.items():
            # guard: skip zero coefficients (shouldn't appear)
            if coeff == 0:
                continue
            a, b = rs.divisors[div_idx]   # these are the atoms used as roots in rs
            ia = get_atom_idx(a)
            ib = get_atom_idx(b)
            atom_row[ia] = atom_row.get(ia, 0) + coeff
            atom_row[ib] = atom_row.get(ib, 0) + coeff

        # prune zeros
        atom_row = {k: v for k, v in atom_row.items() if v != 0}
        if atom_row:
            homogeneous_rows.append(atom_row)

    # invert map to get fb_roots (list indexable by atom index)
    fb_roots = [None] * len(atom_to_idx)
    for atom, idx in atom_to_idx.items():
        fb_roots[idx] = atom

    return homogeneous_rows, fb_roots, atom_to_idx

# Top-level helpers (no nested defs)

def jacobian_to_atoms(J_elem, p, f_p):
    """
    Convert a Jacobian element to a list of atom tuples compatible with atom_to_idx format.
    Uses jacobian_to_dict (assumed present in module scope).
    Returns degree-1 atoms ('d1', x_int, y_can) when possible; otherwise returns one d2 atom.
    """
    d = jacobian_to_dict(J_elem, p)
    atoms: List[Tuple] = []

    # Prefer explicit roots if present
    roots = d.get('roots', None)
    if roots:
        for x_val in roots:
            x_int = int(x_val)
            y2 = int(f_p(GF(p)(x_int)))  # f_p polynomial evaluated at x_int in GF(p)
            if y2 == 0:
                y_can = 0
            else:
                # safe-guard: ensure QR before calling canonical
                if pow(y2, (p - 1) // 2, p) != 1:
                    continue
                y_can = int(canonical_y(y2, p))
            atoms.append(('d1', x_int, int(y_can)))
        if atoms:
            return atoms

    # Fallback: return a single degree-2 style atom using Mumford invariants.
    s = int(d.get('s', 0))
    pval = int(d.get('p', 0))
    v1 = int(d.get('v_1', 0))
    v0 = int(d.get('v_0', 0))
    atoms.append(('d2', s, pval, v1, v0))
    return atoms

def nullspace_mod_p(rows: List[Dict[int, int]], ncols: int, p_mod: int) -> List[List[int]]:
    """
    Compute a nullspace basis (mod p_mod) for the given row-sparse list.
    Returns list of basis vectors, each length ncols. Uses dense Gaussian elimination.
    Suitable for moderate sizes (few thousands).
    """
    if not rows:
        return []
    m = len(rows)
    A = [[0] * ncols for _ in range(m)]
    for i, r in enumerate(rows):
        for c, v in r.items():
            if 0 <= c < ncols:
                A[i][c] = v % p_mod

    # Row reduction to RREF
    row = 0
    pivots = []
    for col in range(ncols):
        if row >= m:
            break
        sel = None
        for r in range(row, m):
            if A[r][col] % p_mod != 0:
                sel = r
                break
        if sel is None:
            continue
        A[row], A[sel] = A[sel], A[row]
        inv = pow(A[row][col], p_mod - 2, p_mod)
        A[row] = [(x * inv) % p_mod for x in A[row]]
        for r in range(m):
            if r != row and A[r][col] != 0:
                factor = A[r][col]
                A[r] = [ (A[r][c] - factor * A[row][c]) % p_mod for c in range(ncols) ]
        pivots.append(col)
        row += 1

    pivot_set = set(pivots)
    free_vars = [j for j in range(ncols) if j not in pivot_set]
    basis: List[List[int]] = []
    for fv in free_vars:
        vec = [0] * ncols
        vec[fv] = 1
        for r_idx, pc in enumerate(pivots):
            vec[pc] = (-A[r_idx][fv]) % p_mod
        basis.append(vec)
    return basis

# ---------------------------------------------------------------------
# perform_dlp_attack (refactored): uses recursive smoothing unconditionally
# ---------------------------------------------------------------------

def _filter_d2_atoms(atom_to_idx: Dict) -> Dict:
    """
    Helper to extract only d2 atoms from the master factor base.
    """
    return {atom: i for i, atom in enumerate(
        [a for a in atom_to_idx.keys() if isinstance(a, tuple) and a[0] == 'd2']
    )}

def split_atom_maps(master_atom_to_idx):
    """
    Split the master factor base into d1 and d2 maps.

    Returns
    -------
    d1_atom_to_idx : Dict
    d2_atom_to_idx : Dict
    """

    d1_atom_to_idx = {}
    d2_atom_to_idx = {}

    for atom in master_atom_to_idx.keys():

        if not isinstance(atom, tuple):
            raise RuntimeError(f"Unexpected atom type in factor base: {atom}")

        tag = atom[0]

        if tag == 'd1':
            d1_atom_to_idx[atom] = len(d1_atom_to_idx)

        elif tag == 'd2':
            d2_atom_to_idx[atom] = len(d2_atom_to_idx)

        else:
            raise RuntimeError(f"Unknown atom tag: {tag}")

    return d1_atom_to_idx, d2_atom_to_idx

def _legacy_build_relations_from_mumford(smooth_divs, G, Q, p, f_coeffs, verbose=True):
    """
    Build legacy relations for index calculus on genus 2 curves.

    Uses d1-only atom map as the unified column space so that Mumford
    relations, fiber augment relations, and G/Q rows all share the same
    indices.  All Mumford divisors split over GF(p), so d2 atoms decompose
    cleanly into d1 pairs via get_relation_row's fallback path.

    Returns:
        valid_rows:      homogeneous relations indexed by d1_atom_to_idx
        rhs_values:      RHS values (all 0)
        fb_roots:        list of d1 atom x-values for diagnostics
        d1_atom_to_idx:  d1-only atom map (shared column space)
        fb_y_cache:      y-values cache for factor base
        d2_to_d1_map:    each d2 atom -> tuple of its two d1 atoms
    """
    if smooth_divs is None or not isinstance(smooth_divs, (list, tuple)):
        raise RuntimeError("_legacy_build_relations_from_mumford: expected list of divisors")

    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)

    extended_divs = [jacobian_to_dict(G, p), jacobian_to_dict(Q, p)] + list(smooth_divs)

    master_atom_to_idx, fb_y_cache = extract_factor_base(extended_divs, p, f_p, verbose=verbose)
    initialize_global_factor_base(master_atom_to_idx)

    d1_atoms = [atom for atom in master_atom_to_idx.keys() if isinstance(atom, tuple) and atom[0] == 'd1']
    d1_atom_to_idx = {atom: i for i, atom in enumerate(d1_atoms)}

    # Diagnostic: count how many pool divisors have both roots in d1_atom_to_idx
    d1_xs = {int(a[1]) for a in d1_atom_to_idx}
    pool_divs = [d for d in smooth_divs if isinstance(d, dict) and d.get('type') != 'relation']
    n_both_in = sum(1 for d in pool_divs if all(r in d1_xs for r in d.get('roots', [])))
    print('[diag_legacy] d1 atom x-set size=%d, pool divs with both roots in d1=%d / %d' % (len(d1_xs), n_both_in, len(pool_divs)))

    d2_atoms = [atom for atom in master_atom_to_idx.keys() if isinstance(atom, tuple) and atom[0] == 'd2']
    d2_to_d1_map = {}
    for atom in d2_atoms:
        d2_to_d1_map[atom] = (('d1', atom[1]), ('d1', atom[2]))

    fb_roots = [atom[1] for atom in d1_atoms]

    if verbose:
        print(f"  [Legacy] Master FB: {len(master_atom_to_idx)} atoms")
        print(f"  [Legacy] d1 atoms (column space): {len(d1_atom_to_idx)}, d2 atoms: {len(d2_atoms)}")
        print(f"  [Legacy] d2 -> d1 mapping size: {len(d2_to_d1_map)}")
    use_collision_walks=False
    valid_rows, rhs_values = build_homogeneous_relations_no_rebase(
        smooth_divs,
        d1_atom_to_idx,
        f_p,
        p,
        fb_y_cache,
        f_coeffs,
        verbose=verbose,
        use_collision_walks=use_collision_walks
    )

    if not valid_rows and use_collision_walks:
        raise RuntimeError("_legacy_build_relations_from_mumford: no valid relations built")

    return valid_rows, rhs_values, fb_roots, d1_atom_to_idx, fb_y_cache, d2_to_d1_map

def check_gq_connectivity(homogeneous_rows, row_g, row_q, verbose=True):
    adj = defaultdict(set)
    for row in homogeneous_rows:
        support = [idx for idx, val in row.items() if val != 0]
        for i in range(len(support)):
            for j in range(i + 1, len(support)):
                adj[support[i]].add(support[j])
                adj[support[j]].add(support[i])

    g_support = set(idx for idx, val in row_g.items() if val != 0)
    q_support = set(idx for idx, val in row_q.items() if val != 0)

    visited = set(g_support)
    queue = deque(g_support)

    while queue:
        node = queue.popleft()
        if node in q_support:
            if verbose:
                print("[connectivity] G and Q are CONNECTED in the relations graph")
            return True, g_support, q_support

        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    if verbose:
        print("[connectivity] G and Q are DISCONNECTED")
        print("[connectivity] Atoms reachable from G:", len(visited))
        print("[connectivity] Q reachable:", q_support & visited)

    return False, g_support, q_support

def _dict_to_jacobian(d, J, R, p):
    K = GF(p)
    x = R.gen()

    # Priority 1: Use explicit coefficients if they exist
    if 'u_coeffs' in d and 'v_coeffs' in d:
        u_poly = R(d['u_coeffs'])
        v_poly = R(d['v_coeffs'])
    # Priority 2: Fallback to Mumford parameters with defaults to prevent KeyError
    else:
        s = int(d.get('s', 0))
        pp = int(d.get('p', 0))
        v0 = int(d.get('v_0', 0))
        v1 = int(d.get('v_1', 0))
        u_poly = x**2 - K(s) * x + K(pp)
        v_poly = K(v1) * x + K(v0)

    return J([u_poly, v_poly])

def _is_precomputed_relations_pack(smooth_divs_or_rels):
    """
    Detect the packed precomputed-relations format:
      [ { 'type': 'relations', 'relations': ..., 'fb_roots': ..., 'fb_map': ... } ]
    """
    return (
        isinstance(smooth_divs_or_rels, (list, tuple))
        and len(smooth_divs_or_rels) == 1
        and isinstance(smooth_divs_or_rels[0], dict)
        and smooth_divs_or_rels[0].get("type") == "relations"
    )

def _compute_full_order_data(full_order, verbose=True):
    """
    Compute the largest prime factor ell and cofactor h from |J|.
    """
    factors = factor(Integer(full_order))
    ell = int(max(int(pp) for pp, _ in factors))
    h = int(Integer(full_order) // Integer(ell))
    if verbose:
        print(f"Largest prime ℓ: {ell}")
        print(f"Cofactor h: {h}")
    return ell, h

def _prepare_curve_and_jacobian(p, f_coeffs):
    """
    Build GF(p), the defining polynomial f_p, the hyperelliptic curve C,
    and its Jacobian J.
    """
    K = GF(p)
    R = PolynomialRing(K, "x")
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    C = HyperellipticCurve(f_p)
    J = C.jacobian()
    return K, R, f_p, C, J

def _build_fb_y_cache(atom_to_idx, f_p, p, K):
    """
    Reconstruct y-cache for factor-base x-coordinates.
    """
    fb_y_cache = {}
    for atom in atom_to_idx:
        if not (isinstance(atom, tuple) and len(atom) >= 3):
            continue
        if atom[0] != "d1":
            continue

        x_int = int(atom[1])
        y_int = int(atom[2])

        # Keep the explicit FB y value if already present.
        fb_y_cache[x_int] = y_int

    # Defensive fill: if the map contains x but not a y entry for some reason,
    # evaluate it from f_p when possible.
    for x_int in list(fb_y_cache.keys()):
        if fb_y_cache[x_int] is not None:
            continue
        try:
            y2 = int(f_p(K(x_int)))
            if y2 == 0:
                fb_y_cache[x_int] = 0
            else:
                y = tonelli_shanks(y2, p)
                fb_y_cache[x_int] = int(min(y, p - y))
        except Exception:
            fb_y_cache[x_int] = None

    return fb_y_cache

def _load_or_build_relations(
    smooth_divs_or_rels,
    G,
    Q,
    p,
    f_coeffs,
    f_p,
    verbose=True,
):
    """
    Returns:
      homogeneous_rows, homogeneous_rhs, fb_roots, atom_to_idx, fb_y_cache, d2_to_d1_map
    """
    if _is_precomputed_relations_pack(smooth_divs_or_rels):
        data = smooth_divs_or_rels[0]
        homogeneous_rows = list(data.get("relations", []))
        homogeneous_rhs = list(data.get("rhs", [])) if "rhs" in data else [0] * len(homogeneous_rows)
        fb_roots = list(data.get("fb_roots", []))
        atom_to_idx = dict(data.get("fb_map", {}))
        fb_y_cache = _build_fb_y_cache(atom_to_idx, f_p, p, GF(p))
        d2_to_d1_map = dict(data.get("d2_to_d1_map", {}))
        return homogeneous_rows, homogeneous_rhs, fb_roots, atom_to_idx, fb_y_cache, d2_to_d1_map

    if verbose:
        print("  [Legacy] Building factor base and relations from Mumford divisors...")

    (
        homogeneous_rows,
        homogeneous_rhs,
        fb_roots,
        atom_to_idx,
        fb_y_cache,
        d2_to_d1_map,
    ) = _legacy_build_relations_from_mumford(
        smooth_divs_or_rels, G, Q, p, f_coeffs, verbose=verbose
    )

    return homogeneous_rows, homogeneous_rhs, fb_roots, atom_to_idx, fb_y_cache, d2_to_d1_map

def _maybe_log_fb_samples(atom_to_idx, f_p, p, verbose=True):
    """
    Print a small diagnostic sample of d1 atoms and verify they match the curve.
    """
    if not verbose:
        return

    d1_sample = [(a[1], a[2]) for a in atom_to_idx if isinstance(a, tuple) and len(a) >= 3 and a[0] == "d1"][:5]
    print("[diag] sample d1 atoms (x, y):", d1_sample)

    K = GF(p)
    for x_int, y_int in d1_sample:
        y2 = int(f_p(K(x_int)))
        if y2 == 0:
            y_check = 0
        else:
            y1 = int(tonelli_shanks(y2, p))
            y_check = min(y1, p - y1)
        print("[diag]   x=%d y_fb=%d y_from_f=%d match=%s" % (x_int, y_int, y_check, y_int == y_check))

def _augment_relations_from_fibers(
    E_rhs_m,
    f_shifted_poly,
    x_b,
    p,
    atom_to_idx,
    fb_y_cache,
    full_order,
    ell,
    x_coords=None,
    num_workers=None,
    verbose=True,
    promote_atom=None,
    lp_state=None,
):
    """
    Run the fiber augmentation pipeline and return:
      fiber_rows, fiber_stats, lp_state
    """
    fiber_rows, fiber_stats, lp_state = build_fiber_augmented_relations(
        E_rhs_m=E_rhs_m,
        f_shifted_fp=f_shifted_poly,
        x_b=x_b,
        p=p,
        atom_to_idx=atom_to_idx,
        fb_y_cache=fb_y_cache,
        full_order=full_order,
        ell=ell,
        x_coords=x_coords,
        num_workers=num_workers,
        verbose=verbose,
        promote_atom=promote_atom,
        lp_state=lp_state,
    )

    if fiber_rows and not all(isinstance(r, dict) for r in fiber_rows):
        raise TypeError("fiber augmentation returned non-dict row(s)")

    return fiber_rows, fiber_stats, lp_state

def _append_lp_promotions(
    homogeneous_rows,
    atom_to_idx,
    fb_y_cache,
    fiber_stats,
    ell,
    verbose=True,
):
    """
    Promote frequently seen large primes into the factor base and append rows
    that contain those primes.
    """
    promoted_count = 0

    lp_counter = fiber_stats.get("large_prime_counter", {})
    lp_table = fiber_stats.get("large_prime_table_single", {})

    if not lp_table:
        return homogeneous_rows, atom_to_idx, fb_y_cache, promoted_count

    max_idx = max(atom_to_idx.values()) if atom_to_idx else -1

    for lp_key, stored_list in lp_table.items():
        if not stored_list:
            continue

        x_lp, y_lp = lp_key
        lp_atom = ("d1", int(x_lp), int(y_lp))

        if lp_atom not in atom_to_idx:
            max_idx += 1
            atom_to_idx[lp_atom] = max_idx
            fb_y_cache[int(x_lp)] = int(y_lp)
            if verbose:
                print(
                    "[LP promote] atom (%d, %d) -> idx %d, freq=%d"
                    % (x_lp, y_lp, max_idx, lp_counter.get(lp_key, 0))
                )

        lp_idx = atom_to_idx[lp_atom]

        for row_fb, lp_mult in stored_list:
            full_row = dict(row_fb)
            val = (full_row.get(lp_idx, 0) + int(lp_mult)) % int(ell)
            if val:
                full_row[lp_idx] = val
            else:
                full_row.pop(lp_idx, None)

            if full_row:
                homogeneous_rows.append(full_row)
                promoted_count += 1

    if verbose:
        print("[LP promote] %d promoted relations added" % promoted_count)

    return homogeneous_rows, atom_to_idx, fb_y_cache, promoted_count

def _build_gq_rows(G, Q, atom_to_idx, f_p, p, verbose=True):
    """
    Build the factor-base rows for G and Q.
    """
    R_fp = PolynomialRing(GF(p), "x")

    G_dict = jacobian_to_dict(G, p)
    Q_dict = jacobian_to_dict(Q, p)

    row_g = get_relation_row(
        [R_fp(G_dict["u_coeffs"]), R_fp(G_dict["v_coeffs"])],
        atom_to_idx, f_p, p,
        require_signed_d2=False,
    )
    if not row_g:
        raise RuntimeError("G not representable in factor base")

    row_q = get_relation_row(
        [R_fp(Q_dict["u_coeffs"]), R_fp(Q_dict["v_coeffs"])],
        atom_to_idx, f_p, p,
        require_signed_d2=False,
    )
    if not row_q:
        raise RuntimeError("Q not representable in factor base")

    if verbose:
        print(f"  [G row] {len(row_g)} atoms")
        print(f"  [Q row] {len(row_q)} atoms")

    return row_g, row_q

def _union_find_connectivity(homogeneous_rows, g_support, q_support):
    """
    Diagnostic connectivity check over the support graph induced by rows.
    """
    parent = {}

    def find_union(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find_union(x)] = find_union(y)

    for row in homogeneous_rows:
        support = [idx for idx, val in row.items() if val != 0]
        if len(support) >= 2:
            a0 = support[0]
            for i in range(1, len(support)):
                union(a0, support[i])

    g_roots = {find_union(idx) for idx in g_support if idx in parent or True}
    q_roots = {find_union(idx) for idx in q_support if idx in parent or True}
    connected = bool(g_roots & q_roots)
    return connected

def perform_dlp_attack(
    G,
    Q,
    smooth_divs_or_rels,
    p,
    f_coeffs,
    order,
    verbose: bool = True,
    force_index_calculus: bool = False,
    E_rhs_m=None,
    x_b=None,
    f_shifted_poly=None,
    x_coords=None,
    num_workers=None,
    promote_atom=None,
    lp_state=None,
):
    """
    Index-calculus / cofactor-projection DLP attack on the Jacobian.

    This version:
      - supports precomputed relations packs,
      - keeps legacy Mumford relations,
      - optionally augments with fiber relations,
      - optionally promotes a repeated LP into the factor base,
      - cleanly builds G/Q rows,
      - performs the cofactor-projection precheck, and
      - solves the discrete log modulo ℓ.

    Required module-scope helpers:
      sage_poly_from_coeffs, _legacy_build_relations_from_mumford,
      filter_forbidden_relations, precheck_cofactor_projection,
      detect_nontrivial_character_from_projection, verify_character_vectors,
      apply_cofactor_filter, solve_dlp_mod_l_cofactor_projection,
      nullspace_mod_p, check_gq_connectivity, jacobian_to_dict,
      get_relation_row, tonelli_shanks, build_fiber_augmented_relations
    """
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")
    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    full_order = Integer(order)

    if verbose:
        print("\n" + "=" * 70)
        print("INDEX CALCULUS DLP ATTACK (Full Jacobian Group)")
        print("=" * 70)
        print(f"Full Jacobian order |J|: {full_order}")

    ell, h = _compute_full_order_data(full_order, verbose=verbose)
    K, R, f_p, C, J = _prepare_curve_and_jacobian(p, f_coeffs)

    # Build/load relations and FB
    homogeneous_rows, homogeneous_rhs, fb_roots, atom_to_idx, fb_y_cache, d2_to_d1_map = _load_or_build_relations(
        smooth_divs_or_rels=smooth_divs_or_rels,
        G=G,
        Q=Q,
        p=p,
        f_coeffs=f_coeffs,
        f_p=f_p,
        verbose=verbose,
    )

    if verbose:
        print(f"  [Legacy] Master FB: {len(atom_to_idx)} atoms")
        d1_atoms = sum(1 for a in atom_to_idx if isinstance(a, tuple) and len(a) >= 3 and a[0] == "d1")
        d2_atoms = sum(1 for a in atom_to_idx if isinstance(a, tuple) and len(a) >= 3 and a[0] == "d2")
        print(f"  [Legacy] d1 atoms (column space): {d1_atoms}, d2 atoms: {d2_atoms}")
        print(f"  [Legacy] d2 -> d1 mapping size: {len(d2_to_d1_map)}")

    if any(r != 0 for r in homogeneous_rhs):
        raise RuntimeError("_legacy_build_relations_from_mumford returned non-homogeneous relations")

    # Register x_b in atom_to_idx before fiber augment so it isn't treated as a LP.
    if x_b is not None and f_shifted_poly is not None:
        x_b_int = int(x_b)
        atom_xb = None
        for a in atom_to_idx:
            if isinstance(a, tuple) and a[0] == 'd1' and a[1] == x_b_int:
                atom_xb = a
                break
        if atom_xb is None:
            K_fb = GF(p)
            y2_xb = int(f_shifted_poly(K_fb(x_b_int)))
            assert pow(y2_xb, (p - 1) // 2, p) == 1, "x_b not on shifted curve"
            y_xb = tonelli_shanks(y2_xb, p)
            y_xb_can = int(min(y_xb, p - y_xb))
            max_idx = max(atom_to_idx.values()) if atom_to_idx else -1
            atom_to_idx[('d1', x_b_int, y_xb_can)] = max_idx + 1
            fb_y_cache[x_b_int] = y_xb_can
            if verbose:
                print(f"[fiber_pre] Registered x_b={x_b_int} y={y_xb_can} into FB at idx {max_idx + 1}")

    # Optional fiber augmentation.
    if True:
        fiber_rows = []
        if E_rhs_m is not None and x_b is not None and f_shifted_poly is not None:
            _maybe_log_fb_samples(atom_to_idx, f_p, p, verbose=verbose)

            fiber_rows, fiber_stats, lp_state = _augment_relations_from_fibers(
                E_rhs_m=E_rhs_m,
                f_shifted_poly=f_shifted_poly,
                x_b=x_b,
                p=p,
                atom_to_idx=atom_to_idx,
                fb_y_cache=fb_y_cache,
                full_order=full_order,
                ell=ell,
                x_coords=x_coords,
                num_workers=num_workers,
                verbose=verbose,
                promote_atom=promote_atom,
                lp_state=lp_state,
            )

            homogeneous_rows.extend(fiber_rows)

            # Promote repeated LPs into the factor base, and keep those rows too.
            homogeneous_rows, atom_to_idx, fb_y_cache, promoted_count = _append_lp_promotions(
                homogeneous_rows=homogeneous_rows,
                atom_to_idx=atom_to_idx,
                fb_y_cache=fb_y_cache,
                fiber_stats=fiber_stats,
                ell=ell,
                verbose=verbose,
            )

            if verbose:
                print(f"  [Fiber] Added {len(fiber_rows)} smooth relations")
                print("  [Legacy] Factor base and relations prepared")
    else:
        if verbose:
            print("  [Legacy] Factor base and relations prepared")

    if not homogeneous_rows:
        raise RuntimeError("No valid homogeneous relations available")

    # Filter relations that would directly expose G/Q dependence.
    homogeneous_rows = filter_forbidden_relations(
        homogeneous_rows, atom_to_idx, f_p, p, G, Q, verbose=verbose
    )
    if not homogeneous_rows:
        raise RuntimeError("All relations were filtered out (G/Q dependent)")

    if verbose:
        print(f"  [Relations] Loaded {len(homogeneous_rows)} homogeneous relations")
        print(f"  [Factor Base] Size: {len(atom_to_idx)}")
        sys.stdout.flush()

    # Build G and Q rows
    if verbose:
        print("  [non-Recursive] Building G and Q rows directly from factor base...")

    row_g, row_q = _build_gq_rows(G, Q, atom_to_idx, f_p, p, verbose=verbose)

    if verbose:
        print("  [Recursive] Finished smoothing. Checking connectivity...")

    connected, g_support, q_support = check_gq_connectivity(
        homogeneous_rows, row_g, row_q, verbose=verbose
    )
    # Keep the union-find diagnostic too.
    connected_uf = _union_find_connectivity(homogeneous_rows, g_support, q_support)
    if verbose:
        print("graph is connected t/f:", bool(connected and connected_uf))

    basis = nullspace_mod_p(homogeneous_rows, len(atom_to_idx), ell)
    if verbose:
        print("nullspace dimension:", len(basis))

    # Cofactor-projection precheck
    precheck = precheck_cofactor_projection(
        atom_to_idx=atom_to_idx,
        homogeneous_rows=homogeneous_rows,
        row_g=row_g,
        row_q=row_q,
        full_order=full_order,
        J=J,
        f_coeffs=f_coeffs,
        p=p,
        verbose=verbose,
    )

    if "alive_fb_indices" in precheck and "filtered_rows" in precheck:
        char_res = detect_nontrivial_character_from_projection(
            precheck["filtered_rows"],
            precheck["alive_fb_indices"],
            precheck["ell"],
            verbose=True,
        )
        if char_res["found"]:
            verify_character_vectors(
                precheck["filtered_rows"],
                char_res["alive_idx_list"],
                char_res["basis"],
                precheck["ell"],
                verbose=True,
            )
        else:
            if verbose:
                print("No nontrivial character found by linear algebra on projected homogeneous matrix.")
    else:
        if verbose:
            print("[precheck] alive_fb_indices missing (projection unsafe); skipping character detection.")

    if not precheck.get("safe_to_project", False):
        if verbose:
            print(f"\n[Strategy] Pre-check FAILED: {precheck.get('reason', 'unknown reason')}")
            print("  Skipping ℓ-torsion solve")
            sys.stdout.flush()
        raise RuntimeError(f"Cofactor projection unsafe: {precheck.get('reason', 'unknown reason')}")

    filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q = apply_cofactor_filter(
        precheck, atom_to_idx, homogeneous_rows, row_g, row_q, verbose=verbose
    )

    if verbose:
        print("\n[Strategy] Using FILTERED ℓ-torsion solve")
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
            G,
            Q,
            filtered_atom_to_idx,
            J,
            verbose=verbose,
        )
    except Exception as e:
        if verbose:
            print(f"  ! Filtered ℓ-torsion solve failed: {e}")
        raise

    if verbose:
        print(f"  [Result] Discrete log (mod ℓ) = {dlog}")

    # Final verification
    D = Integer(dlog) * G - Q
    if not D.is_zero():
        raise RuntimeError(f"[Verify] ✗ Final verification FAILED: dlog * G ≠ Q (dlog={dlog}, ℓ={ell})")

    if verbose:
        print("  [Verify] ✓ Exact equality dlog * G == Q")

    return Integer(dlog)


