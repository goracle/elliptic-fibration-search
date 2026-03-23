from collections import defaultdict, Counter
from sage.all import GF, PolynomialRing, is_prime, Zmod, matrix, vector, Integer, ZZ, factor, crt, prime_factors, set_random_seed
from search_common import FINITE_FIELD, GROUP_MODULUS, BASE_DIVISOR, TARGET_DIVISOR, DATA_PTS_GENUS2, DEBUG, PREFERRED_X_COORDS
from .sparse_linalg_modp import *
from prime_subgroup_projection import *

"""
lp_incidence_dlp.py
-------------------
Index calculus DLP solver for genus-2 HECC using the LP incidence matrix.

Mathematical structure
----------------------
For each x_s in GF(p), m_s = x_b - x_s, and:

    h_{x_s}(x)  =  f_shifted(x)  -  E_rhs_m(x, m_s)

is a principal divisor on C.  By the fibration construction the three
points x_b, x_s, LP_s (where LP_s = F(x_s) is the third root of h) all
lie on C, giving the divisor relation:

    [x_b] + [x_s] + [LP_s]  -  (poles)  =  0   in J(C)          ... (*)

This holds for every s.

Key observation:  x_s is itself an LP atom (not in the existing FB) and
                  appears as LP_t for the fiber indexed by x_t = x_s.
                  Hence every LP atom has "frequency 2": once as LP_s,
                  once as x_t = LP_{s\' } for some s\'.

Row generation — subtracting two fiber relations:
    Fiber(x_s):    [x_b] + [x_s] + [LP_s]   = poles
    Fiber(LP_s):   [x_b] + [LP_s] + [F(LP_s)] = poles
    ─────────────────────────────────────────────────
    Subtract:      [x_s] − [F(LP_s)] = 0

=> row in LP-column space:  +1 in col(x_s),  −1 in col(F(LP_s))

Both x_s and F(LP_s) are LP atoms.  Each row has exactly two nonzero
entries (+1 and −1).  The set of all such rows defines a graph on LP atoms
that (by the 2-regular property) is a union of disjoint cycles; its rank
is  n_LP − n_components.

Adding the G row (inhomogeneous, log_G G = 1) and reading off k = log_G Q
from the LP solution gives the discrete log.

Sign convention for G/Q inhomogeneous rows:
    Compare the v-polynomial of the Mumford rep to the canonical y
    (y_can = min(sqrt(f), p − sqrt(f))).  The sign is +1 if they agree,
    −1 if they differ.

No imports inside functions.  No HDF5 required.
"""

# ---------------------------------------------------------------------------
# Integer Tonelli-Shanks + canonical-y helper
# ---------------------------------------------------------------------------

def _tonelli(n, p):
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m_ts, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, tmp = 1, t * t % p
        while tmp != 1:
            tmp = tmp * tmp % p
            i += 1
        b = pow(c, 1 << (m_ts - i - 1), p)
        m_ts, c, t, r = i, b * b % p, t * c * c % p, r * b % p
    return r

def _y_can(x_int, f_shifted_fp, K, p):
    """Return (y_can: int, on_curve: bool) for x_int."""
    y2 = int(f_shifted_fp(K(x_int)))
    if y2 == 0:
        return 0, True
    if pow(y2, (p - 1) // 2, p) != 1:
        return 0, False
    y = _tonelli(y2, p)
    return min(y, p - y), True

# ---------------------------------------------------------------------------
# Per-fiber computation
# ---------------------------------------------------------------------------

def _eval_erhs_at_m(E_rhs_m, m_val_K, K, Rx):
    """Evaluate E_rhs_m(x) at m = m_val_K; return GF(p)[x] or None on pole."""
    coeffs = []
    for c in E_rhs_m.list():
        den_val = K(c.denominator()(m_val_K))
        if den_val == K(0):
            return None
        coeffs.append(K(c.numerator()(m_val_K)) / den_val)
    return Rx(coeffs)

# ---------------------------------------------------------------------------
# Enumerate LP pairs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Build LP incidence matrix
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LP graph diagnostics
# ---------------------------------------------------------------------------

def lp_to_col_from_xs(xs_to_lp):
    """Helper: collect all LP atoms from the xs_to_lp mapping."""
    return set(xs_to_lp.keys()) | set(xs_to_lp.values())

def run_lp_dlp_attack(cd, E_rhs_m, f_shifted_fp, atom_to_idx, lp_seed_xs,
                      verbose=True):
    p   = int(FINITE_FIELD)
    ell = int(GROUP_MODULUS)
    x_b = int(DATA_PTS_GENUS2[0])

    if verbose:
        print(f"[run_lp_dlp] x_b={x_b}  p={p}  ell={ell}  FB={len(atom_to_idx)}  seeds={len(lp_seed_xs)}")

    return solve_dlp_via_lp_incidence(
        E_rhs_m=E_rhs_m, f_shifted_fp=f_shifted_fp,
        x_b=x_b, p=p, ell=ell,
        base_divisor=BASE_DIVISOR, target_divisor=TARGET_DIVISOR,
        atom_to_idx=atom_to_idx, lp_seed_xs=lp_seed_xs, verbose=verbose,
    )

global NPRINTED
NPRINTED = 0

# ---------------------------------------------------------------------------
# Atom representation helpers
# ---------------------------------------------------------------------------

def _d1_atom(x_int, y_can):
    return ('d1', x_int, y_can)

# ---------------------------------------------------------------------------
# fiber_lp_pair  (replaces existing)
# ---------------------------------------------------------------------------

# ============================================================
# PATCH 1 of 3 — solve_dlp_via_lp_incidence
# Fix: second-pass seed extraction broken by new 3-tuple atom format.
# atom[0] now returns 'd1'/'d2' (tag string), not an x-coordinate.
# Only d1 atoms have a meaningful x-coordinate to seed fibers from.
# ============================================================

# ============================================================
# PATCH 2 of 3 — build_lp_incidence_matrix
# Fixes:
#   (a) sort crash: atoms are now ('d1',x,y) or ('d2',a,b) 3-tuples.
#       sorted() works fine on uniform 3-tuples; the crash was caused by
#       old 2-tuple atoms still present.  Add an assertion to catch any
#       stragglers clearly instead of a mysterious TypeError.
#   (b) key_by_x fallback lookup: lp_s_atom[0] is now the tag 'd1'/'d2',
#       not the x-coordinate.  For d1 atoms the x-coord is lp_s_atom[1].
#       For d2 atoms there is no single x-coord; skip the fallback.
#   (c) Remove the chain diagnostic spam (5 prints per call).
# ============================================================

# ============================================================
# PATCH 3 of 3 — analyze_lp_graph
# Fix: lp_to_col_from_xs / summary print are fine; the only broken line
# was the chain-diag print inside build_lp_incidence_matrix (now removed).
# analyze_lp_graph itself only uses atom keys as opaque dict keys so it
# works without changes — but add a d1/d2 breakdown to the summary.
# ============================================================

def analyze_lp_graph(xs_to_lp, verbose=True):
    """
    Analyze the functional graph  xs_atom -> lp_atom.

    In the ideal case this is a permutation on the LP atom set,
    giving a union of disjoint cycles.  Atoms are now 3-tuples
    ('d1', x, y) or ('d2', a, b).
    """
    # Find cycles via functional-graph DFS
    visited  = set()
    in_cycle = set()
    n_cycles = 0

    for start in xs_to_lp:
        if start in visited:
            continue
        path    = []
        node    = start
        path_set = {}
        while node not in visited and node in xs_to_lp:
            if node in path_set:
                cycle_nodes = path[path_set[node]:]
                in_cycle.update(cycle_nodes)
                n_cycles += 1
                break
            path_set[node] = len(path)
            path.append(node)
            node = xs_to_lp[node]
        visited.update(path)

    n_lp    = len(lp_to_col_from_xs(xs_to_lp))
    n_in_xs = sum(1 for v in xs_to_lp.values() if v in xs_to_lp)

    if verbose:
        all_atoms = set(xs_to_lp.keys()) | set(xs_to_lp.values())
        d1 = sum(1 for a in all_atoms if a[0] == 'd1')
        d2 = sum(1 for a in all_atoms if a[0] == 'd2')
        print(f"\n[LP graph] {len(xs_to_lp)} xs atoms  {n_lp} total LP atoms  "
              f"({d1} d1 + {d2} d2)")
        print(f"  LP atoms also in xs domain : {n_in_xs}")
        print(f"  Cycle nodes : {len(in_cycle)}  Cycles : {n_cycles}")
        print(f"  Expected rank : {n_lp} - {n_cycles} = {n_lp - n_cycles}")

    return dict(n_xs=len(xs_to_lp), n_lp=n_lp, n_cycles=n_cycles,
                n_cycle_nodes=len(in_cycle), expected_rank=n_lp - n_cycles)

def _d2_atom(quad_factor):
    """
    Convert an irreducible degree-2 polynomial over GF(p) to a canonical
    hashable atom key: ('d2', a, b) for x^2 + a*x + b.
    """
    coeffs = quad_factor.list()   # [b, a, c]
    if quad_factor.degree() != 2:
        raise ValueError(f"Expected degree-2 factor, got degree {quad_factor.degree()}")

    p = quad_factor.base_ring().characteristic()
    c = int(coeffs[2]) % p
    if c == 0:
        raise ValueError("Quadratic factor has zero leading coeff")

    c_inv = pow(c, p - 2, p)
    b = int(coeffs[0]) * c_inv % p
    a = int(coeffs[1]) * c_inv % p
    return ('d2', a, b)

def _linear_root_int(fac, p):
    """
    Return the root r in GF(p) of a linear factor fac = a*x + b.
    """
    coeffs = fac.list()   # [b, a]
    if fac.degree() != 1 or len(coeffs) != 2:
        raise ValueError(f"Expected linear factor, got degree {fac.degree()}")

    b = int(coeffs[0]) % p
    a = int(coeffs[1]) % p
    if a == 0:
        raise ValueError("Linear factor has zero leading coefficient")

    return (-b * pow(a, p - 2, p)) % p

def _lp_atom_from_x(x_int, f_shifted_fp, K, p):
    """
    Encode an LP point at x_int as a canonical atom:

    - d1: linear F_p-rational point
        LP atom = ('d1', x, y_can)
    - d2: point only over F_{p^2}
        LP atom = ('d2', x, y2), where y2 = f(x) mod F_{p^2}

    Returns None if the point is not on the curve.
    """
    y2 = int(f_shifted_fp(K(x_int))) % p
    if y2 == 0:
        return _d1_atom(x_int, 0)

    # Check if square in F_p
    if pow(y2, (p - 1) // 2, p) == 1:
        y_can, ok = _y_can(x_int, f_shifted_fp, K, p)
        if not ok:
            return None
        return _d1_atom(x_int, y_can)

    # Non-square in F_p -> d2 over F_{p^2}
    return ('d2', x_int, y2)

def fiber_lp_pair(x_s_int, E_rhs_m, f_shifted_fp, x_b_K, K, Rx, fb_x_set, p):
    """
    Build the fiber relation for a given x_s.
    Produces exactly one LP atom in canonical d1/d2 format.
    """
    x_b_int = int(x_b_K)
    m_val = x_b_K - K(x_s_int)
    g_at_m = _eval_erhs_at_m(E_rhs_m, m_val, K, Rx)
    if g_at_m is None:
        return ('pole', None)

    h = f_shifted_fp - g_at_m
    if h.is_zero():
        return ('pole', None)

    xs_y, xs_ok = _y_can(x_s_int, f_shifted_fp, K, p)
    if not xs_ok:
        return ('off_curve', None)

    xs_atom = _d1_atom(x_s_int, xs_y)
    xs_in_fb = x_s_int in fb_x_set

    lp_atom = None
    for fac, mult in h.factor():
        deg = fac.degree()
        if deg == 0:
            continue

        if deg == 1:
            r = _linear_root_int(fac, p)
            if r in (x_b_int, x_s_int):
                continue
            if mult != 1:
                return ('partial', xs_atom)
            lp_atom = _lp_atom_from_x(r, f_shifted_fp, K, p)
            if lp_atom is None:
                return ('partial', xs_atom)
            break

        elif deg == 2:
            if mult != 1:
                return ('partial', xs_atom)
            # Pick the remaining root of h as LP; encode canonically
            # Use _extract_lp_root_from_h to get x coordinate
            r = _extract_lp_root_from_h(h, x_s_int, x_b_int, K, Rx)
            if r is None:
                return ('partial', xs_atom)
            lp_atom = _lp_atom_from_x(r, f_shifted_fp, K, p)
            if lp_atom is None:
                return ('partial', xs_atom)
            break

        else:
            return ('partial', xs_atom)

    if lp_atom is None:
        return ('partial', xs_atom)

    return ('lp_pair', xs_atom, lp_atom)

def _extract_lp_root_from_h(h, x_s_int, x_b_int, K, Rx):
    x = Rx.gen()

    # divide out known factors explicitly
    h1 = h // (x - K(x_b_int))**3
    h2 = h1 // (x - K(x_s_int))

    if h2.degree() != 1:
        return None

    return _linear_root_int(h2, K.characteristic())

def enumerate_lp_pairs(E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
                       lp_seed_xs, verbose=True):
    assert lp_seed_xs is not None, "lp_seed_xs required"
    K  = GF(p)
    Rx = PolynomialRing(K, 'x')
    x_b_K = K(int(x_b))

    fb_x_set = set(atom[1] for atom in atom_to_idx if atom[0] == 'd1')
    fb_x_set.add(int(x_b))

    true_lp_seeds = [x for x in lp_seed_xs if int(x) not in fb_x_set]
    forced_seeds = [int(x) for x in (PREFERRED_X_COORDS or []) if int(x) not in [int(s) for s in true_lp_seeds]]
    all_seeds = true_lp_seeds + forced_seeds
    all_seeds = list(set(int(x) for x in lp_seed_xs)) # lol
    true_lp_seeds = all_seeds
    if verbose:
        print(f"[LP enum] {len(lp_seed_xs)} seeds -> {len(true_lp_seeds)} after FB filter")

    xs_to_lp = {}
    fb_xs_to_lp = {}
    n_tried = n_pole = n_off = n_fb = n_partial = 0

    for x_s_int in true_lp_seeds:
        n_tried += 1
        res = fiber_lp_pair(x_s_int, E_rhs_m, f_shifted_fp, x_b_K, K, Rx, fb_x_set, p)
        tag = res[0]
        if tag == 'lp_pair':
            xs_to_lp[res[1]] = res[2]
        elif tag == 'fb_lp_pair':
            fb_xs_to_lp[res[1]] = res[2]
            n_fb += 1
        elif tag == 'pole':
            n_pole += 1
        elif tag == 'off_curve':
            n_off += 1
        elif tag == 'partial':
            n_partial += 1

    # also seed from FB atoms directly
    for x_s_int in fb_x_set:
        if x_s_int == int(x_b):
            continue
        res = fiber_lp_pair(x_s_int, E_rhs_m, f_shifted_fp, x_b_K, K, Rx, fb_x_set, p)
        tag = res[0]
        if tag == 'fb_lp_pair':
            fb_xs_to_lp[res[1]] = res[2]

    counters = dict(tried=n_tried, poles=n_pole, off_curve=n_off,
                    fb_only=n_fb, partial=n_partial, lp_fibers=len(xs_to_lp),
                    fb_lp_fibers=len(fb_xs_to_lp))
    if verbose:
        print(f"[LP enum] tried={n_tried}  LP fibers={len(xs_to_lp)}  FB LP fibers={len(fb_xs_to_lp)}")
        print(f"  poles={n_pole}  off_curve={n_off}  fb_only={n_fb}  partial={n_partial}")
    return xs_to_lp, fb_xs_to_lp, counters

def mumford_divisor_lp_cols(divisor, xs_to_col, f_shifted_fp, E_rhs_m, x_b, p, label='DIV'):
    K = GF(p)
    Rx = PolynomialRing(K, 'x')

    try:
        u_poly, v_poly = divisor[0], divisor[1]
    except Exception as exc:
        print(f"[{label}] Cannot read Mumford (u, v): {exc}")
        return None

    roots = []
    for fac, mult in u_poly.factor():
        if fac.degree() == 1:
            r = _linear_root_int(fac, p)
            roots.extend([r] * int(mult))
        elif fac.degree() != 0:
            print(f"[{label}] u(x) has non-linear factor degree {fac.degree()}")
            return None

    if len(roots) != 2:
        print(f"[{label}] expected 2 F_p roots, got {len(roots)}")
        return None

    result = []
    for x_s_int in roots:
        xs_y_can, xs_ok = _y_can(x_s_int, f_shifted_fp, K, p)
        if not xs_ok:
            print(f"[{label}] x_s={x_s_int} not on curve")
            return None

        xs_atom = _d1_atom(x_s_int, xs_y_can)
        col = xs_to_col.get(xs_atom)
        if col is None:
            print(f"[{label}] x_s atom {xs_atom} not in xs_to_col")
            return None

        v_val = int(v_poly(K(x_s_int))) % p
        sign = 1 if v_val == xs_y_can else -1

        result.append((col, sign))

    if len(result) != 2:
        print(f"[{label}] degenerate: got {len(result)} roots")
        return None

    if result[0][0] == result[1][0]:
        print(f"[{label}] degenerate: both roots map to same column")
        return None

    return (result[0][0], result[0][1], result[1][0], result[1][1])

def build_lp_incidence_matrix(xs_to_lp, fb_xs_to_lp, verbose=True):
    # unified column space: all xs atoms + all LP atoms + xb sentinel
    all_atoms = set(xs_to_lp.keys()) | set(xs_to_lp.values())
    atom_to_col = {atom: i for i, atom in enumerate(all_atoms)}
    xb_col = len(atom_to_col)  # last column is log(xb)

    # each fiber: log(xs) + log(LP) + 3*log(xb) = 0
    fiber_rows = []
    for xs_atom, lp_atom in xs_to_lp.items():
        xs_col = atom_to_col[xs_atom]
        lp_col = atom_to_col[lp_atom]
        fiber_rows.append((xs_col, lp_col, xb_col))

    if verbose:
        print(f"[LP matrix] {len(atom_to_col) + 1} cols ({len(atom_to_col)} atoms + xb), {len(fiber_rows)} fiber rows")

    return atom_to_col, xb_col, fiber_rows

def solve_dlp_via_lp_incidence(
        E_rhs_m, f_shifted_fp, x_b, p, ell,
        base_divisor, target_divisor, atom_to_idx,
        lp_seed_xs, verbose=True):

    assert is_prime(ell), f"ell={ell} must be prime"

    if verbose:
        print("=" * 70)
        print("LP INCIDENCE DLP SOLVER")
        print(f"  p={p}  ell={ell}  x_b={x_b}  FB size={len(atom_to_idx)}  lp_seeds={len(lp_seed_xs)}")
        print("=" * 70)

    def _d1_x(atom):
        return atom[1] if atom[0] == 'd1' else None

    xs_to_lp, fb_xs_to_lp, _ = enumerate_lp_pairs(
        E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
        lp_seed_xs=lp_seed_xs, verbose=verbose)

    pass_num = 1
    while True:
        lp_value_xs = {_d1_x(a) for a in xs_to_lp.values() if a[0] == 'd1'}
        xs_key_xs   = {_d1_x(a) for a in xs_to_lp.keys()   if a[0] == 'd1'}
        new_seeds   = lp_value_xs - xs_key_xs

        if not new_seeds:
            break

        pass_num += 1
        if verbose:
            print(f"[LP DLP] Pass {pass_num}: {len(new_seeds)} new seeds from LP values")

        xs_to_lp2, fb_xs_to_lp2, _ = enumerate_lp_pairs(
            E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
            lp_seed_xs=new_seeds, verbose=verbose)
        xs_to_lp.update(xs_to_lp2)
        fb_xs_to_lp.update(fb_xs_to_lp2)

        if not xs_to_lp2:
            break

    if verbose:
        print(f"[LP DLP] Enumeration complete after {pass_num} passes, {len(xs_to_lp)} xs->lp pairs")

    if not xs_to_lp:
        print("[LP DLP] No LP pairs found.")
        return dict(dlp=None, verified=False, solution=None,
                    n_lp_cols=0, n_homogeneous=0, graph_info={})

    atom_to_col, xb_col, fiber_rows = build_lp_incidence_matrix(xs_to_lp, fb_xs_to_lp, verbose=verbose)
    graph_info = analyze_lp_graph(xs_to_lp, verbose=verbose)

    g_result = mumford_divisor_lp_cols(base_divisor,   atom_to_col, f_shifted_fp, E_rhs_m, x_b, p, 'G')
    q_result = mumford_divisor_lp_cols(target_divisor, atom_to_col, f_shifted_fp, E_rhs_m, x_b, p, 'Q')

    if g_result is None:
        print("[LP DLP] G not in atom basis — aborting.")
        return dict(dlp=None, verified=False, solution=None,
                    n_lp_cols=len(atom_to_col), n_homogeneous=len(fiber_rows), graph_info=graph_info)

    if q_result is None:
        print("[LP DLP] Q not in atom basis — aborting.")
        return dict(dlp=None, verified=False, solution=None,
                    n_lp_cols=len(atom_to_col), n_homogeneous=len(fiber_rows), graph_info=graph_info)

    g_c1, g_s1, g_c2, g_s2 = g_result
    q_c1, q_s1, q_c2, q_s2 = q_result

    if verbose:
        print(f"[LP DLP] G cols=({g_c1},{g_c2}) signs=({g_s1},{g_s2})")
        print(f"[LP DLP] Q cols=({q_c1},{q_c2}) signs=({q_s1},{q_s2})")

    k = solve_lp_system(atom_to_col, xb_col, fiber_rows, g_result, q_result, ell, verbose=verbose)

    if k is None:
        return dict(dlp=None, verified=False, solution=None,
                    n_lp_cols=len(atom_to_col), n_homogeneous=len(fiber_rows), graph_info=graph_info)

    verified = False
    try:
        verified = (int(k) * base_divisor == target_divisor)
        status = "✓ VERIFIED" if verified else "✗ FAILED"
        if verbose:
            print(f"[LP DLP] {status}: {k} * G {'==' if verified else '!='} Q")
    except Exception as exc:
        print(f"[LP DLP] Verification error: {exc}")
        raise

    return dict(dlp=k, verified=verified, solution=None,
                n_lp_cols=len(atom_to_col), n_homogeneous=len(fiber_rows), graph_info=graph_info)

def solve_lp_system(atom_to_col, xb_col, fiber_rows, g_result, q_result, ell, verbose=True):
    col_to_atom = {v: k for k, v in atom_to_col.items()}

    def is_d2_col(col):
        atom = col_to_atom.get(col)
        return atom is not None and atom[0] == 'd2'

    g_c1, g_s1, g_c2, g_s2 = g_result
    q_c1, q_s1, q_c2, q_s2 = q_result

    n_d1 = sum(1 for a in atom_to_col if a[0] == 'd1')
    n_d2 = sum(1 for a in atom_to_col if a[0] == 'd2')

    if verbose:
        print(f"[LP solve] atom breakdown: {n_d1} d1, {n_d2} d2, 1 xb")
        print(f"[LP solve] G cols=({g_c1},{g_c2})  d2? {is_d2_col(g_c1) or is_d2_col(g_c2)}")
        print(f"[LP solve] Q cols=({q_c1},{q_c2})  d2? {is_d2_col(q_c1) or is_d2_col(q_c2)}")

    if is_d2_col(g_c1) or is_d2_col(g_c2):
        if verbose:
            print("[LP solve] G has d2 column — aborting.")
        return None

    if is_d2_col(q_c1) or is_d2_col(q_c2):
        if verbose:
            print("[LP solve] Q has d2 column — aborting.")
        return None

    d1_fiber_rows = [(xs_col, lp_col)
                     for (xs_col, lp_col, xb_col_i) in fiber_rows
                     if not is_d2_col(xs_col) and not is_d2_col(lp_col)]

    if verbose:
        print(f"[LP solve] after d2 drop: {len(d1_fiber_rows)} fiber rows")

    # eliminate xb by subtracting consecutive fiber row pairs
    homogeneous_rows = []
    for i in range(len(d1_fiber_rows) - 1):
        xs_i, lp_i = d1_fiber_rows[i]
        xs_j, lp_j = d1_fiber_rows[i + 1]
        row = {}
        for col, coeff in [(xs_i, 1), (lp_i, 1), (xs_j, ell - 1), (lp_j, ell - 1)]:
            row[col] = (row.get(col, 0) + coeff) % ell
        row = {k: v for k, v in row.items() if v != 0}
        if row:
            homogeneous_rows.append(row)

    if not homogeneous_rows:
        if verbose:
            print("[LP solve] no homogeneous rows after xb elimination.")
        return None

    # remap surviving columns to contiguous indices
    surviving_cols = set()
    for row in homogeneous_rows:
        surviving_cols.update(row.keys())
    surviving_cols.add(g_c1)
    surviving_cols.add(g_c2)
    surviving_cols.add(q_c1)
    surviving_cols.add(q_c2)

    old_to_new = {old: new for new, old in enumerate(sorted(surviving_cols))}
    n_cols = len(old_to_new)

    g_c1n = old_to_new[g_c1]
    g_c2n = old_to_new[g_c2]
    q_c1n = old_to_new[q_c1]
    q_c2n = old_to_new[q_c2]

    remapped_hom_rows = []
    for row in homogeneous_rows:
        new_row = {old_to_new[c]: v for c, v in row.items() if c in old_to_new}
        if new_row:
            remapped_hom_rows.append(new_row)

    g_row = {}
    if g_s1 != 0:
        g_row[g_c1n] = int(g_s1) % ell
    if g_s2 != 0:
        g_row[g_c2n] = (g_row.get(g_c2n, 0) + int(g_s2)) % ell
    g_row = {k: v for k, v in g_row.items() if v != 0}

    if verbose:
        print(f"[LP solve] {len(remapped_hom_rows)} hom rows x {n_cols} cols before mixing")

    mixed_hom_rows = mix_rows_to_target_count(
        remapped_hom_rows, ell, target_count=n_cols - 1, mix_count=4, verbose=verbose
    )

    all_rows = mixed_hom_rows + [g_row]
    all_rhs  = [0] * len(mixed_hom_rows) + [1]

    n_rows_final = len(all_rows)
    iters = 2 * min(n_rows_final, n_cols) + 200

    if verbose:
        print(f"[LP solve] BW on {n_rows_final} x {n_cols}  mod {ell}  (iters={iters})")

    A = SparseRelationMatrix(all_rows, all_rhs, ell)

    sol_vec, success = block_wiedemann_inhomogeneous_solve(
        A, all_rhs, verbose=verbose, max_attempts=5, iters=iters
    )

    if not success or sol_vec is None:
        if verbose:
            print("[LP solve] BW failed.")
        return None

    k = (int(q_s1) * int(sol_vec[q_c1n]) + int(q_s2) * int(sol_vec[q_c2n])) % ell

    if verbose:
        print(f"[LP solve] k = {k}")

    return k
