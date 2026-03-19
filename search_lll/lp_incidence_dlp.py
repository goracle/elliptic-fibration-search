from collections import defaultdict, Counter
from sage.all import GF, PolynomialRing, is_prime
from search_common import FINITE_FIELD, GROUP_MODULUS, BASE_DIVISOR, TARGET_DIVISOR, DATA_PTS_GENUS2, DEBUG
from .sparse_linalg_modp import SparseRelationMatrix, block_wiedemann_inhomogeneous_solve

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

def analyze_lp_graph(xs_to_lp, verbose=True):
    """
    Analyze the functional graph  x_s → LP_s.

    In the ideal case this is a permutation on the LP set (every LP atom is
    also an x_s atom), giving a union of disjoint cycles.
    """
    in_degree  = Counter(xs_to_lp.values())
    out_degree = Counter(xs_to_lp.keys())   # all 1 by construction

    # Find cycles via functional-graph DFS
    visited  = set()
    in_cycle = set()
    n_cycles = 0

    for start in xs_to_lp:
        if start in visited:
            continue
        path = []
        node = start
        path_set = {}
        while node not in visited and node in xs_to_lp:
            if node in path_set:
                # found cycle
                cycle_start = node
                i = path_set[node]
                cycle_nodes = path[i:]
                in_cycle.update(cycle_nodes)
                n_cycles += 1
                break
            path_set[node] = len(path)
            path.append(node)
            node = xs_to_lp[node]
        visited.update(path)

    n_lp = len(lp_to_col_from_xs(xs_to_lp))
    if verbose:
        print(f"\n[LP graph] {len(xs_to_lp)} xs atoms  {n_lp} total LP atoms")
        print(f"  LP atoms also in xs domain : {sum(1 for v in xs_to_lp.values() if v in xs_to_lp)}")
        print(f"  Cycle nodes : {len(in_cycle)}  Cycles : {n_cycles}")
        print(f"  Expected rank : {n_lp} - {n_cycles} = {n_lp - n_cycles}")

    return dict(n_xs=len(xs_to_lp), n_lp=n_lp, n_cycles=n_cycles,
                n_cycle_nodes=len(in_cycle), expected_rank=n_lp - n_cycles)

def lp_to_col_from_xs(xs_to_lp):
    """Helper: collect all LP atoms from the xs_to_lp mapping."""
    return set(xs_to_lp.keys()) | set(xs_to_lp.values())

# ---------------------------------------------------------------------------
# Map a Mumford divisor into LP columns (with sign)
# ---------------------------------------------------------------------------

def mumford_divisor_lp_cols(divisor, lp_to_col, f_shifted_fp, p, label='DIV'):
    """
    Return (col1, sign1, col2, sign2) for the two roots of divisor's u-polynomial.

    sign = +1 if v(root) matches y_can,  -1 if it matches p - y_can.
    Returns None if any root is absent from lp_to_col.
    """
    K = GF(p)
    try:
        u_poly, v_poly = divisor[0], divisor[1]
    except Exception as exc:
        print(f"[{label}] Cannot read Mumford (u, v): {exc}")
        return None

    roots_data = u_poly.roots(K)
    flat = [x_r for x_r, m in roots_data for _ in range(int(m))]
    if len(flat) != 2:
        print(f"[{label}] u(x) must have exactly 2 roots; got {len(flat)}")
        return None

    result = []
    for x_r in flat:
        x_int  = int(x_r)
        y_can, ok = _y_can(x_int, f_shifted_fp, K, p)
        if not ok:
            print(f"[{label}] root x={x_int} off-curve")
            return None
        pt  = (x_int, y_can)
        col = lp_to_col.get(pt)
        if col is None:
            print(f"[{label}] root {pt} not in LP basis")
            return None
        v_val = int(v_poly(K(x_int))) % p
        sign  = 1 if v_val == y_can else -1
        result.append((col, sign))

    if result[0][0] == result[1][0]:
        print(f"[{label}] degenerate: both roots share the same LP column")
        return None

    return (result[0][0], result[0][1], result[1][0], result[1][1])

# ---------------------------------------------------------------------------
# Solve the LP system mod ell via Block-Wiedemann
# ---------------------------------------------------------------------------

def solve_lp_system(lp_to_col, row_pairs, inhom_rows, ell, verbose=True):
    """
    Solve the LP incidence system mod ell using Block-Wiedemann.

    Homogeneous rows (fiber subtractions):
        +1 in col_pos,  -1 in col_neg,  RHS = 0

    Inhomogeneous rows (G, Q):
        sign1 * col1  +  sign2 * col2  =  rhs

    row_pairs  : list of (col_pos, col_neg)
    inhom_rows : list of (col1, sign1, col2, sign2, rhs_int)

    Returns list of ints (solution) or None.
    """
    n_cols = len(lp_to_col)
    n_hom  = len(row_pairs)
    n_inh  = len(inhom_rows)
    n_rows = n_hom + n_inh

    if verbose:
        print(f"[LP solve] {n_rows} x {n_cols}  mod {ell}  "
              f"({n_hom} fiber rows  +  {n_inh} inhomogeneous)")

    row_dicts = []
    rhs_list  = []

    for (cp, cn) in row_pairs:
        d = {cp: 1}
        d[cn] = d.get(cn, 0) + (ell - 1)   # -1 mod ell
        row_dicts.append({k: v % ell for k, v in d.items() if v % ell != 0})
        rhs_list.append(0)

    for (c1, s1, c2, s2, rhs) in inhom_rows:
        d = {}
        if s1 != 0:
            d[c1] = int(s1) % ell
        if s2 != 0:
            d[c2] = (d.get(c2, 0) + int(s2)) % ell
        row_dicts.append({k: v for k, v in d.items() if v != 0})
        rhs_list.append(int(rhs) % ell)

    A = SparseRelationMatrix(row_dicts, rhs_list, ell)

    if verbose:
        print(f"[LP solve] SparseRelationMatrix: {A.n_rows} rows x {A.n_cols} cols")

    sol_vec, success = block_wiedemann_inhomogeneous_solve(A, rhs_list, verbose=verbose)

    if not success or sol_vec is None:
        if verbose:
            print("[LP solve] Block-Wiedemann failed.")
        return None

    return [int(sol_vec[i]) for i in range(n_cols)]

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

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

    xs_to_lp, _ = enumerate_lp_pairs(
        E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
        lp_seed_xs=lp_seed_xs, verbose=verbose)

    # Second pass: seed with the LP values to close the functional graph
    lp_value_xs = set(atom[0] for atom in xs_to_lp.values())
    new_seeds = lp_value_xs - set(atom[0] for atom in xs_to_lp.keys())
    if new_seeds:
        if verbose:
            print(f"[LP DLP] Second pass: {len(new_seeds)} new seeds from LP values")
        xs_to_lp2, _ = enumerate_lp_pairs(
            E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
            lp_seed_xs=new_seeds, verbose=verbose)
        xs_to_lp.update(xs_to_lp2)

    if not xs_to_lp:
        print("[LP DLP] No LP pairs found.")
        return dict(dlp=None, verified=False, lp_to_col={}, solution=None,
                    n_lp_cols=0, n_homogeneous=0, graph_info={})

    lp_to_col, row_pairs = build_lp_incidence_matrix(xs_to_lp, verbose=verbose)
    graph_info = analyze_lp_graph(xs_to_lp, verbose=verbose)

    g_result = mumford_divisor_lp_cols(base_divisor,   lp_to_col, f_shifted_fp, p, 'G')
    q_result = mumford_divisor_lp_cols(target_divisor, lp_to_col, f_shifted_fp, p, 'Q')

    if g_result is None:
        print("[LP DLP] G not in LP basis — aborting.")
        return dict(dlp=None, verified=False, lp_to_col=lp_to_col, solution=None,
                    n_lp_cols=len(lp_to_col), n_homogeneous=len(row_pairs), graph_info=graph_info)

    if q_result is None:
        print("[LP DLP] Q not in LP basis — aborting.")
        return dict(dlp=None, verified=False, lp_to_col=lp_to_col, solution=None,
                    n_lp_cols=len(lp_to_col), n_homogeneous=len(row_pairs), graph_info=graph_info)

    g_c1, g_s1, g_c2, g_s2 = g_result
    q_c1, q_s1, q_c2, q_s2 = q_result

    if verbose:
        print(f"[LP DLP] G cols=({g_c1},{g_c2}) signs=({g_s1},{g_s2})")
        print(f"[LP DLP] Q cols=({q_c1},{q_c2}) signs=({q_s1},{q_s2})")

    inhom_rows = [(g_c1, g_s1, g_c2, g_s2, 1)]
    solution   = solve_lp_system(lp_to_col, row_pairs, inhom_rows, ell, verbose=verbose)

    if solution is None:
        return dict(dlp=None, verified=False, lp_to_col=lp_to_col, solution=None,
                    n_lp_cols=len(lp_to_col), n_homogeneous=len(row_pairs), graph_info=graph_info)

    k = (q_s1 * solution[q_c1] + q_s2 * solution[q_c2]) % ell

    if verbose:
        print(f"\n[LP DLP] log Q = {q_s1}*{solution[q_c1]} + {q_s2}*{solution[q_c2]} = {k}")

    verified = False
    try:
        verified = (int(k) * base_divisor == target_divisor)
        status = "✓ VERIFIED" if verified else "✗ FAILED"
        if verbose:
            print(f"[LP DLP] {status}: {k} * G {'==' if verified else '!='} Q")
    except Exception as exc:
        print(f"[LP DLP] Verification error: {exc}")

    return dict(dlp=k, verified=verified, lp_to_col=lp_to_col, solution=solution,
                n_lp_cols=len(lp_to_col), n_homogeneous=len(row_pairs), graph_info=graph_info)

def fiber_lp_pair(x_s_int, E_rhs_m, f_shifted_fp, x_b_K, K, Rx, fb_x_set, p):
    m_val = x_b_K - K(x_s_int)
    g_at_m = _eval_erhs_at_m(E_rhs_m, m_val, K, Rx)
    if g_at_m is None:
        return ('pole', None)

    h = f_shifted_fp - g_at_m
    if h.is_zero():
        return ('pole', None)

    roots_with_mult = h.roots()
    if not roots_with_mult:
        return ('pole', None)

    tagged = []
    for x_r, _mult in roots_with_mult:
        x_int = int(x_r)
        y_can, ok = _y_can(x_int, f_shifted_fp, K, p)
        if not ok:
            return ('off_curve', None)
        tagged.append(((x_int, y_can), x_int in fb_x_set))

    xs_y, xs_ok = _y_can(x_s_int, f_shifted_fp, K, p)
    if not xs_ok:
        return ('off_curve', None)

    xs_lp = (x_s_int, xs_y)
    xs_in_fb = x_s_int in fb_x_set

    lp_pts = [pt for pt, in_fb in tagged if not in_fb and pt != xs_lp]

    if xs_in_fb:
        if len(lp_pts) == 1:
            return ('partial', lp_pts[0])
        return ('fb_only', None)

    if not lp_pts:
        return ('partial', xs_lp)
    return ('lp_pair', xs_lp, lp_pts[0])

def build_lp_incidence_matrix(xs_to_lp, verbose=True):
    all_atoms = set(xs_to_lp.keys()) | set(xs_to_lp.values())
    lp_to_col = {atom: i for i, atom in enumerate(sorted(all_atoms))}

    key_by_x = {atom[0]: atom for atom in xs_to_lp.keys()}

    # Diagnose why chains are failing on first 5 entries
    sample = list(xs_to_lp.items())[:5]
    for xs_atom, lp_s_atom in sample:
        exact = xs_to_lp.get(lp_s_atom)
        by_x  = key_by_x.get(lp_s_atom[0])
        print(f"[LP chain diag] xs={xs_atom}  lp_s={lp_s_atom}")
        print(f"  exact lookup: {exact}")
        print(f"  key_by_x[lp_s.x]={by_x}  y_match={by_x == lp_s_atom if by_x else 'N/A'}")

    row_pairs = []
    skipped = 0

    for xs_atom, lp_s_atom in xs_to_lp.items():
        next_key = xs_to_lp.get(lp_s_atom)
        if next_key is None:
            candidate = key_by_x.get(lp_s_atom[0])
            if candidate is not None:
                next_key = xs_to_lp.get(candidate)

        if next_key is None:
            skipped += 1
            continue

        col_pos = lp_to_col[xs_atom]
        col_neg = lp_to_col[next_key]

        if col_pos != col_neg:
            row_pairs.append((col_pos, col_neg))

    if verbose:
        print(f"[LP matrix] {len(lp_to_col)} LP cols  {len(row_pairs)} rows  "
              f"({skipped} chain-ends skipped)")

    return lp_to_col, row_pairs

def enumerate_lp_pairs(E_rhs_m, f_shifted_fp, x_b, p, atom_to_idx,
                       lp_seed_xs, verbose=True):
    assert lp_seed_xs is not None, "lp_seed_xs required"
    K = GF(p)
    Rx = PolynomialRing(K, 'x')
    x_b_K = K(int(x_b))

    fb_x_set = set(atom[1] for atom in atom_to_idx if atom[0] == 'd1')
    fb_x_set.add(int(x_b))

    # Only seed with x-values that are genuinely outside the full FB
    true_lp_seeds = [x for x in lp_seed_xs if int(x) not in fb_x_set]

    if verbose:
        print(f"[LP enum] {len(lp_seed_xs)} seeds -> {len(true_lp_seeds)} after FB filter")

    xs_to_lp = {}
    n_tried = n_pole = n_off = n_fb = n_partial = 0

    for x_s_int in true_lp_seeds:
        n_tried += 1
        res = fiber_lp_pair(x_s_int, E_rhs_m, f_shifted_fp, x_b_K, K, Rx, fb_x_set, p)
        tag = res[0]
        if tag == 'lp_pair':
            xs_to_lp[res[1]] = res[2]
        elif tag == 'pole':
            n_pole += 1
        elif tag == 'off_curve':
            n_off += 1
        elif tag == 'fb_only':
            n_fb += 1
        elif tag == 'partial':
            n_partial += 1

    counters = dict(tried=n_tried, poles=n_pole, off_curve=n_off,
                    fb_only=n_fb, partial=n_partial, lp_fibers=len(xs_to_lp))

    if verbose:
        print(f"[LP enum] tried={n_tried}  LP fibers={len(xs_to_lp)}")
        print(f"  poles={n_pole}  off_curve={n_off}  fb_only={n_fb}  partial={n_partial}")

    return xs_to_lp, counters
