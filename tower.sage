import operator, random, math, sys
from functools import reduce
from sage.all import *
from sage.functions.other import binomial
from search_common import *
from stats import *
from sage.rings.rational_field import QQ
from math import comb as _int_binom

# tower.sage
# Numeric-first fibration tower builder (strict, exact QQ arithmetic)
# - substitute numeric inputs (curve coeffs, x_i) as QQ immediately
# - interpolate Q(x) exactly from rational (x,y) points before building fibration
# - every solve must return exactly one branch;
#   solution values must be exact QQ (or error)
# - plain python int seed (safe with `sage tower.sage`)
#
# Usage: sage tower.sage

# CONFIG: tune these to trade runtime vs accuracy
_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41]   # primes to test for rejections and collisions
_SAMPLE_M_VALUES = [ -7, -3, -1, 0, 1, 2, 3, 7 ]      # small integer m samples to probe modular behaviour
_NUM_RANDOM_M = 5                                     # additional random m samples (drawn small)
_MAX_DEGREE_PENALTY = 5.0
_WEIGHT_HEIGHT = 1.0
_WEIGHT_DEG = 1.2
_WEIGHT_DISC = 1.5
_WEIGHT_BADPRIME = 6.0    # each small bad prime is expensive -> large penalty
_WEIGHT_COLLISION = 8.0   # collisions are deadly for consensus; heavy penalty

# ---------- Utilities ----------
try:
    PROFILE = profile
except NameError:
    PROFILE = profile

# === normalize and check helpers (drop-in) ===

def _pick_field(step, names):
    # pick first present field name or None
    try:
        for n in names:
            if hasattr(step, 'get') and n in step:
                return step[n]
            if hasattr(step, n):
                try:
                    return getattr(step, n)
                except Exception:
                    raise
        # sequence fallback common positions
        if hasattr(step, '__len__') and not hasattr(step, 'get'):
            if len(step) > 0:
                return step[0]
    except Exception:
        raise
    return None

def normalize_step(step):
    # step is a dict with keys like 'f_i', 'Q_i', 'r_expr', 'info'
    fx = step.get('f_i')
    r_expr = step.get('r_expr')
    param = 'm'   # your tower construction always uses m
    double_root_x = step.get('x0')  # not present in your dump, may be None
    return {'fx': fx, 'r_expr': r_expr, 'param': param,
            'double_root_x': double_root_x, 'raw': step}

# Helpers for robust assertions in tower.sage

# -----------------------
# Deterministic branch chooser for solve() results
# -----------------------
@PROFILE
def pick_solution_by_degree(solutions, target_var, prefer_max_degree=True):
    """Given many dict solutions returned by solve(..., solution_dict=True),
       pick the solution whose 'target_var' has the largest (or smallest) polynomial degree
       in the symbolic parameter (heuristic). Raises RuntimeError if ambiguous.
       solutions: list of dicts
    """
    if not solutions:
        raise RuntimeError("No solutions provided to pick_solution_by_degree")
    scored = []
    for sol in solutions:
        if target_var not in sol:
            raise RuntimeError("candidate solution missing target_var: %s" % target_var)
        sval = sol[target_var]
        # try to measure 'degree' in the parameter(s) by counting occurrences of symbolic variables
        try:
            deg = 0
            for v in SR(sval).variables():
                # crude proxy: degree in each variable via polynomial degree if possible
                try:
                    PR = PolynomialRing(QQ, str(v))
                    deg = max(deg, PR(SR(sval)).degree())
                except Exception:
                    # fallback: count appearance
                    deg = max(deg, str(sval).count(str(v)))
                    raise
        except Exception:
            deg = 0
            raise
        scored.append((deg, sol))
    # pick solution by degree
    if prefer_max_degree:
        scored.sort(key=lambda t: t[0], reverse=True)
    else:
        scored.sort(key=lambda t: t[0])
    # ensure unique top candidate
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise RuntimeError("Ambiguous solution selection: top two candidates have same score.")
    return scored[0][1]

# -----------------------
# High-level comparison wrapper for a layer vs Mathematica output
# -----------------------
# -----------------------
# Robust extraction + comparison wrapper for a layer vs Mathematica output
# -----------------------
# keys we commonly store generated expressions under in layer dicts

@PROFILE
def assert_layer_match(layer, expected_Q=None, expected_r=None):
    """
    Compare layer['Q_i'] (QQ polynomial) and layer['r_expr'] (SR) to expected values.
    - expected_Q: either QQ polynomial or list of coefficients [c0,c1,...]
    - expected_r: SR expression or string; comparison uses SR equality (simplify)
    Raises RuntimeError if mismatch.
    """
    if expected_Q is not None:
        Qgot = layer.get('Q_i')
        if Qgot is None:
            raise RuntimeError("Layer missing 'Q_i'")
        if not isinstance(expected_Q, (Polynomial)):
            # assume list of coefficients
            R = PolynomialRing(QQ, 'x')
            expected_Qpoly = R([QQ(v) for v in expected_Q])
        else:
            expected_Qpoly = expected_Q
        if Qgot != expected_Qpoly:
            raise RuntimeError(f"Q mismatch.\n got: {Qgot}\n expected: {expected_Qpoly}")

    if expected_r is not None:
        rgot = layer.get('r_expr')
        if rgot is None:
            raise RuntimeError("Layer missing 'r_expr'")
        # try SR comparison
        if isinstance(expected_r, str):
            expected_r_sr = SR(expected_r)
        else:
            expected_r_sr = expected_r
        if (SR(rgot) - SR(expected_r_sr)).simplify() != 0:
            raise RuntimeError(f"r_expr mismatch.\n got: {rgot}\n expected: {expected_r_sr}")

# small helper to coerce to QQ and raise helpful error if not possible
@PROFILE
def force_QQ(val, name=''):
    try:
        return QQ(val)
    except Exception as e:
        raise RuntimeError(f"Could not coerce {name!s} to QQ: {val!r} — error: {e}")

# ---------- Core Tower Builder ----------

# ---------- Replacement: interpolate_Q_general ----------
# Required imports (add these to your file if not already present)
# from sage.functions.other import binomial
# from sage.rings.rational_field import QQ
# from sage.rings.polynomial.polynomial_ring import PolynomialRing
# import random

# In tower.sage

@PROFILE
def _verify_fibration_step_properties(fx, r_expr, param):
    # fx: polynomial in x
    # r_expr: expression in m
    # param: expected to be 'm' (string or symbol)
    x = var('x')
    try:
        m = var(str(param))
    except Exception:
        m = var('m')
        raise

    fx_sr = SR(fx)

    # derivative wrt x
    #dfx_dx = fx_sr.derivative(x)
    dfx_dx = kth_derivative(fx_sr, 1, x)

    # derivative wrt m (if r_expr uses m)
    if r_expr is not None:
        r_sr = SR(r_expr)
        if 'm' in [str(v) for v in r_sr.variables()]:
            dr_dm = r_sr.derivative(m)
        else:
            dr_dm = None
    else:
        dr_dm = None

    return {'dfx_dx': dfx_dx, 'dr_dm': dr_dm}

# ---------- Main Driver ----------

###############################################################
# Minimal jet checker for tower.sage
# Runs automatically, safe, pure-Python syntax, no new interface
###############################################################

# Replace previous jet_check_safe with this exact function (top-level in tower.sage)
def jet_check_tower_deep(tower, pts_xy, max_order=5, m0=0):
    """
    Deep jet analysis: expand x(m) and y(m) to high order
    and verify consistency across all tower layers.
    """

    x0, y0 = pts_xy[0]

    # Symbolic setup
    x, y, m = var('x y m')
    t = var('t')  # local parameter

    # Create symbolic coefficients for the series
    a_coeffs = [var(f'a{i}') for i in range(2, max_order+1)]
    b_coeffs = [var(f'b{i}') for i in range(1, max_order+1)]

    # Build series expansions
    x_series = SR(x0) - t + sum(a_coeffs[i-2] * t**i for i in range(2, max_order+1))
    y_series = SR(y0) + sum(b_coeffs[i-1] * t**i for i in range(1, max_order+1))
    m_series = t + SR(m0)

    tower_jets = []
    for layer_idx, layer in enumerate(tower):
        print(f"\n[DEEP JET] Layer {layer_idx+1}")
        F_i = y**2 - layer['f_i']  # Construct full curve equation

        # Substitute series
        expr = F_i.subs({x: x_series, y: y_series, m: m_series}).expand()

        # Extract Taylor coefficients
        coeffs = {}
        for order in range(max_order + 1):
            c = expr.diff(t, order).subs({t: 0})
            if order > 0:
                c = c / factorial(order)
            coeffs[order] = c.simplify()

        # Solve order-by-order
        eqs = [coeffs[i] == 0 for i in range(max_order + 1) if coeffs[i] != 0]
        unknowns = a_coeffs + b_coeffs

        try:
            sol = solve(eqs, unknowns, solution_dict=True)
            if sol:
                first_sol = sol[0] if len(sol) == 1 else sol
                #free_params = [str(u) for u in unknowns if u not in first_sol]
                free_params = []
                for u in unknowns:
                    if u not in first_sol:
                        free_params.append(str(u))
                    else:
                        val = first_sol[u]
                        try:
                            val_vars = {str(v) for v in val.variables()}
                            unknown_names = {str(unk) for unk in unknowns}
                            new_free_vars = val_vars - unknown_names - {'m'}
                            if new_free_vars:
                                free_params.append(f"{str(u)}→{','.join(new_free_vars)}")
                        except Exception:
                            raise

                if free_params:
                    print(f"  ✓ Solution found. Free parameters: {', '.join(free_params)}")
                else:
                    print(f"  ✓ Solution found. Fully determined (no free parameters)")

                # Print the series coefficients
                print(f"  Series expansion x(m) = {x0} - m + ...")
                for i in range(2, min(4, max_order+1)):  # Show a2, a3
                    coeff_var = var(f'a{i}')
                    if coeff_var in first_sol:
                        val = first_sol[coeff_var]
                        print(f"    a{i} = {val}")
                    else:
                        print(f"    a{i} = free")

                print(f"  Series expansion y(m) = {y0} + ...")
                for i in range(1, min(3, max_order)):  # Show b1, b2
                    coeff_var = var(f'b{i}')
                    if coeff_var in first_sol:
                        val = first_sol[coeff_var]
                        print(f"    b{i} = {val}")
                    else:
                        print(f"    b{i} = free")

                tower_jets.append({
                    'layer': layer_idx,
                    'solution': first_sol,
                    'free_params': free_params,
                    'obstructed': False
                })
            else:
                print(f"  ❌ OBSTRUCTED: No solution to higher-order equations")
                tower_jets.append({
                    'layer': layer_idx,
                    'obstructed': True,
                    'reason': 'No solution to higher-order equations'
                })
        except Exception as e:
            print(f"  ❌ OBSTRUCTED: {e}")
            tower_jets.append({
                'layer': layer_idx,
                'obstructed': True,
                'reason': str(e)
            })
            raise

    print("\n" + "="*70)
    print("TOWER JET SUMMARY")
    print("="*70)
    print(f"Total layers: {len(tower)}")
    obstructed = sum(1 for j in tower_jets if j.get('obstructed', False))
    if obstructed == 0:
        print("✓ All layers formally smooth (no obstructions)")
    else:
        print(f"❌ {obstructed} layer(s) have obstructions")

    return tower_jets

def jet_check_safe(F_sr, pts_xy, m0=0):
    """
    Minimal jet checker for tower.sage.
    - F_sr: SR expression for the current layer polynomial (f_i or F_i).
    - pts_xy: list-like of rational (x,y) pairs; uses pts_xy[0] as the point on the rail.
    - m0: base m-value (default 0).
    Prints one short line reporting a2 or obstruction. Let errors propagate.
    """
    assert pts_xy and len(pts_xy) >= 1, "pts_xy must contain at least one (x,y) pair"
    x0, y0 = pts_xy[0]

    # Declare symbols that must match those used in F_sr
    x, y, m = var('x y m')
    t = var('t')
    a2 = var('a2')
    b1 = var('b1')
    b2 = var('b2')

    # Convert F_sr to symbolic and expand to ensure proper form
    proto = SR(F_sr).expand()

    # local series ansatz: rail x = x0 - t (since x = x1 - m)
    x_series = SR(x0) - t + a2 * t * t
    y_series = SR(y0) + b1 * t + b2 * t * t
    m_series = t + SR(m0)

    # Substitute using the same symbolic names as F_sr
    try:
        expr = proto.subs({x: x_series, y: y_series, m: m_series})
    except TypeError as e:
        print(f" [JET] Substitution failed: {e}")
        print(f" [JET] F_sr variables: {proto.variables()}")
        raise
        return

    # Expand the substituted expression
    expr = expr.expand()

    # compute Taylor coefficients via derivatives at t=0
    try:
        c0 = expr.subs({t: 0}).simplify()
        c1 = expr.diff(t).subs({t: 0}).simplify()
        c2 = (expr.diff(t, 2).subs({t: 0}) / 2).simplify()
    except Exception as e:
        print(f" [JET] Taylor expansion failed: {e}")
        raise
        return

    eqs = []
    if c0 != 0:
        eqs.append(c0 == 0)
    if c1 != 0:
        eqs.append(c1 == 0)
    if c2 != 0:
        eqs.append(c2 == 0)

    if not eqs:
        print(" [JET] no local equations found")
        return

    try:
        sol = solve(eqs, [a2, b1, b2], solution_dict=True)
    except Exception as e:
        print(f" [JET] solve failed: {e}")
        raise
        return

    if not sol:
        print(" [JET] obstruction: no local lift at this point")
        return

    first = sol[0] if isinstance(sol, (list, tuple)) and sol else sol
    if isinstance(first, dict) and 'a2' in first:
        print(" [JET] a2 =", first['a2'])
    else:
        print(" [JET] a2 free (curvature unconstrained by double-root)")

# Utility: Print consensus effectiveness
@PROFILE
def build_multiple_fibrations(fx_PR, pts_xy, num_fibrations, max_steps=3,
                               base_seed=SEED_INT, verbose=DEBUG):
    """
    Build multiple independent fibrations with different anchor points.
    Each fibration should find the same rational points (conjecturally).

    The number of anchor points is automatically determined to maximize
    diversity while maintaining degree drop constraints.
    """
    if not USE_ANCHOR_POINTS:
        raise RuntimeError("build_multiple_fibrations requires USE_ANCHOR_POINTS=True")

    # Calculate optimal number of anchor points
    # For a degree n curve reducing to degree 4, we have multiple steps
    # At each step, we can vary anchor points to create diversity
    from sage.all import PolynomialRing, QQ
    R_x = PolynomialRing(QQ, 'x')
    x = R_x.gen()
    #n = int(fx_PR.degree(x))
    n = int(fx_PR.degree())

    # For the first step (n -> n-1), degQ ~ (n-2)/2
    # We need degQ+1 constraints total, have 1 base point
    # So we have degQ degrees of freedom to distribute between tangency and anchors
    initial_degQ = (n - 2) // 2 if (n - 2) % 2 == 0 else (n - 1) // 2
    available_dof = initial_degQ  # After using 1 base point

    # Use 50% of DOF as anchor points (rest for tangency)
    # This balances diversity with computational stability
    num_anchors = available_dof

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-FIBRATION CONSENSUS MODE")
        print(f"Building {num_fibrations} independent fibrations")
        print(f"Using {num_anchors} anchor points per fibration")
        print(f"{'='*70}")

    # Temporarily override the global settings
    global NUM_ANCHOR_POINTS
    original_num_anchors = NUM_ANCHOR_POINTS
    NUM_ANCHOR_POINTS = num_anchors

    try:
        fibrations = []
        for k in range(num_fibrations):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Building Fibration {k+1}/{num_fibrations} (seed={base_seed + k})")
                print(f"{'='*70}")

            tower = iterate_tower(
                fx_PR=fx_PR,
                pts_xy=pts_xy,
                max_steps=max_steps,
                seed_int=base_seed + k,
                verbose=verbose,
                use_anchor_points=USE_ANCHOR_POINTS
            )

            import copy

            # inside build_multiple_fibrations(...) loop, replace the append with:
            # old:
            # fibrations.append({
            #     'tower': tower,
            #     'seed': base_seed + k,
            #     'id': k
            # })
            #
            # new (deepcopy snapshot + debug id print):
            _fib_snapshot = {
                'tower': copy.deepcopy(tower),
                'seed': base_seed + k,
                'id': k
            }
            fibrations.append(_fib_snapshot)

            # quick debug check (temporary; remove once verified)
            try:
                print("DEBUG appended tower id:", id(_fib_snapshot['tower']), "seed:", _fib_snapshot['seed'])
                # also show Q_i repr of last tower step if available (safe to call repr)
                if isinstance(_fib_snapshot['tower'], (list, tuple)) and _fib_snapshot['tower']:
                    last_step = _fib_snapshot['tower'][-1]
                    print("DEBUG last step Q_i repr:", repr(last_step.get('Q_i', '<no Q_i>')))
            except Exception:
                raise

        return fibrations
    finally:
        # Restore original setting
        NUM_ANCHOR_POINTS = original_num_anchors

if __name__ == '__main__':
    pass
    #main() # only for testing

def print_consensus_effectiveness(consensus_stats, cumulative_stats):
    """
    Print how effective the consensus filter was at reducing junk.
    """
    print(f"\n{'='*70}")
    print("CONSENSUS FILTER EFFECTIVENESS")
    print(f"{'='*70}")

    cs = consensus_stats
    print(f"\nResidues filtered: {cs['total_before'] - cs['total_after']:,} / {cs['total_before']:,}")
    print(f"Reduction: {100*cs['reduction_ratio']:.1f}%")

    # Compare to rationality test results
    total_tests = cumulative_stats.counters.get('rationality_tests_total', 0)
    successes = cumulative_stats.counters.get('rationality_tests_success', 0)

    if total_tests > 0:
        hit_rate = successes / total_tests
        print(f"\nRationality tests:")
        print(f"  Total: {total_tests:,}")
        print(f"  Successes: {successes:,}")
        print(f"  Hit rate: {100*hit_rate:.2f}%")

        # Estimate how many tests we saved
        tests_saved = int(cs['total_before'] - cs['total_after'])

        # Calculate average time per test from search phase
        search_time = cumulative_stats.phase_times.get('search_subsets_and_check', 0)
        if total_tests > 0:
            time_per_test = search_time / total_tests
            time_saved_est = tests_saved * time_per_test

            print(f"\nEstimated tests saved: ~{tests_saved:,}")
            print(f"Estimated time saved: ~{time_saved_est:.1f}s")

@PROFILE
def compute_consensus_residues(precomputed_residues_list, prime_pool,
                                consensus_threshold=CONSENSUS_THRESHOLD,
                                debug=DEBUG):
    """
    Compute consensus residues across multiple fibrations.
    A residue is kept if it appears in >= consensus_threshold fraction of
    *participating* fibrations (those that successfully computed residues for that prime).

    Args:
        precomputed_residues_list: List of precomputed_residues dicts (one per fibration)
        prime_pool: List of primes
        consensus_threshold: Minimum fraction of participating fibrations that must agree

    Returns:
        consensus_residues: Dict in same format as precomputed_residues
        stats: Dict with filtering statistics
    """
    from collections import defaultdict, Counter

    num_fibrations = len(precomputed_residues_list)

    if debug:
        print(f"\n{'='*70}")
        print(f"CONSENSUS FILTER: {num_fibrations} fibrations, threshold={consensus_threshold:.1%}")
        print(f"Policy: Primes with no data in a fibration abstain from voting.")
        print(f"{'='*70}")

    # 1. Determine participation per prime
    # participating_counts[p] = number of fibrations that have non-empty data for p
    participating_counts = defaultdict(int)

    for precomp in precomputed_residues_list:
        for p in prime_pool:
            # Check if p exists and has any vectors/residues
            if p in precomp and precomp[p]:
                participating_counts[p] += 1

    # 2. Track votes
    # residue_votes[(p, v_tuple, rhs_idx)][residue] = count
    residue_votes = defaultdict(Counter)

    # Track max RHS index per (p, v_tuple) to properly initialize lists later
    max_rhs_indices = defaultdict(int)

    for precomp in precomputed_residues_list:
        for p in prime_pool:
            if p not in precomp or not precomp[p]:
                continue

            mapping = precomp[p]
            for v_tuple, rhs_lists in mapping.items():
                # Track max rhs index
                if len(rhs_lists) - 1 > max_rhs_indices[(p, v_tuple)]:
                    max_rhs_indices[(p, v_tuple)] = len(rhs_lists) - 1

                for rhs_idx, residue_set in enumerate(rhs_lists):
                    key = (p, v_tuple, rhs_idx)
                    for r in residue_set:
                        if isinstance(r, int):
                            residue_votes[key][r] += 1

    # 3. Filter
    consensus_residues = {}
    stats = {
        'total_before': 0,
        'total_after': 0,
        'per_prime_before': {},
        'per_prime_after': {},
        'reduction_ratio': 0.0,
        'participation': {}
    }

    # Helper to organize keys by prime for efficiency
    keys_by_prime = defaultdict(list)
    for k in residue_votes.keys():
        keys_by_prime[k[0]].append(k)

    for p in prime_pool:
        n_participating = participating_counts[p]
        stats['participation'][p] = n_participating

        if n_participating == 0:
            continue

        # Calculate votes needed for THIS prime
        # Use ceil to ensure we don't accept 0 votes, and strictness matches intent
        # e.g. 0.8 * 2 = 1.6 -> 2. 0.8 * 1 = 0.8 -> 1.
        min_votes_needed = int(math.ceil(consensus_threshold * n_participating))
        # Ensure at least 1 vote is needed if anyone participated
        min_votes_needed = max(1, min_votes_needed)

        consensus_residues[p] = {}

        prime_before = 0
        prime_after = 0

        p_keys = keys_by_prime[p]

        for key in p_keys:
            _, v_tuple, rhs_idx = key
            vote_counter = residue_votes[key]

            original_set = set(vote_counter.keys())
            consensus_set = {r for r, c in vote_counter.items() if c >= min_votes_needed}

            prime_before += len(original_set)
            prime_after += len(consensus_set)

            if consensus_set:
                if v_tuple not in consensus_residues[p]:
                    # Initialize list with empty sets up to max needed
                    needed = max_rhs_indices[(p, v_tuple)] + 1
                    consensus_residues[p][v_tuple] = [set() for _ in range(needed)]

                # Just in case logic implies we need to extend (should be covered by max_rhs_indices)
                while len(consensus_residues[p][v_tuple]) <= rhs_idx:
                    consensus_residues[p][v_tuple].append(set())

                consensus_residues[p][v_tuple][rhs_idx] = consensus_set

        stats['per_prime_before'][p] = prime_before
        stats['per_prime_after'][p] = prime_after
        stats['total_before'] += prime_before
        stats['total_after'] += prime_after

    if stats['total_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_after'] / stats['total_before'])

    if debug:
        print(f"\nConsensus Filter Results:")
        print(f"  Total residues before: {stats['total_before']:,}")
        print(f"  Total residues after:  {stats['total_after']:,}")
        print(f"  Filtered out: {stats['total_before'] - stats['total_after']:,} "
              f"({100*stats['reduction_ratio']:.1f}%)")

        # Show per-prime breakdown for top primes
        sorted_primes = sorted(stats['per_prime_before'].items(),
                              key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 primes by original residue count:")
        for p, before in sorted_primes:
            after = stats['per_prime_after'].get(p, 0)
            part = stats['participation'].get(p, 0)
            reduction = 1.0 - (after / before) if before > 0 else 0.0
            print(f"  p={p}: {before} -> {after} ({100*reduction:.1f}% filtered) [Participating: {part}/{num_fibrations}]")

    return consensus_residues, stats

# Replace measure_poly_complexity with this more robust geometry scorer.
# Uses Sage objects but written in plain Python style.

    # 0. Prepare RHS (constant terms) and Matrix rows
    # Since equations are linear, eq = c0*

# Compute k-th Hasse derivative of an SR / polynomial-like `expr` wrt symbol `x_sym`.
# Works by expanding coefficients in x_sym and applying binomial(i, k).

# Dispatcher: return the k-th derivative-like object appropriate for the arithmetic mode.

# Convenience: returns SR equality constraint "kth_deriv(expr) at pt == 0"

# Compute k-th Hasse derivative of an SR / polynomial-like `expr` wrt symbol `x_sym`.
# Works by expanding coefficients in x_sym and applying binomial(i, k).
# In FINITE_FIELD mode this returns a polynomial over GF(FINITE_FIELD) (no SR).

def _interpolate_Q_finite_field(curve_poly, pts, deg_Q, p):
    """
    Interpolates Q(x) of degree deg_Q over GF(p) such that Q(x_i) = y_i,
    using derivative constraints from the curve y^2 = f(x).
    """
    F = GF(p)
    R_ff = PolynomialRing(F, 'x')

    # Innovative purge: ensure curve_poly is in the finite field ring
    # This prevents SR from sneaking into the matrix or derivative calls.
    try:
        f_poly = R_ff(curve_poly)
    except Exception:
        # Fallback if SR is being particularly stubborn about coefficients
        x_sym = var('x')
        d = int(curve_poly.degree(x_sym))
        coeffs = [F(curve_poly.coefficient(x_sym, i)) for i in range(d + 1)]
        f_poly = R_ff(coeffs)
        raise

    ncoeff = deg_Q + 1
    rows = []
    rhs = []

    # Value constraints: Q(xi) = yi
    for xi, yi in pts:
        xi_f, yi_f = F(xi), F(yi)
        rows.append([xi_f**i for i in range(ncoeff)])
        rhs.append(yi_f)

    # Derivative constraints: 2 * yi * Q'(xi) = f'(xi)
    f_deriv = f_poly.derivative()
    deriv_rows = []
    for xi, yi in pts:
        xi_f, yi_f = F(xi), F(yi)
        coeff_2yi = 2 * yi_f
        # Row for Q'(x) coefficients: [0, 1, 2*x, 3*x^2, ...]
        row = [F(0)] + [coeff_2yi * i * (xi_f**(i-1)) for i in range(1, ncoeff)]
        deriv_rows.append(row)

    num_needed = ncoeff - len(rows)
    if num_needed > 0:
        if len(deriv_rows) < num_needed:
            raise RuntimeError(f"Underdetermined system: need {ncoeff} constraints, have {len(rows) + len(deriv_rows)}.")
        for i in range(num_needed):
            rows.append(deriv_rows[i])
            # Match the derivative constraint to the corresponding point's f'(xi)
            rhs.append(f_deriv(F(pts[i][0])))

    A = Matrix(F, rows)
    b = Matrix(F, [[r] for r in rhs])

    # Ensure square system for the solver
    if A.nrows() > ncoeff:
        A, b = A[:ncoeff, :], b[:ncoeff, :]

    try:
        sol_vec = A.solve_right(b)
    except Exception as e:
        raise RuntimeError(f"Finite-field interpolation linear solve failed: {e}")

    # Build Qx using the ring constructor directly to avoid symbolic sum dispatch
    coeffs = [sol_vec[i, 0] for i in range(ncoeff)]
    return R_ff(coeffs)

def lift_coeff(c):
    """
    Convert a coefficient in Frac(GF(p)[m]) to a symbolic expression in m.
    Scalars lift normally; nonconstant rational functions are mapped
    numerator/denominator -> SR.
    """
    # Finite field scalar
    if hasattr(c, "lift") and not hasattr(c, "numerator"):
        return c.lift()

    # Fraction field element in m
    if hasattr(c, "numerator") and hasattr(c, "denominator"):
        num = c.numerator()
        den = c.denominator()
        return SR(num) / SR(den)

    # Integer-like
    return SR(c)

# Replace iterate_tower with this bimodal version

# Replace measure_poly_complexity with this FINITE_FIELD-oriented function.
# Use this version only for FINITE_FIELD (field-native polynomials); it raises on misuse.

def _build_one_fibration_context(fx_SR, verbose):
    ctx = {}
    if FINITE_FIELD is not None:
        ctx['mode'] = 'FF'
        ctx['base_field'] = GF(FINITE_FIELD)
        ctx['PR_m'] = PolynomialRing(ctx['base_field'], 'm')
        ctx['m_poly'] = ctx['PR_m'].gen()
        ctx['Fm'] = ctx['PR_m'].fraction_field()
        ctx['R_xm'] = PolynomialRing(ctx['Fm'], 'x')
        ctx['x_var'] = ctx['R_xm'].gen()

        if verbose:
            print(f"[build_step] Working over GF({FINITE_FIELD})(m)")

        try:
            ctx['n'] = int(fx_SR.degree())
        except Exception:
            try:
                coeffs_fx = list(fx_SR.list())
                while coeffs_fx and int(coeffs_fx[-1]) == 0:
                    coeffs_fx.pop()
                ctx['n'] = max(0, len(coeffs_fx) - 1)
            except Exception:
                raise RuntimeError("Cannot determine degree of fx_SR in finite-field mode")
            raise
    else:
        ctx['mode'] = 'QQ'
        ctx['base_field'] = QQ
        ctx['PR_m'] = PolynomialRing(QQ, 'm')
        ctx['m_poly'] = ctx['PR_m'].gen()
        ctx['Fm'] = ctx['PR_m'].fraction_field()
        ctx['R_xm'] = PolynomialRing(ctx['Fm'], 'x')
        ctx['x_var'] = ctx['R_xm'].gen()
        ctx['xSR'] = SR.var('x')
        ctx['m_sym'] = SR.var('m')

        if verbose:
            print("[build_step] Working over QQ(m)")

        try:
            ctx['n'] = int(fx_SR.degree(ctx['xSR']))
        except Exception:
            try:
                ctx['n'] = int(fx_SR.degree())
            except Exception:
                raise RuntimeError("Cannot determine degree of fx_SR in QQ mode")
            raise
    return ctx

def _coerce_build_one_points(pts_x, ctx):
    if ctx['mode'] == 'FF':
        xs_chosen = []
        for item in pts_x:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                xs_chosen.append(ctx['base_field'](item[0]))
            else:
                xs_chosen.append(ctx['base_field'](item))
    else:
        xs_chosen = [QQ(xv) for xv in pts_x]

    if len(xs_chosen) == 0:
        raise RuntimeError("build_one_fibration_step: pts_x must contain at least one point.")
    return xs_chosen

def _determine_build_one_degQ(n, forced_Qpoly, ctx):
    max_degQ = (n - 1) // 2
    initial_degQ = choose_degQ(n)
    degQ = min(initial_degQ, max_degQ)

    if forced_Qpoly is not None:
        forced_deg = None
        if ctx['mode'] == 'FF':
            try:
                Rtmp = PolynomialRing(ctx['base_field'], 'x')
                forced_deg = int(Rtmp(forced_Qpoly).degree())
            except Exception:
                raise RuntimeError("Could not coerce forced_Qpoly into base_field polynomial in finite-field mode")
        else:
            try:
                forced_Q_SR = SR(forced_Qpoly)
                forced_deg = int(forced_Q_SR.degree(ctx['xSR']))
            except Exception:
                try:
                    Rtmp = PolynomialRing(ctx['base_field'], str(ctx['xSR']))
                    forced_deg = int(Rtmp(forced_Qpoly).degree())
                except Exception:
                    raise RuntimeError("Could not determine degree of forced_Qpoly")
        if forced_deg > max_degQ:
            raise RuntimeError(f"forced_Qpoly has degree {forced_deg} > allowed max {max_degQ}")
        degQ = forced_deg
    return degQ

def _coerce_build_one_f0(f0, ctx):
    if ctx['mode'] == 'FF':
        R_base_x = PolynomialRing(ctx['base_field'], 'x')
        try:
            coeffs_f0 = list(f0.list())
            coeffs_f0_mapped = [ctx['base_field'](int(c)) for c in coeffs_f0]
            f0_base = R_base_x(coeffs_f0_mapped)
        except Exception:
            try:
                deg_f0 = int(f0.degree())
                coeffs_f0 = [f0.coefficient(i) for i in range(deg_f0 + 1)]
                coeffs_f0_mapped = [ctx['base_field'](int(c)) for c in coeffs_f0]
                f0_base = R_base_x(coeffs_f0_mapped)
            except Exception:
                raise RuntimeError("Cannot coerce f0 into a polynomial over GF(p)")
        coeffs_f0_Fm = [ctx['Fm'](ctx['base_field'](c)) for c in f0_base.list()]
        return PolynomialRing(ctx['Fm'], 'x')(coeffs_f0_Fm)

    try:
        R_QQ = PolynomialRing(QQ, str(ctx['xSR']))
        coeffs_f0 = list(f0.list())
        return R_QQ(coeffs_f0)
    except Exception:
        raise

def _build_build_one_Qpoly(pts_x, xs_chosen, f0, degQ, forced_Qpoly,
                           force_Q_constraint_indices, seed_int, use_anchor_points,
                           ctx, f0_coerced):
    if forced_Qpoly is not None:
        if ctx['mode'] == 'FF':
            try:
                R_field = PolynomialRing(ctx['base_field'], 'x')
                Qpoly_field_base = R_field(forced_Qpoly)
            except Exception:
                raise RuntimeError("Cannot coerce forced_Qpoly into base_field polynomial in finite-field mode")
            Q_coeffs_base = list(Qpoly_field_base.list())
            Q_coeffs_Fm = [ctx['Fm'](ctx['base_field'](int(c))) for c in Q_coeffs_base]
            return PolynomialRing(ctx['Fm'], 'x')(Q_coeffs_Fm)
        try:
            return SR(forced_Qpoly)
        except Exception:
            R_field = PolynomialRing(QQ, 'x')
            return R_field(forced_Qpoly)

    if ctx['mode'] == 'FF':
        chosen_pts_xy = []
        for item in pts_x:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                xv, yv = item
                chosen_pts_xy.append((ctx['base_field'](xv), ctx['base_field'](yv)))
            else:
                raise RuntimeError("Finite-field mode requires (x,y) pairs in pts_x")

        if use_anchor_points:
            total_needed = degQ + 1
            base_pts_count = 1
            remaining_dof = total_needed - base_pts_count
            num_anchors_needed = min(NUM_ANCHOR_POINTS, max(0, remaining_dof))

            anchor_pts = []
            tries = 0
            while len(anchor_pts) < num_anchors_needed and tries < num_anchors_needed * 30:
                tries += 1
                x_anchor = ctx['base_field'].random_element()
                y_val = f0_coerced(x_anchor)
                try:
                    y_anchor = y_val.sqrt()
                    anchor_pts.append((x_anchor, y_anchor))
                except Exception:
                    continue

            if len(anchor_pts) < num_anchors_needed:
                raise RuntimeError(f"Could not generate {num_anchors_needed} anchor points in GF({FINITE_FIELD})")
        else:
            anchor_pts = []

        Qpoly_base = interpolate_Q_with_anchors(chosen_pts_xy, degQ, 'x', anchor_pts, seed_int=seed_int) if use_anchor_points else interpolate_Q_general(chosen_pts_xy, f0, degQ, 'x', seed_int=seed_int, force_constraint_indices=force_Q_constraint_indices)
        try:
            R_base_x = PolynomialRing(ctx['base_field'], 'x')
            Qpoly_field_base = R_base_x(Qpoly_base)
        except Exception:
            try:
                Q_coeffs = list(Qpoly_base.list())
                Q_coeffs_base = [ctx['base_field'](int(c)) for c in Q_coeffs]
                Qpoly_field_base = PolynomialRing(ctx['base_field'], 'x')(Q_coeffs_base)
            except Exception:
                raise RuntimeError("Interpolation returned something that could not be coerced to base_field[x]")
            raise

        Q_coeffs_base = list(Qpoly_field_base.list())
        Q_coeffs_Fm = [ctx['Fm'](ctx['base_field'](int(c))) for c in Q_coeffs_base]
        return PolynomialRing(ctx['Fm'], 'x')(Q_coeffs_Fm)

    chosen_pts_xy = []
    for xv in xs_chosen:
        y_val_expr = f0_coerced(xv) if f0_coerced is not None else SR(f0).subs({ctx['xSR']: SR(xv)})
        try:
            yi = sqrt(QQ(y_val_expr))
        except Exception:
            yi = SR(sqrt(y_val_expr))
            raise
        chosen_pts_xy.append((QQ(xv), yi))

    if use_anchor_points:
        total_needed = degQ + 1
        base_pts_count = 1
        remaining_dof = total_needed - base_pts_count
        num_anchors_needed = min(NUM_ANCHOR_POINTS, max(0, remaining_dof))
        anchor_pts = generate_anchor_points(num_anchors_needed, seed=seed_int, exclude_x=[QQ(xv) for xv in xs_chosen])
        return interpolate_Q_with_anchors(chosen_pts_xy, degQ, ctx['xSR'], anchor_pts, seed_int=seed_int)
    return interpolate_Q_general(chosen_pts_xy, f0, degQ, ctx['xSR'], seed_int=seed_int, force_constraint_indices=force_Q_constraint_indices)

def _build_one_eval_poly_at(poly_Rxm, xpoint, Fm):
    try:
        return poly_Rxm(xpoint)
    except Exception:
        deg = int(poly_Rxm.degree())
        s = Fm(0)
        for i in range(deg + 1):
            s += Fm(int(poly_Rxm.coefficient(i))) * (xpoint ** i)
        return s

def _build_one_tangency_points(xs_chosen, num_tangency_eqs, base_field, forced_tangency_seq=None, avoid_x=None):
    """Choose tangency points while optionally avoiding a preferred base point."""
    if num_tangency_eqs <= 0:
        return []

    xs_norm = [base_field(x) for x in xs_chosen]
    avoid_norm = base_field(avoid_x) if avoid_x is not None else None

    pool = [x for x in xs_norm if avoid_norm is None or x != avoid_norm]
    if not pool:
        pool = xs_norm[:]

    def normalize_candidate(x):
        x = base_field(x)
        if avoid_norm is not None and x == avoid_norm and len(pool) > 0:
            return random.choice(pool)
        return x

    if forced_tangency_seq is not None:
        seq = []
        for x in forced_tangency_seq:
            seq.append(normalize_candidate(x))
            if len(seq) >= num_tangency_eqs:
                break
        while len(seq) < num_tangency_eqs:
            seq.append(random.choice(pool))
        return seq[:num_tangency_eqs]

    return [random.choice(pool) for _ in range(num_tangency_eqs)]

def _solve_build_one_ff(fx_SR, Qpoly_field, xs_chosen, degQ, parameter_m,
                        forced_tangency_seq, use_anchor_points, verbose, ctx):
    Fm, R_xm, x_var = ctx['Fm'], ctx['R_xm'], ctx['x_var']
    base_field, n = ctx['base_field'], ctx['n']
    x1 = xs_chosen[0]

    try:
        fx_field = R_xm(fx_SR)
    except Exception:
        try:
            coeffs_fx = list(fx_SR.list())
            coeffs_fx_Fm = [Fm(base_field(int(c))) for c in coeffs_fx]
            fx_field = R_xm(coeffs_fx_Fm)
        except Exception:
            raise RuntimeError("Could not coerce fx_SR into R_xm polynomial in finite-field mode")

    try:
        Q_poly_Fm = R_xm(list(Qpoly_field.list()))
    except Exception:
        try:
            Q_poly_Fm = R_xm(Qpoly_field)
        except Exception:
            raise RuntimeError("Could not coerce Qpoly_field into R_xm")

    prod_Fm = R_xm(1)
    for xi in xs_chosen:
        prod_Fm *= (x_var - Fm(base_field(xi)))

    deg_prod = int(prod_Fm.degree())
    rest_deg = int(n - 1 - deg_prod)
    if rest_deg < 0:
        raise RuntimeError(f"rest polynomial degree would be negative: rest_deg={rest_deg}")

    num_unknowns = rest_deg + 1
    T_list = [(prod_Fm * x_var**i) for i in range(num_unknowns)]
    Q2 = (Q_poly_Fm**2)

    rows = []
    rhs = []

    if parameter_m is None:
        m_symbol = ctx['m_poly']
    else:
        if isinstance(parameter_m, str):
            m_symbol = ctx['PR_m'](parameter_m).gen()
        else:
            m_symbol = parameter_m

    # ---- REPLACE lines 1086-1121 with this block ----

    if RLINEAR:
        r_expr = Fm(base_field(x1)) - m_symbol
    else:
        # Quadratic rail: r(m) = x1 - m + m^2
        # c=1 keeps things field-native; tweak if a specific c is preferred.
        _c_ff = Fm(base_field(1))
        r_expr = Fm(base_field(x1)) - m_symbol + _c_ff * m_symbol**2

    fx_at_r = _build_one_eval_poly_at(fx_field, r_expr, Fm)
    q2_at_r = _build_one_eval_poly_at(Q2, r_expr, Fm)
    T_at_r = [_build_one_eval_poly_at(Ti, r_expr, Fm) for Ti in T_list]
    rows.append([T_at_r[i] for i in range(num_unknowns)])
    rhs.append(fx_at_r - q2_at_r)

    # FF-only: impose one fewer tangency condition at the base point.
    # QQ mode always adds the derivative row (double-root / full tangency).
    # FF mode skips it (simple-root condition only), freeing one slot that
    # the mixing equation absorbs below.
    if FINITE_FIELD is None:
        dT_at_r = [_build_one_eval_poly_at(Ti.derivative(x_var), r_expr, Fm) for Ti in T_list]
        dQ2_at_r = _build_one_eval_poly_at(Q2.derivative(x_var), r_expr, Fm)
        dfx_at_r = _build_one_eval_poly_at(fx_field.derivative(x_var), r_expr, Fm)
        rows.append([dT_at_r[i] for i in range(num_unknowns)])
        rhs.append(dfx_at_r - dQ2_at_r)

    num_base_rows = len(rows)          # 1 (FF) or 2 (QQ)
    unknowns_order = num_unknowns
    remaining = unknowns_order - num_base_rows

    if FINITE_FIELD is not None:
        # FF mode: 1 base row leaves (unknowns_order - 1) slots.
        # Reserve one for mixing; rest are tangency conditions.
        num_tangency_eqs = max(0, remaining - 1)
        use_mixing = (remaining >= 1)
    elif use_anchor_points:
        # QQ mode with anchors: same as before (reserve 1 slot for mixing).
        if remaining <= 0 and verbose:
            print("Warning: Not enough DOF for Q-mixing strategy, reverting to full tangency.")
        num_tangency_eqs = max(0, remaining - 1)
        use_mixing = (remaining >= 1)
    else:
        # QQ mode without anchors: fill entirely with tangency, no mixing.
        num_tangency_eqs = remaining
        use_mixing = False

    # ---- end replacement ----

    sel_points = _build_one_tangency_points(
        xs_chosen,
        num_tangency_eqs,
        base_field,
        forced_tangency_seq=forced_tangency_seq,
        avoid_x=x1,
    )

    tangency_counts = {x: 0 for x in xs_chosen}
    for xv in sel_points:
        tangency_counts[xv] += 1
        order = tangency_counts[xv]
        xpoint = Fm(base_field(xv))
        Ti_derivs_at_x = [_build_one_eval_poly_at(Ti.derivative(x_var, order), xpoint, Fm) for Ti in T_list]
        q2_deriv_at_x = _build_one_eval_poly_at(Q2.derivative(x_var, order), xpoint, Fm)
        fx_deriv_at_x = _build_one_eval_poly_at(fx_field.derivative(x_var, order), xpoint, Fm)
        rows.append([Ti_derivs_at_x[i] for i in range(num_unknowns)])
        rhs.append(fx_deriv_at_x - q2_deriv_at_x)

    if use_mixing:
        x1_int = int(base_field(x1))
        x_mix_num = 2 * x1_int + 3
        inv2 = Fm(1) / Fm(2)
        x_mix = Fm(base_field(x_mix_num)) * inv2
        xmix_pows = [x_mix**i for i in range(num_unknowns)]
        Q_at_mix = _build_one_eval_poly_at(Q_poly_Fm, x_mix, Fm)
        rows.append([xmix_pows[i] for i in range(num_unknowns)])
        rhs.append(Q_at_mix)

    if len(rows) != num_unknowns:
        raise RuntimeError(f"Equation/unknown mismatch: {len(rows)} equations vs {num_unknowns} unknowns")

    try:
        M = matrix(Fm, rows)
        b = vector(Fm, rhs)
        sol_vec = M.solve_right(b)
    except Exception as e:
        raise RuntimeError(f"Linear solve over Fm failed: {e}")

    rest_coeffs_Fm = [sol_vec[i] for i in range(num_unknowns)]
    rest_poly_Fm = R_xm(rest_coeffs_Fm)

    Q_poly_Fm = R_xm(list(Q_poly_Fm.list()))
    fibration_Fm = (Q_poly_Fm**2 + prod_Fm * rest_poly_Fm)

    deg_fib = int(fibration_Fm.degree())
    target_deg = n - 1
    if deg_fib != target_deg:
        raise RuntimeError(f"Degree drop failed: expected {target_deg}, got {deg_fib}")

    return {
        'f_i': fibration_Fm,
        'Q_i': Q_poly_Fm,
        'rest_poly': rest_poly_Fm,
        'f_i_SR': fibration_Fm,
        'Q_SR': Q_poly_Fm,
        'rest_SR': rest_poly_Fm,
        'r_expr': r_expr,
        'info': f"n={n} degProd={deg_prod} rest_deg={rest_deg} anchor_mode={use_anchor_points} num_anchors={NUM_ANCHOR_POINTS if use_anchor_points else 0} mixed={use_mixing} field={base_field}",
        'base_field': base_field,
        'Fm': Fm,
        'R_xm': R_xm,
    }

def _solve_build_one_qq(fx_SR, Qpoly_field, xs_chosen, degQ, f0, parameter_m,
                        forced_tangency_seq, use_anchor_points, verbose, ctx):
    xSR = ctx['xSR']
    prod1 = poly_prod_numeric(xs_chosen, xSR)
    deg_prod = int(prod1.degree(xSR))
    rest_deg = int(ctx['n'] - 1 - deg_prod)
    if rest_deg < 0:
        raise RuntimeError(f"rest polynomial degree would be negative: rest_deg={rest_deg}")

    rest_coeff_names = [f"b_rest_{i}" for i in range(rest_deg + 1)]
    rest_coeff_syms = [SR.var(name) for name in rest_coeff_names]
    rest_poly_SR = sum(rest_coeff_syms[i] * xSR**i for i in range(rest_deg + 1))

    Q_SR = SR(Qpoly_field) if not isinstance(Qpoly_field, type(SR(0))) else Qpoly_field
    prod1_SR = SR(prod1)
    fibration_SR = (Q_SR**2).expand() + (prod1_SR * rest_poly_SR).expand()
    diff_poly = (SR(fx_SR) - fibration_SR).expand()

    if parameter_m is None:
        m = SR.var('m')
    else:
        m = SR(parameter_m) if not isinstance(parameter_m, type(SR(parameter_m))) else parameter_m

    if RLINEAR:
        if FINITE_FIELD is None:
            r_expr = SR(xs_chosen[0]) - m
        else:
            r_expr = SR(int(xs_chosen[0])) - m
    else:
        # Quadratic rail: r(m) = x1 - m + RLINEAR_C * m^2
        x1_val = SR(xs_chosen[0]) if FINITE_FIELD is None else SR(int(xs_chosen[0]))
        r_expr = x1_val - m + SR(RLINEAR_C) * m**2

    eqs = [diff_poly.subs({xSR: r_expr}), kth_derivative(diff_poly, 1, xSR).subs({xSR: r_expr})]
    unknowns = rest_coeff_syms[:]

    num_tangency_eqs = len(unknowns) - 2 - 1 if use_anchor_points else len(unknowns) - 2
    if num_tangency_eqs < 0 or not use_anchor_points:
        if use_anchor_points:
            print("Warning: Not enough DOF for Q-mixing strategy, reverting to full tangency.")
        num_tangency_eqs = len(unknowns) - 2
        use_mixing = False
    else:
        use_mixing = True

    sel_points = _build_one_tangency_points(
        xs_chosen,
        num_tangency_eqs,
        QQ,
        forced_tangency_seq=forced_tangency_seq,
        avoid_x=xs_chosen[0],
    )

    tangency_counts = {x: 0 for x in xs_chosen}
    for xv in sel_points:
        tangency_counts[xv] += 1
        current_order = tangency_counts[xv]
        xv_sr = SR(xv) if FINITE_FIELD is None else SR(int(xv))
        eq_t = kth_derivative(diff_poly, current_order, xSR).subs({xSR: xv_sr}).expand()
        eqs.append(eq_t)

    if use_mixing:
        if FINITE_FIELD is not None:
            x1_int = int(xs_chosen[0])
            x_mix_num = 2 * x1_int + 3
            x_mix_sr = SR(x_mix_num) / SR(2)
        else:
            x_mix_num = 2 * int(xs_chosen[0]) + 3
            x_mix = QQ(x_mix_num) / QQ(2)
            x_mix_sr = SR(x_mix)

        val_Q = Q_SR.subs({xSR: x_mix_sr})
        val_R = rest_poly_SR.subs({xSR: x_mix_sr})
        eq_mix = (val_R - val_Q).expand()
        eqs.append(eq_mix)

    if len(eqs) != len(unknowns):
        raise RuntimeError(f"Equation/unknown mismatch: {len(eqs)} equations vs {len(unknowns)} unknowns")

    zero_sub = {u: 0 for u in unknowns}
    rhs_vec = []
    rows = []
    for eq in eqs:
        c_term = eq.subs(zero_sub)
        rhs_vec.append(-c_term)
        row = []
        for u in unknowns:
            try:
                coeff_u = eq.coefficient(u)
            except Exception:
                coeff_u = SR(eq).coefficient(u)
                raise
            row.append(coeff_u)
        rows.append(row)

    try:
        rows_SR = matrix(SR, rows)
        rhs_SR = vector(SR, rhs_vec)
        sol_vec = rows_SR.solve_right(rhs_SR)
        sol = {u: sol_vec[i] for i, u in enumerate(unknowns)}
    except Exception as e:
        raise RuntimeError(f"SR linear solve failed: {e}")

    rest_coeffs_Fm = []
    for s in rest_coeff_syms:
        rest_coeffs_Fm.append(sol[s])
    rest_poly_Fm = ctx['R_xm']([QQ(c) if hasattr(c, 'denominator') else c for c in rest_coeffs_Fm])

    fibration_Fm = SR(fx_SR) - (SR(Q_SR)**2 + SR(prod1) * rest_poly_SR)
    return {
        'f_i': fibration_Fm,
        'Q_i': Qpoly_field,
        'rest_poly': rest_poly_Fm,
        'f_i_SR': fibration_Fm,
        'Q_SR': Qpoly_field,
        'rest_SR': rest_poly_Fm,
        'r_expr': r_expr,
        'info': f"n={ctx['n']} degProd={deg_prod} rest_deg={rest_deg} anchor_mode={use_anchor_points} num_anchors={NUM_ANCHOR_POINTS if use_anchor_points else 0} mixed={use_mixing} field={ctx['base_field']}",
        'base_field': ctx['base_field'],
        'Fm': ctx['Fm'],
        'R_xm': ctx['R_xm'],
    }

@PROFILE
def build_one_fibration_step(fx_SR, f0, pts_x, g2, seed_int=SEED_INT,
                             verbose=False, forced_tangency_seq=None,
                             forced_Qpoly=None, force_Q_constraint_indices=None,
                             parameter_m=None, use_anchor_points=USE_ANCHOR_POINTS):
    """
    Field-aware version that avoids SR in finite-field mode.

    Returns field-native polynomials (over Fm = GF(p)(m) or QQ(m)) and
    symbolic SR copies only when FINITE_FIELD is None.

    NOTE: helper functions used by the original (choose_degQ, interpolate_*, etc.)
    are expected to behave sensibly in both modes or to return field-native polynomials.
    """
    random.seed(int(seed_int))

    ctx = _build_one_fibration_context(fx_SR, verbose)
    xs_chosen = _coerce_build_one_points(pts_x, ctx)
    degQ = _determine_build_one_degQ(ctx['n'], forced_Qpoly, ctx)
    f0_coerced = _coerce_build_one_f0(f0, ctx)
    Qpoly_field = _build_build_one_Qpoly(
        pts_x, xs_chosen, f0, degQ, forced_Qpoly,
        force_Q_constraint_indices, seed_int, use_anchor_points,
        ctx, f0_coerced
    )

    if ctx['mode'] == 'FF':
        return _solve_build_one_ff(
            fx_SR, Qpoly_field, xs_chosen, degQ, parameter_m,
            forced_tangency_seq, use_anchor_points, verbose, ctx
        )
    return _solve_build_one_qq(
        fx_SR, Qpoly_field, xs_chosen, degQ, f0, parameter_m,
        forced_tangency_seq, use_anchor_points, verbose, ctx
    )
def check_fibration_step(step, prev_fx=None, layer_index=None):
    L = "Layer[%s]" % (layer_index if layer_index is not None else "unknown")
    s = normalize_step(step)
    fx = s['fx']
    assert fx is not None, L + ": missing f_i in step. repr(step)=" + repr(s['raw'])
    r_expr = s['r_expr']

    ff_mode = (FINITE_FIELD is not None)

    # ------------------------------------------------------------
    # r_expr sanity (NO SR in finite field mode)
    # ------------------------------------------------------------
    if r_expr is not None:
        if ff_mode:
            # r_expr must be constant or a function of m only
            try:
                parent = r_expr.parent()
                gens = parent.gens() if hasattr(parent, "gens") else []
                gen_names = [str(g) for g in gens]
            except Exception:
                gen_names = []

            assert 'x' not in gen_names, (
                L + ": r_expr depends on x in finite-field mode. r_expr=" + repr(r_expr)
            )
        else:
            rv = [str(v) for v in SR(r_expr).variables()]
            assert 'x' not in rv, L + ": r_expr depends on x. r_expr=" + repr(r_expr)
            if len(rv) > 0:
                assert 'm' in rv, L + ": r_expr vars " + repr(rv) + " missing 'm'"

    # ------------------------------------------------------------
    # degree drop check
    # ------------------------------------------------------------
    if prev_fx is not None:
        if ff_mode:
            try:
                dprev = prev_fx.degree()
                dcur = fx.degree()
            except Exception:
                raise RuntimeError(L + ": unable to compute degrees in finite-field mode")

            assert dcur <= dprev - 1, (
                L + ": degree drop failed prev=%s cur=%s" %
                (repr(dprev), repr(dcur))
            )
        else:
            x = var('x')
            try:
                dprev = SR(prev_fx).degree(x)
                dcur = SR(fx).degree(x)
            except Exception:
                raise
            if dprev is not None and dcur is not None:
                assert dcur <= dprev - 1, (
                    L + ": degree drop failed prev=%s cur=%s" %
                    (repr(dprev), repr(dcur))
                )

    # ------------------------------------------------------------
    # optional double root check
    # ------------------------------------------------------------
    dr = s['double_root_x']
    if dr is not None:
        if ff_mode:
            # algebraic double root test: f(dr) = 0 and f'(dr) = 0
            try:
                v1 = fx(dr)
                dfx = fx.derivative()
                v2 = dfx(dr)
            except Exception as e:
                raise RuntimeError(L + ": double root check failed in finite-field mode: " + str(e))

            assert v1 == 0, L + ": x0 not root. fx(x0)=%s" % repr(v1)
            assert v2 == 0, L + ": x0 not double root. fx'(x0)=%s" % repr(v2)

        else:
            x = var('x')
            fx_sr = SR(fx)
            dfx = kth_derivative(fx_sr, 1, x)
            v1 = fx_sr.subs({x: dr})
            v2 = dfx.subs({x: dr})
            assert SR(v1).simplify() == 0, L + ": x0 not root. fx(x0)=%s" % repr(v1)
            assert SR(v2).simplify() == 0, L + ": x0 not double root. fx'(x0)=%s" % repr(v2)

    return True

# tower.sage - Refactored with fail-fast, diagnostics, and defensive assertions
# Numeric-first fibration tower builder (strict, exact QQ arithmetic)

# === Configuration ===
_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41]
_SAMPLE_M_VALUES = [-7, -3, -1, 0, 1, 2, 3, 7]
_NUM_RANDOM_M = 5
_MAX_DEGREE_PENALTY = 5.0
_WEIGHT_HEIGHT = 1.0
_WEIGHT_DEG = 1.2
_WEIGHT_DISC = 1.5
_WEIGHT_BADPRIME = 6.0
_WEIGHT_COLLISION = 8.0

# === Profiler stub ===
try:
    PROFILE = profile
except NameError:
    def profile(arg2):
        return arg2
    PROFILE = profile

# === Field mode diagnostics ===
def _report_mode():
    if FINITE_FIELD is not None:
        print(f"[MODE] Operating in FINITE FIELD mode over GF({FINITE_FIELD})")
        sys.stdout.flush()
    else:
        print("[MODE] Operating in RATIONAL (QQ/SR) mode")
        sys.stdout.flush()

# === Symbol utilities ===
def ensure_symbol(obj, name_hint):
    """Guarantee proper symbolic variable. Crashes if cannot make symbol."""
    assert obj is not None, f"ensure_symbol: received None (hint: {name_hint})"

    if isinstance(obj, str):
        return var(obj)

    try:
        _ = SR(obj)
    except Exception as e:
        raise AssertionError(f"ensure_symbol: cannot coerce {obj!r} to SR (hint: {name_hint}): {e}")

    try:
        nm = getattr(obj, 'name', None)
        if nm:
            return var(str(nm))
    except Exception:
        pass

    try:
        return var(str(obj))
    except Exception as e:
        raise AssertionError(f"ensure_symbol: failed creating symbol from {obj!r} (hint: {name_hint}): {e}")

def expr_variables(expr):
    """Return set of variable names in expression. Fail-fast on errors."""
    if expr is None:
        return set()

    try:
        syms = expr.variables()
        result = set([str(s) for s in syms])
        return result
    except Exception:
        try:
            s = SR(expr)
            syms = s.variables()
            result = set([str(sv) for sv in syms])
            return result
        except Exception as e:
            raise RuntimeError(f"expr_variables: cannot extract variables from {expr!r}: {e}")

# === Solution selection ===
@PROFILE
def require_single_solution(sol_list, context=""):
    """Ensure solver returned exactly one branch. Fail immediately otherwise."""
    assert isinstance(sol_list, (list, tuple)), \
        f"Solver output not list/tuple. Context: {context}. Output type: {type(sol_list)}"

    if len(sol_list) != 1:
        raise RuntimeError(
            f"Solver returned {len(sol_list)} branches (expected exactly 1).\n"
            f"Context: {context}\n"
            f"Solutions: {sol_list}"
        )

    return sol_list[0]

@PROFILE
def has_free_variables(expr):
    """Check if expression has free variables."""
    try:
        return len(expr.free_variables()) > 0
    except AttributeError:
        return False

# === Degree utilities ===
@PROFILE
def choose_degQ(n):
    """Choose degQ so 2*degQ is n-2 if possible, else n-1. Fail if impossible."""
    if (n - 2) % 2 == 0:
        result = (n - 2) // 2
        #print(f"[choose_degQ] n={n} -> degQ={(n-2)//2} (path: 2*degQ=n-2)")
        sys.stdout.flush()
        return result
    if (n - 1) % 2 == 0:
        result = (n - 1) // 2
        #print(f"[choose_degQ] n={n} -> degQ={(n-1)//2} (path: 2*degQ=n-1)")
        sys.stdout.flush()
        return result

    raise ValueError(f"choose_degQ: no integer degQ with 2*degQ in {{n-1,n-2}} for n={n}")

@PROFILE
def poly_prod_numeric(xs, x_sym):
    """Build (x - x1)(x - x2)... with numeric xi (QQ) substituted into SR."""
    assert xs, "poly_prod_numeric: empty xs list"

    prod = SR(1)
    for xi in xs:
        prod *= (x_sym - SR(QQ(xi)))

    result = prod.expand()

    # Diagnostic: verify degree
    expected_deg = len(xs)
    actual_deg = result.degree(x_sym)
    assert actual_deg == expected_deg, \
        f"poly_prod_numeric: degree mismatch. Expected {expected_deg}, got {actual_deg}"

    #print(f"[poly_prod_numeric] Built product of degree {actual_deg} from {len(xs)} points")
    sys.stdout.flush()

    return result

# === Derivative computation (field-aware) ===
def _int_log(x):
    """Safe integer log."""
    if x <= 1:
        return 0.0
    return math.log(float(x))

def kth_derivative(expr, k, x_sym):
    """
    Dispatcher: return k-th derivative appropriate for arithmetic mode.
    - FINITE_FIELD: Hasse derivative
    - QQ/SR: ordinary symbolic derivative
    """
    assert k >= 0, f"kth_derivative: negative order k={k}"

    if FINITE_FIELD is not None:
        return hasse_deriv_sr(expr, k, x_sym)
    else:
        result = SR(expr).diff(x_sym, k)
        return result

def jet_vanish_constraint(expr, order, x_sym, pt):
    """Convenience: kth_deriv(expr) at pt == 0"""
    return kth_derivative(expr, order, x_sym).subs({x_sym: pt}) == 0

def hasse_deriv_sr(expr, k, x_sym):
    """
    k-th Hasse derivative of expr wrt x_sym.

    Behavior:
      * If FINITE_FIELD is None -> symbolic SR mode
      * If FINITE_FIELD != None:
            - numeric expr -> compute in GF(p)[x]
            - symbolic coeffs -> formal symbolic Hasse derivative
    """
    # === Helper: detect symbolic coefficients ===
    def _has_symbolic_coeffs(e):
        e_sr = SR(e).expand()
        syms = e_sr.variables()
        return any(v != x_sym for v in syms)

    # === Characteristic-0 / SR mode ===
    if FINITE_FIELD is None or _has_symbolic_coeffs(expr):
        expr_sr = SR(expr).expand()

        try:
            deg = int(expr_sr.degree(x_sym))
        except Exception:
            coeffs = expr_sr.coefficients(x_sym)
            deg = max(p[1] for p in coeffs) if coeffs else 0

        out = SR(0)
        for i in range(deg + 1):
            ci = expr_sr.coefficient(x_sym, i)
            if i >= k:
                out += _int_binom(i, k) * ci * (x_sym ** (i - k))

        result = out.expand()

        # Diagnostic
        #print(f"[hasse_deriv_sr] SR mode: k={k}, input_deg={deg}, output_deg={result.degree(x_sym) if result != 0 else 0}")
        sys.stdout.flush()

        return result

    # === Finite-field numeric mode ===
    p = int(FINITE_FIELD)
    F = GF(p)
    varname = str(x_sym)
    R = PolynomialRing(F, varname)
    t = R.gen()

    try:
        poly = R(expr)
    except Exception as e:
        raise RuntimeError(f"hasse_deriv_sr: cannot coerce expr to GF({p})[{varname}]: {e}")

    coeffs = poly.list()
    out = R(0)
    for i, a in enumerate(coeffs):
        if i >= k:
            out += F(_int_binom(i, k)) * a * t**(i - k)

    # Diagnostic
    #print(f"[hasse_deriv_sr] GF({p}) mode: k={k}, input_deg={poly.degree()}, output_deg={out.degree()}")
    sys.stdout.flush()

    return out

# === Implicit derivative constraint builder ===
def compute_implicit_derivative_constraint(order, xi_val, yi_val, f_derivs, Q_derivs, x_sym):
    """
    Compute derivative constraint for interpolation.

    - QQ symbolic mode: implicit differentiation from y^2 = f(x)
    - FINITE_FIELD mode: Hasse-jet vanishing
    """
    assert order >= 0, f"compute_implicit_derivative_constraint: negative order {order}"

    # === FINITE_FIELD mode: Hasse-jet vanishing ===
    if FINITE_FIELD is not None:
        p = int(FINITE_FIELD)
        F = GF(p)

        try:
            xi_f = F(xi_val)
            yi_f = F(yi_val)
        except Exception as e:
            raise RuntimeError(f"compute_implicit_derivative: cannot coerce xi={xi_val}, yi={yi_val} to GF({p}): {e}")

        if order == 0:
            Q0 = Q_derivs[0]
            try:
                q_at_xi = Q0(xi_f)
            except Exception:
                q_at_xi = Q0.subs({x_sym: xi_f})

            constraint = (q_at_xi == yi_f)
            #print(f"[compute_implicit_constraint] FF mode: order=0 constraint at xi={xi_f}")
            sys.stdout.flush()
            return constraint

        # order >= 1: Hasse derivative vanishes at xi
        Qk = Q_derivs[order]
        try:
            qk_at_xi = Qk(xi_f)
        except Exception:
            qk_at_xi = Qk.subs({x_sym: xi_f})

        constraint = (qk_at_xi == 0)
        #print(f"[compute_implicit_constraint] FF mode: order={order} Hasse constraint at xi={xi_f}")
        sys.stdout.flush()
        return constraint

    # === QQ / SR mode: original implicit-diff approach ===
    xi_sr = xi_val
    yi_sr = yi_val

    if yi_sr == 0:
        #print(f"[compute_implicit_constraint] QQ mode: skipping order {order} at x={xi_sr} due to y=0")
        sys.stdout.flush()
        return None

    # Compute y derivatives
    y_derivs_at_point = compute_y_derivatives_at_point(xi_sr, yi_sr, f_derivs, order, x_sym)

    Q_nth_expr = Q_derivs[order].subs({x_sym: xi_sr})
    expected = y_derivs_at_point[order]

    constraint = (Q_nth_expr == expected)
    #print(f"[compute_implicit_constraint] QQ mode: order={order} at x={xi_sr}, y={yi_sr}")
    sys.stdout.flush()
    return constraint

def compute_y_derivatives_at_point(xi_sr, yi_sr, f_derivs, max_order, x_sym):
    """Compute y^(k) for k=0..max_order using implicit differentiation of y^2=f(x)."""
    assert max_order >= 0, f"compute_y_derivatives: negative max_order {max_order}"

    f_vals = [f_derivs[i].subs({x_sym: xi_sr}) for i in range(max_order + 1)]

    y_derivs = [yi_sr]

    for n in range(1, max_order + 1):
        if n == 1:
            y_n = f_vals[1] / (2 * yi_sr)
        else:
            cross_sum = 0
            for k in range(1, n):
                cross_sum += binomial(n, k) * y_derivs[k] * y_derivs[n - k]

            y_n = (f_vals[n] - cross_sum) / (2 * yi_sr)

        y_derivs.append(y_n)

    #print(f"[compute_y_derivatives] Computed {len(y_derivs)} derivatives at (x={xi_sr}, y={yi_sr})")
    sys.stdout.flush()

    return y_derivs

# === Interpolation (field-aware) ===
@PROFILE
def solve_for_Q(x_sym, y_sym, base_pts, degQ, constraints=None, derivative_constraints=None):
    """
    Interpolate Q(x) such that y_i = Q(x_i) for base points.
    Respects FINITE_FIELD mode.
    """
    assert degQ >= 0, f"solve_for_Q: negative degQ {degQ}"
    assert base_pts, "solve_for_Q: empty base_pts"

    #print(f"[solve_for_Q] Interpolating Q of degree {degQ} from {len(base_pts)} base points")
    sys.stdout.flush()

    if FINITE_FIELD is not None:
        target_field = GF(FINITE_FIELD)
        #print(f"[solve_for_Q] Using target field GF({FINITE_FIELD})")
    else:
        target_field = QQ
        #print("[solve_for_Q] Using target field QQ")
    sys.stdout.flush()

    coeffs_sym = [var(f'q{i}') for i in range(degQ + 1)]
    def Q_sym(val):
        return sum(coeffs_sym[i] * val**i for i in range(degQ + 1))

    chosen_eqs = [Q_sym(pt[0]) == pt[1] for pt in base_pts]

    if constraints:
        chosen_eqs += constraints
    if derivative_constraints:
        chosen_eqs += derivative_constraints

    #print(f"[solve_for_Q] Solving system of {len(chosen_eqs)} equations in {len(coeffs_sym)} unknowns")
    sys.stdout.flush()

    sols = solve(chosen_eqs, coeffs_sym, solution_dict=True)

    if not sols:
        raise RuntimeError(
            f"solve_for_Q: no solution found.\n"
            f"degQ={degQ}, base_pts={base_pts}\n"
            f"num_equations={len(chosen_eqs)}, num_unknowns={len(coeffs_sym)}"
        )

    sol = require_single_solution(sols, "solving for Q coefficients")

    R = PolynomialRing(target_field, str(x_sym))
    solved_coeffs = []

    for c in coeffs_sym:
        v = sol[c]
        try:
            solved_coeffs.append(target_field(v))
        except (TypeError, ValueError) as e:
            if FINITE_FIELD is None:
                raise RuntimeError(f"solve_for_Q: cannot coerce coefficient {v} to QQ: {e}")
            else:
                solved_coeffs.append(v)

    try:
        Qx = R(solved_coeffs)
    except (TypeError, ValueError):
        Qx = sum(SR(solved_coeffs[i]) * x_sym**i for i in range(len(solved_coeffs)))

    # Verify degree
    try:
        actual_deg = Qx.degree()
        assert actual_deg <= degQ, \
            f"solve_for_Q: result degree {actual_deg} exceeds requested {degQ}"
        #print(f"[solve_for_Q] Result polynomial has degree {actual_deg}")
    except Exception:
        print("[solve_for_Q] Could not verify degree (symbolic result)")
        sys.stdout.flush()

    return Qx, sol

# === General interpolation (bimodal) ===
@PROFILE
def interpolate_Q_general(pts_xy, f_expr, degQ, x_sym, seed_int=SEED_INT, force_constraint_indices=None):
    """
    Interpolate Q(x) of degree degQ from points pts_xy.
    - QQ/SR mode: symbolic solve
    - FINITE_FIELD mode: linear system over GF(p)

    Returns polynomial over correct field.
    """
    assert degQ >= 0, f"interpolate_Q_general: negative degQ {degQ}"
    assert pts_xy, "interpolate_Q_general: empty pts_xy"

    random.seed(int(seed_int))

    ncoeff = degQ + 1
    coeff_names = [f"q{i}" for i in range(ncoeff)]

    #print(f"[interpolate_Q_general] Starting interpolation: degQ={degQ}, num_pts={len(pts_xy)}, mode={'FF' if FINITE_FIELD else 'QQ'}")
    sys.stdout.flush()

    # === QQ / SR mode ===
    if FINITE_FIELD is None:
        coeffs_sym = [SR.var(name) for name in coeff_names]
        Q_poly_sym = sum(coeffs_sym[i] * (x_sym ** i) for i in range(ncoeff))

        max_order = min(5, degQ)

        f_derivs = {0: f_expr}
        Q_derivs = {0: Q_poly_sym}
        for order in range(1, max_order + 1):
            f_derivs[order] = kth_derivative(f_expr, order, x_sym)
            Q_derivs[order] = kth_derivative(Q_poly_sym, order, x_sym)

        mandatory_constraints = []
        derivative_pool = []

        for xi, yi in pts_xy:
            xi_sr = SR(xi)
            yi_sr = SR(yi)

            mandatory_constraints.append(Q_derivs[0].subs({x_sym: xi_sr}) == yi_sr)

            for order in range(1, max_order + 1):
                if order > degQ:
                    break
                constraint = compute_implicit_derivative_constraint(order, xi_sr, yi_sr, f_derivs, Q_derivs, x_sym)
                if constraint is not None:
                    derivative_pool.append(constraint)

        num_constraints_needed = ncoeff
        num_remaining_needed = num_constraints_needed - len(mandatory_constraints)

        assert num_remaining_needed >= 0, \
            f"interpolate_Q_general: too many mandatory constraints ({len(mandatory_constraints)}) for degQ={degQ}"
        assert len(derivative_pool) >= num_remaining_needed, \
            f"interpolate_Q_general: not enough derivative constraints ({len(derivative_pool)}) for degQ={degQ}"

        chosen_derivs = derivative_pool[:num_remaining_needed]
        all_constraints = mandatory_constraints + chosen_derivs

        #print(f"[interpolate_Q_general] QQ mode: {len(mandatory_constraints)} value + {len(chosen_derivs)} derivative constraints")
        sys.stdout.flush()

        sol_list = solve(all_constraints, coeffs_sym, solution_dict=True)

        if not sol_list:
            raise RuntimeError("interpolate_Q_general: no solution found during symbolic interpolation")

        sol_map = sol_list[0]
        solved_coeffs = [QQ(sol_map[name_sym]) for name_sym in coeffs_sym]

        R = PolynomialRing(QQ, 'x')
        Qx = R(solved_coeffs)

        # Dual computation check: verify at all input points
        for xi, yi in pts_xy:
            eval_result = Qx(xi)
            yi_qq = QQ(yi)
            assert eval_result == yi_qq, \
                f"interpolate_Q_general: verification failed at x={xi}: Q(x)={eval_result} != y={yi_qq}"

        #print(f"[interpolate_Q_general] QQ mode: verified Q at {len(pts_xy)} points")
        sys.stdout.flush()

        return Qx

    # === FINITE_FIELD mode ===
    p = int(FINITE_FIELD)
    F = GF(p)
    max_order = min(5, degQ)

    pts_f = [(F(xi), F(yi)) for xi, yi in pts_xy]

    rows = []
    rhs = []

    # Mandatory value constraints
    for xi, yi in pts_f:
        row = [F(0)] * ncoeff
        xi_pow = F(1)
        for i in range(ncoeff):
            row[i] = xi_pow
            xi_pow = xi_pow * xi
        rows.append(row)
        rhs.append(yi)

    # Derivative constraints pool
    deriv_rows = []
    for xi, yi in pts_f:
        for k in range(1, max_order + 1):
            if k > degQ:
                break
            row = [F(0)] * ncoeff
            xi_pow = F(1)
            for j in range(0, ncoeff - k):
                i = j + k
                b = F(_int_binom(i, k))
                row[i] = b * xi_pow
                xi_pow = xi_pow * xi
            deriv_rows.append(row)

    num_mand = len(rows)
    num_needed = ncoeff - num_mand

    assert num_needed >= 0, \
        f"interpolate_Q_general: too many mandatory constraints for degQ={degQ}"
    assert len(deriv_rows) >= num_needed, \
        f"interpolate_Q_general: not enough derivative constraints"

    for i in range(num_needed):
        rows.append(deriv_rows[i])
        rhs.append(F(0))

    #print(f"[interpolate_Q_general] FF mode: {num_mand} value + {num_needed} Hasse constraints")
    sys.stdout.flush()

    A = Matrix(GF(p), rows)
    b = Matrix(GF(p), [[r] for r in rhs])

    assert A.nrows() == A.ncols() == ncoeff, \
        f"interpolate_Q_general: matrix not square ({A.nrows()}x{A.ncols()})"

    try:
        sol_vec = A.solve_right(b)
    except Exception as e:
        raise RuntimeError(f"interpolate_Q_general: FF linear solve failed: {e}")

    gf_solved_coeffs = [F(sol_vec[i, 0]) for i in range(ncoeff)]
    plain_int_coeffs = [int(c) for c in gf_solved_coeffs]

    R = PolynomialRing(GF(p), 'x')
    Qx = R(plain_int_coeffs)

    # Dual computation check: verify at all input points
    for xi_f, yi_f in pts_f:
        eval_result = Qx(xi_f)
        assert eval_result == yi_f, \
            f"interpolate_Q_general: FF verification failed at x={xi_f}: Q(x)={eval_result} != y={yi_f}"

    #print(f"[interpolate_Q_general] FF mode: verified Q at {len(pts_f)} points")
    sys.stdout.flush()

    return Qx

# === Anchor point generation ===
def generate_anchor_points(num_points, seed=SEED_INT, exclude_x=None):
    """Generate anchor points with small denominators to minimize blowup."""
    assert num_points >= 0, f"generate_anchor_points: negative num_points {num_points}"

    random.seed(int(seed))
    anchor_pts = []

    used_x = set() if exclude_x is None else set(QQ(x) for x in exclude_x)

    allowed_denoms = [2, 3, 5, 7, 11, 13] + (PRIME_POOL[:-60] if len(PRIME_POOL) > 60 else PRIME_POOL[:-10])

    max_attempts = 100
    attempts = 0

    print(f"[generate_anchor_points] Generating {num_points} anchor points (seed={seed})")
    sys.stdout.flush()

    while len(anchor_pts) < num_points and attempts < max_attempts:
        attempts += 1

        num_x = random.randint(-10, 10)
        den_x = random.choice(allowed_denoms)
        x_val = QQ(num_x) / QQ(den_x)

        if x_val in used_x:
            continue

        num_y = random.randint(-10, 10)
        den_y = random.choice(allowed_denoms)
        y_val = QQ(num_y) / QQ(den_y)

        anchor_pts.append((x_val, y_val))
        used_x.add(x_val)

    if len(anchor_pts) < num_points:
        raise RuntimeError(f"generate_anchor_points: could not generate {num_points} unique points after {attempts} attempts")

    print(f"[generate_anchor_points] Generated {len(anchor_pts)} points in {attempts} attempts")
    sys.stdout.flush()

    return anchor_pts

def interpolate_Q_with_anchors(base_pts, degQ, x_sym, anchor_pts, seed_int=SEED_INT):
    """Compute Q(x) using base + anchor points (no tangency)."""
    assert degQ >= 0, f"interpolate_Q_with_anchors: negative degQ {degQ}"

    all_pts = list(base_pts) + list(anchor_pts)

    expected_num = degQ + 1
    if len(all_pts) != expected_num:
        raise RuntimeError(
            f"interpolate_Q_with_anchors: need exactly {expected_num} points for degree {degQ} Q, "
            f"but have {len(all_pts)} (base: {len(base_pts)}, anchors: {len(anchor_pts)})"
        )

    xs = [QQ(pt[0]) for pt in all_pts]
    ys = [QQ(pt[1]) for pt in all_pts]

    if len(set(xs)) != len(xs):
        raise RuntimeError(f"interpolate_Q_with_anchors: duplicate x-coordinates: {xs}")

    #print(f"[interpolate_Q_with_anchors] Lagrange interpolation with {len(all_pts)} points")
    sys.stdout.flush()

    R = PolynomialRing(QQ, str(x_sym))

    Qx = R(0)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        Li = R(1)
        for j, xj in enumerate(xs):
            if i != j:
                Li *= (R.gen() - xj) / (xi - xj)
        Qx += yi * Li

    # Dual check: verify at all points
    for xi, yi in zip(xs, ys):
        eval_result = Qx(xi)
        assert eval_result == yi, \
            f"interpolate_Q_with_anchors: verification failed at x={xi}: Q(x)={eval_result} != y={yi}"

    #print(f"[interpolate_Q_with_anchors] Verified Lagrange Q at {len(all_pts)} points")
    sys.stdout.flush()

    return Qx

def measure_poly_complexity(expr_ff):
    """
    FINITE_FIELD-oriented complexity score. Lower is better.
    Raises on unexpected conditions.
    """
    assert expr_ff is not None, "measure_poly_complexity: expr_ff is None"

    if FINITE_FIELD is None:
        raise RuntimeError("measure_poly_complexity: called finite-field scorer while FINITE_FIELD is None")

    assert hasattr(expr_ff, "parent"), "measure_poly_complexity: expr_ff must have .parent() method"

    R = expr_ff.parent()
    base = R.base_ring()

    try:
        coeffs = expr_ff.coefficients(sparse=False)
    except Exception:
        try:
            coeffs = list(expr_ff.list())
        except Exception as e:
            raise RuntimeError(f"measure_poly_complexity: cannot extract coefficients: {e}")

    if not coeffs:
        coeffs = [base.zero()]

    height_score = 0.0
    for c in coeffs:
        if c == 0:
            continue
        if hasattr(c, "numerator") and hasattr(c, "denominator"):
            num = c.numerator()
            den = c.denominator()
            deg_num = num.degree() if hasattr(num, "degree") else 0
            deg_den = den.degree() if hasattr(den, "degree") else 0
            height_score += 1.0 + 0.3 * (deg_num + deg_den)
        else:
            height_score += 0.5

    try:
        deg_x = expr_ff.degree()
    except Exception as e:
        raise RuntimeError(f"measure_poly_complexity: failed to compute degree: {e}")

    degree_penalty = _int_log(1 + int(deg_x))

    collision_penalty = 0.0
    try:
        if deg_x > 1:
            deriv = expr_ff.derivative()
            if not deriv.is_zero():
                g = expr_ff.gcd(deriv)
                if g.degree() > 0:
                    collision_penalty += 1.0
    except Exception as e:
        raise RuntimeError(f"measure_poly_complexity: gcd/derivative check failed: {e}")

    bad_denominator_penalty = 0.0
    for c in coeffs:
        if hasattr(c, "denominator"):
            den = c.denominator()
            try:
                if hasattr(den, "is_zero") and den.is_zero():
                    bad_denominator_penalty += 1.0
            except Exception:
                raise RuntimeError("measure_poly_complexity: failed inspecting denominator")

    total = (
        _WEIGHT_HEIGHT * height_score +
        _WEIGHT_DEG * degree_penalty +
        _WEIGHT_COLLISION * collision_penalty +
        _WEIGHT_BADPRIME * bad_denominator_penalty
    )

    #print(f"[measure_poly_complexity] Score={total:.2f} (height={height_score:.2f}, deg={degree_penalty:.2f}, collision={collision_penalty:.2f}, bad_denom={bad_denominator_penalty:.2f})")
    sys.stdout.flush()

    return float(total)

# === Tower-level verification ===
@PROFILE
def verify_tower_consistency(tower):
    """
    Verify all tower steps use parameter m consistently.
    Fail immediately on inconsistency.
    """
    assert tower, "verify_tower_consistency: empty tower"

    ff_mode = (FINITE_FIELD is not None)

    print(f"[verify_tower_consistency] Checking {len(tower)} layers (mode={'FF' if ff_mode else 'QQ'})")
    sys.stdout.flush()

    for i, step in enumerate(tower):
        assert 'r_expr' in step, f"Layer {i}: missing r_expr in step dict"
        r = step['r_expr']
        assert r is not None, f"Layer {i}: r_expr is None"

        if ff_mode:
            # r_expr must be in Fm (fraction field over GF(p)[m])
            try:
                parent = r.parent()
                gens = parent.gens() if hasattr(parent, "gens") else []
                gen_names = [str(g) for g in gens]
            except Exception as e:
                raise RuntimeError(f"Layer {i}: cannot inspect r_expr parent: {e}")

            assert 'x' not in gen_names, \
                f"Layer {i}: r_expr depends on x in FF mode: {r}"

            if gen_names:
                assert 'm' in gen_names, \
                    f"Layer {i}: r_expr vars {gen_names} missing 'm'"

        else:
            # QQ/SR mode: check symbolic variables
            try:
                rv = expr_variables(r)
            except Exception as e:
                raise RuntimeError(f"Layer {i}: cannot extract r_expr variables: {e}")

            assert 'x' not in rv, \
                f"Layer {i}: r_expr depends on x: vars={rv}, r={r}"

            if rv:
                assert 'm' in rv, \
                    f"Layer {i}: r_expr vars {rv} missing 'm'"

    print(f"[verify_tower_consistency] ✓ All {len(tower)} layers consistent")
    sys.stdout.flush()

    return True

@PROFILE
def verify_y2_consistency_on_rail(tower, x1, m_vals):
    """
    Verify y²ᵢ = y²ᵢ₊₁ along x = x₁ - m for consecutive layers.
    Tests at specific m values.
    """
    assert tower, "verify_y2_consistency: empty tower"
    assert len(tower) > 1, "verify_y2_consistency: need at least 2 layers to check consistency"
    assert m_vals, "verify_y2_consistency: empty m_vals list"

    ff_mode = (FINITE_FIELD is not None)

    print(f"[verify_y2_consistency] Checking {len(tower)-1} layer transitions at {len(m_vals)} m-values")
    sys.stdout.flush()

    if ff_mode:
        p = int(FINITE_FIELD)
        F = GF(p)
        x1_f = F(x1)
        m_vals_f = [F(m) for m in m_vals]

        for i in range(len(tower) - 1):
            f_i = tower[i]['f_i']
            f_i_plus_1 = tower[i+1]['f_i']

            assert hasattr(f_i, 'parent'), f"Layer {i}: f_i has no parent method"
            assert hasattr(f_i_plus_1, 'parent'), f"Layer {i+1}: f_i has no parent method"

            for m_val in m_vals_f:
                # Rail: x = x1 - m
                x_val = x1_f - m_val

                try:
                    y2_i = f_i(x_val)
                    y2_i_plus_1 = f_i_plus_1(x_val)
                except Exception as e:
                    raise RuntimeError(
                        f"Layer {i}->{i+1}: evaluation failed at x={x_val}, m={m_val}: {e}"
                    )

                diff = y2_i - y2_i_plus_1

                assert diff == 0, \
                    f"Layer {i}->{i+1}: y² mismatch at m={m_val}\n" \
                    f"  x={x_val}\n" \
                    f"  y²_{i}={y2_i}\n" \
                    f"  y²_{i+1}={y2_i_plus_1}\n" \
                    f"  diff={diff}"

    else:
        # QQ/SR mode
        x, m = var('x m')
        x1_sr = SR(x1)
        r_expr = x1_sr - m

        for i in range(len(tower) - 1):
            # Force SR conversion
            try:
                f_i = SR(tower[i]['f_i'])
                f_i_plus_1 = SR(tower[i+1]['f_i'])
            except Exception as e:
                raise RuntimeError(f"Layer {i}: cannot convert f_i to SR: {e}")

            for m_val in m_vals:
                m_val_qq = QQ(m_val)

                # Evaluate along rail x = x₁ - m
                try:
                    y2_i = f_i.subs({x: r_expr}).subs({m: m_val_qq})
                    y2_i_plus_1 = f_i_plus_1.subs({x: r_expr}).subs({m: m_val_qq})
                except Exception as e:
                    raise RuntimeError(
                        f"Layer {i}->{i+1}: substitution failed at m={m_val_qq}: {e}"
                    )

                try:
                    diff = (y2_i - y2_i_plus_1).expand().simplify()
                except Exception as e:
                    raise RuntimeError(
                        f"Layer {i}->{i+1}: simplification failed at m={m_val_qq}: {e}"
                    )

                assert diff == 0, \
                    f"Layer {i}->{i+1}: y² mismatch at m={m_val_qq}\n" \
                    f"  rail: x={r_expr}\n" \
                    f"  y²_{i}={y2_i}\n" \
                    f"  y²_{i+1}={y2_i_plus_1}\n" \
                    f"  diff={diff}"

    print(f"[verify_y2_consistency] ✓ All layer transitions consistent")
    sys.stdout.flush()

# === Entry point ===
@PROFILE
def main():
    """Main execution function with comprehensive diagnostics."""
    return # idk why claude put all this stuff down there
    _report_mode()

    print("="*70)
    print("TOWER.SAGE - Fibration Tower Builder")
    print("="*70)
    sys.stdout.flush()

    seed_int = 0

    # Test curve: y² = x⁶ + 4x⁵ - 2x⁴ - 18x³ + x² + 38x + 25
    COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(-2), QQ(-18), QQ(1), QQ(38), QQ(25)]
    DATA_PTS_GENUS2 = [(QQ(0), QQ(5))]

    PR = PolynomialRing(QQ, 'x')
    x = PR.gen()

    # Build polynomial f(x)
    fx_PR = sum(c * x**e for e, c in reversed(list(enumerate(reversed(COEFFS_GENUS2)))))

    # Verify initial polynomial
    assert fx_PR.degree() == 6, f"Expected degree 6, got {fx_PR.degree()}"
    print(f"Initial curve: y² = {fx_PR}")
    print(f"Base point: {DATA_PTS_GENUS2[0]}")

    # Verify base point is on curve
    x0, y0 = DATA_PTS_GENUS2[0]
    y0_squared = y0**2
    f_at_x0 = fx_PR(x0)
    assert y0_squared == f_at_x0, \
        f"Base point not on curve: y₀²={y0_squared}, f(x₀)={f_at_x0}"

    print(f"✓ Base point verified on curve")
    sys.stdout.flush()

    # Build tower
    print("\n" + "="*70)
    print("STARTING TOWER CONSTRUCTION")
    print("="*70)
    sys.stdout.flush()

    try:
        tower = iterate_tower(
            fx_PR=fx_PR,
            pts_xy=DATA_PTS_GENUS2[:1],
            max_steps=2,
            seed_int=seed_int,
            verbose=True,
            use_anchor_points=USE_ANCHOR_POINTS
        )
    except Exception as e:
        print("\n" + "="*70)
        print("TOWER CONSTRUCTION FAILED")
        print("="*70)
        print(f"Error: {e}")
        sys.stdout.flush()
        raise

    # Display results
    print("\n" + "="*70)
    print("TOWER CONSTRUCTION COMPLETE")
    print("="*70)

    for i, step in enumerate(tower):
        print(f"\n--- Layer {i+1} ---")
        print(f"Info: {step['info']}")
        print(f"Q(x): {step['Q_i']}")
        print(f"r(m): {step['r_expr']}")
        print(f"f_{i+1}(x,m): {step['f_i']}")
        sys.stdout.flush()

    print(f"\n✓ Successfully constructed {len(tower)} fibration layers")
    sys.stdout.flush()

@PROFILE
def iterate_tower(fx_PR, pts_xy, max_steps=3, seed_int=SEED_INT, verbose=DEBUG, use_anchor_points=USE_ANCHOR_POINTS):
    """
    Iterates through fibration tower construction.

    Bimodal operation:
    - FINITE_FIELD=None: QQ/SR mode with symbolic derivatives
    - FINITE_FIELD=p: Pure finite-field mode (no SR)

    Args:
        fx_PR: Initial polynomial (QQ[x] or GF(p)[x])
        pts_xy: List of (x,y) points on the curve
        max_steps: Maximum tower layers to build
        seed_int: Random seed for geometry selection
        verbose: Enable diagnostic output
        use_anchor_points: Use anchor point strategy

    Returns:
        List of tower steps, each containing:
        - f_i: polynomial for this layer
        - Q_i: interpolated Q polynomial
        - rest_poly: remainder polynomial
        - r_expr: root expression (x1 - m)
        - info: metadata string
    """
    assert fx_PR is not None, "iterate_tower: fx_PR is None"
    assert pts_xy, "iterate_tower: pts_xy is empty"
    assert max_steps >= 0, f"iterate_tower: negative max_steps={max_steps}"

    _report_mode()

    tower = []
    CANDIDATES_PER_STEP = 10

    # ========================================================================
    # MODE DETECTION AND SETUP
    # ========================================================================

    ff_mode = (FINITE_FIELD is not None)

    if ff_mode:
        # ====================================================================
        # FINITE FIELD MODE
        # ====================================================================
        p = int(FINITE_FIELD)
        F = GF(p)

        # Verify input polynomial is over correct field
        try:
            poly_parent = fx_PR.parent()
            poly_base = poly_parent.base_ring()
        except Exception as e:
            raise RuntimeError(f"iterate_tower (FF): cannot inspect fx_PR parent: {e}")

        assert poly_base == F, \
            f"iterate_tower (FF): fx_PR base ring {poly_base} != GF({p})"

        # Verify points are in correct field
        for i, (xi, yi) in enumerate(pts_xy):
            try:
                xi_f = F(xi)
                yi_f = F(yi)
            except Exception as e:
                raise RuntimeError(
                    f"iterate_tower (FF): point {i} ({xi},{yi}) cannot coerce to GF({p}): {e}"
                )

        current_fx = fx_PR
        f0 = fx_PR
        m_parameter = None  # Will be set from first step

        if verbose:
            print(f"[iterate_tower FF] Starting with degree {current_fx.degree()} polynomial")
            print(f"[iterate_tower FF] Target: {max_steps} steps")
            sys.stdout.flush()

        for step in range(max_steps):
            n = int(current_fx.degree())
            g2 = len(pts_xy)

            if verbose:
                print(f"\n{'='*70}")
                print(f"[FF Step {step+1}/{max_steps}] Building fibration for degree {n} curve")
                print(f"{'='*70}")
                sys.stdout.flush()

            best_step_result = None
            best_score = float('inf')
            pts_subset = pts_xy[:g2]

            # Try multiple geometries, pick best
            for attempt in range(CANDIDATES_PER_STEP):
                attempt_seed = seed_int * 1000 + step * 100 + attempt

                try:
                    step_result = build_one_fibration_step(
                        current_fx, f0,
                        pts_subset,
                        g2,
                        seed_int=attempt_seed,
                        verbose=False,
                        parameter_m=m_parameter,
                        use_anchor_points=use_anchor_points
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [FF Step {step+1} attempt {attempt+1}] Failed: {e}")
                    continue

                # Verify step integrity
                try:
                    check_fibration_step(step_result, prev_fx=current_fx, layer_index=step)
                except Exception as e:
                    if verbose:
                        print(f"  [FF Step {step+1} attempt {attempt+1}] Verification failed: {e}")
                    continue

                # Score geometry (uses FF-aware scorer)
                try:
                    score = measure_poly_complexity(step_result['f_i'])
                except Exception as e:
                    if verbose:
                        print(f"  [FF Step {step+1} attempt {attempt+1}] Scoring failed: {e}")
                    continue

                if score < best_score:
                    best_score = score
                    best_step_result = step_result

            # Ensure we found valid geometry
            assert best_step_result is not None, \
                f"iterate_tower (FF): failed to build valid geometry for step {step+1} after {CANDIDATES_PER_STEP} attempts"

            if verbose:
                print(f"  [FF Step {step+1}] Selected geometry (Score={best_score:.1f})")
                sys.stdout.flush()

            tower.append(best_step_result)
            current_fx = best_step_result['f_i']

        # Verify tower consistency
        verify_tower_consistency(tower)

        if verbose:
            print(f"\n[iterate_tower FF] ✓ Built {len(tower)} layers")
            sys.stdout.flush()

        return tower

    else:
        # ====================================================================
        # QQ / SR MODE
        # ====================================================================

        # Setup symbolic variables
        try:
            poly_x_gen = fx_PR.parent().gen()
            x = SR.var(str(poly_x_gen))
        except Exception as e:
            raise RuntimeError(f"iterate_tower (QQ): cannot extract polynomial variable: {e}")

        # Convert initial polynomial to SR
        try:
            f0 = SR(fx_PR)
            current_fx = SR(fx_PR)
        except Exception as e:
            raise RuntimeError(f"iterate_tower (QQ): cannot convert fx_PR to SR: {e}")

        # Verify points are rational
        for i, (xi, yi) in enumerate(pts_xy):
            try:
                xi_qq = QQ(xi)
                yi_qq = QQ(yi)
            except Exception as e:
                raise RuntimeError(
                    f"iterate_tower (QQ): point {i} ({xi},{yi}) not rational: {e}"
                )

        m_parameter = None  # Will be set from first step

        if verbose:
            try:
                deg = int(current_fx.degree(x))
            except Exception:
                deg = "unknown"
            print(f"[iterate_tower QQ] Starting with degree {deg} polynomial")
            print(f"[iterate_tower QQ] Target: {max_steps} steps")
            sys.stdout.flush()

        for step in range(max_steps):
            try:
                n = int(current_fx.degree(x))
            except Exception as e:
                raise RuntimeError(f"iterate_tower (QQ): cannot determine degree at step {step}: {e}")

            g2 = len(pts_xy)

            if verbose:
                print(f"\n{'='*70}")
                print(f"[QQ Step {step+1}/{max_steps}] Building fibration for degree {n} curve")
                print(f"{'='*70}")
                sys.stdout.flush()

            best_step_result = None
            best_score = float('inf')
            pts_x_subset = [p[0] for p in pts_xy[:g2]]

            # Try multiple geometries, pick best
            for attempt in range(CANDIDATES_PER_STEP):
                attempt_seed = seed_int * 1000 + step * 100 + attempt

                try:
                    step_result = build_one_fibration_step(
                        current_fx, f0,
                        pts_x_subset,
                        g2,
                        seed_int=attempt_seed,
                        verbose=False,
                        parameter_m=m_parameter,
                        use_anchor_points=use_anchor_points
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [QQ Step {step+1} attempt {attempt+1}] Failed: {e}")
                    continue

                # Verify step integrity
                try:
                    check_fibration_step(step_result, prev_fx=current_fx, layer_index=step)
                except Exception as e:
                    if verbose:
                        print(f"  [QQ Step {step+1} attempt {attempt+1}] Verification failed: {e}")
                    continue

                # Extract/set m parameter from first successful step
                temp_m = m_parameter
                if temp_m is None and has_free_variables(step_result['r_expr']):
                    try:
                        temp_m = list(step_result['r_expr'].variables())[0]
                    except Exception as e:
                        raise RuntimeError(f"iterate_tower (QQ): cannot extract m parameter: {e}")

                # Verify fibration properties (symbolic mode)
                try:
                    _verify_fibration_step_properties(current_fx, step_result['r_expr'], temp_m)
                except Exception as e:
                    if verbose:
                        print(f"  [QQ Step {step+1} attempt {attempt+1}] Property check failed: {e}")
                    continue

                # Score geometry (QQ mode doesn't have scorer yet, use placeholder)
                score = float(attempt)  # Simpler: first valid geometry wins

                if score < best_score:
                    best_score = score
                    best_step_result = step_result
                    if m_parameter is None:
                        m_parameter = temp_m

            # Ensure we found valid geometry
            assert best_step_result is not None, \
                f"iterate_tower (QQ): failed to build valid geometry for step {step+1} after {CANDIDATES_PER_STEP} attempts"

            if verbose:
                print(f"  [QQ Step {step+1}] Selected geometry")
                sys.stdout.flush()

            tower.append(best_step_result)
            current_fx = best_step_result['f_i']

        # Verify tower consistency
        verify_tower_consistency(tower)

        # Verify y² consistency along rail at sample m-values
        try:
            verify_y2_consistency_on_rail(
                tower,
                x1=pts_xy[0][0],
                m_vals=[0, 1, -1, QQ(1)/QQ(2)]
            )
        except Exception as e:
            raise RuntimeError(f"iterate_tower (QQ): y² consistency check failed: {e}")

        if verbose:
            print(f"\n[iterate_tower QQ] ✓ Built {len(tower)} layers")
            sys.stdout.flush()

        return tower

