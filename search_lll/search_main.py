import numpy as np, os as _os, math, time
from .search_config import *
from .archimedean_optim import *
from .rational_arithmetic import *
from .search_analysis import *
# [fix] modularthread.py and search_analysis.py both define
# _batch_check_rationality (modularthread's is the one actually used at the
# call sites below). `from module import *` silently skips names starting
# with "_" unless the module declares __all__, so despite line 6 below doing
# `from .modularthread import *`, _batch_check_rationality was never actually
# bound in this module's namespace -- hence NameError at the call sites in
# run_standard_lattice_search, even though the function exists and works
# fine when called from within modularthread.py itself (no import boundary
# there). Import it explicitly by name so the wildcard's underscore rule
# doesn't silently drop it.
# NOTE: several imports in this module look unused here but are RE-EXPORTED to
# standard_search.py, which deliberately imports all its collaborators from
# search_main so it binds exactly the objects this namespace resolved (the
# wildcard chains contain same-named duplicates).  Don't delete without
# checking standard_search.py's import block: _batch_check_rationality,
# filter_residues_by_rail, predict_qc_distribution, and everything pulled in
# by the wildcard imports.
from .modularthread import _batch_check_rationality
from .modularthread import filter_residues_by_rail
from .modularthread import *
from .ll_utilities import *
from .diagnostics_univariate import *
from collections import namedtuple, Counter
from .mumford import *
from .selmer_genus2 import *
from .smoothness import *
from .index_calculus import *
from sage.all import QQ, PolynomialRing, SR
from .riemann_roch_localization import *
from search_common import *
from bounds import predict_qc_distribution
from .fiber_augment_hdf5 import build_fiber_augmented_relations as _orig_bfar
from .fiber_augment import *
from . import height_bound as _hb_mod


def _resolve_height_bound(cd, current_sections, search_vecs, sconf, height_pairing_H=None,
                           known_vectors_and_m=None):
    """
    Build the per-vector {v_orig_tuple: bound} dict from the Shioda-Tate
    canonical height pairing on the elliptic surface (search_lll/height_bound.py),
    falling back to the old flat sconf['HEIGHT_BOUND'] constant if a real H
    isn't available, there's no calibration data yet, or the derived bound
    can't be trusted for this vector set.

    known_vectors_and_m: list of (v_orig_tuple, m_value) pairs already
    CONFIRMED rational by a prior round of this same search (e.g.
    all_final_rational_pairs from earlier anomalous-sweep rounds). Used to
    calibrate the naive-vs-canonical height discrepancy constant c
    empirically (see height_bound.empirical_c_from_known_points) -- there
    is no closed-form Silverman-style bound available for a curve over a
    function field, so this MUST come from real data. None or empty means
    no calibration data exists yet (e.g. round 0 of a fresh search, or
    markov mode, which has no such accumulator at all) -- correctly falls
    back to the flat bound rather than guessing c=0, which is exactly the
    UNDER-estimate that would silently drop real points. The filter
    therefore activates progressively: it's inert on the very first round
    and switches on automatically once that round has found anything to
    calibrate against.

    *** WHY THIS NO LONGER CALLS E.height_pairing_matrix() ***
    cd.E_weier is an EllipticCurve over the function field Frac(QQ[m]) (this
    is a fibration -- m is the base coordinate), not over QQ or a number
    field. Sage's E.height_pairing_matrix()/E.silverman_height_bound() only
    exist for EllipticCurve_rational_field / EllipticCurve_number_field, so
    calling them here always raised AttributeError and silently fell back
    to the flat bound -- i.e. the per-vector filter was never actually
    active, despite being fully wired into modularthread.py's Stage 2
    prefilter and acceptance path. The real analogue for a fibration is the
    Shioda-Tate height pairing <P_i, P_j> = chi + (P.O) + (Q.O) - (P.Q) -
    sum_v contr_v(P,Q), which the driver loop already computes every
    iteration via check_independence -> compute_canonical_height_matrix
    (that's the "Height Pairing Matrix H:" print) and uses to build the
    search lattice itself (compute_search_vectors(H, height_bound)). H is
    now passed straight through here instead of being (impossibly)
    recomputed via a QQ-only Sage API.

    m_map_height_factor=1: valid for the linear/shift-only r_m seen so far
    in this codebase (e.g. r_m = -m-1) -- a linear map x = m + b has
    numerator/denominator growth bounded by a factor of 1 relative to its
    argument's. THIS IS NOT VALID for a Mobius transform T with nontrivial
    coefficients or any higher-degree r_m -- if a fibration using either
    of those is run through this, the resulting bound is not proven and
    should be re-derived (see height_bound.py's module docstring) before
    being trusted.
    """
    flat_fallback = sconf.get('HEIGHT_BOUND')

    if height_pairing_H is None:
        print("[height_bound] no Shioda-Tate H supplied for this iteration; "
              f"falling back to flat HEIGHT_BOUND={flat_fallback}")
        return flat_fallback

    try:
        n = height_pairing_H.nrows()
    except Exception as e:
        print(f"[height_bound] supplied H is not a matrix ({e}); "
              f"falling back to flat HEIGHT_BOUND={flat_fallback}")
        return flat_fallback

    if n != len(current_sections):
        print(f"[height_bound] H is {n}x{n} but current_sections has "
              f"{len(current_sections)} entries -- ordering mismatch, "
              f"refusing to use it. Falling back to flat HEIGHT_BOUND="
              f"{flat_fallback}")
        return flat_fallback

    try:
        bounds = _hb_mod.build_vector_height_bounds_from_matrix(
            height_pairing_H, search_vecs, m_map_height_factor=1
        )
    except Exception as e:
        print(f"[height_bound] per-vector bound computation failed ({e}); "
              f"falling back to flat HEIGHT_BOUND={flat_fallback}")
        return flat_fallback

    # Validate against every known point before trusting this to reject
    # anything (see height_bound.py's module docstring). all_found_x/known
    # m-values aren't threaded into this helper's args, so this checks the
    # cheap invariant we *can* check here -- that no bound came out
    # non-positive/degenerate, which would indicate H itself is bad (e.g.
    # not positive definite for the sections given) -- and defers the
    # against-known-points check to validate_height_bound_or_raise, called
    # once per iteration right after new points are found (see
    # run_qq_mode_diagnostics / the call added in search7_genus2.sage).
    for v_tuple, b in bounds.items():
        if b is not None and b <= 0:
            print(f"[height_bound] degenerate non-positive bound {b} for "
                  f"vector {v_tuple} -- H is likely not positive definite "
                  f"for these sections. Falling back to flat HEIGHT_BOUND="
                  f"{flat_fallback}")
            return flat_fallback

    return bounds
if FINITE_FIELD:
    from .lp_incidence_dlp import *
from markov.mumford_oscar_bridge import mumford_precompute_residues_oscar as _oscar_residues

_OSCAR_AVAILABLE = False


def naive_height_of_rational(q):
    """
    log(max(|numerator|, |denominator|)) of a rational number q, in lowest
    terms. This is the standard "naive height" h(q) used throughout the
    completeness-proof machinery (bounds.py's h_x / h_can) -- NOT the
    canonical height on the section lattice. Returns 0.0 for q == 0.
    Accepts anything QQ() can coerce.
    """
    try:
        qv = QQ(q)
    except Exception:
        return float('nan')
    if qv == 0:
        return 0.0
    n = abs(qv.numerator())
    d = abs(qv.denominator())
    return float(log(max(int(n), int(d))))


def log10_of_naive_height(h):
    """Convert a natural-log naive height h(q) into an approximate number
    of base-10 digits of max(|numerator|, |denominator|), i.e. log10(q)."""
    return h / math.log(10.0)


def _record_rational_candidate(m_val, v_tuple, r_m, shift, rationality_test_func,
                                all_candidate_records, all_processed_m_vals,
                                all_candidate_xs, known_x_before=None):
    """
    Resolve a single (m_val, v_tuple) candidate into a rational point and,
    if genuinely new, append it to the shared accumulators threaded through
    run_standard_lattice_search (all_candidate_records / all_processed_m_vals
    / all_candidate_xs).

    This is the single place a candidate pair joins the results.  It is called
    both from the residue-CRT-graph discovery step (which runs before round 0
    of the anomalous sweep) and from the per-round anomalous-sweep resolve
    loop, so points found either way land in the same bookkeeping: section
    reconstruction, x-height stats and completeness-proof accounting.

    known_x_before, if given, is compared against instead of the live
    all_candidate_xs -- lets a caller (like the per-round sweep loop) freeze
    "known before this round" for correct new-vs-repeat bookkeeping while
    still writing into the live set.

    Returns None if the pair fails the rationality re-check (shouldn't
    normally happen -- callers should already have confirmed rationality --
    but m_val alone doesn't carry y, so it's re-derived and re-checked here
    defensively). Otherwise returns a dict:
        {'x': x_val_q, 'y': y_val, 'is_new_x': bool, 'already_known_m': bool}
    """
    already_known_m = m_val in all_processed_m_vals
    try:
        x_val = r_m(m=m_val) - shift
        y_val = rationality_test_func(x_val)
    except (TypeError, ZeroDivisionError, ArithmeticError):
        return None
    if y_val is None:
        return None

    x_val_q = QQ(x_val)
    compare_against = known_x_before if known_x_before is not None else all_candidate_xs
    is_new_x = x_val_q not in compare_against

    if already_known_m:
        # Same m rediscovered from a different source -- point is already
        # recorded, nothing new to append, but the caller still wants to
        # know x/y and new-vs-repeat status for its own reporting.
        return {'x': x_val_q, 'y': y_val, 'is_new_x': is_new_x, 'already_known_m': True}

    v = vector(QQ, v_tuple)
    all_candidate_records.append({
        "m": m_val,
        "xj": x_val_q,
        "y": y_val,
        "v": tuple(v_tuple),
        "section": None,
    })
    all_processed_m_vals[m_val] = v
    all_candidate_xs.add(x_val_q)
    return {'x': x_val_q, 'y': y_val, 'is_new_x': is_new_x, 'already_known_m': False}


# State inherited by forked graph workers.  It is filled in by the parent
# immediately before the pool is created, so workers read it through fork's
# copy-on-write memory instead of receiving it by pickling.
_GRAPH_WORKER_STATE = {}


def _graph_vector_worker(task):
    """
    Run the residue-graph discovery for one vector and return plain data.

    task is (index, v_tuple, verbose).  Everything the worker needs beyond
    that comes from _GRAPH_WORKER_STATE.  Output printed during discovery is
    captured and returned as 'log' so the parent can replay it in vector
    order rather than interleaved across processes.

    Returns a dict with:
        'index', 'v'  : the task index and vector
        'ncand'       : number of confirmed chains
        'ms'          : list of (m_num, m_den), one per reconstruction
        'trace'       : the graph's known-m trace entries
        'log'         : captured stdout
        'secs'        : wall time for this vector
    """
    import contextlib
    import io
    index, v_tuple, verbose = task
    st = _GRAPH_WORKER_STATE
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rg_result = discover_candidates_via_residue_graph(
            st['residues'], st['prime_pool'], height_bound=st['height_bound'],
            v_tuple=v_tuple, debug=True, known_m=st['trace_ms'],
            verbose_graph=verbose, print_header=(index == 1),
        )
    ms = [(r['m_num'], r['m_den'])
          for cand in rg_result['candidates'] for r in cand['reconstructions']]
    return {
        'index': index,
        'v': v_tuple,
        'ncand': len(rg_result['candidates']),
        'ms': ms,
        'trace': (rg_result.get('graph') or {}).get('trace', []),
        'log': buf.getvalue(),
        'secs': time.time() - t0,
    }


def _run_graph_vectors(tasks, num_workers):
    """
    Yield _graph_vector_worker results for tasks in task order.

    Uses a fork-based process pool when there is more than one task and more
    than one worker, and runs inline otherwise.
    """
    n_workers = max(1, min(int(num_workers), len(tasks)))
    if n_workers == 1:
        for task in tasks:
            yield _graph_vector_worker(task)
        return
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        yield from executor.map(_graph_vector_worker, tasks)


def _call_residues(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                   rhs_modp_list, vecs_list, num_workers, debug, pool, chunk_size,
                   section_poly_dict=None):
    """
    Dispatch to Oscar bridge if available and USE_OSCAR_RESIDUES env var is set,
    otherwise fall back to the original Python implementation.

    To enable Oscar: set USE_OSCAR_RESIDUES=1 in your environment before
    starting the Sage/Python process, and make sure JULIA_NUM_THREADS is also set.
    """
    use_oscar = _OSCAR_AVAILABLE
    assert use_oscar

    if use_oscar:
        return _oscar_residues(
            eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
            rhs_modp_list, vecs_list,
            debug=debug, chunk_size=chunk_size,
            section_poly_dict=section_poly_dict,
            # num_workers and pool are ignored by the bridge (Julia handles threading)
        )
    else:
        from search_lll.mumford.mumford_parallel import mumford_precompute_residues_parallel
        ret = mumford_precompute_residues_parallel(
            eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
            rhs_modp_list, vecs_list,
            num_workers=num_workers,
            debug=debug, pool=pool, chunk_size=chunk_size,
        )
        print(ret)
        return ret

# After your Mumford search in FINITE_FIELD mode:

def search_lattice_symbolic(cd, current_sections, vecs, rhs_list, r_m, shift,
                            all_found_x, rationality_test_func, stats):
    """
    Symbolic search for rational points via solving x_sv == rhs(m) over QQ(m).

    Controlled by the SYMBOLIC_SEARCH flag from search_common.py. If SYMBOLIC_SEARCH is False,
    this is a no-op and returns empty results quickly.
    """
    # Respect the global flag; search_common.py should define SYMBOLIC_SEARCH (all-caps).
    # We do not import here; search_common is already imported at top of file.
    SYMBOLIC_ENABLED = globals().get('SYMBOLIC_SEARCH', False)
    if not SYMBOLIC_ENABLED:
        if DEBUG:
            print("Symbolic search disabled by SYMBOLIC_SEARCH flag.")
        return set(), []

    if not current_sections:
        if DEBUG:
            print("Symbolic search: no current sections provided, skipping.")
        return set(), []

    # === NEW: RUN UNIVARIATE DIAGNOSTICS ===
    try:
        run_univariate_diagnostics(
            cd=cd,
            current_sections=current_sections,
            rhs_list=rhs_list,
            vecs=vecs,
            max_n=len(vecs)  # Analyze up to [12]P
        )
    except Exception as e:
        print(f"Univariate diagnostics failed: {e}")
        raise
    # === END NEW DIAGNOSTICS ===

    print("--- Starting symbolic search over QQ ---")
    stats.start_phase('symbolic_search') # <-- STATS

    # Canonical setup for m (use PR_m and its fraction field so arithmetic stays in QQ(m))
    PR_m = PolynomialRing(QQ, 'm')
    SR_m = var('m')
    Fm = PR_m.fraction_field()

    newly_found_x = set()
    new_sections = []
    found_x_to_section_map = {}

    # Quick sanity: ensure sections are projective-like and have x/z
    # (use assert to make developer intent explicit)
    assert all(len(sec) >= 3 for sec in current_sections), "current_sections entries must be 3-coord sections"

    # Main search: iterate over integer vectors (vecs) and solve numerator==0 over QQ
    # NOTE: we do NOT loop over rational m values; instead we solve for m via polynomial roots.
    for v_tuple in tqdm(vecs, desc="Symbolic Search"):
        if all(int(c) == 0 for c in v_tuple):
            continue

        v = vector(ZZ, [int(c) for c in v_tuple])
        #print("trying search vector:", v) # Reduced verbosity
        S_v = sum(v[i] * current_sections[i] for i in range(len(current_sections)))

        # skip degenerate/new-section-zero cases
        if S_v.is_zero():
            #print("search section is zero; skipping.")
            continue
        if S_v[2].is_zero():
            # projective z==0 (point at infinity) — skip
            #print("search section is point at infinity; skipping.")
            continue

        # Affine x-coordinate in QQ(m) (attempt to coerce)
        try:
            x_sv_raw = S_v[0] / S_v[2]
            x_coerced = Fm(SR(x_sv_raw))
        except Exception:
            # If coercion fails, skip this vector (diagnostic if DEBUG)
            if DEBUG:
                print("Symbolic coercion failed for a section; skipping vector:", v_tuple)
            raise # Let's not raise here unless debugging is critical
            continue
        #print("search x:", x_coerced)

        for rhs_func in rhs_list:
            stats.incr('symbolic_solves_attempted') # <-- STATS
            try:
                rhs_coerced = Fm(SR(rhs_func))
                diff = x_coerced - rhs_coerced
                num = diff.numerator()
            except Exception:
                if DEBUG:
                    print("Symbolic coercion of rhs failed; skipping this rhs.")
                raise
                continue

            # If numerator is constant, there is no m-solution
            if num.degree() == 0:
                #print("numerator is constant; no solution")
                continue

            # Build polynomial in PR_m and get rational roots
            try:
                num_poly = PR_m(num)   # coerce numerator into QQ[m]
            except Exception:
                if DEBUG:
                    print("Could not coerce numerator into PR_m; skipping.")
                raise
                continue

            try:
                roots = num_poly.roots(ring=QQ, multiplicities=False)
            except Exception:
                # If root-finding over QQ fails, skip (better to fail loudly during debugging)
                if DEBUG:
                    print("num_poly.roots(...) failed for polynomial:", num_poly)
                raise
                continue

            if not roots:
                #print("no roots found")
                pass # This happens often, no need to print
            else:
                stats.incr('symbolic_solves_success', n=len(roots)) # <-- STATS
                if DEBUG: print("Symbolic solve success! Found root(s):", roots)

            # For each rational root m0, verify equality by evaluation (clearing denominators),
            # then test rationality and add the point.
            for m_val in roots:
                m_q = QQ(m_val)   # ensure rational

                # Evaluate LHS and RHS using SR substitution to get exact rationals where possible
                try:
                    lhs_at = SR(x_sv_raw).subs({SR_m: m_q})
                    rhs_at = SR(rhs_func).subs({SR_m: m_q})
                except Exception:
                    if DEBUG:
                        print("SR substitution failed at m=", m_q)
                    raise
                    continue

                # Try coercion to QQ for reliable equality checks
                try:
                    lhs_q = QQ(lhs_at)
                    rhs_q = QQ(rhs_at)
                except Exception:
                    # If we cannot coerce either side, fall back to clearing denominators
                    try:
                        lhs_q = QQ(r_m(m=m_q) - shift)
                    except Exception:
                        if DEBUG:
                            print("Failed to compute numeric r_m at m=", m_q)
                        raise
                        continue
                    raise
                    # We cannot easily compute rhs numeric without r_m; but if lhs_q is defined,
                    # we can proceed to rationality test as before.
                    rhs_q = None
                    raise

                # If we have both sides as QQ check equality; otherwise trust the root machinery but still verify via r_m
                if rhs_q is not None and lhs_q != rhs_q:
                    if DEBUG:
                        print("Symbolic-match FAIL for root m =", m_q, "; lhs != rhs after coercion.")
                    raise
                    continue

                # Compute x via r_m (exact rational) and apply shift
                try:
                    x_val = r_m(m=m_q) - shift
                except Exception:
                    if DEBUG:
                        print("r_m evaluation failed at m=", m_q)
                    raise
                    continue

                # Avoid duplicates
                try:
                    x_val_q = QQ(x_val)
                except Exception:
                    # if not rational-coercible, skip
                    if DEBUG:
                        print("x_val not coercible to QQ at m=", m_q, "; skipping")
                    raise
                    continue

                if x_val_q in all_found_x or x_val_q in newly_found_x:
                    #print("found x already seen:", x_val_q)
                    continue

                # Check rationality of y via rationality_test_func
                stats.incr('rationality_tests_total') # <-- STATS (Symbolic path)
                y_val = rationality_test_func(x_val_q)
                if y_val is None:
                    stats.record_failure(m_q, reason='y_not_rational_symbolic') # <-- STATS
                    #print("yval is None; x value found does not give rational point.")
                    # not a rational point
                    continue

                # Found a new rational point
                stats.record_success(m_q, point=x_val_q) # <-- STATS (Symbolic path)
                newly_found_x.add(x_val_q)
                found_x_to_section_map[x_val_q] = S_v
                new_sections.append(S_v)

                if DEBUG:
                    print("Found new rational point via symbolic m =", m_q, " x =", x_val_q)

    # OPTIONAL ASSERT: if the user expects the base m to be discovered, allow caller to check
    # The assert function lives in this module: assert_base_m_found(...)
    stats.end_phase('symbolic_search') # <-- STATS
    return newly_found_x, new_sections

def search_prime_subsets_unified(prime_subsets, worker_func, num_workers=8, debug=DEBUG):
    """
    Process prime subsets in parallel using ProcessPoolExecutor (unified).
    Replaces the multiprocessing.Pool call in search_lattice_modp_lll_subsets.

    Args:
        prime_subsets (list): Prime subsets to search
        worker_func (callable): Worker function (from functools.partial)
        num_workers (int): Number of workers
        debug (bool): Print diagnostics

    Returns:
        list: A list of tuples, one for each subset processed:
              [(subset, candidates_set, worker_stats_dict), ...]
        Counter: Merged stats_counter dict from all workers (Redundant, can be rebuilt from list)
    """
    try:
        ctx = multiprocessing.get_context("fork")
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}
        raise

    # List to store results per subset
    subset_results_list = []
    merged_stats = Counter() # Keep merging stats here too for now
    all_crt_classes = set()  # <-- NEW

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(worker_func, subset): subset for subset in prime_subsets}

        with tqdm(total=len(futures), desc="Searching Prime Subsets") as pbar:
            for future in as_completed(futures):
                original_subset = futures[future]
                try:
                    # Worker now returns three items
                    candidates_set, stats_dict, crt_classes  = future.result()
                    # Append the result tuple to the list
                    subset_results_list.append((original_subset, candidates_set, stats_dict))
                    merged_stats.update(stats_dict) # Keep merging here
                    all_crt_classes.update(crt_classes)  # <-- Collect
                except Exception as e:
                    if debug:
                        print(f"Subset worker failed for subset {original_subset}: {e}")
                    # Append a failure placeholder if needed, or just skip
                    subset_results_list.append((original_subset, set(), Counter()))
                    raise
                finally:
                    pbar.update(1)

    # Return the list of per-subset results and the merged stats
    return subset_results_list, merged_stats, all_crt_classes  # <-- Return classes

def _run_index_calculus_attack(mumford_divisors, coeffs_genus2, tower_data, found_xs,
                               mumford_residues, stats, num_workers, x_b, shifted_coeffs,
                               lp_seed_xs=None):
    """Sub-handler for the Index Calculus execution phase."""
    p = int(FINITE_FIELD)
    f_poly = sage_poly_from_coeffs(coeffs_genus2, PolynomialRing(GF(p), 'x'))

    atom_to_idx, fb_y_cache = extract_factor_base(mumford_divisors, p, f_poly, verbose=True)

    fb_roots = []
    for atom, idx in atom_to_idx.items():
        if atom[0] == 'd1':
            x_val = atom[1]
            if x_val not in fb_roots:
                fb_roots.append(x_val)

    fb_roots_set = set(fb_roots)
    L = compute_jacobian_order(coeffs_genus2, p)

    print(f"  [Setup] Curve: y^2 = {f_poly}")
    G, Q, true_d = BASE_DIVISOR, TARGET_DIVISOR, SECRET_KEY

    print("\n" + "="*70)
    print("TESTING FACTOR BASE HOMOMORPHISM PROPERTY")
    print("="*70)
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    if not homomorphism_test(J, atom_to_idx, f_poly, p, check_divisors=None):
        print("CRITICAL: Homomorphism test FAILED!")
        raise RuntimeError("Factor base homomorphism test failed")
    print(" Homomorphism test PASSED")
    print("="*70 + "\n")

    print(f"  [Phase 0] Attempting RR-Localization for target Q (Parallel)...")
    pole_range = [6, 7, 8, 9, 10]
    rr_tasks = [(Q, fb_roots_set, f_poly, p, n) for n in pole_range]
    found_rr_solution = None

    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = None

    with ProcessPoolExecutor(max_workers=min(len(rr_tasks), num_workers), mp_context=ctx) as executor:
        futures = {executor.submit(localize_wrapper, args): args[-1] for args in rr_tasks}
        for future in as_completed(futures):
            n_pole_val = futures[future]
            try:
                roots, poly_a, poly_b, vec = future.result()
                if roots is not None:
                    print(f"  [!] Phase 0 Success: Target Q decomposed via RR(n={n_pole_val})!")
                    found_rr_solution = (roots, poly_a, poly_b, vec)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            except Exception as e:
                print(f"  [!] RR Worker (n={n_pole_val}) failed: {e}")
                raise e

    if found_rr_solution:
        roots, poly_a, poly_b, vec = found_rr_solution
        x_to_idx = {atom[1]: idx for atom, idx in atom_to_idx.items() if atom[0] == 'd1'}
        log_v = resolve_log_from_rr_decomposition(roots, x_to_idx, fb_y_cache, poly_a, poly_b, p)
        print(f"  [!] SUCCESS: Discrete Log recovered via geometric corridor: {log_v}")
        return found_xs, [], mumford_residues, stats

    print("  [Phase 0] RR did not find a short relation. Falling back to Index Calculus.")
    try:
        #f_shifted_poly = sage_poly_from_coeffs(shifted_coeffs, PolynomialRing(GF(p), 'x')) if shifted_coeffs else f_poly
        f_shifted_poly = sage_poly_from_coeffs(list(reversed(shifted_coeffs)), PolynomialRing(GF(p), 'x')) if shifted_coeffs else f_poly
        E_rhs_m_for_aug = tower_data[-1]['f_i'] if tower_data is not None else None

        if lp_seed_xs is None:
            lp_seed_xs = set()

        # Phase 1
        if E_rhs_m_for_aug is not None:
            print("\n" + "="*70)
            print("PHASE 1: LP INCIDENCE DLP ATTACK")
            print(f"  LP seeds available: {len(lp_seed_xs)}")
            print("="*70)
            lp_result = solve_dlp_via_lp_incidence(
                E_rhs_m=E_rhs_m_for_aug,
                f_shifted_fp=f_shifted_poly,
                x_b=x_b,
                p=p,
                ell=int(GROUP_MODULUS),
                base_divisor=BASE_DIVISOR,
                target_divisor=TARGET_DIVISOR,
                atom_to_idx=atom_to_idx,
                lp_seed_xs=lp_seed_xs,
                verbose=True,
            )

            if lp_result['verified']:
                print(f"  [!] Phase 1 SUCCESS: k = {lp_result['dlp']}")
                return found_xs, [], mumford_residues, stats
            print("  [Phase 1] LP incidence attack did not verify. Falling through to Phase 2.")
        else:
            print("  [Phase 1] Skipped: E_rhs_m not available.")

        log_v = perform_dlp_attack(
            G, Q, mumford_divisors, p, coeffs_genus2, L,
            verbose=True, force_index_calculus=True,
            E_rhs_m=E_rhs_m_for_aug, x_b=x_b, f_shifted_poly=f_shifted_poly,
        )
        print(f"✓ Confirmed Discrete Log: {log_v}")
    except Exception as e:
        print(f"Attack failed: {e}")
        raise

    return found_xs, [], mumford_residues, stats

def search_candidates_adapter(xi, search_context):
    """
    Thin wrapper around existing search code.
    Returns a list of candidate dicts.
    """

    new_xs, new_sections, residues, stats = search_lattice_modp_unified_parallel(
        **search_context(xi)
    )

    candidates = []

    # YOU NEED to expose this from inside search:
    # currently it's buried as final_rational_candidates

    for m_val, v_tuple in stats.get('final_pairs', []):
        x_val = search_context['r_m'](m=m_val) - search_context['shift']

        candidates.append({
            "xj": x_val,
            "m": m_val,
            "v": v_tuple,
        })

    return candidates

def search_lattice_modp_unified_parallel(cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
                                         all_found_x, num_subsets, rationality_test_func,
                                         sconf, coeffs_genus2,
                                         tower_data=None,
                                         num_workers=PARALLEL_PRIME_WORKERS, debug=False,
                                         precomputed_residues=None,
                                         x_b=None, shifted_coeffs=None,
                                         markov_mode=False,
                                         height_pairing_H=None):
    """
    Unified parallel search router.

    If markov_mode=True, always use the lightweight standard-lattice path and
    return candidate pools early, skipping expensive downstream attack logic.

    height_pairing_H: Shioda-Tate height-pairing matrix for current_sections
    (see run_standard_lattice_search / _resolve_height_bound). Not used on
    the Mumford/finite-field branch -- that mode works over a finite field,
    where this QQ-height machinery doesn't apply.
    """
    USE_MUMFORD = globals().get('MUMFORD_SEARCH', False) and tower_data is not None and not markov_mode
    print("USE_MUMFORD", USE_MUMFORD)
    assert len(vecs) > 1, vecs

    if USE_MUMFORD:
        return run_mumford_search(
            cd, current_sections, prime_pool, vecs, rhs_list, shift,
            rationality_test_func, coeffs_genus2, tower_data,
            num_workers, debug, x_b, shifted_coeffs
        )
    else:
        return run_standard_lattice_search(
            cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
            all_found_x, num_subsets, rationality_test_func, sconf, coeffs_genus2,
            num_workers, debug, precomputed_residues,
            markov_mode=markov_mode,
            height_pairing_H=height_pairing_H,
        )

def run_standard_lattice_search(cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
                                 all_found_x, num_subsets, rationality_test_func, sconf, coeffs_genus2,
                                 num_workers, debug, precomputed_residues,
                                 markov_mode=False, height_pairing_H=None):
    """
    Standard lattice search.  The implementation lives in
    search_lll/standard_search.py (staged into small functions); this wrapper
    keeps the historical import location and signature.

    Normal mode: prep -> residues -> Brauer -> residue-graph discovery ->
    anomalous-residue sweep.  Markov mode: skips expensive tuning / Brauer /
    attack plumbing and returns a compact candidate pool.

    Imported lazily because standard_search itself imports its helpers from
    this module.
    """
    from .standard_search import run_standard_lattice_search as _impl
    return _impl(
        cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
        all_found_x, num_subsets, rationality_test_func, sconf, coeffs_genus2,
        num_workers, debug, precomputed_residues,
        markov_mode=markov_mode, height_pairing_H=height_pairing_H,
    )


def run_mumford_search(cd, current_sections, prime_pool, vecs, rhs_list, shift,
                      rationality_test_func, coeffs_genus2, tower_data,
                      num_workers, debug, x_b, shifted_coeffs,
                      markov_mode=False, pool=None, chunk_size=8):
    """
    Mumford / Finite Field search.

    Modes:
    - markov_mode=True  → return raw Mumford residues + metadata ONLY
    - markov_mode=False → full pipeline (reconstruction + attack)
    """

    print("\n" + "="*70)
    print("MUMFORD SEARCH")
    print("Mode:", "MARKOV (residues only)" if markov_mode else "FULL PIPELINE")
    print("="*70 + "\n")

    stats = SearchStats()

    # --------------------------------------------------
    # Phase 1: Build Mumford system
    # --------------------------------------------------
    stats.start_phase('mumford_setup')
    eqs_dict = build_mumford_equations_from_fibration(tower_data, coeffs_genus2)
    stats.end_phase('mumford_setup')

    # --------------------------------------------------
    # Phase 2: Modular prep
    # --------------------------------------------------
    stats.start_phase('prep_mod_data')
    # search_main.py line 1184 — add section_poly_dict
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll, section_poly_dict = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=prime_pool
    )
    stats.end_phase('prep_mod_data')

    if not Ep_dict:
        return {
            "residues": {},
            "prime_list": [],
            "vecs": [],
            "Ep_dict": {},
            "stats": stats,
            "metadata": {}
        } if markov_mode else (set(), [], {}, stats)

    prime_list = sorted(Ep_dict.keys())
    vecs_list = list(vecs)

    # --------------------------------------------------
    # Phase 3: Mumford residues (CORE)
    # --------------------------------------------------
    stats.start_phase('mumford_residues')

    mumford_residues = _call_residues(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list,
        num_workers=num_workers,
        debug=debug, pool=pool, chunk_size=chunk_size,
        section_poly_dict=section_poly_dict,
    )

    stats.end_phase('mumford_residues')

    # --------------------------------------------------
    # MARKOV FAST EXIT 🚀
    # --------------------------------------------------
    if markov_mode:
        total_supports = 0
        for pmap in mumford_residues.values():
            for vmap in pmap.values():
                total_supports += len(vmap)

        return {
            "residues": mumford_residues,
            "prime_list": prime_list,
            "vecs": vecs_list,
            "Ep_dict": Ep_dict,
            "stats": stats,
            "metadata": {
                "num_primes": len(prime_list),
                "num_vectors": len(vecs_list),
                "total_supports": total_supports,
            }
        }

    # --------------------------------------------------
    # Phase 4: (optional) diagnostics
    # --------------------------------------------------
    if FINITE_FIELD:
        compute_zeta_direct(coeffs_genus2, int(FINITE_FIELD))
        compute_zeta_from_fibration(mumford_residues, vecs_list, int(FINITE_FIELD))

    # --------------------------------------------------
    # Phase 5: Reconstruction (heavy)
    # --------------------------------------------------
    stats.start_phase('mumford_reconstruction')

    found_xs, mumford_divisors, lp_seed_xs = reconstruct_and_verify_mumford(
        mumford_residues, prime_list, coeffs_genus2, shift, rationality_test_func
    )

    stats.end_phase('mumford_reconstruction')

    # --------------------------------------------------
    # Phase 6: Optional filtering
    # --------------------------------------------------
    if FINITE_FIELD:
        mumford_divisors = filter_g_q_from_list(
            mumford_divisors,
            BASE_DIVISOR,
            TARGET_DIVISOR,
            FINITE_FIELD,
            coeffs_genus2
        )

    print(f"\nMumford search reconstructed {len(mumford_divisors)} divisors")

    # --------------------------------------------------
    # Phase 7: Attack (only in FULL mode)
    # --------------------------------------------------
    if FINITE_FIELD:
        return _run_index_calculus_attack(
            mumford_divisors,
            coeffs_genus2,
            tower_data,
            found_xs,
            mumford_residues,
            stats,
            num_workers,
            x_b,
            shifted_coeffs,
            lp_seed_xs
        )

    # Non-FF case
    print(f"Mumford search found {len(found_xs)} rational points")
    stats.incr('rational_points_unique', n=len(found_xs))

    return found_xs, [], mumford_residues, stats
