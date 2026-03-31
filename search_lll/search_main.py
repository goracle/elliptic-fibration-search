import numpy as np
from .search_config import *
from .archimedean_optim import *
from .rational_arithmetic import *
from .search_analysis import *
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
from .fiber_augment_hdf5 import build_fiber_augmented_relations as _orig_bfar
from .fiber_augment import *
from .lp_incidence_dlp import *

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
                                         num_workers=20, debug=False,
                                         precomputed_residues=None,
                                         x_b=None, shifted_coeffs=None,
                                         markov_mode=False):
    """
    Unified parallel search router.

    If markov_mode=True, always use the lightweight standard-lattice path and
    return candidate pools early, skipping expensive downstream attack logic.
    """
    USE_MUMFORD = globals().get('MUMFORD_SEARCH', False) and tower_data is not None and not markov_mode
    print("USE_MUMFORD", USE_MUMFORD)
    assert USE_MUMFORD
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
            markov_mode=markov_mode
        )

def run_standard_lattice_search(cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
                                 all_found_x, num_subsets, rationality_test_func, sconf, coeffs_genus2,
                                 num_workers, debug, precomputed_residues,
                                 markov_mode=False):
    """
    Standard lattice search.

    Normal mode: keeps the existing pipeline.
    Markov mode: skips expensive tuning / Brauer / attack plumbing and returns
    a compact candidate pool suitable for transition selection.
    """
    # === UNPACK: SCONF ===
    min_prime_subset_size = sconf['MIN_PRIME_SUBSET_SIZE']
    min_max_prime_subset_size = sconf['MIN_MAX_PRIME_SUBSET_SIZE']
    max_modulus = sconf['MAX_MODULUS']
    tmax = sconf['TMAX']

    # === VECTOR-BLIND CONSENSUS OPTIMIZATION ===
    optimized_vecs = vecs

    if precomputed_residues and precomputed_residues is not True:
        first_prime = next(iter(precomputed_residues), None)
        if first_prime and precomputed_residues[first_prime]:
            first_vector = next(iter(precomputed_residues[first_prime]))
            if all(v == 0 for v in first_vector):
                dim = len(current_sections)
                zero_vec_tuple = tuple([0] * dim)
                optimized_vecs = [zero_vec_tuple]
                if debug:
                    print("OPTIMIZATION: Vector-Blind Consensus detected. Forcing search vectors (`vecs`) to just the zero vector key.")

    search_vecs = optimized_vecs

    # === STATS: INIT ===
    stats = SearchStats()

    print("prime pool used for search:", prime_pool)

    # === PHASE: PREP MOD DATA ===
    stats.start_phase('prep_mod_data')
    print("--- Preparing modular data for LLL search ---")
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=prime_pool
    )
    stats.end_phase('prep_mod_data')

    if not Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return {
            "candidates": [],
            "candidate_xs": set(),
            "new_sections": [],
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": [],
        }

    # === PHASE: PRECOMPUTE RESIDUES ===
    vecs_list = list(search_vecs)
    if precomputed_residues is None:
        stats.start_phase('precompute_residues')
        primes_to_compute = list(Ep_dict.keys())
        num_rhs_fns = len(rhs_list)

        args_list = [
            (
                p,
                Ep_dict[p],
                mult_lll.get(p, {}),
                vecs_lll.get(p, [tuple([0] * len(current_sections)) for _ in vecs_list]),
                vecs_list,
                rhs_modp_list,
                num_rhs_fns,
                stats
            )
            for p in primes_to_compute
        ]

        precomputed_residues = {}
        total_modular_checks = 0

        try:
            ctx = multiprocessing.get_context("fork")
            exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
        except Exception:
            exec_kwargs = {"max_workers": num_workers}

        with ProcessPoolExecutor(**exec_kwargs) as executor:
            if TORSION_SLOPPY:
                futures = {executor.submit(compute_residues_for_prime_worker, args): args[0] for args in args_list}
            else:
                futures = {executor.submit(compute_residues_for_prime_worker_old, args): args[0] for args in args_list}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
                p = futures[future]
                try:
                    p_ret, mapping, local_modular_checks = future.result()
                    mapping = mapping or {}
                    precomputed_residues[p_ret] = mapping
                    total_modular_checks += int(local_modular_checks or 0)

                    residues_union = set()
                    for vtuple, rhs_lists in mapping.items():
                        for rl in rhs_lists:
                            for r in rl:
                                if isinstance(r, int):
                                    residues_union.add(r)

                    stats.residues_by_prime[p_ret].update(residues_union)
                    stats.counters['modular_checks'] += int(local_modular_checks or 0)
                    stats.counters[f'modular_checks_p_{p_ret}'] += int(local_modular_checks or 0)
                    stats.counters[f'residues_seen_p_{p_ret}'] = len(stats.residues_by_prime[p_ret])

                except Exception as e:
                    if debug:
                        print(f"[precompute fail] p={p}: {e}")
                    precomputed_residues[p] = {}
                    stats.residues_by_prime[p].update(set())
                    stats.counters[f'modular_checks_p_{p}'] = 0
                    stats.counters[f'residues_seen_p_{p}'] = 0
                    raise

        if debug:
            print(f"[precompute] total_modular_checks={total_modular_checks}, primes precomputed={len(precomputed_residues)}")

        stats.end_phase('precompute_residues')
    else:
        print(f"Using provided precomputed residues ({len(precomputed_residues)} primes)")
        stats.incr('using_consensus_residues', n=1)

    # Populate stats.residues_by_prime from precomputed residues
    for p, p_mapping in precomputed_residues.items():
        for v_tuple, rhs_lists in p_mapping.items():
            for rhs_list_item in rhs_lists:
                for residue in rhs_list_item:
                    if isinstance(residue, (int, Integer)):
                        stats.add_residue(p, residue)

    # ------------------------------------------------------------------
    # MARKOV MODE: stop early, no Brauer/autotune/attack plumbing.
    # ------------------------------------------------------------------
    if markov_mode:
        stats.start_phase('markov_subset_search')

        # Keep this deliberately simple for transition generation:
        # one subset per prime. This avoids the expensive combinatorial
        # subset generation and still gives you a candidate pool.
        prime_subsets_to_process = [[p] for p in prime_pool if p in precomputed_residues]

        if not prime_subsets_to_process:
            stats.end_phase('markov_subset_search')
            return {
                "candidates": [],
                "candidate_xs": set(),
                "new_sections": [],
                "precomputed_residues": precomputed_residues,
                "stats": stats,
                "final_rational_pairs": [],
            }

        worker_func = partial(
            process_prime_subset_precomputed,
            vecs=search_vecs,
            r_m=r_m,
            shift=shift,
            tmax=tmax,
            combo_cap=max_modulus,
            precomputed_residues=precomputed_residues,
            prime_pool=prime_pool,
            num_rhs_fns=len(rhs_list),
            coeffs_genus2=coeffs_genus2
        )

        subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
            prime_subsets_to_process, worker_func, num_workers=num_workers, debug=debug
        )

        stats.merge_dict(worker_stats_dict)
        stats.crt_classes_tested = all_crt_classes
        stats.incr('subsets_processed', n=len(subset_results_list))

        overall_found_candidates_from_workers = set()
        for subset, candidates_set, _ in subset_results_list:
            overall_found_candidates_from_workers.update(candidates_set)

        stats.incr('crt_candidates_found', n=len(overall_found_candidates_from_workers))

        if not overall_found_candidates_from_workers:
            stats.end_phase('markov_subset_search')
            return {
                "candidates": [],
                "candidate_xs": set(),
                "new_sections": [],
                "precomputed_residues": precomputed_residues,
                "stats": stats,
                "final_rational_pairs": [],
            }

        # Keep the rationality check, but skip all the heavier analytics.
        candidate_list = list(overall_found_candidates_from_workers)
        final_rational_candidates = set()

        batch_size = max(1, floor(0.05 * len(candidate_list)))
        for i in range(0, len(candidate_list), batch_size):
            batch = candidate_list[i:i + batch_size]
            newly_rational = _batch_check_rationality(
                batch, r_m, shift, rationality_test_func, current_sections, stats
            )
            final_rational_candidates.update(newly_rational)

        candidate_records = []
        candidate_xs = set()
        new_sections_raw = []
        processed_m_vals = {}

        for m_val, v_tuple in final_rational_candidates:
            if m_val in processed_m_vals:
                continue
            try:
                x_val = r_m(m=m_val) - shift
                y_val = rationality_test_func(x_val)
                if y_val is None:
                    continue

                x_val_q = QQ(x_val)
                v = vector(QQ, v_tuple)

                if x_val_q in all_found_x:
                    continue

                rec = {
                    "m": m_val,
                    "xj": x_val_q,
                    "y": y_val,
                    "v": tuple(v_tuple),
                    "section": None,
                }
                if any(c != 0 for c in v):
                    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                    rec["section"] = new_sec
                    new_sections_raw.append(new_sec)

                candidate_records.append(rec)
                processed_m_vals[m_val] = v
                candidate_xs.add(x_val_q)

            except Exception:
                raise

        new_sections = list({s: None for s in new_sections_raw}.keys())
        stats.incr('rational_points_unique', n=len(candidate_xs))
        stats.incr('new_sections_unique', n=len(new_sections))
        stats.end_phase('markov_subset_search')

        return {
            "candidates": candidate_records,
            "candidate_xs": candidate_xs,
            "new_sections": new_sections,
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": list(final_rational_candidates),
        }

    # ------------------------------------------------------------------
    # ORIGINAL HEAVY PATH BELOW
    # ------------------------------------------------------------------
    residue_counts = compute_residue_counts_for_primes(cd, rhs_list, prime_pool, max_primes=30)
    coverage_estimator = CoverageEstimator(prime_pool, residue_counts)

    stats.start_phase('brauer')

    report = estimate_completeness_probability(precomputed_residues, PRIME_POOL)
    print(f"[brauer] estimated survival fraction ≈ {report['estimate_survive']:.6f}")
    print(f"[brauer] estimated ruled-out fraction ≈ {report['estimate_ruled_out']:.6f}")

    mc = probe_algebraic_brauer_obstructions(precomputed_residues, PRIME_POOL, sample_size=1000)
    print(f"[brauer] Monte Carlo blocked fraction ≈ {mc['monte_carlo']['blocked_fraction_est']:.6f}")

    some_m = QQ(1)
    allowed, details = m_is_locally_allowed(some_m, precomputed_residues, PRIME_POOL)
    print(f"[brauer] example m={some_m} locally allowed? {allowed}")

    stats.end_phase('brauer')

    if TARGETED_X:
        ret = diagnose_missed_point(TARGETED_X, r_m, shift, precomputed_residues, prime_pool, vecs)
        matched_subset = None
        if 'matched_primes' in ret:
            matched_subset = ret['matched_primes']

        const = r_m(m=0)
        mtarget = QQ(-1) * TARGETED_X + const

        cov1 = compute_residue_coverage_for_m(mtarget, precomputed_residues, PRIME_POOL)
        print("cov1: m = ", mtarget, " coverage:", cov1['coverage_fraction'])
        print("cov1: matched primes:", cov1['matched_primes'])

    residues_by_prime_numeric = {}
    for p, mapping in precomputed_residues.items():
        residues_set = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        residues_set.add(r)
        residues_by_prime_numeric[p] = residues_set

    usable_primes = [p for p in prime_pool if p in residues_by_prime_numeric and residues_by_prime_numeric[p]]
    if not usable_primes:
        print("No primes have numeric precomputed residues. Aborting.")
        return {
            "candidates": [],
            "candidate_xs": set(),
            "new_sections": [],
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": [],
        }
    if len(usable_primes) < len(prime_pool):
        if debug:
            print(f"[filter] Removed {len(prime_pool) - len(usable_primes)} primes with no numeric data. Using {len(usable_primes)} usable primes.")
        prime_pool = usable_primes

    stats.start_phase('autotune_primes')
    prime_stats = estimate_prime_stats(prime_pool, precomputed_residues, vecs_list, num_rhs=len(rhs_list))
    auto_extra_primes = choose_extra_primes(
        prime_stats,
        target_density=EXTRA_PRIME_TARGET_DENSITY,
        max_extra=EXTRA_PRIME_MAX,
        skip_small=EXTRA_PRIME_SKIP
    )
    extra_primes_for_filtering = auto_extra_primes
    stats.end_phase('autotune_primes')

    combo_cap = ceil(50000**(7*min_prime_subset_size/3))
    roots_threshold = ROOTS_THRESHOLD
    if debug:
        print("combo_cap:", combo_cap, "roots_threshold:", roots_threshold)

    PR_m = PolynomialRing(QQ, 'm')
    try:
        Delta_poly = cd.discriminant if hasattr(cd, 'discriminant') else (-16 * (4 * cd.a4**3 + 27 * cd.a6**2))
        if hasattr(Delta_poly, 'numerator'):
            Delta_poly = Delta_poly.numerator()
        Delta_pr = PR_m(SR(Delta_poly))
    except Exception as e:
        print(f"[WARNING] Could not compute Delta_pr: {e}")
        Delta_pr = None
        raise

    predicted_qc_ratio = None
    if Delta_pr is not None:
        prime_sample = prime_pool[:min(30, len(prime_pool))]
        predicted_qc_ratio = predict_qc_distribution(Delta_pr, prime_sample, debug=debug)

    target_qc_ratio = predicted_qc_ratio if predicted_qc_ratio is not None else 1.2
    print(f"[QC Target] Using QC ratio: {target_qc_ratio:.3f} ({'predicted' if predicted_qc_ratio else 'default'})")

    collision_primes = []
    if hasattr(stats, 'rejected_primes'):
        collision_primes = [p for p, reason in stats.rejected_primes if 'collision' in str(reason)]

    density_count = 0
    total_pairs = 0
    for p, mapping in precomputed_residues.items():
        for v_tuple in vecs_list:
            total_pairs += 1
            v_tuple_normalized = tuple(v_tuple)
            roots_lists = mapping.get(v_tuple_normalized, [])
            has_roots = any(roots for roots in roots_lists)
            if has_roots:
                density_count += 1

    empirical_density = density_count / total_pairs if total_pairs > 0 else 0.08

    fiber_collision_fraction = len(collision_primes) / len(prime_pool) if prime_pool else 0.0
    num_subsets_adaptive = compute_adaptive_num_subsets(
        fiber_collision_fraction,
        avg_density=empirical_density,
        target_coverage=0.40,
        base_subsets=num_subsets
    )

    print(f"[Adaptive] Fiber collisions: {len(collision_primes)}/{len(prime_pool)} ({100*fiber_collision_fraction:.1f}%)")
    print(f"[Adaptive] Empirical density: {empirical_density:.4f}")
    print(f"[Adaptive] Recommended NUM_SUBSETS: {num_subsets_adaptive} (configured: {num_subsets})")

    num_subsets_to_use = max(num_subsets, num_subsets_adaptive)

    stats.start_phase('gen_subsets')
    prime_subsets_initial = generate_biased_prime_subsets_by_coverage_v2(
        prime_pool=prime_pool,
        precomputed_residues=precomputed_residues,
        vecs=vecs_list,
        rhs_list=rhs_list,
        num_subsets=num_subsets_to_use,
        min_size=min_prime_subset_size,
        max_size=min_max_prime_subset_size,
        combo_cap=combo_cap,
        seed=SEED_INT,
        force_full_pool=False,
        debug=debug,
        use_qc_bias=True,
        target_qc_ratio=target_qc_ratio
    )

    stats.incr('subsets_generated_initial', n=len(prime_subsets_initial))

    filtered_subsets = []
    for subset in prime_subsets_initial:
        est = 1
        is_viable = True
        for p in subset:
            residues_set = residues_by_prime_numeric.get(p, set())
            roots_count = len(residues_set)
            if roots_count == 0:
                is_viable = False
                break
            if roots_count > roots_threshold:
                est *= roots_count
                if est > combo_cap:
                    is_viable = False
                    break
            else:
                est *= max(1, roots_count)
                if est > combo_cap:
                    is_viable = False
                    break
        if is_viable and est <= combo_cap:
            filtered_subsets.append(subset)

    filtered_out_count = len(prime_subsets_initial) - len(filtered_subsets)
    stats.incr('subsets_filtered_out_combo', n=filtered_out_count)
    if debug:
        print("Generated", len(prime_subsets_initial), "prime_subsets -> filtered to", len(filtered_subsets))

    prime_subsets_to_process = filtered_subsets
    stats.prime_subsets = prime_subsets_to_process

    if TARGETED_X:
        assert matched_subset is None or matched_subset in prime_subsets_to_process, (prime_subsets_to_process, matched_subset)

    count_subsets = {}
    for subset in prime_subsets_to_process:
        key = len(subset)
        if key in count_subsets:
            count_subsets[key] += 1
        else:
            count_subsets[key] = 0

    for key in sorted(list(count_subsets)):
        print("using", count_subsets[key], "subsets of len =", key)

    if not prime_subsets_to_process:
        if debug:
            print("[fallback] coverage-based filtering removed all subsets. Building deterministic fallback subsets.")
        fallback = []
        max_k = min(6, len(prime_pool))
        for k in range(3, max_k + 1):
            for comb in combinations(prime_pool, k):
                good = True
                for p in comb:
                    if not residues_by_prime_numeric.get(p):
                        good = False
                        break
                if not good:
                    continue
                est = 1
                for p in comb:
                    est *= max(1, len(residues_by_prime_numeric[p]))
                    if est > combo_cap:
                        good = False
                        break
                if good:
                    fallback.append(list(comb))
                if len(fallback) >= max(1, num_subsets):
                    break
            if len(fallback) >= max(1, num_subsets):
                break
        if fallback:
            prime_subsets_to_process = fallback[:num_subsets]
            if debug:
                print(f"[fallback] Using {len(prime_subsets_to_process)} deterministic fallback subsets.")
        else:
            print("No viable prime subsets generated or remaining after filtering. Aborting.")
            stats.end_phase('gen_subsets')
            print("\n--- Search Statistics (No Subsets) ---")
            print(stats.summary_string())
            return {
                "candidates": [],
                "candidate_xs": set(),
                "new_sections": [],
                "precomputed_residues": precomputed_residues,
                "stats": stats,
                "final_rational_pairs": [],
            }

    stats.end_phase('gen_subsets')

    stats.start_phase('search_subsets_and_check')
    worker_func = partial(
        process_prime_subset_precomputed,
        vecs=search_vecs,
        r_m=r_m,
        shift=shift,
        tmax=tmax,
        combo_cap=combo_cap,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        num_rhs_fns=len(rhs_list),
        coeffs_genus2=coeffs_genus2
    )

    subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
        prime_subsets_to_process, worker_func, num_workers=num_workers, debug=debug
    )

    stats.crt_classes_tested = all_crt_classes
    coverage_estimator.tested_classes = all_crt_classes
    coverage_report = coverage_estimator.estimate_coverage(prime_subsets_to_process)

    if debug:
        print("\n--- Coverage Estimate ---")
        if coverage_report.get('direct_coverage') is not None:
            print(f"  Direct coverage: {100 * coverage_report['direct_coverage']:.2f}%")
        if coverage_report.get('birthday_coverage') is not None:
            print(f"  Birthday estimate: {100 * coverage_report['birthday_coverage']:.2f}%")
        print(f"  Heuristic (density): {100 * coverage_report.get('heuristic_coverage', 0):.4f}%")
        print(f"  CRT classes tested: {coverage_report.get('classes_tested', 0):,}")
        print(f"  Search space size: ~{coverage_report.get('space_size_estimate', 0):.2e}")
        additional_runs = coverage_estimator.recommend_additional_runs(prime_subsets_to_process, target_coverage=0.95)
        if additional_runs > 0:
            print(f"  ⚠️  Recommend {additional_runs} more run(s) to reach 95% coverage")

    stats.merge_dict(worker_stats_dict)
    stats.incr('subsets_processed', n=len(subset_results_list))

    overall_found_candidates_from_workers = set()
    productive_subsets_data = []
    for subset, candidates_set, _ in subset_results_list:
        overall_found_candidates_from_workers.update(candidates_set)
        if candidates_set:
            productive_subsets_data.append({
                'primes': subset,
                'size': len(subset),
                'candidates': len(candidates_set)
            })

    stats.incr('crt_candidates_found', n=len(overall_found_candidates_from_workers))

    print(f"\nChecking rationality for {len(overall_found_candidates_from_workers)} unique candidates...")
    final_rational_candidates = set()
    candidate_list = list(overall_found_candidates_from_workers)
    if not candidate_list:
        stats.end_phase('search_subsets_and_check')
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return {
            "candidates": [],
            "candidate_xs": set(),
            "new_sections": [],
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": [],
        }

    batch_size = max(1, floor(0.05 * len(candidate_list)))
    for i in range(0, len(candidate_list), batch_size):
        batch = candidate_list[i:i + batch_size]
        newly_rational = _batch_check_rationality(
            batch, r_m, shift, rationality_test_func, current_sections, stats
        )
        final_rational_candidates.update(newly_rational)
        if debug:
            print(f"[batch check] processed {min(i + batch_size, len(candidate_list))}/{len(candidate_list)}, found {len(final_rational_candidates)} rational so far")

    stats.end_phase('search_subsets_and_check')

    try:
        print_subset_productivity_stats(productive_subsets_data, prime_subsets_to_process)
    except Exception as e:
        if debug:
            print(f"Failed to print productivity stats: {e}")
        raise

    if not final_rational_candidates:
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return {
            "candidates": [],
            "candidate_xs": set(),
            "new_sections": [],
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": [],
        }

    print(f"\nFound {len(final_rational_candidates)} rational (m, vector) pairs after checking.")

    stats.start_phase('post_process')

    candidate_records = []
    candidate_xs = set()
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in final_rational_candidates:
        if m_val in processed_m_vals:
            continue
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                x_val_q = QQ(x_val)
                v = vector(QQ, v_tuple)
                candidate_records.append({
                    "m": m_val,
                    "xj": x_val_q,
                    "y": y_val,
                    "v": tuple(v_tuple),
                    "section": None,
                })
                processed_m_vals[m_val] = v
                candidate_xs.add(x_val_q)
                if any(c != 0 for c in v):
                    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                    new_sections_raw.append(new_sec)
                    candidate_records[-1]["section"] = new_sec
        except (TypeError, ZeroDivisionError, ArithmeticError):
            raise
            continue

    analysis = analyze_unused_residue_orders(
        precomputed_residues=precomputed_residues,
        rhs_list=rhs_list,
        found_m_set=processed_m_vals,
        prime_pool=prime_pool,
        max_lift_k=3,
        Delta_pr=Delta_pr,
        Ep_dict=Ep_dict
    )

    print_residue_analysis(analysis)

    new_sections = list({s: None for s in new_sections_raw}.keys())
    stats.incr('rational_points_unique', n=len(candidate_xs))
    stats.incr('new_sections_unique', n=len(new_sections))
    stats.end_phase('post_process')

    print("\n--- Search Statistics ---")
    print(stats.summary_string())

    return {
        "candidates": candidate_records,
        "candidate_xs": candidate_xs,
        "new_sections": new_sections,
        "precomputed_residues": precomputed_residues,
        "stats": stats,
        "final_rational_pairs": list(final_rational_candidates),
    }

def run_mumford_search(cd, current_sections, prime_pool, vecs, rhs_list, shift,
                      rationality_test_func, coeffs_genus2, tower_data,
                      num_workers, debug, x_b, shifted_coeffs,
                      markov_mode=False):
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
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
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

    mumford_residues = mumford_precompute_residues_parallel(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list,
        num_workers=num_workers,
        debug=debug
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
