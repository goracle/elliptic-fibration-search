from .search_config import *
from .archimedean_optim import *
from .rational_arithmetic import *
from .search_analysis import *
from .modularthread import *
from .modularthread import _compute_residues_for_prime_worker_old
from .modularthread import _compute_residues_for_prime_worker
from .modularthread import _process_prime_subset_precomputed
from .modularthread import _batch_check_rationality
from .ll_utilities import *
from .diagnostics_univariate import *
from collections import namedtuple, Counter # <-- IMPORTED COUNTER
from .mumford import *
from .mumford import analyze_active_dead_vectors
from .selmer_genus2 import *
# After your Mumford search in FINITE_FIELD mode:
from .smoothness import *
from .smoothness import extract_factor_base
from .index_calculus import *
from sage.all import QQ, PolynomialRing
from sage.all import PolynomialRing, SR, QQ
from .riemann_roch_localization import localize_target_via_rr
from search_common import BASE_DIVISOR, TARGET_DIVISOR, SECRET_KEY

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


def search_lattice_modp_unified_parallel(cd, current_sections, prime_pool, height_bound,
                                         vecs, rhs_list, r_m, shift,
                                         all_found_x, num_subsets, rationality_test_func,
                                         sconf, coeffs_genus2,
                                         tower_data=None,  # <-- ADD THIS
                                         num_workers=20, debug=DEBUG,
                                         precomputed_residues=None):
    """
    Unified parallel search using ProcessPoolExecutor throughout.
    Hardened against the "filtered to 0 subsets" failure:
      - require primes to have actual residues (not just empty mappings)
      - compute numeric residue sets per-prime and use those counts for combo estimates
      - fall back deterministically if coverage-based generator returns nothing
    Returns: new_xs, new_sections, precomputed_residues, stats
    """
    
    # === CHECK FOR MUMFORD MODE ===
    USE_MUMFORD = (
        globals().get('MUMFORD_SEARCH', False) and 
        tower_data is not None
    )
    #print("USE_MUMFORD:", globals().get('MUMFORD_SEARCH', False),MUMFORD_AVAILABLE,tower_data)
    if MUMFORD_SEARCH:
        assert USE_MUMFORD
    
    if USE_MUMFORD:

        print("\n" + "="*70)
        print("MUMFORD JACOBIAN ELEMENT SEARCH MODE")
        print("Searching for (s,p,v_0,v_1) instead of rational m")
        print("="*70 + "\n")
        
        stats = SearchStats()
        
        # Build equation system
        stats.start_phase('mumford_setup')
        print("Building 5-equation Mumford system from fibration...")
        eqs_dict = build_mumford_equations_from_fibration(tower_data, coeffs_genus2)
        stats.end_phase('mumford_setup')
        
        # Prepare modular data
        # NOTE: For Finite Field consensus mode, prime_pool must contain the target FF primes!
        stats.start_phase('prep_mod_data')
        Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
            cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=prime_pool
        )
        stats.end_phase('prep_mod_data')
        
        if not Ep_dict:
            return set(), [], {}, stats
        
        # Ensure 'prime_list' is the full list of valid primes from the pool
        prime_list = sorted(list(Ep_dict.keys()))
        
        # [MODIFIED] Removed the "Forcing search to single prime" block.
        # We now pass the entire list of compatible primes to precompute and reconstruction.
        if FINITE_FIELD and len(prime_list) > 1:
            print(f"[FF MODE] Multi-prime consensus enabled. Using {len(prime_list)} primes.")
        
        stats.start_phase('mumford_residues')
        vecs_list = list(vecs)
        
        mumford_residues = mumford_precompute_residues_parallel(
            eqs_dict, 
            prime_list, 
            Ep_dict, 
            mult_lll, 
            vecs_lll,
            rhs_modp_list, 
            vecs_list, 
            num_workers=num_workers, 
            debug=True
        )
        stats.end_phase('mumford_residues')

        if FINITE_FIELD:
            # 2. Analyze vector activity for a specific prime
            # Ensure vecs_lll is a list of tuples
            vecs_list_for_p_safe = [tuple(v) if hasattr(v, '__iter__') else (v,) for v in vecs_lll]

            summary, per_vec, cumulative_supports = analyze_active_dead_vectors(
                mumford_residues,
                vecs_generated_list=vecs_lll,   # the canonical vectors you generated (80 in your example)
                vecs_list_for_p=vecs_list_for_p_safe,       # the same list passed to the parallel routine
                prime=FINITE_FIELD                       # the prime you’re analyzing
            )

            # 3. Optionally, print or plot
            print(summary)
            pass

        
        # Reconstruct (Consensus Strategy)
        stats.start_phase('mumford_reconstruction')
        
        # We pass the full prime_list. The reconstruction function now iterates them all.
        found_xs, mumford_divisors = reconstruct_and_verify_mumford(
            mumford_residues, prime_list, coeffs_genus2, shift, rationality_test_func
        )
        stats.end_phase('mumford_reconstruction')

        print(f"\nMumford search reconstructed {len(mumford_divisors)} divisors")
        if mumford_divisors:
            rational_roots_count = sum(1 for div in mumford_divisors 
                                      if 'has_rational_roots' in div and div.get('has_rational_roots'))
            print(f"  {rational_roots_count} divisors had rational roots in u(x)")
            
            # Note: diagnose_finite_field_search expects a single prime for matrix analysis.
            # If we are in multi-prime mode, that analysis is less relevant (or needs refactoring).
            # For now, we skip detailed matrix diagnostics if multiple primes are used.
            if FINITE_FIELD:
                if len(prime_list) == 1:
                    from .smoothness import diagnose_finite_field_search
                    diagnose_finite_field_search(mumford_divisors, verbose=True)
                else:
                    print("[Info] Skipping single-prime matrix diagnostics (Multi-prime mode active).")

        if not FINITE_FIELD:
            print(f"Mumford search found {len(found_xs)} rational points")
            stats.incr('rational_points_unique', n=len(found_xs))
        print(stats.summary_string())

        if not FINITE_FIELD:
            # === NEW: SELMER UPPER BOUND COMPARISON === (only for Q, not F_p)
            rank_analysis = analyze_genus2_rank(
                f_coeffs=coeffs_genus2,
                mumford_divisors=mumford_divisors,
                bad_primes=prime_pool[:20], 
                verbose=True
            )

            stats.counters['rank_lower_bound'] = rank_analysis['lower_bound']
            stats.counters['rank_upper_bound'] = rank_analysis['upper_bound']
            stats.counters['rank_exact'] = 1 if rank_analysis['exact'] else 0

        if FINITE_FIELD:

            p = int(FINITE_FIELD)
            # 1. Extract the factor base
            fb = extract_factor_base(mumford_divisors, p)
            # FIX: Define the mapping to avoid NameError
            root_to_idx = {r: i for i, r in enumerate(fb)}
            fb_roots_set = set(root_to_idx.keys())

            # 2. Compute Jacobian Order
            L = compute_jacobian_order(coeffs_genus2, p)

            # 3. Setup the challenge
            f_poly = PolynomialRing(GF(p), 'x')(list(reversed(list(coeffs_genus2))))
            print(f"  [Setup] Curve: y^2 = {f_poly}")
            #G, Q, true_d = generate_test_keypair(f_poly, p)
            G, Q, true_d = BASE_DIVISOR, TARGET_DIVISOR, SECRET_KEY

            # NEW: RR-Local Pre-check
            print(f"  [Phase 0] Attempting RR-Localization for target Q...")
            
            # Try n_pole = 6, 7, 8 for increasing neighborhood depth
            for n in [6, 7, 8]:
                roots, poly_a, poly_b, vec = localize_target_via_rr(Q, fb_roots_set, f_poly, p, n_pole=n)
                if roots is not None:
                    print(f"  [!] Phase 0 Success: Target Q decomposed via RR(n={n})!")
                    # Convert roots to log_v directly
                    log_v = resolve_log_from_rr_decomposition(roots, root_to_idx, poly_a, poly_b, p)
                    print(f"  [!] SUCCESS: Discrete Log recovered via geometric corridor: {log_v}")
                    return found_xs, [], mumford_residues, stats

            print("  [Phase 0] RR-Localization did not find a short relation. Falling back to Index Calculus.")

            # 4. Solve the system using perform_dlp_attack
            try:
                log_v = perform_dlp_attack(G, Q, mumford_divisors, p, coeffs_genus2, L, verbose=True)
                print(f"✓ Confirmed Discrete Log: {log_v}")
            except Exception as e:
                print(f"Attack failed: {e}")
                raise

        return found_xs, [], mumford_residues, stats

    # === UNPACK: SCONF ===
    min_prime_subset_size = sconf['MIN_PRIME_SUBSET_SIZE']
    min_max_prime_subset_size = sconf['MIN_MAX_PRIME_SUBSET_SIZE']
    max_modulus = sconf['MAX_MODULUS']
    tmax = sconf['TMAX']

    # === VECTOR-BLIND CONSENSUS OPTIMIZATION ===
    # Check if we are running in the optimized Vector-Blind Consensus mode.
    optimized_vecs = vecs
    
    # Heuristic: Check the first prime in the precomputed map (if it exists)
    # to see if the single dummy vector key (all zeros) was used.
    if precomputed_residues and precomputed_residues is not True: # precomputed_residues=True means only compute
        first_prime = next(iter(precomputed_residues), None)
        if first_prime and precomputed_residues[first_prime]:
            first_vector = next(iter(precomputed_residues[first_prime]))
            
            # The dummy vector is a tuple of zeros
            if all(v == 0 for v in first_vector):
                # We found the single dummy vector key. 
                
                dim = len(current_sections)
                zero_vec_tuple = tuple([0] * dim)
                
                # --- THIS IS THE FIX ---
                # If the dummy vector is detected, we *must* use it as the *only* search vector.
                # The original `vecs` list (which may not contain the zero vector)
                # is no longer relevant, as the residues are all mapped to this dummy key.
                optimized_vecs = [zero_vec_tuple]
                if debug:
                    print("OPTIMIZATION: Vector-Blind Consensus detected. Forcing search vectors (`vecs`) to just the zero vector key.")
                # --- END FIX ---
                        
    # The actual vectors list used for the main search loop
    search_vecs = optimized_vecs

    # === STATS: INIT ===
    stats = SearchStats()

    from bounds import compute_residue_counts_for_primes  # if not already imported
    residue_counts = compute_residue_counts_for_primes(cd, rhs_list, prime_pool, max_primes=30)
    coverage_estimator = CoverageEstimator(prime_pool, residue_counts)

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
        return set(), [], {}, stats  # <-- Return stats

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
                stats  # pass the stats object (worker ignores if not used)
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
            raise

        with ProcessPoolExecutor(**exec_kwargs) as executor:
            if TORSION_SLOPPY:
                futures = {executor.submit(_compute_residues_for_prime_worker, args): args[0] for args in args_list}
            else:
                futures = {executor.submit(_compute_residues_for_prime_worker_old, args): args[0] for args in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
                p = futures[future]
                try:
                    p_ret, mapping, local_modular_checks = future.result()
                    mapping = mapping or {}
                    precomputed_residues[p_ret] = mapping
                    total_modular_checks += int(local_modular_checks or 0)

                    # Now compute the union of numeric residues (ignore non-int markers)
                    residues_union = set()
                    for vtuple, rhs_lists in mapping.items():
                        for rl in rhs_lists:
                            # ignore sentinel markers like "DEN_ZERO" or other non-integer entries
                            for r in rl:
                                if isinstance(r, int):
                                    residues_union.add(r)

                    stats.residues_by_prime[p_ret].update(residues_union)

                    # update main counters per-prime
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
        # Use provided residues (already consensus-filtered)
        print(f"Using provided precomputed residues ({len(precomputed_residues)} primes)")
        stats.incr('using_consensus_residues', n=1)


    # ADD THIS: Populate stats.residues_by_prime from precomputed residues
    for p, p_mapping in precomputed_residues.items():
        for v_tuple, rhs_lists in p_mapping.items():
            for rhs_list in rhs_lists:
                for residue in rhs_list:
                    if isinstance(residue, (int, Integer)):
                        stats.add_residue(p, residue)

    stats.start_phase('brauer')
    from brauer import (
        estimate_completeness_probability,
        probe_algebraic_brauer_obstructions,
        m_is_locally_allowed,
    )

    report = estimate_completeness_probability(precomputed_residues, PRIME_POOL)
    print(f"[brauer] estimated survival fraction ≈ {report['estimate_survive']:.6f}")
    print(f"[brauer] estimated ruled-out fraction ≈ {report['estimate_ruled_out']:.6f}")

    mc = probe_algebraic_brauer_obstructions(precomputed_residues, PRIME_POOL, sample_size=1000)
    print(f"[brauer] Monte Carlo blocked fraction ≈ {mc['monte_carlo']['blocked_fraction_est']:.6f}")

    # optional diagnostic: test a representative m from the search range
    some_m = QQ(1)  # or any rational under study
    allowed, details = m_is_locally_allowed(some_m, precomputed_residues, PRIME_POOL)
    print(f"[brauer] example m={some_m} locally allowed? {allowed}")

    stats.end_phase('brauer')


    ##### WHY ISN'T A PARTICULAR FIBRATION FINDING A POINT?  FIND OUT HERE!

    if TARGETED_X: # comment out when not in use

        ret = diagnose_missed_point(TARGETED_X, r_m, shift, precomputed_residues, prime_pool, vecs)
        #print("ret=", ret)
        matched_subset = None
        if 'matched_primes' in ret:
            matched_subset = ret['matched_primes']

        const = r_m(m=0)
        mtarget = QQ(-1)*TARGETED_X+const

        cov1 = compute_residue_coverage_for_m(mtarget, precomputed_residues, PRIME_POOL)
        print("cov1: m = ", mtarget, " coverage:", cov1['coverage_fraction'])
        print("cov1: matched primes:", cov1['matched_primes'])


    # Build a per-prime numeric residue set for later use (and require non-empty)
    residues_by_prime_numeric = {}
    for p, mapping in precomputed_residues.items():
        residues_set = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        residues_set.add(r)
        residues_by_prime_numeric[p] = residues_set

    # Only keep primes that actually gave numeric residues (not merely empty mappings)
    usable_primes = [p for p in prime_pool if p in residues_by_prime_numeric and residues_by_prime_numeric[p]]
    if not usable_primes:
        print("No primes have numeric precomputed residues. Aborting.")
        return set(), [], precomputed_residues, stats
    if len(usable_primes) < len(prime_pool):
        if debug:
            print(f"[filter] Removed {len(prime_pool) - len(usable_primes)} primes with no numeric data. Using {len(usable_primes)} usable primes.")
        prime_pool = usable_primes

    # === PHASE: AUTOTUNE PRIMES ===
    stats.start_phase('autotune_primes')
    prime_stats = estimate_prime_stats(prime_pool, precomputed_residues, vecs_list, num_rhs=len(rhs_list))
    auto_extra_primes = choose_extra_primes(prime_stats,
                                            target_density=EXTRA_PRIME_TARGET_DENSITY,
                                            max_extra=EXTRA_PRIME_MAX,
                                            skip_small=EXTRA_PRIME_SKIP)
    extra_primes_for_filtering = auto_extra_primes
    stats.end_phase('autotune_primes')

    # Filtering stage: compute product estimate using distinct numeric residues per prime
    combo_cap = ceil(50000**(7*min_prime_subset_size/3)) # too many residues for this prime subset, too many possibilities, modular constraints are too loose
    roots_threshold = ROOTS_THRESHOLD
    if debug:
        print("combo_cap:", combo_cap, "roots_threshold:", roots_threshold)


    # In search_main.py, before analyze_unused_residue_orders
    PR_m = PolynomialRing(QQ, 'm')
    Delta_poly = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
    if hasattr(Delta_poly, 'numerator'):
        Delta_poly = Delta_poly.numerator()
    Delta_pr = PR_m(SR(Delta_poly))

    # Get Delta polynomial from cd
    PR_m = PolynomialRing(QQ, 'm')
    try:
        # cd.discriminant or cd.Delta should exist
        Delta_poly = cd.discriminant if hasattr(cd, 'discriminant') else (-16 * (4 * cd.a4**3 + 27 * cd.a6**2))
        if hasattr(Delta_poly, 'numerator'):
            Delta_poly = Delta_poly.numerator()
        Delta_pr = PR_m(SR(Delta_poly))
    except Exception as e:
        print(f"[WARNING] Could not compute Delta_pr: {e}")
        Delta_pr = None
        raise

    # *** NEW: Predict target QC ratio from discriminant structure ***
    predicted_qc_ratio = None
    if Delta_pr is not None:
        from bounds import predict_qc_distribution  # Add this function to bounds.py
        # Use first 20-30 primes from prime_pool as sample
        prime_sample = prime_pool[:min(30, len(prime_pool))]
        predicted_qc_ratio = predict_qc_distribution(Delta_pr, prime_sample, debug=debug)

    # Use predicted ratio if available, otherwise fall back to empirical 1.2
    target_qc_ratio = predicted_qc_ratio if predicted_qc_ratio is not None else 1.2
    print(f"[QC Target] Using QC ratio: {target_qc_ratio:.3f} ({'predicted' if predicted_qc_ratio else 'default'})")

    # In search_main.py, after prepare_modular_data_lll and before GEN SUBSETS phase:

    # === COMPUTE ADAPTIVE NUM_SUBSETS ===
    collision_primes = []
    if hasattr(stats, 'rejected_primes'):
        collision_primes = [p for p, reason in stats.rejected_primes 
                        if 'collision' in str(reason)]

    # CORRECTED density calculation
    density_count = 0
    total_pairs = 0
    for p, mapping in precomputed_residues.items():
        for v_tuple in vecs_list:
            total_pairs += 1
            v_tuple_normalized = tuple(v_tuple)
            roots_lists = mapping.get(v_tuple_normalized, [])
            # Has roots if any RHS function has roots for this (p, v) pair
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

    # Use the adaptive value (or keep your configured value, up to you)
    num_subsets_to_use = num_subsets_adaptive  # Or: min(num_subsets, num_subsets_adaptive)

    # Conservative: use the max of adaptive and configured
    num_subsets_to_use = max(num_subsets, num_subsets_adaptive)

    # === PHASE: GEN SUBSETS ===
    stats.start_phase('gen_subsets')
    prime_subsets_initial = generate_biased_prime_subsets_by_coverage_v2(
        prime_pool=prime_pool,
        precomputed_residues=precomputed_residues,
        vecs=vecs_list,
        rhs_list=rhs_list,
        num_subsets=num_subsets_to_use,  # <-- Use adaptive value
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
            # if any single prime has more residues than the threshold, it's likely to explode
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

    #### if missing a point, assert your matched subset is contained in the used ones
    if TARGETED_X: # commented out when not using/debugging
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

    # If filtering removed everything, build a deterministic fallback pool of small subsets.
    if not prime_subsets_to_process:
        if debug:
            print("[fallback] coverage-based filtering removed all subsets. Building deterministic fallback subsets.")
        from itertools import combinations
        fallback = []
        max_k = min(6, len(prime_pool))
        # prefer sizes 3..max_k
        for k in range(3, max_k + 1):
            for comb in combinations(prime_pool, k):
                # only keep combos with at least one residue per prime
                good = True
                for p in comb:
                    if not residues_by_prime_numeric.get(p):
                        good = False
                        break
                if not good:
                    continue
                # estimate as above
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
            # give up cleanly
            print("No viable prime subsets generated or remaining after filtering. Aborting.")
            stats.end_phase('gen_subsets')
            print("\n--- Search Statistics (No Subsets) ---")
            print(stats.summary_string())
            return set(), [], precomputed_residues, stats

    stats.end_phase('gen_subsets')

    # === PHASE: SEARCH & CHECK ===
    stats.start_phase('search_subsets_and_check')
    worker_func = partial(
        _process_prime_subset_precomputed,
        vecs=search_vecs,
        r_m=r_m,
        shift=shift,
        tmax=tmax,
        combo_cap=combo_cap,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,  # current (filtered) prime_pool
        num_rhs_fns=len(rhs_list),
        coeffs_genus2=coeffs_genus2 # <-- ADDED THIS ARGUMENT
    )

    subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
        prime_subsets_to_process, worker_func, num_workers=num_workers, debug=debug
    )

    # *** THIS IS THE FIX for "CRT-consistent samples: 0" ***
    # Save the collected CRT classes to the main stats object
    stats.crt_classes_tested = all_crt_classes

    # update coverage estimator
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

    # Merge worker stats collected by the manager
    stats.merge_dict(worker_stats_dict)
    stats.incr('subsets_processed', n=len(subset_results_list))

    # aggregate worker candidates
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

    # Batch check rationality
    print(f"\nChecking rationality for {len(overall_found_candidates_from_workers)} unique candidates...")
    final_rational_candidates = set()
    candidate_list = list(overall_found_candidates_from_workers)
    if not candidate_list:
        stats.end_phase('search_subsets_and_check')
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return set(), [], precomputed_residues, stats

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

    # print productivity stats
    try:
        print_subset_productivity_stats(productive_subsets_data, prime_subsets_to_process)
    except Exception as e:
        if debug:
            print(f"Failed to print productivity stats: {e}")
        raise

    if not final_rational_candidates:
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return set(), [], precomputed_residues, stats

    print(f"\nFound {len(final_rational_candidates)} rational (m, vector) pairs after checking.")

    # === PHASE: POST PROCESS ===
    stats.start_phase('post_process')
    sample_pts = []
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in final_rational_candidates:
        if m_val in processed_m_vals:
            continue
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                v = vector(QQ, v_tuple)
                sample_pts.append((x_val, y_val))
                processed_m_vals[m_val] = v
                if any(c != 0 for c in v):
                    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                    new_sections_raw.append(new_sec)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            raise
            continue


    # <<< MODIFIED/NEW SECTION >>>
    analysis = analyze_unused_residue_orders(
        precomputed_residues=precomputed_residues,
        rhs_list=rhs_list,
        found_m_set=processed_m_vals,
        prime_pool=prime_pool,
        max_lift_k=3,
        Delta_pr=Delta_pr,  # <-- Now actually computed
        Ep_dict=Ep_dict
    )

    print_residue_analysis(analysis)


    new_xs = {pt[0] for pt in sample_pts}
    new_sections = list({s: None for s in new_sections_raw}.keys())
    stats.incr('rational_points_unique', n=len(new_xs))
    stats.incr('new_sections_unique', n=len(new_sections))
    stats.end_phase('post_process')

    print("\n--- Search Statistics ---")
    print(stats.summary_string())

    return new_xs, new_sections, precomputed_residues, stats


