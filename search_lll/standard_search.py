"""
standard_search.py: staged implementation of the rational-point lattice search.

This is the body that used to live inline in
``search_main.run_standard_lattice_search`` (a single ~1200-line function).
``search_main.run_standard_lattice_search`` is now a thin wrapper that calls
``run_standard_lattice_search`` here, with an unchanged signature and return
value, so no caller needs to change.

Stages, in the order ``run_standard_lattice_search`` runs them
---------------------------------------------------------------
 1. ``_resolve_search_vecs``       vector-blind-consensus optimisation
 2. ``_prepare_modular_data``      reduce the fibration mod every prime
 3. ``_precompute_residues``       per-prime residues (fork pool)   [skipped if supplied]
 4. ``_run_markov_mode``           early exit for markov_mode=True
 5. ``_report_brauer_estimates``   informational Brauer/density estimates
 6. ``_discover_via_residue_graph``bottom-up CRT-graph candidate discovery
 7. ``_run_targeted_diagnostics``  TARGETED_X debugging (incl. CHEAT / BAND SWEEP)
 8. ``_filter_usable_primes`` / ``_autotune_extra_primes`` / ``_combo_cap`` /
    ``_report_adaptive_subset_count``   sweep parameter setup
 9. ``_run_anomalous_sweep``       outer loop; one ``_run_sweep_round`` per round

Where names come from
---------------------
Every collaborator is imported from ``search_main``'s own namespace rather than
from its defining module.  ``search_main`` and this package are built out of
``from x import *`` chains with several same-named duplicates (see
``_batch_check_rationality``, ``choose_extra_primes``, ...), so importing from
``search_main`` is what guarantees this module binds the *same* object the old
inline code did.  If you later de-duplicate those, retarget these imports.
"""
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations

from tqdm import tqdm
from crt_bounds import pool_m_height_bound
from .residue_signal import report_target_signal
from sage.all import QQ, Integer, PolynomialRing, SR, primes, vector

from .search_main import (
    # --- helpers that stay in search_main (shared with the Mumford path / workers) ---
    _GRAPH_WORKER_STATE,
    _record_rational_candidate,
    _resolve_height_bound,
    _run_graph_vectors,
    log10_of_naive_height,
    naive_height_of_rational,
    search_prime_subsets_unified,
    # --- search-config / global run constants ---
    CHEAT_CANDIDATE_PRIME_BOUND,
    CHEAT_FILTER_POOL_TO_TARGET_M,
    EXTRA_PRIME_MAX,
    EXTRA_PRIME_SKIP,
    EXTRA_PRIME_TARGET_DENSITY,
    HEIGHT_BOUND,
    M_HEIGHT_BOUND,
    MAX_ANOMALOUS_SWEEP_ROUNDS,
    MEASURE_COMPATIBLE_PRIME_RATE_BANDS,
    MEASURE_COMPATIBLE_PRIME_RATE_BY_BAND,
    PARALLEL_PRIME_WORKERS,
    PRIME_POOL,
    ROOTS_THRESHOLD,
    SEED_INT,
    TARGETED_X,
    TORSION_SLOPPY,
    # --- pipeline pieces ---
    CoverageEstimator,
    SearchStats,
    _batch_check_rationality,
    analyze_unused_residue_orders,
    build_cheat_prime_pool,
    choose_extra_primes,
    compute_adaptive_num_subsets,
    compute_residue_counts_for_primes,
    compute_residue_coverage_for_m,
    compute_residues_for_prime_worker,
    compute_residues_for_prime_worker_old,
    diagnose_missed_point,
    estimate_completeness_probability,
    estimate_prime_stats,
    filter_residues_by_rail,
    generate_biased_prime_subsets_by_coverage_v2,
    m_is_locally_allowed,
    predict_qc_distribution,
    prepare_modular_data_lll,
    print_residue_analysis,
    print_subset_productivity_stats,
    probe_algebraic_brauer_obstructions,
    process_prime_subset_precomputed,
    prune_explained_residues,
)


# =============================================================================
# Data carriers
# =============================================================================

@dataclass
class SearchInputs:
    """Arguments of run_standard_lattice_search that never change during a run."""
    cd: object
    current_sections: list
    vecs: object
    rhs_list: list
    r_m: object
    shift: object
    all_found_x: object
    num_subsets: int
    rationality_test_func: object
    sconf: dict
    coeffs_genus2: object
    num_workers: int
    debug: bool
    search_vecs: object
    height_pairing_H: object = None


@dataclass
class ModularData:
    """Output of prepare_modular_data_lll."""
    Ep_dict: dict
    rhs_modp_list: list
    mult_lll: dict
    vecs_lll: dict
    section_poly_dict: dict


@dataclass
class Accumulators:
    """
    Shared accumulators for every rational (m, v_tuple) candidate found by any
    method (residue-graph discovery, then each anomalous-sweep round).  Both
    record into these through _record_rational_candidate.
    """
    records: list = field(default_factory=list)
    xs: set = field(default_factory=set)
    final_pairs: list = field(default_factory=list)
    processed_m_vals: dict = field(default_factory=dict)

    def record(self, m_val, v_tuple, inp, known_x_before=None):
        """Thin wrapper over _record_rational_candidate bound to these accumulators."""
        return _record_rational_candidate(
            m_val, v_tuple, inp.r_m, inp.shift, inp.rationality_test_func,
            self.records, self.processed_m_vals, self.xs,
            known_x_before=known_x_before,
        )

    def as_result(self, precomputed_residues, stats):
        # NOTE: new_sections is always empty on the standard path: the code that
        # used to fill it from `v`-weighted sums of current_sections was disabled
        # ("this section hangs for some reason") and has been removed.
        return {
            "candidates": self.records,
            "candidate_xs": self.xs,
            "new_sections": [],
            "precomputed_residues": precomputed_residues,
            "stats": stats,
            "final_rational_pairs": self.final_pairs,
        }


@dataclass
class TargetedOutcome:
    """What the TARGETED_X debug block hands back to the main flow."""
    prime_pool: list
    precomputed_residues: dict
    matched_subset: object = None
    abort: bool = False


@dataclass
class SweepSetup:
    """Everything the anomalous-sweep rounds need that is fixed before round 0."""
    search_vecs: object
    vecs_list: list
    prep: ModularData
    combo_cap: int
    min_prime_subset_size: int
    min_max_prime_subset_size: int
    tmax: int
    target_qc_ratio: float
    num_subsets_to_use: int
    Delta_pr: object
    coverage_estimator: object
    matched_subset: object = None


# =============================================================================
# Small shared helpers (each one replaces 2-4 copy-pasted blocks)
# =============================================================================

def _empty_result(precomputed_residues, stats):
    """The 'found nothing' result dict (was written out by hand ~10 times)."""
    return {
        "candidates": [],
        "candidate_xs": set(),
        "new_sections": [],
        "precomputed_residues": precomputed_residues,
        "stats": stats,
        "final_rational_pairs": [],
    }


def _fork_executor(num_workers):
    """ProcessPoolExecutor on a fork context when available (else default)."""
    try:
        ctx = multiprocessing.get_context("fork")
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}
    return ProcessPoolExecutor(**exec_kwargs)


def _submit_residue_workers(executor, args_list):
    """Submit one per-prime residue job each; returns {future: p}."""
    worker = compute_residues_for_prime_worker if TORSION_SLOPPY else compute_residues_for_prime_worker_old
    return {executor.submit(worker, args): args[0] for args in args_list}


def _residue_worker_args(prep, vecs_list, n_sections, num_rhs_fns, stats):
    """Per-prime argument tuples for the residue workers (one per prime in prep.Ep_dict)."""
    zero_vec = tuple([0] * n_sections)
    return [
        (
            p,
            prep.Ep_dict[p],
            prep.mult_lll.get(p, {}),
            prep.vecs_lll.get(p, [zero_vec for _ in vecs_list]),
            vecs_list,
            prep.rhs_modp_list,
            num_rhs_fns,
            stats,
        )
        for p in prep.Ep_dict.keys()
    ]


def _numeric_residue_set(mapping):
    """All plain-int residues appearing anywhere in one prime's {vector: [[r, ...], ...]} mapping."""
    out = set()
    for rhs_lists in mapping.values():
        for rl in rhs_lists:
            for r in rl:
                if isinstance(r, int):
                    out.add(r)
    return out


def _numeric_residues_by_prime(precomputed_residues):
    """{p: set of int residues} view of precomputed_residues."""
    return {p: _numeric_residue_set(mapping) for p, mapping in precomputed_residues.items()}


def _check_rationality_in_batches(candidates, inp, stats, report_progress=False):
    """Run _batch_check_rationality over ~20 batches; returns the set of rational (m, v) pairs."""
    candidate_list = list(candidates)
    rational = set()
    batch_size = max(1, math.floor(0.05 * len(candidate_list)))
    for i in range(0, len(candidate_list), batch_size):
        batch = candidate_list[i:i + batch_size]
        newly_rational = _batch_check_rationality(
            batch, inp.r_m, inp.shift, inp.rationality_test_func, inp.current_sections, stats
        )
        rational.update(newly_rational)
        if report_progress:
            print(f"[batch check] processed {min(i + batch_size, len(candidate_list))}/{len(candidate_list)}, "
                  f"found {len(rational)} rational so far")
    return rational


def _make_subset_worker(inp, setup_vecs, tmax, combo_cap, residues, prime_pool):
    """The partial handed to search_prime_subsets_unified (same in markov mode and the sweep)."""
    return partial(
        process_prime_subset_precomputed,
        vecs=setup_vecs,
        r_m=inp.r_m,
        shift=inp.shift,
        tmax=tmax,
        combo_cap=combo_cap,
        precomputed_residues=residues,
        prime_pool=prime_pool,
        num_rhs_fns=len(inp.rhs_list),
        coeffs_genus2=inp.coeffs_genus2,
        height_bound=_resolve_height_bound(inp.cd, inp.current_sections, setup_vecs, inp.sconf, inp.height_pairing_H),
        bad_primes=frozenset(int(p) for p in getattr(inp.cd, 'bad_primes', []) or []),
    )


# =============================================================================
# Stages 1-3: vectors, modular data, residues
# =============================================================================

def _resolve_search_vecs(vecs, precomputed_residues, current_sections, debug):
    """
    Vector-blind consensus optimisation: if the supplied residues are keyed by
    the zero vector only, search over just the zero-vector key.
    """
    if precomputed_residues and precomputed_residues is not True:
        first_prime = next(iter(precomputed_residues), None)
        if first_prime and precomputed_residues[first_prime]:
            first_vector = next(iter(precomputed_residues[first_prime]))
            if all(v == 0 for v in first_vector):
                dim = len(current_sections)
                if debug:
                    print("OPTIMIZATION: Vector-Blind Consensus detected. Forcing search vectors (`vecs`) to just the zero vector key.")
                return [tuple([0] * dim)]
    return vecs


def _prepare_modular_data(inp, prime_pool, stats):
    stats.start_phase('prep_mod_data')
    print("--- Preparing modular data for LLL search ---")
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll, section_poly_dict = prepare_modular_data_lll(
        inp.cd, inp.current_sections, prime_pool, inp.rhs_list, inp.vecs, stats, search_primes=prime_pool
    )
    stats.end_phase('prep_mod_data')
    return ModularData(Ep_dict, rhs_modp_list, mult_lll, vecs_lll, section_poly_dict)


def _precompute_residues(inp, prep, vecs_list, stats):
    """Compute {p: {vector: [[residue, ...] per rhs]}} for every prime in prep.Ep_dict."""
    stats.start_phase('precompute_residues')
    args_list = _residue_worker_args(prep, vecs_list, len(inp.current_sections), len(inp.rhs_list), stats)

    precomputed_residues = {}
    total_modular_checks = 0

    with _fork_executor(inp.num_workers) as executor:
        futures = _submit_residue_workers(executor, args_list)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
            p = futures[future]
            try:
                p_ret, mapping, local_modular_checks = future.result()
                mapping = mapping or {}
                precomputed_residues[p_ret] = mapping
                total_modular_checks += int(local_modular_checks or 0)

                stats.residues_by_prime[p_ret].update(_numeric_residue_set(mapping))
                stats.counters['modular_checks'] += int(local_modular_checks or 0)
                stats.counters[f'modular_checks_p_{p_ret}'] += int(local_modular_checks or 0)
                stats.counters[f'residues_seen_p_{p_ret}'] = len(stats.residues_by_prime[p_ret])
            except Exception as e:
                print(f"[precompute fail] p={p}: {e}")
                raise

    if inp.debug:
        print(f"[precompute] total_modular_checks={total_modular_checks}, primes precomputed={len(precomputed_residues)}")

    stats.end_phase('precompute_residues')
    return precomputed_residues


def _add_residues_to_stats(precomputed_residues, stats):
    """Populate stats.residues_by_prime from precomputed residues."""
    for p, p_mapping in precomputed_residues.items():
        for rhs_lists in p_mapping.values():
            for rhs_list_item in rhs_lists:
                for residue in rhs_list_item:
                    if isinstance(residue, (int, Integer)):
                        stats.add_residue(p, residue)


# =============================================================================
# Stage 4: markov mode
# =============================================================================

def _run_markov_mode(inp, prep, search_vecs, prime_pool, precomputed_residues, stats):
    """
    Markov mode: stop early, no Brauer/autotune/attack plumbing.  One subset per
    prime, rationality check, compact candidate pool for transition selection.
    """
    sconf = inp.sconf
    stats.start_phase('markov_subset_search')

    prime_subsets_to_process = [[p] for p in prime_pool if p in precomputed_residues]
    if not prime_subsets_to_process:
        stats.end_phase('markov_subset_search')
        return _empty_result(precomputed_residues, stats)

    worker_func = _make_subset_worker(
        inp, search_vecs, sconf['TMAX'], sconf['MAX_MODULUS'], precomputed_residues, prime_pool)
    subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
        prime_subsets_to_process, worker_func, num_workers=inp.num_workers, debug=inp.debug
    )

    stats.merge_dict(worker_stats_dict)
    stats.crt_classes_tested = all_crt_classes
    stats.incr('subsets_processed', n=len(subset_results_list))

    found_from_workers = set()
    for subset, candidates_set, _ in subset_results_list:
        found_from_workers.update(candidates_set)
    stats.incr('crt_candidates_found', n=len(found_from_workers))

    if not found_from_workers:
        stats.end_phase('markov_subset_search')
        return _empty_result(precomputed_residues, stats)

    # Keep the rationality check, but skip all the heavier analytics.
    final_rational_candidates = _check_rationality_in_batches(found_from_workers, inp, stats)

    candidate_records = []
    candidate_xs = set()
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in final_rational_candidates:
        if m_val in processed_m_vals:
            continue
        x_val = inp.r_m(m=m_val) - inp.shift
        y_val = inp.rationality_test_func(x_val)
        if y_val is None:
            continue

        x_val_q = QQ(x_val)
        v = vector(QQ, v_tuple)
        if x_val_q in inp.all_found_x:
            continue

        rec = {"m": m_val, "xj": x_val_q, "y": y_val, "v": tuple(v_tuple), "section": None}
        if any(c != 0 for c in v):
            new_sec = sum(v[i] * inp.current_sections[i] for i in range(len(inp.current_sections)))
            rec["section"] = new_sec
            new_sections_raw.append(new_sec)

        candidate_records.append(rec)
        processed_m_vals[m_val] = v
        candidate_xs.add(x_val_q)

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


# =============================================================================
# Stage 5: Brauer estimates (informational)
# =============================================================================

def _report_brauer_estimates(precomputed_residues, stats):
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


# =============================================================================
# Stage 6: residue-CRT-graph discovery
# =============================================================================

def _known_point_trace_ms(inp, mtarget_known):
    """
    m-values of every already-known x (m = -x + r_m(0) - shift, valid because r_m is
    linear here -- see the m_map_height_factor note in _resolve_height_bound),
    plus the configured target's m.  Reporting only; never steers the search.
    """
    trace_ms = []
    try:
        # x = r_m(m) - shift with r_m linear of slope -1, so m = r_m(0) - shift - x.
        r_m0 = QQ(inp.r_m(m=0)) - QQ(inp.shift)
        for kx in list(inp.all_found_x):
            trace_ms.append(QQ(-1) * QQ(kx) + r_m0)
    except Exception as e:
        print(f"[residue_graph] could not derive m for known x's ({e}); tracing disabled")
    if mtarget_known is not None and mtarget_known not in trace_ms:
        trace_ms.append(mtarget_known)
    trace_ms = list(dict.fromkeys(trace_ms))
    print(f"[residue_graph] tracing {len(trace_ms)} known m-value(s): {trace_ms}")
    return trace_ms


def _print_graph_scoreboard(t_graph0, found_by_vector, trace_ms, trace_summary, graph_vecs):
    """Which vectors recovered which known m."""
    print(f"\n[residue_graph] ===== SCOREBOARD ({time.time() - t_graph0:.0f}s total) =====")
    print("[residue_graph] points recovered by the graph:")
    for x, vs in found_by_vector.items():
        uniq = list(dict.fromkeys(vs))
        print(f"[residue_graph]   x={x}: {len(vs)} chain hit(s) across "
              f"{len(uniq)} vector(s) {uniq[:8]}{'...' if len(uniq) > 8 else ''}")
    for m in trace_ms:
        rows = [(v, trace_summary[(m, v)]) for v in graph_vecs if (m, v) in trace_summary]
        conf = [v for v, (st, _) in rows if st == 'confirmed']
        unconf = [v for v, (st, _) in rows if st == 'unconfirmed']
        lost = {}
        for v, (st, g) in rows:
            if st == 'lost':
                lost.setdefault(g, []).append(v)
        print(f"[residue_graph]   m={m}: confirmed in {len(conf)}/{len(rows)} vectors"
              + (f" {conf[:8]}{'...' if len(conf) > 8 else ''}" if conf else "")
              + (f"; alive-but-unconfirmed in {len(unconf)}" if unconf else "")
              + (f"; lost at gen: { {g: len(vs) for g, vs in sorted(lost.items())} }" if lost else ""))
    print("[residue_graph] ===================================================\n")


def _square_den_applicable(inp):
    """
    True iff every rational m the residue graph can meet has a perfect-square
    reduced denominator, so the graph may use the square-denominator lift test
    and bound (crt_bounds.py).  Requires (all checked here, failing closed):
      * SQUARE_DENOMINATORS on, RLINEAR (x = xi - m - shift), no Mobius map;
      * xi = r_m(0) and shift integers (with a non-integral xi, xi - x can
        have a non-square denominator: 1/9 + 2/9 = 1/3);
      * the curve polynomial (inp.coeffs_genus2) integral, monic, odd degree.
    A wrong "True" would silently drop real points, so if this ever looks
    suspicious watch the known-m trace: a real point shows up as
    "LOST at gen ..." instead of CONFIRMED.
    """
    from search_common import SQUARE_DENOMINATORS, RLINEAR, MOBIUS_TRANS
    if not SQUARE_DENOMINATORS or not RLINEAR or MOBIUS_TRANS:
        return False
    try:
        xi = QQ(inp.r_m(m=0))
        sh = QQ(inp.shift)
        cs = [QQ(c) for c in inp.coeffs_genus2]
        return bool(
            xi.denominator() == 1 and sh.denominator() == 1
            and all(c.denominator() == 1 for c in cs)
            and cs and cs[0] == 1 and (len(cs) - 1) % 2 == 1
        )
    except Exception:
        return False


def _report_rail_effect(raw, filt, trace_ms, graph_vecs, top=3):
    """
    For each known m and vector: at how many primes is m mod p in the RAW
    residue domain vs. after the rail_ok filter?  raw > filtered means the
    filter is deleting true residues (it must never do that); raw < #primes
    means the residue computation itself has no root for m at those primes.
    Only the top few vectors per m are printed.
    """
    def dom(res, p, v):
        out = set()
        for rl in (res.get(p) or {}).get(v, []):
            out.update(int(a) % p for a in rl)
        return out
    for m in trace_ms or []:
        try:
            num, den = int(m.numerator()), int(m.denominator())
        except Exception:
            continue
        rows = []
        for v in graph_vecs:
            n_raw = n_filt = n_tot = 0
            dropped = []
            for p in raw:
                if den % p == 0:
                    continue
                n_tot += 1
                a = (num * pow(den, -1, p)) % p
                in_raw = a in dom(raw, p, v)
                in_filt = a in dom(filt, p, v)
                n_raw += in_raw
                n_filt += in_filt
                if in_raw and not in_filt:
                    dropped.append(p)
            rows.append((n_raw, n_filt, n_tot, v, dropped))
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        for n_raw, n_filt, n_tot, v, dropped in rows[:top]:
            print(f"[rail_ok] m={m} v={v}: residue present at {n_raw}/{n_tot} primes raw, "
                  f"{n_filt}/{n_tot} after filter"
                  + (f"  !! FILTER DROPPED TRUE RESIDUE at {dropped[:12]}" if dropped else ""))


def _discover_via_residue_graph(inp, acc, vecs_list, precomputed_residues):
    """
    Bottom-up candidate discovery via the residue CRT-consistency graph
    (search_lll/residue_crt_graph.py).  Takes no target: it only looks at
    precomputed_residues and PRIME_POOL, builds the cross-prime consistency
    graph per vector, and reports the candidate m values that fall out.
    Confirmed finds are recorded into `acc`, so they feed the same
    section/coverage/completeness bookkeeping as anomalous-sweep finds.

    Each vector is scanned independently: CRT-combining prime p's residue for
    one vector with prime q's residue for another would give numerically
    coincidental agreement that says nothing about either (this matches
    modularthread.process_prime_subset_precomputed, which fixes v_orig for the
    whole inner CRT loop of a prime subset).
    """
    # mtarget is only used to flag a match when a target is configured.
    mtarget_known = None
    if TARGETED_X:
        mtarget_known = QQ(-1) * TARGETED_X + QQ(inp.r_m(m=0)) - QQ(inp.shift)

    trace_ms = _known_point_trace_ms(inp, mtarget_known)

    # The m box the graph certifies m in.  Independent of HEIGHT_BOUND, which only
    # decides which multiples [n]P are scanned.  If M_HEIGHT_BOUND is None (the
    # default) m is NOT bounded by hand: the box is whatever the residue pool can
    # certify (crt_bounds.pool_m_height_bound: product of all pool primes that
    # carry residues).  The graph still only confirms a chain at its own H(M),
    # so this is a ceiling on reach, not a cost knob.
    square_den = _square_den_applicable(inp)
    _m_cfg = inp.sconf.get('M_HEIGHT_BOUND')
    if _m_cfg is None:
        _m_cfg = M_HEIGHT_BOUND
    _pool_with_data = [int(p) for p in PRIME_POOL
                       if precomputed_residues.get(p) or precomputed_residues.get(int(p))]
    _pool_box = pool_m_height_bound(_pool_with_data, square_den=square_den)
    if _m_cfg is None:
        m_box = int(_pool_box)
        print(f"[residue_graph] m box: derived from residue pool ({len(_pool_with_data)} primes, "
              f"log10(prod)={sum(math.log10(p) for p in _pool_with_data):.1f}): "
              f"|num|, den <= {m_box} (~10^{math.log10(max(m_box, 1)):.1f}); "
              f"[n]P scan cutoff HEIGHT_BOUND={inp.sconf.get('HEIGHT_BOUND')}")
    else:
        m_box = int(_m_cfg)
        print(f"[residue_graph] m box: M_HEIGHT_BOUND={m_box} (manual; pool could certify "
              f"{_pool_box}); [n]P scan cutoff HEIGHT_BOUND={inp.sconf.get('HEIGHT_BOUND')}")
    if m_box <= 0:
        print("[residue_graph] !!! pool cannot certify any m height (too few primes with "
              "residues); skipping graph discovery.")
        return
    if mtarget_known is not None:
        _mt = QQ(mtarget_known)
        _h_mt = max(abs(int(_mt.numerator())), abs(int(_mt.denominator())))
        if _h_mt > m_box:
            print(f"[residue_graph] !!! target m={_mt} has naive height {_h_mt} > "
                  f"m box {m_box}: it is outside the box and cannot be found this run.  "
                  f"Enlarge PRIME_POOL (or raise M_HEIGHT_BOUND if set manually).")

    graph_vecs = [tuple(v) for v in vecs_list
                  if not (len(vecs_list) > 1 and all(c == 0 for c in v))]
    t_graph0 = time.time()
    found_by_vector = {}       # x -> [vectors that produced it]
    null_tot = {'trials': 0, 'expected': 0.0, 'passed': 0, 'vectors': 0}
    trace_summary = {}         # (m, vector) -> ('confirmed'|'lost'|'unconfirmed', gen)

    # Vectors are independent, read-only graph builds, so they run in parallel.
    # Workers only compute; all recording into `acc` happens here, in vector order.
    # (_GRAPH_WORKER_STATE is search_main's dict; forked workers read it via COW.)
    _GRAPH_WORKER_STATE.clear()
    print(f"[residue_graph] square-denominator mode: {square_den}")
    # rail_ok: drop residues whose induced x can't carry a rational y
    # (G(x) a non-residue mod p).  Real points always survive; ~half of
    # everything else goes, which shrinks every clique the graph builds.
    graph_residues = filter_residues_by_rail(
        precomputed_residues, inp.coeffs_genus2, inp.shift, inp.r_m)
    _GRAPH_WORKER_STATE.update({
        'residues': graph_residues,
        'prime_pool': PRIME_POOL,
        'height_bound': m_box,          # the m box (M_HEIGHT_BOUND), not the [n]P cutoff
        'trace_ms': trace_ms or None,
        'square_den': square_den,
    })
    _report_rail_effect(precomputed_residues, graph_residues, trace_ms, graph_vecs)
    # Is there any residue signal for the traced m at all, or only chance?
    # Null = random m that passes the rail test, so count rail-passing residues per prime.
    rail_pass = {}
    try:
        from .modularthread import _kronecker_prefilter_domain
        for p in PRIME_POOL:
            p = int(p)
            if p <= 200000:
                rail_pass[p] = len(_kronecker_prefilter_domain(
                    p, range(p), inp.coeffs_genus2, inp.shift, None, inp.r_m))
    except Exception as e:
        print(f"[signal] exact rail counts unavailable ({e}); using ~(p+1)/2")
        rail_pass = {}
    report_target_signal(graph_residues, PRIME_POOL, trace_ms, graph_vecs,
                         rail_pass=rail_pass or None)
    graph_tasks = [(vi, v, vi == 1) for vi, v in enumerate(graph_vecs, 1)]
    graph_workers = max(1, min(PARALLEL_PRIME_WORKERS, len(graph_tasks)))
    print(f"[residue_graph] scanning {len(graph_tasks)} vector(s) "
          f"with {graph_workers} worker(s)")

    for res in _run_graph_vectors(graph_tasks, graph_workers):
        vi = res['index']
        v_orig_tuple = res['v']
        elapsed = time.time() - t_graph0
        eta = elapsed / vi * (len(graph_vecs) - vi)
        print(f"[residue_graph] === vector {vi}/{len(graph_vecs)} {v_orig_tuple} "
              f"| elapsed {elapsed:.0f}s | ETA ~{eta:.0f}s ===")
        print(res['log'], end="")

        n_lifted = 0
        n_on_curve = 0
        for num, den in res['ms']:
            m_val = QQ(num) / QQ(den)
            n_lifted += 1
            rec = acc.record(m_val, v_orig_tuple, inp)
            if rec is None:
                continue
            n_on_curve += 1
            acc.final_pairs.append((m_val, v_orig_tuple))
            found_by_vector.setdefault(rec['x'], []).append(v_orig_tuple)
            if rec['is_new_x']:
                h_x = naive_height_of_rational(rec['x'])
                print(f"[residue_graph] *** NEW POINT x={rec['x']}  (naive x-height h(x) ≈ {h_x:.2f}, "
                      f"~10^{log10_of_naive_height(h_x):.1f} digits)  from m={m_val}, "
                      f"vector={v_orig_tuple}  [t={elapsed:.0f}s] ***")

        _nl = res.get('null') or {}
        if _nl.get('null_expected_passes') is not None:
            null_tot['trials'] += int(_nl['null_trials'])
            null_tot['expected'] += float(_nl['null_expected_passes'])
            null_tot['passed'] += int(_nl['observed_passes'])
            null_tot['vectors'] += 1

        for t in res['trace']:
            trace_summary[(t['m'], v_orig_tuple)] = (
                ('confirmed', t['confirmed_gen']) if t.get('confirmed_gen') is not None
                else ('lost', t['lost_gen']) if t['lost_gen'] is not None
                else ('unconfirmed', None))

        print(f"[residue_graph] v={v_orig_tuple} done in {res['secs']:.1f}s: "
              f"{res['ncand']} confirmed chain(s) -> {n_lifted} reconstructed m -> "
              f"{n_on_curve} on the curve (y rational)")

    if null_tot['vectors']:
        from crt_bounds import poisson_upper_tail as _put
        _e, _o = null_tot['expected'], null_tot['passed']
        print(f"[residue_graph] NULL CHECK over {null_tot['vectors']} vector(s): "
              f"{null_tot['trials']:,} states lift-tested; {_o:,} passed vs ~{_e:,.1f} expected "
              f"if the residues carried no information about any real m "
              f"(excess {_o - _e:+,.1f}, P(>=obs | chance) = {_put(_e, _o):.2g}).")
        print("[residue_graph]   Reading it: 'expected' assumes each tested class is uniform mod M, "
              "counts every fraction as its own class (denominators coprime to M), so it is an estimate (~5-10% high); it also treats states as independent, so the p-value is indicative only; a genuine chain adds "
              "~1 pass before its children stable-stop.  Passes ~ expected => the confirmed chains are "
              "what chance alone produces; a big excess is the only thing that would indicate signal. "
              "Exact y-rationality remains the only confirmation of a point.")
    _print_graph_scoreboard(t_graph0, found_by_vector, trace_ms, trace_summary, graph_vecs)
    # (A per-target arc-consistency trace used to live here behind `and False`
    #  -- "too spammy, don't turn back on".  Removed; see
    #  residue_crt_graph.trace_target_through_arc_consistency to call it by hand.)


# =============================================================================
# Stage 7: TARGETED_X diagnostics (debugging only)
# =============================================================================

def _compute_residues_over_primes(inp, prime_list, stats, desc, fail_tag):
    """
    Run the normal modular-prep + residue precompute over an arbitrary list of
    primes (used by the CHEAT and BAND SWEEP diagnostics).  Per-prime failures
    are reported and recorded as empty, not raised.
    """
    prep = ModularData(*prepare_modular_data_lll(
        inp.cd, inp.current_sections, prime_list, inp.rhs_list, inp.vecs, stats, search_primes=prime_list
    ))
    vecs_list = list(inp.search_vecs)
    args_list = _residue_worker_args(prep, vecs_list, len(inp.current_sections), len(inp.rhs_list), stats)

    out = {}
    with _fork_executor(inp.num_workers) as executor:
        futures = _submit_residue_workers(executor, args_list)
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            p = futures[future]
            try:
                p_ret, mapping, _ = future.result()
                out[p_ret] = mapping or {}
            except Exception as e:
                print(f"[{fail_tag} precompute fail] p={p}: {e}")
                out[p] = {}
    return out


def _cheat_restrict_pool(inp, mtarget, prime_pool, precomputed_residues, stats):
    """
    DEBUG-ONLY CHEAT ("spike the ball" pipeline sanity check).

    Restrict prime_pool to primes whose residue for mtarget already matched.
    Filtering the existing small PRIME_POOL is useless for large targets (at
    most a handful of ~5000-6000 primes ever coincidentally match, and
    log10(product) tops out near 15 vs. the 40+ digits a 10^20-10^100 target
    needs), so this first WIDENS the candidate set to
    primes(CHEAT_CANDIDATE_PRIME_BOUND), recomputes residues there, and then
    filters.

    Circular by construction: it only tells you whether the CRT-lift /
    lattice-reduction / rational-reconstruction machinery is *capable* of
    recovering mtarget when handed a pool guaranteed compatible with it.  It
    says NOTHING about whether the library can find this point honestly.
    Never treat a positive result from this branch as a real discovery.

    Returns a TargetedOutcome (abort=True if even the widened pool can't clear
    the required modulus size).
    """
    log_M_required = math.log10(2 * max(abs(QQ(mtarget).numerator()), abs(QQ(mtarget).denominator()))**2)
    print(f"[CHEAT] Target requires log10(M) > {log_M_required:.2f}. "
          f"Widening candidate pool to bound={CHEAT_CANDIDATE_PRIME_BOUND} "
          f"before filtering (the original {len(prime_pool)}-prime pool cannot supply enough matches).")

    candidate_primes = list(primes(CHEAT_CANDIDATE_PRIME_BOUND))
    # Drop primes already covered by precomputed_residues so we don't recompute work we already have.
    new_candidates = [p for p in candidate_primes if p not in precomputed_residues]

    print(f"[CHEAT] Computing residues for {len(new_candidates)} additional candidate primes "
          f"(this reuses the normal precompute machinery -- may take a while for a large bound).")

    wide_precomputed_residues = dict(precomputed_residues)
    wide_precomputed_residues.update(_compute_residues_over_primes(
        inp, new_candidates, stats,
        desc="[CHEAT] Pre-computing residues over widened candidate pool", fail_tag="CHEAT"))

    cheat_pool = build_cheat_prime_pool(
        mtarget, wide_precomputed_residues, candidate_primes, debug=True
    )

    log_M_cheat = sum(math.log10(p) for p in cheat_pool) if cheat_pool else 0.0
    print(f"[CHEAT] Cheat pool capacity: log10(M) = {log_M_cheat:.2f} "
          f"(required > {log_M_required:.2f})")

    if not cheat_pool or log_M_cheat <= log_M_required:
        print(f"[CHEAT] *** Cheat pool still cannot clear the requirement even with "
              f"bound={CHEAT_CANDIDATE_PRIME_BOUND}. Raise CHEAT_CANDIDATE_PRIME_BOUND "
              f"and re-run before concluding anything about the pipeline. Aborting cheat path. ***")
        return TargetedOutcome(prime_pool, precomputed_residues, abort=True)

    print(f"[CHEAT] Restricting prime_pool from {len(prime_pool)} to "
          f"{len(cheat_pool)} primes matching target m (circular sanity check only).")
    return TargetedOutcome(
        cheat_pool,
        {p: wide_precomputed_residues[p] for p in cheat_pool if p in wide_precomputed_residues},
    )


def _measure_compatible_prime_rate_by_band(inp, mtarget, stats):
    """
    MEASUREMENT (not a cheat): run the real residue precompute over successive
    prime bands and report q = compatible/candidates per band, so the decay
    law of q (constant, 1/p, 1/p^2, ...) is measured rather than assumed.
    """
    bands = MEASURE_COMPATIBLE_PRIME_RATE_BANDS
    print(f"[BAND SWEEP] Measuring compatible-prime rate q across {len(bands)} bands: {bands}")
    band_results = []
    for (lo, hi) in bands:
        band_primes = [p for p in primes(lo, hi)]
        if not band_primes:
            print(f"[BAND SWEEP] band [{lo},{hi}) has no primes, skipping")
            continue
        band_precomputed_residues = _compute_residues_over_primes(
            inp, band_primes, stats, desc=f"[BAND SWEEP] band [{lo},{hi})", fail_tag="BAND SWEEP")

        # 'good' primes: ones we actually got residue data for (excludes
        # denom_zero/no_data cases so q isn't diluted by unrelated failures)
        cov_band = compute_residue_coverage_for_m(mtarget, band_precomputed_residues, band_primes)
        good = [p for p in band_primes if cov_band['per_prime'].get(int(p), {}).get('status') in ('matched', 'unseen')]
        compatible = cov_band['matched_primes']
        q = (len(compatible) / len(good)) if good else float('nan')
        print(f"[BAND SWEEP] band [{lo},{hi}): candidates={len(band_primes)} good={len(good)} "
              f"compatible={len(compatible)} q={q:.6g}")
        band_results.append({
            'lo': lo, 'hi': hi, 'candidates': len(band_primes), 'good': len(good),
            'compatible': len(compatible), 'q': q, 'matched_primes': compatible
        })

    print("[BAND SWEEP] summary (band, q):")
    for r in band_results:
        print(f"  [{r['lo']},{r['hi']}): q={r['q']:.6g}  ({r['compatible']}/{r['good']})")
    print("[BAND SWEEP] Compare q across bands to see whether it's roughly constant, ~1/p, ~1/p^2, or "
          "something else -- do NOT extrapolate a required candidate-pool size until this is measured.")


def _run_targeted_diagnostics(inp, stats, prime_pool, precomputed_residues):
    """
    Everything that only runs when TARGETED_X is configured.  Returns a
    TargetedOutcome whose prime_pool / precomputed_residues replace the caller's
    (the CHEAT path narrows both) and whose matched_subset is later asserted to
    be among the generated subsets.
    """
    ret = diagnose_missed_point(TARGETED_X, inp.r_m, inp.shift, precomputed_residues, prime_pool, inp.vecs)
    matched_subset = ret['matched_primes'] if 'matched_primes' in ret else None

    mtarget = QQ(-1) * TARGETED_X + QQ(inp.r_m(m=0)) - QQ(inp.shift)

    cov1 = compute_residue_coverage_for_m(mtarget, precomputed_residues, PRIME_POOL)
    print("cov1: m = ", mtarget, " coverage:", cov1['coverage_fraction'])
    print("cov1: matched primes:", cov1['matched_primes'])

    outcome = TargetedOutcome(prime_pool, precomputed_residues, matched_subset)
    if CHEAT_FILTER_POOL_TO_TARGET_M:
        cheat = _cheat_restrict_pool(inp, mtarget, prime_pool, precomputed_residues, stats)
        if cheat.abort:
            return TargetedOutcome(prime_pool, precomputed_residues, matched_subset, abort=True)
        outcome = TargetedOutcome(cheat.prime_pool, cheat.precomputed_residues, matched_subset)

    if MEASURE_COMPATIBLE_PRIME_RATE_BY_BAND:
        _measure_compatible_prime_rate_by_band(inp, mtarget, stats)

    return outcome


# =============================================================================
# Stage 8: sweep parameter setup
# =============================================================================

def _filter_usable_primes(prime_pool, precomputed_residues, debug):
    """Drop primes with no numeric residue data.  Returns the (possibly shorter) pool, or None if nothing is usable."""
    numeric = _numeric_residues_by_prime(precomputed_residues)
    usable = [p for p in prime_pool if p in numeric and numeric[p]]
    if not usable:
        return None
    if len(usable) < len(prime_pool):
        if debug:
            print(f"[filter] Removed {len(prime_pool) - len(usable)} primes with no numeric data. Using {len(usable)} usable primes.")
        return usable
    return prime_pool


def _autotune_extra_primes(prime_pool, precomputed_residues, vecs_list, rhs_list, stats):
    """
    Runs the extra-prime autotuner for its reporting side effects.
    (Its result -- the extra primes for filtering -- was assigned but never
    used by the old code, and still isn't; kept as a call so the output is
    unchanged.)
    """
    stats.start_phase('autotune_primes')
    prime_stats = estimate_prime_stats(prime_pool, precomputed_residues, vecs_list, num_rhs=len(rhs_list))
    choose_extra_primes(
        prime_stats,
        target_density=EXTRA_PRIME_TARGET_DENSITY,
        max_extra=EXTRA_PRIME_MAX,
        skip_small=EXTRA_PRIME_SKIP
    )
    stats.end_phase('autotune_primes')


def _combo_cap(min_prime_subset_size):
    """
    Sanity ceiling on the estimated number of CRT residue combinations for a
    subset (product of per-prime root counts), used to skip subsets that would
    blow up itertools.product.  Growth law is ~7/3 decimal digits of headroom
    per prime in the subset (50000**(7*k/3)), but with the EXPONENT CLAMPED at 40.

    Why clamped: with min_prime_subset_size raised well past its old 3-12 range
    (to reach large CRT moduli for high naive-height targets), the unclamped
    float**float raises OverflowError (50000**151.7 >> 1.8e308) before ceil()
    runs; and even with big-int exponentiation the cap grows to thousands of
    digits at subset sizes 60-80, which defeats the guard (it never triggers)
    and makes every `est > combo_cap` comparison pointless overhead.  Real
    per-prime root-count products for these fibrations are small (avg_roots
    ~1-3 per prime per the [galois/empirical] log lines), so a plateau around
    10^188 is still enormous headroom while staying a meaningful, cheap bound.
    """
    return 50000 ** min(40, (7 * int(min_prime_subset_size)) // 3)


def _discriminant_polynomial(cd):
    """Discriminant of the fibration as a polynomial in QQ[m]; prints and re-raises on failure."""
    PR_m = PolynomialRing(QQ, 'm')
    try:
        Delta_poly = cd.discriminant if hasattr(cd, 'discriminant') else (-16 * (4 * cd.a4**3 + 27 * cd.a6**2))
        if hasattr(Delta_poly, 'numerator'):
            Delta_poly = Delta_poly.numerator()
        return PR_m(SR(Delta_poly))
    except Exception as e:
        print(f"[WARNING] Could not compute Delta_pr: {e}")
        raise


def _target_qc_ratio(Delta_pr, prime_pool, debug):
    predicted_qc_ratio = predict_qc_distribution(Delta_pr, prime_pool[:min(30, len(prime_pool))], debug=debug)
    target = predicted_qc_ratio if predicted_qc_ratio is not None else 1.2
    print(f"[QC Target] Using QC ratio: {target:.3f} ({'predicted' if predicted_qc_ratio else 'default'})")
    return target


def _report_adaptive_subset_count(stats, prime_pool, precomputed_residues, vecs_list, num_subsets):
    """
    Prints the adaptive NUM_SUBSETS recommendation.  ADVISORY ONLY: the run
    always uses the user's configured num_subsets (the adaptive value was
    previously commented out because it didn't respect user choices).
    """
    collision_primes = []
    if hasattr(stats, 'rejected_primes'):
        collision_primes = [p for p, reason in stats.rejected_primes if 'collision' in str(reason)]

    density_count = 0
    total_pairs = 0
    for p, mapping in precomputed_residues.items():
        for v_tuple in vecs_list:
            total_pairs += 1
            roots_lists = mapping.get(tuple(v_tuple), [])
            if any(roots for roots in roots_lists):
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


# =============================================================================
# Stage 9: the anomalous-residue sweep
# =============================================================================

def _filter_viable_subsets(subsets, numeric_residues, combo_cap):
    """
    Keep subsets where every prime has at least one residue and the product of
    per-prime residue counts stays <= combo_cap.
    (The old code branched on ROOTS_THRESHOLD here but both branches did
    `est *= roots_count`, so the threshold had no effect; folded together.)
    """
    viable = []
    for subset in subsets:
        est = 1
        is_viable = True
        for p in subset:
            roots_count = len(numeric_residues.get(p, set()))
            if roots_count == 0:
                is_viable = False
                break
            est *= roots_count
            if est > combo_cap:
                is_viable = False
                break
        if is_viable:
            viable.append(subset)
    return viable


def _deterministic_fallback_subsets(prime_pool, numeric_residues, combo_cap, num_subsets):
    """Small deterministic k=3..6 subsets, used when coverage-based generation leaves nothing viable."""
    fallback = []
    max_k = min(6, len(prime_pool))
    want = max(1, num_subsets)
    for k in range(3, max_k + 1):
        for comb in combinations(prime_pool, k):
            good = True
            for p in comb:
                if not numeric_residues.get(p):
                    good = False
                    break
            if not good:
                continue
            est = 1
            for p in comb:
                est *= max(1, len(numeric_residues[p]))
                if est > combo_cap:
                    good = False
                    break
            if good:
                fallback.append(list(comb))
            if len(fallback) >= want:
                break
        if len(fallback) >= want:
            break
    return fallback


def _print_subset_size_histogram(subsets):
    # NOTE: preserved as-is from the original -- the first subset of each size
    # initialises its counter to 0, not 1, so every printed count is one too low
    # (and a size with a single subset prints 0).  Log-only; fix by using Counter.
    count_subsets = {}
    for subset in subsets:
        key = len(subset)
        if key in count_subsets:
            count_subsets[key] += 1
        else:
            count_subsets[key] = 0
    for key in sorted(list(count_subsets)):
        print("using", count_subsets[key], "subsets of len =", key)


def _print_coverage_report(coverage_estimator, coverage_report, subsets):
    print("\n--- Coverage Estimate ---")
    if coverage_report.get('direct_coverage') is not None:
        print(f"  Direct coverage: {100 * coverage_report['direct_coverage']:.2f}%")
    if coverage_report.get('birthday_coverage') is not None:
        print(f"  Birthday estimate: {100 * coverage_report['birthday_coverage']:.2f}%")
    print(f"  Heuristic (density): {100 * coverage_report.get('heuristic_coverage', 0):.4f}%")
    print(f"  CRT classes tested: {coverage_report.get('classes_tested', 0):,}")
    print(f"  Search space size: ~{coverage_report.get('space_size_estimate', 0):.2e}")
    additional_runs = coverage_estimator.recommend_additional_runs(subsets, target_coverage=0.95)
    if additional_runs > 0:
        print(f"  ⚠️  Recommend {additional_runs} more run(s) to reach 95% coverage")


def _record_round_results(inp, acc, final_rational_candidates, sweep_round):
    """
    Fold a round's rational (m, v) pairs into `acc`, classify each as new vs
    refound, print the round report; returns the list of NEW (x, y) points.
    """
    # Snapshot what was known *before* this round touches anything.  is_new_x
    # must compare against this frozen snapshot, not the live acc.xs --
    # otherwise the first occurrence of a point within this very round marks it
    # "known", and a second (m, v) pair in the SAME round landing on the same x
    # gets miscategorized as "refound from an earlier round" when it's really a
    # duplicate discovered twice in one round.
    known_x_before_round = set(acc.xs)

    round_new_points = []       # (x, y) newly discovered this round
    round_repeat_points = []    # (x, y) rediscovered this round (already known)
    round_resolve_failures = 0  # pairs that failed the rationality re-check (shouldn't normally happen)

    for m_val, v_tuple in final_rational_candidates:
        rec = acc.record(m_val, v_tuple, inp, known_x_before=known_x_before_round)
        if rec is None:
            round_resolve_failures += 1
            continue

        x_val_q, y_val, is_new_x = rec['x'], rec['y'], rec['is_new_x']
        if is_new_x:
            round_new_points.append((x_val_q, y_val))
        else:
            round_repeat_points.append((x_val_q, y_val))

        if rec['already_known_m']:
            # Same m rediscovered (by the residue-graph step or an earlier round):
            # already recorded, but it still counts as "resolved this round".
            continue

        if is_new_x:
            h_x = naive_height_of_rational(x_val_q)
            print(f"[height] new point x={x_val_q}  (naive x-height h(x) ≈ {h_x:.2f}, "
                  f"~10^{log10_of_naive_height(h_x):.1f} digits)  from m={m_val}, multiplier v={tuple(v_tuple)}")
        # (Section reconstruction from `v` used to live here behind `and False`
        #  -- "this section hangs for some reason".  Removed; records keep section=None.)

    round_resolved_total = len(round_new_points) + len(round_repeat_points)
    print(f"[anomalous-sweep] Round {sweep_round} affine points: "
          f"{round_resolved_total} resolved from {len(final_rational_candidates)} (m, vector) pair(s)"
          + (f" ({round_resolve_failures} failed re-check)" if round_resolve_failures else "") + ".")

    if round_new_points:
        pts_str = ", ".join(f"({x}, {y})" for x, y in sorted(set(round_new_points)))
        print(f"[anomalous-sweep] Round {sweep_round}: {len(set(round_new_points))} NEW point(s): {pts_str}")
    else:
        print(f"[anomalous-sweep] Round {sweep_round}: 0 new points.")

    if round_repeat_points:
        pts_str = ", ".join(f"({x}, {y})" for x, y in sorted(set(round_repeat_points)))
        print(f"[anomalous-sweep] Round {sweep_round}: {len(set(round_repeat_points))} point(s) REFOUND (already known): {pts_str}")
    else:
        print(f"[anomalous-sweep] Round {sweep_round}: 0 points refound.")

    return round_new_points


def _run_sweep_round(inp, setup, acc, stats, prime_pool, precomputed_residues, sweep_round):
    """
    One round: prune explained residues, generate + filter prime subsets, CRT
    search them, rationality-check, record.

    Every round re-runs exactly the same search over the same full prime_pool;
    the only thing that changes is that roots reducing to an already-found m
    (mod p) are pruned first, so explained residues can't keep re-surfacing.

    Returns (round_new_points, aborted).  aborted=True means no viable subset
    could be built at all (statistics already printed).
    """
    debug = inp.debug

    residues_this_round = prune_explained_residues(
        precomputed_residues, acc.processed_m_vals, prime_pool=prime_pool
    )
    # Recompute the numeric view from THIS round's pruned residues, or a subset
    # whose only residues were just explained would still look viable.
    numeric_this_round = _numeric_residues_by_prime(residues_this_round)
    residues_remaining = sum(len(s) for s in numeric_this_round.values())

    print(f"\n{'='*70}")
    print(f"[anomalous-sweep] Round {sweep_round}"
          + (f" | re-running full prime_pool with {len(acc.processed_m_vals)} known m-value(s) pruned from residues "
             f"({residues_remaining} residue(s) remaining across the pool)"
             if acc.processed_m_vals else f" | baseline (round 0: nothing found yet, no pruning; {residues_remaining} residue(s) in the pool)"))
    print(f"{'='*70}")

    # ---- generate + filter subsets -------------------------------------
    stats.start_phase('gen_subsets')
    prime_subsets_initial = generate_biased_prime_subsets_by_coverage_v2(
        prime_pool=prime_pool,
        precomputed_residues=residues_this_round,
        vecs=setup.vecs_list,
        rhs_list=inp.rhs_list,
        num_subsets=setup.num_subsets_to_use,
        min_size=setup.min_prime_subset_size,
        max_size=min(setup.min_max_prime_subset_size, len(prime_pool)),
        combo_cap=setup.combo_cap,
        seed=SEED_INT + sweep_round,
        force_full_pool=False,
        debug=debug,
        use_qc_bias=True,
        target_qc_ratio=setup.target_qc_ratio
    )
    stats.incr('subsets_generated_initial', n=len(prime_subsets_initial))

    prime_subsets_to_process = _filter_viable_subsets(prime_subsets_initial, numeric_this_round, setup.combo_cap)
    stats.incr('subsets_filtered_out_combo', n=len(prime_subsets_initial) - len(prime_subsets_to_process))
    if debug:
        print("Generated", len(prime_subsets_initial), "prime_subsets -> filtered to", len(prime_subsets_to_process))

    # extend() in place: the old list(...) + list(...) form copied the whole
    # growing history every round (O(n^2) over the sweep).
    stats.prime_subsets.extend(prime_subsets_to_process)

    if TARGETED_X:
        assert setup.matched_subset is None or setup.matched_subset in prime_subsets_to_process, \
            (prime_subsets_to_process, setup.matched_subset)

    _print_subset_size_histogram(prime_subsets_to_process)

    if not prime_subsets_to_process:
        if debug:
            print("[fallback] coverage-based filtering removed all subsets. Building deterministic fallback subsets.")
        fallback = _deterministic_fallback_subsets(prime_pool, numeric_this_round, setup.combo_cap, inp.num_subsets)
        if fallback:
            prime_subsets_to_process = fallback[:inp.num_subsets]
            if debug:
                print(f"[fallback] Using {len(prime_subsets_to_process)} deterministic fallback subsets.")
        else:
            print("No viable prime subsets generated or remaining after filtering. Aborting.")
            stats.end_phase('gen_subsets')
            print("\n--- Search Statistics (No Subsets) ---")
            print(stats.summary_string())
            return [], True

    stats.end_phase('gen_subsets')

    # ---- CRT search over the subsets -----------------------------------
    stats.start_phase('search_subsets_and_check')
    worker_func = _make_subset_worker(
        inp, setup.search_vecs, setup.tmax, setup.combo_cap, residues_this_round, prime_pool)
    subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
        prime_subsets_to_process, worker_func, num_workers=inp.num_workers, debug=debug
    )

    # in-place union: the old set(...) | set(...) rebuilt a full copy of the
    # accumulated set every round, doubling peak memory for an already-large structure.
    stats.crt_classes_tested |= set(all_crt_classes)
    setup.coverage_estimator.tested_classes = stats.crt_classes_tested
    coverage_report = setup.coverage_estimator.estimate_coverage(prime_subsets_to_process)
    if debug:
        _print_coverage_report(setup.coverage_estimator, coverage_report, prime_subsets_to_process)

    stats.merge_dict(worker_stats_dict)
    stats.incr('subsets_processed', n=len(subset_results_list))

    found_from_workers = set()
    productive_subsets_data = []
    for subset, candidates_set, _ in subset_results_list:
        found_from_workers.update(candidates_set)
        if candidates_set:
            productive_subsets_data.append({
                'primes': subset,
                'size': len(subset),
                'candidates': len(candidates_set)
            })
    stats.incr('crt_candidates_found', n=len(found_from_workers))

    # ---- rationality check ---------------------------------------------
    print(f"\n[anomalous-sweep] Round {sweep_round}: checking rationality for {len(found_from_workers)} unique candidates...")
    final_rational_candidates = _check_rationality_in_batches(found_from_workers, inp, stats, report_progress=debug)
    stats.end_phase('search_subsets_and_check')

    try:
        print_subset_productivity_stats(productive_subsets_data, prime_subsets_to_process)
    except Exception as e:
        if debug:
            print(f"Failed to print productivity stats: {e}")
        raise

    acc.final_pairs.extend(final_rational_candidates)

    # ---- record --------------------------------------------------------
    round_new_points = []   # stays [] if no candidates at all
    if final_rational_candidates:
        print(f"\nFound {len(final_rational_candidates)} rational (m, vector) pairs after checking.")
        stats.start_phase('post_process')
        round_new_points = _record_round_results(inp, acc, final_rational_candidates, sweep_round)
        stats.end_phase('post_process')
    else:
        print(f"[anomalous-sweep] Round {sweep_round} affine points: 0 (no (m, vector) pairs survived the rationality check).")
        print("\n--- No new rational points this round ---")

    return round_new_points, False


def _run_anomalous_sweep(inp, setup, acc, stats, prime_pool, precomputed_residues):
    """
    Outer loop.  Terminates when (a) every observed residue is explained by a
    known point, (b) a round finds no new points (nothing else has changed, so
    re-running would search the identical space), or (c)
    MAX_ANOMALOUS_SWEEP_ROUNDS is hit.

    Returns True if a round aborted for lack of any viable subset.
    """
    for sweep_round in range(MAX_ANOMALOUS_SWEEP_ROUNDS):
        round_new_points, aborted = _run_sweep_round(
            inp, setup, acc, stats, prime_pool, precomputed_residues, sweep_round)
        if aborted:
            return True

        analysis = analyze_unused_residue_orders(
            precomputed_residues=precomputed_residues,
            rhs_list=inp.rhs_list,
            found_m_set=acc.processed_m_vals,
            prime_pool=prime_pool,
            max_lift_k=3,
            Delta_pr=setup.Delta_pr,
            Ep_dict=setup.prep.Ep_dict
        )
        print_residue_analysis(analysis)

        total_unused = analysis['global']['total_unused_residues']
        if total_unused == 0:
            print(f"\n[anomalous-sweep] All residues explained by known rational points after round {sweep_round}. Terminating sweep.")
            break

        if not round_new_points:
            print(f"\n[anomalous-sweep] {total_unused} residue(s) remain unexplained, but round {sweep_round} found no new "
                  "points even with previously-explained residues pruned out. Stopping sweep.")
            break

        print(f"\n[anomalous-sweep] {total_unused} residue(s) still unexplained after round {sweep_round}. "
              f"Re-running round {sweep_round + 1} over the full prime pool with all {len(acc.processed_m_vals)} "
              "known m-value(s) pruned from the residues.")
    else:
        print(f"\n[anomalous-sweep] Reached MAX_ANOMALOUS_SWEEP_ROUNDS ({MAX_ANOMALOUS_SWEEP_ROUNDS}) without explaining all residues. Stopping.")
    return False


# =============================================================================
# Entry point
# =============================================================================

def run_standard_lattice_search(cd, current_sections, prime_pool, vecs, rhs_list, r_m, shift,
                                all_found_x, num_subsets, rationality_test_func, sconf, coeffs_genus2,
                                num_workers, debug, precomputed_residues,
                                markov_mode=False, height_pairing_H=None):
    """
    Standard lattice search.

    Normal mode: prep -> residues -> Brauer -> residue-graph discovery ->
    anomalous-residue sweep.
    Markov mode: skips expensive tuning / Brauer / attack plumbing and returns a
    compact candidate pool suitable for transition selection.

    Returns a dict with keys candidates, candidate_xs, new_sections,
    precomputed_residues, stats, final_rational_pairs.
    """
    # 1. vectors
    search_vecs = _resolve_search_vecs(vecs, precomputed_residues, current_sections, debug)
    vecs_list = list(search_vecs)

    inp = SearchInputs(
        cd=cd, current_sections=current_sections, vecs=vecs, rhs_list=rhs_list,
        r_m=r_m, shift=shift, all_found_x=all_found_x, num_subsets=num_subsets,
        rationality_test_func=rationality_test_func, sconf=sconf,
        coeffs_genus2=coeffs_genus2, num_workers=num_workers, debug=debug,
        search_vecs=search_vecs, height_pairing_H=height_pairing_H,
    )

    stats = SearchStats()
    print("prime pool used for search:", prime_pool)

    # 2. modular data
    prep = _prepare_modular_data(inp, prime_pool, stats)
    if not prep.Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return _empty_result(precomputed_residues, stats)

    # 3. residues
    if precomputed_residues is None:
        precomputed_residues = _precompute_residues(inp, prep, vecs_list, stats)
    else:
        print(f"Using provided precomputed residues ({len(precomputed_residues)} primes)")
        stats.incr('using_consensus_residues', n=1)
    _add_residues_to_stats(precomputed_residues, stats)

    # 4. markov mode exits here
    if markov_mode:
        return _run_markov_mode(inp, prep, search_vecs, prime_pool, precomputed_residues, stats)

    # ------------------------------------------------------------------
    # ORIGINAL HEAVY PATH
    # ------------------------------------------------------------------
    residue_counts = compute_residue_counts_for_primes(cd, rhs_list, prime_pool, max_primes=30)
    coverage_estimator = CoverageEstimator(prime_pool, residue_counts)

    # 5. Brauer (informational)
    _report_brauer_estimates(precomputed_residues, stats)

    acc = Accumulators()

    # 6. residue-graph discovery (diagnostic + candidate source; failures are fatal but reported)
    try:
        _discover_via_residue_graph(inp, acc, vecs_list, precomputed_residues)
    except Exception as e:
        print(f"[residue_graph] discovery failed (non-fatal, diagnostic only): {e}")
        raise

    # 7. TARGETED_X debugging
    matched_subset = None
    if TARGETED_X:
        targeted = _run_targeted_diagnostics(inp, stats, prime_pool, precomputed_residues)
        if targeted.abort:
            return _empty_result(precomputed_residues, stats)
        prime_pool = targeted.prime_pool
        precomputed_residues = targeted.precomputed_residues
        matched_subset = targeted.matched_subset

    # 8. sweep setup
    usable = _filter_usable_primes(prime_pool, precomputed_residues, debug)
    if usable is None:
        print("No primes have numeric precomputed residues. Aborting.")
        return _empty_result(precomputed_residues, stats)
    prime_pool = usable

    _autotune_extra_primes(prime_pool, precomputed_residues, vecs_list, rhs_list, stats)

    combo_cap = _combo_cap(sconf['MIN_PRIME_SUBSET_SIZE'])
    if debug:
        print("combo_cap:", combo_cap, "roots_threshold:", ROOTS_THRESHOLD)

    Delta_pr = _discriminant_polynomial(cd)
    target_qc_ratio = _target_qc_ratio(Delta_pr, prime_pool, debug)

    _report_adaptive_subset_count(stats, prime_pool, precomputed_residues, vecs_list, num_subsets)

    setup = SweepSetup(
        search_vecs=search_vecs,
        vecs_list=vecs_list,
        prep=prep,
        combo_cap=combo_cap,
        min_prime_subset_size=sconf['MIN_PRIME_SUBSET_SIZE'],
        min_max_prime_subset_size=sconf['MIN_MAX_PRIME_SUBSET_SIZE'],
        tmax=sconf['TMAX'],
        target_qc_ratio=target_qc_ratio,
        num_subsets_to_use=num_subsets,   # adaptive recommendation is advisory only
        Delta_pr=Delta_pr,
        coverage_estimator=coverage_estimator,
        matched_subset=matched_subset,
    )

    # 9. anomalous sweep
    aborted = _run_anomalous_sweep(inp, setup, acc, stats, prime_pool, precomputed_residues)
    if aborted:
        return acc.as_result(precomputed_residues, stats)

    if not acc.xs:
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return _empty_result(precomputed_residues, stats)

    stats.incr('rational_points_unique', n=len(acc.xs))
    stats.incr('new_sections_unique', n=0)   # new_sections is always [] on this path

    print("\n--- Search Statistics ---")
    print(stats.summary_string())

    return acc.as_result(precomputed_residues, stats)
