"""
search_lll/residue_crt_graph.py

try again.
"""

from collections import defaultdict
import itertools
import time
from tqdm import tqdm

from .rational_arithmetic import (
    crt_cached,
    lattice_rational_lift_exists,
    modulus_is_informative,
    rational_reconstruct,
)
try:
    from search_common import MIN_PRIME_SUBSET_SIZE, MIN_MAX_PRIME_SUBSET_SIZE
except ImportError:
    print("[residue_crt_graph] WARNING: could not import MIN_PRIME_SUBSET_SIZE/"
          "MIN_MAX_PRIME_SUBSET_SIZE from search_common (path issue?); "
          "falling back to 3/9.")
    MIN_PRIME_SUBSET_SIZE, MIN_MAX_PRIME_SUBSET_SIZE = 3, 9
import math

def estimate_ktuple_cost(prime_pool, k):
    """
    Satisfies search_analysis.py's logging requirement before it hits 
    the intercepted build_residue_graph_ktuple method.
    """
    n = len(prime_pool)
    if k > n or k < 0:
        return 0
    return math.comb(n, k)

def _iter_residue_nodes(precomputed_residues, prime_pool, v_tuple=None):
    """
    Flatten precomputed_residues into (p, v_tuple, rhs_idx, r) node tuples.
    """
    for p in prime_pool:
        p_map = precomputed_residues.get(p, {})
        if not p_map:
            continue
        items = p_map.items() if v_tuple is None else (
            [(v_tuple, p_map[v_tuple])] if v_tuple in p_map else []
        )
        for vt, roots_lists in items:
            for rhs_idx, roots in enumerate(roots_lists):
                for r in roots:
                    yield p, (vt, rhs_idx), int(r) % p


def false_lift_rate(M, H, num_samples=2000, seed=0):
    """
    Empirically measure what fraction of residues mod M admit SOME small
    lift under lattice_rational_lift_exists(c, M, H).
    """
    import random
    M, H = int(M), int(H)
    if M <= 2 * H + 1:
        return 1.0
    rng = random.Random(seed)
    if M <= num_samples * 5:
        hits = sum(1 for c in range(M) if lattice_rational_lift_exists(c, M, H))
        return hits / M
    sample = [rng.randrange(M) for _ in range(num_samples)]
    hits = sum(1 for c in sample if lattice_rational_lift_exists(c, M, H))
    return hits / num_samples


MIN_MARGIN_OVER_BOX = 15

_STRONG_INFORMATIVE_MARGIN = 50


def _min_attempts_for_informative(primes_with_data, H, p=None,
                                   margin=_STRONG_INFORMATIVE_MARGIN):
    """
    no
    """
    H = int(H)
    threshold = margin * H * H
    others = sorted((int(q) for q in primes_with_data if q != p), reverse=True)
    prod = 1
    for i, q in enumerate(others, 1):
        prod *= q
        if prod > threshold:
            return i
    return len(others)


def _has_witness_chain(p, r_p, nodes_by_prime, primes_with_data, H, sample_cap=None,
                        rng=None, min_success_rate=0.8, min_attempts=6,
                        beam_width=4, max_calls_per_prime_step=32):
    """
    learn to write docs lol
    """
    others = [q for q in primes_with_data if q != p]
    if rng is not None:
        others = list(others)
        rng.shuffle(others)
    if sample_cap is not None:
        others = others[:sample_cap]

    min_attempts = max(min_attempts, _min_attempts_for_informative(primes_with_data, H, p=p))

    # Beam of live accumulator states: list of (M, c).
    beam = [(p, r_p)]
    attempts = 0
    successes = 0
    for q in others:
        q_nodes = nodes_by_prime.get(q)
        if not q_nodes:
            continue

        attempts += 1
        children = []
        calls_this_step = 0
        q_nodes_this_step = list(q_nodes)
        if rng is not None:
            rng.shuffle(q_nodes_this_step)
        effective_cap = max(max_calls_per_prime_step, 4 * len(q_nodes_this_step) // max(1, len(beam)))
        for (M, c) in beam:
            new_M = M * q
            for nk_q in q_nodes_this_step:
                if calls_this_step >= effective_cap:
                    break
                r_q = nk_q[2]
                calls_this_step += 1
                new_c = crt_cached((c, r_q), (M, q))
                if lattice_rational_lift_exists(int(new_c) % new_M, new_M, H):
                    children.append((new_M, new_c))
            if calls_this_step >= effective_cap:
                break

        if children:
            successes += 1
            uniq = {}
            for (M2, c2) in children:
                uniq[(M2, c2)] = None  # dedupe exact repeats
            beam = sorted(uniq.keys(), key=lambda mc: mc[0])[:beam_width]

        if attempts >= min_attempts and successes / attempts < min_success_rate:
            return False  # failing badly enough that more attempts can't recover it

        if attempts >= min_attempts and successes / attempts >= min_success_rate \
                and any(M > _STRONG_INFORMATIVE_MARGIN * H * H for (M, _c) in beam):
            return True

    if attempts == 0:
        return False
    return successes / attempts >= min_success_rate


def arc_consistency_prune_domains(precomputed_residues, prime_pool, height_bound,
                                   v_tuple=None, max_rounds=None, stats_counter=None,
                                   progress=True, witness_sample_cap=None, seed=0,
                                   witness_beam_width=4, witness_max_calls_per_prime_step=32,
                                   witness_min_success_rate=0.8, witness_min_attempts=6):
    """
    figure it out lol
    """
    import random
    rng = random.Random(seed)

    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)

    nodes_by_prime = defaultdict(list)
    for p, node_id, r in _iter_residue_nodes(precomputed_residues, pool, v_tuple=v_tuple):
        nodes_by_prime[p].append((p, node_id, r))

    primes_with_data = [p for p in pool if nodes_by_prime.get(p)]
    if len(primes_with_data) < 2:
        return nodes_by_prime, 0

    total_before = sum(len(v) for v in nodes_by_prime.values())
    round_num = 0

    while True:
        round_num += 1
        if max_rounds is not None and round_num > max_rounds:
            round_num -= 1
            break

        any_dropped = False
        new_nodes_by_prime = {}
        residues_checked_this_round = 0
        checkpoint_start = time.time()

        for p in primes_with_data:
            p_nodes = nodes_by_prime[p]
            if not p_nodes:
                new_nodes_by_prime[p] = []
                continue

            survivors = []
            for nk in p_nodes:
                r_p = nk[2]
                if _has_witness_chain(p, r_p, nodes_by_prime, primes_with_data, H,
                                       sample_cap=witness_sample_cap, rng=rng,
                                       beam_width=witness_beam_width,
                                       max_calls_per_prime_step=witness_max_calls_per_prime_step,
                                       min_success_rate=witness_min_success_rate,
                                       min_attempts=witness_min_attempts):
                    survivors.append(nk)
                else:
                    any_dropped = True
                    if stats_counter is not None:
                        stats_counter['residue_graph_arc_consistency_dropped'] += 1
                residues_checked_this_round += 1
                if progress and residues_checked_this_round % 200 == 0:
                    elapsed = time.time() - checkpoint_start
                    rate = residues_checked_this_round / elapsed if elapsed > 0 else 0
                    print(f"  [residue_graph_arc_consistency] round {round_num} progress: "
                          f"{residues_checked_this_round}/{total_before} residues checked "
                          f"({rate:.1f}/sec)", flush=True)
            new_nodes_by_prime[p] = survivors

        nodes_by_prime = new_nodes_by_prime

        if progress:
            total_now = sum(len(v) for v in nodes_by_prime.values())
            print(f"[residue_graph_arc_consistency] round {round_num}: "
                  f"{total_now} residues survive (of {total_before} originally)")

        if not any_dropped:
            break

    return nodes_by_prime, round_num


def trace_target_through_arc_consistency(target_m, precomputed_residues, prime_pool,
                                          height_bound, v_tuple=None, max_rounds=None,
                                          witness_sample_cap=None, seed=0):
    """
    Diagnostic: given a KNOWN target rational m (e.g. from a point you found
    some other way, like the anomalous-sweep fallback), report whether m's
    true residue at each prime is even present in precomputed_residues, and
    if so, whether it survives each round of arc_consistency_prune_domains
    -- and if it's dropped, at which round.

    This never feeds target_m into the graph-building itself (the point of
    this module is to work without a target); it's purely an after-the-fact
    check on the pruning, to answer "did arc-consistency wrongly kill the
    real answer's residues, and when".

    Returns a dict: {prime: {'residue': r_or_None, 'in_domain': bool,
                              'survived_round': last round it was seen alive,
                              'dropped_round': round it disappeared, or None}}
    """
    from sage.all import QQ, Zmod

    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    target_m = QQ(target_m)

    nodes_by_prime = defaultdict(list)
    for p, node_id, r in _iter_residue_nodes(precomputed_residues, pool, v_tuple=v_tuple):
        nodes_by_prime[p].append((p, node_id, r))

    report = {}
    for p in pool:
        num, den = target_m.numerator(), target_m.denominator()
        if den % p == 0:
            report[p] = {'residue': None, 'in_domain': False,
                          'note': 'target has a pole mod p (den divisible by p)'}
            continue
        true_r = int(Zmod(p)(num) / Zmod(p)(den))
        present = any(r == true_r for (_p, _nid, r) in nodes_by_prime.get(p, []))
        report[p] = {'residue': true_r, 'in_domain': present,
                      'survived_round': None, 'dropped_round': None}

    print(f"[trace_target] target m={target_m}")
    for p in pool:
        info = report[p]
        if info.get('in_domain') is False and 'note' in info:
            print(f"  p={p}: {info['note']}")
        else:
            tag = "PRESENT in precomputed_residues" if info['in_domain'] else \
                  "*** MISSING from precomputed_residues entirely (never a candidate) ***"
            print(f"  p={p}: true residue={info['residue']}  {tag}")

    primes_with_data = [p for p in pool if nodes_by_prime.get(p)]
    if len(primes_with_data) < 2:
        return report

    import random
    rng = random.Random(seed)
    round_num = 0
    while True:
        round_num += 1
        if max_rounds is not None and round_num > max_rounds:
            round_num -= 1
            break
        any_dropped = False
        new_nodes_by_prime = {}
        for p in primes_with_data:
            p_nodes = nodes_by_prime[p]
            survivors = []
            for nk in p_nodes:
                r_p = nk[2]
                ok = _has_witness_chain(p, r_p, nodes_by_prime, primes_with_data, H,
                                         sample_cap=witness_sample_cap, rng=rng)
                if ok:
                    survivors.append(nk)
                    if report.get(p, {}).get('residue') == r_p:
                        report[p]['survived_round'] = round_num
                else:
                    any_dropped = True
                    if report.get(p, {}).get('residue') == r_p and \
                            report[p].get('dropped_round') is None:
                        report[p]['dropped_round'] = round_num
                        print(f"[trace_target]   *** p={p} true residue r={r_p} "
                              f"DROPPED at arc-consistency round {round_num} ***")
            new_nodes_by_prime[p] = survivors
        nodes_by_prime = new_nodes_by_prime
        if not any_dropped:
            break

    print(f"[trace_target] finished after {round_num} round(s). Summary:")
    for p in pool:
        info = report[p]
        if 'note' in info:
            continue
        status = ("still alive" if info.get('dropped_round') is None
                   and info.get('in_domain') else
                   f"dropped at round {info.get('dropped_round')}" if info.get('dropped_round')
                   else "never in domain")
        print(f"  p={p}: {status}")
    return report


def min_tuple_size_for_margin(prime_pool, height_bound, margin=MIN_MARGIN_OVER_BOX):
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = margin * box
    sorted_primes = sorted(int(p) for p in prime_pool)
    prod = 1
    for i, p in enumerate(sorted_primes):
        prod *= p
        if prod > threshold:
            return i + 1
    return None


def build_residue_graph_incremental(precomputed_residues, prime_pool, height_bound,
                                     v_tuple=None, margin=MIN_MARGIN_OVER_BOX,
                                     max_chains=None, stats_counter=None,
                                     progress=True, use_arc_consistency=False,
                                     arc_consistency_max_rounds=None,
                                     max_calls_per_generation=2_000_000,
                                     min_clique_size=MIN_PRIME_SUBSET_SIZE,
                                     max_clique_size=MIN_MAX_PRIME_SUBSET_SIZE,
                                     reconcile_components=False,
                                     max_reconcile_pairs=2_000_000):
    """
    verbose much lol.  i deleted the docs here because it was just a list of bug fixes
    """
    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = margin * box

    if use_arc_consistency:
        _tmp_nodes = defaultdict(list)
        for p, node_id, r in _iter_residue_nodes(precomputed_residues, pool, v_tuple=v_tuple):
            _tmp_nodes[p].append((p, node_id, r))
        _primes_with_data = [p for p in pool if _tmp_nodes.get(p)]
        avg_roots_observed = (sum(len(v) for v in _tmp_nodes.values()) / len(_primes_with_data)
                              if _primes_with_data else 1.0)
        auto_min_success_rate = 0.35
        if progress:
            print(f"[residue_graph_arc_consistency] observed avg_roots={avg_roots_observed:.2f} "
                  f"over {len(_primes_with_data)} primes with data -> "
                  f"using fixed witness_min_success_rate={auto_min_success_rate} "
                  f"(avg_roots is not a reliable proxy for per-residue coverage; see SEVENTH BUG note)")

        nodes_by_prime, ac_rounds = arc_consistency_prune_domains(
            precomputed_residues, pool, H, v_tuple=v_tuple,
            max_rounds=arc_consistency_max_rounds,
            stats_counter=stats_counter, progress=progress,
            witness_min_success_rate=auto_min_success_rate)
        nodes_by_prime = defaultdict(list, nodes_by_prime)
    else:
        nodes_by_prime = defaultdict(list)
        for p, node_id, r in _iter_residue_nodes(precomputed_residues, pool, v_tuple=v_tuple):
            nodes_by_prime[p].append((p, node_id, r))

    all_nodes = [nk for lst in nodes_by_prime.values() for nk in lst]

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for nk in all_nodes:
        find(nk)

    primes_with_data = [p for p in pool if nodes_by_prime.get(p)]
    prime_set = frozenset(primes_with_data)

    # Chains are keyed by the FROZENSET of primes already used, not by a
    # sorted-order index -- this is what lets a chain extend into any
    # remaining prime, in any order, rather than only "the next one".
    chains = []
    for p in primes_with_data:
        for nk in nodes_by_prime[p]:
            chains.append((frozenset((p,)), p, nk[2], [nk]))

    edges_tested = 0
    edges_kept = 0
    chains_confirmed = 0
    generation = 1
    generation_log = []

    confirmed_chains = []

    if progress:
        print(f"[residue_graph_incremental] start: {len(chains)} gen-1 chains "
              f"over {len(primes_with_data)} primes, threshold={threshold:.3g}")

    try:
        while chains:
            gen_start = time.monotonic()

            if max_chains is not None and len(chains) > max_chains:
                chains.sort(key=lambda ch: (len(ch[0]), ch[1]), reverse=True)
                dropped = len(chains) - max_chains
                chains = chains[:max_chains]
                if stats_counter is not None:
                    stats_counter['residue_graph_incremental_beam_capped'] += dropped
                if progress:
                    print(f"[residue_graph_incremental] gen {generation}: beam capped, "
                          f"dropped {dropped} chains (kept {max_chains} deepest-then-widest)")

            still_growing = [ch for ch in chains
                             if ch[1] <= threshold
                             and len(ch[0]) < min_clique_size
                             and len(ch[0]) < max_clique_size]
            already_confirmed_this_gen = len(chains) - len(still_growing)

            def _fanout_cost(ch):
                used_primes = ch[0]
                return sum(len(nodes_by_prime[p_remaining])
                           for p_remaining in primes_with_data
                           if p_remaining not in used_primes)

            inner_total = sum(_fanout_cost(ch) for ch in still_growing)

            deferred = []
            if inner_total > max_calls_per_generation:
                still_growing.sort(key=lambda ch: (len(ch[0]), ch[1]), reverse=True)
                budget = max_calls_per_generation
                affordable = []
                for ch in still_growing:
                    cost = _fanout_cost(ch)
                    if cost <= budget:
                        affordable.append(ch)
                        budget -= cost
                    else:
                        deferred.append(ch)
                if progress:
                    print(f"[residue_graph_incremental] gen {generation}: projected "
                          f"{inner_total} calls exceeds max_calls_per_generation="
                          f"{max_calls_per_generation}; extending {len(affordable)} "
                          f"chains this generation, deferring {len(deferred)} to next")
                still_growing = affordable
                inner_total = sum(_fanout_cost(ch) for ch in still_growing)
                if stats_counter is not None:
                    stats_counter['residue_graph_incremental_generation_deferred'] += len(deferred)

            if progress:
                print(f"[residue_graph_incremental] gen {generation}: "
                      f"{len(chains)} live chains ({already_confirmed_this_gen} already "
                      f"confirmed-or-overgrown, {len(still_growing)} extending this round, "
                      f"{len(deferred)} deferred) "
                      f"-> {inner_total} CRT/lift calls this generation")

            next_chains = []
            any_extended = False

            deferred_ids = {id(ch) for ch in deferred}

            inner_pbar = tqdm(total=inner_total, desc=f"  gen {generation} extend",
                               disable=not progress, leave=False)
            try:
                for ch in chains:
                    used_primes, M, c, node_keys = ch
                    if len(used_primes) >= min_clique_size or M > threshold:
                        first = node_keys[0]
                        for other in node_keys[1:]:
                            union(first, other)
                        chains_confirmed += 1
                        confirmed_chains.append({
                            'primes': sorted(used_primes),
                            'modulus': M,
                            'residue': c,
                            'node_keys': list(node_keys),
                        })
                        continue

                    if len(used_primes) >= max_clique_size:
                        if stats_counter is not None:
                            stats_counter['residue_graph_incremental_overgrown_dropped'] += 1
                        continue

                    if id(ch) in deferred_ids:
                        # Deferred this generation (EIGHTH BUG fix): carry
                        # forward unextended so it gets first priority
                        # (by the same depth-first ordering) next round,
                        # rather than losing the work already invested.
                        next_chains.append(ch)
                        continue

                    for p_next in primes_with_data:
                        if p_next in used_primes:
                            continue
                        for nk in nodes_by_prime[p_next]:
                            r_next = nk[2]
                            new_M = M * p_next
                            new_c = crt_cached((c, r_next), (M, p_next))
                            edges_tested += 1
                            inner_pbar.update(1)
                            if lattice_rational_lift_exists(int(new_c) % new_M, new_M, H):
                                edges_kept += 1
                                any_extended = True
                                next_chains.append((used_primes | {p_next}, new_M, new_c, node_keys + [nk]))
                            elif stats_counter is not None:
                                stats_counter['residue_graph_incremental_pruned'] += 1
            finally:
                inner_pbar.close()

            gen_elapsed = time.monotonic() - gen_start
            generation_log.append({
                'generation': generation,
                'chains_in': len(chains),
                'chains_confirmed_this_gen': already_confirmed_this_gen,
                'chains_out': len(next_chains),
                'inner_calls': inner_total,
                'elapsed_sec': gen_elapsed,
            })
            if progress:
                rate = inner_total / gen_elapsed if gen_elapsed > 0 else float('inf')
                print(f"[residue_graph_incremental] gen {generation} done in "
                      f"{gen_elapsed:.2f}s ({rate:.0f} calls/sec) -> "
                      f"{len(next_chains)} chains survive to gen {generation + 1}")

            if not any_extended:
                break

            chains = next_chains
            generation += 1
    except KeyboardInterrupt:
        if progress:
            print(f"[residue_graph_incremental] INTERRUPTED at gen {generation}: "
                  f"{len(chains)} chains were live, "
                  f"see generation_log below for where time went")
            for row in generation_log:
                print(f"    {row}")
        raise

    comp_map = defaultdict(list)
    for nk in all_nodes:
        comp_map[find(nk)].append(nk)

    components_pre_reconcile = list(comp_map.values())

    if reconcile_components and len(components_pre_reconcile) > 1:
        reconciled = 0
        pairs_checked = 0
        budget_hit = False
        # Use one representative (arbitrary) node per component; if a
        # component spans several primes already, fold its own residues
        # into a single running (M, c) first so the cross-component check
        # is informative on the first try where possible.
        def component_modulus_and_residue(comp):
            by_prime = defaultdict(list)
            for (p, _vt, r) in comp:
                by_prime[p].append(r)
            primes_here = sorted(by_prime.keys())
            M, c = primes_here[0], by_prime[primes_here[0]][0]
            for p in primes_here[1:]:
                r = by_prime[p][0]
                new_M = M * p
                c = crt_cached((c, r), (M, p))
                M = new_M
            return M, c, set(primes_here)

        comp_reps = [component_modulus_and_residue(comp) for comp in components_pre_reconcile]

        for i in range(len(components_pre_reconcile)):
            if budget_hit:
                break
            M_i, c_i, primes_i = comp_reps[i]
            for j in range(i + 1, len(components_pre_reconcile)):
                if pairs_checked >= max_reconcile_pairs:
                    budget_hit = True
                    if stats_counter is not None:
                        stats_counter['residue_graph_reconcile_budget_exhausted'] += 1
                    break
                pairs_checked += 1
                M_j, c_j, primes_j = comp_reps[j]
                if primes_i & primes_j:
                    continue  # already shared a prime; beam loop already compared these
                M_ij = M_i * M_j
                c_ij = crt_cached((c_i, c_j), (M_i, M_j))
                if lattice_rational_lift_exists(int(c_ij) % M_ij, M_ij, H):
                    union(components_pre_reconcile[i][0], components_pre_reconcile[j][0])
                    reconciled += 1
                    if stats_counter is not None:
                        stats_counter['residue_graph_reconciled_components'] += 1
        if progress and budget_hit:
            print(f"[residue_graph_incremental] reconciliation pass: "
                  f"stopped early, max_reconcile_pairs={max_reconcile_pairs} exhausted "
                  f"({reconciled} merges done so far)")
        if progress and reconciled:
            print(f"[residue_graph_incremental] reconciliation pass: "
                  f"merged {reconciled} disjoint-prime component pair(s)")

    comp_map = defaultdict(list)
    for nk in all_nodes:
        comp_map[find(nk)].append(nk)

    components = sorted(comp_map.values(), key=len, reverse=True)
    component_primes = [set(p for (p, _vt, _r) in comp) for comp in components]

    return {
        'components': components,
        'component_primes': component_primes,
        'edges_tested': edges_tested,
        'edges_kept': edges_kept,
        'nodes': len(all_nodes),
        'max_generation_reached': generation,
        'chains_confirmed': chains_confirmed,
        'confirmed_chains': confirmed_chains,
        'generation_log': generation_log,
    }


def build_residue_graph_ktuple(precomputed_residues, prime_pool, height_bound,
                                k=None, v_tuple=None, margin=MIN_MARGIN_OVER_BOX,
                                max_tuples=None, stats_counter=None, progress=False,
                                **kwargs):
    """
    The combinatorial k-tuple generator was ripped out due to massive stalling.
    This now automatically routes to the incremental beam-search method so you
    don't have to alter caller signatures elsewhere.

    progress defaults to False: this is called once PER VECTOR by
    discover_candidates_via_residue_graph (see search_analysis.py), and with
    dozens of vectors in play, per-generation beam-search progress lines
    multiply into an unreadable wall of near-identical output. Pass
    progress=True only when you're debugging a single vector's beam search
    specifically (e.g. temporarily, for one known-target vector) -- not as
    the default for a full multi-vector scan.
    """
    if progress:
        print(f"\n[residue_graph] WARNING: build_residue_graph_ktuple intercepted. Routing to incremental beam search to avoid combinatorial explosion...")
    return build_residue_graph_incremental(
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        height_bound=height_bound,
        v_tuple=v_tuple,
        margin=margin,
        max_chains=20000, 
        stats_counter=stats_counter,
        progress=progress,
    )


def build_residue_graph(precomputed_residues, prime_pool, height_bound,
                         v_tuple=None, require_unique_modulus=False,
                         require_margin_over_box=True,
                         max_primes=None, stats_counter=None):
    """
    Build the pairwise CRT-compatibility graph over ALL residues at ALL
    primes in prime_pool.
    """
    pool = list(prime_pool) if max_primes is None else list(prime_pool)[:max_primes]
    H = int(height_bound)

    nodes_by_prime = defaultdict(list)
    for p, node_id, r in _iter_residue_nodes(precomputed_residues, pool, v_tuple=v_tuple):
        node_key = (p, node_id, r)
        nodes_by_prime[p].append((node_key, r))

    all_nodes = [nk for lst in nodes_by_prime.values() for nk, _ in lst]

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for nk in all_nodes:
        find(nk)

    primes_sorted = sorted(nodes_by_prime.keys())
    edges_tested = 0
    edges_kept = 0

    for i, p in enumerate(primes_sorted):
        p_nodes = nodes_by_prime[p]
        for q in primes_sorted[i + 1:]:
            q_nodes = nodes_by_prime[q]
            if not p_nodes or not q_nodes:
                continue
            M = p * q
            if require_margin_over_box:
                box = (2 * H + 1) ** 2
                informative = M > MIN_MARGIN_OVER_BOX * box
            elif require_unique_modulus:
                informative = M > 2 * H * H
            else:
                informative = modulus_is_informative(M, H)
            if stats_counter is not None and not informative:
                stats_counter['residue_graph_uninformative_pair'] += 1
            if not informative:
                continue

            for nk_p, r_p in p_nodes:
                for nk_q, r_q in q_nodes:
                    edges_tested += 1
                    c = crt_cached((r_p, r_q), (p, q))
                    if lattice_rational_lift_exists(int(c) % M, M, H):
                        edges_kept += 1
                        union(nk_p, nk_q)

    comp_map = defaultdict(list)
    for nk in all_nodes:
        comp_map[find(nk)].append(nk)

    components = sorted(comp_map.values(), key=len, reverse=True)
    component_primes = [set(p for (p, _vt, _r) in comp) for comp in components]

    return {
        'components': components,
        'component_primes': component_primes,
        'edges_tested': edges_tested,
        'edges_kept': edges_kept,
        'nodes': len(all_nodes),
    }


def summarize_components(graph_result, top_k=10):
    rows = []
    for comp, comp_primes in zip(graph_result['components'][:top_k],
                                  graph_result['component_primes'][:top_k]):
        rows.append({
            'num_nodes': len(comp),
            'num_primes': len(comp_primes),
            'primes': sorted(comp_primes),
        })
    return rows


def refine_component_chained(comp, height_bound, stats_counter=None):
    H = int(height_bound)
    by_prime = defaultdict(list)
    for (p, node_id, r) in comp:
        by_prime[p].append(r)

    primes = sorted(by_prime.keys())
    if len(primes) < 2:
        return {'confirmed': False, 'modulus_reached': 0, 'primes_used': []}

    best_modulus_reached = 0

    for combo in itertools.product(*(by_prime[p] for p in primes)):
        modulus = primes[0]
        residue = combo[0] % modulus
        chain_ok = True
        for p, r in zip(primes[1:], combo[1:]):
            new_modulus = modulus * p
            residue = crt_cached((residue, r), (modulus, p))
            modulus = new_modulus
            if not lattice_rational_lift_exists(int(residue) % modulus, modulus, H):
                chain_ok = False
                if stats_counter is not None:
                    stats_counter['residue_graph_chain_broke'] += 1
                break
        best_modulus_reached = max(best_modulus_reached, modulus if chain_ok else 0)
        if chain_ok and modulus > 2 * H * H:
            return {
                'confirmed': True,
                'modulus_reached': modulus,
                'primes_used': list(primes),
            }

    return {
        'confirmed': False,
        'modulus_reached': best_modulus_reached,
        'primes_used': [],
    }


def reconstruct_candidate_from_chain(chain, height_bound, max_den=None):
    """
    Reconstruct the single m implied by one confirmed chain (as recorded
    in build_residue_graph_incremental's 'confirmed_chains'). Unlike
    reconstruct_candidates_from_component, there is no search here --
    M and c are already the fully-reduced CRT accumulation over exactly
    this chain's own primes, so it's one rational_reconstruct call.
    Returns a single {'m_num', 'm_den', 'primes', 'modulus'} dict, or
    None if reconstruction fails (e.g. no a/b within max_den survives
    the height bound -- can happen for a chain that was confirmed via
    the M > threshold path rather than clique size, on primes that
    don't actually carry a real point).
    """
    H = int(height_bound)
    M = chain['modulus']
    c = chain['residue']
    try:
        a, b = rational_reconstruct(int(c) % M, M, max_den=max_den or H)
        return {'m_num': a, 'm_den': b, 'primes': chain['primes'], 'modulus': M}
    except Exception:
        return None


def reconstruct_candidates_from_component(comp, height_bound, max_den=None,
                                           max_results=5, stats_counter=None,
                                           max_visits=2_000_000):
    """
    write better docs and then i won't delete them all, claude
    """
    by_prime = defaultdict(list)
    for (p, node_id, r) in comp:
        by_prime[p].append(r)

    primes = sorted(by_prime.keys(), key=lambda p: len(by_prime[p]))
    if len(primes) < 2:
        return []

    H = int(height_bound)
    results = []
    visits = [0]
    budget_exhausted = [False]

    def recurse(idx, modulus, residue):
        if budget_exhausted[0] or (max_results is not None and len(results) >= max_results):
            return
        visits[0] += 1
        if visits[0] > max_visits:
            budget_exhausted[0] = True
            if stats_counter is not None:
                stats_counter['reconstruct_visit_budget_exhausted'] += 1
            return
        if idx == len(primes):
            try:
                a, b = rational_reconstruct(int(residue) % modulus, modulus,
                                             max_den=max_den or H)
                results.append({'m_num': a, 'm_den': b, 'primes': primes,
                                 'modulus': modulus})
            except Exception:
                pass
            return
        p = primes[idx]
        for r in by_prime[p]:
            if idx == 0:
                new_modulus, new_residue = p, r % p
            else:
                new_modulus = modulus * p
                new_residue = crt_cached((residue, r), (modulus, p))
            if not lattice_rational_lift_exists(int(new_residue) % new_modulus, new_modulus, H):
                if stats_counter is not None:
                    stats_counter['reconstruct_branch_pruned'] += 1
                continue
            recurse(idx + 1, new_modulus, new_residue)
            if budget_exhausted[0] or (max_results is not None and len(results) >= max_results):
                return

    recurse(0, 1, 0)
    return results
