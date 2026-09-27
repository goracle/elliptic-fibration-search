"""
search_lll/residue_crt_graph.py

Cross-prime residue compatibility graph (arc-consistency-style), built
WITHOUT committing to a target m first.

Motivation (see aitools/aimist.txt discussion / chat log):
`compute_residue_coverage_for_m` and `diagnose_missed_point`'s "Compatible
Primes" both answer "is *this specific* m compatible with the residues at
each prime" -- they take a candidate m as input and check it against the
precomputed root sets. That means the search only ever discovers a rational
point by first guessing (or CRT-reconstructing from a subset) a concrete m
and then verifying it: an m never in the sampled subsets is never checked at
all, and one bad prime in a random subset kills the whole candidate.

This module inverts that: instead of "given m, which primes agree with it",
it asks "which residues at DIFFERENT primes are mutually consistent with
EACH OTHER" -- i.e. could be projections of the same rational m -- using
only pairwise CRT-lift feasibility (_pairwise_crt_survivors /
lattice_rational_lift_exists), never touching a specific target m at all.
The residues that mutually agree across many primes form a connected
component in this graph; a large component is a candidate worth actually
CRT-reconstructing, discovered "bottom-up" rather than "guess-then-verify".

This is deliberately a *diagnostic / candidate-generation* layer that sits
alongside process_prime_subset_precomputed, not a replacement for it. It
reuses the exact primitives already validated in modularthread.py and
rational_arithmetic.py:
    - lattice_rational_lift_exists(c, M, H)
    - modulus_is_informative(M, H)
    - crt_cached(residues, moduli)
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
                                     progress=True):
    """
    Replaces build_residue_graph_ktuple's exhaustive C(n,k) strategy with 
    incremental beam growth to strictly avoid combinatorial blowup.
    """
    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = margin * box

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
    prime_index = {p: i for i, p in enumerate(primes_with_data)}

    chains = []
    for p in primes_with_data:
        i = prime_index[p]
        for nk in nodes_by_prime[p]:
            chains.append((i, p, nk[2], [nk])) 

    edges_tested = 0
    edges_kept = 0
    chains_confirmed = 0
    generation = 1
    generation_log = [] 

    if progress:
        print(f"[residue_graph_incremental] start: {len(chains)} gen-1 chains "
              f"over {len(primes_with_data)} primes, threshold={threshold:.3g}")

    try:
        while chains:
            gen_start = time.monotonic()

            if max_chains is not None and len(chains) > max_chains:
                chains.sort(key=lambda c: c[1], reverse=True)
                dropped = len(chains) - max_chains
                chains = chains[:max_chains]
                if stats_counter is not None:
                    stats_counter['residue_graph_incremental_beam_capped'] += dropped
                if progress:
                    print(f"[residue_graph_incremental] gen {generation}: beam capped, "
                          f"dropped {dropped} chains (kept {max_chains} widest-modulus)")

            still_growing = [ch for ch in chains if ch[1] <= threshold]
            already_confirmed_this_gen = len(chains) - len(still_growing)

            inner_total = 0
            for (last_idx, _M, _c, _nk) in still_growing:
                for next_idx in range(last_idx + 1, len(primes_with_data)):
                    inner_total += len(nodes_by_prime[primes_with_data[next_idx]])

            if progress:
                print(f"[residue_graph_incremental] gen {generation}: "
                      f"{len(chains)} live chains ({already_confirmed_this_gen} already "
                      f"past threshold, {len(still_growing)} still extending) "
                      f"-> {inner_total} CRT/lift calls this generation")

            next_chains = []
            any_extended = False

            inner_pbar = tqdm(total=inner_total, desc=f"  gen {generation} extend",
                               disable=not progress, leave=False)
            try:
                for (last_idx, M, c, node_keys) in chains:
                    if M > threshold:
                        first = node_keys[0]
                        for other in node_keys[1:]:
                            union(first, other)
                        chains_confirmed += 1
                        continue

                    # BUG (found after the run that produced only garbage
                    # single-prime components): this used to fan out into
                    # EVERY remaining prime here (range(last_idx+1, len(...))),
                    # which makes the branching factor ~(remaining primes) x
                    # (avg roots) per generation -- that's what actually
                    # produced 1894 -> 1.7M chains in one step, not real
                    # pruning failing. Extending into only the SINGLE next
                    # prime (last_idx+1) makes this genuinely O(depth), not
                    # O(remaining_primes) per generation, so the beam can only
                    # grow by ~avg_roots per generation (~1894 * 1.04^depth),
                    # not by (avg_roots * remaining_primes)^depth.
                    if last_idx + 1 >= len(primes_with_data):
                        continue  # no more primes to extend into
                    next_idx = last_idx + 1
                    p_next = primes_with_data[next_idx]
                    for nk in nodes_by_prime[p_next]:
                        r_next = nk[2]
                        new_M = M * p_next
                        new_c = crt_cached((c, r_next), (M, p_next))
                        edges_tested += 1
                        inner_pbar.update(1)
                        if lattice_rational_lift_exists(int(new_c) % new_M, new_M, H):
                            edges_kept += 1
                            any_extended = True
                            next_chains.append((next_idx, new_M, new_c, node_keys + [nk]))
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
        'generation_log': generation_log,
    }


def build_residue_graph_ktuple(precomputed_residues, prime_pool, height_bound,
                                k=None, v_tuple=None, margin=MIN_MARGIN_OVER_BOX,
                                max_tuples=None, stats_counter=None, **kwargs): 
    """
    The combinatorial k-tuple generator was ripped out due to massive stalling.
    This now automatically routes to the incremental beam-search method so you 
    don't have to alter caller signatures elsewhere.
    """
    print(f"\n[residue_graph] WARNING: build_residue_graph_ktuple intercepted. Routing to incremental beam search to avoid combinatorial explosion...")
    return build_residue_graph_incremental(
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        height_bound=height_bound,
        v_tuple=v_tuple,
        margin=margin,
        max_chains=20000, 
        stats_counter=stats_counter,
        progress=True
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


def reconstruct_candidates_from_component(comp, height_bound, max_den=None):
    by_prime = defaultdict(list)
    for (p, node_id, r) in comp:
        by_prime[p].append(r)

    primes = sorted(by_prime.keys())
    if len(primes) < 2:
        return []

    results = []
    residue_choices = [by_prime[p] for p in primes]
    for combo in itertools.product(*residue_choices):
        moduli = tuple(primes)
        c = crt_cached(tuple(combo), moduli)
        M = 1
        for p in primes:
            M *= p
        try:
            a, b = rational_reconstruct(int(c) % M, M, max_den=max_den or height_bound)
            results.append({'m_num': a, 'm_den': b, 'primes': primes, 'modulus': M})
        except Exception:
            continue

    return results
