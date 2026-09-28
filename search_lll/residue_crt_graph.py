"""
Cross-prime residue consistency search.

This module searches for small-height rational points by gluing residue classes
across primes with CRT and testing whether the resulting class can contain a
rational of bounded height.

The implementation is deliberately organized around the expensive operation
`lattice_rational_lift_exists`.  The old version repeatedly recomputed the
same liftability questions and could spend a very long time constructing a
single generation before yielding any observable progress.  This version adds:

* a bounded LRU cache for liftability tests;
* a round-robin chain work queue, so one huge residue domain cannot monopolize
  a generation;
* hard per-generation and optional total/time budgets;
* randomized, information-per-branching "prime mixing" to diversify CRT
  orders without giving up eventual coverage of the pool;
* state deduplication for equivalent CRT states;
* shared search statistics and cache statistics;
* a small Union-Find implementation instead of repeated nested closures.

The public entry points from the original module are retained.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict, deque
from dataclasses import dataclass
import itertools
import math
import random
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
    print(
        "[residue_crt_graph] WARNING: could not import "
        "MIN_PRIME_SUBSET_SIZE/MIN_MAX_PRIME_SUBSET_SIZE from search_common; "
        "falling back to 3/9."
    )
    MIN_PRIME_SUBSET_SIZE, MIN_MAX_PRIME_SUBSET_SIZE = 3, 9


MIN_MARGIN_OVER_BOX = 15
_STRONG_INFORMATIVE_MARGIN = 50


# ---------------------------------------------------------------------------
# Small infrastructure
# ---------------------------------------------------------------------------

class _UnionFind:
    """Tiny path-compressing / union-by-size disjoint-set structure."""

    __slots__ = ("parent", "size")

    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True


class _LiftCache:
    """
    Bounded LRU cache for lattice_rational_lift_exists.

    The result depends only on (c mod M, M, H), so this is a very high-value
    cache for this search: many different chains reach the same CRT state.
    """

    __slots__ = ("maxsize", "_data", "hits", "misses")

    def __init__(self, maxsize=250_000):
        self.maxsize = max(0, int(maxsize))
        self._data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __call__(self, c, M, H):
        M = int(M)
        H = int(H)
        if M <= 0:
            return False

        c = int(c) % M
        key = (M, c, H)

        cached = self._data.get(key)
        if cached is not None:
            self.hits += 1
            self._data.move_to_end(key)
            return cached

        self.misses += 1
        value = bool(lattice_rational_lift_exists(c, M, H))

        if self.maxsize:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

        return value

    def stats(self):
        return {
            "lift_cache_hits": self.hits,
            "lift_cache_misses": self.misses,
            "lift_cache_size": len(self._data),
            "lift_cache_maxsize": self.maxsize,
        }


@dataclass
class _ChainTask:
    """
    Partially expanded chain.

    `prime_pos` and `node_pos` are resumable cursors, so hitting the generation
    budget no longer means throwing away all the work done on the chain and
    restarting its Cartesian product next generation.
    """

    used_primes: frozenset
    modulus: int
    residue: int
    node_keys: tuple
    prime_order: tuple
    prime_start: int = 0
    prime_pos: int = 0
    node_pos: int = 0
    calls_this_generation: int = 0
    primes_this_generation: int = 0


def _iter_residue_nodes(precomputed_residues, prime_pool, v_tuple=None):
    """Yield `(p, (v_tuple, rhs_idx), residue)` triples."""
    for p in prime_pool:
        p_map = precomputed_residues.get(p, {})
        if not p_map:
            continue

        if v_tuple is None:
            items = p_map.items()
        elif v_tuple in p_map:
            items = ((v_tuple, p_map[v_tuple]),)
        else:
            continue

        for vt, roots_lists in items:
            for rhs_idx, roots in enumerate(roots_lists):
                for r in roots:
                    yield p, (vt, rhs_idx), int(r) % int(p)


def _nodes_by_prime(precomputed_residues, prime_pool, v_tuple=None):
    nodes = defaultdict(list)
    for p, node_id, r in _iter_residue_nodes(
        precomputed_residues, prime_pool, v_tuple=v_tuple
    ):
        nodes[p].append((p, node_id, r))
    return nodes


def _component_map(all_nodes, uf):
    comp_map = defaultdict(list)
    for nk in all_nodes:
        comp_map[uf.find(nk)].append(nk)
    return comp_map


def _union_paths(uf, path_a, path_b):
    """
    Merge equivalent CRT paths prime-by-prime.

    This is used only when two search states have the same `(used_primes, CRT
    residue)`; they therefore represent the same residue at every used prime.
    """
    by_prime_a = {nk[0]: nk for nk in path_a}
    by_prime_b = {nk[0]: nk for nk in path_b}
    for p in by_prime_a.keys() & by_prime_b.keys():
        uf.union(by_prime_a[p], by_prime_b[p])


def _cache_lift_from_optional(cache, c, M, H):
    return cache(c, M, H) if cache is not None else lattice_rational_lift_exists(
        int(c) % int(M), int(M), int(H)
    )


# ---------------------------------------------------------------------------
# Cheap helpers / diagnostics
# ---------------------------------------------------------------------------

def estimate_ktuple_cost(prime_pool, k):
    """Return the combinatorial number of k-subsets of the supplied pool."""
    n = len(prime_pool)
    if k > n or k < 0:
        return 0
    return math.comb(n, k)


def false_lift_rate(M, H, num_samples=2000, seed=0):
    """
    Empirically estimate the fraction of residues mod M admitting a small
    rational lift.
    """
    M, H = int(M), int(H)
    if M <= 2 * H + 1:
        return 1.0

    rng = random.Random(seed)
    if M <= num_samples * 5:
        hits = sum(
            1 for c in range(M) if lattice_rational_lift_exists(c, M, H)
        )
        return hits / M

    hits = sum(
        1
        for _ in range(num_samples)
        if lattice_rational_lift_exists(rng.randrange(M), M, H)
    )
    return hits / num_samples


def _min_attempts_for_informative(
    primes_with_data, H, p=None, margin=_STRONG_INFORMATIVE_MARGIN
):
    """Smallest number of largest partner primes whose product beats margin*H²."""
    H = int(H)
    threshold = int(margin) * H * H
    others = sorted((int(q) for q in primes_with_data if q != p), reverse=True)

    product = 1
    for i, q in enumerate(others, 1):
        product *= q
        if product > threshold:
            return i
    return len(others)


# ---------------------------------------------------------------------------
# Witness / arc consistency
# ---------------------------------------------------------------------------

def _has_witness_chain(
    p,
    r_p,
    nodes_by_prime,
    primes_with_data,
    H,
    sample_cap=None,
    rng=None,
    min_success_rate=0.8,
    min_attempts=6,
    beam_width=4,
    max_calls_per_prime_step=32,
    lift_cache=None,
):
    """
    Decide whether residue `r_p` at `p` has support through other primes.

    This retains the original beam-search idea but routes every lift test through
    the shared cache when one is supplied.
    """
    others = [q for q in primes_with_data if q != p]
    if rng is not None:
        rng.shuffle(others)
    if sample_cap is not None:
        others = others[: int(sample_cap)]

    min_attempts = max(
        int(min_attempts),
        _min_attempts_for_informative(primes_with_data, H, p=p),
    )

    beam = [(int(p), int(r_p) % int(p))]
    attempts = 0
    successes = 0

    for q in others:
        q_nodes = nodes_by_prime.get(q)
        if not q_nodes:
            continue

        attempts += 1
        children = []
        q_nodes_this_step = list(q_nodes)
        if rng is not None:
            rng.shuffle(q_nodes_this_step)

        effective_cap = max(
            int(max_calls_per_prime_step),
            4 * len(q_nodes_this_step) // max(1, len(beam)),
        )

        calls_this_step = 0
        for M, c in beam:
            new_M = M * int(q)
            for nk_q in q_nodes_this_step:
                if calls_this_step >= effective_cap:
                    break

                r_q = nk_q[2]
                calls_this_step += 1
                new_c = crt_cached((c, r_q), (M, q))

                if _cache_lift_from_optional(
                    lift_cache, int(new_c) % new_M, new_M, H
                ):
                    children.append((new_M, new_c))

            if calls_this_step >= effective_cap:
                break

        if children:
            successes += 1
            uniq = {}
            for state in children:
                uniq[state] = None
            beam = sorted(uniq.keys(), key=lambda mc: mc[0])[: int(beam_width)]

        if attempts >= min_attempts and successes / attempts < min_success_rate:
            return False

        if (
            attempts >= min_attempts
            and successes / attempts >= min_success_rate
            and any(M > _STRONG_INFORMATIVE_MARGIN * H * H for M, _ in beam)
        ):
            return True

    return attempts > 0 and successes / attempts >= min_success_rate


def arc_consistency_prune_domains(
    precomputed_residues,
    prime_pool,
    height_bound,
    v_tuple=None,
    max_rounds=None,
    stats_counter=None,
    progress=True,
    witness_sample_cap=None,
    seed=0,
    witness_beam_width=4,
    witness_max_calls_per_prime_step=32,
    witness_min_success_rate=0.8,
    witness_min_attempts=6,
    lift_cache_size=250_000,
):
    """
    Iteratively remove residue nodes that lack a cross-prime witness.

    Returns `(nodes_by_prime, rounds_run)`.
    """
    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    rng = random.Random(seed)
    lift_cache = _LiftCache(lift_cache_size)

    nodes_by_prime = _nodes_by_prime(
        precomputed_residues, pool, v_tuple=v_tuple
    )
    primes_with_data = [p for p in pool if nodes_by_prime.get(p)]

    if len(primes_with_data) < 2:
        return nodes_by_prime, 0

    total_before = sum(len(v) for v in nodes_by_prime.values())
    round_num = 0

    while True:
        round_num += 1
        if max_rounds is not None and round_num > int(max_rounds):
            round_num -= 1
            break

        any_dropped = False
        new_nodes_by_prime = defaultdict(list)
        checked = 0
        checkpoint_start = time.monotonic()

        for p in primes_with_data:
            for nk in nodes_by_prime[p]:
                ok = _has_witness_chain(
                    p,
                    nk[2],
                    nodes_by_prime,
                    primes_with_data,
                    H,
                    sample_cap=witness_sample_cap,
                    rng=rng,
                    beam_width=witness_beam_width,
                    max_calls_per_prime_step=witness_max_calls_per_prime_step,
                    min_success_rate=witness_min_success_rate,
                    min_attempts=witness_min_attempts,
                    lift_cache=lift_cache,
                )
                if ok:
                    new_nodes_by_prime[p].append(nk)
                else:
                    any_dropped = True
                    if stats_counter is not None:
                        stats_counter[
                            "residue_graph_arc_consistency_dropped"
                        ] += 1

                checked += 1
                if progress and checked % 200 == 0:
                    elapsed = max(time.monotonic() - checkpoint_start, 1e-9)
                    rate = checked / elapsed
                    print(
                        "[residue_graph_arc_consistency] "
                        f"round {round_num} progress: {checked}/{total_before} "
                        f"({rate:.1f}/sec)",
                        flush=True,
                    )

        nodes_by_prime = new_nodes_by_prime

        if progress:
            total_now = sum(len(v) for v in nodes_by_prime.values())
            cache_stats = lift_cache.stats()
            print(
                "[residue_graph_arc_consistency] "
                f"round {round_num}: {total_now} residues survive "
                f"(of {total_before} originally); "
                f"lift-cache hits={cache_stats['lift_cache_hits']:,}, "
                f"misses={cache_stats['lift_cache_misses']:,}"
            )

        if not any_dropped:
            break

    return nodes_by_prime, round_num


def trace_target_through_arc_consistency(
    target_m,
    precomputed_residues,
    prime_pool,
    height_bound,
    v_tuple=None,
    max_rounds=None,
    witness_sample_cap=None,
    seed=0,
):
    """
    Diagnostic: trace a known target through the same pruning procedure.
    """
    from sage.all import QQ, Zmod

    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    target_m = QQ(target_m)
    nodes_by_prime = _nodes_by_prime(
        precomputed_residues, pool, v_tuple=v_tuple
    )

    report = {}
    for p in pool:
        num, den = target_m.numerator(), target_m.denominator()
        if den % p == 0:
            report[p] = {
                "residue": None,
                "in_domain": False,
                "note": "target has a pole mod p (den divisible by p)",
            }
            continue

        true_r = int(Zmod(p)(num) / Zmod(p)(den))
        present = any(
            r == true_r for (_p, _nid, r) in nodes_by_prime.get(p, [])
        )
        report[p] = {
            "residue": true_r,
            "in_domain": present,
            "survived_round": None,
            "dropped_round": None,
        }

    print(f"[trace_target] target m={target_m}")
    for p in pool:
        info = report[p]
        if info.get("in_domain") is False and "note" in info:
            print(f"  p={p}: {info['note']}")
        else:
            tag = (
                "PRESENT in precomputed_residues"
                if info["in_domain"]
                else "*** MISSING from precomputed_residues entirely "
                "(never a candidate) ***"
            )
            print(
                f"  p={p}: true residue={info['residue']}  {tag}"
            )

    primes_with_data = [p for p in pool if nodes_by_prime.get(p)]
    if len(primes_with_data) < 2:
        return report

    rng = random.Random(seed)
    lift_cache = _LiftCache()
    round_num = 0

    while True:
        round_num += 1
        if max_rounds is not None and round_num > int(max_rounds):
            round_num -= 1
            break

        any_dropped = False
        new_nodes_by_prime = defaultdict(list)

        for p in primes_with_data:
            for nk in nodes_by_prime[p]:
                r_p = nk[2]
                ok = _has_witness_chain(
                    p,
                    r_p,
                    nodes_by_prime,
                    primes_with_data,
                    H,
                    sample_cap=witness_sample_cap,
                    rng=rng,
                    lift_cache=lift_cache,
                )
                if ok:
                    new_nodes_by_prime[p].append(nk)
                    if report.get(p, {}).get("residue") == r_p:
                        report[p]["survived_round"] = round_num
                else:
                    any_dropped = True
                    if (
                        report.get(p, {}).get("residue") == r_p
                        and report[p].get("dropped_round") is None
                    ):
                        report[p]["dropped_round"] = round_num
                        print(
                            f"[trace_target]   *** p={p} true residue r={r_p} "
                            f"DROPPED at arc-consistency round {round_num} ***"
                        )

        nodes_by_prime = new_nodes_by_prime
        if not any_dropped:
            break

    print(f"[trace_target] finished after {round_num} round(s). Summary:")
    for p in pool:
        info = report[p]
        if "note" in info:
            continue

        if info.get("dropped_round") is None and info.get("in_domain"):
            status = "still alive"
        elif info.get("dropped_round"):
            status = f"dropped at round {info['dropped_round']}"
        else:
            status = "never in domain"
        print(f"  p={p}: {status}")

    return report


# ---------------------------------------------------------------------------
# Prime selection / mixed beam search
# ---------------------------------------------------------------------------

def min_tuple_size_for_margin(
    prime_pool, height_bound, margin=MIN_MARGIN_OVER_BOX
):
    """Smallest k whose k smallest primes have product > margin*(2H+1)^2."""
    H = int(height_bound)
    threshold = int(margin) * (2 * H + 1) ** 2

    product = 1
    for i, p in enumerate(sorted(int(p) for p in prime_pool), 1):
        product *= p
        if product > threshold:
            return i
    return None


def _mixed_prime_order(
    primes,
    nodes_by_prime,
    rng,
    temperature=0.35,
    branch_penalty=0.5,
):
    """
    Produce one shared randomized ranking of primes.

    The score rewards large modulus gain and penalizes large residue domains.
    A single tuple is shared by every chain; each chain gets a different random
    starting offset, so we get N-way mixing without storing N copies of the
    permutation.
    """
    scored = []
    temperature = max(0.0, float(temperature))
    branch_penalty = max(0.0, float(branch_penalty))

    for p in primes:
        p = int(p)
        domain = max(1, len(nodes_by_prime.get(p, ())))
        information = math.log1p(p)
        branch_cost = domain ** branch_penalty
        jitter = (
            math.exp(temperature * (rng.random() - 0.5))
            if temperature
            else 1.0
        )
        scored.append(
            (information / branch_cost * jitter, p)
        )

    scored.sort(reverse=True)
    return tuple(p for _score, p in scored)


def _advance_task(task, nodes_by_prime, used_limit=None):
    """
    Consume one candidate edge from a resumable task.

    Returns `(p_next, node)` or `None` if the task has no remaining work.
    """
    if used_limit is not None and task.primes_this_generation >= used_limit:
        return None

    order_len = len(task.prime_order)
    while task.prime_pos < order_len:
        p_next = task.prime_order[
            (task.prime_start + task.prime_pos) % order_len
        ]
        p_nodes = nodes_by_prime.get(p_next, [])

        if p_next in task.used_primes:
            task.prime_pos += 1
            task.node_pos = 0
            continue

        if task.node_pos >= len(p_nodes):
            task.prime_pos += 1
            task.node_pos = 0
            continue

        nk = p_nodes[task.node_pos]
        task.node_pos += 1
        task.calls_this_generation += 1

        if task.node_pos == 1:
            task.primes_this_generation += 1

        # Keep the cursor canonical: once a prime's domain is exhausted, move
        # immediately to the next prime so the hot-path "do I have more work?"
        # check does not rescan the exhausted domain.
        if task.node_pos >= len(p_nodes):
            task.prime_pos += 1
            task.node_pos = 0

        return p_next, nk

    return None


def _task_has_more_work(task, nodes_by_prime):
    order_len = len(task.prime_order)
    pos = task.prime_pos
    node_pos = task.node_pos

    while pos < order_len:
        p = task.prime_order[
            (task.prime_start + pos) % order_len
        ]
        if p in task.used_primes:
            pos += 1
            node_pos = 0
            continue

        if node_pos < len(nodes_by_prime.get(p, ())):
            return True

        pos += 1
        node_pos = 0

    return False


def _make_component_primes(components):
    return [set(p for p, _node_id, _r in comp) for comp in components]


# ---------------------------------------------------------------------------
# Main incremental graph builder
# ---------------------------------------------------------------------------

def build_residue_graph_incremental(
    precomputed_residues,
    prime_pool,
    height_bound,
    v_tuple=None,
    margin=MIN_MARGIN_OVER_BOX,
    max_chains=None,
    stats_counter=None,
    progress=True,
    use_arc_consistency=False,
    arc_consistency_max_rounds=None,
    max_calls_per_generation=2_000_000,
    min_clique_size=MIN_PRIME_SUBSET_SIZE,
    max_clique_size=MIN_MAX_PRIME_SUBSET_SIZE,
    reconcile_components=False,
    max_reconcile_pairs=2_000_000,
    *,
    seed=0,
    prime_mix_temperature=0.35,
    prime_mix_branch_penalty=0.5,
    max_prime_choices_per_chain_generation=8,
    max_chain_calls_per_generation=256,
    lift_cache_size=500_000,
    max_total_calls=None,
    time_budget_sec=None,
    **_compat_kwargs,
):
    """
    Build a residue graph using a budgeted, mixed-order incremental CRT search.

    Important behavioral changes from the old implementation:

    1. The generation budget is a *hard call budget*.  We do not first compute
       an enormous projected fanout and then wait for it before doing useful
       work.
    2. Chain work is round-robin and resumable.  A giant residue domain cannot
       monopolize the process.
    3. Equivalent CRT states are deduplicated.
    4. Liftability tests are cached.
    5. Prime order is mixed using an information/branching score plus seeded
       randomization.
    6. `max_total_calls` and `time_budget_sec` provide hard global escape hatches.

    Defaults preserve the existing public thresholds while making the search
    substantially less repetitive.
    """
    # Defensive compatibility boundary: some older callers may still pass
    # diagnostic-only keywords directly to this low-level builder.  They have
    # no effect on the CRT search, so consume them here rather than crashing
    # inside worker processes.
    _compat_kwargs.pop("known_m", None)
    _compat_kwargs.pop("label", None)
    _compat_kwargs.pop("debug", None)
    _compat_kwargs.pop("verbose", None)
    _compat_kwargs.pop("print_header", None)

    pool = sorted({int(p) for p in prime_pool})
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = int(margin) * box
    rng = random.Random(seed)

    if int(max_calls_per_generation) <= 0:
        raise ValueError("max_calls_per_generation must be positive")
    if int(max_prime_choices_per_chain_generation) <= 0:
        raise ValueError("max_prime_choices_per_chain_generation must be positive")
    if int(max_chain_calls_per_generation) <= 0:
        raise ValueError("max_chain_calls_per_generation must be positive")
    if int(min_clique_size) < 1:
        raise ValueError("min_clique_size must be >= 1")
    if int(max_clique_size) < int(min_clique_size):
        raise ValueError("max_clique_size must be >= min_clique_size")

    lift_cache = _LiftCache(lift_cache_size)

    # ------------------------------------------------------------------
    # Build / optionally prune residue domains.
    # ------------------------------------------------------------------
    if use_arc_consistency:
        nodes_by_prime, ac_rounds = arc_consistency_prune_domains(
            precomputed_residues,
            pool,
            H,
            v_tuple=v_tuple,
            max_rounds=arc_consistency_max_rounds,
            stats_counter=stats_counter,
            progress=progress,
            seed=seed,
            lift_cache_size=lift_cache_size,
        )
        nodes_by_prime = defaultdict(list, nodes_by_prime)
    else:
        nodes_by_prime = _nodes_by_prime(
            precomputed_residues, pool, v_tuple=v_tuple
        )
        ac_rounds = 0

    all_nodes = [
        nk
        for p in pool
        for nk in nodes_by_prime.get(p, ())
    ]

    uf = _UnionFind()
    for nk in all_nodes:
        uf.add(nk)

    primes_with_data = [
        p for p in pool if nodes_by_prime.get(p)
    ]

    if progress:
        print(
            f"[residue_graph_incremental] start: {len(all_nodes):,} nodes, "
            f"{len(primes_with_data)} primes, threshold={threshold:,}, "
            f"lift-cache={lift_cache.maxsize:,}"
        )

    if len(primes_with_data) < 2:
        return {
            "components": [[nk] for nk in all_nodes],
            "component_primes": [{nk[0]} for nk in all_nodes],
            "edges_tested": 0,
            "edges_kept": 0,
            "nodes": len(all_nodes),
            "max_generation_reached": 0,
            "chains_confirmed": 0,
            "confirmed_chains": [],
            "generation_log": [],
            "cache_stats": lift_cache.stats(),
            "arc_consistency_rounds": ac_rounds,
        }

    mixed_prime_order = _mixed_prime_order(
        primes_with_data,
        nodes_by_prime,
        rng=rng,
        temperature=prime_mix_temperature,
        branch_penalty=prime_mix_branch_penalty,
    )

    def make_task(p, nk):
        return _ChainTask(
            used_primes=frozenset((p,)),
            modulus=int(p),
            residue=int(nk[2]) % int(p),
            node_keys=(nk,),
            prime_order=mixed_prime_order,
            prime_start=(
                rng.randrange(len(mixed_prime_order))
                if mixed_prime_order
                else 0
            ),
        )

    work = deque()
    for p in primes_with_data:
        for nk in nodes_by_prime[p]:
            work.append(make_task(p, nk))

    edges_tested = 0
    edges_kept = 0
    chains_confirmed = 0
    generation = 1
    total_calls = 0
    generation_log = []
    confirmed_chains = []
    start_wall = time.monotonic()

    # Deduplication keyed by the complete CRT state.  Same state means the same
    # residue at every used prime, so one representative path is enough for the
    # expensive search.  Equivalent paths are still unioned immediately.
    generated_state_seen = set()
    confirmed_state_seen = set()

    stopped_reason = None

    pbar = None

    try:
        while work:
            gen_start = time.monotonic()
            gen_start_calls = total_calls
            if (
                time_budget_sec is not None
                and time.monotonic() - start_wall >= float(time_budget_sec)
            ):
                stopped_reason = "time_budget"
                break

            # Beam cap is applied to tasks before expansion.
            if max_chains is not None and len(work) > int(max_chains):
                work_list = list(work)

                def beam_key(task):
                    # Prefer deeper chains, then larger modulus, then smaller
                    # residue-domain cost.
                    remaining_cost = sum(
                        len(nodes_by_prime.get(p, ()))
                        for p in task.prime_order
                        if p not in task.used_primes
                    )
                    return (
                        len(task.used_primes),
                        task.modulus,
                        -remaining_cost,
                    )

                work_list.sort(key=beam_key, reverse=True)
                dropped = len(work_list) - int(max_chains)
                work = deque(work_list[: int(max_chains)])

                if stats_counter is not None:
                    stats_counter[
                        "residue_graph_incremental_beam_capped"
                    ] += dropped

                if progress:
                    print(
                        f"[residue_graph_incremental] gen {generation}: "
                        f"beam capped; dropped {dropped:,}, kept "
                        f"{len(work):,}"
                    )

            # Confirmed / overgrown tasks can be peeled before doing any calls.
            pending = deque()
            already_done = 0

            while work:
                task = work.popleft()
                used_count = len(task.used_primes)

                if (
                    used_count >= int(min_clique_size)
                    or task.modulus > threshold
                ):
                    first = task.node_keys[0]
                    for other in task.node_keys[1:]:
                        uf.union(first, other)

                    chains_confirmed += 1
                    confirmed_chains.append(
                        {
                            "primes": sorted(task.used_primes),
                            "modulus": task.modulus,
                            "residue": task.residue,
                            "node_keys": list(task.node_keys),
                        }
                    )
                    already_done += 1
                    continue

                if used_count >= int(max_clique_size):
                    if stats_counter is not None:
                        stats_counter[
                            "residue_graph_incremental_overgrown_dropped"
                        ] += 1
                    continue

                # Reset generation-local fairness counters.
                task.calls_this_generation = 0
                task.primes_this_generation = 0
                pending.append(task)

            work = pending

            calls_budget = int(max_calls_per_generation)
            pbar = tqdm(
                total=calls_budget,
                desc=f"  gen {generation} CRT/lift",
                disable=not progress,
                leave=False,
                mininterval=0.5,
            )

            next_state = {}

            # Round-robin at the individual edge/candidate level.  This is the
            # main change that prevents a single enormous fanout from looking
            # like the program has frozen.
            while work:
                if (
                    max_total_calls is not None
                    and total_calls >= int(max_total_calls)
                ):
                    stopped_reason = "max_total_calls"
                    break

                if total_calls - gen_start_calls >= calls_budget:
                    break

                if (
                    time_budget_sec is not None
                    and time.monotonic() - start_wall >= float(time_budget_sec)
                ):
                    stopped_reason = "time_budget"
                    break

                task = work.popleft()

                edge = _advance_task(
                    task,
                    nodes_by_prime,
                    used_limit=max_prime_choices_per_chain_generation,
                )

                if edge is None:
                    continue

                p_next, nk = edge
                r_next = nk[2]
                new_M = task.modulus * int(p_next)
                new_c = crt_cached(
                    (task.residue, r_next),
                    (task.modulus, int(p_next)),
                )

                edges_tested += 1
                total_calls += 1
                pbar.update(1)

                if _cache_lift_from_optional(
                    lift_cache, int(new_c) % new_M, new_M, H
                ):
                    edges_kept += 1

                    child_used = task.used_primes | {int(p_next)}
                    state_key = (child_used, int(new_c) % new_M)

                    child_path = task.node_keys + (nk,)

                    old_path = next_state.get(state_key)
                    if old_path is None:
                        next_state[state_key] = child_path
                    else:
                        _union_paths(uf, old_path, child_path)

                    # Keep growing the parent task too, unless its prime search
                    # is exhausted.  It is resumed fairly in the same generation
                    # if budget remains.
                elif stats_counter is not None:
                    stats_counter[
                        "residue_graph_incremental_pruned"
                    ] += 1

                if (
                    task.calls_this_generation
                    < int(max_chain_calls_per_generation)
                    and _task_has_more_work(task, nodes_by_prime)
                ):
                    work.append(task)

                if (
                    max_total_calls is not None
                    and total_calls >= int(max_total_calls)
                ):
                    stopped_reason = "max_total_calls"
                    break

            if stopped_reason is None:
                # Budget exhaustion is represented by unfinished `work`; those
                # tasks are resumed next generation rather than restarted.
                if work:
                    next_work = deque(work)
                else:
                    next_work = deque()
            else:
                next_work = deque(work)

            # Add the children created this generation.  Confirm immediately
            # when a child already satisfies the stopping criterion; this avoids
            # making the user wait for an otherwise-empty next generation.
            for (used, c), node_keys in next_state.items():
                modulus = 1
                for p in used:
                    modulus *= int(p)
                residue = int(c) % modulus
                dedup_key = (frozenset(used), residue)

                if (
                    len(used) >= int(min_clique_size)
                    or modulus > threshold
                ):
                    if dedup_key not in confirmed_state_seen:
                        confirmed_state_seen.add(dedup_key)
                        first = node_keys[0]
                        for other in node_keys[1:]:
                            uf.union(first, other)
                        chains_confirmed += 1
                        confirmed_chains.append(
                            {
                                "primes": sorted(used),
                                "modulus": modulus,
                                "residue": residue,
                                "node_keys": list(node_keys),
                            }
                        )
                    continue

                if dedup_key in generated_state_seen:
                    continue
                generated_state_seen.add(dedup_key)

                next_work.append(
                    _ChainTask(
                        used_primes=frozenset(used),
                        modulus=modulus,
                        residue=residue,
                        node_keys=tuple(node_keys),
                        prime_order=mixed_prime_order,
                        prime_start=(
                            rng.randrange(len(mixed_prime_order))
                            if mixed_prime_order
                            else 0
                        ),
                    )
                )

            pbar.close()
            pbar = None

            gen_elapsed = time.monotonic() - gen_start
            generation_calls = total_calls - gen_start_calls

            # If no child and no resumable parent remain, search is done.
            generation_log.append(
                {
                    "generation": generation,
                    "chains_in": len(next_work),
                    "chains_confirmed_this_gen": already_done,
                    "chains_out": len(next_work),
                    "inner_calls": generation_calls,
                    "elapsed_sec": gen_elapsed,
                    "total_calls": total_calls,
                    "lift_cache_hits": lift_cache.hits,
                    "lift_cache_misses": lift_cache.misses,
                    "stopped_reason": stopped_reason,
                }
            )

            if progress:
                rate = (
                    generation_calls / gen_elapsed
                    if gen_elapsed > 0
                    else float("inf")
                )
                print(
                    f"[residue_graph_incremental] gen {generation}: "
                    f"{generation_calls:,} calls in {gen_elapsed:.2f}s "
                    f"({rate:,.0f}/sec), "
                    f"next frontier={len(next_work):,}, "
                    f"children={len(next_state):,}, "
                    f"confirmed_total={chains_confirmed:,}, "
                    f"cache_hits={lift_cache.hits:,}"
                )

            if stopped_reason is not None:
                break

            work = next_work
            if not work:
                break

            generation += 1

    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"
        if progress:
            print(
                f"[residue_graph_incremental] INTERRUPTED at gen {generation}: "
                f"{len(work):,} work items remain"
            )
    finally:
        if pbar is not None:
            pbar.close()

    # ------------------------------------------------------------------
    # Optional component reconciliation.
    # ------------------------------------------------------------------
    comp_map = _component_map(all_nodes, uf)
    components_pre_reconcile = list(comp_map.values())

    if reconcile_components and len(components_pre_reconcile) > 1:
        reconciled = 0
        pairs_checked = 0
        budget_hit = False

        def component_modulus_and_residue(comp):
            by_prime = defaultdict(list)
            for (p, _vt, r) in comp:
                by_prime[p].append(r)

            primes_here = sorted(by_prime.keys())
            M = primes_here[0]
            c = by_prime[primes_here[0]][0]

            for p in primes_here[1:]:
                r = by_prime[p][0]
                c = crt_cached((c, r), (M, p))
                M *= p

            return M, c, set(primes_here)

        comp_reps = [
            component_modulus_and_residue(comp)
            for comp in components_pre_reconcile
        ]

        for i in range(len(components_pre_reconcile)):
            if budget_hit:
                break

            M_i, c_i, primes_i = comp_reps[i]
            for j in range(i + 1, len(components_pre_reconcile)):
                if pairs_checked >= int(max_reconcile_pairs):
                    budget_hit = True
                    if stats_counter is not None:
                        stats_counter[
                            "residue_graph_reconcile_budget_exhausted"
                        ] += 1
                    break

                pairs_checked += 1
                M_j, c_j, primes_j = comp_reps[j]
                if primes_i & primes_j:
                    continue

                M_ij = M_i * M_j
                c_ij = crt_cached((c_i, c_j), (M_i, M_j))
                if lift_cache(c_ij, M_ij, H):
                    uf.union(
                        components_pre_reconcile[i][0],
                        components_pre_reconcile[j][0],
                    )
                    reconciled += 1
                    if stats_counter is not None:
                        stats_counter[
                            "residue_graph_reconciled_components"
                        ] += 1

        if progress and budget_hit:
            print(
                "[residue_graph_incremental] reconciliation pass stopped early: "
                f"max_reconcile_pairs={max_reconcile_pairs:,}"
            )
        if progress and reconciled:
            print(
                "[residue_graph_incremental] reconciliation pass: "
                f"merged {reconciled} component pair(s)"
            )

    comp_map = _component_map(all_nodes, uf)
    components = sorted(comp_map.values(), key=len, reverse=True)
    component_primes = _make_component_primes(components)

    if stopped_reason and progress:
        print(
            f"[residue_graph_incremental] stopped early: {stopped_reason}; "
            f"frontier={len(work):,}, total_calls={total_calls:,}"
        )

    result = {
        "components": components,
        "component_primes": component_primes,
        "edges_tested": edges_tested,
        "edges_kept": edges_kept,
        "nodes": len(all_nodes),
        "max_generation_reached": generation,
        "chains_confirmed": chains_confirmed,
        "confirmed_chains": confirmed_chains,
        "generation_log": generation_log,
        "cache_stats": lift_cache.stats(),
        "arc_consistency_rounds": ac_rounds,
        "total_calls": total_calls,
        "stopped_reason": stopped_reason,
        "seed": seed,
        "prime_mix_temperature": prime_mix_temperature,
        "prime_mix_branch_penalty": prime_mix_branch_penalty,
    }
    return result


def build_residue_chains_ordered(
    precomputed_residues,
    prime_pool,
    height_bound,
    v_tuple=None,
    margin=MIN_MARGIN_OVER_BOX,
    max_chains=None,
    stats_counter=None,
    progress=True,
    windows=12,
    max_states_per_layer=2_000_000,
    lift_cache_size=500_000,
    **_ignored,
):
    """
    Fixed-order layered CRT sieve (largest primes first).

    Each window starts one prime further down the descending prime list and
    multiplies residue domains layer by layer until the modulus exceeds
    margin*(2H+1)^2, so only as many primes are used as are needed to make
    the lift test informative (typically ~5 large primes, not k smallest).
    Layers whose modulus is still <= box are not lift-tested (the test is a
    tautology there).  A prime with no true residue kills every window that
    spans it; more windows = more robustness to bad primes, at linear cost.

    Returns a dict shaped like build_residue_graph_incremental's result
    (`confirmed_chains` holds one chain per surviving CRT state).
    """
    pool = sorted({int(p) for p in prime_pool})
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = int(margin) * box
    nodes_by_prime = _nodes_by_prime(precomputed_residues, pool, v_tuple=v_tuple)
    order = sorted((p for p in pool if nodes_by_prime.get(p)), reverse=True)
    n_nodes = sum(len(v) for v in nodes_by_prime.values())
    lift_cache = _LiftCache(lift_cache_size)

    tested = kept = 0
    chains, seen = [], set()
    stopped_reason = None

    for w in range(min(int(windows), len(order))):
        frontier = [(1, 0, ())]
        reached = False
        for p in order[w:]:
            nxt = []
            for (M, c, path) in frontier:
                inv = pow(M, -1, p)
                M2 = M * p
                for nk in nodes_by_prime[p]:
                    c2 = c + M * (((nk[2] - c) * inv) % p)
                    tested += 1
                    if M2 <= box or lift_cache(c2, M2, H):
                        kept += 1
                        nxt.append((M2, c2, path + (nk,)))
            frontier = nxt
            if not frontier:
                break
            if len(frontier) > int(max_states_per_layer):
                stopped_reason = "max_states_per_layer"
                break
            if frontier[0][0] > threshold:
                reached = True
                break
        if reached:
            for (M, c, path) in frontier:
                key = (M, c)
                if key in seen:
                    continue
                seen.add(key)
                chains.append({
                    "primes": sorted(k[0] for k in path),
                    "modulus": M,
                    "residue": c,
                    "node_keys": list(path),
                })
        if max_chains is not None and len(chains) >= int(max_chains):
            stopped_reason = "max_chains"
            break

    if progress:
        print(
            f"[residue_graph_ordered] {n_nodes:,} nodes, {len(order)} primes, "
            f"windows={min(int(windows), len(order))}, tested={tested:,}, "
            f"kept={kept:,}, chains={len(chains):,}"
            + (f", stopped: {stopped_reason}" if stopped_reason else "")
        )

    return {
        "components": [],
        "component_primes": [],
        "edges_tested": tested,
        "edges_kept": kept,
        "nodes": n_nodes,
        "max_generation_reached": 0,
        "chains_confirmed": len(chains),
        "confirmed_chains": chains,
        "generation_log": [],
        "cache_stats": lift_cache.stats(),
        "stopped_reason": stopped_reason,
        "strategy": "ordered",
    }



def build_residue_graph_ktuple(
    precomputed_residues,
    prime_pool,
    height_bound,
    k=None,
    v_tuple=None,
    margin=MIN_MARGIN_OVER_BOX,
    max_tuples=None,
    stats_counter=None,
    progress=False,
    **kwargs,
):
    """
    Compatibility entry point.

    Historically this function was intercepted by the search-analysis layer;
    keep its broad signature while routing to the refactored incremental search.
    """
    # This entry point is intentionally permissive because the surrounding
    # search stack has historically passed diagnostic/compatibility keywords
    # such as `known_m`, `label`, and `debug`.  They are meaningful to the
    # caller/diagnostics, but they are not search parameters for the low-level
    # incremental builder and must not leak into its signature.
    kwargs = dict(kwargs)
    strategy = kwargs.pop("strategy", "incremental")
    honor_k = bool(kwargs.pop("honor_k", False))
    k_needed = k if k is not None else kwargs.pop("k", None)
    kwargs.pop("k", None)
    kwargs.pop("max_tuples", None)
    compatibility_only = {
        "known_m",
        "label",
        "debug",
        "verbose",
        "print_header",
    }
    for key in compatibility_only:
        kwargs.pop(key, None)

    # Only forward parameters explicitly understood by the incremental
    # implementation.  Silently ignoring unknown kwargs preserves the old
    # k-tuple API's compatibility behavior instead of turning an otherwise
    # successful graph search into a TypeError.
    ordered_keys = {"max_chains", "windows", "max_states_per_layer", "lift_cache_size"}
    if strategy == "ordered":
        fwd = {key: kwargs[key] for key in ordered_keys if key in kwargs}
        fwd.setdefault("max_chains", 20_000)
        return build_residue_chains_ordered(
            precomputed_residues=precomputed_residues,
            prime_pool=prime_pool,
            height_bound=height_bound,
            v_tuple=v_tuple,
            margin=margin,
            stats_counter=stats_counter,
            progress=progress,
            **fwd,
        )

    incremental_keys = {
        "max_chains",
        "use_arc_consistency",
        "arc_consistency_max_rounds",
        "max_calls_per_generation",
        "min_clique_size",
        "max_clique_size",
        "reconcile_components",
        "max_reconcile_pairs",
        "seed",
        "prime_mix_temperature",
        "prime_mix_branch_penalty",
        "max_prime_choices_per_chain_generation",
        "max_chain_calls_per_generation",
        "lift_cache_size",
        "max_total_calls",
        "time_budget_sec",
    }
    forward = {key: kwargs.pop(key) for key in list(kwargs) if key in incremental_keys}
    # OPT-IN (honor_k=True): `k` has always been dropped here, so chains are
    # confirmed at min_clique_size (default 3) and the exact on-curve check
    # does the real filtering.  That is what finds points present at only a
    # few primes; requiring k primes needs the point at >= k primes.
    if honor_k and k_needed is not None:
        forward.setdefault("min_clique_size", int(k_needed))
        forward["max_clique_size"] = max(
            int(forward.get("max_clique_size", MIN_MAX_PRIME_SUBSET_SIZE)),
            int(forward["min_clique_size"]),
        )

    return build_residue_graph_incremental(
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        height_bound=height_bound,
        v_tuple=v_tuple,
        margin=margin,
        max_chains=forward.pop("max_chains", 20_000),
        stats_counter=stats_counter,
        progress=progress,
        **forward,
    )


# ---------------------------------------------------------------------------
# Legacy pairwise graph builder
# ---------------------------------------------------------------------------

def build_residue_graph(
    precomputed_residues,
    prime_pool,
    height_bound,
    v_tuple=None,
    require_unique_modulus=False,
    require_margin_over_box=True,
    max_primes=None,
    stats_counter=None,
):
    """
    Build the legacy pairwise CRT-compatibility graph.

    This path is retained for callers that specifically want pairwise edges.
    """
    pool = (
        list(prime_pool)
        if max_primes is None
        else list(prime_pool)[: int(max_primes)]
    )
    H = int(height_bound)
    nodes_by_prime = defaultdict(list)

    for p, node_id, r in _iter_residue_nodes(
        precomputed_residues, pool, v_tuple=v_tuple
    ):
        nodes_by_prime[p].append(((p, node_id, r), r))

    all_nodes = [
        nk
        for lst in nodes_by_prime.values()
        for nk, _ in lst
    ]

    uf = _UnionFind()
    for nk in all_nodes:
        uf.add(nk)

    lift_cache = _LiftCache()
    primes_sorted = sorted(nodes_by_prime.keys())

    edges_tested = 0
    edges_kept = 0
    box = (2 * H + 1) ** 2

    for i, p in enumerate(primes_sorted):
        p_nodes = nodes_by_prime[p]
        for q in primes_sorted[i + 1:]:
            q_nodes = nodes_by_prime[q]
            if not p_nodes or not q_nodes:
                continue

            M = int(p) * int(q)
            if require_margin_over_box:
                informative = M > MIN_MARGIN_OVER_BOX * box
            elif require_unique_modulus:
                informative = M > 2 * H * H
            else:
                informative = modulus_is_informative(M, H)

            if stats_counter is not None and not informative:
                stats_counter["residue_graph_uninformative_pair"] += 1
            if not informative:
                continue

            for nk_p, r_p in p_nodes:
                for nk_q, r_q in q_nodes:
                    edges_tested += 1
                    c = crt_cached((r_p, r_q), (p, q))
                    if lift_cache(c, M, H):
                        edges_kept += 1
                        uf.union(nk_p, nk_q)

    components = sorted(
        _component_map(all_nodes, uf).values(),
        key=len,
        reverse=True,
    )
    component_primes = _make_component_primes(components)

    return {
        "components": components,
        "component_primes": component_primes,
        "edges_tested": edges_tested,
        "edges_kept": edges_kept,
        "nodes": len(all_nodes),
        "cache_stats": lift_cache.stats(),
    }


# ---------------------------------------------------------------------------
# Component summaries / reconstruction
# ---------------------------------------------------------------------------

def summarize_components(graph_result, top_k=10):
    """Return compact rows for the largest components."""
    rows = []
    for comp, comp_primes in zip(
        graph_result["components"][: int(top_k)],
        graph_result["component_primes"][: int(top_k)],
    ):
        rows.append(
            {
                "num_nodes": len(comp),
                "num_primes": len(comp_primes),
                "primes": sorted(comp_primes),
            }
        )
    return rows


def refine_component_chained(
    comp,
    height_bound,
    stats_counter=None,
    lift_cache_size=250_000,
):
    """
    Check whether a component contains a full CRT-consistent chain.
    """
    H = int(height_bound)
    by_prime = defaultdict(list)
    for p, _node_id, r in comp:
        by_prime[p].append(r)

    primes = sorted(by_prime.keys())
    if len(primes) < 2:
        return {
            "confirmed": False,
            "modulus_reached": 0,
            "primes_used": [],
        }

    lift_cache = _LiftCache(lift_cache_size)
    best_modulus_reached = 0

    for combo in itertools.product(*(by_prime[p] for p in primes)):
        modulus = int(primes[0])
        residue = int(combo[0]) % modulus
        chain_ok = True

        for p, r in zip(primes[1:], combo[1:]):
            new_modulus = modulus * int(p)
            residue = crt_cached(
                (residue, r),
                (modulus, int(p)),
            )
            modulus = new_modulus

            if not lift_cache(residue, modulus, H):
                chain_ok = False
                if stats_counter is not None:
                    stats_counter["residue_graph_chain_broke"] += 1
                break

        best_modulus_reached = max(
            best_modulus_reached,
            modulus if chain_ok else 0,
        )

        if chain_ok and modulus > 2 * H * H:
            return {
                "confirmed": True,
                "modulus_reached": modulus,
                "primes_used": list(primes),
                "cache_stats": lift_cache.stats(),
            }

    return {
        "confirmed": False,
        "modulus_reached": best_modulus_reached,
        "primes_used": [],
        "cache_stats": lift_cache.stats(),
    }


def reconstruct_candidate_from_chain(
    chain,
    height_bound,
    max_den=None,
):
    """Reconstruct one rational from an already-confirmed CRT chain."""
    H = int(height_bound)
    M = int(chain["modulus"])
    c = int(chain["residue"])

    try:
        a, b = rational_reconstruct(
            c % M,
            M,
            max_den=max_den or H,
        )
        return {
            "m_num": a,
            "m_den": b,
            "primes": chain["primes"],
            "modulus": M,
        }
    except Exception:
        return None


def reconstruct_candidates_from_component(
    comp,
    height_bound,
    max_den=None,
    max_results=5,
    stats_counter=None,
    max_visits=2_000_000,
    lift_cache_size=250_000,
):
    """
    Enumerate bounded CRT-consistent candidates from one component.

    The component's primes are visited from smallest residue domain to largest.
    """
    by_prime = defaultdict(list)
    for p, _node_id, r in comp:
        by_prime[p].append(r)

    primes = sorted(
        by_prime.keys(),
        key=lambda p: len(by_prime[p]),
    )
    if len(primes) < 2:
        return []

    H = int(height_bound)
    results = []
    visits = 0
    budget_exhausted = False
    lift_cache = _LiftCache(lift_cache_size)

    def recurse(idx, modulus, residue):
        nonlocal visits, budget_exhausted

        if budget_exhausted:
            return
        if max_results is not None and len(results) >= int(max_results):
            return

        visits += 1
        if max_visits is not None and visits > int(max_visits):
            budget_exhausted = True
            if stats_counter is not None:
                stats_counter[
                    "reconstruct_visit_budget_exhausted"
                ] += 1
            return

        if idx == len(primes):
            try:
                a, b = rational_reconstruct(
                    int(residue) % int(modulus),
                    int(modulus),
                    max_den=max_den or H,
                )
            except Exception:
                return

            results.append(
                {
                    "m_num": a,
                    "m_den": b,
                    "primes": primes,
                    "modulus": modulus,
                }
            )
            return

        p = int(primes[idx])

        for r in by_prime[p]:
            if idx == 0:
                new_modulus = p
                new_residue = int(r) % p
            else:
                new_modulus = modulus * p
                new_residue = crt_cached(
                    (residue, r),
                    (modulus, p),
                )

            if not lift_cache(new_residue, new_modulus, H):
                if stats_counter is not None:
                    stats_counter[
                        "reconstruct_branch_pruned"
                    ] += 1
                continue

            recurse(
                idx + 1,
                new_modulus,
                new_residue,
            )
            if budget_exhausted or (
                max_results is not None
                and len(results) >= int(max_results)
            ):
                return

    recurse(0, 1, 0)

    return results
