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

*** NINTH BUG (found via trace_target on a real run: true residue r for
known target m=-1/7 dropped at EVERY prime, arc-consistency round 1-11,
despite a manual 3-prime CRT reconstruction of the same target succeeding
immediately) ***
Both arc_consistency_prune_domains/_has_witness_chain AND this module's
own confirmation condition (`M > threshold`, threshold = margin*box) were
built assuming solutions are "global": a true point's residue should hold
up against a MAJORITY of the pool (witness-chain's success-rate vote), and
a confirmed component's modulus should exceed margin * (2H+1)^2, which for
a 23-prime pool requires ~10 primes multiplied together (see
min_tuple_size_for_margin). Confirmed empirically (see project chat log)
that this assumption is simply false for this problem: genuine points are
consistently LOCAL solutions supported by only MIN_PRIME_SUBSET_SIZE..
MIN_MAX_PRIME_SUBSET_SIZE primes (3-9, per search_common.py's own tuning
of the CRT-subset sampler used elsewhere in the pipeline) -- there is
essentially always at least one prime that locally blocks global
agreement, so no filter that rewards broad multi-prime consensus can ever
accept a real point; it will always look, by that filter's own logic,
like "insufficient support" or "modulus too small to trust yet". Beyond
~12 primes the true-support requirement becomes empirically unreachable
(the same "always at least one blocking prime" phenomenon guarantees a
clique that large can't form).

Fix, in three parts:
  1. use_arc_consistency defaults to False here now -- the pre-prune step
     is diagnostic/optional, not a default gate, since its majority-vote
     design actively removes genuine small-clique solutions before the
     graph-building step ever sees them.
  2. build_residue_graph_incremental confirms a chain once it reaches
     MIN_PRIME_SUBSET_SIZE primes (not once its modulus clears
     margin*box) -- clique SIZE, not accumulated modulus magnitude, is
     the right proxy for "this is a real local solution" in this regime.
  3. Chain growth is capped at MIN_MAX_PRIME_SUBSET_SIZE primes -- once a
     chain would extend past that, stop growing it instead of continuing
     to fan out; per this problem's own structure that many primes agreeing
     is already at the edge of what's achievable, and letting an
     over-extended chain keep competing for beam/budget slots only
     starves genuine 3-9-prime cliques of the room they need to survive
     (see "deepest-then-widest" beam-priority comment below -- that
     ordering was tuned for a global-solution model and now systematically
     favors chains that have overgrown past where real solutions live).
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

# _has_witness_chain's own stopping bound, deliberately STRONGER than the
# general-purpose modulus_is_informative(M,H) (which is just M > H --
# see rational_arithmetic.py's docstring: that function's docstring itself
# says callers wanting a genuine uniqueness guarantee should require
# M > 2*H*H instead, but _has_witness_chain was calling the weak M > H
# form). Empirically (see "FOURTH BUG" below), M > H is barely-informative:
# ~99.98% of random residues mod M still pass lattice_rational_lift_exists
# at that threshold, and even M > 2*H*H alone still passes ~45% of random
# residues. Real selectivity (<1% false-positive rate) needs roughly two to
# three primes' worth of extra margin past 2*H*H. 50x is a conservative
# empirical choice (see check_selectivity.py-style benchmarking in chat);
# it costs 1-2 more primes per witness-chain call but is what actually
# makes "modulus is informative" mean something.
_STRONG_INFORMATIVE_MARGIN = 50


def _min_attempts_for_informative(primes_with_data, H, p=None,
                                   margin=_STRONG_INFORMATIVE_MARGIN):
    """
    Lower bound on how many "other" primes a witness chain must successfully
    accumulate before its modulus can POSSIBLY exceed margin*H^2 -- i.e. the
    earliest attempt count at which _has_witness_chain's success condition
    becomes achievable in principle.

    *** BUG THIS FIXES (FIFTH BUG, found after the FOURTH BUG fix above) ***
    _has_witness_chain's failure branch (`successes/attempts < min_success_rate`)
    and its success branch (rate AND modulus > margin*H^2) were both gated on
    the same `attempts >= min_attempts` with a single fixed min_attempts
    (default 6). But the modulus bar and the rate bar need very different
    amounts of accumulated evidence: with H=3700 and margin=50, clearing
    margin*H^2 ~= 6.85e8 requires at least 8 chained primes even in the
    BEST case (taking the 8 largest primes in a 23-prime, max-97 pool) --
    taking them in the order actually offered to the beam (ascending, or
    shuffled) it's typically more. A fixed min_attempts=6 lets the failure
    branch fire two-plus primes before the success branch could possibly
    fire even once, so every residue -- including the true one -- gets
    judged on a success-rate that's structurally unable to succeed yet.
    At avg_roots~1, a couple of coincidental early misses is normal even
    for the correct chain, so this reliably kills genuine chains at round 1
    (confirmed via trace_target_through_arc_consistency on a known point).

    Fix: compute the true best-case floor from the actual prime pool
    (largest primes first, since that's the fastest any chain could clear
    the bar) and use that as a floor under whatever min_attempts the
    caller passed, so the failure branch can never fire before the
    success branch has had a chance to.
    """
    H = int(H)
    threshold = margin * H * H
    others = sorted((int(q) for q in primes_with_data if q != p), reverse=True)
    prod = 1
    for i, q in enumerate(others, 1):
        prod *= q
        if prod > threshold:
            return i
    # Even chaining every available prime doesn't clear the bar -- the
    # pool itself is too small/low at this height bound (see the
    # PRIME_POOL sufficiency diagnostic). Return the full count so the
    # rate check at least isn't the thing that kills it early; the chain
    # will legitimately fail to reach "informative" regardless.
    return len(others)


def _has_witness_chain(p, r_p, nodes_by_prime, primes_with_data, H, sample_cap=None,
                        rng=None, min_success_rate=0.8, min_attempts=6,
                        beam_width=4, max_calls_per_prime_step=32):
    """
    Does residue r_p at prime p have GENUINE cross-prime support -- i.e.
    does it survive lift checks against most of the other primes it's
    tested against (not just SOME prime found anywhere in the pool)?

    *** BUG THIS REPLACED (found via a real run with avg_roots≈1) ***
    The original version walked other primes and, on failing to find a
    compatible partner at prime q, simply skipped q and tried the next
    prime -- "a single bad prime shouldn't kill an otherwise-good
    witness". That's fine for one or two unlucky primes, but at low root
    density (few residues per prime) it makes the check nearly vacuous:
    with e.g. 23 primes and ~10 needed to reach an informative modulus, a
    residue can freely skip the 13 primes where it has no real partner
    and cherry-pick the ~10 where a lift happens to succeed by pigeonhole
    alone -- at avg_roots≈1 that's easy to find coincidentally for almost
    EVERY residue. Confirmed empirically: on a real run with avg_roots=1.0
    arc-consistency reported 1894/1894 residues survived (pruned nothing),
    which is why chain-building then exploded to 34.5M calls/generation
    and OOM'd.

    *** SECOND BUG THIS REPLACED (found via a run where the known target
    m=-1/7 was pruned while a spurious component survived) ***
    The single-accumulator version that replaced the first bug picked,
    at each prime q, the FIRST q-residue that happened to CRT-lift against
    the running (M, c), then permanently committed to it: `M, c = found`.
    At low root density that first accepted witness is frequently the
    WRONG residue at q (coincidentally lift-compatible by pigeonhole, not
    because it's q's true reduction of the real point) -- once folded in,
    every later prime is tested against this now-contaminated (M, c)
    instead of against the true point's CRT value, so a genuine residue
    can be steered onto a self-consistent-but-wrong chain at the very
    first partner prime and then correctly (but wrongly) fail the
    majority-rate test against everything after. Meanwhile some other,
    entirely spurious chain that happened to cohere all the way through
    survives and is reported as the unique candidate.

    Fix: track a small BEAM of live (M, c) accumulator states in parallel
    rather than collapsing to a single greedy choice. At each prime q, every
    live state spawns one child per compatible q-residue; we keep up to
    beam_width children (smallest modulus first) and drop the rest. This
    keeps the true chain alive alongside decoy chains for a few primes, so
    a spurious pigeonhole match early on no longer permanently derails a
    genuine residue.

    *** THIRD BUG (perf hang) THIS REPLACED ***
    The first beam version computed, for EVERY live beam state, a
    crt_cached+lift check against EVERY q-residue at the next prime, with
    no bound on total work per prime step. At the pool's actual pre-prune
    density (~40-80 residues/prime, per the real run's "1780 residues
    originally" over ~23-25 primes) that's beam_width * avg_roots calls
    per prime, compounding across ~22 "other" primes, for EVERY residue
    being tested in the outer arc_consistency_prune_domains loop -- in a
    plain-Python simulation this alone was ~500K crt/lift calls for one
    round; the real Sage crt()/CRT_list() (Integer coercion, generic
    dispatch, growing-multiprecision lcm) is markedly more expensive per
    call than that, which is what actually hung. Bounding fan-out only
    somewhat (beam_width*4) roughly halved the call count but did not
    remove the multiplicative blowup.

    Fix: enforce a hard cap (max_calls_per_prime_step) on total
    crt+lift calls per prime, independent of beam_width and root
    density -- once the cap is hit we stop early with whatever children
    were found so far, so a single _has_witness_chain call is bounded by
    O(others * max_calls_per_prime_step) regardless of how dense the
    pool is. We also keep beam_width itself small (default lowered) since
    a genuine chain only needs to survive with ONE live correct state --
    a wide beam mostly just carries more decoys through more expensive
    per-step work without materially improving the odds the true chain
    is among the survivors.

    Returns True/False. sample_cap optionally caps how many OTHER primes
    are tried (None = try all), trading a small chance of a false
    negative for speed on very large pools.

    *** FOURTH BUG (found via decoy-vs-true simulation after the first
    three fixes above still left every true residue in a real run dying
    in round 1-2 regardless of min_success_rate) ***
    The exit condition `modulus_is_informative(M, H)` calls that function
    at its WEAK threshold (M > H -- see rational_arithmetic.py's own
    docstring, which explicitly says callers wanting a real uniqueness
    guarantee should use M > 2*H*H instead, not M > H). Empirically, at
    M > H a random garbage residue still passes lattice_rational_lift_exists
    ~99.98% of the time; even at M > 2*H*H it's still ~45%. So a chain
    could satisfy "modulus is informative" after just 3-4 primes, at a
    point where the lift-check has essentially no power to distinguish a
    true chain from an accumulating coincidence -- lowering
    min_success_rate didn't fix decoy survival because the underlying
    "have we accumulated enough real evidence" signal was itself firing
    far too early, independent of the success-rate bar. Fixed by requiring
    _STRONG_INFORMATIVE_MARGIN * H^2 (empirically <1% false-positive rate)
    before trusting the modulus as a stopping point, rather than deferring
    to the general-purpose helper's weaker default.
    """
    others = [q for q in primes_with_data if q != p]
    if rng is not None:
        others = list(others)
        rng.shuffle(others)
    if sample_cap is not None:
        others = others[:sample_cap]

    # FIFTH BUG fix (see _min_attempts_for_informative docstring): don't
    # let the failure branch below fire before the success branch could
    # possibly fire, i.e. before enough primes have even been offered to
    # reach an informative modulus in the best case.
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
        # SIXTH BUG fix (found via a run with avg_roots~82, much denser
        # than the avg_roots~1 regime the caps above were tuned against):
        # max_calls_per_prime_step (default 32) was applied by walking
        # q_nodes in whatever order precomputed_residues happened to
        # store them in -- NOT randomized. At avg_roots~82 with a single
        # live beam state, that means only the FIRST 32 of ~82 candidate
        # residues at q are ever even tested; if the true residue isn't
        # among those first 32 (nothing guarantees it is -- storage order
        # has no relation to correctness), it's silently never checked
        # at all, indistinguishable from a genuine incompatibility in the
        # success/failure accounting. Confirmed via trace_target: true
        # residues that survived round 1 (so were genuinely being found)
        # were still dying at round 2, consistent with this kind of
        # order-dependent starvation rather than real incompatibility.
        # Fix: shuffle q_nodes per step so the cap is an unbiased
        # subsample of q's residues rather than a systematic truncation
        # of whichever ones happen to sort first.
        q_nodes_this_step = list(q_nodes)
        if rng is not None:
            rng.shuffle(q_nodes_this_step)
        # Also scale the per-state call budget to the observed root
        # density at q, so a state still gets a reasonable *fraction* of
        # q's residues examined even when avg_roots is far above the
        # regime max_calls_per_prime_step's default was tuned for,
        # instead of an ever-shrinking fraction as density grows.
        effective_cap = max(max_calls_per_prime_step, 4 * len(q_nodes_this_step) // max(1, len(beam)))
        # Hard cap: total crt+lift calls this prime step, regardless of
        # beam_width or how many residues q has. This is what actually
        # bounds worst-case cost -- beam_width and dedup alone don't,
        # since both scale with root density.
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
            # Keep the SMALLEST-modulus survivors, capped at beam_width --
            # but note this alone doesn't protect the true chain from
            # eviction at high root density; the shuffle above (so a
            # dropped true-chain child had a fair chance of being FOUND
            # in the first place) is the primary fix for that. A
            # genuinely correct chain has no reason to prefer
            # accumulating a large modulus, and preferring small M keeps
            # subsequent crt_cached/lcm calls cheap.
            uniq = {}
            for (M2, c2) in children:
                uniq[(M2, c2)] = None  # dedupe exact repeats
            beam = sorted(uniq.keys(), key=lambda mc: mc[0])[:beam_width]
        # else: genuine failure -- NO live beam state had a compatible
        # partner at q (within the call budget). Counted against the
        # rate, and the beam is left as-is (carried forward unchanged) so
        # a single bad prime doesn't wipe out otherwise-live states.

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
    Arc-consistency-style prune over the residue *domains*, before any
    chain-building happens.

    For each prime p and each residue a in its domain D_p, a survives only
    if _has_witness_chain finds SOME sequence of compatible partner
    residues at other primes whose accumulated CRT modulus becomes
    informative. A residue with no such witness chain cannot possibly
    participate in a genuine cross-prime solution at this height bound, so
    it's dead weight for the main chain-building pass and gets dropped.

    NOTE: this checks against MULTI-prime witness chains, not bare pairs --
    see _has_witness_chain's docstring for why a pairwise-only version is
    unusable at realistic prime sizes (no pair alone clears
    modulus_is_informative, so a pairwise prune would report zero support
    for every residue, always, regardless of correctness).

    This is classic AC-3 in spirit: dropping a residue can remove the only
    witness that kept some OTHER residue (at a third prime) alive, so we
    iterate to a fixed point rather than doing one sweep.

    Returns (pruned_nodes_by_prime, rounds_run) where pruned_nodes_by_prime
    is a dict prime -> list of (p, node_id, r) node tuples, i.e. the same
    shape nodes_by_prime has in build_residue_graph_incremental, just with
    unsupported residues removed.

    witness_beam_width / witness_max_calls_per_prime_step tune
    _has_witness_chain's cost/thoroughness tradeoff (see its docstring for
    why both are needed -- beam_width alone doesn't bound worst-case cost
    at high root density, only the per-step call cap does). Lower them if
    a run is too slow at high root density; raise witness_beam_width if
    you suspect a real chain is being missed because a decoy is crowding
    it out of a too-narrow beam.

    witness_min_success_rate / witness_min_attempts tune how strict the
    majority-vote test is (see _has_witness_chain's docstring, "FOURTH
    BUG" section). The defaults (0.8, 6) were tuned assuming denser root
    data than avg_roots~1 actually provides -- at that density a true
    chain has NO fallback root at a prime where the recorded data is
    off, so it needs real tolerance for a couple of early misses. Lower
    witness_min_success_rate (e.g. 0.5-0.6) at low avg_roots; the
    trade-off is weaker pruning of genuine decoys, not a correctness
    issue -- chain-building afterward still requires actual CRT
    consistency to survive, this is only a coarse pre-filter.
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
    Replaces build_residue_graph_ktuple's exhaustive C(n,k) strategy with
    incremental beam growth to strictly avoid combinatorial blowup.

    use_arc_consistency=True (default): before chain-building, run
    arc_consistency_prune_domains to drop residues with no compatible
    partner at ANY other prime. This shrinks each prime's domain -- often
    drastically -- which is what makes it safe to then let chains extend
    into ANY remaining prime (not just the next one in sorted order): the
    old restriction to next_idx = last_idx + 1 was a hack to bound the
    branching factor, but it also meant a chain starting at prime p_i could
    only ever combine with p_{i+1}, p_{i+2}, ... in that fixed order --
    genuinely correct cross-prime combinations that happened to require
    "going backwards" in sorted-prime order were structurally unreachable,
    which is almost certainly why runs were producing only degenerate
    single-prime components. Pruning first makes full fan-out affordable
    again without reintroducing the O(remaining_primes)^depth blowup that
    motivated the next_idx restriction in the first place.

    *** EIGHTH BUG (OOM, found via a real run at avg_roots~82) ***
    arc_consistency_prune_domains and this function's fan-out are in
    tension: the SEVENTH BUG fix lowered witness_min_success_rate to 0.35
    so true chains with ~60% coverage wouldn't be wrongly killed, but at
    that density almost EVERY residue (true or decoy) clears such a
    permissive bar too -- lattice_rational_lift_exists is essentially a
    rubber stamp at 1-2 chained primes (M far below the (2H+1)^2 box), so
    the rate check can't discriminate signal from noise until many primes
    deep, well past where min_attempts forces an early decision. Result:
    arc-consistency pruned almost nothing (1892/1894 -> 1884/1894 residues
    survived, not the ~80% reduction it achieves at lower avg_roots), so
    build_residue_graph_incremental's chain-extension loop received nearly
    the FULL unpruned pool. max_chains (beam cap) bounds how many chains
    are CARRIED FORWARD between generations, but does nothing to bound the
    cost of COMPUTING the next generation from that many chains: with
    max_chains=20000 and avg_roots~82 over ~21 remaining primes, one
    generation alone is ~20000*82*21 ~= 34M CRT/lift calls, which is what
    actually OOM'd (each call retains state -- new (M,c) tuples -- so
    memory, not just time, scales with total calls attempted per
    generation, not just the survivors kept at the end of it).

    Fix: cap total CRT/lift calls PER GENERATION directly
    (max_calls_per_generation), independent of max_chains and root
    density, the same way SIXTH BUG's max_calls_per_prime_step bounds
    _has_witness_chain. When the cap would be exceeded, shrink how many
    chains are extended this generation (not how many candidate residues
    each chain is tested against, since a genuine chain needs its actual
    true partner to be among those tested) -- prioritizing the same
    depth-first ordering already used for the max_chains cap, so it's the
    least-promising (shallowest) chains that get deferred/dropped first,
    not likely-genuine deep ones.

    *** NINTH BUG fix (see module docstring) ***
    use_arc_consistency now defaults to False: this problem's solutions
    are consistently LOCAL (supported by only min_clique_size..
    max_clique_size primes, per search_common's own MIN_PRIME_SUBSET_SIZE/
    MIN_MAX_PRIME_SUBSET_SIZE), so the majority-vote pre-prune removes
    genuine residues before they ever reach chain-building. Pass
    use_arc_consistency=True explicitly if you want the old pre-prune
    behavior for comparison/diagnostics.

    Confirmation is now by CLIQUE SIZE (min_clique_size), not accumulated
    modulus (`M > margin*box`) -- a chain that has reached min_clique_size
    mutually-lift-compatible primes IS a candidate local solution in this
    regime, regardless of whether its modulus happens to be large. margin/
    threshold are kept only as an optional secondary/stronger bar (see
    below) for anyone who still wants the old global-modulus behavior.
    Growth is separately capped at max_clique_size primes: once a chain
    reaches that many primes without yet confirming, it's abandoned rather
    than extended further, since per this problem's structure a clique
    that large is already past where real solutions are found, and
    letting it keep competing for beam/budget slots only starves smaller,
    genuinely-plausible chains (see the "deepest-then-widest" comment
    below, which now interacts with this cap).
    """
    pool = sorted(int(p) for p in prime_pool)
    H = int(height_bound)
    box = (2 * H + 1) ** 2
    threshold = margin * box

    if use_arc_consistency:
        # Auto-tune witness_min_success_rate from the pool's OWN observed
        # root density, rather than trusting a fixed 0.8 default. At low
        # avg_roots (each prime has ~1 residue, no fallback if that one
        # happens to be recorded wrong / from the "wrong" Galois branch)
        # a strict 80%-of-6 majority vote kills genuine chains just as
        # readily as decoys -- see _has_witness_chain's docstring,
        # "FOURTH BUG" section, and the trace_target_through_arc_consistency
        # diagnostic that first surfaced this on a real run (every true
        # residue of a known target m was dropped in round 1-2 while a
        # decoy component survived). Higher avg_roots gives real fallback
        # redundancy, so the strict default is appropriate there.
        #
        # *** SEVENTH BUG (found via trace_target on a real run with
        # avg_roots~82) ***
        # avg_roots (total residues / primes-with-data) measures how many
        # candidate residues EACH PRIME has data for, across ALL candidate
        # points -- it says nothing about what fraction of primes contain
        # any ONE specific residue/target's true value. Those are nearly
        # independent quantities: this run had avg_roots=82.35 (dense
        # overall) but the known target m=-1/7's true residue was only
        # actually PRESENT (per trace_target) at 14 of 23 primes -- ~61%
        # coverage. min_success_rate=0.8 (selected because avg_roots was
        # "high") is mathematically unreachable for a residue with <80%
        # true coverage: every prime where the residue is absent walks as
        # an "attempt" in _has_witness_chain's denominator (nonempty
        # q_nodes, just none of them the true value) and can essentially
        # only register as a failure, so the rate is guaranteed to fall
        # below 0.8 eventually regardless of correctness. Using avg_roots
        # to justify a STRICTER bar was backwards: high avg_roots (many
        # residues/prime across many candidates) does not imply any one
        # residue has high cross-prime coverage, and per-residue coverage
        # is what the rate check actually needs to tolerate.
        #
        # An attempted fix that measured coverage by comparing raw residue
        # INTEGERS across different primes (r=12 at p=17 vs r=12 at p=41)
        # was tried and reverted -- residues live in different Z/pZ per
        # prime, so numeric equality across primes is meaningless
        # coincidence, not a coverage signal. There is no cheap proxy for
        # true per-residue coverage without already knowing which residue
        # is the real point (circular). So instead of inferring the rate
        # from ANY density proxy, use a materially lower fixed default
        # that stays viable even at the ~60% coverage actually observed
        # on real targets, and lean on the min_attempts floor (FIFTH BUG
        # fix) plus the per-step shuffle/cap scaling (SIXTH BUG fix) to
        # do the real work of keeping true chains alive long enough to
        # reach an informative modulus.
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

    if progress:
        print(f"[residue_graph_incremental] start: {len(chains)} gen-1 chains "
              f"over {len(primes_with_data)} primes, threshold={threshold:.3g}")

    try:
        while chains:
            gen_start = time.monotonic()

            if max_chains is not None and len(chains) > max_chains:
                # FIX (found via synthetic ground-truth test): sorting by
                # raw modulus M ("widest-modulus") systematically favors
                # chains that happened to touch a couple of LARGE primes
                # early over chains that have survived MORE independent
                # lift checks at smaller primes -- e.g. a 2-prime chain
                # using p=89,q=97 (M~8600) can outrank a genuine 5-prime
                # chain using p=5..19 (M~1.6M) by raw M alone at shallow
                # depth, wait, the reverse: a chain touching few large
                # primes can have LARGER M than a chain touching many small
                # primes despite the latter having survived more
                # independent tests and being the more probable real
                # signal. Confirmed empirically: on a synthetic pool with
                # one planted true cross-prime point, the true chain was
                # present and growing correctly through gen 5 (1->2->3->4
                # ->5 primes deep) but got squeezed out of the top-20000
                # slots by unrelated noise chains whose accidental prime
                # choices gave them a numerically larger M at the SAME
                # depth, and once those noise chains individually crossed
                # `threshold` they got marked confirmed and stopped
                # competing -- leaving the beam saturated with false
                # positives while the true chain was discarded before it
                # could reach threshold itself.
                #
                # Sort by DEPTH (primes used) first, tie-broken by M, so a
                # chain that has survived more independent CRT/lift checks
                # is preferred over one that merely landed on numerically
                # larger primes. This is a much better proxy for "actually
                # converging on a real point" than raw modulus size.
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

            # EIGHTH BUG fix: bound total CRT/lift calls for this
            # generation directly, independent of max_chains and root
            # density -- max_chains alone doesn't bound this (see
            # docstring). If projected cost exceeds the budget, extend
            # only as many of the still_growing chains (in the same
            # depth-first-then-widest priority already used for the
            # max_chains cap) as fit in the budget; the rest are carried
            # over UNEXTENDED to the next generation rather than dropped,
            # so a chain merely deferred this round still gets a chance
            # once the beam thins out.
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
                    # NINTH BUG fix: confirm by CLIQUE SIZE, not modulus
                    # magnitude -- a chain that has survived
                    # lattice_rational_lift_exists checks across
                    # min_clique_size mutually-consistent primes is a
                    # genuine local-solution candidate in this problem's
                    # regime, whether or not M has crossed margin*box.
                    # The old M > threshold bar is kept as an OR: a chain
                    # that happens to clear it is still confirmed too
                    # (strictly more informative, never wrong to accept),
                    # it's just no longer the ONLY way to confirm.
                    if len(used_primes) >= min_clique_size or M > threshold:
                        first = node_keys[0]
                        for other in node_keys[1:]:
                            union(first, other)
                        chains_confirmed += 1
                        continue

                    if len(used_primes) >= max_clique_size:
                        # NINTH BUG fix: this chain has grown past where
                        # real solutions are found (max_clique_size, per
                        # search_common's MIN_MAX_PRIME_SUBSET_SIZE) without
                        # confirming -- stop extending it. Letting it keep
                        # fanning out only spends budget/beam slots that
                        # smaller, still-plausible chains need, and per
                        # this problem's structure a clique this large
                        # confirming at all is empirically not expected.
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

                    # FIX (see arc_consistency_prune_domains / module
                    # docstring): the old version restricted extension to
                    # ONLY the next prime in sorted order (next_idx =
                    # last_idx + 1), which was a correctness bug wearing a
                    # performance fix's clothes -- it made cross-prime
                    # combinations that required "going backwards" in
                    # sorted order structurally unreachable, which is why
                    # every run bottomed out in degenerate single-prime
                    # components. Now that arc-consistency pruning has
                    # already shrunk each prime's domain down to only
                    # residues with genuine cross-prime support, full
                    # fan-out into every remaining (unused) prime is
                    # affordable again -- domains are small, not O(original
                    # roots) -- so we extend into ANY prime not yet used by
                    # this chain, in pool order, rather than only the next
                    # one after the chain's most recent prime.
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

    # RECONCILIATION PASS (found via synthetic ground-truth test): the beam
    # only has room to grow ONE lineage to confirmation at a time under
    # heavy competition. In a run with a real point but many decoy/noise
    # residues, one lucky confirmed chain (say, covering primes 53..101)
    # wins the beam early, gets unioned and stops contributing further
    # comparisons -- while the TRUE residues at the *other* primes
    # (5..47), still correct, still mutually consistent with each other
    # AND with the confirmed chain, get squeezed out of beam slots before
    # they get a chance to grow their OWN confirmed sub-chain. The result
    # is the true point's information ends up scattered across many small
    # disconnected/singleton components instead of one big one -- verified
    # directly: in the synthetic test this produced 13 separate components
    # jointly covering all 23 true residues, instead of 1.
    #
    # Fix: after the main beam loop, explicitly test cross-compatibility
    # BETWEEN components that share no primes (so nothing checked this
    # pair yet) using a representative node from each side and the same
    # lattice_rational_lift_exists test the main loop uses, and union them
    # if compatible. This is only O(components^2) representative checks,
    # not O(chains^2), since by this point most components have already
    # collapsed via the beam into a small number of confirmed chains plus
    # leftover singletons/small fragments.
    #
    # *** NINTH BUG interaction (found via real run after min_clique_size
    # confirmation was added) *** This pass's whole premise -- that
    # fragments belong to ONE true global solution and should be stitched
    # back together -- is exactly the "global solution" assumption module
    # docstring's NINTH BUG note says is false for this problem. With
    # confirmation now firing at min_clique_size (3), a real run produces
    # potentially THOUSANDS of small confirmed cliques instead of the 1-2
    # giant components this pass's O(components^2) cost estimate assumed
    # ("most components have already collapsed... into a small number" --
    # no longer true). Confirmed empirically: 1884 nodes/23 primes produced
    # components^2 ~ millions of reconciliation checks, each a real
    # multiprecision crt_cached+lift call, and this pass then merged
    # ~1.4M pairs back into one 21-prime blob -- silently undoing the
    # min_clique_size fix's entire point (recovering small local solutions
    # instead of insisting on global ones) AND costing real time doing it.
    # Fix: default this pass OFF (reconcile_components=False). It's still
    # available for the old global-solution regime (pass True explicitly,
    # e.g. if you have independent reason to believe solutions DO span
    # most of the pool for a different problem instance), with a hard cap
    # (max_reconcile_pairs) on top since even opted-in it should not be
    # allowed to blow up the same way max_calls_per_generation guards the
    # main loop.
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


def reconstruct_candidates_from_component(comp, height_bound, max_den=None,
                                           max_results=5, stats_counter=None,
                                           max_visits=2_000_000):
    """
    Bottom-up CRT reconstruction over a component, with the search itself
    bounded -- not just pruned after the fact.

    History: the original version built the full
    itertools.product(*residue_choices) up front (unreachable: avg_roots~82
    per prime, up to 23 primes). A first fix added incremental pruning via
    lattice_rational_lift_exists, modeled on refine_component_chained --
    but that still hung, because lift-existence is nearly a tautology at
    small partial moduli (see _STRONG_INFORMATIVE_MARGIN's comment above:
    ~99.98% of residues pass at M > H, ~45% still pass at M > 2*H*H).
    With ~82-wide fan-out per prime and pruning that doesn't meaningfully
    bite for the first several primes, the tree is still exponential --
    the second version only made each dead branch a little cheaper to
    reach, not the tree itself smaller.

    So this version bounds total recursive calls directly
    (max_visits), the same style fix as EIGHTH BUG's
    max_calls_per_generation in build_residue_graph_incremental --
    trusting a budget rather than trusting the lift check to prune early.
    Two things make the budget spend well instead of just failing after N
    calls:
      - primes are visited in ASCENDING order of residue count, so the
        narrowest (cheapest, most-constraining) primes are folded in
        first -- by the time you reach the wide-fanout primes, most
        branches are already dead from the earlier narrow ones, instead
        of paying full 82-wide fanout before any real constraint applies.
      - lattice_rational_lift_exists still prunes whatever it can (free
        wins once M is large enough to be genuinely informative), it's
        just no longer the ONLY thing standing between this and a hang.

    Returns whatever confirmed reconstructions were found before hitting
    max_results or exhausting max_visits -- it may come back empty or
    partial on a component that's mostly noise; that's expected, this is
    a bounded best-effort search, not a certificate of "no point exists".
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
