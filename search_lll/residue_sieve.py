"""
search_lll/residue_sieve.py

Consensus sieve for the cross-prime residue search (replacement core for
residue_crt_graph.build_residue_graph_ktuple).

Problem
-------
For a vector v (or, in mixed mode, for the pool of all vectors) every prime p
gives a domain R_p of residues.  A true rational point m = a/b with
1 <= b <= H, |a| <= H has m mod p in R_p at (almost) every prime p that does
not divide b.  We want every such (a, b) without being told the target.

Why the old beam search only found height ~0 points
---------------------------------------------------
* A chain was "confirmed" after MIN_PRIME_SUBSET_SIZE (=3) primes, when its
  modulus (~1e5..1e6 for primes < 100) is still far below the box (2H+1)^2.
  Below the box, lattice_rational_lift_exists passes almost every class, so
  the lift test prunes nothing, and rational_reconstruct(c, M, max_den=H) then
  only recovers a/b with roughly 2ab < M, i.e. tiny height.
* Chains were keyed by frozenset but extended "in any order" with no
  canonical ordering, so each k-subset was generated k! times; the
  20000-chain cap then discarded most of them by (depth, modulus), a criterion
  uncorrelated with being a true point.

What this module does instead
-----------------------------
1. Pick a small "base" subset of primes whose product M clears a fraction of
   the box.  For each choice of one residue per base prime, CRT gives a class c
   mod M, and every (a, b) in the box with a = c*b (mod M) is enumerated
   EXACTLY (Gauss-reduced 2-D lattice, not just continued-fraction convergents).
   A true point lies in the enumeration of any base subset on which it has all
   its residues, so completeness needs only one such subset.
2. Every enumerated (a, b) is scored against ALL primes with a log-likelihood
   ratio.  A true point has a root at a given prime only with probability rho
   (for a point whose n is global this is ~1 minus Hensel drops; the 0.54
   'coverage' printed by cov1 is pooled over ALL vectors and is not a
   per-vector rate), so a hard "may miss k primes"
   rule discards real points.  Hypothesis H1: hit with prob max(rho, f_q);
   H0: hit with prob f_q = |R_q|/q.  A hit adds ln(rho/f_q), a miss adds
   ln((1-rho)/(1-f_q)); primes with f_q >= rho carry no weight.  Under H0,
   P(LLR >= t) <= e^-t, so t = ln(#tested) + margin bounds the expected
   number of false accepts.
3. Mixed-n mode pools R_q over all vectors.  The pooled domains are much less
   selective, so instead of thresholding, candidates are ranked by evidence
   and the top ones are handed to the (cheap) on-curve test.  When the box is
   small enough a numpy box sieve replaces base-subset enumeration entirely,
   which makes mixed mode complete instead of budgeted.

Output is a dict shaped like build_residue_graph_incremental's, so
discover_candidates_via_residue_graph and search_main need no changes to
consume it ('confirmed_chains' entries carry 'm_num'/'m_den' directly).
"""

import itertools
import math
import random
import time
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:  # numpy is a stated dependency, but degrade gracefully
    np = None

DEFAULT_LATTICE_MARGIN = 0.25      # base modulus >= margin * (2H+1)^2
DEFAULT_HIT_RATE = 0.9             # assumed P(a true point has a root at a prime), per-vector; see _Scorer
DEFAULT_MAX_BASE_SUBSETS = 64      # diversified base subsets per call
DEFAULT_MAX_COMBOS = 200_000       # CRT combos across all base subsets
DEFAULT_MIX_MAX_COMBOS = 400_000   # same, mixed-n lattice fallback
DEFAULT_BOX_LIMIT = 80_000_000     # max (2H+1)*H points for the numpy sieve
DEFAULT_MIX_MAX_CANDIDATES = 3000  # mixed mode: keep this many best-scored
DEFAULT_MIN_HITS = 3
_MAX_LATTICE_ITER = 20_000


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num_den(m):
    """(numerator, denominator) of a Sage QQ / Fraction / int, denominator > 0."""
    num, den = m.numerator, m.denominator
    if callable(num):
        num, den = num(), den()
    num, den = int(num), int(den)
    if den < 0:
        num, den = -num, -den
    return num, den


def sieve_base_size(prime_pool, height_bound, margin=DEFAULT_LATTICE_MARGIN):
    """
    Fewest primes (largest first) whose product reaches margin*(2H+1)^2, or
    None if the whole pool cannot.  This is the depth at which the sieve can
    start to discriminate; it replaces min_tuple_size_for_margin's
    smallest-primes-first count, which overestimates by using the weakest
    primes.
    """
    H = int(height_bound)
    target = margin * (2 * H + 1) ** 2
    prod = 1
    for i, p in enumerate(sorted((int(p) for p in prime_pool), reverse=True), 1):
        prod *= p
        if prod >= target:
            return i
    return None


def _build_domains(precomputed_residues, pool, v_tuple):
    """
    {p: {residue: [(vector, rhs_idx), ...]}} for the primes that have any
    residue data at all.  v_tuple=None pools every vector (mixed mode);
    otherwise only that vector's roots are used and a prime with no roots for
    it gets an empty domain.
    """
    dom = {}
    for p in pool:
        p_map = precomputed_residues.get(p, {})
        if not p_map:
            continue                      # prime was rejected / never computed
        if v_tuple is None:
            items = p_map.items()
        else:
            items = [(v_tuple, p_map[v_tuple])] if v_tuple in p_map else []
        d = {}
        for vt, roots_lists in items:
            for rhs_idx, roots in enumerate(roots_lists):
                for r in roots:
                    d.setdefault(int(r) % p, []).append((vt, rhs_idx))
        dom[p] = d
    return dom


# ---------------------------------------------------------------------------
# exact enumeration of small rationals in a CRT class
# ---------------------------------------------------------------------------

def _i_interval(coef, off, lo, hi):
    """Integers i with lo <= coef*i + off <= hi, as (imin, imax); None = unbounded side."""
    if coef == 0:
        return (None, None) if lo <= off <= hi else (1, 0)
    if coef > 0:
        return (-((-(lo - off)) // coef), (hi - off) // coef)
    return (-((-(hi - off)) // coef), (lo - off) // coef)


def lattice_box_points(c, M, H, max_iter=_MAX_LATTICE_ITER):
    """
    All primitive (a, b) with 1 <= b <= H, |a| <= H and a = c*b (mod M).

    The solutions are the points of the lattice spanned by (M, 0) and (c, 1)
    inside the box.  The basis is Gauss/Lagrange-reduced and the box is walked
    exactly (interval of i for every j), so unlike walking continued-fraction
    convergents this returns every solution when M < 2H^2 and several exist.
    Returns (points, truncated).
    """
    c %= M
    u0, u1, w0, w1 = M, 0, c, 1
    nu, nw = u0 * u0 + u1 * u1, w0 * w0 + w1 * w1
    if nw < nu:
        u0, u1, w0, w1 = w0, w1, u0, u1
        nu, nw = nw, nu
    while True:
        dot = u0 * w0 + u1 * w1
        q = (2 * dot + nu) // (2 * nu)          # round(dot / nu)
        if q:
            w0 -= q * u0
            w1 -= q * u1
            nw = w0 * w0 + w1 * w1
        if nw >= nu:
            break
        u0, u1, w0, w1 = w0, w1, u0, u1
        nu, nw = nw, nu

    # a = i*u0 + j*w0,  b = i*u1 + j*w1,  det = +-M  =>  |j| <= H(|u0|+|u1|)/M
    jmax = (H * (abs(u0) + abs(u1))) // M
    out = []
    remaining = max_iter
    truncated = False
    for j in range(-jmax, jmax + 1):
        lo_a, hi_a = _i_interval(u0, j * w0, -H, H)
        lo_b, hi_b = _i_interval(u1, j * w1, 1, H)
        lo = max(x for x in (lo_a, lo_b) if x is not None) if (lo_a is not None or lo_b is not None) else None
        hi = min(x for x in (hi_a, hi_b) if x is not None) if (hi_a is not None or hi_b is not None) else None
        if lo is None or hi is None or lo > hi:
            continue
        if hi - lo + 1 > remaining:
            hi = lo + remaining - 1
            truncated = True
        remaining -= hi - lo + 1
        for i in range(lo, hi + 1):
            a = i * u0 + j * w0
            b = i * u1 + j * w1
            if math.gcd(a, b) == 1:
                out.append((a, b))
        if remaining <= 0:
            truncated = True
            break
    return out, truncated


# ---------------------------------------------------------------------------
# candidate scoring
# ---------------------------------------------------------------------------

class _Scorer:
    """
    Log-likelihood-ratio score of a rational a/b against every domain.

    hit  : a * b^-1 mod q lies in dom[q]      -> + ln(rho / f_q)
    miss : it does not                        -> + ln((1-rho) / (1-f_q))
    with f_q = |dom[q]| / q.  Primes with an empty domain add the same
    constant to every candidate and are ignored; primes with f_q >= rho are
    uninformative and ignored.  A prime dividing b is neutral (a/b has a pole
    there, no residue exists).  Primes are visited by decreasing hit weight so
    a caller-supplied threshold can abort hopeless candidates early: the
    remaining hit weights bound what is still attainable.
    """

    def __init__(self, dom, rho=DEFAULT_HIT_RATE):
        self.dom = dom
        self.rho = rho
        self.f = {q: len(d) / q for q, d in dom.items() if d}
        info_primes = [q for q, f in self.f.items() if f < rho]
        self.w_hit = {q: math.log(rho / self.f[q]) for q in info_primes}
        self.w_miss = {q: math.log((1.0 - rho) / (1.0 - self.f[q])) for q in info_primes}
        self.order = sorted(info_primes, key=lambda q: -self.w_hit[q])
        # suffix[i] = total hit weight of order[i:]
        self.suffix = [0.0] * (len(self.order) + 1)
        for i in range(len(self.order) - 1, -1, -1):
            self.suffix[i] = self.suffix[i + 1] + self.w_hit[self.order[i]]
        self.max_llr = self.suffix[0]

    def score(self, a, b, threshold=None, exclude=None):
        """
        (hit_primes, misses, llr), or None if threshold is set and unreachable.

        exclude: primes whose residues were used to ENUMERATE this candidate
        (the lattice base subset).  They are hits by construction, so they are
        listed in hit_primes but contribute nothing to llr; otherwise every
        candidate starts with the base primes' weight (~4 x 3-4 nats) and the
        significance threshold means nothing.
        """
        hits = []
        misses = 0
        llr = 0.0
        dom, order, suffix = self.dom, self.order, self.suffix
        for i, q in enumerate(order):
            if exclude is not None and q in exclude:
                hits.append(q)
                continue
            bq = b % q
            if bq == 0:
                continue
            if (a * pow(bq, -1, q)) % q in dom[q]:
                hits.append(q)
                llr += self.w_hit[q]
            else:
                misses += 1
                llr += self.w_miss[q]
                if threshold is not None and llr + suffix[i + 1] < threshold:
                    return None
        return hits, misses, llr


def _make_chain(a, b, hits, misses, llr, dom):
    """confirmed_chains entry (same keys as the beam search) for a scored (a, b)."""
    primes = sorted(hits)
    M = 1
    for q in primes:
        M *= q
    residue = (a * pow(b, -1, M)) % M if M > 1 else 0
    node_keys = []
    votes = Counter()
    for q in primes:
        r = (a * pow(b % q, -1, q)) % q
        tags = dom[q][r]
        node_keys.append((q, tags[0], r))
        for vt, _rhs in tags:
            votes[vt] += 1
    v_mode = votes.most_common(1)[0][0] if votes else None
    return {
        'primes': primes, 'modulus': M, 'residue': residue,
        'node_keys': node_keys,
        'm_num': a, 'm_den': b,
        'misses': misses, 'bits': llr, 'llr': llr,
        'v_mode': v_mode,
        'v_votes': dict(votes),
    }


# ---------------------------------------------------------------------------
# engine 1: base-subset CRT + exact lattice enumeration
# ---------------------------------------------------------------------------

def _choose_base_subsets(dom, H, margin, max_subsets, max_combos, rng,
                         max_enum=200_000):
    """
    Diversified base subsets: minimal size k with product >= margin*box, cheap
    (small product of domain sizes) first, low mutual overlap so one prime
    with a dropped residue cannot sink every subset.

    Returns (subsets, k, deferred) where subsets is a list of
    (primes, M, crt_coeffs, cost) and deferred counts subsets skipped for
    budget.  k is None if the pool cannot reach the target modulus.
    """
    box = (2 * H + 1) ** 2
    target = margin * box
    ps = sorted((p for p, d in dom.items() if d), reverse=True)
    prod, k = 1, None
    for i, p in enumerate(ps, 1):
        prod *= p
        if prod >= target:
            k = i
            break
    if k is None:
        return [], None, 0

    if math.comb(len(ps), k) <= max_enum:
        cand = itertools.combinations(ps, k)
    else:
        seen = set()
        for _ in range(max_enum):
            seen.add(tuple(sorted(rng.sample(ps, k), reverse=True)))
        cand = seen

    scored = []
    for sub in cand:
        M = 1
        cost = 1
        for p in sub:
            M *= p
            cost *= len(dom[p])
        if M >= target:
            scored.append((cost, -M, sub))
    scored.sort()

    idx_of = {p: i for i, p in enumerate(ps)}

    def mask(sub):
        m = 0
        for p in sub:
            m |= 1 << idx_of[p]
        return m

    chosen, masks, spent, deferred = [], [], 0, 0
    for overlap in (max(1, k // 2), k):          # strict pass, then relax
        for cost, _negM, sub in scored:
            if len(chosen) >= max_subsets:
                break
            mk = mask(sub)
            if mk in masks:
                continue
            if any(bin(mk & m).count('1') > overlap for m in masks):
                continue
            if spent + cost > max_combos and chosen:
                deferred += 1
                continue
            chosen.append(sub)
            masks.append(mk)
            spent += cost
        if len(chosen) >= max_subsets or spent >= max_combos:
            break

    out = []
    for sub in chosen:
        M = 1
        for p in sub:
            M *= p
        coeffs = []
        for p in sub:
            Mi = M // p
            coeffs.append((Mi * pow(Mi % p, -1, p)) % M)
        cost = 1
        for p in sub:
            cost *= len(dom[p])
        out.append((sub, M, coeffs, cost))
    return out, k, deferred


def _sieve_lattice(dom, H, scorer, margin, max_subsets, max_combos, rng,
                   time_limit, threshold, keep):
    """
    Run engine 1.  threshold (single mode) enables early abort inside the
    scorer; keep (mixed mode) trims the result dict to the best `keep` scores.
    Returns ({(a,b): (hits, misses, llr)}, info).
    """
    subsets, k, deferred = _choose_base_subsets(dom, H, margin, max_subsets,
                                                max_combos, rng)
    info = {'engine': 'lattice', 'base_size': k, 'subsets': len(subsets),
            'deferred_subsets': deferred, 'combos': 0, 'lattice_points': 0,
            'lattice_truncated': 0, 'tested': 0}
    found = {}
    if k is None:
        info['pool_too_weak'] = True
        return found, info

    seen = set()
    t0 = time.time()
    stop = False
    for sub, M, coeffs, _cost in subsets:
        lists = [list(dom[p].keys()) for p in sub]
        base_set = frozenset(sub)
        for combo in itertools.product(*lists):
            c = 0
            for r, e in zip(combo, coeffs):
                c += r * e
            info['combos'] += 1
            pts, trunc = lattice_box_points(c, M, H)
            info['lattice_truncated'] += int(trunc)
            for ab in pts:
                info['lattice_points'] += 1
                if ab in seen:
                    continue
                seen.add(ab)
                res = scorer.score(ab[0], ab[1], threshold, exclude=base_set)
                if res is not None:
                    found[ab] = res
            if keep is not None and len(found) > 4 * keep:
                top = sorted(found.items(), key=lambda kv: -kv[1][2])[:keep]
                found = dict(top)
            if time_limit is not None and info['combos'] % 512 == 0 \
                    and time.time() - t0 > time_limit:
                stop = True
                break
        if stop:
            info['time_limit_hit'] = True
            break
    info['tested'] = len(seen)
    return found, info


# ---------------------------------------------------------------------------
# engine 2: numpy box sieve (mixed-n, small H)
# ---------------------------------------------------------------------------

def _sieve_box(dom, H, scorer, keep, time_limit):
    """
    Complete LLR scoring of every (a, b), 1 <= b <= H, |a| <= H, gcd = 1,
    vectorised over a for each b, keeping the best `keep` overall.  Cost is
    about (#informative primes) * (2H+1) * H gathers.  There is no survivor
    cap: the running top-`keep` list is trimmed as it grows, so the answer is
    exact (up to ties) unless time_limit is hit.
    Returns ({(a,b): (hits, misses, llr)}, info).
    """
    order = scorer.order
    A = np.arange(-H, H + 1, dtype=np.int64)
    n = A.size
    W, amod = {}, {}
    for q in order:
        t = np.full(q, scorer.w_miss[q], dtype=np.float32)
        t[list(dom[q].keys())] = scorer.w_hit[q]
        W[q] = t
        amod[q] = A % q
    info = {'engine': 'box', 'base_size': None, 'subsets': 0, 'combos': 0,
            'box_points': n * H}
    best_s = np.empty(0, dtype=np.float32)
    best_a = np.empty(0, dtype=np.int64)
    best_b = np.empty(0, dtype=np.int64)
    thr = -np.inf
    t0 = time.time()
    for b in range(1, H + 1):
        sc = np.zeros(n, dtype=np.float32)
        for q in order:
            if b % q == 0:
                continue
            inv = pow(b % q, -1, q)
            sc += W[q][(amod[q] * inv) % q]
        sc[np.gcd(A, b) != 1] = -np.inf
        cand = np.nonzero(sc > thr)[0]
        if cand.size > keep:
            cand = cand[np.argpartition(sc[cand], -keep)[-keep:]]
        if cand.size:
            best_s = np.concatenate([best_s, sc[cand]])
            best_a = np.concatenate([best_a, A[cand]])
            best_b = np.concatenate([best_b, np.full(cand.size, b, dtype=np.int64)])
            if best_s.size > 4 * keep:
                top = np.argpartition(best_s, -keep)[-keep:]
                best_s, best_a, best_b = best_s[top], best_a[top], best_b[top]
                thr = float(best_s.min())
        if time_limit is not None and b % 64 == 0 and time.time() - t0 > time_limit:
            info['time_limit_hit'] = True
            info['b_reached'] = b
            break
    if best_s.size > keep:
        top = np.argpartition(best_s, -keep)[-keep:]
        best_s, best_a, best_b = best_s[top], best_a[top], best_b[top]
    found = {}
    for a_, b_ in zip(best_a.tolist(), best_b.tolist()):
        found[(a_, b_)] = scorer.score(a_, b_)
    info['tested'] = n * H
    info['lattice_points'] = len(found)
    return found, info


# ---------------------------------------------------------------------------
# tracing of known targets (reporting only; never steers the search)
# ---------------------------------------------------------------------------

def _trace_known(known_m, dom, H, scorer, found_keys, chains_by_ab, accepted_keys,
                 engine, threshold):
    rows = []
    for m in known_m:
        try:
            a, b = _num_den(m)
        except Exception:
            continue
        row = {'m': m, 'confirmed_gen': None, 'lost_gen': None, 'lost_why': None}
        if (a, b) in accepted_keys:
            row['confirmed_gen'] = len(chains_by_ab[(a, b)]['primes'])
            rows.append(row)
            continue
        if b > H or abs(a) > H:
            row['lost_gen'] = 0
            row['lost_why'] = f"height exceeds H={H}"
            rows.append(row)
            continue
        res = scorer.score(a, b)
        if res is None:
            row['lost_gen'] = 0
            row['lost_why'] = "no informative primes"
            rows.append(row)
            continue
        hits, misses, llr = res
        row['lost_gen'] = len(hits)
        desc = (f"LLR={llr:.1f} over all informative primes (max {scorer.max_llr:.1f}; "
                f"acceptance threshold applies after removing the base primes), "
                f"{len(hits)} hits / {misses} misses over informative primes")
        if (a, b) in found_keys:
            row['lost_why'] = desc + "; scored but below the acceptance threshold / candidate cap"
        elif engine == 'box':
            row['lost_why'] = desc + "; outside the top-K by LLR"
        elif threshold is not None and llr < threshold:
            row['lost_why'] = desc + f"; below threshold {threshold:.1f}"
        else:
            row['lost_why'] = desc + "; not enumerated by any base subset (budget/diversity)"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def sieve_residue_candidates(precomputed_residues, prime_pool, height_bound,
                             v_tuple=None, mixed=None, known_m=None,
                             hit_rate=DEFAULT_HIT_RATE,
                             lattice_margin=DEFAULT_LATTICE_MARGIN,
                             max_base_subsets=DEFAULT_MAX_BASE_SUBSETS,
                             max_combos=None, max_candidates=None,
                             min_llr=None, min_hits=DEFAULT_MIN_HITS,
                             box_limit=DEFAULT_BOX_LIMIT, time_limit=None,
                             force_mixed=False, capacity_margin=3.0,
                             seed=0, stats_counter=None, progress=False,
                             **_ignored):
    """
    Find rationals a/b (|a|,b <= H) whose residues agree with the domains.

    v_tuple given  -> single-vector mode: exact lattice engine, accept
                      LLR >= ln(expected #tested) + 2 (expected false accepts
                      per vector <~ 0.14), abort hopeless candidates early.
    v_tuple None   -> mixed-n mode: every vector's roots are pooled per prime;
                      the best max_candidates by LLR are returned.  Uses the
                      complete numpy box sieve when (2H+1)*H <= box_limit,
                      else the lattice engine with a combo budget.

    hit_rate (rho): assumed probability that a true point has a root at a
    given prime.  Only primes with density |R_q|/q < rho carry weight.
    min_hits: minimum number of informative primes that must hit.
    Returns a dict shaped like build_residue_graph_incremental's result.
    """
    H = int(height_bound)
    pool = sorted(int(p) for p in prime_pool)
    if mixed is None:
        mixed = v_tuple is None
    if mixed:
        v_tuple = None
    rng = random.Random(seed)
    t_start = time.time()
    if mixed and time_limit is None:
        time_limit = 600.0     # mixed lattice fallback can be ~1e6 combos per subset

    dom = _build_domains(precomputed_residues, pool, v_tuple)
    dom = {q: d for q, d in dom.items() if d}      # empty domains carry no evidence
    n_nodes = sum(len(v) for d in dom.values() for v in d.values())
    n_primes = len(dom)

    empty_result = {
        'components': [], 'component_primes': [], 'edges_tested': 0,
        'edges_kept': 0, 'nodes': n_nodes, 'max_generation_reached': 0,
        'chains_confirmed': 0, 'confirmed_chains': [], 'generation_log': [],
        'trace': [], 'counters': {},
    }
    if n_primes < 2:
        return empty_result
    scorer = _Scorer(dom, hit_rate)

    # Information check.  A candidate can only be told apart from the
    # (2H+1)*H others by evidence, and total evidence available from hits is at
    # most sum_q ln(q/|R_q|).  Pooling every vector makes |R_q|/q close to 1 at
    # small primes, and a point whose n is not global is no more likely to hit
    # a pooled domain than a random m, so flat pooling can be information-free.
    capacity = sum(math.log(1.0 / (len(d) / q)) for q, d in dom.items())
    box_nats = math.log(2.0 * H * H)
    if mixed and not force_mixed and capacity < box_nats + capacity_margin:
        res = dict(empty_result)
        res['counters'] = {'mixed_skipped_insufficient_capacity': 1}
        res['sieve'] = {
            'engine': 'skipped', 'primes': n_primes,
            'informative_primes': len(scorer.order),
            'capacity_nats': capacity, 'box_nats': box_nats, 'accepted': 0,
            'reason': (f"pooled domains carry {capacity:.1f} nats of evidence in "
                       f"total but the box has {box_nats:.1f}; ranking would be "
                       f"noise (force with RESIDUE_GRAPH_MIX_FORCE=True, or add "
                       f"larger primes)"),
            'elapsed_sec': time.time() - t_start,
        }
        return res
    if len(scorer.order) < 2:
        empty_result['counters']['vector_skipped_uninformative'] = 1
        return empty_result

    cap = max_candidates if max_candidates is not None else (
        DEFAULT_MIX_MAX_CANDIDATES if mixed else None)
    threshold = None
    use_box = (mixed and np is not None and (2 * H + 1) * H <= box_limit)
    if use_box:
        found, info = _sieve_box(dom, H, scorer, cap or DEFAULT_MIX_MAX_CANDIDATES,
                                 time_limit)
    else:
        if max_combos is None:
            max_combos = DEFAULT_MIX_MAX_COMBOS if mixed else DEFAULT_MAX_COMBOS
        if not mixed:
            # planned points ~ combos * max(1, 2H^2 / M); used only to set the
            # significance threshold, so a rough count is fine
            _sub, _k, _ = _choose_base_subsets(dom, H, lattice_margin,
                                               max_base_subsets, max_combos,
                                               random.Random(seed))
            planned = sum(c for (_s, _M, _co, c) in _sub) or 1
            per_class = max(1.0, 2.0 * H * H / (_sub[0][1] if _sub else 1))
            threshold = (min_llr if min_llr is not None
                         else math.log(planned * per_class) + 2.0)
        found, info = _sieve_lattice(dom, H, scorer, lattice_margin,
                                     max_base_subsets, max_combos, rng,
                                     time_limit, threshold,
                                     keep=(cap if mixed else None))

    # ---- acceptance ------------------------------------------------------
    ranked = [(llr, -misses, ab, hits, misses)
              for ab, (hits, misses, llr) in found.items()
              if len(hits) >= min_hits]
    ranked.sort(reverse=True)
    if not mixed and threshold is not None:
        ranked_ok = [r for r in ranked if r[0] >= threshold]
    else:
        ranked_ok = ranked
    cap_dropped = 0
    if cap is not None and len(ranked_ok) > cap:
        cap_dropped = len(ranked_ok) - cap
        ranked_ok = ranked_ok[:cap]

    chains = []
    chains_by_ab = {}
    for llr, _nm, ab, hits, misses in ranked_ok:
        ch = _make_chain(ab[0], ab[1], hits, misses, llr, dom)
        chains.append(ch)
        chains_by_ab[ab] = ch

    if stats_counter is not None:
        stats_counter['residue_sieve_candidates'] += len(chains)
        stats_counter['residue_sieve_combos'] += info.get('combos', 0)

    trace = []
    if known_m:
        trace = _trace_known(known_m, dom, H, scorer, set(found), chains_by_ab,
                             set(chains_by_ab), info['engine'], threshold)

    elapsed = time.time() - t_start
    info.update({'primes': n_primes, 'informative_primes': len(scorer.order),
                 'capacity_nats': capacity, 'box_nats': box_nats,
                 'hit_rate': hit_rate, 'max_llr': scorer.max_llr, 'mixed': mixed,
                 'threshold': threshold,
                 'accepted': len(chains), 'verified': len(found),
                 'best_llr': ranked[0][0] if ranked else None,
                 'elapsed_sec': elapsed})
    if progress:
        print(f"[residue_sieve] v={v_tuple} mode={'mixed' if mixed else 'single'} "
              f"engine={info['engine']} {info}")

    counters = {}
    if cap_dropped:
        counters['cap_dropped'] = cap_dropped
    if info.get('deferred_subsets'):
        counters['deferred'] = info['deferred_subsets']

    return {
        'components': [], 'component_primes': [],
        'edges_tested': info.get('combos', 0) or info.get('tested', 0),
        'edges_kept': len(chains),
        'nodes': n_nodes,
        'max_generation_reached': info.get('base_size') or 0,
        'chains_confirmed': len(chains),
        'confirmed_chains': chains,
        'generation_log': [info],
        'trace': trace,
        'counters': counters,
        'sieve': info,
    }
