"""
Is there residue signal for a given m, or only chance?  (pure Python, no Sage)

Why this exists
---------------
A residue "clique" is evidence only relative to how often a RANDOM m would form
one.  For a vector v and prime p let D_{p,v} be the residue set.  A random m
lands in D_{p,v} with probability q_p = |D_{p,v}| / p, so the number of primes
at which a random m is "present" is Poisson-binomial with those q_p.  A real
point m* with [n*]P = R is present at EVERY good prime for its own vector n*
(m=0 at v=(1,) is 19/19 in the log) and only by chance elsewhere.

Consequences (used by report_target_signal below):
  * The per-vector observed count k is compared to the exact Poisson-binomial
    null, Bonferroni-corrected over the number of vectors scanned
    ("look-elsewhere").  k=3 of 19 at the best of 52 vectors is typically NOT
    significant when the null mean is ~0.3.
  * Pooling presence ACROSS vectors (pooled_signal_across_vectors below) is a
    complement, not a replacement.  A clean genuine point is present at all
    primes for ONE vector and at chance level for the rest, so pooling dilutes
    it (the pooled null mean grows with the number of vectors) and the per-vector
    rows catch it better.  Pooling wins when evidence is spread thin: partial
    matches at many vectors, or residue equations that are only partly right
    for n >= 2, where no single vector clears its own bar but the total does.
  * Capacity: even a perfect clique only certifies a height H if
        sum_{p in clique} ln(p / |D_p|)  >  ln(margin * (2H+1)^2).
    (p, not |D_p|-blind: choosing one of |D_p| residues per prime costs
    ln|D_p| of the modulus.)

Null model (rail-filtered residues)
-----------------------------------
The residue sets fed in here have already been through the rail_ok filter, which
keeps every genuine point and about half of everything else.  So the right null
is "a random m THAT PASSES THE RAIL TEST":  q_p = |D_{p,v}| / N_p, where N_p is
the number of residues mod p passing the rail test (~ (p+1)/2 if not supplied),
NOT |D|/p.  Using |D|/p understates the null mean by roughly 2x and overstates
significance.  Pass rail_pass={p: N_p} for exact counts; rail_filtered=False if
the residues were not rail-filtered.

Seed / identity row
-------------------
m = 0 at v = (1,) is the seed and the identity relation.  It is present at every
prime by construction, so it is at most a weak check that the residue code works
at n = 1.  It is flagged (verdict "seed identity check") and is neither ranked
as evidence nor counted in the Bonferroni factor.

The exact test "y is rational" remains the only confirmation of a point.
Residue cliques only decide WHICH candidates get that exact test.
"""
import math


def _domain(residues, p, v):
    """Residues of prime p for vector v, unioned over rhs functions, as ints mod p."""
    out = set()
    for rl in (residues.get(p) or {}).get(v, []):
        out.update(int(a) % p for a in rl)
    return out


def poisson_binomial_upper_tail(qs, k):
    """P(X >= k) for X = sum of independent Bernoulli(q_i).  Exact DP."""
    dp = [1.0]
    for q in qs:
        q = min(1.0, max(0.0, float(q)))
        nxt = [0.0] * (len(dp) + 1)
        for j, pj in enumerate(dp):
            nxt[j] += pj * (1.0 - q)
            nxt[j + 1] += pj * q
        dp = nxt
    if k <= 0:
        return 1.0
    return float(sum(dp[k:])) if k < len(dp) else 0.0


def _null_size(p, rail_pass, rail_filtered):
    """Number of residues mod p a 'random m' can take under the null."""
    if not rail_filtered:
        return p
    if rail_pass and p in rail_pass and rail_pass[p]:
        return int(rail_pass[p])
    return p if p == 2 else (p + 1) // 2   # QR-or-zero classes, approx


def vector_null_profile(residues, primes, v, den=1, rail_pass=None, rail_filtered=True):
    """(ps, qs, sizes) over primes usable for this m (den % p != 0):
    q_p = |D| / N_p with N_p the rail-passing residue count (see module doc)."""
    qs, sizes, ps = [], [], []
    for p in primes:
        p = int(p)
        if den % p == 0:
            continue
        d = len(_domain(residues, p, v))
        qs.append(min(1.0, d / _null_size(p, rail_pass, rail_filtered)))
        sizes.append(d)
        ps.append(p)
    return ps, qs, sizes


def is_seed_row(m, v):
    """m = 0 at v = (1,): the identity relation, present by construction."""
    try:
        return int(m.numerator()) == 0 and tuple(int(x) for x in v) == (1,)
    except Exception:
        return False


def m_signal_by_vector(residues, primes, m, vectors, margin=15,
                       rail_pass=None, rail_filtered=True):
    """
    For a rational m, one row per vector, sorted best first:
      dict(v, k, n, null_mean, p, p_adj, info_nats, need_nats, capacity_ok)
    p_adj = min(1, p * V), V = number of NON-seed vectors (seed row is flagged, not ranked).
    """
    num, den = int(m.numerator()), int(m.denominator())
    h = max(abs(num), abs(den), 1)
    need = math.log(margin) + 2.0 * math.log(2 * h + 1)
    V = max(1, sum(1 for v in vectors if not is_seed_row(m, v)))
    rows = []
    for v in vectors:
        ps, qs, sizes = vector_null_profile(residues, primes, v, den, rail_pass, rail_filtered)
        k, info = 0, 0.0
        for p, d in zip(ps, sizes):
            if d and (num * pow(den, -1, p)) % p in _domain(residues, p, v):
                k += 1
                info += math.log(p / d)
        pval = poisson_binomial_upper_tail(qs, k)
        rows.append({
            'v': v, 'k': k, 'n': len(ps), 'null_mean': float(sum(qs)),
            'p': pval, 'p_adj': min(1.0, pval * V),
            'info_nats': info, 'need_nats': need, 'capacity_ok': info > need,
            'seed': is_seed_row(m, v),
        })
    rows.sort(key=lambda r: (r['seed'], r['p'], -r['k']))   # seed row last
    return rows


def verdict(row):
    if row.get('seed'):
        return "seed identity check (not evidence)"
    if row['p_adj'] < 1e-3 and row['capacity_ok']:
        return "SIGNAL (significant and enough modulus)"
    if row['p_adj'] < 1e-3:
        return "significant but capacity-limited"
    return "consistent with chance"


# ---------------------------------------------------------------------------
# Cross-vector (pooled) evidence
# ---------------------------------------------------------------------------
def _convolve_upper_tail(pmfs, s_obs):
    """P(sum of independent discrete variables >= s_obs); pmfs = list of {value: prob}."""
    dist = {0: 1.0}
    for pmf in pmfs:
        nxt = {}
        for a, pa in dist.items():
            for b, pb in pmf.items():
                nxt[a + b] = nxt.get(a + b, 0.0) + pa * pb
        dist = nxt
    if s_obs <= 0:
        return 1.0
    return float(sum(w for val, w in dist.items() if val >= s_obs))


def pooled_signal_across_vectors(residues, primes, m, vectors, rail_pass=None,
                                 rail_filtered=True):
    """
    Does m's residue presence, summed over ALL vectors, exceed chance?

    Two statistics, both with exact nulls that handle the dependence between
    vectors at the same prime (several D_{p,v} can contain the same residue,
    and v / -v may give identical sets):

      K_any : number of primes at which m mod p lies in the UNION of D_{p,v}
              over vectors.  Null: independent Bernoulli(|union| / N_p).
      S     : total presence count  sum_p c_p,  c_p = #{v : m mod p in D_{p,v}}
              (= sum_v k_v).  Null: at prime p, a random rail-passing residue a
              has c_p(a) = #{v : a in D_{p,v}}; its exact pmf is read off the
              residue sets (residues in no set have c = 0, N_p - |union| of
              them), and the pmfs are convolved over primes.

    K_any is robust to duplicate vectors; S is more sensitive to spread-out
    partial matches.  p_adj = min(1, 2 * min(p_any, p_S)) (two statistics).
    The seed row (m = 0, v = (1,)) is excluded.  Primes with den % p == 0 are
    skipped, as in the per-vector test.  Not corrected for looking at several m.
    """
    num, den = int(m.numerator()), int(m.denominator())
    vecs = [v for v in vectors if not is_seed_row(m, v)]
    per_vec = {v: 0 for v in vecs}
    qs_any, pmfs, k_any, S, n_used = [], [], 0, 0, 0
    for p in primes:
        p = int(p)
        if den % p == 0:
            continue
        a = (num * pow(den, -1, p)) % p
        doms = [(v, _domain(residues, p, v)) for v in vecs]
        counts = {}
        for _, d in doms:
            for r in d:
                counts[r] = counts.get(r, 0) + 1
        U = len(counts)
        N = max(_null_size(p, rail_pass, rail_filtered), U, 1)
        pmf = {}
        if N - U > 0:
            pmf[0] = (N - U) / N
        for c in counts.values():
            pmf[c] = pmf.get(c, 0.0) + 1.0 / N
        pmfs.append(pmf)
        qs_any.append(U / N)
        hit = [v for v, d in doms if a in d]
        for v in hit:
            per_vec[v] += 1
        S += len(hit)
        k_any += 1 if hit else 0
        n_used += 1
    p_any = poisson_binomial_upper_tail(qs_any, k_any)
    p_S = _convolve_upper_tail(pmfs, S)
    mean_S = sum(sum(val * w for val, w in pmf.items()) for pmf in pmfs)
    top = sorted(((k, v) for v, k in per_vec.items() if k), reverse=True)[:3]
    return {
        'n': n_used, 'n_vectors': len(vecs),
        'k_any': k_any, 'null_mean_any': float(sum(qs_any)), 'p_any': p_any,
        'S': S, 'null_mean_S': float(mean_S), 'p_S': p_S,
        'p_adj': min(1.0, 2.0 * min(p_any, p_S)),
        'top_vectors': [(v, k) for k, v in top],
    }


def pooled_verdict(row):
    if row['p_adj'] < 1e-3:
        return "POOLED SIGNAL (locate via per-vector rows; still needs exact y test)"
    return "pooled: consistent with chance"


def report_target_signal(residues, primes, trace_ms, vectors, top=3, margin=15,
                         rail_pass=None, rail_filtered=True):
    """Print, per traced m, its most significant vectors (look-elsewhere corrected)."""
    if not trace_ms:
        return
    for m in trace_ms:
        try:
            rows = m_signal_by_vector(residues, primes, m, vectors, margin, rail_pass, rail_filtered)
        except Exception as e:
            print(f"[signal] m={m}: could not evaluate ({e})")
            continue
        nv = sum(1 for r in rows if not r['seed'])
        print(f"[signal] m={m}: best of {nv} non-seed vectors "
              f"(p_adj = Bonferroni over vectors; need {rows[0]['need_nats'] / math.log(10):.1f} digits of modulus):")
        for r in [r for r in rows if r['seed']] + [r for r in rows if not r['seed']][:top]:
            print(f"[signal]   v={r['v']}: present at {r['k']}/{r['n']} primes "
                  f"(null mean {r['null_mean']:.2f}), p={r['p']:.2e}, p_adj={r['p_adj']:.2e}, "
                  f"info {r['info_nats'] / math.log(10):.1f} digits -> {verdict(r)}")
        try:
            pr = pooled_signal_across_vectors(residues, primes, m, vectors, rail_pass, rail_filtered)
            print(f"[signal]   POOLED over {pr['n_vectors']} vectors: "
                  f"any-vector at {pr['k_any']}/{pr['n']} primes (null {pr['null_mean_any']:.2f}, p={pr['p_any']:.2e}); "
                  f"total hits S={pr['S']} (null {pr['null_mean_S']:.2f}, p={pr['p_S']:.2e}); "
                  f"p_adj={pr['p_adj']:.2e} -> {pooled_verdict(pr)}"
                  + (f"; top vectors {pr['top_vectors']}" if pr['top_vectors'] else ""))
        except Exception as e:
            print(f"[signal]   pooled test failed ({e})")
