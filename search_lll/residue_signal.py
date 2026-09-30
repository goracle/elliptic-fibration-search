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
  * Pooling presence ACROSS vectors does not add power for a genuine point: it
    is present at all primes for one vector and at chance level for the rest,
    while the pooled null mean grows linearly in the number of vectors.
  * Capacity: even a perfect clique only certifies a height H if
        sum_{p in clique} ln(p / |D_p|)  >  ln(margin * (2H+1)^2).
    (p, not |D_p|-blind: choosing one of |D_p| residues per prime costs
    ln|D_p| of the modulus.)

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


def vector_null_profile(residues, primes, v, den=1):
    """(qs, sizes) over primes usable for this m (den % p != 0): q_p=|D|/p, |D|."""
    qs, sizes, ps = [], [], []
    for p in primes:
        p = int(p)
        if den % p == 0:
            continue
        d = len(_domain(residues, p, v))
        qs.append(d / p)
        sizes.append(d)
        ps.append(p)
    return ps, qs, sizes


def m_signal_by_vector(residues, primes, m, vectors, margin=15):
    """
    For a rational m, one row per vector, sorted best first:
      dict(v, k, n, null_mean, p, p_adj, info_nats, need_nats, capacity_ok)
    p_adj = min(1, p * len(vectors)).
    """
    num, den = int(m.numerator()), int(m.denominator())
    h = max(abs(num), abs(den), 1)
    need = math.log(margin) + 2.0 * math.log(2 * h + 1)
    V = max(1, len(vectors))
    rows = []
    for v in vectors:
        ps, qs, sizes = vector_null_profile(residues, primes, v, den)
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
        })
    rows.sort(key=lambda r: (r['p'], -r['k']))
    return rows


def verdict(row):
    if row['p_adj'] < 1e-3 and row['capacity_ok']:
        return "SIGNAL (significant and enough modulus)"
    if row['p_adj'] < 1e-3:
        return "significant but capacity-limited"
    return "consistent with chance"


def report_target_signal(residues, primes, trace_ms, vectors, top=3, margin=15):
    """Print, per traced m, its most significant vectors (look-elsewhere corrected)."""
    if not trace_ms:
        return
    for m in trace_ms:
        try:
            rows = m_signal_by_vector(residues, primes, m, vectors, margin)
        except Exception as e:
            print(f"[signal] m={m}: could not evaluate ({e})")
            continue
        print(f"[signal] m={m}: best of {len(vectors)} vectors "
              f"(p_adj = Bonferroni over vectors; need {rows[0]['need_nats'] / math.log(10):.1f} digits of modulus):")
        for r in rows[:top]:
            print(f"[signal]   v={r['v']}: present at {r['k']}/{r['n']} primes "
                  f"(null mean {r['null_mean']:.2f}), p={r['p']:.2e}, p_adj={r['p_adj']:.2e}, "
                  f"info {r['info_nats'] / math.log(10):.1f} digits -> {verdict(r)}")
