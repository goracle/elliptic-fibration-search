#!/usr/bin/env python3
"""
planted_analyze.py -- does a thin per-prime signal compound into a useful sieve?

Inputs (per curve idx in manifest.json, in --dir):
    run_<idx>.log     stdout/stderr of the search (parsed for recovery + multiplier)
    run_<idx>.jsonl   written by residue_signal.dump_signal_table
                      (run with  FIB_SIGNAL_DUMP=run_<idx>.jsonl )
Run each curve so that only the FIRST fibration (seed x0) matters; only block 0 is used.

Reports, bucketed by (genus, height):
  * recovery rate of the planted point (any of: '[height] new point', graph 'NEW POINT')
  * for recovered points with a dump row at the discovering vector: k/n, f-bar,
    rank of that vector among all vectors by raw p
  * rho-hat: MLE of the per-prime presence probability at the TRUE vector, corrected for
    selection (recovery needs >= s_min present primes, so ordinary k/n is biased up):
    truncated-binomial MLE.  LR per prime = rho-hat / f-bar.
  * prediction at each height: s primes needed, expected fully-present s-subsets
    C(N,s)*rho^s, and the candidate count C(N,s)*V vs the ~1.22*H^2 rationals of
    height <= H (the enumeration baseline).
"""
import argparse, json, math, os, re
from collections import defaultdict
from fractions import Fraction
from math import comb, log

HEIGHT_RE = re.compile(r"\[height\] new point x=(-?\d+(?:/\d+)?)\s.*?multiplier v=\(([-\d, ]+)\)")
GRAPH_RE = re.compile(r"NEW POINT x=(-?\d+(?:/\d+)?)\s.*?vector=\(([-\d, ]+)\)")

def parse_log(path):
    """{Fraction(x): [vector tuples]} for every reported new point."""
    found = defaultdict(list)
    if not os.path.exists(path): return found
    for line in open(path, errors='replace'):
        for rx in (HEIGHT_RE, GRAPH_RE):
            m = rx.search(line)
            if m:
                v = tuple(int(t) for t in m.group(2).replace(' ', '').split(',') if t)
                found[Fraction(m.group(1))].append(v)
    return found

def binom_pmf(k, n, r): return comb(n, k) * r**k * (1 - r)**(n - k)
def trunc_loglik(r, data):        # data: [(k, n, smin)]
    ll = 0.0
    for k, n, smin in data:
        tail = sum(binom_pmf(j, n, r) for j in range(smin, n + 1))
        ll += log(max(binom_pmf(k, n, r), 1e-300)) - log(max(tail, 1e-300))
    return ll
def trunc_mle(data):
    if not data: return None
    grid = [i / 1000 for i in range(5, 900)]
    return max(grid, key=lambda r: trunc_loglik(r, data))

def s_needed(ps, a, b):
    """fewest primes whose product (taking the largest pool primes) exceeds 2|a|b."""
    need, prod, s = 2 * abs(a) * b, 1, 0
    for p in sorted(ps, reverse=True):
        prod *= p; s += 1
        if prod > need: return s
    return len(ps)

def raw_p(row):
    qs = row['q']; dp = [1.0]
    for q in qs:
        nxt = [0.0] * (len(dp) + 1)
        for j, pj in enumerate(dp):
            nxt[j] += pj * (1 - q); nxt[j + 1] += pj * q
        dp = nxt
    return sum(dp[row['k']:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='manifest.json'); ap.add_argument('--dir', default='.')
    A = ap.parse_args()
    man = json.load(open(A.manifest))
    buckets = defaultdict(lambda: dict(n=0, rec=0, rows=[], ranks=[], Hs=[]))
    for e in man:
        key = (e['genus'], e['height']); B = buckets[key]; B['n'] += 1; B['Hs'].append((e['a'], e['b']))
        xp = Fraction(e['a'], e['b']); m_target = Fraction(e['seed_x']) - xp
        found = parse_log(os.path.join(A.dir, f"run_{e['idx']}.log"))
        if xp not in found: continue
        B['rec'] += 1
        dump = os.path.join(A.dir, f"run_{e['idx']}.jsonl")
        if not os.path.exists(dump): continue
        rows = [json.loads(l) for l in open(dump)]
        rows = [r for r in rows if r['block'] == 0 and Fraction(r['m']) == m_target]
        if not rows: continue
        true_vs = set(found[xp]); rows.sort(key=raw_p)
        for rank, r in enumerate(rows, 1):
            if tuple(r['v']) in true_vs:
                B['rows'].append(dict(k=r['k'], n=r['n'], fbar=sum(r['q']) / max(1, len(r['q'])),
                                      smin=s_needed(r['ps'], e['a'], e['b']), p=raw_p(r), V=len(rows)))
                B['ranks'].append((rank, len(rows))); break
    print(f"{'genus':>5} {'H':>7} {'curves':>6} {'recovered':>9} {'rate':>6} | {'k/n':>7} {'f-bar':>6} "
          f"{'rho-hat':>7} {'LR/prime':>8} | {'s':>2} {'E[full subsets]':>15} {'cands/enum':>12}")
    for (g, H), B in sorted(buckets.items()):
        rate = B['rec'] / B['n']; rows = B['rows']
        if rows:
            kn = sum(r['k'] for r in rows) / sum(r['n'] for r in rows)
            fb = sum(r['fbar'] for r in rows) / len(rows)
            ok = [r for r in rows if r['k'] >= r['smin']]       # consistent with 'recovered via >= s_min primes'
            excl = len(rows) - len(ok)                          # k < s_min: found another way (chance/tiny m)
            rho = trunc_mle([(r['k'], r['n'], r['smin']) for r in ok]) or float('nan')
            N = round(sum(r['n'] for r in rows) / len(rows)); V = rows[0]['V']
            a, b = B['Hs'][0]; s = s_needed(range(11, 100), a, b)       # crude: primes 11..97
            s = max(s, 1); cands = comb(N, s) * V; enum = 1.22 * H * H
            print(f"{g:>5} {H:>7} {B['n']:>6} {B['rec']:>9} {rate:>6.2f} | {kn:>7.3f} {fb:>6.3f} "
                  f"{rho:>7.3f} {rho / fb:>8.1f} | {s:>2} {comb(N, s) * rho**s:>15.3g} {cands / enum:>12.2e}")
            if len(ok) < 10: print(f"      (only {len(ok)} usable row(s): rho-hat is unreliable, want >= 30 per bucket)")
            if excl: print(f"      ({excl} recovered row(s) had k < s_min and were excluded from rho-hat)")
            rk = B['ranks']; print(f"      rank of true vector by raw p: "
                                   + ", ".join(f"{r}/{t}" for r, t in rk[:10]))
        else:
            print(f"{g:>5} {H:>7} {B['n']:>6} {B['rec']:>9} {rate:>6.2f} | (no recovered point with a dump row)")
    print("\nReading it: 'rho-hat' is selection-corrected presence probability at the true vector; "
          "LR/prime = rho-hat/f-bar.\n'cands/enum' < 1 means fewer subset-CRT candidates than enumerating all rationals of "
          "that height.\nRecovered points only: unrecovered ones may be out of the seed's span, so rho-hat is conditional "
          "on being in-span and detected.")

if __name__ == '__main__':
    main()
