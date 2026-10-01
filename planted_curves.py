#!/usr/bin/env python3
"""
planted_curves.py -- hyperelliptic curves y^2 = F(x) of genus g with a SEED point
and one PLANTED non-integer point of controlled height.  Pure Python (fractions).

Construction (exact, integer coefficients, no luck needed):
    F(x) = s(x)^2 + (x - x0) * (b*x - a) * G(x)
  s monic of degree g+1, G small integer poly, deg F = 2g+2, F monic.
  Then F(x0) = s(x0)^2      -> seed    (x0,  s(x0))
       F(a/b) = s(a/b)^2    -> planted (a/b, s(a/b)),  gcd(a,b)=1, b>=2.
Coefficients of F are ~H (not H^deg) because only the product (b*x - a) carries H.

Usage:
  python planted_curves.py --genus 2,3 --heights 100,1000,10000 --per-height 10 \
        --seed 1 --out manifest.json
  python planted_curves.py --show manifest.json 7     # print config for entry 7

CAVEAT: nothing forces the planted point to lie in the span of the seed in J(Q).
A point independent of the seed can never show a 'true vector'.  The analyzer
therefore reports recovery rate AND per-vector statistics only for recovered points.


NOTE:  KNOWN TO BE BUGGED. DO NOT USE.
"""
import argparse, json, math, random
from fractions import Fraction
from math import gcd

# polynomials: list of Fractions/ints, LOW -> HIGH degree
def trim(p):
    while p and p[-1] == 0: p = p[:-1]
    return p
def padd(p, q):
    n = max(len(p), len(q)); return trim([(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)])
def pmul(p, q):
    if not p or not q: return []
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q): r[i + j] += a * b
    return trim(r)
def peval(p, x):
    r = Fraction(0)
    for c in reversed(p): r = r * x + c
    return r
def pder(p): return trim([i * p[i] for i in range(1, len(p))])
def pmod(p, q):
    p = [Fraction(c) for c in p]; q = [Fraction(c) for c in q]
    while len(p) >= len(q) and p:
        c = p[-1] / q[-1]; d = len(p) - len(q)
        for i in range(len(q)): p[i + d] -= c * q[i]
        p = trim(p)
    return p
def gcd_degree(p, q):
    p, q = trim(list(p)), trim(list(q))
    while q: p, q = q, pmod(p, q)
    return len(p) - 1
def squarefree(F): return gcd_degree(F, pder(F)) == 0

def make_curve(g, H, rng, x0=0, sb=3, gb=2, tries=200):
    d = 2 * g + 2
    for _ in range(tries):
        lo = max(2, H // 2)
        b = rng.randint(lo, H); a = rng.randint(lo, H) * rng.choice([-1, 1])
        if gcd(abs(a), b) != 1 or Fraction(a, b) == x0: continue
        s = [rng.randint(-sb, sb) for _ in range(g + 1)] + [1]          # monic, deg g+1
        Gd = 2 * g - 1                                                    # 2 + Gd <= d-1
        G = [rng.randint(-gb, gb) for _ in range(Gd)] + [rng.choice([-gb, -1, 1, gb])]
        F = padd(pmul(s, s), pmul(pmul([-x0, 1], [-a, b]), G))
        if len(F) != d + 1 or F[-1] != 1 or not squarefree(F): continue
        xp = Fraction(a, b)
        assert peval(F, xp) == peval(s, xp) ** 2 and peval(F, Fraction(x0)) == peval(s, Fraction(x0)) ** 2
        return dict(genus=g, height=H, coeffs=[int(c) for c in reversed(F)], seed_x=x0,
                    seed_y=str(peval(s, Fraction(x0))), planted_x=f"{a}/{b}",
                    planted_y=str(peval(s, xp)), a=a, b=b, s=s, G=G)
    raise RuntimeError("could not build a squarefree curve")

def config_snippet(e):
    cs = ", ".join(f"QQ({c})" for c in e['coeffs'])
    return (f"COEFFS_GENUS2 = [{cs}]\nDATA_PTS_GENUS2 = [QQ({e['seed_x']})]\n"
            f"TARGETED_X = QQ({e['a']})/QQ({e['b']})   # planted; adjust to TARGETED_X's expected type\n"
            f"# expected m = seed_x - x = {Fraction(e['seed_x']) - Fraction(e['a'], e['b'])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--genus', default='2'); ap.add_argument('--heights', default='100,1000,10000')
    ap.add_argument('--per-height', type=int, default=5); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--x0', type=int, default=0); ap.add_argument('--out', default='manifest.json')
    ap.add_argument('--show', nargs=2, metavar=('MANIFEST', 'IDX'))
    A = ap.parse_args()
    if A.show:
        e = json.load(open(A.show[0]))[int(A.show[1])]; print(config_snippet(e)); return
    rng = random.Random(A.seed); out = []
    for g in map(int, A.genus.split(',')):
        for H in map(int, A.heights.split(',')):
            for _ in range(A.per_height):
                e = make_curve(g, H, rng, A.x0); e['idx'] = len(out); out.append(e)
    json.dump(out, open(A.out, 'w'), indent=1)
    print(f"wrote {len(out)} curves to {A.out}")

if __name__ == '__main__':
    main()
