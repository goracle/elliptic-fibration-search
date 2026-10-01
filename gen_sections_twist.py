"""
gen_sections_twist.py -- n = 9 constant-x sections on a K3  y^2 = h~(x; t)  (chi = 2, Y_j(t) of degree <= 2).

WRITTEN FROM SCRATCH; tested in plain Python only (exact verify_ydeg2), NOT through Sage / baseline_quartic.sage.

IDEA
----
Start from a rank-8 rational elliptic surface  y^2 = h(x; mu) = h2 mu^2 + h1 mu + h0  with constant-x sections
(x_j, a_j mu + b_j) (gen_sections_conic.generate_conic).  A degree-2 base change mu = N(t)/D(t) gives a K3 with
MW rank = rank E(Q(mu)) + rank of the QUADRATIC TWIST of E by the discriminant d(mu) of N - mu D in t.
A generic twist has rank 0 (generic rho = 10), so we CHOOSE the base change so that the twist has a section.
Constant-x sections of the twist:

  (a) x0 = rho_i, a rational root of h2 (the four rho_i come from the conic construction).  Then
      h(x0; mu) = h1 (mu - mub) is LINEAR; take  mu = c t^2 + mub  with  c = squarefree class of h1(x0).
      Then h(x0; phi(t)) = h1 c t^2 = (s t)^2 :  new section  (x0, s t).      [d = c (mu - mub), branch pts mub, oo]
  (b) any other rational x0 with  disc(x0) = h1^2 - 4 h2 h0  a square (rational points on y^2 = kappa P(x), genus 3):
      h(x0; mu) = h2 (mu - mua)(mu - mub), mua, mub rational.  Take  mu = (mua t^2 - c mub)/(t^2 - c),
      c = class of h2(x0).  Then D^2 h(x0; mu) = h2 c (mua - mub)^2 t^2 : new section (x0, s t).
In both cases the 8 old sections pull back to degree-2 polynomials in t, the new one is (x0, s t); the K3 has
the involution t -> -t, old sections are invariant, the new one anti-invariant, so it is independent of them
unless torsion.  (Branch points of the base change are the two special fibres mua, mub (or mub, oo).)

CAVEAT: still not generic (the K3 is a pullback), and rho >= 11 is only a LOWER bound until the pipeline's
height matrix confirms independence.
"""
import random
from fractions import Fraction as Fr
from math import gcd

from gen_sections import (STATS, build_h, poly_eval, is_square, sqrt_q, clear_denoms)
import gen_sections_conic as GC
from gen_sections_conic import generate_conic, _mask_frac, _sqfree_from_mask


def _cls(v):
    return Fr(_sqfree_from_mask(_mask_frac(v)))


def _y(a, b, mu):
    return [aj * mu + bj for aj, bj in zip(a, b)]


def _case_a(nodes, a, b, h, rho):
    h2, h1, h0 = h
    out = []
    for r in rho:
        if poly_eval(h2, r) != 0 or poly_eval(h1, r) == 0:
            continue
        mub = -poly_eval(h0, r) / poly_eval(h1, r)
        c = _cls(poly_eval(h1, r))
        s = sqrt_q(poly_eval(h1, r) * c)
        cols = [[aj * mub + bj, Fr(0), aj * c] for aj, bj in zip(a, b)] + [[Fr(0), s, Fr(0)]]
        out.append(("a", r, cols))
    return out


def _case_b(nodes, a, b, h, Q=24, P=150):
    h2, h1, h0 = h
    seen = set()
    out = []
    for q in range(1, Q + 1):
        for p in range(-P, P + 1):
            if gcd(p, q) != 1:
                continue
            x0 = Fr(p, q)
            if x0 in nodes or x0 in seen:
                continue
            seen.add(x0)
            H2, H1, H0 = poly_eval(h2, x0), poly_eval(h1, x0), poly_eval(h0, x0)
            if H2 == 0:
                continue
            disc = H1 * H1 - 4 * H2 * H0
            if disc == 0 or not is_square(disc):
                continue
            r = sqrt_q(disc)
            mua, mub = (-H1 + r) / (2 * H2), (-H1 - r) / (2 * H2)
            c = _cls(H2)
            s = sqrt_q(H2 * c) * abs(mua - mub)
            cols = [[-c * (aj * mub + bj), Fr(0), aj * mua + bj] for aj, bj in zip(a, b)] + [[Fr(0), s, Fr(0)]]
            out.append(("b", x0, cols))
    return out


def generate_ydeg2_n9(seed=1, verbose=True, max_surfaces=100, scan_b=False):
    """Return (nodes, coeffs): 9 constant-x sections, coeffs[j] = [c0, c1, c2], Y_j(t) = c0 + c1 t + c2 t^2."""
    from gen_sections_deg2 import verify_ydeg2
    for k in range(max_surfaces):
        nodes, a, b = generate_conic(seed + k, verbose=False)
        rho = GC.LAST["rho"]
        h = build_h(nodes, a, b)
        cands = _case_a(nodes, a, b, h, rho)
        if scan_b:
            cands += _case_b(nodes, a, b, h)
        best = None
        for kind, x0, cols in cands:
            allx = [Fr(x) for x in nodes] + [Fr(x0)]
            c0, c1, c2 = clear_denoms([[cj[0] for cj in cols], [cj[1] for cj in cols], [cj[2] for cj in cols]])
            coeffs = [[c0[j], c1[j], c2[j]] for j in range(len(cols))]
            if not verify_ydeg2(allx, coeffs):
                STATS["twist_verify_failed"] += 1
                continue
            size = max(abs(c) for cj in coeffs for c in cj)
            if best is None or size < best[0]:
                best = (size, kind, x0, allx, coeffs)
        if best:
            _, kind, x0, allx, coeffs = best
            order = sorted(range(9), key=lambda i: allx[i])
            if verbose:
                print(f"[gen_sections_twist] n=9: surface #{k + 1}, case ({kind}), new node x0={x0}, max|coef|={best[0]}")
            return [allx[i] for i in order], [coeffs[i] for i in order]
    raise RuntimeError(f"generate_ydeg2_n9: nothing found; reasons={dict(STATS)}")


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nodes, coeffs = generate_ydeg2_n9(seed)
    for x, c in zip(nodes, coeffs):
        print(f"x={x}: ({c[0]}) + ({c[1]})*t + ({c[2]})*t^2")
