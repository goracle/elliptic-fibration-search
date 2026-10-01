"""
gen_sections_conic.py -- n = 8 constant-x sections on a rational elliptic surface
        y^2 = h(x; mu) = e * ( A(x) mu^2 + B(x) mu + C(x) ),     A, B, C of degree 4 in x,
with a CROSS TERM B != 0 (the evenodd family has B = 0).  Same return convention as
gen_sections.generate:  (nodes, a, b)  with  Y_j(mu) = a_j mu + b_j,  Y_j^2 = h(x_j; mu).

WRITTEN FROM SCRATCH (the earlier gen_sections_conic.py was never saved).  Tested only in plain Python,
not in Sage and not through baseline_quartic.sage.

MATH
----
A section (x_j, a_j mu + b_j) needs the binary form  A_j U^2 + B_j U V + C_j V^2  (U = mu, V = 1) to be
a square times e, i.e.
        (i)  B_j^2 - 4 A_j C_j = 0,          (ii)  e * A_j is a rational square.
Choose A(x) = prod_{i=1..4} (x - rho_i)  (monic, four rational roots).  Put  D = B^2 - 4AC = kappa * P,
P(x) = prod_{j=1..8} (x - x_j).  Then for each root rho_i of A we need  B(rho_i)^2 = kappa P(rho_i), and
conversely, if that holds, A | B^2 - kappa P and  C = (B^2 - kappa P) / (4A)  is a polynomial of degree 4.
So the data is
        nodes x_j, roots rho_i, constants kappa, e   with
        (R) e * prod_i (x_j - rho_i)      a square   for every j          [rows]
        (K) kappa * prod_j (x_j - rho_i)  a square   for every i          [columns]
(R) says every x_j is the x-coordinate of a rational point on the genus-1 curve  e y^2 = prod_i (x - rho_i).
(K) says the 2-descent classes of the 8 points multiply to the same class for all four rho_i; descent
classes live in a SMALL F_2-space (dimension <= rank + 2), so given ~10-40 points on one such curve a
subset of 8 satisfying (K) is found by linear algebra / meet-in-the-middle over F_2.
Then B is the cubic through (rho_i, +-sqrt(kappa P(rho_i))) (plus an optional t*A), C as above, and
        a_j = s_j,   b_j = s_j * B_j / (2 A_j),    s_j = sqrt(e A_j).

CAVEAT: this is still a pullback situation if you base-change mu = phi(t): the K3 obtained that way is NOT
generic (see generate_ydeg2_n8_basechange).  It only widens the family of rank-8 rational surfaces.
"""
import random
from fractions import Fraction as Fr
from itertools import combinations
from math import gcd

from gen_sections import (STATS, verify, clear_denoms, poly_mul, poly_add, poly_scale, poly_eval,
                          poly_divmod, poly_trim, interpolate, sqrt_q, is_square, build_h, all_I1)

_SHIFT = 256            # bit-width of one F_2 block (primes + sign bit); assert below if exceeded
_PIDX = {}


def _bit(p):
    if p not in _PIDX:
        if len(_PIDX) >= _SHIFT:
            raise OverflowError("too many distinct primes")
        _PIDX[p] = len(_PIDX)
    return 1 << _PIDX[p]


_MCACHE = {}


def _mask(n):
    """F_2 mask of the squarefree class of the nonzero integer n (bit for -1 plus one bit per odd-power prime)."""
    r = _MCACHE.get(n)
    if r is not None:
        return r
    m = _bit(-1) if n < 0 else 0
    k = abs(n)
    p = 2
    while p * p <= k:
        if k % p == 0:
            c = 0
            while k % p == 0:
                k //= p
                c += 1
            if c & 1:
                m ^= _bit(p)
        p += 1 if p == 2 else 2
    if k > 1:
        m ^= _bit(k)
    _MCACHE[n] = m
    return m


def _sqfree_from_mask(m):
    """Signed squarefree integer whose class has mask m."""
    r = 1
    for p, i in _PIDX.items():
        if (m >> i) & 1:
            r *= p
    return r


def _scan_curve(rho, Q, P):
    """Bucket coprime (p, q), q >= 1, by the class of prod_i (p - rho_i q)  (class of e)."""
    buckets = {}
    for q in range(1, Q + 1):
        for p in range(-P, P + 1):
            if gcd(p, q) != 1:
                continue
            fs = [p - r * q for r in rho]
            if 0 in fs:
                continue
            row = 0
            for f in fs:
                row ^= _mask(f)
            mq = _mask(q)
            cols = [_mask(f) ^ mq for f in fs]                # class of (x - rho_i), x = p/q
            buckets.setdefault(row, []).append((p, q, cols))
    return buckets


def _pick_subset(pts, k=8):
    """k points (distinct x) whose column classes multiply to the same class for i = 1..4, or None."""
    h = k // 2
    w = []
    for (_, _, c) in pts:
        w.append((c[0] ^ c[1]) | ((c[0] ^ c[2]) << _SHIFT) | ((c[0] ^ c[3]) << (2 * _SHIFT)))
    seen = {}
    idx = range(len(pts))
    for comb in combinations(idx, h):
        x = 0
        for i in comb:
            x ^= w[i]
        for other in seen.get(x, ()):
            if not set(other) & set(comb):
                return list(other) + list(comb)
        seen.setdefault(x, []).append(comb)
    return None


def _search(rng, rho_range=6, Q=40, P=120, max_bucket=30, min_bucket=8):
    """Yield (rho, xs): rho = 4 rationals, xs = 8 nodes satisfying (R) and (K)."""
    while True:
        rho = sorted(rng.sample(range(-rho_range, rho_range + 1), 4))
        buckets = _scan_curve(rho, Q, P)
        STATS["conic_curves"] += 1
        for row, pts in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            if len(pts) < min_bucket:
                break
            STATS["conic_buckets_ge8"] += 1
            pts = pts[:max_bucket]
            sel = _pick_subset(pts)
            if sel is None:
                STATS["conic_no_subset"] += 1
                continue
            xs = [Fr(pts[i][0], pts[i][1]) for i in sel]
            yield [Fr(r) for r in rho], sorted(xs)


def generate_conic(seed=1, verbose=True, require_i1=False, max_curves=400):
    """Return (nodes, a, b), verified exactly (gen_sections.verify), with B != 0."""
    rng = random.Random(seed)
    for cnt, (rho, xs) in enumerate(_search(rng)):
        if cnt >= max_curves:
            break
        Pn = [Fr(1)]
        for x in xs:
            Pn = poly_mul(Pn, [-x, Fr(1)])
        A = [Fr(1)]
        for r in rho:
            A = poly_mul(A, [-r, Fr(1)])
        Pr = [poly_eval(Pn, r) for r in rho]
        kap = _sqfree_from_mask(_mask_frac(Pr[0]))
        if not all(is_square(kap * v) for v in Pr):
            STATS["conic_kappa_fail"] += 1
            continue
        Aj = [poly_eval(A, x) for x in xs]
        e = _sqfree_from_mask(_mask_frac(Aj[0]))
        if not all(is_square(e * v) for v in Aj):
            STATS["conic_e_fail"] += 1
            continue
        roots = [sqrt_q(kap * v) for v in Pr]
        for _ in range(40):
            sg = [rng.choice([1, -1]) for _ in rho]
            Bv = interpolate(rho, [s * r for s, r in zip(sg, roots)])        # degree <= 3
            Bv = poly_add(Bv, poly_scale(A, Fr(rng.randint(-2, 2))))          # optionally + t*A
            num = poly_add(poly_mul(Bv, Bv), poly_scale(Pn, -kap))
            C, rem = poly_divmod(num, poly_scale(A, 4))
            if poly_trim(rem) not in ([], [Fr(0)]):
                STATS["conic_not_divisible"] += 1
                continue
            Bj = [poly_eval(Bv, x) for x in xs]
            if any(b == 0 for b in Bj):
                STATS["conic_B_zero_at_node"] += 1
                continue
            s = [sqrt_q(e * v) for v in Aj]
            a = [sj for sj in s]
            b = [sj * bj / (2 * aj) for sj, bj, aj in zip(s, Bj, Aj)]
            a, b = clear_denoms([a, b])
            if not verify(xs, a, b):
                STATS["conic_verify_failed"] += 1
                continue
            if require_i1 and not all_I1(*build_h(xs, a, b)):
                STATS["conic_not_all_I1"] += 1
                continue
            if verbose:
                print(f"[gen_sections_conic] rho={[str(r) for r in rho]} nodes={[str(x) for x in xs]} "
                      f"e={e} kappa={kap} after {cnt + 1} curves")
            return xs, a, b
    raise RuntimeError(f"generate_conic: nothing found; reasons={dict(STATS)}")


def _mask_frac(v):
    v = Fr(v)
    return _mask(v.numerator) ^ _mask(v.denominator)


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nodes, a, b = generate_conic(seed)
    for x, aj, bj in zip(nodes, a, b):
        print(f"  x={x}: ({aj})*m + ({bj})")
    print("verified:", verify(nodes, a, b), " all_I1:", all_I1(*build_h(nodes, a, b)))
