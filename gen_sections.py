"""
gen_sections.py -- exact generator for quartic fibrations  y^2 = h(x; m)  carrying
n >= 6 sections (x_j, Y_j(m)) with x_j constant and Y_j(m) = a_j*m + b_j.

WHY THIS IS NEEDED
------------------
For n <= 5 nodes the values Y_j(m)^2 can be interpolated by a quartic h(x) freely.
For n >= 6 the interpolant of the Y_j^2 has degree up to n-1, and we need it to have
degree <= 4.  Writing h = h2*m^2 + h1*m + h0 (quartics h_i in x) and Y_j = a_j m + b_j,
the requirement is  h2(x_j)=a_j^2,  h1(x_j)=2 a_j b_j,  h0(x_j)=b_j^2.  With

        w_j = 1 / prod_{i != j} (x_j - x_i)

a vector s in Q^n is the value vector of a polynomial of degree <= 4 iff

        sum_j w_j x_j^e s_j = 0     for e = 0, ..., n-6 .

So we need (a, b) in Q^n with  F_e(a) = F_e(a,b) = F_e(b) = 0  for the quadratic forms
F_e(y) = sum_j w_j x_j^e y_j^2:  a TOTALLY ISOTROPIC PLANE span(a, b).

n = 6 :  one form F_0 on Q^6.  Solved exactly with PARI's qfsolve (an isotropic a, then an
         isotropic b inside a^perp / a).  Solvable iff the 4-dim residual form is isotropic
         over Q (checked; other node sets are tried otherwise).
n = 7, 8: take a 6-node solution and look for the remaining nodes among the roots of
         R(x) = D(x) / prod_{j<=6}(x - x_j),  D = h1^2 - 4 h0 h2  (degree 8).  A root rho of R
         gives a new section iff h2(rho) is a rational square.  This last step is a
         SEARCH, not a construction: it succeeds only when disc(R) and h2(rho) happen to be
         squares, so it is randomized with a budget and may legitimately fail.

n = 7, 8 (default method 'evenodd'): the random-extension idea above was tried and is far too
         unlikely to hit (it needs disc(R) and h2(rho) all to be squares).  Instead use the
         structured family  h = u(x) m^2 + v(x),  u = c_u prod(x - alpha_i),  v = c_v prod(x - beta_j):
           * at x = alpha_i:  h = v(alpha_i)        -> section Y = b_i (constant)     if v(alpha_i) is a square
           * at x = beta_j :  h = u(beta_j) m^2     -> section Y = a_j * m            if u(beta_j) is a square
         This is a quadratic base change (mu = m^2) of a rational elliptic surface, 4 + 4 sections.
         The square conditions say: the four row products  prod_j (alpha_i - beta_j)  lie in ONE
         square class (that class is c_v) and the four column products  prod_i (alpha_i - beta_j)
         lie in ONE square class (that is c_u).  That is a small combinatorial search over integers
         using square-class bitmasks.  Finally m -> m + t is applied so no Y_j is the zero polynomial
         at m = 0 (the beta-sections otherwise vanish at m = 0).  Sections for n < 8 are a subset.

Everything returned is verified in exact rational arithmetic before it is handed back.

NOT TESTED inside Sage by me; the pure-Python core was tested with cypari (PARI).
"""

import os
import random
import re
import time
from itertools import combinations
from collections import Counter
from fractions import Fraction as Fr
from math import gcd, isqrt
from functools import reduce

REQUIRE_ALL_I1 = os.environ.get("GEN_REQUIRE_I1", "0") == "1"

# reject-reason counters, printed in the progress line / final error to localize failures
STATS = Counter()

# --------------------------------------------------------------------------- PARI glue
_PARI = None


def _get_pari():
    global _PARI
    if _PARI is not None:
        return _PARI
    try:                                   # Sage ships cypari2
        import cypari2
        _PARI = cypari2.Pari()
    except Exception:
        try:                               # standalone testing
            from cypari import pari as _p
            _PARI = _p
        except Exception:
            try:
                from sage.all import pari as _p
                _PARI = _p
            except Exception as exc:
                raise RuntimeError("need cypari2, cypari or Sage for qfsolve") from exc
    return _PARI


def _gp_matrix(G):
    n = len(G)
    rows = ";".join(",".join(str(Fr(G[i][j])) for j in range(n)) for i in range(n))
    return _get_pari()("[" + rows + "]")


def qfsolve(G):
    """Nontrivial rational solution of v^T G v = 0 (G symmetric, rational), or None."""
    n = len(G)
    if n == 1:
        return [Fr(1)] if G[0][0] == 0 else None
    res = _gp_matrix(G).qfsolve()
    s = str(res).strip()
    if not s.startswith("["):
        STATS["qfsolve_obstruction(dim=%d):%s" % (n, s[:12])] += 1
        return None                        # integer obstruction (-1 or a prime)
    toks = re.findall(r"-?\d+(?:/\d+)?", s)
    v = [Fr(t) for t in toks]
    if len(v) != n or all(c == 0 for c in v):
        STATS["qfsolve_unparsable:%s" % s[:40]] += 1
        return None
    return v


# --------------------------------------------------------------------------- exact helpers
def is_square(q):
    q = Fr(q)
    if q < 0:
        return False
    return isqrt(q.numerator) ** 2 == q.numerator and isqrt(q.denominator) ** 2 == q.denominator


def sqrt_q(q):
    q = Fr(q)
    return Fr(isqrt(q.numerator), isqrt(q.denominator))


def clear_denoms(vs):
    """Scale the list of Fraction vectors by a common rational so all entries are coprime ints."""
    allv = [c for v in vs for c in v]
    L = reduce(lambda x, y: x * y // gcd(x, y), [c.denominator for c in allv], 1)
    ints = [[int(c * L) for c in v] for v in vs]
    g = reduce(gcd, [abs(c) for v in ints for c in v], 0) or 1
    return [[Fr(c // g) for c in v] for v in ints]


def poly_mul(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def poly_add(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def poly_scale(p, c):
    return [c * a for a in p]


def poly_trim(p):
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def poly_eval(p, x):
    r = Fr(0)
    for c in reversed(p):
        r = r * x + c
    return r


def poly_divmod(p, d):
    p = poly_trim(p)
    d = poly_trim(d)
    if len(p) < len(d):
        return [Fr(0)], p
    q = [Fr(0)] * (len(p) - len(d) + 1)
    p = list(p)
    for k in range(len(p) - len(d), -1, -1):
        c = p[k + len(d) - 1] / d[-1]
        q[k] = c
        for i, di in enumerate(d):
            p[k + i] -= c * di
    return q, poly_trim(p)


def interpolate(nodes, vals):
    """Lagrange interpolant (coefficient list, low->high) of the data (nodes, vals)."""
    n = len(nodes)
    total = [Fr(0)]
    for j in range(n):
        num = [Fr(1)]
        den = Fr(1)
        for i in range(n):
            if i != j:
                num = poly_mul(num, [-Fr(nodes[i]), Fr(1)])
                den *= Fr(nodes[j]) - Fr(nodes[i])
        total = poly_add(total, poly_scale(num, Fr(vals[j]) / den))
    return poly_trim(total) or [Fr(0)]


def weights(nodes):
    ws = []
    for j, xj in enumerate(nodes):
        d = Fr(1)
        for i, xi in enumerate(nodes):
            if i != j:
                d *= Fr(xj) - Fr(xi)
        ws.append(1 / d)
    return ws


def kernel_basis(rows, ncols):
    """Basis of {y : rows . y = 0} over Q (rows = list of Fraction lists)."""
    M = [list(map(Fr, r)) for r in rows]
    piv = []
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [v / M[r][c] for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        v = [Fr(0)] * ncols
        v[f] = Fr(1)
        for i, c in enumerate(piv):
            v[c] = -M[i][f]
        basis.append(v)
    return basis


# --------------------------------------------------------------------------- quadratic-form core
def _form6(nodes):
    ws = weights(nodes)
    n = len(nodes)
    return [[ws[i] if i == j else Fr(0) for j in range(n)] for i in range(n)]


def _bil(G, u, v):
    return sum(G[i][i] * u[i] * v[i] for i in range(len(u)))      # G diagonal here


def _random_isotropic_from(G, v0, rng, span=4):
    """Random rational isotropic vector via the line through known isotropic v0."""
    n = len(v0)
    for _ in range(50):
        d = [Fr(rng.randint(-span, span)) for _ in range(n)]
        qd = _bil(G, d, d)
        if qd == 0:
            continue
        s = -2 * _bil(G, v0, d) / qd
        v = [v0[i] + s * d[i] for i in range(n)]
        if any(c != 0 for c in v):
            return v
    return v0


def _perp_complement(G, a):
    """Four vectors spanning a^perp modulo a (G diagonal 6x6, a isotropic)."""
    K = kernel_basis([[G[i][i] * a[i] for i in range(6)]], 6)     # a^perp, dim 5, contains a
    pv = next(i for i in range(6) if a[i] != 0)
    E = []
    for k in K:
        kk = [k[i] - (k[pv] / a[pv]) * a[i] for i in range(6)]
        if any(c != 0 for c in kk):
            E.append(kk)
    ind = []
    for v in E:
        cand = ind + [v]
        if len(kernel_basis([list(r) for r in zip(*cand)], len(cand))) == 0:
            ind = cand
    return ind if len(ind) == 4 else None


def _random_iso_full(H, v0, rng, span=4):
    """Random rational isotropic vector for a full (non-diagonal) Gram matrix H, from known v0."""
    n = len(v0)

    def bil(u, v):
        return sum(H[i][j] * u[i] * v[j] for i in range(n) for j in range(n))
    for _ in range(50):
        d = [Fr(rng.randint(-span, span)) for _ in range(n)]
        qd = bil(d, d)
        if qd == 0:
            continue
        s = -2 * bil(v0, d) / qd
        v = [v0[i] + s * d[i] for i in range(n)]
        if any(c != 0 for c in v):
            return v
    return v0


def isotropic_plane_6(nodes, rng):
    """
    One random totally isotropic plane (a, b) for F_0 = sum w_j y_j^2 on Q^6, or None if this
    node set admits none (F_0 anisotropic, or the 4-dim residual form a^perp/a anisotropic).
    Returns (a, b) as coprime integer-valued Fraction lists.
    """
    G = _form6(nodes)
    a0 = qfsolve(G)
    if a0 is None:
        return None
    a = _random_isotropic_from(G, a0, rng)
    E = _perp_complement(G, a)
    if E is None:
        STATS["perp_complement_failed"] += 1
        return None
    H4 = [[sum(G[t][t] * E[i][t] * E[j][t] for t in range(6)) for j in range(4)] for i in range(4)]
    w0 = qfsolve(H4)
    if w0 is None:
        return None
    w = _random_iso_full(H4, w0, rng)
    b = [sum(w[i] * E[i][t] for i in range(4)) for t in range(6)]
    if any(c == 0 for c in a):
        STATS["a_has_zero"] += 1
        return None
    # E was reduced so that coordinate pv of every basis vector is 0, hence b[pv] == 0 always.
    # b -> b + t*a keeps the plane (it is the shift m -> m + t) and makes all b_j nonzero.
    for _ in range(40):
        t = Fr(rng.randint(-5, 5), rng.randint(1, 3))
        b2 = [bj + t * aj for aj, bj in zip(a, b)]
        if all(c != 0 for c in b2):
            b = b2
            break
    a_i, b_i = clear_denoms([a, b])
    if not _check_plane(G, a_i, b_i):
        STATS["check_plane_failed"] += 1
        return None
    return (a_i, b_i)


def _check_plane(G, a, b):
    return _bil(G, a, a) == 0 and _bil(G, a, b) == 0 and _bil(G, b, b) == 0 and \
        any(c != 0 for c in a) and any(c != 0 for c in b)


# --------------------------------------------------------------------------- h and verification
def build_h(nodes, a, b):
    """Return (h2, h1, h0) quartic coefficient lists (low->high) with h(x_j) = (a_j m + b_j)^2."""
    h2 = interpolate(nodes, [aj * aj for aj in a])
    h1 = interpolate(nodes, [2 * aj * bj for aj, bj in zip(a, b)])
    h0 = interpolate(nodes, [bj * bj for bj in b])
    return h2, h1, h0


def verify(nodes, a, b):
    """Exact check: all three interpolants have degree <= 4 and reproduce the squares."""
    h2, h1, h0 = build_h(nodes, a, b)
    if max(len(h2), len(h1), len(h0)) > 5:
        return False
    for j, xj in enumerate(nodes):
        if poly_eval(h2, Fr(xj)) != a[j] ** 2 or poly_eval(h1, Fr(xj)) != 2 * a[j] * b[j] \
                or poly_eval(h0, Fr(xj)) != b[j] ** 2:
            return False
    return True


# --------------------------------------------------------------------------- all-I1 screen
def poly_deriv(p):
    return [i * c for i, c in enumerate(p)][1:] or [Fr(0)]


def poly_gcd(p, q):
    p, q = poly_trim(p), poly_trim(q)
    while q:
        _, r = poly_divmod(p, q)
        p, q = q, r
    return p


def all_I1(h2, h1, h0):
    """
    True iff the discriminant of the quartic  h = h2(x) m^2 + h1(x) m + h0(x)  (in x), viewed
    as a polynomial in m, has degree exactly 12 and is squarefree: i.e. twelve I1 fibers,
    smooth fiber at infinity, Sigma(m_v - 1) = 0.  Uses  Disc ~ 4 I^3 - J^2  with
    I = 12ae - 3bd + c^2,  J = 72ace + 9bcd - 27ad^2 - 27b^2 e - 2c^3.
    """
    def coef(k):                                   # coefficient of x^k as a polynomial in m
        g = lambda p: p[k] if k < len(p) else Fr(0)
        return poly_trim([g(h0), g(h1), g(h2)]) or [Fr(0)]
    a, b, c, d, e = coef(4), coef(3), coef(2), coef(1), coef(0)
    mul, add, sc = poly_mul, poly_add, poly_scale
    I = add(add(sc(mul(a, e), 12), sc(mul(b, d), -3)), mul(c, c))
    J = sc(mul(mul(a, c), e), 72)
    J = add(J, sc(mul(mul(b, c), d), 9))
    J = add(J, sc(mul(a, mul(d, d)), -27))
    J = add(J, sc(mul(mul(b, b), e), -27))
    J = add(J, sc(mul(c, mul(c, c)), -2))
    D = poly_trim(add(sc(mul(I, mul(I, I)), 4), sc(mul(J, J), -1)))
    if len(D) != 13:
        return False
    # Squarefree over Q  <=  squarefree mod some prime p with the degree (12) preserved.
    # Plain integer arithmetic mod p; exact rational Euclid blows up and is far too slow.
    L = reduce(lambda x, y: x * y // gcd(x, y), [c.denominator for c in D], 1)
    Di = [int(c * L) for c in D]
    for p in (10007, 10009, 10037, 10039, 10061):
        if Di[-1] % p == 0:
            continue
        if _squarefree_mod_p(Di, p):
            return True
    return False                                   # conservative: treated as degenerate


def _gf_trim(a):
    while a and a[-1] == 0:
        a.pop()
    return a


def _gf_mod(a, b, p):
    a = _gf_trim(list(a))
    inv = pow(b[-1], -1, p)
    while len(a) >= len(b):
        c = a[-1] * inv % p
        k = len(a) - len(b)
        for i, bc in enumerate(b):
            a[k + i] = (a[k + i] - c * bc) % p
        _gf_trim(a)
    return a


def _squarefree_mod_p(coeffs, p):
    """coeffs: integer coefficients low->high, leading one nonzero mod p, p > degree."""
    f = _gf_trim([c % p for c in coeffs])
    df = _gf_trim([(i * c) % p for i, c in enumerate(f)][1:])
    a, b = f, df
    while b:
        a, b = b, _gf_mod(a, b, p)
    return len(a) == 1


# --------------------------------------------------------------------------- extension to n = 7, 8
def extension_candidates(nodes6, a, b):
    """
    Candidate additional sections from the roots of R = D / prod(x - x_j).
    Returns a list of (rho, a_new, b_new) with h(rho; m) = (a_new m + b_new)^2 exactly.
    """
    h2, h1, h0 = build_h(nodes6, a, b)
    D = poly_add(poly_mul(h1, h1), poly_scale(poly_mul(h0, h2), -4))
    P = [Fr(1)]
    for xj in nodes6:
        P = poly_mul(P, [-Fr(xj), Fr(1)])
    R, rem = poly_divmod(D, P)
    if rem:
        raise AssertionError("D not divisible by prod(x - x_j): internal inconsistency")
    R = poly_trim(R)
    roots = []
    if len(R) == 3:
        disc = R[1] ** 2 - 4 * R[2] * R[0]
        if is_square(disc):
            s = sqrt_q(disc)
            roots = [(-R[1] + s) / (2 * R[2]), (-R[1] - s) / (2 * R[2])]
            if s == 0:
                roots = roots[:1]
    elif len(R) == 2:
        roots = [-R[0] / R[1]]
    out = []
    for rho in roots:
        if rho in [Fr(x) for x in nodes6]:
            continue
        A = poly_eval(h2, rho)
        if A == 0 or not is_square(A):
            continue
        an = sqrt_q(A)
        bn = poly_eval(h1, rho) / (2 * an)
        if poly_eval(h0, rho) != bn * bn:
            continue
        out.append((rho, an, bn))
    return out


# --------------------------------------------------------------------------- even/odd family (n = 5..8)
_PRIME_BIT = {}          # prime -> bit position (bit 0 is the sign)
_BIT_PRIME = [None]      # bit position -> prime
_MASK_CACHE = {}


def class_mask(d):
    """Square class of the nonzero integer d as a bitmask: bit 0 = (d < 0), bit k = parity of the
    exponent of the k-th prime that has been seen.  XOR of masks = mask of the product."""
    d = int(d)
    if d in _MASK_CACHE:
        return _MASK_CACHE[d]
    m = 1 if d < 0 else 0
    n = abs(d)
    p = 2
    while p * p <= n:
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e & 1:
                if p not in _PRIME_BIT:
                    _PRIME_BIT[p] = len(_BIT_PRIME)
                    _BIT_PRIME.append(p)
                m ^= 1 << _PRIME_BIT[p]
        p += 1 if p == 2 else 2
    if n > 1:
        if n not in _PRIME_BIT:
            _PRIME_BIT[n] = len(_BIT_PRIME)
            _BIT_PRIME.append(n)
        m ^= 1 << _PRIME_BIT[n]
    _MASK_CACHE[d] = m
    return m


def mask_to_squarefree(mask):
    """Squarefree integer (with sign) whose class is `mask`."""
    v = -1 if (mask & 1) else 1
    k = 1
    mask >>= 1
    while mask:
        if mask & 1:
            v *= _BIT_PRIME[k]
        mask >>= 1
        k += 1
    return v


def _poly_from_roots(roots):
    p = [Fr(1)]
    for r in roots:
        p = poly_mul(p, [-Fr(r), Fr(1)])
    return p


def _prod(xs):
    r = 1
    for x in xs:
        r *= x
    return r


def search_even_odd(rng, N_start=8, N_max=60, N_step=4, time_budget=300, verbose=True):
    """
    Find integers alpha_1..4, beta_1..4 (distinct, in [-N, N]) and squarefree c_u, c_v with
      c_v * prod_j (alpha_i - beta_j) a perfect square for every i,
      c_u * prod_i (alpha_i - beta_j) a perfect square for every j.
    Returns (alphas, betas, c_u, c_v) or None when the time budget runs out.
    """
    t0 = time.time()
    last_beat = t0
    tried = 0
    N = N_start
    while N <= N_max:
        pool = list(range(-N, N + 1))
        stage_end = min(t0 + time_budget, time.time() + max(10.0, time_budget / 6.0))
        while time.time() < stage_end:
            alpha = sorted(rng.sample(pool, 4))
            aset = set(alpha)
            if verbose and time.time() - last_beat > 10:
                last_beat = time.time()
                print(f"[gen_sections] even/odd: N={N}, {tried} alpha-sets, "
                      f"{time.time() - t0:.0f}s, reasons={dict(STATS)}")
            groups = {}
            for bt in pool:
                if bt in aset:
                    continue
                vs = [class_mask(a - bt) for a in alpha]
                u = vs[0] ^ vs[1] ^ vs[2] ^ vs[3]            # class of prod_i (alpha_i - beta)
                groups.setdefault(u, []).append((bt, vs))
            tried += 1
            for u, g in groups.items():
                if len(g) < 4:
                    continue
                for combo in combinations(g, 4):
                    r = [0, 0, 0, 0]
                    for _, vs in combo:
                        for i in range(4):
                            r[i] ^= vs[i]
                    if r[0] == r[1] == r[2] == r[3]:
                        betas = [c[0] for c in combo]
                        c_v = mask_to_squarefree(r[0])
                        c_u = mask_to_squarefree(u)
                        ok = all(is_square(Fr(c_v * _prod(a - bt for bt in betas))) for a in alpha) and \
                            all(is_square(Fr(c_u * _prod(a - bt for a in alpha))) for bt in betas)
                        if ok:
                            STATS["evenodd_square_class_hits"] += 1
                            uu = poly_scale(_poly_from_roots(alpha), Fr(c_u))
                            vv = poly_scale(_poly_from_roots(betas), Fr(c_v))
                            # all-I1 is NOT required (this family has I2 pairs from m -> -m);
                            # only the number of independent sections matters. Opt in with GEN_REQUIRE_I1=1.
                            if REQUIRE_ALL_I1 and not all_I1(uu, [Fr(0)], vv):
                                STATS["evenodd_not_all_I1"] += 1
                                continue
                            # stage-boundary invariants (cheap): u, v squarefree of degree 4, distinct roots
                            assert len(set(alpha)) == 4 and len(set(betas)) == 4
                            assert not (set(alpha) & set(betas)), "alpha/beta overlap"
                            assert len(uu) == 5 and len(vv) == 5 and uu[4] != 0 and vv[4] != 0
                            STATS["evenodd_accepted"] += 1
                            if verbose:
                                print(f"[gen_sections] even/odd: N={N}, {tried} alpha-sets tried, "
                                      f"alpha={alpha}, beta={betas}, c_u={c_u}, c_v={c_v}")
                            return alpha, betas, c_u, c_v
                        STATS["evenodd_mask_mismatch"] += 1
            if time.time() - t0 > time_budget:
                break
        if verbose:
            print(f"[gen_sections] even/odd: N={N} exhausted stage ({tried} alpha-sets, "
                  f"{time.time() - t0:.0f}s); widening")
        if time.time() - t0 > time_budget:
            break
        N += N_step
    return None


def generate_even_odd(n, seed=1, shift=1, time_budget=300, verbose=True):
    """5 <= n <= 8 sections from h = u m^2 + v; returns (nodes, a, b) with Y_j = a_j (m) + b_j."""
    if not 5 <= n <= 8:
        raise ValueError("even/odd generator is for 5 <= n <= 8")
    rng = random.Random(seed)
    sol = search_even_odd(rng, time_budget=time_budget, verbose=verbose)
    if sol is None:
        raise RuntimeError(f"gen_sections(even/odd): nothing found in {time_budget}s; "
                           f"raise time_budget or change seed. reasons={dict(STATS)}")
    alpha, betas, c_u, c_v = sol
    nodes, A, B = [], [], []
    for a in alpha:                                   # Y = b (constant), b^2 = v(alpha)
        val = Fr(c_v * _prod(a - bt for bt in betas))
        nodes.append(Fr(a)); A.append(Fr(0)); B.append(sqrt_q(val))
    for bt in betas:                                  # Y = a*m,     a^2 = u(beta)
        val = Fr(c_u * _prod(bt - a for a in alpha))
        nodes.append(Fr(bt)); A.append(sqrt_q(val)); B.append(Fr(0))
    t = Fr(shift)                                     # m -> m + t:  a*m + b  ->  a*m + (b + a*t)
    B = [bj + aj * t for aj, bj in zip(A, B)]
    nodes, A, B = nodes[:n], A[:n], B[:n]
    if n < 8:                                         # n=5..7: any subset of the 8 sections is valid
        pass
    if not verify(nodes, A, B):
        raise RuntimeError("gen_sections(even/odd): exact verification failed (bug)")
    return nodes, A, B


# --------------------------------------------------------------------------- public driver
def generate(n, seed=1, node_range=6, max_node_sets=200, planes_per_set=300,
             verbose=True, want_entries_below=None, method=None):
    """
    Return (nodes, a, b): nodes[j] = x_j (Fractions), Y_j(m) = a[j]*m + b[j] (integers or
    rationals), n sections, verified exactly.  Raises RuntimeError if nothing is found.

      method: 'isotropic' (qfsolve plane; default for n=6, with random extension for n=7,8 -- not
              recommended) or 'evenodd' (default for n=7,8; see module docstring).
      n = 6 : always found if some node set in [-node_range, node_range] admits an isotropic plane.
      n = 7, 8 : randomized search; may fail within the budget (then try another seed/budget).
    """
    if n < 6 or n > 8:
        raise ValueError("generate() is for 6 <= n <= 8 (rank of a rational elliptic surface is <= 8)")
    if method is None:
        method = "isotropic" if n == 6 else "evenodd"
    if method == "evenodd":
        return generate_even_odd(n, seed=seed, verbose=verbose)
    rng = random.Random(seed)
    pool = list(range(-node_range, node_range + 1))
    best = None
    for ns in range(max_node_sets):
        nodes6 = sorted(rng.sample(pool, 6))
        got_plane = False
        for pl in range(planes_per_set):
            res = isotropic_plane_6(nodes6, rng)
            if res is None:
                if pl == 0:
                    break                       # node set admits no plane at all
                continue
            got_plane = True
            a, b = res
            if any(c == 0 for c in a) or any(c == 0 for c in b):
                STATS["zero_entry"] += 1
                continue                        # keep Y_j nonconstant & nonzero at m=0
            if not verify(nodes6, a, b):
                STATS["verify_failed"] += 1
                continue
            h2_, h1_, h0_ = build_h(nodes6, a, b)
            if not all_I1(h2_, h1_, h0_):
                STATS["not_all_I1"] += 1
                continue
            STATS["plane_ok"] += 1
            if n == 6:
                size = max(abs(c) for c in a + b)
                if best is None or size < best[0]:
                    best = (size, [Fr(x) for x in nodes6], a, b)
                if pl >= 20 or (want_entries_below and size <= want_entries_below):
                    break
                continue
            cands = extension_candidates(nodes6, a, b)
            need = n - 6
            if len(cands) >= need:
                take = cands[:need]
                nodes = [Fr(x) for x in nodes6] + [c[0] for c in take]
                A = list(a) + [c[1] for c in take]
                B = list(b) + [c[2] for c in take]
                A, B = clear_denoms([A, B])
                if verify(nodes, A, B):
                    if verbose:
                        print(f"[gen_sections] n={n}: found after {ns + 1} node sets, plane #{pl}")
                    return nodes, A, B
        if n == 6 and best is not None:
            if verbose:
                print(f"[gen_sections] n=6: nodes={best[1]}, max |entry|={best[0]}")
            return best[1], best[2], best[3]
        if verbose and ns % 5 == 4:
            print(f"[gen_sections] n={n}: {ns + 1} node sets tried, no solution yet; reasons={dict(STATS)}")
    raise RuntimeError(f"gen_sections: no solution for n={n} within budget "
                       f"(node sets={max_node_sets}, planes/set={planes_per_set}); "
                       f"try a different seed or larger budget; reasons={dict(STATS)}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nodes, a, b = generate(n, seed)
    print("nodes:", [str(x) for x in nodes])
    print("Y_j(m) = a_j*m + b_j:")
    for x, aj, bj in zip(nodes, a, b):
        print(f"  x={x}: ({aj})*m + ({bj})")
    print("verified:", verify(nodes, a, b))
