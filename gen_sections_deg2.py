"""
gen_sections_deg2.py -- quartic fibrations y^2 = h(x; m) over QQ(m) with chi = 2 (K3) and n = 6 or 7
sections (x_j, Y_j(m)), x_j constant, Y_j(m) = y0_j + y1_j m + y2_j m^2.

MATH
----
h has degree <= 4 in x iff the value vector s_j(m) = Y_j(m)^2 satisfies, for e = 0..n-6,
    F_e(s) = sum_j w_j x_j^e s_j = 0,      w_j = 1 / prod_{i != j} (x_j - x_i).
n = 6: ONE quadratic form F = F_0 on Q^6.  Writing B for its bilinear form, the m-coefficients
of F(Y(m)) are
    m^0: F(y0)            m^1: 2B(y0,y1)          m^2: F(y1) + 2B(y0,y2)
    m^3: 2B(y1,y2)        m^4: F(y2)
all of which must vanish.  Construction:
    1. y0, y2 isotropic with t = B(y0,y2) != 0           (qfsolve + line-through-a-point)
    2. W = {y0, y2}^perp  (dim 4);   y1 in W with F(y1) = -2t     (qfsolve on  F|_W (+) <2t>)
Then m^1, m^3 hold because y1 is in W, m^0, m^4 because y0, y2 are isotropic, and m^2 because
F(y1) = -2t = -2B(y0,y2).  Everything is re-verified in exact arithmetic before returning.

n = 7: TWO forms F_0, F_1 (weights w_j and w_j x_j).  Y(m) = y0 + y1 m + y2 m^2 lies on both iff the
Gram matrices of F_0 AND F_1 on (y0,y1,y2) both equal lambda_e * [[0,0,1/2],[0,-1,0],[1/2,0,0]]
(the conic X0 X2 = X1^2).  So on V = span(y0,y1,y2) the two forms are proportional:
F_1|_V = mu0 * F_0|_V, i.e. V is TOTALLY ISOTROPIC for the pencil member
        F_{mu0} = F_1 - mu0 F_0 = sum_j w_j (x_j - mu0) y_j^2        (diagonal, 7 variables).
Construction:
    1. pick nodes x_j and a rational mu0 (not a node); the signs of w_j (x_j - mu0) always split 3/4,
       so F_{mu0} has a totally isotropic 3-space iff the last 3-dim form in the flag below is isotropic;
    2. build V by a flag of isotropic vectors v1, v2 (5-dim form: always isotropic by Meyer), v3 (conic);
    3. need F_0|_V nondegenerate and isotropic: take y0 isotropic, y1 in y0^perp, y2 isotropic in y1^perp,
       and rescale y2 so that F_0(y1) + 2 B_0(y0,y2) = 0.
Then every condition for e = 0 holds by construction and for e = 1 because F_1|_V = mu0 F_0|_V.
(n = 8 would need a 3-space isotropic for TWO pencil members: not attempted.)

NOT RUN by me (compile-checked only).  Needs qfsolve (cypari2 / Sage), same as gen_sections.py.
"""
import random
from fractions import Fraction as Fr

from gen_sections import (qfsolve, weights, kernel_basis, interpolate, poly_eval,
                          clear_denoms, _random_isotropic_from, _random_iso_full, _bil, STATS)


from math import gcd
from functools import reduce


def _prim(v):
    """Scale a Fraction vector to a primitive integer vector (isotropy etc. are scale-invariant)."""
    L = reduce(lambda a, b: a * b // gcd(a, b), [Fr(c).denominator for c in v], 1)
    ints = [int(Fr(c) * L) for c in v]
    g = reduce(gcd, [abs(c) for c in ints], 0) or 1
    return [Fr(c // g) for c in ints]


def _intgram(H):
    """Rescale a symmetric rational Gram matrix to a primitive integer one (same isotropy)."""
    k = len(H)
    L = reduce(lambda a, b: a * b // gcd(a, b), [Fr(H[i][j]).denominator for i in range(k) for j in range(k)], 1)
    M = [[int(Fr(H[i][j]) * L) for j in range(k)] for i in range(k)]
    g = reduce(gcd, [abs(c) for r in M for c in r], 0) or 1
    return [[Fr(c // g) for c in r] for r in M]


def _bits_det(H):
    """Bit length of |det H| (exact, Fraction Gaussian elimination)."""
    M = [list(map(Fr, r)) for r in H]
    k = len(M)
    det = Fr(1)
    for c in range(k):
        p = next((i for i in range(c, k) if M[i][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            M[c], M[p] = M[p], M[c]
            det = -det
        det *= M[c][c]
        for i in range(c + 1, k):
            f = M[i][c] / M[c][c]
            if f != 0:
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return max(abs(det.numerator).bit_length(), abs(det.denominator).bit_length())


MAX_DET_BITS = 90      # skip qfsolve when |disc| is bigger (PARI must factor it and can hang)


def _gram_on_basis(G, basis):
    k = len(basis)
    return [[_bil(G, basis[i], basis[j]) for j in range(k)] for i in range(k)]


def _solve_once(nodes, rng):
    n = len(nodes)
    w = weights(nodes)
    G = [[w[i] if i == j else Fr(0) for j in range(n)] for i in range(n)]
    a0 = qfsolve(G)
    if a0 is None:
        STATS["deg2_no_isotropic"] += 1
        return None
    y0 = _random_isotropic_from(G, a0, rng)
    y2 = _random_isotropic_from(G, a0, rng)
    t = _bil(G, y0, y2)
    if t == 0:
        STATS["deg2_t_zero"] += 1
        return None
    rows = [[G[i][i] * y0[i] for i in range(n)], [G[i][i] * y2[i] for i in range(n)]]
    W = kernel_basis(rows, n)                              # dim n-2
    H = _gram_on_basis(G, W)
    k = len(W)
    big = [[H[i][j] if i < k and j < k else Fr(0) for j in range(k + 1)] for i in range(k + 1)]
    big[k][k] = 2 * t
    sol = qfsolve(big)
    if sol is None:
        STATS["deg2_no_y1"] += 1
        return None
    s = sol[k]
    if s == 0:
        STATS["deg2_s_zero"] += 1
        return None
    y1 = [sum(sol[i] / s * W[i][c] for i in range(k)) for c in range(n)]
    # stage-boundary invariants (exact):
    F = lambda u: _bil(G, u, u)
    B = lambda u, v: _bil(G, u, v)
    assert F(y0) == 0 and F(y2) == 0, "y0/y2 not isotropic"
    assert B(y0, y1) == 0 and B(y1, y2) == 0, "y1 not in {y0,y2}^perp"
    assert F(y1) + 2 * B(y0, y2) == 0, "m^2 coefficient of F(Y(m)) nonzero"
    return y0, y1, y2


# ----------------------------------------------------------------------------- n = 7
def _Bf(G, u, v):
    n = len(u)
    return sum(G[i][j] * u[i] * v[j] for i in range(n) for j in range(n) if G[i][j] != 0)


def _independent(vs):
    return len(kernel_basis([list(r) for r in zip(*vs)], len(vs))) == 0


def _perp_mod(G, A):
    """Basis E of A^perp / A (A = independent isotropic vectors, pairwise orthogonal)."""
    n = len(G)
    K = kernel_basis([[sum(G[i][j] * a[j] for j in range(n)) for i in range(n)] for a in A], n)
    cur = list(A)
    E = []
    for k in K:
        if _independent(cur + [k]):
            cur.append(k)
            E.append(_prim(k))
    assert len(E) == n - 2 * len(A), "perp/quotient dimension mismatch"
    return E


def _isotropic_flag(G, rng):
    """[v1, v2, v3] spanning a random totally isotropic 3-space of the 7-dim form G, or None."""
    n = len(G)
    A = []
    for step in range(3):
        if not A:
            H, E = G, [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
        else:
            E = _perp_mod(G, A)
            H = [[_Bf(G, E[i], E[j]) for j in range(len(E))] for i in range(len(E))]
        H = _intgram(H)
        if _bits_det(H) > 4 * MAX_DET_BITS:
            STATS["deg2n7_flag_det_too_big"] += 1
            return "skip"
        c = qfsolve(H)
        if c is None:
            STATS["deg2n7_flag_step%d_fail" % (step + 1)] += 1
            return None
        c = _random_iso_full(H, c, rng, span=2)
        v = _prim([sum(c[i] * E[i][t] for i in range(len(E))) for t in range(n)])
        A.append(v)
    for i in range(3):
        for j in range(i, 3):
            assert _Bf(G, A[i], A[j]) == 0, "flag not totally isotropic"
    assert _independent(A), "flag vectors dependent"
    return A


def _conic_basis_in_V(Q0, rng):
    """In coordinates of a 3-dim V with Gram Q0 (3x3): (c0, c1, c2) with Gram
    [[0,0,t],[0,d,0],[t,0,0]], t != 0, d + 2 t = 0 after scaling.  None if Q0 is degenerate/anisotropic."""
    def q(u, v):
        return _Bf(Q0, u, v)
    Q0 = _intgram(Q0)
    if _bits_det(Q0) > MAX_DET_BITS:
        STATS["deg2n7_Q0_det_too_big"] += 1
        return None
    c = qfsolve(Q0)
    if c is None:
        STATS["deg2n7_F0_anisotropic_on_V"] += 1
        return None
    c = _random_iso_full(Q0, c, rng, span=2)
    def prop(u, v):                                   # u, v parallel?
        return all(u[i] * v[j] == u[j] * v[i] for i in range(3) for j in range(3))
    Kc = kernel_basis([[sum(Q0[i][j] * c[j] for j in range(3)) for i in range(3)]], 3)
    cands = [Kc[0], Kc[1], [a + b for a, b in zip(Kc[0], Kc[1])]]
    y1 = next((k for k in cands if not prop(k, c)), None)
    if y1 is None:
        return None
    d1 = q(y1, y1)
    if d1 == 0:
        STATS["deg2n7_F0_degenerate_on_V"] += 1
        return None
    Ky = kernel_basis([[sum(Q0[i][j] * y1[j] for j in range(3)) for i in range(3)]], 3)
    cands = [Ky[0], Ky[1], [a + b for a, b in zip(Ky[0], Ky[1])]]
    e = next((k for k in cands if not prop(k, c)), None)
    if e is None or q(c, e) == 0:
        STATS["deg2n7_bad_e"] += 1
        return None
    y2 = [e[i] - q(e, e) / (2 * q(c, e)) * c[i] for i in range(3)]       # isotropic, in y1^perp
    t = q(c, y2)
    if t == 0:
        return None
    kappa = -d1 / (2 * t)
    y2 = [kappa * v for v in y2]
    assert q(c, c) == 0 and q(y2, y2) == 0 and q(c, y1) == 0 and q(y1, y2) == 0
    assert q(y1, y1) + 2 * q(c, y2) == 0
    return c, y1, y2


def _solve_n7(nodes, mu0, rng, flags_per_form=10):
    n = 7
    w = weights(nodes)
    D = [w[j] * (Fr(nodes[j]) - mu0) for j in range(n)]
    if any(d == 0 for d in D):
        return None
    G = _intgram([[D[i] if i == j else Fr(0) for j in range(n)] for i in range(n)])
    G0 = [[w[i] if i == j else Fr(0) for j in range(n)] for i in range(n)]
    G1 = [[w[i] * Fr(nodes[i]) if i == j else Fr(0) for j in range(n)] for i in range(n)]
    for _ in range(flags_per_form):
        V = _isotropic_flag(G, rng)
        if V is None:
            return None                              # Witt index < 3: a property of (nodes, mu0), stop
        if V == "skip":
            continue
        Q0 = [[_Bf(G0, V[i], V[j]) for j in range(3)] for i in range(3)]
        res = _conic_basis_in_V(Q0, rng)
        if res is None:
            continue
        ys = [[sum(cc[i] * V[i][t] for i in range(3)) for t in range(n)] for cc in res]
        y0, y1, y2 = ys
        for Ge in (G0, G1):                            # stage-boundary invariants, both forms, exact
            assert _Bf(Ge, y0, y0) == 0 and _Bf(Ge, y2, y2) == 0, "y0/y2 not isotropic"
            assert _Bf(Ge, y0, y1) == 0 and _Bf(Ge, y1, y2) == 0, "y1 not orthogonal"
            assert _Bf(Ge, y1, y1) + 2 * _Bf(Ge, y0, y2) == 0, "m^2 coefficient nonzero"
        return y0, y1, y2
    STATS["deg2n7_no_F0_conic_V"] += 1
    return None


def generate_ydeg2_n7(seed=1, node_range=8, max_tries=3000, verbose=True):
    rng = random.Random(seed)
    pool = list(range(-node_range, node_range + 1))
    for it in range(max_tries):
        nodes = sorted(rng.sample(pool, 7))
        mu0 = Fr(rng.randint(-4 * node_range, 4 * node_range), rng.choice([1, 2, 3, 4]))
        if mu0 in nodes:
            continue
        res = _solve_n7(nodes, mu0, rng)
        if verbose and it % 50 == 49:
            print(f"[gen_sections_deg2] n=7: {it + 1} (nodes, mu0) tried, reasons={dict(STATS)}")
        if res is None:
            continue
        y0, y1, y2 = clear_denoms(list(res))
        coeffs = [[y0[j], y1[j], y2[j]] for j in range(7)]
        if any(all(c == 0 for c in cj) for cj in coeffs):
            STATS["deg2_zero_section"] += 1
            continue
        if all(cj[1] == 0 and cj[2] == 0 for cj in coeffs):
            STATS["deg2_constant"] += 1
            continue
        nodesF = [Fr(x) for x in nodes]
        if verify_ydeg2(nodesF, coeffs):
            if verbose:
                print(f"[gen_sections_deg2] n=7: nodes={nodes}, mu0={mu0}, after {it + 1} tries")
            return nodesF, coeffs
        STATS["deg2_verify_failed"] += 1
    raise RuntimeError(f"generate_ydeg2(7): nothing in {max_tries} tries; reasons={dict(STATS)}")



# ----------------------------------------------------------------------------- n = 8 (symmetric nodes)
def generate_ydeg2_n8_sym(seed=1, node_range=14, max_tries=4000, verbose=True):
    """
    n = 8 with SYMMETRIC nodes {+-a_1, .., +-a_4} and Y_{-a_k}(m) = Y_{a_k}(m).

    Then h(-x) and h(x) agree at all 8 nodes, so h is EVEN: h = H(z; m), z = x^2, H quadratic in z.
    The 8 value conditions collapse to ONE form on Q^4 in the z-nodes z_k = a_k^2:
            F(Y) = sum_k W_k Y_k^2,   W_k = 1 / prod_{i != k} (z_k - z_i)     (zero iff deg_z H <= 2).
    Y(m) = y0 + y1 m + y2 m^2 with F(Y(m)) = 0 for all m is exactly the n = 6 construction
    (_solve_once) run on the 4 nodes z_k; y_k are shared by the pair (+a_k, -a_k).

    CAVEAT (important for a *baseline*): this family is NOT generic.  h even means the surface carries the
    extra involution x -> -x, and it is a quadratic base change of a simpler fibration.  Ranks / rho may
    exceed the generic count; compare with n = 7 only with that in mind.
    """
    rng = random.Random(seed)
    pool = list(range(1, node_range + 1))
    for it in range(max_tries):
        a = sorted(rng.sample(pool, 4))
        z = [Fr(t * t) for t in a]
        res = _solve_once(z, rng)
        if verbose and it % 50 == 49:
            print(f"[gen_sections_deg2] n=8 sym: {it + 1} node sets tried, reasons={dict(STATS)}")
        if res is None:
            continue
        y0, y1, y2 = clear_denoms(list(res))
        per = [[y0[k], y1[k], y2[k]] for k in range(4)]
        if any(all(c == 0 for c in cj) for cj in per):
            STATS["deg2_zero_section"] += 1
            continue
        if any(cj[1] == 0 and cj[2] == 0 for cj in per):
            STATS["deg2_constant_pair"] += 1
            continue
        pairs = sorted([(Fr(-t), per[k]) for k, t in enumerate(a)] + [(Fr(t), per[k]) for k, t in enumerate(a)],
                       key=lambda e: e[0])
        nodes = [e[0] for e in pairs]
        coeffs = [list(e[1]) for e in pairs]
        if verify_ydeg2(nodes, coeffs):
            if verbose:
                print(f"[gen_sections_deg2] n=8 sym: a={a} found after {it + 1} node sets")
            return nodes, coeffs
        STATS["deg2_verify_failed"] += 1
    raise RuntimeError(f"generate_ydeg2(8): nothing in {max_tries} tries; reasons={dict(STATS)}")


def has_x_involution(nodes, coeffs):
    """True iff h(c - x; m) = h(x; m) for some constant c (translation-reflection symmetry of the quartic).
    Such an involution has fixed points on the generic fibre, acts as -1 there, and makes P and its mirror
    image dependent -- the same rank-halving seen in the even (c = 0) family."""
    n = len(nodes)
    sq = [[sum(cj[i] * cj[k - i] for i in range(3) if 0 <= k - i <= 2) for cj in coeffs] for k in range(5)]
    hs = [interpolate(nodes, sq[k]) for k in range(5)]
    pts = [Fr(t) for t in (0, 1, 2, 3, 5, 7)]
    for c in sorted({Fr(nodes[i]) + Fr(nodes[j]) for i in range(n) for j in range(i, n)}):
        if all(poly_eval(hk, c - x) == poly_eval(hk, x) for hk in hs for x in pts):
            return True
    return False


# ----------------------------------------------------------------------------- n = 8 (base change)
def generate_ydeg2_n8_basechange(seed=1, verbose=True):
    """
    n = 8, NON-symmetric nodes, x_j constant, Y_j(t) of degree <= 2 in t, by QUADRATIC BASE CHANGE of a
    rank-8 rational elliptic surface.

    gen_sections.generate(8, method='evenodd') gives h(x; m) = u(x) m^2 + v(x) (chi = 1) with 8 sections
    (x_j, a_j m + b_j).  Substituting m = phi(t) = c2 t^2 + c1 t + c0 gives
            h(x; phi(t)),      Y_j(t) = a_j phi(t) + b_j = [a_j c0 + b_j,  a_j c1,  a_j c2],
    a K3 (chi = 2) with the same 8 nodes.  Pullback of sections is injective on Mordell-Weil, so the 8
    sections are independent iff they were on the rational surface.  Everything is verified exactly.

    CAVEAT: this family is NOT generic.  The K3 is a degree-4 base change of y^2 = u nu + v
    (nu = phi(t)^2), so it has extra automorphisms and rho may be larger than 2 + Sigma + 8.
    Sections at the four 'alpha' nodes are constant in t.
    """
    from gen_sections import generate
    import os
    rng = random.Random(seed)
    use_conic = os.environ.get("BASE_N8_SRC", "conic") == "conic"      # conic: cross term B != 0 (gen_sections_conic.py)
    for k in range(200):                 # skip rational surfaces with an x-involution (they give rank 4, not 8)
        if use_conic:
            from gen_sections_conic import generate_conic
            nodes, a, b = generate_conic(seed + k, verbose=False)
        else:
            nodes, a, b = generate(8, seed=seed + k, method="evenodd", verbose=False)
        if not has_x_involution([Fr(x) for x in nodes], [[Fr(bj), Fr(aj), Fr(0)] for aj, bj in zip(a, b)]):
            break
        STATS["x_involution_rejected"] += 1
    else:
        raise RuntimeError("generate_ydeg2(8, basechange): every evenodd solution had an x-involution")
    for _ in range(100):
        c2 = rng.choice([1, 2, 3, 5, 6, -1, -2, -3])
        c1 = rng.choice([1, 2, 3, -1, -2, -3])
        c0 = rng.randint(-3, 3)
        coeffs = [[Fr(aj * c0 + bj), Fr(aj * c1), Fr(aj * c2)] for aj, bj in zip(a, b)]
        if any(all(c == 0 for c in cj) for cj in coeffs):
            continue
        if not verify_ydeg2([Fr(x) for x in nodes], coeffs):
            STATS["deg2_verify_failed"] += 1
            continue
        if verbose:
            print(f"[gen_sections_deg2] n=8 basechange: phi(t) = {c2} t^2 + {c1} t + {c0}")
        return [Fr(x) for x in nodes], coeffs
    raise RuntimeError(f"generate_ydeg2(8, basechange): no usable phi; reasons={dict(STATS)}")


def generate_ydeg2(n=6, seed=1, node_range=8, max_tries=2000, verbose=True):
    """Return (nodes, coeffs): coeffs[j] = [c0, c1, c2], Y_j(m) = c0 + c1 m + c2 m^2."""
    if n == 7:
        return generate_ydeg2_n7(seed=seed, node_range=node_range, verbose=verbose)
    if n == 8:
        import os
        if os.environ.get("BASE_N8", "basechange") == "sym":     # even-h family: only rank 4 (see docstring)
            return generate_ydeg2_n8_sym(seed=seed, verbose=verbose)
        return generate_ydeg2_n8_basechange(seed=seed, verbose=verbose)
    if n != 6:
        raise NotImplementedError("generate_ydeg2: n = 6, 7 or 8 (8 = symmetric/even family only)")
    rng = random.Random(seed)
    pool = list(range(-node_range, node_range + 1))
    for it in range(max_tries):
        nodes = sorted(rng.sample(pool, n))
        res = _solve_once(nodes, rng)
        if res is None:
            continue
        y0, y1, y2 = res
        y0, y1, y2 = clear_denoms([y0, y1, y2])
        coeffs = [[y0[j], y1[j], y2[j]] for j in range(n)]
        if any(all(c == 0 for c in cj) for cj in coeffs):
            STATS["deg2_zero_section"] += 1
            continue
        if all(cj[1] == 0 and cj[2] == 0 for cj in coeffs):
            STATS["deg2_constant"] += 1
            continue
        if verify_ydeg2([Fr(x) for x in nodes], coeffs):
            if verbose:
                print(f"[gen_sections_deg2] n={n}: nodes={nodes} found after {it + 1} node sets")
            return [Fr(x) for x in nodes], coeffs
        STATS["deg2_verify_failed"] += 1
    raise RuntimeError(f"generate_ydeg2: nothing in {max_tries} tries; reasons={dict(STATS)}")


def verify_ydeg2(nodes, coeffs):
    """Exact: the 5 m-coefficient interpolants all have degree <= 4 and reproduce Y_j^2."""
    n = len(nodes)
    sq = []                                                # sq[k][j] = coeff of m^k in Y_j^2
    for k in range(5):
        row = []
        for cj in coeffs:
            row.append(sum(cj[i] * cj[k - i] for i in range(3) if 0 <= k - i <= 2))
        sq.append(row)
    for k in range(5):
        hk = interpolate(nodes, sq[k])
        if len(hk) > 5:
            return False
        for j, xj in enumerate(nodes):
            if poly_eval(hk, Fr(xj)) != sq[k][j]:
                return False
    return True


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    nodes, coeffs = generate_ydeg2(n, seed)
    for x, c in zip(nodes, coeffs):
        print(f"x={x}: ({c[0]}) + ({c[1]})*m + ({c[2]})*m^2")
    print("verified:", verify_ydeg2(nodes, coeffs))
