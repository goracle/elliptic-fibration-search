"""
rational_arithmetic.py: Core number theory utilities.
"""
import math
from .search_config import gcd, lru_cache, RationalReconstructionError, DEFAULT_MAX_CACHE_SIZE, floor, sqrt, QQ, crt, Integer

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def crt_cached(residues, moduli):
    """
    Cached Chinese Remainder Theorem computation.

    residues/moduli arrive here as plain Python ints (they're built up via
    native int arithmetic -- M * q, etc. -- all over residue_crt_graph.py,
    never wrapped back into Sage Integers). Sage's crt() -> CRT_list ->
    xgcd() calls .xgcd() directly on its arguments rather than coercing
    them first, so a plain int blows up with
    "AttributeError: 'int' object has no attribute 'xgcd'" the moment
    XGCD hits a native-int/native-int pair. Wrap in Integer(...) here,
    once, at the single chokepoint every caller already goes through,
    instead of hunting down every construction site.
    """
    return crt([Integer(r) for r in residues], [Integer(m) for m in moduli])

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def rational_reconstruct(c, N, max_den=None):
    """
    Rational reconstruction using the Extended Euclidean Algorithm.
    Given integers c and N > 0, finds a rational number a/b such that
    a/b ≡ c (mod N), with |a| and |b| bounded.
    """
    if max_den is None:
        max_den = floor(sqrt(N / QQ(2)))

    c = c % N
    if c == 0: return 0, 1
    if c == 1 and max_den >= 1: return 1, 1

    # Standard Extended Euclidean Algorithm setup
    r0, r1 = N, c
    t0, t1 = 0, 1

    while r1 != 0:
        # Check denominator bound before next iteration
        if abs(t1) > max_den:
             # We've overshot the bound.
             a, b = r0, t0
             break

        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    else:
        # Loop finished because r1 == 0.
        a, b = r0, t0

    # Final checks on the result (a, b)
    if abs(b) > max_den or b == 0:
        raise RationalReconstructionError(f"No reconstruction for c={c}, N={N}, max_den={max_den}")

    if b < 0:
        a, b = -a, -b

    if (a - c * b) % N != 0:
        raise RationalReconstructionError(f"Validation failed for c={c}, N={N}: got a={a}, b={b}")

    g = gcd(abs(a), abs(b))
    return int(a // g), int(b // g)

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def lattice_rational_lift_exists(c, M, H):
    """
    CRT-then-lattice-reduction small-rational test.

    Decide whether there EXISTS a rational r/s with |r| <= H and |s| <= H
    such that r - c*s == k*M for some integer k -- i.e. whether the
    residue class c (mod M) can contain a small-height rational at all.

    This is the two-integer lattice problem from the "CRT then lattice
    reduction" scheme: r - c*s is a vector in the 2D lattice
        L = { (r,s) : r - c*s in M*Z }
    spanned by (M, 0) and (c, 1). We want to know if L contains a nonzero
    point inside the box [-H,H] x [-H,H] (nonzero because (0,0) is the
    trivial/uninformative solution). The standard tool for this is the
    Euclidean algorithm on (M, c), which for a rank-2 lattice IS lattice
    reduction (it's the 2D case of LLL): each convergent (r_i, s_i) of the
    continued fraction expansion of c/M is, up to the usual optimality
    theorem for rational approximation, the shortest lattice vector once
    |r_i| drops below the previous |r_{i-1}|. So unlike rational_reconstruct
    (which only bounds s and lets r float free), this walks the full
    Euclidean chain and checks EVERY convergent's (r, s) pair against BOTH
    bounds, and reports failure unless some convergent satisfies both.

    Contrast with the old rational_reconstruct(c, M, max_den=H) sieve: that
    call only enforces |s| <= H and then accepts whatever |r| < M falls out
    of the terminating step -- so whenever M itself is small (which it
    always is for a 2-3 small-prime CRT modulus), nearly every residue
    passes trivially, because "some numerator under M" is not a real
    constraint. This function is only meaningful, i.e. only actually
    rejects a nontrivial fraction of residues, once M is large enough that
    a generic class mod M has no representative with BOTH |r|,|s| <= H --
    which requires M to comfortably exceed H (2*H^2 is the standard bound
    for guaranteed uniqueness of a small-height representative, but the
    filter is still informative, just not exhaustive/unique, well below
    that -- see require_informative_modulus below for the recommended
    minimum to bother calling this at all).

    Returns True iff some convergent (r, s), including the endpoints
    (M, 0) and (c mod M, 1), satisfies |r| <= H and |s| <= H and r != 0
    (excluding only the fully-degenerate all-zero case, which can't arise
    here since r0 starts at M > 0).

    Both r and s are checked -- this is the fix for the numerator gap in
    rational_reconstruct.
    """
    if M <= 0:
        raise ValueError(f"lattice_rational_lift_exists requires M > 0, got M={M}")
    if H < 0:
        raise ValueError(f"lattice_rational_lift_exists requires H >= 0, got H={H}")

    c = int(c) % int(M)
    M = int(M)
    H = int(H)

    # Continued-fraction / Euclidean chain on (M, c): (r0,s0)=(M,0),
    # (r1,s1)=(c,1), r_{i+1} = r_{i-1} - q*r_i, s_{i+1} = s_{i-1} - q*s_i.
    # Every (r_i, s_i) satisfies r_i - c*s_i ≡ 0 (mod M) by construction,
    # and |r_i| is strictly decreasing while |s_i| is strictly increasing,
    # so this sweeps out exactly the lattice's successive minima -- we only
    # need to test each one against the box, not search further.
    r_prev, r_cur = M, c
    s_prev, s_cur = 0, 1

    if r_prev <= H and abs(s_prev) <= H:
        return True  # (M, 0): trivially the "k*M" solution; only relevant if H >= M

    while r_cur != 0:
        if abs(r_cur) <= H and abs(s_cur) <= H:
            return True
        q = r_prev // r_cur
        r_prev, r_cur = r_cur, r_prev - q * r_cur
        s_prev, s_cur = s_cur, s_prev - q * s_cur

    # The loop above stops the moment r_cur == 0, so it never tests the LAST
    # Euclid vector (0, s_cur). That vector is a perfectly good nonzero
    # lattice point (0 - c*s_cur = -c*s_cur is a multiple of M): it is the
    # small rational 0/s. In particular for c == 0 (the true point m = 0)
    # the very first (r_cur, s_cur) is (0, 1) and, before this check, was
    # skipped, so the lift was reported as impossible for every M > H and a
    # chain following the true point m = 0 died as soon as M exceeded H.
    # (Found by brute force: every true point a/b with a != 0 was already
    # accepted; every rejected true point had a == 0.)
    if abs(s_cur) <= H:
        return True
    return False


def lattice_small_rational(c, M, H):
    """
    Return the small rational a/b (as a pair (a, b), b > 0, gcd(a, b) = 1)
    with |a|, |b| <= H, gcd(b, M) = 1 and a == c*b (mod M), or None.

    Same Euclidean walk as lattice_rational_lift_exists, but it returns the
    vector instead of a bool and insists gcd(b, M) = 1. That second condition
    matters: a lattice vector (r, s) whose s shares a factor with M makes the
    plain bool test pass, but r/s does NOT reduce to c modulo that shared
    prime, so it is not a lift of the CRT residue at all.

    UNIQUENESS: if M > 2*H^2 there is at most one such fraction (two of them,
    a1/b1 != a2/b2, would give a nonzero integer a1*b2 - a2*b1 of size
    <= 2*H^2 divisible by M, impossible). So for M > 2*H^2 this is THE
    candidate rational, which is what lets the caller verify it against the
    other primes and stop the chain.
    """
    if M <= 0:
        raise ValueError(f"lattice_small_rational requires M > 0, got M={M}")
    c, M, H = int(c) % int(M), int(M), int(H)

    def _accept(r, s):
        if s < 0:
            r, s = -r, -s
        if s == 0 or s > H or abs(r) > H or gcd(s, M) != 1:
            return None
        g = gcd(abs(r), s)
        return (r // g, s // g)

    r_prev, r_cur = M, c
    s_prev, s_cur = 0, 1
    while True:
        hit = _accept(r_cur, s_cur)         # includes the final (0, s) vector
        if hit is not None:
            return hit
        if r_cur == 0:
            return None
        q = r_prev // r_cur
        r_prev, r_cur = r_cur, r_prev - q * r_cur
        s_prev, s_cur = s_cur, s_prev - q * s_cur


def square_den_small_rationals(c, M, H):
    """
    All reduced fractions a/d^2 (d >= 1, d^2 <= H, |a| <= H, gcd(a, d) = 1,
    gcd(d, M) = 1) with a == c*d^2 (mod M), as a sorted list of (a, d*d).

    This is the square-denominator analogue of lattice_small_rational.  It is
    EXHAUSTIVE (one modular multiply per d <= sqrt(H)), so unlike the
    Euclidean walk it also finds hits that are not lattice minimal points,
    and it does not need M > 2*H^2 for the answer to be the complete list.

    Valid only when every true m is known to have the form a/d^2 (monic
    integral curve, RLINEAR, integral xi and shift; see crt_bounds.py).
    gcd(a, d) = 1 forces the reduced denominator to be exactly d^2, and
    gcd(d, M) = 1 discards d divisible by a prime of the clique (a true m
    with p | d has no finite residue at p, so it is never a node there).
    """
    M, H = int(M), int(H)
    if M <= 0:
        raise ValueError(f"square_den_small_rationals requires M > 0, got M={M}")
    c %= M
    half = M // 2
    out = []
    for d in range(1, math.isqrt(H) + 1):
        s = d * d
        a = (c * s) % M
        if a > half:
            a -= M
        if abs(a) > H:
            continue
        if math.gcd(abs(a), d) != 1 or math.gcd(d, M) != 1:
            continue
        out.append((a, s))
    return out


def lattice_square_den_lift_exists(c, M, H):
    """
    Like lattice_rational_lift_exists, but the denominator must be a perfect
    square d^2 <= H (and the fraction reduced, coprime to M).  Strictly
    stronger than the plain test; used only when square denominators are
    justified (see crt_bounds.py).

    Fast paths: M <= 2H+1 -> d = 1 always works (centered residue fits);
    otherwise the O(log) convergent test is a necessary condition (a hit is
    a box vector, and the convergents contain a dominating vector for every
    box vector), so most classes are rejected before the O(sqrt(H)) scan.
    """
    if M <= 0:
        raise ValueError(f"lattice_square_den_lift_exists requires M > 0, got M={M}")
    M, H = int(M), int(H)
    c = int(c) % M
    if M <= 2 * H + 1:
        return True
    if not lattice_rational_lift_exists(c, M, H):
        return False
    return bool(square_den_small_rationals(c, M, H))


def modulus_is_informative(M, H):
    """
    Whether M is large enough for lattice_rational_lift_exists(., M, H) to
    be a *meaningful* filter rather than a near-tautology.

    The box [-H,H]x[-H,H] contains ~(2H+1)^2 lattice points but there are only
    M residue classes, so by pigeonhole most classes have a representative in
    the box for free once (2H+1)^2 >> M, independent of any number-theoretic
    structure. Returns True once M > H, i.e. once "found a small lift" is
    doing real work. Callers wanting the stronger *uniqueness* guarantee
    should instead require M > 2*H*H, the classical rational-reconstruction
    bound.
    """
    return M > H


def find_minimal_abs_representative(t_mod_Q, Q, T):
    """
    Find if there exists k such that |t_mod_Q + k*Q| <= T
    Returns True if such k exists, False otherwise.

    Exact: the k minimizing |t + k*Q| is floor(-t/Q) or that plus one, so test
    those two with integer floor division rather than a float estimate (a float
    k loses precision once |k| > 2^53, which happens for CRT-sized Q).
    """
    if Q == 0:
        return abs(t_mod_Q) <= T

    k_floor = (-t_mod_Q) // Q
    return any(abs(t_mod_Q + k * Q) <= T for k in (k_floor, k_floor + 1))

def assert_base_m_found(base_m, expected_x, r_m_callable, shift, T=None):
    """
    Ensure that x = T^-1(r_m(base_m)) - shift equals expected_x.
    This checks that the base point (mtest, xtest) relationship is respected
    by the parametrization, handling the global shift and optional Mobius transform T.

    r_m_callable(m) returns the x-coordinate (x'') on the most-transformed curve.
    If T is present, the shifted x-coordinate is T^-1(x'').
    Then the original x-coordinate is x_shifted - shift.

    Always raises AssertionError on failure (the old allow_raise=False mode,
    which returned False instead, had no callers).
    """
    assert base_m is not None, "assert_base_m_found requires a base_m (rational) to check"

    try:
        # r_m_callable(m=QQ(base_m)) evaluates to the final transformed x-coordinate (x'')
        x_final_transformed = r_m_callable(m=QQ(base_m))
    except Exception as e:
        raise AssertionError(
            f"assert_base_m_found: r_m_callable evaluation failed at m,shift={base_m},{shift}: {e}"
        ) from e

    # 1. Apply Inverse Mobius transform T^-1 to get the shifted x-coordinate (x')
    if T is not None:
        try:
            # We must use inverse_transform, as T maps x' -> x''
            x_shifted = T.inverse_transform(x_final_transformed)
        except Exception as e:
            raise AssertionError(
                f"assert_base_m_found: Mobius inverse transform failed at x={x_final_transformed}: {e}"
            ) from e
    else:
        # If no T, x_final_transformed is x'
        x_shifted = x_final_transformed

    # 2. Subtract shift to get the original x-coordinate (x_orig)
    # Since x_shifted = x_orig + shift, we have x_orig = x_shifted - shift.
    x_orig = x_shifted - shift

    try:
        x_orig_q = QQ(x_orig)
        expected_x_q = QQ(expected_x)
    except Exception as e:
        raise AssertionError("assert_base_m_found: coercion to QQ failed") from e

    if x_orig_q != expected_x_q:
        raise AssertionError(
            f"assert_base_m_found: mismatch. m={base_m} expected x={expected_x} got x={x_orig}"
        )
    return True
