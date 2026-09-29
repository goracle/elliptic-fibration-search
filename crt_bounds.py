"""
crt_bounds.py -- principled CRT-modulus bounds for the residue-clique search.

Pure Python (no Sage), so brauer.py, search_common.py and search_lll/ can all
import it without cycles.

Setting
-------
A candidate m is a rational a/s with |a| <= H and s <= H (s = the reduced
denominator).  A clique of primes S gives the class c mod M, M = prod(S).
The lift test asks whether c mod M contains such a rational.

Two numbers matter, and they are NOT the same thing:

  * confirmation threshold  T = margin * B(H)
        B(H) = number of (numerator, denominator) slots in the box.  A
        uniformly random class has a box representative with probability
        about B/M, so at M > T a false clique survives the lift test with
        probability <~ 1/margin.  This is what decides when a chain is
        "done"; the exact on-curve check afterwards does the real filtering.

  * rigorous uniqueness  M > 2*H^2.  Two distinct fractions in the box differ
        by a nonzero integer of size <= 2*H^2, so above this bound at most
        one fraction lies in the class.  (Weaker than T in practice, and it
        does not improve for square denominators: |a1*d2^2 - a2*d1^2| can
        still be ~ 2*H^2.)

Square denominators
-------------------
If the curve is y^2 = f(x) with f monic, integral and of ODD degree n, every
rational point has x = a/d^2 (y = c/d^n): write x = a/b in lowest terms, then
y^2 = N/b^n with gcd(N, b) = 1, so b^n is a square, so b is.  For even degree
(e.g. a monic sextic) b^n is always a square and nothing is forced.  With x = xi - m, xi an integer, the same
holds for m.  Then only about (2H+1)*isqrt(H) slots are occupied instead of
(2H+1)*H, i.e. the false-positive rate drops by ~sqrt(H) and T drops by the
same factor.  This needs: RLINEAR, no Mobius transform, xi and shift integral,
f monic integral.  (With a non-integral xi, 1/9 + 2/9 = 1/3 shows the
denominator of xi - x need not be a square.)  The caller must check that.
"""
import math


def box_slots(H, square_den=False):
    """(2A+1)*(2S+1) with A = H and S = H (general) or isqrt(H) (square dens).

    For square_den, S counts the root d of the denominator d^2 <= H.
    The general case is exactly (2H+1)^2, the box the code has always used.
    """
    H = int(H)
    if H < 0:
        raise ValueError("H must be >= 0")
    S = math.isqrt(H) if square_den else H
    return (2 * H + 1) * (2 * S + 1)


def informative_threshold(H, margin=15, square_den=False):
    """Modulus above which a surviving clique is confirmed: margin * box_slots."""
    return int(margin) * box_slots(H, square_den)


def uniqueness_threshold(H):
    """Rigorous: M > 2*H^2 => at most one fraction a/s (|a|,s <= H) per class."""
    H = int(H)
    return 2 * H * H


def clique_size_bounds(prime_pool, H, margin=15, square_den=False):
    """
    How many primes a clique needs to clear informative_threshold.

    Returns dict:
      threshold : the modulus bound
      k_worst   : smallest k such that ANY k pool primes clear it
                  (product of the k smallest primes) -- a chain of this size
                  is certain to be confirmed; None if the whole pool cannot.
      k_best    : smallest k such that the k LARGEST primes clear it -- the
                  fewest primes any clique can get away with; None likewise.
    """
    T = informative_threshold(H, margin, square_den)
    pool = sorted({int(p) for p in prime_pool})

    def count(order):
        prod = 1
        for i, p in enumerate(order, 1):
            prod *= p
            if prod > T:
                return i
        return None

    return {
        "threshold": T,
        "k_worst": count(pool),
        "k_best": count(reversed(pool)),
    }


def certified_height(M, margin=15, square_den=False):
    """Largest H whose informative_threshold is still < M (0 if none)."""
    M = int(M)
    if informative_threshold(1, margin, square_den) >= M:
        return 0
    lo, hi = 1, 2
    while informative_threshold(hi, margin, square_den) < M:
        lo, hi = hi, hi * 2
    while hi - lo > 1:               # invariant: threshold(lo) < M <= threshold(hi)
        mid = (lo + hi) // 2
        if informative_threshold(mid, margin, square_den) < M:
            lo = mid
        else:
            hi = mid
    return lo


def describe(prime_pool, H, margin=15, square_den=False):
    """One human-readable line for logs."""
    b = clique_size_bounds(prime_pool, H, margin, square_den)
    return (f"H={int(H)}, square_den={bool(square_den)}, margin={margin}: "
            f"confirm at M > {b['threshold']:,} "
            f"(~10^{math.log10(b['threshold']):.1f}); "
            f"clique size {b['k_best']}..{b['k_worst']} primes "
            f"(fewest with largest primes .. guaranteed with smallest); "
            f"rigorous uniqueness M > {uniqueness_threshold(H):,}")
