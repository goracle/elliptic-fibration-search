"""
rational_arithmetic.py: Core number theory utilities.
"""
from .search_config import (
    gcd, lru_cache, RationalReconstructionError, DEFAULT_MAX_CACHE_SIZE,
    floor, sqrt, QQ, crt
)

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def crt_cached(residues, moduli):
    """Cached Chinese Remainder Theorem computation."""
    return crt(list(residues), list(moduli))

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


def find_minimal_abs_representative(t_mod_Q, Q, T):
    """
    Find if there exists k such that |t_mod_Q + k*Q| <= T
    Returns True if such k exists, False otherwise.
    """
    if Q == 0:
        return abs(t_mod_Q) <= T
    
    k_opt_float = -t_mod_Q / Q
    k_candidates = [int(k_opt_float), int(k_opt_float) + 1, 0]
    
    for k in k_candidates:
        t = t_mod_Q + k * Q
        if abs(t) <= T:
            return True
    return False


def assert_base_m_found(base_m, expected_x, r_m_callable, shift, allow_raise=True):
    """
    Ensure that x = r_m(base_m) - shift equals expected_x.
    This checks that the base point (mtest, xtest) relationship is respected
    by the parametrization. It does not scan through newly_found_x; instead
    it asserts consistency between r_m and the supplied base point.
    """
    assert base_m is not None, "assert_base_m_found requires a base_m (rational) to check"
    try:
        x_base = r_m_callable(m=QQ(base_m)) - shift
    except Exception as e:
        msg = f"assert_base_m_found: r_m_callable evaluation failed at m,shift={base_m},{shift}: {e}"
        if allow_raise:
            raise AssertionError(msg)
        return False

    try:
        x_base_q = QQ(x_base)
        expected_x_q = QQ(expected_x)
    except Exception:
        msg = "assert_base_m_found: coercion to QQ failed"
        if allow_raise:
            raise AssertionError(msg)
        return False

    if x_base_q == expected_x_q:
        return True

    msg = (f"assert_base_m_found: mismatch.\n"
           f"  m = {base_m}\n"
           f"  expected x = {expected_x_q}\n"
           f"  got x = {x_base_q}")
    if allow_raise:
        raise AssertionError(msg)
    return False

