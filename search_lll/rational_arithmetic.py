from .search_config import gcd, lru_cache, RationalReconstructionError, DEFAULT_MAX_CACHE_SIZE, floor, sqrt, QQ, crt

"""
rational_arithmetic.py: Core number theory utilities.
"""

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def crt_cached(residues, moduli):
    """Cached Chinese Remainder Theorem computation."""
    return crt(list(residues), list(moduli))

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


def assert_base_m_found(base_m, expected_x, r_m_callable, shift, T=None, allow_raise=True):
    """
    Ensure that x = T^-1(r_m(base_m)) - shift equals expected_x.
    This checks that the base point (mtest, xtest) relationship is respected
    by the parametrization, handling the global shift and optional Mobius transform T.
    
    r_m_callable(m) returns the x-coordinate (x'') on the most-transformed curve.
    If T is present, the shifted x-coordinate is T^-1(x'').
    Then the original x-coordinate is x_shifted - shift.
    """
    assert base_m is not None, "assert_base_m_found requires a base_m (rational) to check"
    
    try:
        # r_m_callable(m=QQ(base_m)) evaluates to the final transformed x-coordinate (x'')
        x_final_transformed = r_m_callable(m=QQ(base_m))
    except Exception as e:
        msg = f"assert_base_m_found: r_m_callable evaluation failed at m,shift={base_m},{shift}: {e}"
        if allow_raise:
            raise AssertionError(msg)
        return False
        
    # 1. Apply Inverse Mobius transform T^-1 to get the shifted x-coordinate (x')
    if T is not None:
        try:
            # We must use inverse_transform, as T maps x' -> x''
            x_shifted = T.inverse_transform(x_final_transformed)
        except Exception as e:
             msg = f"assert_base_m_found: Mobius inverse transform failed at x={x_final_transformed}: {e}"
             if allow_raise:
                 raise AssertionError(msg)
             return False
    else:
        # If no T, x_final_transformed is x'
        x_shifted = x_final_transformed

    # 2. Subtract shift to get the original x-coordinate (x_orig)
    # Since x_shifted = x_orig + shift, we have x_orig = x_shifted - shift.
    x_orig = x_shifted - shift
    
    try:
        x_orig_q = QQ(x_orig)
        expected_x_q = QQ(expected_x)
    except Exception:
        msg = "assert_base_m_found: coercion to QQ failed"
        if allow_raise:
            raise AssertionError(msg)
        return False

    if x_orig_q != expected_x_q:
        msg = f"assert_base_m_found: mismatch. m={base_m} expected x={expected_x} got x={x_orig}"
        if allow_raise:
            raise AssertionError(msg)
        return False
    return True
