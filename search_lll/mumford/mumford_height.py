from sage.all import QQ, Matrix, RDF
from fractions import Fraction
import math
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL

def naive_height_safe(s, p, v0, v1, debug=DEBUG):
    """
    Compute naive height from Mumford representation without building Jacobian.
    Returns log(max(|coeffs of u|, |coeffs of v|)).
    """
    from fractions import Fraction
    import math
    
    # Force conversion to QQ first, then to Fraction
    s_qq = QQ(s)
    p_qq = QQ(p)
    v0_qq = QQ(v0)
    v1_qq = QQ(v1)
    
    # Convert QQ to Fraction using numerator/denominator
    s_frac = Fraction(int(s_qq.numerator()), int(s_qq.denominator()))
    p_frac = Fraction(int(p_qq.numerator()), int(p_qq.denominator()))
    v0_frac = Fraction(int(v0_qq.numerator()), int(v0_qq.denominator()))
    v1_frac = Fraction(int(v1_qq.numerator()), int(v1_qq.denominator()))
    
    # u(x) = x^2 - s*x + p has coefficients [1, -s, p]
    # v(x) = v1*x + v0 has coefficients [v1, v0]
    
    all_coeffs = [
        Fraction(1, 1),  # leading coeff of u
        -s_frac,
        p_frac,
        v1_frac,
        v0_frac
    ]
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))  # include denominator
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return float(math.log(max_abs))

def naive_height_exact(D):
    """
    Compute a naive height in exact rationals from a Mumford divisor D = [u, v].
    Returns a QQ number (log of max coefficient magnitude, exact).
    """
    from fractions import Fraction
    import math
    
    u_coeffs = D[0].list()
    v_coeffs = D[1].list()
    
    # Convert to exact fractions
    all_coeffs = []
    for c in u_coeffs + v_coeffs:
        c_qq = QQ(c)
        all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return QQ(math.log(max_abs))

def manual_naive_height(P):
    """
    Robust naive logarithmic height from Mumford u-polynomial.
    Returns float(log(max_abs)), always finite for valid input, else raises.
    """
    try:
        u = P[0]  # Mumford u polynomial
    except Exception:
        raise

    fracs = _extract_u_coeffs_as_fractions(u)
    # Convert to integer projective coordinates by clearing denominators
    dens = [f.denominator for f in fracs]
    L = 1
    for d in dens:
        L = (L * d) // math.gcd(L, d)

    int_coeffs = [int((f * L).numerator) for f in fracs]  # numerators after clearing denom
    # append the implicit leading coefficient L*1 (monic)
    int_coeffs.append(int(L))

    if not int_coeffs:
        return 0.0

    max_abs = max(abs(c) for c in int_coeffs)
    # defensive: ensure positive integer
    max_abs = max(1, int(max_abs))
    return math.log(max_abs)

def manual_canonical_height(P, limit=8, debug=DEBUG):
    """
    Approximate canonical height by computing h(2^n P)/4^n for n=0..limit and returning last value.
    Re-raises any exceptions from doubling. Prints intermediate heights if debug=True.
    """
    if P.is_zero():
        return 0.0

    Q = P
    vals = []
    try:
        for n in range(limit + 1):
            hQ = manual_naive_height(Q)
            vals.append(float(hQ) / (4.0 ** n))
            if debug:
                print(f"[canon] n={n} naive_h={hQ:.6g} ratio={vals[-1]:.6g}")
            Q = 2 * Q
    except Exception:
        # Re-raise after printing diagnostic if possible
        if debug:
            print("Doubling failed at step", n)
        raise

    # Return the last computed ratio; caller can examine intermediate vals if needed.
    return float(vals[-1])

def compute_manual_height_pairing(P, Q, limit=8, debug=DEBUG):
    """
    <P, Q> = 1/2 * (h_hat(P+Q) - h_hat(P) - h_hat(Q))
    Uses the manual canonical-height approximation.
    """
    try:
        if P.is_zero() or Q.is_zero():
            return float(0.0)

        # Use manual canonical height approximation for all three
        h_p = manual_canonical_height(P, limit=limit, debug=debug)
        h_q = manual_canonical_height(Q, limit=limit, debug=debug)
        h_sum = manual_canonical_height(P + Q, limit=limit, debug=debug)
        val = 0.5 * (h_sum - h_p - h_q)
        return float(val)
    except Exception:
        raise


def compute_height_pairing_simple(D1, D2, num_doublings=NUM_DOUBLINGS):
    """
    Compute <D1, D2> using LIMITED doublings to avoid coefficient explosion.
    Uses: <D1, D2> = (h(D1+D2) - h(D1) - h(D2)) / 2
    where h is naive height.
    
    Only does `num_doublings` iterations instead of 8.
    """
    from fractions import Fraction
    import math
    
    def naive_height_from_jacobian(D):
        u, v = D[0], D[1]
        u_coeffs = u.list()
        v_coeffs = v.list()
        
        all_coeffs = []
        for c in u_coeffs + v_coeffs:
            c_qq = QQ(c)
            all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
        
        # Clear denominators
        lcm_den = 1
        for f in all_coeffs:
            lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
        
        int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
        int_coeffs.append(int(lcm_den))
        
        max_abs = max(abs(c) for c in int_coeffs if c != 0)
        max_abs = max(1, max_abs)
        
        return float(math.log(max_abs))
    
    if D1.is_zero() or D2.is_zero():
        return 0.0
    
    # Compute heights with limited doublings
    vals = []
    P, Q, S = D1, D2, D1 + D2
    
    for n in range(num_doublings):
        hP = naive_height_from_jacobian(P)
        hQ = naive_height_from_jacobian(Q)
        hS = naive_height_from_jacobian(S)
        
        pairing = (hS - hP - hQ) / 2.0
        vals.append(pairing / (4.0 ** n))
        
        P = P + P
        Q = Q + Q
        S = S + S
    
    # Return the last value (most refined estimate)
    return vals[-1]


def compute_height_pairing_exact(D1, D2, f_coeffs, num_doublings=NUM_DOUBLINGS, primes_list=PRIME_POOL, debug=False):
    """
    Exact height pairing <D1, D2> using modular doubling to approximate canonical height.
    This replaces the slow D+D doubling with the robust CRT method.
    Returns a QQ number.
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    # 1. Compute D1+D2
    D_sum = D1 + D2

    # 2. Compute the final doubled points using the robust modular method
    P_final = compute_doubled_point_modular(D1, f_coeffs, num_doublings, primes_list, debug=debug)
    Q_final = compute_doubled_point_modular(D2, f_coeffs, num_doublings, primes_list, debug=debug)
    S_final = compute_doubled_point_modular(D_sum, f_coeffs, num_doublings, primes_list, debug=debug)

    # 3. Calculate the naive height of the final doubled points
    h_P_final = naive_height_exact(P_final)
    h_Q_final = naive_height_exact(Q_final)
    h_S_final = naive_height_exact(S_final)
    
    # 4. Apply the canonical height definition
    # h_hat(D) approx h(2^n D) / 4^n 
    scaling_factor = QQ(4**num_doublings)
    canonical_D1 = h_P_final / scaling_factor
    canonical_D2 = h_Q_final / scaling_factor
    canonical_D_sum = h_S_final / scaling_factor
    
    # 5. Compute the pairing
    pairing_value = (canonical_D_sum - canonical_D1 - canonical_D2) / QQ(2)
    
    return pairing_value

def _extract_u_coeffs_as_fractions(u):
    """
    Return list of coefficients of u (highest-to-lowest) as Fraction objects.
    Accepts:
      - Sage polynomial (use .list() or .coefficients?)
      - Python list/tuple of coeffs
      - tuple-like from mumford (already rational objects)
    Ensures monic by appending the implicit leading 1 if needed.
    """
    # If u is a Sage polynomial, try u.list() (coeffs lowest-first)
    try:
        if hasattr(u, 'list'):
            coeffs_low = u.list()    # lowest-degree first
            coeffs = list(reversed(coeffs_low))  # highest-first
        elif hasattr(u, 'coefficients'):
            coeffs = u.coefficients(sparse=False)
            # coefficients may not include zeros; try to detect degree
            if hasattr(u, 'degree') and u.degree() is not None:
                deg = u.degree()
                # create full list
                full = [QQ(0)] * (deg+1)
                for i, c in enumerate(u.coefficients(sparse=False)):
                    # this is fragile in some Sage versions; fallback below
                    pass
        else:
            # Fallback: treat u as an iterable of coeffs (highest-first)
            coeffs = list(u)
    except Exception:
        # Let exceptions bubble: user asked to raise them
        raise

    # Coerce each coefficient to Fraction robustly
    frac_coeffs = []
    for c in coeffs:
        # If it's a Sage rational (QQ), get numerator/denominator
        try:
            if hasattr(c, 'numerator') and hasattr(c, 'denominator'):
                n = int(c.numerator())
                d = int(c.denominator())
                frac_coeffs.append(Fraction(n, d))
            else:
                # For floats or RDF, convert via Fraction.from_float if necessary
                frac_coeffs.append(Fraction(c))
        except Exception:
            # last resort: try string conversion
            frac_coeffs.append(Fraction(str(c)))
            raise

    # Ensure monic: if leading coeff != 1, check if implicit monic (some code gives only lower terms)
    if not frac_coeffs:
        return [Fraction(1,1)]
    # If leading coeff equals 1, fine. If not, assume monic poly was given lacking leading 1:
    if frac_coeffs[0] != 1:
        # If the polynomial *is* monic but the leading 1 is missing (common if only lower coefs were returned),
        # then append an explicit leading 1.
        # Heuristic: if len(frac_coeffs) == 2 and frac_coeffs[0] < 1 and frac_coeffs[1] != 0, we try appending 1.
        # Safer: don't silently mutate; prefer to return as-is and let caller handle if degree mismatch.
        # For now, if leading coeff is not 1 but <= 1 in magnitude, append an explicit 1 to represent monic.
        frac_coeffs = [Fraction(1,1)] + frac_coeffs

    return frac_coeffs

