def _poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    """
    Reduce polynomial modulo u(x) = x^2 - s*x + p.
    Input: poly_coeffs in HIGH -> LOW order [a_n, a_{n-1}, ..., a_0].
    Returns: [r1, r0] where result = r1*x + r0.
    """
    coeffs = list(poly_coeffs)
    
    # 1. Trim leading zeros
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)
    
    # 2. Horner-like reduction
    while len(coeffs) > 2:
        a = coeffs.pop(0)
        # x^d -> x^{d-2}(s*x - p)
        # Add a*s to next term (x^{d-1})
        coeffs[0] = coeffs[0] + a * s
        # Subtract a*p from term after that (x^{d-2})
        coeffs[1] = coeffs[1] - a * p
        
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    # 3. Final normalization to [r1, r0]
    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs


def poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    coeffs = list(poly_coeffs)
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)

    while len(coeffs) > 2:
        a = coeffs.pop(0)
        coeffs[0] = coeffs[0] + a * s
        coeffs[1] = coeffs[1] - a * p
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs


def _poly_mod_quad_fast(f_coeffs, s_val, p_val, mod_p):
    """
    Computes f(x) mod (x^2 - s*x + p) efficiently using Horner's method.
    Input f_coeffs must be in HIGH -> LOW order.
    Returns (r1, r0) such that f(x) = r1*x + r0.
    """
    r1 = 0
    r0 = 0
    for coeff in f_coeffs:
        # x(r1*x + r0) = r1*x^2 + r0*x
        #              = r1(s*x - p) + r0*x
        #              = (r1*s + r0)*x - r1*p
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        r1, r0 = new_r1, new_r0
    return r1, r0

def _normalize_sign(s, p, v0, v1):
    if v1 < 0 or (v1 == 0 and v0 < 0):
        return (s, p, -v0, -v1)
    return (s, p, v0, v1)


def _poly_from_coeffs_qq(R, coeffs):
    """Reconstructs a polynomial in R from highest-to-lowest QQ coefficients."""
    p = R(0)
    # Handle the case where u=[1] (x-r1), u=[1, -s, p] (x^2-sx+p), etc.
    # The reconstruction must handle varying degree in the loop.
    for c in coeffs:
        p = p * R.gen() + c
    return p

def _get_divisor_coeffs_qq(D):
    """Extracts rational coefficients (QQ) from a Sage Jacobian element D=[u, v]."""
    u = D[0]
    v = D[1]
    # Sage's .list() returns coeffs lowest-to-highest, so reverse them.
    return u.list()[::-1], v.list()[::-1]

def make_monic(u):
    lc = u.leading_coefficient()
    if lc == 1:
        return u
    return (u / lc).change_ring(QQ)   # make monic over QQ


def reduce_v_mod_u(v, u):
    # ensure deg v < deg u by polynomial remainder
    _, r = v.quo_rem(u)
    return r.change_ring(QQ)


def is_divisor_on_curve(u, v, f):
    """
    Tests Mumford divisor conditions:
      1) u monic
      2) deg v < deg u
      3) v^2 - f is divisible by u (exactly)
    Returns (True, None) or (False, reason_string)
    """
    # Ensure polynomial rings are in QQ[x]
    u = u.change_ring(QQ)
    v = v.change_ring(QQ)
    f = f.change_ring(QQ)

    # 1) u must be monic
    if u.leading_coefficient() != 1:
        return False, "u not monic"

    # 2) deg v < deg u
    if v.degree() >= u.degree():
        return False, f"deg v ({v.degree()}) >= deg u ({u.degree()})"

    # 3) divisibility: check remainder of v^2 - f on division by u
    rem = (v**2 - f).quo_rem(u)[1]
    if rem != 0:
        return False, f"v^2 - f mod u != 0 (rem={rem})"

    return True, None

