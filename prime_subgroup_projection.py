# === prime_subgroup_projection.py ===
"""
Projects the hyperelliptic curve Jacobian setup into its largest prime-order subgroup.
For HECC index calculus, we want to work in J(F_p)[ℓ] from the beginning,
not in the full J(F_p) and project later.
"""

from sage.all import GF, PolynomialRing, HyperellipticCurve, factor, Integer, QQ
from time import sleep

# [Deleted unused/duplicate setup_prime_subgroup_system function]


# -------------------------
# Helper: canonical Sage polynomial from user coeff list
# -------------------------
def sage_poly_from_coeffs(coeffs, R):
    """
    Build a polynomial in PolynomialRing R from `coeffs`,
    where coeffs[-1] is the constant term and coeffs[0] the leading coeff.

    Args:
        coeffs: iterable of coefficients in user order [a_n, ..., a_0]
        R: a Sage PolynomialRing instance, e.g. PolynomialRing(GF(p), 'x')
    Returns:
        polynomial in R (exact type of R)
    """
    x = R.gen()
    deg = len(coeffs) - 1
    # Construct explicitly to avoid ambiguity about list ordering
    poly = R(0)
    for i, c in enumerate(coeffs):
        coeff = R(int(c))
        power = deg - i
        poly += coeff * x**power
    return poly


# === prime_subgroup_projection.py ===
"""
Projects the hyperelliptic curve Jacobian setup into its largest prime-order subgroup.
For HECC index calculus, we want to work in J(F_p)[ℓ] from the beginning.
"""


# === prime_subgroup_projection.py ===
"""
Projects the hyperelliptic curve Jacobian setup into its largest prime-order subgroup.
For HECC index calculus, we want to work in J(F_p)[ℓ] from the beginning.
"""


def generate_keypair_from_secret(coeffs_genus2, p, secret_key, data_pts_genus2):
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_poly = R([K(c) for c in reversed(coeffs_genus2)])
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    x_base_qq = data_pts_genus2[0]
    x_base = K(x_base_qq)

    y2_val = f_poly(x_base)
    if not y2_val.is_square():
        raise ValueError("Base point not quadratic residue")

    y_base = y2_val.sqrt()
    P_base = C((x_base, y_base))
    G = J(P_base)
    Q = Integer(secret_key) * G

    preferred_x_values = set()
    for D in (G, Q):
        u = D[0]
        for root, _ in u.roots():
            preferred_x_values.add(int(root))

    return G, Q, preferred_x_values


def get_random_x_on_hyperelliptic(coeffs, p):
    Fp = GF(p)
    for _ in range(1000):
        try_x = Fp.random_element()
        val = Fp(0)
        deg = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            val += Fp(c) * (try_x**(deg - i))
        
        if val.is_square() and val:
            return QQ(int(try_x))
            
    raise ValueError(f"Failed to find a valid point on the curve mod {p}.")


def compute_jacobian_order(f_coeffs, p):
    """
    Computes the exact order of the Jacobian J(F_p).
    For Genus 2, the order is P(1) where P is the frobenius polynomial.
    """
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    
    # Construct the curve
    deg = len(f_coeffs) - 1
    f = P_x(0)
    for i, c in enumerate(f_coeffs):
        f += K(c) * P_x.gen()**(deg - i)
    
    C = HyperellipticCurve(f)
    
    try:
        # The characteristic polynomial of Frobenius P(t)
        # The number of points on the Jacobian is P(1)
        frob_poly = C.frobenius_polynomial()
        return Integer(frob_poly(1))
    except Exception as e:
        print(f"Standard order computation failed: {e}. Falling back...")
        # Fallback for very small p or specific Sage versions
        return Integer(C.jacobian().order())


def generate_random_curve_point(f_poly, p):
    F = GF(p)
    R = PolynomialRing(F, 'x')
    f = R(f_poly)
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    for _ in range(1000):
        x_coord = F.random_element()
        y2 = f(x_coord)
        if y2.is_square() and not y2.is_zero():
            y_coord = y2.sqrt()
            P = J(C((x_coord, y_coord)))
            if not (2 * P).is_zero():
                return P, int(x_coord), int(y_coord)
    
    raise ValueError("Failed to generate random curve point")


def setup_prime_subgroup_cryptosystem(p, coeffs_genus2, base_pts_x, secret_key):
    F = GF(p)
    R = PolynomialRing(F, 'x')
    f_poly = R([F(c) for c in reversed(coeffs_genus2)])
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    order = compute_jacobian_order(coeffs_genus2, p)
    factorization = factor(order)
    ell = max([Integer(prime) for prime, _ in factorization])
    cofactor = order // ell
    
    print(f"Jacobian order: {order}")
    print(f"Factorization: {factorization}")
    print(f"Largest prime ℓ: {ell}")
    print(f"Cofactor h: {cofactor}")
    
    G_original, basex, basey = generate_random_curve_point(f_poly, p)
    G = Integer(cofactor) * G_original
    
    if G.is_zero():
        raise RuntimeError("G projected to identity.")
    
    current_secret = Integer(secret_key) % ell
    Q = None
    final_secret = None
    
    # Search for a split target Q
    for offset in range(1000):
        test_secret = (current_secret + offset) % ell
        if test_secret == 0: continue
            
        Q_candidate = Integer(test_secret) * G
        if Q_candidate.is_zero(): continue
            
        u_poly = Q_candidate[0]
        if u_poly.degree() == 1:
            Q = Q_candidate
            final_secret = test_secret
            break
        elif u_poly.degree() == 2:
            disc = u_poly.discriminant()
            if disc.is_square() and disc != 0:
                Q = Q_candidate
                final_secret = test_secret
                break
    
    if Q is None:
        raise RuntimeError("Failed to find split Q")
    
    preferred_x_coords = set()
    for D in [G, Q]:
        for root, _ in D[0].roots():
            preferred_x_coords.add(int(root))
    
    return ell, [basex], G, Q, preferred_x_coords, final_secret
