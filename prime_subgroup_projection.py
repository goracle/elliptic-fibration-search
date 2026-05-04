from sage.all import GF, PolynomialRing, HyperellipticCurve, factor, Integer, QQ
from time import sleep
from math import ceil, sqrt
from multiprocessing import cpu_count

# === prime_subgroup_projection.py ===
"""
Projects the hyperelliptic curve Jacobian setup into its largest prime-order subgroup.
For HECC index calculus, we want to work in J(F_p)[ℓ] from the beginning,
not in the full J(F_p) and project later.
"""

# [Deleted unused/duplicate setup_prime_subgroup_system function]

# -------------------------
# Helper: canonical Sage polynomial from user coeff list
# -------------------------

def sage_poly_from_coeffs(coeffs, R):
    """Build polynomial from coefficient list (highest degree first)"""
    result = R(0)
    for c in coeffs:
        result = result * R.gen() + R(c)
    return result

"""
Projects the hyperelliptic curve Jacobian setup into its largest prime-order subgroup.
For HECC index calculus, we want to work in J(F_p)[ℓ] from the beginning.
"""

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

    preferred_atoms = set()
    for D in (G, Q):
        u = D[0]
        for root, _ in u.roots():
            x_int = int(root)
            y2_val = f_poly(K(x_int))
            if y2_val.is_square() and not y2_val.is_zero():
                y_int = int(y2_val.sqrt())
                y_canon = min(y_int, p - y_int)
                preferred_atoms.add((x_int, y_canon))

    return G, Q, preferred_atoms

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

# ---------------------------------------------------------------------
# Helper: ensure G and Q are in the ℓ-subgroup (or project them)
# ---------------------------------------------------------------------
def ensure_prime_subgroup_elements(G, Q, full_order, verbose=False):
    """
    Ensure G and Q lie in J(F_p)[ell], where ell is the largest prime factor of full_order.
    If they are not, attempt to project via the cofactor h = |J| // ell and return projected elements.
    Returns (ell, h, G_used, Q_used).
    """
    J_order = Integer(full_order)
    facs = factor(J_order)
    ell = int(max(int(p) for p, _ in facs))
    h = int(J_order // ell)

    # Quick checks: ell*G and ell*Q should be zero if already in subgroup
    ellG_zero = bool((Integer(ell) * G).is_zero())
    ellQ_zero = bool((Integer(ell) * Q).is_zero())

    if verbose:
        print(f"[ensure] ell={ell}, cofactor h={h}")
        print(f"[ensure] ell*G == 0? {ellG_zero}; ell*Q == 0? {ellQ_zero}")

    if ellG_zero and ellQ_zero:
        return ell, h, G, Q

    # Not in ℓ-subgroup: attempt to project by cofactor h
    if verbose:
        print("[ensure] Warning: G/Q not in ℓ-subgroup. Projecting by cofactor h to move into ℓ-subgroup.")

    Gp = Integer(h) * G
    Qp = Integer(h) * Q

    if not (Integer(ell) * Gp).is_zero() or not (Integer(ell) * Qp).is_zero():
        # Projection failed: something strange; raise to force user to inspect the setup
        raise RuntimeError("[ensure] Failed to project G/Q into ℓ-subgroup (post-projection still not ℓ-torsion).")

    return ell, h, Gp, Qp

# ---------------------------------------------------------------------
# Main solver rewrite (Model A: operate directly mod ell)
# ---------------------------------------------------------------------

def setup_prime_subgroup_cryptosystem(p, coeffs_genus2, base_pts_x, secret_key, verbose=False):
    F = GF(p)
    R = PolynomialRing(F, 'x')
    f_poly = R([F(c) for c in reversed(coeffs_genus2)])
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    f = R(f_poly)

    order = compute_jacobian_order(coeffs_genus2, p)
    factorization = factor(order)
    ell = max([Integer(prime) for prime, _ in factorization])
    cofactor = order // ell

    if verbose:
        print(f"Jacobian order: {order}")
        print(f"Factorization: {factorization}")
        print(f"Largest prime ℓ: {ell}")
        print(f"Cofactor h: {cofactor}")

    def has_split_degree2_u(D):
        """Check if D has degree-2 u(x) that splits over F_p"""
        if D.is_zero():
            return False
        u_poly = D[0]
        if u_poly.degree() != 2:
            return False
        disc = u_poly.discriminant()
        return disc != 0 and disc.is_square()

    # Search for G_original that projects to a split divisor
    max_attempts = 10000
    G = None

    for attempt in range(max_attempts):
        if base_pts_x[0] is None:
            try:
                G_original, basex, basey = generate_random_curve_point(f_poly, p)
            except Exception:
                continue
        else:
            # Start from provided base point
            x_coord = base_pts_x[0]
            y2 = f(F(x_coord))
            if not y2.is_square():
                raise ValueError("Base point not quadratic residue")
            y_coord = y2.sqrt()
            G_original_base = J(C((x_coord, y_coord)))

            # Try multiples to find one that projects well
            # Try G_original = [k] * base for k = 1, 2, 3, ...
            G_original = Integer(attempt + 1) * G_original_base

        # Project into ℓ-subgroup
        G_candidate = Integer(cofactor) * G_original

        if G_candidate.is_zero():
            continue

        # Check if it has split u(x)
        if has_split_degree2_u(G_candidate):
            G = G_candidate
            if base_pts_x[0] is None:
                base_pts_x = [basex]
            break

        # If using provided base point, keep trying multiples
        if base_pts_x[0] is None:
            # For random generation, just try a new random point
            pass

    if G is None:
        raise RuntimeError(f"Failed to find G with split u(x) after {max_attempts} attempts")

    # Verify G is in ℓ-subgroup
    assert (Integer(ell) * G).is_zero(), "G not in ℓ-subgroup"

    # Search for Q = [k]*G with split u(x)
    current_secret = Integer(secret_key) % ell
    Q = None
    final_secret = None

    for offset in range(max_attempts):
        test_secret = (current_secret + offset) % ell
        if test_secret == 0:
            continue

        Q_candidate = Integer(test_secret) * G

        if has_split_degree2_u(Q_candidate):
            Q = Q_candidate
            final_secret = test_secret
            break

    if Q is None:
        raise RuntimeError(f"Failed to find Q with split u(x) after {max_attempts} attempts")

    # Extract atoms — (x, y) tuples using the canonical (smaller) y branch.
    preferred_atoms = set()
    for D in [G, Q]:
        u_poly = D[0]
        for root, _ in u_poly.roots():
            x_int = int(root)
            y2_val = f(F(x_int))
            if y2_val.is_square() and not y2_val.is_zero():
                y_int = int(y2_val.sqrt())
                y_canon = min(y_int, p - y_int)
                preferred_atoms.add((x_int, y_canon))

    assert len(preferred_atoms) == 4, f"Expected 4 atoms, got {len(preferred_atoms)}: {preferred_atoms}"

    if verbose:
        print(f"Generated {len(preferred_atoms)} preferred atoms: {preferred_atoms}")
        print(f"G has u(x) = {G[0]} (splits)")
        print(f"Q has u(x) = {Q[0]} (splits)")

    return ell, base_pts_x, G, Q, preferred_atoms, final_secret, cofactor
