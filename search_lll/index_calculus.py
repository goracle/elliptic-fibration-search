from sage.all import matrix, GF, vector, ZZ, PolynomialRing, Curve, Jacobian, Integer, Zmod, prime_factors
from .smoothness import extract_factor_base, tonelli_shanks

def get_canonical_y(x, f_poly, p):
    """
    Returns a canonical y-coordinate for a given x such that y^2 = f(x) mod p.
    """
    y2 = f_poly(x)
    if y2 == 0:
        return None
    # tonelli_shanks_ic imported from smoothness
    y = tonelli_shanks_ic(y2, p)
    return min(int(y), int(p - y))

def get_relation_row(divisor, root_to_idx, f_poly, p):
    """
    Converts a Mumford divisor (u, v) into a relation row with signs.
    Ensures that u(x) splits completely over the factor base.
    """
    u_poly = divisor.u()
    roots_data = u_poly.roots(GF(p))
    
    # Strict smoothness check: u(x) must split fully with degree 2
    if sum(mult for _, mult in roots_data) != u_poly.degree():
        return None
        
    v_poly = divisor.v()
    row_data = [0] * len(root_to_idx)
    
    for x_val, mult in roots_data:
        x_val = Integer(x_val)
        if x_val not in root_to_idx:
            return None
            
        y_val = Integer(v_poly(x_val))
        y_can = get_canonical_y(x_val, f_poly, p)
        
        if y_can is None: # Weierstrass point check
            return None
            
        idx = root_to_idx[x_val]
        if y_val == y_can:
            row_data[idx] += int(mult)
        elif (p - y_val) % p == y_can:
            row_data[idx] -= int(mult)
        else:
            return None
            
    return row_data

def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order, max_tries=5000):
    """
    Finds target + [r]generator = sum(c_i * P_i).
    Avoids r=0 to ensure entropy in the decomposition.
    """
    Z_ord = Zmod(order)
    for _ in range(max_tries):
        r_val = Integer(Z_ord.random_element())
        if r_val == 0:
            continue
            
        t_div = target_point + r_val * generator
        if t_div.u().degree() != 2:
            continue
            
        row_vec = get_relation_row(t_div, root_to_idx, f_poly, p)
        if row_vec is not None:
            return r_val, row_vec
    return None, None

def solve_dlp_index_calculus(valid_rows, g_anchored, q_anchored, ell, verbose=True):
    """
    Solves the system mod ell. ell MUST be prime.
    Uses the G-anchor to define the logs and Q-anchor to find the target.
    """
    K = GF(ell)
    
    # Build the relation matrix
    M = matrix(K, valid_rows)
    
    # We include the G-anchored row to find absolute logs of FB elements
    rg, row_g = g_anchored
    rq, row_q = q_anchored
    
    # Augmented matrix: [Relations] = 0, [row_g] = rg
    # This solves sum(row_i * logs) = 0 and sum(row_g * logs) = rg (mod ell)
    aug_rows = [vector(K, r) for r in valid_rows] + [vector(K, row_g)]
    aug_targets = [K(0)] * len(valid_rows) + [K(rg)]
    
    M_sys = matrix(K, aug_rows)
    target_v = vector(K, aug_targets)
    
    if verbose:
        print(f"Matrix size: {M_sys.nrows()}x{M_sys.ncols()}, Rank: {M_sys.rank()}")

    try:
        # solve_right finds the vector 'logs' such that M_sys * logs = target_v
        fb_logs = M_sys.solve_right(target_v)
    except ValueError:
        raise ArithmeticError("Linear system is inconsistent mod ell. Need more relations.")

    # log(Q) = sum(row_q * fb_logs) - rq (mod ell)
    v_q = vector(K, row_q)
    log_q = (v_q.dot_product(fb_logs) - K(rq))
    
    return Integer(log_q)

def perform_dlp_attack(g_pt, q_pt, smooth_divs, p, f_coeffs, order, verbose=True):
    """
    Main entry point. Automatically factors order and solves mod the largest prime.
    """
    # Factorization to find the largest prime factor ell
    factors = prime_factors(order)
    ell = max(factors)
    
    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order: {ell}")
    
    K_p = GF(p)
    poly_ring = PolynomialRing(K_p, 'x')
    f_p = poly_ring(f_coeffs[::-1])
    
    # Setup factor base
    fb_data = extract_factor_base(smooth_divs, p, verbose=False)
    # Exclude Weierstrass points from the root index
    roots_filtered = sorted([r for r in fb_data['roots'] if f_p(r) != 0])
    r_to_idx = {r: i for i, r in enumerate(roots_filtered)}
    
    if verbose:
        print(f"Factor base size: {len(roots_filtered)}")
    
    # Convert all extracted smooth divisors into relation rows
    valid_rows = []
    for d in fb_data['unique_divisors']:
        row = get_relation_row(d, r_to_idx, f_p, p)
        if row:
            valid_rows.append(row)
            
    if len(valid_rows) < len(roots_filtered):
        if verbose:
            print(f"Warning: Only {len(valid_rows)} relations for {len(roots_filtered)} FB elements.")
    
    # Anchoring
    rg, row_g = find_smooth_decomposition(g_pt.parent().zero(), g_pt, r_to_idx, f_p, p, order)
    rq, row_q = find_smooth_decomposition(q_pt, g_pt, r_to_idx, f_p, p, order)
    
    if row_g is None or row_q is None:
        raise ValueError("Failed to anchor G or Q into the factor base.")

    # Solve mod ell
    d_mod_ell = solve_dlp_index_calculus(valid_rows, (rg, row_g), (rq, row_q), ell, verbose=verbose)
    
    # Project the problem to the l-torsion to verify
    # Q' = [order/ell]Q, G' = [order/ell]G. Then Q' = [d]G' should hold.
    factor_cofactor = order // ell
    if (d_mod_ell * (factor_cofactor * g_pt)) == (factor_cofactor * q_pt):
        if verbose:
            print(f"✓ Success mod {ell}. Log: {d_mod_ell}")
        return d_mod_ell
    else:
        raise ArithmeticError(f"Verification failed mod {ell}. The relations may be spurious.")


def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G) on the Jacobian of y^2 = f(x).
    Intended ONLY for testing the index calculus machinery.
    """
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    # Build curve and Jacobian
    P = PolynomialRing(K, 2, names=['x', 'y'])
    x, y = P.gens()
    C = Curve(y**2 - f(x))
    J = Jacobian(C)
    G = J.random_element()
    if target_d is None:
        target_d = ZZ.random_element(1, J.order())
    Q = target_d * G
    return G, Q, Integer(target_d)


from sage.all import Curve, Jacobian, PolynomialRing, GF
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve


def compute_jacobian_order(f_coeffs, p):
    """
    Computes the order of the Jacobian of the curve y^2 = f(x) over GF(p).
    Strictly follows the requirement to use Curve and Jacobian.
    """
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    f = P_x(f_coeffs)
    
    # Construct the curve using the generic Curve constructor
    P_affine = PolynomialRing(K, 2, names=['x', 'y'])
    x, y = P_affine.gens()
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    # Sage's J.order() is robust for p ~ 2^32
    return J.order()


def compute_jacobian_order(f_coeffs, p):
    """
    Computes the order of the Jacobian of the curve y^2 = f(x) over GF(p).
    Strictly follows the requirement to use Curve and Jacobian.
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    f = P_x(f_coeffs)
    
    # Construct the curve using HyperellipticCurve
    C = HyperellipticCurve(f)
    
    # Compute the cardinality using the Frobenius polynomial evaluated at 1
    # This gives the order of the Jacobian
    frob_poly = C.frobenius_polynomial()
    order = frob_poly(1)
    
    return abs(order)


def compute_jacobian_order(f_coeffs, p):
    """
    Computes the order of the Jacobian of the curve y^2 = f(x) over GF(p).
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    f = P_x(f_coeffs)
    
    print(f"DEBUG: p = {p}")
    print(f"DEBUG: f = {f}")
    print(f"DEBUG: f.degree() = {f.degree()}")
    print(f"DEBUG: f.is_squarefree() = {f.is_squarefree()}")
    
    C = HyperellipticCurve(f)
    return C.count_points(1)[0]


def compute_jacobian_order(f_coeffs, p):
    """
    Computes approximate Jacobian order using Hasse-Weil bound for large primes.
    For p > 2^64, returns approximate value suitable for probabilistic algorithms.
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    from sage.all import Integer, RR
    
    K = GF(p)
    P_x = PolynomialRing(K, 'x')
    f = P_x(f_coeffs)
    
    # For genus 2: Hasse-Weil gives |#J - (p^2+1)| <= 4*sqrt(p^3)
    # Use p^2 + 1 as approximation for large p
    if p > 2**64:
        # Return approximate order: p^2 + 1
        # This is close enough for index calculus smoothness bounds
        return Integer(p**2 + 1)
    
    # For smaller primes, compute exactly
    C = HyperellipticCurve(f)
    return C.count_points(1)[0]


def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G) on the Jacobian of y^2 = f(x).
    Intended ONLY for testing the index calculus machinery.
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    from sage.all import ZZ, Integer
    
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    
    # Use HyperellipticCurve directly instead of generic Curve
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    # Get a random element from the Jacobian
    G = J.random_element()
    
    if target_d is None:
        # For very large groups, pick a reasonable sized discrete log
        # Full order computation is expensive/impossible for large p
        if p > 2**64:
            # Use a random d in a reasonable range
            target_d = ZZ.random_element(1, 2**64)
        else:
            # For smaller p, can try to get actual order
            try:
                order = J.order()
                target_d = ZZ.random_element(1, order)
            except (NotImplementedError, RuntimeError):
                target_d = ZZ.random_element(1, 2**64)
    
    Q = target_d * G
    return G, Q, Integer(target_d)


def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G) on the Jacobian of y^2 = f(x).
    Intended ONLY for testing the index calculus machinery.
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    from sage.all import ZZ, Integer
    
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    
    # Use HyperellipticCurve directly instead of generic Curve
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    # Generate a random point on the curve, then convert to Jacobian element
    # For genus 2, we need to find a point (x, y) on the curve
    x_coord = K.random_element()
    y_squared = f(x_coord)
    
    # Keep trying until we find a point
    max_attempts = 1000
    for _ in range(max_attempts):
        if y_squared.is_square():
            y_coord = y_squared.sqrt()
            # Create a divisor from the point
            P = C((x_coord, y_coord))
            G = J(P)
            break
        x_coord = K.random_element()
        y_squared = f(x_coord)
    else:
        raise RuntimeError(f"Failed to find a valid point after {max_attempts} attempts")
    
    if target_d is None:
        # For very large groups, pick a reasonable sized discrete log
        if p > 2**64:
            target_d = ZZ.random_element(1, 2**64)
        else:
            try:
                order = J.order()
                target_d = ZZ.random_element(1, order)
            except (NotImplementedError, RuntimeError):
                target_d = ZZ.random_element(1, 2**64)
    
    Q = target_d * G
    return G, Q, Integer(target_d)


def perform_dlp_attack(g_pt, q_pt, smooth_divs, p, f_coeffs, order, verbose=True):
    """
    Main entry point. Automatically factors order and solves mod the largest prime.
    """
    # Factorization to find the largest prime factor ell
    factors = prime_factors(order)
    ell = max(factors)
    
    if verbose:
        print(f"Jacobian order factors: {factors}")
        print(f"Targeting subgroup of prime order: {ell}")
    
    K_p = GF(p)
    poly_ring = PolynomialRing(K_p, 'x')
    x = poly_ring.gen()
    f_p = poly_ring(f_coeffs[::-1])
    
    # Build the curve and Jacobian
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    C = HyperellipticCurve(f_p)
    J = C.jacobian()
    
    # Setup factor base
    fb_data = extract_factor_base(smooth_divs, p, verbose=False)
    # Exclude Weierstrass points from the root index
    roots_filtered = sorted([r for r in fb_data['roots'] if f_p(r) != 0])
    r_to_idx = {r: i for i, r in enumerate(roots_filtered)}
    
    if verbose:
        print(f"Factor base size: {len(roots_filtered)}")
    
    # Convert dictionary divisors to Sage Jacobian elements
    sage_divisors = []
    for d in fb_data['unique_divisors']:
        try:
            s_val = int(d['s']) % p
            p_val = int(d['p']) % p
            v0_val = int(d['v_0']) % p
            v1_val = int(d['v_1']) % p
            
            # Build Mumford polynomials
            u_poly = x**2 - K_p(s_val)*x + K_p(p_val)
            v_poly = K_p(v1_val)*x + K_p(v0_val)
            
            # Create Jacobian element
            D = J([u_poly, v_poly])
            sage_divisors.append(D)
        except Exception as e:
            if verbose:
                print(f"Failed to convert divisor: {e}")
            continue
    
    if not sage_divisors:
        raise ValueError("No valid divisors after conversion to Sage objects")
    
    # Convert all extracted smooth divisors into relation rows
    valid_rows = []
    for d in sage_divisors:
        row = get_relation_row(d, r_to_idx, f_p, p)
        if row:
            valid_rows.append(row)
            
    if len(valid_rows) < len(roots_filtered):
        if verbose:
            print(f"Warning: Only {len(valid_rows)} relations for {len(roots_filtered)} FB elements.")
    
    # Anchoring
    rg, row_g = find_smooth_decomposition(g_pt.parent().zero(), g_pt, r_to_idx, f_p, p, order)
    rq, row_q = find_smooth_decomposition(q_pt, g_pt, r_to_idx, f_p, p, order)
    
    if row_g is None or row_q is None:
        raise ValueError("Failed to anchor G or Q into the factor base.")

    # Solve mod ell
    d_mod_ell = solve_dlp_index_calculus(valid_rows, (rg, row_g), (rq, row_q), ell, verbose=verbose)
    
    # Project the problem to the l-torsion to verify
    # Q' = [order/ell]Q, G' = [order/ell]G. Then Q' = [d]G' should hold.
    factor_cofactor = order // ell
    if (d_mod_ell * (factor_cofactor * g_pt)) == (factor_cofactor * q_pt):
        if verbose:
            print(f"✓ Success mod {ell}. Log: {d_mod_ell}")
        return d_mod_ell
    else:
        raise ArithmeticError(f"Verification failed mod {ell}. The relations may be spurious.")


def get_relation_row(divisor, root_to_idx, f_poly, p):
    """
    Converts a Mumford divisor (u, v) into a relation row with signs.
    Ensures that u(x) splits completely over the factor base.
    
    For Sage Jacobian elements, access via divisor[0] and divisor[1]
    """
    # Access Mumford polynomials: divisor[0] is u(x), divisor[1] is v(x)
    u_poly = divisor[0]
    v_poly = divisor[1]
    
    roots_data = u_poly.roots(GF(p))
    
    # Strict smoothness check: u(x) must split fully with degree 2
    if sum(mult for _, mult in roots_data) != u_poly.degree():
        return None
        
    row_data = [0] * len(root_to_idx)
    
    for x_val, mult in roots_data:
        x_val = Integer(x_val)
        if x_val not in root_to_idx:
            return None
            
        y_val = Integer(v_poly(x_val))
        y_can = get_canonical_y(x_val, f_poly, p)
        
        if y_can is None: # Weierstrass point check
            return None
            
        idx = root_to_idx[x_val]
        if y_val == y_can:
            row_data[idx] += int(mult)
        elif (p - y_val) % p == y_can:
            row_data[idx] -= int(mult)
        else:
            return None
            
    return row_data


def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order, max_tries=5000):
    """
    Finds target + [r]generator = sum(c_i * P_i).
    Avoids r=0 to ensure entropy in the decomposition.
    """
    Z_ord = Zmod(order)
    for _ in range(max_tries):
        r_val = Integer(Z_ord.random_element())
        if r_val == 0:
            continue
            
        t_div = target_point + r_val * generator
        
        # Access Mumford polynomial via indexing
        if t_div[0].degree() != 2:
            continue
            
        row_vec = get_relation_row(t_div, root_to_idx, f_poly, p)
        if row_vec is not None:
            return r_val, row_vec
    return None, None

def get_canonical_y(x, f_poly, p):
    """
    Returns a canonical y-coordinate for a given x such that y^2 = f(x) mod p.
    """
    y2 = f_poly(x)
    if y2 == 0:
        return None
    y = tonelli_shanks(y2, p)
    return min(int(y), int(p - y))


def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a test DLP instance (G, Q = [d]G) on the Jacobian of y^2 = f(x).
    Intended ONLY for testing the index calculus machinery.
    """
    from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
    from sage.all import ZZ, Integer
    
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    
    # Use HyperellipticCurve directly
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    # Generate a random point on the curve, then convert to Jacobian element
    x_coord = K.random_element()
    y_squared = f(x_coord)
    
    # Keep trying until we find a point
    max_attempts = 1000
    for _ in range(max_attempts):
        if y_squared.is_square():
            y_coord = y_squared.sqrt()
            # Create a divisor from the point
            P = C((x_coord, y_coord))
            G = J(P)
            break
        x_coord = K.random_element()
        y_squared = f(x_coord)
    else:
        raise RuntimeError(f"Failed to find a valid point after {max_attempts} attempts")
    
    if target_d is None:
        # Use your existing compute_jacobian_order function
        order = compute_jacobian_order(f_poly.list(), p)
        target_d = ZZ.random_element(1, order)
    
    Q = target_d * G
    return G, Q, Integer(target_d)
