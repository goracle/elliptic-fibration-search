from sage.all import matrix, GF, vector, ZZ, PolynomialRing, HyperellipticCurve, Integer
import random
from smoothness import tonelli_shanks, extract_factor_base
from sage.all import matrix, GF, vector, ZZ, QQ
from collections import Counter
from sage.all import matrix, GF, vector, ZZ, PolynomialRing, HyperellipticCurve, Integer, Zmod, gcd
from smoothness import tonelli_shanks_ic, extract_factor_base

def solve_dlp_index_calculus(mumford_divisors, factor_base, L, G, Q):
    """
    Performs the linear algebra phase of the index calculus attack.
    L: Order of the Jacobian (or a large prime factor of it).
    G: Base point (divisor).
    Q: Target point (divisor).
    """
    # 1. Map factor base to matrix columns
    root_to_col = {root: i for i, root in enumerate(factor_base)}
    num_vars = len(factor_base)
    
    # 2. Build Sparse Relation Matrix
    # Each row: sum(c_ij * P_j) ~ 0 in Jacobian
    rows = []
    for div in mumford_divisors:
        row_data = {}
        for r in div['roots']:
            if r in root_to_col:
                col = root_to_col[r]
                row_data[col] = row_data.get(col, 0) + 1
        
        # In the Jacobian, relations are D ~ 0, so we solve the homogeneous system
        # matrix * log(P) = 0 (mod L)
        rows.append(row_data)
    
    M = matrix(GF(L), len(rows), num_vars, rows, sparse=True)
    
    print(f"  Matrix size: {M.nrows()}x{M.ncols()} (Sparse)")
    
    # 3. Find kernel to get relative logs of factor base elements
    # kernel() is efficient for sparse matrices in Sage
    ker = M.kernel()
    if ker.dimension() == 0:
        raise ValueError("Insufficient relations to find kernel.")
    
    log_vector = ker.random_element()
    
    # 4. Express G and Q in terms of the factor base
    # This requires finding 'smooth' relations for G and Q specifically
    # For testing, we assume they are provided or found via 'shaking'
    # d = log(Q) / log(G) mod L
    
    return log_vector

def decompose_target_over_basis(target_divisor, factor_base, log_vector, L):
    """
    Helper to compute the 'log' of a specific divisor once the 
    factor base logs are known.
    """
    # Decompose divisor: D = sum n_i * P_i
    # log(D) = sum n_i * log(P_i) mod L
    roots = target_divisor.get('roots', [])
    total_log = GF(L)(0)
    
    root_to_col = {root: i for i, root in enumerate(factor_base)}
    
    for r in roots:
        if r in root_to_col:
            total_log += log_vector[root_to_col[r]]
        else:
            return None # Target not smooth over current factor base
            
    return total_log


def solve_for_fb_logs(matrix_rows, order):
    """
    Solves the system M * x = 0 mod order.
    Returns a vector of logarithms for the factor base elements.
    """
    M = matrix(Zmod(order), matrix_rows)
    K = M.right_kernel()
    if K.dimension() == 0:
        return None
    
    fb_logs = K.basis()[0]
    return fb_logs


def solve_for_fb_logs_anchored(matrix_rows, G_row, order):
    """
    Solves the inhomogeneous system M * x = b mod order, where:
    - M contains relations: sum(row[i] * log(P_i)) = 0
    - b encodes the G-relation: sum(G_row[i] * log(P_i)) = log(G)
    
    We normalize by setting log(P_0) = 1, removing that column, and solving.
    
    Returns: vector of logs with log(P_0) implicitly = 1
    """
    if not matrix_rows:
        raise ValueError("No relations provided")
    if G_row is None:
        raise ValueError("No G-relation provided")
    
    n = len(G_row)
    if n == 0:
        raise ValueError("Empty G_row")
    
    M = matrix(Zmod(order), matrix_rows)
    g_vec = vector(Zmod(order), G_row)
    
    M_reduced = M.delete_columns([0])
    b_reduced = -g_vec[0] * vector(Zmod(order), [row[0] for row in matrix_rows])
    
    try:
        logs_reduced = M_reduced.solve_right(b_reduced)
    except ValueError as e:
        raise ValueError(f"System not solvable: {e}")
    
    fb_logs = [Zmod(order)(1)] + list(logs_reduced)
    return fb_logs


def solve_dlp_via_relation_ratio(G_relation, Q_relation, order, verbose=True):
    """
    Solve for log_G(Q) using two smooth decompositions WITHOUT computing absolute logs.
    
    Given:
        [r_G]G = sum(a_i * P_i)     =>  r_G * log(G) = sum(a_i * log(P_i))
        Q + [r_Q]G = sum(b_i * P_i) =>  log(Q) + r_Q * log(G) = sum(b_i * log(P_i))
    
    Subtracting:
        log(Q) + (r_Q - r_G) * log(G) = sum((b_i - a_i) * log(P_i))
    
    This is still one equation with many unknowns. But if we assume the factor base
    relations form a consistent system (which they should if the factor base is large
    enough), then the difference vector (b - a) should equal a scalar multiple of the
    relation that expresses log(Q) / log(G).
    
    The correct approach is to solve the system:
        M * logs = 0  (homogeneous relations from factor base)
        a_vec · logs = r_G  (G-relation)
        b_vec · logs = log(Q) + r_Q  (Q-relation, unknown log(Q))
    
    We eliminate logs by expressing them via the kernel and solving for log(Q).
    """
    r_G, a_vec = G_relation
    r_Q, b_vec = Q_relation
    
    assert len(a_vec) == len(b_vec), "Relation vectors must have same dimension"
    
    a_sum = sum(Integer(x) for x in a_vec) % order
    b_sum = sum(Integer(x) for x in b_vec) % order
    
    g = gcd(a_sum, order)
    if g != 1:
        raise ValueError(f"G-relation sum {a_sum} not coprime to order {order}. gcd={g}. Cannot invert.")
    
    inv_a = pow(Integer(a_sum), -1, order)
    log_result = ((b_sum - r_Q * a_sum) * inv_a * r_G) % order
    
    return log_result


from sage.all import matrix, PolynomialRing, HyperellipticCurve, Integer, Zmod, gcd, GF


def get_canonical_y(x, f_poly, p):
    """
    Returns a canonical y-coordinate for a given x such that y^2 = f(x) mod p.
    This is used to define the 'positive' representative for each factor base point.
    """
    y2 = f_poly(x)
    if y2 == 0:
        return 0
    y = tonelli_shanks_ic(y2, p)
    return min(int(y), int(p - y))

def get_relation_row(divisor, root_to_idx, f_poly, p):
    """
    Converts a Mumford divisor (u, v) into a relation row for the index calculus matrix.
    Correctly accounts for the y-coordinate sign relative to the factor base.
    """
    s_val = int(divisor['s']) % p
    p_val = int(divisor['p']) % p
    disc = (s_val*s_val - 4*p_val) % p
    
    if disc == 0:
        roots = [(s_val * pow(2, -1, p)) % p] * 2
    elif pow(disc, (p-1)//2, p) == 1:
        sq = tonelli_shanks_ic(disc, p)
        roots = [(s_val + sq) * pow(2, -1, p) % p, (s_val - sq) * pow(2, -1, p) % p]
    else:
        return None
    
    v1 = int(divisor['v_1']) % p
    v0 = int(divisor['v_0']) % p
    
    row = [0] * len(root_to_idx)
    for x in roots:
        if x not in root_to_idx:
            return None
        
        y_val = (v1 * x + v0) % p
        y_can = get_canonical_y(x, f_poly, p)
        
        idx = root_to_idx[x]
        if y_val == y_can:
            row[idx] += 1
        else:
            row[idx] -= 1
            
    return row

def find_smooth_decomposition(target_point, generator, root_to_idx, f_poly, p, order, max_tries=5000, rng=None):
    """
    Finds a relation: target + [r]generator = sum(c_i * P_i) by randomizing r.
    Returns (r, row) where row encodes the factor base decomposition.
    """
    rng = rng or random
    for _ in range(max_tries):
        r = rng.randint(1, order - 1)
        T = target_point + r * generator
        
        u, v = T.polys()
        if u.degree() != 2:
            continue
            
        u_coeffs = u.list()
        p_const = u_coeffs[0]
        s_val = -u_coeffs[1]
        
        div_data = {
            's': s_val,
            'p': p_const,
            'v_1': v.list()[1] if v.degree() >= 1 else 0,
            'v_0': v.list()[0]
        }
        
        row = get_relation_row(div_data, root_to_idx, f_poly, p)
        if row is not None:
            return r, row
            
    return None, None

def to_int(x):
    """Convert Sage objects to Python int safely."""
    if hasattr(x, 'lift'):
        return int(x.lift())
    return int(x)

def solve_dlp_from_relations(matrix_rows, G_relation, Q_relation, order,
                             generator=None, root_to_idx=None, f_poly=None, p=None,
                             max_extra=16, verbose=True):
    """
    Solve DLP using relation matrix with proper kernel dimension handling.
    
    Args:
        matrix_rows: list of relation rows from factor base
        G_relation: (r_G, a_vec) - decomposition of [r_G]G
        Q_relation: (r_Q, b_vec) - decomposition of Q + [r_Q]G
        order: modulus (preferably prime factor of #J)
        generator: Jacobian element G (needed if kdim > 1)
        root_to_idx: factor base index (needed if kdim > 1)
        f_poly: curve polynomial (needed if kdim > 1)
        p: field characteristic (needed if kdim > 1)
        max_extra: max additional G-decompositions to collect if kdim > 1
    
    Returns:
        d such that Q = [d]G
    """
    R = Zmod(order)
    num_vars = max(len(row) for row in matrix_rows) if matrix_rows else 0
    
    dense_rows = []
    for row in matrix_rows:
        if isinstance(row, dict):
            v = [0] * num_vars
            for j, val in row.items():
                v[j] = to_int(val) % order
        else:
            v = [to_int(x) % order for x in row]
        dense_rows.append([R(x) for x in v])
    
    if not dense_rows:
        raise ValueError("No relations to build matrix")
    
    M = matrix(R, dense_rows)
    
    K = M.right_kernel()
    kdim = K.dimension()
    
    if verbose:
        print(f"[DLP solver] Kernel dimension = {kdim}")
    
    if kdim == 0:
        raise ValueError("No kernel found: relations inconsistent or insufficient.")
    
    if kdim == 1:
        kvec = K.basis()[0]
        r_G, a_vec = G_relation
        r_Q, b_vec = Q_relation
        
        S_G = R(sum(to_int(a_vec[i]) * to_int(kvec[i]) for i in range(len(a_vec))))
        S_Q = R(sum(to_int(b_vec[i]) * to_int(kvec[i]) for i in range(len(b_vec))))
        
        S_Gi = int(S_G) % order
        g = gcd(S_Gi, order)
        if g != 1:
            raise ValueError(f"S_G={S_Gi} not invertible mod order (gcd={g}). Bad kernel vector or subgroup issue.")
        
        inv_SG = pow(S_Gi, -1, order)
        d = (Integer(r_G) * Integer(S_Q) * Integer(inv_SG) - Integer(r_Q)) % order
        return int(d)
    
    if generator is None or root_to_idx is None or f_poly is None or p is None:
        raise ValueError(f"Kernel dimension = {kdim} > 1. Need generator, root_to_idx, f_poly, p to collect extra anchored decompositions.")
    
    if verbose:
        print(f"[DLP solver] Kernel dim > 1, collecting up to {max_extra} extra G-decompositions...")
        print("[warning] Kernel dim > 1 usually indicates incomplete relation lattice or symmetry.")
    
    basis = K.basis()
    collected_a = []
    collected_rG = []
    
    rG0, a0 = G_relation
    collected_rG.append(Integer(rG0) % order)
    collected_a.append([to_int(x) % order for x in a0])
    
    tries = 0
    needed = kdim + 1
    while len(collected_a) < needed and tries < max_extra:
        tries += 1
        r_e, a_e = find_smooth_decomposition(generator.parent()(0), generator, root_to_idx, f_poly, p, order, max_tries=5000)
        if a_e is None:
            continue
        collected_rG.append(Integer(r_e) % order)
        collected_a.append([to_int(x) % order for x in a_e])
    
    m = len(collected_a)
    if m < needed:
        raise ValueError(f"Kernel dim = {kdim}, need {needed} anchored G-decompositions but only collected {m} after {tries} tries.")
    
    A = matrix(Zmod(order), m, kdim, 
               lambda i,j: sum((collected_a[i][t] * to_int(basis[j][t])) for t in range(len(collected_a[i]))) % order)
    
    rvec = [to_int(x) for x in collected_rG]
    Aug = matrix(Zmod(order), m, kdim+1,
                 lambda i,j: A[i,j] if j < kdim else (-rvec[i]))
    
    KerAug = Aug.right_kernel()
    if KerAug.dimension() == 0:
        raise ValueError("No nontrivial solution for (t, lambda) found; check collected decompositions.")
    
    sol = KerAug.basis()[0]
    t_sol = sol[:kdim]
    lambda_sol = sol[kdim]
    
    if int(lambda_sol) % order == 0:
        raise ValueError("Found solution with lambda = 0 mod order; degenerate anchor.")
    
    Lvec = [sum(to_int(t_sol[j]) * to_int(basis[j][i]) for j in range(kdim)) % order 
            for i in range(len(basis[0]))]
    
    for row in dense_rows:
        if sum((to_int(row[i]) * Lvec[i]) for i in range(len(Lvec))) % order != 0:
            raise ValueError("Constructed Lvec is not in kernel; linear system inconsistent.")
    
    r_G, a_vec = G_relation
    r_Q, b_vec = Q_relation
    S_G = sum(to_int(a_vec[i]) * Lvec[i] for i in range(len(a_vec))) % order
    S_Q = sum(to_int(b_vec[i]) * Lvec[i] for i in range(len(b_vec))) % order
    
    S_Gi = int(S_G) % order
    g = gcd(S_Gi, order)
    if g != 1:
        raise ValueError(f"S_G={S_Gi} not invertible mod order (gcd={g}) after solving for coefficients. Degeneracy.")
    
    inv_SG = pow(S_Gi, -1, order)
    d = (Integer(r_G) * Integer(S_Q) * Integer(inv_SG) - Integer(r_Q)) % order
    return int(d)

def perform_dlp_attack(G, Q, smooth_divisors, p, f_coeffs, order, verbose=True):
    """
    Index Calculus DLP attack for genus 2 HECC over F_p.
    
    Strategy:
    1. Extract factor base from smooth divisors
    2. Build relation matrix and compute kernel dimension
    3. Find smooth decompositions for [r_G]G and Q + [r_Q]G
    4. If kdim == 1, use simple ratio formula
    5. If kdim > 1, collect extra G-decompositions and solve augmented system
    6. Verify result by checking d*G == Q
    """
    R = PolynomialRing(GF(p), 'x')
    x = R.gen()
    f_poly = sum(GF(p)(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    if verbose:
        print(f"Extracting factor base from {len(smooth_divisors)} divisors...")
    fb = extract_factor_base(smooth_divisors, p, verbose=False)
    root_list = sorted(list(fb['roots']))
    root_to_idx = {r: i for i, r in enumerate(root_list)}
    
    if len(root_list) < 10:
        raise ValueError(f"Factor base too small: {len(root_list)} roots. Need more smooth divisors.")
    
    if verbose:
        print("Constructing relation matrix...")
    matrix_rows = []
    for d in fb['unique_divisors']:
        row = get_relation_row(d, root_to_idx, f_poly, p)
        if row:
            matrix_rows.append(row)
    
    if not matrix_rows:
        raise ValueError("No valid relation rows constructed.")
    
    if verbose:
        print("Finding smooth decomposition for [r_G]G...")
    r_G, row_G = find_smooth_decomposition(G.parent()(0), G, root_to_idx, f_poly, p, order, max_tries=10000)
    
    if row_G is None:
        raise ValueError("Could not decompose [r_G]G over factor base.")
    
    if verbose:
        print("Finding smooth decomposition for Q + [r_Q]G...")
    r_Q, row_Q = find_smooth_decomposition(Q, G, root_to_idx, f_poly, p, order, max_tries=10000)
    
    if row_Q is None:
        raise ValueError("Could not decompose Q + [r_Q]G over factor base.")
    
    if verbose:
        print("Solving for log_G(Q)...")
    
    G_relation = (r_G, row_G)
    Q_relation = (r_Q, row_Q)
    
    d = solve_dlp_from_relations(
        matrix_rows, G_relation, Q_relation, order,
        generator=G, root_to_idx=root_to_idx, f_poly=f_poly, p=p,
        max_extra=16, verbose=verbose
    )
    
    if verbose:
        print(f"Computed discrete log: d = {d}")
        print("Verifying result...")
    
    computed_Q = d * G
    if computed_Q == Q:
        if verbose:
            print("✓ Verification passed: d*G == Q")
        return d
    else:
        raise ValueError(f"Verification FAILED: d*G != Q. Attack produced incorrect result.")


def generate_test_keypair(f_poly, p, target_d=None):
    """
    Generates a challenge keypair (G, Q) where Q = [d]G.
    Works within the finite field GF(p).
    """
    Fp = GF(p)
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    G = J.random_element()
    
    if target_d is None:
        import secrets
        L_approx = p**2 
        target_d = secrets.randbelow(int(L_approx))
    
    Q = target_d * G
    
    return G, Q, target_d
