from sage.all import GF, PolynomialRing, HyperellipticCurve, factor, Integer, QQ
from time import sleep
from math import ceil, sqrt

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

# Put at top of file (if not already imported)

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
def solve_dlp_mod_l_block_wiedemann(
    valid_rows,
    rhs_values,
    row_q_dict,
    beta_q,
    full_order,
    G, Q,
    *,
    verbose=False,
    block_size=32,
    nprocs=None,
    max_iters=None
):
    """
    Solve discrete log using your sparse Block-Wiedemann core, under the assumption
    we want to operate in the prime-order subgroup J(F_p)[ell].

    Differences from the previous version:
      - We do NOT multiply relation coefficients by the cofactor `h`.
      - We verify with exact equality d*G == Q (because G and Q should already be in the ℓ-subgroup).
      - If G/Q are not in ℓ-subgroup, we project them by h (defensive).
    Returns:
      full integer discrete log d (0 <= d < ell) such that d*G == Q.
    """
    if nprocs is None:
        from multiprocessing import cpu_count
        nprocs = max(1, cpu_count() - 1)

    if verbose:
        print(f"  [BW] block_size={block_size}, nprocs={nprocs}")
        sys.stdout.flush()

    # Compute ell and h (largest prime factor)
    J_order = Integer(full_order)
    facs = factor(J_order)
    ell = int(max(int(p) for p, _ in facs))
    h = int(J_order // ell)

    if verbose:
        print(f"  [BW] Computed Jacobian order factors: ell={ell}, cofactor h={h}")
        sys.stdout.flush()

    # Ensure G and Q are in the ell-subgroup (or project them)
    ell, h, G_used, Q_used = ensure_prime_subgroup_elements(G, Q, full_order, verbose=verbose)

    # Convert relations INTO mod ell (no cofactor multiplication)
    projected_rows = []
    projected_rhs = []

    for row, rhs in zip(valid_rows, rhs_values):
        # Coerce coefficients into ints modulo ell
        new_row = {}
        for k, v in row.items():
            vk = int(Integer(v) % Integer(ell))
            if vk:
                new_row[k] = vk
        rhs_proj = int(Integer(rhs) % Integer(ell))
        # If relation is 0 = nonzero (mod ell), that's inconsistent
        if not new_row:
            if rhs_proj % ell != 0:
                raise ValueError(f"Block-Wiedemann: inconsistent zero-row: 0 == {rhs_proj} (mod {ell})")
            # else redundant relation, skip
            continue
        projected_rows.append(new_row)
        projected_rhs.append(rhs_proj)

    if not projected_rows:
        raise ValueError("Block-Wiedemann: no nonzero relations after projection (unexpected)")

    n_cols = max(k for row in projected_rows for k in row) + 1

    if verbose:
        nnz = sum(len(r) for r in projected_rows)
        print(f"  [BW] Sparse system: {len(projected_rows)} x {n_cols}, nnz={nnz}")
        sys.stdout.flush()

    # Build SparseRelationMatrix (expects modulus=ell)
    A = SparseRelationMatrix(projected_rows, projected_rhs, ell)

    if verbose:
        print("  [BW] Starting Krylov iterations...")
        sys.stdout.flush()

    t0 = time.time()
    solution = block_wiedemann_solve(
        A=A,
        b=projected_rhs,
        block_size=block_size,
        iters=max_iters,
        verbose=verbose,
    )
    dt = time.time() - t0

    if solution is None:
        raise RuntimeError("Block-Wiedemann failed to converge (returned None)")

    if verbose:
        print(f"  [BW] Solved in {dt:.2f}s")
        sys.stdout.flush()

    # Reconstruct discrete log modulo ell
    beta_q_l = int(Integer(beta_q) % Integer(ell))
    row_q_l = {k: int(Integer(v) % Integer(ell)) for k, v in row_q_dict.items()}

    dlog = Integer(beta_q_l)
    for k, v in row_q_l.items():
        coeff = int(solution[k])  # solution is a vector over Zmod(ell); int() coerces
        dlog = (dlog - Integer(v) * Integer(coeff)) % Integer(ell)
    dlog = int(dlog)

    if verbose:
        print("  [BW] Candidate d (mod ell):", dlog)
        sys.stdout.flush()

    # Verify exact equality (since G_used and Q_used are in ell-subgroup)
    D = Integer(dlog) * G_used - Q_used
    if not D.is_zero():
        # This is a fatal error for Model A (we expect exact equality)
        raise AssertionError(
            "[BW-Verify] Verification failed: d * G != Q in prime subgroup\n"
            f"  candidate d (mod ell) = {dlog}\n"
            f"  ell = {ell}, h = {h}\n"
            "  This indicates relation construction or linear algebra corruption."
        )

    if verbose:
        print("  [Verify] ✓ Exact equality d*G == Q (prime subgroup).")
    return int(dlog)

# ---------------------------------------------------------------------
# Optional helper (kept for completeness): BSGS lift in case you ever need to lift
# ---------------------------------------------------------------------
def lift_discrete_log_via_bsgs(d_mod_ell, ell, h, G, Q, verbose=False):
    """
    Attempt to lift d = d_mod_ell (mod ell) to full d = d_mod_ell + t*ell, 0 <= t < h,
    solving t*(ell*G) = Q - d_mod_ell*G using baby-step giant-step.
    Returns full d if found, or None if not found.
    (Not used in Model A where G,Q are already in ell-subgroup.)
    """
    # Compute correction target R = Q - d_mod_ell * G
    R = Q - Integer(d_mod_ell) * G
    if R.is_zero():
        if verbose:
            print("[lift] Already exact: d_mod_ell is the full discrete log.")
        return int(d_mod_ell)

    H = Integer(ell) * G
    if H.is_zero():
        if verbose:
            print("[lift] ell * G is zero (degenerate); cannot lift.")
        return None

    bound = int(h)
    m = int(ceil(sqrt(bound)))
    if verbose:
        print(f"[lift] BSGS parameters: bound={bound}, m={m}")

    # baby steps
    baby = {}
    cur = Integer(0) * H
    for j in range(m):
        key = str(cur)  # use string representation as hashable key
        if key not in baby:
            baby[key] = j
        cur = cur + H

    # giant steps
    factor = Integer(m) * H
    giant = R
    for i in range(0, m + 1):
        key = str(giant)
        if key in baby:
            j = baby[key]
            t = i * m + j
            if t < bound:
                full_d = int(Integer(d_mod_ell) + Integer(t) * Integer(ell))
                if full_d * G == Q:
                    if verbose:
                        print(f"[lift] Found lift: t={t}, full_d={full_d}")
                    return full_d
        giant = giant - factor

    if verbose:
        print("[lift] BSGS failed to find a lift in [0,h).")
    return None

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

    # Extract x-coordinates (now guaranteed to exist since u(x) splits)
    preferred_x_coords = set()
    for D in [G, Q]:
        u_poly = D[0]
        for root, _ in u_poly.roots():
            preferred_x_coords.add(int(root))

    assert len(preferred_x_coords) == 4, f"Expected 4 x-coords, got {len(preferred_x_coords)}: {preferred_x_coords}"

    if verbose:
        print(f"Generated {len(preferred_x_coords)} preferred x-coordinates: {preferred_x_coords}")
        print(f"G has u(x) = {G[0]} (splits)")
        print(f"Q has u(x) = {Q[0]} (splits)")

    return ell, base_pts_x, G, Q, preferred_x_coords, final_secret
