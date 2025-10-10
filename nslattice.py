# Run inside Sage (sage -python or a Sage notebook)
from sage.all import (
    matrix, vector, QQ, ZZ, sqrt, ceil, floor, gcd, Integer, Matrix
)
import itertools, math
from search_common import *
from yau import *

# --- NEW: find isotropic fibrations and test height improvements ---
from sage.all import gcd as _gcd

def _is_primitive_tuple(tpl):
    """Return True if integer tuple tpl is primitive (gcd == 1)."""
    g = 0
    for x in tpl:
        g = _gcd(g, abs(int(x)))
    return g == 1

def find_isotropic_fibration_candidates(cd, current_sections, rho, mw_rank, chi, max_coeff=None, Hvec=None):
    """
    Search small integer boxes for tuples a = (a1,...,ak) such that

        v = S + sum_i a_i * phi(P_i)

    satisfies Q(v) == 0 (isotropic), is primitive, and v·H > 0.

    Returns: list of dicts with keys
      'a'      : tuple of ints (the coefficients)
      'v'      : Matrix(QQ,n,1) the NS column vector for v
      'Qv'     : exact QQ value of Q(v) (should be 0)
      'v_dot_H': exact QQ value of v·H

    Notes:
      - uses the same sections_to_ns_vectors and solve_shioda_image helpers in this module.
      - max_coeff controls search box [-max_coeff..max_coeff]^k; keep small (2..5).
    """
    assert 'sections_to_ns_vectors' in globals(), "sections_to_ns_vectors must be present."
    basis_labels, Q, h_vec = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    n = Q.nrows()

    # S and F (original) as column Matrices
    S_vec = Matrix(ZZ, n, 1, [1 if i == 0 else 0 for i in range(n)])
    F_vec_old = Matrix(ZZ, n, 1, [1 if i == 1 else 0 for i in range(n)])

    # get φ(P_i) vectors using existing routine
    sect_vecs = sections_to_ns_vectors(cd, current_sections, rho, mw_rank, chi)
    if not sect_vecs:
        raise ValueError("No section vectors found.")
    sect_mats = [ Matrix(QQ, n, 1, v) for v in sect_vecs ]

    # create Theta_vecs for reducible fibers (same logic as compute_search_vectors_hodge)
    Theta_vecs = []
    fibers = cd.singfibs.get('fibers', [])
    for idx in range(2, n):
        label = basis_labels[idx]
        if label.startswith('fib'):
            try:
                parts = label.split('_')
                fib_idx = int(parts[0][3:])
                if fib_idx < len(fibers):
                    m_v = fibers[fib_idx].get('m_v', 1)
                    if m_v > 1:
                        Theta_vecs.append(Matrix(ZZ, n, 1, [1 if i == idx else 0 for i in range(n)]))
            except (ValueError, IndexError):
                continue

    # original φ vectors (w.r.t. original F)
    P_phi_vecs = []
    for sect in sect_mats:
        phi = solve_shioda_image(sect, Q, S_vec, F_vec_old, Theta_vecs)
        P_phi_vecs.append(phi)

    # H vector to use
    if Hvec is None:
        H_use = Matrix(QQ, n, 1, [h_vec[i] for i in range(n)])
    else:
        H_use = Matrix(QQ, n, 1, [Hvec[i] for i in range(n)])

    # Build exact ingredients for enumerating v = S + sum a_i * phi_i
    Pmats = [ Matrix(QQ, p) for p in P_phi_vecs ]
    c0_exact = (S_vec.transpose() * Q * S_vec)[0,0]
    b_exact = [ (Pmats[i].transpose() * Q * S_vec)[0,0] for i in range(len(Pmats)) ]
    M_exact = Matrix(QQ, len(Pmats), len(Pmats), lambda i,j: (Pmats[i].transpose() * Q * Pmats[j])[0,0])

    PH_exact = [ (Pmats[i].transpose() * Q * H_use)[0,0] for i in range(len(Pmats)) ]
    QH_exact = (H_use.transpose() * Q * H_use)[0,0]

    k = len(Pmats)
    results = []
    if max_coeff is None:
        if chi != 1:
            # K3 surface: try provable Gershgorin radius
            Rprov = _provable_radius_gershgorin(Q, S_vec, P_phi_vecs, H_use)
            max_coeff = ceil(Rprov)
        else:
            # RES: use a small reasonable box
            max_coeff = 100

    rng = [ range(-max_coeff, max_coeff+1) for _ in range(k) ]
    for tpl in itertools.product(*rng):
        if not _is_primitive_tuple(tpl):
            continue

        # Q(v) = c0 + 2 a^T b + a^T M a
        two_aTb = 2 * sum(Integer(tpl[i]) * b_exact[i] for i in range(k))
        aMa = Integer(0)
        for i in range(k):
            for j in range(k):
                aMa += Integer(tpl[i]) * Integer(tpl[j]) * M_exact[i,j]
        Qv_exact = c0_exact + two_aTb + aMa

        if Qv_exact != 0:
            continue

        # v·H
        vQH_exact = (S_vec.transpose() * Q * H_use)[0,0] + sum(Integer(tpl[i]) * PH_exact[i] for i in range(k))
        if vQH_exact <= 0:
            # require positive intersection with ample H
            continue

        # build the actual v column vector
        v = S_vec.copy()
        for i in range(k):
            v = v + Integer(tpl[i]) * Pmats[i]

        results.append({
            'a': tuple(int(x) for x in tpl),
            'v': v,
            'Qv': QQ(Qv_exact),
            'v_dot_H': QQ(vQH_exact),
        })

    return results


def evaluate_fibration_height_reduction(cd, current_sections, rho, mw_rank, chi, candidates=None, max_coeff=3, shioda_sign=-1):
    """
    For every isotropic candidate (or searches if candidates None), compute the
    canonical heights of `current_sections` before and after switching the
    fiber class to that candidate.

    Returns list of dicts:
      {
        'a': tuple(...)           # coefficients producing candidate v
        'v': Matrix(...)          # the NS vector for candidate fiber
        'heights_old': [...],     # canonical heights (floats or QQ) per section (old fibration)
        'heights_new': [...],     # canonical heights with new fiber used in solve_shioda_image
        'delta': [old-new,...],   # per-section height differences (positive means reduction)
        'total_reduction': sum(old-new)
      }

    Parameters:
      - shioda_sign: multiply result by this sign. Default -1 matches
        hat{h} = - (phi^T Q phi)/2. If your code uses +, pass +1.
    """
    basis_labels, Q, h_vec = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    n = Q.nrows()
    S_vec = Matrix(ZZ, n, 1, [1 if i == 0 else 0 for i in range(n)])
    F_vec_old = Matrix(ZZ, n, 1, [1 if i == 1 else 0 for i in range(n)])

    # Theta components as before
    Theta_vecs = []
    fibers = cd.singfibs.get('fibers', [])
    for idx in range(2, n):
        label = basis_labels[idx]
        if label.startswith('fib'):
            try:
                parts = label.split('_')
                fib_idx = int(parts[0][3:])
                if fib_idx < len(fibers):
                    m_v = fibers[fib_idx].get('m_v', 1)
                    if m_v > 1:
                        Theta_vecs.append(Matrix(ZZ, n, 1, [1 if i == idx else 0 for i in range(n)]))
            except (ValueError, IndexError):
                continue

    # Build φ vectors w.r.t. old fibration for 'old heights'
    sect_vecs = sections_to_ns_vectors(cd, current_sections, rho, mw_rank, chi)
    sect_mats = [ Matrix(QQ, n, 1, v) for v in sect_vecs ]
    P_phi_old = [ solve_shioda_image(sect, Q, S_vec, F_vec_old, Theta_vecs) for sect in sect_mats ]

    def canonical_height_from_phi(phi):
        # \hat h = (shioda_sign) * (phi^T Q phi) / 2
        val = (phi.transpose() * Q * phi)[0,0]
        return QQ(shioda_sign) * QQ(val) / QQ(2)

    heights_old = [ canonical_height_from_phi(p) for p in P_phi_old ]

    # gather candidates if not provided
    if candidates is None:
        candidates = find_isotropic_fibration_candidates(cd, current_sections, max_coeff=max_coeff, Hvec=h_vec)

    outputs = []
    for cand in candidates:
        v = cand['v']            # this is the candidate fiber class (column Matrix)
        # sanity: v·v == 0
        assert (v.transpose() * Q * v)[0,0] == 0

        # compute new φ with F_vec = v (we keep same S_vec and Theta_vecs)
        P_phi_new = []
        ok = True
        try:
            for sect in sect_mats:
                phi_new = solve_shioda_image(sect, Q, S_vec, v, Theta_vecs)
                P_phi_new.append(phi_new)
        except Exception as e:
            # If Shioda projection fails for this candidate, skip it
            ok = False

        if not ok or len(P_phi_new) != len(sect_mats):
            continue

        heights_new = [ canonical_height_from_phi(p) for p in P_phi_new ]
        deltas = [ QQ(h_old - h_new) for (h_old, h_new) in zip(heights_old, heights_new) ]
        total_reduction = sum(deltas)

        outputs.append({
            'a': cand['a'],
            'v': v,
            'heights_old': heights_old,
            'heights_new': heights_new,
            'delta': deltas,
            'total_reduction': QQ(total_reduction),
        })

    # sort descending by total_reduction (largest improvement first)
    outputs.sort(key=lambda d: float(d['total_reduction']), reverse=True)
    return outputs

# --- END NEW ---


def _provable_radius_gershgorin(Q, S_vec, P_vecs, Hvec):
    """
    Return provable Euclidean radius R (float) such that any integer tuple a
    with ||a||_2 > R cannot satisfy the Hodge inequality.

    Uses exact rational Gershgorin lower bound for M = (P_i^T Q P_j).
    Raises ValueError if a finite provable bound cannot be certified.
    """
    # Q, S_vec, P_vecs, Hvec are Matrices over QQ
    k = len(P_vecs)
    assert k >= 1

    # Build exact rational M (k x k)
    M_exact = Matrix(QQ, k, k, lambda i, j: (Matrix(QQ, P_vecs[i]).transpose() * Q * Matrix(QQ, P_vecs[j]))[0,0])

    # Gershgorin lower bounds (exact rational arithmetic)
    gersh_list = []
    for i in range(k):
        Mii = M_exact[i, i]
        row_sum = QQ(0)
        for j in range(k):
            if j == i:
                continue
            row_sum += abs(M_exact[i, j])
        gersh_list.append(Mii - row_sum)

    # lambda_min is min of Gershgorin bounds
    lambda_min_rational = min(gersh_list)
    if lambda_min_rational <= 0:
        raise ValueError("Cannot certify positive definiteness of M via Gershgorin (lambda_min <= 0).")

    lambda_min = float(lambda_min_rational)  # use float for quadratic roots

    # Compute exact rational quantities and convert to float for root solving
    S_vec = Matrix(QQ, S_vec)
    Hvec = Matrix(QQ, Hvec)
    Pmats = [Matrix(QQ, p) for p in P_vecs]

    c0_rational = (S_vec.transpose() * Q * S_vec)[0,0]
    b_rational = [ (Pmats[i].transpose() * Q * S_vec)[0,0] for i in range(k) ]
    b_norm_sq_rational = sum(b*b for b in b_rational)

    S_H_rational = (S_vec.transpose() * Q * Hvec)[0,0]
    PH_rational = [ (Pmats[i].transpose() * Q * Hvec)[0,0] for i in range(k) ]
    gamma_sq_rational = sum(p*p for p in PH_rational)
    QH_rational = (Hvec.transpose() * Q * Hvec)[0,0]

    # Convert to floats for solving quadratic (exact detection conditions were done above)
    b_norm = math.sqrt(float(b_norm_sq_rational))
    S_H = float(S_H_rational)
    gamma = math.sqrt(float(gamma_sq_rational))
    QH = float(QH_rational)
    c0 = float(c0_rational)

    if QH <= 0.0:
        raise ValueError("Q(H) is non-positive; cannot use H as ample vector in Hodge bound.")

    # Quadratic coefficients A,B,C for r variable (r = ||a||_2)
    A = lambda_min - (gamma*gamma) / QH
    B = -2.0 * b_norm - 2.0 * S_H * gamma / QH
    C = c0 - (S_H * S_H) / QH

    if A <= 0.0:
        raise ValueError("Quadratic coefficient A <= 0; inequality does not provide finite bound (A={}).".format(A))

    disc = B*B - 4.0*A*C
    if disc < 0.0:
        raise ValueError("Discriminant negative; cannot produce finite provable radius (disc < 0).")

    sqrt_disc = math.sqrt(disc)
    r1 = (-B - sqrt_disc) / (2.0 * A)
    r2 = (-B + sqrt_disc) / (2.0 * A)
    R = max(r1, r2)
    if R < 0.0:
        R = 0.0
    return float(R)


def compute_search_vectors_hodge(cd, current_sections, rho, mw_rank, chi, Hvec=None):
    """
    Main function: from cd and a list of current_sections (Weierstrass points),
    produce integer coefficient tuples a = (a1,...,ak) for MW combinations

        SUM = sum_i a_i * φ(P_i)

    that pass the exact linear constraint v·(Q*F) == 1 (for v = S + Σ a_i φ(P_i))
    and the exact Hodge inequality Q(v)*Q(H) <= (v·H)^2.

    Returns:
        candidates : list of integer tuples (a1,...,ak), sorted by l2-norm

    Raises:
        ValueError if the provable bounding quadratic cannot certify a finite box.
        ValueError / AssertionError for missing helper functions or singular systems.
    """
    # Retrieve NS basis and Q
    basis_labels, Q, h_vec = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    n = Q.nrows()

    # Build basis unit vectors for S, F, and fiber components
    S_vec = Matrix(ZZ, n, 1, [1 if i == 0 else 0 for i in range(n)])
    F_vec = Matrix(ZZ, n, 1, [1 if i == 1 else 0 for i in range(n)])
    
    # Only include fiber components for actually reducible fibers (m_v > 1)
    Theta_vecs = []
    fibers = cd.singfibs.get('fibers', [])
    
    for idx in range(2, n):
        label = basis_labels[idx]
        if label.startswith('fib'):
            # Parse fib{i}_c{j} to get fiber index i
            try:
                parts = label.split('_')
                fib_idx = int(parts[0][3:])  # extract i from "fibi"
                
                # Check if this fiber actually has m_v > 1 (is reducible)
                if fib_idx < len(fibers):
                    m_v = fibers[fib_idx].get('m_v', 1)
                    if m_v > 1:  # only include for truly reducible fibers
                        Theta_vecs.append(Matrix(ZZ, n, 1, [1 if i == idx else 0 for i in range(n)]))
            except (ValueError, IndexError):
                # If parsing fails, skip this component
                continue
    
    print(f"DEBUG: n={n}, basis_labels={basis_labels}")
    print(f"DEBUG: len(Theta_vecs)={len(Theta_vecs)} (after filtering)")
    print(f"DEBUG: Fiber m_v values: {[f.get('m_v', 1) for f in fibers]}")
    print(f"DEBUG: S_vec.T = {S_vec.transpose()}")
    print(f"DEBUG: F_vec.T = {F_vec.transpose()}")

    # Map input sections -> NS section-class vectors via caller helper
    # (caller must provide sections_to_ns_vectors)
    if 'sections_to_ns_vectors' not in globals():
        raise ValueError("sections_to_ns_vectors(cd, sections, rho, mw_rank, chi) not found in global scope; please provide it.")
    sect_vecs = sections_to_ns_vectors(cd, current_sections, rho, mw_rank, chi)
    if not sect_vecs:
        raise ValueError("No section-class vectors produced by sections_to_ns_vectors; nothing to do.")

    # Convert to Matrix(QQ, n,1) - ensure column vectors
    sect_vecs = [ Matrix(QQ, n, 1, v) for v in sect_vecs ]

    # Compute Shioda images φ(P) exactly for each section
    P_phi_vecs = []
    for idx, sect in enumerate(sect_vecs):
        phi = solve_shioda_image(sect, Q, S_vec, F_vec, Theta_vecs)
        # sanity check: phi·F == 0 and phi·Theta == 0 (enforced in solver)
        P_phi_vecs.append(phi)

    # If user provided explicit Hvec, use it; else use h_vec from build_ns_basis_and_Q
    if Hvec is None:
        Hvec_use = h_vec
    else:
        Hvec_use = Matrix(QQ, Hvec)

    # Now enumerate candidate integer tuples
    k = len(P_phi_vecs)
    if k == 0:
        return []

    # compute provable Euclidean radius R
    Rprov = _provable_radius_gershgorin(Q, S_vec, P_phi_vecs, Hvec_use)
    Bbox = int(ceil(Rprov))

    # Precompute exact quantities for checks
    Pmats = [ Matrix(QQ, p) for p in P_phi_vecs ]
    c0_exact = (S_vec.transpose() * Q * S_vec)[0,0]
    b_exact = [ (Pmats[i].transpose() * Q * S_vec)[0,0] for i in range(k) ]
    M_exact = Matrix(QQ, k, k, lambda i,j: (Pmats[i].transpose() * Q * Pmats[j])[0,0])

    S_Q_H_exact = (S_vec.transpose() * Q * Hvec_use)[0,0]
    PH_exact = [ (Pmats[i].transpose() * Q * Hvec_use)[0,0] for i in range(k) ]
    QH_exact = (Hvec_use.transpose() * Q * Hvec_use)[0,0]

    # linear check helper: compute (S + Σ a_i P_i) · (Q*F) exactly and require == 1
    QF_vec = Q * F_vec
    S_Q_F_exact = (S_vec.transpose() * Q * F_vec)[0,0]
    PF_exact = [ (Pmats[i].transpose() * Q * F_vec)[0,0] for i in range(k) ]

    # Enumerate all integer tuples in box [-Bbox..Bbox]^k
    candidates = []
    rng = [ range(-Bbox, Bbox+1) for _ in range(k) ]
    for tpl in itertools.product(*rng):
        # exact linear constraint v·(Q*F) == 1
        vQF = S_Q_F_exact
        for i in range(k):
            vQF += Integer(tpl[i]) * PF_exact[i]
        if vQF != 1:
            continue

        # exact Q(v) = c0 + 2 a^T b + a^T M a
        two_aTb = 2 * sum(Integer(tpl[i]) * b_exact[i] for i in range(k))
        aMa = Integer(0)
        for i in range(k):
            for j in range(k):
                aMa += Integer(tpl[i]) * Integer(tpl[j]) * M_exact[i,j]
        Qv_exact = c0_exact + two_aTb + aMa

        # exact v·H
        vQH_exact = S_Q_H_exact + sum(Integer(tpl[i]) * PH_exact[i] for i in range(k))

        # exact Hodge check: Q(v)*Q(H) <= (v·H)^2
        lhs = QQ(Qv_exact) * QQ(QH_exact)
        rhs = QQ(vQH_exact)**2
        if lhs > rhs:
            continue

        candidates.append(tuple(int(x) for x in tpl))

    # sort by Euclidean length (small -> large)
    candidates.sort(key=lambda t: sum(x*x for x in t))
    return candidates
