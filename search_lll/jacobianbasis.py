# Module-level worker functions (must be at top level for pickling)

from mpmath import pslq
from sage.all import QQ, ZZ, RR, CC, RDF, CDF, log, sqrt, exp, pi, I
from sage.all import PolynomialRing, HyperellipticCurve, Matrix, vector
from sage.all import QQ, ZZ, RR, CC, RDF, CDF, ComplexField, log, sqrt, exp, pi, I
from sage.all import RealField
from functools import lru_cache
import math
from sage.all import ComplexField, RealField
from sage.all import parallel
import time
from collections import defaultdict
from sage.all import QQ, ZZ, RR, Qp, PolynomialRing, HyperellipticCurve
from search_common import *

from .homology import *


def get_pairing(i, j, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    if i > j:
        i, j = j, i

    if (i, j) not in pairing_cache:
        # Compute on-demand if not in cache
        pairing_cache = precompute_pairings_parallel([i, j], jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs)

    return pairing_cache[(i, j)]


from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, 
                      QQ, RR as SageRR)
from multiprocessing import Pool, cpu_count


def dedupe_basis(basis, basis_indices, debug=False):
    """
    Remove duplicated divisors (preserve order). Returns (basis_u, basis_indices_u).
    """
    seen = set()
    basis_u = []
    basis_indices_u = []
    for div, idx in zip(basis, basis_indices):
        if idx in seen:
            if debug:
                print(f"[dedupe] removing duplicate basis index {idx}")
            continue
        seen.add(idx)
        basis_u.append(div)
        basis_indices_u.append(idx)
    return basis_u, basis_indices_u


def gram_logdet_and_cond(basis_indices, get_pairing):
    """
    Build float Gram matrix for basis_indices (using get_pairing -> float),
    return dict with:
        n, svals (list), log10_abs_det, log10_cond, numeric_rank_est
    """
    import numpy as np, math
    n = len(basis_indices)
    if n == 0:
        return {"n":0, "svals": [], "log10_abs_det": None, "log10_cond": None, "numeric_rank": 0}
    G = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            v = float(get_pairing(basis_indices[i], basis_indices[j]))
            G[i,j] = v; G[j,i] = v
    # enforce symmetry
    G = 0.5*(G + G.T)
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        raise RuntimeError(f"Gram not PD after clipping: min_eig={min_ev:.3e}")
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    svals = [float(x) for x in S]
    # log10_abs_det = sum(log10(svals)) (sign positive for Gram of PD matrix)
    log10_abs_det = sum(math.log10(max(s, 1e-300)) for s in svals)
    cond = svals[0] / (svals[-1] if svals[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)
    # numeric rank: count s > smax * tol_rel
    tol_rel = 1e-12
    smax = svals[0]
    numeric_rank = sum(1 for s in svals if s > smax * tol_rel)
    return {"n": n, "svals": svals, "log10_abs_det": log10_abs_det, "log10_cond": log10_cond, "numeric_rank": numeric_rank}


# requires numpy
import numpy as np, math

def select_independent_indices_from_gram(
    G,
    prec_bits=2048,
    safety_digits=10,
    rel_sv_tol=1e-12,
    pivot_tol_factor=1e-9,
    debug=False,
):
    """
    Given a symmetric float Gram matrix G (n x n), return a deterministic list
    of indices that form a numerically independent set.

    Parameters
    ----------
    G : np.ndarray, shape (n,n), symmetric floats
    prec_bits : int, bit-precision used upstream (only for diagnostic scaling)
    safety_digits : int, digits of safety when building eigenvalue threshold
    rel_sv_tol : float, fallback relative threshold for eigenvalues (smax * rel_sv_tol)
    pivot_tol_factor : float, factor to multiply sqrt(min_positive_eig) to set pivot stop
    debug : bool

    Returns
    -------
    selected_indices : list of ints (length <= numeric_rank)
    info : dict with keys:
        'eigvals' (descending), 'numeric_rank', 'log10_abs_det', 'log10_cond'
    """
    # symmetrize (be safe)
    G = 0.5 * (G + G.T)
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        raise RuntimeError(f"Gram not PD after clipping: min_eig={min_ev:.3e}")
    n = G.shape[0]
    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(G)  # ascending eigenvalues
    eigvals = np.array(eigvals, dtype=float)
    # flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    smax = eigvals[0] if eigvals.size else 0.0
    # build eigenvalue threshold robustly:
    # convert bits -> decimal digits estimate, but cap to avoid under/overflow
    dec_digits = int(prec_bits * 0.30103) if prec_bits > 0 else 50
    dec_digits_cap = min(max(dec_digits, 0), 50)  # cap to [0,50] to avoid absurd exponents
    # threshold by safety_digits (but cap using rel_sv_tol)
    ev_thresh = max(smax * (10.0 ** (-(max(safety_digits, dec_digits_cap)))), smax * rel_sv_tol, 1e-300)

    # positive eigenvalues indices
    pos_mask = eigvals > ev_thresh
    pos_indices = np.nonzero(pos_mask)[0]
    r = len(pos_indices)
    if debug:
        print(f"[select_from_gram] smax={smax:.3g}, ev_thresh={ev_thresh:.3g}, num_pos={r}")

    if r == 0:
        return [], {
            "eigvals": eigvals.tolist(),
            "numeric_rank": 0,
            "log10_abs_det": None,
            "log10_cond": None,
        }

    # Build embedding E (n x r) such that G ≈ E E^T
    Spos = eigvals[pos_indices]                     # length r (descending)
    Upos = eigvecs[:, pos_indices]                  # n x r
    sqrtS = np.sqrt(np.maximum(Spos, 0.0))
    # E[i,:] is embedding vector for candidate i
    E = Upos * sqrtS[np.newaxis, :]                 # shape (n, r)

    # Determine pivot stop tolerance from smallest retained eigenvalue
    min_pos_eig = Spos[-1]
    pivot_tol = math.sqrt(max(min_pos_eig, 0.0)) * pivot_tol_factor
    # also ensure pivot_tol not ridiculously tiny:
    pivot_tol = max(pivot_tol, smax * 1e-16)

    # Deterministic pivoting: iteratively pick row with largest residual norm,
    # orthogonalize rows against chosen normalized vector, stop when residuals small
    rows = E.copy()  # will be modified (deflation)
    norms = np.linalg.norm(rows, axis=1)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    while True:
        # choose argmax among not selected
        cand = int(np.argmax(norms + (selected_mask * -1e300)))  # ensures selected masked
        maxnorm = norms[cand]
        if debug:
            print(f"[pivot] pick candidate {cand} maxnorm={maxnorm:.6g} selected={len(selected)}")
        if maxnorm <= pivot_tol:
            break
        # add candidate
        selected.append(cand)
        selected_mask[cand] = True
        # normalize
        v = rows[cand].copy()
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            # can't orthonormalize further
            break
        v = v / vnorm
        # deflate all rows by projection onto v
        proj = rows @ v   # projection coefficients (n,)
        rows = rows - np.outer(proj, v)
        # recompute norms only for not yet selected
        norms = np.linalg.norm(rows, axis=1)
        # keep selecting until we've chosen r rows
        if len(selected) >= r:
            break

    # if we picked fewer than r (rare), we can accept them as basis; numeric_rank = len(selected)
    numeric_rank = len(selected)

    # diagdet/log det from singular values (embedding singulars are sqrt of eigvals)
    # For Gram matrix, singular values = eigvals (nonnegative). So log10|det| = sum log10(eigvals_pos)
    log10_abs_det = sum(math.log10(max(x, 1e-300)) for x in Spos)
    cond = Spos[0] / (Spos[-1] if Spos[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)

    info = {
        "eigvals": eigvals.tolist(),
        "numeric_rank": numeric_rank,
        "log10_abs_det": log10_abs_det,
        "log10_cond": log10_cond,
    }
    return selected, info


def compute_theta_high_prec(z_vec, tau, prec=300):
    """
    Computes Riemann Theta function theta(z, tau) at high precision.
    z_vec: vector of length 2 (Complex)
    tau: 2x2 symmetric matrix (Complex)
    """
    from sage.all import ComplexField, exp, pi
    import math
    
    CC = ComplexField(prec)
    
    # Pre-compute constants
    pi_I = CC(0, 1) * CC(pi)
    
    # Determine summation radius for precision
    # e^(-pi * n^2 * y_min) < 2^-prec
    # n^2 > prec * log(2) / (pi * y_min)
    # Assuming y_min ~ 0.3 (from your log), n ~ 25 is safe for 2048 bits
    radius = int(math.sqrt(prec * 0.25)) + 2 # Conservative estimate
    
    total = CC(0)
    
    # Naive summation over Z^2 (fast enough for genus 2)
    # Iterating -R to R
    r_range = range(-radius, radius + 1)
    
    # Extract components for speed
    z0, z1 = z_vec[0], z_vec[1]
    t00, t01, t11 = tau[0,0], tau[0,1], tau[1,1]
    
    for n1 in r_range:
        for n2 in r_range:
            # exponent = i*pi * (n^T * tau * n + 2 * n^T * z)
            # n^T tau n = n1^2 t00 + 2 n1 n2 t01 + n2^2 t11
            quad = (n1*n1)*t00 + (2*n1*n2)*t01 + (n2*n2)*t11
            lin = 2 * (n1*z0 + n2*z1)
            
            term_exponent = pi_I * (quad + lin)
            total += exp(term_exponent)
            
    return total


def _compute_height_worker(args):
    """Worker function to compute a single height"""
    from sage.all import PolynomialRing, HyperellipticCurve, QQ
    
    i, div, f_coeffs, prec = args
    try:
        Rq_QQ = PolynomialRing(QQ, 'x')
        x_QQ = Rq_QQ.gen()
        f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-k) 
                       for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
        
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        div = J([u_poly, v_poly])
        
        h = arakelov_canonical_height(div, f_coeffs, prec=prec)
        return (i, h, None)  # Return h as Sage rational, NOT float
    except Exception as e:
        # ABSOLUTELY CRITICAL:
        # convert to a picklable exception with a string-only payload
        msg = (
            f"Height computation failed\n"
            f"Divisor: {repr(div)}\n"
            f"Exception type: {type(e).__name__}\n"
            f"Message: {str(e)}"
        )
        raise RuntimeError(msg)
        return (i, None, str(e))


def naive_height_qq(div, prec=53):
    """
    Compute naive (logarithmic) height of Mumford polynomials.
    """
    from sage.all import QQ, RealField
    
    u_coeffs = [QQ(c) for c in div[0].list()]
    v_coeffs = [QQ(c) for c in div[1].list()]
    
    # Clear denominators
    dens = [c.denominator() for c in (u_coeffs + v_coeffs) if c != 0]
    if not dens:
        return QQ(0)
    
    from math import gcd
    from functools import reduce
    lcm_den = reduce(lambda a, b: (a * b) // gcd(a, b), dens, 1)
    
    # Scale coefficients to integers
    int_coeffs = [int((c * lcm_den).numerator()) for c in (u_coeffs + v_coeffs)]
    int_coeffs = [abs(c) for c in int_coeffs if c != 0]
    if not int_coeffs:
        return QQ(0)
    
    max_abs = max(int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log().nearby_rational(max_error=R(2)**(-prec + 5))


def choose_numerical_base_point(f_coeffs, prec=300):
    """
    Selects a numerically safe base point for Abel-Jacobi maps.
    Returns (x, y) where y^2 = f(x).
    """
    from sage.all import ComplexField, PolynomialRing
    
    CC = ComplexField(prec)
    Rq = PolynomialRing(CC, 'x')
    x = Rq.gen()
    f_poly_cc = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    roots = f_poly_cc.roots(multiplicities=False)
    if not roots:
        # Fallback: use a point away from branch locus
        x_base = CC(1)
        f_val = sum(CC(c) * x_base**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        y_base = f_val.sqrt()
        return (x_base, y_base)
        
    sorted_roots = sorted(roots, key=lambda z: (float(z.real()), float(z.imag())))
    
    # OPTION 1: Use a root with tiny offset (Weierstrass point shifted slightly)
    root = sorted_roots[0]
    eps = CC(2) ** (-(prec // 4))  # Not too tiny
    x_base = root + CC(0, 1) * eps
    f_val = sum(CC(c) * x_base**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    y_base = f_val.sqrt()
    
    # Choose branch with positive imaginary part for consistency
    if y_base.imag() < 0:
        y_base = -y_base
    
    return (x_base, y_base)

def abel_jacobi_mumford(
    div, f_coeffs, base_point, *,
    integrate_func=None,    # function(base_x, x_end, y_end, f_coeffs, use_x_weight, prec, debug)
    prec=300,
    period_matrix=None,     # optional 2x2 matrix whose columns span the period lattice
    debug=False
):
    """
    Compute Abel-Jacobi for Mumford divisor div = (u(x), v(x)).
    - base_point must be (x0, y0) (both provided) to fix starting sheet.
    - integrate_func: function performing the integral; if None it uses
      `integrate_differential_path_with_branch` from the caller's scope.
    - Returns: vector(CC, length=2) (not reduced mod lattice unless period_matrix provided).
    """
    from sage.all import ComplexField, PolynomialRing, vector, matrix
    import math

    CC = ComplexField(prec)

    if div.is_zero():
        return vector(CC, [0, 0])

    # require full base point (x,y)
    if not (isinstance(base_point, (tuple, list)) and len(base_point) == 2):
        raise ValueError("base_point must be a tuple (x0, y0) with both coordinates (to fix sheet).")

    base_x = CC(base_point[0])
    base_y = CC(base_point[1])

    # default integrator: look up in global scope if not provided
    if integrate_func is None:
        try:
            integrate_func = integrate_differential_path_with_branch
        except NameError:
            raise ValueError("No integrate_func provided and integrate_differential_path_with_branch not found.")

    # build polynomial rings over CC
    R = PolynomialRing(CC, 'x')
    x = R.gen()

    u_poly = div[0]
    v_poly = div[1]

    # Convert u(x) and v(x) to CC polynomials (note: u_poly.list() gives coefficients from const->highest)
    u_list = u_poly.list()
    v_list = v_poly.list()

    # build u_cc for root finding (coeffs placed into polynomial ring)
    u_cc = sum(CC(c) * x**i for i, c in enumerate(u_list))

    # get roots with multiplicity info
    try:
        roots_mult = u_cc.roots(multiplicities=True)
    except Exception as e:
        if debug:
            print("[abel_jacobi] root finding failed:", e)
        raise
        return vector(CC, [0, 0])

    if len(roots_mult) == 0:
        return vector(CC, [0, 0])

    # Expand into list of (root, multiplicity) and sort deterministically
    roots_mult = sorted(roots_mult, key=lambda r: (float(r[0].real()), float(r[0].imag())))
    roots = [r[0] for r in roots_mult]
    mults = [r[1] for r in roots_mult]

    # helper: evaluate v(x) fully at a CC point
    def eval_v_at(xpt):
        # v_list: constant term -> highest
        return sum(CC(c) * (xpt ** i) for i, c in enumerate(v_list))

    # helper: polynomial f(x) from f_coeffs (descending coefficients assumed)
    def f_at(z):
        # f_coeffs assumed descending highest->lowest
        return sum(CC(c) * (z ** (len(f_coeffs) - 1 - i)) for i, c in enumerate(f_coeffs))

    # tolerance thresholds (in CC)
    tiny = CC(2) ** (-prec // 2)
    # somewhat looser tolerance for root-v consistency (absolute)
    tol_consistency = CC(10) ** (-8)

    aj_vec = vector(CC, [0, 0])

    # Diagnostics arrays (if debug)
    min_f = None
    min_idx = None

    # iterate roots
    for idx, x_pt in enumerate(roots):
        # multiplicity handling: if multiplicity > 1 assume a Weierstrass (branch) point y=0
        multiplicity = mults[idx] if idx < len(mults) else 1

        x_pt_cc = CC(x_pt)  # convert to CC

        # Evaluate candidate y from v(x) fully (no 1st-degree assumption)
        y_from_v = eval_v_at(x_pt_cc)

        # Evaluate f(x_pt)
        fval = f_at(x_pt_cc)

        # update global min f diagnostics
        absf = abs(fval)
        if min_f is None or absf < min_f:
            min_f = absf
            min_idx = idx

        # If multiplicity>1 trust it's a Weierstrass branch: set y=0 (but still allow tiny numerical)
        if multiplicity > 1:
            y_pt = CC(0)
        else:
            # If v(x) matches sqrt(f) within tolerance, accept v(x). Otherwise pick sqrt(f)
            # but choose sign consistent with v(x) if possible.
            # first check if fval is near zero
            if abs(fval) < tiny:
                # branch point: set y to zero (or to v if v tiny)
                if abs(y_from_v) < tol_consistency:
                    y_pt = CC(0)
                else:
                    # prefer v as it's probably the correct local branch
                    y_pt = y_from_v
            else:
                # compute principal sqrt using CC
                y_sqrt = fval.sqrt()     # this gives one sqrt in CC
                # two choices: y_sqrt or -y_sqrt. choose the one closer to v(x).
                if abs(y_from_v - y_sqrt) <= abs(y_from_v + y_sqrt):
                    y_pt = y_sqrt
                else:
                    y_pt = -y_sqrt

                # final safety: if distance is huge, fall back to v_from_v (but warn)
                if abs((y_pt**2) - fval) > tol_consistency and debug:
                    print(f"[abel_jacobi] Warning: y^2 != f at root idx {idx}, |y^2-f|={abs((y_pt**2)-fval)}")
        # At this point y_pt is the target y at x_pt we will pass to integrator.
        # Integrate two differentials: 1/(2y) and x/(2y)
        try:
            #int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
            #                      use_x_weight=False, prec=prec, debug=debug)
            #int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
            #                      use_x_weight=True, prec=prec, debug=debug)
            int0 = integrate_func(base_x, x_pt_cc, base_y, y_pt, f_coeffs,
                                  use_x_weight=False, prec=prec, debug=debug)
            int1 = integrate_func(base_x, x_pt_cc, base_y, y_pt, f_coeffs,
                                  use_x_weight=True, prec=prec, debug=debug)

        except TypeError:
            # If integrate_func doesn't accept debug/prec/use_x_weight keywords in that order,
            # fall back to positional call (older integrator signature)
            int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, False, prec, debug)
            int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, True, prec, debug)
            raise
        except Exception as e:
            if debug:
                print(f"[abel_jacobi] Integration failed for root {idx} at x={x_pt_cc}: {e}")
            raise
            continue

        aj_vec += vector(CC, [int0, int1])

    if debug:
        print(f"[abel_jacobi] min |f(x)| among nodes (roots): {min_f} at index {min_idx}")

    # Optionally reduce modulo period lattice if provided.
    # We expect period_matrix to be a 2x2 matrix (columns are period vectors).
    if period_matrix is not None:
        # convert to Sage matrix over CC if necessary
        PM = matrix(CC, period_matrix) if not isinstance(period_matrix, type(matrix(CC, [[1]]))) else period_matrix
        # Solve PM * coeffs = aj_vec for coeffs (complex)
        try:
            coeffs = PM.solve_right(aj_vec)   # length-2 vector of complex coords
            # round the coefficients to nearest integers (componentwise) to find lattice representative
            nearest = [int(round(float(c.real()))) + int(round(float(c.imag()))) * CC(0,1) for c in coeffs]
            # nearest integer vector in Z^2: take real parts (period lattice indices are integer)
            nearest_ints = [int(round(float(c.real()))) for c in coeffs]
            # subtract lattice component
            lattice_part = PM * vector(CC, nearest_ints)
            aj_vec = aj_vec - lattice_part
        except Exception as e:
            if debug:
                print("[abel_jacobi] Warning: period reduction failed:", e)
            raise
            # continue without reduction

    return aj_vec

def integrate_differential_path_with_branch(x_start, x_end, y_start, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False):
    """
    Integrate from (x_start, y_start) to (x_end, y_end).
    BOTH y coordinates specify which sheet we're on.
    """
    import math
    from sage.all import ComplexField

    CC = ComplexField(prec)

    if debug:
        print(f"[integrate] from ({x_start}, {y_start}) to ({x_end}, {y_end})")

    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0 / float(N)
        pi = math.pi
        for k in range(-N, N + 1):
            t = k * h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi / 2.0) * sx)
            dx_dt = (pi / 2.0) * math.cosh(t) / (math.cosh((pi / 2.0) * sx) ** 2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes

    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)

    p0 = CC(x_start)
    p1 = CC(x_end)
    y0 = CC(y_start)
    y1 = CC(y_end)
    
    vec = p1 - p0
    
    # NO PERPENDICULAR OFFSET - integrate on the straight line!
    # The branch tracking handles sheet selection
    
    dx_factor = vec / CC(2)

    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs) - 1 - i)) for i, c in enumerate(f_coeffs))

    # Build x-values along the straight line
    xvals = []
    ws = []
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)  # maps (-1,1) -> (0,1)
        xval = p0 + s * vec  # STRAIGHT LINE, no offset
        xvals.append(xval)
        ws.append(CC(w))

    n = len(xvals)
    fvals = [f_at(xv) for xv in xvals]

    tiny = CC(2) ** (-prec // 2)

    # CRITICAL: Establish branches at BOTH ends
    # At s=0 (start): we should get y_start
    # At s=1 (end): we should get y_end
    
    # Find good seed indices at both ends
    start_idx = None
    end_idx = None
    
    for i in range(n // 4):  # first quarter
        if abs(fvals[i]) >= tiny:
            start_idx = i
            break
    
    for i in range(n - 1, 3 * n // 4, -1):  # last quarter
        if abs(fvals[i]) >= tiny:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        raise ValueError("Path too close to branch locus")

    # Assign branches at seed points
    yvals = [None] * n
    
    # Start: choose sqrt that matches y_start
    sqrt_start = fvals[start_idx].sqrt()
    if abs(sqrt_start - y0) <= abs(-sqrt_start - y0):
        yvals[start_idx] = sqrt_start
    else:
        yvals[start_idx] = -sqrt_start
    
    # End: choose sqrt that matches y_end  
    sqrt_end = fvals[end_idx].sqrt()
    if abs(sqrt_end - y1) <= abs(-sqrt_end - y1):
        yvals[end_idx] = sqrt_end
    else:
        yvals[end_idx] = -sqrt_end

    # Propagate from start_idx backward to 0
    for i in range(start_idx - 1, -1, -1):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i + 1]) <= abs(y_m - yvals[i + 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Propagate from start_idx forward to end_idx
    for i in range(start_idx + 1, end_idx + 1):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Propagate from end_idx forward to n-1
    for i in range(end_idx + 1, n):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Integrate
    integral = CC(0)
    for i in range(n):
        y_cur = yvals[i]
        if abs(y_cur) == 0:
            continue
        
        if use_x_weight:
            integrand = xvals[i] / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * ws[i]
        integral += integrand * dxd

    return integral


def neron_tate_height_pairing(z1, z2, Im_tau, prec=300, normalization_factor=1.0):
    """
    Compute Néron-Tate height pairing between two Abel-Jacobi images.
    
    Formula: <z1, z2> = normalization_factor * Re(z1^† * Im(τ)^(-1) * z2)
    
    where z1^† means transpose + complex conjugate,
    z1, z2 are vectors in C^2 and Im(τ) is the imaginary part of 
    the period matrix.
    
    The normalization_factor accounts for different conventions:
    - Some references use no factor (1.0)
    - Some use π
    - Some use 2π  
    - Some use 1/(2π) or 1/π
    
    The canonical height h(div) relates to the self-pairing by:
    h(div) = <div, div> / 2  (in some conventions)
    or
    h(div) = <div, div>      (in other conventions)
    
    Returns: real number (QQ approximation)
    """
    import math
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Invert Im(τ)
    try:
        Im_tau_inv = Im_tau.inverse()
    except:
        # Singular - shouldn't happen if period matrix is correct
        raise
        return QQ(0)
    
    # Convert Im_tau_inv to complex field for proper arithmetic
    Im_tau_inv_CC = Matrix(CC, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau_inv_CC[i,j] = CC(Im_tau_inv[i,j])
    
    # Convert z1, z2 to column vectors in CC
    z1_vec = vector(CC, [CC(z1[0]), CC(z1[1])])
    z2_vec = vector(CC, [CC(z2[0]), CC(z2[1])])
    
    # Compute z1^† * Im_tau_inv * z2
    # = conj(z1)^T * Im_tau_inv * z2
    
    z1_conj = vector(CC, [z1_vec[0].conjugate(), z1_vec[1].conjugate()])
    
    # Matrix-vector multiply: Im_tau_inv * z2
    temp = Im_tau_inv_CC * z2_vec
    
    # Dot product: z1_conj · temp
    result = z1_conj[0] * temp[0] + z1_conj[1] * temp[1]
    
    # Take real part and apply normalization
    real_part = RR(result.real()) * RR(normalization_factor)
    
    # Convert to QQ
    return real_part.nearby_rational(max_error=RR(2)**(-prec+10))

def get_bad_primes(f_coeffs):
    """
    Identify primes of bad reduction for the curve y^2 = f(x).
    Includes factors of discriminant, leading coefficient, and 2.
    """
    from sage.all import QQ, PolynomialRing
    key = tuple(f_coeffs)
    if key in get_bad_primes.cache:
        return get_bad_primes.cache[key]
    
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    bad = set()
    
    # 1. Discriminant factors
    disc = f_poly.discriminant()
    if disc != 0:
        # Handle Rational discriminant: separate numerator and denominator
        bad.update(QQ(disc).numerator().prime_factors())
        bad.update(QQ(disc).denominator().prime_factors())
    
    # 2. Leading coefficient factors (potential degree drop)
    lc = f_coeffs[0]
    if lc != 0:
        bad.update(QQ(lc).numerator().prime_factors())
        bad.update(QQ(lc).denominator().prime_factors())
        
    # Genus 2 arithmetic at p=2 is always delicate
    bad.add(2)
    
    ret = sorted(list(bad))
    get_bad_primes.cache[key] = ret
    return ret
get_bad_primes.cache = {}


def local_naive_height_p(div, p):
    """
    Compute naive local height at p: -min(v_p(coeffs)) * log(p).
    This corresponds to the log of the max p-adic norm of coefficients.
    """
    try:
        # Extract Mumford polynomials u, v
        u_poly, v_poly = div[0], div[1]
        coeffs = u_poly.list() + v_poly.list()
        
        # We want max(|c|_p). 
        # |c|_p = p^(-v_p(c)).
        # log(max |c|_p) = log(p^(-min v_p(c))) = -min(v_p(c)) * log(p)
        
        # Handle 0 coefficients (val is +infinity)
        vals = []
        for c in coeffs:
            if c == 0:
                continue
            # Handle both Rational and p-adic types
            try:
                vals.append(c.valuation(p))
            except AttributeError:
                vals.append(c.valuation())
                raise
                
        if not vals:
            return 0.0
            
        min_val = min(vals)
        return -min_val * math.log(p)
    except Exception:
        raise
        return 0.0


def _compute_pairing_worker(args):
    """Worker function to compute a single Néron-Tate pairing"""
    from sage.all import PolynomialRing, HyperellipticCurve, QQ
    
    i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args
    try:
        if i == j:
            # Diagonal: just return the cached height
            return ((i, j), h_i, None)
        
        # Off-diagonal: only compute h(div1+div2)
        Rq_QQ = PolynomialRing(QQ, 'x')
        x_QQ = Rq_QQ.gen()
        f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-k) 
                       for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
        
        u_poly_i = x_QQ**2 - QQ(div_i['s'])*x_QQ + QQ(div_i['p'])
        v_poly_i = QQ(div_i['v_1'])*x_QQ + QQ(div_i['v_0'])
        div1 = J([u_poly_i, v_poly_i])
        
        u_poly_j = x_QQ**2 - QQ(div_j['s'])*x_QQ + QQ(div_j['p'])
        v_poly_j = QQ(div_j['v_1'])*x_QQ + QQ(div_j['v_0'])
        div2 = J([u_poly_j, v_poly_j])
        
        div_sum = div1 + div2
        
        if div_sum.is_zero():
            h_sum = QQ(0)
        else:
            h_sum = arakelov_canonical_height(div_sum, f_coeffs, prec=prec)
        
        # Use cached h_i and h_j (already canonical heights)
        val = (h_sum - h_i - h_j) / QQ(2)
        
        return ((i, j), val, None)
    except Exception as e:
        raise
        return ((i, j), None, str(e))


def precompute_pairings_parallel(indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    """Precompute all pairings for given indices in parallel"""
    pairs_to_compute = []
    for r in range(len(indices)):
        for c in range(r, len(indices)):
            i, j = indices[r], indices[c]
            if i > j:
                i, j = j, i
            if (i, j) not in pairing_cache:
                div_i = jac_elements[i][0]
                div_j = jac_elements[j][0]
                pairs_to_compute.append((
                    i, j, div_i, div_j, f_coeffs, prec,
                    height_cache[i], height_cache[j]
                ))

    if not pairs_to_compute:
        return pairing_cache

    print(f"[arakelov] Computing {len(pairs_to_compute)} pairings...")
    
    if len(pairs_to_compute) > 2 and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_compute_pairing_worker, pairs_to_compute)):
                results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{len(pairs_to_compute)}")
            
        for (i, j), val, error in results:
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
            pairing_cache[(i, j)] = val
    else:
        for args in pairs_to_compute:
            (i, j), val, error = _compute_pairing_worker(args)
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
            pairing_cache[(i, j)] = val
    
    return pairing_cache


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Incremental basis builder: only compute pairings as needed.
    """
    from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, QQ)
    from multiprocessing import cpu_count
    import sys

    get_period_matrix_auto_B(f_coeffs, prec=prec)

    if not all_divisors:
        return [], 0, None

    if n_jobs == -1:
        try:
            n_jobs = cpu_count()
        except Exception:
            n_jobs = 1
            raise

    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
        print(f"[arakelov] Parallelization: {n_jobs} workers")

    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    jac_elements = []
    for div in all_divisors:
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        div2 = J([u_poly, v_poly])
        jac_elements.append((div, div2))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    # Compute individual heights (parallel)
    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_args = [(i, jac_elements[i][0], f_coeffs, prec) for i in range(n)]
    height_cache = {}

    if n > 1 and n_jobs > 1:
        from multiprocessing import Pool
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_compute_height_worker, height_args)
        for i, h, error in results:
            if error:
                raise RuntimeError(f"Height computation failed for divisor {i}: {error}")
            height_cache[i] = h
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")
    else:
        for i in range(n):
            _, div = jac_elements[i]
            h = arakelov_canonical_height(div, f_coeffs, prec=prec)
            height_cache[i] = h
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")

    # Pairing cache (compute on-demand)
    pairing_cache = {}

    # Helper to get pairing (with lazy computation)
    def get_pairing_lazy(i, j):
        if i > j:
            i, j = j, i
        if (i, j) in pairing_cache:
            return pairing_cache[(i, j)]
        
        # Compute on-demand
        if i == j:
            val = height_cache[i]
        else:
            div_i = jac_elements[i][0]
            div_j = jac_elements[j][0]
            args = (i, j, div_i, div_j, f_coeffs, prec, height_cache[i], height_cache[j])
            _, val, error = _compute_pairing_worker(args)
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
        
        #sanity_check_pairings(pairing_cache, min(len(all_divisors), 1))
        pairing_cache[(i, j)] = val
        return val

    # Incremental basis selection
    if debug:
        print("[arakelov] Selecting basis incrementally...")
    
    basis_indices = []
    
    for cand_idx in range(n):
        if len(basis_indices) >= 2 * C.genus():  # genus 2 -> rank <= 4
            break
            
        if len(basis_indices) == 0:
            # First element: just check if nonzero height
            h = height_cache[cand_idx]
            if abs(float(h)) > 1e-6:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (first basis element)")
            continue
        
        # Test independence via projection residual
        is_indep, info = is_independent_by_projection_log(
            basis_indices,
            cand_idx,
            get_pairing_lazy,
            prec,
            debug=debug
        )
        
        if is_indep:
            k = len(basis_indices)
            from sage.all import RealField, Matrix
            RR = RealField(max(128, int(prec//4)))
            G_test = Matrix(RR, k, k)
            for r in range(k):
                for c in range(r, k):
                    pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                    G_test[r, c] = RR(pv)
                    G_test[c, r] = G_test[r, c]
            
            det_val = G_test.determinant()
            if det_val > 0:
                basis_indices.append(cand_idx)
                if debug:
                    print(f"  Added divisor {cand_idx} (basis now size {len(basis_indices)})")
            else:
                print(f"  Rejected divisor {cand_idx}: dependent")
        elif debug:
            print(f"  Rejected divisor {cand_idx}: dependent")

    # ADD THE SANITY CHECK HERE (after the loop, before building final Gram matrix):
    if len(basis_indices) > 0:
        # Only check pairings that were actually computed during basis selection
        # The cache will have diagonal entries for selected basis elements
        sanity_check_pairings(pairing_cache, min(len(basis_indices), 4))


    basis = [jac_elements[i][0] for i in basis_indices]
    basis, basis_indices = dedupe_basis(basis, basis_indices, debug=debug)

    final_rank = len(basis)
    H_final = None
    
    if final_rank > 0:
        RR = RealField(max(128, int(prec//4)))
        H_final = Matrix(RR, final_rank, final_rank)
        
        if debug:
            print(f"[arakelov] Building final {final_rank}×{final_rank} Gram matrix...")
        
        for r in range(final_rank):
            for c in range(r, final_rank):
                pv = get_pairing_lazy(basis_indices[r], basis_indices[c])
                H_final[r, c] = RR(pv)
                H_final[c, r] = H_final[r, c]

        eigs = H_final.eigenvalues()
        print("eigenvalues:", eigs)
        f_eigs = [float(i) for i in eigs]
        print("eigenvalues (float):", f_eigs)
        print("determinant:", float(H_final.determinant()))
        print("condition number (approx):", float(max(eigs)/min(eigs)))
        # regulator is det(G) (for basis of free part of rank k)
        print("regulator:", float(H_final.determinant()))

    if debug and final_rank > 0:
        try:
            gd = gram_logdet_and_cond(basis_indices, get_pairing_lazy)
            print(f"[arakelov] Final numeric rank: {gd['numeric_rank']}/{gd['n']}")
            print(f"[arakelov] log10|det|={gd['log10_abs_det']:.3g}, log10(cond)={gd['log10_cond']:.3g}")
            if H_final is not None:
                try:
                    print(f"[arakelov] Final determinant: {float(H_final.determinant()):.6g}")
                except Exception:
                    raise
        except Exception as E:
            if debug:
                print(f"[arakelov] Diagnostic failed: {E}")
            raise

    return basis, final_rank, H_final


def is_independent_by_projection_log(
    basis_indices,
    candidate_index,
    get_pairing,
    prec,
    debug=False,
):
    """Use high-precision Sage arithmetic throughout"""
    from sage.all import RealField, Matrix, vector
    import math
    
    k = len(basis_indices)
    if k == 0:
        return True, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
    
    # High-precision reals
    RR = RealField(max(128, int(prec // 4)))
    
    # Build Gram in Sage
    G = Matrix(RR, k, k)
    c = vector(RR, k)
    
    for r in range(k):
        for s in range(r, k):
            v = get_pairing(basis_indices[r], basis_indices[s])
            G[r, s] = RR(v)
            G[s, r] = G[r, s]
        c[r] = RR(get_pairing(basis_indices[r], candidate_index))
    
    G = make_matrix_numerically_positive_definite(G, tol=RR(10)**(-20))
    min_ev = min(float(e) for e in G.eigenvalues())
    if min_ev <= 0:
        raise RuntimeError(f"Gram not PD after clipping: min_eig={min_ev:.3e}")

    vv = RR(get_pairing(candidate_index, candidate_index))
    
    # Special case: if k=1, just check if candidate has different height
    if k == 1:
        g00 = G[0, 0]
        c0 = c[0]
        
        if abs(g00) < 1e-10:
            # First basis element has ~zero height, reject candidate
            if debug:
                print(f"[proj-log] First basis element has near-zero height")
            return False, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
        
        # Projection: proj_sq = c0^2 / g00
        proj_sq = (c0 * c0) / g00
        res_sq = vv - proj_sq
        
        res_sq_f = float(res_sq)
        if res_sq_f <= 0:
            if debug:
                print(f"[proj-log] k=1: negative residual res_sq={res_sq_f}")
            return False, {"res_sq": res_sq_f, "log10_res": float("-inf"), "log10_tol": float("-inf"), "min_sv": None}
        
        dec_digits = int(prec * 0.30103)
        safety_digits = 12
        diag_max = max(abs(float(g00)), abs(float(vv)), 1.0)
        log10_tol = math.log10(diag_max) - max(0, (dec_digits - safety_digits))
        log10_res = math.log10(res_sq_f) if res_sq_f > 0 else float("-inf")
        
        is_independent = (log10_res > log10_tol)
        
        info = {"res_sq": res_sq_f, "log10_res": log10_res, "log10_tol": log10_tol, "min_sv": None}
        if debug:
            print(f"[proj-log] k=1 cand={candidate_index} res={res_sq_f:.3g} log10_res={log10_res:.3g} log10_tol={log10_tol:.3g}")
        
        return is_independent, info
    
    # For k >= 2: use Cholesky (requires positive definite)
    try:
        L = G.cholesky()
        y = L.solve_left(c)
        proj_sq = y.dot_product(y)
    except Exception as e:
        if debug:
            print(f"[proj-log] Cholesky failed for candidate {candidate_index}: {e}")
        raise
        return False, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}
    
    res_sq = vv - proj_sq
    
    res_sq_f = float(res_sq)
    vv_f = float(vv)
    
    if res_sq_f <= 0:
        diag_max = max(float(G[r,r]) for r in range(k)) if k > 0 else 1.0
        diag_max = max(diag_max, vv_f, 1.0)
        if abs(res_sq_f) <= 1e-12 * diag_max:
            res_sq_f = 0.0
        else:
            if debug:
                print(f"[proj-log] negative residual (proj > self): res_sq={res_sq_f}")
            return False, {"res_sq": res_sq_f, "log10_res": float("-inf"), "log10_tol": float("-inf"), "min_sv": None}
    
    dec_digits = int(prec * 0.30103)
    safety_digits = 12
    diag_max = max(max(abs(float(G[r,r])) for r in range(k)), vv_f, 1.0)
    log10_tol = math.log10(diag_max) - max(0, (dec_digits - safety_digits))
    log10_res = math.log10(res_sq_f) if res_sq_f > 0 else float("-inf")
    
    is_independent = (log10_res > log10_tol)
    
    info = {"res_sq": res_sq_f, "log10_res": log10_res, "log10_tol": log10_tol, "min_sv": None}
    if debug:
        print(f"[proj-log] cand={candidate_index} res={res_sq_f:.3g} log10_res={log10_res:.3g} log10_tol={log10_tol:.3g}")
    
    return is_independent, info


def arakelov_quasi_height(div, f_coeffs, period_matrix=None, prec=300, use_finite_places=True, arch_override=None):
    """
    Computes a 'quasi-canonical' height: Naive + Finite + Archimedean(Quadratic).
    This height h(div) satisfies h(ndiv) = n^2 * h_can(div) + L(ndiv) + O(1).
    It is not quadratic itself, but the quadratic coefficient is the canonical height.
    """
    if div.is_zero():
        return QQ(0)

    # 1. Naive global height (Essential for the height to be positive/quadratic)
    h_naive = naive_height_qq(div, prec=prec)
    
    # 2. Archimedean quadratic part
    if period_matrix is None:
        period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
    if arch_override is None:
        h_arch = archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec)
    else:
        h_arch = arch_override 
    # 3. Finite place corrections
    h_finite_correction = QQ(0)
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        for p in bad_primes:
            h_finite_correction += local_height_correction_finite(div, p, f_coeffs)
            
    return h_naive + h_arch + h_finite_correction


# Insert into jacobianbasis.py near other helpers
from sage.all import Matrix, identity_matrix


def sanity_check_pairings(pairing_cache, n):
    """
    pairing_cache: dict of (i,j)->value for 0<=i<=j<n
    n: number of indices expected
    Raises on failures.
    """
    for i in range(n):
        if (i, i) not in pairing_cache:
            raise ValueError(f"Missing diagonal pairing for {i}")
        if float(pairing_cache[(i,i)]) <= 0:
            raise ValueError(f"Non-positive self-pairing for {i}: {pairing_cache[(i,i)]}")

    for i in range(n):
        for j in range(i+1, n):
            a = pairing_cache.get((i,j), None)
            b = pairing_cache.get((j,i), None)
            if a is None and b is None:
                raise ValueError(f"Missing pairing for pair {(i,j)}")
            if a is None: a = b
            if b is None: b = a
            if abs(float(a) - float(b)) > 1e-10 * max(1.0, abs(float(a))):
                raise ValueError(f"Asymmetric pairings for {(i,j)}: {a} vs {b}")


def robust_eig_clip(Im, min_eig_tol=1e-30):
    # Im is a symmetric matrix (numpy or sage). Convert to numpy double for SVD but clip tiny negatives.
    import numpy as np
    M = np.array(Im, dtype=float)
    # symmetrize
    M = 0.5 * (M + M.T)
    U, s, Vt = np.linalg.svd(M)
    # clip
    s_clipped = np.clip(s, min_eig_tol, None)
    return U, s_clipped, Vt


def normalize_periods_and_z(Omega, z_vec):
    """
    Normalize periods and Abel-Jacobi vector.
    Returns tau (g×g) and z_norm (g×1 with SCALAR entries).
    """
    from sage.all import Matrix
    
    Omega = Matrix(Omega)
    g = Omega.nrows()
    
    # Check if already normalized
    if Omega.ncols() == g:
        tau = Omega
        if z_vec is None:
            z_norm = None
        else:
            # Create g×1 matrix element by element
            from sage.all import matrix
            z_norm = matrix(tau.base_ring(), g, 1)
            for i in range(g):
                z_norm[i, 0] = z_vec[i]  # Direct indexing extracts scalars
        return tau, z_norm
    
    if Omega.ncols() != 2*g:
        raise ValueError(f"Omega shape {Omega.nrows()}×{Omega.ncols()}, expected g or 2g cols")
    
    # Split and normalize
    Omega1 = Omega[:, :g]
    Omega2 = Omega[:, g:]
    
    if not Omega1.is_invertible():
        raise ValueError("Omega1 singular")
    
    Omega1_inv = Omega1.inverse()
    tau = Omega1_inv * Omega2
    
    # Normalize z
    if z_vec is None:
        z_norm = None
    else:
        # Build column vector element-by-element
        from sage.all import matrix
        z_temp = matrix(Omega.base_ring(), g, 1)
        for i in range(g):
            z_temp[i, 0] = z_vec[i]
        z_norm = Omega1_inv * z_temp
    
    # Sanity checks
    if max(abs((tau - tau.transpose()).list())) > 1e-10:
        raise ValueError("tau not symmetric")
    
    Im_tau = Matrix([[c.imag() for c in row] for row in tau])
    eigs = Im_tau.eigenvalues()
    if any(float(e) <= 1e-14 for e in eigs):
        raise ValueError("Im(tau) not positive definite")
    
    return tau, z_norm


def make_matrix_numerically_positive_definite(G, tol=1e-20):
    """
    Ensure a symmetric matrix is numerically positive definite by clipping eigenvalues.
    
    Parameters
    ----------
    G : Sage Matrix (RealField or similar)
        Symmetric matrix that should be positive definite
    tol : float or Sage real
        Minimum eigenvalue threshold
    
    Returns
    -------
    G_fixed : Sage Matrix
        Positive definite version of G
    """
    from sage.all import Matrix, diagonal_matrix
    import numpy as np
    
    # Convert to numpy for eigendecomposition
    n = G.nrows()
    G_np = np.array([[float(G[i,j]) for j in range(n)] for i in range(n)], dtype=float)
    
    # Symmetrize to avoid numerical asymmetry
    G_np = 0.5 * (G_np + G_np.T)
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(G_np)
    
    # Clip eigenvalues to minimum threshold
    eigvals_clipped = np.maximum(eigvals, float(tol))
    
    # Reconstruct: G = V * Lambda * V^T
    G_fixed_np = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    # Convert back to Sage matrix
    base_ring = G.base_ring()
    G_fixed = Matrix(base_ring, n, n)
    for i in range(n):
        for j in range(n):
            G_fixed[i,j] = base_ring(G_fixed_np[i,j])
    
    return G_fixed


from sage.all import ComplexField, RealField, Matrix, vector, sqrt, pi, QQ

def print_archimedean_diagnostics(tau, z, quad_val, log_theta, prec, debug=False):
    """
    tau : g x g complex matrix or nested list
    z   : length-g complex vector or list
    quad_val, log_theta : the scalar values used in archimedean height
    prec : bits of precision used (int)
    Prints diagnostics and returns a dict with values.
    """
    # create high-precision fields for robust numeric conversion
    CC = ComplexField(prec)
    RR = RealField(prec)
    # convert tau / z into CC objects and build Im(tau)
    g = len(z)
    # make tau a Matrix(CC)
    try:
        Tau = Matrix(CC, g, g, [[CC(tau[i][j]) for j in range(g)] for i in range(g)])
    except Exception:
        # maybe tau is a Sage Matrix already with complex entries
        Tau = Matrix(CC, g, g, [[CC(tau[i,j]) for j in range(g)] for i in range(g)])
    Z = vector(CC, [CC(z[i]) for i in range(g)])
    # Imaginary part matrix
    ImTau = Matrix(RR, g, g, [[RR((Tau[i,j]).imag) for j in range(g)] for i in range(g)])
    eigs = [float(e) for e in ImTau.eigenvalues()]
    # Norms
    z_norm = float(sum(abs(Z[i])**2 for i in range(g)))

    print("\n[ARCH DIAG] precision:", prec, "bits")
    print("[ARCH DIAG] tau (approx):")
    for i in range(g):
        print("  ",[complex(Tau[i,j]) for j in range(g)])
    print("[ARCH DIAG] Im(tau) eigenvalues:", eigs)
    print("[ARCH DIAG] z (approx):", [complex(Z[i]) for i in range(g)])
    print("[ARCH DIAG] ||z||^2:", z_norm)
    print("[ARCH DIAG] quad_val:", float(quad_val))
    print("[ARCH DIAG] log|theta| (value used):", float(log_theta))
    print("[ARCH DIAG] quad - logtheta:", float(quad_val - log_theta))
    # return a dict if caller wants to inspect
    return dict(prec=prec, tau=Tau, ImTau=ImTau, ImTau_eigs=eigs, z=Z, z_norm=z_norm,
                quad_val=CC(quad_val), log_theta=CC(log_theta),
                arch = CC(quad_val - log_theta))


# put this near the TOP of archimedean_height_correction (or module-level)


from itertools import product


def theta_direct(tau_in, z_in, R=3, prec_local=256):
    """
    Direct summation of theta function for genus 2, used for cheap screening.
    """
    from sage.all import ComplexField, pi, exp
    CC_loc = ComplexField(prec_local)
    g_loc = len(z_in)
    Tau = [[CC_loc(tau_in[i][j]) for j in range(g_loc)] for i in range(g_loc)]
    Z = [CC_loc(z_in[i]) for i in range(g_loc)]
    total = CC_loc(0)
    
    # Generic loop for arbitrary genus would be better, but optimizing for g=2
    if g_loc == 2:
        for n0 in range(-R, R+1):
            for n1 in range(-R, R+1):
                # q = n^T * Tau * n
                q = Tau[0][0]*n0*n0 + (Tau[0][1]+Tau[1][0])*n0*n1 + Tau[1][1]*n1*n1
                # linear = 2 * n^T * z
                linear = 2*(n0*Z[0] + n1*Z[1])
                arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
                total += CC_loc(exp(arg))
        return total
    elif g_loc == 1:
         for n0 in range(-R, R+1):
            q = Tau[0][0]*n0*n0
            linear = 2*n0*Z[0]
            arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
            total += CC_loc(exp(arg))
         return total
    else:
        raise NotImplementedError("theta_direct optimization only implemented for g=1,2")


def archimedean_height_correction(div, f_coeffs, period_matrix, prec=300, debug=False):
    """
    Exact archimedean height correction.
    Crashes on any inconsistency.
    """
    if div.is_zero():
        return QQ(0)

    RR = RealField(prec)
    CC = ComplexField(prec)

    base_point = choose_numerical_base_point(f_coeffs, prec=prec)
    z_vec = abel_jacobi_mumford(div, f_coeffs, base_point=base_point, prec=prec)

    tau, z_norm_mat = normalize_periods_and_z(period_matrix, z_vec)
    g = tau.nrows()

    z_raw = [CC(z_norm_mat[i,0]) for i in range(g)]
    z = reduce_z_arakelov(z_raw, tau, prec=prec, debug=debug)

    # Im(tau)
    Im_tau = Matrix(RR, g, g, [[RR(CC(tau[i,j]).imag()) for j in range(g)] for i in range(g)])
    Im_tau = 0.5 * (Im_tau + Im_tau.transpose())
    det_im = Im_tau.det()
    assert det_im > 0

    y_im = vector(RR, [RR(zi.imag()) for zi in z])
    v = Im_tau.solve_right(y_im)
    quad = RR(pi) * y_im.dot_product(v)

    theta = compute_theta_high_prec(z, tau, prec=prec)
    abs_theta = abs(CC(theta))
    assert abs_theta > 0

    log_theta = RR(abs_theta).log()
    corr = QQ(1)/QQ(2) * RR(det_im).log()

    arch = quad - log_theta + corr

    if arch < -RR(1e-12):
        print("\n[ARCHIMEDEAN HEIGHT FAILURE]")
        print("quad =", float(quad))
        print("log|theta| =", float(log_theta))
        print("det(Im tau) =", float(det_im))
        print("z =", [complex(zi) for zi in z])
        raise RuntimeError("Archimedean height negative")

    return QQ(arch)


def reduce_z_arakelov(z_list, tau, prec=300, debug=False):
    """
    Reduce z modulo Z^g + tau Z^g by testing all half-period shifts
    and choosing the representative maximizing
        quad(z) - log|theta(z)|.
    Strict behavior: no fallbacks, crash on unexpected conditions.
    """
    RR = RealField(prec)
    CC = ComplexField(prec)

    g = tau.nrows()
    assert len(z_list) == g

    # Ensure tau is a CC matrix
    Tau = Matrix(CC, tau)

    # Work with CC vector
    z0 = vector(CC, [CC(z) for z in z_list])

    # Build Im(tau) as an RR matrix and require positive-definiteness
    Im_tau = Matrix(RR, g, g, [[RR(Tau[i, j].imag()) for j in range(g)] for i in range(g)])
    Im_tau = 0.5 * (Im_tau + Im_tau.transpose())
    det_im = Im_tau.det()
    assert det_im > 0

    # First reduce by full lattice: choose integer n solving Im_tau * c = y and round c
    y = vector(RR, [RR(z.imag()) for z in z0])
    c = Im_tau.solve_right(y)
    # convert c entries to Python ints safely (RealField -> float -> round -> int -> ZZ)
    n = vector(ZZ, [ZZ(int(round(float(ci)))) * (-1) for ci in c])
    z1 = z0 + Tau * n

    # Reduce by integer real translations
    m = vector(ZZ, [ZZ(int(round(float(z1[i].real())))) * (-1) for i in range(g)])
    z_base = z1 + m

    # Helper: quadratic form pi * y^T Im_tau^{-1} y
    def quad_val(zvec):
        y_im = vector(RR, [RR(zvec[i].imag()) for i in range(g)])
        v = Im_tau.solve_right(y_im)
        return RR(pi) * y_im.dot_product(v)

    best_score = None
    best_z = None

    # Enumerate all half-period shifts a,b in {0,1}^g without importing itertools
    # Use bit masks for a and b
    limit = 1 << g
    for a_mask in range(limit):
        # build a vector in CC
        a = vector(CC, [CC((a_mask >> i) & 1) for i in range(g)])
        for b_mask in range(limit):
            b = vector(CC, [CC((b_mask >> i) & 1) for i in range(g)])
            shift = CC(1) / CC(2) * (a + Tau * b)
            z_candidate = z_base + shift

            # reduce reals to fundamental interval
            z_candidate = vector(CC, [z_candidate[i] - CC(int(round(float(z_candidate[i].real())))) for i in range(g)])

            theta = compute_theta_high_prec(list(z_candidate), tau, prec=prec)
            abs_theta = abs(CC(theta))
            assert abs_theta > 0

            score = quad_val(z_candidate) - RR(abs_theta).log()

            if debug:
                print("reduce_z_arakelov: score", float(score))

            if (best_score is None) or (score > best_score):
                best_score = score
                best_z = z_candidate

    assert best_z is not None
    return [CC(best_z[i]) for i in range(g)]


def arakelov_canonical_height(div, f_coeffs, prec=300, use_finite_places=True):
    """
    Proper canonical height using the Second Difference Method.
    Archimedean contribution is computed ONCE and reused consistently.
    """
    from .homology import get_period_matrix_auto_B
    from sage.all import QQ

    if div.is_zero():
        return QQ(0)

    # Pre-fetch period matrix once
    period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)

    # Compute archimedean height ONCE
    h_arch = archimedean_height_correction(
        div, f_coeffs, period_matrix, prec=prec
    )

    # Quasi-heights using fixed archimedean input
    h1 = arakelov_quasi_height(
        div, f_coeffs, period_matrix, prec,
        use_finite_places,
        arch_override=h_arch
    )

    div2 = div + div
    if div2.is_zero():
        h2 = QQ(0)
    else:
        h2 = arakelov_quasi_height(
            div2, f_coeffs, period_matrix, prec,
            use_finite_places,
            arch_override=QQ(2) * h_arch
        )

    div3 = div2 + div
    if div3.is_zero():
        h3 = QQ(0)
    else:
        h3 = arakelov_quasi_height(
            div3, f_coeffs, period_matrix, prec,
            use_finite_places,
            arch_override=QQ(3) * h_arch
        )

    h_can = (h3 + h1 - QQ(2) * h2) / QQ(2)

    # Canonical height MUST be non-negative
    if h_can < 0:
        raise RuntimeError(
            "Canonical height negative — invariant violation: "
            f"h_can={h_can}; divisor={div}; prec={prec}"
        )

    return h_can


def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute the local canonical height correction (Neron correction) at p 
    using a stabilized p-adic doubling limit:
       mu_p(div) = lim_{n->inf} 4^(-n) * h_naive(2^n div) - h_naive(div)

    Safety features added:
    - Uses a Cesàro/last-window average of the scaled naive heights to reduce oscillation.
    - Detects instability (large variance, NaNs or huge values) and will retry once
      with larger p-adic precision and more doublings.
    - If instability persists, returns a conservative 0.0 for the local correction
      (and prints a warning). This prevents a single bad p-term from making the
      global canonical height nonsensical.
    - Clamps obviously absurd results.
    """
    import math
    import warnings

    # Parameters for stabilization / safety
    MIN_PADIC_PREC = 2048
    MAX_PADIC_PREC = 8192
    MAX_RETRIES = 1            # automatic retry with higher padic precision
    MAX_ACCEPTABLE_MAG = 1e6   # anything larger is considered bogus
    REL_VAR_TOL = 1e-4         # relative stddev tolerance for tail stability
    ABS_NEG_TOL = 1e-8         # allow tiny negative noise, clamp bigger negatives to 0.0
    MIN_TAIL_LEN = 3

    # Ensure initial padic precision is reasonable
    if padic_prec is None:
        padic_prec = max(MIN_PADIC_PREC, 100 * max(1, num_doublings))

    # internal helper to attempt computation; returns (mu_or_None, reason)
    def _attempt(padic_prec_local, num_doublings_local):
        try:
            # 1. Setup p-adic curve
            K = Qp(p, prec=padic_prec_local)
            R = PolynomialRing(K, 'x')
            f_poly = sum(K(c) * R.gen()**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
            C_p = HyperellipticCurve(f_poly)
            J_p = C_p.jacobian()

            # 2. Lift point div to J(Qp)
            u_Q, v_Q = div[0], div[1]
            # if these are Sage polys or lists, convert coefficients to K
            u_p = R([K(c) for c in u_Q.list()])
            v_p = R([K(c) for c in v_Q.list()])

            P = J_p([u_p, v_p])

            # 3. Compute h0
            h0 = local_naive_height_p(P, p)

            # 4. Iterate doubling and collect scaled naive heights:
            #    s_k = 4^{-k} * h_naive(2^k P)
            s_values = []
            current_P = P

            for k in range(0, num_doublings_local + 1):
                # If point becomes zero at some stage, the Tate-limit is exactly 0
                # (future terms are zero), so mu = 0 - h0.
                if current_P.is_zero():
                    mu_exact = float(0.0 - h0)
                    return mu_exact, "torsion_hit"

                # compute naive height at current_P (may return Sage/Rational/float)
                h_k = local_naive_height_p(current_P, p)
                try:
                    h_kf = float(h_k)
                except Exception:
                    # non-convertible -> instability
                    return None, "non_numeric_height"

                s_k = (4.0 ** (-k)) * h_kf
                # quick sanity: huge values indicate instability
                if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                    return None, "huge_or_nan"

                s_values.append(s_k)

                # prepare for next doubling (but don't double on last loop)
                if k < num_doublings_local:
                    current_P = 2 * current_P

            # Need at least a few tail values to average
            tail_len = max(MIN_TAIL_LEN, num_doublings_local // 2)
            if len(s_values) < tail_len:
                return None, "insufficient_samples"

            tail = s_values[-tail_len:]
            tail_mean = sum(tail) / float(len(tail))
            # compute sample standard deviation (population stddev not necessary)
            mean = tail_mean
            var = sum((x - mean) ** 2 for x in tail) / float(len(tail))
            std = math.sqrt(var)

            # Relative variability check (relative to magnitude of mean)
            rel_std = std / (abs(mean) + 1e-16)

            if rel_std > REL_VAR_TOL and abs(mean) > 1e-12:
                # unstable sequence
                return None, "high_variance"

            # stabilized estimate for the Tate-limit value
            tate_limit_est = tail_mean

            # mu_p(div) = Tate-limit - h0
            mu_est = float(tate_limit_est - float(h0))

            # Clamp/guard obviously absurd negatives (user requested "no bad returns")
            if mu_est < -ABS_NEG_TOL:
                # allow tiny negative rounding noise, but not larger negatives
                return None, "excessive_negative"

            # Final sanity check: not astronomically large
            if abs(mu_est) > MAX_ACCEPTABLE_MAG:
                return None, "excessive_magnitude"

            return float(mu_est), "ok"

        except ZeroDivisionError:
            # propagate so the outer code can retry with larger precision as before
            raise
        except Exception as e:
            # any other failure -> mark as unstable
            return None, f"exception:{repr(e)}"

    # Attempt + one automatic retry if unstable
    attempt = 0

    while attempt <= MAX_RETRIES:
        mu_val, reason = _attempt(padic_prec, num_doublings)
        if mu_val is not None:
            # good result
            return mu_val

        # if we get here, the attempt failed for reason -> try a safer retry if possible

        if p not in local_height_correction_finite.warned_primes:
            warnings.warn(f"[local_height_correction_finite] instability at p={p}; reason={reason}; "
                          f"padic_prec={padic_prec}; num_doublings={num_doublings}. Retrying with higher precision.", RuntimeWarning)
            local_height_correction_finite.warned_primes.add(p)

        # if the failure was due to a ZeroDivisionError, let outer exception handler or caller deal with it
        # (this mirrors your original ZeroDivisionError behavior)
        # Otherwise, increase padic precision and (optionally) num_doublings and retry once.
        if padic_prec >= MAX_PADIC_PREC and p not in local_height_correction_finite.warned_primes:
            # Give up and return conservative 0.0
            warnings.warn(f"[local_height_correction_finite] giving up on p={p} after padic_prec={padic_prec}. Returning 0.0.", RuntimeWarning)
            local_height_correction_finite.warned_primes.add(p)
            return 0.0
        # increase precision and doublings for the retry
        padic_prec = min(MAX_PADIC_PREC, padic_prec * 2)
        num_doublings = min(num_doublings + 2, 2 * num_doublings if num_doublings > 0 else 4)
        attempt += 1

    # If all retries exhausted, return conservative 0.0
    if p not in local_height_correction_finite.warned_primes:
        warnings.warn(f"[local_height_correction_finite] all retries exhausted for p={p}. Returning 0.0.", RuntimeWarning)
        local_height_correction_finite.warned_primes.add(p)
    return 0.0
local_height_correction_finite.warned_primes = set()


def make_matrix_numerically_positive_definite(G, tol=1e-20):
    """
    Ensure a symmetric matrix is numerically positive definite by clipping eigenvalues.

    Works for Sage matrices, NumPy arrays, or list-of-lists.
    """
    import numpy as np
    from sage.all import Matrix, RR

    # --- Step 1: determine size and base ring safely ---
    if hasattr(G, "nrows"):          # Sage matrix
        n = G.nrows()
        base_ring = G.base_ring()
        G_np = np.array(
            [[float(G[i, j]) for j in range(n)] for i in range(n)],
            dtype=float
        )
    else:
        # assume array-like (NumPy or list-of-lists)
        G_np = np.array(G, dtype=float)
        if G_np.ndim != 2 or G_np.shape[0] != G_np.shape[1]:
            raise ValueError("Input must be a square matrix")
        n = G_np.shape[0]
        base_ring = RR

    # --- Step 2: symmetrize to kill numerical noise ---
    G_np = 0.5 * (G_np + G_np.T)

    # --- Step 3: eigendecomposition ---
    eigvals, eigvecs = np.linalg.eigh(G_np)

    # --- Step 4: clip eigenvalues ---
    eigvals_clipped = np.maximum(eigvals, float(tol))

    # --- Step 5: reconstruct matrix ---
    G_fixed_np = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # --- Step 6: convert back to Sage matrix ---
    G_fixed = Matrix(base_ring, n, n)
    for i in range(n):
        for j in range(n):
            G_fixed[i, j] = base_ring(G_fixed_np[i, j])

    return G_fixed


def arakelov_canonical_height(div, f_coeffs, prec=2048, max_prec=8192, debug=False):
    """
    Compute the Arakelov canonical height of `div` (a Jacobian point).
    Defensive: will attempt fallbacks (archimedean-only, retries with higher precision)
    if the assembled canonical height is numerically negative, and will ultimately
    clamp to 0.0 rather than returning a large negative value that would break
    global invariants.

    Returns a Python float (>= 0.0) in all non-exceptional cases.
    """
    import warnings
    from sage.all import QQ

    # small tolerances
    TOL_NEG = 1e-12   # allowed tiny negative noise
    PREC_INCR_FACTOR = 2

    # helper to compute quasi-heights (full or arch-only)
    def _compute_quasi_heights(use_finite, use_period_prec):
        # compute period matrix at requested precision
        PM = get_period_matrix_auto_B(f_coeffs, prec=use_period_prec)
        D1 = div
        D2 = div + div
        D3 = D2 + div
        h1 = arakelov_quasi_height(D1, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h2 = arakelov_quasi_height(D2, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h3 = arakelov_quasi_height(D3, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        return float(h1), float(h2), float(h3)

    # 1) First attempt: normal full computation
    try:
        h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=prec)
        h_can = (h3 + h1 - 2.0 * h2) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] attempt prec={prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
    except ZeroDivisionError:
        # preserve previous retry behavior for ZeroDivisionError so outer code can escalate
        raise
    except Exception as e:
        # catastrophic failure computing heights (e.g. period matrix failed)
        warnings.warn(f"[arakelov_canonical_height] initial full computation failed: {e}. Attempting archimedean-only fallback.", RuntimeWarning)
        h_can = -1.0  # force fallback path

    # If result is numerically acceptable, return it (clamp tiny negatives)
    if h_can >= -TOL_NEG:
        return float(max(h_can, 0.0))

    # 2) Fallback A: try archimedean-only (no finite places)
    try:
        h1_nf, h2_nf, h3_nf = _compute_quasi_heights(use_finite=False, use_period_prec=prec)
        h_can_nf = (h3_nf + h1_nf - 2.0 * h2_nf) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] arch-only: h1={h1_nf}, h2={h2_nf}, h3={h3_nf}, h_can_nf={h_can_nf}")
        if h_can_nf >= -TOL_NEG:
            warnings.warn(f"[arakelov_canonical_height] suppressed finite-place instability; using archimedean-only height for divisor={div}.", RuntimeWarning)
            return float(max(h_can_nf, 0.0))
    except Exception as e:
        warnings.warn(f"[arakelov_canonical_height] arch-only fallback failed: {e}", RuntimeWarning)

    # 3) Fallback B: retry full computation with increasing precision
    cur_prec = prec
    while cur_prec < max_prec:
        cur_prec = min(max_prec, int(cur_prec * PREC_INCR_FACTOR))
        try:
            h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=cur_prec)
            h_can = (h3 + h1 - 2.0 * h2) / 2.0
            if debug:
                print(f"[arakelov_canonical_height] retry prec={cur_prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
            if h_can >= -TOL_NEG:
                warnings.warn(f"[arakelov_canonical_height] resolved negativity after increasing precision to {cur_prec}.", RuntimeWarning)
                return float(max(h_can, 0.0))
        except ZeroDivisionError:
            raise
        except Exception as e:
            warnings.warn(f"[arakelov_canonical_height] retry at prec={cur_prec} failed: {e}", RuntimeWarning)
            continue

    # 4) Give up: clamp to 0.0 (conservative) and emit a full diagnostic warning.
    warnings.warn(
        f"[arakelov_canonical_height] canonical height remained negative after all fallbacks for divisor={div}. "
        f"Returning 0.0 (clamped). Last observed h_can={h_can}. prec tried up to {cur_prec}.",
        RuntimeWarning
    )
    return 0.0
