"""Period matrix and Abel-Jacobi map computations."""

from sage.all import (
    ComplexField, RealField, PolynomialRing,
    Matrix, vector, QQ, ZZ
)

from .integration import integrate_differential_path_with_branch

# Functions: choose_numerical_base_point, abel_jacobi_mumford,
# normalize_periods_and_z




"""Period matrix and Abel-Jacobi map computations."""


# Functions: choose_numerical_base_point, abel_jacobi_mumford,
# normalize_periods_and_z

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
    key = (str(div), tuple(f_coeffs), base_point, prec)
    if key in abel_jacobi_mumford.cache:
        return abel_jacobi_mumford.cache[key]
    from sage.all import ComplexField, PolynomialRing, vector, matrix
    import math

    CC = ComplexField(prec)

    # [Fix] Handle dictionary input (Mumford dict) or Sage Jacobian element
    u_poly = None
    v_poly = None
    
    # Build a polynomial ring for reconstruction if needed
    R = PolynomialRing(CC, 'x')
    x = R.gen()

    if isinstance(div, dict):
        # Dictionary input: assume standard Mumford keys s, p, v_1, v_0
        try:
            s_val = CC(div.get('s', 0))
            p_val = CC(div.get('p', 0))
            # u = x^2 - s*x + p
            u_poly = x**2 - s_val*x + p_val
            
            v1_val = CC(div.get('v_1', 0))
            v0_val = CC(div.get('v_0', 0))
            v_poly = v1_val*x + v0_val
            
            # Check for trivial divisor (degree 0 u)
            if u_poly.degree() == 0:
                 return vector(CC, [0, 0])
        except Exception as e:
            if debug:
                print(f"[abel_jacobi] Failed to reconstruct from dict: {e}")
            raise
    elif hasattr(div, 'is_zero') and div.is_zero():
        return vector(CC, [0, 0])
    else:
        # Standard Jacobian element or tuple-like
        try:
            u_poly = div[0]
            v_poly = div[1]
        except (IndexError, TypeError):
             # Fallback check for is_zero if indexing failed
             if hasattr(div, 'is_zero') and div.is_zero():
                  return vector(CC, [0, 0])
             raise ValueError(f"abel_jacobi_mumford: cannot parse input type {type(div)}")

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

    # Convert u(x) and v(x) to CC polynomials (note: u_poly.list() gives coefficients from const->highest)
    u_list = u_poly.list()
    v_list = v_poly.list()

    # build u_cc for root finding (coeffs placed into polynomial ring)
    # Note: if u_poly was built above in R, u_list are already CC. 
    # If it was a QQ poly, they are QQ. We strictly cast to CC here.
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
        # multiplicity handling
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

        # Determine y_pt
        # Strategy: Trust v(x) if it satisfies y^2 = f(x), even if multiplicity > 1.
        # This fixes the bug where non-Weierstrass double points (2P) were forced to y=0.
        
        is_consistent = (abs(y_from_v**2 - fval) < tol_consistency) or (abs(y_from_v**2 - fval) < 1e-5 * max(1, abs(fval)))
        
        if is_consistent:
            y_pt = y_from_v
        else:
            # v is inconsistent. Try to recover by taking sqrt(f) closest to v.
            if debug:
                print(f"[abel_jacobi] Warning: v(x)^2 != f(x) at root {idx}. Using sqrt(f). Diff={abs(y_from_v**2 - fval)}")
            
            y_sqrt = fval.sqrt()
            if abs(y_from_v - y_sqrt) <= abs(y_from_v + y_sqrt):
                y_pt = y_sqrt
            else:
                y_pt = -y_sqrt
            
            # Double check for true branch points (y ~ 0) if v was inconsistent
            if abs(y_pt) < tol_consistency and abs(fval) < tiny:
                y_pt = CC(0)

        # At this point y_pt is the target y at x_pt we will pass to integrator.
        # Integrate two differentials: 1/(2y) and x/(2y)
        try:
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

        # [cite_start]CRITICAL FIX: Multiply contribution by multiplicity [cite: 52]
        # A double root in u means the point appears twice in the divisor sum.
        # Previously this was missing, causing h(2P) to be computed as h(P).
        aj_vec += CC(multiplicity) * vector(CC, [int0, int1])

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

    ret = aj_vec
    abel_jacobi_mumford.cache[key] = ret
    return ret

abel_jacobi_mumford.cache = {}
