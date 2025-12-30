"""Period matrix and Abel-Jacobi map computations."""

from sage.all import (
    ComplexField, RealField, PolynomialRing,
    Matrix, vector, QQ, ZZ
)
from sage.all import ComplexField, PolynomialRing, matrix
from .integration import *

# Functions: choose_numerical_base_point, abel_jacobi_mumford,
# normalize_periods_and_z


"""Period matrix and Abel-Jacobi map computations."""


# Functions: choose_numerical_base_point, abel_jacobi_mumford,
# normalize_periods_and_z

def choose_numerical_base_point(f_coeffs, prec=300):
    """
    Select a numerically safe base point (x, y) with y^2 = f(x).
    Strategy:
      - compute polynomial roots (branch points) in CC(prec).
      - choose the centroid of branch points and move it slightly off the real/branch
        locus by a controlled imaginary offset proportional to the local scale.
      - evaluate y = sqrt(f(x_base)) and choose a stable branch (prefer imag>0).
    Returns a pair (x_base, y_base) in the ComplexField(prec).
    """
    key = (tuple(map(complex, f_coeffs)), int(prec))
    if key in choose_numerical_base_point.cache:
        return choose_numerical_base_point.cache[key]

    CC = ComplexField(prec)
    R = PolynomialRing(CC, 'x')
    x = R.gen()

    # build polynomial with coefficients given descending (highest -> lowest)
    # ensure we handle python lists and sage vectors
    try:
        # create polynomial using Horner (more stable than sum of powers)
        f_poly = CC(f_coeffs[0])
        for c in f_coeffs[1:]:
            f_poly = f_poly * x + CC(c)
    except Exception:
        # fallback to constructing in Python then casting
        f_poly = sum(CC(c) * x**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
        raise

    # attempt to find roots
    try:
        roots = [r for r, m in f_poly.roots(multiplicities=True)]
    except Exception:
        roots = []
        raise

    # fallback simple safe point if we couldn't find roots
    if not roots:
        x_base = CC(1)
        fval = sum(CC(c) * x_base**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
        y_base = fval.sqrt()
        # pick the branch with positive imaginary part for consistency
        if y_base.imag() < 0:
            y_base = -y_base
        return (x_base, y_base)

    # convert to CC (they may already be CC) and compute centroid
    roots_cc = [CC(r) for r in roots]
    centroid = sum(roots_cc) / CC(len(roots_cc))

    # compute a length scale: max distance between roots (or 1 if tiny)
    max_dist = max([abs(roots_cc[i] - roots_cc[j]) for i in range(len(roots_cc)) for j in range(i+1, len(roots_cc))] + [CC(1)])
    # choose offset magnitude as a modest power of two relative to precision, but scaled to geometry
    eps = CC(2) ** (-(prec // 4))
    offset_mag = max(max_dist * CC(1e-6), eps)

    # move centroid slightly in the imaginary direction to avoid branch cuts
    x_base = centroid + CC(0, 1) * offset_mag

    # evaluate f and take a stable sqrt branch
    def f_at(z):
        s = CC(0)
        for c in f_coeffs:
            s = s * z + CC(c)
        return s

    fval = f_at(x_base)
    y_base = fval.sqrt()

    # prefer branch with positive imaginary part (unless both nearly real)
    if (abs(y_base.imag()) < abs((-y_base).imag())):
        # keep y_base
        pass
    else:
        y_base = -y_base

    # final guard: if y is extremely small, nudge x_base further away
    tiny = CC(2) ** (-prec // 2)
    if abs(y_base) < tiny:
        x_base = centroid + CC(0, 1) * (offset_mag * CC(10))
        fval = f_at(x_base)
        y_base = fval.sqrt()
        if y_base.imag() < 0:
            y_base = -y_base

    return (x_base, y_base)
choose_numerical_base_point.cache = {}

def normalize_periods_and_z(Omega, z_vec):
    """
    Normalize period matrix Omega and Abel-Jacobi vector z_vec.
    Omega may be either:
      - g x g (already tau)
      - g x 2g (Omega1 | Omega2) where tau = Omega1^{-1} * Omega2

    Returns (tau, z_norm) where:
      - tau is a g x g complex symmetric matrix with Im(tau) positive definite
      - z_norm is a g x 1 column vector (or None if z_vec is None)
    """
    from sage.all import Matrix, ComplexField, RealField, matrix, vector, QQ
    Omega = Matrix(Omega)  # canonicalize
    g = Omega.nrows()

    if Omega.ncols() not in (g, 2 * g):
        raise ValueError(f"Omega shape {Omega.nrows()}×{Omega.ncols()}, expected g or 2g cols")

    # If already g×g, treat as tau
    if Omega.ncols() == g:
        tau = Omega
        if z_vec is None:
            z_norm = None
        else:
            # Accept a sequence or matrix for z_vec and coerce to column vector
            z_temp = matrix(tau.base_ring(), g, 1)
            for i in range(g):
                z_temp[i, 0] = z_vec[i]
            z_norm = z_temp
    else:
        # Omega has 2g cols: split into Omega1 | Omega2
        Omega1 = Omega[:, :g]
        Omega2 = Omega[:, g:]
        # check invertibility with a numerically stable tolerance
        if Omega1.det() == 0:
            raise ValueError("Omega1 singular (determinant zero). Can't normalize periods.")
        Omega1_inv = Omega1.inverse()
        tau = Omega1_inv * Omega2

        if z_vec is None:
            z_norm = None
        else:
            z_temp = matrix(Omega.base_ring(), g, 1)
            for i in range(g):
                z_temp[i, 0] = z_vec[i]
            z_norm = Omega1_inv * z_temp

    # symmetry check (numerical tolerance based on base_ring precision if possible)
    try:
        prec = getattr(tau.base_ring(), 'precision', None)
    except Exception:
        prec = None
        raise
    tol = 10 ** (-10) if prec is None else 2 ** (-(prec // 2))

    # check symmetry: max |tau_ij - tau_ji|
    max_asym = max([abs(tau[i, j] - tau[j, i]) for i in range(g) for j in range(g)])
    if max_asym > tol:
        raise ValueError(f"tau not symmetric (max asymmetry = {max_asym})")

    # check Im(tau) positive definite: construct a real matrix and test eigenvalues
    RF = RealField(53 if prec is None else max(53, int(prec // 2)))
    Im_mat = Matrix(RF, g, g, [float((tau[i, j].imag()).real()) if hasattr(tau[i, j], 'imag') else float(complex(tau[i, j]).imag) for i in range(g) for j in range(g)])
    eigs = Im_mat.eigenvalues()
    min_eig = min([float(e) for e in eigs])
    if min_eig <= 1e-14:
        raise ValueError(f"Im(tau) not positive definite (smallest eigenvalue {min_eig})")

    return tau, z_norm



def abel_jacobi_mumford(
    div, f_coeffs, base_point,
    *, integrate_func=None, prec=300,
    period_matrix=None, debug=False
):
    """
    Abel–Jacobi map for a Mumford divisor on y^2 = f(x).
    FIXES:
      - No lattice reduction (raw coordinates only, to avoid rank collapse).
      - Basepoint must be global + consistent across calls.
      - Uses joint-path integration that returns both coordinates together.
    """
    from sage.all import ComplexField, PolynomialRing, vector

    CC = ComplexField(prec)
    R = PolynomialRing(CC, 'x')
    x = R.gen()

    # ---- cache keyed ONLY by divisor + curve + prec (basepoint must be global) ----
    key = (str(div), tuple(map(CC, f_coeffs)), int(prec))
    if key in abel_jacobi_mumford.cache:
        return abel_jacobi_mumford.cache[key]

    # -------- parse Mumford divisor ----------
    if isinstance(div, dict):
        s = CC(div.get("s", 0))
        p = CC(div.get("p", 0))
        v1 = CC(div.get("v_1", 0))
        v0 = CC(div.get("v_0", 0))
        u = x**2 - s*x + p
        v = v1*x + v0
    else:
        u, v = div[0], div[1]
        u = sum(CC(c) * x**i for i, c in enumerate(u.list()))
        v = sum(CC(c) * x**i for i, c in enumerate(v.list()))

    if u.degree() < 1:
        res = vector(CC, [0, 0])
        abel_jacobi_mumford.cache[key] = res
        return res

    x0, y0 = CC(base_point[0]), CC(base_point[1])

    if integrate_func is None:
        integrate_func = integrate_differential_path_joint
        #integrate_func = integrate_differential_path_joint

    # ---- roots of u(x) ----
    t = PolynomialRing(CC, 't').gen()
    u_cc = sum(CC(c) * t**i for i, c in enumerate(u.list()))
    roots = [CC(r) for r, m in u_cc.roots(multiplicities=False)]

    def f_at(z):
        s = CC(0)
        for c in f_coeffs:
            s = s * z + CC(c)
        return s

    def v_at(z):
        return sum(CC(c) * (z**i) for i, c in enumerate(v.list()))

    AJ = vector(CC, [0, 0])

    for xP in roots:
        fval = f_at(xP)
        vval = v_at(xP)

        # choose consistent sheet
        if abs(vval**2 - fval) < 1e-8 * max(1, abs(fval)):
            yP = vval
        else:
            s = fval.sqrt()
            yP = s if abs(s - vval) <= abs(-s - vval) else -s

        # 🚨 NEW: integrate BOTH ω₀, ω₁ together along ONE tracked path
        w0, w1 = integrate_func(x0, xP, y0, yP, f_coeffs, prec=prec, debug=debug)
        AJ += vector(CC, [w0, w1])

    # ❗ NO LATTICE REDUCTION — return raw AJ coordinates
    abel_jacobi_mumford.cache[key] = AJ
    return AJ

abel_jacobi_mumford.cache = {}


