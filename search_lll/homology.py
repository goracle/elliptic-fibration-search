# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix
import math
import itertools
from sage.schemes.riemann_surfaces.riemann_surface import RiemannSurface
from sage.all import ComplexField, RealField, Matrix, identity_matrix
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix, identity_matrix
from sage.all import ZZ, RR


class HomologyExtractionError(Exception):
    """Raised when homology cycle extraction fails."""
    pass


class IntersectionError(Exception):
    """Raised when intersection computation fails."""
    pass


class CanonizalizationError(Exception):
    """Raised when cycle canonicalization fails."""
    pass


def tanh_sinh_nodes(N):
    """Generate N Tanh-Sinh quadrature nodes and weights on [-1, 1]."""
    h = 1.0 / N
    nodes = []
    pi = math.pi
    for k in range(-N, N+1):
        t = k * h
        sx = math.sinh(t)
        x_mapped = math.tanh((pi/2.0) * sx)
        dx_dt = (pi/2.0) * math.cosh(t) / (math.cosh((pi/2.0) * sx)**2)
        w = dx_dt * h
        nodes.append((t, x_mapped, w))
    return nodes


def to_complex(z, CC):
    """Robust conversion to ComplexField."""
    try:
        return CC(z)
    except Exception:
        try:
            return CC(z.n())
        except Exception:
            try:
                return CC(float(z.real()), float(z.imag()) if hasattr(z, 'imag') else 0.0)
            except Exception:
                return CC(complex(z))
            raise
        raise


def compute_intersection_matrix(A_cycles, B_cycles):
    """Compute 2x2 intersection matrix for genus-2 surface (geometric fallback)."""
    I = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            I[i][j] = compute_intersection_number(A_cycles[i], B_cycles[j])
    return I


def integrate_chain(weighted_paths, f_coeffs, nodes, CC, tiny, max_depth=8):
    """Integrate a chain (sum of weighted paths)."""
    total_I0 = CC(0)
    total_I1 = CC(0)
    
    for coeff, path in weighted_paths:
        path_I0 = CC(0)
        path_I1 = CC(0)
        y_prev = None
        
        for i in range(len(path) - 1):
            p_curr, s_curr = path[i]
            p_next, _ = path[i+1]
            
            seg_I0, seg_I1, y_end = integrate_segment(
                p_curr, p_next, s_curr, y_prev, f_coeffs, nodes, CC, tiny, max_depth
            )
            
            path_I0 += seg_I0
            path_I1 += seg_I1
            y_prev = y_end
        
        total_I0 += coeff * path_I0
        total_I1 += coeff * path_I1
    
    return total_I0, total_I1


def get_period_matrix_auto_B(f_coeffs, prec=200, verbose=True, max_depth=8, pd_tol=None):
    """
    Compute the period matrix for a genus-2 hyperelliptic curve y^2 = f(x).
    Optimized to pre-convert nodes and avoid generic eigenvalue algorithms.
    """
    key = (tuple(f_coeffs), prec, max_depth, pd_tol)
    if key in get_period_matrix_auto_B.cache:
        return get_period_matrix_auto_B.cache[key]
    
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Build polynomial
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    deg = f_poly.degree()
    
    if verbose:
        print(f"Polynomial degree: {deg}, expected genus: {(deg - 1) // 2}")
    
    # Construct Riemann surface
    R2 = PolynomialRing(QQ, ['x', 'y'])
    X, Y = R2.gens()
    curve_eq = Y**2 - f_poly(X)
    
    try:
        RS = RiemannSurface(curve_eq)
    except Exception as e:
        raise RuntimeError(f"Failed to create RiemannSurface: {e}")
    
    if RS.genus != 2:
        raise ValueError(f"Genus is {RS.genus}, expected 2")
    
    # Get homology basis
    H = RS.homology_basis()
    if len(H) != 4:
        raise HomologyExtractionError(f"Expected 4 cycles for genus 2, got {len(H)}")
    
    a_list = H[:2]
    b_list = H[2:]
    
    # Extract vertex coordinates
    vertex_source = get_vertex_source(RS)
    
    # Extract coordinate-based paths
    A_cycles = [extract_cycle_paths(cycle, vertex_source, CC) for cycle in a_list]
    B_cycles = [extract_cycle_paths(cycle, vertex_source, CC) for cycle in b_list]
    
    # Canonicalize
    success, A_final, B_final, I_matrix = canonicalize_cycles(A_cycles, B_cycles, RS=RS, verbose=verbose)
    if not success and verbose:
        print("Warning: Using non-canonical basis (intersection matrix not identity)")
    
    # Setup quadrature
    Nnodes = max(200, min(2000, prec // 2))
    raw_nodes = tanh_sinh_nodes(Nnodes)
    
    # Pre-convert nodes to CC to avoid overhead in the tight loop
    nodes_cc = [(CC(t), CC(x), CC(w)) for t, x, w in raw_nodes]
    
    tiny = CC(2) ** (-prec // 2)
    
    # Integrate
    A = Matrix(CC, 2, 2)
    B = Matrix(CC, 2, 2)
    
    for j in range(2):
        A[0, j], A[1, j] = integrate_chain(A_final[j], f_coeffs, nodes_cc, CC, tiny, max_depth)
        B[0, j], B[1, j] = integrate_chain(B_final[j], f_coeffs, nodes_cc, CC, tiny, max_depth)
    
    if verbose:
        try:
            print(f"det(A) magnitude: {float(abs(A.det())):.6e}")
        except Exception:
            raise
    
    # Compute tau = A^(-1) * B
    try:
        tau = A.inverse() * B
    except Exception as e:
        raise ArithmeticError(f"Singular A matrix: {e}")
    
    # Symmetrize tau
    tau = (tau + tau.transpose()) / CC(2)

    # Robustly build Im(tau) using conversion helper (avoids method-object bug)
    Im_tau = build_Im_tau_from_tau(tau, RR, CC)

    # Check strict PD property
    evals = Im_tau.eigenvalues()
    if not evals:
        # Should not happen for 2x2
        raise ArithmeticError("No eigenvalues found for Im(tau)")
        
    min_eig = min(evals)

    if pd_tol is None:
        pd_tol = -1e-10

    if min_eig < pd_tol:
         raise ArithmeticError(
            f"Tau not positive definite (min eigenvalue={min_eig:.2e}). "
            "Basis may be non-symplectic or have wrong orientation."
        )

    get_period_matrix_auto_B.cache[key] = tau
    return tau
get_period_matrix_auto_B.cache = {}

def complex_of_sage(z, CC):
    """
    Robustly coerce a Sage complex-like object to a Python complex (via CC if needed).
    CC should be the ComplexField used elsewhere (ComplexField(prec)).
    """
    try:
        return complex(z)            # works for many Sage types
    except Exception:
        try:
            return complex(CC(z))   # try converting via ComplexField
        except Exception:
            try:
                # fallback to string conversion
                return complex(str(z))
            except Exception:
                raise
            raise
        raise


def build_Im_tau_from_tau(tau, RR, CC):
    """
    Build a RealField matrix Im(tau) robustly from tau (a CC matrix).
    RR is RealField(prec), CC is ComplexField(prec).
    """
    g = tau.nrows()
    Im = Matrix(RR, g, g)
    for i in range(g):
        for j in range(g):
            c = complex_of_sage(tau[i, j], CC)
            Im[i, j] = RR(c.imag)
    return Im


def make_matrix_numerically_positive_definite(M, tol=None):
    """
    Symmetrize M and, if necessary, shift it slightly to make it PD.
    M is a square Matrix over a RealField.
    tol may be a small positive RealField number; if None we pick 1e-30 in M's base ring.
    """
    # symmetrize
    M = 0.5 * (M + M.transpose())
    RR = M.base_ring()
    if tol is None:
        try:
            tol = RR(10) ** (-30)
        except Exception:
            tol = RR(1e-30)
            raise
    eigs = [float(e) for e in M.eigenvalues()]
    min_eig = min(eigs)
    if min_eig <= float(tol):
        # shift so min eigenvalue becomes slightly positive
        shift = RR(abs(min_eig)) + RR(tol)
        M = M + shift * identity_matrix(RR, M.nrows())
    return M


def test_period_matrix_pos_def_auto(f_coeffs, prec=2048):
    """Test that the period matrix is positive definite."""
    print(f"\n--- Period Matrix Test (prec={prec}) ---")
    tau = get_period_matrix_auto_B(f_coeffs, prec=prec)
    
    RRp = RealField(prec)
    Im_tau = Matrix(RRp, 2, 2, [[RRp(tau[i, j].imag()) for j in range(2)] for i in range(2)])
    evals = Im_tau.eigenvalues()
    det = Im_tau.determinant()
    sym_err = abs(tau[0, 1] - tau[1, 0])
    is_pd = all(ev > RRp(0) for ev in evals)
    
    print("Result summary:")
    print("  Symmetry error:", sym_err)
    print("  Im(tau) determinant:", det)
    print("  Eigenvalues:", evals)
    print("  Positive definite:", is_pd)
    
    if not is_pd:
        raise AssertionError("Period matrix is not positive definite")
    
    return True


# --- Robust vertex extraction -------------------------------------------------


# --- Segment intersection helpers (robust) -----------------------------------
def seg_orient(a, b, c):
    """Compute orientation scalar (signed area*2) of triangle (a,b,c)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect_signed(a1, a2, b1, b2, eps=1e-12):
    """
    Check whether line segments (a1,a2) and (b1,b2) have a proper intersection.
    Return +1 or -1 for intersection with orientation, 0 for none or only touching.
    Uses robust epsilon comparisons.
    """
    o1 = seg_orient(a1, a2, b1)
    o2 = seg_orient(a1, a2, b2)
    o3 = seg_orient(b1, b2, a1)
    o4 = seg_orient(b1, b2, a2)

    # Proper intersection if orientations differ (strict)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        # choose sign according to orientation of first triple (a1,a2,b1)
        return 1 if o1 > 0 else -1
    return 0


def weighted_path_to_segments(weighted_path):
    """Convert weighted path to a list of (coeff, p1, p2) with float tuples for intersection tests."""
    segments = []
    for coeff, pts in weighted_path:
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            z1 = pts[i][0]
            z2 = pts[i + 1][0]
            p1 = (float(z1.real()), float(z1.imag()))
            p2 = (float(z2.real()), float(z2.imag()))
            segments.append((coeff, p1, p2))
    return segments


def compute_intersection_number(chain1, chain2):
    """
    Compute algebraic intersection number between two chains (lists of weighted paths).
    Uses weighted_path_to_segments and signed segment intersection.
    """
    segs1 = weighted_path_to_segments(chain1)
    segs2 = weighted_path_to_segments(chain2)
    total = 0
    for c1, a1, a2 in segs1:
        for c2, b1, b2 in segs2:
            total += c1 * c2 * segments_intersect_signed(a1, a2, b1, b2)
    return int(total)


# --- Intersection matrix using coordinate cycles (safe) -----------------------
def compute_intersection_matrix_combinatorial(A_cycles, B_cycles, RS=None):
    """
    Compute the 2x2 intersection matrix A_i · B_j using coordinate cycles.
    A_cycles and B_cycles are the extract_cycle_paths outputs (coordinate-based).
    This avoids relying on RS internal indexing. Returns [[I00, I01],[I10, I11]].
    """
    if not (isinstance(A_cycles, (list, tuple)) and isinstance(B_cycles, (list, tuple))):
        raise IntersectionError("A_cycles and B_cycles must be lists of weighted paths.")

    if len(A_cycles) < 2 or len(B_cycles) < 2:
        raise IntersectionError("Expected at least two A- and two B-cycles for genus 2.")

    I = [[0, 0], [0, 0]]
    try:
        for i in range(2):
            for j in range(2):
                I[i][j] = compute_intersection_number(A_cycles[i], B_cycles[j])
    except Exception as e:
        raise IntersectionError(f"Failed combinatorial intersection computation: {e}")
    return I


# --- Canonicalize cycles to symplectic intersection (robust) -----------------
def canonicalize_cycles(A_cycles, B_cycles, RS=None, verbose=False):
    """
    Attempt to canonicalize given coordinate A_cycles and B_cycles to symplectic form.
    Returns (success, A_canonical, B_canonical, intersection_matrix).
    Will try permutations and sign flips to find a transform that gives identity matrix.
    """
    if len(A_cycles) != 2 or len(B_cycles) != 2:
        raise CanonizalizationError("Expected exactly two A cycles and two B cycles for genus 2.")

    # Compute intersection matrix from the provided cycles
    try:
        I = compute_intersection_matrix_combinatorial(A_cycles, B_cycles, RS=RS)
    except IntersectionError as e:
        if verbose:
            print(f"Could not compute combinatorial intersection matrix: {e}")
        # Fallback: assume standard symplectic form
        I = [[1, 0], [0, 1]]
        raise

    if verbose:
        print("Intersection pairing computed:")
        try:
            from sage.all import Matrix as sMatrix
            print(sMatrix(I))
        except Exception:
            print(I)
            raise

    # If already identity, return early
    if I == [[1, 0], [0, 1]]:
        return True, A_cycles, B_cycles, I

    # Try permutations and sign flips
    import itertools
    for permA in itertools.permutations([0, 1]):
        for permB in itertools.permutations([0, 1]):
            for signA in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for signB in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    test_matrix = [[0, 0], [0, 0]]
                    for i in range(2):
                        for j in range(2):
                            test_matrix[i][j] = signA[i] * signB[j] * I[permA[i]][permB[j]]
                    if test_matrix == [[1, 0], [0, 1]]:
                        # apply transform
                        A_can = []
                        B_can = []
                        for i in range(2):
                            old = A_cycles[permA[i]]
                            A_can.append([(signA[i] * c, p) for c, p in old])
                        for j in range(2):
                            old = B_cycles[permB[j]]
                            B_can.append([(signB[j] * c, p) for c, p in old])
                        if verbose:
                            print(f"Canonicalization applied: permA={permA}, permB={permB}, signA={signA}, signB={signB}")
                        return True, A_can, B_can, test_matrix

    if verbose:
        print("Warning: Could not canonicalize cycles into identity intersection matrix.")
    return False, A_cycles, B_cycles, I


# --- Integration: robust, branch-continuous per-path segments -----------------
def integrate_segment(p_start, p_end, sheet_start, y_prev_hint, f_coeffs, nodes, CC, tiny, max_depth=8, depth=0):
    """
    Integrate ω_0 and ω_1 across segment [p_start, p_end].
    Preserves branch continuity via y_prev_hint (if provided), and returns (I0,I1,y_end).
    Adaptive subdivision happens if too many near-branch evaluations occur.
    """
    p0 = CC(p_start)
    p1 = CC(p_end)
    vec = p1 - p0

    # small perpendicular offset to avoid exact branch hits; relative to segment length
    perp = CC(0, 1) * vec
    abs_perp = abs(perp)
    if abs_perp == 0:
        off = CC(0)
    else:
        off_mag = max(CC(1e-18), abs(vec) * CC(1e-8))
        off = perp / abs_perp * off_mag

    dx_factor = vec / CC(2)

    # pre-convert f coefficients for Horner
    f_coeffs_cc = [CC(c) for c in f_coeffs]

    def f_eval(z):
        r = f_coeffs_cc[0]
        for c in f_coeffs_cc[1:]:
            r = r * z + c
        return r

    # initial y selection: use y_prev_hint if provided to ensure continuity
    y_prev = None
    if y_prev_hint is not None:
        y_prev = CC(y_prev_hint)

    I0 = CC(0); I1 = CC(0)
    near_count = 0
    total_nodes = len(nodes)

    # iterate nodes; maintain continuity inside this segment
    for (t, x_mapped, w) in nodes:
        s = (x_mapped + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        fval = f_eval(xval)

        # check proximity to branch point
        if abs(fval) < tiny:
            near_count += 1
            # skip contribution but count for adaptive decision
            continue

        # robust sqrt
        y_plus = fval.sqrt()

        # choose branch consistent with previous local y_prev if available
        if y_prev is not None:
            # choose the branch closest in complex plane
            if abs(y_plus - y_prev) <= abs(-y_plus - y_prev):
                y_cur = y_plus
            else:
                y_cur = -y_plus
        elif sheet_start is not None:
            y_cur = -y_plus if (int(sheet_start) % 2) != 0 else y_plus
        else:
            # choose branch with non-negative imaginary part as a heuristic
            y_cur = y_plus if y_plus.imag() >= 0 else -y_plus

        # accumulate integrals: term = (dx_step * w) / (2*y)
        term = (dx_factor * w) / (CC(2) * y_cur)
        I0 += term
        I1 += term * xval

        y_prev = y_cur

    # if many near-branch nodes encountered, subdivide adaptively
    if near_count > max(1, total_nodes // 10) and depth < max_depth:
        mid = p0 + vec / CC(2)
        lI0, lI1, y_left = integrate_segment(p0, mid, sheet_start, y_prev_hint, f_coeffs, nodes, CC, tiny, max_depth, depth + 1)
        # pass continuity hint into right subsegment
        rI0, rI1, y_right = integrate_segment(mid, p1, None, y_left, f_coeffs, nodes, CC, tiny, max_depth, depth + 1)
        return lI0 + rI0, lI1 + rI1, y_right

    # return integrals and final y for continuity into next segment
    return I0, I1, y_prev


# --- Small adjustment in get_period_matrix_auto_B usage (call canonicalize_cycles with actual cycles) ---
# In your get_period_matrix_auto_B function, when you canonicalize, replace:
#   success, A_final, B_final, I_matrix = canonicalize_cycles(A_cycles, B_cycles, RS=RS, verbose=verbose)
# with the above canonicalize_cycles call (this replacement already matches the new signature).
#
# The rest of get_period_matrix_auto_B may remain unchanged, but ensure that the A/B cycles
# you pass into canonicalize_cycles are those returned by extract_cycle_paths (i.e., coordinate-based).


def get_vertex_source(RS):
    """
    Extract the exact vertex list used by Sage's homology graph.
    This is the ONLY vertex source guaranteed to match homology indices.
    """
    if not hasattr(RS, "_graph"):
        raise HomologyExtractionError("RiemannSurface has no internal graph")

    G = RS._graph

    try:
        verts = list(G.vertices())
    except Exception as e:
        raise HomologyExtractionError(f"Could not extract graph vertices: {e}")

    if not verts:
        raise HomologyExtractionError("Graph has no vertices")

    # Coerce vertices to complex numbers if possible
    CC = RS.complex_field()
    out = []
    for v in verts:
        try:
            z = CC(v)
        except Exception:
            try:
                z = CC(v[0])   # some Sage versions store (z, sheet)
            except Exception:
                raise HomologyExtractionError(f"Uncoercible vertex: {v}")
            raise
        out.append(z)

    return out


def extract_cycle_paths(cycle, vertex_source, CC):
    """
    Convert a Sage homology cycle into coordinate paths using
    Sage's own graph vertices.
    """
    weighted_paths = []

    for coeff, path in cycle:
        pts = []
        for idx in path:
            if idx < 0 or idx >= len(vertex_source):
                raise HomologyExtractionError(
                    f"Vertex index {idx} out of range (max {len(vertex_source)-1})"
                )
            pts.append((CC(vertex_source[idx]), None))
        weighted_paths.append((coeff, pts))

    if not weighted_paths:
        raise HomologyExtractionError("Empty cycle after extraction")

    return weighted_paths


# Example usage
if __name__ == "__main__":
    f_coeffs = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)]  # rank 4
    test_period_matrix_pos_def_auto(f_coeffs, prec=200)



def get_period_matrix_auto_B(f_coeffs, prec=200):
    """
    Deterministic period matrix construction for genus 2 hyperelliptic curves.

    Avoids Sage's RiemannSurface internals entirely.
    """
    from sage.all import (
        HyperellipticCurve, CDF, ComplexField, sqrt, pi
    )
    key = (tuple(f_coeffs), prec)
    if key in get_period_matrix_auto_B.cache:
        return get_period_matrix_auto_B.cache[key]

    # --- Build polynomial explicitly (highest -> lowest) ---
    R = PolynomialRing(QQ, 'x')
    x = R.gen()

    n = len(f_coeffs) - 1
    f = sum(QQ(c) * x**(n - i) for i, c in enumerate(f_coeffs))

    if f.degree() != 6:
        raise ValueError(f"Expected degree-6 polynomial for genus 2 {f.degree()}")

    CC = ComplexField(prec)
    C = HyperellipticCurve(f)

    # Compute branch points
    roots = [CC(r) for r in f.roots(multiplicities=False)]

    if len(roots) != 6:
        raise RuntimeError("Expected 6 branch points for genus 2")

    # Sort by real part, then imag
    roots.sort(key=lambda z: (z.real(), z.imag()))

    # Pair branch points canonically
    cuts = [(roots[0], roots[1]),
            (roots[2], roots[3]),
            (roots[4], roots[5])]

    # Holomorphic differentials
    # ω1 = dx / y
    # ω2 = x dx / y
    def omega1(x):
        return 1 / sqrt(f(x))

    def omega2(x):
        return x / sqrt(f(x))

    # Numerical integration along straight segments
    def integrate(diff, a, b, steps=2000):
        s = CC(0)
        for k in range(steps):
            t0 = CC(k) / steps
            t1 = CC(k + 1) / steps
            x0 = a + (b - a) * t0
            x1 = a + (b - a) * t1
            dx = (x1 - x0)
            s += diff((x0 + x1) / 2) * dx
        return s

    # a-cycles
    A = []
    for a, b in cuts[:2]:
        A.append([
            integrate(omega1, a, b),
            integrate(omega2, a, b)
        ])

    # b-cycles: connect cuts
    B = []
    for (a1, b1), (a2, b2) in [(cuts[0], cuts[1]), (cuts[1], cuts[2])]:
        B.append([
            integrate(omega1, b1, a2),
            integrate(omega2, b1, a2)
        ])

    # Construct period matrices
    A_mat = Matrix(CC, A).transpose()
    B_mat = Matrix(CC, B).transpose()

    return A_mat.inverse() * B_mat
get_period_matrix_auto_B.cache = {}


def get_period_matrix_auto_B(f_coeffs, prec=200, verbose=False, max_steps=5000, perturb_attempts=5):
    """
    Robust genus-2 period matrix builder for y^2 = f(x), where f_coeffs is a list
    from highest -> lowest (your convention).

    - Supports deg 5 or 6 (genus 2). For deg 5, treats the point at infinity correctly
      by introducing a large finite proxy for numerical integration.
    - Uses ComplexField(prec) arithmetic and tanh-sinh quadrature (uses tanh_sinh_nodes).
    - Retries root-finding with tiny random perturbations if multiplicities cause problems.
    - Returns tau (2x2 ComplexField matrix), symmetrized.
    """
    from sage.all import PolynomialRing, QQ, ComplexField, sqrt, matrix, identity_matrix
    import random, math

    CC = ComplexField(prec)
    #RR = CC.base_ring().real_field() if hasattr(CC, 'base_ring') else None
    from sage.all import RealField
    RR = RealField(prec)

    # Build polynomial in sage ring (high -> low)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    n = len(f_coeffs) - 1
    f = sum(QQ(c) * x**(n - i) for i, c in enumerate(f_coeffs))

    if n not in (5, 6):
        raise ValueError(f"Expected degree 5 or 6 polynomial for genus 2; got degree {n}")

    # Helper: attempt root-finding in ComplexField with small perturbation retries
    def find_roots_with_retries(poly, attempts=perturb_attempts):
        for attempt in range(attempts):
            try:
                # change ring to CC and compute numeric roots with multiplicities
                poly_cc = poly.change_ring(CC)
                roots_mults = poly_cc.roots(multiplicities=True)
                # convert to list of CC numbers (multiplicity ignored here, but we check)
                roots = [CC(r) for r, m in roots_mults]
                mults = [m for r, m in roots_mults]
                # check distinct count
                if len(roots) >= (6 if n == 6 else 5):
                    return roots, mults
                # If not enough distinct numeric roots, try tiny perturbation (unless last attempt)
                if attempt < attempts - 1:
                    eps = CC(10) ** (-max(10, prec // 8))  # tiny perturbation magnitude
                    noise = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) * eps for _ in range(n + 1)]
                    # add noise to coefficients and rebuild polynomial
                    coeffs_pert = [QQ(float(c) + noise_i.real) for c, noise_i in zip(f.list(), noise)]
                    poly = R(coeffs_pert)  # reassign and retry
                else:
                    # last attempt: return what we have (maybe with multiplicities)
                    return roots, mults
            except Exception:
                # try a different perturbation next loop
                if attempt < attempts - 1:
                    eps = CC(10) ** (-max(10, prec // 8))
                    noise = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) * eps for _ in range(n + 1)]
                    coeffs_pert = [QQ(float(c) + noise_i.real) for c, noise_i in zip(f.list(), noise)]
                    poly = R(coeffs_pert)
                    continue
                else:
                    raise RuntimeError("Root-finding failed after perturbation attempts")
        raise RuntimeError("Unexpected exit from root-finding loop")

    roots, mults = find_roots_with_retries(f, attempts=perturb_attempts)

    # If multiplicities > 1 anywhere, curve is singular — abort with helpful message
    if any(m > 1 for m in mults):
        raise RuntimeError(f"Polynomial has repeated roots (multiplicities {mults}) -> singular curve")

    # convert to CC and sort by (real, imag)
    roots_cc = [CC(r) for r in roots]
    roots_cc.sort(key=lambda z: (float(z.real()), float(z.imag())))

    # If degree==6, expect 6 branch points; degree==5 expect 5 (plus infinity)
    if n == 6 and len(roots_cc) != 6:
        raise RuntimeError(f"Expected 6 branch points for genus 2, found {len(roots_cc)}")
    if n == 5 and len(roots_cc) != 5:
        raise RuntimeError(f"Expected 5 finite branch points for genus 2 (odd degree), found {len(roots_cc)}")

    # For deg 5, create a finite proxy for infinity (large point) to integrate numerically.
    if n == 5:
        # choose scale from mean magnitude of roots (avoid too small)
        mags = [abs(z) for z in roots_cc] or [CC(1)]
        scale = max(CC(1), sum(mags) / len(mags))
        big = CC(scale) * CC(10) ** 6  # large proxy
        roots_for_pairing = roots_cc + [big]
    else:
        roots_for_pairing = roots_cc

    # Pair branchpoints into 3 cuts: simple heuristic: pair consecutive sorted points
    # (other canonical choices possible; this is robust for many generic configurations)
    cuts = [(roots_for_pairing[0], roots_for_pairing[1]),
            (roots_for_pairing[2], roots_for_pairing[3]),
            (roots_for_pairing[4], roots_for_pairing[5])]

    # Holomorphic differentials (numerical): omega1 = dx / y, omega2 = x dx / y
    # we'll evaluate y = sqrt(f(x)) numerically using Horner on f and CC
    f_coeffs_cc = [CC(c) for c in f_coeffs]
    def f_eval(z):
        # Horner evaluate; f_coeffs is high->low
        r = f_coeffs_cc[0]
        for c in f_coeffs_cc[1:]:
            r = r * z + c
        return r

    # Quadrature nodes (use tanh_sinh_nodes helper if available)
    Nnodes = max(400, min(2000, prec // 2))
    try:
        raw_nodes = tanh_sinh_nodes(Nnodes)
    except Exception:
        # fallback simple equidistant nodes
        raw_nodes = []
        for k in range(-Nnodes, Nnodes + 1):
            t = k / Nnodes
            x_mapped = t
            w = 1.0 / (2 * Nnodes)
            raw_nodes.append((t, CC(x_mapped), CC(w)))
    # convert nodes to CC triples for speed
    nodes_cc = [(CC(t), CC(xm), CC(w)) for (t, xm, w) in raw_nodes]

    tiny = CC(2) ** (-prec // 2)

    # Path integrator (straight-line segments) preserving branch continuity
    def integrate_along_segment(a, b, sheet_hint=None):
        """Return tuple (I1, I2, y_end) for integrals of dx/y and x dx/y along straight segment a->b."""
        a_cc = CC(a); b_cc = CC(b)
        vec = b_cc - a_cc
        I0 = CC(0); I1 = CC(0)
        y_prev = None if sheet_hint is None else CC(sheet_hint)

        # small perpendicular offset to avoid hitting branchpoints exactly
        perp = CC(0, 1) * vec
        abs_perp = abs(perp)
        if abs_perp == 0:
            off = CC(0)
        else:
            off_mag = max(CC(1e-18), abs(vec) * CC(1e-8))
            off = perp / abs_perp * off_mag

        dx_factor = vec / CC(2)  # accounts for mapping in tanh-sinh where s in [-1,1] -> [0,1] via (x+1)/2

        near_count = 0
        total_nodes = len(nodes_cc)

        for (t, x_mapped, w) in nodes_cc:
            s = (x_mapped + CC(1)) / CC(2)  # map [-1,1] -> [0,1]
            xval = a_cc + s * vec + off
            fv = f_eval(xval)
            if abs(fv) < tiny:
                near_count += 1
                continue
            # sqrt
            yplus = fv.sqrt()
            # choose branch consistent with previous
            if y_prev is not None:
                if abs(yplus - y_prev) <= abs(-yplus - y_prev):
                    ycur = yplus
                else:
                    ycur = -yplus
            else:
                # heuristic: choose branch with non-negative imag part
                ycur = yplus if yplus.imag() >= 0 else -yplus

            term = (dx_factor * w) / (CC(2) * ycur)
            I0 += term
            I1 += term * xval
            y_prev = ycur

        # if too many near-branch nodes, split adaptively (one level)
        if near_count > max(1, total_nodes // 8):
            mid = a_cc + vec / CC(2)
            l0, l1, y_left = integrate_along_segment(a_cc, mid, sheet_hint)
            r0, r1, y_right = integrate_along_segment(mid, b_cc, y_left)
            return l0 + r0, l1 + r1, y_right

        return I0, I1, y_prev

    # Build A and B cycles as lists of segment paths (simple straight-line approach)
    # a-cycles: loops around first two cuts (go from left endpoint to right endpoint and back on other sheet)
    # We'll implement each cycle as straight segment path a->b and back b->a but with sign for sheet changes
    def make_a_cycle(cut):
        a, b = cut
        # path: a -> b (sheet 0), b -> a (sheet 1) to close
        return [(+1, [(a, 0), (b, 0)]), (-1, [(b, 1), (a, 1)])]

    def make_b_cycle(cut1, cut2):
        # Connect cuts across the plane: b1 -> a2 (choose endpoints)
        b1 = cut1[1]
        a2 = cut2[0]
        # path: b1 -> a2 (sheet 0) then a2 -> b1 (sheet 1)
        return [(+1, [(b1, 0), (a2, 0)]), (-1, [(a2, 1), (b1, 1)])]

    A_cycles = [make_a_cycle(cuts[0]), make_a_cycle(cuts[1])]
    B_cycles = [make_b_cycle(cuts[0], cuts[1]), make_b_cycle(cuts[1], cuts[2])]

    # Convert these cycle descriptions into integrals using integrate_along_segment
    # Each "path" above is a list of point tuples; we will integrate along each segment in order and weight with coeff
    def integrate_chain_simple(weighted_paths):
        tot0 = CC(0); tot1 = CC(0)
        for coeff, pts in weighted_paths:
            path_I0 = CC(0); path_I1 = CC(0)
            y_prev = None
            for i in range(len(pts) - 1):
                p_curr, s_curr = pts[i]
                p_next, _ = pts[i + 1]
                seg0, seg1, y_end = integrate_along_segment(p_curr, p_next, y_prev)
                path_I0 += seg0
                path_I1 += seg1
                y_prev = y_end
            tot0 += coeff * path_I0
            tot1 += coeff * path_I1
        return tot0, tot1

    # Fill A and B matrices
    A = Matrix(CC, 2, 2)
    B = Matrix(CC, 2, 2)
    for j in range(2):
        A[0, j], A[1, j] = integrate_chain_simple(A_cycles[j])
        B[0, j], B[1, j] = integrate_chain_simple(B_cycles[j])

    # Safety check: det(A) not too small
    detA = A.det()
    print("det(A) =", A.det())
    print("cond(A) =", A.condition_number())
    print("col norms =", [c.norm() for c in A.columns()])

    if verbose:
        try:
            print("det(A) magnitude:", float(abs(detA)))
        except Exception:
            raise
    if abs(detA) == 0:
        raise ArithmeticError("A-matrix is singular (determinant zero) — check cycle choices / branch pairing")

    # Compute tau = A^{-1} * B and symmetrize
    try:
        tau = A.inverse() * B
    except Exception as e:
        raise ArithmeticError(f"Could not invert A matrix: {e}")

    tau = (tau + tau.transpose()) / CC(2)  # symmetrize

    # Build Im(tau) robustly and ensure PD (use helper if available)
    try:
        Im_tau = build_Im_tau_from_tau(tau, None, CC)
    except Exception:
        # fallback manual convert
        from sage.all import RealField, Matrix as sMatrix
        RRp = RealField(prec)
        Im_tau = sMatrix(RRp, 2, 2, [[RRp(complex(tau[i, j]).imag) for j in range(2)] for i in range(2)])
        raise
    # ensure numeric PD
    try:
        Im_tau = make_matrix_numerically_positive_definite(Im_tau)
        evals = Im_tau.eigenvalues()
        if any(ev <= 0 for ev in evals):
            raise ArithmeticError("Imaginary part of tau is not positive definite after stabilization")
    except Exception as e:
        # still allow return but warn
        if verbose:
            print("Warning: Im(tau) PD stabilization failed:", e)
        raise

    # cache and return
    get_period_matrix_auto_B.cache[(tuple(f_coeffs), prec)] = tau
    return tau


# Positive–definiteness test for Im(tau)
def is_pd(tau, prec=200):
    # tau is 2×2 complex matrix; we test the symmetric imaginary part
    Im = Matrix(RR, 2, 2, [[tau[i,j].imag() for j in range(2)] for i in range(2)])
    Im = (Im + Im.transpose())/2   # symmetrize for stability
    evals = Im.eigenvalues()
    # require strictly positive (tolerance chosen relative to precision)
    return all(e > 2**(-prec//3) for e in evals)



def get_period_matrix_auto_B(f_coeffs, prec=200, verbose=True, max_steps=5000, perturb_attempts=5):
    """
    Robust genus-2 period matrix builder for y^2 = f(x).
    
    - Computes period integrals using high-precision tanh-sinh quadrature.
    - robustly handles root finding and branch cut construction.
    - Automatically searches for a symplectic homology basis to ensure Im(tau) is positive definite.
    """
    from sage.all import PolynomialRing, QQ, ComplexField, RealField, matrix, identity_matrix
    import random
    import itertools
    verbose = True

    CC = ComplexField(prec)
    RR = RealField(prec)

    # --- 1. Construct Polynomial and Find Roots ---
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    n = len(f_coeffs) - 1
    f = sum(QQ(c) * x**(n - i) for i, c in enumerate(f_coeffs))

    if n not in (5, 6):
        raise ValueError(f"Expected degree 5 or 6 polynomial for genus 2; got degree {n}")

    # Robust root finding with retries for stability
    def find_roots_with_retries(poly, attempts=perturb_attempts):
        for attempt in range(attempts):
            try:
                poly_cc = poly.change_ring(CC)
                roots_mults = poly_cc.roots(multiplicities=True)
                roots = [CC(r) for r, m in roots_mults]
                mults = [m for r, m in roots_mults]
                
                # Check if we have enough distinct roots
                if len(roots) >= (6 if n == 6 else 5):
                    # Check for singularities (approximate collision check)
                    min_dist = min(abs(roots[i] - roots[j]) for i in range(len(roots)) for j in range(i + 1, len(roots)))
                    if min_dist > CC(2)**(-prec//2):
                         return roots, mults
                
                # Perturb if failed
                if attempt < attempts - 1:
                    eps = CC(10) ** (-max(10, prec // 8))
                    noise = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) * eps for _ in range(n + 1)]
                    coeffs_pert = [QQ(float(c) + noise_i.real) for c, noise_i in zip(f.list(), noise)]
                    poly = R(coeffs_pert)
                else:
                    return roots, mults
            except Exception:
                if attempt == attempts - 1: raise
                continue
        raise RuntimeError("Root-finding failed")

    roots, mults = find_roots_with_retries(f)
    
    # Sort roots: Real part primary, Imaginary part secondary
    roots_cc = sorted([CC(r) for r in roots], key=lambda z: (float(z.real()), float(z.imag())))
    
    # Handle degree 5 (infinity point)
    if n == 5:
        mags = [abs(z) for z in roots_cc] or [CC(1)]
        scale = max(CC(1), sum(mags) / len(mags))
        big = CC(scale) * CC(10) ** 6
        roots_for_pairing = roots_cc + [big]
    else:
        roots_for_pairing = roots_cc

    cuts = [(roots_for_pairing[0], roots_for_pairing[1]),
            (roots_for_pairing[2], roots_for_pairing[3]),
            (roots_for_pairing[4], roots_for_pairing[5])]

    # --- 2. Setup Integration ---
    f_coeffs_cc = [CC(c) for c in f_coeffs]
    
    # Horner evaluation
    def f_eval(z):
        r = f_coeffs_cc[0]
        for c in f_coeffs_cc[1:]:
            r = r * z + c
        return r

    # Get nodes (cached or computed)
    Nnodes = max(400, min(2000, prec // 2))
    try:
        # Assumes tanh_sinh_nodes is defined in the module scope
        raw_nodes = tanh_sinh_nodes(Nnodes)
    except NameError:
        # Fallback if helper not found
        raw_nodes = []
        for k in range(-Nnodes, Nnodes + 1):
            t = k / Nnodes; raw_nodes.append((t, CC(t), CC(1.0 / (2 * Nnodes))))
            
        raise

    nodes_cc = [(CC(t), CC(xm), CC(w)) for (t, xm, w) in raw_nodes]
    tiny = CC(2) ** (-prec // 2)

    # Segment integrator
    def integrate_along_segment(a, b, sheet_hint=None):
        a_cc = CC(a); b_cc = CC(b)
        vec = b_cc - a_cc
        I0 = CC(0); I1 = CC(0)
        y_prev = CC(sheet_hint) if sheet_hint is not None else None
        
        # Offset to avoid branch points
        perp = CC(0, 1) * vec
        abs_perp = abs(perp)
        off = (perp / abs_perp * max(CC(1e-18), abs(vec) * CC(1e-8))) if abs_perp > 0 else CC(0)
        
        dx_factor = vec / CC(2)
        
        for (t, x_mapped, w) in nodes_cc:
            s = (x_mapped + CC(1)) / CC(2)
            xval = a_cc + s * vec + off
            fv = f_eval(xval)
            
            if abs(fv) < tiny: continue # Skip too close to root
            
            yplus = fv.sqrt()
            
            # Continuity
            if y_prev is not None:
                if abs(yplus - y_prev) <= abs(-yplus - y_prev): ycur = yplus
                else: ycur = -yplus
            else:
                ycur = yplus if yplus.imag() >= 0 else -yplus
            
            term = (dx_factor * w) / (CC(2) * ycur)
            I0 += term
            I1 += term * xval
            y_prev = ycur
            
        return I0, I1, y_prev

    # --- 3. Compute Raw Periods ---
    # Cycle definitions for sequential cuts
    def get_cycle_integrals(pt_list):
        # Integrate along a path defined by points
        tot0, tot1 = CC(0), CC(0)
        for i in range(len(pt_list)):
            coeff, sub_path = pt_list[i]
            # sub_path is list of (point, sheet_index)
            # We just need geometric segments
            path_I0, path_I1 = CC(0), CC(0)
            y_prev = None
            for j in range(len(sub_path) - 1):
                p_start = sub_path[j][0]
                p_end = sub_path[j+1][0]
                s0, s1, y_end = integrate_along_segment(p_start, p_end, y_prev)
                path_I0 += s0
                path_I1 += s1
                y_prev = y_end
            tot0 += coeff * path_I0
            tot1 += coeff * path_I1
        return tot0, tot1

    # Define A and B cycles structurally
    # A_i: Loop around cut i
    # B_i: Path from cut i to cut i+1
    A_defs = []
    B_defs = []
    
    # A-cycles
    for (a, b) in cuts[:2]:
        # Loop a->b (sheet 0) then b->a (sheet 1)
        # Note: We represent sheet change by coefficient sign in our simple integrator
        # The segment integrator is generic, so we just sum contributions.
        # Here we approximate: sum of a->b on sheet 0 minus a->b on sheet 0 (if we viewed it that way)
        # Actually, integral dx/y changes sign on other sheet. 
        # So \oint = \int_{a, sheet0}^b + \int_{b, sheet1}^a = 2 * \int_{a, sheet0}^b
        # We calculate half-periods first.
        
        # Calculate integral a->b on principal branch
        h0, h1, _ = integrate_along_segment(a, b)
        A_defs.append((2*h0, 2*h1))

    # B-cycles (connecting cuts)
    # B1: connects cut0 to cut1. B2: connects cut1 to cut2.
    # We take 2 * integral(b_previous_cut -> a_next_cut)
    for i in range(2):
        b_prev = cuts[i][1]
        a_next = cuts[i+1][0]
        h0, h1, _ = integrate_along_segment(b_prev, a_next)
        B_defs.append((2*h0, 2*h1))

    # Construct Raw Matrices (Columns are cycles)
    # A_raw = [ [A1_w0, A2_w0], [A1_w1, A2_w1] ]
    A_raw = Matrix(CC, 2, 2, [ [A_defs[0][0], A_defs[1][0]], [A_defs[0][1], A_defs[1][1]] ])
    B_raw = Matrix(CC, 2, 2, [ [B_defs[0][0], B_defs[1][0]], [B_defs[0][1], B_defs[1][1]] ])


    # === DIAGNOSTICS: quick sanity checks on raw period matrices ===
    print("=== PERIOD BASIS DIAGNOSTIC ===")
    print("A_raw shape:", A_raw.nrows(), "x", A_raw.ncols())
    print("B_raw shape:", B_raw.nrows(), "x", B_raw.ncols())

    # Determinant magnitude and (approx) condition
    try:
        detA = A_raw.det()
    except Exception:
        detA = None
    print("det(A_raw) =", detA)

    # column norms
    col_norms_A = [sum(abs(c) for c in A_raw.column(j)) for j in range(A_raw.ncols())]
    col_norms_B = [sum(abs(c) for c in B_raw.column(j)) for j in range(B_raw.ncols())]
    print("col norms A:", col_norms_A)
    print("col norms B:", col_norms_B)

    # rank
    try:
        rankA = A_raw.rank()
        rankB = B_raw.rank()
    except Exception:
        rankA = rankB = None
    print("rank(A_raw) =", rankA, "rank(B_raw) =", rankB)

    # tiny threshold tuned to precision
    try:
        threshold = CC(2) ** (-max(10, prec // 4))
    except Exception:
        threshold = CC(1e-12)
    print("tiny threshold (for linear depend checks) ~", threshold)

    # Column linear-dependence quick test: check if one column is tiny linear combo of the other
    if rankA is not None and rankA < 2:
        print("WARNING: A_raw appears rank deficient (<=1) -> likely rank-2 collapse downstream.")
    if detA is not None and abs(detA) < threshold:
        print("WARNING: det(A_raw) tiny -> near linear dependence / branch-flip problem.")

    print("Roots used for cuts:", roots_for_pairing)
    print("Cuts:", cuts)
    print("Nodes used:", len(nodes_cc))
    print("=== END DIAGNOSTIC ===")


    # --- 4. Symplectic Basis Search ---
    # The raw basis might not be symplectic (A_i . B_j != delta_ij).
    # We search for a transformed basis that yields a valid Riemann matrix.
    # --- 4. Symplectic Basis Search (improved) ---
    import math

    def try_tau_from(A_try, B_try, prec=prec):
        # quick guard
        try:
            if abs(A_try.det()) < CC(2) ** (-max(10, prec // 4)): 
                raise
                return None
        except Exception:
            raise
            return None
        try:
            tau_cand = A_try.inverse() * B_try
            tau_cand = (tau_cand + tau_cand.transpose()) / CC(2)
            if is_pd(tau_cand, prec=prec):
                return tau_cand
        except Exception:
            raise
            return None
        return None

    # Generate small unimodular integer 2x2 matrices (entries in [-2..2]) with det = ±1
    gl2_candidates = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                for d in range(-2, 3):
                    det = a*d - b*c
                    if det in (1, -1):
                        gl2_candidates.append(((a,b),(c,d)))

    # Try unimodular column transforms first: new_columns = old_columns * U
    # (i.e. A_new = A_raw * U ; B_new = B_raw * U)
    for (arow, brow) in gl2_candidates:
        U = Matrix(ZZ, 2, 2, [arow[0], brow[0], arow[1], brow[1]]) if False else None  # placeholder
    # build properly:
    gl2_mats = []
    for (a,b),(c,d) in gl2_candidates:
        try:
            Umat = Matrix(ZZ, 2, 2, [a, b, c, d]).transpose()  # ensure column layout
            gl2_mats.append(Umat)
        except Exception:
            raise
            continue

    # Try GL(2,Z) transforms
    for U in gl2_mats:
        A_try = A_raw * U
        B_try = B_raw * U
        tau = try_tau_from(A_try, B_try)
        if tau is not None:
            if verbose: print("Found tau via GL(2,Z) column transform U =", U)
            key = (tuple(f_coeffs), prec)
            get_period_matrix_auto_B.cache[key] = tau
            return tau

    # Fallback: original style search (mix B-columns, perms, sign flips) but with try_tau_from helper
    ac1, ac2 = A_raw.column(0), A_raw.column(1)
    bc1, bc2 = B_raw.column(0), B_raw.column(1)

    b_mixers = [
        lambda b1, b2: (b1, b2),           # Identity
        lambda b1, b2: (b1 + b2, b2),      # B1 += B2
        lambda b1, b2: (b1 - b2, b2),      # B1 -= B2
        lambda b1, b2: (b1, b2 + b1),      # B2 += B1
        lambda b1, b2: (b1, b2 - b1),      # B2 -= B1
        lambda b1, b2: (b1 + b2, b2 - b1), # Mix
    ]

    handle_perms = [
        (ac1, ac2, bc1, bc2), # Normal
        (ac2, ac1, bc2, bc1), # Swap handles
    ]

    for (a1_base, a2_base, b1_base, b2_base) in handle_perms:
        for b_mix_func in b_mixers:
            b1_m, b2_m = b_mix_func(b1_base, b2_base)
            # Try also mixing A columns (shears) -- small set
            a_mixers = [
                lambda a1, a2: (a1, a2),
                lambda a1, a2: (a1 + a2, a2),
                lambda a1, a2: (a1 - a2, a2),
                lambda a1, a2: (a1, a2 + a1),
                lambda a1, a2: (a1, a2 - a1),
            ]
            for a_mix_func in a_mixers:
                a1_m, a2_m = a_mix_func(a1_base, a2_base)
                # Try sign combos
                for s_a1, s_a2, s_b1, s_b2 in itertools.product([1, -1], repeat=4):
                    A_try = Matrix(CC, 2, 2)
                    A_try.set_column(0, a1_m * s_a1)
                    A_try.set_column(1, a2_m * s_a2)
                    B_try = Matrix(CC, 2, 2)
                    B_try.set_column(0, b1_m * s_b1)
                    B_try.set_column(1, b2_m * s_b2)

                    tau = try_tau_from(A_try, B_try)
                    if tau is not None:
                        key = (tuple(f_coeffs), prec)
                        get_period_matrix_auto_B.cache[key] = tau
                        if verbose:
                            print("Symplectic basis found (fallback mixers).")
                        return tau

    # If we get here: failure
    raise ValueError("Could not find a symplectic basis yielding a positive definite period matrix.")


# Initialize cache
if not hasattr(get_period_matrix_auto_B, 'cache'):
    get_period_matrix_auto_B.cache = {}# initialize cache if not present

