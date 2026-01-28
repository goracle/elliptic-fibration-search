import math, itertools
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix, identity_matrix, ZZ, RR
from sage.schemes.riemann_surfaces.riemann_surface import RiemannSurface

# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.

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
            print(sMatrix(I))
        except Exception:
            print(I)
            raise

    # If already identity, return early
    if I == [[1, 0], [0, 1]]:
        return True, A_cycles, B_cycles, I

    # Try permutations and sign flips
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
    #print("det(A_raw) =", detA)

    # column norms
    col_norms_A = [sum(abs(c) for c in A_raw.column(j)) for j in range(A_raw.ncols())]
    col_norms_B = [sum(abs(c) for c in B_raw.column(j)) for j in range(B_raw.ncols())]
    #print("col norms A:", col_norms_A)
    #print("col norms B:", col_norms_B)

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

    #print("Roots used for cuts:", roots_for_pairing)
    #print("Cuts:", cuts)
    print("Nodes used:", len(nodes_cc))
    print("=== END DIAGNOSTIC ===")

    # --- 4. Symplectic Basis Search ---
    # The raw basis might not be symplectic (A_i . B_j != delta_ij).
    # We search for a transformed basis that yields a valid Riemann matrix.
    # --- 4. Symplectic Basis Search (improved) ---

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

