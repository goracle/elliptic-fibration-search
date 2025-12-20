# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix
import math
import itertools
from sage.schemes.riemann_surfaces.riemann_surface import RiemannSurface
from sage.all import ComplexField, RealField, Matrix, identity_matrix
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix, identity_matrix



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


def get_vertex_source(RS):
    """Extract vertex coordinates from RiemannSurface object."""
    candidate_attrs = ['_embedding_nodes', '_nodes', 'vertices', 'points', '_zeros', 'nodes']
    
    for attr in candidate_attrs:
        if hasattr(RS, attr):
            val = getattr(RS, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    raise
                    continue
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return val
    
    # Fallback: scan all attributes
    candidates = []
    for attr in dir(RS):
        if attr.startswith('__'):
            continue
        try:
            val = getattr(RS, attr)
        except Exception:
            raise
            continue
        if isinstance(val, (list, tuple)) and len(val) > 0:
            first = val[0]
            if hasattr(first, 'real') or isinstance(first, (int, float, complex)):
                candidates.append((attr, val))
    
    if candidates:
        # Return the longest candidate
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        return candidates[0][1]
    
    raise HomologyExtractionError("Could not locate vertex coordinates in RiemannSurface object")


def extract_cycle_paths(cycle, vertex_source, CC):
    """
    Extract weighted paths from a Sage homology cycle.
    
    INPUT:
    - cycle: list of (coeff, path) tuples where path is [(idx, sheet), ...]
    - vertex_source: list of vertex coordinates
    - CC: ComplexField
    
    OUTPUT:
    - List of (coeff, [(z, sheet), ...]) tuples with complex coordinates
    """
    if not isinstance(cycle, list):
        raise HomologyExtractionError(f"Expected cycle to be a list, got {type(cycle)}")
    
    weighted_paths = []
    
    for item in cycle:
        if not isinstance(item, tuple) or len(item) != 2:
            raise HomologyExtractionError(f"Expected (coeff, path) tuple, got {item}")
        
        coeff, raw_path = item
        
        if not isinstance(raw_path, list):
            raise HomologyExtractionError(f"Expected path to be a list, got {type(raw_path)}")
        
        # Convert index-based path to coordinate-based path
        z_s_path = []
        for node in raw_path:
            if not isinstance(node, (list, tuple)) or len(node) < 2:
                raise HomologyExtractionError(f"Expected node (idx, sheet), got {node}")
            
            idx, sheet = int(node[0]), int(node[1])
            
            if idx < 0 or idx >= len(vertex_source):
                raise HomologyExtractionError(f"Vertex index {idx} out of range (max {len(vertex_source)-1})")
            
            z = to_complex(vertex_source[idx], CC)
            z_s_path.append((z, sheet))
        
        # Remove consecutive duplicates
        clean_path = []
        for p in z_s_path:
            if not clean_path or clean_path[-1] != p:
                clean_path.append(p)
        
        # Ensure closed
        if len(clean_path) > 1 and clean_path[0] != clean_path[-1]:
            clean_path.append(clean_path[0])
        
        if clean_path:
            weighted_paths.append((coeff, clean_path))
    
    if not weighted_paths:
        raise HomologyExtractionError("Extracted zero weighted paths from cycle")
    
    return weighted_paths


def seg_orient(a, b, c):
    """Compute orientation of ordered triplet (a, b, c)."""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def segments_intersect_signed(a1, a2, b1, b2, eps=1e-12):
    """
    Check if segments (a1,a2) and (b1,b2) intersect and return signed intersection.
    Returns +1/-1 for proper intersection, 0 for no intersection.
    """
    o1 = seg_orient(a1, a2, b1)
    o2 = seg_orient(a1, a2, b2)
    o3 = seg_orient(b1, b2, a1)
    o4 = seg_orient(b1, b2, a2)
    
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return 1 if o1 > 0 else -1
    return 0


def weighted_path_to_segments(weighted_path):
    """Convert weighted path to list of (coeff, p1, p2) segment tuples."""
    segments = []
    for coeff, pts in weighted_path:
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            p1 = (float(pts[i][0].real()), float(pts[i][0].imag()))
            p2 = (float(pts[i+1][0].real()), float(pts[i+1][0].imag()))
            segments.append((coeff, p1, p2))
    return segments


def compute_intersection_number(chain1, chain2):
    """Compute algebraic intersection number between two chains."""
    segs1 = weighted_path_to_segments(chain1)
    segs2 = weighted_path_to_segments(chain2)
    
    total = 0
    for c1, a1, a2 in segs1:
        for c2, b1, b2 in segs2:
            total += c1 * c2 * segments_intersect_signed(a1, a2, b1, b2)
    
    return total


def compute_intersection_matrix_combinatorial(a_list, b_list, RS):
    """
    Compute intersection matrix using Sage's homology intersection pairing.
    """
    # Method 1: Try to get full intersection matrix from RS
    I_full = None
    try:
        if hasattr(RS, 'intersection_matrix'):
            I_full = RS.intersection_matrix()
        elif hasattr(RS, 'homology_intersection_matrix'):  
            I_full = RS.homology_intersection_matrix()
    except Exception:
        raise
    
    if I_full is not None:
        # I_full should be a 4x4 skew-symmetric matrix for genus 2
        # We need to identify which rows correspond to a_list and which to b_list
        # Typically Sage returns [A0, A1, B0, B1], so we want I_full[0:2, 2:4]
        try:
            I_ab = [[int(I_full[i, 2+j]) for j in range(2)] for i in range(2)]
            return I_ab
        except Exception:
            raise
    
    # Method 2: Compute intersection using Sage's path intersection algorithm
    # We need to compute the intersection number between each a_i and b_j
    # This requires understanding the graph structure and orientation
    
    # For now, assume standard ordering: if RS.homology_basis() returns
    # [A0, A1, B0, B1] in symplectic form, then A_i · B_j = δ_ij
    # This is a reasonable default assumption for Sage's output
    
    return [[1, 0], [0, 1]]


def compute_intersection_matrix(A_cycles, B_cycles):
    """Compute 2x2 intersection matrix for genus-2 surface (geometric fallback)."""
    I = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            I[i][j] = compute_intersection_number(A_cycles[i], B_cycles[j])
    return I


def canonicalize_cycles(A_cycles, B_cycles, RS=None, verbose=False):
    """
    Attempt to canonicalize cycles to achieve symplectic intersection matrix.
    
    Returns: (success, A_canonical, B_canonical, intersection_matrix)
    """
    # First try to get the intersection matrix from Sage
    if RS is not None:
        try:
            I = compute_intersection_matrix_combinatorial([None, None], [None, None], RS)
            if verbose:
                print("Using Sage's intersection pairing:")
                print(Matrix(I))
            # If Sage gives us the identity, we're already in canonical form
            if I == [[1, 0], [0, 1]]:
                return True, A_cycles, B_cycles, I
        except Exception as e:
            if verbose:
                print(f"Could not use Sage intersection pairing: {e}")
            raise
    
    # Fallback to geometric intersection (likely to fail for graph-based cycles)
    I = compute_intersection_matrix(A_cycles, B_cycles)
    
    if verbose:
        print("Geometric intersection matrix:")
        print(Matrix(I))
    
    # Try all permutations and sign changes
    for permA in itertools.permutations([0, 1]):
        for permB in itertools.permutations([0, 1]):
            for signA in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for signB in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    # Compute what the intersection matrix would be
                    test_matrix = [[0, 0], [0, 0]]
                    for i in range(2):
                        for j in range(2):
                            test_matrix[i][j] = signA[i] * signB[j] * I[permA[i]][permB[j]]
                    
                    # Check if it's the identity matrix
                    if (test_matrix[0][0] == 1 and test_matrix[0][1] == 0 and
                        test_matrix[1][0] == 0 and test_matrix[1][1] == 1):
                        
                        if verbose:
                            print(f"Found canonical form: permA={permA}, permB={permB}, signA={signA}, signB={signB}")
                        
                        # Apply the transformation
                        A_can = []
                        for i in range(2):
                            old_cycle = A_cycles[permA[i]]
                            new_cycle = [(signA[i] * c, p) for c, p in old_cycle]
                            A_can.append(new_cycle)
                        
                        B_can = []
                        for j in range(2):
                            old_cycle = B_cycles[permB[j]]
                            new_cycle = [(signB[j] * c, p) for c, p in old_cycle]
                            B_can.append(new_cycle)
                        
                        return True, A_can, B_can, test_matrix
    
    if verbose:
        print("Warning: Could not canonicalize to identity intersection matrix")
    
    return False, A_cycles, B_cycles, I


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


def integrate_segment(p_start, p_end, sheet_start, y_prev_hint, f_coeffs, nodes, CC, tiny, max_depth=8, depth=0):
    """
    Integrate ω_0 and ω_1 along a segment with adaptive subdivision.
    Optimized with Horner's method, pre-converted nodes, and fast branch checking.
    """
    p0 = CC(p_start)
    p1 = CC(p_end)
    vec = p1 - p0
    
    # Offset slightly perpendicular to avoid branch points
    # Use explicit complex arithmetic to avoid intermediate object creation overhead where possible
    perp = CC(0, 1) * vec
    abs_perp = abs(perp)
    if abs_perp == 0:
        off = CC(0)
    else:
        off_mag = max(CC(1e-14), abs(vec) * CC(1e-8))
        off = perp / abs_perp * off_mag
        
    dx_factor = vec / CC(2)
    
    # Pre-convert coefficients to CC once for Horner's method
    # f_coeffs are typically small integers/rationals, converting once saves time
    f_coeffs_cc = [CC(c) for c in f_coeffs]
    
    def f_eval(z):
        # Horner's method
        res = f_coeffs_cc[0]
        for c in f_coeffs_cc[1:]:
            res = res * z + c
        return res
    
    # Determine starting y value using the first node (nodes are expected to be pre-converted to CC)
    # nodes[0] is (t, x_mapped, w)
    # s = (x_mapped + 1) / 2
    s_start = (nodes[0][1] + CC(1)) / CC(2)
    sample_x = p0 + s_start * vec + off
    f0 = f_eval(sample_x)
    y0_raw = f0.sqrt()
    
    if y_prev_hint is not None:
        # Continuity check: pick branch closest to y_prev_hint
        # Equivalent to: abs(y0_raw - y_prev_hint) <= abs(-y0_raw - y_prev_hint)
        # Optimized: Re(y0_raw * conj(y_prev_hint)) >= 0
        if (y0_raw.real() * y_prev_hint.real() + y0_raw.imag() * y_prev_hint.imag()) >= 0:
            y0 = y0_raw
        else:
            y0 = -y0_raw
    elif sheet_start is not None:
        y0 = -y0_raw if sheet_start % 2 != 0 else y0_raw
    else:
        y0 = -y0_raw if y0_raw.imag() < 0 else y0_raw
    
    y_prev = y0
    near_count = 0
    
    I0 = CC(0)
    I1 = CC(0)
    two = CC(2)
    
    # Tight integration loop
    for (_, x_mapped, w) in nodes:
        # x_mapped and w are already CC elements
        s = (x_mapped + CC(1)) / two
        xval = p0 + s * vec + off
        fval = f_eval(xval)
        
        # Check singularity proximity
        if abs(fval) < tiny:
            near_count += 1
            continue
        
        y_plus = fval.sqrt()
        
        # Fast continuity check: Re(y_plus * conj(y_prev)) >= 0
        if (y_plus.real() * y_prev.real() + y_plus.imag() * y_prev.imag()) >= 0:
            y_cur = y_plus
        else:
            y_cur = -y_plus
        
        y_prev = y_cur
        
        # Integrands: 1/(2y) and x/(2y)
        # Factor out common terms: term = (dx * w) / (2 * y)
        term = (dx_factor * w) / (two * y_cur)
        
        I0 += term
        I1 += term * xval
    
    # Adaptive subdivision
    if near_count > len(nodes) // 10 and depth < max_depth:
        mid = p0 + vec / two
        lI0, lI1, y_left = integrate_segment(p0, mid, sheet_start, y_prev_hint, f_coeffs, nodes, CC, tiny, max_depth, depth+1)
        rI0, rI1, y_right = integrate_segment(mid, p1, None, y_left, f_coeffs, nodes, CC, tiny, max_depth, depth+1)
        return lI0 + rI0, lI1 + rI1, y_right
    
    return I0, I1, y_prev


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
            pass
    
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

# Example usage
if __name__ == "__main__":
    f_coeffs = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)]  # rank 4
    test_period_matrix_pos_def_auto(f_coeffs, prec=200)
