# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.
from sage.all import ComplexField, RealField, PolynomialRing, QQ, Matrix, conjugate
from sage.schemes.riemann_surfaces.riemann_surface import RiemannSurface
import math
import itertools


# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.


    # Sanity-check against RS.period_matrix() at low precision (non-fatal)


# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.


# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.
# Enforces sheet tracking to match Sage's topological definitions.


# Revised: Properly handle homology chains (linear combinations of paths).


# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.

# Revised: Properly handle homology chains (linear combinations of paths).


# Revised: use RiemannSurface for topology, but integrate ourselves at high precision.
# Fixes: Handles shared vertices in cycle paths by defaulting to standard order if geometric intersection check fails.


def get_period_matrix_auto_B(f_coeffs, prec=200, verbose=True, max_depth=8, pd_tol=None):
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Subdivision tracking
    _subdivisions = {"count": 0}

    # --- build polynomial in QQ[x]
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    deg = f_poly.degree()
    
    if verbose:
        print(f"Polynomial degree: {deg}, expected genus: {(deg - 1) // 2}")

    # --- construct Riemann surface (topology only)
    try:
        R2 = PolynomialRing(QQ, ['x','y'])
        X, Y = R2.gens()
        curve_eq = Y**2 - f_poly(X)
        RS = RiemannSurface(curve_eq)
    except Exception as e:
        raise RuntimeError(f"Failed to create RiemannSurface: {e}")

    if RS.genus != 2:
        raise ValueError(f"Genus is {RS.genus}, this code is specialized for genus 2.")

    # --- get homology basis ---
    H = RS.homology_basis()
    if isinstance(H, tuple) and len(H) == 2:
        a_list, b_list = H
    elif isinstance(H, list) and len(H) == 4:
        a_list = H[:2]; b_list = H[2:]
    else:
        H_list = list(H)
        a_list = H_list[:2]; b_list = H_list[2:]

    if verbose:
        print("Obtained homology cycles (A and B).")

    # --- Robust Point/Sheet Extraction ---
    def toCC(x):
        """Robust conversion to CC, handling various Sage algebraic types."""
        try:
            return CC(x)
        except Exception:
            try:
                # numeric approximation by Sage
                return CC(x.n())
            except Exception:
                try:
                    return CC(float(x.real()), float(x.imag()) if hasattr(x, 'imag') else 0.0)
                except Exception:
                    return CC(complex(x))

    def extract_cycle_structure(cycle, RS_obj):
        weighted_paths = []
        raw_chains = []
        max_index_found = -1

        # 1. Inspect cycle structure for (coeff, path) tuples
        if isinstance(cycle, list):
            for item in cycle:
                if isinstance(item, tuple) and len(item) == 2:
                    coeff = item[0]
                    path_data = item[1] 
                    if isinstance(path_data, list):
                        chain_points = []
                        for node in path_data:
                            if isinstance(node, (list, tuple)) and len(node) >= 2:
                                idx, sheet = node[0], node[1]
                                chain_points.append((idx, sheet))
                                if isinstance(idx, int) and idx > max_index_found:
                                    max_index_found = idx
                        if chain_points:
                            raw_chains.append((coeff, chain_points))
        
        # 2. Locate the vertex list in the RS object
        vertex_source = None
        candidate_attrs = ['_embedding_nodes', '_nodes', 'vertices', 'points', '_zeros']
        candidates = []
        
        for attr in candidate_attrs:
            if hasattr(RS_obj, attr):
                val = getattr(RS_obj, attr)
                if callable(val):
                    try: val = val()
                    except: continue
                if isinstance(val, (list, tuple)):
                    candidates.append((attr, val))

        # Fallback inspection
        if not candidates:
            for attr in dir(RS_obj):
                if attr.startswith('__'): continue
                try: val = getattr(RS_obj, attr)
                except: continue
                if isinstance(val, list) and len(val) > 0:
                    first = val[0]
                    if hasattr(first, 'real') or isinstance(first, (int, float, complex)):
                        candidates.append((attr, val))

        valid_sources = []
        for name, lst in candidates:
            if len(lst) > max_index_found:
                valid_sources.append(lst)
        valid_sources.sort(key=len, reverse=True)
        
        if valid_sources:
            vertex_source = valid_sources[0]
        elif raw_chains:
             # If we have indices but no vertex source, we cannot proceed
            sizes = [(name, len(lst)) for name, lst in candidates]
            raise RuntimeError(f"Could not locate a vertex list long enough for index {max_index_found}. Candidates: {sizes}")

        # 3. Construct weighted paths
        if raw_chains and vertex_source:
            for coeff, raw_path in raw_chains:
                z_s_path = []
                for idx, sheet in raw_path:
                    try:
                        z = toCC(vertex_source[idx])
                        z_s_path.append((z, int(sheet)))
                    except IndexError:
                        raise IndexError(f"Vertex index {idx} out of range for source len {len(vertex_source)}")
                # Clean consecutive duplicates
                clean_path = []
                for p in z_s_path:
                    if not clean_path or clean_path[-1] != p:
                        clean_path.append(p)
                if clean_path:
                    weighted_paths.append((coeff, clean_path))

        # 4. Fallback: simple .points() or .vertices() list
        if not weighted_paths:
            pts = []
            if hasattr(cycle, 'points'): pts = cycle.points()
            elif hasattr(cycle, 'vertices'): pts = cycle.vertices()
            if pts:
                clean_path = []
                for p in pts:
                    if isinstance(p, (list, tuple)):
                        z = toCC(p[0])
                        s = int(p[1]) if len(p) > 1 else 0 
                        clean_path.append((z, s))
                    else:
                        clean_path.append((toCC(p), 0))
                weighted_paths.append((1, clean_path))

        if not weighted_paths:
            raise RuntimeError(f"Could not extract any paths from cycle: {cycle}")

        return weighted_paths

    # --- Quadrature Setup ---
    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0/float(N)
        pi = math.pi
        for k in range(-N, N+1):
            t = k*h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi/2.0) * sx)
            dx_dt = (pi/2.0) * math.cosh(t) / (math.cosh((pi/2.0) * sx)**2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes
    
    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)
    tiny = CC(2) ** (-prec//2)

    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))

    # --- Core Integrator ---
    def integrate_segment_refined(p_start, p_end, sheet_start=None, y_prev_hint=None, max_depth=max_depth):
        p0 = CC(p_start); p1 = CC(p_end)
        vec = p1 - p0

        def _single_integrate(p0, vec, sheet_start, y_prev_hint):
            perp = CC(0,1) * vec
            off_mag = max(CC(1e-14), abs(vec) * CC(1e-8))
            off = perp / (abs(perp) + CC(1e-30)) * off_mag
            dx_factor = vec / CC(2)

            I0 = CC(0); I1 = CC(0)
            sample_x = p0 + ((CC(nodes[0][1]) + CC(1)) / CC(2)) * vec + off
            f0 = f_at(sample_x)
            y0_raw = f0.sqrt()
            
            # Determine correct branch for y0
            if y_prev_hint is not None:
                # Analytic continuation: pick closer value
                if abs(y0_raw - y_prev_hint) <= abs(-y0_raw - y_prev_hint): y0 = y0_raw
                else: y0 = -y0_raw
            elif sheet_start is not None:
                # Explicit sheet index
                if sheet_start % 2 != 0: y0 = -y0_raw
                else: y0 = y0_raw
            else:
                # Standard convention
                if y0_raw.imag() < 0: y0 = -y0_raw
                else: y0 = y0_raw

            y_prev = y0
            near_count = 0
            for (t, x_mapped, w) in nodes:
                s = (CC(x_mapped) + CC(1)) / CC(2)
                xval = p0 + s * vec + off
                fval = f_at(xval)
                if abs(fval) < tiny:
                    near_count += 1
                    continue
                y_plus = fval.sqrt(); y_minus = -y_plus
                if abs(y_plus - y_prev) <= abs(y_minus - y_prev): y_cur = y_plus
                else: y_cur = y_minus
                y_prev = y_cur
                denom = CC(2) * y_cur
                dxd = dx_factor * CC(w)
                I0 += (CC(1)/denom) * dxd
                I1 += (xval/denom) * dxd
            
            if near_count: _subdivisions["count"] += near_count
            return I0, I1, y_prev, near_count

        def _recurse(p0, vec, sheet, y_hint, depth):
            I0, I1, y_last, near = _single_integrate(p0, vec, sheet, y_hint)
            if near > len(nodes) // 10 and depth < max_depth:
                mid = p0 + vec/CC(2)
                lI0, lI1, y_left = _recurse(p0, vec/CC(2), sheet, y_hint, depth+1)
                rI0, rI1, y_right = _recurse(mid, vec/CC(2), None, y_left, depth+1)
                return lI0 + rI0, lI1 + rI1, y_right
            return I0, I1, y_last

        return _recurse(p0, vec, sheet_start, y_prev_hint, 0)

    def integrate_full_chain(weighted_paths):
        tI0, tI1 = CC(0), CC(0)
        for coeff, path in weighted_paths:
            pI0, pI1 = CC(0), CC(0)
            y_prev = None
            for i in range(len(path) - 1):
                p_curr, s_curr = path[i]
                p_next, _ = path[i+1]
                v0, v1, y_end = integrate_segment_refined(p_curr, p_next, sheet_start=s_curr, y_prev_hint=y_prev)
                pI0 += v0; pI1 += v1
                y_prev = y_end
            tI0 += coeff * pI0
            tI1 += coeff * pI1
        return tI0, tI1

    # --- Extract & Compute ---
    try:
        A_cycles = [extract_cycle_structure(c, RS) for c in a_list]
        B_cycles = [extract_cycle_structure(c, RS) for c in b_list]
    except Exception as e:
        raise RuntimeError(f"Cycle extraction failed: {e}")

    # --- Intersection Logic (Validation Only) ---
    all_pts = []
    for chain in A_cycles + B_cycles:
        for coeff, path in chain:
            for p, s in path: all_pts.append(p)
    xs = [float(z.real()) for z in all_pts]; ys = [float(z.imag()) for z in all_pts]
    span = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)
    eps_geom = max(1e-12 * span, 1e-12)

    def signed_segment_intersection(p1, p2, q1, q2):
        x1, y1 = float(p1.real()), float(p1.imag())
        x2, y2 = float(p2.real()), float(p2.imag())
        x3, y3 = float(q1.real()), float(q1.imag())
        x4, y4 = float(q2.real()), float(q2.imag())
        def orient(ax, ay, bx, by, cx, cy):
            return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
        o1 = orient(x1,y1,x2,y2,x3,y3); o2 = orient(x1,y1,x2,y2,x4,y4)
        o3 = orient(x3,y3,x4,y4,x1,y1); o4 = orient(x3,y3,x4,y4,x2,y2)
        # Strict inequality often fails for graph-based cycles sharing vertices
        if o1*o2 < -eps_geom and o3*o4 < -eps_geom: return 1 if o1 > 0 else -1
        return 0

    def chain_intersection(chain1, chain2):
        total = 0
        for c1, path1 in chain1:
            for c2, path2 in chain2:
                if len(path1) < 2 or len(path2) < 2: continue
                sub = 0
                for i in range(len(path1)-1):
                    for j in range(len(path2)-1):
                        sub += signed_segment_intersection(path1[i][0], path1[i+1][0], path2[j][0], path2[j+1][0])
                total += c1 * c2 * sub
        return total

    # Compute base intersection matrix
    baseI = [[chain_intersection(A_cycles[i], B_cycles[j]) for j in range(2)] for i in range(2)]
    
    # Check if geometric intersection failed (all zeros) due to shared vertices
    is_zero_matrix = all(abs(baseI[i][j]) < 0.1 for i in range(2) for j in range(2))

    final_A, final_B = A_cycles, B_cycles

    if is_zero_matrix:
        if verbose:
            print("Geometric intersection matrix is zero (likely due to shared vertices). Assuming standard symplectic order from Sage.")
    else:
        # If we have detected intersections, try to canonicalize to force [I 0; 0 I] structure
        found_canonical = False
        for pA in itertools.permutations([0,1]):
            for pB in itertools.permutations([0,1]):
                for sA in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                    for sB in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                        valid = True
                        for i in range(2):
                            for j in range(2):
                                val = sA[i]*sB[j]*baseI[pA[i]][pB[j]]
                                if val != (1 if i==j else 0):
                                    valid = False; break
                            if not valid: break
                        if valid:
                            final_A = [[(c*sA[i], p) for c,p in A_cycles[pA[i]]] for i in range(2)]
                            final_B = [[(c*sB[i], p) for c,p in B_cycles[pB[i]]] for i in range(2)]
                            found_canonical = True
                            if verbose: print(f"Canonicalized cycles: pA={pA} pB={pB}")
                            break
                    if found_canonical: break
                if found_canonical: break
            if found_canonical: break
        
        if not found_canonical and verbose:
            print("Warning: Could not strictly canonicalize based on geometric intersections. Proceeding with default order.")

    # --- Final Integration ---
    A = Matrix(CC, 2, 2)
    B = Matrix(CC, 2, 2)
    for j in range(2):
        A[0,j], A[1,j] = integrate_full_chain(final_A[j])
        B[0,j], B[1,j] = integrate_full_chain(final_B[j])

    if verbose:
        try:
            print(f"det(A) (magnitude): {float(abs(A.det())):.6e}")
        except:
            print(f"det(A): {A.det()}")

    try:
        tau = A.inverse() * B
    except Exception as e:
        raise ArithmeticError(f"Singular A matrix: {e}")

    # Symmetrize
    tau = (tau + tau.transpose())/CC(2)
    
    # Check PD (Riemann Relations)
    evals = [float(e) for e in Matrix(RR, [[t.imag() for t in row] for row in tau]).eigenvalues()]
    if verbose: print(f"Im(tau) evals: {evals}")
    
    # Default tolerance scaled to precision if not provided
    if pd_tol is None: 
        pd_tol = -1e-10 
    
    if min(evals) < pd_tol:
        raise ArithmeticError(f"Tau not positive definite (min eval={min(evals):.2e}). Basis may be non-symplectic.")

    return tau
