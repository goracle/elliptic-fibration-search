from sage.all import QQ, PolynomialRing, HyperellipticCurve, Matrix, CDF, RealField
from .mumford_height import compute_height_pairing_exact, naive_height_exact
from ..arakelov import arakelov_build_basis_with_heights, arakelov_height_pairing, clear_period_cache
from .mumford_core import _poly_from_coeffs_qq


def build_mumford_basis_incremental(all_divisors, f_coeffs, num_doublings=NUM_DOUBLINGS, debug=True):
    """
    Build independent basis using height pairing checks.
    
    Will use Arakelov heights if available, otherwise falls back to exact doubling method.
    
    Args:
        num_doublings: Number of doubling iterations (only used for fallback method)
    """
    # run the smoothness tests
    p_test = 2_000_003  # large random-ish prime

    diagnostic_x_root_distribution(all_divisors, p_test)
    diagnostic_section_collapse(all_divisors)
    diagnostic_smoothness_proxy(all_divisors, p_test)
    diagnostic_factor_base_saturation(all_divisors, p_test)
    diagnostic_mod_p_coverage(all_divisors, p_test, genus=2)

    if len(all_divisors) > MAX_BASIS_CANDIDATES:
        all_divisors = all_divisors[:MAX_BASIS_CANDIDATES]
        print(f"[basis] Truncating candidate divisors to {MAX_BASIS_CANDIDATES}")


    if ARAKELOV_AVAILABLE:
        if debug:
            print("[basis] Using Arakelov heights for basis construction")
        
        # INCREASED DEFAULT PRECISION to prevent false rank increases
        prec = 2048
        max_attempts = 2
        
        for attempt in range(max_attempts):
            try:
                # Use the parallel version which has the new checks
                result = arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=prec, debug=True)
                return result
            except Exception as e:
                if attempt < max_attempts - 1:
                    prec *= 2
                    if debug:
                        print(f"[basis] Attempt {attempt+1} failed with prec={prec//2}, retrying with prec={prec}")
                    clear_period_cache()
                    raise
                else:
                    if debug:
                        print(f"[basis] All Arakelov attempts failed, falling back to exact method")
                        # DIAGNOSTIC: Print the actual error causing the fallback
                        print(f"[basis] Arakelov Failure Reason: {type(e).__name__}: {e}")
                        # traceback.print_exc() # Uncomment for full stack trace if needed
                    raise
                    return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings, debug)
    else:
        assert None, "deprecated"
        if debug:
            print("[basis] Using exact doubling method for basis construction")
        return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings, debug)


def build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=NUM_DOUBLINGS, debug=True):
    """
    OLD METHOD: Build independent basis using EXACT height pairing checks via doubling.
    This is the fallback when Arakelov module is not available.
    """
    if not all_divisors:
        return [], 0, None
    if len(all_divisors) > MAX_BASIS_CANDIDATES:
        all_divisors = all_divisors[:MAX_BASIS_CANDIDATES]
        print(f"[basis] Truncating candidate divisors to {MAX_BASIS_CANDIDATES}")

    print(f"\n[basis] Starting with {len(all_divisors)} total divisors")
    print(f"[basis] Using {num_doublings} doublings for height pairing approximation")
    
    # Build curve once
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Filter out torsion with HIGHER order bound
    non_torsion = []
    torsion_count = 0
    
    max_torsion_order = 100
    
    for div in all_divisors:
        is_tors, order = is_mumford_torsion_fast(
            div['s'], div['p'], div['v_0'], div['v_1'], 
            f_coeffs, max_order=max_torsion_order, debug=False
        )
        
        if is_tors:
            torsion_count += 1
            if debug and torsion_count <= 5:
                print(f"[basis] Filtered torsion divisor (order {order}): s={div['s']}, p={div['p']}")
        else:
            non_torsion.append(div)
    
    print(f"[basis] Filtered {torsion_count} torsion divisors -> {len(non_torsion)} candidates")
    
    if not non_torsion:
        return [], 0, None
    
    # Convert to Jacobian elements
    jac_elements = []
    for div in non_torsion:
        u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        jac_elements.append((div, D))
    
    # Build basis using EXACT independence checks
    basis = []
    basis_jac = []
    
    typical_height = None
    
    for i, (div, D) in enumerate(jac_elements):
        if not basis:
            # First divisor - just check self-pairing is nonzero
            try:
                h_exact = compute_height_pairing_exact(D, D, f_coeffs, num_doublings=num_doublings)
            except (ValueError, RationalReconstructionError) as e:
                if debug:
                    print(f"[basis] compute_height_pairing_exact failed for candidate {i+1}: {e}")
                raise
                continue

            h_float = float(h_exact) # Keep sign!
            
            # DIAGNOSTIC: Check negativity
            if h_float < 0:
                 print(f"[basis] WARNING: Negative self-pairing for divisor {i+1}: {h_float:.6g}")
                 # Strictly reject negative heights
                 continue

            if h_float < 1e-8:
                if debug:
                    print(f"[basis] Skipping divisor {i+1}: self-pairing too small ({h_float:.3g})")
                continue
            
            typical_height = h_float
            basis.append(div)
            basis_jac.append(D)
            if debug:
                print(f"[basis] Added divisor 1 (self-pairing {h_float:.3g})")
        else:
            # Check independence by computing height pairing matrix
            # before loop: prepare cache and choose numeric precision & tolerance
            pairing_cache = {}   # optional cache to avoid recomputing the same pairings
            prec_bits_for_test = 2048   # tune: 256..2048 depending on machine
            # tolerance rule (use decimal-digits heuristic)
            def _auto_tol(scale, prec_bits):
                # convert bits->decimal digits roughly: digits ~ prec_bits*log10(2)
                dec_digits = int(prec_bits * 0.30103)
                # margin 6 digits of safety
                return float(scale) * (10.0 ** (-(dec_digits - 6)))

            # inside candidate loop, instead of determinant-based check:
            # compute residual squared against current basis_jac
            res_sq = _projection_residual_sq(basis_jac, D, f_coeffs, 
                                            prec_bits=prec_bits_for_test,
                                            pairing_func=arakelov_height_pairing, 
                                            pairing_cache=pairing_cache, debug=debug)
            # pick scale = typical_height or diag max
            if typical_height is None or typical_height <= 0:
                # fallback scale: max diagonal of G if available, else 1.0
                scale = 1.0
            else:
                scale = typical_height
            tol = _auto_tol(scale, prec_bits_for_test)

            if res_sq > tol:
                basis.append(div)
                basis_jac.append(D)
                if debug:
                    print(f"[basis] Added divisor {len(basis)} (res_sq={res_sq:.3g} tol={tol:.3g})")
            else:
                if debug:
                    print(f"[basis] Rejected divisor {i+1}: residual {res_sq:.3g} <= tol {tol:.3g}")

    rank = len(basis)
    
    # Build final height matrix with EXACT rationals
    if rank > 0:
        H_exact = Matrix(QQ, rank, rank)
        for i in range(rank):
            for j in range(i, rank):
                h_ij_exact = compute_height_pairing_exact(
                    basis_jac[i], 
                    basis_jac[j], 
                    f_coeffs,
                    num_doublings=num_doublings
                )
                H_exact[i, j] = h_ij_exact
                H_exact[j, i] = h_ij_exact
        
        if debug:
            print(f"\n[basis] Final rank: {rank}")
            print(f"[basis] Checked {len(jac_elements)} candidates total")
            det_exact = H_exact.determinant()
            print(f"[basis] Determinant (exact): {det_exact}")
            print(f"[basis] Determinant (float): {float(det_exact):.6g}")
            
            # DIAGNOSTIC: Final Matrix Dump
            #print("[basis] Final Height Matrix:")
            #print(H_exact.str())
            try:
                evals = H_exact.change_ring(CDF).eigenvalues()
                print(f"[basis] Final Eigenvalues: {evals}")
            except:
                raise

    else:
        H_exact = None
    
    return basis, rank, H_exact

def check_mumford_independence(divisors, f_coeffs, debug=DEBUG):
    """
    Build Jacobian elements and compute pairing matrix.
    Uses Arakelov if available, otherwise falls back to manual method.
    
    Returns (is_indep, rank, H_matrix)
    """
    if not divisors:
        return True, 0, None

    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)

    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(div['s'], div['p'], div['v_0'], div['v_1'], C)
            if not elem.is_zero():
                jac_elements.append(elem)
            else:
                if debug:
                    print("[check] element is zero, skipping.")
        except Exception:
            if debug:
                print("[check] failed to convert divisor to jac element:", div)
            raise

    if not jac_elements:
        return True, 0, None

    n = len(jac_elements)
    
    if ARAKELOV_AVAILABLE:
        if debug:
            print("[check] Using Arakelov heights")
        is_indep, rank, H = arakelov_check_independence(jac_elements, f_coeffs, prec=300, debug=debug)
        return is_indep, rank, H
    else:
        if debug:
            print("[check] Using manual height computation")
        H = Matrix(RDF, n, n)
        for i in range(n):
            for j in range(i, n):
                try:
                    val = compute_manual_height_pairing(jac_elements[i], jac_elements[j], debug=debug)
                except Exception:
                    if debug:
                        print(f"[check] height pairing failed for indices {i},{j}")
                    raise
                H[i, j] = val
                H[j, i] = val

        if n == 1:
            is_indep = abs(H[0, 0]) > 1e-8
            rank = 1 if is_indep else 0
        else:
            rank = H.rank()
            is_indep = (rank == n)
        return is_indep, rank, H


def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Create a Jacobian element while coercing the u,v polynomials into the curve's polynomial ring.
    Raises on failure (user preference).
    """
    try:
        f_curve, h_curve = C.hyperelliptic_polynomials()
        R = f_curve.parent()   # polynomial ring of the curve
        x = R.gen()

        # Coerce inputs to rational numbers in Python Fraction and then to QQ for the ring
        def to_QQ_obj(a):
            try:
                return QQ(a)
            except Exception:
                raise
                return QQ(Fraction(str(a)))

        s_q = to_QQ_obj(s)
        p_q = to_QQ_obj(p)
        v0_q = to_QQ_obj(v0)
        v1_q = to_QQ_obj(v1)

        u_poly = x**2 - s_q * x + p_q
        v_poly = v1_q * x + v0_q

        # Make sure polynomials live in the same parent as the curve
        u_poly = R(u_poly)
        v_poly = R(v_poly) if v_poly.parent() == R else R(v_poly)  # coerce v into same ring

        return C.jacobian()([u_poly, v_poly])
    except Exception:
        # re-raise so user sees the problem
        raise


def _projection_residual_sq(basis_jac, candidate_D, f_coeffs, 
                            prec_bits=512, pairing_func=None, pairing_cache=None, debug=False):
    """
    Return residual^2 = <v, v> - c^T * G^{-1} * c where:
      - G is Gram matrix of basis_jac (size m x m) under canonical height pairing
      - c is vector of pairings <basis_jac[i], candidate_D>
      - pairing_func(D1, D2, f_coeffs, prec=...) should return QQ/float pairing
    Uses high-precision RealField arithmetic for numeric stability.
    pairing_cache: optional dict to avoid recomputation, keyed by (id(D1), id(D2)) or indices
    """
    if pairing_func is None:
        # default to arakelov path if available
        from .arakelov import arakelov_height_pairing as pairing_func

    m = len(basis_jac)
    if m == 0:
        # residual is just candidate self-pairing
        val = pairing_func(candidate_D, candidate_D, f_coeffs, prec=prec_bits)
        return float(val)

    RR = RealField(prec_bits)
    # build G (m x m) and c (m)
    Gnum = matrix(RR, m, m)
    cnum = vector(RR, m)
    for i in range(m):
        for j in range(i, m):
            key = (id(basis_jac[i]), id(basis_jac[j]))
            if pairing_cache and key in pairing_cache:
                v = pairing_cache[key]
            else:
                v = pairing_func(basis_jac[i], basis_jac[j], f_coeffs, prec=prec_bits)
                if pairing_cache is not None:
                    pairing_cache[key] = v
            Gnum[i, j] = RR(v)
            Gnum[j, i] = Gnum[i, j]
        # c_i
        keyc = (id(basis_jac[i]), id(candidate_D))
        if pairing_cache and keyc in pairing_cache:
            ci = pairing_cache[keyc]
        else:
            ci = pairing_func(basis_jac[i], candidate_D, f_coeffs, prec=prec_bits)
            if pairing_cache is not None:
                pairing_cache[keyc] = ci
        cnum[i] = RR(ci)
    vv = RR(pairing_func(candidate_D, candidate_D, f_coeffs, prec=prec_bits))

    # Try Cholesky (fast & stable for PD G); fallback to SVD pseudo-inverse
    try:
        L = Gnum.cholesky()
        # solve L * y = c  (L lower-triangular)
        y = L.solve_left(cnum)
        proj_sq = float((y.dot_product(y)))
    except Exception:
        # fallback: pseudoinverse via SVD
        try:
            U, S, Vt = Gnum.SVD()
            # build pseudo-inverse
            # invert non-zero singular values safely
            S_inv = [ (1.0 / float(si)) if float(si) > 0 else 0.0 for si in S ]
            # create diagonal matrix and compute G_inv = Vt^T * diag(S_inv) * U^T
            from sage.all import diagonal_matrix
            S_inv_mat = diagonal_matrix(RR, [RR(s) for s in S_inv])
            Ginv = Vt.transpose() * S_inv_mat * U.transpose()
            # proj_sq = c^T * Ginv * c
            proj_sq = float((cnum * (Ginv * cnum)))
        except Exception:
            # last resort: pessimistically return zero projection -- force acceptance check to rely on self-pairing alone
            proj_sq = 0.0

    res_sq = float(vv) - proj_sq
    # numerical floor at zero
    if res_sq < 0 and abs(res_sq) < 10**(- (prec_bits // 3)):
        res_sq = 0.0
    if debug:
        print(f"[proj] vv={float(vv):.4g} proj={proj_sq:.4g} res_sq={res_sq:.4g}")
    return float(res_sq)

def dbg_poly_info(poly):
    # poly is a sage polynomial
    coeffs = poly.list()          # lowest-first
    if not coeffs:
        return "deg=-inf"
    deg = poly.degree()
    # get bit sizes
    def bits_of(c):
        try:
            return int(c.nbits()) if hasattr(c, 'nbits') else int(Fraction(c).numerator).bit_length()
        except Exception:
            try:
                return int(abs(int(c)).bit_length())
            except Exception:
                raise
                return -1
            raise
    bits = [bits_of(c) for c in coeffs]
    maxbits = max(bits) if bits else 0
    return f"deg={deg}, maxcoeff_bits={maxbits}, len={len(coeffs)}"

def dump_jacobian_mumford_info(JP, label="P"):
    # JP is jacobian element [u,v]
    try:
        u = JP[0]   # polynomial
        v = JP[1]
        print(f"[DBG] {label} u: {dbg_poly_info(u)}; v: {dbg_poly_info(v)}; parents: {type(u.parent())}")
    except Exception as e:
        print("[DBG] failed to print mumford info:", e)
        raise

