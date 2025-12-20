"""Parallel worker functions for multiprocessing."""

from sage.all import PolynomialRing, HyperellipticCurve, QQ

from .heights import arakelov_canonical_height


"""Parallel worker functions for multiprocessing."""

from sage.all import QQ


def compute_pairing_worker(args):
    """
    Worker function to compute a single Néron-Tate pairing.
    Uses the parallelogram law for robustness:
      <P, Q> = (h(P+Q) - h(P-Q)) / 4
    Falls back to one-sided formulas if one addition fails.
    """
    from sage.all import PolynomialRing, HyperellipticCurve, QQ
    
    i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args
    
    try:
        if i == j:
            return ((i, j), h_i, None)
        
        # Reconstruct Jacobian and divisors
        Rq_QQ = PolynomialRing(QQ, 'x')
        x_QQ = Rq_QQ.gen()
        f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-k) 
                       for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
        
        # Note: We assume standard Mumford form (deg u = 2) for reconstruction.
        # If u is lower degree, this might need care, but usually 's' and 'p' handle it.
        u_poly_i = x_QQ**2 - QQ(div_i['s'])*x_QQ + QQ(div_i['p'])
        v_poly_i = QQ(div_i['v_1'])*x_QQ + QQ(div_i['v_0'])
        div1 = J([u_poly_i, v_poly_i])
        
        u_poly_j = x_QQ**2 - QQ(div_j['s'])*x_QQ + QQ(div_j['p'])
        v_poly_j = QQ(div_j['v_1'])*x_QQ + QQ(div_j['v_0'])
        div2 = J([u_poly_j, v_poly_j])
        
        # Strategy: Try both P+Q and P-Q
        # h(P+Q)
        h_sum = None
        try:
            div_sum = div1 + div2
            if div_sum.is_zero():
                h_sum = QQ(0)
            else:
                h_sum = arakelov_canonical_height(div_sum, f_coeffs, prec=prec)
        except Exception:
            h_sum = None
            raise

        # h(P-Q)
        h_diff = None
        try:
            div_diff = div1 - div2
            if div_diff.is_zero():
                h_diff = QQ(0)
            else:
                h_diff = arakelov_canonical_height(div_diff, f_coeffs, prec=prec)
        except Exception:
            h_diff = None
            raise

        # Logic to combine results
        val = None
        
        # Case 1: Both succeeded (Best accuracy)
        if h_sum is not None and h_diff is not None:
            val = (h_sum - h_diff) / QQ(4)
            
        # Case 2: Only Sum succeeded
        elif h_sum is not None:
            val = (h_sum - h_i - h_j) / QQ(2)
            
        # Case 3: Only Diff succeeded
        elif h_diff is not None:
            # <P,Q> = (h(P) + h(Q) - h(P-Q)) / 2
            val = (h_i + h_j - h_diff) / QQ(2)
            
        else:
            raise RuntimeError(f"Both P+Q and P-Q height computations failed for pair ({i},{j})")

        return ((i, j), val, None)

    except Exception as e:
        # Return error as string so parent process can handle it (or crash efficiently)
        raise
        return ((i, j), None, str(e))


def compute_height_worker(args):
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
        return (i, h, None)  # Return h as Sage rational
    except Exception as e:
        msg = (
            f"Height computation failed\n"
            f"Divisor: {repr(div)}\n"
            f"Exception type: {type(e).__name__}\n"
            f"Message: {str(e)}"
        )
        # We propagate the crash as requested, but also return the tuple structure
        # in case the pool catches it differently.
        raise
        return (i, None, msg)
