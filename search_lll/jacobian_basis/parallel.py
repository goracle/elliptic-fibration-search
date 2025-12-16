"""Parallel worker functions for multiprocessing."""

from sage.all import PolynomialRing, HyperellipticCurve, QQ

from .heights import arakelov_canonical_height


def compute_pairing_worker(args):
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


