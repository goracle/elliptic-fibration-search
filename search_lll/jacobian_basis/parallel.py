"""Parallel worker functions for multiprocessing."""

from sage.all import PolynomialRing, HyperellipticCurve, QQ

from .heights import arakelov_canonical_height


"""Parallel worker functions for multiprocessing."""

from sage.all import QQ


from sage.all import PolynomialRing, HyperellipticCurve, QQ, RealField


def compute_pairing_worker(args):
    """
    Robust worker for a single Néron–Tate pairing.
    Uses parallelogram law with guard precision.
    """
    i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args

    if i == j:
        return ((i, j), h_i, None)

    # Guard bits against cancellation
    guard_bits = 2048
    work_prec = prec + guard_bits
    RR = RealField(work_prec)

    # Rebuild curve (exact)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs) - 1 - k)
                 for k, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    def rebuild_div(div):
        u = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v = QQ(div['v_1'])*x + QQ(div['v_0'])
        return J([u, v])

    P = rebuild_div(div_i)
    Q = rebuild_div(div_j)

    h_sum = None
    h_diff = None

    # P + Q
    try:
        D = P + Q
        if D.is_zero():
            h_sum = RR(0)
        else:
            h_sum = RR(arakelov_canonical_height(D, f_coeffs, prec=work_prec))
    except Exception:
        raise

    # P − Q
    try:
        D = P - Q
        if D.is_zero():
            h_diff = RR(0)
        else:
            h_diff = RR(arakelov_canonical_height(D, f_coeffs, prec=work_prec))
    except Exception:
        raise

    if h_sum is not None and h_diff is not None:
        val = (h_sum - h_diff) / RR(4)

    elif h_sum is not None:
        val = (h_sum - RR(h_i) - RR(h_j)) / RR(2)

    elif h_diff is not None:
        val = (RR(h_i) + RR(h_j) - h_diff) / RR(2)

    else:
        raise RuntimeError(
            f"Both P+Q and P−Q height computations failed for ({i},{j})"
        )

    # Round back to requested precision
    return ((i, j), RealField(prec)(val), None)


def compute_height_worker(args):
    i, div, f_coeffs, prec = args

    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs) - 1 - k)
                 for k, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    u = x**2 - QQ(div['s'])*x + QQ(div['p'])
    v = QQ(div['v_1'])*x + QQ(div['v_0'])
    D = J([u, v])

    h = arakelov_canonical_height(D, f_coeffs, prec=prec)
    return (i, h, None)
