
def compute_S_of_m(fi, G_poly, curve_degree):
    """Return the x^(d-1) coefficient of (G(x) - f_i(x, m)) as a symbolic
    rational function in m, without evaluating m numerically.

    fi lives in R_xm = PolynomialRing(Frac(GF(p)['m']), 'x'), so its
    coefficients are already rational functions in m.  G_poly lives in
    GF(p)[x] with constant coefficients.  We lift G into the same ring and
    subtract to get the bivariate intersection polynomial, then read off the
    x^(d-1) coefficient.

    Returns (S_of_m, inter_sym) where:
      S_of_m   -- negated x^(d-1) coeff of monic inter, a rational function in m
      inter_sym -- full symbolic intersection poly in R_xm[x]

    Returns (None, None) if fi or G_poly are unavailable.
    """
    if fi is None or G_poly is None:
        return None, None
    try:
        R_xm = fi.parent()                          # PolynomialRing(Frac(Fp[m]), 'x')
        base = R_xm.base_ring()                     # Frac(Fp[m])
        # Lift G_poly coefficients into Frac(Fp[m]) so subtraction is valid.
        G_lifted = R_xm([base(c) for c in G_poly.list()])
        inter_sym = G_lifted - fi
        lc = inter_sym.leading_coefficient()
        monic_sym = inter_sym / lc
        deg = int(monic_sym.degree())
        coeffs = monic_sym.list()                   # low-to-high
        # x^(d-1) coeff is at index deg-1; sum-of-roots = -that coeff
        a_dm1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else base(0)
        S_of_m = -a_dm1
        return S_of_m, inter_sym
    except Exception:
        raise

def compute_xk_from_fiber(xi_val, m_val, xj_val, fi, G_poly, curve_degree):
    if fi is None or G_poly is None or m_val is None:
        assert None, None
        return None, None

    try:
        Rx = G_poly.parent()
        Fp = Rx.base_ring()
        m_fp = Fp(m_val)

        def eval_at_m(obj):
            try:
                return obj(m_fp) if callable(obj) else obj
            except TypeError:
                raise
                return obj

        coeffs = []
        for c in fi.list():
            num = c.numerator()
            den = c.denominator()

            nv = eval_at_m(num)
            dv = eval_at_m(den)

            dv_fp = Fp(dv)
            if dv_fp == Fp(0):
                raise ZeroDivisionError(
                    f"compute_xk_from_fiber: fiber pole at m={m_val} "
                    f"(denominator of fi coefficient vanished).\n"
                    f"  x_src={xi_val}  x_step={xj_val}\n"
                    f"  fi={fi}\n"
                    f"  G_poly={G_poly}\n"
                    f"  coeffs so far={coeffs}"
                )

            coeffs.append(Fp(nv) / dv_fp)

        fi_at_m = Rx(coeffs)
        inter = G_poly - fi_at_m

        if inter.degree() != curve_degree:
            assert False, f"inter.degree()={inter.degree()} != curve_degree={curve_degree}, x_src={xi_val} m={m_val} x_step={xj_val}"
            return None, None

        # Determine x_src's multiplicity in the intersection poly.
        roots_wm = inter.roots()  # Sage: [(root, mult), ...]
        actual_src_mult = 0
        for r, m in roots_wm:
            if Fp(r) == Fp(xi_val):
                actual_src_mult = int(m)
                break
        assert actual_src_mult > 0, (
            f"compute_xk_from_fiber: x_src={xi_val} is not a root of the fiber intersection poly "
            f"at m={m_val}, x_step={xj_val}. roots={roots_wm}. "
            f"This means the fiber was constructed for a different x_src or the multiplicity "
            f"assumptions are wrong."
        )

        known = [xi_val] * actual_src_mult + [xj_val]
        return missing_root_by_vieta(inter, known), inter

    except Exception:
        raise

def missing_root_by_vieta(poly, known_roots: Sequence[Any]) -> Any:
    """Given a degree-5 polynomial and 4 known roots (with multiplicity), recover the fifth.

    Works for monic or non-monic polynomials over a field.
    """
    if poly is None:
        raise ValueError("missing_root_by_vieta requires a polynomial")
    R = poly.parent()
    x = R.gen()
    deg = int(poly.degree())
    if deg < 1:
        raise ValueError("polynomial degree too small for Vieta recovery")

    lc = poly.leading_coefficient()
    if lc == 0:
        raise ValueError("leading coefficient is zero")
    monic = poly / lc
    coeffs = monic.list()  # low-to-high
    # For monic x^d + a_{d-1}x^{d-1} + ... the sum of roots is -a_{d-1}
    a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else R.base_ring()(0)
    total_sum = -a_d_minus_1
    return total_sum - sum(known_roots)

def poly_roots_with_multiplicity(poly) -> List[Tuple[Any, int]]:
    """Return roots as (root, multiplicity) pairs over the polynomial's base field."""
    roots = poly.roots(multiplicities=True)
    assert roots, roots
    return [(r, int(m)) for r, m in roots]

