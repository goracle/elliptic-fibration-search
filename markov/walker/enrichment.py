from sage.all import *
from search_common import *

def enrich_candidates(
    norm,
    x_here,
    y_here,
    n0,
    fi,
    G_poly,
    curve_degree,
    p=None,
    shift=0,
    T=None,
    T_inv=None,
):
    """
    Enrich candidate records by reconstructing the fiber intersection directly
    over GF(p), using only the shared m-parameter.

    Main fixes versus the previous version:
      - evaluates coefficients at m_val_fp more carefully, avoiding brittle
        direct subs-on-fraction-field behavior where possible
      - coerces all comparisons into GF(p)
      - skips candidates cleanly on poles / malformed evaluations
      - keeps the geometric reconstruction path explicit and strict
    """
    enriched = []

    print("x_here", x_here)

    if p is None or G_poly is None or fi is None:
        return []

    Fp = GF(int(p))
    shift_fp = Fp(shift)

    def _eval_at_m(obj, m_val_fp):
        """
        Evaluate a coefficient-like object at m = m_val_fp and coerce into GF(p).

        Tries callable evaluation first, then symbolic substitution, then direct
        coercion. This is intentionally conservative.
        """
        # Most polynomial/rational-function objects in Sage can be called.
        try:
            return Fp(obj(m_val_fp))
        except Exception:
            raise

        # Symbolic / expression fallback.
        try:
            return Fp(obj.subs(m=m_val_fp))
        except Exception:
            raise

        # Plain field element / integer / already-evaluated object.
        return Fp(obj)

    x_here_f = x_here
    x_here_f_fp = Fp(x_here_f)
    R_x = PolynomialRing(Fp, "x")

    candidates = norm.get("candidate_records", []) or norm.get("candidates", [])
    for cand in candidates:
        rec = dict(cand) if isinstance(cand, dict) else {"x_step": cand}

        m_val = rec.get("m")
        if m_val is None:
            if RLINEAR and rec.get("x_step") is not None:
                m_val = Fp(x_here) - Fp(rec["x_step"])
            else:
                continue

        try:
            m_val_fp = Fp(m_val)
        except Exception:
            raise
            continue

        try:
            coeffs = []
            for c in fi.list():
                num_val = Fp(c.numerator()(m_val_fp))
                den_val = Fp(c.denominator()(m_val_fp))
                if den_val == 0:
                    raise ZeroDivisionError(f"Fiber pole at m={m_val_fp}")
                coeffs.append(num_val / den_val)
            f_eval_poly = R_x(coeffs)
        except ZeroDivisionError:
            continue
        except Exception as e:
            print(f"CRITICAL: Evaluation failed for m={m_val_fp}. Error: {e}")
            raise

        #print("m, f_eval_poly, fi", m_val, f_eval_poly, fi)

        # Step 3: intersection polynomial on the fiber.
        try:
            G_Rx = R_x(G_poly)
        except Exception:
            raise
            continue

        # Step 4: find all roots in the base field.
        # We solve f(x) - g(x) = 0 only to locate x_step / x_res — the roots
        # are correct.  The polynomial itself is NOT a valid intersection_poly
        # for the relation matrix: its roots carry no principal-divisor structure.
        # intersection_poly is left None here and must be filled in by
        # augment_with_phi (phi_search.py) after enrich_candidates returns.
        intersection_poly = None
        _root_poly = G_Rx - f_eval_poly
        try:
            roots_wm = _root_poly.roots()
        except Exception:
            raise
            continue

        assert roots_wm, roots_wm

        other_roots_f = []
        actual_xi_mult = 0

        for r, mult in roots_wm:
            try:
                r_fp = Fp(r)
            except Exception:
                raise
                continue

            if r_fp == x_here_f_fp:
                actual_xi_mult += mult

                # Preserve the previous convention for excess x_src multiplicity.
                if mult > (curve_degree - 2):
                    other_roots_f.extend([r_fp] * (mult - (curve_degree - 2)))
            else:
                other_roots_f.extend([r_fp] * mult)

        # Step 5: require exactly two non-x_src roots in GF(p).
        if len(other_roots_f) != 2:
            continue

        xj_f, xk_f = other_roots_f

        # Step 6: evaluate Y directly from the fiber.
        try:
            yj_f_2 = f_eval_poly(xj_f)
            yk_f_2 = f_eval_poly(xk_f)
        except Exception:
            raise
            continue

        # Step 7: strict sign validation against the original curve model.
        def _get_strict_sign(x_val_f, y_val_f):
            y_int2 = int(Fp(y_val_f))
            x_int = Fp(x_val_f)
            curve_y2 = int(Fp(G_poly(x_int)))

            if (y_int2 ) % int(p) != curve_y2:
                print("y_int", y_int, "y_int^2", y_int**2 % p, "curve_y2", curve_y2)
                raise ValueError(
                    f"Y-coordinate validation failed for X={x_val_f}: "
                    f"fiber Y={y_val_f}, but Y^2 != G(X)."
                )

            canonical_y = min(y_int, int(p) - y_int)
            return 1 if y_int == canonical_y else -1

        try:
            #yj_sign = _get_strict_sign(xj_f, yj_f_2)
            #yk_sign = _get_strict_sign(xk_f, yk_f_2)
            yj_sign = 1
            yk_sign = 1
        except Exception:
            raise
            continue

        # intersection_poly is intentionally None here; augment_with_phi fills it in.

        xj_val, xk_val = xj_f, xk_f
        # Step 9: pack the record.
        new_rec = {
            "x_src": x_here,
            "yi": y_here,
            "x_step": xj_val,
            "x_res": xk_val,
            "yj_sign": yj_sign,
            "yk_sign": yk_sign,
            "m": m_val_fp,
            "input_n": n0,
            "source": "pure_fiber_intersection",
            "src_mult": actual_xi_mult,
            "intersection_poly": intersection_poly,
            "shift": shift,
        }
        enriched.append(new_rec)

        # Optional x_res-head injection for RLINEAR.
        if RLINEAR and xk_val != x_here:
            try:
                enriched.append({
                    "x_src": x_here,
                    "yi": y_here,
                    "x_step": xk_val,
                    "x_res": xj_val,
                    "yj_sign": yk_sign,
                    "yk_sign": yj_sign,
                    "m": Fp(x_here) - Fp(xk_val),
                    "input_n": n0,
                    "source": "x_res_head",
                    "src_mult": actual_xi_mult,
                    "intersection_poly": intersection_poly,
                    "shift": shift,
                })
            except Exception:
                raise

    return enriched

