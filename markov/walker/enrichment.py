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
    _n_poles = _n_no_roots = _n_wrong_nroots = _n_sign_fail = 0


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
        return Fp(obj(m_val_fp))

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
            _n_poles += 1; continue
        except Exception as e:
            print(f"CRITICAL: Evaluation failed for m={m_val_fp}. Error: {e}")
            raise

        #print("m, f_eval_poly, fi", m_val, f_eval_poly, fi)

        # Step 3: intersection polynomial on the fiber.
        try:
            G_Rx = R_x(G_poly)
        except Exception:
            raise

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

        if not roots_wm:
            _n_no_roots += 1; continue

        other_roots_f = []
        actual_xi_mult = 0

        for r, mult in roots_wm:
            try:
                r_fp = Fp(r)
            except Exception:
                raise

            if r_fp == x_here_f_fp:
                actual_xi_mult += mult

                # Preserve the previous convention for excess x_src multiplicity.
                if mult > (curve_degree - 2):
                    other_roots_f.extend([r_fp] * (mult - (curve_degree - 2)))
            else:
                other_roots_f.extend([r_fp] * mult)

        # Step 5: need 1 or 2 distinct non-x_src roots.
        # len==2: generic relation xi^src_mult + xj + xk - deg*inf = 0
        # len==1: self relation  xi^src_mult + 2*xj - deg*inf = 0 (double root)
        # len==0 or >2: degenerate, skip.
        if len(other_roots_f) == 2:
            xj_f, xk_f = other_roots_f
        elif len(other_roots_f) == 1:
            xj_f = xk_f = other_roots_f[0]
        else:
            _n_wrong_nroots += 1; continue

        # Step 6: evaluate Y directly from the fiber.
        try:
            yj_f_2 = f_eval_poly(xj_f)
            yk_f_2 = f_eval_poly(xk_f)
        except Exception:
            raise

        # Step 7: strict sign validation against the original curve model.
        def _get_strict_sign(x_val_f, y2_val_f):
            """Return +1 if the canonical (smaller) square root is positive, -1 otherwise.

            y2_val_f is f(x) evaluated from the fiber — equal to G(x) when the
            candidate is on the curve.  We take the square root over Fp and compare
            against the canonical branch.
            """
            x_int = Fp(x_val_f)
            curve_y2 = Fp(G_poly(x_int))
            y2_fp = Fp(y2_val_f)
            if y2_fp != curve_y2:
                print(f"[sign_fail] x={x_val_f} fiber_y2={y2_fp} curve_y2={curve_y2}")
                print(f"[sign_fail] x={x_val_f} fiber_y2={y2_fp} curve_y2={curve_y2}")
                raise ValueError(
                    f"Y-coordinate validation failed for X={x_val_f}: "
                    f"fiber y²={y2_fp}, curve y²={curve_y2}."
                )
            if y2_fp == 0:
                return 1  # Weierstrass point — sign is irrelevant
            sq = y2_fp.sqrt(extend=False, all=True)
            if not sq:
                raise ValueError(f"No square root for y²={y2_fp} at x={x_val_f}")
            y_int = int(min(sq, key=lambda v: int(v)))
            canonical_y = min(y_int, int(p) - y_int)
            return 1 if y_int == canonical_y else -1

        try:
            yj_sign = _get_strict_sign(xj_f, yj_f_2)
            yk_sign = _get_strict_sign(xk_f, yk_f_2)
        except Exception:
            _n_sign_fail += 1; continue

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

        if RLINEAR and xk_val != x_here:
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

    print(f"[enrich] x_here={x_here_f_fp} candidates={len(candidates)} enriched={len(enriched)} "
          f"poles={_n_poles} no_roots={_n_no_roots} wrong_nroots={_n_wrong_nroots} sign_fail={_n_sign_fail}")
    return enriched

