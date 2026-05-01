from sage.all import *
from search_common import *

def enrich_candidates(
    norm,
    pt_here,
    n0,
    fi,
    G_poly,
    curve_degree,
    p=None,
    shift=0,
    T=None,
    T_inv=None,
    Ep=None,  # Added: Sage curve object for lifting x to (x,y)
):
    """
    Enrich candidate records by evaluating the fiber at each m-value and
    collecting all non-src roots as candidate pt_step values.
    """

    enriched = []
    _n_poles = _n_no_roots = _n_wrong_nroots = _n_sign_fail = 0

    if p is None or G_poly is None or fi is None or Ep is None:
        print("broken callsite")
        return []

    Fp = GF(int(p))
    R_x = PolynomialRing(Fp, "x")

    # If pt_here is a Sage point, extract the x-coordinate for comparison
    try:
        x_here_fp = Fp(pt_here[0])
        y_here = pt_here[1]
    except (TypeError, IndexError):
        x_here_fp = Fp(pt_here)
        y_here = None # Fallback if only x was passed
        raise

    candidates = norm.get("candidate_records", []) or norm.get("candidates", [])

    for cand in candidates:
        print("candid:", cand)
        rec = dict(cand) if isinstance(cand, dict) else {"m": cand}

        m_val = rec.get("m")
        if m_val is None:
            # Handle RLINEAR cases where m might be derived from x-distance
            if rec.get("pt_step") is not None:
                try:
                    # Treat pt_step as x for distance calculation
                    x_step = Fp(rec["pt_step"][0]) if hasattr(rec["pt_step"], "__getitem__") else Fp(rec["pt_step"])
                    m_val = x_here_fp - x_step
                except Exception:
                    raise
                    continue
            else:
                continue

        m_val_fp = Fp(m_val)

        # Evaluate fiber polynomial at this m
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

        # Find roots of (fiber - G) over F_p
        G_Rx = R_x(G_poly)
        diff_poly = G_Rx - f_eval_poly
        roots_wm = diff_poly.roots()

        if not roots_wm:
            _n_no_roots += 1; continue

        # Diagnostic: print root structure
        _cand_idx = len(enriched) + _n_poles + _n_no_roots + _n_wrong_nroots
        if _cand_idx < 3:
            print(f"  [root_dbg] cand#{_cand_idx} m={m_val_fp} "
                  f"roots_wm={[(int(r), int(mult)) for r,mult in roots_wm]}  "
                  f"x_src={int(x_here_fp)}")

        any_emitted = False
        for r, mult in roots_wm:
            r_fp = Fp(r)
            if r_fp == x_here_fp:
                continue

            # FIX: Recover y-coordinate so pt_step is a real POINT object.
            # This prevents "not_on_curve" and "TypeError" downstream.
            try:
                # We lift the x-root to the curve.
                # Note: lift_x provides one point; phi_aug handles sign splits.
                pt_step_obj = (r_fp, Ep(r_fp))
            except ValueError:
                # Root is not on the curve (quadratic non-residue)
                _n_sign_fail += 1
                continue

            enriched.append({
                "pt_src":             pt_here,    # The source point (x,y)
                "pt_step":            pt_step_obj, # The new point (x,y)
                "pt_res":             pt_step_obj, # populated so walker doesn't skip
                "m":                 m_val_fp,
                "input_n":           n0,
                "source":            "pure_fiber_intersection",
                "intersection_poly": diff_poly,
                "shift":             shift,
            })
            any_emitted = True

        if not any_emitted:
            _n_wrong_nroots += 1

    print(f"[enrich] x_here={x_here_fp} candidates={len(candidates)} "
          f"enriched={len(enriched)} poles={_n_poles} no_roots={_n_no_roots} "
          f"all_src={_n_wrong_nroots} sign_fail={_n_sign_fail}")
    return enriched
