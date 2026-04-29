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
    Enrich candidate records by evaluating the fiber at each m-value and
    collecting all non-src roots as candidate x_step values.

    Root multiplicities and divisor structure are intentionally ignored here —
    phi (augment_with_phi) is the only source of valid relations.  This function
    just feeds x-coordinates to phi; any candidate that phi can't build a
    relation for will be dropped downstream.

    The only early rejections are:
      - fiber pole at m (ZeroDivisionError in coefficient evaluation)
      - fiber has no F_p roots at all
      - every root equals x_src (no candidate step exists)
    """
    enriched = []
    _n_poles = _n_no_roots = _n_wrong_nroots = _n_sign_fail = 0

    if p is None or G_poly is None or fi is None:
        return []

    Fp = GF(int(p))
    R_x = PolynomialRing(Fp, "x")

    x_here_f_fp = Fp(x_here)

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

        # Evaluate fiber polynomial at this m.
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

        # Find roots of (fiber - G) over F_p.
        try:
            G_Rx = R_x(G_poly)
            roots_wm = (G_Rx - f_eval_poly).roots()
            raw = [r for r,_ in roots_wm]
            for xraw in raw:
                assert get_y_unshifted_genus2(xraw), (xraw, roots_wm, get_y_unshifted_genus2(xraw), m_val, m_val_fp)
        except Exception:
            raise

        if not roots_wm:
            _n_no_roots += 1; continue

        # Diagnostic: print root structure for first 3 candidates.
        _cand_idx = _n_poles + _n_no_roots + _n_wrong_nroots + _n_sign_fail + len(enriched)
        if _cand_idx < 300:
            print(f"  [root_dbg] cand#{_cand_idx} m={m_val_fp} "
                  f"roots_wm={[(int(r), int(mult)) for r,mult in roots_wm]}  "
                  f"x_src={int(x_here_f_fp)}")

        # Collect every distinct non-src root as a candidate x_step.
        # Multiplicity is irrelevant — phi is the only thing that determines
        # whether a valid relation exists for a given (x_src, x_step) pair.
        any_emitted = False
        for r, mult in roots_wm:
            r_fp = Fp(r)
            if r_fp == x_here_f_fp:
                continue

            enriched.append({
                "x_src":             x_here,
                "yi":                y_here,
                "x_step":            r_fp,
                "x_res":             None,
                "m":                 m_val_fp,
                "input_n":           n0,
                "source":            "pure_fiber_intersection",
                "intersection_poly": None,
                "shift":             shift,
            })
            any_emitted = True

        if not any_emitted:
            _n_wrong_nroots += 1  # all roots were x_src

    print(f"[enrich] x_here={x_here_f_fp} candidates={len(candidates)} "
          f"enriched={len(enriched)} poles={_n_poles} no_roots={_n_no_roots} "
          f"all_src={_n_wrong_nroots} sign_fail={_n_sign_fail}")
    return enriched
