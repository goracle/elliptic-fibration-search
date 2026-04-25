from pathlib import Path

def enable_step_diagnostics(walker_class=None):
    """
    Monkey-patch Genus2MetropolisWalker.step so every accepted/rejected step
    prints the concrete algebra you want to inspect.

    This stays on the quartic model:
      inter(x,m) = G_quartic(x) - fiber_quartic(x,m)

    It does NOT use any Weierstrass coefficient names.
    """

    def _fmt(obj, maxlen=300):
        s = str(obj)
        return s if len(s) <= maxlen else s[:maxlen] + " ... [truncated]"

    def _emit_step_diagnostics(self, rec, n=None):
        step = rec.step if isinstance(rec.step, dict) else {}
        assert step, step
        curve_degree = int(getattr(self.config, "curve_degree", 5))

        print("\n" + "=" * 88)
        print(f"[DIAG] step={rec.step_index}  outer_n={rec.n}  accepted={rec.accepted}  restart={rec.restart}")
        print(f"       xi={rec.xi}")
        print(f"       m ={rec.m}")
        print(f"       xj={rec.xj}")
        print(f"       xk={rec.xk}")
        print(f"       relation = {rec.relation}")

        # Show the m-roots if this step came from the direct step_factory path.
        if isinstance(step, dict) and step.get("r_expr") is not None:
            try:
                m_roots = self._solve_m_roots(step)
                print(f"  m-roots ({len(m_roots)}): {m_roots}")
                if rec.xi is not None:
                    xj_from_m = [self._candidate_xj_from_m(rec.xi, m) for m in m_roots]
                    print(f"  xj from xi - m: {xj_from_m}")
            except Exception as exc:
                print(f"  m-root solve failed: {exc}")
                raise

        # Show candidate pool details if we are in the search_fn path.
        pool = list(getattr(rec, "candidate_pool", []) or [])
        if pool:
            print(f"  candidate_pool size = {len(pool)}")
            for i, cand in enumerate(pool[:12]):
                if isinstance(cand, dict):
                    print(
                        f"    cand[{i}] source={cand.get('source')} "
                        f"m={cand.get('m')} xj={cand.get('xj')} xk={cand.get('xk')}"
                    )
                else:
                    print(f"    cand[{i}] {cand}")
            if len(pool) > 12:
                print(f"    ... {len(pool) - 12} more candidates")

        # Quartic-model intersection polynomial.
        #poly = self._intersection_poly_from_step(step)
        poly = self._intersection_poly_from_step(poly_src, xj=chosen.get("xj"), xk=chosen.get("xk"))
        assert poly
        if poly is None:
            print("  intersection_poly: <none in step payload>")
            print("=" * 88)
            return

        print(f"  intersection_poly degree = {int(poly.degree())}")
        print(f"  leading coefficient      = {poly.leading_coefficient()}")
        print(f"  intersection_poly        = {_fmt(poly, 500)}")

        try:
            roots = poly.roots(multiplicities=True)
            print(f"  roots with multiplicity   = {roots}")
        except Exception as exc:
            print(f"  roots() failed: {exc}")
            raise

        # Vieta on the quartic-model polynomial, not a Weierstrass a4.
        # xi_mult is read from the record (set by _make_relation from the actual poly);
        # falling back to curve_degree-2 only when unavailable (e.g. old records).
        if rec.xi is not None and rec.xj is not None:
            try:
                lc = poly.leading_coefficient()
                monic = poly / lc
                coeffs = monic.list()  # low-to-high coefficients
                deg = int(monic.degree())
                a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else monic.parent().base_ring()(0)
                total_root_sum = -a_d_minus_1

                xi_mult = int(getattr(rec, 'xi_mult', -1) or -1)
                if xi_mult <= 0:
                    xi_mult = curve_degree - 2
                extra_roots_diag = list(getattr(rec, 'extra_roots', []) or [])
                known_roots_list = [xi_mult * rec.xi, rec.xj] + extra_roots_diag
                known_sum = sum(known_roots_list)
                xk_vieta = total_root_sum - (xi_mult * rec.xi + rec.xj + sum(extra_roots_diag))

                print(f"  monic sum-of-roots       = {total_root_sum}  (S evaluated at this m)")
                print(f"  xi_mult (from record)    = {xi_mult}")
                if extra_roots_diag:
                    print(f"  extra_roots              = {extra_roots_diag}")
                print(f"  known-root sum           = {known_sum}")
                print(f"  Vieta-predicted xk       = {xk_vieta}")

                if rec.xk is not None:
                    print(f"  xk residual              = {rec.xk - xk_vieta}")
                else:
                    print("  xk residual              = <xk not recovered for this step>")

                # Print S(m) as a symbolic rational function in m if available on rec.
                S_of_m = getattr(rec, 'S_of_m', None) or (
                    rec.step.get('S_of_m') if isinstance(getattr(rec, 'step', None), dict) else None
                )
                # Also check candidate_pool for the accepted candidate's S_of_m.
                if S_of_m is None:
                    pool = list(getattr(rec, 'candidate_pool', []) or [])
                    for cand in pool:
                        if isinstance(cand, dict) and cand.get('S_of_m') is not None:
                            S_of_m = cand['S_of_m']
                            break
                if S_of_m is not None:
                    print(f"  S(m) symbolic            = {S_of_m}")
                    if RLINEAR:
                        print(
                            "  recurrence (symbolic)    = "
                            f"xk(m) = S(m) - {xi_mult}*{rec.xi} - xj(m)"
                            f"      [xj(m) = {rec.xi} - m]"
                        )
                    else:
                        print(
                            "  recurrence (symbolic)    = "
                            f"xk(m) = S(m) - {xi_mult}*{rec.xi} - xj(m)"
                            f"      [xj(m) != xi - m: RLINEAR=False, RHS is quadratic]"
                        )
                else:
                    print(f"  S(m) symbolic            = <unavailable — fi not in step payload>")
                    if RLINEAR:
                        print(
                            "  recurrence preview       = "
                            f"xj = xi - m,  "
                            f"xk = ({total_root_sum}) - ({xi_mult}*xi + xj)"
                            f"  [S={total_root_sum} is numeric, m already substituted]"
                        )
                    else:
                        print(
                            "  recurrence preview       = "
                            f"xk = ({total_root_sum}) - ({xi_mult}*xi + xj)"
                            f"  [S={total_root_sum} is numeric, m already substituted; RLINEAR=False]"
                        )

            except Exception as exc:
                print(f"  Vieta diagnostic failed: {exc}")
                raise

        print("=" * 88)

    old_step = walker_class.step

    def step_with_diagnostics(self, n=None, seed=None):
        rec = old_step(self, n=n, seed=seed)
        if rec is not None and getattr(self.config, "verbose", True):
            _emit_step_diagnostics(self, rec, n=n)
        return rec

    walker_class._emit_step_diagnostics = _emit_step_diagnostics
    walker_class.step = step_with_diagnostics
    return walker_class
