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
        print(f"       x_src={rec.x_src}")
        print(f"       m ={rec.m}")
        print(f"       x_step={rec.x_step}")
        print(f"       x_res={rec.x_res}")
        print(f"       relation = {rec.relation}")

        # Show the m-roots if this step came from the direct step_factory path.
        if isinstance(step, dict) and step.get("r_expr") is not None:
            try:
                m_roots = self._solve_m_roots(step)
                print(f"  m-roots ({len(m_roots)}): {m_roots}")
                if rec.x_src is not None:
                    xj_from_m = [self._candidate_xj_from_m(rec.x_src, m) for m in m_roots]
                    print(f"  x_step from x_src - m: {xj_from_m}")
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
                        f"m={cand.get('m')} x_step={cand.get('x_step')} x_res={cand.get('x_res')}"
                    )
                else:
                    print(f"    cand[{i}] {cand}")
            if len(pool) > 12:
                print(f"    ... {len(pool) - 12} more candidates")

        # Quartic-model intersection polynomial.
        #poly = self._intersection_poly_from_step(step)
        poly = self._intersection_poly_from_step(poly_src, x_step=chosen.get("x_step"), x_res=chosen.get("x_res"))
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
        # src_mult is read from the record (set by _make_relation from the actual poly);
        # falling back to curve_degree-2 only when unavailable (e.g. old records).
        if rec.x_src is not None and rec.x_step is not None:
            try:
                lc = poly.leading_coefficient()
                monic = poly / lc
                coeffs = monic.list()  # low-to-high coefficients
                deg = int(monic.degree())
                a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else monic.parent().base_ring()(0)
                total_root_sum = -a_d_minus_1

                src_mult = int(getattr(rec, 'src_mult', -1) or -1)
                if src_mult <= 0:
                    src_mult = curve_degree - 2
                extra_roots_diag = list(getattr(rec, 'extra_roots', []) or [])
                known_roots_list = [src_mult * rec.x_src, rec.x_step] + extra_roots_diag
                known_sum = sum(known_roots_list)
                xk_vieta = total_root_sum - (src_mult * rec.x_src + rec.x_step + sum(extra_roots_diag))

                print(f"  monic sum-of-roots       = {total_root_sum}  (S evaluated at this m)")
                print(f"  src_mult (from record)    = {src_mult}")
                if extra_roots_diag:
                    print(f"  extra_roots              = {extra_roots_diag}")
                print(f"  known-root sum           = {known_sum}")
                print(f"  Vieta-predicted x_res       = {xk_vieta}")

                if rec.x_res is not None:
                    print(f"  x_res residual              = {rec.x_res - xk_vieta}")
                else:
                    print("  x_res residual              = <x_res not recovered for this step>")

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
                            f"x_res(m) = S(m) - {src_mult}*{rec.x_src} - x_step(m)"
                            f"      [x_step(m) = {rec.x_src} - m]"
                        )
                    else:
                        print(
                            "  recurrence (symbolic)    = "
                            f"x_res(m) = S(m) - {src_mult}*{rec.x_src} - x_step(m)"
                            f"      [x_step(m) != x_src - m: RLINEAR=False, RHS is quadratic]"
                        )
                else:
                    print(f"  S(m) symbolic            = <unavailable — fi not in step payload>")
                    if RLINEAR:
                        print(
                            "  recurrence preview       = "
                            f"x_step = x_src - m,  "
                            f"x_res = ({total_root_sum}) - ({src_mult}*x_src + x_step)"
                            f"  [S={total_root_sum} is numeric, m already substituted]"
                        )
                    else:
                        print(
                            "  recurrence preview       = "
                            f"x_res = ({total_root_sum}) - ({src_mult}*x_src + x_step)"
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
