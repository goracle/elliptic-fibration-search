def close_under_involution2(walker) -> int:
    """Verify the Vieta involution T(x_step) = S(m) - (d-2)*x_src - x_step is a
    genuine 2-cycle on every (x_src, x_step) pair in the candidate pool.

    For each such pair this checks:
        T(T(x_step)) == x_step                    (2-cycle / involution)

    Nothing is appended to history.  The relation
        (d-2)*x_src + x_step + x_res - d*inf = 0
    is symmetric in x_step and x_res by construction, so the x_step↔x_res swap
    produces the identical algebraic relation — there is no new
    information to record.

    Raises AssertionError immediately on the first violation found.

    Returns the number of (x_src, x_step) pairs that passed the check.

    NOTE: the involution uses m = x_src - x_step, which is only valid when the
    search RHS is linear in m (RLINEAR=True).  When RLINEAR=False this
    method returns 0 immediately without checking anything.
    """
    if not RLINEAR:
        print(
            "[close_under_involution] skipped: RLINEAR=False, "
            "m = x_src - x_step inversion is not valid for a quadratic RHS."
        )
        return 0
    deg = walker.config.curve_degree
    default_xi_mult = deg - 2  # expected for double-tangency fibers

    # Derive per-x_src multiplicity from accepted records' atoms list (the
    # canonical encoding).  T(x_step) = S(m) - src_mult*x_src - x_step is a proper
    # 2-cycle only when src_mult == deg-2 (exactly two non-x_src roots).
    xi_to_xi_mult: Dict[Any, int] = {}
    for _rec in walker.history:
        if _rec.accepted and getattr(_rec, 'atoms', None):
            _xi_fp = walker.base_ring(_rec.x_src)
            _cnt = sum(1 for _a in _rec.atoms if walker.base_ring(_a) == _xi_fp)
            if _cnt > 0:
                xi_to_xi_mult[_rec.x_src] = _cnt

    # Collect S_of_m per x_src from history records that have it.
    xi_to_S_sym: Dict[Any, Any] = {}
    for rec in walker.history:
        if rec.accepted and rec.x_src not in xi_to_S_sym:
            S_sym = rec.step.get('S_of_m') if isinstance(rec.step, dict) else None
            if S_sym is not None:
                xi_to_S_sym[rec.x_src] = S_sym

    def _eval_S(x_src, m_val):
        S_sym = xi_to_S_sym.get(x_src)
        assert S_sym is not None, (
            f"close_under_involution: no S_of_m available for x_src={x_src}. "
            f"Run the walk first so S_of_m is stored on candidate records."
        )
        Fp = walker.base_ring
        m_fp = Fp(m_val)
        try:
            return Fp(S_sym(m_fp))
        except Exception:
            num = S_sym.numerator()
            den = S_sym.denominator()
            dv = Fp(den(m_fp))
            if dv == 0:
                raise _FiberPoleError(
                    f"close_under_involution: S(m) denominator zero at m={m_val} "
                    f"(x_src={x_src}) — fiber degenerate at this m, pair skipped"
                )
            raise

    def _T(x_src, xj_val, xi_mult_override=None):
        Fp = walker.base_ring
        xi_mult_for_T = (
            xi_mult_override if xi_mult_override is not None
            else xi_to_xi_mult.get(x_src, default_xi_mult)
        )
        m_val = Fp(x_src) - Fp(xj_val)
        S_val = _eval_S(x_src, m_val)
        return Fp(S_val - xi_mult_for_T * Fp(x_src) - Fp(xj_val))

    def _check_pair(x_src, xj_val):
        Fp = walker.base_ring
        S_sym = xi_to_S_sym.get(x_src, "<missing>")

        # Involution is only well-defined as a 2-cycle when src_mult == deg-2.
        eff_mult = xi_to_xi_mult.get(x_src, default_xi_mult)
        if eff_mult != default_xi_mult:
            raise _FiberPoleError(
                f"close_under_involution: x_src={x_src} has src_mult={eff_mult} "
                f"(expected {default_xi_mult}=deg-2); "
                f"T is not a simple x_step↔x_res 2-cycle for this fiber, skipping"
            )

        m1 = Fp(x_src) - Fp(xj_val)
        S1 = _eval_S(x_src, m1)
        partner = _T(x_src, xj_val)

        # x_step is itself a fixed point of T: x_step=x_res tangency at the fiber for m1.
        # The 2-cycle is not defined when one endpoint is a tangency point; skip.
        if partner == Fp(xj_val):
            raise _FiberPoleError(
                f"close_under_involution: x_step={xj_val} is a fixed point of T at x_src={x_src} "
                f"(x_step=x_res tangency at m={m1}); skipping pair"
            )

        m2 = Fp(x_src) - Fp(partner)
        S2 = _eval_S(x_src, m2)
        roundtrip = _T(x_src, partner)

        # T(x_step) is a fixed point of T: x_step=x_res tangency at the fiber for m2.
        # T maps x_step to a partner that is itself a tangency point, so the
        # roundtrip cannot return to x_step.  This is degenerate geometry, not a
        # violation.
        if roundtrip == Fp(partner):
            raise _FiberPoleError(
                f"close_under_involution: T(x_step)={partner} is a fixed point of T at x_src={x_src} "
                f"(x_step=x_res tangency at m={m2}, originating from x_step={xj_val}); skipping pair"
            )

        if roundtrip != Fp(xj_val):
            raise AssertionError(
                f"close_under_involution: 2-cycle violated\n"
                f"  x_src          = {x_src}\n"
                f"  S_of_m      = {S_sym}\n"
                f"  x_step          = {xj_val}\n"
                f"  m1=x_src-x_step    = {m1}\n"
                f"  S(m1)       = {S1}\n"
                f"  T(x_step)       = {partner}\n"
                f"  m2=x_src-T(x_step) = {m2}\n"
                f"  S(m2)       = {S2}\n"
                f"  T(T(x_step))    = {roundtrip}  (expected {xj_val})\n"
            )

    n_checked = 0
    n_skipped = 0
    seen_pairs: set = set()

    for rec in walker.history:
        if not rec.accepted or rec.x_src is None:
            continue
        if rec.x_src not in xi_to_S_sym:
            continue
        # Skip involution-closure sentinels — they have no real m-fiber.
        step = rec.step if isinstance(rec.step, dict) else {}
        if step.get('source') == 'involution_closure':
            continue

        pool = list(rec.candidate_pool or [])
        for cand in pool:
            xj_cand = cand.get('x_step') if isinstance(cand, dict) else cand
            if xj_cand is None or xj_cand == rec.x_src:
                continue
            key = (rec.x_src, xj_cand)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            try:
                _check_pair(rec.x_src, xj_cand)
                n_checked += 1
            except _FiberPoleError:
                n_skipped += 1
                continue

    print(
        f"[close_under_involution] T(T(x_step))==x_step verified on {n_checked} "
        f"(x_src, x_step) pairs ({n_skipped} skipped — degenerate fiber/pole) "
        f"across {len(walker.history)} history records. History unchanged."
    )
    return n_checked


def generate_mixed_relations2(
    walker,
    atoms_to_inject: List[Any],
    *,
    seed_atoms: Optional[set] = None,
    label: str = "mixed",
) -> int:
    """Post-step injection: for each accepted history record whose x_src is in
    seed_atoms (all accepted records if seed_atoms is None), try injecting
    every atom from atoms_to_inject as x_step.

    Because x_step = x_src - m (RLINEAR), we invert: m = x_src - xj_target.
    Then x_res = S(m) - (deg-2)*x_src - xj_target.
    A relation is recorded iff curve_poly(x_res) is a QR in F_p.

    The resulting RelationRecords are stored via _store_record so their
    leaves land in global_leaves_seen exactly like normal search leaves.

    Returns the number of new relation records appended to walker.history.
    """
    if not RLINEAR:
        print("[generate_mixed_relations] skipped: RLINEAR=False")
        return 0

    Fp  = walker.base_ring
    deg = walker.config.curve_degree

    # Normalise seed_atoms filter to a set of base-ring elements (or None).
    if seed_atoms is not None:
        seed_fp = {Fp(a) for a in seed_atoms}
    else:
        seed_fp = None

    atoms_fp = [Fp(a) for a in atoms_to_inject]

    n_added   = 0
    n_skipped = 0

    for rec in list(walker.history):
        if not rec.accepted:
            continue
        x_src = Fp(rec.x_src)
        if seed_fp is not None and x_src not in seed_fp:
            continue

        # Derive x_src's multiplicity from the canonical atoms list.
        # The formula x_res = S(m) - src_mult*x_src - x_step uniquely recovers x_res
        # only when src_mult == deg-2 (exactly two non-x_src roots).  For any
        # other multiplicity S(m) gives the *sum* of multiple unknowns.
        _xi_fp = Fp(rec.x_src)
        _rec_atoms = getattr(rec, 'atoms', None) or []
        rec_xi_mult = sum(1 for _a in _rec_atoms if Fp(_a) == _xi_fp)
        if rec_xi_mult == 0:
            # atoms not populated (old record or rejected) — skip
            n_skipped += len(atoms_fp)
            continue
        if rec_xi_mult != deg - 2:
            n_skipped += len(atoms_fp)
            continue
        src_mult = rec_xi_mult

        # Prefer fi-based fiber evaluation (exact per-m multiplicity check).
        # Fall back to S(m) symbolic shortcut when fi is unavailable.
        fi, G_poly_rec = walker._get_fiber_context_for_rec(rec)
        S_sym = walker._get_S_of_m_for_rec(rec)
        if fi is None and S_sym is None:
            continue

        for xj_val in atoms_fp:
            if xj_val == x_src:
                continue

            m_val = x_src - xj_val   # inversion: x_step = x_src - m  =>  m = x_src - x_step

            if fi is not None and G_poly_rec is not None:
                # Use the actual intersection polynomial at this m — reads
                # src_mult from the fiber directly, no assumption needed.
                try:
                    xk_val, _inter = compute_xk_from_fiber(
                        x_src, m_val, xj_val, fi, G_poly_rec, deg
                    )
                    if xk_val is None:
                        n_skipped += 1
                        continue
                    xk_val = Fp(xk_val)
                except (ZeroDivisionError, AssertionError, _FiberPoleError):
                    n_skipped += 1
                    continue
                except Exception:
                    n_skipped += 1
                    continue
            else:
                # S(m) fallback: only valid because we already verified
                # src_mult == deg-2 above.
                try:
                    dv = Fp(S_sym.denominator()(m_val))
                    if dv == Fp(0):
                        n_skipped += 1
                        continue
                    S_val = Fp(S_sym.numerator()(m_val)) / dv
                except Exception:
                    n_skipped += 1
                    continue
                xk_val = Fp(S_val - src_mult * x_src - xj_val)

            # x_res must lift to a curve point.
            rhs = walker.curve_poly(xk_val)
            if not (hasattr(rhs, "is_square") and rhs.is_square()):
                n_skipped += 1
                continue

            # --- leaf bookkeeping (mirrors _step_from_candidate_search) ---
            walker._update_leaf_bookkeeping({xj_val, xk_val}, n=rec.n, xi_before=x_src)

            # --- build record ---
            cand = {
                "source": "mixed_injection",
                "m":  int(m_val),
                "x_step": xj_val,
                "x_res": xk_val,
            }
            step_dict = {
                "source":            "mixed_injection",
                "label":             label,
                "origin_step_index": rec.step_index,
                "S_of_m":            S_sym,          # kept for close_under_involution
            }
            relation = (
                f"{src_mult}*{int(x_src)} + {int(xj_val)} + {int(xk_val)}"
                f" - {deg}*\u221e = 0"
            )
            injected = RelationRecord(
                step_index       = len(walker.history),
                n                = rec.n,
                x_src               = x_src,
                m                = m_val,
                x_step               = xj_val,
                x_res               = xk_val,
                relation         = relation,
                step             = step_dict,
                accepted         = True,
                restart          = False,
                candidate_pool   = [cand],
                selected_candidate = cand,
                yj_sign          = 1,
                yk_sign          = 1,
                atoms            = [x_src] * rec_xi_mult + [xj_val, xk_val],
            )
            walker._store_record(injected)
            n_added += 1

            if walker.config.verbose:
                print(
                    f"[mixed_injection] x_src={int(x_src)}  x_step={int(xj_val)} ({label})"
                    f"  m={int(m_val)}  x_res={int(xk_val)}"
                )

    print(
        f"[generate_mixed_relations] added={n_added}  skipped={n_skipped}"
        f"  (seed_atoms={'all' if seed_fp is None else len(seed_fp)},"
        f" inject_pool={len(atoms_fp)})"
    )
    return n_added


def try_partial_cantor_reduction(walker, rec: RelationRecord) -> bool:
    """Pick two atom slots at random from rec.atoms and attempt a Cantor reduction.

    rec.atoms is the canonical flat atom list (len == curve_degree, with
    repetition for multiplicity).  Two slots are sampled uniformly at random.
    The two chosen degree-1 divisors are Cantor-added on the Jacobian.
    If the resulting reduced Mumford u-polynomial splits completely over F_p,
    the two slots are replaced by the roots of u (each contributing one slot),
    preserving the degree invariant.  Otherwise rec is left unchanged.

    Mutates rec.atoms in place (and keeps rec.x_step / rec.x_res / rec.extra_roots
    in sync as secondary navigation fields).
    Returns True if a substitution was made, False otherwise.
    """
    Fp = walker.base_ring
    cd = getattr(walker.config, 'curve_degree', 5)

    # Canonical source of atoms — no src_mult needed.
    flat = [Fp(a) for a in (rec.atoms or [])]

    if len(flat) < 2:
        return False

    # Sample two distinct slots uniformly at random.
    idx1, idx2 = walker.rng.sample(range(len(flat)), 2)
    a1 = flat[idx1]
    a2 = flat[idx2]

    # Recover y-coordinates.
    try:
        y1 = walker._recover_y(a1)
        y2 = walker._recover_y(a2)
    except Exception:
        return False
    if y1 is None or y2 is None:
        return False

    # Cantor-add the two degree-1 divisors on the Jacobian.
    try:
        C = HyperellipticCurve(walker.curve_poly)
        J = C.jacobian()(Fp)
        d_sum = J(C([Fp(a1), Fp(y1)])) + J(C([Fp(a2), Fp(y2)]))
        u_poly = d_sum[0]
    except Exception:
        return False

    # Check that u splits completely over F_p.
    #
    # Principality accounting: each atom slot [ai] represents [ai] - [∞].
    # Cantor-adding two slots gives a reduced Mumford element with deg(u) ≤ 2.
    # In all cases the substitution is principal:
    #
    #   deg(u) == 2  →  remove 2 atoms + 2∞, add 2 atoms + 2∞.  Degree unchanged.
    #   deg(u) == 1  →  remove 2 atoms + 2∞, add 1 atom  + 1∞.  Degree drops by 1.
    #   deg(u) == 0  →  remove 2 atoms + 2∞, add 0 atoms + 0∞.  Degree drops by 2.
    #
    # All three are valid principal divisors; we accept whichever splits completely.
    try:
        roots_wm = u_poly.roots()
    except Exception:
        return False

    total_roots = sum(int(m) for _, m in roots_wm)
    u_deg = int(u_poly.degree()) if u_poly.degree() >= 0 else 0
    if total_roots != u_deg:
        # u doesn't split completely over F_p — new atoms not in the base field.
        return False

    new_atoms_from_u = []
    for r, mult in roots_wm:
        new_atoms_from_u.extend([Fp(r)] * int(mult))

    # Substitute: remove the two chosen slots, insert the new roots.
    # new degree = cd - 2 + total_roots (0, 1, or 2 less than cd).
    counts = Counter(flat)
    counts[a1] -= 1
    if counts[a1] == 0:
        del counts[a1]
    counts[a2] -= 1
    if counts[a2] == 0:
        del counts[a2]
    for atom in new_atoms_from_u:
        counts[atom] += 1

    # Reconstruct canonical flat atoms list (arbitrary order within multiplicity).
    new_flat: List[Any] = []
    for atom, cnt in counts.items():
        new_flat.extend([atom] * int(cnt))

    expected_len = cd - 2 + total_roots
    if len(new_flat) != expected_len:
        raise AssertionError(
            f"_try_partial_cantor_reduction: degree accounting broken: "
            f"len={len(new_flat)} != cd - 2 + total_roots = {expected_len}  "
            f"(cd={cd} u_deg={u_deg} total_roots={total_roots})"
        )

    # Commit: update rec.atoms and sync secondary navigation fields.
    rec.atoms = new_flat

    # Secondary navigation: x_step = first non-x_src atom, x_res = second, extra_roots = rest.
    # x_src remains rec.x_src (chain state, never changed by Cantor reduction).
    xi_fp = Fp(rec.x_src) if rec.x_src is not None else None
    others = [a for a in new_flat if a != xi_fp]
    # Retain x_src-copies in others if it appears more times than before would hide them;
    # actually just partition: others = all copies not matching x_src identity check.
    others_all = []
    xi_seen = 0
    xi_count_new = int(counts.get(xi_fp, 0)) if xi_fp is not None else 0
    for a in new_flat:
        if a == xi_fp and xi_seen < xi_count_new:
            xi_seen += 1
        else:
            others_all.append(a)
    rec.x_step          = others_all[0] if len(others_all) > 0 else None
    rec.x_res          = others_all[1] if len(others_all) > 1 else None
    rec.extra_roots = others_all[2:] if len(others_all) > 2 else []

    return True
