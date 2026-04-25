def _step_direct(walker, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
    xi_before = walker.current_x
    current_point = (walker.current_x, walker.current_y)

    step = walker.step_factory(walker.current_x, n, seed=seed, current_point=current_point)
    m_roots = walker._solve_m_roots(step)

    if not m_roots:
        return walker._reject_direct_step(
            step_payload=step if isinstance(step, dict) else {},
            stage="direct_step",
            reason="no_m_roots",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            extra={"step": _jsonable(step)},
        )

    if not RLINEAR:
        raise RuntimeError(
            "direct step() path (step_factory) does not support RLINEAR=False: "
            "x_step cannot be recovered from m via x_src - m when the RHS is quadratic. "
            "Supply a search_fn that returns explicit x_step values in candidate records."
        )

    xj_candidates = [walker._candidate_xj_from_m(walker.current_x, m_val) for m_val in m_roots]
    if not xj_candidates:
        return walker._reject_direct_step(
            step_payload=step if isinstance(step, dict) else {},
            stage="direct_step",
            reason="no_xj_candidates",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            extra={"m_roots": _jsonable(m_roots)},
        )

    valid_leaves = {cx for cx in xj_candidates if cx is not None}
    xk_per_xj = []
    for xj_c in xj_candidates:
        xk_c, _ = walker._recover_x_res(step, walker.current_x, xj_c)
        xk_per_xj.append(xk_c)
        if xk_c is not None:
            valid_leaves.add(xk_c)

    missing_xj = [m for m, xj_c in zip(m_roots, xj_candidates) if xj_c is None]
    assert not missing_xj, (
        f"[ASSERT-a direct] {len(missing_xj)}/{len(m_roots)} m-roots produced no x_step leaf "
        f"at step={len(walker.history)+1} x_src={xi_before}. missing m-roots: {_missing_xj}"
    )
    assert valid_leaves, (
        f"[ASSERT-a direct] leaf set is empty after processing {len(m_roots)} m-roots "
        f"at step={len(walker.history)+1} x_src={xi_before}."
    )

    missing_xk = [(m, xj_c) for m, xj_c, xk_c in zip(m_roots, xj_candidates, xk_per_xj) if xk_c is None]
    if missing_xk:
        print(
            f"[WARN-a direct] {len(missing_xk)}/{len(m_roots)} x_step leaves have no recoverable x_res "
            f"at step={len(walker.history)+1} x_src={xi_before}: "
            f"(m, x_step) pairs without x_res = {missing_xk[:5]}"
            + (" ..." if len(missing_xk) > 5 else "")
        )

    print(
        f"[LEAVES direct] step={len(walker.history)+1} x_src={xi_before} "
        f"m-roots={len(m_roots)} x_step-leaves={len(xj_candidates)} "
        f"x_res-recovered={sum(1 for xk_c in xk_per_xj if xk_c is not None)} "
        f"total-leaves={len(valid_leaves)}"
    )

    valid_leaves, new_leaves_this_step, leaf_collisions_this_step = walker._update_leaf_bookkeeping(
        valid_leaves, n=n, xi_before=xi_before
    )

    step_payload = walker._make_direct_step_payload(
        step,
        step_leaves_found=len(valid_leaves),
        step_leaves_new=new_leaves_this_step,
        step_leaf_collisions=leaf_collisions_this_step,
    )

    if walker.config.verbose:
        sqrt_p = (walker.p ** 0.5) if walker.p is not None else float("nan")
        collision_frac = len(walker.global_leaves_seen) / sqrt_p if walker.p is not None else float("nan")
        print(
            f"\n[CANDIDATES] step={len(walker.history)+1} n={n} | "
            f"x_src={walker.current_x} | "
            f"m-roots: {len(xj_candidates)} -> {new_leaves_this_step} new leaves "
            f"(novelty {new_leaves_this_step / len(valid_leaves):.1%} if valid_leaves else 0.0%) | "
            f"Graph volume: {len(walker.global_leaves_seen)} "
            f"({collision_frac:.3f}×√p)"
        )

    x_step = xj_candidates[0]
    m_val = m_roots[0]
    x_res, _src_mult_unused = walker._recover_x_res(step, walker.current_x, x_step)

    if x_res is None:
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="missing_xk_recovery",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            chosen={"source": "direct_step"},
            extra={"step": _jsonable(step)},
        )

    xj_diag = walker._point_check_details(x_step, label="x_step")
    xk_diag = walker._point_check_details(x_res, label="x_res")

    try:
        next_y_xj = walker._recover_y(x_step, y_sign=1)
    except Exception as exc:
        raise
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="no_y",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "role": "x_step"},
            extra={
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
                "error": repr(exc),
            },
        )

    if next_y_xj == walker.base_ring(0):
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="weierstrass_y0",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "role": "x_step"},
            extra={
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    if not xk_is_fp_point(x_res, walker.curve_poly):
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="non_fp_xk",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "role": "x_res"},
            extra={
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    chosen = walker._choose_between(
        x_step,
        x_res,
        {
            "n": n,
            "step": step,
            "current_x": walker.current_x,
            "current_y": walker.current_y,
        },
    )
    if chosen is None:
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="no_move_choice",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step"},
            extra={
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    chosen_sign = 1 if chosen == x_step else int(step_payload.get("yk_sign", 1))
    try:
        next_y = walker._recover_y(chosen, y_sign=chosen_sign)
    except Exception as exc:
        raise
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="move_target_no_y",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "chosen": _jsonable(chosen)},
            extra={
                "chosen_target": _jsonable(chosen),
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
                "error": repr(exc),
            },
        )

    if next_y == walker.base_ring(0):
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="chosen_target_weierstrass_y0",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "chosen": _jsonable(chosen)},
            extra={
                "chosen_target": _jsonable(chosen),
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    if not walker._x_src_is_fresh(chosen):
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="repeated_xi_forbidden",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "chosen": _jsonable(chosen)},
            extra={
                "chosen_target": _jsonable(chosen),
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    if step_payload.get("intersection_poly") is None:
        return walker._reject_direct_step(
            step_payload=step_payload,
            stage="direct_step",
            reason="missing_intersection_poly",
            x_src=xi_before,
            n=n,
            current_point=current_point,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen={"source": "direct_step", "chosen": _jsonable(chosen)},
            extra={
                "xj_diagnostic": xj_diag,
                "xk_diagnostic": xk_diag,
            },
        )

    walker.current_x, walker.current_y = chosen, next_y
    walker.visited_x.add(chosen)
    walker.x_src_visit_count[xi_before] += 1

    unique_x_step_new, _ = walker._annotate_step_counts(step_payload, chosen, accepted=True)
    if not unique_x_step_new:
        walker.collision_count += 1

    step_payload["xj_diagnostic"] = xj_diag
    step_payload["xk_diagnostic"] = xk_diag
    step_payload["chosen_target"] = _jsonable(chosen)
    step_payload["move_committed"] = True

    rec = walker._accept_direct_step(
        step_payload=step_payload,
        n=n,
        x_src=xi_before,
        m_val=m_val,
        x_step=x_step,
        x_res=x_res,
        yj_sign=1,
        yk_sign=int(step_payload.get("yk_sign", 1)),
    )

    assert walker.history[-1] is rec, (
        f"[ASSERT-b direct] _store_record did not append rec at step={rec.step_index} x_src={xi_before}"
    )
    assert rec.accepted, (
        f"[ASSERT-b direct] accepted record has accepted=False at step={rec.step_index}"
    )
    assert rec.x_step is not None, (
        f"[ASSERT-b direct] accepted record has x_step=None at step={rec.step_index} x_src={xi_before}"
    )
    assert rec.x_res is not None, (
        f"[ASSERT-b direct] accepted record has x_res=None at step={rec.step_index} x_src={xi_before}"
    )
    print(
        f"[STORED direct] step_index={rec.step_index} x_src={rec.x_src} x_step={rec.x_step} x_res={rec.x_res} "
        f"history_len={len(walker.history)} accepted=True"
    )
    return rec
