from .candidate_utils import *
from .curve_helpers import *
from .candidate_utils import _normalize_candidate_output

def step_from_candidate_search(walker, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
    x_src = walker.current_x
    pt = (walker.current_x, walker.current_y)
    # Mark x_src exhausted immediately — its fiber is deterministic, so any
    # subsequent run as current_x would produce no new information.
    walker.exhausted_x_src.add(x_src)

    def _poly_aux(poly, x_step, x_res):
        """Optional: extract src_mult and extra_roots from intersection_poly.

        The candidate record is the authoritative source for x_step, x_res,
        yj_sign, and yk_sign.  The poly (when present) is used only to:
          - count the multiplicity of x_src (for atom construction), and
          - collect any extra roots beyond x_step and x_res.

        Returns (src_mult, extra_roots) or None if the poly is absent,
        unfactorable, or doesn't contain x_src as a root.
        """
        if poly is None:
            return None
        try:
            roots_wm = poly.roots(multiplicities=True)
        except Exception:
            return None

        src_mult = 0
        others = []
        for r, m in roots_wm:
            if r == x_src:
                src_mult += int(m)
            else:
                others.extend([r] * int(m))

        if src_mult <= 0:
            return None

        # Extra roots: remove exactly one occurrence of x_step and one of x_res
        # from others (the expected single appearance of each in the divisor).
        # Any remaining copies — e.g. a second x_step when it appears as a
        # double root — are genuine extra atoms and must be kept.
        remaining = list(others)
        for expected in (x_step, x_res):
            try:
                remaining.remove(expected)
            except ValueError:
                pass  # not present (degenerate geometry), leave unchanged
        extra_roots = remaining
        return src_mult, extra_roots

    def reject(reason, *, m_val=None, x_step=None, x_res=None, chosen=None, extra=None):
        step_payload = walker._reject_step_payload(
            search_out,
            stage="candidate_search",
            reason=reason,
            x_src=x_src,
            n=n,
            current_point=pt,
            m_val=m_val,
            x_step=x_step,
            x_res=x_res,
            chosen=chosen,
            extra=extra or {},
        )
        rec = walker._make_relation(
            len(walker.history), n, x_src, m_val, x_step, x_res,
            step_payload, accepted=False, restart=False,
        )
        walker._store_record(rec)
        return rec

    # --- search ---
    raw = walker._call_search_fn(n=n, seed=seed, current_point=pt)
    search_out = _normalize_candidate_output(raw)
    C = list(search_out.get("candidate_records") or search_out.get("candidates") or [])
    X = {x for x in search_out.get("candidate_xs", set()) if x is not None}

    # --- leaf bookkeeping ---
    # Novelty counts only genuine next-step candidates: x_step values that differ
    # from x_src.  Self-steps (x_step == x_src) are a geometry failure in the phi
    # branch, not a mixing failure, and must never trigger zero_novelty_thermal.
    X_novel_candidates = {x for x in X if x != x_src}
    organic = X_novel_candidates - walker._injected_xs
    new_leaves_count = len(organic - walker.global_leaves_seen)

    # During thermalization, zero novelty means we haven't mixed yet — escape.
    # Post-thermalization, zero novelty is normal (graph is saturated near this x_src);
    # fall through to candidate selection and commit as a regular step.
    #
    # Exception 1: if x_src has never been used as x_src before (visit count == 0),
    # this is a freshly-restarted position.  Escaping immediately would cause an
    # infinite restart loop because the restart point's leaves are all already seen.
    # Let the step commit so the walk actually advances from the new position.
    #
    # Exception 2: if X_novel_candidates is empty, there are no non-self leaves at
    # all — this is a pure geometry failure, not a mixing failure.  Fall through so
    # the dead-end path handles it cleanly instead of looping through restarts.
    _x_src_prior_visits = walker.x_src_visit_count.get(x_src, 0)
    if (new_leaves_count == 0 and len(X_novel_candidates) > 0
            and n < walker.config.nthermal and _x_src_prior_visits > 0):
        walker.dead_end_count += 1
        walker.dead_end_reasons["zero_novelty_thermal"] += 1
        walker.exhausted_x_src.add(x_src)
        rec = reject(
            "zero_novelty_thermal",
            extra={
                "leaves_found": len(X_novel_candidates),
                "leaves_found_including_self": len(X),
                "thermal_threshold": walker.config.nthermal,
                "thermalized": False,
            },
        )
        walker._restart_after_dead_end(
            x_src=x_src,
            n=n,
            reason="zero_novelty_thermal",
            current_point=pt,
        )
        return rec

    _, new_leaves_this_step, leaf_collisions_this_step = \
        walker._update_leaf_bookkeeping(X, n=n, xi_before=x_src)

    search_out.update({
        "step_leaves_found": len(X),
        "step_leaves_new": new_leaves_this_step,
        "step_leaf_collisions": leaf_collisions_this_step,
        "global_leaves_total": len(walker.global_leaves_seen),
        "global_leaf_collisions": walker.leaf_collision_count,
    })

    # --- dead end ---
    if not C:
        walker.dead_end_count += 1
        r = search_out.get("dead_end_reason", "unknown")
        walker.dead_end_reasons[r] += 1
        rec = reject("no_candidates", extra={"dead_end_reason": r})

        if walker.config.restart_on_dead_end:
            nxt = walker._restart_from_valid_curve_point(exclude={x_src})
            if nxt:
                walker.current_x, walker.current_y = nxt
                walker.visited_x.add(nxt[0])
            else:
                if walker.config.verbose:
                    print(
                        "[dead_end restart] no valid curve point found in base points or leaf pool; "
                        "leaving current state unchanged."
                    )
        return rec

    # --- choose candidate ---
    def is_fp(c):
        if not isinstance(c, dict):
            return False
        return walker._point_check_details(c.get("x_res"), "x_res").get("is_fp_point", False)

    pool = [c for c in C if is_fp(c)] or C
    pool = list(pool)

    valid_candidate_found = False

    while pool:
        chosen = walker._choose_candidate_record(
            pool,
            {"n": n, "step": search_out, "current_x": x_src, "current_y": pt[1]},
        )

        if chosen is None:
            return reject("selection_failed", extra={"candidate_count": len(C)})

        if not isinstance(chosen, dict):
            chosen = {"x_step": chosen}

        try:
            # --- geometry from candidate record (authoritative) ---
            x_step = chosen.get("x_step")
            x_res  = chosen.get("x_res")
            m      = chosen.get("m")

            if x_step is None or x_res is None:
                print(f"  [cand_skip] missing x_step or x_res  x_src={x_src} rec={chosen.get('source')}")
                pool.remove(chosen); continue

            # --- degenerate: both neighbors are x_src (phi self-loop geometry) ---
            # This happens when the fiber has a triple root at x_src and the phi
            # branch sets x_step=x_res=x_src.  No fresh move is possible; skip
            # before entering the full validation gauntlet.
            if x_step == x_src and x_res == x_src:
                print(f"  [cand_skip] degenerate self-loop: x_step=x_res=x_src={x_src}  source={chosen.get('source')}")
                pool.remove(chosen); continue

            # --- src_mult + extra_roots: candidate record is authoritative ---
            # enrich_candidates stores actual_xi_mult on each record; use it.
            # Fall back to poly factoring only when the record doesn't carry it.
            # Final fallback: curve_degree - 2 (generic 2P+2Q+R divisor).
            cand_src_mult = chosen.get("src_mult")
            poly = chosen.get("intersection_poly")
            aux  = _poly_aux(poly, x_step, x_res)
            if aux is not None:
                src_mult, extra_roots = aux
            elif cand_src_mult is not None and int(cand_src_mult) > 0:
                src_mult    = int(cand_src_mult)
                extra_roots = list(chosen.get("extra_roots") or [])
                poly        = None
            else:
                src_mult    = walker.config.curve_degree - 2
                extra_roots = []
                poly        = None

            # --- validate x_step ---
            if not xk_is_fp_point(x_step, walker.curve_poly):
                print(f"  [cand_skip] x_step={x_step} not on curve  x_src={x_src}")
                pool.remove(chosen); continue

            try:
                yj = walker._recover_y(x_step, y_sign=int(chosen.get("yj_sign", 1)))
            except Exception as e:
                print(f"  [cand_skip] _recover_y failed for x_step={x_step}: {e}")
                pool.remove(chosen); continue

            if yj == walker.base_ring(0):
                print(f"  [cand_skip] x_step={x_step} is Weierstrass point (y=0)")
                pool.remove(chosen); continue

            # --- validate x_res ---
            if not xk_is_fp_point(x_res, walker.curve_poly):
                print(f"  [cand_skip] x_res={x_res} not on curve  x_src={x_src}")
                pool.remove(chosen); continue

            try:
                yk = walker._recover_y(x_res, y_sign=int(chosen.get("yk_sign", 1)))
            except Exception as e:
                print(f"  [cand_skip] _recover_y failed for x_res={x_res}: {e}")
                pool.remove(chosen); continue

            if yk == walker.base_ring(0):
                print(f"  [cand_skip] x_res={x_res} is Weierstrass point (y=0)")
                pool.remove(chosen); continue

            # --- choose move target ---
            # Build fresh-options list over both neighbors uniformly, avoiding
            # the old coin-flip bias that silently discarded half the fibers.
            # x_src itself is never a valid move target (exhausted at top of fn).
            # x_res may equal x_step in the phi double-root geometry (2P+2Q+R);
            # in that case it still contributes a valid move via x_step.
            fresh_opts = []
            for _tgt, _sign_key in ((x_step, "yj_sign"), (x_res, "yk_sign")):
                if _tgt is None:
                    continue
                if _tgt == x_src:
                    # Self-loop half: skip this neighbor but keep the other.
                    continue
                if not walker._x_src_is_fresh(_tgt):
                    continue
                try:
                    _y = walker._recover_y(_tgt, y_sign=int(chosen.get(_sign_key, 1)))
                except Exception:
                    continue
                if _y == walker.base_ring(0):
                    continue
                fresh_opts.append((_tgt, int(chosen.get(_sign_key, 1)), _y))

            if not fresh_opts:
                _why = []
                for _tgt, _sk in ((x_step, "yj_sign"), (x_res, "yk_sign")):
                    if _tgt is None: _why.append("None")
                    elif _tgt == x_src: _why.append(f"{_tgt}==x_src")
                    elif not walker._x_src_is_fresh(_tgt): _why.append(f"{_tgt}=not_fresh(visited={_tgt in walker.visited_x},exhausted={_tgt in walker.exhausted_x_src})")
                    else: _why.append(f"{_tgt}=y_fail")
                print(f"  [cand_skip] fresh_opts empty  x_src={x_src} x_step={x_step} x_res={x_res}  reasons={_why}")
                pool.remove(chosen); continue

            tgt, sign, y = walker.rng.choice(fresh_opts)

            if x_step != x_res: print(f"[generic_cand] x_src={x_src} x_step={x_step} x_res={x_res} poly={poly is not None}")
            # --- verify relation is principal ---
            # Only run when phi has fired and produced a real intersection_poly.
            # Pre-phi atoms (x_step==x_res double-root geometry) are not expected
            # to be principal and must not be checked here.
            if poly is not None:
                _tentative_atoms = (
                    [walker.base_ring(x_src)] * src_mult
                    + [walker.base_ring(x_step), walker.base_ring(x_res)]
                    + [walker.base_ring(xr) for xr in extra_roots]
                )
                _aux_raw = _poly_aux(poly, x_step, x_res)
                print(f"  [verify_dbg] x_src={x_src} x_step={x_step} x_res={x_res} "
                      f"src_mult={src_mult} "
                      f"aux={'None' if _aux_raw is None else f'src_mult={_aux_raw[0]} extra={_aux_raw[1]}'} "
                      f"cand_src_mult={cand_src_mult} "
                      f"poly_roots={[(int(r), int(m)) for r, m in poly.roots(multiplicities=True)] if poly is not None else 'N/A'} "
                      f"atoms={[int(a) for a in _tentative_atoms]}")
                if not walker._verify_atoms_principal(_tentative_atoms):
                    print(f"  [verify] candidate non-principal, trying next  "
                          f"x_src={x_src} x_step={x_step} x_res={x_res}")
                    pool.remove(chosen); continue

            valid_candidate_found = True
            break

        except Exception:
            pool.remove(chosen)
            continue

    if not valid_candidate_found:
        walker.dead_end_count += 1
        walker.dead_end_reasons["all_candidates_failed_validation"] += 1
        rec = reject("all_candidates_failed_validation", extra={"candidate_count": len(C)})
        if walker.config.restart_on_dead_end and not walker.walk_terminated:
            nxt = walker._restart_after_dead_end(
                x_src=x_src,
                n=n,
                reason="all_candidates_failed_validation",
                current_point=pt,
            )
            if nxt is None:
                walker.walk_terminated = True
        return rec

    # --- commit ---
    walker.current_x, walker.current_y = tgt, y
    walker.visited_x.add(tgt)
    walker.x_src_visit_count[x_src] += 1

    new_flag, _ = walker._annotate_step_counts(search_out, tgt, accepted=True)
    if not new_flag:
        walker.collision_count += 1

    payload = dict(search_out)
    payload["move_committed"] = True
    payload["intersection_poly"] = poly  # may be None if phi didn't fire

    rec = walker._make_relation(
        len(walker.history), n, x_src, m, x_step, x_res,
        payload, accepted=True, restart=False,
        yj_sign=int(chosen.get("yj_sign", 1)),
        yk_sign=int(chosen.get("yk_sign", 1)),
        src_mult=src_mult,
        extra_roots=extra_roots,
    )

    rec.candidate_pool = C
    rec.selected_candidate = dict(chosen)

    walker._store_record(rec)
    return rec
