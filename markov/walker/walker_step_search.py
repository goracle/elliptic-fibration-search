from .candidate_utils import _normalize_candidate_output
from .candidate_utils import *
from .curve_helpers import *

def step_from_candidate_search(walker, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
    x_src = walker.current_x
    pt = (walker.current_x, walker.current_y)
    # Mark x_src exhausted immediately — its fiber is deterministic, so any
    # subsequent run as current_x would produce no new information.
    walker.exhausted_x_src.add(x_src)

    def _poly_from(obj):
        if not isinstance(obj, dict):
            return None
        for key in ["intersection_poly"]:
            poly = obj.get(key)
            if poly is not None:
                return poly
        return None

    def _derive_from_poly(obj):
        poly = _poly_from(obj)
        if poly is None:
            assert None, "poly is missing!! gang."
            return None

        try:
            roots_wm = poly.roots(multiplicities=True)
        except Exception:
            raise
            return None

        src_mult = 0
        others = []
        for r, m in roots_wm:
            if r == x_src:
                src_mult += int(m)
            else:
                others.extend([r] * int(m))

        if src_mult <= 0:
            print(roots_wm, x_src, poly)
            raise ValueError
            return None

        # No multiplicity pattern assumed — derive directly from the root list.
        if not others:
            return None
        if len(others) == 1:
            x_step = others[0]
            x_res = x_src
            src_mult -= 1  # fold one copy of x_src into the x_res slot
            extra_roots = []
        elif len(others) == 2:
            x_step, x_res = others[0], others[1]
            extra_roots = []
        else:
            # 3+ non-x_src roots: x_step/x_res get the first two, rest go in extra_roots
            x_step = others[0]
            x_res = others[1]
            extra_roots = others[2:]

        return x_step, x_res, src_mult, poly, extra_roots

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
    organic = X - walker._injected_xs
    new_leaves_count = len(organic - walker.global_leaves_seen)

    # During thermalization, zero novelty means we haven't mixed yet — escape.
    # Post-thermalization, zero novelty is normal (graph is saturated near this x_src);
    # fall through to candidate selection and commit as a regular step.
    if new_leaves_count == 0 and len(X) > 0 and n < walker.config.nthermal:
        walker.dead_end_count += 1
        walker.dead_end_reasons["zero_novelty_thermal"] += 1
        walker.exhausted_x_src.add(x_src)
        rec = reject(
            "zero_novelty_thermal",
            extra={
                "leaves_found": len(X),
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
    #pool = walker._prefer_unvisited_candidates(pool) # Use the built-in tiering
    pool = list(pool)  # Make a copy so we can pop from it

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

        # Geometry must come from the intersection polynomial.
        poly_src = dict(search_out)
        poly_src.update(chosen)

        try:
            derived = _derive_from_poly(poly_src)
            if derived is None:
                pool.remove(chosen); continue

            x_step, x_res, src_mult, poly, extra_roots = derived

            # If the candidate carried explicit roots, require them to match the poly.
            cand_xj = chosen.get("x_step")
            cand_xk = chosen.get("x_res")
            if cand_xj is not None and cand_xk is not None and {cand_xj, cand_xk} != {x_step, x_res}:
                pool.remove(chosen); continue

            m = chosen.get("m")

            # --- validate x_step ---
            try:
                yj = walker._recover_y(x_step, y_sign=int(chosen.get("yj_sign", 1)))
            except Exception as e:
                pool.remove(chosen); continue

            if yj == walker.base_ring(0):
                pool.remove(chosen); continue

            if not xk_is_fp_point(x_step, walker.curve_poly):
                pool.remove(chosen); continue

            # --- validate x_res ---
            if not xk_is_fp_point(x_res, walker.curve_poly):
                pool.remove(chosen); continue

            try:
                yk = walker._recover_y(x_res, y_sign=int(chosen.get("yk_sign", 1)))
            except Exception as e:
                raise
                pool.remove(chosen); continue

            if yk == walker.base_ring(0):
                pool.remove(chosen); continue

            # --- choose move ---
            # Build the list of fresh options before selecting so the
            # choice is uniform over reachable neighbors.  The old code
            # flipped a coin first and then rejected on freshness, which
            # silently discarded an entire fiber whenever the coin landed
            # on the already-visited root — a systematic sampling bias.
            fresh_opts = []
            for _tgt_cand, _sign_key in ((x_step, "yj_sign"), (x_res, "yk_sign")):
                if _tgt_cand is None:
                    continue
                if not walker._x_src_is_fresh(_tgt_cand):
                    continue
                try:
                    _y_cand = walker._recover_y(
                        _tgt_cand,
                        y_sign=int(chosen.get(_sign_key, 1)),
                    )
                except Exception:
                    raise
                    continue
                if _y_cand == walker.base_ring(0):
                    continue
                fresh_opts.append((_tgt_cand, int(chosen.get(_sign_key, 1)), _y_cand))

            if not fresh_opts:
                pool.remove(chosen); continue

            tgt, sign, y = walker.rng.choice(fresh_opts)

            # Verify the relation is a principal divisor before committing.
            # Build atoms tentatively the same way _make_relation will, then check.
            # If it fails, try the next candidate rather than rejecting the whole step.
            _tentative_atoms = (
                [walker.base_ring(x_src)] * src_mult
                + [walker.base_ring(x_step), walker.base_ring(x_res)]
                + [walker.base_ring(xr) for xr in extra_roots]
            )
            if not walker._verify_atoms_principal(_tentative_atoms):
                print(f"  [verify] candidate non-principal, trying next  "
                      f"x_src={x_src} x_step={x_step} x_res={x_res}")
                pool.remove(chosen); continue

            # If we made it here, everything is valid!
            valid_candidate_found = True
            break

        except Exception:
                # Catch any _recover_y or internal errors for this specific candidate
                pool.remove(chosen)
                raise
                continue
            # --- END OF EXISTING VALIDATION LOGIC ---

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
    payload["intersection_poly"] = poly

    rec = walker._make_relation(
        len(walker.history), n, x_src, m, x_step, x_res,
        payload, accepted=True, restart=False,
        yj_sign=int(chosen.get("yj_sign", 1)),
        yk_sign=int(chosen.get("yk_sign", 1)),
    )

    rec.candidate_pool = C
    rec.selected_candidate = dict(chosen)

    walker._store_record(rec)
    return rec

