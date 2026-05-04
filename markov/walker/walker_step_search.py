from .candidate_utils import *
from .curve_helpers import *
from typing import Optional, List, Dict, Any, Tuple
from .dlp_anchor_injection import inject_dlp_relations

def step_from_candidate_search(walker, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
    """
    High-level Markov step logic. Fully transitioned to (x, y) coordinate pairs.
    """
    # 1. Initialize State as full points
    pt_src_tuple = (walker.current_x, walker.current_y)
    # We still track exhausted x for fiber-level pruning
    walker.exhausted_pt_src.add(walker.current_x)

    # 2. Search & Normalize
    raw = walker._call_search_fn(n=n, seed=seed, current_point=pt_src_tuple)
    search_out = normalize_candidate_output(raw)

    # 3. Leaf & Novelty Bookkeeping
    _handle_thermal_novelty(walker, search_out, pt_src_tuple, n)

    # 4. Candidate Pool Filtering
    candidates = search_out.get("candidate_records", [])
    if not candidates:
        return _handle_dead_end(walker, search_out, pt_src_tuple, n, "no_candidates")

    # 5. Validation Loop
    valid_record = _select_and_validate_candidate(walker, candidates, search_out, pt_src_tuple, n)

    if not valid_record:
        return _handle_dead_end(walker, search_out, pt_src_tuple, n, "all_candidates_failed_validation")

    # 6. Commit Step
    return _commit_step(walker, valid_record, search_out, pt_src_tuple, n)

def _commit_step(walker, chosen, search_out, pt_src, n):
    """
    Commits the validated move to the walker's history.
    """
    # Choose target from neighbors
    fresh_opts = []
    for key, pt in [("yj_sign", chosen["pt_step"]), ("yk_sign", chosen["pt_res"])]:
        if pt != pt_src and walker._pt_src_is_fresh(pt[0]):
            fresh_opts.append(pt)

    if not fresh_opts:
        # Should be rare if the candidate passed validation
        return _handle_dead_end(walker, search_out, pt_src, n, "no_fresh_neighbors")

    target_pt = walker.rng.choice(fresh_opts)

    # Update Walker state
    walker.current_x, walker.current_y = target_pt
    walker.visited_x.add(target_pt[0])
    walker.pt_src_visit_count[pt_src] += 1

    # Create Relation
    # Pass chosen as step_metadata so _make_relation finds _validated_atoms.
    # Merge search_out keys into chosen so logging fields are preserved.
    step_metadata = dict(search_out)
    step_metadata.update(chosen)
    rec = walker._make_relation(
            len(walker.history), n, pt_src, chosen.get("m"),
            chosen["pt_step"], chosen["pt_res"],
            step_metadata, accepted=True,
            src_mult=chosen["_src_mult"],
            extra_pts=chosen.get("_extra_roots")
        )

    walker._store_record(rec)

    # Attempt synthetic phi-relations anchored at each G-atom and T-atom so
    # those x-coordinates appear as columns in the relation matrix.
    # No-op if set_dlp_points() has not been called on the walker.
    inject_dlp_relations(walker, rec, n)

    return rec

def _handle_dead_end(walker, search_out, pt_src, n, reason):
    """
    Standardized dead-end handling with tuple-aware logging.
    """
    walker.dead_end_count += 1
    r = search_out.get("dead_end_reason", reason)
    walker.dead_end_reasons[r] += 1

    # Payload expects x-scalar for history but full point for context
    step_payload = walker._reject_step_payload(
        search_out, stage="candidate_search", reason=r,
        pt_src=pt_src[0], n=n, current_point=pt_src
    )

    rec = walker._make_relation(
        len(walker.history), n, pt_src[0], None, None, None,
        step_payload, accepted=False, restart=False
    )
    walker._store_record(rec)

    if walker.config.restart_on_dead_end and not walker.walk_terminated:
        # Pass the full point to the restart handler
        nxt = walker._restart_after_dead_end(
            pt_src=pt_src[0], n=n, reason=r, current_point=pt_src
        )
        if nxt:
            walker.current_x, walker.current_y = nxt
            walker.visited_x.add(nxt[0])
        else:
            walker.walk_terminated = True

    return rec

def _handle_thermal_novelty(walker, search_out, pt_src, n):
    """
    Enforces mixing requirements during thermalization.
    """
    candidate_pts = search_out.get("candidate_pt", set())
    # candidate_pt may contain (x, y) tuples now; extract x-coords for
    # graph-volume / birthday accounting (global_leaves_seen is bare x-scalars).
    candidate_xs = set()
    for pt in candidate_pts:
        candidate_xs.add(pt[0] if isinstance(pt, (tuple, list)) else pt)
    # Ensure comparison is tuple-to-tuple for novelty check
    X_novel = {p for p in candidate_pts if p != pt_src}
    organic = X_novel - walker._injected_pts
    new_leaves = len(organic - walker.global_leaves_seen)

    _visits = walker.pt_src_visit_count.get(pt_src[0], 0)

    # Thermalization check: escape if we aren't finding new ground
    if (new_leaves == 0 and len(X_novel) > 0
            and n < walker.config.nthermal and _visits > 0):
        # We raise a controlled Exception to be caught by the restart logic
        # or return a specific signal for the caller to handle.
        search_out["dead_end_reason"] = "zero_novelty_thermal"

    _, new_this_step, collisions = walker._update_leaf_bookkeeping(candidate_xs, n=n, xi_before=pt_src[0])

    search_out.update({
        "step_leaves_found": len(candidate_pts),
        "step_leaves_new": new_this_step,
        "step_leaf_collisions": collisions,
    })

def _is_well_formed_candidate(c: Any) -> bool:
    """
    Firewall check: Ensures the record is a dict and points are subscriptable tuples.
    """
    if not isinstance(c, dict):
        return False

    # Requirement: pt_step and pt_res must be present and must be tuples/lists of length 2
    for field in ("pt_step", "pt_res"):
        p = c.get(field)
        if p is None:
            return False
        if not isinstance(p, (tuple, list)) or len(p) < 2:
            return False

    return True

def _build_atoms(walker, pt_src, pt_step, pt_res, poly, src_mult=None, extra_roots=None):
    """
    Constructs the atom list for a phi-step.
    Authoritative: Uses poly if present.
    Fallback: Uses src_mult/extra_roots if poly is None.

    y-sign rule: pt_src, pt_step, pt_res carry correct branches from upstream.
    Extra roots (R, S for generic phi-geometry) have unknown branch; we try
    all 2^k combinations (k = number of extra roots, typically 2) and return
    the first combination that passes _verify_atoms_principal.  If none pass,
    return [] so the candidate is dropped cleanly.
    """
    import itertools

    # 1. Authoritative Path: Polynomial roots
    if poly is not None:
        try:
            roots_wm = poly.roots(multiplicities=True)
            all_x_roots = []
            for r, m in roots_wm:
                all_x_roots.extend([r] * int(m))

            if len(all_x_roots) != 5:
                return []

            # Split roots into named (known branch) and extras (branch unknown).
            # pt_src must be consumed as many times as it appears in the poly
            # (its multiplicity); pt_step and pt_res are consumed once each.
            remaining_roots = list(all_x_roots)
            named_atoms = []

            # Consume pt_src at its full multiplicity first.
            for pt in (pt_src, pt_step, pt_res):
                if pt is None or not isinstance(pt, (tuple, list)) or len(pt) < 2:
                    continue
                x_pt = pt[0]
                if pt is pt_src:
                    # Consume every occurrence of pt_src[0] in remaining_roots.
                    while x_pt in remaining_roots:
                        remaining_roots.remove(x_pt)
                        named_atoms.append((x_pt, pt[1]))
                else:
                    if x_pt in remaining_roots:
                        remaining_roots.remove(x_pt)
                        named_atoms.append((x_pt, pt[1]))
            extra_xs = remaining_roots

            if not extra_xs:
                # All roots were named — no branch ambiguity.
                if len(named_atoms) != 5:
                    return []
                return named_atoms

            # Guard: total must still be 5.
            if len(named_atoms) + len(extra_xs) != 5:
                return []

            # Build both branch options for each extra root.
            p = int(walker.p)
            extra_branches = []
            for xr in extra_xs:
                y_can = walker._recover_y(xr)
                y_neg = walker.base_ring(p - int(y_can))
                extra_branches.append([(xr, y_can), (xr, y_neg)])

            # Try all 2^k branch combinations; return first that is principal.
            for branch_combo in itertools.product(*extra_branches):
                atoms = named_atoms + list(branch_combo)
                if walker._verify_atoms_principal(atoms):
                    return atoms

            return []
        except Exception:
            return []

    # 2. Fallback Path: Metadata-based
    if src_mult is not None:
        atoms = [pt_src] * src_mult
        atoms.append(pt_step)
        atoms.append(pt_res)
        for xr in (extra_roots or []):
            yr = walker._recover_y(xr)
            atoms.append((xr, yr))
        return atoms

    # 3. Legacy Path
    atoms = [pt_src, pt_step]
    if pt_res:
        atoms.append(pt_res)
    return atoms

def _select_and_validate_candidate(walker, candidates, search_out, pt_src, n):
    """
    Firewalled selection loop.
    Standardizes on (x, y) tuples and ensures NO SILENT FAILURES.
    """
    pool = [c for c in candidates if _is_well_formed_candidate(c)]

    if not pool:
        return None

    while pool:
        chosen = walker._choose_candidate_record(
            pool,
            {"n": n, "step": search_out, "current_pt": pt_src}
        )
        if not chosen:
            break

        try:
            pt_step = chosen["pt_step"]
            pt_res  = chosen["pt_res"]
            poly    = chosen.get("intersection_poly")

            # Degenerate Self-Loop Check
            if pt_step == pt_src and pt_res == pt_src:
                pool.remove(chosen)
                continue

            # Get Metadata (Needed for Fallback or Commit)
            src_mult, extra_roots = _get_geometry_metadata(walker, chosen, pt_src, pt_step, pt_res)

            # Build Atoms (Passes 6 arguments: walker + 5 parameters)
            atoms = _build_atoms(walker, pt_src, pt_step, pt_res, poly, src_mult, extra_roots)

            if not atoms:
                pool.remove(chosen)
                continue

            # Verify Principality
            if not walker._verify_atoms_principal(atoms):
                pool.remove(chosen)
                continue

            # Success: Store metadata for _commit_step
            chosen["_validated_atoms"] = atoms
            chosen["_src_mult"] = src_mult
            chosen["_extra_roots"] = extra_roots

            return chosen

        except (TypeError, KeyError, ValueError, AssertionError) as e:
            if chosen in pool:
                pool.remove(chosen)
            continue

    return None

def _get_geometry_metadata(walker, chosen, pt_src, pt_step, pt_res):
    """
    Extracts divisor metadata. Transitioned to (x, y) paradigm.
    Strictly purges primary points to prevent len(atoms) errors.
    """
    poly = chosen.get("intersection_poly")
    if poly is not None:
        try:
            roots_data = poly.roots(multiplicities=True)

            # 1. Identify source multiplicity
            src_mult = 0
            for x_val, m in roots_data:
                if x_val == pt_src[0]:
                    src_mult = int(m)
                    break

            # 2. Build full pool
            pool = []
            for x_val, m in roots_data:
                pool.extend([x_val] * int(m))

            # UPSTREAM ASSERT: Ensure the poly has exactly curve_degree F_p roots
            # (with multiplicity).  A degree-5 poly with R,S in F_p² will have
            # only 3 F_p roots here; those candidates are dropped cleanly.
            if len(pool) != walker.config.curve_degree:
                raise AssertionError(
                    f"[GEOM_UPSTREAM] Insufficient F_p roots: {len(pool)} != {walker.config.curve_degree} "
                    f"(R,S may be in F_p² extension). Roots: {roots_data}"
                )

            # 3. Purge the named primary points from the pool.
            # We purge pt_src (×src_mult) and pt_step (×1).
            # pt_res is NOT purged here: it is the enrichment-derived xk residual
            # and may not coincide with a phi-poly root; _build_atoms will match
            # it via known_y if it appears as a root, or it will stand as an extra.
            for _ in range(src_mult):
                if pt_src[0] in pool:
                    pool.remove(pt_src[0])

            if pt_step[0] in pool:
                pool.remove(pt_step[0])

            # UPSTREAM ASSERT: After removing src_mult + 1 points, remaining
            # count must equal curve_degree - src_mult - 1.
            expected_extra = walker.config.curve_degree - src_mult - 1
            if len(pool) != expected_extra:
                raise AssertionError(
                    f"[GEOM_UPSTREAM] Purge failed. Expected {expected_extra} extras, found {len(pool)}. "
                    f"Pool: {pool}, src_mult: {src_mult}"
                )

            return src_mult, pool

        except Exception as e:
            print(f"[CRITICAL] Geometry extraction failed: {e}")
            raise e

    # Fallback logic
    src_mult = chosen.get("src_mult") or (walker.config.curve_degree - 2)
    extra_roots = chosen.get("extra_roots") or []
    return int(src_mult), list(extra_roots)
