from .candidate_utils import *
from .curve_helpers import *
from typing import Optional, List, Dict, Any, Tuple

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

def _get_geometry_metadata(walker, chosen, pt_src, pt_step, pt_res):
    """
    Extracts divisor metadata, prioritizing the intersection polynomial.
    """
    poly = chosen.get("intersection_poly")
    if poly is not None:
        try:
            roots = poly.roots(multiplicities=True)
            src_mult = next((m for r, m in roots if r == pt_src[0]), 0)

            # Collect extra roots by removing one instance of pt_step and pt_res
            others = []
            for r, m in roots: others.extend([r] * int(m))

            for expected in (pt_step[0], pt_res[0]):
                if expected in others: others.remove(expected)

            return int(src_mult), others
        except Exception:
            pass # Fall back to record metadata

    src_mult = chosen.get("src_mult") or (walker.config.curve_degree - 2)
    extra_roots = chosen.get("extra_roots") or []
    return int(src_mult), list(extra_roots)

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
    rec = walker._make_relation(
        len(walker.history), n, pt_src[0], chosen.get("m"),
        chosen["pt_step"], chosen["pt_res"],
        search_out, accepted=True,
        src_mult=chosen["_src_mult"],
        extra_roots=chosen["_extra_roots"]
    )

    walker._store_record(rec)
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
    # Ensure comparison is tuple-to-tuple
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

    _, new_this_step, collisions = walker._update_leaf_bookkeeping(candidate_pts, n=n, xi_before=pt_src[0])

    search_out.update({
        "step_leaves_found": len(candidate_pts),
        "step_leaves_new": new_this_step,
        "step_leaf_collisions": collisions,
    })

def _select_and_validate_candidate(walker, candidates, search_out, pt_src, n):
    """
    Firewalled selection loop.
    Standardizes on (x, y) tuples and ensures NO SILENT FAILURES.
    """
    # 1. Filter out malformed records immediately (Firewall)
    # This removes the 'pt_step: None' records seen in your logs.
    pool = [c for c in candidates if _is_well_formed_candidate(c)]

    if not pool:
        if walker.config.verbose:
            print(f"  [warn] Pool empty after well-formed check. Raw count: {len(candidates)}")
        return None

    while pool:
        # 2. Select a candidate using the walker's selection strategy
        # FIX: Standardized on 'pt_src' to avoid NameError: 'pt'
        chosen = walker._choose_candidate_record(
            pool,
            {"n": n, "step": search_out, "current_pt": pt_src}
        )

        if not chosen:
            break

        try:
            # 3. Extract Points (Guaranteed to be tuples by the firewall)
            pt_step = chosen["pt_step"]
            pt_res  = chosen["pt_res"]

            # 4. Geometry Check: Degenerate Self-Loops
            # In a phi-step, if both roots collapse back to the source, no move is possible.
            if pt_step == pt_src and pt_res == pt_src:
                if walker.config.verbose:
                    print(f"  [cand_skip] degenerate self-loop at {pt_src}")
                pool.remove(chosen)
                continue

            # 5. Extract Multiplicity & Extra Roots (Divisor Metadata)
            # Prioritizes the intersection polynomial; falls back to record metadata.
            src_mult, extra_roots = _get_geometry_metadata(walker, chosen, pt_src, pt_step, pt_res)

            # 6. Branch Verification (Principality)
            # Ensure the divisor (src_mult)P + Q + R + ... is principal.
            atoms = _build_atoms(walker, pt_src, pt_step, pt_res, src_mult, extra_roots)

            if not walker._verify_atoms_principal(atoms):
                if walker.config.verbose:
                    print(f"  [verify_fail] non-principal relation for {pt_step}")
                pool.remove(chosen)
                continue

            # 7. Package Validated Metadata
            # Store calculated values back on the record to prevent re-calculation in commit
            chosen["_validated_atoms"] = atoms
            chosen["_src_mult"] = src_mult
            chosen["_extra_roots"] = extra_roots

            return chosen

        except (TypeError, KeyError, ValueError) as e:
            # NO SILENT FAILURES: Trace specific geometry or data corruption issues
            print(f"  [geom_exception] {type(e).__name__}: {e}")
            if chosen in pool:
                pool.remove(chosen)
            continue

    return None

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

def _build_atoms(walker, pt_src, pt_step, pt_res, poly):
    """
    Constructs the atom list for a phi-step.
    Ensures exactly 5 atoms by treating the polynomial as authoritative.
    """
    # 1. If we have a polynomial, it DEFINES the atoms.
    # We do not manually add pt_step or pt_res because they are roots of this poly.
    if poly is not None:
        try:
            roots_wm = poly.roots(multiplicities=True)
            all_x_roots = []
            for r, m in roots_wm:
                all_x_roots.extend([r] * int(m))

            # A phi-step MUST have 5 roots (2P + Q + R + S)
            if len(all_x_roots) != 5:
                return []

            atoms = []
            # Recover y-branches for each root.
            # In phi-steps, y is determined by the intersection y = A(x).
            for xr in all_x_roots:
                # Use the walker's branch recovery
                yr = walker._recover_y(xr)
                atoms.append((xr, yr))

            # This returns EXACTLY 5 atoms.
            return atoms

        except Exception as e:
            print(f"  [build_fail] Poly factoring error: {e}")
            return []

    # 2. Fallback for legacy/generic steps without a polynomial
    # This remains 3 or 4 atoms depending on the move type.
    atoms = [pt_src, pt_step]
    if pt_res:
        atoms.append(pt_res)
    return atoms
