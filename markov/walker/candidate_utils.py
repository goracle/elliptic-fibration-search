import dataclasses
from collections import Counter
from .fiber_geometry import *
from sage.all import Integer

def safe_solve_univariate_roots(poly, ring=None) -> List[Any]:
    """Solve poly=0 in its base ring, returning roots if Sage can see them."""
    roots = poly.roots(multiplicities=False)
    assert roots, roots
    return list(roots)

def _as_set(values):
    if values is None:
        return set()
    if isinstance(values, set):
        return set(values)
    if isinstance(values, (list, tuple)):
        return {v for v in values if v is not None}
    return {values}

def _solve_m_roots(step: Dict[str, Any]) -> List[Any]:
    r_expr = step.get("r_expr")
    if r_expr is None:
        return []
    try:
        poly = r_expr if hasattr(r_expr, "roots") else SR(r_expr)
    except Exception:
        poly = r_expr
        raise
        return []
    try:
        return safe_solve_univariate_roots(poly)
    except Exception:
        raise
        return []

def _get_S_of_m_for_rec(rec) -> Optional[Any]:
    """Return the S(m) symbolic rational function for the pt_src of *rec*.

    Priority order (mirrors _emit_step_diagnostics):
    1. rec.step['S_of_m']  – stored by the search path on accepted steps
    2. first candidate_pool entry that carries S_of_m
    Returns None if unavailable (injection is silently skipped for that step).
    """
    step = rec.step if isinstance(rec.step, dict) else {}
    S_sym = step.get('S_of_m')
    if S_sym is not None:
        return S_sym
    for cand in (rec.candidate_pool or []):
        if isinstance(cand, dict):
            S_sym = cand.get('S_of_m')
            if S_sym is not None:
                return S_sym
    return None

def _get_fiber_context_for_rec(rec):
    """Return (fi, G_poly) for the pt_src of *rec*, or (None, None) if unavailable.

    fi is the symbolic fiber poly in x over Frac(Fp[m]).
    G_poly is the curve poly in x over Fp.
    Both are stored on the step payload by make_project_markov_search_fn.
    """
    step = rec.step if isinstance(rec.step, dict) else {}
    fi = step.get('fi')
    G_poly = step.get('G_poly')
    if fi is not None and G_poly is not None:
        return fi, G_poly
    # Fall back to candidate pool entries.
    for cand in (rec.candidate_pool or []):
        if isinstance(cand, dict):
            fi = fi or cand.get('fi')
            G_poly = G_poly or cand.get('G_poly')
            if fi is not None and G_poly is not None:
                return fi, G_poly
    return None, None

def _intersection_poly_from_step(step: Dict[str, Any], *, pt_step=None, pt_res=None):
    """Best-effort access to a degree-5 intersection polynomial.

    Priority:
    1. top-level step payload
    2. matching candidate record
    3. any candidate record with a poly
    """
    if not isinstance(step, dict):
        return None

    poly_keys = ("intersection_poly", "fiber_poly", "intersection", "poly_x")

    # 1) top-level payload first
    for key in poly_keys:
        poly = step.get(key)
        if poly is not None:
            return poly

    # 2) search candidate records / pool
    pools = []
    for key in ("candidate_records", "candidates", "candidate_pool"):
        pool = step.get(key)
        if pool:
            pools.extend(pool)

    def _cand_poly(cand):
        if not isinstance(cand, dict):
            return None
        for key in poly_keys:
            poly = cand.get(key)
            if poly is not None:
                return poly
        return None

    # 2a) exact-ish match first
    if pt_step is not None or pt_res is not None:
        for cand in pools:
            if not isinstance(cand, dict):
                continue
            cand_xj = cand.get("pt_step")
            cand_xk = cand.get("pt_res")
            if pt_step is not None and cand_xj == pt_step:
                poly = _cand_poly(cand)
                if poly is not None:
                    return poly
            if pt_res is not None and cand_xk == pt_res:
                poly = _cand_poly(cand)
                if poly is not None:
                    return poly

    # 2b) any candidate with a poly
    for cand in pools:
        poly = _cand_poly(cand)
        if poly is not None:
            return poly

    return None

def _derive_relation_from_intersection_poly(step: Dict[str, Any], pt_src):
    """
    Return (pt_step, pt_res, src_mult, poly) derived only from the intersection polynomial.

    This is the only place pt_step/pt_res/src_mult should be trusted from.
    """
    poly = _intersection_poly_from_step(step)
    #poly = self._intersection_poly_from_step(poly_src, pt_step=chosen.get("pt_step"), pt_res=chosen.get("pt_res"))
    if poly is None:
        #assert None, "poly is missing, gang!"
        return None

    # Handle shifted tower coordinates, if present.
    shift = step.get("shift") if isinstance(step, dict) else None
    if shift is not None:
        try:
            shift_int = int(shift)
        except Exception:
            shift_int = 0
            raise
        if shift_int != 0:
            x_var = poly.parent().gen()
            poly = poly(x_var + shift_int)

    try:
        roots_wm = poly_roots_with_multiplicity(poly)  # [(root, mult), ...]
    except Exception:
        raise
        return None

    src_mult = 0
    leftovers = []
    for r, m in roots_wm:
        if r == pt_src:
            src_mult += int(m)
        else:
            leftovers.extend([r] * int(m))

    if src_mult <= 0:
        return None

    if not leftovers:
        return None  # All roots are pt_src; no usable relation.

    # Dispatch on the number of non-pt_src roots.  No multiplicity pattern is
    # assumed in advance — the actual root list drives the relation.
    if len(leftovers) == 1:
        # Tangency: one non-pt_src root.  Fold one copy of pt_src into the pt_res slot
        # so that pt_res==pt_src and src_mult is decremented by one.  The relation
        # matrix adds +1 to the pt_src column for pt_res, giving the right total.
        pt_step = leftovers[0]
        pt_res = pt_src
        src_mult -= 1
        extra_roots = []
    elif len(leftovers) == 2:
        pt_step, pt_res = leftovers[0], leftovers[1]
        extra_roots = []
    else:
        # General case: 3+ non-pt_src roots (pt_src has lower-than-expected multiplicity).
        # pt_step/pt_res carry the first two; extra_roots carries the remainder.
        # Each extra root contributes +1 in the relation matrix, same as pt_step/pt_res.
        pt_step = leftovers[0]
        pt_res = leftovers[1]
        extra_roots = leftovers[2:]

    return pt_step, pt_res, src_mult, poly, extra_roots

def _jsonable(obj: Any):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, complex):
        return str(obj)
    if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        try:
            return _jsonable(obj.item())
        except Exception:
            raise
            return str(obj)
    if dataclasses.is_dataclass(obj):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)

def _candidate_xj_from_m(base_ring, pt_src, m_val):
    return base_ring(pt_src) - base_ring(m_val)

def _score_candidate_record(score_fn, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
    if score_fn is None:
        return 0.0
    pt_step = candidate.get("pt_step")
    # Raises on failure instead of silently returning 0.0
    return float(score_fn(pt_step, context | {"candidate": candidate}))

def _score_candidate(score_fn, candidate_x, context: Dict[str, Any]) -> float:
    if score_fn is None:
        assert None, None
        return 0.0
    # Raises on failure instead of silently returning 0.0
    return float(score_fn(candidate_x, context))

def _collect_mumford_candidate_pts(obj, out=None):
    """
    Recursively collect candidate (x, y) points or atoms from a Mumford payload.
    """
    if out is None:
        out = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        # Check for explicit point/atom keys first
        for key in ("pt_step", "atom_record", "point", "candidate_pt", "pt_src"):
            if key in obj and obj[key] is not None:
                out.append(obj[key])

        # Recurse into children
        for value in obj.values():
            _collect_mumford_candidate_pts(value, out)
        return out

    if isinstance(obj, (list, tuple, set)):
        # Treat 2-tuples as points if they aren't containers themselves
        if len(obj) == 2 and not isinstance(obj[0], (dict, list, tuple)):
             out.append(tuple(obj))
        else:
            for value in obj:
                _collect_mumford_candidate_pts(value, out)
        return out

    return out

def _dedupe_preserve_order(values):
    seen = set()
    out_vals = []
    for v in values:
        if v is None:
            continue
        try:
            # Handle non-hashable coordinates via repr
            key = v if hash(v) is not None else repr(v)
        except Exception:
            key = repr(v)
        if key in seen:
            continue
        seen.add(key)
        out_vals.append(v)
    return out_vals

def _candidate_pt_from_obj(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        # Look for pt_step first, then fall back to x for reconstruction
        for key in ("pt_step", "pt_src", "x", "candidate_x"):
            if key in obj and obj[key] is not None:
                return obj[key]
    return obj if not isinstance(obj, (dict, list, tuple, set)) else None

def _candidate_record_from_pt(pt, source="mumford_residue", **extra):
    """The central record generator. Use this instead of _candidate_record_from_x."""
    rec = {"pt_step": pt, "source": source}
    rec.update(extra)
    return rec

def candidates_from_residues(residues, p):
    """Extract candidate records from residues. Signs reconstructed later."""
    records = []
    seen = set()
    pmap = residues.get(p, {})
    for vtup, rhs_map in pmap.items():
        if not isinstance(rhs_map, dict):
            continue
        for rhs_idx, m_roots in rhs_map.items():
            if not m_roots:
                continue
            rhs_idx = int(rhs_idx)
            for m_root in m_roots:
                m_root = int(m_root)
                if (m_root, rhs_idx) in seen:
                    continue
                seen.add((m_root, rhs_idx))
                records.append({
                    "pt_step": None,  # Populated by enrich_candidates
                    "m": m_root,
                    "rhs_idx": rhs_idx,
                    "source": "mumford_residue",
                })
    return records

def normalize_markov_mumford_result(result, fallback_step=None):
    """
    Normalize the legacy Mumford-search return payload into a walker-friendly dict.
    """
    # 1. Initialize the standardized state
    out = _initialize_empty_normalization_payload()

    if result is None:
        return out

    # 2. Extract raw data based on the input type
    if isinstance(result, dict):
        out = _extract_from_dict_result(result, out)
    elif isinstance(result, (tuple, list)):
        out = _extract_from_sequence_result(result, out)
    else:
        out["raw_mumford_residues"] = result

    # 3. Finalize candidates: if records are empty, try to scrape them from residues
    if not out["candidate_pt"]:
        out = _populate_candidates_from_scraping(out, fallback_step)

    # 4. Final assembly: ensure candidate list and counts are in sync
    out["candidates"] = list(out["candidate_records"])
    out["candidate_counts"] = _calculate_candidate_counts(out["candidate_records"])

    return out

def _initialize_empty_normalization_payload():
    return {
        "candidates": [],
        "candidate_records": [],
        "candidate_pt": set(),
        "new_sections": [],
        "precomputed_residues": None,
        "residues": None,
        "stats": None,
        "raw_mumford_residues": None,
        "found_pt": set(),
    }

def _extract_from_dict_result(result, out):
    """Handles dictionary-style payloads."""
    out["raw_mumford_residues"] = result.get("raw_mumford_residues", result)
    out["precomputed_residues"] = result.get("precomputed_residues")
    out["residues"] = result.get("residues")
    out["stats"] = result.get("stats")
    out["new_sections"] = result.get("new_sections", [])

    # Map core search keys
    for key in ("input_n", "vecs", "tower_context", "current_pt",
                "pt_src", "shift", "r_expr", "n_with_roots", "per_n_roots"):
        if key in result:
            out[key] = result[key]

    # Handle explicit point sets
    if "found_pt" in result:
        out["found_pt"] = _as_set(result.get("found_pt"))
    if "candidate_pt" in result:
        out["candidate_pt"] = _as_set(result.get("candidate_pt"))

    # Process existing records
    raw_cands = result.get("candidate_records") or result.get("candidates")
    if raw_cands:
        out["candidate_records"] = list(raw_cands) if isinstance(raw_cands, (list, tuple)) else [raw_cands]
        for cand in out["candidate_records"]:
            pt = _candidate_pt_from_obj(cand) # Fixed to use PT version
            if pt is not None:
                out["candidate_pt"].add(pt)
                out["found_pt"].add(pt)
    return out

def _calculate_candidate_counts(records):
    """Safe calculation of candidate frequencies."""
    try:
        return Counter(
            cand.get("pt_step")
            for cand in records
            if isinstance(cand, dict) and cand.get("pt_step") is not None
        )
    except Exception:
        return Counter()

def _validate_is_point(obj, source_context=""):
    """
    Enforces 'NO SILENT FAILURES'.
    Ensures obj is a subscriptable point (x, y), not a bare x.
    """
    if isinstance(obj, (int, Integer, float)):
        raise TypeError(
            f"Upstream Failure: {source_context} provided a scalar '{type(obj).__name__}' "
            f"instead of a point tuple (x, y). Value: {obj}"
        )
    if obj is not None and not isinstance(obj, (tuple, list)):
         raise TypeError(f"Point must be tuple or list, got {type(obj).__name__}: {obj}")
    return obj

def normalize_candidate_output(result):
    """
    Normalize any search result into the walker-friendly dict shape.
    Strictly enforces that points are coordinates, not scalars.
    """
    # 1. Handle Null Result
    if result is None:
        return {
            "candidates": [],
            "candidate_records": [],
            "candidate_pt": set(),
            "new_sections": [],
            "precomputed_residues": None,
            "stats": None,
        }

    # 2. Handle Dictionary Result (Most common from phi/scraping)
    if isinstance(result, dict):
        out = dict(result)
        out.setdefault("candidates", [])
        out.setdefault("candidate_records", list(out.get("candidates", [])))
        out.setdefault("new_sections", [])
        out.setdefault("precomputed_residues", None)
        out.setdefault("stats", None)

        # Build candidate_pt set while enforcing point integrity
        found_pts = set()
        for cand in out["candidate_records"]:
            if isinstance(cand, dict):
                # Check keys in order of precedence
                pt = next((cand.get(k) for k in ["pt_step", "x", "candidate_x", "x_value"]
                          if cand.get(k) is not None), None)
                if pt is not None:
                    found_pts.add(_validate_is_point(pt, f"Record Source: {cand.get('source', 'unknown')}"))
            else:
                # If the record itself is just a value, it better be a tuple
                found_pts.add(_validate_is_point(cand, "Raw Candidate List"))

        out["candidate_pt"] = found_pts
        return out

    # 3. Handle Legacy 4-tuple Result (a, b, c, d)
    if isinstance(result, (tuple, list)) and len(result) == 4:
        cands, news, resids, stats = result

        # Case A: List of dictionaries
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            pts = {
                _validate_is_point(c.get("pt_step"), "4-tuple dict list")
                for c in cands if c.get("pt_step") is not None
            }
            return {
                "candidates": cands,
                "candidate_records": cands,
                "candidate_pt": pts,
                "new_sections": news,
                "precomputed_residues": resids,
                "stats": stats,
            }

        # Case B: List of raw objects (must be point tuples)
        if isinstance(cands, (set, list, tuple)):
            processed_pts = { _validate_is_point(p, "4-tuple raw list") for p in cands }
            records = [{"pt_step": p} for p in processed_pts]
            return {
                "candidates": records,
                "candidate_records": records,
                "candidate_pt": processed_pts,
                "new_sections": news,
                "precomputed_residues": resids,
                "stats": stats,
            }

    raise TypeError(f"Unsupported search result type: {type(result)!r}")

def _populate_candidates_from_scraping(out, fallback_step):
    """
    Scrapes raw residues if no explicit records were provided.
    Strictly enforces that scraped 'pts' are coordinate tuples (x, y).
    """
    # 1. Attempt to collect points from primary residue source
    pts = _collect_mumford_candidate_pts(out.get("raw_mumford_residues", []), [])
    source = "mumford_residue"

    # 2. Fallback logic
    if not pts and fallback_step is not None:
        pts = _collect_mumford_candidate_pts(fallback_step, [])
        source = "fallback_step"

    if pts:
        pts = _dedupe_preserve_order(pts)

        # Validate every point before committing to 'out'
        # This prevents Integers from leaking into pt_step
        validated_pts = []
        for p in pts:
            # Re-using the validation logic to ensure subscriptability
            _validate_is_point(p, source_context=f"Scraper Source: {source}")
            validated_pts.append(p)

        out["candidate_pt"] = set(validated_pts)
        out["candidate_records"] = [
            _candidate_record_from_pt(p, source=source)
            for p in validated_pts
        ]

    # 3. Final check: Synchronize records if points exist but records don't
    elif out.get("candidate_pt") and not out.get("candidate_records"):
        # Validate existing points in set for good measure
        pts_list = list(out["candidate_pt"])
        for p in pts_list:
            _validate_is_point(p, source_context="Pre-existing candidate_pt set")

        out["candidate_records"] = [
            _candidate_record_from_pt(p, source="unknown_recovery")
            for p in pts_list
        ]

    return out
