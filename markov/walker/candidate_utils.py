from collections import Counter
import dataclasses
from .fiber_geometry import *

def _collect_mumford_candidate_x_values(obj, out=None):
    """
    Recursively collect candidate x-values from a Mumford payload.
    """
    if out is None:
        out = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        for key in ("x_step", "x", "x_val", "xcoord", "candidate_x", "x_value"):
            if key in obj and obj[key] is not None:
                out.append(obj[key])

        if obj and all(not isinstance(v, dict) for v in obj.values()):
            for k, v in obj.items():
                if isinstance(v, (list, tuple, set)) and not isinstance(k, (list, tuple, set, dict)):
                    out.append(k)

        for value in obj.values():
            _collect_mumford_candidate_x_values(value, out)
        return out

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            _collect_mumford_candidate_x_values(value, out)
        return out

    return out

def _dedupe_preserve_order(values):
    seen = set()
    out_vals = []
    for v in values:
        if v is None:
            continue
        try:
            key = v if hash(v) is not None else repr(v)
        except Exception:
            key = repr(v)
            raise
        if key in seen:
            continue
        seen.add(key)
        out_vals.append(v)
    return out_vals

def _candidate_x_from_obj(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for key in ("x_step", "x", "x_val", "xcoord", "candidate_x", "x_value"):
            if key in obj and obj[key] is not None:
                return obj[key]
    return obj if not isinstance(obj, (dict, list, tuple, set)) else None

def _candidate_record_from_x(x, source="mumford_residue", **extra):
    rec = {"x_step": x, "source": source}
    rec.update(extra)
    return rec

def _candidates_from_residues(residues, p):
    """Extract candidate records from mumford_residues {p: {vtup: {rhs_idx: [m_root, ...]}}}.

    Julia now returns only m_root values — no Mumford pairs, no sign computation.
    enrich_candidates reconstructs x_step/x_res and signs from the fiber geometry.

    Returns a list of dicts with keys: x_step, yj_sign, m, rhs_idx, source.
    """
    records = []
    seen = set()  # (m_root, rhs_idx) dedup

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
                dedup_key = (m_root, rhs_idx)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                records.append({
                    "x_step":      None,   # reconstructed by enrich_candidates from m
                    "yj_sign": 1,      # enrich_candidates computes true sign from fiber
                    "m":       m_root,
                    "rhs_idx": rhs_idx,
                    "source":  "mumford_residue",
                })

    return records

def _normalize_candidate_output(result):
    """
    Normalize any search result into the walker-friendly dict shape.
    """
    if result is None:
        return {
            "candidates": [],
            "candidate_records": [],
            "candidate_xs": set(),
            "new_sections": [],
            "precomputed_residues": None,
            "stats": None,
        }

    if isinstance(result, dict):
        out = dict(result)
        out.setdefault("candidates", [])
        out.setdefault("candidate_records", out.get("candidates", []))
        out.setdefault("candidate_xs", set())
        out.setdefault("new_sections", [])
        out.setdefault("precomputed_residues", None)
        out.setdefault("stats", None)

        if not out.get("candidate_records") and out.get("candidates"):
            out["candidate_records"] = list(out["candidates"])

        if not out.get("candidate_xs"):
            xs = set()
            for cand in out.get("candidate_records", []):
                if isinstance(cand, dict):
                    x = cand.get("x_step", None)
                    if x is None:
                        x = cand.get("x", None)
                    if x is None:
                        x = cand.get("candidate_x", None)
                    if x is None:
                        x = cand.get("x_value", None)
                    if x is not None:
                        xs.add(x)
                else:
                    if cand is not None:
                        xs.add(cand)
            out["candidate_xs"] = xs

        return out

    if isinstance(result, (tuple, list)) and len(result) == 4:
        a, b, c, d = result
        if isinstance(a, list) and a and isinstance(a[0], dict):
            xs = {cand.get("x_step") for cand in a if cand.get("x_step") is not None}
            return {
                "candidates": a,
                "candidate_records": a,
                "candidate_xs": xs,
                "new_sections": b,
                "precomputed_residues": c,
                "stats": d,
            }
        if isinstance(a, (set, list, tuple)):
            records = [{"x_step": x} for x in a]
            return {
                "candidates": records,
                "candidate_records": records,
                "candidate_xs": set(a),
                "new_sections": b,
                "precomputed_residues": c,
                "stats": d,
            }

    raise TypeError(f"Unsupported search result type: {type(result)!r}")



def _normalize_markov_mumford_result(result, fallback_step=None):
    """
    Normalize the legacy Mumford-search return payload into a walker-friendly dict.
    """
    out = {
        "candidates": [],
        "candidate_records": [],
        "candidate_xs": set(),
        "new_sections": [],
        "precomputed_residues": None,
        "residues": None,          # markov_mode fast-exit: {p: {vtup: {x_val: [(sol, yj_sign, v0, v1)]}}}
        "stats": None,
        "raw_mumford_residues": None,
        "found_xs": set(),
    }

    if result is None:
        return out

    if isinstance(result, dict):
        out["raw_mumford_residues"] = result.get("raw_mumford_residues", result)
        out["precomputed_residues"] = result.get("precomputed_residues", None)
        out["residues"] = result.get("residues", None)   # signed residues from markov fast-exit
        out["stats"] = result.get("stats", None)
        out["new_sections"] = result.get("new_sections", [])

        for key in (
            "input_n", "vecs", "tower_context", "current_x", "current_y",
            "x_src", "yi", "shift", "r_expr", "n_with_roots", "per_n_roots",
        ):
            if key in result:
                out[key] = result[key]

        if "found_xs" in result:
            out["found_xs"] = _as_set(result.get("found_xs"))
        if "candidate_xs" in result:
            out["candidate_xs"] = _as_set(result.get("candidate_xs"))

        raw_candidates = result.get("candidate_records", None)
        if raw_candidates is None:
            raw_candidates = result.get("candidates", None)

        if raw_candidates is not None:
            if isinstance(raw_candidates, (list, tuple)):
                out["candidate_records"] = list(raw_candidates)
            else:
                out["candidate_records"] = [raw_candidates]

        for cand in out["candidate_records"]:
            x = _candidate_x_from_obj(cand)
            if x is not None:
                out["candidate_xs"].add(x)
                out["found_xs"].add(x)

        if not out["candidate_xs"]:
            xs = _collect_mumford_candidate_x_values(out["raw_mumford_residues"], [])
            xs = _dedupe_preserve_order(xs)
            if xs:
                out["candidate_xs"] = set(xs)
                out["candidate_records"] = [_candidate_record_from_x(x) for x in xs]

        if not out["candidate_xs"] and fallback_step is not None:
            xs = _collect_mumford_candidate_x_values(fallback_step, [])
            xs = _dedupe_preserve_order(xs)
            if xs:
                out["candidate_xs"] = set(xs)
                out["candidate_records"] = [_candidate_record_from_x(x, source="fallback_step") for x in xs]

        if not out["candidate_records"] and out["candidate_xs"]:
            out["candidate_records"] = [_candidate_record_from_x(x) for x in out["candidate_xs"]]

        out["candidates"] = list(out["candidate_records"])

        try:
            out["candidate_counts"] = Counter(
                cand.get("x_step")
                for cand in out["candidate_records"]
                if isinstance(cand, dict) and cand.get("x_step") is not None
            )
        except Exception:
            out["candidate_counts"] = Counter()
            raise

        return out

    if isinstance(result, (tuple, list)):
        items = list(result)
        out["raw_mumford_residues"] = items

        xs = []
        found_xs = set()

        for item in items:
            if isinstance(item, (list, tuple, set)):
                for v in item:
                    if v is not None:
                        found_xs.add(v)
            xs.extend(_collect_mumford_candidate_x_values(item, []))

        xs = _dedupe_preserve_order(xs)

        if not xs and found_xs:
            xs = _dedupe_preserve_order(list(found_xs))

        out["found_xs"] = set(found_xs) if found_xs else set(xs)
        out["candidate_xs"] = set(xs)
        out["candidate_records"] = [{"x_step": x, "source": "mumford_residue"} for x in xs]
        out["candidates"] = list(out["candidate_records"])

        for item in reversed(items):
            if isinstance(item, dict):
                if out["stats"] is None and "stats" in item:
                    out["stats"] = item["stats"]
                if out["precomputed_residues"] is None and "precomputed_residues" in item:
                    out["precomputed_residues"] = item["precomputed_residues"]
                if not out["new_sections"] and "new_sections" in item:
                    out["new_sections"] = item["new_sections"]

        try:
            out["candidate_counts"] = Counter(
                cand.get("x_step")
                for cand in out["candidate_records"]
                if isinstance(cand, dict) and cand.get("x_step") is not None
            )
        except Exception:
            out["candidate_counts"] = Counter()
            raise

        return out

    xs = _collect_mumford_candidate_x_values(result, [])
    xs = _dedupe_preserve_order(xs)
    out["raw_mumford_residues"] = result
    out["candidate_xs"] = set(xs)
    out["found_xs"] = set(xs)
    out["candidate_records"] = [{"x_step": x, "source": "scalar_fallback"} for x in xs]
    out["candidates"] = list(out["candidate_records"])

    try:
        out["candidate_counts"] = Counter(xs)
    except Exception:
        out["candidate_counts"] = Counter()
        raise

    return out


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
    """Return the S(m) symbolic rational function for the x_src of *rec*.

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
    """Return (fi, G_poly) for the x_src of *rec*, or (None, None) if unavailable.

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



def _intersection_poly_from_step(step: Dict[str, Any], *, x_step=None, x_res=None):
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
    if x_step is not None or x_res is not None:
        for cand in pools:
            if not isinstance(cand, dict):
                continue
            cand_xj = cand.get("x_step")
            cand_xk = cand.get("x_res")
            if x_step is not None and cand_xj == x_step:
                poly = _cand_poly(cand)
                if poly is not None:
                    return poly
            if x_res is not None and cand_xk == x_res:
                poly = _cand_poly(cand)
                if poly is not None:
                    return poly

    # 2b) any candidate with a poly
    for cand in pools:
        poly = _cand_poly(cand)
        if poly is not None:
            return poly

    return None


def _derive_relation_from_intersection_poly(step: Dict[str, Any], x_src):
    """
    Return (x_step, x_res, src_mult, poly) derived only from the intersection polynomial.

    This is the only place x_step/x_res/src_mult should be trusted from.
    """
    poly = _intersection_poly_from_step(step)
    #poly = self._intersection_poly_from_step(poly_src, x_step=chosen.get("x_step"), x_res=chosen.get("x_res"))
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
        if r == x_src:
            src_mult += int(m)
        else:
            leftovers.extend([r] * int(m))

    if src_mult <= 0:
        return None

    if not leftovers:
        return None  # All roots are x_src; no usable relation.

    # Dispatch on the number of non-x_src roots.  No multiplicity pattern is
    # assumed in advance — the actual root list drives the relation.
    if len(leftovers) == 1:
        # Tangency: one non-x_src root.  Fold one copy of x_src into the x_res slot
        # so that x_res==x_src and src_mult is decremented by one.  The relation
        # matrix adds +1 to the x_src column for x_res, giving the right total.
        x_step = leftovers[0]
        x_res = x_src
        src_mult -= 1
        extra_roots = []
    elif len(leftovers) == 2:
        x_step, x_res = leftovers[0], leftovers[1]
        extra_roots = []
    else:
        # General case: 3+ non-x_src roots (x_src has lower-than-expected multiplicity).
        # x_step/x_res carry the first two; extra_roots carries the remainder.
        # Each extra root contributes +1 in the relation matrix, same as x_step/x_res.
        x_step = leftovers[0]
        x_res = leftovers[1]
        extra_roots = leftovers[2:]

    return x_step, x_res, src_mult, poly, extra_roots


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


def _candidate_xj_from_m(base_ring, x_src, m_val):
    return base_ring(x_src) - base_ring(m_val)


def _score_candidate_record(score_fn, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
    if score_fn is None:
        return 0.0
    x_step = candidate.get("x_step")
    # Raises on failure instead of silently returning 0.0
    return float(score_fn(x_step, context | {"candidate": candidate}))

def _score_candidate(score_fn, candidate_x, context: Dict[str, Any]) -> float:
    if score_fn is None:
        assert None, None
        return 0.0
    # Raises on failure instead of silently returning 0.0
    return float(score_fn(candidate_x, context))

