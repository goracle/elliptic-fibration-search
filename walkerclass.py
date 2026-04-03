from __future__ import annotations
import argparse, json, dataclasses, math, random, itertools, bounds, warnings, sys, inspect
from dataclasses import dataclass, field
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from sage.all import *
from search_common import *
from collections import Counter
from sympy import symbols, expand
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from tate import *
from search_lll import *
from sat import *
from picard import *
from automorph import *
from torsion import *
from nslattice import *
from yau import *
from bounds import *
from selmer import *
from stats import *
from consensus import *
from mobius import *
from sage.misc.verbose import set_verbose
from relation_matrix import *
from functools import partial

def build_project_tower_context_for_point(
    xi,
    yi=None,
    *,
    coeffs_genus2=None,
    base_points=None,
    p: Optional[int] = None,
    debug: bool = False,
):
    """Rebuild the tower / fibration context for a single current point.

    This mirrors the search7_genus2.doloop_genus2 setup path but keeps only the
    ingredients needed by the Markov candidate-search branch.
    """
    # Tower construction requires xi to be an F_p point (no field extensions supported).
    # This is satisfied because xi comes from a prior m-root search over F_p, guaranteeing
    # it is a point on C(F_p).
    setup_field_and_rings = resolve_project_symbol('setup_field_and_rings', required=True)
    apply_shift_transformation = resolve_project_symbol('apply_shift_transformation', required=True)
    apply_mobius_transformation = resolve_project_symbol('apply_mobius_transformation', required=True)
    build_tower_and_fibrations = resolve_project_symbol('build_tower_and_fibrations', required=True)
    extract_geometry_from_tower = resolve_project_symbol('extract_geometry_from_tower', required=True)
    build_curve_data = resolve_project_symbol('build_curve_data', required=True)
    configure_search_parameters = resolve_project_symbol('configure_search_parameters', required=True)
    build_search_rhs_list = resolve_project_symbol('build_search_rhs_list', required=True)
    setup_rationality_test_function = resolve_project_symbol('setup_rationality_test_function', required=True)
    compute_base_sections_m = resolve_project_symbol('compute_base_sections_m', required=True)
    lll_reduce_mw_basis = resolve_project_symbol('lll_reduce_mw_basis', required=True)

    coeffs_genus2 = coeffs_genus2 if coeffs_genus2 is not None else COEFFS_GENUS2
    print("building tower search for point:", (xi, yi))

    #base_points = list(base_points or _project_base_points_from_globals(xi, yi, p=p))
    base_points = [(xi, yi)]
    if yi is None:
        yfun = resolve_project_symbol('get_y_unshifted_genus2', default=None)
        if yfun is not None:
            try:
                yi = yfun(xi)
            except Exception:
                yi = None
                raise

    if yi is None:
        raise ValueError(f"Could not recover y-value for xi={xi!r}; please supply base_points or yi.")

    data_pts = [(xi, yi)]
    for pt in base_points:
        if pt is None:
            continue
        if len(pt) >= 2 and pt[0] is not None and pt[1] is not None:
            data_pts.append((pt[0], pt[1]))

    # Deduplicate while preserving order
    seen = set()
    uniq_data_pts = []
    for pt in data_pts:
        key = (str(pt[0]), str(pt[1]))
        if key in seen:
            continue
        seen.add(key)
        uniq_data_pts.append(pt)

    field_data = setup_field_and_rings(coeffs_genus2, uniq_data_pts)
    shifted_G_poly, base_pts, shift = apply_shift_transformation(
        field_data['G'], field_data['real_pts'], field_data['base_field']
    )
    assert len(base_pts) == 1, base_pts
    shifted_G_poly, base_pts, T, T_inv, _all_known_x = apply_mobius_transformation(
        shifted_G_poly, {xi}, base_pts
    )

    #print("base_pts, non-legacy", base_pts)
    primary_tower, fibrations, tower_for_mumford = build_tower_and_fibrations(
        shifted_G_poly, base_pts
    )
    #print("primary_tower, fibrations, tower_for_mumford")
    #print(primary_tower, fibrations, tower_for_mumford)

    E_rhs_m, r_m, roots = extract_geometry_from_tower(primary_tower, field_data['Fm'])

    cd, morphism_data = build_curve_data(E_rhs_m, roots, base_pts)
    one, two, three = morphism_data

    if False:
        print("E_rhs_m, r_m, roots")
        print(E_rhs_m, r_m, roots)
        print("cd, morphism_data")
        print(cd, morphism_data)
        sys.exit()

    sconf, prime_pool = configure_search_parameters(cd, {xi}, base_pts, field_data['base_field'])
    E_rhs_m_symbolic = primary_tower[-1]['f_i'] if not resolve_project_symbol('FINITE_FIELD', default=None) else None
    search_rhs_list = build_search_rhs_list(cd, roots, E_rhs_m_symbolic, one, two, three)
    testfunc, shift = setup_rationality_test_function(shift, T, T_inv)

    base_sections = compute_base_sections_m(cd, base_pts, tower=primary_tower)
    if not base_sections:
        raise RuntimeError('compute_base_sections_m returned no sections for the rebuilt tower')
    base_sections = lll_reduce_mw_basis(cd, base_sections)
    current_sections = list(set(base_sections))
    if not current_sections:
        raise RuntimeError('No usable current sections after LLL reduction')

    # Markov mode: keep a single section so vecs can remain one-dimensional.
    current_sections = [current_sections[0]]

    if debug:
        print(f"[tower] rebuilt for xi={xi}; sections={len(current_sections)}; primes={len(prime_pool)}")

    return {
        'cd': cd,
        'current_sections': current_sections,
        'prime_pool': prime_pool,
        'r_m': r_m,
        'shift': shift,
        'search_rhs_list': search_rhs_list,
        'testfunc': testfunc,
        'field_data': field_data,
        'shifted_G_poly': shifted_G_poly,
        'base_pts': base_pts,
        'T': T,
        'T_inv': T_inv,
        'primary_tower': primary_tower,
        'fibrations': fibrations,
        'tower_for_mumford': tower_for_mumford,
        'roots': roots,
        'morphism_data': morphism_data,
        'sconf': sconf,
        'xi': xi,
        'yi': yi,
    }

def _collect_candidate_x_like_values(obj: Any, out: Optional[List[Any]] = None) -> List[Any]:
    """Fallback collector for x-like payloads in legacy return values.

    This is intentionally permissive and is only used when the Mumford result
    does not expose its x-residue map in a recognizable shape.
    """
    if out is None:
        out = []

    if obj is None:
        return out

    scalar_types = (int, float, complex, str)
    try:
        scalar_types = scalar_types + (Integer,)
    except Exception:
        raise

    if isinstance(obj, dict):
        for key in ('xj', 'x', 'x_val', 'xcoord', 'candidate_x', 'x_value'):
            if key in obj and obj[key] is not None:
                out.append(obj[key])
        for value in obj.values():
            _collect_candidate_x_like_values(value, out)
        return out

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) in (1, 2, 3) and all(not isinstance(v, (dict, list, tuple, set)) for v in seq):
            out.extend(seq)
            return out
        for value in seq:
            _collect_candidate_x_like_values(value, out)
        return out

    try:
        if isinstance(obj, scalar_types):
            out.append(obj)
            return out
    except Exception:
        raise

    try:
        if hasattr(obj, 'parent') or hasattr(obj, 'degree') or hasattr(obj, 'numerator'):
            out.append(obj)
            return out
    except Exception:
        raise

    return out

# ---------------------------------------------------------------------------
# Small algebra helpers
# ---------------------------------------------------------------------------

def _coerce_base_ring(p: Optional[int], base_ring: Optional[Any] = None):
    if base_ring is not None:
        return base_ring
    if p is None:
        return QQ
    return GF(int(p))

def build_hyperelliptic_poly(coeffs: Sequence[Any], x_sym=None, base_ring=None, descending: bool = True):
    """Build a polynomial from coefficients or pass through an existing polynomial."""
    if hasattr(coeffs, "parent") and hasattr(coeffs, "degree"):
        return coeffs

    base_ring = base_ring or QQ
    if x_sym is None:
        R = PolynomialRing(base_ring, "x")
        x_sym = R.gen()
    else:
        R = x_sym.parent()

    coeffs = list(coeffs)
    if descending:
        deg = len(coeffs) - 1
        poly = sum(base_ring(coeffs[i]) * x_sym ** (deg - i) for i in range(len(coeffs)))
    else:
        poly = sum(base_ring(c) * x_sym ** i for i, c in enumerate(coeffs))
    return poly

def flatten_roots(roots_with_mult: Sequence[Tuple[Any, int]]) -> List[Any]:
    out: List[Any] = []
    for root, mult in roots_with_mult:
        out.extend([root] * int(mult))
    return out

def missing_root_by_vieta(poly, known_roots: Sequence[Any]) -> Any:
    """Given a degree-5 polynomial and 4 known roots (with multiplicity), recover the fifth.

    Works for monic or non-monic polynomials over a field.
    """
    if poly is None:
        raise ValueError("missing_root_by_vieta requires a polynomial")
    R = poly.parent()
    x = R.gen()
    deg = int(poly.degree())
    if deg < 1:
        raise ValueError("polynomial degree too small for Vieta recovery")

    lc = poly.leading_coefficient()
    if lc == 0:
        raise ValueError("leading coefficient is zero")
    monic = poly / lc
    coeffs = monic.list()  # low-to-high
    # For monic x^d + a_{d-1}x^{d-1} + ... the sum of roots is -a_{d-1}
    a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else R.base_ring()(0)
    total_sum = -a_d_minus_1
    return total_sum - sum(known_roots)

def build_default_walker(
    coeffs: Sequence[Any],
    initial_x: Any,
    p: Optional[int] = None,
    initial_y: Optional[Any] = None,
    base_points: Optional[Sequence[Tuple[Any, Any]]] = None,
    seed: int = 0,
    load_sources: bool = True,
    verbose: bool = True,
    search_fn: Optional[Callable[..., Any]] = None,
    step_factory: Optional[StepFactory] = None,
    score_fn: Optional[ScoreFn] = None,
    log_path: Optional[str] = None,
    log_full_candidates: bool = False,
    log_candidate_limit: int = 25*infinity,
) -> Genus2MetropolisWalker:
    if load_sources:
        load_project_sources(verbose=verbose)
    curve_poly = build_hyperelliptic_poly(coeffs, base_ring=_coerce_base_ring(p))
    cfg = WalkConfig(
        seed=seed,
        verbose=verbose,
        log_path=log_path,
        log_full_candidates=log_full_candidates,
        log_candidate_limit=log_candidate_limit,
    )
    return Genus2MetropolisWalker(
        curve_poly=curve_poly,
        p=p,
        initial_x=initial_x,
        initial_y=initial_y,
        base_points=base_points,
        step_factory=step_factory,
        search_fn=search_fn,
        score_fn=score_fn,
        config=cfg,
    )

PROJECT_REGISTRY: Dict[str, Any] = {}

def resolve_project_symbol(name: str, default: Any = None, required: bool = False):
    """Resolve a symbol loaded from tower.sage / search7_genus2.sage.

    Looks in PROJECT_REGISTRY first (populated by load_project_sources),
    then falls back to the module globals for any symbol that was already
    defined before load_project_sources was called.
    """
    if name in PROJECT_REGISTRY:
        return PROJECT_REGISTRY[name]
    g = globals()
    if name in g:
        return g[name]
    if required:
        raise RuntimeError(
            f"Required project symbol {name!r} not found. "
            "Call load_project_sources() before using this function."
        )
    return default

def project_base_points_from_globals(current_x=None, current_y=None, p: Optional[int] = None):
    """Build a base-point list from project globals such as DATA_PTS_GENUS2."""
    data_pts = DATA_PTS_GENUS2
    yfun = get_y_unshifted_genus2
    finite_field = FINITE_FIELD

    pts = []
    for x in data_pts or []:
        y = None
        try:
            y = yfun(x) if yfun is not None else None
        except Exception:
            y = None
            raise
        if y is None:
            continue
        try:
            if finite_field is not None and p is not None:
                pts.append((GF(int(p))(x), GF(int(p))(y)))
            else:
                pts.append((QQ(x), QQ(y)))
        except Exception:
            try:
                pts.append((x, y))
            except Exception:
                raise
            raise

    if current_x is not None and current_y is not None:
        try:
            cx = GF(int(p))(current_x) if (p is not None and finite_field is not None) else QQ(current_x)
            cy = GF(int(p))(current_y) if (p is not None and finite_field is not None) else QQ(current_y)
            if (cx, cy) not in pts:
                pts.insert(0, (cx, cy))
        except Exception:
            raise

    if not pts and current_x is not None and current_y is not None:
        pts = [(current_x, current_y)]

    return pts

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RelationRecord:
    step_index: int
    n: int
    xi: Any
    m: Optional[Any] = None
    xj: Optional[Any] = None
    xk: Optional[Any] = None
    relation: str = ""
    step: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    restart: bool = False
    candidate_pool: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WalkConfig:
    max_n: int = 80
    coin_bias_for_xj: float = 0.5
    metropolis_temperature: float = 1.0
    restart_on_dead_end: bool = True
    allow_branching: bool = False
    branch_width: int = 2
    seed: int = 0
    # Degree of the hyperelliptic curve polynomial (y^2 = f(x), deg f = curve_degree).
    # The divisor relation from a double-tangency fiber intersection is:
    #   (curve_degree - 2)*xi + xj + xk - curve_degree*∞ = 0
    # and Vieta recovery needs curve_degree - 3 copies of xi as known roots (triple root
    # because of the double tangency, so 3 copies for degree 5, etc.).
    curve_degree: int = 5
    # Alias kept for backward compat — always equal to curve_degree.
    degree_for_intersection: int = 5
    verbose: bool = True
    log_path: Optional[str] = None
    from sage.all import infinity
    log_candidate_limit: int = 25*infinity
    log_full_candidates: bool = True
    diagnostic_print: bool = True
    diagnostic_show_poly: bool = True
    diagnostic_show_roots: bool = True

    def __post_init__(self):
        # Keep the two degree fields in sync whichever one the caller set.
        if self.degree_for_intersection != self.curve_degree:
            # If both were set to non-default values and differ, curve_degree wins.
            self.degree_for_intersection = self.curve_degree

class Genus2MetropolisWalker:
    """Run the proposed x-coordinate Markov walk on a genus-2 curve.

    This version keeps a running total of unique xj values seen so far. The
    counter is stored in each step record's `step` dict and in the JSONL log, so
    you can inspect growth step by step without changing the rest of the file.
    """

    def __init__(
        self,
        curve_poly,
        p: Optional[int] = None,
        initial_x: Optional[Any] = None,
        initial_y: Optional[Any] = None,
        base_points: Optional[Sequence[Tuple[Any, Any]]] = None,
        step_factory: Optional[StepFactory] = None,
        search_fn: Optional[Callable[..., Any]] = None,
        score_fn: Optional[ScoreFn] = None,
        config: Optional[WalkConfig] = None,
        rng: Optional[random.Random] = None,
        base_ring: Optional[Any] = None,
    ):
        self.config = config or WalkConfig()
        self.rng = rng or random.Random(self.config.seed)
        self.base_ring = _coerce_base_ring(p, base_ring)
        self.curve_poly = build_hyperelliptic_poly(curve_poly, base_ring=self.base_ring)
        self.p = p
        self.x = self.curve_poly.parent().gen()
        self.step_factory = step_factory or self._default_step_factory
        self.search_fn = search_fn
        self.score_fn = score_fn
        self.base_points = list(base_points or [])
        self.log_path = Path(self.config.log_path).expanduser() if self.config.log_path else None
        self.relation_matrix = self._relation_matrix

        if initial_x is None:
            raise ValueError("initial_x is required")

        self.current_x = self.base_ring(initial_x)
        self.current_y = self._recover_y(self.current_x, initial_y)

        self.visited_x = {self.current_x}
        self.unique_xj_seen = {self.current_x}
        # NEW: Track ALL candidate leaves across the entire walk
        self.global_leaves_seen = {self.current_x}
        # How many times each x has been stepped *through* as xi (i.e. used as chain state).
        # We avoid re-using high-visit-count nodes as the next xi when fresher candidates exist.
        self.xi_visit_count: Counter = Counter({self.current_x: 1})

        self.history: List[RelationRecord] = []
        self.dead_end_count = 0
        self.collision_count = 0      # path collisions: chosen xj already on chain path
        self.leaf_collision_count = 0 # graph collisions: any leaf already in global_leaves_seen
        self._restart_cursor = 0

        if not self.base_points:
            self.base_points.append((self.current_x, self.current_y))

    def _recover_y(self, x_val, explicit_y=None):
        if explicit_y is not None:
            return self.base_ring(explicit_y)

        rhs = self.curve_poly(x_val)
        if self.p is not None:
            if hasattr(rhs, "is_square") and rhs.is_square():
                try:
                    return rhs.sqrt()
                except Exception:
                    raise
            raise ValueError(f"No y given and f(x) is not a square at x={x_val!r}")

        sq = sqrt(rhs)
        if sq * sq != rhs:
            raise ValueError(f"No rational y at x={x_val!r}")
        return sq

    def _default_step_factory(self, current_x, n, seed=None, current_point=None):
        if "build_one_fibration_step" not in globals():
            raise RuntimeError(
                "No step_factory was provided and build_one_fibration_step is not loaded. "
                "Call load_project_sources() or pass a custom step_factory."
            )

        pts_x = [current_x]
        if self.base_points:
            pts_x.extend([x for x, _y in self.base_points if x is not None and x != current_x])

        pts_x = pts_x[: max(1, min(len(pts_x), 5))]
        g2 = len(pts_x)
        fx_SR = self.curve_poly
        f0 = self.curve_poly

        return build_one_fibration_step(
            fx_SR,
            f0,
            pts_x,
            g2,
            seed_int=int(seed or self.config.seed),
            verbose=False,
            parameter_m=None,
            use_anchor_points=True,
        )

    def _normalize_candidate_output(self, result: Any) -> Dict[str, Any]:
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
                        x = cand.get("xj", None)
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
                xs = {cand.get("xj") for cand in a if cand.get("xj") is not None}
                return {
                    "candidates": a,
                    "candidate_records": a,
                    "candidate_xs": xs,
                    "new_sections": b,
                    "precomputed_residues": c,
                    "stats": d,
                }
            if isinstance(a, (set, list, tuple)):
                records = [{"xj": x} for x in a]
                return {
                    "candidates": records,
                    "candidate_records": records,
                    "candidate_xs": set(a),
                    "new_sections": b,
                    "precomputed_residues": c,
                    "stats": d,
                }

        raise TypeError(f"Unsupported search result type: {type(result)!r}")

    def _prefer_unvisited_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Partition candidates by xi_visit_count and return the least-visited tier.

        We walk through visit counts 0, 1, 2, ... and return the first non-empty
        tier, so we always prefer candidates whose xj has never (or least often)
        been used as xi before.  Falls back to the full list if every candidate
        has been visited.
        """
        if not candidates:
            return candidates
        def _count(c):
            xj = c.get("xj") if isinstance(c, dict) else c
            return self.xi_visit_count.get(xj, 0)
        min_count = min(_count(c) for c in candidates)
        preferred = [c for c in candidates if _count(c) == min_count]
        return preferred

    def _register_unique_xj(self, xj):
        if xj is None:
            return False, len(self.unique_xj_seen)
        was_new = xj not in self.unique_xj_seen
        self.unique_xj_seen.add(xj)
        return was_new, len(self.unique_xj_seen)

    def _annotate_step_counts(self, step: Dict[str, Any], xj, accepted: bool) -> Tuple[bool, int]:
        unique_new = False
        unique_total = len(self.unique_xj_seen)

        if accepted and xj is not None:
            unique_new, unique_total = self._register_unique_xj(xj)

        if isinstance(step, dict):
            step["unique_xj_new"] = bool(unique_new)
            step["unique_xj_total"] = int(unique_total)

        return unique_new, unique_total

    def _solve_m_roots(self, step: Dict[str, Any]) -> List[Any]:
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

    def _candidate_xj_from_m(self, xi, m_val):
        return self.base_ring(xi) - self.base_ring(m_val)

    def _intersection_poly_from_step(self, step: Dict[str, Any]):
        """Best-effort access to a degree-5 intersection polynomial."""
        for key in ("intersection_poly", "fiber_poly", "intersection", "poly_x"):
            if key in step and step[key] is not None:
                return step[key]
            else:
                assert None, (step, key)
        return None

    def _choose_between(self, xj, xk, context: Dict[str, Any]):
        candidates = [c for c in (xj, xk) if c is not None]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if self.score_fn is None:
            return xj if self.rng.random() < self.config.coin_bias_for_xj else xk

        scores = [self._score_candidate(c, context) for c in candidates]
        temp = max(1e-12, float(self.config.metropolis_temperature))
        weights = [math.exp(-s / temp) for s in scores]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(candidates)

        u = self.rng.random() * total
        acc = 0.0
        for cand, w in zip(candidates, weights):
            acc += w
            if u <= acc:
                return cand
        return candidates[-1]

    def _jsonable(self, obj: Any):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, complex):
            return str(obj)
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            try:
                return self._jsonable(obj.item())
            except Exception:
                raise
                return str(obj)
        if dataclasses.is_dataclass(obj):
            return {k: self._jsonable(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._jsonable(v) for v in obj]
        return str(obj)

    def _record_to_log_dict(self, rec: RelationRecord) -> Dict[str, Any]:
        step = rec.step if isinstance(rec.step, dict) else {}
        candidate_pool = list(getattr(rec, 'candidate_pool', []) or [])
        selected = dict(getattr(rec, 'selected_candidate', {}) or {})

        limit = int(getattr(self.config, 'log_candidate_limit', 25*infinity) or 25*infinity)
        pool_summary = candidate_pool if self.config.log_full_candidates else candidate_pool[:limit]

        return {
            'step_index': rec.step_index,
            'n': rec.n,
            'xi': self._jsonable(rec.xi),
            'm': self._jsonable(rec.m),
            'xj': self._jsonable(rec.xj),
            'xk': self._jsonable(rec.xk),
            'accepted': bool(rec.accepted),
            'restart': bool(rec.restart),
            'relation': rec.relation,
            'candidate_count': len(candidate_pool),
            'candidate_pool': self._jsonable(pool_summary),
            'selected_candidate': self._jsonable(selected),
            'step': self._jsonable(step),
            'unique_xj_new': bool(step.get('unique_xj_new', False)),
            'unique_xj_total': int(step.get('unique_xj_total', len(self.unique_xj_seen))),
        }

    def _append_jsonl_log(self, rec: RelationRecord) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._record_to_log_dict(rec)
            with self.log_path.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            if self.config.verbose:
                print(f"[walk] JSONL logging failed: {exc}")
            raise

    def _next_restart_point(self):
        while self._restart_cursor < len(self.base_points):
            x, y = self.base_points[self._restart_cursor]
            self._restart_cursor += 1
            if x not in self.visited_x:
                return x, y
        return None

    def run_branching(self, num_steps: int, width: Optional[int] = None) -> List[List[RelationRecord]]:
        """Small breadth-style helper for the parallel-branch idea."""
        width = int(width or self.config.branch_width)
        branches: List[Tuple[Any, Any, List[RelationRecord]]] = [(self.current_x, self.current_y, [])]
        n_values = list(range(1, min(self.config.max_n, 80) + 1))

        for step_idx in range(num_steps):
            new_branches = []
            for bx, by, hist in branches:
                saved = (self.current_x, self.current_y, list(self.history), set(self.visited_x), set(self.unique_xj_seen))
                self.current_x, self.current_y = bx, by
                self.history = list(hist)
                self.visited_x = {r.xi for r in hist if r.xi is not None} | {bx}
                self.unique_xj_seen = set(self.visited_x)
                rec = self.step(n=n_values[step_idx % len(n_values)])
                if rec is not None and rec.accepted:
                    new_branches.append((self.current_x, self.current_y, list(self.history)))
                self.current_x, self.current_y, self.history, self.visited_x, self.unique_xj_seen = saved
            if not new_branches:
                break
            branches = new_branches[:width]
        return [hist for _x, _y, hist in branches]

    def summary(self) -> str:
        accepted = sum(1 for r in self.history if r.accepted)
        restarts = sum(1 for r in self.history if r.restart)
        unique_path_nodes = len(self.unique_xj_seen)
        total_leaves = len(self.global_leaves_seen)

        return (
            f"\n--- WALK SUMMARY ---\n"
            f"Steps taken: {len(self.history)}\n"
            f"Path accepted: {accepted}\n"
            f"Path collisions (xj revisited on chain): {self.collision_count}\n"
            f"Graph/birthday collisions (leaf already seen): {self.leaf_collision_count}\n"
            f"Restarts: {restarts}\n"
            f"Dead ends: {self.dead_end_count}\n"
            f"Nodes in chosen path: {unique_path_nodes}\n"
            f"Total unique leaves discovered (Graph Volume): {total_leaves}\n"
            f"--------------------"
        )

    def _make_relation(
            self,
            step_index: int,
            n: int,
            xi,
            m_val,
            xj,
            xk,
            step: Dict[str, Any],
            accepted=True,
            restart=False,
        ):
            deg = self.config.curve_degree
            xi_mult = deg - 2  # double tangency → multiplicity 3 for deg-5, etc.
            if xj is not None and xk is not None:
                relation = f"{xi_mult}*{xi} + {xj} + {xk} - {deg}*∞ = 0"
            elif xj is not None:
                relation = f"{xi_mult}*{xi} + {xj} + ? - {deg}*∞ = 0"
            else:
                relation = "no xj"

            # Memory Fix: Scrub heavy keys from the step dictionary to prevent history leaks
            clean_step = {}
            if isinstance(step, dict):
                bad_keys = {'raw_mumford_residues', 'precomputed_residues', 'context', 'candidates', 'candidate_records'}
                for k, v in step.items():
                    if k not in bad_keys:
                        clean_step[k] = v

            return RelationRecord(
                step_index=step_index,
                n=n,
                xi=xi,
                m=m_val,
                xj=xj,
                xk=xk,
                relation=relation,
                step=clean_step,
                accepted=accepted,
                restart=restart,
            )

    def _score_candidate_record(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
        if self.score_fn is None:
            return 0.0
        xj = candidate.get("xj")
        # Raises on failure instead of silently returning 0.0
        return float(self.score_fn(xj, context | {"candidate": candidate}))

    def _score_candidate(self, candidate_x, context: Dict[str, Any]) -> float:
        if self.score_fn is None:
            assert None, None
            return 0.0
        # Raises on failure instead of silently returning 0.0
        return float(self.score_fn(candidate_x, context))

    def _recover_xk(self, step: Dict[str, Any], xi, xj):
        poly = self._intersection_poly_from_step(step)
        assert poly, poly
        if poly is None:
            assert None, None
            return None

        xi_mult = self.config.curve_degree - 2  # double tangency → triple root for deg-5

        roots = flatten_roots(poly_roots_with_multiplicity(poly))

        if roots:
            leftovers = []
            xi_count = 0
            xj_count = 0
            for r in roots:
                if r == xi and xi_count < xi_mult:
                    xi_count += 1
                    continue
                if xj is not None and r == xj and xj_count < 1:
                    xj_count += 1
                    continue
                leftovers.append(r)
            if leftovers:
                return leftovers[0]

        if poly.degree() != self.config.curve_degree:
            return None
        known = [xi] * xi_mult
        if xj is not None:
            known.append(xj)
        return missing_root_by_vieta(poly, known)

    def print_relation_summary(self, **kwargs):
        """Prints the shape, column mapping, and rank of the relation matrix."""
        mat, atoms, used = self.relation_matrix()
        print_relation_matrix_summary(mat, atoms, used, **kwargs)

    def _relation_matrix(self, **kwargs):
        cd = getattr(self.config, "curve_degree", 5)
        return build_relation_matrix2(self.history, curve_degree=cd, **kwargs)

    def close_under_involution(self) -> int:
        """For every accepted step in history, compute the 2-cycle partner xk'
        via the T-involution T(xj) = 5*xi - xj - S(xj→m), and append a free
        RelationRecord for the pair (xj, xk').

        Asserts T(T(xj)) == xj for every point — crashes if the 2-cycle
        assumption is violated.

        Iterates to closure: newly added xk' values are themselves queued for
        partner computation until no new atoms are produced.

        Returns the number of free relations appended.
        """
        from genus2_markov_module import compute_S_of_m

        p = self.p
        deg = self.config.curve_degree
        xi_mult = deg - 2

        # Collect S_of_m per xi from history records that have it.
        # S_of_m is a rational function in m over F_p; we need to evaluate it
        # numerically at a given m value.  We stash the symbolic object keyed
        # by xi so we can reuse it across the pool.
        xi_to_S_sym: Dict[Any, Any] = {}
        xi_to_G_poly: Dict[Any, Any] = {}
        for rec in self.history:
            if rec.accepted and rec.xi not in xi_to_S_sym:
                S_sym = rec.step.get('S_of_m') if isinstance(rec.step, dict) else None
                if S_sym is not None:
                    xi_to_S_sym[rec.xi] = S_sym

        def _eval_S(xi, m_val):
            """Evaluate S(m) at a numeric m, using the symbolic S stored for xi."""
            S_sym = xi_to_S_sym.get(xi)
            assert S_sym is not None, (
                f"close_under_involution: no S_of_m available for xi={xi}. "
                f"Run the walk first so compute_S_of_m is stored on candidate records."
            )
            Fp = self.base_ring
            m_fp = Fp(m_val)
            try:
                # S_sym is a Sage rational function in m; call it directly.
                return Fp(S_sym(m_fp))
            except Exception:
                # Fallback: evaluate numerator/denominator separately.
                num = S_sym.numerator()
                den = S_sym.denominator()
                dv = Fp(den(m_fp))
                assert dv != 0, f"close_under_involution: S(m) denominator zero at m={m_val}"
                raise
                return Fp(num(m_fp)) / dv

        def _T(xi, xj_val):
            """One application of the involution: xj → xk' = T(xj)."""
            Fp = self.base_ring
            # m = xi - xj  (from xj = xi - m)
            m_val = Fp(xi) - Fp(xj_val)
            S_val = _eval_S(xi, m_val)
            return Fp(5 * Fp(xi) - Fp(xj_val) - S_val)

        def _assert_2cycle(xi, xj_val, partner):
            roundtrip = _T(xi, partner)
            assert roundtrip == self.base_ring(xj_val), (
                f"close_under_involution: 2-cycle violated for xi={xi}, "
                f"xj={xj_val}, T(xj)={partner}, T(T(xj))={roundtrip} != xj"
            )

        # Collect all (xi, xj) pairs already in history so we don't duplicate.
        existing_pairs: set = set()
        for rec in self.history:
            if rec.accepted and rec.xi is not None and rec.xj is not None:
                existing_pairs.add((rec.xi, rec.xj))

        # Queue of (xi, xj) pairs to process — seeded from accepted history.
        from collections import deque
        queue: deque = deque()
        for rec in self.history:
            if rec.accepted and rec.xi is not None and rec.xj is not None:
                queue.append((rec.xi, rec.xj))

        n_added = 0
        step_base = len(self.history)

        while queue:
            xi_val, xj_val = queue.popleft()

            # Skip if we have no S_of_m for this xi.
            if xi_val not in xi_to_S_sym:
                continue

            partner = _T(xi_val, xj_val)
            _assert_2cycle(xi_val, xj_val, partner)

            pair_fwd = (xi_val, partner)
            if pair_fwd in existing_pairs:
                continue

            # New atom: build a free RelationRecord for  xi_val → partner.
            existing_pairs.add(pair_fwd)
            xk_of_partner = xj_val   # by the 2-cycle: T(partner) = xj_val

            relation_str = (
                f"{xi_mult}*{xi_val} + {partner} + {xk_of_partner} - {deg}*∞ = 0"
                f"  [free/involution]"
            )
            free_rec = RelationRecord(
                step_index=step_base + n_added,
                n=-1,                      # sentinel: not from an m-root search
                xi=xi_val,
                m=None,                    # no fiber solve needed
                xj=partner,
                xk=xk_of_partner,
                relation=relation_str,
                step={'source': 'involution_closure'},
                accepted=True,
                restart=False,
            )
            self.history.append(free_rec)
            n_added += 1

            # Queue the partner itself so its own partner gets computed.
            queue.append((xi_val, partner))

        print(
            f"[close_under_involution] appended {n_added} free relations "
            f"({n_added} new d1 atoms). History size now {len(self.history)}."
        )
        return n_added

    def _call_search_fn(self, n: int, seed: Optional[int] = None, current_point=None):
        if self.search_fn is None:
            return None

        kwargs = {
            "xi": self.current_x,
            "current_x": self.current_x,
            "n": n,
            "seed": seed if seed is not None else self.config.seed,
            "current_point": current_point,
            "walker": self,
            "curve_poly": self.curve_poly,
            "base_points": self.base_points,
            "p": self.p,
            "base_ring": self.base_ring,
        }

        attempts = [
            lambda: self.search_fn(**kwargs),
            lambda: self.search_fn(self.current_x, n=n, seed=seed, current_point=current_point),
            lambda: self.search_fn(self.current_x, n),
            lambda: self.search_fn(self.current_x),
        ]

        last_exc = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last_exc = exc
                raise
                continue

        if last_exc is not None:
            raise last_exc
        return None

    def _choose_candidate_record(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]):
        if not candidates:
            return None

        # Prefer candidates whose xj has been stepped through least often.
        pool = self._prefer_unvisited_candidates(candidates)

        if len(pool) == 1:
            return pool[0]

        if self.score_fn is None:
            return self.rng.choice(pool)

        scores = [self._score_candidate_record(c, context) for c in pool]
        temp = max(1e-12, float(self.config.metropolis_temperature))
        weights = [math.exp(-s / temp) for s in scores]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(pool)

        u = self.rng.random() * total
        acc = 0.0
        for cand, w in zip(pool, weights):
            acc += w
            if u <= acc:
                return cand
        return pool[-1]

    def _step_from_candidate_search(self, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
        current_point = (self.current_x, self.current_y)
        raw = self._call_search_fn(n=n, seed=seed, current_point=current_point)
        search_out = self._normalize_candidate_output(raw)

        candidates = list(search_out.get("candidate_records") or search_out.get("candidates") or [])
        candidate_xs = search_out.get("candidate_xs", set())

        if self.config.verbose:
            print(
                f"[walk] n={n} candidates={len(candidates)} "
                f"candidate_xs={len(candidate_xs) if hasattr(candidate_xs, '__len__') else 'unk'}"
            )

        # Leaf bookkeeping for the search branch.
        # We treat the candidate x-values as the leaves discovered by this step.
        old_leaves_count = len(self.global_leaves_seen)
        valid_leaves = {cx for cx in candidate_xs if cx is not None}

        if valid_leaves:
            self.global_leaves_seen.update(valid_leaves)

        new_leaves_this_step = len(self.global_leaves_seen) - old_leaves_count
        leaf_collisions_this_step = len(valid_leaves) - new_leaves_this_step if valid_leaves else 0
        if leaf_collisions_this_step < 0:
            leaf_collisions_this_step = 0

        self.leaf_collision_count += leaf_collisions_this_step

        search_out["step_leaves_found"] = len(valid_leaves)
        search_out["step_leaves_new"] = new_leaves_this_step
        search_out["step_leaf_collisions"] = leaf_collisions_this_step
        search_out["global_leaves_total"] = len(self.global_leaves_seen)
        search_out["global_leaf_collisions"] = self.leaf_collision_count

        if not candidates:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    rec = self._make_relation(
                        len(self.history), n, self.current_x, None, None, None,
                        search_out, accepted=False, restart=True
                    )
                    rec.candidate_pool = []
                    rec.selected_candidate = {}
                    self._store_record(rec)
                    return rec

            rec = self._make_relation(
                len(self.history), n, self.current_x, None, None, None,
                search_out, accepted=False
            )
            rec.candidate_pool = []
            rec.selected_candidate = {}
            self._store_record(rec)
            return rec

        chosen = self._choose_candidate_record(
            candidates,
            {
                "n": n,
                "step": search_out,
                "current_x": self.current_x,
                "current_y": self.current_y,
            },
        )

        if chosen is None:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    rec = self._make_relation(
                        len(self.history), n, self.current_x, None, None, None,
                        search_out, accepted=False, restart=True
                    )
                    rec.candidate_pool = candidates
                    rec.selected_candidate = {}
                    self._store_record(rec)
                    return rec

            rec = self._make_relation(
                len(self.history), n, self.current_x, None, None, None,
                search_out, accepted=False
            )
            rec.candidate_pool = candidates
            rec.selected_candidate = {}
            self._store_record(rec)
            return rec

        if not isinstance(chosen, dict):
            chosen = {"xj": chosen}

        m_val = chosen.get("m")
        xj = chosen.get("xj")
        xk = chosen.get("xk")

        if xj is None and m_val is not None:
            xj = self._candidate_xj_from_m(self.current_x, m_val)

        if xj is None and "x" in chosen:
            xj = chosen.get("x")

        if xk is None and xj is not None:
            xk = self._recover_xk(search_out if isinstance(search_out, dict) else {}, self.current_x, xj)

        accepted = xj is not None
        xi_before = self.current_x

        if accepted:
            next_x = xj
            try:
                next_y = self._recover_y(next_x)
            except Exception:
                raise

            if next_y is not None:
                self.current_x, self.current_y = next_x, next_y
                self.visited_x.add(next_x)
            else:
                self.current_x = next_x

        step_payload = dict(search_out) if isinstance(search_out, dict) else {}
        if chosen.get('intersection_poly') is not None:
            step_payload['intersection_poly'] = chosen['intersection_poly']
        unique_xj_new, unique_xj_total = self._annotate_step_counts(
            step_payload,
            xj if accepted else None,
            accepted=accepted,
        )
        if accepted and not unique_xj_new:
            self.collision_count += 1

        rec = self._make_relation(
            len(self.history), n, xi_before, m_val, xj, xk,
            step_payload, accepted=accepted, restart=False
        )
        rec.candidate_pool = candidates
        rec.selected_candidate = dict(chosen)
        self._store_record(rec)
        return rec

    def step(self, n: Optional[int] = None, seed: Optional[int] = None) -> Optional[RelationRecord]:
        n = int(n or (len(self.history) + 1))
        if n < 1:
            n = 1
        if n > self.config.max_n:
            n = self.config.max_n

        if self.search_fn is not None:
            try:
                return self._step_from_candidate_search(n=n, seed=seed)
            except Exception as exc:
                if self.config.verbose:
                    print(f"[walk] candidate search failed at n={n}: {exc}")
                if self.config.restart_on_dead_end:
                    restart = self._next_restart_point()
                    if restart is not None:
                        self.current_x, self.current_y = restart
                        self.dead_end_count += 1
                        step_payload = {}
                        self._annotate_step_counts(step_payload, None, accepted=False)
                        rec = self._make_relation(
                            len(self.history), n, self.current_x, None, None, None,
                            step_payload, accepted=False, restart=True
                        )
                        self._store_record(rec)
                        raise
                        return rec
                raise

        current_point = (self.current_x, self.current_y)
        try:
            step = self.step_factory(self.current_x, n, seed=seed, current_point=current_point)
        except Exception as exc:
            if self.config.verbose:
                print(f"[walk] step factory failed at n={n}: {exc}")
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    step_payload = {}
                    self._annotate_step_counts(step_payload, None, accepted=False)
                    rec = self._make_relation(
                        len(self.history), n, self.current_x, None, None, None,
                        step_payload, accepted=False, restart=True
                    )
                    self._store_record(rec)
                    raise
                    return rec
            raise

        m_roots = self._solve_m_roots(step)
        assert m_roots, m_roots

        xj_candidates = [self._candidate_xj_from_m(self.current_x, m_val) for m_val in m_roots]
        assert xj_candidates, xj_candidates

        old_leaves_count = len(self.global_leaves_seen)
        valid_leaves = {cx for cx in xj_candidates if cx is not None}

        for _xj in xj_candidates:
            _xk = self._recover_xk(step, self.current_x, _xj)
            if _xk is not None:
                valid_leaves.add(_xk)

        if valid_leaves:
            self.global_leaves_seen.update(valid_leaves)

        new_leaves_this_step = len(self.global_leaves_seen) - old_leaves_count
        leaf_collisions_this_step = len(valid_leaves) - new_leaves_this_step if valid_leaves else 0
        if leaf_collisions_this_step < 0:
            leaf_collisions_this_step = 0

        self.leaf_collision_count += leaf_collisions_this_step
        step_novelty_ratio = (new_leaves_this_step / len(valid_leaves)) if valid_leaves else 0.0

        step_payload = dict(step) if isinstance(step, dict) else {}
        if isinstance(step_payload, dict):
            step_payload["step_leaves_found"] = len(valid_leaves)
            step_payload["step_leaves_new"] = new_leaves_this_step
            step_payload["step_leaf_collisions"] = leaf_collisions_this_step
            step_payload["global_leaves_total"] = len(self.global_leaves_seen)
            step_payload["global_leaf_collisions"] = self.leaf_collision_count

        if self.config.verbose:
            sqrt_p = (self.p ** 0.5) if self.p is not None else float("nan")
            collision_frac = len(self.global_leaves_seen) / sqrt_p if self.p is not None else float("nan")
            print(
                f"\n[CANDIDATES] step={len(self.history)+1} n={n} | "
                f"xi={self.current_x} | "
                f"m-roots: {len(xj_candidates)} -> {new_leaves_this_step} new leaves "
                f"(novelty {step_novelty_ratio:.1%}) | "
                f"Graph volume: {len(self.global_leaves_seen)} "
                f"({collision_frac:.3f}×√p)"
            )

        xj = xj_candidates[0] if xj_candidates else None
        if xj is None:
            raise ValueError(f"No xj candidate produced at n={n}")
        m_val = m_roots[0] if m_roots else None

        xk = self._recover_xk(step, self.current_x, xj)
        if xk is None:
            raise ValueError(f"No xk recovered at n={n} for xi={self.current_x}, xj={xj}")

        chosen = self._choose_between(
            xj,
            xk,
            {
                "n": n,
                "step": step,
                "current_x": self.current_x,
                "current_y": self.current_y,
            },
        )
        accepted = chosen is not None

        unique_xj_new = False
        unique_xj_total = len(self.unique_xj_seen)

        xi_before = self.current_x
        if accepted:
            next_x = chosen
            self.xi_visit_count[next_x] += 1
            unique_xj_new, unique_xj_total = self._register_unique_xj(next_x)
            if not unique_xj_new:
                self.collision_count += 1

            try:
                next_y = self._recover_y(next_x)
            except Exception:
                raise

            if next_y is not None:
                self.current_x, self.current_y = next_x, next_y
                self.visited_x.add(next_x)
            else:
                self.current_x = next_x
        else:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    self._annotate_step_counts(step_payload, None, accepted=False)
                    rec = self._make_relation(
                        len(self.history), n, xi_before, m_val, xj, xk,
                        step_payload, accepted=False, restart=True
                    )
                    self._store_record(rec)
                    return rec

        self._annotate_step_counts(step_payload, chosen if accepted else None, accepted=accepted)

        rec = self._make_relation(
            len(self.history), n, xi_before, m_val, xj, xk,
            step_payload, accepted=accepted
        )
        self._store_record(rec)
        return rec

    def run(self, num_steps: int, n_values: Optional[Sequence[int]] = None) -> List[RelationRecord]:
        results: List[RelationRecord] = []
        if num_steps <= 0:
            return results

        for i in range(num_steps):
            n = None
            if n_values:
                n = n_values[i % len(n_values)]

            rec = self.step(n=n)
            if rec is None:
                break

            results.append(rec)

            step_no = len(self.history)
            accepted_count = sum(1 for r in self.history if r.accepted)
            restarts = sum(1 for r in self.history if r.restart)

            step_dict = rec.step if isinstance(rec.step, dict) else {}
            step_leaves = int(step_dict.get("step_leaves_found", 0) or 0)
            new_leaves = int(step_dict.get("step_leaves_new", 0) or 0)
            total_leaves = int(step_dict.get("global_leaves_total", len(self.global_leaves_seen)) or 0)

            expansion_rate = (total_leaves / step_no) if step_no > 0 else 0.0
            sqrt_p = (self.p ** 0.5) if self.p is not None else float("nan")
            collision_frac = total_leaves / sqrt_p if self.p is not None else float("nan")
            novelty_ratio = (new_leaves / step_leaves) if step_leaves > 0 else 0.0

            xj_str = str(rec.xj) if rec.xj is not None else "—"
            xk_str = str(rec.xk) if rec.xk is not None else "—"
            m_str = str(rec.m) if rec.m is not None else "—"
            rel_str = rec.relation if rec.relation else "—"
            xj_visits = self.xi_visit_count.get(rec.xj, 0) if rec.xj is not None else 0
            xi_visits = self.xi_visit_count.get(rec.xi, 0) if rec.xi is not None else 0

            path_collision = (rec.xj is not None and not step_dict.get("unique_xj_new", False))
            leaf_collisions_this_step = step_dict.get("step_leaf_collisions", 0)

            n_with_roots = step_dict.get("n_with_roots")
            n_total = step_dict.get("n_total") or self.config.max_n
            total_roots = step_dict.get("total_roots")
            per_n_roots_map = step_dict.get("per_n_roots") or {}
            if n_with_roots is None and per_n_roots_map:
                n_with_roots = len(per_n_roots_map)

            xj_leaves_count = step_dict.get("step_xj_leaves", step_leaves)
            xk_new_count = step_dict.get("step_xk_leaves_new", 0)
            xk_overlap = step_dict.get("step_xk_leaves_overlap", 0)

            if n_with_roots is not None:
                frac_fertile = n_with_roots / n_total if n_total > 0 else 0.0
                avg_roots = (total_roots / n_with_roots) if (total_roots and n_with_roots) else None
                avg_str = f" avg {avg_roots:.1f} roots/fertile-n" if avg_roots is not None else ""
                if per_n_roots_map and isinstance(per_n_roots_map, dict):
                    dist = Counter(per_n_roots_map.values())
                    dist_str = " ".join(f"{cnt}n×{nr}r" for nr, cnt in sorted(dist.items()))
                    frac_fertile_str = (
                        f"{n_with_roots}/{n_total} ({frac_fertile:.0%})"
                        f" | {total_roots} m-roots{avg_str}"
                        f" | dist [{dist_str}]"
                    )
                else:
                    frac_fertile_str = (
                        f"{n_with_roots}/{n_total} ({frac_fertile:.0%})"
                        + (f" | {total_roots} m-roots{avg_str}" if total_roots is not None else "")
                    )
            else:
                frac_fertile_str = "n/a"

            xk_leaf_note = (
                f" (+{xk_new_count} xk-new, {xk_overlap} xk↔xj overlap)"
                if xk_new_count or xk_overlap else ""
            )

            pool = getattr(rec, "candidate_pool", []) or []
            n_xk_head_pool = sum(1 for c in pool if isinstance(c, dict) and c.get("source") == "xk_head")
            n_xj_head_pool = len(pool) - n_xk_head_pool
            if n_xk_head_pool:
                rel_annotation = f"  (pool: {n_xj_head_pool} xj-head + {n_xk_head_pool} xk-head)"
            elif len(pool) > 1:
                rel_annotation = f"  (chosen from pool of {len(pool)})"
            else:
                rel_annotation = ""

            print(
                f"\n{'='*70}",
                f"\n[WALK] STEP {step_no} COMPLETE  (outer n={rec.n})",
                f"\n  Path:      xi → xj  |  xi={rec.xi}  (visited {xi_visits}×)",
                f"\n             xj={xj_str}  (visited {xj_visits}×)  |  xk={xk_str}  |  m={m_str}",
                f"\n  Relation (example):  {rel_str}{rel_annotation}",
                f"\n  This step: accepted={rec.accepted}  path_collision={'YES' if path_collision else 'no'}  | leaf_collisions_this_step={leaf_collisions_this_step}",
                f"\n  Collisions: path={self.collision_count} total  | graph/birthday={self.leaf_collision_count} total  (first expected near √p={sqrt_p:.0f} graph volume)",
                f"\n  Totals:    steps_accepted={accepted_count}  restarts={restarts}  dead_ends={self.dead_end_count}",
                f"\n  Leaves:    xj={xj_leaves_count}{xk_leaf_note}  total={step_leaves}  new={new_leaves}  novelty={novelty_ratio:.1%}",
                f"\n  Graph vol: {total_leaves} unique x-coords seen across all leaves  ({collision_frac:.4f}×√p  [√p={sqrt_p:.1f}])",
                f"\n  Rate:      {expansion_rate:.2f} unique leaves/step",
                f"\n  Fertility: {frac_fertile_str} of n-values had F_p roots",
                f"\n{'='*70}\n",
                sep="",
                flush=True,
            )

        return results

    def _store_record(self, rec: RelationRecord) -> RelationRecord:
        # Only trim the candidate pool if the user hasn't requested full candidates.
        # If we trim it, those leaves won't make it into the relation matrix!
        if not getattr(self.config, 'log_full_candidates', True):
            limit = getattr(self.config, 'log_candidate_limit', 25*infinity) or 25*infinity
            if hasattr(rec, 'candidate_pool') and rec.candidate_pool and len(rec.candidate_pool) > limit:
                if limit < infinity:
                    rec.candidate_pool = rec.candidate_pool[:limit]
                else:
                    pass

        self.history.append(rec)
        self._append_jsonl_log(rec)
        return rec

def enable_step_diagnostics(walker_class=Genus2MetropolisWalker):
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
        poly = self._intersection_poly_from_step(step)
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
        # For degree-5 with triple xi root:
        #   known roots = [xi, xi, xi, xj]
        #   missing root = xk
        if rec.xi is not None and rec.xj is not None:
            try:
                lc = poly.leading_coefficient()
                monic = poly / lc
                coeffs = monic.list()  # low-to-high coefficients
                deg = int(monic.degree())
                a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else monic.parent().base_ring()(0)
                total_root_sum = -a_d_minus_1

                xi_mult = curve_degree - 2
                known_sum = xi_mult * rec.xi + rec.xj
                xk_vieta = total_root_sum - known_sum

                print(f"  monic sum-of-roots       = {total_root_sum}  (S evaluated at this m)")
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
                    print(
                        "  recurrence (symbolic)    = "
                        f"xk(m) = S(m) - {xi_mult}*{rec.xi} - xj(m)"
                        f"      [xj(m) = {rec.xi} - m]"
                    )
                else:
                    print(f"  S(m) symbolic            = <unavailable — fi not in step payload>")
                    print(
                        "  recurrence preview       = "
                        f"xj = xi - m,  "
                        f"xk = ({total_root_sum}) - ({xi_mult}*xi + xj)"
                        f"  [S={total_root_sum} is numeric, m already substituted]"
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

enable_step_diagnostics()
