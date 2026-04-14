from __future__ import annotations
import argparse, json, dataclasses, math, random, itertools, bounds, warnings, sys, inspect, json as _json
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
from .relation_matrix import *
from functools import partial
from .cantor_cache import *
from .mixing_diagnostics import *
from .adjacency_matrix import *

class _FiberPoleError(Exception):
    """Raised when S(m) has a pole at a specific m value.

    A zero denominator in S(m) means the secant-line fiber is degenerate
    (the rational function parametrisation breaks down) at that particular m.
    This is not a 2-cycle violation; the (xi, xj) pair is simply unevaluable
    and should be skipped rather than treated as an error.
    """

def poly_roots_with_multiplicity(poly) -> List[Tuple[Any, int]]:
    """Return roots as (root, multiplicity) pairs over the polynomial's base field."""
    roots = poly.roots(multiplicities=True)
    assert roots, roots
    return [(r, int(m)) for r, m in roots]

def compute_S_of_m(fi, G_poly, curve_degree):
    """Return the x^(d-1) coefficient of (G(x) - f_i(x, m)) as a symbolic
    rational function in m, without evaluating m numerically.

    fi lives in R_xm = PolynomialRing(Frac(GF(p)['m']), 'x'), so its
    coefficients are already rational functions in m.  G_poly lives in
    GF(p)[x] with constant coefficients.  We lift G into the same ring and
    subtract to get the bivariate intersection polynomial, then read off the
    x^(d-1) coefficient.

    Returns (S_of_m, inter_sym) where:
      S_of_m   -- negated x^(d-1) coeff of monic inter, a rational function in m
      inter_sym -- full symbolic intersection poly in R_xm[x]

    Returns (None, None) if fi or G_poly are unavailable.
    """
    if fi is None or G_poly is None:
        return None, None
    try:
        R_xm = fi.parent()                          # PolynomialRing(Frac(Fp[m]), 'x')
        base = R_xm.base_ring()                     # Frac(Fp[m])
        # Lift G_poly coefficients into Frac(Fp[m]) so subtraction is valid.
        G_lifted = R_xm([base(c) for c in G_poly.list()])
        inter_sym = G_lifted - fi
        lc = inter_sym.leading_coefficient()
        monic_sym = inter_sym / lc
        deg = int(monic_sym.degree())
        coeffs = monic_sym.list()                   # low-to-high
        # x^(d-1) coeff is at index deg-1; sum-of-roots = -that coeff
        a_dm1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else base(0)
        S_of_m = -a_dm1
        return S_of_m, inter_sym
    except Exception:
        raise

def compute_xk_from_fiber(xi_val, m_val, xj_val, fi, G_poly, curve_degree):
    if fi is None or G_poly is None or m_val is None:
        assert None, None
        return None, None

    try:
        Rx = G_poly.parent()
        Fp = Rx.base_ring()
        m_fp = Fp(m_val)

        def eval_at_m(obj):
            try:
                return obj(m_fp) if callable(obj) else obj
            except TypeError:
                raise
                return obj

        coeffs = []
        for c in fi.list():
            num = c.numerator()
            den = c.denominator()

            nv = eval_at_m(num)
            dv = eval_at_m(den)

            dv_fp = Fp(dv)
            if dv_fp == Fp(0):
                raise ZeroDivisionError(
                    f"compute_xk_from_fiber: fiber pole at m={m_val} "
                    f"(denominator of fi coefficient vanished).\n"
                    f"  xi={xi_val}  xj={xj_val}\n"
                    f"  fi={fi}\n"
                    f"  G_poly={G_poly}\n"
                    f"  coeffs so far={coeffs}"
                )

            coeffs.append(Fp(nv) / dv_fp)

        fi_at_m = Rx(coeffs)
        inter = G_poly - fi_at_m

        if inter.degree() != curve_degree:
            assert False, f"inter.degree()={inter.degree()} != curve_degree={curve_degree}, xi={xi_val} m={m_val} xj={xj_val}"
            return None, None

        #xi_mult = curve_degree - 2

        # Replace the hardcoded xi_mult with the actual multiplicity from the fiber poly.
        roots_wm = inter.roots()  # Sage: [(root, mult), ...]
        actual_xi_mult = 0
        for r, m in roots_wm:
            if Fp(r) == Fp(xi_val):
                actual_xi_mult = int(m)
                break
        assert actual_xi_mult > 0, (
            f"compute_xk_from_fiber: xi={xi_val} is not a root of the fiber intersection poly "
            f"at m={m_val}, xj={xj_val}. roots={roots_wm}. "
            f"This means the fiber was constructed for a different xi or the multiplicity "
            f"assumptions are wrong."
        )

        known = [xi_val] * actual_xi_mult + [xj_val]
        return missing_root_by_vieta(inter, known), inter

    except Exception:
        raise

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
    assert xi is not None, xi
    assert yi is not None, yi
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
    E_rhs_m_symbolic = primary_tower[-1]['f_i'] if primary_tower else None
    search_rhs_list = build_search_rhs_list(cd, roots, E_rhs_m_symbolic, one, two, three)

    # Add xk(m) as second RHS via Vieta: xk = S(m) - (d-1)*xi - xj(m).
    # S(m) is the negated x^(d-1) coefficient of the monic fiber intersection poly,
    # which equals xi + xj + xk for a degree-5 curve (d-1 = 4 roots sum to S).
    # We use the actual xj(m) RHS from the search rather than the RLINEAR=True
    # shortcut xi-m, so this is valid regardless of RLINEAR.
    _fi_for_xk = primary_tower[-1].get('f_i') if primary_tower else None
    _curve_degree = int(resolve_project_symbol('CURVE_DEGREE', default=5))
    if _fi_for_xk is not None and shifted_G_poly is not None and len(search_rhs_list) == 1:
        S_of_m, _ = compute_S_of_m(_fi_for_xk, shifted_G_poly, _curve_degree)
        if S_of_m is not None:
            try:
                _base = S_of_m.parent()           # Frac(GF(p)[m])
                _xj_rhs = _base(r_m)
                _xi_lifted = _base(xi)
                xk = S_of_m - (_curve_degree - 1) * _xi_lifted - _xj_rhs
                lastrhs = E_rhs_m(x=xk)
                last_phi_x = get_phi_x(one, two, three, xk, lastrhs)
                search_rhs_list = list(search_rhs_list) + [last_phi_x]
            except Exception as e:
                print(f"[build_project_tower_context] warning: could not build xk RHS: {e}")
                raise

    assert len(search_rhs_list) > 1, search_rhs_list

    testfunc, shift = setup_rationality_test_function(shift, T, T_inv)

    base_sections = compute_base_sections_m(cd, base_pts, tower=primary_tower)
    if not base_sections:
        raise RuntimeError('compute_base_sections_m returned no sections for the rebuilt tower')
    if len(base_sections) > 1:
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
    preferred_xs: Optional[list] = None,
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
        preferred_xs=preferred_xs,
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
    # y-branch signs for xj and xk: +1 = canonical (positive) root, -1 = conjugate.
    # Default to +1 (old behaviour) when signs are not available (e.g. preferred_injection).
    yj_sign: int = 1
    yk_sign: int = 1
    # Actual multiplicity of xi in the fiber intersection poly.
    # Defaults to -1 (sentinel: not set); build_relation_matrix2 falls back to curve_degree-2.
    xi_mult: int = -1

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
    log_candidate_limit: int = 25*infinity
    log_full_candidates: bool = True
    diagnostic_print: bool = True
    diagnostic_show_poly: bool = True
    diagnostic_show_roots: bool = True

    # Spectral gap reporting via adjacency matrix.
    # spectral_enabled=False turns the whole thing off silently.
    spectral_report_every: int = 10
    spectral_min_collisions: int = 3
    spectral_enabled: bool = True
    # How many eigenvalues to request from the sparse Arnoldi solver.
    # 50 is cheap (<<1s) for matrices up to ~10k nodes; raise further to
    # probe deeper into the tail for expander-bound analysis.
    spectral_n_eigenvalues: int = 50

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
        preferred_xs: Optional[List[Any]] = None,
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
        self.preferred_xs = list(preferred_xs or [])
        # Nullity-derived injection atoms with a finite budget.
        # Each entry x -> remaining is decremented on use and deleted at zero.
        # Divisor seeds go in preferred_xs (permanent); nullity hints go here.
        self.preferred_xs_budget: Dict[Any, int] = {}

        if initial_x is None:
            raise ValueError("initial_x is required")

        self.current_x = self.base_ring(initial_x)
        self.current_y = self._recover_y(self.current_x, initial_y)

        self.visited_x = {self.current_x}
        self.unique_xj_seen = {self.current_x}
        # NEW: Track ALL candidate leaves across the entire walk
        self.global_leaves_seen = {self.current_x}
        # Cumulative count of leaf insertions (not deduplicated) — the true
        # "effort" metric for merge-time analysis, independent of step count.
        self.total_leaf_insertions: int = 1  # seed x counts as one insertion
        # Xs that are structurally pre-known (seed, preferred_xs, budget keys).
        # Birthday collision accounting excludes these: collisions on injected
        # xs are guaranteed by construction and not diagnostic of walk mixing.
        self._injected_xs: set = {self.current_x} | {self.base_ring(x) for x in (preferred_xs or [])}
        # How many times each x has been stepped *through* as xi (i.e. used as chain state).
        # We avoid re-using high-visit-count nodes as the next xi when fresher candidates exist.
        self.xi_visit_count: Counter = Counter({self.current_x: 1})

        self.history: List[RelationRecord] = []
        self.cantor_cache: Optional[CantorPairCache] = None
        self.dead_end_count = 0
        self.dead_end_reasons: Counter = Counter()  # reason -> count

        # Cross-chain merge tracking.
        # Set via load_foreign_leaves(); once a leaf from this walk hits
        # foreign_leaves, merge_log records (step_index, graph_vol, leaf).
        self.foreign_leaves: Optional[set] = None     # leaf set from walk A
        self.merge_log: list = []                     # [(step_index, graph_vol, leaf), ...]
        self.first_merge_step: Optional[int] = None   # step_index of first hit
        self.first_merge_vol: Optional[int] = None    # total_leaf_insertions at first hit
        self._merged_leaves: set = set()              # dedup across all merge hits
        self.collision_count = 0      # path collisions: chosen xj already on chain path
        self.leaf_collision_count = 0 # graph collisions: any leaf already in global_leaves_seen
        self.first_birthday_step: Optional[int] = None  # step_index of first graph/birthday collision
        self.first_birthday_n: Optional[int] = None     # outer n of first graph/birthday collision
        self.collision_log: list = []  # [(step_index, outer_n, graph_vol, count, colliding_xs[:10]), ...]
        self._restart_cursor = 0

        # Adjacency / transition matrices for spectral gap estimation.
        # mat_chain = accepted steps only          (path diagnostic, d~1)
        # mat_graph = full candidate pool per xi   (row-truncated average operator)
        if getattr(self.config, 'spectral_enabled', True):
            _p   = self.p
            _re  = getattr(self.config, 'spectral_report_every', 10)
            _mc  = getattr(self.config, 'spectral_min_collisions', 3)
            _nev = getattr(self.config, 'spectral_n_eigenvalues', 50)
            self.mat_chain = MarkovAdjacencyMatrix(
                p=_p, label="chain",
                use_candidate_pool=False,
                normalize_per_step=False,
                report_every=_re, min_collisions=_mc,
                n_eigenvalues=_nev,
            )
            self.mat_graph = MarkovAdjacencyMatrix(
                p=_p, label="graph",
                use_candidate_pool=True,
                normalize_per_step=True,
                report_every=_re, min_collisions=_mc,
                n_eigenvalues=_nev,
            )
        else:
            self.mat_chain = None
            self.mat_graph = None

        if not self.base_points:
            self.base_points.append((self.current_x, self.current_y))

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

        limit = getattr(self.config, 'log_candidate_limit', 25*infinity) or 25*infinity
        pool_summary = candidate_pool if self.config.log_full_candidates else candidate_pool[:limit]

        return {
            'step_index': rec.step_index,
            'n': rec.n,
            'xi': self._jsonable(rec.xi),
            'm': self._jsonable(rec.m),
            'xj': self._jsonable(rec.xj),
            'xk': self._jsonable(rec.xk),
            'xi_mult': int(getattr(rec, 'xi_mult', -1)),
            'yj_sign': int(getattr(rec, 'yj_sign', 1)),
            'yk_sign': int(getattr(rec, 'yk_sign', 1)),
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

        base = (
            f"\n--- WALK SUMMARY ---\n"
            f"Steps taken: {len(self.history)}\n"
            f"Path accepted: {accepted}\n"
            f"Path collisions (xj revisited on chain): {self.collision_count}\n"
            f"Graph/birthday collisions (leaf already seen): {self.leaf_collision_count}\n"
            f"First birthday collision: step={self.first_birthday_step}  outer_n={self.first_birthday_n}  (graph vol at that point: {self.collision_log[0][2] if self.collision_log else 'none'})\n"
            f"Restarts: {restarts}\n"
            f"Dead ends: {self.dead_end_count}  {dict(self.dead_end_reasons) if self.dead_end_reasons else ''}\n"
            f"Nodes in chosen path: {unique_path_nodes}\n"
            f"Total unique leaves discovered (Graph Volume): {total_leaves}\n"
            f"--------------------"
        )
        extras = []
        if getattr(self, 'mat_chain', None) is not None:
            extras.append(
                f"Chain matrix  : {self.mat_chain.n_atoms} atoms, "
                f"{self.mat_chain.n_steps} steps ingested"
            )
        if getattr(self, 'mat_graph', None) is not None:
            extras.append(
                f"Graph matrix  : {self.mat_graph.n_atoms} atoms, "
                f"{self.mat_graph.n_steps} steps ingested"
            )

        return base + ("\n" + "\n".join(extras) if extras else "")

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
            yj_sign: int = 1,
            yk_sign: int = 1,
            xi_mult: int = -1,
        ):
            deg = self.config.curve_degree
            # Use provided xi_mult if valid, else fall back to double-tangency assumption.
            effective_xi_mult = xi_mult if xi_mult > 0 else (deg - 2)
            if xj is not None and xk is not None:
                relation = f"{effective_xi_mult}*{xi} + {xj} + {xk} - {deg}*∞ = 0"
            elif xj is not None:
                relation = f"{effective_xi_mult}*{xi} + {xj} + ? - {deg}*∞ = 0"
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
                yj_sign=yj_sign,
                yk_sign=yk_sign,
                xi_mult=xi_mult,
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

    def print_relation_summary(self, **kwargs):
        """Prints the shape, column mapping, and rank of the relation matrix."""
        mat, atoms, used = self.relation_matrix()
        print_relation_matrix_summary(mat, atoms, used, **kwargs)

    def _relation_matrix(self, **kwargs):
        cd = getattr(self.config, "curve_degree", 5)
        return build_relation_matrix2(self.history, curve_degree=cd, **kwargs)

    def close_under_involution(self) -> int:
        """Verify the Vieta involution T(xj) = S(m) - (d-2)*xi - xj is a
        genuine 2-cycle on every (xi, xj) pair in the candidate pool.

        For each such pair this checks:
            T(T(xj)) == xj                    (2-cycle / involution)

        Nothing is appended to history.  The relation
            (d-2)*xi + xj + xk - d*inf = 0
        is symmetric in xj and xk by construction, so the xj↔xk swap
        produces the identical algebraic relation — there is no new
        information to record.

        Raises AssertionError immediately on the first violation found.

        Returns the number of (xi, xj) pairs that passed the check.

        NOTE: the involution uses m = xi - xj, which is only valid when the
        search RHS is linear in m (RLINEAR=True).  When RLINEAR=False this
        method returns 0 immediately without checking anything.
        """
        if not RLINEAR:
            print(
                "[close_under_involution] skipped: RLINEAR=False, "
                "m = xi - xj inversion is not valid for a quadratic RHS."
            )
            return 0
        deg = self.config.curve_degree
        xi_mult = deg - 2

        # Collect S_of_m per xi from history records that have it.
        xi_to_S_sym: Dict[Any, Any] = {}
        for rec in self.history:
            if rec.accepted and rec.xi not in xi_to_S_sym:
                S_sym = rec.step.get('S_of_m') if isinstance(rec.step, dict) else None
                if S_sym is not None:
                    xi_to_S_sym[rec.xi] = S_sym

        def _eval_S(xi, m_val):
            S_sym = xi_to_S_sym.get(xi)
            assert S_sym is not None, (
                f"close_under_involution: no S_of_m available for xi={xi}. "
                f"Run the walk first so S_of_m is stored on candidate records."
            )
            Fp = self.base_ring
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
                        f"(xi={xi}) — fiber degenerate at this m, pair skipped"
                    )
                raise

        def _T(xi, xj_val, xi_mult_override=None):
            Fp = self.base_ring
            xi_mult_for_T = xi_mult_override if xi_mult_override is not None else xi_mult
            m_val = Fp(xi) - Fp(xj_val)
            S_val = _eval_S(xi, m_val)
            return Fp(S_val - xi_mult_for_T * Fp(xi) - Fp(xj_val))

        def _check_pair(xi, xj_val):
            Fp = self.base_ring
            S_sym = xi_to_S_sym.get(xi, "<missing>")
            m1 = Fp(xi) - Fp(xj_val)
            S1 = _eval_S(xi, m1)
            partner = _T(xi, xj_val)
            m2 = Fp(xi) - Fp(partner)
            S2 = _eval_S(xi, m2)
            roundtrip = _T(xi, partner)
            if roundtrip != Fp(xj_val):
                raise AssertionError(
                    f"close_under_involution: 2-cycle violated\n"
                    f"  xi          = {xi}\n"
                    f"  S_of_m      = {S_sym}\n"
                    f"  xj          = {xj_val}\n"
                    f"  m1=xi-xj    = {m1}\n"
                    f"  S(m1)       = {S1}\n"
                    f"  T(xj)       = {partner}\n"
                    f"  m2=xi-T(xj) = {m2}\n"
                    f"  S(m2)       = {S2}\n"
                    f"  T(T(xj))    = {roundtrip}  (expected {xj_val})\n"
                )

        n_checked = 0
        n_skipped = 0
        seen_pairs: set = set()

        for rec in self.history:
            if not rec.accepted or rec.xi is None:
                continue
            if rec.xi not in xi_to_S_sym:
                continue
            # Skip involution-closure sentinels — they have no real m-fiber.
            step = rec.step if isinstance(rec.step, dict) else {}
            if step.get('source') == 'involution_closure':
                continue

            pool = list(rec.candidate_pool or [])
            for cand in pool:
                xj_cand = cand.get('xj') if isinstance(cand, dict) else cand
                if xj_cand is None or xj_cand == rec.xi:
                    continue
                key = (rec.xi, xj_cand)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                try:
                    _check_pair(rec.xi, xj_cand)
                    n_checked += 1
                except _FiberPoleError:
                    n_skipped += 1
                    continue

        print(
            f"[close_under_involution] T(T(xj))==xj verified on {n_checked} "
            f"(xi, xj) pairs ({n_skipped} skipped — degenerate fiber/pole) "
            f"across {len(self.history)} history records. History unchanged."
        )
        return n_checked

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

        # Pick uniformly from all candidates.
        pool = candidates

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

            step_no = sum(
                1 for r in self.history
                if not (isinstance(r.step, dict) and r.step.get('source') == 'preferred_injection')
            )
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
                f"\n  Collisions: path={self.collision_count} total  | graph/birthday={self.leaf_collision_count} total  (first expected near √p={sqrt_p:.0f} graph volume)"
                + (f"  [first birthday: step={self.first_birthday_step} n={self.first_birthday_n} vol={self.collision_log[0][2]} xs={self.collision_log[0][4]}]" if self.collision_log else ""),
                f"\n  Totals:    steps_accepted={accepted_count}  restarts={restarts}  dead_ends={self.dead_end_count}",
                f"\n  Leaves:    xj={xj_leaves_count}{xk_leaf_note}  total={step_leaves}  new={new_leaves}  novelty={novelty_ratio:.1%}",
                f"\n  Graph vol: {total_leaves} unique x-coords seen across all leaves  ({collision_frac:.4f}×√p  [√p={sqrt_p:.1f}])",
                f"\n  Rate:      {expansion_rate:.2f} unique leaves/step",
                f"\n  Fertility: {frac_fertile_str} of n-values had F_p roots",
                f"\n{'='*70}\n",
                sep="",
                flush=True,
            )
            print(mixing_one_liner(self, step_no))

        # Spectral reports only at end-of-run (mid-run printing is too slow).
        if getattr(self, 'mat_chain', None) is not None:
            self.mat_chain.maybe_report(len(self.history), force=True)
        if getattr(self, 'mat_graph', None) is not None:
            self.mat_graph.maybe_report(len(self.history), force=True)

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
        if rec.accepted:
            if self.cantor_cache is None and self.p is not None:
                self.cantor_cache = CantorPairCache(
                    self.curve_poly, self.p,
                    curve_degree=self.config.curve_degree,
                    verbose=getattr(self.config, 'verbose', True),
                )
            if self.cantor_cache is not None:
                self.cantor_cache.on_new_step(rec)

        self._append_jsonl_log(rec)

        # Feed all three adjacency matrices.
        if getattr(self, 'mat_chain', None) is not None:
            self.mat_chain.ingest(rec,
                graph_collision_count=self.leaf_collision_count)
        if getattr(self, 'mat_graph', None) is not None:
            self.mat_graph.ingest(rec,
                graph_collision_count=self.leaf_collision_count)

        # Cross-chain merge detection.
        if self.foreign_leaves is not None:
            step_payload = rec.step if isinstance(rec.step, dict) else {}
            # Collect all leaves touched this step: accepted triple + pool candidates.
            this_step_leaves: set = set()
            if rec.xi is not None:
                this_step_leaves.add(rec.xi)
            if rec.xj is not None:
                this_step_leaves.add(rec.xj)
            if rec.xk is not None:
                this_step_leaves.add(rec.xk)
            for cand in (rec.candidate_pool or []):
                if not isinstance(cand, dict):
                    continue
                for key in ("xj", "x", "candidate_x", "xk"):
                    v = cand.get(key)
                    if v is not None:
                        this_step_leaves.add(v)

            # Exclude base points (trivial hits) and already-reported leaves.
            base_xs = {bp[0] for bp in self.base_points if bp and len(bp) > 0}
            hits = (this_step_leaves & self.foreign_leaves) - base_xs - self._merged_leaves
            if hits:
                vol = len(self.global_leaves_seen)
                ins = self.total_leaf_insertions
                label = getattr(self, '_foreign_label', 'A')
                sqrt_p = (self.p ** 0.5) if self.p is not None else float("nan")
                for leaf in sorted(hits, key=int):
                    self._merged_leaves.add(leaf)
                    entry = (rec.step_index, ins, vol, leaf)
                    self.merge_log.append(entry)
                    if self.first_merge_step is None:
                        self.first_merge_step = rec.step_index
                        self.first_merge_vol = ins
                        print(
                            f"\n*** [MERGE] step={rec.step_index}  "
                            f"leaf_insertions={ins} ({ins/sqrt_p:.4f}×√p)  "
                            f"B-vol={vol} ({vol/sqrt_p:.4f}×√p)  "
                            f"leaf={leaf}  ∩ walk-{label} ***\n",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [merge+] step={rec.step_index}  leaf={leaf}  "
                            f"(total merge hits: {len(self.merge_log)})",
                            flush=True,
                        )

        return rec

    def save_leaf_snapshot(self, path: str) -> None:
        """Serialise global_leaves_seen to a JSON file as a list of ints.

        The elements are GF(p) field elements; we lift them to Python ints so
        the file is plain JSON and doesn't depend on Sage being importable.

        Usage (walk A):
            walker_A.run(5000)
            walker_A.save_leaf_snapshot("walk_A_leaves.json")
        """
        leaves = sorted(int(x) for x in self.global_leaves_seen)
        payload = {
            "p": int(self.p) if self.p is not None else None,
            "n_leaves": len(leaves),
            "step_count": len(self.history),
            "leaves": leaves,
        }
        with open(path, "w") as fh:
            _json.dump(payload, fh)
        print(f"[save_leaf_snapshot] wrote {len(leaves)} leaves → {path}")

    def load_foreign_leaves(self, path_or_set, *, label: str = "A") -> int:
        """Load a foreign leaf set to track cross-chain merge events.

        ``path_or_set`` can be:
          - a str/Path pointing to a JSON file written by save_leaf_snapshot()
          - a plain Python set/frozenset of ints or GF(p) elements

        After loading, every call to _store_record checks whether any new leaf
        from the current step is in foreign_leaves and appends to merge_log.

        Returns the number of foreign leaves loaded.
        """
        if isinstance(path_or_set, (str, Path)):
            with open(path_or_set) as fh:
                data = _json.load(fh)
            raw = data["leaves"]
        else:
            raw = path_or_set

        if self.p is not None:
            Fp = self.base_ring
            self.foreign_leaves = {Fp(x) for x in raw}
        else:
            self.foreign_leaves = set(raw)

        # Exclude base points so trivial shared starting regions don't
        # produce false-positive merge signals.
        base_xs = {bp[0] for bp in self.base_points if bp and len(bp) > 0}
        self.foreign_leaves -= base_xs

        self._foreign_label = label
        print(
            f"[load_foreign_leaves] loaded {len(self.foreign_leaves)} leaves "
            f"from walk {label!r}  (current walk vol = {len(self.global_leaves_seen)})"
        )

        # Sanity check: if seeds are truly independent the initial overlap
        # should be zero.  A nonzero value means the two walkers share starting
        # state and the merge metric will be artificially low.
        initial_overlap = len(self.global_leaves_seen & self.foreign_leaves) - len(
            self.global_leaves_seen & self.foreign_leaves & base_xs
        )
        initial_overlap = len((self.global_leaves_seen - base_xs) & self.foreign_leaves)
        if initial_overlap > 0:
            print(
                f"[load_foreign_leaves] WARNING: initial overlap = {initial_overlap} leaves "
                f"— seeds are not fully independent; merge metric will be artificially low."
            )
        else:
            print(f"[load_foreign_leaves] [sanity] initial overlap = 0  (seeds are independent)")

        return len(self.foreign_leaves)

    def cantor_summary(self) -> None:
        """Print the Cantor pair cache collision summary."""
        if self.cantor_cache is None:
            print("[cantor_summary] No cache yet — run some steps first.")
            return
        self.cantor_cache.summary()

    def _get_S_of_m_for_rec(self, rec) -> Optional[Any]:
        """Return the S(m) symbolic rational function for the xi of *rec*.

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

    def _get_fiber_context_for_rec(self, rec):
        """Return (fi, G_poly) for the xi of *rec*, or (None, None) if unavailable.

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

    def inject_preferred_relations(self, accepted_rec) -> int:
        """After an accepted walk step, inject one synthetic relation per preferred
        x-coord by inverting xj = xi - m  =>  m = xi - t.

        For each t in self.preferred_xs:
          m   = xi - t
          xk, actual_xi_mult = compute_xk_from_fiber(xi, m, t, fi, G_poly, deg)

        Uses the actual fiber intersection poly so xi's multiplicity is computed
        correctly at each specific m, rather than assumed to be deg-2.

        A RelationRecord with source='preferred_injection' is appended via
        _store_record so leaf bookkeeping, merge detection, and JSONL logging
        all happen normally.

        Returns the number of relations successfully injected this call.
        """
        if not self.preferred_xs and not self.preferred_xs_budget:
            return 0

        # Synthetic injection inverts xj = xi - m  =>  m = xi - t, which is only
        # valid when the search RHS is linear in m.  When RLINEAR=False the RHS is
        # quadratic and the inversion is undefined, so skip entirely.
        #
        # Read RLINEAR from the search_common module object at call time rather than
        # from this module's namespace.  The double-assignment in search_common
        # (RLINEAR=True then RLINEAR=False) means a `from search_common import *`
        # done at import time may have captured the intermediate True value if import
        # ordering was unlucky.  Hitting the live module attribute is immune to that.
        import search_common as _sc
        if not getattr(_sc, 'RLINEAR', True):
            return 0

        fi, G_poly = self._get_fiber_context_for_rec(accepted_rec)
        if fi is None or G_poly is None:
            # No fiber context available — fall back to S(m) shortcut with default xi_mult.
            # This preserves backward compat for steps that predate fi/G_poly storage.
            assert None, "do not fallback to incorrect code lol"
            #return self._inject_preferred_relations_legacy(accepted_rec)

        xi    = accepted_rec.xi
        Fp    = self.base_ring
        deg   = self.config.curve_degree

        xi_fp = Fp(xi)
        n_injected = 0

        def _do_inject(t_fp):
            nonlocal n_injected
            if t_fp == xi_fp:
                return
            m_fp = xi_fp - t_fp

            try:
                xk_val, inter = compute_xk_from_fiber(
                    xi_fp, m_fp, t_fp, fi, G_poly, deg
                )
            except ZeroDivisionError:
                return

            if xk_val is None:
                return
            xk_fp = Fp(xk_val)

            # Read actual xi multiplicity from the intersection poly roots.
            actual_xi_mult = deg - 2  # fallback
            if inter is not None:
                for r, mult in inter.roots():
                    if Fp(r) == Fp(xi_fp):
                        actual_xi_mult = int(mult)
                        break

            step_payload = {
                'source': 'preferred_injection',
                'preferred_target': int(t_fp),
                'S_of_m': self._get_S_of_m_for_rec(accepted_rec),
                'fi': fi,
                'G_poly': G_poly,
                'intersection_poly': None,  # not stored to avoid memory cost
            }

            rec = self._make_relation(
                step_index=len(self.history),
                n=accepted_rec.n,
                xi=xi_fp,
                m_val=m_fp,
                xj=t_fp,
                xk=xk_fp,
                step=step_payload,
                accepted=True,
                restart=False,
                xi_mult=actual_xi_mult,
            )
            self._injected_xs.add(t_fp)
            self._injected_xs.add(xk_fp)
            for leaf in (t_fp, xk_fp):
                if leaf not in self.global_leaves_seen:
                    self.global_leaves_seen.add(leaf)
                    self.total_leaf_insertions += 1
            self._store_record(rec)
            n_injected += 1

        for t in self.preferred_xs:
            _do_inject(Fp(t))

        exhausted = []
        for t, remaining in list(self.preferred_xs_budget.items()):
            _do_inject(Fp(t))
            self.preferred_xs_budget[t] = remaining - 1
            if self.preferred_xs_budget[t] <= 0:
                exhausted.append(t)

        for t in exhausted:
            del self.preferred_xs_budget[t]

        return n_injected

    def _inject_preferred_relations_legacy(self, accepted_rec) -> int:
        """Fallback for steps that lack fi/G_poly: uses S(m) Vieta shortcut with
        xi_mult assumed to be curve_degree - 2.  Only reached when the step predates
        fi/G_poly storage on the step payload."""
        assert None, "dead code"
        S_sym = self._get_S_of_m_for_rec(accepted_rec)
        if S_sym is None:
            assert None, "canary"
            return 0

        xi    = accepted_rec.xi
        Fp    = self.base_ring
        deg   = self.config.curve_degree
        xi_mult = deg - 2

        xi_fp = Fp(xi)
        n_injected = 0

        def _eval_S(m_fp):
            try:
                num = S_sym.numerator()
                den = S_sym.denominator()
                den_val = Fp(den(m_fp))
                if den_val == Fp(0):
                    return None
                return Fp(num(m_fp)) / den_val
            except Exception:
                raise

        def _do_inject(t_fp, budget_key=None):
            nonlocal n_injected
            if t_fp == xi_fp:
                return
            m_fp = xi_fp - t_fp
            S_val = _eval_S(m_fp)
            if S_val is None:
                return
            xk_fp = S_val - xi_mult * xi_fp - t_fp
            step_payload = {
                'source': 'preferred_injection',
                'preferred_target': int(t_fp),
                'S_of_m': S_sym,
                'intersection_poly': None,
            }
            rec = self._make_relation(
                step_index=len(self.history),
                n=accepted_rec.n,
                xi=xi_fp,
                m_val=m_fp,
                xj=t_fp,
                xk=xk_fp,
                step=step_payload,
                accepted=True,
                restart=False,
                xi_mult=xi_mult,
            )
            self._injected_xs.add(t_fp)
            self._injected_xs.add(xk_fp)
            for leaf in (t_fp, xk_fp):
                if leaf not in self.global_leaves_seen:
                    self.global_leaves_seen.add(leaf)
                    self.total_leaf_insertions += 1
            self._store_record(rec)
            n_injected += 1

        for t in self.preferred_xs:
            _do_inject(Fp(t))

        exhausted = []
        for t, remaining in list(self.preferred_xs_budget.items()):
            _do_inject(Fp(t))
            self.preferred_xs_budget[t] = remaining - 1
            if self.preferred_xs_budget[t] <= 0:
                exhausted.append(t)
        for t in exhausted:
            del self.preferred_xs_budget[t]

        return n_injected

    def _recover_y(self, x_val, explicit_y=None, y_sign: Optional[int] = None):
        """
        Recover a y-coordinate for x_val.

        Rules:
        - If explicit_y is given, keep it exactly.
        - If y_sign is given, choose that branch consistently.
        - If no sign is given, fall back to a deterministic canonical choice.
        """
        if self.p is not None:
            p = int(self.p)

            # Caller already knows the branch: preserve it exactly.
            if explicit_y is not None:
                return self.base_ring(explicit_y)

            rhs = self.curve_poly(x_val)
            if not (hasattr(rhs, "is_square") and rhs.is_square()):
                raise ValueError(f"No y given and f(x) is not a square at x={x_val!r}")

            y_any = self.base_ring(rhs.sqrt())
            y_can = self.base_ring(min(int(y_any), p - int(y_any)))

            if y_sign is None:
                return y_can

            return y_can if int(y_sign) >= 0 else -y_can

        # Non-finite-field path
        if explicit_y is not None:
            return explicit_y

        rhs = self.curve_poly(x_val)
        sq = sqrt(rhs)
        if sq * sq != rhs:
            raise ValueError(f"No rational y at x={x_val!r}")
        return sq

    def _recover_xk(self, step: Dict[str, Any], xi, xj):
        """Return (xk, actual_xi_mult) where actual_xi_mult is the true multiplicity
        of xi in the fiber intersection poly (not assumed to be curve_degree-2).

        Returns (None, -1) when no intersection poly is available in the step dict
        (e.g. xk_head candidates whose sibling poly could not be found in the pool).
        """
        poly = self._intersection_poly_from_step(step)
        if poly is None:
            return None, -1

        # The tower search may have been built on a shifted curve (x -> x + shift,
        # e.g. shift=1 when xi=0) so that the base point avoids x=0.  The returned
        # intersection_poly has roots in the shifted frame.  Substitute x -> x - shift
        # (i.e. x_shifted -> x_orig) before doing any root-matching against the
        # unshifted xi/xj/xk coordinates.
        shift = step.get('shift') if isinstance(step, dict) else None
        if shift is not None:
            try:
                shift_int = int(shift)
            except (TypeError, ValueError):
                shift_int = 0
            if shift_int != 0:
                x_var = poly.parent().gen()
                poly = poly(x_var + shift_int)

        # Determine actual xi multiplicity from the poly roots.
        roots_wm = poly_roots_with_multiplicity(poly)  # [(root, mult), ...]
        actual_xi_mult = 0
        for r, m in roots_wm:
            if r == xi:
                actual_xi_mult = int(m)
                break
        assert actual_xi_mult > 0, (
            f"_recover_xk: xi={xi} is not a root of intersection poly. "
            f"roots={roots_wm}"
        )

        roots = flatten_roots(roots_wm)

        if roots:
            leftovers = []
            xi_count = 0
            xj_count = 0
            for r in roots:
                if r == xi and xi_count < actual_xi_mult:
                    xi_count += 1
                    continue
                if xj is not None and r == xj and xj_count < 1:
                    xj_count += 1
                    continue
                leftovers.append(r)
            if leftovers:
                return leftovers[0], actual_xi_mult

        # Vieta fallback
        if poly.degree() != self.config.curve_degree:
            return None, actual_xi_mult
        known = [xi] * actual_xi_mult
        if xj is not None:
            known.append(xj)
        return missing_root_by_vieta(poly, known), actual_xi_mult

    def _point_check_details(self, x_val, label: str = "x") -> Dict[str, Any]:
        """Return a compact diagnostic record for whether x lifts to a point on C(F_p)."""
        info: Dict[str, Any] = {
            "label": label,
            "x": self._jsonable(x_val),
            "is_fp_point": False,
        }

        if x_val is None:
            info["reason"] = "missing_x"
            return info

        if self.curve_poly is None:
            info["reason"] = "missing_curve_poly"
            return info

        try:
            rhs = self.curve_poly(x_val)
            info["rhs"] = self._jsonable(rhs)

            if hasattr(rhs, "is_square"):
                is_sq = bool(rhs.is_square())
                info["is_square"] = is_sq
                info["is_fp_point"] = is_sq
                if is_sq:
                    try:
                        y_any = self.base_ring(rhs.sqrt())
                        info["sqrt"] = self._jsonable(y_any)
                    except Exception as exc:
                        info["sqrt_error"] = repr(exc)
            else:
                info["reason"] = "rhs_has_no_is_square"
        except Exception as exc:
            info["error"] = repr(exc)

        return info

    def _reject_step_payload(
        self,
        step_payload: Optional[Dict[str, Any]],
        *,
        stage: str,
        reason: str,
        xi,
        n: int,
        current_point=None,
        m_val=None,
        xj=None,
        xk=None,
        chosen=None,
        move_committed: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach a structured rejection record and print a compact explanation."""
        payload = dict(step_payload) if isinstance(step_payload, dict) else {}
        payload["accepted"] = False
        payload["reject_stage"] = stage
        payload["reject_reason"] = reason
        payload["reject_n"] = int(n)
        payload["reject_xi"] = self._jsonable(xi)
        payload["reject_m"] = self._jsonable(m_val)
        payload["reject_xj"] = self._jsonable(xj)
        payload["reject_xk"] = self._jsonable(xk)

        if current_point is not None:
            try:
                cx, cy = current_point
                payload["reject_current_x"] = self._jsonable(cx)
                payload["reject_current_y"] = self._jsonable(cy)
            except Exception:
                payload["reject_current_point"] = self._jsonable(current_point)

        if chosen is not None:
            payload["reject_selected_candidate"] = self._jsonable(chosen)
            if isinstance(chosen, dict):
                payload["reject_selected_source"] = chosen.get("source", None)

        if move_committed is not None:
            payload["reject_move_committed"] = bool(move_committed)

        if extra:
            for k, v in extra.items():
                payload[f"reject_{k}"] = self._jsonable(v)

        if self.config.verbose:
            print(f"[reject] stage={stage} reason={reason} xi={xi} n={n}")
            if m_val is not None or xj is not None or xk is not None:
                print(f"         m={m_val} xj={xj} xk={xk}")
            if current_point is not None:
                try:
                    cx, cy = current_point
                    print(f"         current=({cx}, {cy})")
                except Exception:
                    print(f"         current={current_point}")
            if extra:
                for k, v in extra.items():
                    print(f"         {k} = {self._jsonable(v)}")

        return payload

    def _step_from_candidate_search(self, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
        xi_before = self.current_x
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

        # Leaf bookkeeping is kept even if the relation later gets rejected.
        valid_leaves = {cx for cx in candidate_xs if cx is not None}
        organic_leaves = valid_leaves - self._injected_xs
        organic_already_seen = organic_leaves & self.global_leaves_seen
        colliding_xs = sorted(organic_already_seen)[:10]
        old_leaves_count = len(self.global_leaves_seen)

        if valid_leaves:
            self.global_leaves_seen.update(valid_leaves)
            self.total_leaf_insertions += len(valid_leaves)

        new_leaves_this_step = len(self.global_leaves_seen) - old_leaves_count
        leaf_collisions_this_step = len(organic_already_seen)

        self.leaf_collision_count += leaf_collisions_this_step
        if leaf_collisions_this_step > 0:
            _step_idx = len(self.history)
            _entry = (_step_idx, n, len(self.global_leaves_seen), leaf_collisions_this_step, colliding_xs)
            self.collision_log.append(_entry)
            if self.first_birthday_step is None:
                self.first_birthday_step = _step_idx
                self.first_birthday_n = n

        search_out["step_leaves_found"] = len(valid_leaves)
        search_out["step_leaves_new"] = new_leaves_this_step
        search_out["step_leaf_collisions"] = leaf_collisions_this_step
        search_out["global_leaves_total"] = len(self.global_leaves_seen)
        search_out["global_leaf_collisions"] = self.leaf_collision_count

        if not candidates:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="no_candidates",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={
                    "n_with_roots": search_out.get("n_with_roots"),
                    "n_total": search_out.get("n_total"),
                    "dead_end_reason": search_out.get("dead_end_reason", "unknown"),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, None, None, None,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        # Prefer candidates whose xk is already a valid F_p point.
        fp_valid_candidates = []
        fp_valid_details = []
        for idx, cand in enumerate(candidates):
            if isinstance(cand, dict):
                xk = cand.get("xk")
                diag = self._point_check_details(xk, label="xk")
                fp_valid_details.append((idx, diag))
                if diag.get("is_fp_point", False):
                    fp_valid_candidates.append(cand)

        selection_pool = fp_valid_candidates if fp_valid_candidates else candidates

        chosen = self._choose_candidate_record(
            selection_pool,
            {
                "n": n,
                "step": search_out,
                "current_x": self.current_x,
                "current_y": self.current_y,
            },
        )

        if chosen is None:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="selection_failed",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={
                    "candidate_count": len(candidates),
                    "fp_valid_candidate_count": len(fp_valid_candidates),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, None, None, None,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        if not isinstance(chosen, dict):
            chosen = {"xj": chosen}

        m_val = chosen.get("m")
        xj = chosen.get("xj")
        xk = chosen.get("xk")

        if xj is None and m_val is not None and RLINEAR:
            xj = self._candidate_xj_from_m(self.current_x, m_val)

        if xj is None and "x" in chosen:
            xj = chosen.get("x")

        chosen_xi_mult = int(chosen.get("xi_mult", -1)) if isinstance(chosen, dict) else -1

        _chosen_for_recover = dict(chosen)
        if _chosen_for_recover.get("intersection_poly") is None:
            if _chosen_for_recover.get("source") == "xk_head":
                orig_xj = _chosen_for_recover.get("xk")
                orig_xk = _chosen_for_recover.get("xj")
                for cand in candidates:
                    if (
                        isinstance(cand, dict)
                        and cand.get("source") != "xk_head"
                        and cand.get("xj") == orig_xj
                        and cand.get("xk") == orig_xk
                        and cand.get("intersection_poly") is not None
                    ):
                        _chosen_for_recover["intersection_poly"] = cand["intersection_poly"]
                        break
            elif isinstance(search_out, dict) and search_out.get("intersection_poly") is not None:
                _chosen_for_recover["intersection_poly"] = search_out["intersection_poly"]

        if xj is None:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="missing_xj",
                xi=xi_before,
                n=n,
                current_point=current_point,
                chosen=chosen,
                extra={"candidate_count": len(candidates)},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, None, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        # Validate xj first: if it is not on C(F_p), the move itself is invalid.
        xj_diag = self._point_check_details(xj, label="xj")
        try:
            next_y_for_xj = self._recover_y(xj, y_sign=int(chosen.get("yj_sign", 1)))
        except Exception as exc:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="no_y",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "xj_diagnostic": xj_diag,
                    "error": repr(exc),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        if next_y_for_xj == self.base_ring(0):
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="weierstrass_y0",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={"xj_diagnostic": xj_diag},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        # Recover xk if needed.
        if xk is None:
            if self.base_ring(self.current_x) == self.base_ring(0):
                step_payload = self._reject_step_payload(
                    search_out,
                    stage="candidate_search",
                    reason="xk_unrecoverable_at_xi_zero",
                    xi=xi_before,
                    n=n,
                    current_point=current_point,
                    m_val=m_val,
                    xj=xj,
                    chosen=chosen,
                    extra={"xj_diagnostic": xj_diag},
                )
                rec = self._make_relation(
                    len(self.history), n, xi_before, m_val, xj, None,
                    step_payload, accepted=False, restart=False,
                )
                self._store_record(rec)
                return rec
            xk, recovered_xi_mult = self._recover_xk(_chosen_for_recover, self.current_x, xj)
            if chosen_xi_mult < 0:
                chosen_xi_mult = recovered_xi_mult if recovered_xi_mult is not None else -1

        if xk is None:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="missing_xk_recovery",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                chosen=chosen,
                extra={"xj_diagnostic": xj_diag},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, None,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        xk_diag = self._point_check_details(xk, label="xk")
        if not xk_is_fp_point(xk, self.curve_poly):
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="non_fp_xk",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                    "fp_valid_candidate_count": len(fp_valid_candidates),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        if _chosen_for_recover.get("intersection_poly") is None:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="missing_intersection_poly",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                    "candidate_source": chosen.get("source", None),
                    "candidate_count": len(candidates),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        chosen_target = self._choose_between(
            xj,
            xk,
            {
                "n": n,
                "step": search_out,
                "current_x": self.current_x,
                "current_y": self.current_y,
            },
        )
        if chosen_target is None:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="no_move_choice",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        chosen_sign = int(chosen.get("yj_sign", 1)) if chosen_target == xj else int(chosen.get("yk_sign", 1))
        chosen_diag = xj_diag if chosen_target == xj else xk_diag

        try:
            next_y = self._recover_y(chosen_target, y_sign=chosen_sign)
        except Exception as exc:
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="move_target_no_y",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "chosen_target": self._jsonable(chosen_target),
                    "chosen_target_diagnostic": chosen_diag,
                    "error": repr(exc),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                yj_sign=int(chosen.get("yj_sign", 1)),
                yk_sign=int(chosen.get("yk_sign", 1)),
                xi_mult=chosen_xi_mult,
            )
            self._store_record(rec)
            return rec

        if next_y == self.base_ring(0):
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason="chosen_target_weierstrass_y0",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra={
                    "chosen_target": self._jsonable(chosen_target),
                    "chosen_target_diagnostic": chosen_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                yj_sign=int(chosen.get("yj_sign", 1)),
                yk_sign=int(chosen.get("yk_sign", 1)),
                xi_mult=chosen_xi_mult,
            )
            self._store_record(rec)
            return rec

        # Commit the move only after all relation gates pass.
        self.current_x, self.current_y = chosen_target, next_y
        self.visited_x.add(chosen_target)
        self.xi_visit_count[xi_before] += 1
        unique_xj_new, unique_xj_total = self._annotate_step_counts(
            search_out,
            chosen_target,
            accepted=True,
        )
        if not unique_xj_new:
            self.collision_count += 1

        step_payload = dict(search_out) if isinstance(search_out, dict) else {}
        step_payload["xj_diagnostic"] = xj_diag
        step_payload["xk_diagnostic"] = xk_diag
        step_payload["chosen_target"] = self._jsonable(chosen_target)
        step_payload["chosen_target_diagnostic"] = chosen_diag
        step_payload["move_committed"] = True

        rec = self._make_relation(
            len(self.history), n, xi_before, m_val, xj, xk,
            step_payload, accepted=True, restart=False,
            yj_sign=int(chosen.get("yj_sign", 1)),
            yk_sign=int(chosen.get("yk_sign", 1)),
            xi_mult=chosen_xi_mult,
        )
        rec.candidate_pool = candidates
        rec.selected_candidate = dict(chosen)
        self._store_record(rec)
        self.inject_preferred_relations(rec)
        return rec

    def step(self, n: Optional[int] = None, seed: Optional[int] = None) -> Optional[RelationRecord]:
        n = int(n or (len(self.history) + 1))
        if n < 1:
            n = 1
        if n > self.config.max_n:
            n = self.config.max_n

        if self.search_fn is not None:
            return self._step_from_candidate_search(n=n, seed=seed)

        xi_before = self.current_x
        current_point = (self.current_x, self.current_y)

        step = self.step_factory(self.current_x, n, seed=seed, current_point=current_point)
        m_roots = self._solve_m_roots(step)
        if not m_roots:
            step_payload = self._reject_step_payload(
                step if isinstance(step, dict) else {},
                stage="direct_step",
                reason="no_m_roots",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={"step": self._jsonable(step)},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, None, None, None,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        if not RLINEAR:
            raise RuntimeError(
                "direct step() path (step_factory) does not support RLINEAR=False: "
                "xj cannot be recovered from m via xi - m when the RHS is quadratic. "
                "Supply a search_fn that returns explicit xj values in candidate records."
            )

        xj_candidates = [self._candidate_xj_from_m(self.current_x, m_val) for m_val in m_roots]
        if not xj_candidates:
            step_payload = self._reject_step_payload(
                step if isinstance(step, dict) else {},
                stage="direct_step",
                reason="no_xj_candidates",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={"m_roots": self._jsonable(m_roots)},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, None, None, None,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        valid_leaves = {cx for cx in xj_candidates if cx is not None}
        for _xj in xj_candidates:
            _xk, _ = self._recover_xk(step, self.current_x, _xj)
            if _xk is not None:
                valid_leaves.add(_xk)

        organic_leaves = valid_leaves - self._injected_xs
        organic_already_seen = organic_leaves & self.global_leaves_seen
        colliding_xs = sorted(organic_already_seen)[:10]
        old_leaves_count = len(self.global_leaves_seen)

        if valid_leaves:
            self.global_leaves_seen.update(valid_leaves)
            self.total_leaf_insertions += len(valid_leaves)

        new_leaves_this_step = len(self.global_leaves_seen) - old_leaves_count
        leaf_collisions_this_step = len(organic_already_seen)

        self.leaf_collision_count += leaf_collisions_this_step
        if leaf_collisions_this_step > 0:
            _step_idx = len(self.history)
            _entry = (_step_idx, n, len(self.global_leaves_seen), leaf_collisions_this_step, colliding_xs)
            self.collision_log.append(_entry)
            if self.first_birthday_step is None:
                self.first_birthday_step = _step_idx
                self.first_birthday_n = n

        step_payload = dict(step) if isinstance(step, dict) else {}
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
                f"(novelty {new_leaves_this_step / len(valid_leaves):.1%} if valid_leaves else 0.0%) | "
                f"Graph volume: {len(self.global_leaves_seen)} "
                f"({collision_frac:.3f}×√p)"
            )

        xj = xj_candidates[0]
        m_val = m_roots[0]

        xk, sf_xi_mult = self._recover_xk(step, self.current_x, xj)
        if sf_xi_mult is None:
            sf_xi_mult = -1

        if xk is None:
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="missing_xk_recovery",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                chosen={"source": "direct_step"},
                extra={"step": self._jsonable(step)},
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, None,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        xj_diag = self._point_check_details(xj, label="xj")
        xk_diag = self._point_check_details(xk, label="xk")

        # xj must be a genuine curve point because it can become the next state.
        try:
            next_y_xj = self._recover_y(xj, y_sign=1)
        except Exception as exc:
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="no_y",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "role": "xj"},
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                    "error": repr(exc),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        if next_y_xj == self.base_ring(0):
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="weierstrass_y0",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "role": "xj"},
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        if not xk_is_fp_point(xk, self.curve_poly):
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="non_fp_xk",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "role": "xk"},
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        # Move choice only happens after all relation gates pass.
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
        if chosen is None:
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="no_move_choice",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step"},
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        chosen_sign = 1 if chosen == xj else int(step_payload.get("yk_sign", 1))
        try:
            next_y = self._recover_y(chosen, y_sign=chosen_sign)
        except Exception as exc:
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="move_target_no_y",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "chosen": self._jsonable(chosen)},
                extra={
                    "chosen_target": self._jsonable(chosen),
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                    "error": repr(exc),
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        if next_y == self.base_ring(0):
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="chosen_target_weierstrass_y0",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "chosen": self._jsonable(chosen)},
                extra={
                    "chosen_target": self._jsonable(chosen),
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        if step_payload.get("intersection_poly") is None:
            step_payload = self._reject_step_payload(
                step_payload,
                stage="direct_step",
                reason="missing_intersection_poly",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen={"source": "direct_step", "chosen": self._jsonable(chosen)},
                extra={
                    "xj_diagnostic": xj_diag,
                    "xk_diagnostic": xk_diag,
                },
            )
            rec = self._make_relation(
                len(self.history), n, xi_before, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
                xi_mult=sf_xi_mult,
            )
            self._store_record(rec)
            return rec

        # Commit now that everything passed.
        self.current_x, self.current_y = chosen, next_y
        self.visited_x.add(chosen)
        self.xi_visit_count[xi_before] += 1
        unique_xj_new, unique_xj_total = self._annotate_step_counts(step_payload, chosen, accepted=True)
        if not unique_xj_new:
            self.collision_count += 1

        step_payload["xj_diagnostic"] = xj_diag
        step_payload["xk_diagnostic"] = xk_diag
        step_payload["chosen_target"] = self._jsonable(chosen)
        step_payload["move_committed"] = True

        rec = self._make_relation(
            len(self.history), n, xi_before, m_val, xj, xk,
            step_payload, accepted=True, restart=False,
            xi_mult=sf_xi_mult,
        )
        self._store_record(rec)
        if rec.accepted:
            self.inject_preferred_relations(rec)
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
                    if RLINEAR:
                        print(
                            "  recurrence (symbolic)    = "
                            f"xk(m) = S(m) - {xi_mult}*{rec.xi} - xj(m)"
                            f"      [xj(m) = {rec.xi} - m]"
                        )
                    else:
                        print(
                            "  recurrence (symbolic)    = "
                            f"xk(m) = S(m) - {xi_mult}*{rec.xi} - xj(m)"
                            f"      [xj(m) != xi - m: RLINEAR=False, RHS is quadratic]"
                        )
                else:
                    print(f"  S(m) symbolic            = <unavailable — fi not in step payload>")
                    if RLINEAR:
                        print(
                            "  recurrence preview       = "
                            f"xj = xi - m,  "
                            f"xk = ({total_root_sum}) - ({xi_mult}*xi + xj)"
                            f"  [S={total_root_sum} is numeric, m already substituted]"
                        )
                    else:
                        print(
                            "  recurrence preview       = "
                            f"xk = ({total_root_sum}) - ({xi_mult}*xi + xj)"
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

def xk_is_fp_point(xk_val, G_poly):
    if G_poly is None or xk_val is None:
        return False

    try:
        rhs = G_poly(xk_val)
        return bool(hasattr(rhs, "is_square") and rhs.is_square())
    except Exception:
        return False
