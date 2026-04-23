from __future__ import annotations
import argparse, json, dataclasses, math, random, itertools, bounds, warnings, sys, inspect, json as _json
from dataclasses import dataclass, field
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from sage.all import *
from collections import Counter
from sympy import symbols, expand
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from tate import *
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
from functools import partial
from .relation_matrix import *
from .cantor_cache import *
from .mixing_diagnostics import *
from .adjacency_matrix import *
from search_common import *

load('tower.sage')
load('search7_genus2.sage')



# DEAR AI:  EVERY RAISE IN HERE IS ON PURPOSE ASK ME IF YOU WANT TO REMOVE ONE.

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
    # y-branch signs for xj and xk: +1 = canonical (positive) root, -1 = conjugate.
    # Default to +1 (old behaviour) when signs are not available (e.g. preferred_injection).
    yj_sign: int = 1
    yk_sign: int = 1
    # Actual multiplicity of xi in the fiber intersection poly.
    # Defaults to -1 (sentinel: not set); build_relation_matrix2 falls back to curve_degree-2.
    xi_mult: int = -1
    # Any non-xi roots beyond the first two (xj, xk).  Non-empty only when the
    # intersection polynomial has 3+ roots other than xi — i.e. xi_mult < deg-2.
    # Each entry contributes a +1 column in the relation matrix, exactly like xj/xk.
    extra_roots: List[Any] = field(default_factory=list)

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
    nthermal: int = 30

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
        # Cumulative count of leaf insertions (not deduplicated) — the true
        # "effort" metric for merge-time analysis, independent of step count.
        self.total_leaf_insertions: int = 1  # seed x counts as one insertion
        # Xs that are structurally pre-known (seed x).
        # Birthday collision accounting excludes these.
        self._injected_xs: set = {self.current_x}
        # How many times each x has been stepped *through* as xi (i.e. used as chain state).
        # We avoid re-using high-visit-count nodes as the next xi when fresher candidates exist.
        self.xi_visit_count: Counter = Counter({self.current_x: 1})

        self.history: List[RelationRecord] = []
        self.cantor_cache: Optional[CantorPairCache] = None
        self.dead_end_count = 0
        self.dead_end_reasons: Counter = Counter()  # reason -> count
        self.walk_terminated: bool = False  # stop the run if no fresh xi remains

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
        # Rolling averages for novelty and fertility (window = last 20 accepted steps).
        self._ROLL_WINDOW = 20
        self._roll_novelty: list = []   # novelty_ratio values, capped at _ROLL_WINDOW
        self._roll_fertility: list = [] # frac_fertile values, capped at _ROLL_WINDOW
        self.first_birthday_step: Optional[int] = None  # step_index of first graph/birthday collision
        self.first_birthday_n: Optional[int] = None     # outer n of first graph/birthday collision
        self.collision_log: list = []  # [(step_index, outer_n, graph_vol, count, colliding_xs[:10]), ...]
        self._restart_cursor = 0
        # Xi values that have been fully exhausted (ran as current_x and produced
        # zero novelty or a dead end).  Since each xi's fiber is deterministic,
        # re-running the same xi yields nothing new; we never select it as the
        # next chain state again.
        self.exhausted_xi: set = set()

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
        assert None, "mistaken ai code"

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

    def _xi_is_fresh(self, x) -> bool:
        """Return True only if x has never been used as xi in this walk."""
        return x is not None and x not in self.visited_x and x not in self.exhausted_xi

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
            if self._xi_is_fresh(x):
                return x, y
        return None

    def _restart_from_valid_curve_point(self, *, exclude: Optional[set] = None):
        """Return a restart point (x, y) that is actually on the curve.

        The earlier fallback picked arbitrary x-values from the leaf set and then
        tried to recover y. That can fail in finite-field mode when the sampled x
        is not a curve point. We now only accept candidates that pass the
        point-on-curve test before recovering y.
        """
        exclude = set(exclude or ())

        # First try any unused base point, because those already carry a y-value.
        nxt = self._next_restart_point()
        if nxt is not None:
            return nxt

        # Otherwise scan the global leaf pool for an actual F_p-point.
        pool = list(self.global_leaves_seen - self._injected_xs - exclude)
        self.rng.shuffle(pool)
        for x in pool:
            try:
                info = self._point_check_details(x, label="restart")
            except Exception:
                continue
            if not info.get("is_fp_point", False):
                continue
            try:
                y = self._recover_y(x)
            except Exception:
                continue
            if y == self.base_ring(0):
                continue
            return x, y

        # As a last resort, allow revisiting a known base point if one exists.
        for x, y in self.base_points:
            if x in exclude:
                continue
            try:
                info = self._point_check_details(x, label="restart-fallback")
            except Exception:
                continue
            if info.get("is_fp_point", False):
                try:
                    yy = self._recover_y(x, y)
                except Exception:
                    continue
                if yy != self.base_ring(0):
                    return x, yy

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
        default_xi_mult = deg - 2  # expected for double-tangency fibers

        # Build a per-xi xi_mult map from accepted history records.
        # T(xj) = S(m) - xi_mult*xi - xj is a proper 2-cycle only when
        # xi_mult == deg-2 (exactly two non-xi roots).  For fibers with
        # other multiplicities the swap is not a valid involution and the
        # pair is skipped rather than falsely asserted.
        xi_to_xi_mult: Dict[Any, int] = {}
        for _rec in self.history:
            if _rec.accepted and int(getattr(_rec, 'xi_mult', -1)) > 0:
                xi_to_xi_mult[_rec.xi] = int(_rec.xi_mult)

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
            xi_mult_for_T = (
                xi_mult_override if xi_mult_override is not None
                else xi_to_xi_mult.get(xi, default_xi_mult)
            )
            m_val = Fp(xi) - Fp(xj_val)
            S_val = _eval_S(xi, m_val)
            return Fp(S_val - xi_mult_for_T * Fp(xi) - Fp(xj_val))

        def _check_pair(xi, xj_val):
            Fp = self.base_ring
            S_sym = xi_to_S_sym.get(xi, "<missing>")

            # Involution is only well-defined as a 2-cycle when xi_mult == deg-2.
            eff_mult = xi_to_xi_mult.get(xi, default_xi_mult)
            if eff_mult != default_xi_mult:
                raise _FiberPoleError(
                    f"close_under_involution: xi={xi} has xi_mult={eff_mult} "
                    f"(expected {default_xi_mult}=deg-2); "
                    f"T is not a simple xj↔xk 2-cycle for this fiber, skipping"
                )

            m1 = Fp(xi) - Fp(xj_val)
            S1 = _eval_S(xi, m1)
            partner = _T(xi, xj_val)

            # xj is itself a fixed point of T: xj=xk tangency at the fiber for m1.
            # The 2-cycle is not defined when one endpoint is a tangency point; skip.
            if partner == Fp(xj_val):
                raise _FiberPoleError(
                    f"close_under_involution: xj={xj_val} is a fixed point of T at xi={xi} "
                    f"(xj=xk tangency at m={m1}); skipping pair"
                )

            m2 = Fp(xi) - Fp(partner)
            S2 = _eval_S(xi, m2)
            roundtrip = _T(xi, partner)

            # T(xj) is a fixed point of T: xj=xk tangency at the fiber for m2.
            # T maps xj to a partner that is itself a tangency point, so the
            # roundtrip cannot return to xj.  This is degenerate geometry, not a
            # violation.
            if roundtrip == Fp(partner):
                raise _FiberPoleError(
                    f"close_under_involution: T(xj)={partner} is a fixed point of T at xi={xi} "
                    f"(xj=xk tangency at m={m2}, originating from xj={xj_val}); skipping pair"
                )

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

    def generate_mixed_relations(
        self,
        atoms_to_inject: List[Any],
        *,
        seed_atoms: Optional[set] = None,
        label: str = "mixed",
    ) -> int:
        """Post-step injection: for each accepted history record whose xi is in
        seed_atoms (all accepted records if seed_atoms is None), try injecting
        every atom from atoms_to_inject as xj.

        Because xj = xi - m (RLINEAR), we invert: m = xi - xj_target.
        Then xk = S(m) - (deg-2)*xi - xj_target.
        A relation is recorded iff curve_poly(xk) is a QR in F_p.

        The resulting RelationRecords are stored via _store_record so their
        leaves land in global_leaves_seen exactly like normal search leaves.

        Returns the number of new relation records appended to self.history.
        """
        if not RLINEAR:
            print("[generate_mixed_relations] skipped: RLINEAR=False")
            return 0

        Fp  = self.base_ring
        deg = self.config.curve_degree

        # Normalise seed_atoms filter to a set of base-ring elements (or None).
        if seed_atoms is not None:
            seed_fp = {Fp(a) for a in seed_atoms}
        else:
            seed_fp = None

        atoms_fp = [Fp(a) for a in atoms_to_inject]

        n_added   = 0
        n_skipped = 0

        for rec in list(self.history):
            if not rec.accepted:
                continue
            xi = Fp(rec.xi)
            if seed_fp is not None and xi not in seed_fp:
                continue

            # The formula xk = S(m) - xi_mult*xi - xj uniquely recovers xk
            # only when xi_mult == deg-2 (exactly two non-xi roots).  For any
            # other multiplicity S(m) gives the *sum* of multiple unknowns, so
            # we cannot determine individual roots.  Skip rather than assume.
            rec_xi_mult = int(rec.xi_mult) if (rec.xi_mult is not None and int(rec.xi_mult) > 0) else (deg - 2)
            if rec_xi_mult != deg - 2:
                n_skipped += len(atoms_fp)
                continue
            xi_mult = rec_xi_mult

            # Prefer fi-based fiber evaluation (exact per-m multiplicity check).
            # Fall back to S(m) symbolic shortcut when fi is unavailable.
            fi, G_poly_rec = self._get_fiber_context_for_rec(rec)
            S_sym = self._get_S_of_m_for_rec(rec)
            if fi is None and S_sym is None:
                continue

            for xj_val in atoms_fp:
                if xj_val == xi:
                    continue

                m_val = xi - xj_val   # inversion: xj = xi - m  =>  m = xi - xj

                if fi is not None and G_poly_rec is not None:
                    # Use the actual intersection polynomial at this m — reads
                    # xi_mult from the fiber directly, no assumption needed.
                    try:
                        xk_val, _inter = compute_xk_from_fiber(
                            xi, m_val, xj_val, fi, G_poly_rec, deg
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
                    # xi_mult == deg-2 above.
                    try:
                        dv = Fp(S_sym.denominator()(m_val))
                        if dv == Fp(0):
                            n_skipped += 1
                            continue
                        S_val = Fp(S_sym.numerator()(m_val)) / dv
                    except Exception:
                        n_skipped += 1
                        continue
                    xk_val = Fp(S_val - xi_mult * xi - xj_val)

                # xk must lift to a curve point.
                rhs = self.curve_poly(xk_val)
                if not (hasattr(rhs, "is_square") and rhs.is_square()):
                    n_skipped += 1
                    continue

                # --- leaf bookkeeping (mirrors _step_from_candidate_search) ---
                self._update_leaf_bookkeeping({xj_val, xk_val}, n=rec.n, xi_before=xi)

                # --- build record ---
                cand = {
                    "source": "mixed_injection",
                    "m":  int(m_val),
                    "xj": xj_val,
                    "xk": xk_val,
                }
                step_dict = {
                    "source":            "mixed_injection",
                    "label":             label,
                    "origin_step_index": rec.step_index,
                    "S_of_m":            S_sym,          # kept for close_under_involution
                }
                relation = (
                    f"{xi_mult}*{int(xi)} + {int(xj_val)} + {int(xk_val)}"
                    f" - {deg}*\u221e = 0"
                )
                injected = RelationRecord(
                    step_index       = len(self.history),
                    n                = rec.n,
                    xi               = xi,
                    m                = m_val,
                    xj               = xj_val,
                    xk               = xk_val,
                    relation         = relation,
                    step             = step_dict,
                    accepted         = True,
                    restart          = False,
                    candidate_pool   = [cand],
                    selected_candidate = cand,
                    yj_sign          = 1,
                    yk_sign          = 1,
                    xi_mult          = xi_mult,
                )
                self._store_record(injected)
                n_added += 1

                if self.config.verbose:
                    print(
                        f"[mixed_injection] xi={int(xi)}  xj={int(xj_val)} ({label})"
                        f"  m={int(m_val)}  xk={int(xk_val)}"
                    )

        print(
            f"[generate_mixed_relations] added={n_added}  skipped={n_skipped}"
            f"  (seed_atoms={'all' if seed_fp is None else len(seed_fp)},"
            f" inject_pool={len(atoms_fp)})"
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

    def run(self, num_steps: int, n_values: Optional[Sequence[int]] = None, label: str = "") -> List[RelationRecord]:
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

            step_no = sum(1 for r in self.history if r.accepted)
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

            # Update rolling window for novelty (only when we have leaf data).
            if step_leaves > 0:
                self._roll_novelty.append(novelty_ratio)
                if len(self._roll_novelty) > self._ROLL_WINDOW:
                    self._roll_novelty.pop(0)
            roll_novelty_avg = (
                sum(self._roll_novelty) / len(self._roll_novelty)
                if self._roll_novelty else None
            )

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

            # Update rolling window for fertility.
            if n_with_roots is not None and n_total and n_total > 0:
                self._roll_fertility.append(n_with_roots / n_total)
                if len(self._roll_fertility) > self._ROLL_WINDOW:
                    self._roll_fertility.pop(0)
            roll_fertility_avg = (
                sum(self._roll_fertility) / len(self._roll_fertility)
                if self._roll_fertility else None
            )

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

            walk_tag = f"[walk-{label}] " if label else ""
            print(
                f"\n{'='*70}",
                f"\n{walk_tag}[WALK] STEP {step_no} COMPLETE  (outer n={rec.n})",
                f"\n  Path:      xi → xj  |  xi={rec.xi}  (visited {xi_visits}×)",
                f"\n             xj={xj_str}  (visited {xj_visits}×)  |  xk={xk_str}  |  m={m_str}",
                f"\n  Relation (example):  {rel_str}{rel_annotation}",
                f"\n  This step: accepted={rec.accepted}  path_collision={'YES' if path_collision else 'no'}"
                + (f"  | repeated x-coords this step (relations still novel): {leaf_collisions_this_step}" if leaf_collisions_this_step else ""),
                f"\n  Collisions: path={self.collision_count} total  | repeated x-coords={self.leaf_collision_count} total  (birthday clock ticks when x-coord repeats; first expected near graph vol=√p={sqrt_p:.0f})"
                + (f"  [first birthday: step={self.first_birthday_step} n={self.first_birthday_n} vol={self.collision_log[0][2]} xs={self.collision_log[0][4]}]" if self.collision_log else ""),
                f"\n  Totals:    steps_accepted={accepted_count}  restarts={restarts}  dead_ends={self.dead_end_count}",
                f"\n  Leaves:    xj={xj_leaves_count}{xk_leaf_note}  total={step_leaves}  new={new_leaves}  novelty={novelty_ratio:.1%} (new x-coords / all leaves this step)"
                + (f"  (avg {roll_novelty_avg:.1%} /{len(self._roll_novelty)})" if roll_novelty_avg is not None else ""),
                f"\n  Graph vol: {total_leaves} unique x-coords seen across all leaves  ({collision_frac:.4f}×√p  [√p={sqrt_p:.1f}])",
                f"\n  Rate:      {expansion_rate:.2f} unique leaves/step",
                f"\n  Fertility: {frac_fertile_str} of n-values had F_p roots"
                + (f"  (avg {roll_fertility_avg:.1%} /{len(self._roll_fertility)})" if roll_fertility_avg is not None else ""),
                f"\n{'='*70}\n",
                sep="",
                flush=True,
            )
            print(f"{walk_tag}" + mixing_one_liner(self, step_no))

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
                #assert get_y_unshifted_genus2(x_val) == explicit_y, (x_val, get_y_unshifted_genus2(x_val), explicit_y)
                return self.base_ring(explicit_y)

            rhs = self.curve_poly(x_val)
            if not (hasattr(rhs, "is_square") and rhs.is_square()):
                raise ValueError(f"No y given and f(x) is not a square at x={x_val!r}")

            #y_any = self.base_ring(rhs.sqrt()) # maybe wrong
            y_any = self.base_ring(rhs).sqrt()
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
        #assert get_y_unshifted_genus2(x_val) == sq, (x_val, get_y_unshifted_genus2(x_val), sq)
        return sq

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
                        raise
            else:
                info["reason"] = "rhs_has_no_is_square"
        except Exception as exc:
            info["error"] = repr(exc)
            raise

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
                raise

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
                    raise
            if extra:
                for k, v in extra.items():
                    print(f"         {k} = {self._jsonable(v)}")

        return payload

    def _derive_relation_from_intersection_poly(self, step: Dict[str, Any], xi):
        """
        Return (xj, xk, xi_mult, poly) derived only from the intersection polynomial.

        This is the only place xj/xk/xi_mult should be trusted from.
        """
        poly = self._intersection_poly_from_step(step)
        #poly = self._intersection_poly_from_step(poly_src, xj=chosen.get("xj"), xk=chosen.get("xk"))
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

        xi_mult = 0
        leftovers = []
        for r, m in roots_wm:
            if r == xi:
                xi_mult += int(m)
            else:
                leftovers.extend([r] * int(m))

        if xi_mult <= 0:
            return None

        if not leftovers:
            return None  # All roots are xi; no usable relation.

        # Dispatch on the number of non-xi roots.  No multiplicity pattern is
        # assumed in advance — the actual root list drives the relation.
        if len(leftovers) == 1:
            # Tangency: one non-xi root.  Fold one copy of xi into the xk slot
            # so that xk==xi and xi_mult is decremented by one.  The relation
            # matrix adds +1 to the xi column for xk, giving the right total.
            xj = leftovers[0]
            xk = xi
            xi_mult -= 1
            extra_roots = []
        elif len(leftovers) == 2:
            xj, xk = leftovers[0], leftovers[1]
            extra_roots = []
        else:
            # General case: 3+ non-xi roots (xi has lower-than-expected multiplicity).
            # xj/xk carry the first two; extra_roots carries the remainder.
            # Each extra root contributes +1 in the relation matrix, same as xj/xk.
            xj = leftovers[0]
            xk = leftovers[1]
            extra_roots = leftovers[2:]

        return xj, xk, xi_mult, poly, extra_roots

    def _recover_xk(self, step: Dict[str, Any], xi, xj):
        """
        Compatibility wrapper.

        The new source of truth is the intersection polynomial; this just
        extracts xk and xi_mult from it.
        """
        derived = self._derive_relation_from_intersection_poly(step, xi)
        if derived is None:
            return None, -1
        _xj, xk, xi_mult, _poly, _extra = derived
        return xk, xi_mult

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
        """
        Build a RelationRecord.

        Accepted records must be derivable from the intersection polynomial.
        Rejected records may carry whatever diagnostic payload they have.
        """
        derived = self._derive_relation_from_intersection_poly(step, xi)
        extra_roots: List[Any] = []

        if accepted:
            if derived is None:
                raise AssertionError(
                    f"[MAKE_RELATION] accepted record missing usable intersection polynomial "
                    f"at step={step_index} xi={xi} xj={xj} xk={xk}"
                )
            xj, xk, xi_mult, poly, extra_roots = derived
        else:
            # For rejected rows, prefer derived geometry when it exists.
            if derived is not None:
                dxj, dxk, dmult, poly, dextra = derived
                if xj is None:
                    xj = dxj
                if xk is None:
                    xk = dxk
                if xi_mult <= 0:
                    xi_mult = dmult
                if not extra_roots:
                    extra_roots = list(dextra)

        deg = self.config.curve_degree
        effective_xi_mult = xi_mult if xi_mult > 0 else (deg - 2)

        # Build relation string generically from whatever roots the poly gave us.
        all_others = []
        if xj is not None:
            all_others.append(xj)
        if xk is not None:
            all_others.append(xk)
        all_others.extend(extra_roots)
        if all_others:
            others_str = " + ".join(str(r) for r in all_others)
            if xj is not None and xk is None and not extra_roots:
                others_str += " + ?"
            relation = f"{effective_xi_mult}*{xi} + {others_str} - {deg}*\u221e = 0"
        elif xj is not None:
            relation = f"{effective_xi_mult}*{xi} + {xj} + ? - {deg}*\u221e = 0"
        else:
            relation = "no xj"

        clean_step = {}
        if isinstance(step, dict):
            bad_keys = {
                "raw_mumford_residues",
                "precomputed_residues",
                "context",
                "candidates",
                "candidate_records",
            }
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
            extra_roots=list(extra_roots),
        )

    def _step_from_candidate_search(self, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
        xi = self.current_x
        pt = (self.current_x, self.current_y)
        # Mark xi exhausted immediately — its fiber is deterministic, so any
        # subsequent run as current_x would produce no new information.
        self.exhausted_xi.add(xi)

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

            xi_mult = 0
            others = []
            for r, m in roots_wm:
                if r == xi:
                    xi_mult += int(m)
                else:
                    others.extend([r] * int(m))

            if xi_mult <= 0:
                print(roots_wm, xi, poly)
                raise ValueError
                return None

            # No multiplicity pattern assumed — derive directly from the root list.
            if not others:
                return None
            if len(others) == 1:
                xj = others[0]
                xk = xi
                xi_mult -= 1  # fold one copy of xi into the xk slot
                extra_roots = []
            elif len(others) == 2:
                xj, xk = others[0], others[1]
                extra_roots = []
            else:
                # 3+ non-xi roots: xj/xk get the first two, rest go in extra_roots
                xj = others[0]
                xk = others[1]
                extra_roots = others[2:]

            return xj, xk, xi_mult, poly, extra_roots

        def reject(reason, *, m_val=None, xj=None, xk=None, chosen=None, extra=None):
            step_payload = self._reject_step_payload(
                search_out,
                stage="candidate_search",
                reason=reason,
                xi=xi,
                n=n,
                current_point=pt,
                m_val=m_val,
                xj=xj,
                xk=xk,
                chosen=chosen,
                extra=extra or {},
            )
            rec = self._make_relation(
                len(self.history), n, xi, m_val, xj, xk,
                step_payload, accepted=False, restart=False,
            )
            self._store_record(rec)
            return rec

        # --- search ---
        raw = self._call_search_fn(n=n, seed=seed, current_point=pt)
        search_out = self._normalize_candidate_output(raw)
        C = list(search_out.get("candidate_records") or search_out.get("candidates") or [])
        X = {x for x in search_out.get("candidate_xs", set()) if x is not None}

        # --- leaf bookkeeping ---
        organic = X - self._injected_xs
        new_leaves_count = len(organic - self.global_leaves_seen)

        # During thermalization, zero novelty means we haven't mixed yet — escape.
        # Post-thermalization, zero novelty is normal (graph is saturated near this xi);
        # fall through to candidate selection and commit as a regular step.
        if new_leaves_count == 0 and len(X) > 0 and n < self.config.nthermal:
            self.dead_end_count += 1
            self.dead_end_reasons["zero_novelty_thermal"] += 1
            self.exhausted_xi.add(xi)
            rec = reject(
                "zero_novelty_thermal",
                extra={
                    "leaves_found": len(X),
                    "thermal_threshold": self.config.nthermal,
                    "thermalized": False,
                },
            )
            self._restart_after_dead_end(
                xi=xi,
                n=n,
                reason="zero_novelty_thermal",
                current_point=pt,
            )
            return rec

        _, new_leaves_this_step, leaf_collisions_this_step = \
            self._update_leaf_bookkeeping(X, n=n, xi_before=xi)

        search_out.update({
            "step_leaves_found": len(X),
            "step_leaves_new": new_leaves_this_step,
            "step_leaf_collisions": leaf_collisions_this_step,
            "global_leaves_total": len(self.global_leaves_seen),
            "global_leaf_collisions": self.leaf_collision_count,
        })

        # --- dead end ---
        if not C:
            self.dead_end_count += 1
            r = search_out.get("dead_end_reason", "unknown")
            self.dead_end_reasons[r] += 1
            rec = reject("no_candidates", extra={"dead_end_reason": r})

            if self.config.restart_on_dead_end:
                nxt = self._restart_from_valid_curve_point(exclude={xi})
                if nxt:
                    self.current_x, self.current_y = nxt
                    self.visited_x.add(nxt[0])
                else:
                    if self.config.verbose:
                        print(
                            "[dead_end restart] no valid curve point found in base points or leaf pool; "
                            "leaving current state unchanged."
                        )
            return rec

        # --- choose candidate ---
        def is_fp(c):
            if not isinstance(c, dict):
                return False
            return self._point_check_details(c.get("xk"), "xk").get("is_fp_point", False)

        pool = [c for c in C if is_fp(c)] or C
        #pool = self._prefer_unvisited_candidates(pool) # Use the built-in tiering
        pool = list(pool)  # Make a copy so we can pop from it

        valid_candidate_found = False

        while pool:
            chosen = self._choose_candidate_record(
                pool,
                {"n": n, "step": search_out, "current_x": xi, "current_y": pt[1]},
            )

            if chosen is None:
                return reject("selection_failed", extra={"candidate_count": len(C)})

            if not isinstance(chosen, dict):
                chosen = {"xj": chosen}

            # Geometry must come from the intersection polynomial.
            poly_src = dict(search_out)
            poly_src.update(chosen)

            try:
                derived = _derive_from_poly(poly_src)
                if derived is None:
                    pool.remove(chosen); continue

                xj, xk, xi_mult, poly, extra_roots = derived

                # If the candidate carried explicit roots, require them to match the poly.
                cand_xj = chosen.get("xj")
                cand_xk = chosen.get("xk")
                if cand_xj is not None and cand_xk is not None and {cand_xj, cand_xk} != {xj, xk}:
                    pool.remove(chosen); continue

                m = chosen.get("m")

                # --- validate xj ---
                try:
                    yj = self._recover_y(xj, y_sign=int(chosen.get("yj_sign", 1)))
                except Exception as e:
                    pool.remove(chosen); continue

                if yj == self.base_ring(0):
                    pool.remove(chosen); continue

                if not xk_is_fp_point(xj, self.curve_poly):
                    pool.remove(chosen); continue

                # --- validate xk ---
                if not xk_is_fp_point(xk, self.curve_poly):
                    pool.remove(chosen); continue

                try:
                    yk = self._recover_y(xk, y_sign=int(chosen.get("yk_sign", 1)))
                except Exception as e:
                    pool.remove(chosen); continue

                if yk == self.base_ring(0):
                    pool.remove(chosen); continue

                # --- choose move ---
                tgt = self._choose_between(
                    xj, xk,
                    {"n": n, "step": search_out, "current_x": xi, "current_y": pt[1]},
                )
                if tgt is None:
                    pool.remove(chosen); continue

                # Never repeat an xi within the same walk.
                if not self._xi_is_fresh(tgt):
                    pool.remove(chosen); continue

                sign = int(chosen.get("yj_sign", 1)) if tgt == xj else int(chosen.get("yk_sign", 1))
                y = self._recover_y(tgt, y_sign=sign)
                if y == self.base_ring(0):
                    pool.remove(chosen); continue

                # If we made it here, everything is valid!
                valid_candidate_found = True
                break

            except Exception:
                    # Catch any _recover_y or internal errors for this specific candidate
                    pool.remove(chosen)
                    continue
                # --- END OF EXISTING VALIDATION LOGIC ---

        if not valid_candidate_found:
            self.dead_end_count += 1
            self.dead_end_reasons["all_candidates_failed_validation"] += 1
            rec = reject("all_candidates_failed_validation", extra={"candidate_count": len(C)})
            if self.config.restart_on_dead_end and not self.walk_terminated:
                nxt = self._restart_after_dead_end(
                    xi=xi,
                    n=n,
                    reason="all_candidates_failed_validation",
                    current_point=pt,
                )
                if nxt is None:
                    self.walk_terminated = True
            return rec

        # --- commit ---
        self.current_x, self.current_y = tgt, y
        self.visited_x.add(tgt)
        self.xi_visit_count[xi] += 1

        new_flag, _ = self._annotate_step_counts(search_out, tgt, accepted=True)
        if not new_flag:
            self.collision_count += 1

        payload = dict(search_out)
        payload["move_committed"] = True
        payload["intersection_poly"] = poly

        rec = self._make_relation(
            len(self.history), n, xi, m, xj, xk,
            payload, accepted=True, restart=False,
            yj_sign=int(chosen.get("yj_sign", 1)),
            yk_sign=int(chosen.get("yk_sign", 1)),
            xi_mult=xi_mult,
        )

        assert rec.xi_mult > 0
        rec.candidate_pool = C
        rec.selected_candidate = dict(chosen)

        self._store_record(rec)
        return rec

    def _clamp_step_n(self, n: Optional[int]) -> int:
        n = int(n or (len(self.history) + 1))
        if n < 1:
            return 1
        if n > self.config.max_n:
            return self.config.max_n
        return n

    def _store_relation_record(
        self,
        *,
        step_index: int,
        n: int,
        xi,
        m_val=None,
        xj=None,
        xk=None,
        step_payload=None,
        accepted: bool,
        restart: bool = False,
        yj_sign: int = 1,
        yk_sign: int = 1,
        xi_mult: int = -1,
    ):
        rec = self._make_relation(
            step_index, n, xi, m_val, xj, xk,
            step_payload or {},
            accepted=accepted,
            restart=restart,
            yj_sign=yj_sign,
            yk_sign=yk_sign,
            xi_mult=xi_mult,
        )
        self._store_record(rec)
        return rec

    def _update_leaf_bookkeeping(self, leaves, *, n: int, xi_before):
        valid_leaves = {cx for cx in leaves if cx is not None}
        organic = valid_leaves - self._injected_xs
        organic_already_seen = organic & self.global_leaves_seen
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
            self.collision_log.append(
                (_step_idx, n, len(self.global_leaves_seen), leaf_collisions_this_step, colliding_xs)
            )
            if self.first_birthday_step is None:
                self.first_birthday_step = _step_idx
                self.first_birthday_n = n

        return valid_leaves, new_leaves_this_step, leaf_collisions_this_step

    def _make_direct_step_payload(self, step, *, step_leaves_found, step_leaves_new, step_leaf_collisions):
        payload = dict(step) if isinstance(step, dict) else {}
        payload["step_leaves_found"] = step_leaves_found
        payload["step_leaves_new"] = step_leaves_new
        payload["step_leaf_collisions"] = step_leaf_collisions
        payload["global_leaves_total"] = len(self.global_leaves_seen)
        payload["global_leaf_collisions"] = self.leaf_collision_count
        return payload

    def _reject_direct_step(self, *, step_payload, stage, reason, xi, n, current_point, m_val=None, xj=None, xk=None, chosen=None, extra=None, xi_mult=-1):
        payload = self._reject_step_payload(
            step_payload if isinstance(step_payload, dict) else {},
            stage=stage,
            reason=reason,
            xi=xi,
            n=n,
            current_point=current_point,
            m_val=m_val,
            xj=xj,
            xk=xk,
            chosen=chosen,
            extra=extra or {},
        )
        return self._store_relation_record(
            step_index=len(self.history),
            n=n,
            xi=xi,
            m_val=m_val,
            xj=xj,
            xk=xk,
            step_payload=payload,
            accepted=False,
            restart=False,
            xi_mult=xi_mult,
        )

    def _accept_direct_step(self, *, step_payload, n, xi, m_val, xj, xk, yj_sign=1, yk_sign=1, xi_mult=-1):
        rec = self._store_relation_record(
            step_index=len(self.history),
            n=n,
            xi=xi,
            m_val=m_val,
            xj=xj,
            xk=xk,
            step_payload=step_payload,
            accepted=True,
            restart=False,
            yj_sign=yj_sign,
            yk_sign=yk_sign,
            xi_mult=xi_mult,
        )
        assert rec.xi_mult > 0
        return rec

    def _step_direct(self, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
        xi_before = self.current_x
        current_point = (self.current_x, self.current_y)

        step = self.step_factory(self.current_x, n, seed=seed, current_point=current_point)
        m_roots = self._solve_m_roots(step)

        if not m_roots:
            return self._reject_direct_step(
                step_payload=step if isinstance(step, dict) else {},
                stage="direct_step",
                reason="no_m_roots",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={"step": self._jsonable(step)},
            )

        if not RLINEAR:
            raise RuntimeError(
                "direct step() path (step_factory) does not support RLINEAR=False: "
                "xj cannot be recovered from m via xi - m when the RHS is quadratic. "
                "Supply a search_fn that returns explicit xj values in candidate records."
            )

        xj_candidates = [self._candidate_xj_from_m(self.current_x, m_val) for m_val in m_roots]
        if not xj_candidates:
            return self._reject_direct_step(
                step_payload=step if isinstance(step, dict) else {},
                stage="direct_step",
                reason="no_xj_candidates",
                xi=xi_before,
                n=n,
                current_point=current_point,
                extra={"m_roots": self._jsonable(m_roots)},
            )

        valid_leaves = {cx for cx in xj_candidates if cx is not None}
        xk_per_xj = []
        for xj_c in xj_candidates:
            xk_c, _ = self._recover_xk(step, self.current_x, xj_c)
            xk_per_xj.append(xk_c)
            if xk_c is not None:
                valid_leaves.add(xk_c)

        missing_xj = [m for m, xj_c in zip(m_roots, xj_candidates) if xj_c is None]
        assert not missing_xj, (
            f"[ASSERT-a direct] {len(missing_xj)}/{len(m_roots)} m-roots produced no xj leaf "
            f"at step={len(self.history)+1} xi={xi_before}. missing m-roots: {_missing_xj}"
        )
        assert valid_leaves, (
            f"[ASSERT-a direct] leaf set is empty after processing {len(m_roots)} m-roots "
            f"at step={len(self.history)+1} xi={xi_before}."
        )

        missing_xk = [(m, xj_c) for m, xj_c, xk_c in zip(m_roots, xj_candidates, xk_per_xj) if xk_c is None]
        if missing_xk:
            print(
                f"[WARN-a direct] {len(missing_xk)}/{len(m_roots)} xj leaves have no recoverable xk "
                f"at step={len(self.history)+1} xi={xi_before}: "
                f"(m, xj) pairs without xk = {missing_xk[:5]}"
                + (" ..." if len(missing_xk) > 5 else "")
            )

        print(
            f"[LEAVES direct] step={len(self.history)+1} xi={xi_before} "
            f"m-roots={len(m_roots)} xj-leaves={len(xj_candidates)} "
            f"xk-recovered={sum(1 for xk_c in xk_per_xj if xk_c is not None)} "
            f"total-leaves={len(valid_leaves)}"
        )

        valid_leaves, new_leaves_this_step, leaf_collisions_this_step = self._update_leaf_bookkeeping(
            valid_leaves, n=n, xi_before=xi_before
        )

        step_payload = self._make_direct_step_payload(
            step,
            step_leaves_found=len(valid_leaves),
            step_leaves_new=new_leaves_this_step,
            step_leaf_collisions=leaf_collisions_this_step,
        )

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
            return self._reject_direct_step(
                step_payload=step_payload,
                stage="direct_step",
                reason="missing_xk_recovery",
                xi=xi_before,
                n=n,
                current_point=current_point,
                m_val=m_val,
                xj=xj,
                chosen={"source": "direct_step"},
                extra={"step": self._jsonable(step)},
                xi_mult=sf_xi_mult,
            )

        xj_diag = self._point_check_details(xj, label="xj")
        xk_diag = self._point_check_details(xk, label="xk")

        try:
            next_y_xj = self._recover_y(xj, y_sign=1)
        except Exception as exc:
            raise
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        if next_y_xj == self.base_ring(0):
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        if not xk_is_fp_point(xk, self.curve_poly):
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

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
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        chosen_sign = 1 if chosen == xj else int(step_payload.get("yk_sign", 1))
        try:
            next_y = self._recover_y(chosen, y_sign=chosen_sign)
        except Exception as exc:
            raise
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        if next_y == self.base_ring(0):
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        if not self._xi_is_fresh(chosen):
            return self._reject_direct_step(
                step_payload=step_payload,
                stage="direct_step",
                reason="repeated_xi_forbidden",
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
                xi_mult=sf_xi_mult,
            )

        if step_payload.get("intersection_poly") is None:
            return self._reject_direct_step(
                step_payload=step_payload,
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
                xi_mult=sf_xi_mult,
            )

        self.current_x, self.current_y = chosen, next_y
        self.visited_x.add(chosen)
        self.xi_visit_count[xi_before] += 1

        unique_xj_new, _ = self._annotate_step_counts(step_payload, chosen, accepted=True)
        if not unique_xj_new:
            self.collision_count += 1

        step_payload["xj_diagnostic"] = xj_diag
        step_payload["xk_diagnostic"] = xk_diag
        step_payload["chosen_target"] = self._jsonable(chosen)
        step_payload["move_committed"] = True

        rec = self._accept_direct_step(
            step_payload=step_payload,
            n=n,
            xi=xi_before,
            m_val=m_val,
            xj=xj,
            xk=xk,
            yj_sign=1,
            yk_sign=int(step_payload.get("yk_sign", 1)),
            xi_mult=sf_xi_mult,
        )

        assert self.history[-1] is rec, (
            f"[ASSERT-b direct] _store_record did not append rec at step={rec.step_index} xi={xi_before}"
        )
        assert rec.accepted, (
            f"[ASSERT-b direct] accepted record has accepted=False at step={rec.step_index}"
        )
        assert rec.xj is not None, (
            f"[ASSERT-b direct] accepted record has xj=None at step={rec.step_index} xi={xi_before}"
        )
        assert rec.xk is not None, (
            f"[ASSERT-b direct] accepted record has xk=None at step={rec.step_index} xi={xi_before}"
        )
        print(
            f"[STORED direct] step_index={rec.step_index} xi={rec.xi} xj={rec.xj} xk={rec.xk} "
            f"history_len={len(self.history)} accepted=True"
        )
        return rec

    def step(self, n: Optional[int] = None, seed: Optional[int] = None) -> Optional[RelationRecord]:
        if self.walk_terminated:
            return None
        n = self._clamp_step_n(n)
        if self.search_fn is not None:
            rec = self._step_from_candidate_search(n=n, seed=seed)
        else:
            rec = self._step_direct(n=n, seed=seed)

        if rec is not None and not rec.accepted and not self.walk_terminated:
            # Never allow the walk to stall on the same xi after a rejection.
            # If the underlying step did not already move to a fresh restart
            # point, force one here.
            if self.current_x == rec.xi:
                step_dict = rec.step if isinstance(rec.step, dict) else {}
                reason = step_dict.get("reason", "rejected")
                nxt = self._restart_after_dead_end(
                    xi=rec.xi,
                    n=rec.n,
                    reason=reason,
                    current_point=(rec.xi, self.current_y),
                )
                if nxt is None:
                    self.walk_terminated = True
        return rec

    def _intersection_poly_from_step(self, step: Dict[str, Any], *, xj=None, xk=None):
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
        if xj is not None or xk is not None:
            for cand in pools:
                if not isinstance(cand, dict):
                    continue
                cand_xj = cand.get("xj")
                cand_xk = cand.get("xk")
                if xj is not None and cand_xj == xj:
                    poly = _cand_poly(cand)
                    if poly is not None:
                        return poly
                if xk is not None and cand_xk == xk:
                    poly = _cand_poly(cand)
                    if poly is not None:
                        return poly

        # 2b) any candidate with a poly
        for cand in pools:
            poly = _cand_poly(cand)
            if poly is not None:
                return poly

        return None

    def _restart_after_dead_end(self, *, xi, n, reason, current_point=None):
        # Mark the incoming xi as exhausted — its fiber is deterministic so
        # re-running it as chain state will produce nothing new.
        self.exhausted_xi.add(xi)

        # Build candidate pool: base_points first, then accumulated leaves.
        # Exclude any xi that is already exhausted so we never loop back to it.
        candidates = [
            (x, y) for x, y in self.base_points
            if x is not None and y is not None and self._xi_is_fresh(x)
        ]

        # If base_points is only the current stuck point (or empty), augment from
        # global_leaves_seen — the actual visited graph.  This is the escape hatch
        # for the single-base-point case: without it the cursor just loops back to
        # the same xi every time.
        if len(candidates) <= 1:
            # Prefer leaves that have never been used as xi (freshest first).
            never_xi = self.global_leaves_seen - self.exhausted_xi - self.visited_x
            pool_order = sorted(never_xi, key=lambda lx: self.xi_visit_count.get(lx, 0))
            # Fall back to any non-exhausted leaf if the fresh pool is empty.
            if not pool_order:
                pool_order = sorted(
                    self.global_leaves_seen - self.exhausted_xi - self.visited_x,
                    key=lambda lx: self.xi_visit_count.get(lx, 0),
                )
            for lx in pool_order:
                if not self._xi_is_fresh(lx):
                    continue
                try:
                    ly = self._recover_y(lx, None)
                    if ly is not None:
                        candidates.append((lx, ly))
                        if len(candidates) >= 32:   # enough variety, stop early
                            break
                except Exception:
                    continue

        # Only fresh xi values are allowed for restarts.
        candidates = [(x, y) for x, y in candidates if self._xi_is_fresh(x)]

        if not candidates:
            self.walk_terminated = True
            if self.config.verbose:
                print(
                    f"[restart] no fresh restart point available after dead end: "
                    f"reason={reason}  exhausted_xi={len(self.exhausted_xi)}  visited_x={len(self.visited_x)}"
                )
            return None

        x, y = candidates[self._restart_cursor % len(candidates)]
        self._restart_cursor += 1
        self.current_x, self.current_y = x, y
        self.visited_x.add(x)
        self.xi_visit_count[x] += 1

        if self.config.verbose:
            print(
                f"[restart] dead-end escape -> ({x}, {y})  reason={reason}  n={n}  "
                f"exhausted_xi={len(self.exhausted_xi)}"
            )

        return (x, y)

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
        #poly = self._intersection_poly_from_step(step)
        poly = self._intersection_poly_from_step(poly_src, xj=chosen.get("xj"), xk=chosen.get("xk"))
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
        # xi_mult is read from the record (set by _make_relation from the actual poly);
        # falling back to curve_degree-2 only when unavailable (e.g. old records).
        if rec.xi is not None and rec.xj is not None:
            try:
                lc = poly.leading_coefficient()
                monic = poly / lc
                coeffs = monic.list()  # low-to-high coefficients
                deg = int(monic.degree())
                a_d_minus_1 = coeffs[deg - 1] if deg - 1 < len(coeffs) else monic.parent().base_ring()(0)
                total_root_sum = -a_d_minus_1

                xi_mult = int(getattr(rec, 'xi_mult', -1) or -1)
                if xi_mult <= 0:
                    xi_mult = curve_degree - 2
                extra_roots_diag = list(getattr(rec, 'extra_roots', []) or [])
                known_roots_list = [xi_mult * rec.xi, rec.xj] + extra_roots_diag
                known_sum = sum(known_roots_list)
                xk_vieta = total_root_sum - (xi_mult * rec.xi + rec.xj + sum(extra_roots_diag))

                print(f"  monic sum-of-roots       = {total_root_sum}  (S evaluated at this m)")
                print(f"  xi_mult (from record)    = {xi_mult}")
                if extra_roots_diag:
                    print(f"  extra_roots              = {extra_roots_diag}")
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
        raise
        return False



def make_project_markov_search_fn(
    *,
    coeffs_genus2=None,
    base_points=None,
    p: Optional[int] = None,
    prime_pool=None,
    sconf=None,
    num_subsets: Optional[int] = None,
    num_workers: Optional[int] = None,
    debug: bool = False,
    max_n: int = 80,
    precomputed_residues=None,
    all_found_x=None, pool=None, chunk_size=8
):
    """Build a search function that rebuilds the tower for every current x."""
    coeffs_genus2 = coeffs_genus2 if coeffs_genus2 is not None else resolve_project_symbol('COEFFS_GENUS2', required=True)
    p = FINITE_FIELD
    prime_pool = PRIME_POOL
    num_workers = 20
    all_found_x = set()

    def search_fn(xi=None, current_x=None, n=None, seed=None, current_point=None, walker=None, **kwargs):
        yfun = get_y_unshifted_genus2
        if current_point is not None and isinstance(current_point, (tuple, list)) and len(current_point) >= 2:
            x_here, y_here = current_point[0], current_point[1]
        else:
            x_here = xi if xi is not None else current_x
            y_here = None
            if walker is not None and hasattr(walker, 'current_y'):
                y_here = walker.current_y
            if y_here is None:
                if yfun is not None:
                    y_here = yfun(x_here)
        assert yfun(x_here) == y_here, (yfun(x_here), y_here, current_point, current_x)

        ctx = build_project_tower_context_for_point(
            x_here,
            y_here,
            coeffs_genus2=coeffs_genus2,
            base_points=base_points if base_points is not None else getattr(walker, 'base_points', None),
            p=p,
            debug=debug,
        )

        assert ctx['search_rhs_list'] != ['INF'], ctx

        if debug:
            for k in ctx:
                print(k, ctx[k])

        n0 = int(n or 1)
        vecs = generate_ff_search_vectors(1)
        if not vecs:
            vecs = [tuple([n0])]

        raw = _run_markov_mumford_search_for_point(
            cd=ctx['cd'],
            current_sections=ctx['current_sections'],
            prime_pool=prime_pool,
            vecs=vecs,
            rhs_list=ctx['search_rhs_list'],
            shift=ctx['shift'],
            rationality_test_func=ctx['testfunc'],
            coeffs_genus2=coeffs_genus2,
            tower_data=ctx['tower_for_mumford'],
            num_workers=num_workers,
            debug=debug,
            x_b=None,
            shifted_coeffs=None, pool=pool, chunk_size=chunk_size
        )

        norm = _normalize_markov_mumford_result(raw, fallback_step=ctx)

        # Override candidate_records with sign-aware records extracted directly
        # from the raw residues dict, where each solution now carries yj_sign
        # and the v-polynomial coefficients needed to compute yk_sign.
        _raw_residues = norm.get('residues') if isinstance(norm, dict) else None
        if _raw_residues:
            sign_records = _candidates_from_residues(_raw_residues, p)
            if sign_records:
                norm['candidate_records'] = sign_records
                norm['candidates'] = sign_records
                norm['candidate_xs'] = {r['xj'] for r in sign_records}

        # Fertility: fraction of n-values (vecs) that had at least one F_p root across any prime.
        # precomputed_residues[p] is keyed only by v_tuples that had roots, so union of keys = fertile set.
        # We try two possible shapes: {prime: {vtup: solutions}} and the flat {vtup: solutions} fallback.
        _precomp = norm.get('precomputed_residues') or (raw.get('precomputed_residues') if isinstance(raw, dict) else None)
        if _precomp and vecs:
            fertile_vtups = set()
            per_n_root_counts: Dict[str, int] = {}  # vtup_str -> number of F_p roots for that n
            # Shape A: {prime: {vtup: solutions}}  (expected from markov_mode search)
            for p_entry in _precomp.values():
                if isinstance(p_entry, dict):
                    fertile_vtups.update(p_entry.keys())
                    for vtup, solutions in p_entry.items():
                        k = str(vtup)
                        if isinstance(solutions, (list, tuple, set)):
                            cnt = len(solutions)
                        elif solutions is not None:
                            cnt = 1
                        else:
                            cnt = 0
                        # Take max across primes (FF mode has 1 prime; guard for multi-prime callers)
                        if cnt > 0:
                            per_n_root_counts[k] = max(per_n_root_counts.get(k, 0), cnt)
            # Shape B: flat {vtup: solutions} — if Shape A found nothing
            if not fertile_vtups:
                for k, v in _precomp.items():
                    if not isinstance(v, dict):
                        fertile_vtups.add(k)
                        per_n_root_counts[str(k)] = 1
            norm['n_with_roots'] = len(per_n_root_counts)
            norm['n_total'] = len(vecs)
            norm['total_roots'] = sum(per_n_root_counts.values())
            norm['per_n_roots'] = per_n_root_counts  # {vtup_str: root_count}
        else:
            # Fallback filled in below, after enriched_candidates is assembled.
            norm['n_with_roots'] = None
            norm['n_total'] = len(vecs) if vecs else None
            norm['total_roots'] = None

        # Grab ingredients for xk computation.
        # f_i is in R_xm = PolynomialRing(Frac(GF(p)['m']), 'x') — poly in x with rational-function-in-m coefficients.
        # shifted_G_poly is the curve poly in x over GF(p).
        # At a specific m_val: evaluate each coeff of f_i at m=m_val -> univariate poly in x over GF(p).
        # Then G(x) - f_i(x, m_val) = 0 has roots xi(x3), xj, xk.
        _G_poly = ctx.get('shifted_G_poly')
        _tower = ctx.get('primary_tower')
        _fi = None
        if _tower and isinstance(_tower, (list, tuple)) and len(_tower) > 0:
            last = _tower[-1]
            if isinstance(last, dict):
                _fi = last.get('f_i')

        # Curve degree from project globals, defaulting to 5.
        _curve_degree = int(resolve_project_symbol('CURVE_DEGREE', default=5))

        # enrich_candidates handles: degenerate-xj skip, m recovery, xk computation
        # via compute_xk_from_fiber, yk_sign computation from v(xk), yj_sign/yk_sign
        # defaults, and xk_head injection with roles swapped.  It replaces the old
        # inline loop which computed xk but never computed yk_sign.
        enriched_candidates = enrich_candidates(
            norm,
            x_here=x_here,
            y_here=y_here,
            n0=n0,
            fi=_fi,
            G_poly=_G_poly,
            curve_degree=_curve_degree,
            p=p,
            shift=ctx.get('shift', 0),
            T=ctx.get('T'),
            T_inv=ctx.get('T_inv'),
        )

        # Attach S_of_m and inter_sym to every record while the tower context is
        # still available.  This is a fibration property of xi (not of any xj), so
        # we compute once and stamp it on all records.
        if _fi is not None and _G_poly is not None:
            _S_of_m_rec, _inter_sym_rec = compute_S_of_m(_fi, _G_poly, _curve_degree)
            for rec in enriched_candidates:
                if isinstance(rec, dict):
                    rec.setdefault('S_of_m', _S_of_m_rec)
                    rec.setdefault('inter_sym', _inter_sym_rec)
        # candidate_xs is the set of xj values derived from actual m-roots only.
        # xk_head records must be excluded here so the leaf-tracking in
        # _step_from_candidate_search can separate xj_set (m-root derived) from
        # xk_set (Vieta derived).  xk_head entries are still in enriched_candidates
        # and eligible for the Metropolis chooser.
        candidate_xs = {
            c.get('xj') for c in enriched_candidates
            if isinstance(c, dict)
            and c.get('xj') is not None
            and c.get('source') != 'xk_head'
        }

        # Dead-end reason classification — emitted into the result dict so the
        # walker can log *why* a step produced no candidates rather than silently
        # restarting.  Distinguishes failure modes:
        #   no_roots    — Mumford search found zero F_p roots for all vecs
        #   torsion     — roots found but all equal xi (xi is torsion / Weierstrass pt)
        #   ok          — candidates available
        # Note: all_inf_xk is no longer a reachable case — compute_xk_from_fiber now
        # raises AssertionError on a fiber pole rather than returning "∞".
        _dead_end_reason = 'ok'
        if not enriched_candidates:
            raw_candidate_records = norm.get('candidate_records', []) or []
            if not raw_candidate_records:
                _dead_end_reason = 'no_roots'
            else:
                xjs = [r.get('xj') for r in raw_candidate_records if isinstance(r, dict)]
                xjs_nondegenerate = [xj for xj in xjs if xj is not None and xj != x_here]
                if not xjs_nondegenerate:
                    _dead_end_reason = 'torsion'
                else:
                    _dead_end_reason = 'no_roots'  # shouldn't happen; fallback

        # Deferred fertility fallback: if precomputed_residues wasn't available, use
        # the m-root-derived candidate count as a lower-bound proxy.
        # Do NOT include xk_head records in this count — they are not m-roots.
        if norm.get('n_with_roots') is None and vecs:
            n_mroot_cands = sum(
                1 for c in enriched_candidates
                if isinstance(c, dict) and c.get('source') != 'xk_head'
            )
            if n_mroot_cands > 0:
                norm['n_with_roots'] = min(n_mroot_cands, len(vecs))
                norm['n_total'] = len(vecs)
                norm['total_roots'] = n_mroot_cands
                norm['per_n_roots'] = {}  # per-vec provenance not available without precomputed_residues

        n_xk_head = sum(1 for c in enriched_candidates
                        if isinstance(c, dict) and c.get('source') == 'xk_head')

        S_of_m_step, _ = compute_S_of_m(_fi, _G_poly, _curve_degree) if _fi is not None else (None, None)
        result = {
            'candidates': enriched_candidates,
            'candidate_records': enriched_candidates,
            'candidate_xs': candidate_xs,       # xj-only (m-root derived), for leaf tracking
            'n_xk_head': n_xk_head,             # how many xk_head alternatives were injected
            'stats': norm.get('stats', None),
            'found_xs': norm.get('found_xs', set()),
            'input_n': n0,
            'S_of_m': S_of_m_step,   # fibration property of this xi, not of any xj
            'fi': _fi,               # symbolic fiber poly in x over Frac(Fp[m]); needed for synthetic injection
            'G_poly': _G_poly,       # curve poly in x over Fp; needed for synthetic injection
            'vecs': vecs,
            'n_with_roots': norm.get('n_with_roots', None),
            'n_total': norm.get('n_total', None),
            'total_roots': norm.get('total_roots', None),
            'per_n_roots': norm.get('per_n_roots', None),
            'dead_end_reason': _dead_end_reason,
            # Memory Fix: Omit 'context', 'raw_mumford_residues', 'new_sections', 'precomputed_residues'
            # which hold uncollectable SageMath Rings and Ideals.
        }

        return result

    return search_fn



def _run_markov_mumford_search_for_point(
    *,
    cd,
    current_sections,
    prime_pool,
    vecs,
    rhs_list,
    shift,
    rationality_test_func,
    coeffs_genus2,
    tower_data,
    num_workers,
    debug=False,
    x_b=None,
    shifted_coeffs=None,
    pool=None, chunk_size=8
):
    """Call the legacy Mumford search and normalize the result for Markov use.

    The important thing here is to keep the raw payload available, because the
    smarter chooser wants provenance such as which n/vector produced each xj.
    """
    raw = run_mumford_search(
        cd, current_sections, prime_pool, vecs, rhs_list, shift,
        rationality_test_func, coeffs_genus2, tower_data,
        num_workers, debug, x_b, shifted_coeffs, markov_mode=True, pool=pool, chunk_size=chunk_size
    )
    return _normalize_markov_mumford_result(raw)


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
            "xi", "yi", "shift", "r_expr", "n_with_roots", "per_n_roots",
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
                cand.get("xj")
                for cand in out["candidate_records"]
                if isinstance(cand, dict) and cand.get("xj") is not None
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
        out["candidate_records"] = [{"xj": x, "source": "mumford_residue"} for x in xs]
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
                cand.get("xj")
                for cand in out["candidate_records"]
                if isinstance(cand, dict) and cand.get("xj") is not None
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
    out["candidate_records"] = [{"xj": x, "source": "scalar_fallback"} for x in xs]
    out["candidates"] = list(out["candidate_records"])

    try:
        out["candidate_counts"] = Counter(xs)
    except Exception:
        out["candidate_counts"] = Counter()
        raise

    return out


def _collect_mumford_candidate_x_values(obj, out=None):
    """
    Recursively collect candidate x-values from a Mumford payload.
    """
    if out is None:
        out = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        for key in ("xj", "x", "x_val", "xcoord", "candidate_x", "x_value"):
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
        for key in ("xj", "x", "x_val", "xcoord", "candidate_x", "x_value"):
            if key in obj and obj[key] is not None:
                return obj[key]
    return obj if not isinstance(obj, (dict, list, tuple, set)) else None

def _candidate_record_from_x(x, source="mumford_residue", **extra):
    rec = {"xj": x, "source": source}
    rec.update(extra)
    return rec

def _candidates_from_residues(residues, p):
    """Extract candidate records from mumford_residues {p: {vtup: {rhs_idx: [m_root, ...]}}}.

    Julia now returns only m_root values — no Mumford pairs, no sign computation.
    enrich_candidates reconstructs xj/xk and signs from the fiber geometry.

    Returns a list of dicts with keys: xj, yj_sign, m, rhs_idx, source.
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
                    "xj":      None,   # reconstructed by enrich_candidates from m
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


def poly_roots_with_multiplicity(poly) -> List[Tuple[Any, int]]:
    """Return roots as (root, multiplicity) pairs over the polynomial's base field."""
    roots = poly.roots(multiplicities=True)
    assert roots, roots
    return [(r, int(m)) for r, m in roots]

def safe_solve_univariate_roots(poly, ring=None) -> List[Any]:
    """Solve poly=0 in its base ring, returning roots if Sage can see them."""
    roots = poly.roots(multiplicities=False)
    assert roots, roots
    return list(roots)

def compute_fertility(norm, raw, vecs):
    precomp = norm.get('precomputed_residues') or (
        raw.get('precomputed_residues') if isinstance(raw, dict) else None
    )

    if not (precomp and vecs):
        return {
            'n_with_roots': None,
            'n_total': len(vecs) if vecs else None,
            'total_roots': None,
            'per_n_roots': None,
        }

    fertile = set()
    per_n = {}

    for entry in precomp.values():
        if isinstance(entry, dict):
            fertile.update(entry.keys())
            for vtup, sols in entry.items():
                k = str(vtup)
                cnt = len(sols) if isinstance(sols, (list, tuple, set)) else (1 if sols else 0)
                if cnt > 0:
                    per_n[k] = max(per_n.get(k, 0), cnt)

    if not fertile:
        for k, v in precomp.items():
            if not isinstance(v, dict):
                fertile.add(k)
                per_n[str(k)] = 1

    return {
        'n_with_roots': len(per_n),
        'n_total': len(vecs),
        'total_roots': sum(per_n.values()),
        'per_n_roots': per_n,
    }

def exec_namespace(src: str) -> Dict[str, Any]:
    """Execute preparsed sage source and return the resulting namespace."""
    ns: Dict[str, Any] = {}
    exec(src, ns)
    return ns

def load_project_sources(base_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, bool]:
    """Load tower.sage and search7_genus2.sage into PROJECT_REGISTRY."""
    here = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    loaded: Dict[str, bool] = {}
    for name in ("tower.sage", "search7_genus2.sage"):
        path = here / name
        if verbose:
            print(f"[bootstrap] loading {path}")
        try:
            with open(path, "r") as f:
                src = f.read()
        except FileNotFoundError:
            if verbose:
                print(f"[bootstrap] WARNING: {path} not found, skipping")
            loaded[name] = False
            raise

        src = src.replace("    main_genus2()", "    pass # main_genus2() disabled")

        # Mutate the shared dict in-place so walkerclass.resolve_project_symbol sees it
        PROJECT_REGISTRY.update(
            {k: v for k, v in exec_namespace(preparse(src)).items()
             if not k.startswith('__')}
        )
        globals().update({k: v for k, v in PROJECT_REGISTRY.items()
                          if not k.startswith('__')})
        loaded[name] = True

    return loaded

def enrich_candidates(
    norm,
    x_here,
    y_here,
    n0,
    fi,
    G_poly,
    curve_degree,
    p=None,
    shift=0,
    T=None,
    T_inv=None,
):
    """
    Enrich candidate records by reconstructing the fiber intersection directly
    over GF(p), using only the shared m-parameter.

    Main fixes versus the previous version:
      - evaluates coefficients at m_val_fp more carefully, avoiding brittle
        direct subs-on-fraction-field behavior where possible
      - coerces all comparisons into GF(p)
      - skips candidates cleanly on poles / malformed evaluations
      - keeps the geometric reconstruction path explicit and strict
    """
    enriched = []

    print("x_here", x_here)

    if p is None or G_poly is None or fi is None:
        return []

    Fp = GF(int(p))
    shift_fp = Fp(shift)

    def _eval_at_m(obj, m_val_fp):
        """
        Evaluate a coefficient-like object at m = m_val_fp and coerce into GF(p).

        Tries callable evaluation first, then symbolic substitution, then direct
        coercion. This is intentionally conservative.
        """
        # Most polynomial/rational-function objects in Sage can be called.
        try:
            return Fp(obj(m_val_fp))
        except Exception:
            raise

        # Symbolic / expression fallback.
        try:
            return Fp(obj.subs(m=m_val_fp))
        except Exception:
            raise

        # Plain field element / integer / already-evaluated object.
        return Fp(obj)

    x_here_f = x_here
    x_here_f_fp = Fp(x_here_f)
    R_x = PolynomialRing(Fp, "x")

    candidates = norm.get("candidate_records", []) or norm.get("candidates", [])
    for cand in candidates:
        rec = dict(cand) if isinstance(cand, dict) else {"xj": cand}

        m_val = rec.get("m")
        if m_val is None:
            if RLINEAR and rec.get("xj") is not None:
                m_val = Fp(x_here) - Fp(rec["xj"])
            else:
                continue

        try:
            m_val_fp = Fp(m_val)
        except Exception:
            raise
            continue

        try:
            coeffs = []
            for c in fi.list():
                num_val = Fp(c.numerator()(m_val_fp))
                den_val = Fp(c.denominator()(m_val_fp))
                if den_val == 0:
                    raise ZeroDivisionError(f"Fiber pole at m={m_val_fp}")
                coeffs.append(num_val / den_val)
            f_eval_poly = R_x(coeffs)
        except ZeroDivisionError:
            continue
        except Exception as e:
            print(f"CRITICAL: Evaluation failed for m={m_val_fp}. Error: {e}")
            raise


        #print("m, f_eval_poly, fi", m_val, f_eval_poly, fi)

        # Step 3: intersection polynomial on the fiber.
        try:
            G_Rx = R_x(G_poly)
        except Exception:
            raise
            continue

        #print("G_Rx", G_Rx)
        intersection_poly = G_Rx - f_eval_poly # no square on f

        # Step 4: find all roots in the base field.
        try:
            roots_wm = intersection_poly.roots()
        except Exception:
            raise
            continue

        assert roots_wm, roots_wm

        other_roots_f = []
        actual_xi_mult = 0

        for r, mult in roots_wm:
            try:
                r_fp = Fp(r)
            except Exception:
                raise
                continue

            if r_fp == x_here_f_fp:
                actual_xi_mult += mult

                # Preserve the previous convention for excess xi multiplicity.
                if mult > (curve_degree - 2):
                    other_roots_f.extend([r_fp] * (mult - (curve_degree - 2)))
            else:
                other_roots_f.extend([r_fp] * mult)

        # Step 5: require exactly two non-xi roots in GF(p).
        if len(other_roots_f) != 2:
            continue

        xj_f, xk_f = other_roots_f

        # Step 6: evaluate Y directly from the fiber.
        try:
            yj_f_2 = f_eval_poly(xj_f)
            yk_f_2 = f_eval_poly(xk_f)
        except Exception:
            raise
            continue

        # Step 7: strict sign validation against the original curve model.
        def _get_strict_sign(x_val_f, y_val_f):
            y_int2 = int(Fp(y_val_f))
            x_int = Fp(x_val_f)
            curve_y2 = int(Fp(G_poly(x_int)))

            if (y_int2 ) % int(p) != curve_y2:
                print("y_int", y_int, "y_int^2", y_int**2 % p, "curve_y2", curve_y2)
                raise ValueError(
                    f"Y-coordinate validation failed for X={x_val_f}: "
                    f"fiber Y={y_val_f}, but Y^2 != G(X)."
                )

            canonical_y = min(y_int, int(p) - y_int)
            return 1 if y_int == canonical_y else -1

        try:
            #yj_sign = _get_strict_sign(xj_f, yj_f_2)
            #yk_sign = _get_strict_sign(xk_f, yk_f_2)
            yj_sign = 1
            yk_sign = 1
        except Exception:
            raise
            continue

        assert intersection_poly

        #print("intersection poly", intersection_poly, actual_xi_mult)

        xj_val, xk_val = xj_f, xk_f
        # Step 9: pack the record.
        new_rec = {
            "xi": x_here,
            "yi": y_here,
            "xj": xj_val,
            "xk": xk_val,
            "yj_sign": yj_sign,
            "yk_sign": yk_sign,
            "m": m_val_fp,
            "input_n": n0,
            "source": "pure_fiber_intersection",
            "xi_mult": actual_xi_mult,
            "intersection_poly": intersection_poly,
            "shift": shift,
        }
        enriched.append(new_rec)

        # Optional xk-head injection for RLINEAR.
        if RLINEAR and xk_val != x_here:
            try:
                enriched.append({
                    "xi": x_here,
                    "yi": y_here,
                    "xj": xk_val,
                    "xk": xj_val,
                    "yj_sign": yk_sign,
                    "yk_sign": yj_sign,
                    "m": Fp(x_here) - Fp(xk_val),
                    "input_n": n0,
                    "source": "xk_head",
                    "xi_mult": actual_xi_mult,
                    "intersection_poly": intersection_poly,
                    "shift": shift,
                })
            except Exception:
                raise

    return enriched


def _as_set(values):
    if values is None:
        return set()
    if isinstance(values, set):
        return set(values)
    if isinstance(values, (list, tuple)):
        return {v for v in values if v is not None}
    return {values}
