"""genus2_markov_module.py

A self-contained Sage-based draft for the genus-2 fibration / Markov-walk idea.

Run under SageMath:
    sage -python genus2_markov_module.py --help

Design goals:
- keep the chain logic and divisor bookkeeping in one place
- optionally load companion project files (tower.sage, search7_genus2.sage)
- accept a fibration-step factory from your existing code
- recover candidate x_j values from the m-root equation
- recover x_k from a degree-5 intersection polynomial via Vieta when available
- support a simple Metropolis-style bias and restart-on-dead-end behavior

This is a draft module, not a proof that the attack works.
"""

from __future__ import annotations

import argparse
import json
import dataclasses
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from sage.all import *  # type: ignore

StepFactory = Callable[..., Dict[str, Any]]
ScoreFn = Callable[[Any, Dict[str, Any]], float]


# ---------------------------------------------------------------------------
# Optional project bootstrap
# ---------------------------------------------------------------------------

def load_project_sources(base_dir: Optional[Path] = None, verbose: bool = True) -> Dict[str, bool]:
    """Load companion Sage files if they exist next to this module.

    The loaded namespace may provide:
      - build_one_fibration_step
      - iterate_tower
      - setup_field_and_rings
      - extract_geometry_from_tower
      - any other helpers your current project defines
    """
    here = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    loaded = {}
    for name in ("tower.sage", "search7_genus2.sage"):
        path = here / name
        if path.exists():
            if verbose:
                print(f"[bootstrap] loading {path}")
            load(str(path))
            loaded[name] = True
        else:
            loaded[name] = False
    return loaded


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


def poly_roots_with_multiplicity(poly) -> List[Tuple[Any, int]]:
    """Return roots as (root, multiplicity) pairs over the polynomial's base field."""
    try:
        roots = poly.roots(multiplicities=True)
        return [(r, int(m)) for r, m in roots]
    except Exception:
        raise
        return []


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


def safe_solve_univariate_roots(poly, ring=None) -> List[Any]:
    """Solve poly=0 in its base ring, returning roots if Sage can see them."""
    try:
        roots = poly.roots(multiplicities=False)
        return list(roots)
    except Exception:
        raise
        return []


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
    degree_for_intersection: int = 5
    verbose: bool = True
    log_path: Optional[str] = None
    log_candidate_limit: int = 25
    log_full_candidates: bool = False


# ---------------------------------------------------------------------------
# Core walker
# ---------------------------------------------------------------------------

class Genus2MetropolisWalker:
    """Run the proposed x-coordinate Markov walk on a genus-2 curve.

    Parameters
    ----------
    curve_poly:
        Hyperelliptic polynomial f(x) for y^2 = f(x).
    p:
        Optional prime for GF(p). If omitted, QQ is used.
    initial_x:
        Starting x-coordinate.
    initial_y:
        Optional starting y-coordinate. If omitted, we try to recover it from f(initial_x).
    base_points:
        Optional list of (x, y) pairs passed into the fibration constructor.
    step_factory:
        Optional callable with the same spirit as build_one_fibration_step.
        If omitted, the module will try to use a loaded build_one_fibration_step
        from tower.sage.
    score_fn:
        Optional scoring function for Metropolis bias.
        It receives (candidate_x, context_dict) and should return a real-valued score.
        Lower is better.
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

        if initial_x is None:
            raise ValueError("initial_x is required")
        self.current_x = self.base_ring(initial_x)
        self.current_y = self._recover_y(self.current_x, initial_y)
        self.visited_x = {self.current_x}
        self.history: List[RelationRecord] = []
        self.dead_end_count = 0
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
        # QQ mode: exact square root is rare; permit only exact squares.
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

        # The companion code expects a list of x-values and a point count.
        # We hand it a small seed list anchored at the current point.
        pts_x = [current_x]
        if self.base_points:
            pts_x.extend([x for x, _y in self.base_points if x is not None and x != current_x])
        # Keep the list reasonably small and deterministic.
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

    def _normalize_candidate_output(self, result: Any) -> Dict[str, Any]:
        if result is None:
            return {
                "candidates": [],
                "candidate_xs": set(),
                "new_sections": [],
                "precomputed_residues": None,
                "stats": None,
            }

        if isinstance(result, dict):
            out = dict(result)
            out.setdefault("candidates", [])
            out.setdefault("candidate_xs", set())
            out.setdefault("new_sections", [])
            out.setdefault("precomputed_residues", None)
            out.setdefault("stats", None)
            return out

        if isinstance(result, (tuple, list)) and len(result) == 4:
            a, b, c, d = result
            if isinstance(a, list) and a and isinstance(a[0], dict):
                return {
                    "candidates": a,
                    "candidate_xs": {cand.get("xj") for cand in a if cand.get("xj") is not None},
                    "new_sections": b,
                    "precomputed_residues": c,
                    "stats": d,
                }
            if isinstance(a, (set, list, tuple)):
                return {
                    "candidates": [{"xj": x} for x in a],
                    "candidate_xs": set(a),
                    "new_sections": b,
                    "precomputed_residues": c,
                    "stats": d,
                }

        raise TypeError(f"Unsupported search result type: {type(result)!r}")

    def _score_candidate_record(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> float:
        if self.score_fn is None:
            return 0.0
        xj = candidate.get("xj")
        try:
            return float(self.score_fn(xj, context | {"candidate": candidate}))
        except Exception:
            raise
            return 0.0

    def _choose_candidate_record(self, candidates: List[Dict[str, Any]], context: Dict[str, Any]):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if self.score_fn is None:
            return self.rng.choice(candidates)

        scores = [self._score_candidate_record(c, context) for c in candidates]
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

    def _step_from_candidate_search(self, n: int, seed: Optional[int] = None) -> Optional[RelationRecord]:
        current_point = (self.current_x, self.current_y)
        raw = self._call_search_fn(n=n, seed=seed, current_point=current_point)
        search_out = self._normalize_candidate_output(raw)

        candidates = list(search_out.get("candidates", []))
        candidate_xs = search_out.get("candidate_xs", set())
        if self.config.verbose:
            print(f"[walk] n={n} candidates={len(candidates)} candidate_xs={len(candidate_xs) if hasattr(candidate_xs, '__len__') else 'unk'}")

        if not candidates:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, search_out, accepted=False, restart=True)
                    rec.candidate_pool = candidates
                    self._store_record(rec)
                    return rec
            rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, search_out, accepted=False)
            rec.candidate_pool = candidates
            self._store_record(rec)
            return rec

        chosen = self._choose_candidate_record(
            candidates,
            {"n": n, "step": search_out, "current_x": self.current_x, "current_y": self.current_y},
        )

        if chosen is None:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, search_out, accepted=False, restart=True)
                    rec.candidate_pool = candidates
                    self._store_record(rec)
                    return rec
            rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, search_out, accepted=False)
            rec.candidate_pool = candidates
            self._store_record(rec)
            return rec

        m_val = chosen.get("m")
        xj = chosen.get("xj")
        xk = chosen.get("xk")

        if xj is None and m_val is not None:
            xj = self._candidate_xj_from_m(self.current_x, m_val)

        if xk is None and xj is not None:
            try:
                xk = self._recover_xk(search_out if isinstance(search_out, dict) else {}, self.current_x, xj)
            except Exception:
                xk = None
                raise

        accepted = xj is not None
        if accepted:
            next_x = xj
            try:
                next_y = self._recover_y(next_x)
            except Exception:
                next_y = None
                raise

            if next_y is not None:
                self.current_x, self.current_y = next_x, next_y
                self.visited_x.add(next_x)
            else:
                self.current_x = next_x

        rec = self._make_relation(len(self.history), n, self.current_x, m_val, xj, xk, search_out, accepted=accepted, restart=False)
        rec.candidate_pool = candidates
        rec.selected_candidate = dict(chosen)
        self._store_record(rec)
        return rec

    def _solve_m_roots(self, step: Dict[str, Any]) -> List[Any]:
        r_expr = step.get("r_expr")
        if r_expr is None:
            return []
        try:
            poly = r_expr if hasattr(r_expr, "roots") else SR(r_expr)
        except Exception:
            poly = r_expr
            raise
        try:
            return safe_solve_univariate_roots(poly)
        except Exception:
            raise
            return []

    def _candidate_xj_from_m(self, xi, m_val):
        return self.base_ring(xi) - self.base_ring(m_val)

    def _intersection_poly_from_step(self, step: Dict[str, Any]):
        """Best-effort access to a degree-5 intersection polynomial.

        The companion tower code stores f_i as the current fibration polynomial.
        If you have a dedicated degree-5 fiber-intersection polynomial, place it in
        step['intersection_poly'] or step['fiber_poly'] and this module will use it.
        """
        for key in ("intersection_poly", "fiber_poly", "intersection", "poly_x"):
            if key in step and step[key] is not None:
                return step[key]
        return None

    def _recover_xk(self, step: Dict[str, Any], xi, xj):
        poly = self._intersection_poly_from_step(step)
        if poly is None:
            return None

        # If the polynomial is univariate in x, use roots or Vieta.
        try:
            roots = flatten_roots(poly_roots_with_multiplicity(poly))
        except Exception:
            raise
            roots = []

        if roots:
            # Remove the triple root and the x_j root if possible.
            leftovers = []
            xi_count = 0
            xj_count = 0
            for r in roots:
                if r == xi and xi_count < 3:
                    xi_count += 1
                    continue
                if xj is not None and r == xj and xj_count < 1:
                    xj_count += 1
                    continue
                leftovers.append(r)
            if leftovers:
                return leftovers[0]

        # Fall back to Vieta if we have exactly 4 known roots with multiplicity.
        try:
            if poly.degree() != self.config.degree_for_intersection:
                return None
            known = [xi, xi, xi]
            if xj is not None:
                known.append(xj)
            return missing_root_by_vieta(poly, known)
        except Exception:
            raise
            return None

    def _score_candidate(self, candidate_x, context: Dict[str, Any]) -> float:
        if self.score_fn is None:
            return 0.0
        try:
            return float(self.score_fn(candidate_x, context))
        except Exception:
            raise
            return 0.0

    def _choose_between(self, xj, xk, context: Dict[str, Any]):
        candidates = [c for c in (xj, xk) if c is not None]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # If no score function is supplied, use the user-specified coin bias.
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

    def _make_relation(self, step_index: int, n: int, xi, m_val, xj, xk, step: Dict[str, Any], accepted=True, restart=False):
        relation = f"3*{xi} + {xj} + {xk} - 5*∞ = 0" if xj is not None and xk is not None else "relation incomplete"
        return RelationRecord(
            step_index=step_index,
            n=n,
            xi=xi,
            m=m_val,
            xj=xj,
            xk=xk,
            relation=relation,
            step=step,
            accepted=accepted,
            restart=restart,
        )

    def _jsonable(self, obj: Any):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        try:
            if isinstance(obj, (complex,)):
                return str(obj)
        except Exception:
            raise
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            try:
                return self._jsonable(obj.item())
            except Exception:
                raise
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

        limit = int(getattr(self.config, 'log_candidate_limit', 25) or 25)
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

    def _store_record(self, rec: RelationRecord) -> RelationRecord:
        self.history.append(rec)
        self._append_jsonl_log(rec)
        return rec

    def _next_restart_point(self):
        # Prefer unused base points first, then any x in the support of the curve if we can find one.
        while self._restart_cursor < len(self.base_points):
            x, y = self.base_points[self._restart_cursor]
            self._restart_cursor += 1
            if x not in self.visited_x:
                return x, y
        return None

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
                        rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, {}, accepted=False, restart=True)
                        self._store_record(rec)
                        return rec
                raise
                return None

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
                    rec = self._make_relation(len(self.history), n, self.current_x, None, None, None, {}, accepted=False, restart=True)
                    self._store_record(rec)
                    return rec
            raise
            return None

        m_roots = self._solve_m_roots(step)
        if self.config.verbose:
            print(f"[walk] n={n} m-roots={len(m_roots)}")

        xj_candidates = [self._candidate_xj_from_m(self.current_x, m_val) for m_val in m_roots]
        xj = xj_candidates[0] if xj_candidates else None
        m_val = m_roots[0] if m_roots else None

        xk = None
        if xj is not None:
            xk = self._recover_xk(step, self.current_x, xj)

        chosen = self._choose_between(xj, xk, {
            "n": n,
            "step": step,
            "current_x": self.current_x,
            "current_y": self.current_y,
        })
        accepted = chosen is not None

        if accepted:
            next_x = chosen
            try:
                next_y = self._recover_y(next_x)
            except Exception:
                next_y = None
                raise
            if next_y is not None:
                self.current_x, self.current_y = next_x, next_y
                self.visited_x.add(next_x)
        else:
            if self.config.restart_on_dead_end:
                restart = self._next_restart_point()
                if restart is not None:
                    self.current_x, self.current_y = restart
                    self.dead_end_count += 1
                    rec = self._make_relation(len(self.history), n, self.current_x, m_val, xj, xk, step, accepted=False, restart=True)
                    self._store_record(rec)
                    return rec

        rec = self._make_relation(len(self.history), n, self.current_x, m_val, xj, xk, step, accepted=accepted)
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
        return results

    def run_branching(self, num_steps: int, width: Optional[int] = None) -> List[List[RelationRecord]]:
        """Small breadth-style helper for the parallel-branch idea.

        This is intentionally conservative: each layer keeps up to `width` active branches.
        """
        width = int(width or self.config.branch_width)
        branches: List[Tuple[Any, Any, List[RelationRecord]]] = [(self.current_x, self.current_y, [])]
        n_values = list(range(1, min(self.config.max_n, 80) + 1))

        for step_idx in range(num_steps):
            new_branches = []
            for bx, by, hist in branches:
                saved = (self.current_x, self.current_y, list(self.history), set(self.visited_x))
                self.current_x, self.current_y = bx, by
                self.history = list(hist)
                self.visited_x = {r.xi for r in hist if r.xi is not None} | {bx}
                rec = self.step(n=n_values[step_idx % len(n_values)])
                if rec is not None and rec.accepted:
                    new_branches.append((self.current_x, self.current_y, list(self.history)))
                self.current_x, self.current_y, self.history, self.visited_x = saved
            if not new_branches:
                break
            branches = new_branches[:width]
        return [hist for _x, _y, hist in branches]

    def summary(self) -> str:
        accepted = sum(1 for r in self.history if r.accepted)
        restarts = sum(1 for r in self.history if r.restart)
        return (
            f"steps={len(self.history)} accepted={accepted} restarts={restarts} "
            f"visited={len(self.visited_x)} dead_ends={self.dead_end_count}"
        )


# ---------------------------------------------------------------------------
# Convenience helpers for direct use
# ---------------------------------------------------------------------------

def relation_string(xi, xj, xk) -> str:
    return f"3*{xi} + {xj} + {xk} - 5*∞ = 0"


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
    log_candidate_limit: int = 25,
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


# ---------------------------------------------------------------------------
# Minimal CLI
# ---------------------------------------------------------------------------

def _parse_coeffs(text: str) -> List[Fraction]:
    # Accept comma-separated rationals/ints.
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [QQ(p) for p in parts]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Genus-2 fibration Markov-walk draft module")
    ap.add_argument("--coeffs", required=True, help="Descending coefficients for y^2=f(x), comma-separated")
    ap.add_argument("--x0", required=True, help="Starting x-coordinate")
    ap.add_argument("--y0", default=None, help="Optional starting y-coordinate")
    ap.add_argument("--p", type=int, default=None, help="Finite-field prime p (omit for QQ)")
    ap.add_argument("--steps", type=int, default=10, help="Number of Markov steps")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--log-jsonl", default=None, help="Write one JSON record per step to this file")
    ap.add_argument("--log-full-candidates", action="store_true", help="Write the full candidate pool to the JSONL log")
    ap.add_argument("--no-load", action="store_true", help="Do not load tower.sage/search7_genus2.sage")
    args = ap.parse_args(argv)

    coeffs = _parse_coeffs(args.coeffs)
    x0 = QQ(args.x0) if args.p is None else GF(args.p)(args.x0)
    y0 = QQ(args.y0) if args.y0 is not None and args.p is None else (GF(args.p)(args.y0) if args.y0 is not None else None)

    walker = build_default_walker(
        coeffs=coeffs,
        initial_x=x0,
        p=args.p,
        initial_y=y0,
        seed=args.seed,
        load_sources=not args.no_load,
        verbose=True,
        log_path=args.log_jsonl,
        log_full_candidates=args.log_full_candidates,
    )

    results = walker.run(args.steps)
    for rec in results:
        print(dataclasses.asdict(rec))
    print(walker.summary())
    return 0



def build_tower_context_for_xi(xi, tower_builder_fn, builder_kwargs=None):
    """
    Rebuild the tower/fibration for a specific xi.

    Expected return from tower_builder_fn:
        either a dict with keys:
            cd, current_sections, rhs_list, r_m, shift
        or a tuple/list in that order.

    Replace tower_builder_fn with the actual entrypoint from tower.sage.
    """
    builder_kwargs = dict(builder_kwargs or {})

    tower_obj = tower_builder_fn(xi, **builder_kwargs)

    if isinstance(tower_obj, dict):
        cd = tower_obj["cd"]
        current_sections = tower_obj["current_sections"]
        rhs_list = tower_obj["rhs_list"]
        r_m = tower_obj["r_m"]
        shift = tower_obj["shift"]
        extra = {k: v for k, v in tower_obj.items() if k not in {
            "cd", "current_sections", "rhs_list", "r_m", "shift"
        }}
        return {
            "cd": cd,
            "current_sections": current_sections,
            "rhs_list": rhs_list,
            "r_m": r_m,
            "shift": shift,
            "extra": extra,
        }

    if isinstance(tower_obj, (tuple, list)) and len(tower_obj) >= 5:
        cd, current_sections, rhs_list, r_m, shift = tower_obj[:5]
        return {
            "cd": cd,
            "current_sections": current_sections,
            "rhs_list": rhs_list,
            "r_m": r_m,
            "shift": shift,
            "extra": {},
        }

    raise TypeError(
        "tower_builder_fn must return either a dict with keys "
        "cd/current_sections/rhs_list/r_m/shift or a tuple/list of length >= 5."
    )


def make_markov_search_fn(
    tower_builder_fn,
    search_kwargs,
    tower_builder_kwargs=None,
):
    """
    Returns a search_fn(xi) that rebuilds the tower for each xi and then runs
    the standard lattice search on that rebuilt context.
    """
    search_kwargs = dict(search_kwargs or {})
    tower_builder_kwargs = dict(tower_builder_kwargs or {})

    def search_fn(xi):
        ctx = build_tower_context_for_xi(
            xi=xi,
            tower_builder_fn=tower_builder_fn,
            builder_kwargs=tower_builder_kwargs,
        )

        result = _run_standard_lattice_search(
            cd=ctx["cd"],
            current_sections=ctx["current_sections"],
            prime_pool=search_kwargs["prime_pool"],
            vecs=search_kwargs["vecs"],
            rhs_list=ctx["rhs_list"],
            r_m=ctx["r_m"],
            shift=ctx["shift"],
            all_found_x=search_kwargs.get("all_found_x", set()),
            num_subsets=search_kwargs.get("num_subsets", 1),
            rationality_test_func=search_kwargs["rationality_test_func"],
            sconf=search_kwargs["sconf"],
            coeffs_genus2=search_kwargs["coeffs_genus2"],
            num_workers=search_kwargs.get("num_workers", 1),
            debug=search_kwargs.get("debug", False),
            precomputed_residues=search_kwargs.get("precomputed_residues", None),
        )

        candidates = result.get("candidates", [])

        # Enrich candidates with tower context so the Markov layer can inspect provenance.
        for c in candidates:
            c["tower_xi"] = xi
            c["tower_shift"] = ctx["shift"]
            c["tower_cd"] = ctx["cd"]
            c["tower_extra"] = ctx["extra"]

        # Keep the last context around for debugging / logging.
        search_fn.last_context = ctx
        search_fn.last_result = result

        return candidates

    search_fn.last_context = None
    search_fn.last_result = None
    return search_fn




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log-jsonl", type=str, default="markov_log.jsonl")
    parser.add_argument("--log-full-candidates", action="store_true")
    args = parser.parse_args()

    # Replace this with the actual tower.sage entrypoint.
    # It must take xi and return either:
    #   (cd, current_sections, rhs_list, r_m, shift)
    # or a dict with those fields.
    def tower_builder_fn(xi, **kwargs):
        # Example placeholder:
        # return build_tower_from_xi(xi, **kwargs)
        return build_tower_from_xi(xi, **kwargs)

    search_kwargs = {
        "prime_pool": PRIME_POOL,
        "vecs": VECS,
        "all_found_x": set(),
        "num_subsets": NUM_SUBSETS,
        "rationality_test_func": RATIONALITY_TEST,
        "sconf": SCONF,
        "coeffs_genus2": COEFFS_GENUS2,
        "num_workers": NUM_WORKERS,
        "debug": False,
        "precomputed_residues": None,
    }

    search_fn = make_markov_search_fn(
        tower_builder_fn=tower_builder_fn,
        search_kwargs=search_kwargs,
        tower_builder_kwargs={},
    )

    walker = Genus2MetropolisWalker(
        search_fn=search_fn,
        log_jsonl=args.log_jsonl,
        log_full_candidates=args.log_full_candidates,
    )

    walker.state = INITIAL_XI

    for i in range(args.steps):
        rec = walker.step()
        if i % 10 == 0:
            print(f"[step {i}] xi={rec.xi}  xj={rec.xj}  cand_count={rec.candidate_count}")
