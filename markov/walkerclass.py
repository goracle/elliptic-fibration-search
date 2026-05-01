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
from .walker import *
from .walker.curve_helpers import _coerce_base_ring
from .walker.candidate_utils import _jsonable, _derive_relation_from_intersection_poly

load('tower.sage')
load('search7_genus2.sage')

# DEAR AI:  EVERY RAISE IN HERE IS ON PURPOSE ASK ME IF YOU WANT TO REMOVE ONE.

# ---------------------------------------------------------------------------
# Small algebra helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RelationRecord:
    """
    Encodes a relation where each point is a (x, y) tuple.
    In the Jacobian of a Genus 2 curve, (x, y) and (x, -y) are distinct.
    """
    step_index: int
    n: int

    # The source atom (walk head) as a (x, y) tuple
    pt_src: Tuple[Any, Any]

    m: Optional[Any] = None

    # The chosen move target as a (x, y) tuple
    pt_step: Optional[Tuple[Any, Any]] = None

    # The first named residual root from the intersection poly as a (x, y) tuple
    pt_res: Optional[Tuple[Any, Any]] = None

    relation: str = ""
    step: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    restart: bool = False
    candidate_pool: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Dict[str, Any] = field(default_factory=dict)

    # Canonical flat atom list: all finite atoms in the relation with multiplicity.
    # Each atom is a (x, y) tuple. (x, y) and (x, -y) are DISTINCT atoms.
    # Sum invariant: sum(atoms) + (-curve_degree * ∞) = 0 in the Jacobian[cite: 1].
    atoms: List[Tuple[Any, Any]] = field(default_factory=list)

    # Any additional residual points beyond pt_step and pt_res as (x, y) tuples.
    extra_pts: List[Tuple[Any, Any]] = field(default_factory=list)

@dataclass
class WalkConfig:
    max_n: int = 80
    coin_bias_for_pt_step: float = 0.5
    metropolis_temperature: float = 1.0
    restart_on_dead_end: bool = True
    allow_branching: bool = False
    branch_width: int = 2
    seed: int = 0
    # Degree of the hyperelliptic curve polynomial (y^2 = f(x), deg f = curve_degree).
    # The divisor relation from the fiber intersection is built from the intersection
    # poly roots with multiplicities; the generic form is:
    #   mult(pt_src)*pt_src + pt_step + pt_res + extra_pts... - curve_degree*∞ = 0
    # Multiplicities are read from the intersection poly, not assumed.
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
    """Run the proposed Markov walk on a genus-2 curve.

    D1 atoms in the relation matrix are (x, y) curve points; (x, +y) and (x, -y)
    are distinct atoms so branch signs are preserved and used in the relation matrix.
    Navigation fields (pt_src, pt_step, pt_res) and graph-volume tracking
    (global_leaves_seen, visited_x, etc.) still use bare x-coordinates for
    birthday-paradox / graph-volume accounting, where sign is irrelevant.

    This version keeps a running total of unique pt_step values seen so far. The
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
        self.unique_pt_step_seen = {self.current_x}
        # Graph-volume / birthday-paradox tracking: bare x-coordinates only.
        # (x, +y) and (x, -y) count as ONE node for collision purposes — we want
        # to know when the walk revisits an x-coordinate, regardless of branch.
        # Relation-matrix atom identity uses (x, y) tuples (see rec.atoms).
        self.global_leaves_seen = {self.current_x}
        # Cumulative count of leaf insertions (not deduplicated) — the true
        # "effort" metric for merge-time analysis, independent of step count.
        self.total_leaf_insertions: int = 1  # seed x counts as one insertion
        # Pts that are structurally pre-known (seed x).
        # Birthday collision accounting excludes these.
        self._injected_pts: set = {self.current_x}
        # How many times each x has been stepped *through* as pt_src (i.e. used as chain state).
        # We avoid re-using high-visit-count nodes as the next pt_src when fresher candidates exist.
        self.pt_src_visit_count: Counter = Counter({self.current_x: 1})

        self.history: List[RelationRecord] = []
        self.cantor_cache: Optional[CantorPairCache] = None
        self.dead_end_count = 0
        self.dead_end_reasons: Counter = Counter()  # reason -> count
        self.walk_terminated: bool = False  # stop the run if no fresh pt_src remains

        # Cross-chain merge tracking.
        # Set via load_foreign_leaves(); once a leaf from this walk hits
        # foreign_leaves, merge_log records (step_index, graph_vol, leaf).
        self.foreign_leaves: Optional[set] = None     # leaf set from walk A
        self.merge_log: list = []                     # [(step_index, graph_vol, leaf), ...]
        self.first_merge_step: Optional[int] = None   # step_index of first hit
        self.first_merge_vol: Optional[int] = None    # total_leaf_insertions at first hit
        self._merged_leaves: set = set()              # dedup across all merge hits
        self.collision_count = 0      # path collisions: chosen pt_step already on chain path
        self.leaf_collision_count = 0 # graph collisions: any leaf already in global_leaves_seen
        # Rolling averages for novelty and fertility (window = last 20 accepted steps).
        self._ROLL_WINDOW = 20
        self._roll_novelty: list = []   # novelty_ratio values, capped at _ROLL_WINDOW
        self._roll_fertility: list = [] # frac_fertile values, capped at _ROLL_WINDOW
        self.first_birthday_step: Optional[int] = None  # step_index of first graph/birthday collision
        self.first_birthday_n: Optional[int] = None     # outer n of first graph/birthday collision
        self.collision_log: list = []  # [(step_index, outer_n, graph_vol, count, colliding_pts[:10]), ...]
        self._restart_cursor = 0
        # pt_src values that have been fully exhausted (ran as current_x and produced
        # zero novelty or a dead end).  Since each pt_src's fiber is deterministic,
        # re-running the same pt_src yields nothing new; we never select it as the
        # next chain state again.
        self.exhausted_pt_src: set = set()

        # Adjacency / transition matrices for spectral gap estimation.
        # mat_chain = accepted steps only          (path diagnostic, d~1)
        # mat_graph = full candidate pool per pt_src   (row-truncated average operator)
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

    def _prefer_unvisited_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Partition candidates by pt_src_visit_count and return the least-visited tier.

        We walk through visit counts 0, 1, 2, ... and return the first non-empty
        tier, so we always prefer candidates whose pt_step has never (or least often)
        been used as pt_src before.  Falls back to the full list if every candidate
        has been visited.
        """
        if not candidates:
            return candidates
        def _count(c):
            pt_step = c.get("pt_step") if isinstance(c, dict) else c
            return self.pt_src_visit_count.get(pt_step, 0)
        min_count = min(_count(c) for c in candidates)
        preferred = [c for c in candidates if _count(c) == min_count]
        return preferred

    def _pt_src_is_fresh(self, x) -> bool:
        """Return True only if x has never been used as pt_src in this walk."""
        return x is not None and x not in self.visited_x and x not in self.exhausted_pt_src

    def _register_unique_pt_step(self, pt_step):
        if pt_step is None:
            return False, len(self.unique_pt_step_seen)
        was_new = pt_step not in self.unique_pt_step_seen
        self.unique_pt_step_seen.add(pt_step)
        return was_new, len(self.unique_pt_step_seen)

    def _annotate_step_counts(self, step: Dict[str, Any], pt_step, accepted: bool) -> Tuple[bool, int]:
        unique_new = False
        unique_total = len(self.unique_pt_step_seen)

        if accepted and pt_step is not None:
            unique_new, unique_total = self._register_unique_pt_step(pt_step)

        if isinstance(step, dict):
            step["unique_pt_step_new"] = bool(unique_new)
            step["unique_pt_step_total"] = int(unique_total)

        return unique_new, unique_total

    def _choose_between(self, pt_step, pt_res, context: Dict[str, Any]):
        candidates = [c for c in (pt_step, pt_res) if c is not None]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if self.score_fn is None:
            return pt_step if self.rng.random() < self.config.coin_bias_for_pt_step else pt_res

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

    def _record_to_log_dict(self, rec: RelationRecord) -> Dict[str, Any]:
        step = rec.step if isinstance(rec.step, dict) else {}
        candidate_pool = list(getattr(rec, 'candidate_pool', []) or [])
        selected = dict(getattr(rec, 'selected_candidate', {}) or {})

        limit = getattr(self.config, 'log_candidate_limit', 25*infinity) or 25*infinity
        pool_summary = candidate_pool if self.config.log_full_candidates else candidate_pool[:limit]

        # Serialize the relation as a flat atom list (with repetition for
        # multiplicity) — the canonical on-disk encoding.  rec.atoms is the
        # authoritative source; each atom is a (x, y) tuple.
        # Degree invariant: len(flat_atoms) == curve_degree for every accepted
        # relation (∞ is implicit, contributing -curve_degree; not listed here).
        Fp = self.base_ring
        cd = getattr(self.config, 'curve_degree', 5)
        flat_atoms: List[Any] = [
            [_jsonable(a[0]), _jsonable(a[1])] if (isinstance(a, (tuple, list)) and len(a) == 2)
            else [_jsonable(a), None]
            for a in (getattr(rec, 'atoms', None) or [])
        ]
        assert not rec.accepted or len(flat_atoms) in (cd, cd - 1), (
            f"[_record_to_log_dict] degree invariant violated at step={rec.step_index}: "
            f"len(atoms)={len(flat_atoms)} not in ({cd-1}, {cd})  "
            f"(pt_src={rec.pt_src!r})"
        )

        return {
            'step_index': rec.step_index,
            'n': rec.n,
            'pt_src': _jsonable(rec.pt_src),
            'm': _jsonable(rec.m),
            # Flat atom list: all finite atoms in the relation with multiplicity.
            # len(atoms) == curve_degree; ∞ contributes -curve_degree implicitly.
            'atoms': flat_atoms,
            'yj_sign': int(getattr(rec, 'yj_sign', 1)),
            'yk_sign': int(getattr(rec, 'yk_sign', 1)),
            'accepted': bool(rec.accepted),
            'restart': bool(rec.restart),
            'relation': rec.relation,
            'candidate_count': len(candidate_pool),
            'candidate_pool': _jsonable(pool_summary),
            'selected_candidate': _jsonable(selected),
            'step': _jsonable(step),
            'unique_pt_step_new': bool(step.get('unique_pt_step_new', False)),
            'unique_pt_step_total': int(step.get('unique_pt_step_total', len(self.unique_pt_step_seen))),
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
            if self._pt_src_is_fresh(x):
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
        pool = list(self.global_leaves_seen - self._injected_pts - exclude)
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
                saved = (self.current_x, self.current_y, list(self.history), set(self.visited_x), set(self.unique_pt_step_seen))
                self.current_x, self.current_y = bx, by
                self.history = list(hist)
                self.visited_x = {r.pt_src for r in hist if r.pt_src is not None} | {bx}
                self.unique_pt_step_seen = set(self.visited_x)
                rec = self.step(n=n_values[step_idx % len(n_values)])
                if rec is not None and rec.accepted:
                    new_branches.append((self.current_x, self.current_y, list(self.history)))
                self.current_x, self.current_y, self.history, self.visited_x, self.unique_pt_step_seen = saved
            if not new_branches:
                break
            branches = new_branches[:width]
        return [hist for _x, _y, hist in branches]

    def summary(self) -> str:
        accepted = sum(1 for r in self.history if r.accepted)
        restarts = sum(1 for r in self.history if r.restart)
        unique_path_nodes = len(self.unique_pt_step_seen)
        total_leaves = len(self.global_leaves_seen)

        base = (
            f"\n--- WALK SUMMARY ---\n"
            f"Steps taken: {len(self.history)}\n"
            f"Path accepted: {accepted}\n"
            f"Path collisions (pt_step revisited on chain): {self.collision_count}\n"
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

    def print_relation_summary(self, **kwargs):
        """Prints the shape, column mapping, and rank of the relation matrix."""
        mat, atoms, used = self.relation_matrix()
        print_relation_matrix_summary(mat, atoms, used, **kwargs)

    def _relation_matrix(self, **kwargs):
        cd = getattr(self.config, "curve_degree", 5)
        return build_relation_matrix2(self.history, curve_degree=cd, **kwargs)

    def _call_search_fn(self, n: int, seed: Optional[int] = None, current_point=None):
        if self.search_fn is None:
            return None

        kwargs = {
            "pt_src": self.current_x,
            "xi": self.current_x,          # backward-compat alias
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

            # Use the full (x, y) tuple if available, otherwise fallback to x or "—"
            pt_src_str = str(getattr(rec, 'p_src', rec.pt_src))
            pt_step_str = str(getattr(rec, 'p_step', rec.pt_step)) if rec.pt_step is not None else "—"
            pt_res_str = str(getattr(rec, 'p_res', rec.pt_res)) if rec.pt_res is not None else "—"

            m_str = str(rec.m) if rec.m is not None else "—"
            rel_str = rec.relation if rec.relation else "—"
            pt_step_visits = self.pt_src_visit_count.get(rec.pt_step, 0) if rec.pt_step is not None else 0
            pt_src_visits = self.pt_src_visit_count.get(rec.pt_src, 0) if rec.pt_src is not None else 0

            path_collision = (rec.pt_step is not None and not step_dict.get("unique_pt_step_new", False))
            leaf_collisions_this_step = step_dict.get("step_leaf_collisions", 0)

            n_with_roots = step_dict.get("n_with_roots")
            n_total = step_dict.get("n_total") or self.config.max_n
            total_roots = step_dict.get("total_roots")
            per_n_roots_map = step_dict.get("per_n_roots") or {}
            if n_with_roots is None and per_n_roots_map:
                n_with_roots = len(per_n_roots_map)

            pt_step_leaves_count = step_dict.get("step_pt_step_leaves", step_leaves)
            pt_res_new_count = step_dict.get("step_pt_res_leaves_new", 0)
            pt_res_overlap = step_dict.get("step_pt_res_leaves_overlap", 0)

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

            pt_res_leaf_note = (
                f" (+{pt_res_new_count} pt_res-new, {pt_res_overlap} pt_res↔pt_step overlap)"
                if pt_res_new_count or pt_res_overlap else ""
            )

            pool = getattr(rec, "candidate_pool", []) or []
            n_pt_res_head_pool = sum(1 for c in pool if isinstance(c, dict) and c.get("source") == "pt_res_head")
            n_pt_step_head_pool = len(pool) - n_pt_res_head_pool
            if n_pt_res_head_pool:
                rel_annotation = f"  (pool: {n_pt_step_head_pool} pt_step-head + {n_pt_res_head_pool} pt_res-head)"
            elif len(pool) > 1:
                rel_annotation = f"  (chosen from pool of {len(pool)})"
            else:
                rel_annotation = ""

            walk_tag = f"[walk-{label}] " if label else ""
            print(
                f"\n{'='*70}",
                f"\n{walk_tag}[WALK] STEP {step_no} COMPLETE  (outer n={rec.n})",
                f"\n  Path:      P_src → P_step  |  P_src={pt_src_str}  (visited {pt_src_visits}×)",
                f"\n             P_step={pt_step_str}  (visited {pt_step_visits}×)  |  P_res={pt_res_str}  |  m={m_str}",
                f"\n  Relation (example):  {rel_str}{rel_annotation}",
                f"\n  This step: accepted={rec.accepted}  path_collision={'YES' if path_collision else 'no'}"
                + (f"  | repeated x-coords this step (relations still novel): {leaf_collisions_this_step}" if leaf_collisions_this_step else ""),
                f"\n  Collisions: path={self.collision_count} total  | repeated x-coords={self.leaf_collision_count} total  (birthday clock ticks when x-coord repeats; first expected near graph vol=√p={sqrt_p:.0f})"
                + (f"  [first birthday: step={self.first_birthday_step} n={self.first_birthday_n} vol={self.collision_log[0][2]} pts={self.collision_log[0][4]}]" if self.collision_log else ""),
                f"\n  Totals:    steps_accepted={accepted_count}  restarts={restarts}  dead_ends={self.dead_end_count}",
                f"\n  Leaves:    pt_step={pt_step_leaves_count}{pt_res_leaf_note}  total={step_leaves}  new={new_leaves}  novelty={novelty_ratio:.1%} (new x-coords / all leaves this step)"
                + (f"  (avg {roll_novelty_avg:.1%} /{len(self._roll_novelty)})" if roll_novelty_avg is not None else ""),
                # If tracking points, sqrt_p isn't the only benchmark; maybe use 2*p or just keep the ratio relative to √p
                f"\n  Graph vol: {total_leaves} unique (x,y) atoms seen across all leaves ({collision_frac:.4f}×√p [√p={sqrt_p:.1f}])"
                f"  [Note: graph vol counts x-coords; relation-matrix atoms are (x,y) pairs]",
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

    def close_under_involution(self):
        return close_under_involution2(self)

    def _try_partial_cantor_reduction(self, rec: RelationRecord) -> bool:
        """
        For each accepted 5-atom relation, try all C(5,2)=10 fixed-pair choices.
        Ensures (x, y) tuples are passed to the Cantor cache to avoid TypeErrors.
        """
        step_dict = rec.step if isinstance(rec.step, dict) else {}
        if step_dict.get("source") == "cantor_triple_reduction":
            return False
        if not rec.accepted:
            return False

        atoms = list(getattr(rec, "atoms", None) or [])
        if len(atoms) != self.config.curve_degree:
            return False
        if self.cantor_cache is None:
            return False

        Fp = self.base_ring
        emitted = False

        # 1. Normalize all atoms to (x, y) tuples immediately to avoid repeated recovery
        normalized_atoms = []
        for a in atoms:
            if isinstance(a, (tuple, list)):
                normalized_atoms.append((Fp(a[0]), Fp(a[1])))
            else:
                x_val = Fp(a)
                normalized_atoms.append((x_val, self._recover_y(x_val)))

        # 2. Iterate through combinations of the 5 atoms
        # We pick 2 to stay "fixed" and reduce the other 3
        for fixed_indices in itertools.combinations(range(len(normalized_atoms)), 2):
            triple_indices = [i for i in range(len(normalized_atoms)) if i not in fixed_indices]

            # Extract the two fixed atoms as full (x, y) tuples
            atom_fa = normalized_atoms[fixed_indices[0]]
            atom_fb = normalized_atoms[fixed_indices[1]]

            # Extract x-coordinates for the triple to be reduced
            xa, xb, xc = (normalized_atoms[i][0] for i in triple_indices)

            # 3. Pass full tuples to the cache to fix the branch/sign issue
            # This prevents the TypeError: 'sage.rings.finite_rings.integer_mod.IntegerMod_int'
            # object is not subscriptable in cantor_cache.py
            result = self.cantor_cache.reduce_triple(
                xa, xb, xc,
                atom_fa,
                atom_fb
            )

            if result is None:
                continue

            # result contains the new reduced x-coordinates r0, r1
            r0, r1 = Fp(result[0]), Fp(result[1])

            # 4. Construct the new 4-atom relation
            # We keep the 2 fixed atoms and add the 2 new reduced atoms
            new_atoms = [
                atom_fa,
                atom_fb,
                (r0, self._recover_y(r0)),
                (r1, self._recover_y(r1)),
            ]

            new_rec = RelationRecord(
                step_index=len(self.history),
                n=rec.n,
                pt_src=atom_fa[0],
                pt_step=atom_fb[0],
                pt_res=r0,
                relation=f"Cantor red: {atom_fa[0]}, {atom_fb[0]}, {r0}, {r1}",
                step={"source": "cantor_triple_reduction", "parent": rec.step_index},
                accepted=True,
                atoms=new_atoms,
                extra_pts=[r1]
            )

            # 5. Verify the new divisor is principal before storing
            if self._verify_atoms_principal(new_atoms):
                self._store_record(new_rec)
                emitted = True

        return emitted

    def generate_mixed_relations(
        self,
        atoms_to_inject: List[Any],
        *,
        seed_atoms: Optional[set] = None,
        label: str = "mixed",
    ) -> int:
        return generate_mixed_relations2(self, atoms_to_inject, label=label, seed_atoms=seed_atoms)

    def _store_record(self, rec: RelationRecord) -> RelationRecord:
        # Only trim the candidate pool if the user hasn't requested full candidates.
        # If we trim it, those leaves won't make it into the relation matrix!
        if not getattr(self.config, 'log_full_candidates', True):
            limit = getattr(self.config, 'log_candidate_limit', 25*infinity) or 25*infinity
            if hasattr(rec, 'candidate_pool') and rec.candidate_pool and len(rec.candidate_pool) > limit:
                if limit < infinity:
                    rec.candidate_pool = rec.candidate_pool[:limit]
                else:
                    raise ValueError

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
            self._try_partial_cantor_reduction(rec)

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
            if rec.pt_src is not None:
                this_step_leaves.add(rec.pt_src)
            if rec.pt_step is not None:
                this_step_leaves.add(rec.pt_step)
            if rec.pt_res is not None:
                this_step_leaves.add(rec.pt_res)
            for cand in (rec.candidate_pool or []):
                if not isinstance(cand, dict):
                    continue
                for key in ("pt_step", "candidate_pt", "pt_res"):
                    v = cand.get(key)
                    if v is not None:
                        this_step_leaves.add(v)

            # Exclude base points (trivial hits) and already-reported leaves.
            base_pts = {bp[0] for bp in self.base_points if bp and len(bp) > 0}
            hits = (this_step_leaves & self.foreign_leaves) - base_pts - self._merged_leaves
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
        base_pts = {bp[0] for bp in self.base_points if bp and len(bp) > 0}
        self.foreign_leaves -= base_pts

        self._foreign_label = label
        print(
            f"[load_foreign_leaves] loaded {len(self.foreign_leaves)} leaves "
            f"from walk {label!r}  (current walk vol = {len(self.global_leaves_seen)})"
        )

        # Sanity check: if seeds are truly independent the initial overlap
        # should be zero.  A nonzero value means the two walkers share starting
        # state and the merge metric will be artificially low.
        initial_overlap = len(self.global_leaves_seen & self.foreign_leaves) - len(
            self.global_leaves_seen & self.foreign_leaves & base_pts
        )
        initial_overlap = len((self.global_leaves_seen - base_pts) & self.foreign_leaves)
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

    def _recover_y(self, x_val, explicit_y=None, y_sign: Optional[int] = None):
        """
        Recover a y-coordinate for x_val.

        Rules:
        - If explicit_y is given, keep it exactly.
        - If y_sign is given, choose that branch consistently.
        - If no sign is given, fall back to a deterministic canonical choice.
        """
        if isinstance(x_val, (tuple, list)):
            x_val = x_val[0]
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
            "x": _jsonable(x_val),
            "is_fp_point": False,
        }

        if isinstance(x_val, tuple):
            x_val = x_val[0]

        if x_val is None:
            info["reason"] = "missing_x"
            return info

        if self.curve_poly is None:
            info["reason"] = "missing_curve_poly"
            return info

        try:
            rhs = self.curve_poly(x_val)
            info["rhs"] = _jsonable(rhs)

            if hasattr(rhs, "is_square"):
                is_sq = bool(rhs.is_square())
                info["is_square"] = is_sq
                info["is_fp_point"] = is_sq
                if is_sq:
                    try:
                        y_any = self.base_ring(rhs.sqrt())
                        info["sqrt"] = _jsonable(y_any)
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
        pt_src,
        n: int,
        current_point=None,
        m_val=None,
        pt_step=None,
        pt_res=None,
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
        payload["reject_pt_src"] = _jsonable(pt_src)
        payload["reject_m"] = _jsonable(m_val)
        payload["reject_pt_step"] = _jsonable(pt_step)
        payload["reject_pt_res"] = _jsonable(pt_res)

        if current_point is not None:
            try:
                cx, cy = current_point
                payload["reject_current_x"] = _jsonable(cx)
                payload["reject_current_y"] = _jsonable(cy)
            except Exception:
                payload["reject_current_point"] = _jsonable(current_point)
                raise

        if chosen is not None:
            payload["reject_selected_candidate"] = _jsonable(chosen)
            if isinstance(chosen, dict):
                payload["reject_selected_source"] = chosen.get("source", None)

        if move_committed is not None:
            payload["reject_move_committed"] = bool(move_committed)

        if extra:
            for k, v in extra.items():
                payload[f"reject_{k}"] = _jsonable(v)

        if self.config.verbose:
            print(f"[reject] stage={stage} reason={reason} pt_src={pt_src} n={n}")
            if m_val is not None or pt_step is not None or pt_res is not None:
                print(f"         m={m_val} pt_step={pt_step} pt_res={pt_res}")
            if current_point is not None:
                try:
                    cx, cy = current_point
                    print(f"         current=({cx}, {cy})")
                except Exception:
                    print(f"         current={current_point}")
                    raise
            if extra:
                for k, v in extra.items():
                    print(f"         {k} = {_jsonable(v)}")

        return payload

    def _recover_pt_res(self, step: Dict[str, Any], pt_src, pt_step):
        """
        Compatibility wrapper.

        The new source of truth is the intersection polynomial; this just
        extracts pt_res and src_mult from it.
        """
        derived = _derive_relation_from_intersection_poly(step, pt_src)
        if derived is None:
            return None, -1
        _pt_step, pt_res, src_mult, _poly, _extra = derived
        return pt_res, src_mult

    def _make_relation(
        self,
        step_index: int,
        n: int,
        pt_src: Tuple[Any, Any],  # Authoritative (x, y) tuple
        m_val: Any,
        pt_step: Tuple[Any, Any], # Authoritative (x, y) tuple
        pt_res: Tuple[Any, Any],  # Authoritative (x, y) tuple
        step_metadata: Dict[str, Any],
        accepted: bool = True,
        restart: bool = False,
        src_mult: Optional[int] = None,
        extra_pts: Optional[List[Tuple[Any, Any]]] = None,
    ):
        """
        Build a RelationRecord using the Point Paradigm.

        This method treats all point inputs as authoritative (x, y) tuples.
        It prioritizes '_validated_atoms' from the step metadata, which
        ensures exact geometric consistency without re-calculating signs.
        """
        Fp = self.base_ring
        deg = self.config.curve_degree

        # 1. Atom Recovery: Prioritize the validator's work
        # If the validator already built and checked the atoms, use them as-is.
        atoms_list = step_metadata.get("_validated_atoms", [])

        # 2. Fallback: If no validated atoms exist (e.g. rejected/legacy steps)
        if not atoms_list and accepted:
            # We must reconstruct them from the provided point tuples
            effective_mult = src_mult if src_mult is not None else (deg - 2)

            # Ensure everything is a clean (Fp, Fp) tuple
            # In markov/walkerclass.py -> _make_relation

            def _clean(pt):
                # Handle (x, y) tuples or lists
                if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                    return (Fp(pt[0]), Fp(pt[1]))

                # Fallback for scalar x: recover y from the walker's current state if it matches
                # otherwise use the standard recovery tool
                x_val = pt
                if hasattr(self, 'current_x') and x_val == self.current_x:
                    return (Fp(x_val), Fp(self.current_y))

                return (Fp(x_val), Fp(self._recover_y(x_val)))

            atoms_list = [_clean(pt_src)] * int(effective_mult)
            if pt_step: atoms_list.append(_clean(pt_step))
            if pt_res:  atoms_list.append(_clean(pt_res))
            if extra_pts:
                atoms_list.extend([_clean(p) for p in extra_pts])

        # 3. String Representation
        if atoms_list:
            # Group by point to make the string readable (e.g., 2*P + Q...)
            from collections import Counter
            counts = Counter(atoms_list)
            parts = [f"{count}*{pt}" if count > 1 else f"{pt}" for pt, count in counts.items()]
            relation_str = " + ".join(parts) + f" - {deg}*∞ = 0"
        else:
            relation_str = "no atoms found" if accepted else "step_rejected"

        # 4. Data Sanitization (Remove heavy search objects)
        clean_metadata = {}
        if isinstance(step_metadata, dict):
            bad_keys = {"candidates", "candidate_records", "fi", "context"}
            clean_metadata = {k: v for k, v in step_metadata.items() if k not in bad_keys}

        # 5. Invariant Check
        if accepted and len(atoms_list) not in (deg, deg - 1):
            raise AssertionError(
                f"[POINT_PARADIGM] Invariant violation: len(atoms)={len(atoms_list)} "
                f"expected {deg} or {deg-1}. Source: {pt_src}"
            )

        return RelationRecord(
            step_index=step_index,
            n=n,
            pt_src=pt_src,
            m=m_val,
            pt_step=pt_step,
            pt_res=pt_res,
            relation=relation_str,
            step=clean_metadata,
            accepted=accepted,
            restart=restart,
            atoms=atoms_list,
            extra_pts=extra_pts or []
        )

    def _step_from_candidate_search(self, n: int, seed: Optional[int] = None):
        return step_from_candidate_search(self, n, seed=seed)

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
        pt_src,
        m_val=None,
        pt_step=None,
        pt_res=None,
        step_payload=None,
        accepted: bool,
        restart: bool = False,
        yj_sign: int = 1,
        yk_sign: int = 1,
    ):
        rec = self._make_relation(
            step_index, n, pt_src, m_val, pt_step, pt_res,
            step_payload or {},
            accepted=accepted,
            restart=restart,
            yj_sign=yj_sign,
            yk_sign=yk_sign,
        )
        self._store_record(rec)
        return rec

    def _update_leaf_bookkeeping(self, leaves, *, n: int, xi_before):
        valid_leaves = {cx for cx in leaves if cx is not None}
        organic = valid_leaves - self._injected_pts
        organic_already_seen = organic & self.global_leaves_seen
        colliding_pts = sorted(organic_already_seen)[:10]
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
                (_step_idx, n, len(self.global_leaves_seen), leaf_collisions_this_step, colliding_pts)
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

    def _reject_direct_step(self, *, step_payload, stage, reason, pt_src, n, current_point, m_val=None, pt_step=None, pt_res=None, chosen=None, extra=None):
        payload = self._reject_step_payload(
            step_payload if isinstance(step_payload, dict) else {},
            stage=stage,
            reason=reason,
            pt_src=pt_src,
            n=n,
            current_point=current_point,
            m_val=m_val,
            pt_step=pt_step,
            pt_res=pt_res,
            chosen=chosen,
            extra=extra or {},
        )
        return self._store_relation_record(
            step_index=len(self.history),
            n=n,
            pt_src=pt_src,
            m_val=m_val,
            pt_step=pt_step,
            pt_res=pt_res,
            step_payload=payload,
            accepted=False,
            restart=False,
        )

    def _verify_atoms_principal(self, atoms_list: List[Tuple[Any, Any]]) -> bool:
        """
        Returns True if the divisor represented by atoms_list is principal.

        Mandate: atoms_list MUST contain (x, y) tuples.
        For phi-steps on Genus 2 (quintic f), the list must have exactly 5 atoms
        to balance the 5-order pole at infinity.
        """
        if not atoms_list:
            return False

        # 1. Setup Curve & Jacobian
        if not hasattr(self, "_hec") or self._hec is None:
            from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
            self._hec = HyperellipticCurve(self.curve_poly)

        C = self._hec
        Fp = self.base_ring
        J = C.jacobian()(Fp)

        # 2. Degree Validation (No Silent Failures)
        # Genus 2 phi-steps define div(phi) = 2P + Q + R + S - 5*inf.
        # If we have 4 or 6 atoms, the sum in the Jacobian will never be zero.
        if len(atoms_list) != 5:
            # 4-atom records are routine non-phi enrichment steps (src_mult=2, no R,S).
            # Pool summary in _select_and_validate_candidate counts these as degree-skip.
            return False

        # 3. Summation in Jacobian
        total = J(0)
        for i, atom in enumerate(atoms_list):
            # Ensure we have a tuple (x, y)
            if not isinstance(atom, (tuple, list)) or len(atom) < 2:
                print(f"  [verify_fail] Atom {i} is not a point tuple: {atom}")
                return False

            x_fp, y_fp = Fp(atom[0]), Fp(atom[1])

            # Weierstrass points (y=0) make the relation degenerate/non-principal for phi
            if y_fp == 0:
                return False

            try:
                # Lift to the curve and add to the running divisor sum
                # J(C(x, y)) is the class [(x, y) - (inf)]
                total += J(C(x_fp, y_fp))
            except Exception as e:
                print(f"  [verify_fail] Sage could not lift {atom}: {e}")
                return False

        # 4. Final Principality Check
        # A sum of 0 in the Jacobian means (Sum P_i) - 5*inf ~ 0
        is_principal = (total == J(0))

        if not is_principal and self.config.verbose:
            atom_strs = [(int(a[0]), int(a[1])) if isinstance(a, (tuple, list)) else a for a in atoms_list]
            print(
                f"  [verify_fail] Jacobian sum ≠ 0: got {total}. "
                f"Likely wrong y-branch on one or more atoms. "
                f"Atoms: {atom_strs}"
            )

        return is_principal

    def _accept_direct_step(self, *, step_payload, n, pt_src, m_val, pt_step, pt_res, yj_sign=1, yk_sign=1):
        rec = self._make_relation(
            len(self.history), n, pt_src, m_val, pt_step, pt_res,
            step_payload or {},
            accepted=True,
            restart=False,
            yj_sign=yj_sign,
            yk_sign=yk_sign,
        )
        if rec.atoms and not self._verify_atoms_principal(rec.atoms):
            print(f"  [verify] REJECT non-principal relation at step={rec.step_index} "
                  f"atoms={[(int(a[0]),int(a[1])) if isinstance(a,(tuple,list)) else int(a) for a in rec.atoms]}")
            return None
        self._store_record(rec)
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
            # Never allow the walk to stall on the same pt_src after a rejection.
            # If the underlying step did not already move to a fresh restart
            # point, force one here.
            if self.current_x == rec.pt_src:
                step_dict = rec.step if isinstance(rec.step, dict) else {}
                reason = step_dict.get("reason", "rejected")
                nxt = self._restart_after_dead_end(
                    pt_src=rec.pt_src,
                    n=rec.n,
                    reason=reason,
                    current_point=(rec.pt_src, self.current_y),
                )
                if nxt is None:
                    self.walk_terminated = True
        return rec

    def _restart_after_dead_end(self, *, pt_src, n, reason, current_point=None):
        # Mark the incoming pt_src as exhausted — its fiber is deterministic so
        # re-running it as chain state will produce nothing new.
        self.exhausted_pt_src.add(pt_src)

        # Build candidate pool: base_points first, then accumulated leaves.
        # Exclude any pt_src that is already exhausted so we never loop back to it.
        candidates = [
            (x, y) for x, y in self.base_points
            if x is not None and y is not None and self._pt_src_is_fresh(x)
        ]

        # If base_points is only the current stuck point (or empty), augment from
        # global_leaves_seen — the actual visited graph.  This is the escape hatch
        # for the single-base-point case: without it the cursor just loops back to
        # the same pt_src every time.
        if len(candidates) <= 1:
            # Prefer leaves that have never been used as pt_src (freshest first).
            never_pt_src = self.global_leaves_seen - self.exhausted_pt_src - self.visited_x
            pool_order = sorted(never_pt_src, key=lambda lx: self.pt_src_visit_count.get(lx, 0))
            # Fall back to any non-exhausted leaf if the fresh pool is empty.
            if not pool_order:
                pool_order = sorted(
                    self.global_leaves_seen - self.exhausted_pt_src - self.visited_x,
                    key=lambda lx: self.pt_src_visit_count.get(lx, 0),
                )
            for lx in pool_order:
                if not self._pt_src_is_fresh(lx):
                    continue
                try:
                    ly = self._recover_y(lx, None)
                    if ly is not None:
                        candidates.append((lx, ly))
                        if len(candidates) >= 32:   # enough variety, stop early
                            break
                except Exception:
                    continue

        # Only fresh pt_src values are allowed for restarts.
        candidates = [(x, y) for x, y in candidates if self._pt_src_is_fresh(x)]

        if not candidates:
            self.walk_terminated = True
            if self.config.verbose:
                print(
                    f"[restart] no fresh restart point available after dead end: "
                    f"reason={reason}  exhausted_pt_src={len(self.exhausted_pt_src)}  visited_x={len(self.visited_x)}"
                )
            return None

        x, y = candidates[self._restart_cursor % len(candidates)]
        self._restart_cursor += 1
        self.current_x, self.current_y = x, y
        # Do NOT add x to visited_x or increment pt_src_visit_count here.
        # The restart point must remain "fresh" so that the next step can use
        # it as pt_src without being blocked by zero_novelty_thermal.
        # visited_x and pt_src_visit_count are updated in walker_step_search.py
        # when the step actually commits (walker.visited_x.add(tgt) etc.).

        if self.config.verbose:
            print(
                f"[restart] dead-end escape -> ({x}, {y})  reason={reason}  n={n}  "
                f"exhausted_pt_src={len(self.exhausted_pt_src)}"
            )

        return (x, y)

    def _step_direct(self, n, seed):
        return _step_direct(self, n, seed)

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

