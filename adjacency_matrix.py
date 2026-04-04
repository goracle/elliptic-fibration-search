"""adjacency_matrix.py

Row-stochastic transition matrix for the genus-2 Markov walk.

Three flavours are maintained in parallel:
  - "chain"  : transitions from accepted (xi -> xj) steps only.
               Rows are the actual chain xi; columns are the chosen xj.
               This is the matrix whose mixing time you care about for the
               cryptographic argument.  Requires path collisions to get
               multi-observation rows; useless until xi is revisited.
  - "graph"  : transitions weighted over the full candidate_pool.
               Every (xi, xj) and (xi, xk) leaf pair contributes weight.
               Denser, better-conditioned for spectral estimation even at
               small step counts.  Suffers from dest-only node pruning.
  - "full"   : the literal Markov process over all observed x-coordinates.
               M_ij = P(next position is j | current position is i).
               Construction: at step t, xi_t was reached because it was a
               leaf at step t-1.  So every leaf x of step t-1 gets a row
               whose outgoing distribution is the pool of step t (the step
               that x "feeds into" as xi).  xi itself also gets a row for
               the same pool.  This makes every observed x-coordinate a
               first-class node with a meaningful outgoing distribution,
               regardless of whether it was ever walked as xi.  The spectral
               gap of this matrix is the correct mixing-time diagnostic.

Spectral gap
------------
Computed via scipy.sparse.linalg.eigs on the sparse row-stochastic matrix.
For a matrix of size n we need at most k=min(n-2, 6) eigenvalues, which is
cheap even when n ~ 10,000.

With 76 leaves/step the graph matrix will have O(steps x 76) distinct atoms.
After 150 steps that is ~11,000 atoms.  The sparse eigensolver runs in well
under a second at that size.

Reporting cadence
-----------------
Call maybe_report(step_no) from the run loop.  The default is to print every
10 steps once 3+ graph collisions are seen, then always at the end.
Adjust with MarkovAdjacencyMatrix(report_every=N, min_collisions=K).

Usage
-----
    from adjacency_matrix import MarkovAdjacencyMatrix

    mat_chain = MarkovAdjacencyMatrix(p=33554467, label="chain")
    mat_graph = MarkovAdjacencyMatrix(p=33554467, label="graph",
                                      use_candidate_pool=True,
                                      normalize_per_step=True)
    mat_full  = MarkovAdjacencyMatrix(p=33554467, label="full",
                                      use_full_markov=True)

    # chain and graph: ingest one record at a time
    for rec in walker.history:
        mat_chain.ingest(rec)
        mat_graph.ingest(rec)

    # full: must be ingested in pairs so leaves of step t point to pool of t+1
    mat_full.ingest_all(walker.history)

    mat_chain.maybe_report(step_no=len(walker.history), force=True)
    mat_graph.maybe_report(step_no=len(walker.history), force=True)
    mat_full.maybe_report(step_no=len(walker.history), force=True)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_key(x) -> Any:
    try:
        return int(x)
    except Exception:
        return x  # fallback: keep as-is for hashable Sage elements


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MarkovAdjacencyMatrix:
    """Incrementally build and analyse a row-stochastic transition matrix.

    Parameters
    ----------
    p : int, optional
        Field prime -- used only for sqrt(p) reference in reports.
    label : str
        Short name printed in reports ("chain" or "graph").
    use_candidate_pool : bool
        True  -> weight over the full candidate_pool (xj + xk of every
                 candidate record).  Good for spectral estimation.
        False -> count only the accepted (xi -> xj) transition.
    normalize_per_step : bool
        True  -> divide each step's leaf weights by the number of leaves
                 before accumulating, so every step contributes total weight 1
                 regardless of fan-out.  Recommended when use_candidate_pool=True.
        False -> raw counts; rows normalised only at query time.
    min_collisions : int
        Graph collision count required before spectral estimation is printed.
    report_every : int
        Minimum steps between spectral reports (default 10).
    """

    def __init__(
        self,
        p: Optional[int] = None,
        label: str = "matrix",
        use_candidate_pool: bool = False,
        normalize_per_step: bool = False,
        use_full_markov: bool = False,
        min_collisions: int = 3,
        report_every: int = 10,
    ):
        self.p = p
        self.label = label
        self.use_candidate_pool = use_candidate_pool
        self.normalize_per_step = normalize_per_step
        self.use_full_markov = use_full_markov
        self.min_collisions = min_collisions
        self.report_every = report_every

        # raw_counts[w_int][r_int] = accumulated float weight
        self._raw: Dict[Any, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))

        # stable ordered index: x_int -> row/col index
        self._index: Dict[Any, int] = {}
        self._atoms: List[Any] = []          # _atoms[i] == i-th x-coordinate

        self._steps_ingested: int = 0
        self._collision_count: int = 0       # graph collisions seen from walker
        self._last_report_at: int = 0

        # full-markov flavor: buffer the previous step's leaves so we can
        # wire them to the current step's pool on the next ingest call.
        self._prev_leaves: List[Any] = []    # leaf x-ints from the last step

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _idx(self, x_int) -> int:
        if x_int not in self._index:
            self._index[x_int] = len(self._atoms)
            self._atoms.append(x_int)
        return self._index[x_int]

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, rec, graph_collision_count: Optional[int] = None) -> None:
        """Add one RelationRecord (dataclass or dict) to the matrix.

        graph_collision_count : pass walker.leaf_collision_count so the
        threshold check uses the walker's authoritative counter.
        """
        if isinstance(rec, dict):
            xi       = rec.get("xi")
            xj       = rec.get("xj")
            pool     = rec.get("candidate_pool") or []
            accepted = rec.get("accepted", True)
            restart  = rec.get("restart", False)
        else:
            xi       = getattr(rec, "xi", None)
            xj       = getattr(rec, "xj", None)
            pool     = getattr(rec, "candidate_pool", None) or []
            accepted = getattr(rec, "accepted", True)
            restart  = getattr(rec, "restart", False)

        if xi is None:
            return

        if graph_collision_count is not None:
            self._collision_count = graph_collision_count

        xi_int = _int_key(xi)

        # ------ full-markov flavor ------
        # Every leaf x from the previous step points to the pool distribution
        # of the current step (the step that x "feeds into" as xi).
        # xi itself also gets a row for the current pool.
        if self.use_full_markov:
            cur_pool_leaves: List[Any] = []
            seen_fm: set = set()
            for cand in pool:
                if isinstance(cand, dict):
                    for key in ("xj", "xk"):
                        v = cand.get(key)
                        if v is not None:
                            v_int = _int_key(v)
                            if v_int != xi_int and v_int not in seen_fm:
                                cur_pool_leaves.append(v_int)
                                seen_fm.add(v_int)

            if cur_pool_leaves:
                weight = 1.0 / len(cur_pool_leaves)
                for v_int in cur_pool_leaves:
                    self._idx(v_int)
                # xi row -> current pool
                self._idx(xi_int)
                for v_int in cur_pool_leaves:
                    self._raw[xi_int][v_int] += weight
                # every previous-step leaf -> same pool distribution
                for src in self._prev_leaves:
                    self._idx(src)
                    for v_int in cur_pool_leaves:
                        self._raw[src][v_int] += weight

            # buffer current pool leaves for the next step
            self._prev_leaves = cur_pool_leaves
            self._steps_ingested += 1
            return

        # ------ collect leaves (chain / graph flavors) ------
        leaves: List[Any] = []

        if self.use_candidate_pool and pool:
            seen_this_step: set = set()
            for cand in pool:
                if isinstance(cand, dict):
                    for key in ("xj", "xk"):
                        v = cand.get(key)
                        if v is not None:
                            v_int = _int_key(v)
                            if v_int != xi_int and v_int not in seen_this_step:
                                leaves.append(v_int)
                                seen_this_step.add(v_int)
        else:
            # accepted-only (chain view): just the chosen xj
            if accepted and not restart and xj is not None:
                xj_int = _int_key(xj)
                if xj_int != xi_int:
                    leaves.append(xj_int)

        if not leaves:
            self._steps_ingested += 1
            return

        self._idx(xi_int)   # ensure source node is indexed

        weight = (1.0 / len(leaves)) if self.normalize_per_step else 1.0

        for xr_int in leaves:
            self._idx(xr_int)
            self._raw[xi_int][xr_int] += weight

        self._steps_ingested += 1

    def ingest_all(self, records, graph_collision_count: Optional[int] = None) -> None:
        for rec in records:
            self.ingest(rec, graph_collision_count=graph_collision_count)

    # ------------------------------------------------------------------
    # Build sparse row-stochastic matrix
    # ------------------------------------------------------------------

    def _source_atoms(self) -> List[Any]:
        """Return the list of atoms that have appeared as xi (have outgoing edges).

        This is the meaningful subspace for spectral analysis.  Atoms that only
        appear as destinations (xj/xk leaves) but have never been walked through
        as xi contribute all-zero rows to the full matrix and pollute the
        eigenvalue computation with spurious zero eigenvalues.
        """
        return [x for x in self._atoms if self._raw.get(x)]

    def _build_sparse(self):
        """Return (P_sparse, source_atoms) restricted to the induced subgraph
        on source nodes.  Returns (None, []) if scipy unavailable."""
        if not _SCIPY_AVAILABLE:
            return None, []

        sources = self._source_atoms()
        n = len(sources)
        if n == 0:
            return None, []

        # Local re-index over source atoms only.
        local_idx = {x: i for i, x in enumerate(sources)}

        rows, cols, data = [], [], []
        for xi_int in sources:
            outgoing = self._raw[xi_int]
            total = sum(outgoing.values())
            if total == 0:
                continue
            i = local_idx[xi_int]
            inv = 1.0 / total
            for xr_int, wt in outgoing.items():
                j = local_idx.get(xr_int)
                if j is None:
                    # Destination not yet a source — skip for spectral purposes.
                    continue
                rows.append(i)
                cols.append(j)
                data.append(wt * inv)

        if not data:
            return None, []

        P = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        return P, sources

    def transition_matrix_dense(self) -> Tuple[np.ndarray, List[Any]]:
        """Return (P_dense, source_atoms) restricted to the induced subgraph.

        Only practical for small matrices (<5000 source atoms).
        """
        sources = self._source_atoms()
        n = len(sources)
        local_idx = {x: i for i, x in enumerate(sources)}

        P = np.zeros((n, n), dtype=np.float64)
        for xi_int in sources:
            outgoing = self._raw[xi_int]
            total = sum(outgoing.values())
            if total == 0:
                continue
            i = local_idx[xi_int]
            inv = 1.0 / total
            for xr_int, wt in outgoing.items():
                j = local_idx.get(xr_int)
                if j is None:
                    continue
                P[i, j] = wt * inv
        return P, sources

    # ------------------------------------------------------------------
    # Spectral gap
    # ------------------------------------------------------------------

    def mean_out_degree(self) -> float:
        """Mean number of distinct destination atoms per source node.

        For the chain matrix this is ~1 (one accepted xj per step).
        For the graph matrix this is ~candidate_pool_size.
        The true mixing time scales as O(1 / (d * gap)) where d is this value,
        because a degree-d walk covers d times as much ground per step.
        """
        sources = [x for x in self._atoms if self._raw.get(x)]
        if not sources:
            return 0.0
        return sum(len(v) for v in self._raw.values() if v) / len(sources)

    def spectral_gap(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (gap, |lam1|, |lam2|).

        Uses scipy sparse eigs (Arnoldi) for speed.  Falls back to dense
        numpy eigvals for small matrices or when scipy is absent.

        gap = |lam1| - |lam2|   (for row-stochastic, lam1 ~= 1.0)
        """
        n = len(self._atoms)
        if n < 4:
            return None, None, None

        n_nonempty = sum(1 for v in self._raw.values() if v)
        if n_nonempty < 4:
            return None, None, None

        k = min(n_nonempty - 2, 6)
        if k < 2:
            return None, None, None

        # --- sparse path (preferred) ---
        if _SCIPY_AVAILABLE and n >= 10:
            P_sparse, sources = self._build_sparse()
            n_src = len(sources)
            if P_sparse is not None and n_src >= 4:
                try:
                    vals = spla.eigs(
                        P_sparse, k=min(n_src - 2, k), which="LM",
                        return_eigenvectors=False,
                        v0=np.ones(n_src) / math.sqrt(n_src),
                        maxiter=n_src * 10,
                        tol=1e-6,
                    )
                    abs_vals = np.sort(np.abs(vals))[::-1]
                    lam1 = float(abs_vals[0])
                    lam2 = float(abs_vals[1])
                    return lam1 - lam2, lam1, lam2
                except Exception:
                    pass  # fall through to dense

        # --- dense fallback ---
        if n > 5000:
            return None, None, None   # too large for dense
        try:
            P, _ = self.transition_matrix_dense()
            eigs = np.linalg.eigvals(P)
            abs_eigs = np.sort(np.abs(eigs))[::-1]
            lam1 = float(abs_eigs[0])
            lam2 = float(abs_eigs[1])
            return lam1 - lam2, lam1, lam2
        except np.linalg.LinAlgError:
            return None, None, None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def maybe_report(self, step_no: int, force: bool = False) -> Optional[float]:
        """Print a spectral-gap report if conditions are met.

        Parameters
        ----------
        step_no : int
            Current step number from the walker (for cadence check).
        force : bool
            Print regardless of thresholds (use at end-of-run).

        Returns the gap value, or None if not printed.
        """
        if not force:
            if self._collision_count < self.min_collisions:
                return None
            if (step_no - self._last_report_at) < self.report_every:
                return None

        self._last_report_at = step_no
        gap, lam1, lam2 = self.spectral_gap()
        self._print_report(gap, lam1, lam2)
        return gap

    def _print_report(self, gap, lam1, lam2) -> None:
        n_total = len(self._atoms)
        n_sources = sum(1 for v in self._raw.values() if v)
        n_dest_only = n_total - n_sources
        sqrt_p = self.p ** 0.5 if self.p else None
        if self.use_full_markov:
            pool_note = "full-markov (leaves -> next pool)"
        elif self.use_candidate_pool:
            pool_note = "candidate-pool weighted"
        else:
            pool_note = "chain (accepted xj only)"

        # --- in-degree distribution ---
        # Count how many times each atom appears as a destination across all rows.
        in_deg: Dict[Any, int] = defaultdict(int)
        for outgoing in self._raw.values():
            for xr in outgoing:
                in_deg[xr] += 1
        n_indeg_ge2 = sum(1 for d in in_deg.values() if d >= 2)
        n_indeg_ge3 = sum(1 for d in in_deg.values() if d >= 3)
        n_indeg_ge5 = sum(1 for d in in_deg.values() if d >= 5)
        multi_frac  = n_indeg_ge2 / n_total if n_total > 0 else 0.0

        # --- C/V density and regime ---
        C = self._collision_count
        V = n_total
        cv_density = C / V if V > 0 else 0.0
        if cv_density < 0.001:
            regime = "forest            [spectral meaningless]"
        elif cv_density < 0.01:
            regime = "sparse overlap     [spectral unreliable]"
        elif cv_density < 0.05:
            regime = "connectivity forming [first signal]"
        else:
            regime = "well-connected     [estimate trustworthy]"

        # Steps to reach next regime thresholds (birthday formula V ~ sqrt(2pC))
        steps_to = {}
        if sqrt_p and self._steps_ingested > 0:
            leaves_per_step = n_total / self._steps_ingested
            for c_target, label in [(20, "sparse"), (200, "first-signal"), (1000, "trustworthy")]:
                if C < c_target and leaves_per_step > 0:
                    v_needed = (2 * self.p * c_target) ** 0.5
                    steps_needed = max(0, (v_needed - n_total) / leaves_per_step)
                    steps_to[label] = int(steps_needed)

        lines = [
            f"",
            f"+-- AdjacencyMatrix [{self.label}]  {pool_note}",
            f"|   atoms total    : {n_total}  (sources: {n_sources}  dest-only: {n_dest_only})",
            f"|   steps ingested : {self._steps_ingested}   collisions (C): {C}",
        ]
        if sqrt_p:
            lines.append(
                f"|   atoms/sqrt(p)  : {n_total/sqrt_p:.4f}   "
                f"sources/sqrt(p): {n_sources/sqrt_p:.4f}   (sqrt(p) = {sqrt_p:.1f})"
            )
        lines.append(
            f"|   C/V density    : {cv_density:.4%}   regime: {regime}"
        )
        lines.append(
            f"|   in-deg >=2     : {n_indeg_ge2}/{n_total} ({multi_frac:.2%})   "
            f">=3: {n_indeg_ge3}   >=5: {n_indeg_ge5}"
        )
        if steps_to:
            eta_parts = "   ".join(f"~{v} steps to {k}" for k, v in steps_to.items())
            lines.append(f"|   ETA            : {eta_parts}")

        d = self.mean_out_degree()
        lines.append(f"|   mean out-degree: {d:.2f}  (distinct dest atoms per source)")

        # Whether this matrix's spectral gap estimates mixing time.
        # Chain matrix (d~1) is a realized sample path -- its gap measures
        # path structure, not the branching walk.  Only the graph/full matrix
        # (row-normalized transition kernel over the full candidate pool) is
        # the correct mixing-time object.
        is_mixing_estimator = self.use_candidate_pool or self.use_full_markov

        if not is_mixing_estimator:
            lines.append(
                f"|   [note] chain gap is a PATH diagnostic, not a mixing estimator.\n"
                f"|          The chain has d~1 by construction (one accepted xj/step).\n"
                f"|          Use the graph/full matrix for mixing-time inference."
            )

        if gap is None:
            lines.append(f"|   spectral gap   : <matrix too sparse / small to estimate>")
        else:
            trustworthy = cv_density >= 0.05
            trust = "" if trustworthy else f"  [regime: {regime.split()[0]}]"
            lines.append(f"|   |lam1|         : {lam1:.6f}  (should be ~= 1){trust}")
            lines.append(f"|   |lam2|         : {lam2:.6f}{trust}")
            lines.append(f"|   spectral gap   : {gap:.6f}  (= |lam1| - |lam2|){trust}")
            if gap > 1e-12:
                t_mix = 1.0 / gap
                if is_mixing_estimator:
                    lines.append(f"|   O(1/gap)       : {t_mix:.1f} steps{trust}")
                    if sqrt_p and trustworthy:
                        lines.append(
                            f"|   O(1/gap)/√p    : {t_mix/sqrt_p:.4f}  (want O(1) or polylog)"
                        )
                else:
                    # Chain: report gap for completeness but suppress mixing interpretation
                    lines.append(
                        f"|   O(1/gap)       : {t_mix:.1f}  (path diagnostic only — not mixing time)"
                    )
            else:
                lines.append(
                    f"|   O(1/gap)       : inf  "
                    + ("(gap ~= 0 -- disconnected?)" if trustworthy
                       else "(expected -- matrix still a forest/sparse)")
                )

        if not _SCIPY_AVAILABLE:
            lines.append(f"|   [note] scipy not found; used dense numpy eigvals")

        lines.append(f"+{'-'*60}")
        print("\n".join(lines), flush=True)

    def summary(self) -> None:
        n = len(self._atoms)
        n_rows = sum(1 for v in self._raw.values() if v)
        print(
            f"[AdjacencyMatrix/{self.label}] "
            f"{n} atoms | {self._steps_ingested} steps | "
            f"{n_rows} rows w/ edges | "
            f"{self._collision_count} collisions"
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def row_for(self, x) -> Optional[np.ndarray]:
        xi_int = _int_key(x)
        if xi_int not in self._index:
            return None
        P, _ = self.transition_matrix_dense()
        return P[self._index[xi_int]]

    def top_transitions(self, x, k: int = 10) -> List[Tuple[Any, float]]:
        """Return the k largest outgoing transition probabilities from atom x."""
        xi_int = _int_key(x)
        outgoing = self._raw.get(xi_int)
        if not outgoing:
            return []
        total = sum(outgoing.values())
        if total == 0:
            return []
        ranked = sorted(outgoing.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(xr, wt / total) for xr, wt in ranked]

    @property
    def n_atoms(self) -> int:
        return len(self._atoms)

    @property
    def n_steps(self) -> int:
        return self._steps_ingested
