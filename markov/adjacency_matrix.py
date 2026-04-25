from __future__ import annotations
import math, numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

"""adjacency_matrix.py

Row-stochastic transition matrix for the genus-2 Markov walk.

Two flavours are maintained in parallel:
  - "chain" : transitions from accepted (x_src -> x_step) steps only.
              One row per walked x_src, column is the chosen x_step.
              Path diagnostic only -- d~1 by construction.
  - "graph" : one row per walked x_src, uniform weight over the full L(x_src)
              candidate pool.  This is the row-truncated average operator:
              P(x_src -> x_step) = 1/|L(x_src)| if x_step in L(x_src), else 0.
              Rows are exact and deterministic (L(x_src) is fixed by the curve
              arithmetic).  Spectral gap of the source-induced subgraph is
              the correct mixing-time diagnostic.

Spectral gap
------------
Computed via scipy.sparse.linalg.eigs on the source-induced sparse subgraph.
Dest-only atoms (leaves never walked as x_src) are excluded -- they have no
outgoing row in the true operator.

Reporting cadence
-----------------
Reports are suppressed until min_collisions graph collisions are seen.
Printed once at end-of-run via maybe_report(force=True).

Usage
-----
    from adjacency_matrix import MarkovAdjacencyMatrix

    mat_chain = MarkovAdjacencyMatrix(p=65537, label="chain")
    mat_graph = MarkovAdjacencyMatrix(p=65537, label="graph",
                                      use_candidate_pool=True,
                                      normalize_per_step=True)

    for rec in walker.history:
        mat_chain.ingest(rec)
        mat_graph.ingest(rec)

    mat_chain.maybe_report(step_no=len(walker.history), force=True)
    mat_graph.maybe_report(step_no=len(walker.history), force=True)
"""

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
        True  -> weight over the full candidate_pool (x_step + x_res of every
                 candidate record).  Good for spectral estimation.
        False -> count only the accepted (x_src -> x_step) transition.
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
        min_collisions: int = 3,
        report_every: int = 10,
        n_eigenvalues: int = 20,
    ):
        self.p = p
        self.label = label
        self.use_candidate_pool = use_candidate_pool
        self.normalize_per_step = normalize_per_step
        self.min_collisions = min_collisions
        self.report_every = report_every
        # How many eigenvalues to request from the sparse solver.
        # Increase this to probe deeper into the spectrum.
        # The actual k used is min(n_eigenvalues, n_sources - 2).
        self.n_eigenvalues = n_eigenvalues

        # raw_counts[w_int][r_int] = accumulated float weight
        self._raw: Dict[Any, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))

        # stable ordered index: x_int -> row/col index
        self._index: Dict[Any, int] = {}
        self._atoms: List[Any] = []          # _atoms[i] == i-th x-coordinate

        self._steps_ingested: int = 0
        self._collision_count: int = 0       # graph collisions seen from walker
        self._last_report_at: int = 0

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
            x_src       = rec.get("x_src")
            x_step       = rec.get("x_step")
            pool     = rec.get("candidate_pool") or []
            accepted = rec.get("accepted", True)
            restart  = rec.get("restart", False)
        else:
            x_src       = getattr(rec, "x_src", None)
            x_step       = getattr(rec, "x_step", None)
            pool     = getattr(rec, "candidate_pool", None) or []
            accepted = getattr(rec, "accepted", True)
            restart  = getattr(rec, "restart", False)

        if x_src is None:
            return

        if graph_collision_count is not None:
            self._collision_count = graph_collision_count

        xi_int = _int_key(x_src)

        # ------ full-markov flavor ------
        # Every leaf x from the previous step points to the pool distribution
        # ------ collect leaves (chain / graph flavors) ------
        leaves: List[Any] = []

        if self.use_candidate_pool and pool:
            seen_this_step: set = set()
            for cand in pool:
                if isinstance(cand, dict):
                    for key in ("x_step", "x_res"):
                        v = cand.get(key)
                        if v is not None:
                            v_int = _int_key(v)
                            if v_int != xi_int and v_int not in seen_this_step:
                                leaves.append(v_int)
                                seen_this_step.add(v_int)
        else:
            # accepted-only (chain view): just the chosen x_step
            if accepted and not restart and x_step is not None:
                xj_int = _int_key(x_step)
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
        """Return the list of atoms that have appeared as x_src (have outgoing edges).

        This is the meaningful subspace for spectral analysis.  Atoms that only
        appear as destinations (x_step/x_res leaves) but have never been walked through
        as x_src contribute all-zero rows to the full matrix and pollute the
        eigenvalue computation with spurious zero eigenvalues.
        """
        return [x for x in self._atoms if self._raw.get(x)]

    def _build_sparse(self):
        """Return (Q_sym, source_atoms, all_atoms).

        Q is the square source-induced row-stochastic submatrix (sources x
        sources).  Transitions to dest-only leaves are absorbed by
        renormalising each row over source-to-source edges only.

        Q_sym = (Q + Q^T) / 2  is the additive reversiblization: a symmetric
        matrix whose eigenvalues are real, eliminating the complex-conjugate
        pairing artifact that appears when taking |eig| of a non-symmetric Q.
        gap(Q_sym) <= gap(Q), so it is a conservative mixing bound.
        eigsh (symmetric Lanczos) is used instead of eigs — faster and more
        numerically stable.

        Returns (None, [], []) if scipy unavailable or no data.
        """
        if not _SCIPY_AVAILABLE:
            return None, [], []

        sources = self._source_atoms()
        n_src = len(sources)
        if n_src == 0:
            return None, [], []

        all_atoms = self._atoms
        src_idx = {x: i for i, x in enumerate(sources)}
        src_set = set(sources)

        rows, cols, data = [], [], []
        for xi_int in sources:
            outgoing = self._raw[xi_int]
            sub_edges = {xr: wt for xr, wt in outgoing.items() if xr in src_set}
            total = sum(sub_edges.values())
            if total == 0:
                continue
            i = src_idx[xi_int]
            inv = 1.0 / total
            for xr_int, wt in sub_edges.items():
                rows.append(i)
                cols.append(src_idx[xr_int])
                data.append(wt * inv)

        if not data:
            return None, [], []

        Q = sp.csr_matrix((data, (rows, cols)), shape=(n_src, n_src), dtype=np.float64)
        # Additive reversiblization: symmetrize to get real eigenvalues.
        Q_sym = (Q + Q.T) * 0.5
        return Q_sym, sources, all_atoms

    def transition_matrix_dense(self) -> Tuple[np.ndarray, List[Any], List[Any]]:
        """Return (P_dense, source_atoms, all_atoms) as honest rectangular
        row-stochastic matrix.  Shape (n_sources, n_all_atoms), rows sum to 1.
        Only practical for small matrices (<5000 source atoms).
        """
        sources = self._source_atoms()
        n_src = len(sources)
        all_atoms = self._atoms
        n_all = len(all_atoms)
        row_idx = {x: i for i, x in enumerate(sources)}
        col_idx = {x: j for j, x in enumerate(all_atoms)}

        P = np.zeros((n_src, n_all), dtype=np.float64)
        for xi_int in sources:
            outgoing = self._raw[xi_int]
            total = sum(outgoing.values())
            if total == 0:
                continue
            i = row_idx[xi_int]
            inv = 1.0 / total
            for xr_int, wt in outgoing.items():
                j = col_idx.get(xr_int)
                if j is None:
                    continue
                P[i, j] = wt * inv

        return P, sources, all_atoms

    def _transition_matrix_square_dense(self) -> Tuple[np.ndarray, List[Any]]:
        """Return (Q_sym_dense, source_atoms): symmetrized square source submatrix.

        Q_sym = (Q + Q^T) / 2 where Q is the source-induced row-stochastic
        submatrix.  Real symmetric — correct input for np.linalg.eigh.
        """
        sources = self._source_atoms()
        n_src = len(sources)
        src_set = set(sources)
        src_idx = {x: i for i, x in enumerate(sources)}

        Q = np.zeros((n_src, n_src), dtype=np.float64)
        for xi_int in sources:
            outgoing = self._raw[xi_int]
            sub = {xr: wt for xr, wt in outgoing.items() if xr in src_set}
            total = sum(sub.values())
            if total == 0:
                continue
            i = src_idx[xi_int]
            inv = 1.0 / total
            for xr_int, wt in sub.items():
                Q[i, src_idx[xr_int]] = wt * inv

        Q_sym = (Q + Q.T) * 0.5
        return Q_sym, sources

    # ------------------------------------------------------------------
    # Spectral gap
    # ------------------------------------------------------------------

    def mean_out_degree(self) -> float:
        """Mean number of distinct destination atoms per source node.

        For the chain matrix this is ~1 (one accepted x_step per step).
        For the graph matrix this is ~candidate_pool_size.
        The true mixing time scales as O(1 / (d * gap)) where d is this value,
        because a degree-d walk covers d times as much ground per step.
        """
        sources = [x for x in self._atoms if self._raw.get(x)]
        if not sources:
            return 0.0
        return sum(len(v) for v in self._raw.values() if v) / len(sources)

    def spectral_gap(
        self,
        n_eigenvalues: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[np.ndarray]]:
        """Return (gap, lambda_1, lambda_2, eigenvalues).

        Computes eigenvalues of Q_sym = (Q + Q^T) / 2, the additive
        reversiblization of the source-induced row-stochastic submatrix Q.

        Q_sym is real symmetric, so:
          - eigenvalues are real (no complex-conjugate pairing artifact)
          - eigsh (symmetric Lanczos) is used — faster and more stable than eigs
          - gap(Q_sym) <= gap(Q), giving a conservative mixing bound

        lambda_1 = 1 for a connected row-stochastic Q (may be slightly <1 for
        Q_sym due to the subgraph restriction; we normalise by lambda_1).
        gap = lambda_1 - lambda_2,  O(1/gap) steps to mix.

        eigenvalues is a sorted descending real array, all in [0,1].
        """
        if n_eigenvalues is None:
            n_eigenvalues = self.n_eigenvalues

        n_nonempty = sum(1 for v in self._raw.values() if v)
        if n_nonempty < 4:
            return None, None, None, None

        k = min(n_nonempty - 2, n_eigenvalues)
        if k < 2:
            return None, None, None, None

        # --- sparse path (preferred) ---
        if _SCIPY_AVAILABLE and n_nonempty >= 10:
            Q_sym, sources, _ = self._build_sparse()
            n_src = len(sources)
            if Q_sym is not None and n_src >= 4:
                k_actual = min(n_src - 2, k)
                try:
                    # eigsh: symmetric Lanczos on Q_sym.
                    # which="LM" returns k largest-magnitude (= largest, since
                    # eigenvalues are real and Q_sym is positive-semidefinite
                    # by construction for a row-stochastic Q).
                    eig_vals = spla.eigsh(
                        Q_sym, k=k_actual, which="LM",
                        return_eigenvectors=False,
                        v0=np.ones(n_src) / math.sqrt(n_src),
                        tol=1e-6,
                    )
                    eig_vals = np.sort(np.abs(eig_vals))[::-1]
                    if eig_vals[0] > 1e-12:
                        eig_vals = eig_vals / eig_vals[0]
                    lam1 = float(eig_vals[0])
                    lam2 = float(eig_vals[1])
                    return lam1 - lam2, lam1, lam2, eig_vals
                except Exception:
                    pass  # fall through to dense

        # --- dense fallback ---
        if n_nonempty > 5000:
            return None, None, None, None
        try:
            Q_sym, sources = self._transition_matrix_square_dense()
            # eigh: full symmetric eigendecomposition (real, sorted ascending).
            eig_vals = np.linalg.eigh(Q_sym)[0]
            eig_vals = np.sort(np.abs(eig_vals))[::-1]
            if eig_vals[0] > 1e-12:
                eig_vals = eig_vals / eig_vals[0]
            lam1 = float(eig_vals[0])
            lam2 = float(eig_vals[1])
            return lam1 - lam2, lam1, lam2, eig_vals
        except np.linalg.LinAlgError:
            return None, None, None, None

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
        gap, lam1, lam2, all_eigs = self.spectral_gap()
        self._print_report(gap, lam1, lam2, all_eigs)
        return gap

    def _print_report(self, gap, lam1, lam2, all_eigs=None) -> None:
        n_total = len(self._atoms)
        n_sources = sum(1 for v in self._raw.values() if v)
        n_dest_only = n_total - n_sources
        sqrt_p = self.p ** 0.5 if self.p else None
        if self.use_candidate_pool:
            pool_note = "candidate-pool weighted"
        else:
            pool_note = "chain (accepted x_step only)"

        # --- in-degree distribution ---
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

        is_mixing_estimator = self.use_candidate_pool

        if not is_mixing_estimator:
            lines.append(
                f"|   [note] chain gap is a PATH diagnostic, not a mixing estimator.\n"
                f"|          The chain has d~1 by construction (one accepted x_step/step).\n"
                f"|          Use the graph/full matrix for mixing-time inference."
            )

        if gap is None:
            lines.append(f"|   spectral gap   : <matrix too sparse / small to estimate>")
        else:
            trustworthy = cv_density >= 0.05
            trust = "" if trustworthy else f"  [regime: {regime.split()[0]}]"

            # ---- full eigenvalue spectrum ----
            lines.append(f"|")
            lines.append(f"|   --- Eigenvalue Spectrum ({len(all_eigs) if all_eigs is not None else chr(63)} values, symmetrized Q_sym) ---")
            if all_eigs is not None and len(all_eigs) >= 2:
                # Print each eigenvalue with its consecutive gap and a bar chart
                bar_width = 30
                lines.append(f"|   {'idx':>4}  {'lambda_i':>10}  {'gap(i,i+1)':>12}  {'ratio l_{i+1}/l_i':>18}  bar")
                lines.append(f"|   " + "-" * 75)
                for i, lam in enumerate(all_eigs):
                    bar_fill = int(float(lam) * bar_width + 0.5)
                    bar = "█" * bar_fill + "░" * (bar_width - bar_fill)
                    if i + 1 < len(all_eigs):
                        consec_gap = float(lam) - float(all_eigs[i + 1])
                        ratio = float(all_eigs[i + 1]) / float(lam) if float(lam) > 1e-12 else float("nan")
                        lines.append(
                            f"|   {i:>4}  {float(lam):>10.6f}  {consec_gap:>12.6f}  {ratio:>18.6f}  {bar}"
                        )
                    else:
                        lines.append(
                            f"|   {i:>4}  {float(lam):>10.6f}  {'(last)':>12}  {'':>18}  {bar}"
                        )

                # ---- spectral gap summary ----
                lines.append(f"|")
                lines.append(f"|   --- Gap Summary ---")
                lines.append(f"|   lambda_1 = {lam1:.6f}  (should be ~= 1 for row-stochastic Q){trust}")
                lines.append(f"|   lambda_2 = {lam2:.6f}{trust}")
                lines.append(f"|   spectral gap (lambda_1 - lambda_2) = {gap:.6f}{trust}")

                # Identify the largest consecutive gap beyond lam1→lam2 (spectral clustering hint)
                if len(all_eigs) >= 3:
                    consec_gaps = [
                        (i, float(all_eigs[i]) - float(all_eigs[i + 1]))
                        for i in range(1, len(all_eigs) - 1)
                    ]
                    if consec_gaps:
                        best_i, best_gap = max(consec_gaps, key=lambda x: x[1])
                        lines.append(
                            f"|   largest sub-gap : gap({best_i},{best_i+1}) = {best_gap:.6f}  "
                            f"(lambda_{best_i} = {float(all_eigs[best_i]):.6f}  "
                            f"→  lambda_{best_i+1} = {float(all_eigs[best_i+1]):.6f})"
                        )
                        lines.append(
                            f"|     → suggests {best_i+1} near-stationary component(s) in the operator"
                        )

                # ---- mixing time estimates ----
                if gap > 1e-12:
                    t_mix = 1.0 / gap
                    if is_mixing_estimator:
                        lines.append(f"|   O(1/gap)       : {t_mix:.1f} steps{trust}")
                        if sqrt_p and trustworthy:
                            lines.append(
                                f"|   O(1/gap)/√p    : {t_mix/sqrt_p:.4f}  (want O(1) or polylog)"
                            )
                    else:
                        lines.append(
                            f"|   O(1/gap)       : {t_mix:.1f}  (path diagnostic only — not mixing time)"
                        )
                else:
                    lines.append(
                        f"|   O(1/gap)       : inf  "
                        + ("(gap ~= 0 -- disconnected?)" if trustworthy
                           else "(expected -- matrix still a forest/sparse)")
                    )
            else:
                # Fallback: no eigenvalue array (shouldn't happen if gap is set)
                lines.append(f"|   lambda_1       : {lam1:.6f}  (should be ~= 1){trust}")
                lines.append(f"|   lambda_2       : {lam2:.6f}{trust}")
                lines.append(f"|   spectral gap   : {gap:.6f}  (= lambda_1 - lambda_2){trust}")

        if not _SCIPY_AVAILABLE:
            lines.append(f"|   [note] scipy not found; used dense numpy eigvals (all eigenvalues computed)")

        lines.append(f"+{'-'*60}")
        print("\n".join(lines), flush=True)

    def full_spectrum(self, n_eigenvalues: Optional[int] = None) -> Optional[np.ndarray]:
        """Return sorted descending array of |eigenvalue| for all computed eigenvalues.

        This is the raw array for downstream analysis — plotting, plateau detection,
        expander-bound checking, etc.  Returns None if the matrix is too small.

        Parameters
        ----------
        n_eigenvalues : int, optional
            How many eigenvalues to request.  Defaults to self.n_eigenvalues.
            Pass a larger value (or None to use the instance default) to probe
            deeper into the tail of the spectrum.
        """
        _, _, _, all_eigs = self.spectral_gap(n_eigenvalues=n_eigenvalues)
        return all_eigs

    def print_spectrum(self, n_eigenvalues: Optional[int] = None) -> None:
        """Print the full eigenvalue spectrum report unconditionally.

        Useful for post-run analysis without triggering the cadence gate.
        """
        gap, lam1, lam2, all_eigs = self.spectral_gap(n_eigenvalues=n_eigenvalues)
        self._print_report(gap, lam1, lam2, all_eigs)

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
        P, sources, _ = self.transition_matrix_dense()
        src_idx = {s: i for i, s in enumerate(sources)}
        i = src_idx.get(xi_int)
        if i is None:
            return None
        return P[i]

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
