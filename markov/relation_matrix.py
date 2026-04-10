from __future__ import annotations
import types
from sage.all import Matrix, ZZ, QQ, GF
from typing import Any, List, Optional, Sequence, Tuple

"""relation_matrix.py

Encode the Markov walk history as an integer divisor-relation matrix and
compute its rank.

Each accepted step produces a principal divisor relation of the form

    (d-2)·[xi] + [xj] + [xk] - d·[∞] = 0          (curve degree d, default 5)

The columns of the matrix are the **d1 atoms**: the unique finite
x-coordinates that appear in any role (xi / xj / xk) across the full set of
accepted, complete relations.  The point at infinity is appended as the last
column.

Each row encodes one relation:
    col[xi]  += (d - 2)
    col[xj]  += 1
    col[xk]  += 1
    col[∞]   += -d

The matrix lives over ZZ.  Rank is computed via Sage's exact arithmetic
(Matrix_integer_dense or Matrix_rational_dense, depending on what Sage gives
us for the small-coefficient case).

Usage
-----
After running the walker:

    from relation_matrix import build_relation_matrix, print_relation_matrix_summary

    mat, atoms, used = build_relation_matrix(walker.history, curve_degree=5)
    print_relation_matrix_summary(mat, atoms, used)

Or attach it as a method:

    # monkey-patch
    from relation_matrix import build_relation_matrix, print_relation_matrix_summary
    import types

    def _walker_relation_matrix(self, **kw):
        return build_relation_matrix(self.history, curve_degree=self.config.curve_degree, **kw)

    Genus2MetropolisWalker.relation_matrix = _walker_relation_matrix

Then just call walker.relation_matrix().
"""
# DEAR AI READING THIS:  RAISE ALL EXCEPTIONS
# DEAR AI READING THIS:  all imports are available

_SAGE_AVAILABLE = True
# ---------------------------------------------------------------------------
# Try to import Sage.  If we are not inside a Sage session, fall back to a
# pure-Python integer matrix so the module is still importable for testing.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_INFINITY_SENTINEL = "∞"

def attach_to_walker(walker_class) -> None:
    """Monkey-patch build_relation_matrix + summary onto a walker class.

    After calling this once:

        attach_to_walker(Genus2MetropolisWalker)

    you can do:

        mat, atoms, used = walker.relation_matrix()
        walker.print_relation_summary()
    """

    def _relation_matrix(self, **kwargs):
        cd = getattr(getattr(self, "config", None), "curve_degree", 5)
        return build_relation_matrix(self.history, curve_degree=cd, **kwargs)

    def _print_relation_summary(self, **kwargs):
        mat, atoms, used = self.relation_matrix()
        print_relation_matrix_summary(mat, atoms, used, **kwargs)

    walker_class.relation_matrix = _relation_matrix
    walker_class.print_relation_summary = _print_relation_summary

def _get(rec, key):
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)

def build_relation_matrix2(
    history: Sequence[Any],
    *,
    curve_degree: int = 5,
    include_infinity: bool = True,
    accepted_only: bool = True,
    require_xk: bool = False,
    include_step_leaves: bool = True,
) -> Tuple[Any, List[Any], List[Any]]:
    """Build the integer divisor-relation matrix from a walker history.

    This version explodes step-level leaf data (candidate_pool) into full rows,
    generating a divisor relation for EVERY candidate found during the step.
    """
    xi_mult = curve_degree - 2
    inf_coeff = -curve_degree

    atom_index: dict[Any, int] = {}
    used_records: List[Any] = []
    skipped_degenerate = 0

    def _iter_step_leaves(rec):
        """Yield extra leaf atoms stored in step/search payloads."""
        if not include_step_leaves:
            return

        step = _get(rec, "step")
        if isinstance(step, dict):
            for key in ("candidate_xs", "found_xs"):
                xs = step.get(key)
                if xs is None:
                    continue
                if isinstance(xs, dict):
                    xs = xs.keys()
                try:
                    for x in xs:
                        if x is not None:
                            yield x
                except TypeError:
                    if xs is not None:
                        yield xs

        pool = _get(rec, "candidate_pool")
        if pool:
            for cand in pool:
                if isinstance(cand, dict):
                    # Added 'xk' so xk candidates are guaranteed columns
                    for key in ("xj", "x", "candidate_x", "x_value", "xk"):
                        x = cand.get(key)
                        if x is not None:
                            yield x
                elif cand is not None:
                    yield cand

        sel = _get(rec, "selected_candidate")
        if isinstance(sel, dict):
            for key in ("xj", "x", "candidate_x", "x_value", "xk"):
                x = sel.get(key)
                if x is not None:
                    yield x

    # Pass 1a: register atoms from ALL accepted non-involution records BEFORE filtering.
    #
    # Involution/free records are excluded here too: their xj values are just
    # T-images of existing walk atoms (xj<->xk swap), so they add no new columns.
    # Excluding them keeps atom_index clean and the column count honest.
    for rec in history:
        if accepted_only and not _get(rec, "accepted"):
            continue

        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            continue

        xi = _get(rec, "xi")
        xj = _get(rec, "xj")
        xk = _get(rec, "xk")

        # Need at least one of xi/xj to be meaningful.
        if xi is None and xj is None:
            continue

        # Register primary atoms unconditionally (no degenerate skip here).
        for x in (xi, xj, xk):
            if x is not None and x not in atom_index:
                atom_index[x] = len(atom_index)

        # Step-level leaf atoms, if available.
        for x in _iter_step_leaves(rec):
            if x is not None and x not in atom_index:
                atom_index[x] = len(atom_index)

    # Pass 1b: build used_records with full validity checks.
    # Involution/free records are excluded: they are proven algebraically identical
    # to existing walk rows (the relation is symmetric in xj and xk, so T(xj)=xk
    # just produces the same row with xj and xk swapped).  Including them adds
    # zero rank and pollutes the atom list with duplicate columns.
    for rec in history:
        if accepted_only and not _get(rec, "accepted"):
            continue

        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            continue

        xi = _get(rec, "xi")
        xj = _get(rec, "xj")

        if xi is None or xj is None:
            continue

        if xi == xj:
            skipped_degenerate += 1
            continue

        used_records.append(rec)

    if skipped_degenerate:
        print(f"[relation_matrix] Skipped {skipped_degenerate} degenerate relations where xi == xj.")

    if not used_records:
        print("[relation_matrix] No usable relations found in history.")
        atoms: List[Any] = list(atom_index.keys())
        if include_infinity:
            atoms.append(_INFINITY_SENTINEL)
        return Matrix(ZZ, 0, len(atoms)), atoms, []

    finite_atoms: List[Any] = sorted(atom_index.keys(), key=lambda a: atom_index[a])
    atoms: List[Any] = list(finite_atoms)

    if include_infinity:
        atoms.append(_INFINITY_SENTINEL)
        inf_col = len(finite_atoms)
    else:
        inf_col = None

    n_cols = len(atoms)

    # Pass 2: build rows.
    rows: List[List[int]] = []
    n_involution_rows = 0
    for rec in used_records:
        rows_before_this_rec = len(rows)
        xi = _get(rec, "xi")
        if xi is None or xi not in atom_index:
            continue

        cands_to_add = []
        if include_step_leaves:
            pool = _get(rec, "candidate_pool")
            if pool:
                for cand in pool:
                    if isinstance(cand, dict):
                        c_xj = next((cand[k] for k in ("xj", "x", "candidate_x", "x_value") if k in cand and cand[k] is not None), None)
                        c_xk = cand.get("xk")
                        cands_to_add.append((c_xj, c_xk))
                    elif cand is not None:
                        cands_to_add.append((cand, None))

        # Always include the primary accepted path
        cands_to_add.append((_get(rec, "xj"), _get(rec, "xk")))

        seen_pairs = set()

        for cxj, cxk in cands_to_add:
            if cxj is None or cxj == xi:
                continue

            if require_xk and cxk is None:
                continue

            # Normalize xk
            if cxk is not None and (cxk == xi or cxk == cxj):
                cxk = None

            if cxj not in atom_index:
                # This should never happen after the two-phase pass-1 fix:
                # pass-1a registers atoms from ALL accepted records before any
                # filtering, so every (xi, xj, xk) from every accepted record
                # (including free/involution records) is in atom_index by now.
                raise AssertionError(
                    f"[relation_matrix] BUG: cxj={cxj!r} not in atom_index "
                    f"(xi={_get(rec, 'xi')!r}, source={_get(rec, 'step')!r}).  "
                    f"Pass-1a missed this atom — please report."
                )

            # Deduplicate identical (xj, xk) and (xk, xj) pairs from the same step
            pair_key = frozenset([cxj, cxk]) if cxk is not None else frozenset([cxj])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            row = [0] * n_cols
            row[atom_index[xi]] += xi_mult
            row[atom_index[cxj]] += 1

            if cxk == "∞":
                if inf_col is not None:
                    row[inf_col] += 1
            elif cxk is not None and cxk in atom_index:
                row[atom_index[cxk]] += 1

            if inf_col is not None:
                row[inf_col] += inf_coeff

            rows.append(row)

        step_src = _get(rec, "step")
        if isinstance(step_src, dict) and step_src.get("source") == "involution_closure":
            # Count every row emitted for involution records (there is normally 1).
            n_involution_rows += (len(rows) - rows_before_this_rec)

    if n_involution_rows:
        print(f"[relation_matrix] Involution/free records contributed {n_involution_rows} rows.")

    mat = Matrix(ZZ, rows)
    return mat, atoms, used_records

def print_relation_matrix_summary(
    mat: Any,
    atoms: List[Any],
    used_records: List[Any],
    *,
    show_matrix: bool = False,
    max_show_rows: int = 20,
    max_show_cols: int = 30,
) -> None:
    """Pretty-print the shape, atom list, and rank of the relation matrix."""
    n_rows, n_cols = mat.nrows(), mat.ncols()

    step_leaf_atoms = set()
    for rec in used_records:
        step = _get(rec, "step")
        if isinstance(step, dict):
            for key in ("candidate_xs", "found_xs"):
                xs = step.get(key)
                if xs is None:
                    continue
                if isinstance(xs, dict):
                    xs = xs.keys()
                try:
                    step_leaf_atoms.update(x for x in xs if x is not None)
                except TypeError:
                    if xs is not None:
                        step_leaf_atoms.add(xs)

        pool = _get(rec, "candidate_pool")
        if pool:
            for cand in pool:
                if isinstance(cand, dict):
                    # Added 'xk' to summary tracker
                    for key in ("xj", "x", "candidate_x", "x_value", "xk"):
                        x = cand.get(key)
                        if x is not None:
                            step_leaf_atoms.add(x)
                elif cand is not None:
                    step_leaf_atoms.add(cand)

    print("\n" + "=" * 70)
    print("DIVISOR RELATION MATRIX SUMMARY")
    print("=" * 70)
    print(f"  Relations encoded  : {n_rows} rows")
    print(f"  Matrix size        : {n_rows} x {n_cols}")
    print(f"  Leaf atoms seen    : {len(step_leaf_atoms)} extra leaf values in step payloads")
    print(f"  D1 atoms (columns) : {n_cols} cols  ({max(0, n_cols - 1)} finite x-coords + ∞)")

    finite_count = n_cols - 1
    print("\n  Column layout:")
    print(f"    cols 0 .. {finite_count - 1}  ->  finite x-coordinates")
    print(f"    col  {finite_count}          ->  ∞")

    if len(atoms) <= 30:
        print("\n  Atom index  (col -> x-value):")
        for i, a in enumerate(atoms):
            print(f"    [{i:3d}]  {a}")
    else:
        print("\n  Atom index  (first 10 / last 5):")
        for i in range(min(10, len(atoms))):
            print(f"    [{i:3d}]  {atoms[i]}")
        print("    ...")
        for i in range(max(10, len(atoms) - 5), len(atoms)):
            print(f"    [{i:3d}]  {atoms[i]}")

    if show_matrix and n_rows > 0:
        print(f"\n  Matrix (up to {max_show_rows}×{max_show_cols}):")
        show_r = min(n_rows, max_show_rows)
        show_c = min(n_cols, max_show_cols)
        print(mat[:show_r, :show_c])
        if show_r < n_rows or show_c < n_cols:
            print(f"  ... truncated; full size {n_rows}×{n_cols}")

    if n_rows == 0 or n_cols == 0:
        print("\n  Rank: N/A (empty matrix)")
    else:
        # Use mod-p rank for memory efficiency.
        # The matrix has only small integer coefficients (-d, 1, d-2),
        # so rank mod a large prime equals the true rank with probability
        # 1 - n_cols/p > 1 - 1e-9 for our choice of p.  This avoids
        # materialising a dense QQ matrix which OOMs at scale.
        _RANK_PRIME = 2**31 - 1  # Mersenne prime, fits in Sage GF
        print(f"\n  Computing rank mod {_RANK_PRIME} (exact w.h.p., O(1) memory vs dense QQ)...")
        mat_modp = mat.change_ring(GF(_RANK_PRIME))
        rank = mat_modp.rank()

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Rows    (relations)    : {n_rows:>6}          │")
        print(f"  │  Columns (atoms + ∞)    : {n_cols:>6}          │")
        print(f"  │  Rank    (mod p)        : {rank:>6}          │")
        print(f"  │  Nullity (mod p)        : {n_cols - rank:>6}          │")
        print(f"  └─────────────────────────────────────────┘")

        if rank < n_cols:
            print(f"\n  Note: the relations span a proper subspace of the atom lattice.")
        else:
            print(f"\n  Note: the relations span the full column space.")

    print("=" * 70 + "\n")

def print_nullity_report(mat, atoms, *, fp_prime=2**31 - 1):
    """Pretty-print a full nullity decomposition report."""
    bottlenecks, report = find_bottleneck_atoms(mat, atoms, fp_prime=fp_prime)

    print("\n" + "=" * 70)
    print("NULLITY DECOMPOSITION REPORT")
    print("=" * 70)
    print(f"  Rank              : {report['rank']}")
    print(f"  Nullity (total)   : {report['nullity']}")
    print(f"  ∞ contribution    : {1 if report['inf_atom_in_null'] else 0}  (irreducible)")
    print(f"  Dest-only atoms   : {len(report['dest_only_atoms'])}  (each needs one xi-step through it)")
    print(f"  Residual nullity  : {report['residual_nullity']}  (disconnected components?)")
    print()

    if report["dest_only_atoms"]:
        print("  Dest-only atoms (restart walker from these):")
        for a in report["dest_only_atoms"]:
            print(f"    xi = {a}")

    if report["residual_nullity"] > 0:
        print(f"\n  Residual bottleneck atoms:")
        for b in bottlenecks:
            if b["reason"] == "residual":
                print(f"    xi = {b['atom']}  ({b['col_nnz']} relations)  — {b['action']}")

    print("=" * 70 + "\n")
    return bottlenecks, report

def prune_dest_only(mat, atoms, protected_atoms=None):
    """
    Iteratively remove dest-only atoms (columns with exactly 1 nonzero entry)
    and their single incident row until fixed point.

    protected_atoms: An optional set of atoms that should NEVER be pruned,
                     even if they are pendant leaves (e.g., target/generator roots).
    """
    inf_sentinel = "∞"
    cur_atoms    = list(atoms)

    n_rows = mat.nrows()
    n_cols = mat.ncols()

    row_data = [{} for _ in range(n_rows)]
    col_rows = [set() for _ in range(n_cols)]

    for (i, j) in mat.nonzero_positions(copy=False):
        row_data[i][j] = int(mat[i, j])
        col_rows[j].add(i)

    live_cols = set(range(n_cols))
    inf_cols  = {j for j, a in enumerate(cur_atoms) if str(a) == inf_sentinel}

    # Map protected atoms to their column indices
    protected_cols = set()
    if protected_atoms:
        protected_strs = {str(pa) for pa in protected_atoms}
        protected_cols = {j for j, a in enumerate(cur_atoms) if str(a) in protected_strs}

    removed   = []

    # Seed worklist: cols with exactly one live row, excluding inf and protected atoms.
    worklist = {
        j for j in live_cols
        if j not in inf_cols and j not in protected_cols and len(col_rows[j]) == 1
    }

    dead_rows: set = set()

    while worklist:
        j = worklist.pop()
        if j not in live_cols:
            continue
        if len(col_rows[j]) != 1:
            continue

        (i,) = col_rows[j]
        removed.append((cur_atoms[j], i))
        dead_rows.add(i)
        live_cols.discard(j)

        # Propagate
        for k, _val in row_data[i].items():
            if k == j:
                continue
            col_rows[k].discard(i)
            # Add to worklist if it became a pendant AND is not protected
            if k in live_cols and k not in inf_cols and k not in protected_cols and len(col_rows[k]) == 1:
                worklist.add(k)

    # --- Reconstruct Sage matrix ---
    sorted_cols  = sorted(live_cols)
    col_remap    = {old_j: new_j for new_j, old_j in enumerate(sorted_cols)}
    pruned_atoms = [cur_atoms[j] for j in sorted_cols]
    n_pruned_cols = len(sorted_cols)

    surviving = {}
    new_row_idx = 0
    for i in range(n_rows):
        if i in dead_rows:
            continue
        for old_j, val in row_data[i].items():
            if old_j in col_remap:
                surviving[(new_row_idx, col_remap[old_j])] = val
        new_row_idx += 1

    n_surviving_rows = new_row_idx
    if n_surviving_rows == 0:
        return Matrix(ZZ, 0, n_pruned_cols), pruned_atoms, removed

    pruned_mat = Matrix(ZZ, n_surviving_rows, n_pruned_cols, surviving)
    return pruned_mat, pruned_atoms, removed
