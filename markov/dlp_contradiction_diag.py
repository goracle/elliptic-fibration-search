from __future__ import annotations
import argparse, json, sys, h5py, numpy as np
from pathlib import Path
from relation_matrix import *
from sage.all import GF, ZZ, Integer, matrix, vector, Matrix

"""dlp_contradiction_diag.py

Post-mortem diagnostics for a failed DLP solve, operating directly on the
HDF5 relation-matrix dump produced by dlp_diagnostics.dump_matrix_hdf5().

Two core questions:

  1. HOMOGENEOUS CHECK
     Solve A_hom * x = 0 (the walk relations only, no gauge/anchor rows).
     Check whether the log-G vector — the known discrete-log solution — lies
     in the kernel, i.e. whether the walk data is arithmetically consistent
     before the anchor is added.

  2. CONTRADICTION EXTRACTOR
     When the full affine system A * x = b has no solution, there exists a
     left-kernel vector y (a "Farkas certificate") such that:
          y^T * A = 0   but   y^T * b != 0
     That y is a linear combination of relation rows (and the anchor) whose
     net coefficient vector is zero — meaning the rows are mutually
     contradictory regardless of x.  Printing which rows enter y (and with
     what weight) tells us exactly which walk steps are in conflict.

Usage
-----
    sage -python dlp_contradiction_diag.py relation_matrix.h5 \
         --group-order 25373 --known-key 802

    # or, if group_order / known_key are stored in the HDF5 metadata:
    sage -python dlp_contradiction_diag.py relation_matrix.h5

The script is read-only; it never modifies the HDF5 file.
"""

# ---------------------------------------------------------------------------
# Lazy Sage import — keep the module importable for linting outside Sage.
# ---------------------------------------------------------------------------
_SAGE = True

_HAS_H5PY = True

_INFINITY = "∞"
_SEP = "=" * 70
_THIN = "-" * 70

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)

def _section(title: str) -> None:
    _log(f"\n{_SEP}")
    _log(f"  {title}")
    _log(_SEP)

def _brief_atom_list(atom_rows, max_items: int = 6) -> str:
    if not atom_rows:
        return "[]"
    items = atom_rows[:max_items]
    s = ", ".join(f"{atom}={coeff}" for atom, coeff in items)
    if len(atom_rows) > max_items:
        s += ", ..."
    return f"[{s}]"

def _matrix_preview(M_ZZ, atoms, max_rows: int = 6, max_atoms: int = 6) -> None:
    _log(f"[matrix] shape={M_ZZ.nrows()}×{M_ZZ.ncols()}  atoms={len(atoms)}")
    row_limit = min(max_rows, M_ZZ.nrows())
    for i in range(row_limit):
        row_atoms = [(str(atoms[j]), int(M_ZZ[i, j])) for j in range(M_ZZ.ncols()) if int(M_ZZ[i, j]) != 0]
        _log(f"[matrix] row {i:5d}: {_brief_atom_list(row_atoms, max_items=max_atoms)}")
    if M_ZZ.nrows() > row_limit:
        _log(f"[matrix] ... {M_ZZ.nrows() - row_limit} more row(s)")


def _dedupe_rows_mod(M_ZZ, atoms, modulus: int, *, keep_zero_rows: bool = False):
    """Collapse exact duplicate rows and scalar multiples over GF(modulus).

    Each row is canonicalized by:
      1) reducing coefficients modulo modulus,
      2) removing zero entries,
      3) dividing by the first nonzero coefficient so the leading coefficient is 1,
      4) sorting the sparse support to form a hashable signature.

    Returns (M_dedup, row_sources) where row_sources maps each kept row index
    to the original row indices merged into it.
    """
    if modulus is None:
        raise ValueError("modulus is required for row deduplication")
    n_rows = M_ZZ.nrows()
    n_cols = M_ZZ.ncols()
    seen = {}
    dedup_rows = []
    row_sources = []

    for i in range(n_rows):
        entries = []
        for j in range(n_cols):
            v = int(M_ZZ[i, j]) % modulus
            if v:
                entries.append((j, v))
        if not entries:
            if keep_zero_rows:
                sig = ((), ())
                if sig not in seen:
                    seen[sig] = len(dedup_rows)
                    dedup_rows.append([0] * n_cols)
                    row_sources.append([i])
                else:
                    row_sources[seen[sig]].append(i)
            continue

        lead = entries[0][1]
        try:
            inv_lead = pow(lead, -1, modulus)
        except ValueError:
            # Fall back to Sage-style inversion if needed.
            inv_lead = int((Integer(lead) ** (-1)) % modulus)
        sig = tuple((j, (v * inv_lead) % modulus) for j, v in entries)
        if sig not in seen:
            seen[sig] = len(dedup_rows)
            row = [0] * n_cols
            for j, v in sig:
                row[j] = int(v)
            dedup_rows.append(row)
            row_sources.append([i])
        else:
            row_sources[seen[sig]].append(i)

    if dedup_rows:
        M_dedup = Matrix(ZZ, dedup_rows)
    else:
        M_dedup = Matrix(ZZ, 0, n_cols)
    return M_dedup, row_sources

# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

def load_matrix_hdf5(path: str):
    """Load the pruned relation matrix and metadata from an HDF5 dump.

    Returns a dict with keys:
        M_ZZ        : Sage integer matrix (rows × cols)
        atoms       : list of str, one per column
        aidx        : dict str -> col index
        group_order : int or None
        divisor_xs  : list of 4 ints or None
        col_inf     : int or None  (column index of ∞, -1 means absent)
        col_gen0    : int or None
        col_gen1    : int or None
        col_tgt0    : int or None
        col_tgt1    : int or None
    """
    if not _HAS_H5PY:
        raise RuntimeError("h5py is not installed — run: pip install h5py")
    if not _SAGE:
        raise RuntimeError("SageMath is required — run with: sage -python ...")

    with h5py.File(path, "r") as f:
        # --- atoms ---
        atoms = [a.decode("utf-8") for a in f["atoms"][:]]
        aidx_raw = f["atom_index"][()].decode("utf-8")
        aidx = json.loads(aidx_raw)

        # --- matrix (prefer dense if present, else reconstruct from CSR) ---
        if "matrix_dense" in f:
            M_np = f["matrix_dense"][:].astype(int)
            M_ZZ = matrix(ZZ, M_np.tolist())
        else:
            data   = f["csr/data"][:]
            indices = f["csr/indices"][:]
            indptr  = f["csr/indptr"][:]
            shape   = tuple(int(x) for x in f["csr/shape"][:])
            nrows, ncols = shape
            rows_list = [[] for _ in range(nrows)]
            for r in range(nrows):
                for idx in range(indptr[r], indptr[r + 1]):
                    c = int(indices[idx])
                    v = int(data[idx])
                    rows_list[r].append((c, v))
            M_ZZ = matrix(ZZ, nrows, ncols, sparse=True)
            for r, entries in enumerate(rows_list):
                for c, v in entries:
                    M_ZZ[r, c] = v

        # --- metadata ---
        group_order = int(f["group_order"][()]) if "group_order" in f else None
        divisor_xs  = [int(x) for x in f["divisor_xs"][:]] if "divisor_xs" in f else None

        def _col(key):
            if key not in f:
                return None
            v = int(f[key][()])
            return v if v >= 0 else None

        col_inf  = _col("col_inf")
        col_gen0 = _col("col_gen0")
        col_gen1 = _col("col_gen1")
        col_tgt0 = _col("col_tgt0")
        col_tgt1 = _col("col_tgt1")

    return {
        "M_ZZ":        M_ZZ,
        "atoms":       atoms,
        "aidx":        aidx,
        "group_order": group_order,
        "divisor_xs":  divisor_xs,
        "col_inf":     col_inf,
        "col_gen0":    col_gen0,
        "col_gen1":    col_gen1,
        "col_tgt0":    col_tgt0,
        "col_tgt1":    col_tgt1,
    }

# ---------------------------------------------------------------------------
# Check 1: Homogeneous system + log-G membership
# ---------------------------------------------------------------------------

def drop_rows(M, rows_to_drop):
    keep = [i for i in range(M.nrows()) if i not in set(rows_to_drop)]
    return M.matrix_from_rows(keep)

def replace_row_append(M, row_idx, new_row):
    M2 = drop_rows(M, [row_idx])
    return M2.stack(matrix(M.base_ring(), [new_row]))

# ---------------------------------------------------------------------------
# Check 2: Left-kernel certificate (Farkas row)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_structural_collapse(M_ZZ, atoms, group_order,
                               col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1):
    """
    Compact structural diagnostics.
    """
    _section("CHECK 3: STRUCTURAL COLLAPSE TRIAGE")

    n  = group_order
    Fp = GF(Integer(n))

    protected = [a for a in [atoms[col_gen0] if col_gen0 is not None else None,
                              atoms[col_gen1] if col_gen1 is not None else None,
                              atoms[col_tgt0] if col_tgt0 is not None else None,
                              atoms[col_tgt1] if col_tgt1 is not None else None]
                 if a is not None]
    M_pruned, pruned_atoms, _ = prune_dest_only(M_ZZ, atoms, protected=protected)
    M_pruned, row_sources = _dedupe_rows_mod(M_pruned, pruned_atoms, group_order)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}
    _log(f"  row dedup    : {sum(len(v) for v in row_sources) - len(row_sources)} duplicates removed")
    def remap(col):
        if col is None:
            return None
        return pruned_aidx.get(str(atoms[col]))

    p_col_inf  = remap(col_inf)
    p_col_gen0 = remap(col_gen0)
    p_col_gen1 = remap(col_gen1)
    p_col_tgt0 = remap(col_tgt0)
    p_col_tgt1 = remap(col_tgt1)

    n_rows = M_pruned.nrows()
    n_cols = M_pruned.ncols()
    A = M_pruned.change_ring(Fp)
    full_null = A.right_kernel().dimension()

    _log(f"  pruned matrix: {n_rows} rows × {n_cols} cols")
    _log(f"  nullity       : {full_null}")

    _log("\n  --- A) Special-column order test ---")
    row_space = A.row_space()
    special = [("inf",  p_col_inf),
               ("gen0", p_col_gen0), ("gen1", p_col_gen1),
               ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]
    for name, col in special:
        if col is None:
            _log(f"  {name:6s}: absent after prune")
            continue
        e_j = vector(Fp, n_cols)
        e_j[col] = Fp(1)
        found_k = None
        for k in range(1, min(25, n)):
            if Fp(k) * e_j in row_space:
                found_k = k
                break
        _log(f"  {name:6s} (col {col:5d}): " + (f"k={found_k}" if found_k is not None else "k>24"))

    _log("\n  --- B) Rank-without-inf test ---")
    if p_col_inf is not None:
        cols_no_inf = [j for j in range(n_cols) if j != p_col_inf]
        null_no_inf = A.matrix_from_columns(cols_no_inf).right_kernel().dimension()
        _log(f"  without-inf nullity: {null_no_inf}")
        if null_no_inf > full_null:
            _log("  inf column is absorbing degrees of freedom.")
        elif null_no_inf == full_null:
            _log("  inf column is not the source of collapse.")
        else:
            _log("  inf column contributed free directions.")
    else:
        _log("  no inf column present; skipped.")

    _log("\n  --- C) Direct fusion audit ---")
    special_by_col = {c: name for name, c in [("inf", p_col_inf), ("gen0", p_col_gen0), ("gen1", p_col_gen1), ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)] if c is not None}
    col_groups: dict = {}
    zero_cols = []
    for j in range(n_cols):
        sig = []
        lead_inv = None
        for i in range(n_rows):
            v = A[i, j]
            if v != Fp(0):
                if lead_inv is None:
                    lead_inv = v ** (-1)
                sig.append((i, int(v * lead_inv)))
        if lead_inv is None:
            zero_cols.append(j)
            continue
        col_groups.setdefault(tuple(sig), []).append(j)

    fusion_groups = [cols for cols in col_groups.values() if len(cols) > 1]
    if zero_cols:
        preview = ", ".join(f"{pruned_atoms[j]}(col={j})" for j in zero_cols[:10])
        _log(f"  zero columns: {len(zero_cols)}")
        _log(f"    {preview}" + (" ..." if len(zero_cols) > 10 else ""))

    if fusion_groups:
        _log(f"  fusion classes: {len(fusion_groups)}")
        for cols in fusion_groups[:8]:
            labels = []
            special_hits = []
            for j in cols:
                atom = str(pruned_atoms[j])
                if j in special_by_col:
                    special_hits.append(special_by_col[j])
                    labels.append(f"{atom}({special_by_col[j]})")
                else:
                    labels.append(atom)
            flag = f"  special={sorted(set(special_hits))}" if special_hits else ""
            _log(f"    {labels}{flag}")
        if len(fusion_groups) > 8:
            _log(f"    ... {len(fusion_groups) - 8} more fusion class(es)")
    else:
        _log("  no proportional column classes found.")

    _log("\n  --- C2) Kernel-basis fusion sanity check ---")
    ker_full = A.right_kernel()
    kernel_fusions = []
    for vec in ker_full.basis():
        support = [(j, int(vec[j])) for j in range(n_cols) if vec[j] != Fp(0)]
        if len(support) == 2:
            (j0, c0), (j1, c1) = support
            if (c0 == 1 and c1 == n - 1) or (c0 == n - 1 and c1 == 1):
                kernel_fusions.append((pruned_atoms[j0], pruned_atoms[j1]))
    if kernel_fusions:
        _log(f"  support-2 kernel fusions: {len(kernel_fusions)}")
        for a0, a1 in kernel_fusions[:8]:
            _log(f"    a[{a0}] = a[{a1}]")
        if len(kernel_fusions) > 8:
            _log(f"    ... {len(kernel_fusions) - 8} more")
    else:
        _log("  no support-2 kernel fusions.")

    _log("\n  --- D) Rows hitting special columns ---")
    for col_name, p_col in [("gen0", p_col_gen0), ("gen1", p_col_gen1), ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]:
        if p_col is None:
            continue
        hitting_rows = [(i, int(A[i, p_col])) for i in range(n_rows) if A[i, p_col] != Fp(0)]
        _log(f"  {col_name}: {len(hitting_rows)} row(s)")
        for row_i, coeff in hitting_rows[:5]:
            row_atoms = [(pruned_atoms[j], int(M_pruned[row_i, j])) for j in range(n_cols) if M_pruned[row_i, j] != 0]
            _log(f"    row {row_i:5d} coeff={coeff:6d} atoms={_brief_atom_list(row_atoms, max_items=5)}")
        if len(hitting_rows) > 5:
            _log(f"    ... {len(hitting_rows) - 5} more rows")

    _log("\n  --- E) Row-subsampling stability ---")
    import random as _random
    _random.seed(42)
    n_drop = max(1, n_rows // 10)
    _log(f"  base nullity={full_null}; dropping {n_drop}/{n_rows} rows per trial")
    for trial in range(5):
        drop = set(_random.sample(range(n_rows), n_drop))
        keep = [i for i in range(n_rows) if i not in drop]
        null_sub = A.matrix_from_rows(keep).right_kernel().dimension()
        delta = null_sub - full_null
        _log(f"  trial {trial+1}: nullity={null_sub} delta={delta:+d}")

def incremental_consistency_filter(
    M_ZZ, atoms, group_order,
    col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1,
):
    """
    Check 4: Incremental consistency filter.

    This version stays conservative and keeps the output brief.
    """
    _section("CHECK 4: INCREMENTAL CONSISTENCY FILTER")

    n   = group_order
    Fp  = GF(Integer(n))
    p   = int(n)

    protected = [a for a in [
        atoms[col_gen0] if col_gen0 is not None else None,
        atoms[col_gen1] if col_gen1 is not None else None,
        atoms[col_tgt0] if col_tgt0 is not None else None,
        atoms[col_tgt1] if col_tgt1 is not None else None,
    ] if a is not None]
    M_pruned, pruned_atoms, _ = prune_dest_only(M_ZZ, atoms, protected=protected)
    M_pruned, row_sources = _dedupe_rows_mod(M_pruned, pruned_atoms, group_order)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}
    _log(f"  row dedup    : {sum(len(v) for v in row_sources) - len(row_sources)} duplicates removed")

    def remap(col):
        if col is None:
            return None
        return pruned_aidx.get(str(atoms[col]))

    p_col_inf  = remap(col_inf)
    p_col_gen0 = remap(col_gen0)
    p_col_gen1 = remap(col_gen1)
    p_col_tgt0 = remap(col_tgt0)
    p_col_tgt1 = remap(col_tgt1)

    n_rows = M_pruned.nrows()
    n_cols = M_pruned.ncols()

    _log(f"  Matrix: {n_rows} rows × {n_cols} cols over GF({p})")

    pivots = {}

    def _row_from_matrix(i):
        return [int(M_pruned[i, j]) % p for j in range(n_cols)]

    def _reduce(row, rhs):
        for pc, (prow, prhs) in pivots.items():
            coeff = row[pc] % p
            if coeff == 0:
                continue
            row = [(row[j] - coeff * prow[j]) % p for j in range(n_cols)]
            rhs = (rhs - coeff * prhs) % p
        return row, rhs

    def _add_pivot(row, rhs):
        for j in range(n_cols):
            if row[j] % p != 0:
                inv = pow(row[j], p - 2, p)
                pivots[j] = ([(row[k] * inv) % p for k in range(n_cols)], (rhs * inv) % p)
                return

    if p_col_inf is not None:
        gauge = [0] * n_cols
        gauge[p_col_inf] = 1
        _add_pivot(gauge, 0)
        _log(f"  Seeded gauge a[∞]=0 (col {p_col_inf})")

    anchor_row, anchor_rhs, anchor_label = _build_balanced_anchor_row(
        Fp, n_cols, p_col_gen0, p_col_gen1, p_col_inf
    )
    if anchor_row is not None:
        _add_pivot([int(x) % p for x in anchor_row], int(anchor_rhs))
        _log(f"  Seeded {anchor_label}")
    else:
        _log("  Balanced anchor unavailable.")

    good_rows  = []
    bad_rows   = []
    dep_rows   = []
    first_bad  = None

    for i in range(n_rows):
        row_r, rhs_r = _reduce(_row_from_matrix(i), 0)
        if all(v == 0 for v in row_r):
            if rhs_r % p == 0:
                dep_rows.append(i)
            else:
                bad_rows.append(i)
                if first_bad is None:
                    first_bad = i
        else:
            _add_pivot(row_r, rhs_r)
            good_rows.append(i)

    _log(f"  good={len(good_rows)}  bad={len(bad_rows)}  dependent={len(dep_rows)}  first_bad={first_bad}")

    if not bad_rows:
        _log("  ✓  No contradictions found in step order.")
        return

    special_names = {}
    for nm, pc in [("inf", p_col_inf), ("gen0", p_col_gen0), ("gen1", p_col_gen1),
                   ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]:
        if pc is not None:
            special_names[pc] = nm

    def _atom_freq(row_list, top=10):
        freq = {}
        for i in row_list:
            for j in range(n_cols):
                if int(M_pruned[i, j]) % p != 0:
                    key = str(pruned_atoms[j])
                    freq[key] = freq.get(key, 0) + 1
        return sorted(freq.items(), key=lambda kv: -kv[1])[:top], freq

    top_bad, bad_freq = _atom_freq(bad_rows, top=10)
    top_good, good_freq = _atom_freq(good_rows, top=10)

    _log("  top bad-row atoms:")
    for atom, cnt in top_bad[:8]:
        col = pruned_aidx.get(atom)
        sp = f" [{special_names[col]}]" if col in special_names else ""
        _log(f"    {atom:>8}  {cnt:5d} rows{sp}")

    _log("  first bad rows:")
    for i in bad_rows[:8]:
        row_atoms = [(str(pruned_atoms[j]), int(M_pruned[i, j])) for j in range(n_cols) if int(M_pruned[i, j]) % p != 0]
        _log(f"    row {i:5d}: {_brief_atom_list(row_atoms, max_items=6)}")
    if len(bad_rows) > 8:
        _log(f"    ... {len(bad_rows) - 8} more bad rows")

    _log("  top enrichment (bad/good):")
    n_bad  = max(len(bad_rows),  1)
    n_good = max(len(good_rows), 1)
    enrichment = []
    all_atoms = set(bad_freq) | set(good_freq)
    for atom in all_atoms:
        br = bad_freq.get(atom,  0) / n_bad
        gr = good_freq.get(atom, 0) / n_good
        if gr > 0:
            enrichment.append((atom, br / gr, br, gr))
        elif br > 0:
            enrichment.append((atom, float("inf"), br, gr))
    enrichment.sort(key=lambda x: -x[1])
    for atom, ratio, br, gr in enrichment[:8]:
        col = pruned_aidx.get(atom)
        sp  = f" [{special_names[col]}]" if col in special_names else ""
        ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "inf"
        _log(f"    {atom:>8}  enrich={ratio_str}  bad={br:.4f} good={gr:.4f}{sp}")



def farkas_delete_rerun(
    M_ZZ, atoms, aidx, group_order,
    farkas_walk_rows,
    col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1,
    known_key=None,
):
    """
    Delete the walk rows that appeared in the Farkas certificate and re-run
    all three checks on the reduced matrix.

    This experiment answers: is the contradiction localized to that certificate
    subset, or is it globally encoded throughout the walk data?

    Expected outcomes:
      - System becomes consistent, nullity >= 2, gen/tgt no longer k=1
        → contradiction was concentrated; those relations are the bad core.
      - System stays inconsistent or gen/tgt remain k=1
        → contradiction is globally encoded; removing one certificate
          witness only exposes the next one.
    """
    _log(f"\n{'#'*70}")
    _log("# FARKAS-DELETE RE-RUN")
    _log(f"#  Deleting {len(farkas_walk_rows)} certificate walk row(s) from M_ZZ")
    _log(f"{'#'*70}")

    n_orig = M_ZZ.nrows()
    farkas_set = set(farkas_walk_rows)
    keep = [i for i in range(n_orig) if i not in farkas_set]
    M_reduced = M_ZZ.matrix_from_rows(keep)
    _log(f"  Original rows: {n_orig}  →  Reduced rows: {M_reduced.nrows()}"
         f"  (deleted {len(farkas_walk_rows)} rows)")

    # --- Check 1 on reduced matrix ---
    if known_key is not None:
        check_homogeneous(
            M_reduced, atoms, aidx, group_order,
            known_key, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf,
        )
    else:
        _section("CHECK 1 (reduced): HOMOGENEOUS SYSTEM  (no --known-key)")
        Fp = GF(Integer(group_order))
        A = M_reduced.change_ring(Fp)
        null = A.right_kernel().dimension()
        _log(f"  rows={M_reduced.nrows()}  cols={M_reduced.ncols()}"
             f"  rank={A.rank()}  nullity={null}")

    # --- Check 2 on reduced matrix ---
    cert_entries2, farkas_walk_rows2 = extract_contradiction_certificate(
        M_reduced, atoms, group_order,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
    )
    if not cert_entries2:
        _log("\n  [farkas-delete] Reduced system is CONSISTENT after deletion.")
        _log("  Contradiction was localized to the certificate rows.")
    else:
        _log(f"\n  [farkas-delete] Reduced system still INCONSISTENT."
             f"  New certificate uses {len(farkas_walk_rows2)} walk row(s).")
        overlap = farkas_set & set(farkas_walk_rows2)
        _log(f"  Overlap with deleted rows: {len(overlap)}  "
             f"(should be 0 — deleted rows are gone)")

    # --- Check 3 on reduced matrix ---
    check_structural_collapse(
        M_reduced, atoms, group_order,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
        col_tgt0=col_tgt0,
        col_tgt1=col_tgt1,
    )

    _log(f"\n{'#'*70}")
    _log("# FARKAS-DELETE RE-RUN COMPLETE")
    _log(f"{'#'*70}")

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyse a failed DLP solve from an HDF5 relation-matrix dump."
    )
    ap.add_argument("hdf5_path", help="Path to the HDF5 file from dump_matrix_hdf5()")
    ap.add_argument("--group-order", type=int, default=None,
                    help="Group order l (overrides HDF5 metadata if given)")
    ap.add_argument("--known-key",   type=int, default=None,
                    help="Known DLP answer for the log-G membership test")
    ap.add_argument("--col-gen0",    type=int, default=None,
                    help="Column index of gen0 atom (overrides HDF5 metadata)")
    ap.add_argument("--col-gen1",    type=int, default=None,
                    help="Column index of gen1 atom (overrides HDF5 metadata)")
    ap.add_argument("--col-tgt0",    type=int, default=None,
                    help="Column index of tgt0 atom (overrides HDF5 metadata)")
    ap.add_argument("--col-tgt1",    type=int, default=None,
                    help="Column index of tgt1 atom (overrides HDF5 metadata)")
    ap.add_argument("--farkas-delete", action="store_true",
                    help="After extracting the certificate, delete those walk rows "
                         "and re-run all checks on the reduced matrix.")
    args = ap.parse_args(argv)

    if not Path(args.hdf5_path).exists():
        sys.exit(f"ERROR: file not found: {args.hdf5_path}")

    _log(f"\n{'#'*70}")
    _log(f"# DLP CONTRADICTION DIAGNOSTICS")
    _log(f"#  file: {args.hdf5_path}")
    _log(f"{'#'*70}")

    _log("\n[load] reading HDF5 matrix ...")
    data = load_matrix_hdf5(args.hdf5_path)

    M_ZZ        = data["M_ZZ"]
    atoms       = data["atoms"]
    aidx        = data["aidx"]
    group_order = args.group_order or data["group_order"]
    divisor_xs  = data["divisor_xs"]

    col_inf  = data["col_inf"]
    col_gen0 = args.col_gen0 if args.col_gen0 is not None else data["col_gen0"]
    col_gen1 = args.col_gen1 if args.col_gen1 is not None else data["col_gen1"]
    col_tgt0 = args.col_tgt0 if args.col_tgt0 is not None else data["col_tgt0"]
    col_tgt1 = args.col_tgt1 if args.col_tgt1 is not None else data["col_tgt1"]

    if group_order is None:
        sys.exit("ERROR: group_order not found in HDF5 and not supplied via --group-order")

    _log(f"[load] matrix shape : {M_ZZ.nrows()} × {M_ZZ.ncols()}")
    _matrix_preview(M_ZZ, atoms, max_rows=4, max_atoms=6)
    _log(f"[load] group_order  : {group_order}")
    _log(f"[load] divisor_xs   : {divisor_xs}")
    _log(f"[load] col_inf={col_inf}  col_gen0={col_gen0}  col_gen1={col_gen1}"
         f"  col_tgt0={col_tgt0}  col_tgt1={col_tgt1}")

    known_key = args.known_key
    if known_key is None:
        _log("[load] --known-key not supplied; log-G membership test will be skipped.")

    # --- Check 1 ---
    if known_key is not None:
        check_homogeneous(
            M_ZZ, atoms, aidx, group_order,
            known_key, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf,
        )
    else:
        _section("CHECK 1: HOMOGENEOUS SYSTEM  (skipped — no --known-key)")
        Fp = GF(Integer(group_order))
        A_hom = M_ZZ.change_ring(Fp)
        rank_hom = A_hom.rank()
        null_hom = A_hom.ncols() - rank_hom
        _log(f"  rows={M_ZZ.nrows()}  cols={M_ZZ.ncols()}  rank={rank_hom}  nullity={null_hom}")

    # --- Check 2 ---
    cert_entries, farkas_walk_rows = extract_contradiction_certificate(
        M_ZZ, atoms, group_order,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
    )

    # --- Check 3 ---
    check_structural_collapse(
        M_ZZ, atoms, group_order,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
        col_tgt0=col_tgt0,
        col_tgt1=col_tgt1,
    )

    # --- Check 4: incremental consistency filter (always runs) ---
    incremental_consistency_filter(
        M_ZZ, atoms, group_order,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
        col_tgt0=col_tgt0,
        col_tgt1=col_tgt1,
    )

    if args.farkas_delete and farkas_walk_rows:
        farkas_delete_rerun(
            M_ZZ, atoms, aidx, group_order,
            farkas_walk_rows,
            col_inf=col_inf,
            col_gen0=col_gen0,
            col_gen1=col_gen1,
            col_tgt0=col_tgt0,
            col_tgt1=col_tgt1,
            known_key=known_key,
        )
    elif args.farkas_delete:
        _log("\n[farkas-delete] No walk rows in certificate — nothing to delete.")

    _log(f"\n{'#'*70}")
    _log("# DIAGNOSTICS COMPLETE")
    _log(f"{'#'*70}\n")

def _extract_pin_rows(ker, pruned_atoms, n_cols, Fp, p_col_inf=None, *, pin_isolated: bool = False, max_preview: int = 12):
    """
    Inspect a right-kernel basis and report structurally important directions.

    By default this function is read-only: it reports isolated atoms, gauge
    directions, and low-support fusion relations, but it does *not* pin any
    free variables into the system.  That is the conservative behavior for
    diagnostics.

    If pin_isolated=True, isolated atoms are converted into explicit pin rows
    a[j] = 0.  That mode is available for experiments, but it is deliberately
    opt-in because it can manufacture contradictions by over-fixing the nullspace.
    """
    n = int(Fp.characteristic())
    pin_rows   = []
    pin_rhs    = []
    pin_labels = []

    counts = {"gauge": 0, "isolated": 0, "fusion": 0, "parity": 0, "other": 0}
    previews = []

    for vi, vec in enumerate(ker.basis()):
        support = [(j, int(vec[j])) for j in range(n_cols) if vec[j] != Fp(0)]
        if not support:
            continue

        msg = None
        if len(support) == 1 and p_col_inf is not None and support[0][0] == p_col_inf:
            counts["gauge"] += 1
            msg = f"kernel[{vi}]: GAUGE (inf) -- keeping free"
        elif len(support) == 1:
            counts["isolated"] += 1
            j, coeff = support[0]
            atom = pruned_atoms[j]
            msg = f"kernel[{vi}]: ISOLATED atom={atom} coeff={coeff}"
            if pin_isolated:
                pin_row = vector(Fp, n_cols)
                pin_row[j] = Fp(1)
                pin_rows.append(pin_row)
                pin_rhs.append(Fp(0))
                pin_labels.append(f"pin a[{atom}]=0")
                msg += " -- pinning"
            else:
                msg += " -- leaving free"
        elif len(support) == 2:
            (j0, c0), (j1, c1) = support
            if (c0 == 1 and c1 == n - 1) or (c0 == n - 1 and c1 == 1):
                counts["fusion"] += 1
                a0, a1 = pruned_atoms[j0], pruned_atoms[j1]
                msg = f"kernel[{vi}]: FUSION a[{a0}] = a[{a1}]"
            else:
                counts["other"] += 1
                msg = f"kernel[{vi}]: OTHER support_size=2 distinct_coeffs={sorted(set(c for _, c in support))}"
        else:
            coeffs_vals = [c for _, c in support]
            is_flat = len(set(coeffs_vals)) == 1
            if is_flat:
                counts["parity"] += 1
                msg = f"kernel[{vi}]: PARITY/CONSERVATION support_size={len(support)} all_coeffs={coeffs_vals[0]}"
            else:
                counts["other"] += 1
                msg = f"kernel[{vi}]: OTHER support_size={len(support)} distinct_coeffs={sorted(set(coeffs_vals))}"

        if len(previews) < max_preview:
            previews.append(msg)

    _log(f"  kernel summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for msg in previews:
        _log(f"  {msg}")
    omitted = sum(counts.values()) - len(previews)
    if omitted > 0:
        _log(f"  ... {omitted} more kernel direction(s) omitted")

    return pin_rows, pin_rhs, pin_labels


def check_homogeneous(M_ZZ, atoms: list, aidx: dict, group_order: int,
                      known_key: int, col_gen0, col_gen1, col_tgt0, col_tgt1,
                      col_inf):
    """
    Prune M_ZZ via prune_dest_only, then solve A_hom * x = 0 over GF(group_order).

    Tests whether the known log-G vector lies in the kernel of the pruned
    homogeneous system, i.e. whether the walk data is consistent with the
    known answer before any normalization rows are added.

    This check is deliberately conservative: isolated atoms are reported, not
    forcibly pinned.  That avoids manufacturing contradictions by collapsing
    nullspace directions that the walk data never touched.
    """
    _section("CHECK 1: HOMOGENEOUS SYSTEM  (walk relations, no anchor)")

    n  = group_order
    Fp = GF(Integer(n))

    # --- prune_dest_only (same as merge_experiment does before the solve) ---
    protected = [a for a in [atoms[col_gen0] if col_gen0 is not None else None,
                              atoms[col_gen1] if col_gen1 is not None else None,
                              atoms[col_tgt0] if col_tgt0 is not None else None,
                              atoms[col_tgt1] if col_tgt1 is not None else None]
                 if a is not None]
    M_pruned, pruned_atoms, removed = prune_dest_only(M_ZZ, atoms, protected=protected)
    M_before_dedup = M_pruned
    M_pruned, row_sources = _dedupe_rows_mod(M_before_dedup, pruned_atoms, group_order)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    n_removed = len(removed)
    n_dedup = M_before_dedup.nrows() - M_pruned.nrows()
    _log(f"  prune_dest_only removed {n_removed} dest-only atoms "
         f"({M_ZZ.nrows()} → {M_before_dedup.nrows()} rows, "
         f"{M_ZZ.ncols()} → {M_before_dedup.ncols()} cols)")
    _log(f"  row dedup      removed {n_dedup} duplicate/scalar-multiple row(s)")

    def remap(col):
        if col is None:
            return None
        return pruned_aidx.get(str(atoms[col]))

    p_col_gen0 = remap(col_gen0)
    p_col_gen1 = remap(col_gen1)
    p_col_tgt0 = remap(col_tgt0)
    p_col_tgt1 = remap(col_tgt1)
    p_col_inf  = remap(col_inf)

    for name, col in [("gen0", p_col_gen0), ("gen1", p_col_gen1),
                      ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]:
        if col is None:
            _log(f"  ⚠  {name} pruned away — it was dest-only (never appeared as xi).")

    n_rows = M_pruned.nrows()
    n_cols = M_pruned.ncols()
    A_hom = M_pruned.change_ring(Fp)

    # --- kernel diagnostics only: do not pin isolated atoms ---
    ker_pre  = A_hom.right_kernel()
    null_pre = ker_pre.dimension()
    _log(f"\n  Pre-normalization nullity: {null_pre} on the {n_rows}×{n_cols} system")

    _extract_pin_rows(
        ker_pre, pruned_atoms, n_cols, Fp, p_col_inf, pin_isolated=False, max_preview=8
    )
    _log("  Isolated atoms are reported as free directions, not pinned.")

    rank_hom = A_hom.rank()
    ker_hom  = A_hom.right_kernel()
    null_hom = ker_hom.dimension()

    _log(f"  rows={M_pruned.nrows()}  cols={n_cols}  rank={rank_hom}  nullity={null_hom}")
    _log(f"  (ideal: nullity >= 2 — gauge direction + DLP direction)")

    if null_hom == 0:
        _log("\n  ✗  Nullity=0 — walk relations alone are already inconsistent over GF(l).")
        _log("     The contradiction is in the relation rows, not the anchor.")
        _log("     → Proceed to Check 2 for root cause.")
    elif null_hom == 1:
        _log("\n  ⚠  Nullity=1 — gauge and DLP directions are fused or one is missing.")
        _log("     A single normalization row may still be too aggressive.")
    else:
        _log(f"\n  ✓  Nullity={null_hom} — at least gauge + DLP directions present.")

    # --- inspect the surviving kernel basis vectors ---
    _log(f"\n  Surviving kernel basis ({ker_hom.dimension()} vector(s)):")
    special_cols = {name: col for name, col in [
        ("gen0", p_col_gen0), ("gen1", p_col_gen1),
        ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1),
        ("inf",  p_col_inf),
    ] if col is not None}
    for bi, bv in enumerate(ker_hom.basis()[:8]):
        support = [(j, int(bv[j])) for j in range(n_cols) if bv[j] != Fp(0)]
        coeffs  = [c for _, c in support]
        is_flat = len(set(coeffs)) == 1
        special_vals = {name: int(bv[col]) for name, col in special_cols.items()}
        all_special_same = len(set(special_vals.values())) == 1
        _log(f"  basis[{bi}]: support_size={len(support)} flat={'yes' if is_flat else 'no'} specials={special_vals}")
        if is_flat:
            _log("    flat vector: all atoms share the same log")
        elif all_special_same:
            _log("    special atoms all equal")
        else:
            _log("    special atoms differ; DLP direction still present")
    if ker_hom.dimension() > 8:
        _log(f"  ... {ker_hom.dimension() - 8} more basis vector(s)")

    # --- log-G membership test ---
    missing = [name for name, col in [("gen0", p_col_gen0), ("gen1", p_col_gen1),
                                       ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]
               if col is None]
    if missing:
        _log(f"\n  ⚠  Cannot build log-G vector — columns missing after prune: {missing}")
        _log("     Skipping log-G membership test.")
        return null_hom

    _log(f"\n  Building log-G candidate vector (known_key={known_key}) ...")

    v_logG = vector(Fp, n_cols)
    v_logG[p_col_gen0] = Fp(1)
    v_logG[p_col_gen1] = Fp(0)
    if p_col_inf is not None:
        v_logG[p_col_inf] = Fp(0)

    inv2 = None
    try:
        inv2 = Fp(2) ** (-1)
    except Exception:
        raise

    if inv2 is not None:
        half_key = Fp(known_key) * inv2
        v_logG[p_col_tgt0] = half_key
        v_logG[p_col_tgt1] = half_key
        _log(f"  a[gen0]=1, a[gen1]=0, a[tgt0]=a[tgt1]={int(half_key)} (={known_key}/2 mod {n})")
    else:
        v_logG[p_col_tgt0] = Fp(known_key)
        v_logG[p_col_tgt1] = Fp(0)
        _log(f"  a[gen0]=1, a[gen1]=0, a[tgt0]={known_key}, a[tgt1]=0")
        _log(f"  (2 not invertible mod {n}; used asymmetric split)")

    residual     = A_hom * v_logG
    nonzero_rows = [(i, int(residual[i])) for i in range(n_rows) if residual[i] != Fp(0)]

    assigned_cols = {c for c in [p_col_gen0, p_col_gen1, p_col_tgt0, p_col_tgt1, p_col_inf]
                     if c is not None}
    true_failures = []
    fb_residuals  = []
    for row_i, resid in nonzero_rows:
        row_support = {j for j in range(n_cols) if M_pruned[row_i, j] != 0}
        if row_support <= assigned_cols:
            true_failures.append((row_i, resid))
        else:
            fb_residuals.append((row_i, resid))

    if not nonzero_rows:
        _log("\n  ✓  log-G vector IS in the kernel of A_hom (pruned).")
        _log("     Walk relations are consistent with the known solution.")
        _log("     The failure is introduced by normalization rows, not the walk data.")
    else:
        if fb_residuals:
            _log(f"\n  ℹ  {len(fb_residuals)} row(s) have nonzero residual because they contain")
            _log("     unassigned factor-base atoms (v_logG leaves them at 0).")
            _log("     This is expected -- the test vector is not a full oracle.")
        if true_failures:
            _log(f"\n  ✗  {len(true_failures)} row(s) fail on assigned atoms only -- genuine contradiction:")
            for row_i, resid in true_failures[:30]:
                row_atoms = [(pruned_atoms[j], int(M_pruned[row_i, j]))
                             for j in range(n_cols) if M_pruned[row_i, j] != 0]
                _log(f"    row {row_i:5d}  residual={resid:5d}  atoms={row_atoms}")
            if len(true_failures) > 30:
                _log(f"    ... and {len(true_failures) - 30} more rows")
            _log("\n     -> The walk data itself contradicts the known key.")
            _log("       Likely cause: wrong xi multiplicity, wrong inf sign, or a bad")
            _log("       involution-closure row in the relation matrix.")
        elif fb_residuals:
            _log("\n  ✓  No true failures -- all residuals are from unassigned fb atoms.")
            _log("     Walk structure is consistent with the known key.")

    return null_hom



def _remap(old_col, old_atoms, pruned_aidx):
    if old_col is None:
        return None
    return pruned_aidx.get(str(old_atoms[old_col]))

def extract_contradiction_certificate(
    M_ZZ, atoms: list, group_order: int,
    col_inf, col_gen0, col_gen1,
    n_anchor_rows: int = 2,
):
    _section("CHECK 2: CONTRADICTION CERTIFICATE  (left-kernel Farkas row)")

    n  = group_order
    Fp = GF(Integer(n))

    protected = [a for a in [atoms[col_gen0] if col_gen0 is not None else None,
                              atoms[col_gen1] if col_gen1 is not None else None]
                 if a is not None]
    M_pruned, pruned_atoms, removed = prune_dest_only(M_ZZ, atoms, protected=protected)
    M_before_dedup = M_pruned
    M_pruned, row_sources = _dedupe_rows_mod(M_before_dedup, pruned_atoms, group_order)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    _log(f"  prune_dest_only: {len(removed)} atoms removed  "
         f"({M_ZZ.nrows()}→{M_before_dedup.nrows()} rows, "
         f"{M_ZZ.ncols()}→{M_before_dedup.ncols()} cols)")
    _log(f"  row dedup      : {M_before_dedup.nrows() - M_pruned.nrows()} duplicate/scalar-multiple row(s) removed")
    _matrix_preview(M_pruned, pruned_atoms, max_rows=4, max_atoms=6)

    p_col_inf  = _remap(col_inf,  atoms, pruned_aidx)
    p_col_gen0 = _remap(col_gen0, atoms, pruned_aidx)
    p_col_gen1 = _remap(col_gen1, atoms, pruned_aidx)

    n_walk = M_pruned.nrows()
    n_cols = M_pruned.ncols()

    A_hom_fp = M_pruned.change_ring(Fp)

    ker_pre  = A_hom_fp.right_kernel()
    null_pre = ker_pre.dimension()
    _log(f"  pre-normalization nullity: {null_pre} on the {n_walk}×{n_cols} homogeneous system")

    _extract_pin_rows(ker_pre, pruned_atoms, n_cols, Fp, p_col_inf, pin_isolated=False, max_preview=10)

    A_pinned = A_hom_fp
    b_pinned = vector(Fp, [Fp(0)] * n_walk)

    extra_rows_fp = []
    extra_rhs     = []
    extra_labels  = []

    if p_col_inf is not None:
        gauge_row = vector(Fp, n_cols)
        gauge_row[p_col_inf] = Fp(1)
        extra_rows_fp.append(gauge_row)
        extra_rhs.append(Fp(0))
        extra_labels.append(f"gauge a[∞]=0  (col={p_col_inf})")
    else:
        _log("  no ∞ column after prune; gauge row omitted")

    # Keep the balanced anchor as a solver-side normalization only.
    # Do NOT stack it into the augmented contradiction system.
    if p_col_gen0 is not None and p_col_gen1 is not None:
        try:
            anchor_rhs = pow(5, -1, group_order)
            _log(
                f"  solver normalization only: a[gen0], a[gen1] are scaled by inv(5) mod {group_order} = {int(anchor_rhs)}"
            )
        except ValueError:
            _log("  solver normalization only: balanced anchor unavailable (5 not invertible mod group_order)")
    else:
        _log("  solver normalization only: balanced anchor unavailable")

    if not extra_rows_fp:
        _log("  no augmentation rows; cannot find certificate")
        return [], []

    A_extra = matrix(Fp, extra_rows_fp)
    A_full  = A_pinned.stack(A_extra)
    b_full  = vector(Fp, list(b_pinned) + extra_rhs)
    n_full  = A_full.nrows()

    row_labels = [f"walk[{i}]" for i in range(n_walk)] + extra_labels

    _log(f"  augmented system: {n_full} rows × {n_cols} cols over GF({n})")
    rank_A   = A_full.rank()
    rank_aug = A_full.augment(b_full.column()).rank()
    _log(f"  rank(A)={rank_A}  rank([A|b])={rank_aug}")

    if rank_A == rank_aug:
        _log("  ✓  system is consistent — no Farkas certificate exists")
        return [], []

    _log("  ✗  inconsistent — extracting left-kernel certificate ...")

    AT        = A_full.transpose()
    left_ker  = AT.right_kernel()
    left_null = left_ker.dimension()
    _log(f"  left kernel dimension: {left_null}")

    if left_null == 0:
        _log("  ✗  left kernel is trivial; unexpected")
        return [], []

    certificate_y = None
    for basis_vec in left_ker.basis():
        if sum(basis_vec[i] * b_full[i] for i in range(n_full)) != Fp(0):
            certificate_y = basis_vec
            break

    if certificate_y is None:
        basis_list = list(left_ker.basis())
        _log("  no basis vector satisfies y·b!=0; trying small linear combinations ...")
        found = False
        for i in range(len(basis_list)):
            for j in range(i + 1, len(basis_list)):
                for ci in range(1, min(n, 5)):
                    for cj in range(1, min(n, 5)):
                        cand = Fp(ci) * basis_list[i] + Fp(cj) * basis_list[j]
                        if sum(cand[k] * b_full[k] for k in range(n_full)) != Fp(0):
                            certificate_y = cand
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

    if certificate_y is None:
        _log("  ✗  could not find certificate")
        return [], []

    dot_b = sum(certificate_y[i] * b_full[i] for i in range(n_full))
    _log(f"  ✓  certificate found; y^T*b={int(dot_b)}")

    nonzero_entries = [(i, int(certificate_y[i])) for i in range(n_full) if certificate_y[i] != Fp(0)]

    walk_entries  = [(i, c) for i, c in nonzero_entries if i < n_walk]
    extra_entries = [(i, c) for i, c in nonzero_entries if i >= n_walk]

    _log(f"  certificate support: {len(nonzero_entries)} total  |  walk={len(walk_entries)}  extra={len(extra_entries)}")

    if walk_entries:
        _log("  walk rows in certificate:")
        for row_i, coeff in walk_entries[:12]:
            row_atoms = [(pruned_atoms[j], int(M_pruned[row_i, j])) for j in range(n_cols) if M_pruned[row_i, j] != 0]
            _log(f"    row {row_i:5d}  weight={coeff:8d}  {_brief_atom_list(row_atoms, max_items=6)}")
        if len(walk_entries) > 12:
            _log(f"    ... {len(walk_entries) - 12} more walk rows")
    else:
        _log("  (no walk rows in certificate)")

    if extra_entries:
        _log("  augmented rows in certificate:")
        for row_i, coeff in extra_entries:
            label = row_labels[row_i] if row_i < len(row_labels) else f"row {row_i}"
            _log(f"    row {row_i:5d}  weight={coeff:8d}  [{label}]")

    _section("CERTIFICATE DIAGNOSIS")

    extra_map = dict(extra_entries)
    anchor_row_idx = None
    gauge_row_idx  = None
    for row_i, coeff in extra_entries:
        lbl = row_labels[row_i] if row_i < len(row_labels) else ""
        if "anchor" in lbl:
            anchor_row_idx = row_i
        if "gauge" in lbl:
            gauge_row_idx = row_i

    _log(f"  anchor row weight in y  : {extra_map.get(anchor_row_idx)}")
    _log(f"  gauge row weight in y   : {extra_map.get(gauge_row_idx)}")

    if anchor_row_idx is not None and extra_map.get(anchor_row_idx):
        _log("  the anchor row participates in the contradiction; normalization is suspect")

    if not walk_entries:
        _log("  contradiction is entirely in normalization rows, not in walk data")
    else:
        atom_freq: dict = {}
        for row_i, _ in walk_entries:
            for j in range(n_cols):
                if M_pruned[row_i, j] != 0:
                    atom_freq[str(pruned_atoms[j])] = atom_freq.get(str(pruned_atoms[j]), 0) + 1
        top_atoms = sorted(atom_freq.items(), key=lambda kv: -kv[1])[:8]
        _log("  most frequent atoms in certificate rows:")
        for atom, freq in top_atoms:
            _log(f"    {atom:>8}  appears in {freq} certificate row(s)")

    walk_row_indices = sorted({i for i, _ in walk_entries})
    return nonzero_entries, walk_row_indices



def _build_balanced_anchor_row(Fp, n_cols, col_gen0, col_gen1, col_inf):
    """
    Build the balanced-anchor coefficients for solver-side normalization only.

    IMPORTANT: this helper returns a row-shaped vector for *inspection* but the
    caller should not append it as a relation row. It is meant to document the
    normalization choice:

        a[gen0] + a[gen1] - 5*a[∞] = 0

    When ∞ is unavailable, the fallback normalization is:
        a[gen0] - a[gen1] = 0
    """
    row = vector(Fp, n_cols)
    if col_gen0 is None or col_gen1 is None:
        return None, None, "anchor omitted"

    row[col_gen0] = Fp(1)
    row[col_gen1] = Fp(1)

    if col_inf is not None:
        row[col_inf] = Fp(-5)
        rhs = Fp(0)
        label = "anchor a[gen0]+a[gen1]-5*a[∞]=0"
    else:
        row[col_gen1] = Fp(-1)
        rhs = Fp(0)
        label = "anchor a[gen0]-a[gen1]=0"

    return row, rhs, label

if __name__ == "__main__":
    main()
