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
    Check 3: Three structural diagnostics from Gemini.

    A) Column-order test: for each special column j, find the smallest k s.t.
       k*e_j lies in the row space of A over GF(p).  k=1 means the column is
       already determined by the walk; k=p means it is completely free.

    B) Rank-without-inf: recompute nullity after dropping the inf column
       entirely.  If nullity jumps back to >=2, the collapse is driven by
       over-use of inf as a balancing term.

    C) Fusion audit: list every pair of atoms that the walk equates
       (support-2, +1/-1 kernel vectors) before any pinning.
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
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

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

    # ------------------------------------------------------------------
    # A) Column-order test for special atoms
    # ------------------------------------------------------------------
    _log("\n  --- A) Column-order test (special atoms) ---")
    _log("  For each special column j, smallest k s.t. k*e_j in RowSpace(A).")
    _log("  k=1 => column fully determined by walk; k=p => completely free.")

    row_space = A.row_space()
    special = [("inf",  p_col_inf),
               ("gen0", p_col_gen0), ("gen1", p_col_gen1),
               ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]

    for name, col in special:
        if col is None:
            _log(f"  {name:6s}: column absent after prune")
            continue
        e_j = vector(Fp, n_cols)
        e_j[col] = Fp(1)
        found_k = None
        for k in range(1, min(50, n)):
            if Fp(k) * e_j in row_space:
                found_k = k
                break
        if found_k is not None:
            _log(f"  {name:6s} (col {col:5d}): k={found_k}  "
                 + ("*** column is over-determined by walk rows" if found_k == 1
                    else f"order-{found_k} dependency"))
        else:
            _log(f"  {name:6s} (col {col:5d}): k>50 -- effectively free (good)")

    # ------------------------------------------------------------------
    # B) Rank without inf
    # ------------------------------------------------------------------
    _log("\n  --- B) Rank-without-inf test ---")
    if p_col_inf is not None:
        cols_no_inf = [j for j in range(n_cols) if j != p_col_inf]
        A_no_inf = A.matrix_from_columns(cols_no_inf)
        ker_no_inf = A_no_inf.right_kernel()
        null_no_inf = ker_no_inf.dimension()
        _log(f"  Full matrix nullity  : {A.right_kernel().dimension()}")
        _log(f"  Without-inf nullity  : {null_no_inf}")
        full_null = A.right_kernel().dimension()
        if null_no_inf > full_null:
            _log(f"  Nullity INCREASES by {null_no_inf - full_null} without inf.")
            _log("  *** inf column is absorbing degrees of freedom -- normalization leakage.")
        elif null_no_inf == full_null:
            _log("  Nullity unchanged -- inf column is not the source of collapse.")
        else:
            _log(f"  Nullity DECREASES by {full_null - null_no_inf} without inf.")
            _log("  inf column was contributing free directions -- unexpected.")
    else:
        _log("  No inf column present -- test skipped.")

    # ------------------------------------------------------------------
    # C) Fusion audit (all +1/-1 kernel pairs before pinning)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # C) Direct fusion audit
    #    Fusion = proportional columns over GF(p).
    #    This is the cleanest test: if col_i = λ col_j, then the walk
    #    relations cannot distinguish those atoms.
    # ------------------------------------------------------------------
    _log("\n  --- C) Direct fusion audit (proportional columns) ---")

    special_by_col = {
        c: name for name, c in [
            ("inf", p_col_inf),
            ("gen0", p_col_gen0),
            ("gen1", p_col_gen1),
            ("tgt0", p_col_tgt0),
            ("tgt1", p_col_tgt1),
        ] if c is not None
    }

    # Map: normalized column signature -> list of columns with that signature.
    # Two columns are in the same bucket iff they are proportional over GF(p).
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

        sig_t = tuple(sig)
        col_groups.setdefault(sig_t, []).append(j)

    fusion_groups = [cols for cols in col_groups.values() if len(cols) > 1]

    if zero_cols:
        _log(f"  {len(zero_cols)} zero column(s) found -- completely absent from walk rows.")
        if len(zero_cols) <= 20:
            _log("    " + ", ".join(f"{pruned_atoms[j]}(col={j})" for j in zero_cols))
        else:
            preview = ", ".join(f"{pruned_atoms[j]}(col={j})" for j in zero_cols[:20])
            _log(f"    {preview}")
            _log("    ...")

    if fusion_groups:
        _log(f"  {len(fusion_groups)} fusion class(es) found (proportional columns):")
        for cols in fusion_groups[:20]:
            labels = []
            special_hits = []
            for j in cols:
                atom = str(pruned_atoms[j])
                if j in special_by_col:
                    special_hits.append(special_by_col[j])
                    labels.append(f"{atom}({special_by_col[j]})")
                else:
                    labels.append(atom)
            flag = ""
            if special_hits:
                flag = f"  *** SPECIAL ATOMS: {sorted(set(special_hits))}"
            _log(f"    {labels}{flag}")
        if len(fusion_groups) > 20:
            _log(f"    ... and {len(fusion_groups) - 20} more fusion class(es)")
    else:
        _log("  No proportional column classes found -- no direct fusion detected.")

    # Keep the old kernel-basis fusion scan as a sanity check.
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
        _log(f"  {len(kernel_fusions)} support-2 kernel fusion(s):")
        for a0, a1 in kernel_fusions:
            _log(f"    a[{a0}] = a[{a1}]")
    else:
        _log("  No support-2 kernel fusions.")
    # ------------------------------------------------------------------
    # D) Which rows overdetermine gen0 / gen1
    #    The row space contains e_gen0 (k=1 above).  Find the actual rows
    #    that have nonzero gen0/gen1 entries — these are the rows "pinning"
    #    the generator columns, which should never happen in a clean setup.
    # ------------------------------------------------------------------
    _log("\n  --- D) Rows that directly constrain gen0 / gen1 ---")
    _log("  (any row with nonzero entry in a special column is suspicious)")
    for col_name, p_col in [("gen0", p_col_gen0), ("gen1", p_col_gen1),
                             ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]:
        if p_col is None:
            continue
        hitting_rows = [(i, int(A[i, p_col])) for i in range(n_rows)
                        if A[i, p_col] != Fp(0)]
        _log(f"  {col_name}: {len(hitting_rows)} row(s) with nonzero entry")
        for row_i, coeff in hitting_rows[:10]:
            row_atoms = [(pruned_atoms[j], int(M_pruned[row_i, j]))
                         for j in range(n_cols) if M_pruned[row_i, j] != 0]
            _log(f"    row {row_i:5d}  coeff={coeff:6d}  atoms={row_atoms}")
        if len(hitting_rows) > 10:
            _log(f"    ... and {len(hitting_rows) - 10} more rows")

    # ------------------------------------------------------------------
    # E) Row-subsampling stability
    #    Remove 10% of rows at random (5 trials) and check if nullity
    #    changes.  If it does, the system is in a brittle over-determined
    #    regime; if it stays the same, the constraints are redundant.
    # ------------------------------------------------------------------
    _log("\n  --- E) Row-subsampling stability (5 trials, 10% row removal) ---")
    import random as _random
    _random.seed(42)
    base_null = A.right_kernel().dimension()
    n_drop = max(1, n_rows // 10)
    _log(f"  Base nullity: {base_null}  (dropping {n_drop} of {n_rows} rows per trial)")
    for trial in range(5):
        drop = set(_random.sample(range(n_rows), n_drop))
        keep = [i for i in range(n_rows) if i not in drop]
        A_sub = A.matrix_from_rows(keep)
        null_sub = A_sub.right_kernel().dimension()
        delta = null_sub - base_null
        flag = ""
        if delta > 2:
            flag = "  *** large jump -- brittle over-determined regime"
        elif delta > 0:
            flag = "  (slight increase -- some redundancy)"
        _log(f"  trial {trial+1}: nullity={null_sub}  delta={delta:+d}{flag}")


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
    extract_contradiction_certificate(
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

    _log(f"\n{'#'*70}")
    _log("# DIAGNOSTICS COMPLETE")
    _log(f"{'#'*70}\n")

def _extract_pin_rows(ker, pruned_atoms, n_cols, Fp, p_col_inf=None):
    """
    Scan a right-kernel basis and return (pin_rows, pin_rhs, pin_labels) for
    every basis vector whose support is a single non-inf atom (isolated atoms).
    Logs each pinned atom.  Does NOT consume the inf/gauge direction.

    Also diagnoses non-isolated kernel directions:
      FUSION   -- support_size=2, coeffs +1/-1: two atoms forced to equal logs
      PARITY   -- all coefficients equal
      OTHER    -- anything else; distinct coefficients listed
    """
    n = int(Fp.characteristic())
    pin_rows   = []
    pin_rhs    = []
    pin_labels = []
    for vi, vec in enumerate(ker.basis()):
        support = [(j, int(vec[j])) for j in range(n_cols) if vec[j] != Fp(0)]
        if not support:
            continue

        # --- gauge direction ---
        if len(support) == 1 and p_col_inf is not None and support[0][0] == p_col_inf:
            _log(f"  kernel[{vi}]: GAUGE (inf) -- skipping")
            continue

        # --- isolated atom: pin it ---
        if len(support) == 1:
            j, _ = support[0]
            atom = pruned_atoms[j]
            pin_row = vector(Fp, n_cols)
            pin_row[j] = Fp(1)
            pin_rows.append(pin_row)
            pin_rhs.append(Fp(0))
            pin_labels.append(f"pin a[{atom}]=0")
            _log(f"  kernel[{vi}]: ISOLATED atom={atom} -- pinning a[{atom}]=0")
            continue

        # --- fusion pair: two atoms forced to identical logs ---
        if len(support) == 2:
            (j0, c0), (j1, c1) = support
            if (c0 == 1 and c1 == n - 1) or (c0 == n - 1 and c1 == 1):
                a0, a1 = pruned_atoms[j0], pruned_atoms[j1]
                _log(f"  kernel[{vi}]: FUSION -- walk data forces "
                     f"a[{a0}] = a[{a1}]  (log-space collapse)")
                continue

        # --- other multi-atom directions ---
        coeffs_vals = [c for _, c in support]
        is_flat = len(set(coeffs_vals)) == 1
        kind = "PARITY/CONSERVATION" if is_flat else "OTHER"
        c0_val = coeffs_vals[0] if is_flat else None
        _log(f"  kernel[{vi}]: {kind}  support_size={len(support)}"
             + (f"  all_coeffs={c0_val}" if is_flat else f"  distinct_coeffs={sorted(set(coeffs_vals))}"))
    return pin_rows, pin_rhs, pin_labels


def check_homogeneous(M_ZZ, atoms: list, aidx: dict, group_order: int,
                      known_key: int, col_gen0, col_gen1, col_tgt0, col_tgt1,
                      col_inf):
    """
    Prune M_ZZ via prune_dest_only, then solve A_hom * x = 0 over GF(group_order).

    Tests whether the known log-G vector lies in the kernel of the pruned
    homogeneous system, i.e. whether the walk data is consistent with the
    known answer before any anchor is added.

    Reports kernel dimension and, if the log-G vector is not in the kernel,
    which pruned rows have nonzero residual.
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
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    n_removed = len(removed)
    _log(f"  prune_dest_only removed {n_removed} dest-only atoms "
         f"({M_ZZ.nrows()} → {M_pruned.nrows()} rows, "
         f"{M_ZZ.ncols()} → {M_pruned.ncols()} cols)")

    # Remap col indices into the pruned column space.

    p_col_gen0 = _remap(col_gen0, atoms, pruned_aidx)
    p_col_gen1 = _remap(col_gen1, atoms, pruned_aidx)
    p_col_tgt0 = _remap(col_tgt0, atoms, pruned_aidx)
    p_col_tgt1 = _remap(col_tgt1, atoms, pruned_aidx)
    p_col_inf  = _remap(col_inf,  atoms, pruned_aidx)

    for name, col in [("gen0", p_col_gen0), ("gen1", p_col_gen1),
                      ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]:
        if col is None:
            _log(f"  ⚠  {name} pruned away — it was dest-only (never appeared as xi).")

    n_rows = M_pruned.nrows()
    n_cols = M_pruned.ncols()

    A_hom = M_pruned.change_ring(Fp)

    # --- pin isolated atoms first (same logic as Check 2) ---
    ker_pre  = A_hom.right_kernel()
    null_pre = ker_pre.dimension()
    _log(f"\n  Pre-pin nullity: {null_pre} on the {n_rows}×{n_cols} system")

    pin_rows, pin_rhs, pin_labels = _extract_pin_rows(
        ker_pre, pruned_atoms, n_cols, Fp, p_col_inf
    )

    if pin_rows:
        A_hom = A_hom.stack(matrix(Fp, pin_rows))
        _log(f"  Pinned {len(pin_rows)} isolated atom(s)")

    rank_hom = A_hom.rank()
    ker_hom  = A_hom.right_kernel()
    null_hom = ker_hom.dimension()

    _log(f"  rows={M_pruned.nrows()}  cols={n_cols}  rank={rank_hom}  nullity={null_hom}"
         f"  (after pinning)")
    _log(f"  (ideal: nullity >= 2 — gauge direction + DLP direction)")

    if null_hom == 0:
        _log("\n  ✗  Nullity=0 — walk relations alone are already inconsistent over GF(l).")
        _log("     The contradiction is in the relation rows, not the anchor.")
        _log("     → Proceed to Check 2 for root cause.")
    elif null_hom == 1:
        _log("\n  ⚠  Nullity=1 — gauge and DLP directions are fused or one is missing.")
        _log("     Adding the anchor may still be inconsistent if the single free")
        _log("     direction is not a[gen0]-a[gen1].")
    else:
        _log(f"\n  ✓  Nullity={null_hom} — at least gauge + DLP directions present.")

    # --- inspect the surviving kernel basis vectors ---
    _log(f"\n  Surviving kernel basis ({ker_hom.dimension()} vector(s)):")
    special_cols = {name: col for name, col in [
        ("gen0", p_col_gen0), ("gen1", p_col_gen1),
        ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1),
        ("inf",  p_col_inf),
    ] if col is not None}
    for bi, bv in enumerate(ker_hom.basis()):
        support = [(j, int(bv[j])) for j in range(n_cols) if bv[j] != Fp(0)]
        coeffs  = [c for _, c in support]
        is_flat = len(set(coeffs)) == 1
        special_vals = {name: int(bv[col]) for name, col in special_cols.items()}
        all_special_same = len(set(special_vals.values())) == 1
        _log(f"  basis[{bi}]:  support_size={len(support)}  flat={'yes' if is_flat else 'no'}")
        _log(f"    special atom values: {special_vals}")
        if is_flat:
            _log("    *** ALL-ONES (flat) kernel vector -- every atom maps to the same log.")
            _log("        Total log-space collapse confirmed: no DLP dimension in walk data.")
        elif all_special_same:
            _log("    *** Special atoms all equal -- DLP direction absent for gen/tgt.")
        else:
            _log("    OK -- special atoms differ, DLP direction present.")

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

    # Partition: rows that only touch assigned atoms are true failures.
    # Rows touching unassigned fb atoms will always have nonzero residual
    # because v_logG is not a full oracle -- that is expected, not a bug.
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
        _log("\n  \u2713  log-G vector IS in the kernel of A_hom (pruned).")
        _log("     Walk relations are consistent with the known solution.")
        _log("     The failure is introduced by the anchor/gauge rows, not the walk data.")
    else:
        if fb_residuals:
            _log(f"\n  \u2139  {len(fb_residuals)} row(s) have nonzero residual because they contain")
            _log( "     unassigned factor-base atoms (v_logG leaves them at 0).")
            _log( "     This is expected -- the test vector is not a full oracle.")
        if true_failures:
            _log(f"\n  \u2717  {len(true_failures)} row(s) fail on assigned atoms only -- genuine contradiction:")
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
            _log("\n  \u2713  No true failures -- all residuals are from unassigned fb atoms.")
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

    # --- Step 1: prune_dest_only ---
    protected = [a for a in [atoms[col_gen0] if col_gen0 is not None else None,
                              atoms[col_gen1] if col_gen1 is not None else None]
                 if a is not None]
    M_pruned, pruned_atoms, removed = prune_dest_only(M_ZZ, atoms, protected=protected)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    _log(f"  prune_dest_only: {len(removed)} atoms removed  "
         f"({M_ZZ.nrows()}→{M_pruned.nrows()} rows, "
         f"{M_ZZ.ncols()}→{M_pruned.ncols()} cols)")

    p_col_inf  = _remap(col_inf,  atoms, pruned_aidx)
    p_col_gen0 = _remap(col_gen0, atoms, pruned_aidx)
    p_col_gen1 = _remap(col_gen1, atoms, pruned_aidx)

    n_walk = M_pruned.nrows()
    n_cols = M_pruned.ncols()

    A_hom_fp = M_pruned.change_ring(Fp)

    # --- Step 2: nullity-prune ---
    ker_pre  = A_hom_fp.right_kernel()
    null_pre = ker_pre.dimension()
    _log(f"\n  Pre-nullity-prune: nullity={null_pre} on the {n_walk}×{n_cols} homogeneous system")

    pin_rows, pin_rhs, pin_labels = _extract_pin_rows(
        ker_pre, pruned_atoms, n_cols, Fp, p_col_inf
    )

    if pin_rows:
        A_pinned = A_hom_fp.stack(matrix(Fp, pin_rows))
        b_pinned = vector(Fp, [Fp(0)] * n_walk + pin_rhs)
        ker_post = A_pinned.right_kernel()
        _log(f"\n  After pinning {len(pin_rows)} isolated atom(s): nullity {null_pre} → {ker_post.dimension()}")
    else:
        A_pinned = A_hom_fp
        b_pinned = vector(Fp, [Fp(0)] * n_walk)
        _log(f"  No isolated atoms to pin.")

    n_pinned_rows = A_pinned.nrows()

    # --- Step 3: gauge-fix and balanced anchor ---
    extra_rows_fp = []
    extra_rhs     = []
    extra_labels  = []

    # Gauge: a[inf] = 0 (identity element of the Jacobian has log 0).
    if p_col_inf is not None:
        gauge_row = vector(Fp, n_cols)
        gauge_row[p_col_inf] = Fp(1)
        extra_rows_fp.append(gauge_row)
        extra_rhs.append(Fp(0))
        _log(f"  Gauge row: a[∞]=0  (col={p_col_inf})")
        extra_labels.append(f"gauge a[∞]=0  (col={p_col_inf})")
    else:
        _log("  No ∞ column after prune — gauge row omitted.")

    # Affine anchor: a[gen0] = 1.  This puts a nonzero value in b_full so
    # that a Farkas certificate can exist if the system is inconsistent.
    # A homogeneous anchor (RHS=0) makes b_full all-zeros, which always
    # looks consistent and can never produce a certificate.
    anchor_added = False
    if p_col_gen0 is not None:
        anchor_row = vector(Fp, n_cols)
        anchor_row[p_col_gen0] = Fp(1)
        anchor_rhs = Fp(1)
        anchor_label = "affine anchor a[gen0]=1"
        extra_rows_fp.append(anchor_row)
        extra_rhs.append(anchor_rhs)
        extra_labels.append(anchor_label)
        _log(f"  Anchor row: {anchor_label}")
        anchor_added = True
    else:
        _log("  ⚠  gen0 missing after prune — anchor row omitted.")

    if not extra_rows_fp:
        _log("  ⚠  No augmentation rows — cannot find certificate.")
        return []

    A_extra = matrix(Fp, extra_rows_fp)
    A_full  = A_pinned.stack(A_extra)
    b_full  = vector(Fp, list(b_pinned) + extra_rhs)
    n_full  = A_full.nrows()

    row_labels = (
        [f"walk[{i}]" for i in range(n_walk)]
        + pin_labels
        + extra_labels
    )

    _log(f"\n  Augmented system: {n_full} rows × {n_cols} cols over GF({n})")

    rank_A   = A_full.rank()
    rank_aug = A_full.augment(b_full.column()).rank()
    _log(f"  rank(A)={rank_A}  rank([A|b])={rank_aug}")

    if rank_A == rank_aug:
        _log("  ✓  System is consistent — no Farkas certificate exists.")
        _log("     (solve_right should succeed after this pipeline; check caller)")
        return []

    _log(f"  ✗  INCONSISTENT — extracting left-kernel certificate ...")

    AT        = A_full.transpose()
    left_ker  = AT.right_kernel()
    left_null = left_ker.dimension()
    _log(f"  Left kernel dimension: {left_null}")

    if left_null == 0:
        _log("  ✗  Left kernel is trivial — unexpected; check field characteristic.")
        return []

    certificate_y = None
    for basis_vec in left_ker.basis():
        dot = sum(basis_vec[i] * b_full[i] for i in range(n_full))
        if dot != Fp(0):
            certificate_y = basis_vec
            break

    if certificate_y is None:
        basis_list = list(left_ker.basis())
        _log("  No single basis vector satisfies b·y!=0; trying linear combinations ...")
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
        _log("  ✗  Could not find certificate — inspect basis vectors manually.")
        return []

    dot_b = sum(certificate_y[i] * b_full[i] for i in range(n_full))
    _log(f"\n  ✓  Certificate found.  y^T * b = {int(dot_b)}  (nonzero — contradiction confirmed)")

    nonzero_entries = [(i, int(certificate_y[i])) for i in range(n_full)
                       if certificate_y[i] != Fp(0)]

    _log(f"\n  Certificate y has {len(nonzero_entries)} nonzero entries:")

    walk_entries  = [(i, c) for i, c in nonzero_entries if i < n_walk]
    extra_entries = [(i, c) for i, c in nonzero_entries if i >= n_walk]

    _log(f"\n  Walk-relation rows in certificate ({len(walk_entries)} rows):")
    if walk_entries:
        _log(f"  {'row':>6}  {'weight':>8}  atoms_in_row")
        for row_i, coeff in walk_entries[:50]:
            row_atoms = [(pruned_atoms[j], int(M_pruned[row_i, j]))
                         for j in range(n_cols) if M_pruned[row_i, j] != 0]
            _log(f"  {row_i:6d}  {coeff:8d}  {row_atoms}")
        if len(walk_entries) > 50:
            _log(f"  ... and {len(walk_entries) - 50} more walk rows")
    else:
        _log("  (none)")

    if extra_entries:
        _log(f"\n  Augmented rows in certificate ({len(extra_entries)} rows):")
        for row_i, coeff in extra_entries:
            label = row_labels[row_i] if row_i < len(row_labels) else f"row {row_i}"
            _log(f"  row {row_i:5d}  weight={coeff:8d}  [{label}]")

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

    _log(f"  Anchor row weight in y  : {extra_map.get(anchor_row_idx)}")
    _log(f"  Gauge row weight in y   : {extra_map.get(gauge_row_idx)}")

    if anchor_row_idx is not None and extra_map.get(anchor_row_idx):
        _log("\n  The anchor row participates in the contradiction.")
        _log("  With the balanced anchor, this means the generator columns")
        _log("  are still not behaving independently in the relation space.")

    if not walk_entries:
        _log("\n  No walk rows in certificate — contradiction is entirely in the")
        _log("  augmented rows (gauge/pin/anchor).  Walk data is self-consistent.")
    else:
        atom_freq: dict = {}
        for row_i, _ in walk_entries:
            for j in range(n_cols):
                if M_pruned[row_i, j] != 0:
                    atom_freq[str(pruned_atoms[j])] = atom_freq.get(str(pruned_atoms[j]), 0) + 1
        top_atoms = sorted(atom_freq.items(), key=lambda kv: -kv[1])[:10]
        _log(f"\n  Most frequent atoms in certificate rows (top 10):")
        for atom, freq in top_atoms:
            _log(f"    {atom:>8}  appears in {freq} certificate row(s)")

    return nonzero_entries

def _build_balanced_anchor_row(Fp, n_cols, col_gen0, col_gen1, col_inf):
    """
    Build an anchor row that is compatible with the row-sum conservation law.

    Preferred form:
        a[gen0] + a[gen1] - 2*a[∞] = 0

    If ∞ is unavailable, fall back to:
        a[gen0] - a[gen1] = 0
    """
    row = vector(Fp, n_cols)
    if col_gen0 is None or col_gen1 is None:
        return None, None, "anchor omitted"

    row[col_gen0] = Fp(1)
    row[col_gen1] = Fp(1)

    if col_inf is not None:
        row[col_inf] = Fp(-2)
        rhs = Fp(0)
        label = "anchor a[gen0]+a[gen1]-2*a[∞]=0"
    else:
        row[col_gen1] = Fp(-1)
        rhs = Fp(0)
        label = "anchor a[gen0]-a[gen1]=0"

    return row, rhs, label

if __name__ == "__main__":
    main()
