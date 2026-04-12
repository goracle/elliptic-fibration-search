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

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy Sage import — keep the module importable for linting outside Sage.
# ---------------------------------------------------------------------------
try:
    from sage.all import GF, ZZ, Integer, matrix, vector, Matrix
    _SAGE = True
except ImportError:
    _SAGE = False

try:
    import h5py
    import numpy as np
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

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

def check_homogeneous(M_ZZ, atoms: list, aidx: dict, group_order: int,
                      known_key: int, col_gen0, col_gen1, col_tgt0, col_tgt1,
                      col_inf):
    """
    Solve A_hom * x = 0 (walk relations only, no gauge/anchor rows) over
    GF(group_order).

    Then test whether the *known* log-G vector satisfies this system:

        v[col_gen0] = 0,  v[col_gen1] = 1,
        v[col_tgt0] = v[col_tgt1] = known_key / 2 mod l,
        v[col_inf]  = 0,
        v[all others] = 0.

    (The exact form of the log-G vector depends on the normalisation convention
    used when building the anchor.  The convention here matches the anchor
      a[gen0] - a[gen1] = 1  used in merge_experiment._dlp_build_affine_system:
      we assign a[gen1]=0, a[gen0]=1 as the canonical generator normalisation,
      then the target's log entries must satisfy a[tgt0]+a[tgt1] = known_key.)

    Reports:
      - kernel dimension of A_hom
      - whether the log-G vector is in the kernel (residual == 0)
      - if not, which rows of A_hom have nonzero residual for that vector
    """
    _section("CHECK 1: HOMOGENEOUS SYSTEM  (walk relations, no anchor)")

    n = group_order
    Fp = GF(Integer(n))
    n_rows = M_ZZ.nrows()
    n_cols = M_ZZ.ncols()

    A_hom = M_ZZ.change_ring(Fp)

    rank_hom  = A_hom.rank()
    ker_hom   = A_hom.right_kernel()
    null_hom  = ker_hom.dimension()

    _log(f"  rows={n_rows}  cols={n_cols}  rank={rank_hom}  nullity={null_hom}")
    _log(f"  (ideal: nullity >= 2 — gauge direction + DLP direction)")

    if null_hom == 0:
        _log("\n  ✗  Nullity=0 on the homogeneous system — the walk relations alone")
        _log("     are already inconsistent over GF(l).  The contradiction is in the")
        _log("     relation rows themselves, not introduced by the anchor.")
        _log("     → Proceed to Check 2 (left-kernel certificate) for root cause.")
    elif null_hom == 1:
        _log("\n  ⚠  Nullity=1 on the homogeneous system — only one free direction.")
        _log("     The gauge (∞) and DLP directions are fused or one is missing.")
        _log("     Adding the anchor (which has coefficient sum 0 but nonzero RHS)")
        _log("     may still be inconsistent if the single free direction is not the")
        _log("     difference a[gen0]-a[gen1].")
    else:
        _log(f"\n  ✓  Nullity={null_hom} — at least gauge + DLP directions present.")

    # --- Build the candidate log-G vector ---
    # Convention: a[gen1]=0, a[gen0]=1 (anchor: a[gen0]-a[gen1]=1).
    # Target: a[tgt0]+a[tgt1] = known_key.
    # We set a[tgt0] = a[tgt1] = known_key * inv(2)  if 2 is invertible,
    # otherwise a[tgt0] = known_key, a[tgt1] = 0  as a fallback.
    # All other atoms (including ∞) get 0.

    _log(f"\n  Building log-G candidate vector (known_key={known_key}) ...")

    missing = []
    for name, col in [("gen0", col_gen0), ("gen1", col_gen1),
                      ("tgt0", col_tgt0), ("tgt1", col_tgt1)]:
        if col is None:
            missing.append(name)
    if missing:
        _log(f"  ⚠  Cannot build log-G vector — columns missing: {missing}")
        _log("     Skipping log-G membership test.")
        return null_hom

    v_logG = vector(Fp, n_cols)
    v_logG[col_gen0] = Fp(1)
    v_logG[col_gen1] = Fp(0)
    if col_inf is not None:
        v_logG[col_inf] = Fp(0)

    # Split known_key evenly if possible.
    inv2 = None
    try:
        inv2 = Fp(2) ** (-1)
    except Exception:
        pass  # n is even — 2 not invertible mod n

    if inv2 is not None:
        half_key = Fp(known_key) * inv2
        v_logG[col_tgt0] = half_key
        v_logG[col_tgt1] = half_key
        _log(f"  a[gen0]=1, a[gen1]=0, a[tgt0]=a[tgt1]={int(half_key)} (= {known_key}/2 mod {n})")
    else:
        v_logG[col_tgt0] = Fp(known_key)
        v_logG[col_tgt1] = Fp(0)
        _log(f"  a[gen0]=1, a[gen1]=0, a[tgt0]={known_key}, a[tgt1]=0")
        _log(f"  (2 not invertible mod {n}; used asymmetric split)")

    # Check A_hom * v_logG == 0
    residual = A_hom * v_logG
    nonzero_rows = [(i, int(residual[i])) for i in range(n_rows) if residual[i] != Fp(0)]

    if not nonzero_rows:
        _log("\n  ✓  log-G vector IS in the kernel of A_hom.")
        _log("     The walk relations are consistent with the known solution.")
        _log("     The failure to solve is introduced by the anchor/gauge rows,")
        _log("     not by the walk data itself.")
    else:
        _log(f"\n  ✗  log-G vector is NOT in the kernel of A_hom.")
        _log(f"     {len(nonzero_rows)} relation row(s) have nonzero residual:")
        _log(f"     (showing up to 30)")
        for row_i, resid in nonzero_rows[:30]:
            # Recover the nonzero atoms in this row for context.
            row_atoms = [(atoms[j], int(M_ZZ[row_i, j]))
                         for j in range(n_cols) if M_ZZ[row_i, j] != 0]
            _log(f"    row {row_i:5d}  residual={resid:5d}  atoms={row_atoms}")
        if len(nonzero_rows) > 30:
            _log(f"    ... and {len(nonzero_rows) - 30} more rows")
        _log("\n     → The walk data itself contradicts the known key.")
        _log("       Most likely cause: a sign/coefficient error in how rows are")
        _log("       written into the relation matrix (e.g. wrong multiplicity for xi,")
        _log("       wrong sign for ∞, or a missing involution-closure row).")

    return null_hom


# ---------------------------------------------------------------------------
# Check 2: Left-kernel certificate (Farkas row)
# ---------------------------------------------------------------------------

def extract_contradiction_certificate(
    M_ZZ, atoms: list, group_order: int,
    col_inf, col_gen0, col_gen1,
    n_anchor_rows: int = 2,   # gauge-fix row + anchor row
):
    """
    Find a Farkas certificate y for the inconsistency of A * x = b.

    A is the augmented system: [A_hom | gauge_row | anchor_row], b has
    zeros for A_hom rows and [0, 1] for the gauge and anchor.

    We build A and b here from first principles using the same construction as
    merge_experiment._dlp_build_affine_system, then find y such that:
        y^T * A = 0    (y is in the left kernel of A)
        y^T * b != 0   (y exposes the contradiction)

    The rows of A are:
        rows 0 .. n_walk-1      : walk relations (RHS = 0)
        row  n_walk             : gauge row  a[∞] = 0         (if ∞ col present)
        row  n_walk + offset    : anchor row a[gen0]-a[gen1] = 1

    y is printed as a sparse combination of these rows.

    Returns the certificate vector y (as a list of (row_index, coeff) pairs).
    """
    _section("CHECK 2: CONTRADICTION CERTIFICATE  (left-kernel Farkas row)")

    if not _SAGE:
        raise RuntimeError("SageMath required")

    n = group_order
    Fp = GF(Integer(n))
    n_walk = M_ZZ.nrows()
    n_cols = M_ZZ.ncols()

    # --- Build augmented A, b ---
    A_hom_fp = M_ZZ.change_ring(Fp)

    extra_rows_fp = []
    extra_rhs     = []

    # Gauge-fix: a[∞] = 0
    if col_inf is not None:
        gauge_row = vector(Fp, n_cols)
        gauge_row[col_inf] = Fp(1)
        extra_rows_fp.append(gauge_row)
        extra_rhs.append(Fp(0))
        _log(f"  Gauge row: a[∞]=0  (col={col_inf})")
    else:
        _log("  No ∞ column found — gauge row omitted.")

    # Anchor: a[gen0] - a[gen1] = 1
    anchor_added = False
    if col_gen0 is not None and col_gen1 is not None:
        anchor_row = vector(Fp, n_cols)
        anchor_row[col_gen0] = Fp(1)
        anchor_row[col_gen1] = Fp(-1)
        extra_rows_fp.append(anchor_row)
        extra_rhs.append(Fp(1))
        _log(f"  Anchor row: a[gen0]-a[gen1]=1  (cols {col_gen0}, {col_gen1})")
        anchor_added = True
    else:
        _log(f"  ⚠  gen0 or gen1 column missing — anchor row omitted.")

    if not extra_rows_fp:
        _log("  ⚠  No extra rows to augment with — cannot identify contradiction source.")
        _log("     Provide --col-gen0 / --col-gen1 or ensure they are in the HDF5 file.")
        return []

    # Full augmented matrix A (rows × cols) and RHS b
    A_extra   = matrix(Fp, extra_rows_fp)
    A_full    = A_hom_fp.stack(A_extra)
    b_full    = vector(Fp, [Fp(0)] * n_walk + extra_rhs)

    n_full    = A_full.nrows()

    _log(f"\n  Augmented system: {n_full} rows × {n_cols} cols over GF({n})")

    # Check consistency first.
    rank_A   = A_full.rank()
    rank_aug = A_full.augment(b_full.column()).rank()

    _log(f"  rank(A)={rank_A}  rank([A|b])={rank_aug}")

    if rank_A == rank_aug:
        _log("  ✓  System is consistent — no Farkas certificate exists.")
        _log("     (solve_right should succeed; check caller for other bugs)")
        return []

    _log(f"  ✗  System is INCONSISTENT — rank(A)={rank_A} < rank([A|b])={rank_aug}")
    _log("     Extracting left-kernel certificate ...")

    # --- Find y in left kernel of A such that y^T b != 0 ---
    # Left kernel of A = right kernel of A^T.
    # We want vectors y with A^T * y = 0 (left null space) and b · y != 0.
    #
    # Strategy: compute the right kernel of A^T, then filter for those y
    # satisfying b · y != 0.  In exact arithmetic the first such basis vector
    # is our certificate.  If the basis vectors all have b·y=0 (should not
    # happen when the system is truly inconsistent) we take linear combinations.

    AT = A_full.transpose()
    left_ker = AT.right_kernel()
    left_null = left_ker.dimension()

    _log(f"  Left kernel dimension: {left_null}")

    if left_null == 0:
        # This should not happen given rank_A < rank_aug, but guard anyway.
        _log("  ✗  Left kernel is trivial — unexpected; check field characteristic.")
        return []

    certificate_y = None
    for basis_vec in left_ker.basis():
        dot = sum(basis_vec[i] * b_full[i] for i in range(n_full))
        if dot != Fp(0):
            certificate_y = basis_vec
            break

    if certificate_y is None:
        # Try linear combinations of the first few basis vectors.
        basis_list = list(left_ker.basis())
        _log("  No single basis vector satisfies b·y!=0; trying linear combinations ...")
        found = False
        for i in range(len(basis_list)):
            for j in range(i + 1, len(basis_list)):
                for ci in range(1, min(n, 5)):
                    for cj in range(1, min(n, 5)):
                        cand = Fp(ci) * basis_list[i] + Fp(cj) * basis_list[j]
                        dot  = sum(cand[k] * b_full[k] for k in range(n_full))
                        if dot != Fp(0):
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
        _log("  ✗  Could not find a certificate with b·y!=0 — the inconsistency")
        _log("     may involve the field characteristic.  Inspect basis vectors manually.")
        return []

    # --- Report the certificate ---
    dot_b = sum(certificate_y[i] * b_full[i] for i in range(n_full))
    _log(f"\n  ✓  Certificate found.  y^T * b = {int(dot_b)}  (nonzero — contradiction confirmed)")

    nonzero_entries = [(i, int(certificate_y[i])) for i in range(n_full)
                       if certificate_y[i] != Fp(0)]

    _log(f"\n  Certificate y has {len(nonzero_entries)} nonzero entries:")
    _log(f"  (Each entry is a weight on one row of the augmented system.)")

    walk_entries   = [(i, c) for i, c in nonzero_entries if i < n_walk]
    extra_entries  = [(i, c) for i, c in nonzero_entries if i >= n_walk]

    _log(f"\n  Walk-relation rows in certificate ({len(walk_entries)} rows):")
    if walk_entries:
        _log(f"  {'row':>6}  {'weight':>8}  atoms_in_row")
        for row_i, coeff in walk_entries[:50]:
            row_atoms = [(atoms[j], int(M_ZZ[row_i, j]))
                         for j in range(n_cols) if M_ZZ[row_i, j] != 0]
            _log(f"  {row_i:6d}  {coeff:8d}  {row_atoms}")
        if len(walk_entries) > 50:
            _log(f"  ... and {len(walk_entries) - 50} more walk rows")
    else:
        _log("  (none)")

    if extra_entries:
        extra_labels = []
        row_idx = n_walk
        if col_inf is not None:
            extra_labels.append(f"gauge (a[∞]=0, row {row_idx})")
            row_idx += 1
        if anchor_added:
            extra_labels.append(f"anchor (a[gen0]-a[gen1]=1, row {row_idx})")

        _log(f"\n  Augmented rows in certificate ({len(extra_entries)} rows):")
        for (i, c), label in zip(extra_entries, extra_labels):
            _log(f"  row {i:5d}  weight={c:8d}  [{label}]")

    # --- Diagnose the certificate structure ---
    _section("CERTIFICATE DIAGNOSIS")

    anchor_weight  = None
    gauge_weight   = None
    anchor_row_idx = n_walk + (1 if col_inf is not None else 0) if anchor_added else None
    gauge_row_idx  = n_walk if col_inf is not None else None

    for i, c in extra_entries:
        if gauge_row_idx is not None and i == gauge_row_idx:
            gauge_weight = c
        if anchor_row_idx is not None and i == anchor_row_idx:
            anchor_weight = c

    _log(f"  Anchor row weight in y  : {anchor_weight}")
    _log(f"  Gauge row weight in y   : {gauge_weight}")

    if anchor_weight is not None and anchor_weight != 0:
        _log("\n  The anchor row participates in the contradiction.")
        _log("  This means the RHS value '1' on the anchor is incompatible")
        _log("  with the walk relations.  Likely causes:")
        _log("   (a) Conservation law: every walk row satisfies Σ coeff = 0,")
        _log("       so any linear combination of walk rows also sums to 0.")
        _log("       The anchor row a[gen0]-a[gen1]=1 has Σ coeff = 0, RHS=1.")
        _log("       If the walk rows force a[gen0]-a[gen1]=0 the system is")
        _log("       inconsistent — check whether a parity/flat kernel direction")
        _log("       equates gen0 and gen1.")
        _log("   (b) The gen0/gen1 columns are aliased (same column index).")
        _log("       In that case the anchor reads 0=1, an immediate contradiction.")
        if col_gen0 == col_gen1:
            _log(f"\n  ✗  DETECTED: col_gen0 == col_gen1 == {col_gen0}.")
            _log("     The two generator atoms map to the same column — the anchor")
            _log("     row literally encodes 0=1.  Check divisor_xs ordering and")
            _log("     that both BASE_DIVISOR roots are distinct in the leaf set.")

    if not walk_entries:
        _log("\n  No walk rows appear in the certificate.")
        _log("  The contradiction is entirely within the extra (gauge/anchor) rows.")
        _log("  The walk data is self-consistent; only the added constraints fight each other.")
    else:
        # Summarise which atoms appear most in the certificate rows.
        atom_freq: dict = {}
        for row_i, _ in walk_entries:
            for j in range(n_cols):
                if M_ZZ[row_i, j] != 0:
                    a = atoms[j]
                    atom_freq[a] = atom_freq.get(a, 0) + 1
        top_atoms = sorted(atom_freq.items(), key=lambda kv: -kv[1])[:10]
        _log(f"\n  Most frequent atoms in certificate rows (top 10):")
        for atom, freq in top_atoms:
            _log(f"    {atom:>8}  appears in {freq} certificate row(s)")

    return nonzero_entries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    _log(f"\n{'#'*70}")
    _log("# DIAGNOSTICS COMPLETE")
    _log(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
