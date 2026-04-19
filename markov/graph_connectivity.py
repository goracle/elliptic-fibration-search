#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

import h5py
import numpy as np

"""
graph_connectivity.py — bipartite connectivity analysis for a DLP relation matrix

This version fixes the solve wiring so fixed variables (anchors) actually
enter the linear system, and inconsistencies are detected in the augmented
system rather than in the homogeneous submatrix.
"""

_XI_COEFF = 3


# ---------------------------------------------------------------------------
# HDF5 load
# ---------------------------------------------------------------------------

def load_hdf5(path: str) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        data = f["csr/data"][:]
        indices = f["csr/indices"][:]
        indptr = f["csr/indptr"][:]
        shape = tuple(int(x) for x in f["csr/shape"][:])

        atoms_raw = f["atoms"][:]
        atoms = [
            a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
            for a in atoms_raw
        ]

        aidx_raw = f["atom_index"][()]
        if isinstance(aidx_raw, (bytes, np.bytes_)):
            aidx_raw = aidx_raw.decode("utf-8")
        aidx: dict[str, int] = json.loads(aidx_raw)

        def _scalar(key: str):
            if key not in f:
                return None
            v = int(f[key][()])
            return v if v >= 0 else None

        group_order = _scalar("group_order")
        divisor_xs = [int(x) for x in f["divisor_xs"][:]] if "divisor_xs" in f else None
        col_inf = _scalar("col_inf")
        col_gen0 = _scalar("col_gen0")
        col_gen1 = _scalar("col_gen1")
        col_tgt0 = _scalar("col_tgt0")
        col_tgt1 = _scalar("col_tgt1")

        def _xs_dataset(*names):
            for name in names:
                if name in f:
                    return [int(x) for x in f[name][:]]
            return None

        ell_torsion_xs = _xs_dataset("ell_torsion_xs", "torsion_xs", "bad_atoms")

    nrows, ncols = shape
    return dict(
        data=data,
        indices=indices,
        indptr=indptr,
        nrows=nrows,
        ncols=ncols,
        atoms=atoms,
        aidx=aidx,
        group_order=group_order,
        divisor_xs=divisor_xs,
        ell_torsion_xs=ell_torsion_xs,
        col_inf=col_inf,
        col_gen0=col_gen0,
        col_gen1=col_gen1,
        col_tgt0=col_tgt0,
        col_tgt1=col_tgt1,
    )


# ---------------------------------------------------------------------------
# Prune dest-only atoms
# ---------------------------------------------------------------------------

def prune_dest_only(
    data,
    indices,
    indptr,
    nrows: int,
    ncols: int,
    atoms,
    protected_cols: set[int],
    excluded_cols: set[int] | None = None,
):
    """Prune columns that never appear as source-like atoms (|coeff| == 3).

    Any row touching an excluded column is discarded entirely, because the
    corresponding relation is not safe to use in log space.

    Returns
    -------
    pdata, pidx, pptr, pnrows, pncols, patoms, paidx, kept_cols, kept_rows
    """
    if excluded_cols is None:
        excluded_cols = set()

    source_cols: set[int] = set(protected_cols) - set(excluded_cols)
    for v, c in zip(data, indices):
        ci = int(c)
        if ci in excluded_cols:
            continue
        if abs(int(v)) == _XI_COEFF:
            source_cols.add(ci)

    kept_cols = sorted(source_cols)
    col_remap = {old: new for new, old in enumerate(kept_cols)}

    new_data, new_idx, new_ptr = [], [], [0]
    kept_rows = []
    n_malformed = 0
    for r in range(nrows):
        rs, re = int(indptr[r]), int(indptr[r + 1])
        row_cols = [int(indices[k]) for k in range(rs, re)]
        if any(c in excluded_cols for c in row_cols):
            continue
        # Drop malformed relations: coefficients must sum to zero (principal divisor).
        if sum(int(data[k]) for k in range(rs, re)) != 0:
            n_malformed += 1
            continue
        row_ents = [
            (col_remap[c], int(data[k]))
            for k, c in ((k, int(indices[k])) for k in range(rs, re))
            if c in col_remap
        ]
        if row_ents:
            kept_rows.append(r)
            for c, v in row_ents:
                new_data.append(v)
                new_idx.append(c)
            new_ptr.append(len(new_data))

    if n_malformed:
        print(f"[prune] dropped {n_malformed} malformed rows (coeff sum ≠ 0)")

    new_atoms = [atoms[c] for c in kept_cols]
    new_aidx = {str(a): i for i, a in enumerate(new_atoms)}
    return (
        np.array(new_data, dtype=np.int32),
        np.array(new_idx, dtype=np.int32),
        np.array(new_ptr, dtype=np.int32),
        len(kept_rows),
        len(kept_cols),
        new_atoms,
        new_aidx,
        kept_cols,
        kept_rows,
    )


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]

    def components(self, n: int) -> dict[int, list[int]]:
        comp: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            comp[self.find(i)].append(i)
        return dict(comp)


def build_atom_graph(data, indices, indptr, nrows, ncols) -> UnionFind:
    uf = UnionFind(ncols)
    for r in range(nrows):
        rs, re = int(indptr[r]), int(indptr[r + 1])
        cols = [int(indices[k]) for k in range(rs, re)]
        for i in range(1, len(cols)):
            uf.union(cols[0], cols[i])
    return uf


def resolve_xs_to_cols(aidx: dict[str, int], xs: list[int] | None) -> set[int]:
    cols: set[int] = set()
    if not xs:
        return cols
    for x in xs:
        c = aidx.get(str(int(x)))
        if c is not None:
            cols.add(int(c))
    return cols


# ---------------------------------------------------------------------------
# Rank over GF(p)
# ---------------------------------------------------------------------------

def rank_mod_p(data, indices, indptr, nrows, ncols, p: int, keep_cols: set[int] | None = None) -> int:
    if keep_cols is not None:
        col_remap = {c: i for i, c in enumerate(sorted(keep_cols))}
    else:
        col_remap = None

    rows: list[dict[int, int]] = []
    for r in range(nrows):
        rs, re = int(indptr[r]), int(indptr[r + 1])
        d: dict[int, int] = {}
        for k in range(rs, re):
            c = int(indices[k])
            v = int(data[k]) % p
            if v == 0:
                continue
            if col_remap is not None:
                if c not in col_remap:
                    continue
                c = col_remap[c]
            d[c] = (d.get(c, 0) + v) % p
        d = {k: v for k, v in d.items() if v}
        if d:
            rows.append(d)

    pivots: dict[int, dict[int, int]] = {}
    rank = 0
    for rd in rows:
        rd = dict(rd)
        for pc, prow in pivots.items():
            coeff = rd.get(pc, 0)
            if not coeff:
                continue
            for k, v in prow.items():
                rd[k] = (rd.get(k, 0) - coeff * v) % p
            rd = {k: v for k, v in rd.items() if v}
            if not rd:
                break
        if not rd:
            continue
        pc = min(rd)
        inv = pow(int(rd[pc]), p - 2, p)
        pivots[pc] = {k: (v * inv) % p for k, v in rd.items()}
        rank += 1

    return rank


# ---------------------------------------------------------------------------
# Solve augmented system A x = b mod p, where fixed_vars are substituted
# ---------------------------------------------------------------------------

def solve_mod_p(
    data,
    indices,
    indptr,
    nrows,
    ncols,
    p: int,
    keep_cols: set[int],
    fixed_vars: dict[int, int] | None = None,
):
    """Solve the linear system on a chosen column subset.

    The matrix is interpreted row-wise as a homogeneous relation
        sum_j a_ij x_j = 0 (mod p)
    but any variables listed in fixed_vars are substituted to the right-hand side.

    Returns a dictionary mapping original column indices to one particular
    solution on the kept columns.
    """
    if fixed_vars is None:
        fixed_vars = {}

    col_list = sorted(keep_cols - set(fixed_vars.keys()))
    col_remap = {c: i for i, c in enumerate(col_list)}
    n = len(col_list)

    rows: list[tuple[dict[int, int], int]] = []
    for r in range(nrows):
        rs, re = int(indptr[r]), int(indptr[r + 1])
        d: dict[int, int] = {}
        rhs = 0

        for k in range(rs, re):
            c_old = int(indices[k])
            v = int(data[k]) % p
            if v == 0:
                continue
            if c_old in fixed_vars:
                rhs = (rhs - v * (int(fixed_vars[c_old]) % p)) % p
                continue
            if c_old not in col_remap:
                continue
            c = col_remap[c_old]
            d[c] = (d.get(c, 0) + v) % p

        d = {k: v for k, v in d.items() if v}
        if d or rhs:
            rows.append((d, rhs))

    # Gaussian elimination on augmented rows.
    pivots: dict[int, tuple[dict[int, int], int]] = {}
    for rd, rhs in rows:
        rd = dict(rd)
        rhs = int(rhs) % p

        for pc in sorted(pivots.keys()):
            if pc not in rd:
                continue
            prow, prhs = pivots[pc]
            factor = rd[pc]
            for k, v in prow.items():
                rd[k] = (rd.get(k, 0) - factor * v) % p
            rhs = (rhs - factor * prhs) % p
            rd = {k: v for k, v in rd.items() if v}
            if not rd:
                break

        if not rd:
            if rhs % p != 0:
                raise ValueError("Inconsistent system (row reduces to 0 = nonzero)")
            continue

        pc = min(rd)
        inv = pow(int(rd[pc]), p - 2, p)
        rd = {k: (v * inv) % p for k, v in rd.items()}
        rhs = (rhs * inv) % p
        pivots[pc] = (rd, rhs)

    # Back substitution: choose free vars = 0 for one particular solution.
    x = [0] * n
    for pc in sorted(pivots.keys(), reverse=True):
        row, rhs = pivots[pc]
        s = 0
        for k, v in row.items():
            if k == pc:
                continue
            s = (s + v * x[k]) % p
        x[pc] = (rhs - s) % p

    return {old: x[new] for old, new in col_remap.items()} | {
        c: int(v) % p for c, v in fixed_vars.items() if c in keep_cols
    }


# ---------------------------------------------------------------------------
# Nullity-basis residual analysis
# ---------------------------------------------------------------------------

def nullity_residuals(
    pdata, pidx, pptr, pnrows, pncols,
    main_comp: set[int],
    special_cols: dict[str, int | None],
    p: int,
    top_k: int = 8,
):
    """Compute the full right-kernel of the pruned matrix, restrict each basis
    vector to main_comp columns, compute residuals M_main @ v mod p, and rank
    them by how much their weight concentrates on divisor/special atoms.

    Parameters
    ----------
    pdata, pidx, pptr : CSR arrays for the pruned matrix
    pnrows, pncols    : shape of the pruned matrix
    main_comp         : set of pruned column indices in the main component
    special_cols      : mapping label -> pruned col index (may have None values)
    p                 : field characteristic
    top_k             : how many top vectors to print
    """
    # ---- 1. Build GF(p) matrix as list-of-dicts (sparse rows), then dedup ----
    print(f"\n  Raw pruned matrix: {pnrows}×{pncols} over GF({p})")

    # Build sparse row dicts, reducing mod p
    rows_raw: list[dict[int, int]] = []
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        d: dict[int, int] = {}
        for k in range(rs, re):
            c = int(pidx[k])
            v = int(pdata[k]) % p
            if v:
                d[c] = (d.get(c, 0) + v) % p
        d = {k: v for k, v in d.items() if v}
        if d:
            rows_raw.append(d)

    # Row dedup: canonicalize each row by dividing by its leading coefficient,
    # then hash the sparse pattern. Keep one representative per equivalence class.
    seen_sigs: dict[tuple, bool] = {}
    rows: list[dict[int, int]] = []
    for d in rows_raw:
        lead = min(d.keys())
        inv_lead = pow(int(d[lead]), p - 2, p)
        sig = tuple(sorted((c, (v * inv_lead) % p) for c, v in d.items()))
        if sig not in seen_sigs:
            seen_sigs[sig] = True
            rows.append(d)

    # Determine actual column count from rows present
    all_cols: set[int] = set()
    for d in rows:
        all_cols.update(d.keys())
    actual_ncols = max(all_cols) + 1 if all_cols else 0

    print(f"  After row dedup : {len(rows)} rows, {len(all_cols)} distinct columns "
          f"(pncols={pncols}, max_col={actual_ncols - 1})")

    # Use actual_ncols as working dimension — pncols may differ from what
    # other tools compute due to different prune criteria.
    ncols = actual_ncols

    # ---- 2. Sparse GF(p) kernel basis via column reduction ----
    print(f"  Computing right kernel ({len(rows)} rows × {ncols} cols) ... ", end="", flush=True)

    # Column reduction of A over GF(p), tracking a right-multiplier.
    # col_vecs2[c] = sparse dict row->value for column c.
    # right2[c]    = sparse dict orig_col->coeff tracking column operations.
    # After reduction: if col_vecs2[c] is empty, right2[c] is a kernel vector.
    col_vecs2: list[dict[int, int]] = [{} for _ in range(ncols)]
    for r, rd in enumerate(rows):
        for c, v in rd.items():
            col_vecs2[c][r] = v

    right2: list[dict[int, int]] = [{c: 1} for c in range(ncols)]
    pivot_col2: dict[int, int] = {}
    pivot_vec2: dict[int, dict[int, int]] = {}
    pivot_right2: dict[int, dict[int, int]] = {}

    for c in range(ncols):
        col = dict(col_vecs2[c])
        rs_c = dict(right2[c])

        for row in sorted(pivot_col2.keys()):
            coeff = col.get(row, 0)
            if not coeff:
                continue
            piv = pivot_vec2[row]
            piv_rs = pivot_right2[row]
            inv_lead = pow(int(piv[row]), p - 2, p)
            factor = (coeff * inv_lead) % p
            for r2, v2 in piv.items():
                col[r2] = (col.get(r2, 0) - factor * v2) % p
            col = {k: v % p for k, v in col.items() if v % p}
            for k, v in piv_rs.items():
                rs_c[k] = (rs_c.get(k, 0) - factor * v) % p
            rs_c = {k: v % p for k, v in rs_c.items() if v % p}

        if not col:
            kernel_vecs.append(rs_c)
        else:
            prow = min(col.keys())
            pivot_col2[prow] = c
            pivot_vec2[prow] = col
            pivot_right2[prow] = rs_c

    print(f"done.  kernel dimension = {len(kernel_vecs)}")

    if not kernel_vecs:
        print("  [warn] kernel is trivial — nothing to score.")
        return

    # ---- 3. Restrict each kernel vector to main_comp ----
    # A kernel vector v has entries indexed by original column indices.
    # We zero out entries outside main_comp.
    main_list = sorted(main_comp)
    main_set  = set(main_comp)
    # Local re-index for M_main columns
    main_remap = {c: i for i, c in enumerate(main_list)}
    n_main = len(main_list)

    # ---- 4. Build M_main: pruned matrix restricted to main_comp columns ----
    # Rows that have ANY entry in main_comp are kept (all their main-comp entries).
    M_main_rows: list[dict[int, int]] = []
    for r, rd in enumerate(rows):
        d = {main_remap[c]: v for c, v in rd.items() if c in main_set}
        if d:
            M_main_rows.append(d)

    n_M = len(M_main_rows)
    print(f"  M_main: {n_M} rows × {n_main} cols")

    # ---- 5. Compute residuals r = M_main @ v_restricted (mod p) ----
    # v_restricted: length n_main, indexed by main_remap

    # Identify special column positions in main_remap
    sp_local: dict[str, int | None] = {}
    for label, orig_col in special_cols.items():
        if orig_col is not None and orig_col in main_remap:
            sp_local[label] = main_remap[orig_col]
        else:
            sp_local[label] = None

    sp_set = {v for v in sp_local.values() if v is not None}

    scored: list[tuple[float, int, list[int], dict[str, int]]] = []
    # scored entries: (score, basis_idx, residual_as_list, sp_values)

    for bi, kv in enumerate(kernel_vecs):
        # kv is sparse dict original_col -> value; restrict to main_comp
        v_main = [0] * n_main
        for orig_c, val in kv.items():
            if orig_c in main_remap:
                v_main[main_remap[orig_c]] = int(val) % p

        # Compute r = M_main @ v_main mod p
        r_vec = [0] * n_M
        for ri, rd in enumerate(M_main_rows):
            s = 0
            for ci, mv in rd.items():
                s += mv * v_main[ci]
            r_vec[ri] = s % p

        # Score: fraction of L1 weight on special columns
        total_l1 = sum(min(v, p - v) for v in r_vec if v)
        if total_l1 == 0:
            continue  # zero residual — skip (exact kernel of M_main too)

        sp_vals = {}
        sp_l1 = 0
        for label, lc in sp_local.items():
            if lc is not None:
                # find rows that correspond to special-atom rows
                # Actually r_vec is indexed by M_main rows, not columns.
                # We want: which entries of r_vec are "at" special atoms?
                # The residual r is a row-indexed vector; special atoms are columns.
                # The natural score is: |v_main[special_col]| / |v_main| total
                pass

        # Revised score: concentrate on v_main entries at special cols
        v_sp_l1 = sum(min(v_main[lc], p - v_main[lc]) for lc in sp_set if v_main[lc])
        v_total_l1 = sum(min(v, p - v) for v in v_main if v)
        if v_total_l1 == 0:
            continue

        score = v_sp_l1 / v_total_l1

        sp_vals = {label: v_main[lc] if lc is not None else None
                   for label, lc in sp_local.items()}

        # Also record residual L1
        scored.append((score, bi, r_vec, sp_vals, v_sp_l1, v_total_l1))

    if not scored:
        print("  [warn] all restricted kernel vectors lie in kernel of M_main too — no nonzero residuals.")
        return

    scored.sort(key=lambda t: -t[0])
    print(f"\n  {len(scored)} kernel vectors with nonzero M_main residual.")
    print(f"  Scoring: v_main weight on special cols / total v_main weight.\n")

    label_w = max(len(k) for k in sp_local) if sp_local else 4

    print(f"  {'rank':>4}  {'score':>8}  {'sp_l1':>7}  {'tot_l1':>7}  {'r_l1':>7}  "
          + "  ".join(f"{k:>{label_w}}" for k in sp_local))
    print("  " + "-" * (4 + 8 + 7 + 7 + 7 + 6 + (label_w + 2) * len(sp_local)))

    for rank_i, (score, bi, r_vec, sp_vals, v_sp_l1, v_total_l1) in enumerate(scored[:top_k]):
        r_l1 = sum(min(v, p - v) for v in r_vec if v)
        sp_str = "  ".join(
            f"{(str(sp_vals[k]) if sp_vals[k] is not None else 'N/A'):>{label_w}}"
            for k in sp_local
        )
        print(f"  {rank_i:>4}  {score:>8.4f}  {v_sp_l1:>7}  {v_total_l1:>7}  {r_l1:>7}  {sp_str}")

    # Detailed dump for top-3
    print(f"\n  --- Detailed top-{min(3, len(scored))} ---")
    for rank_i, (score, bi, r_vec, sp_vals, v_sp_l1, v_total_l1) in enumerate(scored[:3]):
        print(f"\n  [rank {rank_i}]  basis_vec={bi}  score={score:.4f}")
        # nonzero entries of v_main at special cols
        print(f"    Special-col values in v_main:")
        for label, lc in sp_local.items():
            val = sp_vals.get(label)
            sym = min(val, p - val) if val else 0
            print(f"      {label:>{label_w}}: col={lc}  raw={val}  sym={sym}")
        # top nonzero rows of residual
        nonzero_r = [(i, v) for i, v in enumerate(r_vec) if v]
        nonzero_r.sort(key=lambda t: -min(t[1], p - t[1]))
        print(f"    Residual r_vec: {len(nonzero_r)} nonzero rows (top 6 by magnitude):")
        for ri, rv in nonzero_r[:6]:
            sym = min(rv, p - rv)
            print(f"      row={ri:5d}  raw={rv:6d}  sym={sym:6d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Graph connectivity analysis for DLP relation matrix (HDF5).")
    ap.add_argument("matrix_file")
    ap.add_argument("--divisor-xs", type=int, nargs="+", default=None)
    ap.add_argument("--group-order", type=int, default=None)
    ap.add_argument("--show-isolated", action="store_true", help="Print isolated atom values")
    ap.add_argument("--component-detail", action="store_true", help="Print all atoms in divisor component")
    ap.add_argument("--rank-check", action="store_true", help="Compute rank of divisor-component submatrix (slow)")
    ap.add_argument("--no-prune", action="store_true", help="Skip dest-only pruning")
    ap.add_argument("--exclude-xs", type=int, nargs="*", default=None,
                    help="Explicit x-coordinates to remove as torsion/bad atoms")
    ap.add_argument("--exclude-ell-torsion", action="store_true",
                    help="Also exclude torsion atoms listed in the HDF5 file, if present")
    ap.add_argument("--nullity-residuals", action="store_true",
                    help="Compute full nullity basis, restrict to main component, score residuals by divisor-atom overlap")
    ap.add_argument("--top-k", type=int, default=8,
                    help="How many top-scoring residual vectors to print (default: 8)")
    args = ap.parse_args()

    SEP = "=" * 70

    def section(t):
        print(f"\n{SEP}\n{t}\n{SEP}")

    print("[load] reading HDF5 ...")
    h = load_hdf5(args.matrix_file)
    group_order = args.group_order or h["group_order"]
    divisor_xs = args.divisor_xs or h["divisor_xs"] or []

    print(f"[load] raw shape  : {h['nrows']} rows × {h['ncols']} cols")
    print(f"[load] nonzeros   : {len(h['data'])}")
    print(f"[load] group_order: {group_order}")
    print(f"[load] divisor_xs : {divisor_xs}")

    protected_cols: set[int] = set()
    for key in ("col_inf", "col_gen0", "col_gen1", "col_tgt0", "col_tgt1"):
        c = h[key]
        if c is not None:
            protected_cols.add(c)
    for x in divisor_xs:
        c = h["aidx"].get(str(int(x)))
        if c is not None:
            protected_cols.add(c)

    excluded_xs: list[int] = []
    if args.exclude_ell_torsion and h.get("ell_torsion_xs"):
        excluded_xs.extend(int(x) for x in h["ell_torsion_xs"])
    if args.exclude_xs:
        excluded_xs.extend(int(x) for x in args.exclude_xs)
    excluded_cols = resolve_xs_to_cols(h["aidx"], excluded_xs)
    excluded_cols -= protected_cols
    if excluded_cols:
        excluded_atoms = [h["atoms"][c] for c in sorted(excluded_cols)]
        preview = excluded_atoms[:12]
        print(f"[filter] excluding {len(excluded_cols)} torsion/bad atom(s): {preview}" +
              (" ..." if len(excluded_atoms) > 12 else ""))

    if args.no_prune:
        if excluded_cols:
            print("[warn] --no-prune ignores row/column elimination for excluded atoms; proceeding may reintroduce torsion contamination.")
        pdata, pidx, pptr = h["data"], h["indices"], h["indptr"]
        pnrows, pncols = h["nrows"], h["ncols"]
        patoms = h["atoms"]
        print("[prune] skipped")
    else:
        pdata, pidx, pptr, pnrows, pncols, patoms, _paidx, kept_cols, kept_rows = prune_dest_only(
            h["data"], h["indices"], h["indptr"], h["nrows"], h["ncols"], h["atoms"], protected_cols=protected_cols, excluded_cols=excluded_cols
        )
        print(f"[prune] removed {h['ncols'] - pncols} dest-only atoms, {h['nrows'] - pnrows} now-empty rows")
        print(f"[prune] pruned  : {pnrows} rows × {pncols} cols")

    section("CONNECTIVITY ANALYSIS")
    print(f"\n[graph] building co-occurrence graph on {pncols} atoms ...")
    uf = build_atom_graph(pdata, pidx, pptr, pnrows, pncols)
    comps = uf.components(pncols)
    by_size = sorted(((len(v), r) for r, v in comps.items()), reverse=True)
    print(f"[graph] {len(comps)} components, sizes (top 10): {[s for s, _ in by_size[:10]]}")

    pruned_x_to_col: dict[int, int] = {}
    for i, a in enumerate(patoms):
        try:
            pruned_x_to_col[int(a)] = i
        except (ValueError, TypeError):
            pass

    div_cols: list[tuple[int, int]] = []
    for x in divisor_xs:
        c = pruned_x_to_col.get(int(x))
        if c is None:
            if int(x) in excluded_xs:
                print(f"  [warn] divisor x={x} was excluded as torsion/bad and will not be used.")
            else:
                print(f"  [warn] divisor x={x} was pruned as dest-only — it never appeared as xi!")
        else:
            div_cols.append((int(x), c))

    print(f"\n  Divisor atoms surviving prune: {len(div_cols)} / {len(divisor_xs)}")
    div_roots: set[int] = set()
    for x, c in div_cols:
        root = uf.find(c)
        div_roots.add(root)
        print(f"    x={x:6d}  pruned_col={c:5d}  component_root={root:5d}  size={len(comps[root])}")

    if len(div_roots) == 1:
        print(f"\n  ✓  All divisor atoms are in ONE component.")
        main_root = next(iter(div_roots))
    elif len(div_roots) == 0:
        print("\n  ✗  No divisor atoms survived pruning!")
        sys.exit(1)
    else:
        print(f"\n  ✗  Divisor atoms span {len(div_roots)} different components!")
        main_root = max(div_roots, key=lambda r: len(comps[r]))

    main_comp: set[int] = set(comps[main_root])
    print(f"  Main component size: {len(main_comp)} atoms")

    section("ISOLATED ATOM ANALYSIS")
    isolated = [comps[r][0] for s, r in by_size if s == 1 and comps[r][0] not in main_comp]
    print(f"\n  Singletons outside divisor component: {len(isolated)}")
    print(f"  These are exactly the free kernel directions (nullity contributions).")

    overlap = set(isolated) & main_comp
    if overlap:
        print(f"  [BUG] {len(overlap)} singletons overlap with main component!")
    else:
        print(f"  ✓  Zero isolated atoms are reachable from the divisor component.")
        print(f"     Pinning them to 0 is safe and sufficient.")

    if args.show_isolated:
        print(f"\n  Isolated atoms (first 60):")
        for c in isolated[:60]:
            print(f"    pruned_col={c:5d}  atom={patoms[c]}")
        if len(isolated) > 60:
            print(f"    ... {len(isolated) - 60} more")

    other_nontrivial = [(s, r) for s, r in by_size if r != main_root and s > 1]
    if other_nontrivial:
        print(f"\n  Other non-trivial components: {len(other_nontrivial)}")
        for s, r in other_nontrivial[:6]:
            sample = [patoms[c] for c in comps[r][:4]]
            print(f"    size={s}  root={r}  sample={sample}")
    else:
        print(f"\n  ✓  No other non-trivial components.")

    section("ROW ANALYSIS")
    rows_in = rows_cross = rows_out = 0
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        cols = {int(pidx[k]) for k in range(rs, re)}
        in_m = cols & main_comp
        out_m = cols - main_comp
        if in_m and not out_m:
            rows_in += 1
        elif in_m:
            rows_cross += 1
        else:
            rows_out += 1

    n_atoms = len(main_comp)
    needed = n_atoms - 1
    deficit = max(0, needed - rows_in)

    print(f"\n  Total pruned rows           : {pnrows}")
    print(f"  Rows inside divisor comp    : {rows_in}")
    print(f"  Rows crossing boundary      : {rows_cross}")
    print(f"  Rows entirely outside       : {rows_out}")
    print(f"\n  Atoms in divisor component  : {n_atoms}")
    print(f"  Rank needed (atoms - 1)     : {needed}")
    print(f"  Row surplus/deficit         : {rows_in - needed:+d}  ({'ok' if deficit == 0 else f'need {deficit} more'})")

    if args.rank_check:
        if not group_order:
            print("\n  [rank-check] --group-order required. Skipping.")
        else:
            section("RANK CHECK  (divisor-component submatrix)")
            inf_pruned_col = None
            for _i, _a in enumerate(patoms):
                if str(_a) in ("∞", "inf", "\u221e"):
                    if _i in main_comp:
                        inf_pruned_col = _i
                    break

            print(f"\n  Eliminating over GF({group_order}) on {n_atoms}-col component submatrix ...")
            rank = rank_mod_p(pdata, pidx, pptr, pnrows, pncols, p=group_order, keep_cols=main_comp)
            nullity = n_atoms - rank
            print(f"\n  rank={rank}  atoms={n_atoms}  nullity={nullity}")

            if nullity == 1:
                print("  ✓  nullity=1 — FULLY DETERMINED up to gauge.")
                print("     DLP is solvable on this component.")
                print("     Pin isolated atoms to 0 and call solve_right.")
            elif nullity == 0:
                if inf_pruned_col is not None:
                    comp_no_inf = main_comp - {inf_pruned_col}
                    rank_no_inf = rank_mod_p(pdata, pidx, pptr, pnrows, pncols, p=group_order, keep_cols=comp_no_inf)
                    null_no_inf = len(comp_no_inf) - rank_no_inf
                    print(f"  Without-∞: rank={rank_no_inf}  atoms={len(comp_no_inf)}  nullity={null_no_inf}")
                    if null_no_inf == 0:
                        print("  ✓  Gauge (∞) is organically pinned by the walk data.")
                        print("     The non-∞ subspace is fully determined.")
                        print("     DLP is solvable: add anchor and call solve_right.")

                        # Translate original HDF5 column indices → pruned indices.
                        orig_to_pruned: dict[int, int] = {}
                        if not args.no_prune:
                            orig_to_pruned = {orig: pruned for pruned, orig in enumerate(kept_cols)}

                        def pruned_col(key: str) -> int | None:
                            orig = h.get(key)
                            if orig is None:
                                return None
                            return orig if args.no_prune else orig_to_pruned.get(orig)

                        pc_inf  = inf_pruned_col  # already found by patoms scan above
                        pc_gen0 = pruned_col("col_gen0")
                        pc_gen1 = pruned_col("col_gen1")
                        pc_tgt0 = pruned_col("col_tgt0")
                        pc_tgt1 = pruned_col("col_tgt1")

                        print(f"\n  [diag] pruned cols — inf={pc_inf}  gen0={pc_gen0}"
                              f"  gen1={pc_gen1}  tgt0={pc_tgt0}  tgt1={pc_tgt1}")

                        if pc_inf is None:
                            print("\n  [warn] ∞ not found in pruned matrix — cannot solve.")
                        else:
                            # Pin ∞=1. Its coefficient is -5 in every row, so each row
                            # gets rhs += 5, breaking homogeneity and yielding the unique
                            # solution via back-substitution.
                            fixed_vars = {pc_inf: 1}
                            print("\n  [solve] attempting to recover logs...")
                            try:
                                sol = solve_mod_p(
                                    pdata, pidx, pptr,
                                    pnrows, pncols,
                                    p=group_order,
                                    keep_cols=main_comp,
                                    fixed_vars=fixed_vars,
                                )
                                print(f"  [solve] succeeded  gauge={fixed_vars}")
                                print("\n  --- recovered logs (sample) ---")
                                for key, pc in (("col_inf",  pc_inf),
                                                ("col_gen0", pc_gen0),
                                                ("col_gen1", pc_gen1),
                                                ("col_tgt0", pc_tgt0),
                                                ("col_tgt1", pc_tgt1)):
                                    if pc is None:
                                        continue
                                    print(f"    {key} (pruned_col={pc}): log = {sol.get(pc)}")

                                if pc_gen0 is not None and pc_tgt0 is not None:
                                    g = sol.get(pc_gen0, 0)
                                    t = sol.get(pc_tgt0, 0)
                                    if g != 0:
                                        k = (t * pow(int(g), group_order - 2, group_order)) % group_order
                                        print(f"\n  >>> recovered discrete log k ≡ {k} mod {group_order}")
                                    else:
                                        print("\n  [warn] generator log is 0 after solve")
                            except ValueError as e:
                                print(f"\n  [solve] failed: {e}")
                    else:
                        print(f"  ✗  Genuinely underdetermined even without ∞ (nullity={null_no_inf}).")
                        print(f"     Need {null_no_inf} more independent rows.")
                else:
                    print("  ?  nullity=0 and ∞ not in component — unexpected.")
            else:
                print(f"  ✗  nullity={nullity} — need {nullity - 1} more independent in-component rows.")

    if args.component_detail:
        section("DIVISOR COMPONENT ATOMS")
        for c in sorted(main_comp):
            print(f"  col={c:5d}  atom={patoms[c]}")

    section("VERDICT")
    ok = len(div_roots) == 1 and len(overlap) == 0
    print(
        f"""
  Divisor component size : {n_atoms}
  Rows in component      : {rows_in}
  Isolated free atoms    : {len(isolated)}  ← safe to pin to 0
  Other non-trivial comps: {len(other_nontrivial)}
  All divisors colocated : {'YES' if len(div_roots) == 1 else 'NO'}
  Isolated ∩ divisor comp: {len(overlap)}  (want 0)
  Row deficit            : {deficit}
"""
    )
    if ok and deficit == 0:
        print("  ✓  STRUCTURAL VERDICT: solvable pending rank check.")
        print("     Run with --rank-check to confirm nullity=1 in the component.")
    elif ok and deficit > 0:
        print(f"  ✗  Need {deficit} more rows inside the divisor component.")
        print("     Best strategy: more walker steps seeded from atoms already")
        print("     in the component. Involution-closure rows are efficient.")
    else:
        print("  ✗  Structural problem — see warnings above.")
    print()

    if args.nullity_residuals:
        if not group_order:
            print("\n[nullity-residuals] --group-order required. Skipping.")
        else:
            section("NULLITY-BASIS RESIDUAL ANALYSIS")

            # Build pruned-col indices for special columns.
            # kept_cols maps pruned_col -> original_col; invert it.
            if args.no_prune:
                orig_to_pruned = {c: c for c in range(h["ncols"])}
            else:
                orig_to_pruned = {orig: pruned for pruned, orig in enumerate(kept_cols)}

            def _pc(key):
                orig = h.get(key)
                if orig is None:
                    return None
                return orig_to_pruned.get(orig)

            special_cols = {
                "inf":  _pc("col_inf"),
                "gen0": _pc("col_gen0"),
                "gen1": _pc("col_gen1"),
                "tgt0": _pc("col_tgt0"),
                "tgt1": _pc("col_tgt1"),
            }
            # Remove entries with None pruned col
            special_cols = {k: v for k, v in special_cols.items() if v is not None}

            nullity_residuals(
                pdata, pidx, pptr, pnrows, pncols,
                main_comp=main_comp,
                special_cols=special_cols,
                p=group_order,
                top_k=args.top_k,
            )


if __name__ == "__main__":
    main()
