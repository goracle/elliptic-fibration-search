#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any
from relation_matrix import *

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
    """Compute the full right-kernel of the pruned matrix via Sage, then report
    the structure of each basis vector.

    The workflow matches dlp_contradiction_diag.py:

      1. Reconstruct a Sage integer matrix from the CSR data.
      2. Deduplicate rows mod p (collapse scalar multiples).
      3. Call A.change_ring(GF(p)).right_kernel() — no handrolled elimination.
      4. Classify each basis vector (gauge, isolated, fusion, parity, other)
         and report special-column values.
    """
    from sage.all import GF, ZZ, Integer, matrix as sage_matrix, vector as sage_vector, Matrix

    Fp = GF(Integer(p))

    # ---- 1. Reconstruct Sage integer matrix from CSR ----
    print(f"\n  Raw pruned matrix: {pnrows}×{pncols} over GF({p})")
    entries = {}
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        for k in range(rs, re):
            entries[(r, int(pidx[k]))] = int(pdata[k])
    M_ZZ = Matrix(ZZ, pnrows, pncols, entries)

    # ---- 2. Deduplicate rows mod p (collapse scalar multiples) ----
    # Mirrors _dedupe_rows_mod from dlp_contradiction_diag.py.
    seen: dict = {}
    dedup_rows = []
    for i in range(pnrows):
        row_entries = sorted(
            (j, int(M_ZZ[i, j]) % p)
            for j in range(pncols)
            if int(M_ZZ[i, j]) % p != 0
        )
        if not row_entries:
            continue
        lead = row_entries[0][1]
        inv_lead = pow(lead, -1, p)
        sig = tuple((j, (v * inv_lead) % p) for j, v in row_entries)
        if sig not in seen:
            seen[sig] = len(dedup_rows)
            row = {j: int(v) for j, v in sig}
            dedup_rows.append(row)

    if dedup_rows:
        sparse_entries = {}
        for i, row_d in enumerate(dedup_rows):
            for j, v in row_d.items():
                sparse_entries[(i, j)] = v
        M_dedup = Matrix(ZZ, len(dedup_rows), pncols, sparse_entries)
    else:
        M_dedup = Matrix(ZZ, 0, pncols)

    n_dedup = pnrows - M_dedup.nrows()
    print(f"  After row dedup : {M_dedup.nrows()} rows ({n_dedup} duplicate/scalar-multiple rows removed)")

    # ---- 3. Compute right kernel via Sage ----
    print(f"  Computing right kernel ({M_dedup.nrows()} rows × {pncols} cols) ... ", end="", flush=True)
    A = M_dedup.change_ring(Fp)
    ker = A.right_kernel()
    nullity = ker.dimension()
    print(f"done.  kernel dimension = {nullity}")

    if nullity == 0:
        print("  [warn] kernel is trivial.")
        return

    # ---- 4. Classify each basis vector (matches _extract_pin_rows pattern) ----
    # Identify the inf column in the pruned space (special_cols uses original
    # pruned indices, which are the same column indices we're working with here).
    p_col_inf = special_cols.get("inf")

    counts = {"gauge": 0, "isolated": 0, "fusion": 0, "parity": 0, "other": 0}
    previews = []
    max_preview = top_k

    for vi, vec in enumerate(ker.basis()):
        support = [(j, int(vec[j])) for j in range(pncols) if vec[j] != Fp(0)]
        if not support:
            continue

        if len(support) == 1 and p_col_inf is not None and support[0][0] == p_col_inf:
            counts["gauge"] += 1
            msg = f"kernel[{vi}]: GAUGE (inf col={p_col_inf})"
        elif len(support) == 1:
            counts["isolated"] += 1
            j, coeff = support[0]
            msg = f"kernel[{vi}]: ISOLATED  col={j}  coeff={coeff}"
        elif len(support) == 2:
            (j0, c0), (j1, c1) = support
            if (c0 == 1 and c1 == p - 1) or (c0 == p - 1 and c1 == 1):
                counts["fusion"] += 1
                msg = f"kernel[{vi}]: FUSION  col={j0} = col={j1}"
            else:
                counts["other"] += 1
                msg = f"kernel[{vi}]: OTHER  support_size=2  coeffs={sorted(c for _, c in support)}"
        else:
            coeffs_vals = [c for _, c in support]
            if len(set(coeffs_vals)) == 1:
                counts["parity"] += 1
                msg = f"kernel[{vi}]: PARITY/CONSERVATION  support_size={len(support)}  coeff={coeffs_vals[0]}"
            else:
                counts["other"] += 1
                msg = f"kernel[{vi}]: OTHER  support_size={len(support)}  distinct_coeffs={sorted(set(coeffs_vals))}"

        # Annotate with special-column values.
        sp_vals = {label: int(vec[col]) for label, col in special_cols.items()
                   if col is not None and vec[col] != Fp(0)}
        if sp_vals:
            msg += f"  specials={sp_vals}"

        if len(previews) < max_preview:
            previews.append(msg)

    print(f"\n  kernel summary: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for msg in previews:
        print(f"  {msg}")
    omitted = nullity - len(previews)
    if omitted > 0:
        print(f"  ... {omitted} more kernel direction(s) omitted")

    # ---- 5. Restrict matrix and kernel vectors to main_comp ----
    main_list = sorted(main_comp)
    main_idx  = {c: i for i, c in enumerate(main_list)}
    n_main    = len(main_list)

    # Rows of M_dedup restricted to main_comp columns; track which touch gen/tgt.
    gen_tgt_cols  = {c for lbl, c in special_cols.items()
                     if lbl in ("gen0", "gen1", "tgt0", "tgt1") and c in main_idx}
    gen_tgt_local = {main_idx[c] for c in gen_tgt_cols}

    M_main_rows        = []
    row_touches_gen_tgt = []
    for row_d in dedup_rows:
        restricted = {main_idx[j]: v for j, v in row_d.items() if j in main_idx}
        if restricted:
            M_main_rows.append(restricted)
            row_touches_gen_tgt.append(bool(restricted.keys() & gen_tgt_local))

    n_M = len(M_main_rows)
    print(f"\n  M_main: {n_M} rows x {n_main} cols  "
          f"({sum(row_touches_gen_tgt)} rows touch gen/tgt cols)")

    sp_local = {lbl: main_idx[c] for lbl, c in special_cols.items() if c in main_idx}

    def sym(v: int) -> int:
        return min(v, p - v) if v else 0

    # ---- 6. Compute residuals r = M_main @ v_main and score ----
    scored = []  # (score, vi, v_main, r_vec, sp_vals, r_gen_tgt, r_total)

    for vi, vec in enumerate(ker.basis()):
        v_main = [int(vec[c]) % p for c in main_list]

        r_vec = [
            sum(mv * v_main[lc] for lc, mv in row_d.items()) % p
            for row_d in M_main_rows
        ]

        r_sym = [sym(v) for v in r_vec]
        r_total = sum(r_sym)
        if r_total == 0:
            continue  # v_main in kernel of M_main, skip

        r_gen_tgt = sum(r_sym[i] for i, hit in enumerate(row_touches_gen_tgt) if hit)
        score     = r_gen_tgt / r_total

        sp_vals = {lbl: int(vec[c]) % p for lbl, c in special_cols.items() if c in main_idx}
        scored.append((score, vi, v_main, r_vec, sp_vals, r_gen_tgt, r_total))

    if not scored:
        print("  All restricted kernel vectors are in the kernel of M_main -- no residuals.")
        return

    scored.sort(key=lambda t: (-t[0], -t[5], -t[6]))
    print(f"  {len(scored)} kernel vector(s) with nonzero M_main residual.\n")

    # ---- 7. Print top-k ----
    sp_labels = [lbl for lbl in ("inf", "gen0", "gen1", "tgt0", "tgt1") if lbl in sp_local]
    header_sp = "  ".join(f"{lbl:>6}" for lbl in sp_labels)
    print(f"  {'rank':>4}  {'score':>7}  {'r_gt':>7}  {'r_tot':>7}  {header_sp}")
    print("  " + "-" * (30 + 8 * len(sp_labels)))

    for rank_i, (score, vi, v_main, r_vec, sp_vals, r_gen_tgt, r_total) in enumerate(scored[:top_k]):
        sp_str = "  ".join(f"{sym(sp_vals.get(lbl, 0)):>6}" for lbl in sp_labels)
        print(f"  {rank_i:>4}  {score:>7.4f}  {r_gen_tgt:>7}  {r_total:>7}  {sp_str}  basis[{vi}]")

    print(f"\n  --- Detailed top-{min(3, len(scored))} ---")
    for rank_i, (score, vi, v_main, r_vec, sp_vals, r_gen_tgt, r_total) in enumerate(scored[:3]):
        print(f"\n  [rank {rank_i}]  basis[{vi}]  score={score:.4f}"
              f"  r_gen_tgt={r_gen_tgt}  r_total={r_total}")
        print(f"  Special-col values (sym):")
        for lbl in sp_labels:
            raw = sp_vals.get(lbl, 0)
            print(f"    {lbl:>6}: raw={raw:>6}  sym={sym(raw):>6}")
        nonzero_r = sorted(((i, v) for i, v in enumerate(r_vec) if v),
                           key=lambda t: -sym(t[1]))
        print(f"  Residual: {len(nonzero_r)} nonzero rows (top 8 by sym magnitude):")
        for ri, rv in nonzero_r[:8]:
            tag = " <- gen/tgt" if row_touches_gen_tgt[ri] else ""
            print(f"    row={ri:5d}  raw={rv:>6}  sym={sym(rv):>6}{tag}")


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
        raw_data, raw_idx, raw_ptr = h["data"], h["indices"], h["indptr"]
        raw_nrows, raw_ncols = h["nrows"], h["ncols"]
        cur_atoms = list(h["atoms"])

        # Step 1: drop rows touching excluded_cols and remove those columns.
        # Must happen before the malformed-row filter so both tools see the
        # same set of rows when computing coeff sums.
        if excluded_cols:
            kept_data2, kept_idx2, kept_ptr2 = [], [], [0]
            n_excl_rows = 0
            for r in range(raw_nrows):
                rs, re = int(raw_ptr[r]), int(raw_ptr[r + 1])
                if any(int(raw_idx[k]) in excluded_cols for k in range(rs, re)):
                    n_excl_rows += 1
                    continue
                for k in range(rs, re):
                    kept_data2.append(int(raw_data[k]))
                    kept_idx2.append(int(raw_idx[k]))
                kept_ptr2.append(len(kept_data2))
            keep_cols_list = [c for c in range(raw_ncols) if c not in excluded_cols]
            col_remap_excl = {old: new for new, old in enumerate(keep_cols_list)}
            raw_data  = np.array(kept_data2, dtype=np.int32)
            raw_idx   = np.array([col_remap_excl[c] for c in kept_idx2], dtype=np.int32)
            raw_ptr   = np.array(kept_ptr2, dtype=np.int32)
            raw_nrows = raw_nrows - n_excl_rows
            raw_ncols = len(keep_cols_list)
            cur_atoms = [cur_atoms[c] for c in keep_cols_list]
            protected_cols = {col_remap_excl[c] for c in protected_cols if c in col_remap_excl}
            print(f"[filter] dropped {n_excl_rows} row(s) touching excluded cols, "
                  f"{h['ncols'] - raw_ncols} col(s) removed")

        # Step 2: prune dest-only atoms (excluded_cols already removed above).
        # Uses prune_dest_only from relation_matrix.py which takes a SageMath Matrix.
        # Convert CSR → dense SageMath Matrix, call, then convert back to CSR.
        from sage.matrix.constructor import Matrix
        from sage.rings.integer_ring import ZZ
        entries = {}
        for r in range(raw_nrows):
            rs, re = int(raw_ptr[r]), int(raw_ptr[r + 1])
            for k in range(rs, re):
                entries[(r, int(raw_idx[k]))] = int(raw_data[k])
        raw_mat = Matrix(ZZ, raw_nrows, raw_ncols, entries)

        # protected_cols holds integer column indices; pass as a list of atom values.
        protected_atoms = [cur_atoms[c] for c in protected_cols] if protected_cols else None
        pruned_mat, pruned_atoms, _removed = prune_dest_only(
            raw_mat, cur_atoms, protected=protected_atoms
        )

        pnrows, pncols = pruned_mat.nrows(), pruned_mat.ncols()
        patoms = pruned_atoms

        # Derive kept_cols: indices into cur_atoms that survived (matched by identity/value).
        atom_to_orig = {str(a): i for i, a in enumerate(cur_atoms)}
        kept_cols = [atom_to_orig[str(a)] for a in pruned_atoms]

        # Convert pruned SageMath Matrix → CSR numpy arrays.
        p_entries: dict[int, list] = {}
        _pdata_list: list[int] = []
        _pidx_list: list[int] = []
        _pptr_list: list[int] = [0]
        for r in range(pnrows):
            for c in range(pncols):
                v = int(pruned_mat[r, c])
                if v:
                    _pdata_list.append(v)
                    _pidx_list.append(c)
            _pptr_list.append(len(_pdata_list))
        pdata = np.array(_pdata_list, dtype=np.int32)
        pidx  = np.array(_pidx_list,  dtype=np.int32)
        pptr  = np.array(_pptr_list,  dtype=np.int32)

        print(f"[prune] removed {raw_ncols - pncols} dest-only atoms, {raw_nrows - pnrows} now-empty rows")
        print(f"[prune] pruned  : {pnrows} rows x {pncols} cols")

    # Drop malformed rows: a valid principal-divisor relation must have integer
    # coefficients summing to zero.  Rows that fail this are arithmetically
    # unsound (typically because a dest-only atom was pruned away after the
    # relation was written, leaving its column absent from the matrix).
    _malformed = []
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        if sum(int(pdata[k]) for k in range(rs, re)) != 0:
            _malformed.append(r)
    if _malformed:
        print(f"[filter] dropping {len(_malformed)} malformed row(s) (coeff sum != 0): "
              f"{_malformed[:16]}" + (" ..." if len(_malformed) > 16 else ""))
        _keep_rows = [r for r in range(pnrows) if r not in set(_malformed)]
        _new_data, _new_idx, _new_ptr = [], [], [0]
        for r in _keep_rows:
            rs, re = int(pptr[r]), int(pptr[r + 1])
            for k in range(rs, re):
                _new_data.append(int(pdata[k]))
                _new_idx.append(int(pidx[k]))
            _new_ptr.append(len(_new_data))
        pdata  = np.array(_new_data, dtype=np.int32)
        pidx   = np.array(_new_idx,  dtype=np.int32)
        pptr   = np.array(_new_ptr,  dtype=np.int32)
        pnrows = len(_keep_rows)
        print(f"[filter] matrix after malformed-row drop: {pnrows} rows x {pncols} cols")
    else:
        print("[filter] all rows well-formed (coeff sums all zero).")

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
                print("     Pinning a[gen0]=1 and solving...")

                if True:
                    orig_to_pruned_n1: dict[int, int] = {}
                    if not args.no_prune:
                        orig_to_pruned_n1 = {orig: pruned for pruned, orig in enumerate(kept_cols)}
                    _excl_remap_n1 = col_remap_excl if excluded_cols else {}

                    def _pc_n1(key):
                        orig = h.get(key)
                        if orig is None:
                            return None
                        if _excl_remap_n1:
                            orig = _excl_remap_n1.get(orig)
                            if orig is None:
                                return None
                        return orig_to_pruned_n1.get(orig)

                    pc_inf_n1  = inf_pruned_col
                    pc_gen0_n1 = _pc_n1("col_gen0")
                    pc_gen1_n1 = _pc_n1("col_gen1")
                    pc_tgt0_n1 = _pc_n1("col_tgt0")
                    pc_tgt1_n1 = _pc_n1("col_tgt1")

                    print(f"  [solve] pruned cols — inf={pc_inf_n1}  gen0={pc_gen0_n1}"
                          f"  gen1={pc_gen1_n1}  tgt0={pc_tgt0_n1}  tgt1={pc_tgt1_n1}")

                    if pc_inf_n1 is None:
                        raise ValueError("inf column not found — cannot pin gauge")
                    fixed_vars_n1 = {pc_inf_n1: 0, pc_gen0_n1: 1}
                    try:
                        sol_n1 = solve_mod_p(
                            pdata, pidx, pptr,
                            pnrows, pncols,
                            p=group_order,
                            keep_cols=main_comp,
                            fixed_vars=fixed_vars_n1,
                        )
                        print(f"  [solve] succeeded  gauge=a[inf]=0")
                        print("\n  --- recovered logs ---")
                        for key, pc in (("col_inf",  pc_inf_n1),
                                        ("col_gen0", pc_gen0_n1),
                                        ("col_gen1", pc_gen1_n1),
                                        ("col_tgt0", pc_tgt0_n1),
                                        ("col_tgt1", pc_tgt1_n1)):
                            if pc is None:
                                continue
                            print(f"    {key} (pruned_col={pc}): log = {sol_n1.get(pc)}")

                        if pc_gen0_n1 is not None and pc_tgt0_n1 is not None:
                            g = sol_n1.get(pc_gen0_n1, 0)
                            t = sol_n1.get(pc_tgt0_n1, 0)
                            if g != 0:
                                k = (t * pow(int(g), group_order - 2, group_order)) % group_order
                                print(f"\n  >>> recovered discrete log k ≡ {k} mod {group_order}")
                            else:
                                print("\n  [warn] generator log is 0 after solve — gauge may be wrong")
                    except ValueError as e:
                        print(f"\n  [solve] failed: {e}")
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
                        _excl_remap2 = col_remap_excl if excluded_cols else {}

                        def pruned_col(key: str) -> int | None:
                            orig = h.get(key)
                            if orig is None:
                                return None
                            if args.no_prune:
                                return orig
                            if _excl_remap2:
                                orig = _excl_remap2.get(orig)
                                if orig is None:
                                    return None
                            return orig_to_pruned.get(orig)

                        pc_inf  = inf_pruned_col  # already found by patoms scan above
                        pc_gen0 = pruned_col("col_gen0")
                        pc_gen1 = pruned_col("col_gen1")
                        pc_tgt0 = pruned_col("col_tgt0")
                        pc_tgt1 = pruned_col("col_tgt1")

                        print(f"\n  [diag] pruned cols — inf={pc_inf}  gen0={pc_gen0}"
                              f"  gen1={pc_gen1}  tgt0={pc_tgt0}  tgt1={pc_tgt1}")

                        if pc_inf is None:
                            print("\n  [warn] inf not found in pruned matrix — cannot solve.")
                        else:
                            fixed_vars = {pc_inf: 0, pc_gen0: 1}
                            print("\n  [solve] attempting to recover logs...")
                            try:
                                sol = solve_mod_p(
                                    pdata, pidx, pptr,
                                    pnrows, pncols,
                                    p=group_order,
                                    keep_cols=main_comp,
                                    fixed_vars=fixed_vars,
                                )
                                print(f"  [solve] succeeded  gauge=a[inf]=0")
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
            # kept_cols is indexed into the post-exclusion column space.
            # h["col_*"] are original HDF5 indices.  Compose through
            # col_remap_excl (if any exclusions were applied) then kept_cols.
            if args.no_prune:
                orig_to_pruned = {c: c for c in range(h["ncols"])}
                _excl_remap = {}
            else:
                orig_to_pruned = {orig: pruned for pruned, orig in enumerate(kept_cols)}
                # col_remap_excl maps original HDF5 col -> post-exclusion col;
                # it was only built when excluded_cols was non-empty.
                _excl_remap = col_remap_excl if excluded_cols else {}

            def _pc(key):
                orig = h.get(key)
                if orig is None:
                    return None
                # Map original HDF5 col through exclusion remap first.
                if _excl_remap:
                    orig = _excl_remap.get(orig)
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
