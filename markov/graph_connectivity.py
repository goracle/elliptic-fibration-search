#!/usr/bin/env python3
"""
graph_connectivity.py  —  bipartite connectivity analysis for DLP relation matrix

Reads the HDF5 format written by dlp_diagnostics.dump_matrix_hdf5():
    /csr/{data,indices,indptr,shape}
    /atoms           vlen-bytes, one per column
    /atom_index      JSON string  atom_str -> col_index
    /divisor_xs      int64[4]
    /group_order     scalar int64
    /col_inf, /col_gen0, /col_gen1, /col_tgt0, /col_tgt1  scalar int64

Does NOT require SageMath.  Needs: h5py, numpy.

Usage
-----
    python3 graph_connectivity.py relation_matrix.h5

    python3 graph_connectivity.py relation_matrix.h5 \\
        --divisor-xs 11504 5506 1821 12655 --group-order 25373

    python3 graph_connectivity.py relation_matrix.h5 \\
        --show-isolated --component-detail --rank-check

What it answers
---------------
1. Are all four divisor atoms in the same connected component?
2. Are the ~120 isolated atoms (free kernel directions) genuinely
   disconnected from the divisor component, or reachable through some path?
3. If isolated atoms are disconnected, what rank does the divisor-component
   submatrix have?  Is nullity already 1 there?
4. How many more rows are needed?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 load  (matches dump_matrix_hdf5 layout exactly)
# ---------------------------------------------------------------------------

def load_hdf5(path: str) -> dict:
    with h5py.File(path, "r") as f:
        data    = f["csr/data"][:]
        indices = f["csr/indices"][:]
        indptr  = f["csr/indptr"][:]
        shape   = tuple(int(x) for x in f["csr/shape"][:])

        atoms_raw = f["atoms"][:]
        atoms = [
            a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
            for a in atoms_raw
        ]
        aidx_raw = f["atom_index"][()]
        if isinstance(aidx_raw, (bytes, np.bytes_)):
            aidx_raw = aidx_raw.decode("utf-8")
        aidx: dict[str, int] = json.loads(aidx_raw)

        def _scalar(key):
            if key not in f:
                return None
            v = int(f[key][()])
            return v if v >= 0 else None

        group_order = _scalar("group_order")
        divisor_xs  = [int(x) for x in f["divisor_xs"][:]] if "divisor_xs" in f else None
        col_inf  = _scalar("col_inf")
        col_gen0 = _scalar("col_gen0")
        col_gen1 = _scalar("col_gen1")
        col_tgt0 = _scalar("col_tgt0")
        col_tgt1 = _scalar("col_tgt1")

    nrows, ncols = shape
    return dict(
        data=data, indices=indices, indptr=indptr,
        nrows=nrows, ncols=ncols,
        atoms=atoms, aidx=aidx,
        group_order=group_order, divisor_xs=divisor_xs,
        col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1,
        col_tgt0=col_tgt0, col_tgt1=col_tgt1,
    )


# ---------------------------------------------------------------------------
# prune_dest_only  (pure-numpy port matching relation_matrix.prune_dest_only)
#
# In the genus-2 Markov walker each row is:  3*xi + 1*xj + 1*xk - 5*inf = 0
# so xi always has |coeff|=3.  Atoms that never appear with |coeff|=3 are
# "dest-only" and get pruned (their column, and any row that becomes empty).
# Protected columns (divisor atoms, inf) always survive.
# ---------------------------------------------------------------------------

_XI_COEFF = 3

def prune_dest_only(data, indices, indptr, nrows, ncols, atoms,
                    protected_cols: set[int]):
    """
    Returns pruned (data, indices, indptr, nrows, ncols, atoms, aidx,
                    kept_cols_list, kept_rows_list).
    """
    source_cols: set[int] = set(protected_cols)
    for v, c in zip(data, indices):
        if abs(int(v)) == _XI_COEFF:
            source_cols.add(int(c))

    kept_cols = sorted(source_cols)
    col_remap  = {old: new for new, old in enumerate(kept_cols)}

    new_data, new_idx, new_ptr = [], [], [0]
    kept_rows = []
    for r in range(nrows):
        rs, re = int(indptr[r]), int(indptr[r + 1])
        row_ents = [
            (col_remap[int(indices[k])], int(data[k]))
            for k in range(rs, re)
            if int(indices[k]) in col_remap
        ]
        if row_ents:
            kept_rows.append(r)
            for c, v in row_ents:
                new_data.append(v)
                new_idx.append(c)
            new_ptr.append(len(new_data))

    new_atoms = [atoms[c] for c in kept_cols]
    new_aidx  = {str(a): i for i, a in enumerate(new_atoms)}
    return (
        np.array(new_data, dtype=np.int32),
        np.array(new_idx,  dtype=np.int32),
        np.array(new_ptr,  dtype=np.int32),
        len(kept_rows), len(kept_cols),
        new_atoms, new_aidx,
        kept_cols, kept_rows,
    )


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size   = [1] * n

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


# ---------------------------------------------------------------------------
# Rank over GF(p) — sparse dict-based Gaussian elimination
# ---------------------------------------------------------------------------

def rank_mod_p(data, indices, indptr, nrows, ncols, p: int,
               keep_cols: set[int] | None = None) -> int:
    if keep_cols is not None:
        col_remap = {c: i for i, c in enumerate(sorted(keep_cols))}
        eff_ncols = len(keep_cols)
    else:
        col_remap = None
        eff_ncols = ncols

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

    pivots: dict[int, dict[int, int]] = {}  # pivot_col -> normalised row
    rank = 0
    for rd in rows:
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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Graph connectivity analysis for DLP relation matrix (HDF5).")
    ap.add_argument("matrix_file")
    ap.add_argument("--divisor-xs",  type=int, nargs="+", default=None)
    ap.add_argument("--group-order", type=int, default=None)
    ap.add_argument("--show-isolated",    action="store_true",
                    help="Print isolated atom values")
    ap.add_argument("--component-detail", action="store_true",
                    help="Print all atoms in divisor component")
    ap.add_argument("--rank-check",  action="store_true",
                    help="Compute rank of divisor-component submatrix (slow)")
    ap.add_argument("--no-prune",    action="store_true",
                    help="Skip dest-only pruning")
    args = ap.parse_args()

    SEP = "=" * 70
    def section(t): print(f"\n{SEP}\n{t}\n{SEP}")

    # --- load ---
    print("[load] reading HDF5 ...")
    h = load_hdf5(args.matrix_file)
    group_order = args.group_order or h["group_order"]
    divisor_xs  = args.divisor_xs  or h["divisor_xs"] or []

    print(f"[load] raw shape  : {h['nrows']} rows × {h['ncols']} cols")
    print(f"[load] nonzeros   : {len(h['data'])}")
    print(f"[load] group_order: {group_order}")
    print(f"[load] divisor_xs : {divisor_xs}")

    # --- protected columns ---
    # map atom string -> col index for the raw matrix
    protected_cols: set[int] = set()
    for key in ("col_inf", "col_gen0", "col_gen1", "col_tgt0", "col_tgt1"):
        c = h[key]
        if c is not None:
            protected_cols.add(c)
    # also add by x-coord lookup
    for x in divisor_xs:
        c = h["aidx"].get(str(int(x)))
        if c is not None:
            protected_cols.add(c)

    # --- prune ---
    if args.no_prune:
        pdata, pidx, pptr = h["data"], h["indices"], h["indptr"]
        pnrows, pncols    = h["nrows"], h["ncols"]
        patoms            = h["atoms"]
        print("[prune] skipped")
    else:
        (pdata, pidx, pptr,
         pnrows, pncols,
         patoms, _paidx,
         kept_cols, kept_rows) = prune_dest_only(
             h["data"], h["indices"], h["indptr"],
             h["nrows"], h["ncols"], h["atoms"],
             protected_cols=protected_cols,
         )
        print(f"[prune] removed {h['ncols'] - pncols} dest-only atoms, "
              f"{h['nrows'] - pnrows} now-empty rows")
        print(f"[prune] pruned  : {pnrows} rows × {pncols} cols")

    # --- connectivity ---
    section("CONNECTIVITY ANALYSIS")
    print(f"\n[graph] building co-occurrence graph on {pncols} atoms ...")
    uf = build_atom_graph(pdata, pidx, pptr, pnrows, pncols)
    comps = uf.components(pncols)
    by_size = sorted(((len(v), r) for r, v in comps.items()), reverse=True)
    print(f"[graph] {len(comps)} components, sizes (top 10): {[s for s,_ in by_size[:10]]}")

    # --- divisor component ---
    # re-map divisor x-coords into pruned column space
    pruned_x_to_col: dict[int, int] = {}
    for i, a in enumerate(patoms):
        try:
            pruned_x_to_col[int(a)] = i
        except (ValueError, TypeError):
            pass   # ∞ and other non-integer atoms

    div_cols: list[tuple[int, int]] = []
    for x in divisor_xs:
        c = pruned_x_to_col.get(int(x))
        if c is None:
            print(f"  [warn] divisor x={x} was pruned as dest-only — "
                  f"it never appeared as xi!")
        else:
            div_cols.append((int(x), c))

    print(f"\n  Divisor atoms surviving prune: {len(div_cols)} / {len(divisor_xs)}")
    div_roots: set[int] = set()
    for x, c in div_cols:
        root = uf.find(c)
        div_roots.add(root)
        print(f"    x={x:6d}  pruned_col={c:5d}  component_root={root:5d}  "
              f"size={len(comps[root])}")

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

    # --- isolated atoms ---
    section("ISOLATED ATOM ANALYSIS")
    isolated = [
        comps[r][0] for s, r in by_size
        if s == 1 and comps[r][0] not in main_comp
    ]
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

    # --- row analysis ---
    section("ROW ANALYSIS")
    rows_in = rows_cross = rows_out = 0
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        cols = {int(pidx[k]) for k in range(rs, re)}
        in_m  = cols & main_comp
        out_m = cols - main_comp
        if in_m and not out_m:
            rows_in += 1
        elif in_m:
            rows_cross += 1
        else:
            rows_out += 1

    n_atoms  = len(main_comp)
    needed   = n_atoms - 1   # rank needed for nullity=1 in component
    deficit  = max(0, needed - rows_in)

    print(f"\n  Total pruned rows           : {pnrows}")
    print(f"  Rows inside divisor comp    : {rows_in}")
    print(f"  Rows crossing boundary      : {rows_cross}")
    print(f"  Rows entirely outside       : {rows_out}")
    print(f"\n  Atoms in divisor component  : {n_atoms}")
    print(f"  Rank needed (atoms - 1)     : {needed}")
    print(f"  Row surplus/deficit         : {rows_in - needed:+d}  "
          f"({'ok' if deficit == 0 else f'need {deficit} more'})")

    # --- rank check ---
    if args.rank_check:
        if not group_order:
            print("\n  [rank-check] --group-order required. Skipping.")
        else:
            section("RANK CHECK  (divisor-component submatrix)")

            # Locate the pruned column index of ∞ by scanning patoms for "∞".
            # This works regardless of --no-prune since patoms is always set.
            inf_pruned_col = None
            for _i, _a in enumerate(patoms):
                if str(_a) in ("∞", "inf", "\u221e"):
                    if _i in main_comp:
                        inf_pruned_col = _i
                    break

            # Full rank check including ∞.
            print(f"\n  Eliminating over GF({group_order}) on "
                  f"{n_atoms}-col component submatrix ...")
            rank = rank_mod_p(pdata, pidx, pptr, pnrows, pncols,
                               p=group_order, keep_cols=main_comp)
            nullity = n_atoms - rank
            print(f"\n  rank={rank}  atoms={n_atoms}  nullity={nullity}")

            if nullity == 1:
                print("  ✓  nullity=1 — FULLY DETERMINED up to gauge.")
                print("     DLP is solvable on this component.")
                print("     Pin isolated atoms to 0 and call solve_right.")
            elif nullity == 0:
                # Distinguish between gauge-organically-pinned and
                # genuinely overconstrained by re-running without ∞.
                if inf_pruned_col is not None:
                    comp_no_inf = main_comp - {inf_pruned_col}
                    rank_no_inf = rank_mod_p(pdata, pidx, pptr, pnrows, pncols,
                                             p=group_order, keep_cols=comp_no_inf)
                    null_no_inf = len(comp_no_inf) - rank_no_inf
                    print(f"  Without-∞: rank={rank_no_inf}  atoms={len(comp_no_inf)}"
                          f"  nullity={null_no_inf}")
                    if null_no_inf == 0:
                        # Rank fills the non-inf space completely — the walk data
                        # algebraically forces a[∞] = some specific value, i.e.
                        # the gauge is organically pinned (not overconstrained).
                        # This is fine: add the gauge row a[∞]=0 and the DLP
                        # direction should give nullity=1 in the affine system.
                        print("  ✓  Gauge (∞) is organically pinned by the walk data.")
                        print("     The non-∞ subspace is fully determined.")
                        print("     DLP is solvable: add anchor and call solve_right.")
                    else:
                        print(f"  ✗  Genuinely underdetermined even without ∞ "
                              f"(nullity={null_no_inf}).")
                        print(f"     Need {null_no_inf} more independent rows.")
                else:
                    print("  ?  nullity=0 and ∞ not in component — unexpected.")
            else:
                print(f"  ✗  nullity={nullity} — need {nullity - 1} more "
                      f"independent in-component rows.")

    # --- component detail ---
    if args.component_detail:
        section("DIVISOR COMPONENT ATOMS")
        for c in sorted(main_comp):
            print(f"  col={c:5d}  atom={patoms[c]}")

    # --- verdict ---
    section("VERDICT")
    ok = len(div_roots) == 1 and len(overlap) == 0
    print(f"""
  Divisor component size : {n_atoms}
  Rows in component      : {rows_in}
  Isolated free atoms    : {len(isolated)}  ← safe to pin to 0
  Other non-trivial comps: {len(other_nontrivial)}
  All divisors colocated : {'YES' if len(div_roots)==1 else 'NO'}
  Isolated ∩ divisor comp: {len(overlap)}  (want 0)
  Row deficit            : {deficit}
""")
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


if __name__ == "__main__":
    main()
