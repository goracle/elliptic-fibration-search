#!/usr/bin/env python3
import argparse, json, sys, numpy as np, h5py
from scipy.sparse import csr_matrix

"""
inspect_malformed_rows.py  --  triage malformed (coeff-sum ≠ 0) rows in relation_matrix.h5

Run:
    python3 inspect_malformed_rows.py relation_matrix.h5 [--curve-degree 5] [--max-bad 50]
"""

def load_h5(path):
    with h5py.File(path, "r") as f:
        atoms    = [a.decode("utf-8") if isinstance(a, bytes) else a for a in f["atoms"][:]]
        data     = f["csr/data"][:]
        indices  = f["csr/indices"][:]
        indptr   = f["csr/indptr"][:]
        shape    = tuple(int(x) for x in f["csr/shape"][:])
        try:
            aidx = json.loads(f["atom_index"][()].decode("utf-8"))
        except Exception:
            aidx = {a: i for i, a in enumerate(atoms)}
        meta = {}
        for key in ("group_order", "col_inf", "col_gen0", "col_gen1", "col_tgt0", "col_tgt1"):
            if key in f:
                meta[key] = int(f[key][()])
        if "divisor_xs" in f:
            meta["divisor_xs"] = [int(x) for x in f["divisor_xs"][:]]
    M = csr_matrix((data, indices, indptr), shape=shape)
    return M, atoms, aidx, meta

def analyse(path, curve_degree=5, max_bad=50):
    M, atoms, aidx, meta = load_h5(path)
    nrows, ncols = M.shape
    print(f"Loaded: {nrows} rows × {ncols} cols")
    print(f"Meta: {meta}\n")

    expected_sum = 0   # for a valid relation:  (d-2)+1+1 + (-d) = 0
    row_sums = np.array(M.sum(axis=1)).ravel()   # shape (nrows,)

    bad_mask = row_sums != expected_sum
    bad_rows = np.where(bad_mask)[0]
    print(f"Malformed rows (coeff sum ≠ {expected_sum}): {len(bad_rows)} / {nrows}")
    if len(bad_rows) == 0:
        print("No malformed rows — matrix is clean.")
        return

    # Bucket by coeff-sum value to see patterns
    from collections import Counter
    sum_counts = Counter(int(row_sums[i]) for i in bad_rows)
    print(f"\nCoeff-sum histogram for bad rows:")
    for s, cnt in sorted(sum_counts.items()):
        print(f"  sum={s:+d}  count={cnt}")

    # For each bad row, print its full nonzero support with atom labels
    col_inf  = meta.get("col_inf", -1)
    col_gen0 = meta.get("col_gen0", -1)
    col_gen1 = meta.get("col_gen1", -1)
    col_tgt0 = meta.get("col_tgt0", -1)
    col_tgt1 = meta.get("col_tgt1", -1)
    special = {col_inf: "∞", col_gen0: "gen0", col_gen1: "gen1",
               col_tgt0: "tgt0", col_tgt1: "tgt1"}

    def atom_label(j):
        label = atoms[j] if j < len(atoms) else f"col{j}"
        tag = special.get(j)
        return f"{label}({tag})" if tag else label

    print(f"\nFirst {min(max_bad, len(bad_rows))} bad rows (full support):")
    for row_i in bad_rows[:max_bad]:
        row = M.getrow(row_i).toarray().ravel()
        nz  = [(j, int(row[j])) for j in np.where(row != 0)[0]]
        support_str = "  ".join(f"{atom_label(j)}={v}" for j, v in nz)
        print(f"  row {row_i:6d}  sum={int(row_sums[row_i]):+d}  nnz={len(nz)}  [{support_str}]")

    # Structural analysis: which columns appear in bad rows?
    print(f"\nColumn participation in malformed rows:")
    bad_col_counts = Counter()
    inf_missing = 0
    inf_present = 0
    for row_i in bad_rows:
        row = M.getrow(row_i).toarray().ravel()
        nz_cols = set(np.where(row != 0)[0])
        for j in nz_cols:
            bad_col_counts[j] += 1
        if col_inf >= 0:
            if col_inf in nz_cols:
                inf_present += 1
            else:
                inf_missing += 1

    top_cols = bad_col_counts.most_common(20)
    for j, cnt in top_cols:
        print(f"  col {j:5d}  atom={atom_label(j):20s}  bad_row_count={cnt}")

    if col_inf >= 0:
        print(f"\n∞-column (col {col_inf}) presence in bad rows: "
              f"{inf_present} present, {inf_missing} MISSING")
        if inf_missing > 0:
            print("  → rows missing ∞ are the likely source: "
                  "cxk=='∞' hit the early `continue` in build_relation_matrix2 "
                  "and skipped the inf_coeff assignment.")

    # Check for nnz=2 rows (missing one term)
    nnz_counts = Counter()
    for row_i in bad_rows:
        row = M.getrow(row_i).toarray().ravel()
        nnz_counts[int(np.count_nonzero(row))] += 1
    print(f"\nnnz distribution of bad rows: {dict(sorted(nnz_counts.items()))}")
    if nnz_counts.get(2, 0):
        print("  → nnz=2 rows are missing one atom term (x_res dropped or x_step dropped).")
    if nnz_counts.get(3, 0):
        print("  → nnz=3 rows have all three positions but wrong coefficient "
              "(e.g. src_mult wrong, or x_res==x_src caused x_src to absorb the +1).")

    # Check: are all bad rows also valid-looking except for missing ∞?
    if col_inf >= 0:
        inf_vec = np.zeros(ncols, dtype=np.int32)
        inf_vec[col_inf] = -curve_degree
        bad_minus_inf = []
        for row_i in bad_rows:
            row = M.getrow(row_i).toarray().ravel()
            if row[col_inf] == 0:
                # ∞ missing: would adding -d fix the sum?
                adjusted_sum = int(row_sums[row_i]) + (-curve_degree)
                bad_minus_inf.append((row_i, int(row_sums[row_i]), adjusted_sum))
        if bad_minus_inf:
            fixable = [(r, s, a) for r, s, a in bad_minus_inf if a == 0]
            not_fixable = [(r, s, a) for r, s, a in bad_minus_inf if a != 0]
            print(f"\nRows missing ∞ that would be fixed by adding inf_coeff={-curve_degree}: "
                  f"{len(fixable)} / {len(bad_minus_inf)}")
            if not_fixable[:5]:
                print(f"  Rows where even adding ∞ wouldn't help (first 5):")
                for r, s, a in not_fixable[:5]:
                    print(f"    row {r}  current_sum={s}  adjusted_sum={a}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--curve-degree", type=int, default=5)
    ap.add_argument("--max-bad", type=int, default=50)
    args = ap.parse_args()
    analyse(args.path, curve_degree=args.curve_degree, max_bad=args.max_bad)
