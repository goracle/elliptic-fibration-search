#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, h5py, numpy as np
from collections import defaultdict
from typing import Any
from relation_matrix import *

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
# Parity-vector residual analysis
# ---------------------------------------------------------------------------

def parity_residual_analysis(
    pdata, pidx, pptr, pnrows, pncols,
    patoms,
    col_gen0: int | None,
    col_gen1: int | None,
    col_tgt0: int | None,
    col_tgt1: int | None,
    p: int,
    known_key: int | None = None,
):
    """Compute and print M @ v symbolically and numerically.

    v is the parity/DLP-test vector:
        v[col_gen0] = +k,  v[col_gen1] = +k
        v[col_tgt0] = -1,  v[col_tgt1] = -1
        v[j]        =  0   for all other j

    where k is the unknown discrete log.

    For each row i of M the residual entry is:
        r[i] = M[i, gen0]*k + M[i, gen1]*k - M[i, tgt0] - M[i, tgt1]
             = (M[i,gen0] + M[i,gen1]) * k  -  (M[i,tgt0] + M[i,tgt1])

    We print:
      1. Symbolic form: show (A_coeff)*k - B_coeff for each nonzero entry.
      2. If known_key is given: substitute k, reduce mod p, show which rows are
         nonzero (i.e. which atoms appear in the residual we still need to kill).
    """
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  PARITY-VECTOR RESIDUAL ANALYSIS")
    print(f"  v = (gen0+gen1)*k - (tgt0+tgt1)   over GF({p})")
    print(SEP)

    special = {"gen0": col_gen0, "gen1": col_gen1, "tgt0": col_tgt0, "tgt1": col_tgt1}
    missing = [nm for nm, c in special.items() if c is None]
    if missing:
        print(f"  [warn] missing columns for: {missing}  — those terms treated as 0")

    # Build the symbolic row-wise residual.
    # For each row r: A[r] = sum of coefficients hitting gen0/gen1 columns
    #                 B[r] = sum of coefficients hitting tgt0/tgt1 columns (positive sense)
    # residual[r] = A[r]*k - B[r]

    A_coeffs: list[int] = []  # coefficient of k
    B_coeffs: list[int] = []  # constant term (positive)

    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        a = 0
        b = 0
        for ki in range(rs, re):
            c = int(pidx[ki])
            v = int(pdata[ki])
            if c == col_gen0 or c == col_gen1:
                a += v
            if c == col_tgt0 or c == col_tgt1:
                b += v
        A_coeffs.append(a % p)
        B_coeffs.append(b % p)

    # Rows where at least one of A or B is nonzero.
    symbolic_nonzero = [(r, A_coeffs[r], B_coeffs[r])
                        for r in range(pnrows)
                        if A_coeffs[r] or B_coeffs[r]]

    print(f"\n  Rows touching gen/tgt columns: {len(symbolic_nonzero)} / {pnrows}")
    print(f"\n  --- Symbolic residual (showing rows where r[i] != 0 symbolically) ---")
    print(f"  {'row':>6}  {'A (coeff of k)':>16}  {'B (constant)':>14}  expression")
    print("  " + "-" * 60)
    for r, a, b in symbolic_nonzero[:40]:
        # Express as A*k - B  (both reduced mod p already)
        b_neg = (-b) % p
        if a == 0:
            expr = f"-{b} ≡ {b_neg} mod p"
        elif b == 0:
            expr = f"{a}*k"
        else:
            expr = f"{a}*k - {b}"
        print(f"  {r:>6}  {a:>16}  {b:>14}  {expr}")
    if len(symbolic_nonzero) > 40:
        print(f"  ... {len(symbolic_nonzero) - 40} more rows omitted")

    # Group by (A, B) signature to show how many rows share the same structure.
    from collections import Counter
    sig_counts = Counter((a, b) for _, a, b in symbolic_nonzero)
    print(f"\n  --- Symbolic signature histogram (top 12) ---")
    print(f"  {'A':>8}  {'B':>8}  {'count':>8}  expression")
    print("  " + "-" * 44)
    for (a, b), cnt in sig_counts.most_common(12):
        if a == 0:
            expr = f"const {(-b) % p} mod p"
        elif b == 0:
            expr = f"{a}*k"
        else:
            expr = f"{a}*k - {b}"
        print(f"  {a:>8}  {b:>8}  {cnt:>8}  {expr}")

    # ------------------------------------------------------------------
    # K-SENSITIVITY DIAGNOSTIC (k-free)
    #
    # For every row with A_i ≠ 0, the relation A_i*k - B_i ≡ 0 mod p
    # implies a unique candidate value  k_cand = B_i * inverse(A_i) mod p.
    # If many rows agree on the same candidate, the system is highly
    # k-sensitive and that candidate is likely the true discrete log.
    # ------------------------------------------------------------------

    k_sensitive_rows = [(r, a, b) for r, a, b in symbolic_nonzero if a != 0]
    k_candidates: list[int] = []
    for r, a, b in k_sensitive_rows:
        cand = (b * pow(int(a), p - 2, p)) % p
        k_candidates.append(cand)

    cand_counts = _CandCounter(k_candidates)
    n_sensitive  = len(k_sensitive_rows)
    n_distinct   = len(cand_counts)
    top_cand, top_freq = cand_counts.most_common(1)[0] if cand_counts else (None, 0)

    SEP2 = "=" * 70
    print(f"\n{SEP2}")
    print("  K-SENSITIVITY ANALYSIS  (no known key required)")
    print(f"  k_cand = B * inverse(A) mod p  for rows with A ≠ 0")
    print(SEP2)
    print(f"\n  k-sensitive rows (A≠0) : {n_sensitive} / {len(symbolic_nonzero)}")
    print(f"  distinct candidate k   : {n_distinct}")
    if top_cand is not None:
        pct = 100.0 * top_freq / n_sensitive if n_sensitive else 0
        print(f"  dominant candidate     : k = {top_cand}  (appears {top_freq}×, {pct:.1f}% of k-sensitive rows)")

    if n_distinct <= 1 and n_sensitive > 0:
        print("  ✓  All k-sensitive rows agree on a single candidate — system is fully k-pinned.")
    elif n_distinct > 0 and top_freq >= max(2, n_sensitive // 2):
        print("  ~  Strong consensus: dominant candidate accounts for ≥50% of k-sensitive rows.")
    elif n_sensitive == 0:
        print("  [info] No rows have A≠0 — system has zero k-sensitivity (all residual is constant).")

    if n_distinct > 0:
        print(f"\n  --- Candidate k histogram (top 20 by frequency) ---")
        print(f"  {'k_cand':>12}  {'count':>8}  {'% of A≠0':>10}  {'sym(k_cand)':>12}")
        print("  " + "-" * 50)
        for cand, freq in cand_counts.most_common(20):
            pct = 100.0 * freq / n_sensitive if n_sensitive else 0
            sym_cand = min(cand, p - cand)
            marker = "  ← dominant" if cand == top_cand else ""
            print(f"  {cand:>12}  {freq:>8}  {pct:>9.1f}%  {sym_cand:>12}{marker}")

        if n_distinct > 20:
            print(f"  ... {n_distinct - 20} more distinct candidates (each appearing once or twice)")

    if known_key is not None:
        if top_cand == int(known_key) % p:
            print(f"\n  ✓  Dominant candidate matches --known-key={known_key}.")
        elif int(known_key) % p in cand_counts:
            freq_true = cand_counts[int(known_key) % p]
            print(f"\n  ⚠  True key k={known_key} appears in candidates but is NOT dominant "
                  f"(appears {freq_true}× vs dominant {top_freq}×).")
        else:
            print(f"\n  ✗  True key k={known_key} does NOT appear among the {n_distinct} candidates.")

    if known_key is None:
        print("\n  [no --known-key supplied; skipping numeric substitution]")
        return

    # Substitute k = known_key and reduce mod p.
    k = int(known_key) % p
    print(f"\n  --- Numeric residual with known key k={known_key} (mod {p}) ---")
    numeric_nonzero = []
    for r in range(pnrows):
        a, b = A_coeffs[r], B_coeffs[r]
        val = (a * k - b) % p
        if val:
            numeric_nonzero.append((r, val, a, b))

    print(f"  Nonzero residual entries: {len(numeric_nonzero)} / {pnrows}")

    if not numeric_nonzero:
        print("  ✓  Residual is identically zero — v is in the row-null space (consistent with known key).")
        return

    # ------------------------------------------------------------------
    # A/B COUPLING DIAGNOSTIC
    #
    # The question is NOT "are residuals divisible by some d?" (gcd test).
    # The question is "do A and B ever appear in the SAME row?" — i.e. does
    # the system have rows that couple the k-dependent part to the constant
    # part?  If every row is either pure-A (B=0) or pure-B (A=0), then the
    # two halves of the residual are structurally decoupled and the system
    # cannot triangulate k against the constants, no matter how many rows
    # you have.
    #
    # Three sub-tests:
    #   1. Mod-3 histogram of (A%3, B%3) pairs — smoking gun for 3-coupling
    #   2. Mixed-row count: rows with both A≠0 and B≠0
    #   3. Rank of the (A,B) pair matrix over GF(p): rank=1 means all rows
    #      lie on a single line through the origin — pure decoupling
    # ------------------------------------------------------------------

    print(f"\n  --- A/B coupling diagnostic ---")

    # 1. Mod-3 histogram
    mod3_pairs = _Counter((a % 3, b % 3) for _, a, b in symbolic_nonzero)
    print(f"\n  (A mod 3, B mod 3) histogram over symbolic nonzero rows:")
    print(f"  {'(A%3,B%3)':>12}  {'count':>8}  meaning")
    print("  " + "-" * 50)
    meaning = {
        (0, 0): "pure multiple-of-3 in both — fully decoupled",
        (1, 0): "A≢0(3), B=0     — k-only, no constant",
        (2, 0): "A≢0(3), B=0     — k-only, no constant",
        (0, 1): "A=0, B≢0(3)     — constant-only, no k",
        (0, 2): "A=0, B≢0(3)     — constant-only, no k",
        (1, 1): "MIXED ✓          — couples k to constant",
        (1, 2): "MIXED ✓          — couples k to constant",
        (2, 1): "MIXED ✓          — couples k to constant",
        (2, 2): "MIXED ✓          — couples k to constant",
    }
    for pair, cnt in sorted(mod3_pairs.items(), key=lambda x: -x[1]):
        print(f"  {str(pair):>12}  {cnt:>8}  {meaning.get(pair, '')}")

    mixed_mod3 = sum(cnt for pair, cnt in mod3_pairs.items()
                     if pair[0] % 3 != 0 and pair[1] % 3 != 0)
    print(f"\n  Rows with both A≢0(3) and B≢0(3): {mixed_mod3}")

    # 2. Strictly mixed rows (both A and B nonzero, regardless of mod 3)
    mixed_rows = [(r, a, b) for r, a, b in symbolic_nonzero if a != 0 and b != 0]
    print(f"  Rows with both A≠0 and B≠0 (strict): {len(mixed_rows)}")
    if mixed_rows:
        print(f"  All mixed rows:")
        for r, a, b in mixed_rows:
            rs_m, re_m = int(pptr[r]), int(pptr[r + 1])
            terms = []
            for ki_m in range(rs_m, re_m):
                c_m = int(pidx[ki_m])
                v_m = int(pdata[ki_m])
                terms.append(f"{patoms[c_m]}:{v_m}")
            print(f"    row={r}  A={a}  B={b}  ({a}*k - {b})")
            print(f"      atoms: {' | '.join(terms)}")

    # 3. Rank of (A,B) pairs over GF(p)
    # Build a Nx2 matrix from all (A,B) pairs and compute rank.
    # rank=1: all on one line (e.g. all (3,0) and (0,3) both lie on no
    #         single line → actually rank=2, but if they're never mixed
    #         the system is still decoupled). So also check the span.
    ab_pairs_unique = list(set((a, b) for _, a, b in symbolic_nonzero if a or b))
    # Rank over GF(p) via hand-rolled 2-col elimination (no sage dep here).
    def _rank2_gfp(pairs, p):
        pivots = {}
        for a, b in pairs:
            row = [a % p, b % p]
            for pc, prow in pivots.items():
                f = row[pc]
                if f == 0:
                    continue
                row = [(row[j] - f * prow[j]) % p for j in range(2)]
            nz = [j for j in range(2) if row[j]]
            if not nz:
                continue
            pc2 = nz[0]
            inv = pow(int(row[pc2]), p - 2, p)
            pivots[pc2] = [(v * inv) % p for v in row]
            if len(pivots) == 2:
                return 2
        return len(pivots)

    ab_rank = _rank2_gfp(ab_pairs_unique, p)
    print(f"\n  Rank of (A,B) pair matrix over GF({p}): {ab_rank}")
    if ab_rank == 0:
        print("  [warn] all rows have A=B=0 — nothing to analyse")
    elif ab_rank == 1:
        print("  ✗  rank=1: all (A,B) pairs are scalar multiples of a single vector.")
        print("     The k-part and constant-part are COMPLETELY collinear — fully decoupled.")
        print("     The system cannot determine k relative to constants.")
    else:
        # rank == 2
        if len(mixed_rows) == 0:
            print("  ⚠  rank=2 (A and B span GF(p)²), but NO row has both A≠0 and B≠0.")
            print("     The span comes from separate pure-A and pure-B rows.")
            print("     Algebraically the system CAN determine k, but only if a solver")
            print("     chains through multiple rows — e.g. pure-A pins k*atom,")
            print("     pure-B pins atom, combining gives k.  Check if the atoms overlap.")
            # Check atom overlap between A-rows and B-rows.
            a_atoms: set = set()
            b_atoms: set = set()
            for r, a, b in symbolic_nonzero:
                rs2, re2 = int(pptr[r]), int(pptr[r + 1])
                for ki2 in range(rs2, re2):
                    c2 = int(pidx[ki2])
                    nm2 = patoms[c2]
                    if nm2 == "∞":
                        continue
                    if a != 0:
                        a_atoms.add(nm2)
                    if b != 0:
                        b_atoms.add(nm2)
            shared = a_atoms & b_atoms
            print(f"     Atoms in A-rows: {len(a_atoms)}  |  Atoms in B-rows: {len(b_atoms)}  |  Shared: {len(shared)}")
            if shared:
                print(f"     ✓ Shared atoms exist — indirect coupling possible via multi-row chains.")
                print(f"     Sample shared atoms: {sorted(shared)[:8]}")
            else:
                print(f"     ✗ NO shared atoms. A-rows and B-rows are on completely disjoint")
                print(f"       atom sets. The system is structurally decoupled.")

                # ----------------------------------------------------------
                # CHAIN-LENGTH-2 RELAY ANALYSIS
                #
                # Even with no direct shared atoms, k-coupling can happen
                # through a relay row: an ordinary (non-gen/tgt) row that
                # shares one atom with an A-row and another atom with a B-row.
                #
                # Chain:  A-row --[atom_a]--> relay_row --[atom_b]--> B-row
                #
                # We report:
                #   - how many such chains exist
                #   - which relay atoms bridge A-side to B-side
                #   - the shortest such chain (fewest relay hops)
                # ----------------------------------------------------------
                print(f"\n     --- Chain-length-2 relay analysis ---")

                # Build atom -> set of ALL row indices (full matrix).
                atom_to_rows: dict = _dd3(set)
                for row_r in range(pnrows):
                    rs3, re3 = int(pptr[row_r]), int(pptr[row_r + 1])
                    for ki3 in range(rs3, re3):
                        nm3 = patoms[int(pidx[ki3])]
                        if nm3 != "∞":
                            atom_to_rows[nm3].add(row_r)

                # Row sets for A-side and B-side gen/tgt rows.
                a_row_set = {r for r, a, b in symbolic_nonzero if a != 0}
                b_row_set = {r for r, a, b in symbolic_nonzero if b != 0}

                # For each atom in a_atoms, find relay rows (rows that contain
                # that atom but are NOT A-rows themselves).  Then for each relay
                # row, check which of its atoms are in b_atoms.
                relay_chains: list[tuple] = []   # (a_atom, relay_row, b_atom)
                relay_atom_pairs: set = set()    # (a_atom, b_atom) bridge pairs

                for a_atom in a_atoms:
                    relay_rows = atom_to_rows[a_atom] - a_row_set
                    for relay_r in relay_rows:
                        rs4, re4 = int(pptr[relay_r]), int(pptr[relay_r + 1])
                        relay_atoms = {patoms[int(pidx[ki4])]
                                       for ki4 in range(rs4, re4)
                                       if patoms[int(pidx[ki4])] != "∞"}
                        bridged = relay_atoms & b_atoms
                        for b_atom in bridged:
                            relay_chains.append((a_atom, relay_r, b_atom))
                            relay_atom_pairs.add((a_atom, b_atom))

                n_chains = len(relay_chains)
                n_pairs  = len(relay_atom_pairs)

                if n_chains == 0:
                    print(f"     ✗ No length-2 relay chains found.")
                    print(f"       A-atoms and B-atoms are ≥3 hops apart in the co-occurrence graph.")
                    print(f"       This is a deep structural decoupling — new relation types needed.")
                else:
                    # Summarise by (a_atom, b_atom) bridge pair frequency.
                    pair_freq   = _CC2((a, b) for a, _, b in relay_chains)
                    a_atom_freq = _CC2(a for a, _, _ in relay_chains)
                    b_atom_freq = _CC2(b for _, _, b in relay_chains)

                    print(f"     ✓ {n_chains} length-2 relay chain(s) found via {n_pairs} distinct (A-atom, B-atom) bridge pair(s).")
                    print(f"       These are the relay paths through which the bulk matrix")
                    print(f"       can indirectly couple k-rows to constant-rows.")

                    print(f"\n     Top bridge pairs (a_atom → b_atom, by relay chain count):")
                    print(f"     {'a_atom':>10}  {'b_atom':>10}  {'chains':>8}")
                    print(f"     " + "-" * 34)
                    for (a_at, b_at), cnt in pair_freq.most_common(12):
                        print(f"     {str(a_at):>10}  {str(b_at):>10}  {cnt:>8}")
                    if len(pair_freq) > 12:
                        print(f"     ... {len(pair_freq) - 12} more bridge pairs")

                    print(f"\n     Most-bridging A-side atoms (appear in most relay chains):")
                    for atm, cnt in a_atom_freq.most_common(6):
                        print(f"       {str(atm):>10}  chains={cnt}")

                    print(f"\n     Most-bridging B-side atoms (appear in most relay chains):")
                    for atm, cnt in b_atom_freq.most_common(6):
                        print(f"       {str(atm):>10}  chains={cnt}")

                    # Show one concrete example chain in full.
                    ex_a, ex_relay, ex_b = relay_chains[0]
                    rs5, re5 = int(pptr[ex_relay]), int(pptr[ex_relay + 1])
                    ex_atoms = [(patoms[int(pidx[ki5])], int(pdata[ki5]))
                                for ki5 in range(rs5, re5)]
                    ex_str = "  ".join(f"{nm}:{cf}" for nm, cf in ex_atoms[:8])
                    if len(ex_atoms) > 8:
                        ex_str += f"  ...+{len(ex_atoms)-8}"
                    print(f"\n     Example chain:")
                    print(f"       A-atom={ex_a}  →  relay row {ex_relay}  →  B-atom={ex_b}")
                    print(f"       relay row atoms: {ex_str}")
        else:
            print(f"  ✓  rank=2 AND {len(mixed_rows)} mixed rows (A≠0, B≠0) exist.")
            print(f"     The parity system is properly coupled. Residual should collapse")
            print(f"     once the underlying atom logs are determined.")

    # Summary verdict
    print(f"\n  --- Coupling verdict ---")
    if mixed_mod3 == 0 and len(mixed_rows) == 0:
        print("  ✗  DECOUPLED: no row mixes k-dependence with constants (mod 3 or strictly).")
        print("     Every relation touching gen/tgt is either purely k-dependent OR purely constant.")
        print("     This is why the residual is stuck — the walk relation template structurally")
        print("     prevents A and B from appearing together. Need a new relation type")
        print("     (e.g. from a different fiber intersection or an explicit tgt/gen cross-term).")
    elif mixed_rows:
        print(f"  ✓  {len(mixed_rows)} strictly mixed rows found. System is coupled.")
    else:
        print(f"  ⚠  No strictly mixed rows, but mod-3 mixing exists ({mixed_mod3} rows).")
        print(f"     Marginal coupling — may or may not be sufficient.")
    print()

    print(f"\n  These are the rows (relations) not yet killed by existing data:")
    print(f"  {'row':>6}  {'r[i] mod p':>12}  {'sym val':>8}  {'A':>8}  {'B':>8}  atoms in row")
    print("  " + "-" * 80)

    def sym(v: int) -> int:
        return min(v, p - v)

    # Sort by symmetric magnitude descending.
    numeric_nonzero.sort(key=lambda t: -sym(t[1]))

    # Build a quick row->atoms lookup from CSR.
    for r, val, a, b in numeric_nonzero[:30]:
        rs, re = int(pptr[r]), int(pptr[r + 1])
        row_atoms = [(patoms[int(pidx[ki])], int(pdata[ki])) for ki in range(rs, re)]
        # Only non-gen/tgt atoms are "background" — label gen/tgt specially.
        atom_strs = []
        for atm, coef in row_atoms:
            c = int(pidx[pptr[r]])  # placeholder; recompute properly below
            atom_strs.append(f"{atm}:{coef}")
        # recompute properly
        atom_strs = []
        for ki in range(rs, re):
            c = int(pidx[ki])
            coef = int(pdata[ki])
            nm = patoms[c]
            tag = ""
            if c == col_gen0:   tag = "(gen0)"
            elif c == col_gen1: tag = "(gen1)"
            elif c == col_tgt0: tag = "(tgt0)"
            elif c == col_tgt1: tag = "(tgt1)"
            atom_strs.append(f"{nm}{tag}:{coef}")
        preview = "  ".join(atom_strs[:8])
        if len(atom_strs) > 8:
            preview += f"  ...+{len(atom_strs)-8}"
        print(f"  {r:>6}  {val:>12}  {sym(val):>8}  {a:>8}  {b:>8}  {preview}")

    if len(numeric_nonzero) > 30:
        print(f"  ... {len(numeric_nonzero) - 30} more nonzero rows")

    # Atom frequency in nonzero-residual rows — tells you which atoms are blocking.
    print(f"\n  --- Atoms appearing in nonzero-residual rows (top 20 by frequency) ---")
    atom_freq: dict = _dd(int)
    atom_max_sym: dict = _dd(int)
    for r, val, a, b in numeric_nonzero:
        rs, re = int(pptr[r]), int(pptr[r + 1])
        for ki in range(rs, re):
            c = int(pidx[ki])
            nm = patoms[c]
            atom_freq[nm] += 1
            atom_max_sym[nm] = max(atom_max_sym[nm], sym(val))
    top_atoms = sorted(atom_freq.items(), key=lambda kv: (-kv[1], -atom_max_sym[kv[0]]))[:20]
    print(f"  {'atom':>12}  {'freq':>6}  {'max_sym_r':>10}")
    print("  " + "-" * 36)
    for nm, freq in top_atoms:
        print(f"  {nm:>12}  {freq:>6}  {atom_max_sym[nm]:>10}")

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
    ap.add_argument("--parity-residual", action="store_true",
                    help="Compute M @ v where v=(gen0+gen1)*k-(tgt0+tgt1); show symbolically and (with --known-key) numerically")
    ap.add_argument("--known-key", type=int, default=None,
                    help="Known discrete log k to substitute into the parity residual")
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

    # Repair/drop malformed rows.
    #
    # A valid principal-divisor relation has integer coefficients summing to 0.
    #
    # sum == -1  ->  tangent relation (x_res == x_src): the dest atom x_src had
    #   coefficient 3 written, but should be 4 (the tangency accounts for one
    #   extra copy of the root).  We find the unique finite atom with coeff 3
    #   and bump it to 4 to restore sum=0.  "Finite" means not the inf column.
    #
    # any other nonzero sum  ->  genuinely corrupt, drop the row.
    #
    # Find the pruned inf column index so the repair step can skip it.
    _inf_col_pruned: int | None = None
    for _ci, _ca in enumerate(patoms):
        if str(_ca) in ("\u221e", "inf", "infinity"):
            _inf_col_pruned = _ci
            break

    # Materialise each row as a mutable dense list for patching.
    _rows_list: list[list[int]] = []
    for r in range(pnrows):
        rs, re = int(pptr[r]), int(pptr[r + 1])
        row: list[int] = [0] * pncols
        for k in range(rs, re):
            row[int(pidx[k])] = int(pdata[k])
        _rows_list.append(row)

    _repaired:  list[int] = []
    _malformed: list[int] = []
    for r, row in enumerate(_rows_list):
        s = sum(row)
        if s == 0:
            continue
        if s == -1:
            # Tangent row: find the unique finite coeff-3 atom and bump to 4.
            fixed = False
            for c, v in enumerate(row):
                if v == 3 and c != _inf_col_pruned:
                    row[c] = 4
                    _repaired.append(r)
                    fixed = True
                    break
            if not fixed:
                # sum=-1 but no coeff-3 finite atom: unexpected, drop it.
                _malformed.append(r)
        else:
            _malformed.append(r)

    # Rebuild CSR from the (possibly patched) row list, excluding malformed rows.
    _malformed_set = set(_malformed)
    _keep_rows = [r for r in range(pnrows) if r not in _malformed_set]
    _new_data, _new_idx, _new_ptr = [], [], [0]
    for r in _keep_rows:
        for c, v in enumerate(_rows_list[r]):
            if v:
                _new_data.append(v)
                _new_idx.append(c)
        _new_ptr.append(len(_new_data))
    pdata  = np.array(_new_data, dtype=np.int32)
    pidx   = np.array(_new_idx,  dtype=np.int32)
    pptr   = np.array(_new_ptr,  dtype=np.int32)
    pnrows = len(_keep_rows)

    if _repaired:
        print(f"[filter] repaired {len(_repaired)} tangent row(s) (x_res==x_src, x_src coeff 3->4): "
              f"{_repaired[:16]}" + (" ..." if len(_repaired) > 16 else ""))
    if _malformed:
        print(f"[filter] dropping {len(_malformed)} malformed row(s) (coeff sum != 0, not repairable): "
              f"{_malformed[:16]}" + (" ..." if len(_malformed) > 16 else ""))
        print(f"[filter] matrix after malformed-row drop: {pnrows} rows x {pncols} cols")
    if not _repaired and not _malformed:
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
                print(f"  [warn] divisor x={x} was pruned as dest-only — it never appeared as x_src!")
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

    if args.parity_residual:
        if not group_order:
            print("\n[parity-residual] --group-order required. Skipping.")
        else:
            # Build pruned-col indices for gen/tgt columns.
            if args.no_prune:
                orig_to_pruned = {c: c for c in range(h["ncols"])}
                _excl_remap_pr = {}
            else:
                orig_to_pruned = {orig: pruned for pruned, orig in enumerate(kept_cols)}
                _excl_remap_pr = col_remap_excl if excluded_cols else {}

            def _pc_pr(key):
                orig = h.get(key)
                if orig is None:
                    return None
                if _excl_remap_pr:
                    orig = _excl_remap_pr.get(orig)
                    if orig is None:
                        return None
                return orig_to_pruned.get(orig)

            parity_residual_analysis(
                pdata, pidx, pptr, pnrows, pncols,
                patoms=patoms,
                col_gen0=_pc_pr("col_gen0"),
                col_gen1=_pc_pr("col_gen1"),
                col_tgt0=_pc_pr("col_tgt0"),
                col_tgt1=_pc_pr("col_tgt1"),
                p=group_order,
                known_key=args.known_key,
            )

if __name__ == "__main__":
    main()
