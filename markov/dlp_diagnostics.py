from __future__ import annotations
import math
import json
import numpy as np
from typing import Any, List, Optional, Sequence
from collections import Counter
from sage.all import GF, ZZ, Matrix, vector, matrix, Integer
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False
from .relation_matrix import *
from search_common import get_y_unshifted_genus2, COEFFS_GENUS2

"""dlp_diagnostics.py

Post-run consistency checks for the genus-2 Markov DLP solve.

This module does not try to explain causation from collisions or kernel
support; it only reports what the current matrix and walker logs actually show.

Typical usage:

    from dlp_diagnostics import run_all_checks
    run_all_checks(
        walkers       = [walker_a, walker_b, walker_c, walker_d],
        divisor_xs    = divisor_xs,   # [x0_a, x0_b, x0_c, x0_d]
        group_order   = GROUP_MODULUS,
        known_key     = SECRET_KEY,
        p             = FINITE_FIELD,
        coeffs        = COEFFS_GENUS2,
    )

Do not suppress exceptions silently. The caller should see failures.
"""

# ---------------------------------------------------------------------------
# Lazy Sage import (so the module is importable in non-Sage envs for linting)
# ---------------------------------------------------------------------------
_SAGE = True

# Re-use helpers from the project

_SEP = "=" * 70
_INFINITY = "∞"

def check_involution_symmetry(walkers):
    """Run close_under_involution() on each walker and report.

    In merge_experiment.py this check is NEVER called (it only appears in
    genus2_markov_module.py's single-walker path).  This function plugs that gap.

    Also checks whether the hyperelliptic involution (y -> -y) is correctly
    handled: T(xj) swaps to the other fiber root xk, which lies on the
    OPPOSITE y-branch.  So if yj_sign = +1 on the original record, then the
    involution-closure record should have yj_sign reflecting the -y branch
    (conventionally still +1 if we're only tracking x-coordinates, but the
    point is that xj↔xk with the SAME y-branch would be wrong geometry).
    """
    _section("INVOLUTION SYMMETRY CHECK")

    for i, w in enumerate(walkers):
        label = getattr(w, '_label', f"walker[{i}]")
        try:
            n = w.close_under_involution()
            _log(f"  ✓  {label}: T(T(xj))==xj on {n} pairs")
        except AssertionError as exc:
            _log(f"  ✗  {label}: INVOLUTION VIOLATION — {exc}")
            raise
        except Exception as exc:
            _log(f"  ?  {label}: check failed ({exc})")
            raise

    # Does the involution preserve y?  Answer: NO — it should NEGATE y.
    # T maps (xj, yj) -> (xk, yk) where xk = S(m) - xi_mult*xi - xj.
    # The point xk is a DIFFERENT x-coordinate on the curve, with its own y.
    # The hyperelliptic involution (x,y)->(x,-y) is a separate operation.
    # T is the Vieta partner on the SAME fiber (same secant line m),
    # so xk has y²=G(xk) which is generically nonzero and unrelated to yj.
    # Conclusion: the "does T preserve y?" question is malformed — T maps
    # to a different x, so y is recomputed from scratch.  yj_sign and yk_sign
    # are both +1 by convention (we pick the positive square root branch),
    # UNLESS the walker specifically tracks which branch, in which case
    # yk_sign should be the sign of sqrt(G(xk)) on the fiber.
    _log(
        "\n  NOTE on y-preservation:\n"
        "  The Vieta involution T: xj -> xk = S(m) - xi_mult*xi - xj maps to a\n"
        "  DIFFERENT x-coordinate.  y is recomputed at xk from y²=G(xk).\n"
        "  T does NOT preserve y (that would be the hyperelliptic involution x->x, y->-y).\n"
        "  If your walker records yj_sign for the branch at xj, the analogous yk_sign\n"
        "  at xk is independently chosen — there is no forced sign relationship between them.\n"
        "  The matrix coefficient for xk should always be +1 (|yk_sign|=1, not the branch sign)."
    )

def _col(aidx: dict, key):
    """Stable column lookup that does not break when the index is 0."""
    if key is None:
        return None
    if key in aidx:
        return aidx[key]
    return aidx.get(str(key))

_SEP = "=" * 70
_INFINITY = "∞"

def check_fiber_sign(walkers, p: int, coeffs, n_spot_checks: int = 5):
    """
    Conservative sign checker.

    This does NOT assume that yj_sign/yk_sign are errors just because they are
    negative. It checks only what can be justified from the stored line data:
      - Vieta sum consistency
      - sign metadata is ±1
      - if line coefficients v0,v1 are available, yj should match v(xj)
        and yk should match -v(xk), both compared against the canonical branch
        convention used by the project.
    """
    _section("FIBER SIGN CHECK")

    checked = 0
    violations = []

    for w in walkers:
        for rec in w.history:
            if not getattr(rec, "accepted", False):
                continue
            if rec.xi is None or rec.xj is None or rec.xk is None:
                continue

            step = _step_dict(rec)
            if step.get("source") == "involution_closure":
                continue

            xi = int(rec.xi)
            xj = int(rec.xj)
            xk = int(rec.xk)

            deg = w.config.curve_degree
            xi_mult = deg - 2

            # (a) Vieta sum check.
            S_sym = step.get("S_of_m")
            m_val = getattr(rec, "m", None)
            if S_sym is not None and m_val is not None:
                try:
                    Fp = GF(p)
                    m_fp = Fp(m_val)
                    S_val = Fp(S_sym(m_fp))
                    vieta_sum = Fp(xi_mult * xi + xj + xk)
                    if vieta_sum != S_val:
                        violations.append(
                            f"  VIETA MISMATCH  step={getattr(rec, 'step_index', '?')} "
                            f"xi={xi} xj={xj} xk={xk}  "
                            f"xi_mult*xi+xj+xk={vieta_sum}  S(m)={S_val}"
                        )
                except Exception as exc:
                    violations.append(f"  VIETA CHECK ERROR step={getattr(rec, 'step_index', '?')}: {exc}")

            # (b) Metadata sanity.
            yj_sign = getattr(rec, "yj_sign", None)
            yk_sign = getattr(rec, "yk_sign", None)
            if yj_sign is not None and int(yj_sign) not in (1, -1):
                violations.append(
                    f"  BAD SIGN METADATA  step={getattr(rec, 'step_index', '?')}  yj_sign={yj_sign}"
                )
            if yk_sign is not None and int(yk_sign) not in (1, -1):
                violations.append(
                    f"  BAD SIGN METADATA  step={getattr(rec, 'step_index', '?')}  yk_sign={yk_sign}"
                )

            # (c) If we have the line parameters, compare against the actual
            # branch values the group law implies.
            v0 = _first_present(rec, step, "v0", "line_v0", "mumford_v0")
            v1 = _first_present(rec, step, "v1", "line_v1", "mumford_v1")

            line_info = ""
            if v0 is not None and v1 is not None:
                try:
                    v0 = int(v0)
                    v1 = int(v1)
                    yj_from_line = (v0 + v1 * xj) % p
                    yk_from_line = (- (v0 + v1 * xk)) % p
                    yj_branch = _branch_sign_from_residue(yj_from_line, p)
                    yk_branch = _branch_sign_from_residue(yk_from_line, p)
                    line_info = (
                        f"  v(xj)={yj_from_line}  branch={yj_branch:+d}"
                        f"  -v(xk)={yk_from_line}  branch={yk_branch:+d}"
                    )

                    if yj_sign is not None and int(yj_sign) != yj_branch:
                        violations.append(
                            f"  YJ SIGN MISMATCH  step={getattr(rec, 'step_index', '?')} "
                            f"xi={xi} xj={xj} xk={xk}  yj_sign={yj_sign}  expected={yj_branch}"
                        )
                    if yk_sign is not None and int(yk_sign) != yk_branch:
                        violations.append(
                            f"  YK SIGN MISMATCH  step={getattr(rec, 'step_index', '?')} "
                            f"xi={xi} xj={xj} xk={xk}  yk_sign={yk_sign}  expected={yk_branch}"
                        )
                except Exception as exc:
                    violations.append(f"  LINE SIGN CHECK ERROR step={getattr(rec, 'step_index', '?')}: {exc}")

            _log(
                f"  step={getattr(rec, 'step_index', '?'):>4}  xi={xi} xj={xj} xk={xk}  "
                f"yj_sign={yj_sign}  yk_sign={yk_sign}{line_info}"
            )

            checked += 1
            if checked >= n_spot_checks:
                break
        if checked >= n_spot_checks:
            break

    if violations:
        _log(f"\n  ✗  {len(violations)} violation(s) found across {checked} spot-checks:")
        for v in violations:
            _log(v)
    else:
        _log(f"\n  ✓  {checked} spot-checks passed.")

    return violations

def _log(msg: str) -> None:
    print(msg, flush=True)

def _section(title: str) -> None:
    _log(f"\n{_SEP}")
    _log(f"CHECK: {title}")
    _log(_SEP)

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _step_dict(rec):
    step = getattr(rec, "step", None)
    return step if isinstance(step, dict) else {}

def _first_present(rec, step, *names):
    for name in names:
        if isinstance(step, dict) and name in step and step[name] is not None:
            return step[name]
        if hasattr(rec, name):
            val = getattr(rec, name)
            if val is not None:
                return val
    return None

def _branch_sign_from_residue(val, p: int) -> int:
    """
    Canonical branch convention used by this diagnostic:
      +1 if residue is on the canonical representative side,
      -1 otherwise.
    """
    y = int(val) % p
    if y == 0:
        return 0
    return 1 if y <= (p - y) else -1

def check_divisor_injection(walkers, divisor_xs: Sequence):
    """
    Check whether the four divisor atoms appear as xi and xj in the collected walks.

    This is a coverage check, not a causality claim. It reports whether the
    atoms survive as source-side columns in the data used to build the relation
    matrix, and whether any walker merged very early.
    """
    _section("DIVISOR ATOM INJECTION QUALITY")

    for i, w in enumerate(walkers):
        label = getattr(w, "_label", f"walker[{i}]")
        xi_counts = Counter(int(r.xi) for r in w.history if r.accepted and r.xi is not None)
        xj_counts = Counter(int(r.xj) for r in w.history if r.accepted and r.xj is not None)

        _log(f"\n  {label}:")
        all_ok = True
        for x in divisor_xs:
            ix = int(x)
            as_xi = xi_counts[ix]
            as_xj = xj_counts[ix]
            status = "✓" if as_xi > 0 else "✗ NEVER AS xi"
            if as_xi == 0:
                all_ok = False
            _log(f"    x={ix:6d}  as xi: {as_xi:4d}  as xj: {as_xj:4d}  {status}")

        if all_ok:
            _log("  ✓  All divisor atoms appeared as xi at least once.")
        else:
            _log(
                "  ✗  Some divisor atoms never appeared as xi.\n"
                "     Those atoms cannot contribute source-side rows in the matrix.\n"
                "     That may matter for anchors or for any check that assumes the\n"
                "     seed atoms are represented on the xi side."
            )

        fms = getattr(w, "first_merge_step", None)
        fmv = getattr(w, "first_merge_vol", None)
        if fms == 0:
            _log(
                f"\n  note: first_merge_step=0 for {label} (vol={fmv}).\n"
                "        This records an immediate collision in the current run.\n"
                "        By itself it does not prove anything about correctness."
            )

def _classify_kernel_vec(vec, atoms, n_cols, inf_str, div_strs, gen_strs, tgt_strs):
    """Classify a single kernel basis vector.

    Returns a dict with keys:
      kind         : 'gauge' | 'isolated' | 'parity' | 'other'
      support      : list of (atom, coeff) for nonzero entries
      touches_gen  : bool
      touches_tgt  : bool
      is_flat      : bool (all nonzero coefficients equal)
      flat_coeff   : the common coefficient value if is_flat, else None
    """
    support = [(atoms[j], int(vec[j])) for j in range(n_cols) if vec[j] != 0]
    atom_strs = [str(a) for a, _ in support]
    coeffs = [c for _, c in support]

    touches_inf = any(s == inf_str for s in atom_strs)
    touches_gen = any(s in gen_strs for s in atom_strs)
    touches_tgt = any(s in tgt_strs for s in atom_strs)

    is_flat = len(set(coeffs)) == 1
    flat_coeff = coeffs[0] if (is_flat and coeffs) else None

    if len(support) == 1 and touches_inf:
        kind = 'gauge'
    elif len(support) == 1:
        kind = 'isolated'
    elif is_flat:
        kind = 'parity'
    else:
        kind = 'other'

    return {
        'kind': kind,
        'support': support,
        'touches_gen': touches_gen,
        'touches_tgt': touches_tgt,
        'is_flat': is_flat,
        'flat_coeff': flat_coeff,
    }

def check_kernel(walkers, group_order: int, divisor_xs=()):
    _section("KERNEL DECOMPOSITION")

    M, atoms, aidx, _, _ = _build_combined_matrix(
        walkers, protected=divisor_xs if divisor_xs else None
    )
    n_cols = len(atoms)
    Fp = GF(group_order)
    M_fp = M.change_ring(Fp)

    rank = M_fp.rank()
    ker = M_fp.right_kernel()
    null = ker.dimension()

    _log(f"  rows={M.nrows()}  cols={n_cols}  rank={rank}  nullity={null}")
    _log("  (nullity=1 is the ideal case if the only free direction is the gauge.)")

    div_strs = {str(x) for x in divisor_xs}
    # First two divisor_xs are gen, last two are tgt (matches seed convention).
    div_list = [str(x) for x in divisor_xs]
    gen_strs = set(div_list[:2]) if len(div_list) >= 2 else set()
    tgt_strs = set(div_list[2:]) if len(div_list) >= 4 else set()

    kind_counts = Counter()

    for i, vec in enumerate(ker.basis()):
        info = _classify_kernel_vec(vec, atoms, n_cols, _INFINITY, div_strs, gen_strs, tgt_strs)
        kind = info['kind']
        kind_counts[kind] += 1
        support = info['support']

        tags = []
        if any(str(a) == _INFINITY for a, _ in support):
            tags.append("∞-gauge")
        if any(str(a) in div_strs for a, _ in support):
            tags.append("DIVISOR-ATOM")
        tags.append(kind.upper())
        tag_str = f"  [{', '.join(tags)}]"

        _log(f"\n  kernel[{i}]:{tag_str}")

        if kind == 'gauge':
            _log(f"    atom=∞  coeff={support[0][1]}  (expected gauge direction)")

        elif kind == 'isolated':
            atom, coeff = support[0]
            marker = " ← divisor" if str(atom) in div_strs else ""
            _log(f"    atom={atom}  coeff={coeff}{marker}")
            _log(f"    → Isolated atom: no relation pins this column.")
            _log(f"    → Fix: ensure at least one accepted step has xi={atom} (coeff 3).")

        elif kind == 'parity':
            c0 = info['flat_coeff']
            _log(f"    All {len(support)} nonzero coefficients = {c0}  (conservation law)")
            _log(f"    → Enforces: {c0} · Σ a[x] ≡ 0 (mod {group_order}) "
                 f"over this direction's support.")
            _log(f"    → Touches generator atoms: {info['touches_gen']}")
            _log(f"    → Touches target atoms:    {info['touches_tgt']}")
            if info['touches_gen'] and info['touches_tgt']:
                _log(f"    ✗ CRITICAL: anchor a[gen0]+a[gen1]=1 contradicts this law.")
                _log(f"      Each step contributes 3·a[xi]+a[xj]+a[xk]=0 (coeff-sum=5).")
                _log(f"      The conservation forces anchor RHS → 0, not 1.")
                _log(f"      Fix: change anchor RHS from 1 to the inverse of the")
                _log(f"      conserved coefficient (e.g. try RHS = inverse(5) mod {group_order}"
                     f" = {pow(5, -1, group_order) if group_order > 5 else '?'}).")
            # Print atom list (truncated — can be huge)
            for atom, coeff in support[:30]:
                marker = " ← divisor" if str(atom) in div_strs else ""
                _log(f"    atom={atom}  coeff={coeff}{marker}")
            if len(support) > 30:
                _log(f"    atom=[...]")
            for atom, coeff in support[-4:] if len(support) > 30 else []:
                marker = " ← divisor" if str(atom) in div_strs else ""
                _log(f"    atom={atom}  coeff={coeff}{marker}")

        else:  # 'other'
            coeff_vals = sorted(set(c for _, c in support))
            _log(f"    support_size={len(support)}  distinct_coeffs={coeff_vals}")
            for atom, coeff in support:
                marker = " ← divisor" if str(atom) in div_strs else ""
                _log(f"    atom={atom}  coeff={coeff}{marker}")

    _log(f"\n  Direction summary: "
         + "  ".join(f"{k}={v}" for k, v in sorted(kind_counts.items())))

    if null == 1:
        _log("\n  ✓  Nullity=1.")
    elif null == 2:
        _log("\n  ✗  Nullity=2 — one extra free direction beyond the gauge.")
    else:
        _log(f"\n  ✗  Nullity={null} — {null - 1} extra free directions beyond the gauge.")

    return ker, atoms, aidx

def check_zero_compatibility(
    walkers,
    divisor_xs,
    group_order: int,
):
    """
    Inspect whether the divisor atoms appear in the kernel basis support.

    This is a structural summary of the current kernel basis only. It does not
    claim that any particular atom is the cause of inconsistency.
    """
    _section("ZERO COMPATIBILITY OF BASE/TARGET ATOMS")

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]
    M, atoms, aidx, _, _ = _build_combined_matrix(walkers, protected=divisor_xs)
    Fp = GF(group_order)
    M_fp = M.change_ring(Fp)

    ker = M_fp.right_kernel()
    null = ker.dimension()

    labels = {
        str(x0_a): "GEN_x0 (A seed)",
        str(x0_b): "GEN_x1 (B seed)",
        str(x0_c): "TGT_x0 (C seed)",
        str(x0_d): "TGT_x1 (D seed)",
    }

    _log(f"  Kernel dimension: {null}  (1 = gauge-only in the homogeneous system)")
    _log("\n  Divisor atom membership in kernel basis support:")

    kernel_hits = {}
    basis_vecs = list(ker.basis())
    for bi, vec in enumerate(basis_vecs):
        for j, coeff in enumerate(vec):
            if coeff == 0:
                continue
            atom_str = str(atoms[j])
            if atom_str in labels:
                kernel_hits.setdefault(atom_str, []).append((bi, int(coeff)))

    for atom_str, label in labels.items():
        col = aidx.get(atom_str)
        hits = kernel_hits.get(atom_str, [])
        pruned = col is None

        if hits:
            coeffs_in_ker = [c for _, c in hits]
            _log(f"    x={atom_str:6s}  [{label}]  IN KERNEL BASIS  coeffs={coeffs_in_ker}")
        elif pruned:
            _log(f"    x={atom_str:6s}  [{label}]  PRUNED (never appeared as xi)")
        else:
            _log(f"    x={atom_str:6s}  [{label}]  not seen in kernel basis support")

    gen_col = aidx.get(str(x0_a))
    gen_p_col = aidx.get(str(x0_b))
    tgt_col = aidx.get(str(x0_c))
    tgt_p_col = aidx.get(str(x0_d))
    inf_col = aidx.get(_INFINITY)

    _log("\n  Column presence after prune:")
    _log(f"    ∞       : col={inf_col}")
    _log(f"    gen_x0  : col={gen_col}")
    _log(f"    gen_x1  : col={gen_p_col}")
    _log(f"    tgt_x0  : col={tgt_col}")
    _log(f"    tgt_x1  : col={tgt_p_col}")

    missing_cols = [n for n, c in [("gen_x0", gen_col), ("tgt_x0", tgt_col)] if c is None]
    if missing_cols:
        _log(
            f"\n  ✗  Key columns were pruned: {missing_cols}\n"
            "     These atoms need source-side representation to survive pruning."
        )
        return

    mixed_basis = []
    for bi, vec in enumerate(basis_vecs):
        has_gen = any(vec[aidx[str(x0)]] != 0 for x0 in (x0_a, x0_b) if str(x0) in aidx)
        has_tgt = any(vec[aidx[str(x0)]] != 0 for x0 in (x0_c, x0_d) if str(x0) in aidx)
        if has_gen and has_tgt:
            mixed_basis.append(bi)

    if mixed_basis:
        _log(
            "\n  warning: at least one kernel basis vector has nonzero coefficients on both\n"
            f"           generator and target atoms (basis indices: {mixed_basis[:10]}"
            f"{'...' if len(mixed_basis) > 10 else ''})."
        )
    else:
        _log("\n  ✓  No displayed kernel basis vector mixes generator and target atoms.")

    _log("\n  Steps where xi equals a target atom:")
    for i, w in enumerate(walkers):
        label = getattr(w, "_label", f"walker[{i}]")
        tgt_xi_steps = [
            r.step_index for r in w.history
            if getattr(r, "accepted", False) and _safe_int(getattr(r, "xi", None)) in (x0_c, x0_d)
        ]
        preview = f"  (step indices: {tgt_xi_steps[:10]}{'...' if len(tgt_xi_steps) > 10 else ''})" if tgt_xi_steps else ""
        _log(f"    {label}: {len(tgt_xi_steps)} accepted steps with xi in target atoms{preview}")

def dump_matrix_hdf5(
    walkers,
    divisor_xs,
    group_order: int,
    path: str = "relation_matrix.h5",
):
    """Dump the pruned relation matrix and metadata to an HDF5 file.

    Layout
    ------
    /matrix          : int32 sparse CSR data (see below)
    /matrix_dense    : int32 dense matrix (rows × cols), stored if small enough
    /atoms           : variable-length UTF-8 strings, one per column
    /atom_index      : JSON string mapping atom_str -> col_index
    /divisor_xs      : int64 array of the four divisor x-coordinates
    /group_order     : scalar int64
    /col_inf         : scalar int64 (column index of ∞, or -1)
    /col_gen0        : scalar int64
    /col_gen1        : scalar int64
    /col_tgt0        : scalar int64
    /col_tgt1        : scalar int64

    Sparse CSR datasets (always written):
    /csr/data        : int32 nonzero values
    /csr/indices     : int32 column indices
    /csr/indptr      : int32 row pointers (len = nrows+1)
    /csr/shape       : int64 [nrows, ncols]

    The dense matrix is also written unless it exceeds 500 MB
    (rows * cols * 4 bytes).  At that point only CSR is stored.

    Usage after loading
    -------------------
        import h5py, numpy as np, json
        with h5py.File("relation_matrix.h5") as f:
            atoms    = [a.decode() for a in f["atoms"]]
            aidx     = json.loads(f["atom_index"][()].decode())
            data     = f["csr/data"][:]
            indices  = f["csr/indices"][:]
            indptr   = f["csr/indptr"][:]
            shape    = tuple(f["csr/shape"][:])
            # reconstruct scipy sparse:
            from scipy.sparse import csr_matrix
            M = csr_matrix((data, indices, indptr), shape=shape)
            # or dense numpy:
            M_dense  = f["matrix_dense"][:]   # if present
    """
    if not _HAS_H5PY:
        raise RuntimeError(
            "dump_matrix_hdf5: h5py is not installed.  "
            "Install with: pip install h5py"
        )

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]

    M_ZZ, atoms, aidx, M_raw, _ = _build_combined_matrix(
        walkers, protected=divisor_xs,
    )

    nrows = M_ZZ.nrows()
    ncols = M_ZZ.ncols()

    _log(f"[dump_matrix_hdf5] matrix is {nrows}×{ncols}  path={path}")

    # Build numpy dense array first (needed for both dense and CSR paths).
    # Use int32 — coefficients are small (−5 … +3).
    M_np = np.array(M_ZZ, dtype=np.int32)

    # CSR via scipy if available, otherwise manual.
    try:
        from scipy.sparse import csr_matrix as _csr
        sp = _csr(M_np)
        csr_data    = sp.data.astype(np.int32)
        csr_indices = sp.indices.astype(np.int32)
        csr_indptr  = sp.indptr.astype(np.int32)
    except ImportError:
        # Manual CSR construction.
        data_list, idx_list, ptr_list = [], [], [0]
        for r in range(nrows):
            for c in range(ncols):
                v = int(M_np[r, c])
                if v != 0:
                    data_list.append(v)
                    idx_list.append(c)
            ptr_list.append(len(data_list))
        csr_data    = np.array(data_list, dtype=np.int32)
        csr_indices = np.array(idx_list,  dtype=np.int32)
        csr_indptr  = np.array(ptr_list,  dtype=np.int32)

    atom_strs   = [str(a).encode("utf-8") for a in atoms]
    atom_idx_js = json.dumps(aidx).encode("utf-8")

    def _col(x):
        c = aidx.get(str(int(x)))
        return np.int64(-1 if c is None else c)

    with h5py.File(path, "w") as f:
        # CSR group — always written.
        g = f.create_group("csr")
        g.create_dataset("data",    data=csr_data)
        g.create_dataset("indices", data=csr_indices)
        g.create_dataset("indptr",  data=csr_indptr)
        g.create_dataset("shape",   data=np.array([nrows, ncols], dtype=np.int64))

        # Dense matrix — skip if too large.
        dense_bytes = nrows * ncols * 4
        if dense_bytes <= 500 * 1024 * 1024:
            f.create_dataset("matrix_dense", data=M_np, compression="gzip", compression_opts=4)
            _log(f"[dump_matrix_hdf5] dense matrix written ({dense_bytes // (1024*1024)} MB)")
        else:
            _log(f"[dump_matrix_hdf5] dense matrix skipped (would be {dense_bytes // (1024*1024)} MB > 500 MB)")

        # Atom labels.
        dt = h5py.special_dtype(vlen=bytes)
        ds = f.create_dataset("atoms", (len(atom_strs),), dtype=dt)
        ds[:] = atom_strs
        f.create_dataset("atom_index",  data=atom_idx_js)

        # Metadata.
        f.create_dataset("divisor_xs",  data=np.array([x0_a, x0_b, x0_c, x0_d], dtype=np.int64))
        f.create_dataset("group_order", data=np.int64(group_order))
        f.create_dataset("col_inf",     data=_col(_INFINITY if _INFINITY in aidx else -1))
        f.create_dataset("col_gen0",    data=_col(x0_a))
        f.create_dataset("col_gen1",    data=_col(x0_b))
        f.create_dataset("col_tgt0",    data=_col(x0_c))
        f.create_dataset("col_tgt1",    data=_col(x0_d))

    _log(f"[dump_matrix_hdf5] written -> {path}")
    return path


def run_all_checks(
    walkers,
    divisor_xs,
    group_order: int,
    known_key: int,
    p: int,
    coeffs=None,
    n_fiber_spot_checks: int = 5,
    dump_path: str = None,
):
    """Run all diagnostics and print a summary verdict.

    Parameters
    ----------
    dump_path
        If given, dump the pruned relation matrix to this HDF5 file before
        running checks.  Useful for offline gauge-row experimentation without
        rerunning the walk.  Requires h5py.
    """
    _log(f"\n{'#' * 70}")
    _log("# DLP INTEGRATION DIAGNOSTICS")
    _log(f"#  walkers       : {len(walkers)}")
    _log(f"#  divisor_xs    : {[int(x) for x in divisor_xs]}")
    _log(f"#  group_order   : {group_order}")
    _log(f"#  known_key     : {known_key}")
    _log(f"#  p             : {p}  (sqrt_p={math.sqrt(p):.2f})")
    _log(f"{'#' * 70}\n")

    results = {}

    if dump_path is not None:
        try:
            dump_matrix_hdf5(walkers, divisor_xs, group_order, path=dump_path)
            results["matrix_dump"] = f"ok -> {dump_path}"
        except Exception as exc:
            _log(f"  [dump_matrix_hdf5 FAILED: {exc}]")
            results["matrix_dump"] = f"failed: {exc}"
            raise

    try:
        ker, atoms, aidx = check_kernel(walkers, group_order, divisor_xs)
        results["kernel"] = "ok" if ker.dimension() <= 1 else f"nullity={ker.dimension()}"
    except Exception as exc:
        _log(f"  [check_kernel FAILED: {exc}]")
        raise

    try:
        viols = check_fiber_sign(walkers, p, coeffs, n_spot_checks=n_fiber_spot_checks)
        results["fiber_sign"] = "ok" if not viols else f"{len(viols)} violations"
    except Exception as exc:
        _log(f"  [check_fiber_sign FAILED: {exc}]")
        raise

    try:
        check_divisor_injection(walkers, divisor_xs)
        results["divisor_injection"] = "ok"
    except Exception as exc:
        _log(f"  [check_divisor_injection FAILED: {exc}]")
        raise

    try:
        kk = check_known_key(walkers, divisor_xs, group_order, known_key)
        ok = kk.get("ok")
        if ok is True:
            results["known_key"] = "ok"
        elif ok is None:
            results["known_key"] = f"underdetermined (nullity={kk.get('nullity', '?')})"
        else:
            viols = kk.get("violations", [])
            if viols:
                results["known_key"] = f"{len(viols)} hard contradiction(s)"
            else:
                results["known_key"] = f"inconsistent (rank test, nullity={kk.get('nullity', '?')})"
    except Exception as exc:
        _log(f"  [check_known_key FAILED: {exc}]")
        raise

    try:
        check_zero_compatibility(walkers, divisor_xs, group_order)
        results["zero_compat"] = "ok"
    except Exception as exc:
        _log(f"  [check_zero_compatibility FAILED: {exc}]")
        raise

    try:
        check_involution_symmetry(walkers)
        results["involution"] = "ok"
    except Exception as exc:
        _log(f"  [check_involution_symmetry FAILED: {exc}]")
        raise

    _log(f"\n{'#' * 70}")
    _log("# DIAGNOSTIC SUMMARY")
    _log(f"{'#' * 70}")
    for name, status in results.items():
        if status == "ok":
            icon = "✓"
        elif status.startswith("underdetermined"):
            icon = "?"
        else:
            icon = "✗"
        _log(f"  {icon}  {name:<25s}  {status}")
    _log(f"{'#' * 70}\n")

    return results

def _build_combined_matrix(walkers, include_step_leaves: bool = False, protected=None):
    """
    Build the combined pruned ZZ relation matrix from all walkers.

    Returns (M_pruned, pruned_atoms, atom_index, M_raw, all_atoms_raw).

    Parameters
    ----------
    walkers
        Iterable of walker objects.
    include_step_leaves
        Forwarded into each walker's relation_matrix().
    protected
        Optional collection of atom labels that must survive pruning,
        even if they become dest-only.
    """
    mats, atom_lists = [], []
    for w in walkers:
        mat, atoms, _ = w.relation_matrix(include_step_leaves=include_step_leaves)
        if mat.nrows() > 0:
            mats.append(mat)
            atom_lists.append(list(atoms))

    assert mats, "All walkers have empty relation matrices — nothing to work with."

    # Union column spaces, preserving the first-seen order.
    all_atoms = list(atom_lists[0])
    atom_set = set(map(str, all_atoms))
    for atms in atom_lists[1:]:
        for a in atms:
            sa = str(a)
            if sa not in atom_set:
                all_atoms.append(a)
                atom_set.add(sa)

    n_cols = len(all_atoms)
    aidx = {str(a): i for i, a in enumerate(all_atoms)}

    rows = []
    for mat, atms in zip(mats, atom_lists):
        cols_src = [aidx[str(a)] for a in atms]
        for r in range(mat.nrows()):
            row = [0] * n_cols
            for c_src, c_dst in enumerate(cols_src):
                row[c_dst] += int(mat[r, c_src])
            rows.append(row)

    M_raw = Matrix(ZZ, rows)

    M_pruned, pruned_atoms, removed = prune_dest_only(
        M_raw,
        all_atoms,
        protected=protected,
    )
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    _log(f"  Combined matrix (pre-prune):  {M_raw.nrows()} rows × {n_cols} cols")
    _log(
        f"  After prune:                  {M_pruned.nrows()} rows × {len(pruned_atoms)} cols"
        f"  ({len(removed)} dest-only atoms removed)"
    )

    return M_pruned, pruned_atoms, pruned_aidx, M_raw, all_atoms

def check_known_key(
    walkers,
    divisor_xs,
    group_order: int,
    known_key: int,
):
    """
    Check whether the supplied known key is consistent with the pruned relation
    matrix using the balanced anchor a[gen0] - a[gen1] = k.

    The balanced anchor respects the conservation law (coeff-sum = 1-1 = 0).
    Single-atom pinning (a[gen0]=1 etc.) violates the conservation and will
    always produce an inconsistent system regardless of the key.

    We test four hypotheses corresponding to sign conventions on the generator
    and target atoms, using k = ±1 for the anchor and key = ±known_key for
    the target log sum.

    Returns a dict with keys:
        ok         : True / False / None
        nullity    : kernel dimension of the homogeneous system
        violations : list of contradicted rows (for hard-contradiction cases)
        best_label : hypothesis label with fewest violations
    """
    _section(f"KNOWN KEY COMPATIBILITY  (key={known_key})")

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]

    M_ZZ, atoms, aidx, _, _ = _build_combined_matrix(
        walkers, protected=divisor_xs,
    )
    n_rows = M_ZZ.nrows()
    n_cols = M_ZZ.ncols()

    F = GF(Integer(group_order))
    M  = M_ZZ.change_ring(F)

    nullity = n_cols - M.rank()

    def col_of(x):
        return aidx.get(str(int(x)))

    inf_col      = aidx.get(_INFINITY)
    gen_col      = col_of(x0_a)
    gen_p_col    = col_of(x0_b)
    tgt_col      = col_of(x0_c)
    tgt_p_col    = col_of(x0_d)

    missing = []
    if inf_col  is None: missing.append("∞")
    if gen_col  is None: missing.append(f"gen_x0={x0_a}")
    if gen_p_col is None: missing.append(f"gen_x1={x0_b}")
    if tgt_col  is None: missing.append(f"tgt_x0={x0_c}")

    if missing:
        _log(f"  ✗  Cannot test known key — required columns missing after prune: {missing}")
        return {"ok": False, "reason": "missing_columns", "missing": missing}

    # Balanced hypotheses: anchor is a[gen0] - a[gen1] = anchor_k (coeff-sum=0).
    # Target log is a[tgt0] + a[tgt1] = tgt_k (we test ±known_key).
    # We fix {∞=0, gen0-gen1=anchor_k} and substitute into every fully-pinned row.
    # Rows with free variables are skipped (inconclusive when underdetermined).
    hypotheses = [
        ("anchor=+1, key=+k",  F(1),  F(known_key)),
        ("anchor=+1, key=-k",  F(1),  F(-known_key)),
        ("anchor=-1, key=+k",  F(-1), F(known_key)),
        ("anchor=-1, key=-k",  F(-1), F(-known_key)),
    ]

    scored = []

    for label, anchor_k, tgt_k in hypotheses:
        # Build a partial assignment for the fully-determined atoms.
        # a[gen0] - a[gen1] = anchor_k  means we can parameterise as
        # a[gen1] = t, a[gen0] = t + anchor_k for free t.
        # Only rows whose support is contained in {inf, gen0, gen1, tgt0, tgt1}
        # can be directly evaluated; everything else has additional free columns.
        pinned = {}
        if inf_col  is not None: pinned[inf_col]   = F(0)
        # We don't assign absolute values to gen0/gen1 — only their difference
        # is fixed.  So rows that contain *only* gen0 or *only* gen1 (but not
        # both) still have a free variable and are skipped.
        # Rows that contain both gen0 and gen1 can be evaluated via the difference.
        if tgt_col  is not None: pinned[tgt_col]   = tgt_k          # a[tgt0]
        if tgt_p_col is not None: pinned[tgt_p_col] = F(0)           # a[tgt1]=0, sum=tgt_k

        fixed_cols = set(pinned.keys()) | ({gen_col, gen_p_col} if gen_col is not None and gen_p_col is not None else set())

        violations = []
        for i in range(n_rows):
            nz = [j for j in range(n_cols) if M[i, j] != 0]
            if not nz:
                continue
            if not all(j in fixed_cols for j in nz):
                continue  # free variables — inconclusive

            # Evaluate row using pinned values; for gen columns use difference.
            has_gen0 = gen_col   in nz
            has_gen1 = gen_p_col in nz
            if (has_gen0 or has_gen1) and not (has_gen0 and has_gen1):
                continue  # only one gen column — free variable via t

            resid = F(0)
            for j in nz:
                if j == gen_col:
                    # a[gen0] = t + anchor_k; coefficient is M[i,gen_col]
                    # t terms cancel when both gen0 and gen1 appear (checked above)
                    resid += M[i, j] * anchor_k
                elif j == gen_p_col:
                    # a[gen1] = t; coefficient is M[i,gen_p_col]
                    # t term cancels with gen0's t term
                    pass
                else:
                    resid += M[i, j] * pinned[j]

            if resid != F(0):
                violations.append(
                    (i, int(resid), [(atoms[j], int(M[i, j])) for j in nz])
                )

        scored.append({
            "label": label,
            "violations": violations,
            "n_violations": len(violations),
        })

    scored.sort(key=lambda d: d["n_violations"])
    best = scored[0]

    _log(f"  Nullity of homogeneous system: {nullity}")
    _log(f"  (nullity=1 means fully determined up to gauge; "
         f"nullity>1 means underdetermined — only direct contradictions are conclusive)\n")
    _log(f"  Anchor convention: a[gen0] - a[gen1] = k  (balanced, coeff-sum=0)")
    _log(f"  Target convention: a[tgt0] + a[tgt1] = known_key (or negated)\n")

    _log("  Hypothesis scan (direct contradictions only):")
    for item in scored:
        status = "OK" if item["n_violations"] == 0 else f"{item['n_violations']} violation(s)"
        _log(f"    {item['label']:<22s} -> {status}")

    if nullity == 1:
        if best["n_violations"] == 0:
            # Full rank test: substitute anchor and solve for free columns.
            anchor_row = vector(F, n_cols)
            anchor_row[gen_col]   = F(1)
            anchor_row[gen_p_col] = F(-1)
            inf_row = vector(F, n_cols)
            inf_row[inf_col] = F(1)

            rows_aug = [M.row(i) for i in range(n_rows)]
            rows_aug.append(inf_row)
            rows_aug.append(anchor_row)
            rhs_aug  = [F(0)] * (n_rows + 1) + [F(1)]

            A_aug = matrix(F, rows_aug)
            b_aug = vector(F, rhs_aug)
            rank_A   = A_aug.rank()
            rank_aug = A_aug.augment(b_aug.column()).rank()

            if rank_A == rank_aug:
                _log(f"\n  ✓  key={known_key} is CONSISTENT (nullity=1, full system consistent).")
                return {"ok": True, "label": best["label"], "nullity": nullity, "violations": []}
            else:
                _log(f"\n  ✗  key={known_key} is INCONSISTENT (nullity=1, rank test fails). "
                     f"rank(A)={rank_A}  rank([A|b])={rank_aug}")
                return {"ok": False, "label": best["label"], "nullity": nullity,
                        "violations": [], "rank_A": rank_A, "rank_aug": rank_aug}
        else:
            _log(f"\n  ✗  key={known_key} INCONSISTENT — {best['n_violations']} contradicted row(s):")
            for row_i, resid, nz_cols in best["violations"][:10]:
                _log(f"       row {row_i:5d}  residual={resid}  atoms={nz_cols}")
            return {"ok": False, "label": best["label"], "nullity": nullity,
                    "violations": best["violations"][:10]}
    else:
        if best["n_violations"] == 0:
            _log(f"\n  ?  key={known_key} is UNVERIFIABLE — system is underdetermined "
                 f"(nullity={nullity}, need nullity=1).\n"
                 f"     No direct contradiction found under any sign convention.\n"
                 f"     Need {nullity - 1} more independent relation rows before "
                 f"this check is conclusive.")
            return {"ok": None, "reason": "underdetermined", "nullity": nullity, "violations": []}
        else:
            _log(f"\n  ✗  key={known_key} has HARD CONTRADICTION (nullity={nullity}) — "
                 f"{best['n_violations']} row(s) with support in fixed columns fail:")
            for row_i, resid, nz_cols in best["violations"][:10]:
                _log(f"       row {row_i:5d}  residual={resid}  atoms={nz_cols}")
            return {"ok": False, "label": best["label"], "nullity": nullity,
                    "violations": best["violations"][:10]}

def _build_combined_matrix(walkers, include_step_leaves: bool = False, protected=None):
    """
    Build the combined pruned ZZ relation matrix from all walkers.

    Returns (M_pruned, pruned_atoms, atom_index, M_raw, all_atoms_raw).

    Parameters
    ----------
    walkers
        Iterable of walker objects.
    include_step_leaves
        Forwarded into each walker's relation_matrix().
    protected
        Optional collection of atom labels that must survive pruning,
        even if they become dest-only.
    """
    mats, atom_lists = [], []
    for w in walkers:
        mat, atoms, _ = w.relation_matrix(include_step_leaves=include_step_leaves)
        if mat.nrows() > 0:
            mats.append(mat)
            atom_lists.append(list(atoms))

    assert mats, "All walkers have empty relation matrices — nothing to work with."

    # Union column spaces, preserving the first-seen order.
    all_atoms = list(atom_lists[0])
    atom_set = set(map(str, all_atoms))
    for atms in atom_lists[1:]:
        for a in atms:
            sa = str(a)
            if sa not in atom_set:
                all_atoms.append(a)
                atom_set.add(sa)

    n_cols = len(all_atoms)
    aidx = {str(a): i for i, a in enumerate(all_atoms)}

    rows = []
    for mat, atms in zip(mats, atom_lists):
        cols_src = [aidx[str(a)] for a in atms]
        for r in range(mat.nrows()):
            row = [0] * n_cols
            for c_src, c_dst in enumerate(cols_src):
                row[c_dst] += int(mat[r, c_src])
            rows.append(row)

    M_raw = Matrix(ZZ, rows)

    M_pruned, pruned_atoms, removed = prune_dest_only(
        M_raw,
        all_atoms,
        protected=protected,
    )
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    _log(f"  Combined matrix (pre-prune):  {M_raw.nrows()} rows × {n_cols} cols")
    _log(
        f"  After prune:                  {M_pruned.nrows()} rows × {len(pruned_atoms)} cols"
        f"  ({len(removed)} dest-only atoms removed)"
    )

    return M_pruned, pruned_atoms, pruned_aidx, M_raw, all_atoms

def check_known_key(
    walkers,
    divisor_xs,
    group_order: int,
    known_key: int,
):
    """
    Check whether the supplied known key is consistent with the relation matrix.

    Pins the known columns (∞, gen_x0, tgt_x0) and tests whether any row
    whose support is entirely within the pinned columns is contradicted.
    Rows with free variables are inconclusive when the system is underdetermined
    and are not counted as violations.

    Returns a dict with keys:
        ok            : True / False / None (None = underdetermined, no contradiction found)
        nullity       : kernel dimension of the homogeneous system
        violations    : list of directly contradicted rows (empty = no hard contradiction)
        best_label    : sign convention with fewest violations
    """
    _section(f"KNOWN KEY COMPATIBILITY  (key={known_key})")

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]

    M_ZZ, atoms, aidx, _, _ = _build_combined_matrix(
        walkers,
        protected=divisor_xs,
    )
    n_rows = M_ZZ.nrows()
    n_cols = M_ZZ.ncols()

    F = GF(Integer(group_order))
    M = M_ZZ.change_ring(F)

    # Compute nullity of the homogeneous system once.
    nullity = n_cols - M.rank()

    def col_of(x):
        return aidx.get(str(int(x)))

    inf_col  = aidx.get(_INFINITY)
    gen_col  = col_of(x0_a)
    tgt_col  = col_of(x0_c)

    missing = []
    if inf_col is None:
        missing.append("∞")
    if gen_col is None:
        missing.append(f"gen_x0={x0_a}")
    if tgt_col is None:
        missing.append(f"tgt_x0={x0_c}")

    if missing:
        _log(f"  ✗  Cannot test known key — required columns missing after prune: {missing}")
        return {"ok": False, "reason": "missing_columns", "missing": missing}

    hypotheses = [
        ("gen=+1, tgt=+k",  1,  1),
        ("gen=+1, tgt=-k",  1, -1),
        ("gen=-1, tgt=+k", -1,  1),
        ("gen=-1, tgt=-k", -1, -1),
    ]

    scored = []

    for label, gsgn, tsgn in hypotheses:
        fixed = {
            inf_col: F(0),
            gen_col: F(gsgn),
            tgt_col: F(tsgn * known_key),
        }
        fixed_cols = set(fixed.keys())

        # Only rows whose entire nonzero support falls within fixed_cols
        # can be directly tested — they have no free variables.
        violations = []
        for i in range(n_rows):
            nz = [j for j in range(n_cols) if M[i, j] != 0]
            if not nz:
                continue
            if not all(j in fixed_cols for j in nz):
                continue  # has free variables — inconclusive
            resid = sum(M[i, j] * fixed[j] for j in nz)
            if resid != 0:
                violations.append(
                    (i, int(resid), [(atoms[j], int(M[i, j])) for j in nz])
                )

        scored.append({
            "label": label,
            "violations": violations,
            "n_violations": len(violations),
        })

    scored.sort(key=lambda d: d["n_violations"])
    best = scored[0]

    _log(f"  Nullity of homogeneous system: {nullity}")
    _log(f"  (nullity=1 means fully determined up to gauge; "
         f"nullity>1 means underdetermined — only direct contradictions are conclusive)\n")

    _log("  Hypothesis scan (direct contradictions only):")
    for item in scored:
        status = "OK" if item["n_violations"] == 0 else f"{item['n_violations']} violation(s)"
        _log(f"    {item['label']:<18s} -> {status}")

    if nullity == 1:
        # System is fully determined up to gauge. Consistency test is conclusive.
        if best["n_violations"] == 0:
            # Do the full rank test to confirm.
            fixed = {inf_col: F(0), gen_col: F(1), tgt_col: F(known_key)}
            fixed_cols = set(fixed.keys())
            free_cols = [j for j in range(n_cols) if j not in fixed_cols]
            rhs = vector(F, n_rows)
            for j, val in fixed.items():
                if val != 0:
                    rhs -= M.column(j) * val
            A = M.matrix_from_columns(free_cols) if free_cols else matrix(F, n_rows, 0)
            rank_A = A.rank()
            rank_aug = A.augment(rhs.column()).rank()
            if rank_A == rank_aug:
                _log(f"\n  ✓  key={known_key} is CONSISTENT (nullity=1, no contradictions, "
                     f"full system consistent).")
                return {"ok": True, "label": best["label"], "nullity": nullity,
                        "violations": []}
            else:
                _log(f"\n  ✗  key={known_key} is INCONSISTENT (nullity=1, rank test fails). "
                     f"rank(A)={rank_A}  rank([A|b])={rank_aug}")
                return {"ok": False, "label": best["label"], "nullity": nullity,
                        "violations": [], "rank_A": rank_A, "rank_aug": rank_aug}
        else:
            _log(f"\n  ✗  key={known_key} is INCONSISTENT — "
                 f"{best['n_violations']} directly contradicted row(s):")
            for row_i, resid, nz_cols in best["violations"][:10]:
                _log(f"       row {row_i:5d}  residual={resid}  atoms={nz_cols}")
            return {"ok": False, "label": best["label"], "nullity": nullity,
                    "violations": best["violations"][:10]}

    else:
        # Underdetermined. Only hard contradictions (fully-fixed rows) are conclusive.
        if best["n_violations"] == 0:
            _log(f"\n  ?  key={known_key} is UNVERIFIABLE — system is underdetermined "
                 f"(nullity={nullity}, need nullity=1).\n"
                 f"     No direct contradiction found under any sign convention.\n"
                 f"     Need {nullity - 1} more independent relation rows before "
                 f"this check is conclusive.")
            return {"ok": None, "reason": "underdetermined", "nullity": nullity,
                    "violations": []}
        else:
            _log(f"\n  ✗  key={known_key} has a HARD CONTRADICTION even in the "
                 f"underdetermined case — {best['n_violations']} row(s) whose support "
                 f"is entirely within fixed columns fail:\n"
                 f"     (This is a genuine error regardless of nullity.)")
            for row_i, resid, nz_cols in best["violations"][:10]:
                _log(f"       row {row_i:5d}  residual={resid}  atoms={nz_cols}")
            return {"ok": False, "label": best["label"], "nullity": nullity,
                    "violations": best["violations"][:10]}
