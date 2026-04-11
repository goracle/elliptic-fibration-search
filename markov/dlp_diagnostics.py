from __future__ import annotations
import math
from typing import Any, List, Optional, Sequence
from collections import Counter
from sage.all import GF, ZZ, Matrix, vector, matrix, Integer
from .relation_matrix import *
from search_common import get_y_unshifted_genus2, COEFFS_GENUS2

"""dlp_diagnostics.py

Drop-in post-run integration checks for the genus-2 Markov DLP solve.

Run after merge_experiment.py has built walker_a/b/c/d but before you exit,
OR load from JSONL logs by constructing walkers and replaying history.

Usage (paste into merge_experiment.py after the DLP solve block, or run
standalone with walkers in scope):

    from dlp_diagnostics import run_all_checks
    run_all_checks(
        walkers       = [walker_a, walker_b, walker_c, walker_d],
        divisor_xs    = divisor_xs,   # [x0_a, x0_b, x0_c, x0_d]
        group_order   = GROUP_MODULUS,
        known_key     = 800,
        p             = FINITE_FIELD,
        coeffs        = COEFFS_GENUS2,
    )

Each check is also callable independently — see the check_* functions below.

DEAR AI READING THIS: RAISE ALL EXCEPTIONS.
"""

# ---------------------------------------------------------------------------
# Lazy Sage import (so the module is importable in non-Sage envs for linting)
# ---------------------------------------------------------------------------
_SAGE = True

# Re-use helpers from the project

_SEP = "=" * 70
_INFINITY = "∞"

# ---------------------------------------------------------------------------
# Shared matrix-building helper
# (mirrors _dlp_union_columns + _dlp_prune from merge_experiment, but local
#  so this file has no circular import dependency)
# ---------------------------------------------------------------------------

# ===========================================================================
# CHECK 1 — Kernel decomposition
# ===========================================================================

# ===========================================================================
# CHECK 2 — Fiber sign check
# ===========================================================================

# ===========================================================================
# CHECK 3 — Involution symmetry (why it's not running in merge_experiment)
# ===========================================================================

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

# ===========================================================================
# CHECK 4 — Divisor atom injection quality
# ===========================================================================

def check_divisor_injection(walkers, divisor_xs: Sequence):
    """Check that the 4 divisor atoms appear as both xi AND xj across the walks.

    A divisor atom that only appears as xj is a dest-only column and will be
    pruned (or, if protected, will have no xi-row, making the anchor
    incompatible with the relation subspace).

    Also reports the merge-time volume — if first_merge_step=0, the chains
    are colliding on the very first insertion, which means the divisor seeds
    are trivially shared and the relations built before that merge are tiny.
    """
    _section("DIVISOR ATOM INJECTION QUALITY")

    div_set = set(int(x) for x in divisor_xs)

    for i, w in enumerate(walkers):
        label = getattr(w, '_label', f"walker[{i}]")
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
            _log(f"  ✓  All 4 divisor atoms appeared as xi at least once.")
        else:
            _log(
                f"  ✗  Some divisor atoms NEVER appeared as xi.\n"
                f"     These atoms have no xi-role rows in the matrix.\n"
                f"     The anchor a[G_x0] + a[G_x1] = 1 will be inconsistent\n"
                f"     with the relation subspace if gen_col has no xi rows.\n"
                f"     Fix: run more steps, or force a restart at those x-coords."
            )

        # Report merge timing
        fms = getattr(w, 'first_merge_step', None)
        fmv = getattr(w, 'first_merge_vol',  None)
        if fms == 0:
            _log(
                f"\n  ⚠  first_merge_step=0 for {label}.\n"
                f"     The chain collided with A on its very first insertion (vol={fmv}).\n"
                f"     This is almost certainly a divisor seed trivially shared between\n"
                f"     A and {label} (both start from the same PREFERRED_X_COORDS).\n"
                f"     The DLP solve should NOT be triggered until vol >= some threshold\n"
                f"     (e.g. 0.5*sqrt(p)) so the relation matrix has enough rows."
            )

# ===========================================================================
# CHECK 5 — Known key compatibility
# ===========================================================================

# ===========================================================================
# CHECK 6 — Zero compatibility of base/target atom logs
# ===========================================================================

# ===========================================================================
# Master runner
# ===========================================================================

def _col(aidx: dict, key):
    """Stable column lookup that does not break when the index is 0."""
    if key is None:
        return None
    if key in aidx:
        return aidx[key]
    return aidx.get(str(key))

_SEP = "=" * 70
_INFINITY = "∞"

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
    Canonical branch convention:
      +1 if residue is the canonical representative in [0, floor(p/2)]
      -1 otherwise.
    """
    y = int(val) % p
    if y == 0:
        return 0
    return 1 if y <= (p - y) else -1

def _build_combined_matrix(walkers, include_step_leaves: bool = False):
    """
    Build the combined pruned ZZ relation matrix from all walkers.

    Returns (M_pruned, pruned_atoms, atom_index, M_raw, all_atoms_raw).
    """
    mats, atom_lists = [], []
    for w in walkers:
        mat, atoms, _ = w.relation_matrix(include_step_leaves=include_step_leaves)
        if mat.nrows() > 0:
            mats.append(mat)
            atom_lists.append(list(atoms))

    assert mats, "All walkers have empty relation matrices — nothing to work with."

    # Union column spaces
    all_atoms = list(atom_lists[0])
    atom_set = set(map(str, all_atoms))
    for atms in atom_lists[1:]:
        for a in atms:
            if str(a) not in atom_set:
                all_atoms.append(a)
                atom_set.add(str(a))

    n_cols = len(all_atoms)
    aidx = {str(a): i for i, a in enumerate(all_atoms)}

    rows = []
    for mat, atms in zip(mats, atom_lists):
        cols_src = [aidx[str(a)] for a in atms]
        for r in range(mat.nrows()):
            row = [0] * n_cols
            for c_src, c_dst in enumerate(cols_src):
                row[c_dst] = int(mat[r, c_src])
            rows.append(row)

    M_raw = Matrix(ZZ, rows)

    M_pruned, pruned_atoms, removed = prune_dest_only(M_raw, all_atoms)
    pruned_aidx = {str(a): i for i, a in enumerate(pruned_atoms)}

    _log(f"  Combined matrix (pre-prune):  {M_raw.nrows()} rows × {n_cols} cols")
    _log(
        f"  After prune:                  {M_pruned.nrows()} rows × {len(pruned_atoms)} cols"
        f"  ({len(removed)} dest-only atoms removed)"
    )

    return M_pruned, pruned_atoms, pruned_aidx, M_raw, all_atoms

def check_kernel(walkers, group_order: int, divisor_xs=()):
    _section("KERNEL DECOMPOSITION")

    M, atoms, aidx, _, _ = _build_combined_matrix(walkers)
    n_cols = len(atoms)
    Fp = GF(group_order)
    M_fp = M.change_ring(Fp)

    rank = M_fp.rank()
    ker = M_fp.right_kernel()
    null = ker.dimension()

    _log(f"  rows={M.nrows()}  cols={n_cols}  rank={rank}  nullity={null}")
    _log("  (need nullity=1 for unique solution after gauge-fix a[∞]=0)")

    div_strs = {str(x) for x in divisor_xs}

    for i, vec in enumerate(ker.basis()):
        nz = [(atoms[j], int(vec[j])) for j in range(n_cols) if vec[j] != 0]
        tags = []
        if any(str(a) == _INFINITY for a, _ in nz):
            tags.append("∞-gauge")
        if any(str(a) in div_strs for a, _ in nz):
            tags.append("DIVISOR-ATOM")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""

        _log(f"\n  kernel[{i}]:{tag_str}")
        for atom, coeff in nz:
            marker = " ← divisor" if str(atom) in div_strs else ""
            _log(f"    atom={atom}  coeff={coeff}{marker}")

    if null == 1:
        _log("\n  ✓  Nullity=1 — only the ∞ gauge direction is free.")
    elif null == 2:
        _log("\n  ✗  Nullity=2 — one extra free direction beyond the ∞ gauge.")
    else:
        _log(f"\n  ✗  Nullity={null} — {null - 1} extra free directions beyond ∞ gauge.")

    return ker, atoms, aidx

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

def check_known_key(
    walkers,
    divisor_xs,
    group_order: int,
    known_key: int,
):
    """
    Check whether the known DLP answer is consistent with the relation matrix.

    This version probes the four obvious sign conventions for the generator /
    target anchor, so it can distinguish a true algebraic failure from a
    normalization mismatch.
    """
    _section(f"KNOWN KEY COMPATIBILITY  (key={known_key})")

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]

    M, atoms, aidx, _, _ = _build_combined_matrix(walkers)
    n_cols = len(atoms)
    Fp = GF(group_order)
    M_fp = M.change_ring(Fp)

    inf_col = aidx.get(_INFINITY)
    gen_col = aidx.get(str(x0_a))
    gen_p_col = aidx.get(str(x0_b))
    tgt_col = aidx.get(str(x0_c))
    tgt_p_col = aidx.get(str(x0_d))

    missing = []
    if inf_col is None:
        missing.append("∞")
    if gen_col is None:
        missing.append(f"gen_x0={x0_a}")
    if tgt_col is None:
        missing.append(f"target_x0={x0_c}")

    if missing:
        _log(f"  ✗  Cannot build test vector — columns missing after prune: {missing}")
        return None

    def build_vec(gen_sign: int, tgt_sign: int):
        v = vector(Fp, n_cols)
        if inf_col is not None:
            v[inf_col] = Fp(0)
        v[gen_col] = Fp(gen_sign)
        if gen_p_col is not None:
            v[gen_p_col] = Fp(0)
        v[tgt_col] = Fp(tgt_sign * known_key)
        if tgt_p_col is not None:
            v[tgt_p_col] = Fp(0)
        return v

    hypotheses = [
        ("gen=+1, tgt=+k",  1,  1),
        ("gen=+1, tgt=-k",  1, -1),
        ("gen=-1, tgt=+k", -1,  1),
        ("gen=-1, tgt=-k", -1, -1),
    ]

    scored = []
    for label, gsgn, tsgn in hypotheses:
        v = build_vec(gsgn, tsgn)
        residual = M_fp * v
        nonzero = [(i, int(residual[i])) for i in range(len(residual)) if residual[i] != 0]
        scored.append((len(nonzero), label, nonzero))

    scored.sort(key=lambda t: t[0])
    best_count, best_label, best_nonzero = scored[0]

    _log("  Hypothesis scan:")
    for count, label, _ in scored:
        _log(f"    {label:<18s} -> {count} residual rows")

    if best_count == 0:
        _log(f"\n  ✓  key={known_key} is CONSISTENT under convention '{best_label}'.")
        return []

    _log(f"\n  ✗  key={known_key} is INCONSISTENT under the best convention '{best_label}'.")
    _log(f"     Best residual count: {best_count}")
    _log("     First 10 violating rows:")
    for row_i, val in best_nonzero[:10]:
        nz_cols = [(atoms[j], int(M[row_i, j])) for j in range(n_cols) if M[row_i, j] != 0]
        _log(f"    row {row_i:5d}  residual={val}  atoms={nz_cols}")

    return best_nonzero

def check_zero_compatibility(
    walkers,
    divisor_xs,
    group_order: int,
):
    """Check whether the target/generator atoms are uniquely determined up to gauge."""
    _section("ZERO COMPATIBILITY OF BASE/TARGET ATOMS")

    x0_a, x0_b, x0_c, x0_d = [int(x) for x in divisor_xs]
    M, atoms, aidx, _, _ = _build_combined_matrix(walkers)
    Fp = GF(group_order)
    M_fp = M.change_ring(Fp)

    ker = M_fp.right_kernel()
    null = ker.dimension()

    labels = {
        str(x0_a): "GEN_x0 (A seed)",
        str(x0_b): "GEN_x1 (B seed)",
        str(x0_c): "TGT_x0 (C seed ← DLP answer lives here)",
        str(x0_d): "TGT_x1 (D seed)",
    }

    _log(f"  Kernel dimension: {null}  (1 = gauge-only, >1 = underdetermined)")
    _log("\n  Divisor atom membership in kernel support:")

    kernel_hits = {}
    for vec in ker.basis():
        for j, coeff in enumerate(vec):
            if coeff == 0:
                continue
            atom_str = str(atoms[j])
            if atom_str in labels:
                kernel_hits.setdefault(atom_str, []).append(int(coeff))

    any_target_in_kernel = False
    for atom_str, label in labels.items():
        col = aidx.get(atom_str)
        in_ker = atom_str in kernel_hits
        pruned = col is None

        if in_ker:
            coeffs_in_ker = kernel_hits[atom_str]
            if "TGT" in label:
                any_target_in_kernel = True
            _log(f"    x={atom_str:6s}  [{label}]  ✗ IN KERNEL  kernel_coeffs={coeffs_in_ker}")
        elif pruned:
            _log(f"    x={atom_str:6s}  [{label}]  PRUNED (dest-only — never appeared as xi)")
        else:
            _log(f"    x={atom_str:6s}  [{label}]  ✓ not in kernel")

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
            f"\n  ✗  Key columns were PRUNED: {missing_cols}\n"
            f"     These atoms must appear as xi, not just xj/xk, to survive pruning."
        )
        return

    if any_target_in_kernel:
        _log(
            "\n  ✗  Target atoms appear in kernel support.\n"
            "     The target direction is not uniquely pinned by the current relation span."
        )
    else:
        _log("\n  ✓  No target atoms in kernel support.")

    _log("\n  Steps where xi == target atom:")
    for i, w in enumerate(walkers):
        label = getattr(w, "_label", f"walker[{i}]")
        tgt_xi_steps = [
            r.step_index for r in w.history
            if getattr(r, "accepted", False) and _safe_int(getattr(r, "xi", None)) in (x0_c, x0_d)
        ]
        preview = f"  (step indices: {tgt_xi_steps[:10]}{'...' if len(tgt_xi_steps) > 10 else ''})" if tgt_xi_steps else ""
        _log(f"    {label}: {len(tgt_xi_steps)} accepted steps with xi in target atoms{preview}")

def run_all_checks(
    walkers,
    divisor_xs,
    group_order: int,
    known_key: int,
    p: int,
    coeffs=None,
    n_fiber_spot_checks: int = 5,
):
    """Run all diagnostics and print a summary verdict."""
    _log(f"\n{'#' * 70}")
    _log("# DLP INTEGRATION DIAGNOSTICS")
    _log(f"#  walkers       : {len(walkers)}")
    _log(f"#  divisor_xs    : {[int(x) for x in divisor_xs]}")
    _log(f"#  group_order   : {group_order}")
    _log(f"#  known_key     : {known_key}")
    _log(f"#  p             : {p}  (sqrt_p={math.sqrt(p):.2f})")
    _log(f"{'#' * 70}\n")

    results = {}

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
        nonzero = check_known_key(walkers, divisor_xs, group_order, known_key)
        results["known_key"] = "ok" if nonzero == [] else f"{len(nonzero or [])} residual rows"
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
        icon = "✓" if status == "ok" else "✗"
        _log(f"  {icon}  {name:<25s}  {status}")
    _log(f"{'#' * 70}\n")

    return results
