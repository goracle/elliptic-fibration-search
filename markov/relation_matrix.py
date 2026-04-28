from __future__ import annotations
import types
from collections import Counter
from sage.all import Matrix, ZZ, QQ, GF
from typing import Any, List, Optional, Sequence, Tuple

"""relation_matrix.py

Encode the Markov walk history as an integer divisor-relation matrix and
compute its rank.

Each accepted step produces a principal divisor relation of the form

    (d-2)·[x_src] + [x_step] + [x_res] - d·[∞] = 0          (curve degree d, default 5)

The columns of the matrix are the **d1 atoms**: the unique finite
x-coordinates that appear in any role (x_src / x_step / x_res) across the full set of
accepted, complete relations.  The point at infinity is appended as the last
column.

Each row encodes one relation:
    col[x_src]  += (d - 2)*
    col[x_step]  += 1*
    col[x_res]  += 1*
    col[∞]   += -d
*=not always

The matrix lives over ZZ.  Rank is computed via Sage's exact arithmetic
(Matrix_integer_dense or Matrix_rational_dense, depending on what Sage gives
us for the small-coefficient case).

Usage
-----
After running the walker:

    from relation_matrix import build_relation_matrix, print_relation_matrix_summary

    mat, atoms, used = build_relation_matrix(walker.history, curve_degree=5)
    print_relation_matrix_summary(mat, atoms, used)

Or attach it as a method:

    # monkey-patch
    from relation_matrix import build_relation_matrix, print_relation_matrix_summary
    import types

    def _walker_relation_matrix(self, **kw):
        return build_relation_matrix(self.history, curve_degree=self.config.curve_degree, **kw)

    Genus2MetropolisWalker.relation_matrix = _walker_relation_matrix

Then just call walker.relation_matrix().
"""
# DEAR AI READING THIS:  RAISE ALL EXCEPTIONS
# DEAR AI READING THIS:  all imports are available

_SAGE_AVAILABLE = True
# ---------------------------------------------------------------------------
# Try to import Sage.  If we are not inside a Sage session, fall back to a
# pure-Python integer matrix so the module is still importable for testing.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_INFINITY_SENTINEL = "∞"

def attach_to_walker(walker_class) -> None:
    """Monkey-patch build_relation_matrix + summary onto a walker class.

    After calling this once:

        attach_to_walker(Genus2MetropolisWalker)

    you can do:

        mat, atoms, used = walker.relation_matrix()
        walker.print_relation_summary()
    """

    def _relation_matrix(self, **kwargs):
        cd = getattr(getattr(self, "config", None), "curve_degree", 5)
        return build_relation_matrix(self.history, curve_degree=cd, **kwargs)

    def _print_relation_summary(self, **kwargs):
        mat, atoms, used = self.relation_matrix()
        print_relation_matrix_summary(mat, atoms, used, **kwargs)

    walker_class.relation_matrix = _relation_matrix
    walker_class.print_relation_summary = _print_relation_summary

def _get(rec, key):
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)

def verify_relation_is_principal(rec, curve, p) -> dict:
    """Verify that a stored relation record encodes a principal divisor on Jac(C).

    A relation record stores atoms = [x_src]*mult + [x_step] + [x_res] + extra_roots,
    all as x-coordinates only.  To verify the relation is principal we must assign
    y-signs to each atom, lift the x-coordinates to curve points (x, ±y), and check
    that their sum in Jac(C) equals zero.

    x_src appears with multiplicity > 1, so its y-sign is fixed once; the sign
    choices that matter are one per *distinct non-src* x-value (x_step, x_res,
    extra_roots).  In the generic degree-5 case that is 2 free signs → 4 candidates.
    We try all 2^k combinations and return the first one that sums to zero, or report
    failure.

    Parameters
    ----------
    rec   : RelationRecord (or dict with same fields)
    curve : SageMath HyperellipticCurve object, defined over GF(p)
    p     : field characteristic (int), used to lift x-coordinates into GF(p)

    Returns
    -------
    dict with keys:
        'ok'          : bool   — True if any sign assignment makes the sum zero
        'signs'       : dict   — {x_val: sign (+1/-1)} for the winning assignment,
                                 or None on failure
        'n_tried'     : int    — number of sign combinations attempted
        'zero_sum'    : bool   — same as 'ok'
        'msg'         : str    — human-readable summary
    """
    from sage.all import GF, ZZ, PolynomialRing
    import itertools

    atoms_raw = _get(rec, "atoms") or []
    x_src = _get(rec, "x_src")
    if not atoms_raw:
        return {"ok": False, "signs": None, "n_tried": 0, "zero_sum": False,
                "msg": "no atoms in record"}

    Fp = GF(p)
    J = curve.jacobian()(Fp)
    f, _ = curve.hyperelliptic_polynomials()

    def lift_point(x_fp, sign):
        """Return a Jacobian divisor [(x, sign*y)] - [∞], or None if x not on curve."""
        y2 = f(x_fp)
        if y2 == 0:
            # Ramification point: y=0, sign is moot.
            return J(curve.lift_x(x_fp))
        sq = y2.sqrt(extend=False, all=True)
        if not sq:
            return None  # x not on curve over Fp
        y_can = min(sq, key=lambda v: int(v))  # canonical = smaller representative
        y = y_can if sign >= 0 else -y_can
        return J(curve(x_fp, y))

    # Build multiplicity-counted list of x-values.
    atom_counter = Counter(Fp(a) for a in atoms_raw)

    # x_src in Fp.
    x_src_fp = Fp(x_src) if x_src is not None else None

    # Identify atoms whose sign is free vs. fixed.
    # x_src's sign: if src multiplicity >= 2, one sign choice applies to all copies,
    # so it is *one* free bit.  But we only need to flip the non-src atoms because the
    # src sign cancels algebraically only if src_mult is even — for odd src_mult it's
    # also a free bit.  We brute-force *all* distinct x-values to keep this general.
    distinct_xs = list(atom_counter.keys())

    # For each distinct x, try sign +1 or -1; enumerate all 2^k combos.
    n_distinct = len(distinct_xs)
    best = None
    n_tried = 0

    for signs_tuple in itertools.product([1, -1], repeat=n_distinct):
        sign_map = dict(zip(distinct_xs, signs_tuple))
        n_tried += 1

        total = J(0)
        failed = False
        for x_fp, mult in atom_counter.items():
            pt = lift_point(x_fp, sign_map[x_fp])
            if pt is None:
                failed = True
                break
            total += mult * pt

        if failed:
            continue

        if total == J(0):
            best = sign_map
            break

    ok = best is not None
    # Convert sign_map keys back to plain Python ints for readability.
    signs_out = {int(x): s for x, s in best.items()} if best else None
    msg = (f"principal ✓  signs={signs_out}  n_tried={n_tried}"
           if ok else
           f"NOT principal ✗  n_tried={n_tried}  atoms={[int(a) for a in atoms_raw]}")
    return {"ok": ok, "signs": signs_out, "n_tried": n_tried, "zero_sum": ok, "msg": msg}

def verify_history_relations(history, curve, p, *, accepted_only=True, verbose=True) -> dict:
    """Run verify_relation_is_principal on every accepted record in history.

    Parameters
    ----------
    history      : list of RelationRecord
    curve        : HyperellipticCurve over GF(p)
    p            : int, field characteristic
    accepted_only: if True, skip non-accepted records
    verbose      : if True, print per-record results for failures and a summary

    Returns
    -------
    dict with keys:
        'n_checked'  : int
        'n_ok'       : int
        'n_fail'     : int
        'failures'   : list of (step_index, msg) for failed records
    """
    n_checked = 0
    n_ok = 0
    failures = []

    for rec in history:
        if accepted_only and not _get(rec, "accepted"):
            continue
        atoms = _get(rec, "atoms") or []
        if not atoms:
            continue

        step_idx = _get(rec, "step_index")
        result = verify_relation_is_principal(rec, curve, p)
        n_checked += 1
        if result["ok"]:
            n_ok += 1
        else:
            failures.append((step_idx, result["msg"]))
            if verbose:
                print(f"  [verify] FAIL  step={step_idx}  {result['msg']}")

    n_fail = len(failures)
    if verbose:
        print(f"\n[verify_history_relations] checked={n_checked}  ok={n_ok}  fail={n_fail}")
        if n_fail == 0:
            print("  All relations verified as principal divisors ✓")

    return {"n_checked": n_checked, "n_ok": n_ok, "n_fail": n_fail, "failures": failures}

def build_relation_matrix2(
    history: Sequence[Any],
    *,
    curve_degree: int = 5,
    include_infinity: bool = True,
    accepted_only: bool = True,
    require_xk: bool = True,
    include_step_leaves: bool = True,
) -> Tuple[Any, List[Any], List[Any]]:
    """Build the integer divisor-relation matrix from a walker history.

    This version explodes step-level leaf data (candidate_pool) into full rows,
    generating a divisor relation for EVERY candidate found during the step.
    """
    assert require_xk
    inf_coeff = -curve_degree

    atom_index: dict[Any, int] = {}
    used_records: List[Any] = []
    skipped_degenerate = 0

    def _iter_step_leaves(rec):
        """Yield extra leaf atoms stored in step/search payloads."""
        if not include_step_leaves:
            return

        step = _get(rec, "step")
        if isinstance(step, dict):
            for key in ("candidate_xs", "found_xs"):
                xs = step.get(key)
                if xs is None:
                    continue
                if isinstance(xs, dict):
                    xs = xs.keys()
                try:
                    for x in xs:
                        if x is not None:
                            yield x
                except TypeError:
                    if xs is not None:
                        yield xs

        pool = _get(rec, "candidate_pool")
        if pool:
            for cand in pool:
                if isinstance(cand, dict):
                    # Added 'x_res' so x_res candidates are guaranteed columns
                    for key in ("x_step", "x", "candidate_x", "x_value", "x_res"):
                        x = cand.get(key)
                        if x is not None:
                            yield x
                elif cand is not None:
                    yield cand

        sel = _get(rec, "selected_candidate")
        if isinstance(sel, dict):
            for key in ("x_step", "x", "candidate_x", "x_value", "x_res"):
                x = sel.get(key)
                if x is not None:
                    yield x

    # Pass 1a: register atoms from ALL accepted non-involution records BEFORE filtering.
    #
    # Involution/free records are excluded here too: their x_step values are just
    # T-images of existing walk atoms (x_step<->x_res swap), so they add no new columns.
    # Excluding them keeps atom_index clean and the column count honest.
    for rec in history:
        if accepted_only and not _get(rec, "accepted"):
            continue

        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            continue

        x_src = _get(rec, "x_src")
        x_step = _get(rec, "x_step")
        x_res = _get(rec, "x_res")

        # Need at least one of x_src/x_step to be meaningful.
        if x_src is None and x_step is None:
            continue

        # Register primary atoms unconditionally (no degenerate skip here).
        for x in (x_src, x_step, x_res):
            if x is not None and x not in atom_index:
                atom_index[x] = len(atom_index)

        # Register any extra non-x_src roots (3+ root fibers).
        for x in (_get(rec, "extra_roots") or []):
            if x is not None and x not in atom_index:
                atom_index[x] = len(atom_index)

        # Step-level leaf atoms, if available.
        for x in _iter_step_leaves(rec):
            if x is not None and x not in atom_index:
                atom_index[x] = len(atom_index)

    # Pass 1b: build used_records with full validity checks.
    # Involution/free records are excluded: they are proven algebraically identical
    # to existing walk rows (the relation is symmetric in x_step and x_res, so T(x_step)=x_res
    # just produces the same row with x_step and x_res swapped).  Including them adds
    # zero rank and pollutes the atom list with duplicate columns.
    for rec in history:
        if accepted_only and not _get(rec, "accepted"):
            continue

        step = _get(rec, "step")
        if isinstance(step, dict) and step.get("source") == "involution_closure":
            continue

        x_src = _get(rec, "x_src")
        x_step = _get(rec, "x_step")

        if x_src is None or x_step is None:
            continue

        if x_src == x_step:
            skipped_degenerate += 1
            continue

        used_records.append(rec)

    if skipped_degenerate:
        print(f"[relation_matrix] Skipped {skipped_degenerate} degenerate relations where x_src == x_step.")

    if not used_records:
        print("[relation_matrix] No usable relations found in history.")
        atoms: List[Any] = list(atom_index.keys())
        if include_infinity:
            atoms.append(_INFINITY_SENTINEL)
        return Matrix(ZZ, 0, len(atoms)), atoms, []

    finite_atoms: List[Any] = sorted(atom_index.keys(), key=lambda a: atom_index[a])
    atoms: List[Any] = list(finite_atoms)

    if include_infinity:
        atoms.append(_INFINITY_SENTINEL)
        inf_col = len(finite_atoms)
    else:
        inf_col = None

    n_cols = len(atoms)

    # Pass 2: build rows.
    rows: List[List[int]] = []
    n_involution_rows = 0
    for rec in used_records:
        rows_before_this_rec = len(rows)
        x_src = _get(rec, "x_src")
        if x_src is None or x_src not in atom_index:
            continue

        # ── Primary record: emit row directly from rec.atoms (canonical flat list) ──
        _primary_atoms = list(_get(rec, "atoms") or [])
        if _primary_atoms:
            if len(_primary_atoms) != curve_degree:
                raise ValueError(
                    f"[relation_matrix] degree invariant violated at "
                    f"step={_get(rec, 'step_index')}: "
                    f"len(atoms)={len(_primary_atoms)} != curve_degree={curve_degree} "
                    f"(x_src={x_src!r})"
                )
            _cnt = Counter(_primary_atoms)
            primary_row = [0] * n_cols
            for _atom, _c in _cnt.items():
                if _atom not in atom_index:
                    raise AssertionError(
                        f"[relation_matrix] BUG: atom={_atom!r} from rec.atoms "
                        f"not in atom_index (x_src={x_src!r}).  Pass-1a missed it."
                    )
                primary_row[atom_index[_atom]] += _c
            if inf_col is not None:
                primary_row[inf_col] += inf_coeff
            _row_sum = sum(primary_row)
            if _row_sum != 0:
                raise AssertionError(
                    f"[relation_matrix] sum-to-zero violated: sum={_row_sum} "
                    f"atoms={_primary_atoms}  x_src={x_src!r}"
                )
            rows.append(primary_row)

        # ── Pool candidates (step leaves): build rows from x_step/x_res/src_mult ──
        # Pool candidates are raw search dicts; they carry src_mult from the fiber
        # but no atoms list.  We reconstruct the row from x_step/x_res/extra_roots and
        # the candidate's own src_mult.  Signs (yj_sign/yk_sign) are ignored here —
        # the matrix cares only about x-coordinates (atom indices).
        if include_step_leaves:
            pool = _get(rec, "candidate_pool")
            if pool:
                seen_pairs: set = set()
                for cand in pool:
                    if isinstance(cand, dict):
                        c_xj = next((cand[k] for k in ("x_step", "x", "candidate_x", "x_value") if k in cand and cand[k] is not None), None)
                        c_xk = cand.get("x_res")
                        c_extra = list(cand.get("extra_roots") or [])
                        c_xi_mult = int(cand.get("src_mult", -1))
                    elif cand is not None:
                        c_xj, c_xk, c_extra, c_xi_mult = cand, None, [], -1
                    else:
                        continue

                    if c_xj is None or c_xj == x_src:
                        continue
                    if require_xk and c_xk is None:
                        continue
                    if c_xk == "∞":
                        continue
                    if c_xi_mult < 0:
                        continue  # no fiber-derived multiplicity — skip
                    if c_xj not in atom_index:
                        continue  # leaf not in factor base — skip silently

                    all_non_xi = [c_xj]
                    if c_xk is not None:
                        all_non_xi.append(c_xk)
                    all_non_xi.extend(c_extra)
                    pair_key = frozenset(all_non_xi)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    row = [0] * n_cols
                    row[atom_index[x_src]] += c_xi_mult
                    row[atom_index[c_xj]] += 1
                    if c_xk is not None and c_xk in atom_index:
                        row[atom_index[c_xk]] += 1
                    for xr in c_extra:
                        if xr is not None and xr in atom_index:
                            row[atom_index[xr]] += 1
                    if inf_col is not None:
                        row[inf_col] += inf_coeff
                    rows.append(row)

        step_src = _get(rec, "step")
        if isinstance(step_src, dict) and step_src.get("source") == "involution_closure":
            # Count every row emitted for involution records (there is normally 1).
            n_involution_rows += (len(rows) - rows_before_this_rec)

    if n_involution_rows:
        print(f"[relation_matrix] Involution/free records contributed {n_involution_rows} rows.")

    mat = Matrix(ZZ, rows)
    return mat, atoms, used_records

def print_relation_matrix_summary(
    mat: Any,
    atoms: List[Any],
    used_records: List[Any],
    *,
    show_matrix: bool = False,
    max_show_rows: int = 20,
    max_show_cols: int = 30,
) -> None:
    """Pretty-print the shape, atom list, and rank of the relation matrix."""
    n_rows, n_cols = mat.nrows(), mat.ncols()

    step_leaf_atoms = set()
    for rec in used_records:
        step = _get(rec, "step")
        if isinstance(step, dict):
            for key in ("candidate_xs", "found_xs"):
                xs = step.get(key)
                if xs is None:
                    continue
                if isinstance(xs, dict):
                    xs = xs.keys()
                try:
                    step_leaf_atoms.update(x for x in xs if x is not None)
                except TypeError:
                    if xs is not None:
                        step_leaf_atoms.add(xs)

        pool = _get(rec, "candidate_pool")
        if pool:
            for cand in pool:
                if isinstance(cand, dict):
                    # Added 'x_res' to summary tracker
                    for key in ("x_step", "x", "candidate_x", "x_value", "x_res"):
                        x = cand.get(key)
                        if x is not None:
                            step_leaf_atoms.add(x)
                elif cand is not None:
                    step_leaf_atoms.add(cand)

    print("\n" + "=" * 70)
    print("DIVISOR RELATION MATRIX SUMMARY")
    print("=" * 70)
    print(f"  Relations encoded  : {n_rows} rows")
    print(f"  Matrix size        : {n_rows} x {n_cols}")
    print(f"  Leaf atoms seen    : {len(step_leaf_atoms)} extra leaf values in step payloads")
    print(f"  D1 atoms (columns) : {n_cols} cols  ({max(0, n_cols - 1)} finite x-coords + ∞)")

    finite_count = n_cols - 1
    print("\n  Column layout:")
    print(f"    cols 0 .. {finite_count - 1}  ->  finite x-coordinates")
    print(f"    col  {finite_count}          ->  ∞")

    if len(atoms) <= 30:
        print("\n  Atom index  (col -> x-value):")
        for i, a in enumerate(atoms):
            print(f"    [{i:3d}]  {a}")
    else:
        print("\n  Atom index  (first 10 / last 5):")
        for i in range(min(10, len(atoms))):
            print(f"    [{i:3d}]  {atoms[i]}")
        print("    ...")
        for i in range(max(10, len(atoms) - 5), len(atoms)):
            print(f"    [{i:3d}]  {atoms[i]}")

    if show_matrix and n_rows > 0:
        print(f"\n  Matrix (up to {max_show_rows}×{max_show_cols}):")
        show_r = min(n_rows, max_show_rows)
        show_c = min(n_cols, max_show_cols)
        print(mat[:show_r, :show_c])
        if show_r < n_rows or show_c < n_cols:
            print(f"  ... truncated; full size {n_rows}×{n_cols}")

    if n_rows == 0 or n_cols == 0:
        print("\n  Rank: N/A (empty matrix)")
    else:
        # Use mod-p rank for memory efficiency.
        # The matrix has only small integer coefficients (-d, 1, d-2),
        # so rank mod a large prime equals the true rank with probability
        # 1 - n_cols/p > 1 - 1e-9 for our choice of p.  This avoids
        # materialising a dense QQ matrix which OOMs at scale.
        _RANK_PRIME = 2**31 - 1  # Mersenne prime, fits in Sage GF
        print(f"\n  Computing rank mod {_RANK_PRIME} (exact w.h.p., O(1) memory vs dense QQ)...")
        mat_modp = mat.change_ring(GF(_RANK_PRIME))
        rank = mat_modp.rank()

        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Rows    (relations)    : {n_rows:>6}          │")
        print(f"  │  Columns (atoms + ∞)    : {n_cols:>6}          │")
        print(f"  │  Rank    (mod p)        : {rank:>6}          │")
        print(f"  │  Nullity (mod p)        : {n_cols - rank:>6}          │")
        print(f"  └─────────────────────────────────────────┘")

        if rank < n_cols:
            print(f"\n  Note: the relations span a proper subspace of the atom lattice.")
        else:
            print(f"\n  Note: the relations span the full column space.")

    print("=" * 70 + "\n")

def print_nullity_report(mat, atoms, *, fp_prime=2**31 - 1):
    """Pretty-print a full nullity decomposition report."""
    bottlenecks, report = find_bottleneck_atoms(mat, atoms, fp_prime=fp_prime)

    print("\n" + "=" * 70)
    print("NULLITY DECOMPOSITION REPORT")
    print("=" * 70)
    print(f"  Rank              : {report['rank']}")
    print(f"  Nullity (total)   : {report['nullity']}")
    print(f"  ∞ contribution    : {1 if report['inf_atom_in_null'] else 0}  (irreducible)")
    print(f"  Dest-only atoms   : {len(report['dest_only_atoms'])}  (each needs one x_src-step through it)")
    print(f"  Residual nullity  : {report['residual_nullity']}  (disconnected components?)")
    print()

    if report["dest_only_atoms"]:
        print("  Dest-only atoms (restart walker from these):")
        for a in report["dest_only_atoms"]:
            print(f"    x_src = {a}")

    if report["residual_nullity"] > 0:
        print(f"\n  Residual bottleneck atoms:")
        for b in bottlenecks:
            if b["reason"] == "residual":
                print(f"    x_src = {b['atom']}  ({b['col_nnz']} relations)  — {b['action']}")

    print("=" * 70 + "\n")
    return bottlenecks, report

def prune_dest_only(mat, atoms, protected=None):
    """
    Iteratively remove dest-only atoms (columns with exactly 1 nonzero entry)
    and their single incident row, AND degree-1 rows (rows with exactly 1
    nonzero entry in live columns) and their single live column, until fixed
    point. Both directions are rank-preserving over any field.

    Protected atoms are never removed even if they become prunable.
    """
    inf_sentinel = "∞"
    cur_atoms = list(atoms)

    protected_strs = set()
    if protected:
        for p in protected:
            protected_strs.add(str(p))

    n_rows = mat.nrows()
    n_cols = mat.ncols()

    row_data = [{} for _ in range(n_rows)]
    col_rows = [set() for _ in range(n_cols)]

    for (i, j) in mat.nonzero_positions(copy=False):
        val = int(mat[i, j])
        if val != 0:
            row_data[i][j] = val
            col_rows[j].add(i)

    live_cols = set(range(n_cols))
    live_rows = set(range(n_rows))

    immune_cols = {
        j for j, a in enumerate(cur_atoms)
        if str(a) == inf_sentinel or str(a) in protected_strs
    }

    removed = []
    dead_rows = set()

    def col_worklist_seed():
        return {
            j for j in live_cols
            if j not in immune_cols and len(col_rows[j] & live_rows) == 1
        }

    def row_worklist_seed():
        return {
            i for i in live_rows
            if len({k for k in row_data[i] if k in live_cols}) == 1
        }

    col_worklist = col_worklist_seed()
    row_worklist = row_worklist_seed()

    while col_worklist or row_worklist:
        # --- column pass: dest-only atom ---
        while col_worklist:
            j = col_worklist.pop()
            if j not in live_cols or j in immune_cols:
                continue
            live_incident = col_rows[j] & live_rows
            if len(live_incident) != 1:
                continue

            (i,) = live_incident
            removed.append((cur_atoms[j], i))
            dead_rows.add(i)
            live_rows.discard(i)
            live_cols.discard(j)

            for k in list(row_data[i].keys()):
                if k == j or k not in live_cols:
                    continue
                col_rows[k].discard(i)
                live_inc = col_rows[k] & live_rows
                if k not in immune_cols and len(live_inc) == 1:
                    col_worklist.add(k)
                # If a surviving row now has degree 1, add to row worklist.
                for ii in list(live_inc):
                    live_support = {kk for kk in row_data[ii] if kk in live_cols}
                    if len(live_support) == 1:
                        row_worklist.add(ii)

        # --- row pass: degree-1 row forces column ---
        while row_worklist:
            i = row_worklist.pop()
            if i not in live_rows:
                continue
            live_support = {k for k in row_data[i] if k in live_cols}
            if len(live_support) != 1:
                continue

            (j,) = live_support
            if j in immune_cols:
                # Row is degree-1 on a protected column: just drop the row,
                # the column survives (it's pinned to zero by this row but
                # we can't remove it, so we just drop the redundant row).
                dead_rows.add(i)
                live_rows.discard(i)
                col_rows[j].discard(i)
                live_inc = col_rows[j] & live_rows
                if j not in immune_cols and len(live_inc) == 1:
                    col_worklist.add(j)
                continue

            removed.append((cur_atoms[j], i))
            dead_rows.add(i)
            live_rows.discard(i)
            live_cols.discard(j)

            for ii in list(col_rows[j] & live_rows):
                col_rows[j].discard(ii)
                # Check if any other column in that row is now degree-1.
                for k in list(row_data[ii].keys()):
                    if k not in live_cols:
                        continue
                    live_inc = col_rows[k] & live_rows
                    if k not in immune_cols and len(live_inc) == 1:
                        col_worklist.add(k)
                live_support2 = {k for k in row_data[ii] if k in live_cols}
                if len(live_support2) == 1:
                    row_worklist.add(ii)

    # Reconstruct surviving matrix.
    sorted_cols = sorted(live_cols)
    col_remap = {old_j: new_j for new_j, old_j in enumerate(sorted_cols)}
    pruned_atoms = [cur_atoms[j] for j in sorted_cols]
    n_pruned_cols = len(sorted_cols)

    surviving = {}
    new_row_idx = 0
    for i in range(n_rows):
        if i in dead_rows:
            continue
        for old_j, val in row_data[i].items():
            if old_j in col_remap:
                surviving[(new_row_idx, col_remap[old_j])] = val
        new_row_idx += 1

    if new_row_idx == 0:
        return Matrix(ZZ, 0, n_pruned_cols), pruned_atoms, removed

    pruned_mat = Matrix(ZZ, new_row_idx, n_pruned_cols, surviving)
    return pruned_mat, pruned_atoms, removed

build_relation_matrix = build_relation_matrix2
