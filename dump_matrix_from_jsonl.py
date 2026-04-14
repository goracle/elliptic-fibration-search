from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Optional
from markov.walkerclass import RelationRecord, WalkConfig
from markov.relation_matrix import build_relation_matrix2
from markov.dlp_diagnostics import dump_matrix_hdf5

"""dump_matrix_from_jsonl.py

Reconstruct the four relation matrices from the JSONL walk logs and dump
the combined pruned matrix to an HDF5 file — without re-running any walks.

Usage
-----
    sage -python dump_matrix_from_jsonl.py \
        --walks walk_A.jsonl walk_B.jsonl walk_C.jsonl walk_D.jsonl \
        --output relation_matrix.h5

All other parameters (group_order, divisor_xs, curve_degree) are read from
search_common if available, but can be overridden on the command line.

Memory note
-----------
The JSONL files log one line per relation record, including preferred-injection
synthetic relations (~13 rows per outer walk step).  With 4 walks of ~7-10k
steps each that is ~400k rows total.  build_relation_matrix2 allocates a dense
Sage ZZ matrix which OOMs at that scale.

--max-rows-per-walk caps each walker's history before matrix construction.
The relation matrix rank saturates at n_cols (a few thousand), so 20k rows
per walker is more than enough.  Default is 25000.
"""

try:
    from sage.all import GF, ZZ, Integer
except ImportError:
    sys.exit("ERROR: must run under SageMath — use: sage -python ...")

try:
    from search_common import FINITE_FIELD, GROUP_MODULUS, PREFERRED_X_COORDS
    _HAS_SEARCH_COMMON = True
except ImportError:
    _HAS_SEARCH_COMMON = False
    FINITE_FIELD = None
    GROUP_MODULUS = None
    PREFERRED_X_COORDS = None

# ---------------------------------------------------------------------------
# Minimal stub walker
# ---------------------------------------------------------------------------

class _StubWalker:
    def __init__(self, history: List[RelationRecord], curve_degree: int = 5, label: str = "?"):
        self.history = history
        self.config  = WalkConfig(curve_degree=curve_degree)
        self._label  = label
        self.first_merge_step = None
        self.first_merge_vol  = None

    def relation_matrix(self, include_step_leaves: bool = False):
        return build_relation_matrix2(
            self.history,
            curve_degree=self.config.curve_degree,
            include_step_leaves=include_step_leaves,
        )

# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl_history(
    path: str,
    p: int,
    curve_degree: int = 5,
    max_rows: Optional[int] = None,
) -> List[RelationRecord]:
    """Read a JSONL walk log and return RelationRecord objects.

    The JSONL contains one line per relation record — this includes both the
    main accepted walk steps AND synthetic preferred-injection relations logged
    by inject_preferred_relations().  All are valid rows.

    Rejected steps and records missing xj/xk are skipped.

    max_rows caps the number of records returned.  The relation matrix rank
    saturates well before the full log is consumed; 20-25k rows per walker
    is sufficient and avoids the dense-matrix OOM.
    """
    Fp   = GF(Integer(p))
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL log not found: {path}")

    records: List[RelationRecord] = []
    n_skipped = 0
    n_lines   = 0
    n_missing_xi_mult = 0  # rows loaded but with xi_mult=-1 (will be dropped by matrix builder)

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            n_lines += 1

            try:
                d = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: JSON parse error: {exc}") from exc

            if not d.get("accepted", False):
                n_skipped += 1
                continue

            xi_raw = d.get("xi")
            xj_raw = d.get("xj")
            xk_raw = d.get("xk")
            if xi_raw is None or xj_raw is None or xk_raw is None:
                n_skipped += 1
                continue

            try:
                xi = Fp(int(xi_raw))
                xj = Fp(int(xj_raw))
                xk = Fp(int(xk_raw))
            except Exception as exc:
                raise ValueError(
                    f"{path}:{lineno}: cannot coerce xi/xj/xk to Fp: {exc}"
                ) from exc

            if xi == xj:          # degenerate — skip
                n_skipped += 1
                continue

            m_fp = None
            m_raw = d.get("m")
            if m_raw is not None:
                try:
                    m_fp = Fp(int(m_raw))
                except Exception:
                    pass

            step_dict = d.get("step") or {}
            if not isinstance(step_dict, dict):
                step_dict = {}

            # xi_mult priority: top-level d (written by fixed _record_to_log_dict)
            # > step sub-dict (older logs / preferred_injection) > sentinel -1.
            # Sentinel -1 means no fiber-derived multiplicity was logged; such rows
            # will be dropped by build_relation_matrix2.  We count them here so the
            # summary line makes the situation immediately visible.
            xi_mult = int(d.get("xi_mult", -1))
            if xi_mult < 0:
                xi_mult = int(step_dict.get("xi_mult", -1))
            if xi_mult < 0:
                n_missing_xi_mult += 1

            records.append(RelationRecord(
                step_index = int(d.get("step_index", lineno)),
                n          = int(d.get("n", 1)),
                xi         = xi,
                m          = m_fp,
                xj         = xj,
                xk         = xk,
                relation   = str(d.get("relation", "")),
                step       = step_dict,
                accepted   = True,
                restart    = bool(d.get("restart", False)),
                yj_sign    = int(d.get("yj_sign", 1)),
                yk_sign    = int(d.get("yk_sign", 1)),
                xi_mult    = xi_mult,
            ))

            if max_rows is not None and len(records) >= max_rows:
                print(
                    f"[load_jsonl] {path.name}: hit row cap {max_rows} at line {lineno}",
                    flush=True,
                )
                break

    print(
        f"[load_jsonl] {path.name}: {len(records)} rows loaded"
        f"  (skipped {n_skipped}, total lines {n_lines}"
        + (f", {n_missing_xi_mult} rows have xi_mult=-1 and will be dropped by matrix builder"
           f" — re-run walks with fixed walkerclass to populate xi_mult in logs"
           if n_missing_xi_mult else "")
        + ")",
        flush=True,
    )
    return records

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Dump relation matrix HDF5 from existing JSONL walk logs."
    )
    ap.add_argument(
        "--walks", nargs="+",
        default=["walk_A.jsonl", "walk_B.jsonl", "walk_C.jsonl", "walk_D.jsonl"],
        metavar="JSONL",
    )
    ap.add_argument("--output",           default="relation_matrix.h5")
    ap.add_argument("--p",                type=int, default=None)
    ap.add_argument("--group-order",      type=int, default=None)
    ap.add_argument("--divisor-xs",       type=int, nargs=4, default=None,
                    metavar=("X0", "X1", "X2", "X3"))
    ap.add_argument("--curve-degree",     type=int, default=5)
    ap.add_argument("--max-rows-per-walk", type=int, default=25000,
                    help="Cap rows loaded per walker to avoid OOM (default 25000; "
                         "rank saturates long before full log size)")
    args = ap.parse_args(argv)

    p = args.p or (int(FINITE_FIELD) if FINITE_FIELD is not None else None)
    if p is None:
        sys.exit("ERROR: --p not supplied and FINITE_FIELD unavailable")

    group_order = args.group_order or (int(GROUP_MODULUS) if GROUP_MODULUS is not None else None)
    if group_order is None:
        print("WARNING: group_order unknown — pass --group-order when running diagnostics",
              flush=True)

    divisor_xs = args.divisor_xs
    if divisor_xs is None and PREFERRED_X_COORDS is not None:
        try:
            divisor_xs = [int(x) for x in PREFERRED_X_COORDS]
        except Exception:
            pass
    if divisor_xs is None:
        sys.exit("ERROR: --divisor-xs not supplied and PREFERRED_X_COORDS unavailable")
    if len(divisor_xs) != 4:
        sys.exit(f"ERROR: expected 4 divisor x-coordinates, got {len(divisor_xs)}")

    print(
        f"\n[config] p={p}  group_order={group_order}  divisor_xs={divisor_xs}"
        f"  max_rows_per_walk={args.max_rows_per_walk}",
        flush=True,
    )

    labels  = ["A", "B", "C", "D"] + [str(i) for i in range(4, len(args.walks))]
    walkers = []
    for label, jsonl_path in zip(labels, args.walks):
        history = load_jsonl_history(
            jsonl_path, p=p,
            curve_degree=args.curve_degree,
            max_rows=args.max_rows_per_walk,
        )
        if not history:
            print(f"WARNING: {jsonl_path} produced no usable records — skipping.", flush=True)
            continue
        walkers.append(_StubWalker(history, curve_degree=args.curve_degree, label=label))

    if not walkers:
        sys.exit("ERROR: no usable walk histories")

    print(f"\n[build] {len(walkers)} walker(s), building relation matrices ...", flush=True)

    dump_matrix_hdf5(
        walkers=walkers,
        divisor_xs=divisor_xs,
        group_order=group_order if group_order is not None else 0,
        path=args.output,
    )

    print(f"\n[done] HDF5 written to: {args.output}", flush=True)
    print(
        f"[next] run:\n"
        f"  sage -python markov/dlp_contradiction_diag.py {args.output}"
        f" --group-order {group_order or '???'} --known-key <SECRET_KEY>",
        flush=True,
    )

if __name__ == "__main__":
    main()
