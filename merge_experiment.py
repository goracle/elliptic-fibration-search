from __future__ import annotations
import argparse, json, math, random as _random, sys, time
from pathlib import Path
from typing import Optional, List, Tuple
from search_common import FINITE_FIELD, get_y_unshifted_genus2, COEFFS_GENUS2, PRIME_POOL
from sage.all import *
from markov import *

"""merge_experiment.py

Cross-chain merge time measurement for the genus-2 Markov walk.

Seed strategy
-------------
Walks A, B, C, D are each seeded from one of the four x-coordinates that make
up the *base divisor* and *target divisor* of the cryptosystem:

  BASE_DIVISOR   : the generator G of the Jacobian subgroup (degree-2 divisor
                   over F_p).  Its Mumford u-polynomial has two roots x₀, x₁
                   in F_p (the x-coordinates of the two affine points in G).

  TARGET_DIVISOR : the challenge point T = SECRET_KEY·G.  Its Mumford
                   u-polynomial similarly yields roots x₂, x₃.

These four values are exactly PREFERRED_X_COORDS (asserted length-4 by
search_common).  Seeding each walk at one of these roots ensures that the
corresponding factor-base column appears in the relation matrix from step 0,
giving the DLP solve the algebraic anchoring it needs.

DLP stream-merge solve
----------------------
When any of walks B/C/D first touches a leaf that walk A already visited, the
two relation matrices can be stacked and the target DLP solved by:

  1.  Build the combined relation matrix M over Z  (rows = walk steps,
      cols = leaf x-coordinates ≅ Jacobian divisor classes).
  2.  Identify the column corresponding to the unknown generator P.
  3.  Solve M·v ≡ 0 (mod n) for the group order n using SmithNormalForm or a
      modular kernel computation.
  4.  Extract the discrete logarithm coefficient for P.

The function `dlp_from_merged_walks` implements this.  It requires the walker's
`relation_matrix()` method (already present on Genus2MetropolisWalker) and
knowledge of the target generator column index.

Usage
-----
    sage -python merge_experiment.py                  # full run (4 walks)
    sage -python merge_experiment.py --skip-a         # resume with existing snapshot
    sage -python merge_experiment.py --steps-a 500 --steps-bcd 500
"""

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from genus2_markov_module import make_project_markov_search_fn, load_project_sources
try:
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DEFAULT_SNAPSHOT      = "walk_A_leaves.json"
DEFAULT_STEPS_A       = 320
DEFAULT_STEPS_BCD     = 320

_FALLBACK_P      = 33554467
_FALLBACK_COEFFS = None


# ---------------------------------------------------------------------------
# Divisor-root seed computation
# ---------------------------------------------------------------------------

def _divisor_seed_xs() -> List[int]:
    """Return the four x-coordinates from BASE_DIVISOR and TARGET_DIVISOR.

    These are the roots of the Mumford u-polynomials of the two degree-2
    divisors that define the cryptosystem:

        BASE_DIVISOR   -> x-coords [x0, x1]   (indices 0, 1 of PREFERRED_X_COORDS)
        TARGET_DIVISOR -> x-coords [x2, x3]   (indices 2, 3 of PREFERRED_X_COORDS)

    PREFERRED_X_COORDS is populated by setup_prime_subgroup_cryptosystem()
    at import time in search_common and is asserted to have exactly 4 entries.

    Returns a list of 4 ints in the same order as PREFERRED_X_COORDS.
    """
    import search_common as _sc
    xs = getattr(_sc, 'PREFERRED_X_COORDS', None)
    if xs is None:
        # Fallback: try PROJECT_REGISTRY via resolve_project_symbol
        from genus2_markov_module import resolve_project_symbol
        xs = resolve_project_symbol('PREFERRED_X_COORDS', default=None)
    if xs is None:
        raise RuntimeError(
            "_divisor_seed_xs: PREFERRED_X_COORDS not available; "
            "call load_project_sources() before invoking main()."
        )
    result = [int(x) for x in xs]
    assert len(result) == 4, f"Expected 4 divisor roots, got {len(result)}: {result}"
    return result


# ---------------------------------------------------------------------------
# Walker construction
# ---------------------------------------------------------------------------

def _log(msg: str, **kw) -> None:
    print(msg, flush=True, **kw)


def _build_walker(
    seed: int,
    p: int,
    coeffs,
    x0,
    y0=None,
    base_points=None,
    verbose: bool = True,
    log_path: Optional[str] = None,
    search_fn=None,
) -> Genus2MetropolisWalker:
    return build_default_walker(
        coeffs=coeffs,
        initial_x=x0,
        p=p,
        initial_y=y0,
        base_points=base_points,
        seed=seed,
        load_sources=False,
        verbose=verbose,
        search_fn=search_fn,
        log_path=log_path,
        log_full_candidates=True,
    )


# ---------------------------------------------------------------------------
# DLP stream-merge solve
# ---------------------------------------------------------------------------

def dlp_from_merged_walks(
    walker_a: Genus2MetropolisWalker,
    walker_b: Genus2MetropolisWalker,
    target_x,
    group_order: Optional[int] = None,
    generator_x=None,
    verbose: bool = True,
) -> Optional[dict]:
    """Attempt a DLP solve from the combined relation matrices of two walks.

    The merge event means the two walks share a common leaf, giving us a path
    in the Cayley graph from the generator to the target through that leaf.
    The relation matrix encodes each walk step as a linear equation over Z/nZ
    relating the discrete logs of the x-coordinates involved.

    Parameters
    ----------
    walker_a, walker_b : completed walkers with non-empty history.
    target_x           : x-coordinate of the DLP target point T.
    group_order        : order n of the Jacobian (or known subgroup order).
                         If None, we attempt to read it from walker globals.
    generator_x        : x-coordinate of the generator G.  If None, the
                         function uses the first leaf seen by walk A.
    verbose            : print intermediate diagnostics.

    Returns
    -------
    dict with keys:
        "dlp"          : integer k such that T = k·G  (mod n), or None.
        "method"       : description of the solve path used.
        "rank_A"       : rank of A's relation matrix mod a large prime.
        "rank_B"       : rank of B's relation matrix mod a large prime.
        "rank_combined": rank of the stacked matrix.
        "n_cols"       : number of distinct leaf columns.
    """
    _P_TEST = 2**31 - 1   # large prime for rank testing

    result = {
        "dlp": None,
        "method": None,
        "rank_A": None,
        "rank_B": None,
        "rank_combined": None,
        "n_cols": None,
    }

    # --- Build individual relation matrices --------------------------------
    try:
        mat_a, atoms_a, used_a = walker_a.relation_matrix(include_step_leaves=False)
        mat_b, atoms_b, used_b = walker_b.relation_matrix(include_step_leaves=False)
    except Exception as exc:
        _log(f"[dlp] relation_matrix() failed: {exc}")
        return result

    if mat_a.nrows() == 0 or mat_b.nrows() == 0:
        _log("[dlp] one or both walks have empty relation matrices — no solve possible.")
        return result

    # --- Merge the atom (column) sets -------------------------------------
    # atoms_* are lists of x-coordinates (leaves).  We need a unified column
    # ordering so both matrices can be stacked consistently.
    all_atoms_ordered = list(atoms_a)
    atom_set = set(map(str, atoms_a))
    extra_b = [a for a in atoms_b if str(a) not in atom_set]
    all_atoms_ordered.extend(extra_b)
    n_cols = len(all_atoms_ordered)
    atom_index = {str(a): i for i, a in enumerate(all_atoms_ordered)}

    result["n_cols"] = n_cols

    def _reindex_matrix(mat, atoms_src):
        """Re-embed mat (which lives in atom_src's column space) into the
        unified n_cols-wide column space."""
        cols_src = [atom_index[str(a)] for a in atoms_src]
        M = matrix(ZZ, mat.nrows(), n_cols)
        for r in range(mat.nrows()):
            for c_src, c_dst in enumerate(cols_src):
                M[r, c_dst] = mat[r, c_src]
        return M

    M_a = _reindex_matrix(mat_a, atoms_a)
    M_b = _reindex_matrix(mat_b, atoms_b)
    M_combined = M_a.stack(M_b)

    # --- Rank diagnostics -------------------------------------------------
    Fp_test = GF(_P_TEST)
    result["rank_A"]        = mat_a.change_ring(Fp_test).rank()
    result["rank_B"]        = mat_b.change_ring(Fp_test).rank()
    result["rank_combined"] = M_combined.change_ring(Fp_test).rank()

    if verbose:
        _log(f"[dlp] atoms: A={mat_a.ncols()}, B={mat_b.ncols()}, combined={n_cols}")
        _log(f"[dlp] rows:  A={mat_a.nrows()}, B={mat_b.nrows()}, combined={M_combined.nrows()}")
        _log(f"[dlp] rank:  A={result['rank_A']}, B={result['rank_B']}, combined={result['rank_combined']}")

    # --- Identify target and generator columns ----------------------------
    if target_x is None:
        _log("[dlp] target_x is None — cannot solve.")
        return result

    target_col = atom_index.get(str(target_x))
    if target_col is None:
        _log(f"[dlp] target x={target_x} not in combined leaf set — walk did not reach it.")
        return result

    gen_col = None
    if generator_x is not None:
        gen_col = atom_index.get(str(generator_x))
        if gen_col is None:
            _log(f"[dlp] generator x={generator_x} not in combined leaf set.")

    # --- Solve over Z/nZ --------------------------------------------------
    # We want a vector v in the right kernel of M_combined mod n such that
    # v[target_col] ≠ 0.  Then dlp = -v[gen_col] * v[target_col]^{-1} mod n.
    #
    # If group_order is unknown, try Smith Normal Form over Z to read off
    # torsion structure, then reduce.
    n = group_order
    if n is None:
        try:
            n_sym = resolve_project_symbol('GROUP_ORDER', default=None)
            if n_sym is not None:
                n = int(n_sym)
        except Exception:
            pass

    if n is None:
        if verbose:
            _log("[dlp] group_order unknown — computing Smith Normal Form over Z.")
        try:
            D, U, V = M_combined.smith_form()
            # Diagonal of D gives torsion invariants.  The last non-zero diagonal
            # entry is the largest invariant factor and bounds the group order.
            diag = [D[i, i] for i in range(min(D.nrows(), D.ncols()))]
            non_trivial = [d for d in diag if d not in (0, 1)]
            if non_trivial:
                n = non_trivial[-1]
                _log(f"[dlp] inferred group order bound from SNF: n={n}")
            else:
                _log("[dlp] SNF gave trivial invariant factors — cannot extract n.")
                result["method"] = "snf_trivial"
                return result
        except Exception as exc:
            _log(f"[dlp] SNF failed: {exc}")
            return result

    # Right kernel of M_combined mod n
    try:
        Zn = Integers(n)
        M_mod = M_combined.change_ring(Zn)
        K = M_mod.right_kernel()
        if verbose:
            _log(f"[dlp] right kernel dimension mod {n}: {K.dimension()}")
    except Exception as exc:
        _log(f"[dlp] kernel computation failed: {exc}")
        result["method"] = "kernel_failed"
        return result

    # Search kernel basis for a vector with non-zero target_col entry
    dlp_val = None
    for v in K.basis():
        t_entry = Zn(v[target_col])
        if t_entry == 0:
            continue
        if gen_col is not None:
            g_entry = Zn(v[gen_col])
            if g_entry == 0:
                continue
            try:
                dlp_val = int(-t_entry * g_entry.__invert__()) % n
                result["method"] = "kernel_ratio"
                result["dlp"] = dlp_val
                break
            except ZeroDivisionError:
                continue
        else:
            # No generator identified; report the raw target coefficient
            result["method"] = "kernel_target_only"
            result["dlp"] = int(t_entry)
            break

    if verbose:
        if dlp_val is not None:
            _log(f"[dlp] SOLVED: k = {dlp_val}  (T = {dlp_val}·G mod {n})")
        else:
            _log("[dlp] kernel basis exhausted without finding usable vector.")

    return result


# ---------------------------------------------------------------------------
# Merge-path DLP: birthday-paradox collision path
# ---------------------------------------------------------------------------

def _collision_path_dlp(
    walker_a: Genus2MetropolisWalker,
    walker_b: Genus2MetropolisWalker,
    merge_leaf,
    group_order: int,
    verbose: bool = True,
) -> Optional[int]:
    """Extract DLP from the birthday collision directly.

    When B's step t_B first hits a leaf that A visited at its step t_A, we
    have two paths from the respective starting points to the same Jacobian
    element.  If the starting points are known multiples of the generator G:

        x₀_A = a·G,   x₀_B = b·G

    then the relation matrices encode:

        Σ_A (relation coefficients for A's path to merge_leaf) ≡ e_A  (mod n)
        Σ_B (relation coefficients for B's path to merge_leaf) ≡ e_B  (mod n)

    and we can subtract to isolate the target.

    This is a placeholder that calls dlp_from_merged_walks with the merge leaf
    as the target.  A full implementation requires knowing (a, b) in advance;
    document that requirement clearly.
    """
    if verbose:
        _log(f"[collision_dlp] merge leaf = {merge_leaf}")
        _log(f"[collision_dlp] delegating to dlp_from_merged_walks "
             f"(target=merge_leaf, generator=walk_A_start)")
    gen_x = walker_a.history[0].xi if walker_a.history else None
    return dlp_from_merged_walks(
        walker_a, walker_b,
        target_x=merge_leaf,
        group_order=group_order,
        generator_x=gen_x,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Cross-chain merge experiment seeded from BASE/TARGET divisor roots"
    )
    ap.add_argument("--steps-a",   type=int, default=DEFAULT_STEPS_A,
                    help="Steps for walk A")
    ap.add_argument("--steps-bcd", type=int, default=DEFAULT_STEPS_BCD,
                    help="Steps for each of walks B, C, D")
    ap.add_argument("--snapshot",  default=DEFAULT_SNAPSHOT)
    ap.add_argument("--skip-a",    action="store_true",
                    help="Skip walk A and load existing snapshot")
    ap.add_argument("--results",   default="merge_results.json")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--max-n",     type=int, default=80)
    ap.add_argument("--num-subsets", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=16)
    # DLP options
    ap.add_argument("--dlp",        action="store_true",
                    help="Attempt DLP solve from merged relation matrices after the run")
    ap.add_argument("--dlp-target", type=int, default=None,
                    help="x-coordinate of the DLP target point (integer mod p)")
    ap.add_argument("--group-order", type=int, default=None,
                    help="Known group order n for the DLP solve (optional)")
    args = ap.parse_args(argv)

    # -- Resolve project globals -------------------------------------------
    search_fn = None
    pool = None
    if _HAS_PROJECT:
        load_project_sources(verbose=False)
        import multiprocessing
        from search_lll.mumford.mumford_parallel import init_worker

        p          = FINITE_FIELD
        coeffs     = COEFFS_GENUS2
        prime_pool = PRIME_POOL

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=args.num_workers, initializer=init_worker)
        chunk_size = 8

        search_fn = make_project_markov_search_fn(
            coeffs_genus2=coeffs,
            base_points=None,
            p=p,
            prime_pool=prime_pool,
            num_subsets=args.num_subsets,
            num_workers=args.num_workers,
            debug=False,
            max_n=args.max_n,
            precomputed_residues=None,
            all_found_x=set(),
            pool=pool,
            chunk_size=chunk_size,
        )
    else:
        p      = _FALLBACK_P
        coeffs = _FALLBACK_COEFFS
        if coeffs is None:
            sys.exit("ERROR: project sources unavailable and _FALLBACK_COEFFS not set.")

    sqrt_p = math.sqrt(p)

    # -- Divisor root seeds ------------------------------------------------
    # x0, x1 are the Mumford u-poly roots of BASE_DIVISOR  (the generator G)
    # x2, x3 are the Mumford u-poly roots of TARGET_DIVISOR (the challenge T)
    divisor_xs = _divisor_seed_xs()
    x0_a, x0_b, x0_c, x0_d = divisor_xs

    _log(f"\n[seeds] divisor roots (PREFERRED_X_COORDS): {divisor_xs}")
    _log(f"[seeds]   BASE_DIVISOR   roots: x0_A={x0_a}, x0_B={x0_b}")
    _log(f"[seeds]   TARGET_DIVISOR roots: x0_C={x0_c}, x0_D={x0_d}")

    def _bp(x0):
        if _HAS_PROJECT:
            return project_base_points_from_globals(current_x=x0, p=p)
        return []

    # -- Phase A -----------------------------------------------------------
    walker_a = None
    if not args.skip_a:
        _log(f"\n{'='*70}")
        _log(f"PHASE A  x0={x0_a} (BASE_DIVISOR root 0)  steps={args.steps_a}  p={p}  sqrt_p={sqrt_p:.1f}")
        _log(f"{'='*70}\n")

        walker_a = _build_walker(
            seed=0, p=p, coeffs=coeffs, x0=x0_a, y0=None,
            base_points=_bp(x0_a), verbose=True, log_path="walk_A.jsonl",
            search_fn=search_fn,
        )
        walker_a.run(args.steps_a)
        walker_a.save_leaf_snapshot(args.snapshot)

        vol_a = len(walker_a.global_leaves_seen)
        _log(f"\n[A done]  vol={vol_a}  ({vol_a/sqrt_p:.4f}*sqrt_p)  snapshot -> {args.snapshot}\n")
    else:
        _log(f"[skip-a]  loading snapshot from {args.snapshot}")

    # -- Helper: build + run one of the B/C/D walks -----------------------
    def _run_secondary(label, x0, seed, log_path):
        _log(f"\n{'='*70}")
        _log(f"PHASE {label}  x0={x0}  steps={args.steps_bcd}  (quiet mode)")
        _log(f"{'='*70}\n")
        w = _build_walker(
            seed=seed, p=p, coeffs=coeffs, x0=x0, y0=None,
            base_points=_bp(x0), verbose=False, log_path=log_path,
            search_fn=search_fn,
        )
        n_foreign = w.load_foreign_leaves(args.snapshot, label="A")
        _log(f"  [{label}] foreign leaves loaded: {n_foreign}")
        _quiet_run(w, args.steps_bcd, checkpoint_every=args.checkpoint_every)
        return w

    walker_b = _run_secondary("B", x0_b, seed=1, log_path="walk_B.jsonl")
    walker_c = _run_secondary("C", x0_c, seed=2, log_path="walk_C.jsonl")
    walker_d = _run_secondary("D", x0_d, seed=3, log_path="walk_D.jsonl")

    # -- Per-walk merge reports --------------------------------------------
    all_metrics = {}
    first_merger = None   # whichever of B/C/D merged first
    first_merge_step = None

    for label, walker in [("B", walker_b), ("C", walker_c), ("D", walker_d)]:
        m = _merge_report(walker, args.steps_bcd)
        m["x0"] = int(walker.current_x if hasattr(walker, 'current_x') else -1)
        all_metrics[label] = m
        if m["first_merge_step"] is not None:
            if first_merge_step is None or m["first_merge_step"] < first_merge_step:
                first_merge_step = m["first_merge_step"]
                first_merger = (label, walker)

    metrics = {
        "p": p,
        "sqrt_p": sqrt_p,
        "steps_a": args.steps_a,
        "steps_bcd": args.steps_bcd,
        "x0_A": int(x0_a),
        "x0_B": int(x0_b),
        "x0_C": int(x0_c),
        "x0_D": int(x0_d),
        "divisor_xs": [int(x) for x in divisor_xs],
        "walks": all_metrics,
        "first_merge_label": first_merger[0] if first_merger else None,
        "first_merge_step_global": first_merge_step,
    }

    # -- DLP solve (optional) -- use first walk that merged ---------------
    if args.dlp and walker_a is not None and first_merger is not None:
        merge_label, merge_walker = first_merger
        _log(f"\n{'='*70}")
        _log(f"DLP SOLVE using walk {merge_label} (first to merge with A)")
        _log(f"{'='*70}\n")

        merge_leaf = None
        if merge_walker.merge_log:
            merge_leaf = merge_walker.merge_log[0][3]
            _log(f"[dlp] merge leaf (first collision): {merge_leaf}")

        target_x = args.dlp_target if args.dlp_target is not None else merge_leaf

        if target_x is not None:
            dlp_result = dlp_from_merged_walks(
                walker_a, merge_walker,
                target_x=target_x,
                group_order=args.group_order,
                generator_x=x0_a,
                verbose=True,
            )
            metrics["dlp"] = dlp_result
            _log(f"[dlp] result: {dlp_result}")
        else:
            _log("[dlp] no merge occurred and no --dlp-target given; skipping solve.")
    elif args.dlp:
        _log("[dlp] skipped: walk A not run (--skip-a) or no merge detected in B/C/D.")

    # -- Write results -----------------------------------------------------
    with open(args.results, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    _log(f"[results] written -> {args.results}")

    for w in (walker_b, walker_c, walker_d):
        if getattr(w, 'cantor_cache', None) is not None:
            w.cantor_summary()

    # -- Cleanup ----------------------------------------------------------
    if pool is not None:
        pool.close()
        pool.join()

    return metrics


# ---------------------------------------------------------------------------
# Quiet runner (unchanged from original)
# ---------------------------------------------------------------------------

def _quiet_run(walker: Genus2MetropolisWalker, steps: int, checkpoint_every: int = 5) -> None:
    original_verbose = walker.config.verbose
    walker.config.verbose = False

    sqrt_p = math.sqrt(walker.p) if walker.p is not None else float("nan")
    t0 = time.time()

    for i in range(steps):
        walker.step()
        display_step = i + 1

        if display_step % checkpoint_every == 0:
            vol  = len(walker.global_leaves_seen)
            ins  = walker.total_leaf_insertions
            hits = len(walker.merge_log) if walker.foreign_leaves is not None else "n/a"
            elapsed = time.time() - t0
            _log(
                f"  [B step {display_step:>5}]  vol={vol:>7}  ({vol/sqrt_p:.4f}×√p)  "
                f"ins={ins}  ({ins/sqrt_p:.4f}×√p)  "
                f"merge_hits={hits}  elapsed={elapsed:.1f}s"
            )

        if (walker.first_merge_step is not None
                and display_step == walker.first_merge_step
                and display_step > 0):
            _log(f"  [B] first merge recorded at step {walker.first_merge_step}, continuing…")

    walker.config.verbose = original_verbose


def _merge_report(walker: Genus2MetropolisWalker, total_steps: int) -> dict:
    p = walker.p
    sqrt_p = math.sqrt(p)

    first_step = walker.first_merge_step
    first_vol  = walker.first_merge_vol
    total_ins  = walker.total_leaf_insertions

    ins_ratio  = (first_vol / sqrt_p) if first_vol is not None else (total_ins / sqrt_p)
    step_ratio = (first_step / sqrt_p) if first_step is not None else (total_steps / sqrt_p)

    vol_at_merge = 0
    if first_step is not None:
        for entry in walker.merge_log:
            if entry[0] == first_step:
                vol_at_merge = entry[2]
                break

    metrics = {
        "p": p,
        "sqrt_p": sqrt_p,
        "steps_run": total_steps,
        "total_leaf_insertions": total_ins,
        "final_vol": len(walker.global_leaves_seen),
        "first_merge_step": first_step,
        "first_merge_leaf_insertions": first_vol,
        "vol_at_first_merge": vol_at_merge,
        "merge_insertion_ratio": ins_ratio,
        "merge_step_ratio": step_ratio,
        "total_unique_hits": len(walker.merge_log),
    }

    _log("\n" + "="*70)
    _log("CROSS-CHAIN MERGE REPORT")
    _log("="*70)
    _log(f"  p                          = {p}")
    _log(f"  √p                         = {sqrt_p:.1f}")
    _log(f"  B steps run                = {total_steps}")
    _log(f"  B total leaf insertions    = {total_ins} ({total_ins/sqrt_p:.4f}×√p)")
    _log(f"  B final vol (unique)       = {metrics['final_vol']} ({metrics['final_vol']/sqrt_p:.4f}×√p)")
    _log(f"  First merge step           = {first_step if first_step is not None else 'N/A'}")
    _log(f"  Leaf insertions at merge   = {first_vol if first_vol is not None else 'N/A'}"
         + (f" ({first_vol/sqrt_p:.4f}×√p)" if first_vol is not None else ""))
    _log(f"  B vol (unique) at merge    = {vol_at_merge} ({vol_at_merge/sqrt_p:.4f}×√p)")
    _log(f"  Merge insertion ratio      = {ins_ratio:.4f}  ← PRIMARY METRIC")
    _log(f"  Merge step ratio (legacy)  = {step_ratio:.4f}")
    _log(f"  Total unique hits          = {metrics['total_unique_hits']}")
    _log("")

    if first_vol is not None and ins_ratio < 0.5:
        _log("  Verdict: STRONG OVERLAP — multi-chain is clearly viable")
    elif first_vol is not None:
        _log("  Verdict: MODERATE OVERLAP")
    else:
        _log("  Verdict: NO MERGE DETECTED in this window")
    _log("="*70 + "\n")

    return metrics


if __name__ == "__main__":
    main()
