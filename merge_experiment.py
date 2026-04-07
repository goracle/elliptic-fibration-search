from __future__ import annotations
import argparse, json, math, random as _random, sys, time
from pathlib import Path
from typing import Optional
from search_common import FINITE_FIELD, get_y_unshifted_genus2, COEFFS_GENUS2, PRIME_POOL
from sage.all import *
from markov import *

"""merge_experiment.py

Cross-chain merge time measurement for the genus-2 Markov walk.

The experiment is:
    1.  Run walk A to saturation (~√p steps).  Save its leaf snapshot.
    2.  Build walk B with a different seed.  Load A's snapshot as foreign leaves.
    3.  Run B quietly.  Record the step at which B first touches a leaf A already
        explored.  That step count, normalised by √p, is the merge-time ratio.

A merge-time ratio close to 1.0 = independent uniform walks (no speedup).
A ratio << 1.0 = strong leaf-set overlap; multi-chain is viable.

Usage
-----
    # Full two-phase run:
    sage -python merge_experiment.py

    # Resume: skip walk A if snapshot already exists:
    sage -python merge_experiment.py --skip-a --snapshot walk_A_leaves.json

    # Tune step counts:
    sage -python merge_experiment.py --steps-a 5000 --steps-b 2000

The script shares your project's build_default_walker / make_project_markov_search_fn
infrastructure — edit _build_walker() below to match whatever you pass in
genus2_markov_module.py.
"""

# ---------------------------------------------------------------------------
# Project imports — same pattern as genus2_markov_module.py
# ---------------------------------------------------------------------------
# Adjust these imports to match your actual search-fn factory location:
from genus2_markov_module import make_project_markov_search_fn, load_project_sources
try:
    _HAS_PROJECT = True
except ImportError:
    _HAS_PROJECT = False

# get_y_unshifted_genus2 uses COEFFS_GENUS2 / FINITE_FIELD globals from search_common.

# ---------------------------------------------------------------------------
# Experiment configuration — edit to match your project globals
# ---------------------------------------------------------------------------

DEFAULT_SNAPSHOT = "walk_A_leaves.json"
DEFAULT_STEPS_A  = 15    # ≈ √p × 1.0  for p=33554467
DEFAULT_STEPS_B  = 15    # generous headroom; merge usually happens <<√p
DEFAULT_SEED_A   = 12
DEFAULT_SEED_B   = 67

# These are read from your project globals if _HAS_PROJECT; otherwise supply
# them directly here as fallbacks.
_FALLBACK_P      = 33554467
_FALLBACK_COEFFS = None   # set this if running without project sources
# x0 is now always chosen randomly via _random_curve_x(); no fallback needed.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str, **kw) -> None:
    print(msg, flush=True, **kw)

def _random_curve_x(p: int, rng: _random.Random) -> int:
    """Return a random x in F_p where G(x) is a quadratic residue.

    Uses get_y_unshifted_genus2 from search_common, which evaluates G(x) mod p
    via COEFFS_GENUS2 / FINITE_FIELD globals.  For large p (> ~10^6) this linear
    scan can be slow; at p=65537 it typically finds a valid x within a few tries.
    """
    xs = list(range(p))
    rng.shuffle(xs)
    for x in xs:
        if get_y_unshifted_genus2(x) is not None:
            return x
    raise RuntimeError(f"_random_curve_x: no valid x found on curve mod {p}")

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
    """Thin wrapper so both walks are built identically."""
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
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-chain merge time experiment")
    ap.add_argument("--steps-a",   type=int, default=DEFAULT_STEPS_A)
    ap.add_argument("--steps-b",   type=int, default=DEFAULT_STEPS_B)
    ap.add_argument("--seed-a",    type=int, default=DEFAULT_SEED_A)
    ap.add_argument("--seed-b",    type=int, default=DEFAULT_SEED_B)
    ap.add_argument("--snapshot",  type=str, default=DEFAULT_SNAPSHOT,
                    help="Path for A's leaf snapshot JSON")
    ap.add_argument("--skip-a",    action="store_true",
                    help="Skip running walk A; load existing --snapshot instead")
    ap.add_argument("--results",   type=str, default="merge_results.json",
                    help="Where to write the metrics JSON")
    ap.add_argument("--checkpoint-every", type=int, default=500)

    # Sync arguments with genus2_markov_module
    ap.add_argument("--max-n", type=int, default=80)
    ap.add_argument("--num-subsets", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args(argv)

    # -- Resolve project globals -------------------------------------------
    search_fn = None
    pool = None
    if _HAS_PROJECT:
        load_project_sources(verbose=False)
        import multiprocessing
        from search_lll.mumford.mumford_parallel import init_worker

        # Fetch required globals
        p = FINITE_FIELD
        coeffs = COEFFS_GENUS2
        prime_pool = PRIME_POOL

        # Build the persistent pool
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=args.num_workers, initializer=init_worker)
        chunk_size = 8

        search_fn = make_project_markov_search_fn(
            coeffs_genus2=coeffs,
            base_points=None,  # Leaving None lets the search_fn pull it dynamically per-walker
            p=p,
            prime_pool=prime_pool,
            num_subsets=args.num_subsets,
            num_workers=args.num_workers,
            debug=False,
            max_n=args.max_n,
            precomputed_residues=None,
            all_found_x=set(),
            pool=pool,
            chunk_size=chunk_size
        )
    else:
        p      = _FALLBACK_P
        coeffs = _FALLBACK_COEFFS
        if coeffs is None:
            sys.exit(
                "ERROR: project sources unavailable and _FALLBACK_COEFFS not set.  "
                "Edit merge_experiment.py or ensure genus2_markov_module is importable."
            )

    sqrt_p = math.sqrt(p)

    # -- Pick independent random starting points for A and B ---------------
    rng_a = _random.Random(args.seed_a)
    rng_b = _random.Random(args.seed_b)

    x0_a = _random_curve_x(p, rng_a)
    x0_b = _random_curve_x(p, rng_b)
    while x0_b == x0_a:
        x0_b = _random_curve_x(p, rng_b)

    _log(f"[seeds]  x0_A={x0_a}  x0_B={x0_b}")

    if _HAS_PROJECT:
        base_points_a = project_base_points_from_globals(current_x=x0_a, p=p)
        base_points_b = project_base_points_from_globals(current_x=x0_b, p=p)
    else:
        base_points_a = []
        base_points_b = []

    # -- Phase A -----------------------------------------------------------
    if not args.skip_a:
        _log(f"\n{'='*70}")
        _log(f"PHASE A  seed={args.seed_a}  x0={x0_a}  steps={args.steps_a}  p={p}  √p={sqrt_p:.1f}")
        _log(f"{'='*70}\n")

        walker_a = _build_walker(
            seed=args.seed_a,
            p=p, coeffs=coeffs, x0=x0_a, y0=None,
            base_points=base_points_a,
            verbose=True,
            log_path="walk_A.jsonl",
            search_fn=search_fn,
        )
        walker_a.run(args.steps_a)
        walker_a.save_leaf_snapshot(args.snapshot)

        vol_a = len(walker_a.global_leaves_seen)
        _log(f"\n[A done]  vol={vol_a}  ({vol_a/sqrt_p:.4f}×√p)  snapshot → {args.snapshot}\n")
    else:
        _log(f"[skip-a]  loading snapshot from {args.snapshot}")

    # -- Phase B -----------------------------------------------------------
    _log(f"\n{'='*70}")
    _log(f"PHASE B  seed={args.seed_b}  x0={x0_b}  steps={args.steps_b}  (quiet mode)")
    _log(f"{'='*70}\n")

    walker_b = _build_walker(
        seed=args.seed_b,
        p=p, coeffs=coeffs, x0=x0_b, y0=None,
        base_points=base_points_b,
        verbose=False,
        log_path="walk_B.jsonl",
        search_fn=search_fn,
    )
    n_foreign = walker_b.load_foreign_leaves(args.snapshot, label="A")
    _log(f"  Foreign leaves loaded: {n_foreign}")

    _quiet_run(walker_b, args.steps_b, checkpoint_every=args.checkpoint_every)

    # -- Report -----------------------------------------------------------
    metrics = _merge_report(walker_b, args.steps_b)

    with open(args.results, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    _log(f"[results] written → {args.results}")

    if walker_b.cantor_cache is not None:
        walker_b.cantor_summary()

    # -- Cleanup ----------------------------------------------------------
    if pool is not None:
        pool.close()
        pool.join()

    return metrics

def _quiet_run(walker: Genus2MetropolisWalker, steps: int, checkpoint_every: int = 500) -> None:
    """Run the walker with suppressed per-step banners.

    Only prints: checkpoint lines, merge hits (emitted by _store_record),
    and the final summary.
    """
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

        # Stop early once we have a merge — caller can decide whether to continue.
        if walker.first_merge_step is not None and display_step == walker.first_merge_step and display_step > 0:
            _log(f"  [B] first merge recorded at step {walker.first_merge_step}, continuing…")

    walker.config.verbose = original_verbose

def _merge_report(walker: Genus2MetropolisWalker, total_steps: int) -> dict:
    p = walker.p
    sqrt_p = math.sqrt(p)

    first_step = walker.first_merge_step
    first_vol  = walker.first_merge_vol   # total_leaf_insertions at first merge
    total_ins  = walker.total_leaf_insertions

    # Primary ratio: leaf insertions explored before first merge, normalised by √p.
    # This is the correct effort metric — step count is misleading because each
    # step produces O(15-25) leaves, so "step 0" ≠ "zero effort".
    if first_vol is not None:
        ins_ratio = first_vol / sqrt_p
    else:
        ins_ratio = total_ins / sqrt_p

    # Secondary (legacy) ratio based on step count.
    step_ratio = (first_step / sqrt_p) if first_step is not None else (total_steps / sqrt_p)

    # Volume (unique graph nodes) at the time of first merge.
    vol_at_merge = 0
    if first_step is not None:
        for entry in walker.merge_log:
            if entry[0] == first_step:
                vol_at_merge = entry[2]  # entry = (step_index, leaf_insertions, graph_vol, leaf)
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
