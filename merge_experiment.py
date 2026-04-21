from __future__ import annotations
import argparse, json, math, random as _random, sys, time, multiprocessing
from pathlib import Path
from typing import Optional, List, Tuple
from search_common import FINITE_FIELD, get_y_unshifted_genus2, COEFFS_GENUS2, PRIME_POOL, BASE_DIVISOR, TARGET_DIVISOR, GROUP_MODULUS, SECRET_KEY, GENERATE_MIXED_RELATIONS, MAXN
from sage.all import *
from markov import *
from genus2_markov_module import make_project_markov_search_fn, load_project_sources, resolve_project_symbol
from math import sqrt, ceil
from search_lll.mumford.mumford_parallel import init_worker

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
_HAS_PROJECT = True

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DEFAULT_SNAPSHOT      = "walk_A_leaves.json"
DEFAULT_STEPS_A       = 4*int(ceil(0.2*sqrt(FINITE_FIELD)))
DEFAULT_STEPS_BCD     = 4*int(ceil(0.2*sqrt(FINITE_FIELD)))

_FALLBACK_P      = FINITE_FIELD
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
    xs = PREFERRED_X_COORDS
    if xs is None:
        # Fallback: try PROJECT_REGISTRY via resolve_project_symbol
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

# ---------------------------------------------------------------------------
# Force divisor atoms to appear as xi (chain state) in each walker
# ---------------------------------------------------------------------------

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
    # Walk A is seeded from BASE_DIVISOR root 0; walk B from root 1.
    # merge_leaf is the collision point — NOT the DLP target.
    # The DLP target is the TARGET_DIVISOR root (x0_c from PREFERRED_X_COORDS).
    try:
        div_xs = _divisor_seed_xs()
        gen_x, gen_partner_x, target_x_override, target_partner_x = div_xs
    except Exception:
        gen_x             = walker_a.history[0].xi if walker_a.history else None
        gen_partner_x     = walker_b.history[0].xi if walker_b.history else None
        target_x_override = merge_leaf   # last-resort fallback only
        target_partner_x  = None
        raise

    if verbose:
        _log(f"[collision_dlp] merge_leaf={merge_leaf}  (collision, not DLP target)")
        _log(f"[collision_dlp] generator roots: {gen_x}, {gen_partner_x}")
        _log(f"[collision_dlp] target    roots: {target_x_override}, {target_partner_x}")

    protected = set()
    for x in (gen_x, gen_partner_x, target_x_override, target_partner_x):
        if x is not None:
            protected.add(int(x))
    return dlp_from_merged_walks(
        walker_a, walker_b,
        target_x=target_x_override,
        group_order=group_order,
        generator_x=gen_x,
        generator_x_partner=gen_partner_x,
        target_x_partner=target_partner_x,
        verbose=verbose,
        protected=protected,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from sage.all import floor
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
    ap.add_argument("--num-workers", type=int, default=20)
    # DLP options
    ap.add_argument("--dlp",        action="store_true",
                    help="Attempt DLP solve from merged relation matrices after the run")
    ap.add_argument("--dlp-target", type=int, default=None,
                    help="x-coordinate of the DLP target point (integer mod p)")
    ap.add_argument("--group-order", type=int, default=None,
                    help="Known group order n for the DLP solve (optional)")
    ap.add_argument("--seed-xs", type=int, nargs="+", metavar="X",
                    help="Override starting x-coords for walks A/B/C/D (1–4 ints). "
                         "Provided values overwrite the corresponding PREFERRED_X_COORDS "
                         "entries in order; unspecified walks keep their divisor-root seeds.")
    args = ap.parse_args(argv)

    # -- Resolve project globals -------------------------------------------
    search_fn = None
    pool = None
    if _HAS_PROJECT:
        load_project_sources(verbose=False)

        p          = FINITE_FIELD
        coeffs     = COEFFS_GENUS2
        prime_pool = PRIME_POOL

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=args.num_workers, initializer=init_worker)
        chunk_size = floor(MAXN*2/args.num_workers)

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

    # Apply --seed-xs overrides (up to 4 values, in A/B/C/D order).
    if args.seed_xs:
        if len(args.seed_xs) > 4:
            ap.error("--seed-xs accepts at most 4 values (one per walk A/B/C/D)")
        walk_vars = [x0_a, x0_b, x0_c, x0_d]
        for i, val in enumerate(args.seed_xs):
            walk_vars[i] = val
        x0_a, x0_b, x0_c, x0_d = walk_vars
        _log(f"\n[seeds] --seed-xs override applied: {args.seed_xs}")

    _log(f"\n[seeds] divisor roots (PREFERRED_X_COORDS): {divisor_xs}")
    _log(f"[seeds]   BASE_DIVISOR   roots: x0_A={x0_a}, x0_B={x0_b}")
    _log(f"[seeds]   TARGET_DIVISOR roots: x0_C={x0_c}, x0_D={x0_d}")

    # In GENERATE_MIXED_RELATIONS mode the interesting xi is only the seed
    # atom (step 0); after one step the chain leaves G/T territory, so injection
    # yields nothing useful.  G atoms = BASE_DIVISOR roots, T atoms = TARGET.
    G_atoms = {x0_a, x0_b}
    T_atoms = {x0_c, x0_d}

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
        walker_a.config.spectral_enabled = False
        walker_a.mat_chain = None
        walker_a.mat_graph = None

        if GENERATE_MIXED_RELATIONS:
            _log("[A] GENERATE_MIXED_RELATIONS: running 1 step then injecting T atoms")
            _quiet_run(walker_a, 1, checkpoint_every=args.checkpoint_every,
                       label="A", collective_leaves=None,
                       nullity_every=0, peer_walkers=[])
            n_inj = walker_a.generate_mixed_relations(
                list(T_atoms), seed_atoms=G_atoms, label="T"
            )
            _log(f"[A] mixed injection complete: {n_inj} relations added")
        else:
            _quiet_run(walker_a, args.steps_a, checkpoint_every=args.checkpoint_every,
                       label="A", collective_leaves=None,
                       nullity_every=0, peer_walkers=[])
        walker_a.save_leaf_snapshot(args.snapshot)

        vol_a = len(walker_a.global_leaves_seen)
        _log(f"\n[A done]  vol={vol_a}  ({vol_a/sqrt_p:.4f}*sqrt_p)  snapshot -> {args.snapshot}\n")
    else:
        _log(f"[skip-a]  loading snapshot from {args.snapshot}")

    # -- Helper: build + run one of the B/C/D walks -----------------------
    _ROLE = {
        "B": "BASE_DIVISOR root 1",
        "C": "TARGET_DIVISOR root 0",
        "D": "TARGET_DIVISOR root 1",
    }

    # collective_leaves starts as walk A's leaf set (or empty if --skip-a).
    # Each secondary walk adds to it, so novelty is measured against ALL prior
    # chains rather than just the walk's own history.  This prevents B from
    # looking artificially high-novelty when it's really just re-covering A's ground.
    collective_leaves: set = set(walker_a.global_leaves_seen) if walker_a is not None else set()
    # peer_walkers accumulates finished walkers so nullity checks during C/D
    # see the combined rank of A+B, A+B+C, etc. rather than just their own rows.
    peer_walkers_done: List = [walker_a] if walker_a is not None else []

    def _run_secondary(label, x0, seed, log_path):
        role = _ROLE.get(label, "?")

        _log(f"\n{'='*70}")
        _log(f"PHASE {label}  x0={x0}  ({role})  steps={args.steps_bcd}  p={p}  sqrt_p={sqrt_p:.1f}")
        _log(f"{'='*70}\n")
        w = _build_walker(
            seed=seed, p=p, coeffs=coeffs, x0=x0, y0=None,
            base_points=_bp(x0), verbose=False, log_path=log_path,
            search_fn=search_fn,
        )

        n_foreign = w.load_foreign_leaves(args.snapshot, label="A")
        _log(f"[{label}] foreign leaves loaded from A snapshot: {n_foreign}")
        # Spectral reports are expensive and not useful on sub-chains mid-run.
        w.config.spectral_enabled = False
        w.mat_chain = None
        w.mat_graph = None

        if GENERATE_MIXED_RELATIONS:
            # Determine which divisor set this walker's seed belongs to, then
            # inject all atoms from the opposite set.
            seed_fp = w.base_ring(x0)
            if seed_fp in {w.base_ring(a) for a in G_atoms}:
                inject_atoms = list(T_atoms)
                inject_label = "T"
                walker_seed_atoms = G_atoms
            else:
                inject_atoms = list(G_atoms)
                inject_label = "G"
                walker_seed_atoms = T_atoms

            _log(f"[{label}] GENERATE_MIXED_RELATIONS: 1 step then injecting {inject_label} atoms")
            _quiet_run(w, 1, checkpoint_every=args.checkpoint_every,
                       label=label, collective_leaves=collective_leaves,
                       nullity_every=0, peer_walkers=list(peer_walkers_done))
            n_inj = w.generate_mixed_relations(
                inject_atoms, seed_atoms=walker_seed_atoms, label=inject_label
            )
            _log(f"[{label}] mixed injection complete: {n_inj} relations added")
        else:
            _quiet_run(w, args.steps_bcd, checkpoint_every=args.checkpoint_every,
                       label=label, collective_leaves=collective_leaves,
                       nullity_every=0, peer_walkers=list(peer_walkers_done))
        peer_walkers_done.append(w)
        return w

    walker_b = _run_secondary("B", x0_b, seed=1, log_path="walk_B.jsonl")
    walker_c = _run_secondary("C", x0_c, seed=2, log_path="walk_C.jsonl")
    walker_d = _run_secondary("D", x0_d, seed=3, log_path="walk_D.jsonl")

    # -- Close worker pool as soon as walks are done ----------------------
    # All multiprocessing work is finished; release workers immediately so
    # their RAM is reclaimed before the (potentially large) DLP solve.
    if pool is not None:
        pool.close()
        pool.join()
        pool = None
        _log("[pool] worker pool closed")

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

    # -- DLP solve (optional) -- all four walks stacked -------------------
    if args.dlp and walker_a is not None:
        _log(f"\n{'='*70}")
        _log(f"DLP SOLVE  (A + B + C + D relation matrices stacked)")
        _log(f"{'='*70}\n")

        # PREFERRED_X_COORDS order: [x0_A, x0_B, x0_C, x0_D]
        #   x0_A, x0_B = BASE_DIVISOR roots  (generator G)
        #   x0_C, x0_D = TARGET_DIVISOR roots (challenge T)
        target_x         = args.dlp_target if args.dlp_target is not None else x0_c
        target_x_partner = x0_d

        if target_x is not None:
            dlp_result = dlp_from_merged_walks(
                [walker_a, walker_b, walker_c, walker_d],
                target_x=target_x,
                group_order=args.group_order,
                generator_x=x0_a,
                generator_x_partner=x0_b,
                target_x_partner=target_x_partner,
                verbose=True,
                protected=set(int(x) for x in divisor_xs),
            )
            metrics["dlp"] = dlp_result
            _log(f"[dlp] result: {dlp_result}")
        else:
            _log("[dlp] no --dlp-target given; skipping solve.")
    elif args.dlp:
        _log("[dlp] skipped: walk A not run (--skip-a).")

    run_all_checks(
        walkers     = [walker_a, walker_b, walker_c, walker_d],
        divisor_xs  = divisor_xs,
        group_order = int(GROUP_MODULUS),
        known_key   = SECRET_KEY,
        p           = p,
        coeffs      = coeffs,
        dump_path   = "relation_matrix.h5",   # ← add this
        augment     = True,                   # append-not-overwrite
    )

    # -- Write results -----------------------------------------------------
    with open(args.results, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    _log(f"[results] written -> {args.results}")

    for w in (walker_b, walker_c, walker_d):
        if getattr(w, 'cantor_cache', None) is not None:
            w.cantor_summary()

    return metrics

# ---------------------------------------------------------------------------
# Secondary-walk runner (verbose, label-aware)
# ---------------------------------------------------------------------------

def _quiet_run(
    walker: Genus2MetropolisWalker,
    steps: int,
    checkpoint_every: int = 1,
    label: str = "?",
    collective_leaves: Optional[set] = None,
    nullity_every: int = 100,
    peer_walkers: Optional[List] = None,
) -> None:
    """Run a secondary walk with full per-step verbose output.

    Uses walker.run(1) per step so the secondary chain gets the exact same
    ====== block as the primary walk — you can see what chain/walk you're on
    from the [walk-LABEL] header lines bracketing each step.

    collective_leaves -- if supplied, novelty rate vs the shared pool
    (A + all prior secondary chains) is printed after each run(1) call.
    Pass a set that starts from walker_a.global_leaves_seen; it is updated
    in-place as each step runs so later chains see the correct baseline.

    nullity_every -- compute and print combined nullity every N steps.
    peer_walkers  -- walkers already finished (e.g. [walker_a, walker_b]);
    their relation matrices are stacked with the current walker's to get the
    true combined nullity as this walk accumulates rows.
    """
    walker.config.verbose = True

    sqrt_p = math.sqrt(walker.p) if walker.p is not None else float("nan")
    t0     = time.time()
    tag    = f"[walk-{label}]"

    _log(f"\n{tag} starting  steps={steps}  xi={getattr(walker, 'current_x', '?')}  "
         f"foreign_leaves={len(walker.foreign_leaves) if walker.foreign_leaves is not None else 'none'}"
         f"  collective_ref={'yes ('+str(len(collective_leaves))+' leaves)' if collective_leaves is not None else 'own'}")

    # -- Precompute peer relation matrices once, outside the step loop --------
    # Rebuilding all peer walker matrices on every nullity checkpoint is O(peers
    # * rows * cols) per check, growing quadratically.  The peers are finished
    # walkers whose matrices never change, so we fetch them once here and reuse
    # the cached result on every checkpoint.  Only the current walker's matrix
    # is re-fetched inside the loop (it grows by one row per step).
    _Fp_rank = GF(2**31 - 1)   # Mersenne prime; fast rank over this field

    _peer_mats: List       = []
    _peer_atom_lists: List = []
    if nullity_every > 0:
        for w in (peer_walkers or []):
            try:
                mat, atoms, _ = w.relation_matrix(include_step_leaves=False)
                if mat.nrows() > 0:
                    _peer_mats.append(mat)
                    _peer_atom_lists.append(list(atoms))
            except Exception:
                raise

    for i in range(steps):
        # Snapshot only the leaves that are genuinely new-to-collective this step.
        # We compare collective BEFORE vs AFTER the step, but only counting leaves
        # the walker itself just discovered (intersection of new own-leaves with
        # collective).  Do NOT use collective.update(global_leaves_seen) — that
        # would dump the walker's full accumulated history every step, inflating
        # the before→after delta and producing >100% novelty.
        own_before = set(walker.global_leaves_seen)
        coll_size_before = len(collective_leaves) if collective_leaves is not None else None  # noqa: F841

        # Use run(1) so the full ====== diagnostic block is printed, identical
        # to the on-chain (walk A) output.  The walk-label banner before and
        # after makes clear which chain produced each block when outputs from
        # multiple chains are interleaved.
        _log(f"\n{tag} >>>>>> step {i+1}/{steps} <<<<<<")
        walker.run(1, label=label)
        _log(f"{tag} <<<<<< step {i+1}/{steps} done  xi_now={int(walker.current_x)} >>>>>>")
        display_step = i + 1

        # Novelty vs collective: count only the leaves this step actually added
        # to global_leaves_seen that were absent from the collective pool.
        if collective_leaves is not None:
            new_own_this_step = walker.global_leaves_seen - own_before
            new_to_coll = new_own_this_step - collective_leaves
            collective_leaves.update(new_own_this_step)   # only add genuinely new leaves
            leaves_found = len(new_own_this_step)
            novelty_coll = len(new_to_coll) / leaves_found if leaves_found > 0 else 0.0
            novelty_str  = f"novelty={novelty_coll:.1%} vs collective ({len(collective_leaves)} total)"
        else:
            novelty_str = ""

        own_vol  = len(walker.global_leaves_seen)
        ins      = walker.total_leaf_insertions
        merges   = len(walker.merge_log) if walker.foreign_leaves is not None else "n/a"
        elapsed  = time.time() - t0

        # -- merge announcement (fires exactly once, immediately) ----------
        if (walker.first_merge_step is not None
                and len(walker.history) - 1 == walker.first_merge_step):
            merge_leaf = walker.merge_log[0][3] if walker.merge_log else "?"
            _log(
                f"\n{tag} *** FIRST MERGE at step {display_step} ***"
                f"  leaf={merge_leaf}  own_vol={own_vol} ({own_vol/sqrt_p:.4f}\u00d7\u221ap)"
                f"  elapsed={elapsed:.1f}s\n"
            )

        # -- optional nullity checkpoint ----------------------------------
        # Two fixes applied here vs the original:
        #
        # 1. Peer matrices are precomputed above the loop (see _peer_mats /
        #    _peer_atom_lists).  Only the *current* walker's matrix is fetched
        #    fresh; peer matrices are constant and reused each checkpoint.
        #
        # 2. The combined row list is accumulated into a ZZ matrix and then
        #    converted to GF via .change_ring().  The old code passed a Python
        #    list-of-lists directly to sage_matrix(GF(...), rows), which performs
        #    one GF coercion per element in Python — O(rows*cols) transient
        #    objects.  change_ring() does the same conversion entirely in C,
        #    avoiding the object-creation overhead.
        if nullity_every > 0 and display_step % nullity_every == 0:
            try:
                # Fetch only the current walker's fresh matrix.
                cur_mats       = list(_peer_mats)
                cur_atom_lists = list(_peer_atom_lists)
                try:
                    mat, atoms, _ = walker.relation_matrix(include_step_leaves=False)
                    if mat.nrows() > 0:
                        cur_mats.append(mat)
                        cur_atom_lists.append(list(atoms))
                except Exception:
                    raise

                if cur_mats:
                    # Union atom sets across all matrices.
                    all_atoms: List = list(cur_atom_lists[0])
                    atom_set = set(map(str, all_atoms))
                    for atms in cur_atom_lists[1:]:
                        for a in atms:
                            if str(a) not in atom_set:
                                all_atoms.append(a)
                                atom_set.add(str(a))
                    nc   = len(all_atoms)
                    aidx = {str(a): idx for idx, a in enumerate(all_atoms)}

                    # Build a flat list of ZZ rows (Python ints stay as ints
                    # until we call matrix(ZZ, ...) which packs them in one shot).
                    rows_int: List[List[int]] = []
                    for mat, atms in zip(cur_mats, cur_atom_lists):
                        cols_src = [aidx[str(a)] for a in atms]
                        for r in range(mat.nrows()):
                            row = [0] * nc
                            for c_src, c_dst in enumerate(cols_src):
                                row[c_dst] = int(mat[r, c_src])
                            rows_int.append(row)

                    # Build ZZ matrix first (one C-level allocation), then
                    # change_ring to GF (one C-level pass — no per-element Python
                    # objects, unlike sage_matrix(GF(...), rows_int)).
                    M_zz = matrix(ZZ, rows_int)
                    rk      = M_zz.change_ring(_Fp_rank).rank()
                    nullity = nc - rk
                    total_rows = len(rows_int)
                    nullity_str2 = (
                        f"  [nullity check @ step {display_step}]"
                        f"  rows={total_rows}  cols={nc}  rank={rk}  nullity={nullity}"
                        f"  (need {max(0, nullity - 1)} more independent rows)"
                    )
                    _log(f"{tag}{nullity_str2}")
            except Exception as exc:
                _log(f"{tag} [nullity check failed: {exc}]")
                raise

        # Summary line with chain label so you always know which walk this is.
        _log(
            f"{tag} step {display_step:>5}/{steps}"
            f"  own_vol={own_vol} ({own_vol/sqrt_p:.4f}\u00d7\u221ap)"
            f"  ins={ins} ({ins/sqrt_p:.4f}\u00d7\u221ap)"
            f"  merges={merges}"
            f"  {novelty_str}"
            f"  elapsed={elapsed:.1f}s"
        )

    vol_final = len(walker.global_leaves_seen)
    _log(
        f"\n{tag} done  vol={vol_final} ({vol_final/sqrt_p:.4f}\u00d7\u221ap)"
        f"  total_merges={len(walker.merge_log) if walker.foreign_leaves is not None else 'n/a'}"
        f"  elapsed={time.time()-t0:.1f}s"
    )

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

def _log(msg: str) -> None:
    print(msg, flush=True)

_INFINITY_SENTINEL = "∞"
_RANK_PRIME_NULLITY = 2**31 - 1   # Mersenne prime for nullity kernel computation

def _dlp_collect_matrices(walkers, verbose: bool):
    """Step 1: Build per-walker relation matrices, skipping empty ones.

    Returns (mats, atom_lists).  Raises on relation_matrix() failure.
    """
    mats = []
    atom_lists = []
    for i, w in enumerate(walkers):
        try:
            mat, atoms, _used = w.relation_matrix(include_step_leaves=False)
        except Exception as exc:
            _log(f"[dlp_list] walker[{i}] relation_matrix() failed: {exc}")
            raise

        if mat.nrows() == 0:
            if verbose:
                _log(f"[dlp_list] walker[{i}] has empty relation matrix -- skipping")
            continue

        mats.append(mat)
        atom_lists.append(atoms)

    return mats, atom_lists

def _dlp_union_columns(mats, atom_lists, verbose: bool):
    """Step 2: Reindex all matrices into a shared column space and stack them.

    Returns (M_combined, all_atoms_ordered, atom_index, ranks).
    Per-walker rank computation is skipped here (expensive and not needed
    before pruning); only row/col dims are logged.  The single rank that
    matters is computed on the pruned matrix before the solve.
    """

    all_atoms_ordered = list(atom_lists[0])
    atom_set = set(map(str, all_atoms_ordered))
    for atoms in atom_lists[1:]:
        for a in atoms:
            if str(a) not in atom_set:
                all_atoms_ordered.append(a)
                atom_set.add(str(a))

    n_cols = len(all_atoms_ordered)
    atom_index = {str(a): i for i, a in enumerate(all_atoms_ordered)}

    def _reindex(mat, atoms_src):
        cols_src = [atom_index[str(a)] for a in atoms_src]
        M = matrix(ZZ, mat.nrows(), n_cols)
        for r in range(mat.nrows()):
            for c_src, c_dst in enumerate(cols_src):
                M[r, c_dst] = mat[r, c_src]
        return M

    reindexed = []
    ranks = []   # kept for API compatibility; values are None (not computed)
    for mat, atoms in zip(mats, atom_lists):
        M_i = _reindex(mat, atoms)
        reindexed.append(M_i)
        if verbose:
            _log(f"[dlp_list]   walker rows={mat.nrows()} atoms={mat.ncols()}")

    M_combined = reindexed[0]
    for M_i in reindexed[1:]:
        M_combined = M_combined.stack(M_i)

    return M_combined, all_atoms_ordered, atom_index, ranks

def _dlp_prune(M_combined, all_atoms_ordered: list, verbose: bool, protected=None):
    """Step 3: Drop dest-only (pendant) columns and their single incident rows.

    A dest-only atom appears only as xj/xk (coefficient 1), never as xi
    (coefficient d-2 = 3).  It contributes a free variable to the kernel and
    adds directly to the nullity without adding rank.  Pruning removes these
    pendant leaves iteratively to fixed point, which (a) reduces n_cols,
    (b) removes the rows that only served those leaves, and (c) strictly
    reduces nullity without affecting the DLP solution for any non-pruned atom.

    protected: set of atoms that must survive pruning (e.g. divisor roots).

    Returns (M_pruned, pruned_atoms, pruned_atom_index).
    """

    M_pruned, pruned_atoms, removed = prune_dest_only(M_combined, all_atoms_ordered, protected=protected)
    if removed and verbose:
        _log(
            f"[dlp_list] pruned {len(removed)} dest-only atoms "
            f"and {len(removed)} pendant rows"
        )
    pruned_atom_index = {str(a): i for i, a in enumerate(pruned_atoms)}
    return M_pruned, pruned_atoms, pruned_atom_index

def _dlp_rank_check(A, n_cols: int, rank_combined: int, n_rows_combined: int, verbose: bool):
    """Step 6: Compute rank of augmented system; return (rank_aug, kernel_dim).

    kernel_dim > 1 means underdetermined; caller should bail out.
    """
    rank_aug = A.rank()
    kernel_dim = n_cols - rank_aug
    if verbose:
        _log(
            f"[dlp_list] rank_augmented={rank_aug}  nullity={kernel_dim}"
            f"  (need nullity=1 for unique solution)"
        )
    if kernel_dim > 1:
        _log(
            f"[dlp_list] UNDERDETERMINED: nullity={kernel_dim} -- need {kernel_dim - 1} more "
            f"independent relation rows before solve is possible.  "
            f"(rows={n_rows_combined}, cols={n_cols}, rank={rank_combined})"
        )
    return rank_aug, kernel_dim

def _dlp_nullity_prune(
    A,
    b,
    pruned_atoms: list,
    atom_index: dict,
    n: int,
    inf_col,
    gen_col: int,
    gen_partner_col,
    target_col: int,
    verbose: bool = True,
) -> tuple:
    """Pre-solve nullity check.  Returns (A_fixed, b_fixed, fixable).

    Classifies each kernel direction of the current augmented system A:

      - GAUGE (single entry at ∞): leave it, expected.
      - ISOLATED (single entry, not ∞): pin a[atom] = 0 by appending a row.
        Safe because isolated atoms have no relation rows constraining them.
      - PARITY / CONSERVATION (all nonzero coefficients equal): cannot be
        auto-fixed without knowing the conserved value.  Reports the conflict
        and returns fixable=False.
      - OTHER: also reported as unfixable.

    After pinning all isolated atoms the augmented system is rebuilt and its
    nullity re-checked.  If it reaches 0 the solve should succeed.
    """
    Fp = GF(n)
    ker = A.right_kernel()
    nullity = ker.dimension()

    if verbose:
        _log(f"[pre-solve nullity] A is {A.nrows()}×{A.ncols()}  nullity={nullity}")

    if nullity == 0:
        if verbose:
            _log("[pre-solve nullity] ✓ fully determined — no pinning needed")
        return A, b, True

    extra_rows = []
    extra_rhs  = []
    fixable    = True

    for vi, vec in enumerate(ker.basis()):
        support = [(j, int(vec[j])) for j in range(len(vec)) if vec[j] != Fp(0)]

        # Gauge — skip
        if len(support) == 1 and inf_col is not None and support[0][0] == inf_col:
            if verbose:
                _log(f"[pre-solve nullity]   kernel[{vi}]: gauge (∞) — skipping")
            continue

        # Isolated atom — pin to 0
        if len(support) == 1:
            j, _ = support[0]
            atom = pruned_atoms[j]
            pin_row = vector(Fp, A.ncols())
            pin_row[j] = Fp(1)
            extra_rows.append(pin_row)
            extra_rhs.append(Fp(0))
            if verbose and False:
                _log(f"[pre-solve nullity]   kernel[{vi}]: isolated atom={atom} — pinning a[{atom}]=0")
            continue

        # Parity or other — unfixable
        fixable = False
        coeffs = sorted(set(int(c) for _, c in support))
        is_flat = len(coeffs) == 1
        kind = "PARITY" if is_flat else "OTHER"
        if verbose:
            c0 = coeffs[0] if is_flat else None
            _log(
                f"[pre-solve nullity]   kernel[{vi}]: {kind}  support_size={len(support)}"
                + (f"  all_coeffs={c0}" if is_flat else f"  distinct_coeffs={coeffs}")
            )
            if is_flat:
                inv5 = pow(5, -1, n) if n > 5 else None
                _log(
                    f"[pre-solve nullity]     Conservation law: {c0}·Σa[x]=0 over {len(support)} atoms."
                    f"\n[pre-solve nullity]     Anchor likely needs RHS=inv(5) mod {n} = {inv5}"
                    f" instead of 1."
                )

    if not extra_rows:
        if verbose:
            _log("[pre-solve nullity] no isolated atoms found — nothing pinned")
        return A, b, fixable

    A_extra = matrix(Fp, extra_rows)
    b_extra = vector(Fp, extra_rhs)
    A_fixed = A.stack(A_extra)
    b_fixed = vector(Fp, b.list() + extra_rhs)

    new_nullity = A_fixed.right_kernel().dimension()
    if verbose:
        _log(
            f"[pre-solve nullity] pinned {len(extra_rows)} isolated atom(s) — "
            f"nullity {nullity} → {new_nullity}"
        )
        if new_nullity == 0:
            _log("[pre-solve nullity] ✓ fully determined after pinning")
        elif fixable:
            # This would be surprising — isolated pinning didn't finish the job
            fixable = False
            _log(f"[pre-solve nullity] ✗ still underdetermined (nullity={new_nullity}) after pinning")

    return A_fixed, b_fixed, fixable

def _dlp_search_free_parameter(
    A, b, all_atoms_ordered, atom_index, n, target_col, target_partner_col, verbose
):
    """Fallback when solve_right fails with nullity=1 in the augmented system.

    The system A*x = b has a 1-D solution space:
        x = x_particular + t * v_kernel,   t in GF(n)

    where v_kernel is the unique (up to scale) kernel vector of A.

    We find x_particular by pinning one free column to 0 (making the system
    fully determined), then BSGS over t to find the unique t such that the
    implied DLP value satisfies k*G == T in the Jacobian.

    BSGS costs O(sqrt(n)) Jacobian multiplications — for n~25000 that is ~160
    steps, essentially free.

    Returns (dlp_val, t, method_str) or (None, None, reason_str).
    """
    Fp = GF(n)

    # --- find the kernel vector of A ---
    ker = A.right_kernel()
    if ker.dimension() != 1:
        if verbose:
            _log(f"[free_param] kernel dimension={ker.dimension()}, expected 1 — aborting")
        return None, None, f"kernel_dim={ker.dimension()}"

    v_ker = ker.basis()[0]   # the free direction, over GF(n)

    # Find a column with nonzero kernel coefficient to pin.
    # Prefer a non-target, non-divisor column so we don't over-constrain the answer.
    pin_col = None
    for j in range(len(v_ker)):
        if v_ker[j] != Fp(0) and j != target_col and j != target_partner_col:
            pin_col = j
            break
    if pin_col is None:
        if verbose:
            _log("[free_param] could not find a safe column to pin — aborting")
        return None, None, "no_pin_col"

    # Build A_pinned by appending a row that pins a[pin_col] = 0.
    pin_row = vector(Fp, A.ncols())
    pin_row[pin_col] = Fp(1)
    A_pinned = A.stack(matrix(Fp, [pin_row]))
    b_pinned = vector(Fp, b.list() + [Fp(0)])

    try:
        x_part = A_pinned.solve_right(b_pinned)
    except ValueError as exc:
        if verbose:
            _log(f"[free_param] particular solution failed: {exc}")
        return None, None, "particular_solution_failed"

    # DLP value as a function of t:
    #   dlp(t) = (x_part[tgt0] + t*v_ker[tgt0]) + (x_part[tgt1] + t*v_ker[tgt1])
    #          = base_val + t * step_val   (mod n)
    base_val = Fp(0)
    step_val = Fp(0)
    for col in [target_col, target_partner_col]:
        if col is not None:
            base_val += x_part[col]
            step_val += v_ker[col]

    base_val = int(base_val)
    step_val = int(step_val)

    if verbose:
        _log(f"[free_param] dlp(t) = {base_val} + t*{step_val}  (mod {n})")
        _log(f"[free_param] starting BSGS over t in GF({n}), sqrt_n ~ {int(n**0.5)+1} steps")

    if BASE_DIVISOR is None or TARGET_DIVISOR is None:
        if verbose:
            _log("[free_param] BASE_DIVISOR/TARGET_DIVISOR unavailable — falling back to linear scan")
        # Linear scan — only feasible for small n
        if n > 100000:
            return None, None, "no_divisors_n_too_large"
        for t in range(n):
            dlp_val = (base_val + t * step_val) % n
            # can't verify without divisors; just return first candidate
        return None, None, "no_divisors"

    # BSGS: find t such that dlp(t)*G == T
    # dlp(t) = base_val + t*step_val  =>  t*step_val*G = T - base_val*G
    # Let H = T - base_val*G,  Q = step_val*G
    # Find t such that t*Q = H   via BSGS in <G>

    import math as _math
    m = int(_math.isqrt(n)) + 1

    G = BASE_DIVISOR
    T = TARGET_DIVISOR

    # Baby steps: table[j*Q] = j  for j = 0..m-1
    Q = Integer(step_val) * G
    H = T - Integer(base_val) * G

    if verbose:
        _log(f"[free_param] BSGS: m={m} baby steps, then {m} giant steps")

    baby = {}
    cur = H.parent()(0)   # identity in Jacobian
    for j in range(m):
        key = str(cur)
        if key not in baby:
            baby[key] = j
        cur = cur + Q   # baby step: cur = j*Q ... wait, we want j*Q = H - i*(m*Q)
    # Recompute correctly:
    # We want t*Q = H.
    # Baby steps: store (j, j*Q) for j=0..m-1
    # Giant steps: check H - i*m*Q for i=0..m-1
    baby = {}
    cur = cur.parent()(0)
    for j in range(m):
        baby[str(cur)] = j
        cur = cur + Q

    mQ = Integer(m) * Q
    giant_cur = H
    for i in range(m + 1):
        key = str(giant_cur)
        if key in baby:
            j = baby[key]
            t = (i * m + j) % n
            dlp_val = (base_val + t * step_val) % n
            if verbose:
                _log(f"[free_param] BSGS found t={t}  dlp={dlp_val}")
            # Verify
            check = Integer(dlp_val) * G
            if check == T:
                if verbose:
                    _log(f"[free_param] ✓ verified: {dlp_val}*G == T")
                return dlp_val, t, "free_param_bsgs"
            else:
                if verbose:
                    _log(f"[free_param] ✗ t={t} dlp={dlp_val} did not verify — continuing")
        giant_cur = giant_cur - mQ

    if verbose:
        _log("[free_param] BSGS exhausted — no solution found")
    return None, None, "bsgs_exhausted"

def _dlp_solve(A, b, verbose: bool):
    """Solve A * x = b and return the full solution vector."""
    try:
        sol = A.solve_right(b)
    except ValueError as exc:
        msg = (
            f"[dlp_list] no solution for the affine system "
            f"({A.nrows()} rows, {A.ncols()} cols over GF({A.base_ring().characteristic()})).\n"
            f"  The anchor row is not compatible with the relation rows.\n"
            f"  This usually means one of:\n"
            f"    - the generator/target columns were pruned or never included,\n"
            f"    - the anchor was built from atoms not present in the matrix,\n"
            f"    - the group order/modulus is wrong,\n"
            f"    - or the relation set is missing enough independent constraints.\n"
            f"  Original solver error: {exc}"
        )
        _log(msg)
        raise ValueError(msg) from exc
    except Exception as exc:
        msg = f"[dlp_list] solve_right raised an unexpected error: {exc}"
        _log(msg)
        raise

    residual = A * sol - b
    if any(v != 0 for v in residual):
        raise RuntimeError(
            "[dlp_list] solve_right returned a vector, but A*sol != b "
            "(internal consistency check failed)"
        )

    return sol

def _dlp_verify(dlp_val: int, verbose: bool) -> bool:
    """Step 8: Jacobian verification against module-level BASE/TARGET divisors."""

    try:
        if BASE_DIVISOR is not None and TARGET_DIVISOR is not None:
            check = Integer(dlp_val) * BASE_DIVISOR
            verified = bool(check == TARGET_DIVISOR)
            if verbose:
                status = "VERIFIED" if verified else "MISMATCH"
                _log(f"[dlp_list] Jacobian verification: {dlp_val}*G == T?  {status}")
            return verified
    except Exception as exc:
        if verbose:
            _log(f"[dlp_list] verification skipped ({exc})")
        raise
    return False

def dlp_from_merged_walks(
    walkers,
    target_x,
    group_order=None,
    generator_x=None,
    generator_x_partner=None,
    target_x_partner=None,
    verbose: bool = True,
    protected=None,
):
    """DLP solve from the union of all walkers' relation matrices.

    Solves the normalized affine system

        M_combined * alpha = 0
        alpha[gen_x] + alpha[gen_partner_x] = 1

    over GF(l), then reads off alpha[target_x] as the DLP coefficient.
    Dest-only (pendant) atoms are pruned from M_combined before the solve to
    reduce nullity.
    """

    result = {
        "dlp": None,
        "method": None,
        "ranks": [],
        "rank_combined": None,
        "rank_augmented": None,
        "kernel_dim": None,
        "n_cols": None,
        "verified": False,
    }

    if not walkers:
        raise ValueError("dlp_from_merged_walks: no walkers supplied")

    # 1. Per-walker matrices
    mats, atom_lists = _dlp_collect_matrices(walkers, verbose)
    if not mats:
        _log("[dlp_list] all walks have empty relation matrices -- no solve possible.")
        return result

    # 2. Union column spaces, reindex, stack
    M_combined, all_atoms_ordered, atom_index, ranks = _dlp_union_columns(
        mats, atom_lists, verbose
    )
    result["ranks"] = ranks
    if verbose:
        _log(
            f"[dlp_list] combined: rows={M_combined.nrows()} cols={len(all_atoms_ordered)} "
            f"(pruning next, rank deferred to post-solve)"
        )

    # 3. Prune dest-only pendant columns/rows before resolving column indices.
    #    target_col/gen_col must be resolved against the post-prune atom_index.
    M_combined, all_atoms_ordered, atom_index = _dlp_prune(
        M_combined, all_atoms_ordered, verbose, protected=protected
    )
    n_cols = len(all_atoms_ordered)
    result["n_cols"] = n_cols

    # 4. Resolve column indices (after pruning so indices are correct)
    if target_x is None:
        _log("[dlp_list] target_x is None -- cannot solve.")
        return result
    if generator_x is None:
        _log("[dlp_list] generator_x is None -- cannot anchor.")
        return result

    target_col, target_partner_col, gen_col, gen_partner_col = _dlp_resolve_cols(
        atom_index,
        target_x,
        target_x_partner,
        generator_x,
        generator_x_partner,
        verbose,
    )

    if target_col is None:
        _log(f"[dlp_list] target x={target_x} not in combined leaf set after pruning.")
        return result
    if gen_col is None:
        _log(f"[dlp_list] generator x={generator_x} not in combined leaf set after pruning.")
        return result

    # 5. Resolve group order
    n = group_order
    if n is None:
        try:
            if GROUP_MODULUS is not None:
                n = int(GROUP_MODULUS)
        except Exception:
            raise
    if n is None:
        _log("[dlp_list] group_order unknown -- cannot solve mod l.")
        result["method"] = "no_group_order"
        return result
    if verbose:
        _log(f"[dlp_list] working mod l = {n}")

    # 6. Build affine system over GF(n)
    Fp = GF(n)
    _M_fp, A, b = _dlp_build_affine_system(
        M_combined, n_cols, gen_col, gen_partner_col, Fp,
        generator_x, generator_x_partner, verbose,
        atom_index=atom_index,   # NEW
    )

    # 6b. Pre-solve nullity check: compute kernel of A, pin isolated atoms to 0,
    #     report parity/conservation directions that can't be auto-fixed.
    inf_col_val = atom_index.get(_INFINITY_SENTINEL) or atom_index.get(str(_INFINITY_SENTINEL))
    A, b, _fixable = _dlp_nullity_prune(
        A, b,
        pruned_atoms=all_atoms_ordered,
        atom_index=atom_index,
        n=n,
        inf_col=inf_col_val,
        gen_col=gen_col,
        gen_partner_col=gen_partner_col,
        target_col=target_col,
        verbose=verbose,
    )
    if not _fixable and verbose:
        _log(
            "[dlp_list] system has non-trivial free direction(s) that cannot be auto-pinned.\n"
            "[dlp_list] proceeding to solve anyway — expect failure or underdetermined result."
        )

    # 7. Solve, then decode target-root logs from the solution vector
    sol = None
    try:
        sol = _dlp_solve(A, b, verbose)
    except ValueError:
        result["method"] = "solve_right_failed"
        # 7b. Fallback: if augmented system has nullity=1, the solution space is a
        #     1-D coset.  Search the free parameter via BSGS.
        if verbose:
            _log("[dlp_list] solve_right failed — attempting free-parameter BSGS fallback")
        dlp_val, t_free, fb_method = _dlp_search_free_parameter(
            A, b, all_atoms_ordered, atom_index, n,
            target_col, target_partner_col, verbose,
        )
        if dlp_val is not None:
            result["dlp"] = dlp_val
            result["method"] = fb_method
            result["free_parameter_t"] = t_free
            result["verified"] = _dlp_verify(dlp_val, verbose)
        else:
            if verbose:
                _log(f"[dlp_list] free-parameter fallback also failed: {fb_method}")
            result["method"] = f"solve_right_failed+{fb_method}"

    if sol is not None:
        target_log, target_partner_log, dlp_val = _dlp_extract_target_atom_logs(
            sol, target_col, target_partner_col, n, verbose
        )
        result["dlp"] = dlp_val
        result["method"] = "normalized_solve_right"
        result["target_atom_logs"] = {
            "target_x": int(target_x) if target_x is not None else None,
            "target_x_partner": int(target_x_partner) if target_x_partner is not None else None,
            "target_log": target_log,
            "target_partner_log": target_partner_log,
            "target_total_log": dlp_val,
        }

        # 8. Verify using the total target divisor log
        result["verified"] = _dlp_verify(dlp_val, verbose)

    # 9. Rank diagnostics — always last, only after solve attempt
    if verbose:
        _log("[dlp_list] computing rank diagnostics (deferred, may be slow) ...")
    rank_combined = _M_fp.rank()
    result["rank_combined"] = rank_combined
    rank_aug, kernel_dim = _dlp_rank_check(
        A, n_cols, rank_combined, M_combined.nrows(), verbose
    )
    result["rank_augmented"] = rank_aug
    result["kernel_dim"] = kernel_dim
    if kernel_dim > 1 and result.get("method") is None:
        result["method"] = "underdetermined"

    return result

def extract_target_atom_logs(Q, alpha, ell):
    """
    Given:
        Q     : target divisor in J(F_p)[ℓ]
        alpha : dict or array mapping x -> log(x)
        ell   : subgroup order

    Returns:
        (x1, log_x1), (x2, log_x2), total_log

    Raises:
        RuntimeError if structure is not as expected.
    """

    if Q.is_zero():
        raise RuntimeError("Target divisor Q is zero")

    u_poly = Q[0]

    if u_poly.degree() != 2:
        raise RuntimeError(f"Expected degree-2 u(x), got degree {u_poly.degree()}")

    roots = u_poly.roots()

    if len(roots) != 2:
        raise RuntimeError(f"Expected 2 roots, got {roots}")

    x_vals = []
    for root, mult in roots:
        if mult != 1:
            raise RuntimeError(f"Non-simple root in target divisor: {root}^{mult}")
        x_vals.append(int(root))

    x1, x2 = x_vals

    if x1 not in alpha:
        raise RuntimeError(f"x1={x1} not in solution vector")
    if x2 not in alpha:
        raise RuntimeError(f"x2={x2} not in solution vector")

    log_x1 = int(alpha[x1]) % ell
    log_x2 = int(alpha[x2]) % ell

    total = (log_x1 + log_x2) % ell

    return (x1, log_x1), (x2, log_x2), total

def verify_target_log(G, Q, total_log, ell):
    """
    Check that computed log actually reconstructs Q.
    """
    if (Integer(total_log) * G - Q).is_zero():
        return True
    else:
        raise RuntimeError(
            f"Log mismatch: computed log does not reproduce Q\n"
            f"log={total_log}"
        )

def _dlp_resolve_cols(
    atom_index: dict,
    target_x,
    target_x_partner,
    generator_x,
    generator_x_partner,
    verbose: bool,
):
    """Map target/generator x-values to column indices."""
    target_col = atom_index.get(str(target_x)) if target_x is not None else None
    target_partner_col = atom_index.get(str(target_x_partner)) if target_x_partner is not None else None

    gen_col = atom_index.get(str(generator_x)) if generator_x is not None else None

    gen_partner_col = None
    if generator_x_partner is not None:
        gen_partner_col = atom_index.get(str(generator_x_partner))
        if gen_partner_col is None and verbose:
            _log(
                f"[dlp_list] generator partner x={generator_x_partner} not in leaf set "
                f"-- anchor will use single-column form."
            )

    if target_x_partner is not None and target_partner_col is None and verbose:
        _log(
            f"[dlp_list] target partner x={target_x_partner} not in leaf set "
            f"-- target will use single-column form."
        )

    return target_col, target_partner_col, gen_col, gen_partner_col

def _dlp_extract_target_atom_logs(
    sol,
    target_col: int,
    target_partner_col,
    n: int,
    verbose: bool,
):
    """
    Extract the target atom logs from the solution vector.

    Returns:
        target_log, target_partner_log, total_target_log
    """
    if target_col is None:
        raise RuntimeError("[dlp_list] target_col is None in _dlp_extract_target_atom_logs")

    target_log = int(sol[target_col]) % n
    target_partner_log = None

    if target_partner_col is not None:
        target_partner_log = int(sol[target_partner_col]) % n
        total_target_log = (target_log + target_partner_log) % n
    else:
        total_target_log = target_log

    if verbose:
        if target_partner_log is None:
            _log(f"[dlp_list] target atom log: a[T1]={target_log}")
        else:
            _log(
                f"[dlp_list] target atom logs: "
                f"a[T1]={target_log}, a[T2]={target_partner_log}, "
                f"sum={total_target_log}"
            )

    return target_log, target_partner_log, total_target_log

def _dlp_build_affine_system(
    M_combined, n_cols: int, gen_col: int, gen_partner_col, Fp,
    generator_x, generator_x_partner, verbose: bool,
    atom_index=None,   # NEW
):
    """Step 5: Lift M_combined to GF(n), append gauge-fix and anchor rows."""

    char = int(Fp.characteristic())
    if verbose:
        _log(
            f"[dlp_list] combined over GF({char}): "
            f"rows={M_combined.nrows()} cols={n_cols} "
            f"(rank deferred to post-solve)"
        )

    M_fp = M_combined.change_ring(Fp)

    rows = []
    rhs = []

    # Homogeneous walk relations.
    for r in range(M_fp.nrows()):
        rows.append(M_fp.row(r))
        rhs.append(Fp(0))

    # NEW: gauge-fix infinity.
    inf_col = None
    if atom_index is not None:
        inf_col = atom_index.get(_INFINITY_SENTINEL)
        if inf_col is None:
            inf_col = atom_index.get(str(_INFINITY_SENTINEL))

    if inf_col is not None:
        inf_row = vector(Fp, n_cols)
        inf_row[inf_col] = Fp(1)
        rows.append(inf_row)
        rhs.append(Fp(0))
        if verbose:
            _log("[dlp_list] gauge fix   : a[∞] = 0")
    elif verbose:
        _log("[dlp_list] gauge fix   : ∞ column not found, skipping a[∞] = 0")

    # Anchor row: a[gen0] - a[gen1] = 1.
    # Every relation row has coefficient sum 3+1+1-5=0 (conservation law).
    # Any appended row must also have coefficient sum 0, otherwise it directly
    # contradicts the invariant and the system is inconsistent.
    # a[gen0]=0 and a[gen1]=1 each have sum 1 — invalid.
    # a[gen0] - a[gen1] = 1 has sum 1+(-1) = 0 — valid, and breaks translation.
    if gen_partner_col is None:
        raise RuntimeError(
            "_dlp_build_affine_system: gen_partner_col is None — "
            "cannot build balanced anchor to break translation symmetry. "
            "Ensure both BASE_DIVISOR roots are present in the leaf set."
        )

    anchor_row = vector(Fp, n_cols)
    anchor_row[gen_col] = Fp(1)
    anchor_row[gen_partner_col] = Fp(-1)
    rows.append(anchor_row)
    rhs.append(Fp(1))

    A = matrix(Fp, rows)
    b = vector(Fp, rhs)

    if verbose:
        _log(f"[dlp_list] anchor row   : a[{generator_x}] - a[{generator_x_partner}] = 1")
        _log(f"[dlp_list] attempting solve_right on {A.nrows()}x{A.ncols()} system over GF({char}) ...")

    return M_fp, A, b

if __name__ == "__main__":
    main()
