from __future__ import annotations
import math, statistics
from collections import Counter
from typing import Any, List, Optional, Sequence, Tuple
from search_common import *

"""mixing_diagnostics.py

Birthday-law / Pollard-rho style mixing diagnostics for the genus-2 Markov walk.

The spectral gap is the wrong early diagnostic: the graph is a forest until
~sqrt(p) steps and eigenvalues are meaningless until then.  What IS meaningful
early is collision statistics directly compared to the birthday bound.

These functions can be called:
  - at end of run:   print_mixing_report(walker)
  - incrementally:   tracker = CollisionTracker(p); tracker.update(walker); tracker.report()

All diagnostics are plain Python / numpy — no Sage required.
"""
# DEAR AI READING THIS:  RAISE ALL EXCEPTIONS

# ---------------------------------------------------------------------------
# Theoretical reference values
# ---------------------------------------------------------------------------

def birthday_expected_collision_step(p: int, leaves_per_step: float) -> float:
    """Expected step of first collision under uniform random walk.

    Under birthday paradox: E[first collision] ~ sqrt(pi*p/2) / leaves_per_step.
    This is the Pollard-rho expected cycle length on F_p.
    """
    return math.sqrt(math.pi * p / 2.0) / leaves_per_step

def birthday_expected_volume_at_collision(p: int) -> float:
    """Expected number of unique nodes visited at first collision: sqrt(pi*p/2)."""
    return math.sqrt(math.pi * p / 2.0)

def expected_collisions_at_volume(V: float, p: int) -> float:
    """Birthday estimate: E[collisions] ~ V^2 / (2p) for V << p."""
    return V * V / (2.0 * p)

def cv_density(C: int, V: int) -> float:
    return C / V if V > 0 else 0.0

# ---------------------------------------------------------------------------
# Single-run collision report
# ---------------------------------------------------------------------------

def print_mixing_report(walker, label: str = "") -> None:
    """Print a full birthday-law mixing diagnostic for a completed walker run.

    Reads directly from walker fields:
        walker.p
        walker.leaf_collision_count
        walker.global_leaves_seen
        walker.collision_log       -- list of (step_idx, outer_n, vol, count, xs)
        walker.history
        walker.xi_visit_count
    """

    p = getattr(walker, "p", None) or FINITE_FIELD
    if p is None:
        raise ValueError("walker.p is None — diagnostics require a finite field prime")

    sqrt_p = math.sqrt(p)
    V = len(walker.global_leaves_seen)
    C = walker.leaf_collision_count
    n_steps = len([r for r in walker.history if getattr(r, 'accepted', True)])

    # leaves/step from actual data
    leaves_per_step = V / n_steps if n_steps > 0 else 0.0

    # theoretical expected collision volume
    V_theory = birthday_expected_volume_at_collision(p)
    # ratio: 1.0 = perfect birthday scaling
    V_ratio = V / V_theory if C >= 1 else None

    # first collision info
    first_col = walker.collision_log[0] if walker.collision_log else None

    tag = f" [{label}]" if label else ""
    hdr = "=" * 68
    print(f"\n{hdr}")
    print(f"  MIXING DIAGNOSTICS (birthday-law / Pollard-rho){tag}")
    print(hdr)

    # --- basic counts ---
    print(f"\n  Field prime p        : {p}")
    print(f"  sqrt(p)              : {sqrt_p:.1f}")
    print(f"  Steps accepted       : {n_steps}")
    print(f"  Unique leaves seen V : {V}   ({V/sqrt_p:.4f} × sqrt(p))")
    print(f"  Graph collisions C   : {C}")

    # --- birthday law check ---
    print(f"\n  -- Birthday law check --")
    E_C = expected_collisions_at_volume(V, p)
    print(f"  E[C] at V={V} (birthday): {E_C:.2f}")
    print(f"  Actual C             : {C}")
    if E_C > 0:
        ratio = C / E_C
        status = "✓  consistent" if 0.3 < ratio < 3.0 else "✗  anomalous"
        print(f"  C / E[C]             : {ratio:.3f}   {status}")

    # --- first collision ---
    if first_col is not None:
        fc_step, fc_n, fc_vol, fc_cnt, fc_xs = first_col
        V_at_first = fc_vol
        theory_vol = V_theory
        ratio_first = V_at_first / theory_vol
        print(f"\n  -- First collision --")
        print(f"  Step index           : {fc_step}")
        print(f"  Graph volume at hit  : {V_at_first}   ({V_at_first/sqrt_p:.4f} × sqrt(p))")
        print(f"  Theory (sqrt(pi*p/2)): {theory_vol:.1f}")
        print(f"  Ratio V_first/theory : {ratio_first:.4f}   ", end="")
        if 0.5 < ratio_first < 2.0:
            print("✓  within birthday window")
        else:
            print("✗  outside expected range — possible bias or fragmentation")
        print(f"  Colliding x-coords   : {fc_xs[:5]}")
    else:
        # no collision yet — show how far we are and ETA
        if leaves_per_step > 0:
            steps_to_collision = birthday_expected_collision_step(p, leaves_per_step)
            print(f"\n  -- No collision yet --")
            print(f"  E[steps to first collision]: {steps_to_collision:.0f}  "
                  f"(at {leaves_per_step:.1f} leaves/step)")
            pct = n_steps / steps_to_collision * 100
            print(f"  Progress             : {pct:.1f}% of expected collision distance")

    # --- coverage growth regime ---
    print(f"\n  -- Coverage growth regime --")
    print(f"  V / sqrt(p)          : {V/sqrt_p:.4f}")
    print(f"  Leaves/step          : {leaves_per_step:.2f}")
    if V < 0.5 * sqrt_p:
        regime = "early linear growth — collision not yet expected"
    elif V < 1.5 * sqrt_p:
        regime = "birthday window — first collision expected here"
    elif V < 3.0 * sqrt_p:
        regime = "post-birthday — collisions should be accumulating"
    else:
        regime = "dense — multiple collisions expected per step"
    print(f"  Regime               : {regime}")

    # --- revisit distribution ---
    print(f"\n  -- Revisit distribution (xi_visit_count) --")
    vc = walker.xi_visit_count
    if vc:
        counts = list(vc.values())
        n_visited_once = sum(1 for c in counts if c == 1)
        n_visited_2plus = sum(1 for c in counts if c >= 2)
        n_visited_5plus = sum(1 for c in counts if c >= 5)
        max_visits = max(counts)
        mean_visits = statistics.mean(counts)
        print(f"  Unique xi seen       : {len(vc)}")
        print(f"  Visited once         : {n_visited_once}  ({n_visited_once/len(vc):.1%})")
        print(f"  Revisited (>=2)      : {n_visited_2plus}  ({n_visited_2plus/len(vc):.1%})")
        print(f"  Revisited (>=5)      : {n_visited_5plus}")
        print(f"  Max visits to one xi : {max_visits}")
        print(f"  Mean visits/xi       : {mean_visits:.2f}")
        # mixing signal: revisit spacing
        if n_visited_2plus > 0:
            print(f"\n  Note: {n_visited_2plus} nodes revisited — chain is no longer a DAG.")
            print(f"  Spectral gap computation becomes meaningful around this scale.")

    # --- C/V density and ETAs ---
    print(f"\n  -- C/V density and spectral readiness --")
    cv = cv_density(C, V)
    print(f"  C/V density          : {cv:.5%}")
    thresholds = [
        (0.001, "forest       [spectral gap meaningless]"),
        (0.01,  "sparse       [spectral gap unreliable]"),
        (0.05,  "first-signal [spectral gap: first hint]"),
        (0.10,  "trustworthy  [spectral gap reliable]"),
    ]
    for thresh, desc in thresholds:
        if cv < thresh:
            # V needed = thresh * 2p / V  =>  V_needed = sqrt(2*p*thresh) ... wait:
            # C/V = V/(2p) > thresh  =>  V > 2*p*thresh
            V_needed = 2 * p * thresh
            steps_needed = max(0, (V_needed - V) / leaves_per_step) if leaves_per_step > 0 else float('inf')
            print(f"  Next threshold: C/V>{thresh:.3%} needs V>{V_needed:.0f}  "
                  f"(~{steps_needed:.0f} more steps at current rate)")
            break
    else:
        print(f"  C/V density is in the trustworthy regime — spectral gap is meaningful.")

    print(f"\n{hdr}\n")

# ---------------------------------------------------------------------------
# Multi-run collision statistics (Pollard-rho distribution check)
# ---------------------------------------------------------------------------

class CollisionTracker:
    """Accumulate first-collision volumes across multiple independent runs.

    Usage:
        tracker = CollisionTracker(p=33554467)
        for run in runs:
            walker = ... # fresh walker per run
            run_walker(walker)
            tracker.record(walker)
        tracker.report()
    """

    def __init__(self, p: int):
        self.p = p
        self.sqrt_p = math.sqrt(p)
        self.V_theory = birthday_expected_volume_at_collision(p)
        self._first_collision_volumes: List[float] = []
        self._no_collision_runs: int = 0

    def record(self, walker) -> None:
        """Record the first-collision volume from one completed walker run."""
        if not walker.collision_log:
            self._no_collision_runs += 1
            return
        fc_step, fc_n, fc_vol, fc_cnt, fc_xs = walker.collision_log[0]
        self._first_collision_volumes.append(float(fc_vol))

    def record_volume(self, V_at_first_collision: float) -> None:
        """Directly record a first-collision volume (e.g. from a log file)."""
        self._first_collision_volumes.append(float(V_at_first_collision))

    def report(self) -> None:
        """Print multi-run birthday-law statistics."""
        vs = self._first_collision_volumes
        n_runs = len(vs) + self._no_collision_runs
        n_hit = len(vs)

        sqrt_p = self.sqrt_p
        V_theory = self.V_theory

        hdr = "=" * 68
        print(f"\n{hdr}")
        print(f"  MULTI-RUN COLLISION STATISTICS  (n={n_runs} runs)")
        print(hdr)
        print(f"  p                    : {self.p}")
        print(f"  sqrt(p)              : {sqrt_p:.1f}")
        print(f"  E[V_first] (theory)  : {V_theory:.1f}  (= sqrt(pi*p/2))")
        print(f"  Runs with collision  : {n_hit} / {n_runs}")
        print(f"  Runs without         : {self._no_collision_runs}")

        if n_hit == 0:
            print(f"\n  No collision data yet.")
            print(hdr)
            return

        mean_V = statistics.mean(vs)
        ratios = [v / V_theory for v in vs]
        mean_ratio = statistics.mean(ratios)

        print(f"\n  -- First-collision volume V_first --")
        print(f"  Mean V_first         : {mean_V:.1f}   ({mean_V/sqrt_p:.4f} × sqrt(p))")
        print(f"  Mean V / E[V_theory] : {mean_ratio:.4f}   (1.0 = perfect birthday scaling)")

        if n_hit >= 2:
            stdev_V = statistics.stdev(vs)
            stdev_ratio = statistics.stdev(ratios)
            # Theoretical std of birthday first-collision: std ~ sqrt((4-pi)/2) * sqrt(p/2)
            # i.e. std/mean ~ sqrt((4-pi)/pi) ~ 0.5227
            theory_cv = math.sqrt((4 - math.pi) / math.pi)
            empirical_cv = stdev_V / mean_V if mean_V > 0 else 0.0
            print(f"  Stdev V_first        : {stdev_V:.1f}")
            print(f"  Empirical CV         : {empirical_cv:.4f}   (theory: {theory_cv:.4f})")
            cv_ok = abs(empirical_cv - theory_cv) / theory_cv < 0.3
            print(f"  CV consistent?       : {'✓  yes' if cv_ok else '✗  no — possible bias'}")

        if n_hit >= 2:
            print(f"\n  -- Per-run ratios V_first / E[V_theory] --")
            for i, (v, r) in enumerate(zip(vs, ratios)):
                flag = "✓" if 0.5 < r < 2.0 else "✗"
                print(f"    run {i+1:3d}:  V={v:8.0f}  ratio={r:.4f}  {flag}")

        # overall verdict
        print(f"\n  -- Verdict --")
        if 0.7 < mean_ratio < 1.4 and n_hit >= 1:
            print(f"  ✓  Collision timing consistent with uniform birthday law.")
            print(f"     Walk appears to explore F_p without fragmentation or bias.")
        elif mean_ratio < 0.5:
            print(f"  ✗  Collisions arriving earlier than expected.")
            print(f"     Possible: walk confined to a subgroup or structured subset.")
        else:
            print(f"  ?  Collisions later than expected — may need more data.")

        print(hdr)
        print()

    @property
    def n_runs(self) -> int:
        return len(self._first_collision_volumes) + self._no_collision_runs

# ---------------------------------------------------------------------------
# Incremental per-step summary line (drop into walk loop)
# ---------------------------------------------------------------------------

def mixing_one_liner(walker, step: int) -> str:
    """Return a compact one-line mixing status string for logging.

    Intended for insertion into the per-step walk loop print.
    Example output:
        [mix] V=6012 (1.037×√p)  C=1  E[C]=0.54  ratio=1.85  birthday: ✓
    """

    p = getattr(walker, "p", None) or FINITE_FIELD
    if p is None:
        return "[mix] p=None, skipping"
    sqrt_p = math.sqrt(p)
    V = len(walker.global_leaves_seen)
    C = walker.leaf_collision_count
    E_C = expected_collisions_at_volume(V, p)
    ratio = C / E_C if E_C > 0.1 else float('nan')
    bday_ok = (0.3 < ratio < 3.0) if not math.isnan(ratio) else None
    bday_str = ("✓" if bday_ok else ("✗" if bday_ok is False else "–"))
    return (
        f"[mix] V={V} ({V/sqrt_p:.3f}×√p)  "
        f"C={C}  E[C]={E_C:.2f}  "
        f"C/E[C]={ratio:.2f}  birthday:{bday_str}"
    )



def compute_fertility(norm, raw, vecs):
    precomp = norm.get('precomputed_residues') or (
        raw.get('precomputed_residues') if isinstance(raw, dict) else None
    )

    if not (precomp and vecs):
        return {
            'n_with_roots': None,
            'n_total': len(vecs) if vecs else None,
            'total_roots': None,
            'per_n_roots': None,
        }

    fertile = set()
    per_n = {}

    for entry in precomp.values():
        if isinstance(entry, dict):
            fertile.update(entry.keys())
            for vtup, sols in entry.items():
                k = str(vtup)
                cnt = len(sols) if isinstance(sols, (list, tuple, set)) else (1 if sols else 0)
                if cnt > 0:
                    per_n[k] = max(per_n.get(k, 0), cnt)

    if not fertile:
        for k, v in precomp.items():
            if not isinstance(v, dict):
                fertile.add(k)
                per_n[str(k)] = 1

    return {
        'n_with_roots': len(per_n),
        'n_total': len(vecs),
        'total_roots': sum(per_n.values()),
        'per_n_roots': per_n,
    }

