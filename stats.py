# search_stats.py
import time
import json
import math
from collections import defaultdict, Counter
from functools import reduce
from operator import mul
# stats_utils.py (or inside stats.py)
from sage.all import QQ, crt
from functools import reduce
import math


from bounds import build_split_poly_from_cd, compute_residue_counts_for_primes, estimate_galois_signature_modp
from sage.all import *

class SearchStats:
    def __init__(self):
        self.start_time = time.time()
        # Phase timers
        self.phase_times = defaultdict(float)
        self._phase_start = {}
        # Counters
        self.counters = Counter()
        # Mapping prime -> set(residues tested mod p)
        self.residues_by_prime = defaultdict(set)
        self.prime_subsets = []

        # Initialize all counters
        self.counters.update({
            'modular_checks': 0,
            'crt_lift_attempts': 0,
            'rational_recon_attempts_worker': 0,
            'rational_recon_success_worker': 0,
            'rational_recon_failure_worker': 0,
            'rationality_tests_total': 0,
            'rationality_tests_success': 0,
            'rationality_tests_failure': 0,
            'multiply_ops': 0,
            'symbolic_solves_attempted': 0,
            'symbolic_solves_success': 0,
            'subsets_generated_initial': 0,
            'subsets_filtered_out_combo': 0,
            'subsets_processed': 0,
            'crt_candidates_found': 0,
            'rational_points_unique': 0,
            'new_sections_unique': 0,
        })

        # Discard reasons and examples
        self.discard_reasons = Counter()
        self.discard_examples = defaultdict(list)
        # Sample successes/failures
        self.successes = []
        self.failures = []
        # CRT classes tested
        self.crt_classes_tested = set()

    # ---------------- Merging ----------------
    def merge(self, other):
        """Merge another SearchStats object into this one."""
        if not isinstance(other, SearchStats):
            return
        for phase, t in other.phase_times.items():
            self.phase_times[phase] += t
        self.counters.update(other.counters)
        for p, res_set in other.residues_by_prime.items():
            self.residues_by_prime[p].update(res_set)
        self.discard_reasons.update(other.discard_reasons)
        for reason, examples in other.discard_examples.items():
            current_len = len(self.discard_examples[reason])
            needed = 5 - current_len
            if needed > 0:
                self.discard_examples[reason].extend(examples[:needed])
        self.successes.extend(other.successes)
        self.failures.extend(other.failures)
        self.successes = self.successes[-1000:]
        self.failures = self.failures[-1000:]
        self.crt_classes_tested.update(other.crt_classes_tested)
        self.prime_subsets.extend(other.prime_subsets)

    def merge_dict(self, stats_dict):
        """Merge a simple Counter dict into counters."""
        self.counters.update(stats_dict)

    # ---------------- CRT ----------------
    def record_crt_class(self, m_mod_M, M):
        canonical = (int(m_mod_M) % int(M), int(M))
        self.crt_classes_tested.add(canonical)

    # ---------------- Timing ----------------
    def start_phase(self, name):
        self._phase_start[name] = time.time()

    def end_phase(self, name):
        if name in self._phase_start:
            dt = time.time() - self._phase_start.pop(name)
            self.phase_times[name] += dt

    # ---------------- Counters ----------------
    def incr(self, key, n=1):
        self.counters[key] += n

    def add_residue(self, prime, residue):
        self.residues_by_prime[prime].add(int(residue) % int(prime))

    def record_discard(self, reason, example=None):
        self.discard_reasons[reason] += 1
        if example is not None and len(self.discard_examples[reason]) < 5:
            self.discard_examples[reason].append(example)

    def record_success(self, m_value, point=None):
        self.counters['rationality_tests_success'] += 1
        self.successes.append({'m': m_value, 'pt': point})

    def record_failure(self, m_value, reason=None):
        self.counters['rationality_tests_failure'] += 1
        self.failures.append({'m': m_value, 'reason': reason})
        if reason:
            self.record_discard(reason, example=m_value)

    # ---------------- Coverage ----------------
    def prime_coverage_fraction(self):
        """Estimate CRT-class coverage using prime residues."""
        fracs = [len(S)/float(p) for p, S in self.residues_by_prime.items() if p > 0]
        if not fracs:
            return 0.0, {}
        log_prod = sum(math.log(f) for f in fracs if f > 0)
        prod = math.exp(log_prod)
        per_prime = {int(p): len(S)/float(p) for p, S in self.residues_by_prime.items()}
        return prod, per_prime

    def crt_space_ratio(self, prime_list):
        M_log10 = sum(math.log10(p) for p in prime_list)
        coverage_prod, _ = self.prime_coverage_fraction()
        return coverage_prod, M_log10

    def crt_coverage_exact(self, prime_subsets_used):
        total_classes_possible = sum(reduce(mul, subset, 1) for subset in prime_subsets_used)
        classes_tested = len(self.crt_classes_tested)
        return classes_tested / total_classes_possible if total_classes_possible > 0 else 0

    def expected_runs_for_coverage(self, prime_subsets_used, target_coverage=0.99):
        coverage_per_run = self.crt_coverage_exact(prime_subsets_used)
        if coverage_per_run >= target_coverage or coverage_per_run == 0:
            return 1
        p = coverage_per_run
        expected_runs = math.log(1 - target_coverage) / math.log(1 - p)
        return math.ceil(expected_runs)

    # ---------------- Diagnostics ----------------
    def compare_target_m_residues(self, m_value, prime_pool):
        """
        For a target rational m = a/b, compute its residue modulo each prime
        in prime_pool and compare against residues actually seen during search.

        Returns a dict with:
            - 'matched_primes': primes where target residue already tested
            - 'unseen_primes': primes not yet tested for that residue
            - 'denom_zero_primes': primes dividing denominator(b)
            - 'coverage_fraction': |matched_primes| / (# usable primes)
        """
        from sage.all import QQ
        a = int(QQ(m_value).numerator())
        b = int(QQ(m_value).denominator())

        matched_primes = []
        unseen_primes = []
        denom_zero_primes = []

        for p in prime_pool:
            p = int(p)
            if b % p == 0:
                denom_zero_primes.append(p)
                continue
            residue = (a * pow(b, -1, p)) % p
            if residue in self.residues_by_prime.get(p, set()):
                matched_primes.append(p)
            else:
                unseen_primes.append(p)

        usable = len(prime_pool) - len(denom_zero_primes)
        coverage = len(matched_primes) / usable if usable > 0 else 0.0

        return {
            'm': (a, b),
            'matched_primes': matched_primes,
            'unseen_primes': unseen_primes,
            'denom_zero_primes': denom_zero_primes,
            'coverage_fraction': coverage
        }

    # ---------------- Summary ----------------
    def summary(self):
        prod_frac, per_prime = self.prime_coverage_fraction()
        return {
            'elapsed': time.time() - self.start_time,
            'phase_times': dict(self.phase_times),
            'counters': dict(self.counters),
            'discard_reasons': dict(self.discard_reasons),
            'discard_examples': dict(self.discard_examples),
            'success_count': self.counters['rationality_tests_success'],
            'failure_count': self.counters['rationality_tests_failure'],
            'prime_coverage_product_heuristic': prod_frac,
            'prime_coverage_per_prime': per_prime
        }

    def summary_string(self):
        s = self.summary()
        lines = [f"Total time: {s['elapsed']:.2f}s",
                 f"Total Rational Points Found (Unique x): {s['counters'].get('rational_points_unique', 0)}",
                 "\nPhases (s):"]
        if not s['phase_times']:
            lines.append("  (No phases recorded)")
        else:
            for phase, t in sorted(s['phase_times'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {phase:<25}: {t:.2f}s")
        lines.append("\nCounters:")
        if not s['counters']:
            lines.append("  (No counters recorded)")
        else:
            for counter, n in sorted(s['counters'].items()):
                lines.append(f"  {counter:<30}: {n}")
        lines.append(f"\nSuccesses: {s['success_count']}, Failures: {s['failure_count']}")
        lines.append("Discard Reasons (Top 5):")
        top_discards = sorted(s['discard_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]
        if not top_discards:
            lines.append("  (None)")
        else:
            for reason, count in top_discards:
                lines.append(f"  {reason:<30}: {count}")
        lines.append("-" * 32)
        return "\n".join(lines)

    def to_json(self, path):
        def serializer(o):
            if isinstance(o, set):
                return list(o)
            return int(o)
        with open(path, 'w') as fh:
            json.dump(self.summary(), fh, indent=2, default=serializer)


    # Put these inside class SearchStats

    def subset_match_probability(self, subset):
        """
        For one subset (iterable of primes), estimate probability that
        a uniform random rational m avoids denominator primes and matches
        all residues for that subset.

        Returns (p_subset, details) where p_subset is float probability,
        and details is dict {'per_prime': {p: (L_p, p, L_p/p)} }.
        """
        per_prime = {}
        logp = 0.0
        any_zero = False
        for p in subset:
            p = int(p)
            L = len(self.residues_by_prime.get(p, set()))
            per_prime[p] = (L, p, (L / float(p) if p > 0 else 0.0))
            if p == 0:
                any_zero = True
                continue
            if L == 0:
                # If no residues seen mod p, then p_subset = 0
                any_zero = True
                logp = float('-inf')
                break
            logp += log(max(1e-300, L / float(p)))  # safe log
        if any_zero and logp == float('-inf'):
            return 0.0, {'per_prime': per_prime}
        p_subset = exp(logp)
        return p_subset, {'per_prime': per_prime}

    def estimate_overall_visibility(self, prime_subsets):
        """
        Given a list of prime subsets (each subset is a list of primes),
        estimate the probability that a random rational is detected by at least
        one subset. Uses the inclusion-as-independent-subsets approximation:
        P_visible = 1 - product_S (1 - p_S)
        Also returns per-subset probabilities for inspection.
        """
        p_list = []
        subset_details = []
        # compute each subset probability
        for subset in prime_subsets:
            p_s, detail = self.subset_match_probability(subset)
            p_list.append(p_s)
            subset_details.append((list(subset), p_s, detail))

        # compute combined probability robustly using logs when necessary
        # if p_s are small, product(1 - p_s) ≈ exp(sum log(1 - p_s))
        prod_log = 0.0
        for p in p_list:
            # clamp p to [0, 1)
            p = max(0.0, min(0.999999999999, p))
            prod_log += log(1.0 - p)
        P_visible = 1.0 - exp(prod_log)

        return {
            'P_visible': P_visible,
            'per_subset': subset_details,
            'num_subsets': len(p_list),
            'product_density_old': self.prime_coverage_fraction()[0]
        }

    def compare_known_points_visibility(self, known_rationals, prime_subsets, verbose=False):
        """
        For a list of rationals (Fraction, QQ, or (num,den) tuples), compute:
        - how many are CRT-visible using crt_visibility_by_subsets
        - per-point fraction matched (existing visibility_signature)
        Returns a summary dict and per-point list.
        """
        from sage.all import QQ
        analyzer = FindabilityAnalyzer(self, [int(p) for s in prime_subsets for p in s])
        samples = []
        visible_count = 0
        for q in known_rationals:
            try:
                r = QQ(q)
            except Exception:
                # tolerate (num,den) style
                if isinstance(q, tuple) and len(q) == 2:
                    r = QQ(q[0])/QQ(q[1])
                else:
                    continue
            sig = analyzer.visibility_signature(r)
            # sig['crt_visible'] = self.crt_visibility_by_subsets(r, prime_subsets) # This logic is broken
            sig['crt_visible'] = sig['fraction'] > 0.1 # Placeholder: "visible" if matched > 10%
            samples.append(sig)
            if sig['crt_visible']:
                visible_count += 1
            if verbose:
                print(f"{sig['m']} visible:{sig['crt_visible']} frac:{sig['fraction']:.3f} matched:{sig['matched']}/{sig['usable']}")
        return {
            'visible_count': visible_count,
            'total': len(samples),
            'fraction_visible': visible_count / len(samples) if samples else 0.0,
            'samples': samples
        }


# ---------------- BenchmarkStats ----------------
class BenchmarkStats:
    def __init__(self, known_ground_truth):
        self.ground_truth = frozenset(known_ground_truth)
        self.start_time = time.time()
        self.discoveries = []
        self.found_so_far = set()
        self.total_crt_candidates = 0
        self.total_vectors_checked = 0
        self.total_prime_subsets_used = 0
        self.fibration_stats = []
        self.current_fib = None

    def start_fibration(self, base_pts, height_bound):
        self.current_fib = {
            'base_pts': tuple(sorted(base_pts)),
            'height_bound': height_bound,
            'start_time': time.time(),
            'vectors': 0,
            'crt_candidates': 0,
            'found_here': set(),
        }

    def record_discovery(self, x_coord):
        if x_coord not in self.found_so_far:
            t = time.time() - self.start_time
            self.discoveries.append((t, x_coord))
            self.found_so_far.add(x_coord)
            if self.current_fib:
                self.current_fib['found_here'].add(x_coord)

    def record_crt_candidate(self):
        self.total_crt_candidates += 1
        if self.current_fib:
            self.current_fib['crt_candidates'] += 1

    def end_fibration(self):
        if self.current_fib:
            self.current_fib['duration'] = time.time() - self.current_fib['start_time']
            self.fibration_stats.append(self.current_fib)
            self.current_fib = None

    def efficiency_report(self):
        total_time = time.time() - self.start_time
        found = len(self.found_so_far)
        expected = len(self.ground_truth)
        discovery_times = [t for t, x in self.discoveries]
        return {
            'total_time': total_time,
            'points_found': found,
            'points_expected': expected,
            'recall': found / expected if expected > 0 else 0,
            'crt_candidates_tested': self.total_crt_candidates,
            'candidates_per_point': self.total_crt_candidates / found if found > 0 else float('inf'),
            'hit_rate': found / self.total_crt_candidates if self.total_crt_candidates > 0 else 0,
            'time_to_first_new_point': discovery_times[0] if discovery_times else None,
            'time_to_all_points': discovery_times[-1] if len(discovery_times) == expected else None,
            'fibrations_needed': len([f for f in self.fibration_stats if f['found_here']]),
            'avg_time_per_fibration': total_time / len(self.fibration_stats) if self.fibration_stats else 0,
        }

    def print_report(self):
        report = self.efficiency_report()
        print("\n" + "="*70)
        print("BENCHMARK REPORT")
        print("="*70)
        print(f"Time: {report['total_time']:.2f}s")
        print(f"Points: {report['points_found']}/{report['points_expected']} ({report['recall']:.0%} recall)")
        print(f"Efficiency: {report['candidates_per_point']:.1f} CRT candidates per point found")
        print(f"Hit rate: {report['hit_rate']:.1%}")
        if report['time_to_all_points']:
            print(f"Time to find all points: {report['time_to_all_points']:.2f}s")
        print(f"\nFibrations used: {report['fibrations_needed']} / {len(self.fibration_stats)} tried")
        print(f"Avg time per fibration: {report['avg_time_per_fibration']:.2f}s")
        print("\nDiscovery timeline:")
        for t, x in self.discoveries:
            print(f"  {t:6.2f}s: x = {x}")
        print("\nPer-fibration breakdown:")
        for i, fib in enumerate(self.fibration_stats):
            if fib['found_here']:
                print(f"  Fib {i} ({fib['base_pts']}): found {fib['found_here']} in {fib['duration']:.2f}s ({fib['crt_candidates']} candidates)")


# ---------------- QuickBench ----------------
class QuickBench:
    def __init__(self):
        self.runs = []

    def record(self, curve_id, time, candidates, points):
        self.runs.append({
            'curve': curve_id,
            'time': time,
            'candidates': candidates,
            'points': points,
            'hit_rate': points / candidates if candidates > 0 else 0,
        })

    def summary(self):
        if not self.runs:
            print("No runs recorded")
            return
        avg_time = sum(r['time'] for r in self.runs) / len(self.runs)
        avg_hit_rate = sum(r['hit_rate'] for r in self.runs) / len(self.runs)
        print(f"Avg time: {avg_time:.1f}s")
        print(f"Avg hit rate: {100*avg_hit_rate:.1f}%")
        print(f"Curves tested: {len(self.runs)}")


# write_run_summary.py
# Pure stdlib. Call write_run_summary(summary_dict, "summaries") at end of run.

import json
import os
from datetime import datetime
from fractions import Fraction

def _rational_to_pair(q):
    # Accept int, Fraction, or (num,den) tuple
    if isinstance(q, tuple):
        assert len(q) == 2
        return (int(q[0]), int(q[1]))
    if isinstance(q, Fraction):
        return (int(q.numerator), int(q.denominator))
    if isinstance(q, int):
        return (q, 1)
    raise AssertionError("unexpected rational type: " + str(type(q)))

def normalize_summary(run):
    # Expect run to be a dict; coerce some common types
    out = dict(run)  # shallow copy
    out['run_id'] = out.get('run_id') or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out['curve_id'] = str(out.get('curve_id', 'unknown'))
    out['wall_seconds'] = float(out.get('wall_seconds', 0.0))
    out['total_crt_candidates'] = int(out.get('total_crt_candidates', 0))
    out['total_lift_attempts'] = int(out.get('total_lift_attempts', 0))
    out['total_rationality_tests_success'] = int(out.get('total_rationality_tests_success', 0))
    out['total_rationality_tests_failure'] = int(out.get('total_rationality_tests_failure', 0))
    # unique_x_list: coerce each x into (num,den)
    ux = out.get('unique_x_list', [])
    out['unique_x_list'] = [_rational_to_pair(q) for q in ux]
    # per_point_counts: convert keys to str "num/den"
    pcounts = out.get('per_point_counts', {})
    out['per_point_counts'] = {str(k): int(v) for k, v in pcounts.items()}
    # residues_seen: ensure ints
    rs = out.get('residues_seen', {})
    out['residues_seen'] = {str(k): int(v) for k, v in rs.items()}
    # subset_productivity: list of [[p1,p2], count]
    sp = out.get('subset_productivity', [])
    out['subset_productivity'] = [[list(map(int, s)), int(c)] for s, c in sp]
    out['extra_flags'] = out.get('extra_flags', {})
    return out

def write_run_summary(run_dict, outdir="summaries"):
    s = normalize_summary(run_dict)
    os.makedirs(outdir, exist_ok=True)
    fname = "{curve}-{run}.json".format(curve=s['curve_id'], run=s['run_id'])
    tmp = os.path.join(outdir, fname + ".tmp")
    final = os.path.join(outdir, fname)
    with open(tmp, "w") as f:
        json.dump(s, f, sort_keys=True, indent=2)
    os.replace(tmp, final)
    print("wrote summary:", final)


# analyze_summaries.py
# Usage: python analyze_summaries.py summaries/
import json
import os
import math
from collections import defaultdict, Counter
from statistics import mean, pstdev

def chao1_estimator(counts):
    # counts: list of frequencies of coupon types
    # Chao1 = S_obs + (n1^2)/(2*n2). If n2==0 use bias-corrected form
    S_obs = len(counts)
    f1 = sum(1 for c in counts if c == 1)
    f2 = sum(1 for c in counts if c == 2)
    if f2 > 0:
        return S_obs + (f1 * f1) / (2.0 * f2)
    if f1 > 0:
        # conservative upper bound
        return S_obs + (f1 * (f1 - 1)) / 2.0
    return S_obs

def entropy_from_counts(count_map):
    total = sum(count_map.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in count_map.values():
        p = v / total
        if p > 0:
            ent -= p * math.log(p)
    return ent

def analyze_dir(d):
    summaries = []
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn), "r") as f:
            summaries.append(json.load(f))
    summaries.sort(key=lambda s: s.get('run_id'))
    # global per-point frequencies across all runs
    global_point_counts = Counter()
    per_curve_runs = defaultdict(list)
    for s in summaries:
        curve = s['curve_id']
        per_curve_runs[curve].append(s)
        for x in s.get('unique_x_list', []):
            global_point_counts[tuple(x)] += 1
    print("Loaded", len(summaries), "summaries for", len(per_curve_runs), "curves")
    # analyze per-curve
    for curve, runs in per_curve_runs.items():
        print("\n=== Curve:", curve, "runs:", len(runs))
        # accumulation curve per run (unique x after each subset processed)
        # if run contains per_subset cumulative data, use it; otherwise estimate from subset_productivity
        for s in runs:
            ux = s.get('unique_x_list', [])
            n_unique = len(ux)
            print(" run", s['run_id'], "time(s):", s.get('wall_seconds'), "unique_x:", n_unique,
                  "crt_candidates:", s.get('total_crt_candidates', 0),
                  "rational_success:", s.get('total_rationality_tests_success', 0))
        # combine across runs to get per-point frequency
        combined = Counter()
        for s in runs:
            for x in s.get('unique_x_list', []):
                combined[tuple(x)] += 1
        freqs = list(combined.values())
        if not freqs:
            print(" no points found across runs")
            continue
        heterogeneity = (pstdev(freqs)**2) / mean(freqs) if len(freqs) > 1 and mean(freqs) != 0 else 0.0
        print(" points found:", len(freqs), "mean frequency:", mean(freqs), "heterogeneity index:", heterogeneity)
        # per-prime entropy and noise index
        agg_residue_counts = defaultdict(int)
        agg_residue_observations = defaultdict(int)
        prime_reject_counts = Counter()
        for s in runs:
            for p, r in s.get('residues_seen', {}).items():
                agg_residue_counts[int(p)] += int(r)
                agg_residue_observations[int(p)] += 1
            for reason, c in s.get('discard_reasons_count', {}).items():
                prime_reject_counts[reason] += int(c)
        prime_entropy = {p: entropy_from_counts({'seen': agg_residue_counts[p], 'notseen': max(0, sum(agg_residue_counts.values())-agg_residue_counts[p])})
                         for p in agg_residue_counts}
        top_primes = sorted(prime_entropy.items(), key=lambda t: -t[1])[:8]
        print(" top primes by entropy:", top_primes)
        print(" discard reason totals (top 10):", prime_reject_counts.most_common(10))
        # subset productivity -> Chao1 on residue-vectors (approx)
        # if subset_productivity present, use candidate counts per subset as proxy for coupon frequencies
        coupon_freqs = []
        for s in runs:
            for subset, cnt in s.get('subset_productivity', []):
                # treat each subset as a coupon type; cnt = number of CRT candidates for that subset
                coupon_freqs.append(int(cnt))
        if coupon_freqs:
            # transform to frequency-of-frequencies: how many coupon types had count 1,2,...
            f_counts = Counter(coupon_freqs)
            # For chao1 we need counts of coupon types observed how many times; treat coupon_freqs as "observations"
            ch1 = chao1_estimator(coupon_freqs)
            print(" subset coupon-types observed:", len(coupon_freqs), "Chao1 estimate:", ch1)
        # simple hard-curve heuristic:
        # compute slope of unique_x accumulation for the first run if we have per_subset data; as fallback use mean frequency
        first = runs[0]
        acc = None
        if first.get('per_subset_accumulation'):
            acc = first['per_subset_accumulation']
        else:
            # fallback: estimate slope as (unique_count)/(subsets_processed)
            sp = first.get('subsets_processed', None)
            if sp and sp > 0:
                acc = [0, first.get('total_rationality_tests_success', 0)]
        if acc:
            # slope ~ increase per subset; use last-third mean slope
            if len(acc) >= 3:
                n = len(acc)
                slopes = [(acc[i+1]-acc[i]) for i in range(max(0, n//2), n-1)]
                avg_slope = mean(slopes) if slopes else 0.0
            else:
                avg_slope = acc[-1] / max(1, first.get('subsets_processed', 1))
        else:
            avg_slope = mean(freqs)
        hard_flag = avg_slope < 0.02 or heterogeneity > 5.0
        print(" avg_slope:", avg_slope, "hard_flag:", hard_flag)
        # collisions estimate: total candidates / subsets_processed -> high ratio indicates collisions/ambiguity
        tot_cand = sum(s.get('total_crt_candidates', 0) for s in runs)
        tot_subs = sum(s.get('subsets_processed', 1) for s in runs)
        cand_per_subset = tot_cand / max(1, tot_subs)
        print(" cand_per_subset:", cand_per_subset)
        # print a short recommendation line
        if hard_flag or cand_per_subset > 1000:
            print(" RECOMMEND: mark curve as HARD -> increase NUM_SUBSETS and HEIGHT_BOUND, prefer higher-entropy primes, add targeted primes to heavy subsets")
        else:
            print(" RECOMMEND: standard budget OK")
    # global point frequencies summary
    if global_point_counts:
        most_common = global_point_counts.most_common(10)
        print("\nGlobal top found x's:", most_common)
    return


class CurveComplexityPredictor:
    """Predict if a curve will be hard before spending compute"""
    
    def __init__(self):
        self.complexity_signals = {}
    
    def assess_curve_difficulty(self, cd, initial_sections, prime_pool, H):
        """Run cheap diagnostics before heavy search"""
        
        # Signal 1: Discriminant polynomial complexity
        split_poly = build_split_poly_from_cd(cd)
        degree = split_poly.degree()
        
        # Signal 2: Residue density across primes
        residue_counts = compute_residue_counts_for_primes(
            cd, [cd.phi_x], prime_pool[:20]  # Just first 20 primes
        )
        avg_density = sum(r/p for p, r in residue_counts.items()) / len(residue_counts)
        density_variance = variance([r/p for p, r in residue_counts.items()])
        
        # Signal 3: How many primes are "zero-ratio" (no roots)?
        zero_ratio = sum(1 for r in residue_counts.values() if r == 0) / len(residue_counts)
        
        # Signal 4: Canonical height pairing matrix condition number
        #H = compute_height_pairing_matrix(cd, initial_sections)
        try:
            cond = np.linalg.cond(np.array(H.change_ring(RDF)))
        except:
            cond = float('inf')
        
        # Signal 5: Galois complexity
        galois_info = estimate_galois_signature_modp(split_poly, prime_pool[:15])
        splitting_degree = galois_info.get('splitting_field_degree_est', 1)
        
        # Combine into difficulty score
        difficulty_score = (
            0.2 * min(degree / 12, 3.0) +  # Discriminant degree (normalized)
            0.3 * (1.0 - avg_density) +     # Low density = hard
            0.2 * zero_ratio +              # Many zero primes = hard  
            0.1 * min(log(cond) / 10, 3.0) + # Ill-conditioned = hard
            0.2 * min(log(splitting_degree) / 10, 3.0)  # High Galois complexity = hard
        )
        
        return {
            'difficulty_score': difficulty_score,  # 0 (easy) to 3+ (very hard)
            'recommended_height_multiplier': 1.0 + difficulty_score,
            'recommended_subset_multiplier': 1.0 + 0.5 * difficulty_score,
            'signals': {
                'discriminant_degree': degree,
                'avg_residue_density': avg_density,
                'zero_prime_ratio': zero_ratio,
                'height_matrix_condition': cond,
                'galois_complexity': splitting_degree
            }
        }


class CoverageEstimator:
    """Estimate how much of the search space we've covered"""
    
    def __init__(self, prime_pool, residue_counts):
        self.prime_pool = prime_pool
        self.residue_counts = residue_counts
        self.tested_classes = set()  # (m mod M, M) pairs we've tested
    
    def record_crt_class(self, m0, M):
        """Record that we tested this residue class"""
        canonical = (int(m0) % int(M), int(M))
        self.tested_classes.add(canonical)
    
    def estimate_coverage(self, prime_subsets_used):
        """Estimate what fraction of search space we've covered"""
        
        # Method 1: Direct counting (only feasible for small M)
        total_classes_possible = sum(
            prod([int(p) for p in subset]) if subset else 1
            for subset in prime_subsets_used
        )

        classes_tested = len(self.tested_classes)
        
        if total_classes_possible < 10**9:  # Feasible to count
            direct_coverage = classes_tested / total_classes_possible
        else:
            direct_coverage = None
        
        # Method 2: Heuristic via residue density
        # If residue_counts[p] = r_p, then "density" mod p is r_p / p
        # Coverage ≈ product of densities (assumes independence)
        density_product = 1.0
        for p in self.prime_pool:
            r_p = self.residue_counts.get(p, 1)
            density_product *= (r_p / float(p))
        
        # Method 3: Birthday paradox estimate
        # After n random samples from space of size N,
        # expected coverage ≈ 1 - exp(-n/N)
        # Solve for coverage given n = classes_tested
        if total_classes_possible < 10**15:
            birthday_coverage = 1 - math.exp(-classes_tested / total_classes_possible)
        else:
            birthday_coverage = None
        
        return {
            'direct_coverage': direct_coverage,
            'heuristic_coverage': density_product,
            'birthday_coverage': birthday_coverage,
            'classes_tested': classes_tested,
            'space_size_estimate': total_classes_possible
        }
    
    def recommend_additional_runs(self, prime_subsets_used, target_coverage=0.95):
        """How many more runs to reach target coverage?"""
        current = self.estimate_coverage(prime_subsets_used)
        
        if current['direct_coverage'] is not None:
            p = current['direct_coverage']
        elif current['birthday_coverage'] is not None:
            p = current['birthday_coverage']
        else:
            p = current['heuristic_coverage']
        
        if p >= target_coverage:
            return 0
        
        # Coupon collector problem: expected runs to reach coverage c
        # is -log(1-c) / p_per_run
        if not len(self.tested_classes):
            coverage_per_run = 0
            return -1
        else:
            coverage_per_run = p / len(self.tested_classes)  # rough estimate
            try:
                expected_runs = math.log(1 - target_coverage) / math.log(1 - coverage_per_run)
                expected_runs = math.ceil(expected_runs)
            except ZeroDivisionError:
                expected_runs = oo # sage for infinity
        
            return expected_runs

# In stats.py
def analyze_sample_m_list(m_list, analyzer, prime_subsets):
    results = []
    product_density = None  # Will be set by the first sample

    for m in m_list:
        sig = analyzer.visibility_signature(m)
        if product_density is None: # On first iteration, grab the m-independent density
            product_density = sig['coverage']
        # sig['crt_visible'] = analyzer.crt_visibility_by_subsets(m, prime_subsets) # Broken logic
        sig['crt_visible'] = sig['fraction'] > 0.1 # Placeholder
        results.append(sig)

    if product_density is None:
        product_density = 0.0 # Handle case where m_list is empty

    try:
        from search_common import MIN_PRIME_SUBSET_SIZE
    except Exception:
        MIN_PRIME_SUBSET_SIZE = 3

    meet_count = sum(1 for sig in results if sig['matched'] >= MIN_PRIME_SUBSET_SIZE)
    frac_meet = meet_count / len(results) if results else 0.0

    return {
        'product_density_heuristic': product_density,
        'fraction_meet_min_subset': frac_meet,
        'samples': results
    }


class FindabilityAnalyzer:
    """
    Analyzes the "findability" of a rational m-value based on the
    set of residues seen during the search.
    """
    def __init__(self, stats, prime_pool):
        self.stats = stats
        self.prime_pool = list(prime_pool)

    def visibility_signature(self, m_val):
        """
        Compute per-prime match, fraction matched, and global *average* density.
        
        Returns:
            'm': (a, b) tuple for the rational
            'per_prime': {p: (residue, ok)} dict
            'matched': int, number of primes where residue was seen
            'usable': int, number of primes where denom != 0
            'coverage': float, global *average* density heuristic (avg(L_p/p))
            'fraction': float, 'matched' / 'usable' (local findability)
        """
        try:
            a = QQ(m_val).numerator()
            b = QQ(m_val).denominator()
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Cannot coerce m_val={m_val} to QQ: {e}")

        a = int(a)
        b = int(b)

        matched = 0
        usable = 0
        per_prime = {}

        for p in self.prime_pool:
            if b % p == 0:
                per_prime[p] = ('DENOM_ZERO', False)
                continue
            usable += 1

            try:
                residue = (a * pow(b, -1, p)) % p
            except ValueError as e:
                # This shouldn't happen, but good to catch
                per_prime[p] = ('INV_FAIL', False)
                continue

            seen = self.stats.residues_by_prime.get(p, set())
            ok = residue in seen
            per_prime[p] = (residue, ok)
            if ok:
                matched += 1

        # --- MODIFIED: Heuristic *average* density ---
        # This is a global metric, independent of m_val.
        if not self.prime_pool:
            density = 0.0
        else:
            densities = []
            for p in self.prime_pool:
                # Only consider primes where we have residue data
                if p not in self.stats.residues_by_prime:
                    continue
                L = len(self.stats.residues_by_prime[p])
                if p == 0:
                    continue
                densities.append(L / float(p))

            density = sum(densities) / len(densities) if densities else 0.0
        # --- END MODIFICATION ---

        frac = matched / usable if usable > 0 else 0.0

        return {
            'm': (a, b),
            'per_prime': per_prime,
            'matched': matched,
            'usable': usable,
            'coverage': density, # This is now avg(L_p/p)
            'fraction': frac,   # This is the local matched/usable
        }

    def analyze_batch(self, m_list):
        """Analyze a batch of rationals, return list of signatures."""
        results = []
        avg_density = None # Use a more descriptive name

        for m in m_list:
            sig = self.visibility_signature(m)
            if avg_density is None:
                avg_density = sig['coverage'] # Now captures avg(L_p/p)
            results.append(sig)
        
        if avg_density is None:
            avg_density = 0.0

        return {
            'global_average_density': avg_density, # Renamed key
            'samples': results,
            # 'fraction_visible' is no longer well-defined without the broken CRT check
        }


# Add to stats.py - Height-based completeness metric

class CompletenessAnalyzer:
    """
    Estimate what fraction of rational m-values (and thus points) up to 
    canonical height H were "findable" by the search.
    
    Uses two main metrics:
    1. Global Average Density: avg(L_p/p) over all primes (baseline difficulty).
    2. Local Coverage: Average findability fraction (matched/usable) of points we *actually found*.
    """
    
    def __init__(self, stats, prime_pool, prime_subsets, height_bound, r_m_func, shift):
        """
        Args:
            stats: SearchStats object
            prime_pool: List of primes used in search
            prime_subsets: List of prime subsets actually searched
            height_bound: Canonical height bound used for search vectors
            r_m_func: Function to compute x from m (for inversion)
            shift: The x-coordinate shift applied
        """
        self.stats = stats
        self.prime_pool = sorted(list(set(prime_pool)))
        self.prime_subsets = prime_subsets
        self.height_bound = float(height_bound)
        self.r_m_func = r_m_func
        self.shift = shift
        self.analyzer = FindabilityAnalyzer(stats, self.prime_pool)
    
    def canonical_height_of_x(self, x_val):
        """
        Naive canonical height proxy: h(x) ≈ log(max(|num|, |den|))
        """
        from sage.all import QQ
        q = QQ(x_val)
        num = abs(int(q.numerator()))
        den = abs(int(q.denominator()))
        return float(math.log(max(num, den, 1)))
    
    def height_distribution_of_found(self, found_xs):
        """Compute height distribution of found points"""
        if not found_xs:
            return {}
        
        heights = [self.canonical_height_of_x(x) for x in found_xs]
        
        return {
            'min': min(heights),
            'max': max(heights),
            'mean': sum(heights) / len(heights),
            'median': sorted(heights)[len(heights) // 2],
            'count': len(heights)
        }
    
    def m_value_from_x(self, x_val):
        """
        Invert x = r_m(m) - shift to get m from x.
        For linear case: x = -m - const - shift => m = -x - const - shift
        """
        from sage.all import QQ
        const = self.r_m_func(m=QQ(0))
        return -(QQ(x_val) + const + self.shift)
    
    def compute_m_space_coverage(self, found_xs):
        """
        Computes the average "findability fraction" (matched/usable)
        for all m-values corresponding to the *found* x-coordinates.
        Also finds the minimum and maximum findability fractions.
        
        Returns: (avg_coverage, min_coverage_info, max_coverage_info, samples)
                 *_coverage_info = {'x': x_val, 'm': m_val, 'fraction': fraction}
        """
        if not found_xs:
            return 0.0, None, None, []
        
        coverage_samples = []
        min_info = {'fraction': 1.1} # Start min > 1
        max_info = {'fraction': -0.1} # Start max < 0
        
        for x in found_xs:
            m_val = self.m_value_from_x(x)
            sig = self.analyzer.visibility_signature(m_val)
            
            findability_frac = sig['fraction']
            
            sample_data = {
                'x': x,
                'm': m_val,
                'findability_fraction': findability_frac,
                'matched': sig['matched'],
                'usable': sig['usable']
            }
            coverage_samples.append(sample_data)

            # Update min/max
            if findability_frac < min_info['fraction']:
                min_info = {'x': x, 'm': m_val, 'fraction': findability_frac}
            if findability_frac > max_info['fraction']:
                max_info = {'x': x, 'm': m_val, 'fraction': findability_frac}

        if not coverage_samples:
            return 0.0, None, None, []
            
        avg_coverage = sum(s['findability_fraction'] for s in coverage_samples) / len(coverage_samples)
        
        # Handle cases where min/max weren't updated (e.g., only one point)
        min_result = min_info if min_info['fraction'] <= 1.0 else None
        max_result = max_info if max_info['fraction'] >= 0.0 else None
        
        return avg_coverage, min_result, max_result, coverage_samples
    
    def estimate_total_m_space_coverage(self):
        """
        Computes the global m-space coverage heuristic: the *average* density: avg(L_p/p).
        """
        if not self.prime_pool:
            return 0.0
        
        sig = self.analyzer.visibility_signature(QQ(0))
        global_average_density = sig['coverage']
        
        return global_average_density

    
    def full_report(self, found_xs):
        """
        Generate complete completeness analysis report based on new metrics.
        Includes min/max findability among found points.
        """
        if not found_xs:
            return {
                'status': 'no_points_found',
                'completeness_estimate': 'unknown',
                'recommendation': 'No points found - increase height bound or check curve'
            }
        
        # Global m-space metric (average density)
        global_coverage_heuristic = self.estimate_total_m_space_coverage()
        
        # Local coverage metrics (at found points)
        local_avg_coverage, min_findability, max_findability, samples = self.compute_m_space_coverage(found_xs)
        
        # Height distribution
        h_dist = self.height_distribution_of_found(found_xs)
        
        # Reconcile estimates into a single 'completeness_estimate'
        # (This remains subjective, based on comparing global and local)
        final_estimate = (global_coverage_heuristic + local_avg_coverage) / 2 # Simple average for now
        confidence = 'medium'
        bias_warning = False

        if abs(local_avg_coverage - global_coverage_heuristic) > 0.2: # If they differ significantly
            confidence = 'low'
            bias_warning = True
            # Maybe lean towards the lower value if they differ?
            final_estimate = min(global_coverage_heuristic, local_avg_coverage)
        
        if final_estimate > 0.85:
            confidence = 'high'
        elif final_estimate < 0.3:
             confidence = 'low'

        # Generate recommendation (remains the same logic based on final_estimate)
        if final_estimate > 0.95:
            rec = 'Search appears complete ✓'
        elif final_estimate > 0.75:
            rec = 'Likely found most points. (High findability)'
        elif final_estimate > 0.5:
            rec = 'Moderate coverage. Increase NUM_SUBSETS by 2-3x.'
        elif final_estimate > 0.25:
            rec = 'Low coverage. Double both HEIGHT_BOUND and NUM_SUBSETS.'
        else:
            rec = 'Very low coverage. Search parameters may be inadequate. Check CurveComplexityPredictor output.'
        
        # Additional warnings
        warnings = []
        if bias_warning:
            warnings.append(f'⚠️  Coverage may be uneven: Found points avg findability ({local_avg_coverage:.1%}) vs global heuristic ({global_coverage_heuristic:.1%}) differ significantly.')
        
        if h_dist.get('max', 0) > self.height_bound * 0.9:
            warnings.append('⚠️  Found points near height bound - may be more points above bound')
        
        if len(found_xs) < 3:
            warnings.append('⚠️  Very few points found - completeness estimate unreliable')
        
        return {
            'completeness_estimate': final_estimate,
            'confidence': confidence,
            'coverage_breakdown': {
                'global_m_space_heuristic': global_coverage_heuristic,
                'local_coverage_at_found_points': {
                     'average': local_avg_coverage,
                     'min': min_findability,
                     'max': max_findability,
                },
                'agreement': abs(local_avg_coverage - global_coverage_heuristic)
            },
            'found_points': len(found_xs),
            'height_distribution': h_dist,
            'recommendation': rec,
            'warnings': warnings,
            'details': {
                'coverage_samples': samples[:5], # Keep sample for debugging
            }
        }
    
    def print_report(self, found_xs):
        """Pretty-print completeness report, including min/max findability."""
        report = self.full_report(found_xs)
        
        print("\n" + "="*70)
        print("COMPLETENESS ANALYSIS")
        print("="*70)
        
        if report.get('status') == 'no_points_found':
            print(report['recommendation'])
            print("="*70)
            return
        
        est = float(report['completeness_estimate'])
        conf = report['confidence']
        
        print(f"\nEstimated Completeness Score: {100*est:.1f}% (confidence: {conf})")
        
        breakdown = report['coverage_breakdown']
        global_heuristic = float(breakdown['global_m_space_heuristic'])
        local_cov = breakdown['local_coverage_at_found_points']
        local_avg = float(local_cov['average'])
        
        print(f"\nCoverage Details:")
        print(f"  Global m-space heuristic (avg density): {100*global_heuristic:.1f}%")
        print(f"  Findability of Found Points:")
        print(f"    Average: {100*local_avg:.1f}%")
        if local_cov['min']:
            min_f = local_cov['min']
            print(f"    Min:     {100*float(min_f['fraction']):.1f}% (at x={min_f['x']})")
        if local_cov['max']:
            max_f = local_cov['max']
            print(f"    Max:     {100*float(max_f['fraction']):.1f}% (at x={max_f['x']})")

        if breakdown['agreement'] > 0.2:
            print(f"  -> Note: Avg findability differs significantly from global heuristic.")
        
        print(f"\nPoints Found: {report['found_points']}")
        
        print(f"\nHeight Distribution:")
        h = report['height_distribution']
        print(f"  Range: [{float(h.get('min',0)):.2f}, {float(h.get('max',0)):.2f}]")
        print(f"  Mean: {float(h.get('mean',0)):.2f}, Median: {float(h.get('median',0)):.2f}")
        print(f"  (Search height bound: {float(self.height_bound):.2f})")
        
        if report.get('warnings'):
            print(f"\nWarnings:")
            for warn in report['warnings']:
                print(f"  {warn}")
        
        print(f"\nRecommendation:")
        print(f"  {report['recommendation']}")
        
        print("="*70)


def print_unified_completeness_report(stats, prime_pool, prime_subsets, 
                                     height_bound, found_xs, r_m, shift):
    """
    Single unified completeness report to replace the scattered diagnostics.
    
    Call this ONCE at the end of search, after all the technical stats.
    
    Args:
        stats: SearchStats object with crt_classes_tested
        prime_pool: List of primes used
        prime_subsets: List of prime subsets searched
        height_bound: Canonical height bound for search vectors
        found_xs: List/set of found x-coordinates
        r_m: Function to compute x from m (for inversion)
        shift: The x-coordinate shift applied
    """
    try:
        analyzer = CompletenessAnalyzer(stats, prime_pool, prime_subsets, 
                                       height_bound, r_m, shift)
        analyzer.print_report(found_xs)
    except Exception as e:
        print("\n" + "="*70)
        print("COMPLETENESS ANALYSIS FAILED")
        print(f"Error: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full stack trace during debugging
        print("Please check CompletenessAnalyzer implementation.")
        print("="*70)




# --- Add these imports at top of stats.py if not already present ---
import random
import math
from collections import Counter, defaultdict

# --- Bootstrap empirical visibility ---
def bootstrap_visibility(findability_analyzer,
                         N_samples=5000,
                         max_num=10**4,
                         max_den=10**4,
                         seed=None,
                         thresholds=(0.1, 0.5)):
    """
    Sample random rationals m = num/den with num in [-max_num,max_num], den in [1,max_den].
    Uses findability_analyzer.visibility_signature(m) to compute per-prime matches.

    Returns:
      {
        'avg_fraction': float,
        'frac_above_{t}': ... for each t in thresholds,
        'empirical_visible_fraction': fraction of samples visible to at least one prime_subset (if analyzer has prime_subsets),
        'm_samples': list of sampled QQ m's (length N_samples),
        'fractions': list of per-sample fractions
      }
    """
    if seed is not None:
        random.seed(seed)
    fractions = []
    m_samples = []
    visible_count = 0
    per_prime_counts = Counter()
    # If analyzer exposes prime_subsets (list-of-list of primes), use it to check empirical visibility:
    prime_subsets = getattr(findability_analyzer, 'prime_subsets', None)

    for i in range(N_samples):
        num = random.randint(-max_num, max_num)
        den = random.randint(1, max_den)
        m = QQ(num) / QQ(den)
        sig = findability_analyzer.visibility_signature(m)
        frac = float(sig.get('fraction', 0.0))
        fractions.append(frac)
        m_samples.append(m)

        # count per-prime matched occurrences (for empirical densities)
        per_prime = sig.get('per_prime', {})
        for p, (_, ok) in per_prime.items():
            if ok:
                per_prime_counts[p] += 1

        # empirical "visible" if any stored subset is fully matched
        if prime_subsets:
            for S in prime_subsets:
                if all(sig['per_prime'].get(p, (None, False))[1] for p in S):
                    visible_count += 1
                    break

    out = {}
    out['avg_fraction'] = sum(fractions) / len(fractions)
    for t in thresholds:
        out[f'frac_above_{t}'] = sum(1 for f in fractions if f >= t) / len(fractions)
    if prime_subsets:
        out['empirical_visible_fraction'] = visible_count / float(len(fractions))
    else:
        out['empirical_visible_fraction'] = None
    out['m_samples'] = m_samples
    out['fractions'] = fractions
    out['per_prime_counts'] = per_prime_counts
    out['sample_size'] = N_samples
    return out


# --- Pairwise mutual information between matched indicators ---
def pairwise_mutual_info(findability_analyzer,
                         primes,
                         sample_ms=None,
                         N_samples=2000,
                         seed=None,
                         top_k=20,
                         mi_threshold=0.01):
    """
    Compute pairwise mutual information (binary matches) for the given primes.
    If sample_ms is None, it will be generated by sampling rationals (uses default small range).
    Returns a dict with:
      - 'mi_matrix' : dict[(p,q)] = MI (bits)
      - 'top_pairs' : list of (p,q,MI) sorted descending (top_k)
      - 'frac_pairs_above_threshold' : fraction of pairs with MI >= mi_threshold
    """
    if seed is not None:
        random.seed(seed)

    if sample_ms is None:
        # generate sample_ms cheaply
        sample_ms = []
        for _ in range(N_samples):
            num = random.randint(-10000, 10000)
            den = random.randint(1, 10000)
            sample_ms.append(QQ(num)/QQ(den))
    else:
        N_samples = len(sample_ms)

    # Build per-prime matched lists (0/1)
    matched = {p: [] for p in primes}
    for m in sample_ms:
        sig = findability_analyzer.visibility_signature(m)
        perp = sig.get('per_prime', {})
        for p in primes:
            ok = perp.get(p, (None, False))[1]
            matched[p].append(1 if ok else 0)

    # compute pairwise MI on binary variables
    def mi_binary(a_list, b_list):
        # a_list, b_list are lists of 0/1
        N = len(a_list)
        cnt = Counter(zip(a_list, b_list))
        mi = 0.0
        for (a,b), c in cnt.items():
            p_ab = c / N
            p_a = sum(1 for x in a_list if x == a) / N
            p_b = sum(1 for x in b_list if x == b) / N
            # avoid log(0)
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * math.log2(p_ab / (p_a * p_b))
        return mi

    mi_matrix = {}
    primes_list = list(primes)
    total_pairs = 0
    above_count = 0
    for i, p in enumerate(primes_list):
        for q in primes_list[i+1:]:
            M = mi_binary(matched[p], matched[q])
            mi_matrix[(p,q)] = M
            total_pairs += 1
            if M >= mi_threshold:
                above_count += 1

    top_pairs = sorted([(p,q,mi) for (p,q),mi in mi_matrix.items()], key=lambda t: -t[2])[:top_k]
    return {
        'mi_matrix': mi_matrix,
        'top_pairs': top_pairs,
        'frac_pairs_above_threshold': (above_count / total_pairs if total_pairs else 0.0),
        'sample_size': N_samples
    }


# --- Empirical per-subset coverage via sampling (compare to product model) ---
def per_subset_empirical_coverage(findability_analyzer,
                                  subsets,
                                  sample_ms=None,
                                  N_samples=2000):
    """
    For each subset S in subsets (list of lists of primes), estimate:
      - empirical p_S = fraction of sample_ms for which all primes in S match
      - product_model p_S_prod = product of empirical per-prime densities
    If sample_ms is None sample N_samples rationals.
    Returns list of dicts for each subset with ('subset','empirical_p_S','product_p_S','sample_size')
    """
    if sample_ms is None:
        sample_ms = []
        for _ in range(N_samples):
            num = random.randint(-10000, 10000)
            den = random.randint(1, 10000)
            sample_ms.append(QQ(num)/QQ(den))
    else:
        N_samples = len(sample_ms)

    # compute per-prime empirical densities on the sample
    per_prime_ok = defaultdict(int)
    sig_cache = {}
    for m in sample_ms:
        sig = findability_analyzer.visibility_signature(m)
        sig_cache[m] = sig
        for p, (_, ok) in sig.get('per_prime', {}).items():
            if ok:
                per_prime_ok[p] += 1
    per_prime_density = {p: per_prime_ok[p] / float(N_samples) for p in per_prime_ok}

    results = []
    for S in subsets:
        match_count = 0
        for m in sample_ms:
            sig = sig_cache[m]
            if all(sig['per_prime'].get(p, (None, False))[1] for p in S):
                match_count += 1
        emp_p = match_count / float(N_samples)
        # product model using empirical densities; if some p not present in density assume very small density
        prod = 1.0
        for p in S:
            prod *= per_prime_density.get(p, 0.0)
        results.append({'subset': S, 'empirical_p_S': emp_p, 'product_p_S': prod, 'sample_size': N_samples})
    return results


# --- Nice wrapper that prints the "five number" summary and small tables ---
def print_unified_diagnostics(findability_analyzer,
                              prime_pool,
                              prime_subsets,
                              height_bound=None,
                              bootstrap_N=5000,
                              bootstrap_max_num=10**4,
                              bootstrap_max_den=10**4,
                              mi_primes_limit=40,
                              mi_N=2000):
    """
    Run bootstrap visibility, pairwise MI (on first mi_primes_limit primes), and a
    small per-subset empirical coverage check (first 30 subsets).
    Prints concise summary suitable for pasting in logs.
    """
    print("\n=== Unified diagnostics: running bootstrap visibility (this may take a few seconds) ===")
    boot = bootstrap_visibility(findability_analyzer, N_samples=bootstrap_N,
                                max_num=bootstrap_max_num, max_den=bootstrap_max_den)
    print(f"avg_fraction (unbiased sample): {boot['avg_fraction']:.3f}")
    print(f"fraction >= 0.1 : {boot['frac_above_0.1']:.3%}, fraction >= 0.5 : {boot['frac_above_0.5']:.3%}")
    if boot['empirical_visible_fraction'] is not None:
        print(f"empirical visible fraction (any prime_subset): {boot['empirical_visible_fraction']:.3%}")
    else:
        print("empirical visible fraction: (no prime_subsets available in analyzer)")

    # MI
    primes_for_mi = list(prime_pool)[:mi_primes_limit]
    print(f"\n=== Pairwise MI on first {len(primes_for_mi)} primes (binary match indicators) ===")
    mires = pairwise_mutual_info(findability_analyzer, primes_for_mi, sample_ms=boot['m_samples'], N_samples=mi_N)
    print(f"Top MI pairs (p,q,MI bits):")
    for p,q,mi in mires['top_pairs']:
        print(f"  ({p:3},{q:3})  {mi:.4f} bits")
    print(f"fraction of pairs with MI >= 0.01 bits: {mires['frac_pairs_above_threshold']:.3%}")

    # per-subset empirical coverage for first subsets
    if prime_subsets:
        print("\n=== Per-subset empirical coverage (first 30 subsets) ===")
        subsets = prime_subsets[:30]
        subset_res = per_subset_empirical_coverage(findability_analyzer, subsets, sample_ms=boot['m_samples'])
        for r in subset_res:
            S = r['subset']
            print(f" subset {S[:6]}...  emp_p={r['empirical_p_S']:.4g}  product_p={r['product_p_S']:.4g}")
    else:
        print("No prime_subsets available to check.")
    # return raw objects for further programmatic use
    return {'bootstrap': boot, 'mi': mires, 'subset_res': locals().get('subset_res', None)}


def prior_from_arithmetic(k_found,
                          p_visibility,
                          selmer_dim=None,
                          r_found=None,
                          crt_candidates_found=None,
                          rationality_tests_success=None,
                          h_max=None,
                          known_heights=None):
    """
    Produce a suggested geometric-prior parameter q (and component mus) from arithmetic signals.
    Returns dict with 'mu_selmer','mu_local','mu_height','mu_bootstrap','mu_combined','q'.
    All components optional; function uses available inputs.
    """
    # defaults
    mu_selmer = 0.0
    mu_local = 0.0
    mu_height = 0.0
    mu_bootstrap = 0.0

    # Selmer signal
    if selmer_dim is not None and r_found is not None:
        delta_r = max(0, selmer_dim - r_found)
        if delta_r == 0:
            mu_selmer = 0.02
        else:
            mu_selmer = 0.2 * delta_r


    # Replace this block in prior_from_arithmetic
    if crt_candidates_found and rationality_tests_success is not None and crt_candidates_found > 0:
        rho_global = rationality_tests_success / float(crt_candidates_found)
        # Ignore meaningless tiny ratios (< 0.05%) that just reflect modular overgeneration
        if rho_global > 0.0005:
            est_missed_classes = max(0.0, (1.0 / rho_global - 1.0) * k_found)
            mu_local = min(10.0, est_missed_classes * 0.05)
        else:
            mu_local = 0.0

    # Height/regulator signal
    if h_max is not None and known_heights:
        max_known = max(known_heights)
        if max_known >= h_max * 0.95:
            mu_height = 0.01
        else:
            available = max(0.0, (h_max - max_known) / max(1e-12, h_max))
            mu_height = min(10.0, 0.5 * available)

    # Bootstrap / empirical signal
    if p_visibility is not None and p_visibility > 0:
        mu_bootstrap = k_found * ((1.0 - p_visibility) / p_visibility) * 0.01
        mu_bootstrap = min(mu_bootstrap, 50.0)

    mu_combined = max(mu_selmer, mu_local, mu_height, mu_bootstrap)
    q = mu_combined / (1.0 + mu_combined)
    return {
        'mu_selmer': mu_selmer,
        'mu_local': mu_local,
        'mu_height': mu_height,
        'mu_bootstrap': mu_bootstrap,
        'mu_combined': mu_combined,
        'q': q
    }


import math

def completeness_posterior_geometric(k, p, q=0.10, m_max=200):
    """
    Bayesian posterior for true number of points T given:
      - observed found points k
      - per-point detection probability p (0<p<1, e.g. bootstrap avg_fraction)
      - geometric prior on missed count m = T-k: Pr(m) = (1-q) q^m

    Returns dict with:
      - 'posterior': dict mapping T -> posterior prob (T = k..k+m_max)
      - 'P_all' = Pr(T == k)
      - 'P_all_but_1' = Pr(T <= k+1)
      - 'P_all_but_r' for r up to a few if desired
      - 'posterior_mean_T'
    """
    from math import comb

    post = {}
    Z = 0.0
    for m in range(0, m_max+1):
        T = k + m
        prior_m = (1.0 - q) * (q ** m)
        # likelihood: choose k found out of T, each found with prob p
        # if T < k this is 0, but T = k+m with m>=0 so fine
        like = comb(T, k) * (p ** k) * ((1.0 - p) ** (T - k))
        val = prior_m * like
        post[T] = val
        Z += val

    # normalize
    if Z <= 0:
        # numerical failure; return degenerate posterior at T=k
        return {'posterior': {k: 1.0}, 'P_all': 1.0, 'P_all_but_1': 1.0, 'posterior_mean_T': float(k)}

    for T in list(post.keys()):
        post[T] /= Z

    P_all = post.get(k, 0.0)
    P_all_but_1 = sum(v for T, v in post.items() if T <= k + 1)
    # produce a few extra tail probs for convenience
    P_all_but_2 = sum(v for T, v in post.items() if T <= k + 2)
    mean_T = sum(T * v for T, v in post.items())

    return {
        'posterior': post,
        'P_all': P_all,
        'P_all_but_1': P_all_but_1,
        'P_all_but_2': P_all_but_2,
        'posterior_mean_T': mean_T
    }
