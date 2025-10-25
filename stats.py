# search_stats.py
import time
import json
import math
from collections import defaultdict, Counter
from functools import reduce
from operator import mul

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

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "summaries"
    analyze_dir(directory)
