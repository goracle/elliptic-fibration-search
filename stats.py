# search_stats.py
import time
import json
import math
from collections import defaultdict, Counter
from functools import reduce
from operator import mul

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
