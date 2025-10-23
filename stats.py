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
        # phase timers
        self.phase_times = defaultdict(float)
        self._phase_start = {}
        # counters
        self.counters = Counter()
        # mapping prime -> set(residues tested mod p)
        self.residues_by_prime = defaultdict(set)
        
        # Initialize all counters we expect to track
        self.counters.update({
            'modular_checks': 0,
            'crt_lift_attempts': 0,         # (from worker) CRT combos
            'rational_recon_attempts_worker': 0, # (from worker) rational_reconstruct calls
            'rational_recon_success_worker': 0, # (from worker) rational_reconstruct successes
            'rational_recon_failure_worker': 0, # (from worker) rational_reconstruct failures
            'rationality_tests_total': 0,   # (from checker) Total m-vals tested
            'rationality_tests_success': 0, # (from checker) m-vals that gave a y-point
            'rationality_tests_failure': 0, # (from checker) m-vals that failed y-test
            'multiply_ops': 0,
            'symbolic_solves_attempted': 0,
            'symbolic_solves_success': 0,
            'subsets_generated_initial': 0,
            'subsets_filtered_out_combo': 0,
            'subsets_processed': 0,
            'crt_candidates_found': 0,      # (m,v) pairs from workers
            'rational_points_unique': 0,
            'new_sections_unique': 0,
        })
        
        # discard reasons and examples
        self.discard_reasons = Counter()
        self.discard_examples = defaultdict(list)  # reason -> [candidate_examples...]
        # store sample successful points and m-values
        self.successes = []
        self.failures = []

        # In SearchStats.__init__:
        self.crt_classes_tested = set()  # set of tuples (m mod M, M)

    def merge(self, other_stats):
        """Merge another SearchStats object into this one."""
        if not isinstance(other_stats, SearchStats):
            return

        # Merge phase times
        for phase, t in other_stats.phase_times.items():
            self.phase_times[phase] += t
            
        # Merge counters
        self.counters.update(other_stats.counters)
        
        # Merge residue maps
        for p, res_set in other_stats.residues_by_prime.items():
            self.residues_by_prime[p].update(res_set)
            
        # Merge discard reasons
        self.discard_reasons.update(other_stats.discard_reasons)
        for reason, examples in other_stats.discard_examples.items():
            current_len = len(self.discard_examples[reason])
            needed = 5 - current_len
            if needed > 0:
                self.discard_examples[reason].extend(examples[:needed])
                
        # Merge success/failure lists (cap to avoid memory bloat)
        self.successes.extend(other_stats.successes)
        self.failures.extend(other_stats.failures)
        if len(self.successes) > 1000:
            self.successes = self.successes[-1000:]
        if len(self.failures) > 1000:
            self.failures = self.failures[-1000:]
            
        # Merge CRT classes
        self.crt_classes_tested.update(other_stats.crt_classes_tested)

    def merge_dict(self, stats_dict):
        """Merge a simple Counter dict (from a worker) into counters."""
        self.counters.update(stats_dict)

    # When you test a candidate:
    def record_crt_class(self, m_mod_M, M):
        """Record that we tested m ≡ m_mod_M (mod M)"""
        # Normalize to canonical representative
        canonical = (int(m_mod_M) % int(M), int(M))
        self.crt_classes_tested.add(canonical)

    # timing helpers
    def start_phase(self, name):
        self._phase_start[name] = time.time()

    def end_phase(self, name):
        if name in self._phase_start:
            dt = time.time() - self._phase_start.pop(name)
            self.phase_times[name] += dt

    # counters
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

    # simple completeness heuristics
    def prime_coverage_fraction(self):
        """
        For each prime p, compute fraction of residues seen f_p = |tested residues| / p.
        Assuming independence, estimated total CRT-class coverage ~= prod_p(f_p).
        This is a heuristic; primes are not independent in practice, but it's informative.
        """
        fracs = []
        for p, S in self.residues_by_prime.items():
            if p <= 0:
                continue
            fracs.append(len(S) / float(p))
        if not fracs:
            return 0.0, {}
        prod = 1.0
        for f in fracs:
            prod *= f
        per_prime = {int(p): len(S)/float(p) for p, S in self.residues_by_prime.items()}
        return prod, per_prime

    def crt_space_ratio(self, prime_list):
        """
        Direct ratio: (#unique residue tuples tested) / (product(primes)).
        If you are not tracking tuples, this gives an upper bound using independence assumption.
        Track unique tuples in your search if you want an exact value.
        """
        M_log10 = sum(math.log10(p) for p in prime_list)
        # compute estimated tested classes via product of fractions
        coverage_prod, _ = self.prime_coverage_fraction()
        if coverage_prod == 0.0:
            return 0.0, M_log10
        # estimated tested classes = coverage_prod * product(primes)
        # so ratio = coverage_prod
        return coverage_prod, M_log10

    def summary(self):
        """Returns a dict summary of all stats."""
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
        """Returns a formatted string of the summary."""
        s = self.summary()
        lines = []
        lines.append(f"Total time: {s['elapsed']:.2f}s")
        lines.append(f"Total Rational Points Found (Unique x): {s['counters'].get('rational_points_unique', 0)}")
        
        lines.append("\nPhases (s):")
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
                
        lines.append(f"\nSuccesses (rationality tests): {s['success_count']}, Failures: {s['failure_count']}")
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
        with open(path, 'w') as fh:
            json.dump(self.summary(), fh, indent=2, default=int)

    def crt_coverage_exact(self, prime_subsets_used):
        """
        Compute fraction of CRT classes tested for the subsets we actually used.
        """
        total_classes_possible = 0
        for subset in prime_subsets_used:
            M = reduce(mul, subset, 1)
            total_classes_possible += M

        # Count unique m values tested (modulo their respective M)
        classes_tested = len(self.crt_classes_tested)

        return classes_tested / total_classes_possible if total_classes_possible > 0 else 0

    def expected_runs_for_coverage(self, target_coverage=0.99):
        """
        Estimate runs needed to achieve target_coverage of CRT space.

        Assumes: Each run samples a random subset of CRT classes with 
        coverage p = (classes_tested / total_classes_possible).

        Classic coupon collector: E[runs] ≈ log(1 - target) / log(1 - p)
        """
        coverage_per_run = self.crt_coverage_exact(...)
        if coverage_per_run >= target_coverage:
            return 1

        # Coupon collector approximation
        p = coverage_per_run
        expected_runs = math.log(1 - target_coverage) / math.log(1 - p)
        return math.ceil(expected_runs)


class BenchmarkStats:
    def __init__(self, known_ground_truth):
        """
        Args:
            known_ground_truth: set of x-coordinates we expect to find
        """
        self.ground_truth = frozenset(known_ground_truth)
        self.start_time = time.time()
        
        # Discovery timeline
        self.discoveries = []  # [(timestamp, x_coord), ...]
        self.found_so_far = set()
        
        # Efficiency metrics
        self.total_crt_candidates = 0
        self.total_vectors_checked = 0
        self.total_prime_subsets_used = 0
        
        # Per-fibration tracking
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
        """Called when a new rational point is found"""
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
        """Key metrics for benchmarking"""
        total_time = time.time() - self.start_time
        found = len(self.found_so_far)
        expected = len(self.ground_truth)
        
        # Calculate cumulative discovery times
        discovery_times = [t for t, x in self.discoveries]
        
        return {
            'total_time': total_time,
            'points_found': found,
            'points_expected': expected,
            'recall': found / expected if expected > 0 else 0,
            'crt_candidates_tested': self.total_crt_candidates,
            'candidates_per_point': (self.total_crt_candidates / found 
                                    if found > 0 else float('inf')),
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
        print(f"Points: {report['points_found']}/{report['points_expected']} "
              f"({report['recall']:.0%} recall)")
        print(f"Efficiency: {report['candidates_per_point']:.1f} CRT candidates per point found")
        print(f"Hit rate: {report['hit_rate']:.1%}")
        
        if report['time_to_all_points']:
            print(f"Time to find all points: {report['time_to_all_points']:.2f}s")
        
        print(f"\nFibrations used: {report['fibrations_needed']} / {len(self.fibration_stats)} tried")
        print(f"Avg time per fibration: {report['avg_time_per_fibration']:.2f}s")
        
        # Show discovery timeline
        print("\nDiscovery timeline:")
        for t, x in self.discoveries:
            print(f"  {t:6.2f}s: x = {x}")
        
        # Per-fibration breakdown
        print("\nPer-fibration breakdown:")
        for i, fib in enumerate(self.fibration_stats):
            if fib['found_here']:
                print(f"  Fib {i} ({fib['base_pts']}): "
                      f"found {fib['found_here']} in {fib['duration']:.2f}s "
                      f"({fib['crt_candidates']} candidates)")


# Minimal benchmark tracking (no invasive changes)
class QuickBench:
    def __init__(self):
        self.runs = []  # List of {curve_id, time, candidates, points_found}
    
    def record(self, curve_id, time, candidates, points):
        self.runs.append({
            'curve': curve_id,
            'time': time,
            'candidates': candidates,
            'points': points,
            'hit_rate': points / candidates if candidates > 0 else 0,
        })
    
    def summary(self):
        avg_time = sum(r['time'] for r in self.runs) / len(self.runs)
        avg_hit_rate = sum(r['hit_rate'] for r in self.runs) / len(self.runs)
        
        print(f"Avg time: {avg_time:.1f}s")
        print(f"Avg hit rate: {100*avg_hit_rate:.1f}%")
        print(f"Curves tested: {len(self.runs)}")
