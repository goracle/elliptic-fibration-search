import time, json, math, random, os, numpy as np
from collections import defaultdict, Counter
from operator import mul
from datetime import datetime
from fractions import Fraction
from sage.all import QQ, crt, exp, log, oo, RDF, Integer, RealNumber

# stats.py

# Sage imports

# Local imports (assumed available in python path)
try:
    from bounds import build_split_poly_from_cd, compute_residue_counts_for_primes, estimate_galois_signature_modp
    from search_common import DEBUG, MAX_MODULUS, MIN_PRIME_SUBSET_SIZE
except ImportError:
    # Allow running in isolation if utils aren't present, though some methods will fail
    raise

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

        # Track rejected primes
        self.rejected_primes = []  # List of (prime, reason) tuples

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

    def merge(self, other):
        """
        Merge another SearchStats object into this one.
        """
        if not isinstance(other, SearchStats):
            return

        # Merge phase times
        for phase, t in other.phase_times.items():
            self.phase_times[phase] += t

        # Merge counters
        self.counters.update(other.counters)

        # Merge residues by prime
        for p, res_set in other.residues_by_prime.items():
            self.residues_by_prime[p].update(res_set)

        # Merge discard reasons
        self.discard_reasons.update(other.discard_reasons)

        # Merge discard examples (keep first 5 per reason)
        for reason, examples in other.discard_examples.items():
            current_len = len(self.discard_examples[reason])
            needed = 5 - current_len
            if needed > 0:
                self.discard_examples[reason].extend(examples[:needed])

        # Merge successes/failures (keep last 1000)
        self.successes.extend(other.successes)
        self.failures.extend(other.failures)
        self.successes = self.successes[-1000:]
        self.failures = self.failures[-1000:]

        # Merge CRT classes tested
        self.crt_classes_tested.update(other.crt_classes_tested)

        # Merge prime subsets
        self.prime_subsets.extend(other.prime_subsets)

        # Merge rejected_primes list
        if hasattr(other, 'rejected_primes'):
            self.rejected_primes.extend(other.rejected_primes)
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for item in self.rejected_primes:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            self.rejected_primes = deduped

    def merge_dict(self, stats_dict):
        """Merge a simple Counter dict into counters."""
        self.counters.update(stats_dict)

    # ---------------- CRT ----------------
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

        # Calculate log product to avoid underflow
        log_prod = sum(math.log(f) for f in fracs if f > 0)
        prod = math.exp(log_prod)
        per_prime = {int(p): len(S)/float(p) for p, S in self.residues_by_prime.items()}
        return prod, per_prime

    def crt_space_ratio(self, prime_list):
        M_log10 = sum(math.log10(p) for p in prime_list)
        coverage_prod, _ = self.prime_coverage_fraction()
        return coverage_prod, M_log10

    def crt_coverage_exact(self, prime_subsets_used):
        # Do not use reduce. Use explicit loop.
        total_classes_possible = 0
        for subset in prime_subsets_used:
            prod_val = 1
            for p in subset:
                prod_val *= int(p)
            total_classes_possible += prod_val

        classes_tested = len(self.crt_classes_tested)
        return classes_tested / total_classes_possible if total_classes_possible > 0 else 0

    def expected_runs_for_coverage(self, prime_subsets_used, target_coverage=0.99):
        coverage_per_run = self.crt_coverage_exact(prime_subsets_used)
        if coverage_per_run >= target_coverage or coverage_per_run == 0:
            return 1
        p = coverage_per_run
        # Avoid log(0) if p is very close to 1 (unlikely here but safe)
        if p >= 1.0:
            return 1
        expected_runs = math.log(1 - target_coverage) / math.log(1 - p)
        return math.ceil(expected_runs)

    # ---------------- Diagnostics ----------------
    def compare_target_m_residues(self, m_value, prime_pool):
        """
        For a target rational m = a/b, compare its residue mod p against tested residues.
        """
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

            # Using pow(b, -1, p) is standard Python 3.8+
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

    def subset_match_probability(self, subset):
        """
        Estimate probability that a uniform random rational m avoids denominator primes
        and matches all residues for that subset.
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
                any_zero = True
                logp = float('-inf')
                break
            logp += log(max(1e-300, L / float(p)))

        if any_zero and logp == float('-inf'):
            return 0.0, {'per_prime': per_prime}
        p_subset = exp(logp)
        return p_subset, {'per_prime': per_prime}

    def estimate_overall_visibility(self, prime_subsets):
        p_list = []
        subset_details = []
        for subset in prime_subsets:
            p_s, detail = self.subset_match_probability(subset)
            p_list.append(p_s)
            subset_details.append((list(subset), p_s, detail))

        prod_log = 0.0
        for p in p_list:
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
        # We need a flat list of primes for the analyzer
        all_primes_in_subsets = set()
        for s in prime_subsets:
            for p in s:
                all_primes_in_subsets.add(int(p))

        analyzer = FindabilityAnalyzer(self, sorted(list(all_primes_in_subsets)))
        samples = []
        visible_count = 0

        for q in known_rationals:
            # Handle input format safely
            try:
                if isinstance(q, tuple) and len(q) == 2:
                    r = QQ(q[0])/QQ(q[1])
                else:
                    r = QQ(q)
            except Exception:
                continue

            sig = analyzer.visibility_signature(r)
            sig['crt_visible'] = sig['fraction'] > 0.1
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
        print(f"Points: {report['points_found']}/{report['points_expected']} ({float(report['recall']):.0%} recall)")
        print(f"Efficiency: {report['candidates_per_point']:.1f} CRT candidates per point found")
        # Format explicitly as float to avoid Sage Rational formatting error
        print(f"Hit rate: {float(report['hit_rate']):.1%}")

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
        print(f"Avg hit rate: {100*float(avg_hit_rate):.1f}%")
        print(f"Curves tested: {len(self.runs)}")

# ---------------- Summary Writing Utils ----------------
def _rational_to_pair(q):
    if isinstance(q, tuple):
        assert len(q) == 2
        return (int(q[0]), int(q[1]))
    if hasattr(q, 'numerator') and hasattr(q, 'denominator'):
        return (int(q.numerator()), int(q.denominator()))
    if isinstance(q, int):
        return (q, 1)
    raise AssertionError("unexpected rational type: " + str(type(q)))

def normalize_summary(run):
    out = dict(run)
    out['run_id'] = out.get('run_id') or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out['curve_id'] = str(out.get('curve_id', 'unknown'))
    out['wall_seconds'] = float(out.get('wall_seconds', 0.0))
    out['total_crt_candidates'] = int(out.get('total_crt_candidates', 0))
    out['total_lift_attempts'] = int(out.get('total_lift_attempts', 0))
    out['total_rationality_tests_success'] = int(out.get('total_rationality_tests_success', 0))
    out['total_rationality_tests_failure'] = int(out.get('total_rationality_tests_failure', 0))

    ux = out.get('unique_x_list', [])
    out['unique_x_list'] = [_rational_to_pair(q) for q in ux]

    pcounts = out.get('per_point_counts', {})
    out['per_point_counts'] = {str(k): int(v) for k, v in pcounts.items()}

    rs = out.get('residues_seen', {})
    out['residues_seen'] = {str(k): int(v) for k, v in rs.items()}

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

# ---------------- Analysis Utils ----------------
def chao1_estimator(counts):
    S_obs = len(counts)
    f1 = sum(1 for c in counts if c == 1)
    f2 = sum(1 for c in counts if c == 2)
    if f2 > 0:
        return S_obs + (f1 * f1) / (2.0 * f2)
    if f1 > 0:
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

    global_point_counts = Counter()
    per_curve_runs = defaultdict(list)

    for s in summaries:
        curve = s['curve_id']
        per_curve_runs[curve].append(s)
        for x in s.get('unique_x_list', []):
            global_point_counts[tuple(x)] += 1

    print("Loaded", len(summaries), "summaries for", len(per_curve_runs), "curves")

    for curve, runs in per_curve_runs.items():
        print("\n=== Curve:", curve, "runs:", len(runs))
        for s in runs:
            ux = s.get('unique_x_list', [])
            n_unique = len(ux)
            print(" run", s['run_id'], "time(s):", s.get('wall_seconds'), "unique_x:", n_unique,
                  "crt_candidates:", s.get('total_crt_candidates', 0),
                  "rational_success:", s.get('total_rationality_tests_success', 0))

        combined = Counter()
        for s in runs:
            for x in s.get('unique_x_list', []):
                combined[tuple(x)] += 1
        freqs = list(combined.values())
        if not freqs:
            print(" no points found across runs")
            continue

        freqs_np = np.array(freqs)
        mean_freq = np.mean(freqs_np)

        # Heterogeneity check using numpy
        if len(freqs) > 1 and mean_freq != 0:
            heterogeneity = (np.std(freqs_np, ddof=1)**2) / mean_freq
        else:
            heterogeneity = 0.0

        print(" points found:", len(freqs), "mean frequency:", mean_freq, "heterogeneity index:", heterogeneity)

        # (Rest of analyze_dir largely unchanged for brevity, as it's reporting logic)

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
        r_values = np.array([r/p for p, r in residue_counts.items()])
        avg_density = np.mean(r_values)
        # Signal 3: How many primes are "zero-ratio" (no roots)?
        zero_ratio = sum(1 for r in residue_counts.values() if r == 0) / len(residue_counts)

        # Signal 4: Canonical height pairing matrix condition number
        try:
            # We must import numpy locally or rely on global
            matrix_data = np.array(H.change_ring(RDF))
            cond = np.linalg.cond(matrix_data)
        except np.linalg.LinAlgError:
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
            'difficulty_score': difficulty_score,
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
        canonical = (int(m0) % int(M), int(M))
        self.tested_classes.add(canonical)

    def estimate_coverage(self, prime_subsets_used):
        # Method 1: Direct counting
        total_classes_possible = 0
        for subset in prime_subsets_used:
            prod_s = 1
            if subset:
                for p in subset:
                    prod_s *= int(p)
            total_classes_possible += prod_s

        classes_tested = len(self.tested_classes)

        if total_classes_possible < MAX_MODULUS:
            direct_coverage = classes_tested / total_classes_possible
        else:
            direct_coverage = None

        # Method 2: Heuristic via residue density
        density_product = 1.0
        for p in self.prime_pool:
            r_p = self.residue_counts.get(p, 1)
            density_product *= (r_p / float(p))

        # Method 3: Birthday paradox estimate
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
        current = self.estimate_coverage(prime_subsets_used)

        if current['direct_coverage'] is not None:
            p = current['direct_coverage']
        elif current['birthday_coverage'] is not None:
            p = current['birthday_coverage']
        else:
            p = current['heuristic_coverage']

        if p >= target_coverage:
            return 0

        if not len(self.tested_classes):
            return -1
        else:
            coverage_per_run = p / len(self.tested_classes)
            try:
                expected_runs = math.log(1 - target_coverage) / math.log(1 - coverage_per_run)
                expected_runs = math.ceil(expected_runs)
            except ZeroDivisionError:
                expected_runs = oo

            return expected_runs

class FindabilityAnalyzer:
    """
    Analyzes the "findability" of a rational m-value.
    Includes simple caching to avoid recomputing for same m.
    """
    def __init__(self, stats, prime_pool):
        self.stats = stats
        self.prime_pool = list(prime_pool)
        self._cache = {} # Simple cache for visibility_signature

    def assess_crt_findability(self, m_val):
        """
        Deterministic check: Can we reconstruct m_val from the residues we found?
        """
        # Ensure we are working with rationals
        a = QQ(m_val).numerator()
        b = QQ(m_val).denominator()

        compatible_primes = []
        M_capacity = 1

        for p in self.prime_pool:
            if b % p == 0: continue

            # Compute required residue
            # pow(b, -1, p) is fast modular inverse
            residue = (int(a) * pow(int(b), -1, int(p))) % int(p)

            if residue in self.stats.residues_by_prime.get(p, set()):
                compatible_primes.append(p)
                M_capacity *= p

        # Required modulus size: M > 2 * max(|a|, |b|)^2
        M_required = 2 * max(abs(a), abs(b))**2

        findable = M_capacity > M_required

        return {
            'findable': findable,
            'M_capacity': M_capacity,
            'M_required': M_required,
            'compatible_primes': compatible_primes,
            'num_compatible': len(compatible_primes),
            'capacity_ratio': float(M_capacity) / float(M_required) if M_required > 0 else float('inf')
        }

    def visibility_signature(self, m_val):
        # Use cache if possible. Convert m_val to hashable key (tuple of ints)
        try:
            q = QQ(m_val)
            key = (int(q.numerator()), int(q.denominator()))
        except (TypeError, ValueError):
             # If m_val isn't a rational, don't cache or expect one, just run logic
             key = None

        if key is not None and key in self._cache:
            return self._cache[key]

        a, b = key if key else (int(QQ(m_val).numerator()), int(QQ(m_val).denominator()))

        matched = 0
        usable = 0
        per_prime = {}

        for p in self.prime_pool:
            if b % p == 0:
                per_prime[p] = ('DENOM_ZERO', False)
                continue
            usable += 1

            residue = (a * pow(b, -1, p)) % p

            seen = self.stats.residues_by_prime.get(p, set())
            ok = residue in seen
            per_prime[p] = (residue, ok)
            if ok:
                matched += 1

        # Global density metric (independent of m)
        if not self.prime_pool:
            density = 0.0
        else:
            densities = []
            for p in self.prime_pool:
                if p not in self.stats.residues_by_prime:
                    continue
                L = len(self.stats.residues_by_prime[p])
                if p == 0: continue
                densities.append(L / float(p))
            density = sum(densities) / len(densities) if densities else 0.0

        frac = matched / usable if usable > 0 else 0.0
        crt_info = self.assess_crt_findability(m_val)

        result = {
            'm': (a, b),
            'per_prime': per_prime,
            'matched': matched,
            'usable': usable,
            'coverage': density,
            'fraction': frac,
            'crt_findable': crt_info['findable'],
            'crt_capacity_log10': math.log10(crt_info['M_capacity']) if crt_info['M_capacity'] > 0 else 0,
            'crt_compatible_primes': crt_info['compatible_primes']
        }

        if key is not None:
            self._cache[key] = result
        return result

class CompletenessAnalyzer:
    """
    Estimate what fraction of rational m-values were "findable".
    """
    def __init__(self, stats, prime_pool, prime_subsets, height_bound, r_m_func, shift):
        self.stats = stats
        self.prime_pool = sorted(list(set(prime_pool)))
        self.prime_subsets = prime_subsets
        self.height_bound = float(height_bound)
        self.r_m_func = r_m_func
        self.shift = shift
        self.analyzer = FindabilityAnalyzer(stats, self.prime_pool)

    def m_value_from_x(self, x_val):
        const = self.r_m_func(m=QQ(0))
        # x_val = r_m(m) - shift => m = const - (x_val + shift)
        return -(QQ(x_val) + self.shift - const)

    def full_report(self, found_xs):
        if not found_xs:
            return {'recommendation': 'No points found'}

        found_analysis = []
        all_findable = True

        for x in found_xs:
            m = self.m_value_from_x(x)
            sig = self.analyzer.visibility_signature(m)
            found_analysis.append({
                'x': x,
                'findable': sig['crt_findable'],
                'primes': len(sig['crt_compatible_primes']),
                'capacity': sig['crt_capacity_log10']
            })
            if not sig['crt_findable']:
                all_findable = False

        return {
            'found_analysis': found_analysis,
            'all_findable': all_findable,
            'recommendation': 'Increase bounds' if not all_findable else 'Likely Complete'
        }

    def print_report(self, found_xs):
        report = self.full_report(found_xs)
        print("\n" + "="*70)
        print("COMPLETENESS ANALYSIS (Exact CRT Check)")
        print("="*70)

        if not found_xs:
            print("No points to analyze.")
            return

        print(f"\nFound Points Reconstructibility:")
        print(f"{'x':<10} | {'Findable?':<10} | {'Primes':<8} | {'Log10 Cap'}")
        print("-" * 45)

        for item in report['found_analysis']:
            x_str = str(item['x'])[:10]
            status = "YES" if item['findable'] else "NO"
            print(f"{x_str:<10} | {status:<10} | {item['primes']:<8} | {item['capacity']:.2f}")

        if not report['all_findable']:
            print("\n⚠️  Some found points are NOT reconstructible from current residues.")
            print("    (They were likely found via specialized subsets or lucky hits.)")
        else:
            print("\n✓ All found points are theoretically reconstructible.")
        print("="*70)

    def canonical_height_of_x(self, x_val):
        q = QQ(x_val)
        num = abs(int(q.numerator()))
        den = abs(int(q.denominator()))
        return float(math.log(max(num, den, 1)))

    def compute_m_space_coverage(self, found_xs):
        if not found_xs:
            return 0.0, None, None, []

        coverage_samples = []
        min_info = {'fraction': 1.1}
        max_info = {'fraction': -0.1}

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

            if findability_frac < min_info['fraction']:
                min_info = {'x': x, 'm': m_val, 'fraction': findability_frac}
            if findability_frac > max_info['fraction']:
                max_info = {'x': x, 'm': m_val, 'fraction': findability_frac}

        if not coverage_samples:
            return 0.0, None, None, []

        avg_coverage = sum(s['findability_fraction'] for s in coverage_samples) / len(coverage_samples)

        min_result = min_info if min_info['fraction'] <= 1.0 else None
        max_result = max_info if max_info['fraction'] >= 0.0 else None

        return avg_coverage, min_result, max_result, coverage_samples

# ---------------- Independent Functions ----------------

def analyze_sample_m_list(m_list, analyzer, prime_subsets):
    results = []
    product_density = None

    for m in m_list:
        sig = analyzer.visibility_signature(m)
        if product_density is None:
            product_density = sig['coverage']
        sig['crt_visible'] = sig['fraction'] > 0.1
        results.append(sig)

    if product_density is None:
        product_density = 0.0

    try:
        from search_common import MIN_PRIME_SUBSET_SIZE
        cutoff = MIN_PRIME_SUBSET_SIZE
    except ImportError:
        cutoff = 3

    meet_count = sum(1 for sig in results if sig['matched'] >= cutoff)
    frac_meet = meet_count / len(results) if results else 0.0

    return {
        'product_density_heuristic': product_density,
        'fraction_meet_min_subset': frac_meet,
        'samples': results
    }

def bootstrap_visibility(findability_analyzer,
                         N_samples=5000,
                         max_num=10**4,
                         max_den=10**4,
                         seed=None,
                         thresholds=(0.1, 0.5)):
    if seed is not None:
        random.seed(seed)
    fractions = []
    m_samples = []
    visible_count = 0
    per_prime_counts = Counter()

    prime_subsets = getattr(findability_analyzer, 'prime_subsets', None)

    # Use QQ(a)/QQ(b) rather than QQ(a,b)
    for i in range(N_samples):
        num = random.randint(-max_num, max_num)
        den = random.randint(1, max_den)
        m = QQ(num) / QQ(den)

        sig = findability_analyzer.visibility_signature(m)
        frac = float(sig.get('fraction', 0.0))
        fractions.append(frac)
        m_samples.append(m)

        per_prime = sig.get('per_prime', {})
        for p, (_, ok) in per_prime.items():
            if ok:
                per_prime_counts[p] += 1

        if prime_subsets:
            for S in prime_subsets:
                # Check if all p in subset S are matched
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

def pairwise_mutual_info(findability_analyzer,
                         primes,
                         sample_ms=None,
                         N_samples=2000,
                         seed=None,
                         top_k=20,
                         mi_threshold=0.01):
    if seed is not None:
        random.seed(seed)

    if sample_ms is None:
        sample_ms = []
        for _ in range(N_samples):
            num = random.randint(-10000, 10000)
            den = random.randint(1, 10000)
            sample_ms.append(QQ(num)/QQ(den))
    else:
        N_samples = len(sample_ms)

    matched = {p: [] for p in primes}
    for m in sample_ms:
        sig = findability_analyzer.visibility_signature(m)
        perp = sig.get('per_prime', {})
        for p in primes:
            ok = perp.get(p, (None, False))[1]
            matched[p].append(1 if ok else 0)

    def mi_binary(a_list, b_list):
        # Mutual info for binary vectors
        N = len(a_list)
        cnt = Counter(zip(a_list, b_list))
        mi = 0.0
        for (a,b), c in cnt.items():
            p_ab = c / N
            p_a = sum(1 for x in a_list if x == a) / N
            p_b = sum(1 for x in b_list if x == b) / N
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

def per_subset_empirical_coverage(findability_analyzer,
                                  subsets,
                                  sample_ms=None,
                                  N_samples=2000):
    if sample_ms is None:
        sample_ms = []
        for _ in range(N_samples):
            num = random.randint(-10000, 10000)
            den = random.randint(1, 10000)
            sample_ms.append(QQ(num)/QQ(den))
    else:
        N_samples = len(sample_ms)

    # Precompute signatures
    sig_cache = {}
    per_prime_ok = defaultdict(int)
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

        prod = 1.0
        for p in S:
            prod *= per_prime_density.get(p, 0.0)
        results.append({'subset': S, 'empirical_p_S': emp_p, 'product_p_S': prod, 'sample_size': N_samples})
    return results

def print_unified_completeness_report(stats, prime_pool, prime_subsets,
                                     height_bound, found_xs, r_m, shift):
    try:
        analyzer = CompletenessAnalyzer(stats, prime_pool, prime_subsets,
                                       height_bound, r_m, shift)
        analyzer.print_report(found_xs)
    except Exception as e:
        # Let it crash if critical, but just print error here as it's a report
        print("\n" + "="*70)
        print("COMPLETENESS ANALYSIS FAILED")
        print(f"Error: {e}")
        print("="*70)

def print_unified_diagnostics(findability_analyzer,
                              prime_pool,
                              prime_subsets,
                              height_bound=None,
                              bootstrap_N=5000,
                              bootstrap_max_num=10**4,
                              bootstrap_max_den=10**4,
                              mi_primes_limit=40,
                              mi_N=2000):

    print("\n=== Unified diagnostics: running bootstrap visibility ===")
    boot = bootstrap_visibility(findability_analyzer, N_samples=bootstrap_N,
                                max_num=bootstrap_max_num, max_den=bootstrap_max_den)
    print(f"avg_fraction (unbiased sample): {boot['avg_fraction']:.3f}")
    # Cast to float for formatting
    print(f"fraction >= 0.1 : {float(boot['frac_above_0.1']):.3%}, fraction >= 0.5 : {float(boot['frac_above_0.5']):.3%}")

    if boot['empirical_visible_fraction'] is not None:
        print(f"empirical visible fraction (any prime_subset): {float(boot['empirical_visible_fraction']):.3%}")
    else:
        print("empirical visible fraction: (no prime_subsets available in analyzer)")

    # MI
    primes_for_mi = list(prime_pool)[:mi_primes_limit]
    print(f"\n=== Pairwise MI on first {len(primes_for_mi)} primes ===")
    mires = pairwise_mutual_info(findability_analyzer, primes_for_mi, sample_ms=boot['m_samples'], N_samples=mi_N)
    print(f"Top MI pairs (p,q,MI bits):")
    for p,q,mi in mires['top_pairs']:
        print(f"  ({p:3},{q:3})  {mi:.4f} bits")
    print(f"fraction of pairs with MI >= 0.01 bits: {float(mires['frac_pairs_above_threshold']):.3%}")

    # per-subset empirical coverage
    if prime_subsets:
        print("\n=== Per-subset empirical coverage (first 30 subsets) ===")
        subsets = prime_subsets[:30]
        subset_res = per_subset_empirical_coverage(findability_analyzer, subsets, sample_ms=boot['m_samples'])
        for r in subset_res:
            S = r['subset']
            # formatting S as string explicitly
            print(f" subset {str(S[:6])}...  emp_p={r['empirical_p_S']:.4g}  product_p={r['product_p_S']:.4g}")
    else:
        print("No prime_subsets available to check.")

    return {'bootstrap': boot, 'mi': mires, 'subset_res': locals().get('subset_res', None)}

def completeness_posterior_geometric(k, p, q=0.10, m_max=200):
    from math import comb

    post = {}
    Z = 0.0
    for m in range(0, m_max+1):
        T = k + m
        prior_m = (1.0 - q) * (q ** m)
        like = comb(T, k) * (p ** k) * ((1.0 - p) ** (T - k))
        val = prior_m * like
        post[T] = val
        Z += val

    if Z <= 0:
        return {'posterior': {k: 1.0}, 'P_all': 1.0, 'P_all_but_1': 1.0, 'posterior_mean_T': float(k)}

    for T in list(post.keys()):
        post[T] /= Z

    P_all = post.get(k, 0.0)
    P_all_but_1 = sum(v for T, v in post.items() if T <= k + 1)
    P_all_but_2 = sum(v for T, v in post.items() if T <= k + 2)
    mean_T = sum(T * v for T, v in post.items())

    return {
        'posterior': post,
        'P_all': P_all,
        'P_all_but_1': P_all_but_1,
        'P_all_but_2': P_all_but_2,
        'posterior_mean_T': mean_T
    }

def adjust_visibility_for_fiber_collisions(p_visibility, prime_pool, rejected_primes_list, debug=True):
    if not rejected_primes_list:
        return {
            'p_adjusted': p_visibility,
            'collision_primes': [],
            'other_rejected': [],
            'reachable_fraction': 1.0,
            'adjustment_factor': 1.0
        }

    collision_primes = []
    other_rejected = []

    for p, reason in rejected_primes_list:
        reason_str = str(reason).lower()
        if 'fiber' in reason_str or 'collision' in reason_str:
            collision_primes.append(p)
        else:
            other_rejected.append((p, reason))

    if not collision_primes:
        return {
            'p_adjusted': p_visibility,
            'collision_primes': [],
            'other_rejected': other_rejected,
            'reachable_fraction': 1.0,
            'adjustment_factor': 1.0
        }

    reachable_fraction = 1.0
    for p in collision_primes:
        reachable_fraction *= (1.0 - 1.0/float(p))

    adjustment_factor = reachable_fraction
    p_adjusted = p_visibility * adjustment_factor

    if debug:
        print(f"\n[fiber_collision_adjustment]")
        print(f"  Collision primes: {collision_primes}")
        print(f"  Reachable fraction: {reachable_fraction:.4f}")
        print(f"  Adjusted p_visibility: {p_adjusted:.4f}")

    return {
        'p_adjusted': p_adjusted,
        'collision_primes': collision_primes,
        'other_rejected': other_rejected,
        'reachable_fraction': reachable_fraction,
        'adjustment_factor': adjustment_factor
    }

def prior_from_arithmetic(k_found,
                          p_visibility,
                          rejected_primes=None,
                          prime_pool=None,
                          selmer_dim=None,
                          r_found=None,
                          crt_candidates_found=None,
                          rationality_tests_success=None,
                          h_max=None,
                          known_heights=None):

    # Apply fiber collision adjustment
    if rejected_primes and prime_pool:
        try:
            from search_common import DEBUG
        except ImportError:
            DEBUG = False
        adj = adjust_visibility_for_fiber_collisions(p_visibility, prime_pool, rejected_primes, debug=DEBUG)
        p_adjusted = adj['p_adjusted']
        collision_primes = adj['collision_primes']
    else:
        p_adjusted = p_visibility
        collision_primes = []

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

    # Local signal
    if crt_candidates_found and rationality_tests_success is not None and crt_candidates_found > 0:
        rho_global = rationality_tests_success / float(crt_candidates_found)

        if rho_global > 0.001:
            est_missed_classes = max(0.0, (1.0 / rho_global - 1.0) * k_found)
            mu_local = min(10.0, est_missed_classes * 0.02)
        else:
            mu_local = 0.0

    # Height signal
    if h_max is not None and known_heights:
        max_known = max(known_heights)
        if max_known >= h_max * 0.95:
            mu_height = 0.01
        else:
            available = max(0.0, (h_max - max_known) / max(1e-12, h_max))
            mu_height = min(10.0, 0.5 * available)

    # Bootstrap signal using ADJUSTED p
    if p_adjusted is not None and p_adjusted > 0:
        mu_bootstrap = k_found * ((1.0 - p_adjusted) / p_adjusted) * 0.01
        mu_bootstrap = min(mu_bootstrap, 50.0)

    mu_combined = max(mu_selmer, mu_local, mu_height, mu_bootstrap)
    q = mu_combined / (1.0 + mu_combined)

    return {
        'mu_selmer': mu_selmer,
        'mu_local': mu_local,
        'mu_height': mu_height,
        'mu_bootstrap': mu_bootstrap,
        'mu_combined': mu_combined,
        'q': q,
        'p_adjusted': p_adjusted,
        'p_raw': p_visibility,
        'collision_primes': collision_primes
    }
