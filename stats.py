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

    def crt_visibility_by_subsets(self, m_val, prime_subsets):
        """
        Check whether a rational m_val is congruent (mod M) to at least one CRT
        class actually tested by the search. Returns True if visible, else False.
        """
        assert hasattr(self.stats, 'crt_classes_tested'), "stats must have crt_classes_tested set"

        from sage.all import QQ, Integer, crt
        from functools import reduce
        import operator

        # Normalize rational
        try:
            a = int(QQ(m_val).numerator())
            b = int(QQ(m_val).denominator())
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Cannot coerce m_val={m_val} to rational: {e}")

        for subset in prime_subsets:
            # Validate subset structure
            if not subset:
                continue

            try:
                primes = [int(p) for p in subset]
            except (TypeError, ValueError) as e:
                if DEBUG:
                    print(f"[crt_vis] Malformed subset {subset}: {e}")
                raise  # Don't silently skip - this indicates data corruption

            # Drop primes dividing denominator
            usable_primes = [p for p in primes if b % p != 0]
            if not usable_primes:
                continue

            # Compute residue (a * b^{-1}) mod p for each usable prime
            res_vals = []
            mod_vals = []
            for p in usable_primes:
                try:
                    inv = pow(b, -1, p)
                except ValueError as e:
                    # This shouldn't happen since we filtered b % p != 0
                    raise ArithmeticError(f"Modular inverse failed for b={b} mod p={p}: {e}")

                r = (a * inv) % p
                res_vals.append(int(r))
                mod_vals.append(int(p))

            if not res_vals:
                continue

            # Convert to Sage Integers for CRT
            sage_res = [Integer(r) for r in res_vals]
            sage_mods = [Integer(m) for m in mod_vals]

            try:
                if len(sage_res) == 1:
                    m_mod = sage_res[0]
                    M = int(sage_mods[0])
                else:
                    m_mod = crt(sage_res, sage_mods)
                    M = int(reduce(operator.mul, mod_vals, 1))
            except Exception as e:
                # CRT failure is serious - indicates inconsistent residues
                if DEBUG:
                    print(f"[crt_vis] CRT failed for subset {subset}, residues {res_vals}: {e}")
                raise ArithmeticError(f"CRT reconstruction failed: {e}")

            canonical = (int(m_mod) % int(M), int(M))
            if canonical in self.stats.crt_classes_tested:
                return True

        return False


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
            sig['crt_visible'] = self.crt_visibility_by_subsets(r, prime_subsets)
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
        sig['crt_visible'] = analyzer.crt_visibility_by_subsets(m, prime_subsets)
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

# In stats.py

class CRTAnalyzer:
    """
    Analyze CRT visibility and modular coverage for a batch of rational m-values.
    Works with your SearchStats object.
    """

    def __init__(self, stats, prime_pool, prime_subsets):
        self.stats = stats
        self.prime_pool = list(prime_pool)
        self.prime_subsets = prime_subsets  # list of prime subsets used in the search

    def crt_visibility(self, m_val):
        """Return True if m_val is visible via at least one tested CRT class."""
        a, b = QQ(m_val).numerator(), QQ(m_val).denominator()

        for subset in self.prime_subsets:
            usable = [int(p) for p in subset if b % int(p) != 0]
            if not usable:
                continue

            residues, moduli = [], []
            for p in usable:
                try:
                    r = (a * pow(b, -1, p)) % p
                    residues.append(int(r))
                    moduli.append(int(p))
                except Exception:
                    residues = []
                    break

            if not residues:
                continue

            try:
                m_mod = crt(residues, moduli)
                M = reduce(lambda x, y: x*y, moduli, 1)
                if (int(m_mod) % M, M) in self.stats.crt_classes_tested:
                    return True
            except Exception:
                continue

        return False


    def visibility_signature(self, m_val):
    """Compute per-prime match, fraction matched, and CRT consistency."""
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
            continue
        usable += 1
        
        try:
            residue = (a * pow(b, -1, p)) % p
        except ValueError as e:
            raise ArithmeticError(f"Modular inverse failed for b={b} mod p={p}: {e}")
        
        seen = self.stats.residues_by_prime.get(p, set())
        ok = residue in seen
        per_prime[p] = (residue, ok)
        if ok:
            matched += 1

    # Heuristic product density
    if not self.prime_pool:
        density = 0.0
    else:
        log_density = 0.0
        count = 0
        for p in self.prime_pool:
            if p not in self.stats.residues_by_prime:
                continue
            L = len(self.stats.residues_by_prime[p])
            if L == 0:
                continue
            log_density += math.log(L / float(p))
            count += 1
        
        density = math.exp(log_density) if count > 0 else 0.0

    frac = matched / usable if usable > 0 else 0.0
    
    return {
        'm': (a, b),
        'per_prime': per_prime,
        'matched': matched,
        'usable': usable,
        'coverage': density,
        'fraction': frac,
        'crt_visible': self.crt_visibility_by_subsets(m_val, self.stats.prime_subsets)
    }



    def analyze_batch(self, m_list):
        """Analyze a batch of rationals, return list of signatures."""
        results = []
        product_density = None

        for m in m_list:
            sig = self.visibility_signature(m)
            if product_density is None:
                product_density = sig['coverage']
            results.append(sig)

        return {
            'product_density_heuristic': product_density,
            'samples': results,
            'fraction_visible': sum(1 for r in results if r['crt_visible']) / len(results) if results else 0.0
        }



class FindabilityAnalyzer:
    def __init__(self, stats, prime_pool):
        self.stats = stats
        self.prime_pool = list(prime_pool)

    def visibility_signature(self, m_val):
    """Compute per-prime match, fraction matched, and CRT consistency."""
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
            continue
        usable += 1
        
        try:
            residue = (a * pow(b, -1, p)) % p
        except ValueError as e:
            raise ArithmeticError(f"Modular inverse failed for b={b} mod p={p}: {e}")
        
        seen = self.stats.residues_by_prime.get(p, set())
        ok = residue in seen
        per_prime[p] = (residue, ok)
        if ok:
            matched += 1

    # Heuristic product density
    if not self.prime_pool:
        density = 0.0
    else:
        log_density = 0.0
        count = 0
        for p in self.prime_pool:
            if p not in self.stats.residues_by_prime:
                continue
            L = len(self.stats.residues_by_prime[p])
            if L == 0:
                continue
            log_density += math.log(L / float(p))
            count += 1
        
        density = math.exp(log_density) if count > 0 else 0.0

    frac = matched / usable if usable > 0 else 0.0
    
    return {
        'm': (a, b),
        'per_prime': per_prime,
        'matched': matched,
        'usable': usable,
        'coverage': density,
        'fraction': frac,
        'crt_visible': self.crt_visibility_by_subsets(m_val, self.stats.prime_subsets)
    }


    def crt_visibility(self, m_val):
        """Return True if m_val is visible via at least one tested CRT class."""
        a, b = QQ(m_val).numerator(), QQ(m_val).denominator()

        for subset in self.stats.prime_subsets:
            usable = [int(p) for p in subset if b % int(p) != 0]
            if not usable:
                continue

            residues, moduli = [], []
            for p in usable:
                try:
                    r = (a * pow(b, -1, p)) % p
                    residues.append(int(r))
                    moduli.append(int(p))
                except Exception:
                    residues = []
                    break

            if not residues:
                continue

            try:
                m_mod = crt(residues, moduli)
                M = reduce(lambda x, y: x*y, moduli, 1)
                if (int(m_mod) % M, M) in self.stats.crt_classes_tested:
                    return True
            except Exception:
                continue

        return False


    def crt_visibility_by_subsets(self, m_val, prime_subsets):
        """
        Check whether a rational m_val is congruent (mod M) to at least one CRT
        class actually tested by the search. Returns True if visible, else False.
        """
        assert hasattr(self.stats, 'crt_classes_tested'), "stats must have crt_classes_tested set"

        from sage.all import QQ, Integer, crt
        from functools import reduce
        import operator

        # Normalize rational
        try:
            a = int(QQ(m_val).numerator())
            b = int(QQ(m_val).denominator())
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Cannot coerce m_val={m_val} to rational: {e}")

        for subset in prime_subsets:
            # Validate subset structure
            if not subset:
                continue

            try:
                primes = [int(p) for p in subset]
            except (TypeError, ValueError) as e:
                if DEBUG:
                    print(f"[crt_vis] Malformed subset {subset}: {e}")
                raise  # Don't silently skip - this indicates data corruption

            # Drop primes dividing denominator
            usable_primes = [p for p in primes if b % p != 0]
            if not usable_primes:
                continue

            # Compute residue (a * b^{-1}) mod p for each usable prime
            res_vals = []
            mod_vals = []
            for p in usable_primes:
                try:
                    inv = pow(b, -1, p)
                except ValueError as e:
                    # This shouldn't happen since we filtered b % p != 0
                    raise ArithmeticError(f"Modular inverse failed for b={b} mod p={p}: {e}")

                r = (a * inv) % p
                res_vals.append(int(r))
                mod_vals.append(int(p))

            if not res_vals:
                continue

            # Convert to Sage Integers for CRT
            sage_res = [Integer(r) for r in res_vals]
            sage_mods = [Integer(m) for m in mod_vals]

            try:
                if len(sage_res) == 1:
                    m_mod = sage_res[0]
                    M = int(sage_mods[0])
                else:
                    m_mod = crt(sage_res, sage_mods)
                    M = int(reduce(operator.mul, mod_vals, 1))
            except Exception as e:
                # CRT failure is serious - indicates inconsistent residues
                if DEBUG:
                    print(f"[crt_vis] CRT failed for subset {subset}, residues {res_vals}: {e}")
                raise ArithmeticError(f"CRT reconstruction failed: {e}")

            canonical = (int(m_mod) % int(M), int(M))
            if canonical in self.stats.crt_classes_tested:
                return True

        return False



    def completeness_curve(self, xs, all_xs, bins=10):
        # height = log(max(|num|,|den|))
        def H(x):
            num = abs(int(QQ(x).numerator()))
            den = abs(int(QQ(x).denominator()))
            return float(math.log(max(num, den, 1)))

        foundH = [H(x) for x in xs]
        allH = [H(x) for x in all_xs]
        lo, hi = min(allH), max(allH)
        step = (hi - lo) / bins if hi > lo else 1.0
        edges = [lo + i * step for i in range(bins + 1)]
        results = []
        for e in edges[1:]:
            tot = sum(1 for h in allH if h <= e)
            got = sum(1 for h in foundH if h <= e)
            frac = got / tot if tot > 0 else 0.0
            results.append((e, frac))
        return results


# paste into your running environment (where SearchStats + FindabilityAnalyzer are available)

from fractions import Fraction
from sage.all import QQ

def per_point_diagnostics(stats, prime_pool, prime_subsets, m_values):
    analyzer = FindabilityAnalyzer(stats, prime_pool)
    for m in m_values:
        qm = QQ(m)
        sig = analyzer.visibility_signature(qm)
        crt_vis = analyzer.crt_visibility_by_subsets(qm, prime_subsets)
        print(f"m = {sig['m']}, frac_matched = {sig['fraction']:.3f}, crt_visible = {crt_vis}")
        # show a few primes with (residue, ok)
        show_primes = sorted(sig['per_prime'].items())[:20]
        print("  per-prime (p:(residue,ok)) sample:", show_primes)
        # if not visible, enumerate subsets where it *would* match per-prime but wasn't tested
        if not crt_vis:
            matching_subs = []
            for subset in prime_subsets:
                usable = [int(p) for p in subset if int(sig['m'][1]) % int(p) != 0]
                if not usable: 
                    continue
                # check per-prime residue match
                ok_all = True
                for p in usable:
                    a,b = sig['m']
                    r = (int(a) * pow(int(b), -1, p)) % p
                    if r not in stats.residues_by_prime.get(p, set()):
                        ok_all = False
                        break
                if ok_all:
                    # candidate subset would be matching on per-prime counts (but maybe combination not tested)
                    matching_subs.append(tuple(usable))
            if matching_subs:
                print("  subsets that would match per-prime (but may not have actual tested tuple):", matching_subs[:8])
        print()

from functools import reduce
import operator
from collections import defaultdict

def compute_actual_subset_cover(stats, prime_subsets):
    # build map M -> number of tested CRT classes with modulus M
    tested_by_M = defaultdict(int)
    for (res, M) in stats.crt_classes_tested:
        tested_by_M[int(M)] += 1

    per_subset = []
    for subset in prime_subsets:
        M = reduce(operator.mul, [int(p) for p in subset], 1)
        tested_count = tested_by_M.get(M, 0)
        p_s_actual = tested_count / float(M) if M > 0 else 0.0
        per_subset.append((tuple(subset), p_s_actual, tested_count, M))
    # union probability
    prod = 1.0
    for _, p_s, _, _ in per_subset:
        prod *= (1.0 - min(max(p_s, 0.0), 0.999999999999))
    P_visible_actual = 1.0 - prod
    return P_visible_actual, per_subset


if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "summaries"
    analyze_dir(directory)



# Paste this into your running Sage/Python environment (same session as cumulative_stats)
import math
from functools import reduce
import operator
from collections import defaultdict

def compute_product_model(stats, prime_subsets):
    """
    For each prime subset S, compute p_model = prod_p (L_p / p)
    where L_p = len(stats.residues_by_prime[p]).
    Returns list of (tuple(subset), p_model).
    """
    pm = []
    for subset in prime_subsets:
        ps = 1.0
        ok = True
        for p in subset:
            p = int(p)
            L = len(stats.residues_by_prime.get(p, set()))
            if L == 0:
                ok = False
                ps = 0.0
                break
            ps *= (L / float(p))
        pm.append((tuple(int(p) for p in subset), ps))
    return pm

# small helper to compute actual per-subset tested fraction (if not already defined):
def compute_actual_subset_cover_map(stats, prime_subsets):
    # map modulus M -> number of tested classes with that M
    tested_by_M = defaultdict(int)
    for (res, M) in stats.crt_classes_tested:
        tested_by_M[int(M)] += 1
    per_subset_actual = []
    for subset in prime_subsets:
        M = reduce(operator.mul, [int(p) for p in subset], 1)
        tested_count = tested_by_M.get(M, 0)
        p_s_actual = tested_count / float(M) if M > 0 else 0.0
        per_subset_actual.append((tuple(int(p) for p in subset), p_s_actual, tested_count, M))
    return per_subset_actual

# Add to stats.py - Height-based completeness metric

class CompletenessAnalyzer:
    """
    Estimate what fraction of rational points up to canonical height H
    were found by the search.
    
    Key insight: We can't enumerate all rationals up to height H efficiently,
    but we CAN estimate how many we SHOULD have found using:
    1. The CRT classes we tested
    2. The height distribution of points we found
    3. Visibility analysis of our found points
    """
    
    def __init__(self, stats, prime_pool, prime_subsets, height_bound):
        """
        Args:
            stats: SearchStats object with crt_classes_tested populated
            prime_pool: List of primes used in search
            prime_subsets: List of prime subsets actually searched
            height_bound: Canonical height bound used for search vectors
        """
        self.stats = stats
        self.prime_pool = sorted(prime_pool)
        self.prime_subsets = prime_subsets
        self.height_bound = float(height_bound)
        self.analyzer = FindabilityAnalyzer(stats, prime_pool)
    
    def canonical_height_of_x(self, x_val):
        """
        Naive canonical height proxy: h(x) ≈ log(max(|num|, |den|))
        
        For better estimates, use the actual elliptic curve canonical height,
        but this is a reasonable first-order approximation for x-coordinates.
        """
        from sage.all import QQ
        q = QQ(x_val)
        num = abs(int(q.numerator()))
        den = abs(int(q.denominator()))
        return float(math.log(max(num, den, 1)))
    
    def estimate_total_points_in_height_range(self, h_max):
        """
        Heuristic estimate of how many rational x-coordinates exist with h(x) <= h_max.
        
        Number of rationals a/b with max(|a|,|b|) <= N is roughly 6/π² * N²
        (by counting primitive fractions via Euler's totient)
        
        So number with log(max(|a|,|b|)) <= h is roughly (6/π²) * exp(2*h)
        """
        # Farey sequence heuristic
        N_max = math.exp(h_max)
        return (6.0 / (math.pi ** 2)) * (N_max ** 2)
    
    def compute_visibility_rate(self, found_xs):
        """
        For the points we found, compute what fraction are CRT-visible.
        This tells us our "detection efficiency".
        """
        if not found_xs:
            return 0.0, {}
        
        visible_count = 0
        invisible_examples = []
        
        for x in found_xs:
            is_visible = self.analyzer.crt_visibility_by_subsets(x, self.prime_subsets)
            if is_visible:
                visible_count += 1
            elif len(invisible_examples) < 5:
                invisible_examples.append(x)
        
        rate = visible_count / len(found_xs)
        
        return rate, {
            'visible': visible_count,
            'total': len(found_xs),
            'invisible_examples': invisible_examples
        }
    
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
    
    def m_value_from_x(self, x_val, r_m_func, shift):
        """
        Invert x = r_m(m) - shift to get m from x.
        For linear case: x = -m - const - shift => m = -x - const - shift
        """
        from sage.all import QQ
        const = r_m_func(m=QQ(0))
        return -(QQ(x_val) + const + shift)
    
    def compute_m_space_coverage(self, found_xs, r_m_func, shift):
        """
        For each found x, compute its m-value and check what fraction of
        CRT classes near that m were actually tested.
        
        This tells us: "Of the m-values that COULD produce rational points,
        what fraction of their neighborhoods did we test?"
        """
        if not found_xs:
            return 0.0, []
        
        coverage_samples = []
        
        for x in found_xs:
            m_val = self.m_value_from_x(x, r_m_func, shift)
            
            # For this m-value, check coverage across all prime subsets
            # Count: how many subsets have this m in a tested CRT class?
            visible_in_subsets = 0
            total_subsets = len(self.prime_subsets)
            
            for subset in self.prime_subsets:
                # Check if m_val's residue class was tested for this subset
                if self._is_m_covered_by_subset(m_val, subset):
                    visible_in_subsets += 1
            
            local_coverage = visible_in_subsets / total_subsets if total_subsets > 0 else 0.0
            coverage_samples.append({
                'x': x,
                'm': m_val,
                'coverage': local_coverage,
                'visible_subsets': visible_in_subsets,
                'total_subsets': total_subsets
            })
        
        # Average coverage across all found points
        avg_coverage = sum(s['coverage'] for s in coverage_samples) / len(coverage_samples)
        
        return avg_coverage, coverage_samples
    
    def _is_m_covered_by_subset(self, m_val, subset):
        """Check if m_val's CRT class for this subset was tested"""
        from sage.all import QQ, crt
        from functools import reduce
        import operator
        
        a = int(QQ(m_val).numerator())
        b = int(QQ(m_val).denominator())
        
        # Filter out primes dividing denominator
        usable = [p for p in subset if b % int(p) != 0]
        if not usable:
            return False
        
        try:
            # Compute m's residue class for this subset
            residues = [(a * pow(b, -1, p)) % p for p in usable]
            M = reduce(operator.mul, usable, 1)
            m_mod = crt(residues, usable)
            
            canonical = (int(m_mod) % int(M), int(M))
            return canonical in self.stats.crt_classes_tested
        except Exception:
            return False
    
    def completeness_lower_bound(self, found_xs, r_m_func, shift):
        """
        Conservative lower bound based on m-space coverage of found points.
        
        Logic: If we found points at m-values {m1, m2, ...} and those m-values
        have average CRT coverage C, then we've tested fraction C of the relevant
        m-space. Completeness >= C (assuming found points are representative).
        """
        avg_coverage, samples = self.compute_m_space_coverage(found_xs, r_m_func, shift)
        
        # This is actually our BEST estimate, not just a lower bound
        return {
            'estimate': avg_coverage,
            'reasoning': 'Average CRT coverage of m-values corresponding to found points',
            'samples': samples[:5]  # Show first 5 for debugging
        }
    
    def completeness_upper_bound_via_coverage(self, found_xs):
        """
        Upper bound using CRT coverage estimate.
        
        If our CRT coverage is C (fraction of m-space tested), then:
        - Expected points found ≈ C * (total points in height range)
        - Actual points found = |found_xs|
        - Implied total ≈ |found_xs| / C
        - Completeness upper bound ≈ |found_xs| / (|found_xs| / C) = C
        
        But this assumes uniform distribution, which is false. So we
        need to correct for the height distribution bias.
        """
        if not found_xs:
            return {'upper_bound': 1.0, 'reasoning': 'No points found'}
        
        # Estimate CRT coverage (what fraction of m-values we tested)
        # Use the product model as a rough estimate
        prod_density, per_prime = self.stats.prime_coverage_fraction()
        
        # Height-based correction
        h_dist = self.height_distribution_of_found(found_xs)
        h_mean = h_dist['mean']
        
        # If we're searching up to height H, but found points have mean height h_mean,
        # we're biased toward finding LOW height points (they're denser in m-space).
        # Correction factor: points at height h are ~exp(2*h) times as common as at height H
        if h_mean < self.height_bound:
            density_ratio = math.exp(2 * (self.height_bound - h_mean))
            effective_coverage = prod_density * density_ratio
        else:
            effective_coverage = prod_density
        
        # Upper bound: we've covered effective_coverage of the space
        upper = min(1.0, effective_coverage)
        
        return {
            'upper_bound': upper,
            'reasoning': 'CRT coverage adjusted for height distribution',
            'raw_coverage': prod_density,
            'height_adjustment': density_ratio if h_mean < self.height_bound else 1.0,
            'mean_found_height': h_mean,
            'height_bound': self.height_bound
        }
    
    def estimate_missing_points(self, found_xs):
        """
        Estimate how many points we SHOULD have found but didn't.
        
        Method: For each found point, check if it's CRT-visible.
        If visibility rate is V, and we found N points, then:
        - True total ≈ N / V (assuming found points are representative)
        - Missing ≈ (N/V) - N = N*(1-V)/V
        """
        vis_rate, details = self.compute_visibility_rate(found_xs)
        
        if vis_rate < 0.01:  # Can't trust estimate if visibility is too low
            return {
                'estimate': 'unknown',
                'reasoning': f'Visibility rate too low ({vis_rate:.1%}) for reliable estimate'
            }
        
        N = len(found_xs)
        estimated_total = N / vis_rate
        estimated_missing = estimated_total - N
        
        return {
            'found': N,
            'estimated_total': estimated_total,
            'estimated_missing': estimated_missing,
            'confidence': 'low' if vis_rate < 0.3 else 'medium' if vis_rate < 0.7 else 'high',
            'visibility_rate': vis_rate,
            'reasoning': f'Extrapolation from {vis_rate:.1%} visibility rate'
        }
    
    def full_report(self, found_xs):
        """Generate complete completeness analysis report"""
        if not found_xs:
            return {
                'status': 'no_points_found',
                'completeness_estimate': 'unknown',
                'recommendation': 'No points found - increase height bound or check curve'
            }
        
        h_dist = self.height_distribution_of_found(found_xs)
        lower = self.completeness_lower_bound(found_xs)
        upper = self.completeness_upper_bound_via_coverage(found_xs)
        missing = self.estimate_missing_points(found_xs)
        
        # Compute final estimate as midpoint of bounds
        if lower['lower_bound'] > 0 and upper['upper_bound'] < 1:
            estimate = (lower['lower_bound'] + upper['upper_bound']) / 2
            confidence = 'medium'
        elif lower['lower_bound'] > 0.8:
            estimate = lower['lower_bound']
            confidence = 'high'
        else:
            estimate = lower['lower_bound']
            confidence = 'low'
        
        # Generate recommendation
        if estimate > 0.95:
            rec = 'Search appears complete ✓'
        elif estimate > 0.7:
            rec = 'Likely found most points. Consider 1-2 more runs to verify.'
        elif estimate > 0.4:
            rec = 'Significant points may be missing. Increase NUM_SUBSETS or run targeted recovery.'
        else:
            rec = 'Low completeness. Increase HEIGHT_BOUND and NUM_SUBSETS significantly.'
        
        return {
            'completeness_estimate': estimate,
            'confidence': confidence,
            'bounds': {
                'lower': lower['lower_bound'],
                'upper': upper['upper_bound']
            },
            'found_points': len(found_xs),
            'estimated_missing': missing.get('estimated_missing', 'unknown'),
            'height_distribution': h_dist,
            'recommendation': rec,
            'details': {
                'lower_bound_analysis': lower,
                'upper_bound_analysis': upper,
                'missing_points_analysis': missing
            }
        }
    
    def print_report(self, found_xs):
        """Pretty-print completeness report"""
        report = self.full_report(found_xs)
        
        print("\n" + "="*70)
        print("COMPLETENESS ANALYSIS")
        print("="*70)
        
        if report.get('status') == 'no_points_found':
            print(report['recommendation'])
            return
        
        est = report['completeness_estimate']
        conf = report['confidence']
        
        print(f"\nEstimated Completeness: {est:.1%} (confidence: {conf})")
        print(f"  Lower bound: {report['bounds']['lower']:.1%}")
        print(f"  Upper bound: {report['bounds']['upper']:.1%}")
        
        print(f"\nPoints Found: {report['found_points']}")
        if report['estimated_missing'] != 'unknown':
            print(f"Estimated Missing: {report['estimated_missing']:.1f}")
        
        print(f"\nHeight Distribution:")
        h = report['height_distribution']
        print(f"  Range: [{h['min']:.2f}, {h['max']:.2f}]")
        print(f"  Mean: {h['mean']:.2f}, Median: {h['median']:.2f}")
        print(f"  (Height bound used: {self.height_bound:.2f})")
        
        print(f"\n{report['recommendation']}")
        
        print("="*70)


def print_unified_completeness_report(stats, prime_pool, prime_subsets, 
                                     height_bound, found_xs):
    """
    Single unified completeness report to replace the scattered diagnostics.
    
    Call this ONCE at the end of search, after all the technical stats.
    """
    analyzer = CompletenessAnalyzer(stats, prime_pool, prime_subsets, height_bound)
    analyzer.print_report(found_xs)



# Patch for stats.py - Replace silent exception handlers

# In crt_visibility_by_subsets method:

# In visibility_signature method:
