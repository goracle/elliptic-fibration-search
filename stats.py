# search_stats.py
import time
import json
import math
from collections import defaultdict, Counter

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
        # number of CRT lift attempts and rational recon attempts
        self.counters.update({
            'modular_checks': 0,
            'crt_lift_attempts': 0,
            'rational_recon_attempts': 0,
            'rational_recon_success': 0,
            'rational_recon_failure': 0,
            'multiply_ops': 0,
            'symbolic_solves_attempted': 0,
            'symbolic_solves_success': 0
        })
        # discard reasons and examples
        self.discard_reasons = Counter()
        self.discard_examples = defaultdict(list)  # reason -> [candidate_examples...]
        # store sample successful points and m-values
        self.successes = []
        self.failures = []

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
        self.counters['rational_recon_success'] += 1
        self.successes.append({'m': m_value, 'pt': point})

    def record_failure(self, m_value, reason=None):
        self.counters['rational_recon_failure'] += 1
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
        prod_frac, per_prime = self.prime_coverage_fraction()
        return {
            'elapsed': time.time() - self.start_time,
            'phase_times': dict(self.phase_times),
            'counters': dict(self.counters),
            'discard_reasons': dict(self.discard_reasons),
            'discard_examples': dict(self.discard_examples),
            'success_count': len(self.successes),
            'failure_count': len(self.failures),
            'prime_coverage_product_heuristic': prod_frac,
            'prime_coverage_per_prime': per_prime
        }

    def to_json(self, path):
        with open(path, 'w') as fh:
            json.dump(self.summary(), fh, indent=2, default=int)
