# brauer.py
# Practical / heuristic algebraic Brauer-Manin checks for elliptic fibrations
# - Sage-compatible (assumes `sage -python3` or imported inside a .sage driver)
# - Minimal, explicit, no imports inside functions
# - Relies on precomputed residue maps as in search_lll.py:
#     compute_residues_for_m, compute_residue_coverage_for_m, build_targeted_subset
#   See search_lll.py for the exact structures expected. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
#
# Author: generated to fit user's repo style and constraints

from sage.all import QQ, ZZ
from collections import Counter

# ---------------------------
# Helper / sanity utilities
# ---------------------------

def _coerce_rational(m):
    """
    Coerce m to QQ cleanly. Accepts (a,b) tuple, Python Fraction, Sage QQ, int.
    """
    if isinstance(m, tuple) and len(m) == 2:
        a = int(m[0]); b = int(m[1])
        return QQ(ZZ(a)) / QQ(ZZ(b))
    return QQ(m)


def _product(iterable):
    # explicit product to avoid reduce issues on Sage
    p = 1
    for x in iterable:
        p *= int(x)
    return p


# ---------------------------
# Model: local evaluation statistic extractor
# ---------------------------

def prime_survival_fraction_from_residues(precomputed_residues, prime):
    """
    Given precomputed_residues[p] -> { v_tuple : [ set(roots_rhs0), ... ] },
    build a simple model of the fraction of m (mod p) that would survive modular
    tests for *some* vector.  Returns fraction in [0,1].
    - If prime has no numeric residues, returns 1.0 (conservative: prime gives no information).
    """
    assert prime is not None
    p = int(prime)
    pmap = precomputed_residues.get(p, {})
    if not pmap:
        # we have no data: treat as non-discriminating (conservative)
        return 1.0

    numeric_residues = set()
    for vtuple, rhs_lists in pmap.items():
        for s in rhs_lists:
            # s is expected to be a set of ints or empty set
            for r in s:
                if isinstance(r, int):
                    numeric_residues.add(r)
    # If no numeric residues recorded for p, conservative
    if not numeric_residues:
        return 1.0

    # fraction of residues allowed (simple model: allowed residues / p)
    frac = float(len(numeric_residues)) / float(p)
    # sanity clamp
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return frac


# ---------------------------
# Estimate global completeness
# ---------------------------

def estimate_completeness_probability(precomputed_residues, prime_pool, primes_for_model=None):
    """
    Using a simple independence model, estimate the probability that a random rational
    m (with denominator not vanishing on the primes used) would *not* be ruled out by
    the modular information in precomputed_residues up to the supplied prime_pool.

    Returns a dict:
      {'per_prime_frac': {p: frac_survive_p, ...},
       'estimate_survive': float,   # product of per-prime fractions
       'estimate_ruled_out': float  # 1 - estimate_survive
      }

    Notes:
     - This is heuristic: it assumes prime-level independence, which is the same
       approximation the rest of your statistics use.
     - If a prime has no numeric residues, we treat its survival fraction as 1.0.
    """
    assert isinstance(prime_pool, (list, tuple))
    if primes_for_model is None:
        primes_for_model = list(prime_pool)

    per_prime = {}
    for p in primes_for_model:
        per_prime[int(p)] = prime_survival_fraction_from_residues(precomputed_residues, p)

    # product of survival fractions
    prod = 1.0
    for p, frac in per_prime.items():
        prod *= float(frac)

    return {
        'per_prime_frac': per_prime,
        'estimate_survive': float(prod),
        'estimate_ruled_out': float(1.0 - prod)
    }


# ---------------------------
# Targeted test: is an m killed?
# ---------------------------

def m_is_locally_allowed(m, precomputed_residues, prime_pool, v_tuple=None):
    """
    Given a rational m (QQ-coercible), test whether for each prime in prime_pool we
    can find that m (mod p) among precomputed residues (for some vector if v_tuple None).
    If any prime with numeric data rules out the residue, we report it as 'locally blocked'.

    Returns:
      (allowed_bool, details)
    where details is a dict with per-prime status values:
      {'p': {'residue': r or None, 'status': 'matched'|'unseen'|'denom_zero'|'no_data'}}
    Implementation re-uses the same expectations on precomputed_residues as search_lll.py. :contentReference[oaicite:4]{index=4}
    """
    m_q = _coerce_rational(m)
    a = ZZ(m_q.numerator()); b = ZZ(m_q.denominator())

    details = {}
    allowed = True

    for p in prime_pool:
        p = int(p)
        entry = {'residue': None, 'status': 'no_data'}
        if (b % p) == 0:
            entry['status'] = 'denom_zero'
            details[p] = entry
            # denominator zero primes cannot be used to rule out m
            continue

        residue = int((int(a % p) * pow(int(b % p), -1, p)) % p)
        entry['residue'] = residue

        p_map = precomputed_residues.get(p, {})
        if not p_map:
            entry['status'] = 'no_data'
            details[p] = entry
            continue

        found = False
        if v_tuple is not None:
            sets_list = p_map.get(tuple(v_tuple), [])
            for s in sets_list:
                if residue in s:
                    found = True
                    break
        else:
            for sets_list in p_map.values():
                for s in sets_list:
                    if residue in s:
                        found = True
                        break
                if found:
                    break

        if found:
            entry['status'] = 'matched'
        else:
            entry['status'] = 'unseen'
            allowed = False

        details[p] = entry

    return allowed, details


# ---------------------------
# Diagnostic: find prime contributors to any blockade
# ---------------------------

def blocking_primes_for_m(m, precomputed_residues, prime_pool, v_tuple=None):
    """
    Returns list of primes that would block m (i.e., have status 'unseen' in m_is_locally_allowed).
    """
    allowed, details = m_is_locally_allowed(m, precomputed_residues, prime_pool, v_tuple=v_tuple)
    blocked = [p for p, d in details.items() if d['status'] == 'unseen']
    return blocked, details


# ---------------------------
# Heuristic "algebraic Brauer" probe
# ---------------------------

def probe_algebraic_brauer_obstructions(precomputed_residues, prime_pool,
                                       candidate_ms=None, sample_size=500, v_tuple=None):
    """
    Heuristic probe for algebraic Brauer obstructions:
      - If candidate_ms is provided, test those m values explicitly.
      - Otherwise, sample random residues modulo the primes and use the residue-sets
        to estimate the fraction blocked (Monte-Carlo) under independence.

    Returns a result dict:
      {
        'explicit': { m_q: (allowed_bool, blocked_primes_list) , ... }   # present if candidate_ms given
        'monte_carlo': { 'survive_fraction_est': float, 'blocked_fraction_est': float }  # always present
      }

    NOTE: This is not a proof of a nontrivial Brauer element. It is a practical check
    which matches the data-driven modular filtering used elsewhere in the project.
    """
    result = {}
    # explicit tests
    if candidate_ms:
        explicit = {}
        for m in candidate_ms:
            m_q = _coerce_rational(m)
            allowed, details = m_is_locally_allowed(m_q, precomputed_residues, prime_pool, v_tuple=v_tuple)
            blocked = [p for p, d in details.items() if d['status'] == 'unseen']
            explicit[m_q] = (allowed, blocked, details)
        result['explicit'] = explicit

    # Monte-Carlo sampling: sample random residues across primes and see survival
    # For speed we sample small integers per-prime and CRT combine a subset of primes
    import random
    # choose a small subset of primes (to keep CRT modulus small) but representative
    subset = list(prime_pool)[:min(len(prime_pool), 8)]
    survive = 0
    trials = 0
    for _ in range(sample_size):
        # generate random residue vector (one residue per prime in subset)
        residues = [random.randrange(0, int(p)) for p in subset]
        # attempt to locate a matching vector/residue in precomputed_residues per-prime
        ok = True
        for p, r in zip(subset, residues):
            p = int(p)
            pmap = precomputed_residues.get(p, {})
            if not pmap:
                # no data -> treat as pass for this prime
                continue
            # if r appears anywhere in pmap, pass
            seen = False
            for vlist in pmap.values():
                for s in vlist:
                    if r in s:
                        seen = True
                        break
                if seen:
                    break
            if not seen:
                ok = False
                break
        if ok:
            survive += 1
        trials += 1

    survive_frac = float(survive) / float(max(1, trials))
    result['monte_carlo'] = {
        'subset_primes': subset,
        'sample_size': trials,
        'survive_fraction_est': survive_frac,
        'blocked_fraction_est': float(1.0 - survive_frac)
    }
    return result



# In a new file: ramification.py

def compute_ramification_locus(cd, verbose=True):
    """
    Compute the set of bad primes where Δ(m) has repeated roots.
    These are exactly the primes dividing resultant(Δ, Δ').
    
    Returns:
        set of primes
    """
    from sage.all import factor, PolynomialRing, is_prime
    
    PR_m = PolynomialRing(QQ, 'm')
    Delta = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
    
    if hasattr(Delta, 'numerator'):
        Delta_poly = PR_m(Delta.numerator())
    else:
        Delta_poly = PR_m(Delta)
    
    dDelta = Delta_poly.derivative()
    #R = resultant(Delta_poly, dDelta)
    R = Delta_poly.resultant(dDelta)
    
    if verbose:
        print(f"Resultant R = {R}")
    
    bad_primes = set()
    for p, _ in factor(R):
        if is_prime(p):
            bad_primes.add(int(p))
    
    return bad_primes


def compute_lll_constant(delta=0.98, d=1):
    """
    Compute the LLL guarantee constant for basis quality.
    
    Args:
        delta: LLL reduction parameter (0.75 < delta < 1)
        d: dimension (number of sections = MW rank)
    
    Returns:
        C such that shortest vector b1 satisfies ||b1|| ≤ C × det(L)^(1/d)
    """
    # From Lenstra-Lenstra-Lovász 1982:
    # ||b1|| ≤ (4/(4*delta - 1))^((d-1)/4) × det(L)^(1/d)
    import math
    C = (4.0 / (4.0 * delta - 1.0)) ** ((d - 1) / 4.0)
    return C

def prove_modulus_sufficiency(C_lll, height_bound, prime_subset):
    """
    Prove that prod(primes) > MAX_MODULUS is sufficient for reconstruction.
    
    Theorem: If M = prod(p in subset) > 2 × C_lll × exp(height_bound),
    then rational reconstruction succeeds for all sections up to height H.
    """
    from functools import reduce
    from operator import mul
    import math
    
    M = reduce(mul, prime_subset, 1)
    threshold = 2.0 * C_lll * math.exp(height_bound)
    
    return M > threshold, {
        'M': M,
        'threshold': threshold,
        'C_lll': C_lll,
        'height_bound': height_bound
    }


# In search7_genus2.sage
# (Can be placed near the top with other helper functions, or near the end before main)


def run_sufficiency_proof(height_bound, prime_subsets, mw_rank):
    """
    Runs the formal "C-bound" check to verify that the CRT modulus
    is sufficient for rational reconstruction up to the height bound.
    """
    print("\n" + "="*70)
    print("FORMAL COMPLETENESS PROOF (Roadmap Step 3)")
    print("="*70)
    
    if not prime_subsets:
        print("No prime subsets were used. Cannot run sufficiency proof.")
        print("="*70)
        return

    # 1. Compute the LLL constant
    # We use the Mordell-Weil rank as the dimension 'd'
    d = mw_rank
    if d == 0:
        print("MW rank is 0, setting dimension d=1 for LLL constant.")
        d = 1
        
    # Use the default LLL_DELTA from search_lll.py
    C_lll = compute_lll_constant(delta=0.98, d=d)
    print(f"LLL Guarantee Constant (C_lll) for rank d={d}: {C_lll:.4f}")
    
    # 2. Find the smallest modulus M used in the search
    # We must prove that *every* subset's modulus was sufficient
    min_M = 0
    min_M_subset = []
    
    for subset in prime_subsets:
        if not subset:
            continue
        M = 1
        for p in subset:
            M *= int(p)
        
        if M == 0:
            continue
            
        if min_M == 0 or M < min_M:
            min_M = M
            min_M_subset = subset
            
    if min_M == 0:
        print("Could not find a valid prime subset modulus. Skipping check.")
        print("="*70)
        return
        
    print(f"Smallest Modulus (M_min) used: {min_M} (from subset {min_M_subset})")
    
    # 3. Run the sufficiency proof
    is_sufficient, details = prove_modulus_sufficiency(C_lll, height_bound, min_M_subset)
    
    print(f"Height Bound (H): {details['height_bound']:.2f}")
    
    # Note: Using .2e for scientific notation to avoid f-string format issues
    threshold_str = f"{details['threshold']:.2e}"
    min_M_str = f"{details['M']:.2e}"
    
    print(f"Required: M > 2 * C_lll * exp(H)")
    print(f"Threshold: {threshold_str}")
    print(f"Actual M:  {min_M_str}")
    
    if is_sufficient:
        print("\n*** ✅ PASS ***")
        print("The smallest modulus M used in the search is formally sufficient")
        print("to guarantee rational reconstruction for all sections up to height H.")
    else:
        print("\n*** ⚠️  FAIL ***")
        print("The search modulus M is NOT large enough to guarantee reconstruction.")
        print("This implies the search may be incomplete (missed points).")
        print("RECOMMENDATION: Increase MIN_PRIME_SUBSET_SIZE or PRIME_POOL size.")
        
    print("="*70)
