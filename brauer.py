from sage.all import QQ, ZZ, Integer, PolynomialRing, lcm, gcd, GF
from collections import Counter
# [fix] math.log is used in prove_modulus_sufficiency below (and in
# compute_lll_constant's docstring-adjacent code), but the only `import math`
# in this file was local to compute_lll_constant -- a function-local import
# doesn't leak into other functions or module scope, so prove_modulus_sufficiency
# hit NameError: name 'math' is not defined the first time it actually ran.
import math

# ---------------------------
# Helper / sanity utilities
# ---------------------------

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

    Theorem: If M = prod(p in subset) > 2 * C_lll * exp(height_bound),
    then rational reconstruction succeeds for all sections up to height H.

    FIX: This comparison is done in log-space to prevent overflow from exp(H).
    """
    from functools import reduce
    from operator import mul

    M = reduce(mul, [int(p) for p in prime_subset], 1)

    # --- FIX: Use logarithms to avoid overflow ---
    if M <= 0 or C_lll <= 0: # Safety check
        return False, {
            'M': M, 'log_M': 0, 'log_threshold': 0, 'C_lll': C_lll,
            'height_bound': height_bound, 'error': 'Non-positive M or C_lll'
        }

    log_M = math.log(float(M))
    # log(Threshold) = log(2 * C_lll * exp(H)) = log(2) + log(C_lll) + H
    log_threshold = math.log(2.0) + math.log(float(C_lll)) + float(height_bound)

    is_sufficient = log_M > log_threshold
    # --- END FIX ---

    return is_sufficient, {
        'M': M,
        'log_M': log_M,                 # New value for reporting
        'log_threshold': log_threshold, # New value for reporting
        'C_lll': C_lll,
        'height_bound': height_bound
    }

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
    d = mw_rank
    if d == 0:
        print("MW rank is 0, setting dimension d=1 for LLL constant.")
        d = 1

    C_lll = compute_lll_constant(delta=0.98, d=d)
    print(f"LLL Guarantee Constant (C_lll) for rank d={d}: {C_lll:.4f}")

    # 2. Find the smallest modulus M used
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

    # --- FIX: Print log-domain values ---
    log_M_str = f"{details['log_M']:.2f}"
    log_thresh_str = f"{details['log_threshold']:.2f}"

    print(f"Required (in log-space): log(M) > log(2*C_lll) + H")
    print(f"Actual log(M):  {log_M_str}")
    print(f"Required log(M): > {log_thresh_str}")
    # --- END FIX ---

    if is_sufficient:
        print("\n*** ✅ PASS ***")
        print("The smallest modulus M is formally sufficient")
        print("to guarantee rational reconstruction for all sections up to height H.")
    else:
        print("\n*** ⚠️  FAIL ***")
        print("The search modulus M is NOT large enough to guarantee reconstruction.")
        print("This implies the search may be incomplete (missed points).")
        print("RECOMMENDATION: Increase MIN_PRIME_SUBSET_SIZE or PRIME_POOL size.")

    print("="*70)

def analyze_prime_pool_sufficiency(prime_pool, min_subset_size, max_subset_size,
                                    target_naive_height_digits, mw_rank=1,
                                    max_modulus=None, verbose=True):
    """
    Pre-flight check: is the CONFIGURED prime pool / subset-size combination
    actually capable of clearing the modulus the completeness proof will
    demand for a target naive x-height, before you spend hours searching?

    This mirrors exactly what run_sufficiency_proof / prove_modulus_sufficiency
    check at the end of a run, but runs it up front against the worst case
    subset the sampler in bounds.py can draw, so you get a PASS/FAIL and a
    concrete "raise MIN_PRIME_SUBSET_SIZE to N" recommendation instead of
    finding out after the search that M_min was too small.

    Why worst case, not average case: subsets are drawn as
    random.choices(prime_pool, weights=..., k=size) -- i.e. WITH replacement,
    weighted toward high-root-count primes but not guaranteed to avoid small
    primes. A single subset that happens to land on mostly-small primes (or
    on few distinct primes due to replacement) can be the M_min that
    run_sufficiency_proof reports on, since that function scans ALL generated
    subsets for the smallest modulus. Sizing against "smallest possible
    subset size, smallest available primes" is what makes a PASS here a real
    guarantee.

    Also mirrors a real runtime gate: search_lll/modularthread.py drops any
    CRT lift with M > MAX_MODULUS (`if M > MAX_MODULUS: continue`), silently.
    So this also flags subsets whose modulus would be usable in principle but
    gets discarded at runtime because MAX_MODULUS is set too low.

    Args:
        prime_pool: the configured PRIME_POOL (list of ints/Sage primes).
        min_subset_size: MIN_PRIME_SUBSET_SIZE.
        max_subset_size: MIN_MAX_PRIME_SUBSET_SIZE.
        target_naive_height_digits: target naive x-height as "10^D" -- pass D.
            (Matches how the runtime proof actually compares against naive
            x-height in nats, not canonical height -- see the note in
            search_common.py above MIN_PRIME_SUBSET_SIZE.)
        mw_rank: Mordell-Weil rank / number of sections, for C_lll. Only
            weakly affects the threshold (C_lll enters as a log, so even
            rank 10 barely moves the required digit count).
        max_modulus: MAX_MODULUS; if the worst-case M exceeds this, those CRT
            lifts get silently skipped at runtime regardless of the proof.
            Defaults to the module-level MAX_MODULUS if available.
        verbose: print a human-readable report.

    Returns:
        dict with keys: 'pass', 'log10_M_worst', 'log10_M_best',
        'log10_threshold', 'recommended_min_subset_size',
        'worst_case_primes_used', 'max_modulus_ok'.
    """
    if max_modulus is None:
        max_modulus = globals().get('MAX_MODULUS', 10**500)

    pool_sorted = sorted(int(p) for p in prime_pool)
    if not pool_sorted:
        if verbose:
            print("analyze_prime_pool_sufficiency: PRIME_POOL is empty, cannot analyze.")
        return {'pass': False, 'error': 'empty prime pool'}

    # --- Required threshold (mirrors prove_modulus_sufficiency exactly) ---
    d = max(1, int(mw_rank))
    C_lll = compute_lll_constant(delta=0.98, d=d)
    h_x = float(target_naive_height_digits) * math.log(10.0)
    log10_threshold = (math.log(2.0) + math.log(C_lll) + h_x) / math.log(10.0)

    # --- Worst-case subset: the MIN_PRIME_SUBSET_SIZE smallest primes in the
    #     pool (this is what the sampler in bounds.py *could* draw, and what
    #     run_sufficiency_proof scans for as M_min across all subsets) ---
    k_worst = min(int(min_subset_size), len(pool_sorted))
    worst_primes = pool_sorted[:k_worst]
    log10_M_worst = sum(math.log10(p) for p in worst_primes)

    # --- Typical/best-case subset for context: max_subset_size LARGEST
    #     primes, representing a favorably-drawn subset, so you can see the
    #     spread between worst and best case ---
    k_best = min(int(max_subset_size), len(pool_sorted))
    best_primes = pool_sorted[-k_best:]
    log10_M_best = sum(math.log10(p) for p in best_primes)

    passes = log10_M_worst > log10_threshold

    # --- MAX_MODULUS sanity: does the runtime hard-cap silently discard the
    #     very moduli we're relying on? ---
    log10_max_modulus = math.log10(max_modulus) if max_modulus > 0 else 0.0
    max_modulus_ok = log10_max_modulus > log10_M_best + 1.0  # 1 extra digit of margin

    # --- Recommended MIN_PRIME_SUBSET_SIZE if it currently fails ---
    recommended_min_subset_size = k_worst
    if not passes:
        cum = 0.0
        recommended_min_subset_size = None  # pool itself too small even at full size
        for i, p in enumerate(pool_sorted):
            cum += math.log10(p)
            if cum > log10_threshold:
                recommended_min_subset_size = i + 1
                break

    if verbose:
        print("\n" + "=" * 70)
        print("PRIME POOL / SUBSET SIZE SUFFICIENCY ANALYSIS")
        print("=" * 70)
        print(f"Target naive x-height: 10^{target_naive_height_digits}  "
              f"(h_x = {h_x:.2f} nats)")
        print(f"MW rank used for C_lll: d={d}  (C_lll={C_lll:.4f})")
        print(f"Required: log10(M) > {log10_threshold:.2f}  "
              f"i.e. M > 10^{log10_threshold:.2f}")
        print("-" * 70)
        print(f"PRIME_POOL: {len(pool_sorted)} primes, "
              f"range [{pool_sorted[0]}, {pool_sorted[-1]}]")
        print(f"MIN_PRIME_SUBSET_SIZE = {min_subset_size}, "
              f"MIN_MAX_PRIME_SUBSET_SIZE = {max_subset_size}")
        print("-" * 70)
        print(f"WORST-CASE subset (smallest {k_worst} primes in pool, i.e. what "
              f"the weighted-with-replacement sampler could draw):")
        print(f"  primes: {worst_primes[:8]}{'...' if len(worst_primes) > 8 else ''}")
        print(f"  log10(M_worst) = {log10_M_worst:.2f}  "
              f"(M_worst ~ 10^{log10_M_worst:.2f})")
        print(f"BEST-CASE subset (largest {k_best} primes in pool):")
        print(f"  log10(M_best)  = {log10_M_best:.2f}  "
              f"(M_best  ~ 10^{log10_M_best:.2f})")
        print("-" * 70)
        if passes:
            print(f"*** PASS *** worst-case subset modulus (10^{log10_M_worst:.2f}) "
                  f"clears the requirement (10^{log10_threshold:.2f}) with "
                  f"{log10_M_worst - log10_threshold:.1f} digits of margin.")
            print("Even a pessimistically-drawn subset should certify completeness")
            print(f"up to naive x-height 10^{target_naive_height_digits}.")
        else:
            print(f"*** FAIL *** worst-case subset modulus (10^{log10_M_worst:.2f}) "
                  f"does NOT clear the requirement (10^{log10_threshold:.2f}).")
            print(f"Shortfall: {log10_threshold - log10_M_worst:.1f} digits.")
            if recommended_min_subset_size is not None:
                print(f"RECOMMENDATION: raise MIN_PRIME_SUBSET_SIZE to >= "
                      f"{recommended_min_subset_size} (currently {min_subset_size}), "
                      f"and raise MIN_MAX_PRIME_SUBSET_SIZE to stay >= that.")
            else:
                print("RECOMMENDATION: the pool is too small even using every prime "
                      "in it as one subset -- widen PRIME_POOL (e.g. primes(N) for "
                      "larger N) as well as raising MIN_PRIME_SUBSET_SIZE.")
        print("-" * 70)
        if max_modulus_ok:
            print(f"MAX_MODULUS = 10^{log10_max_modulus:.0f} has headroom above the "
                  f"best-case subset modulus (10^{log10_M_best:.2f}) -- CRT lifts "
                  "won't be silently dropped by the `M > MAX_MODULUS` runtime gate.")
        else:
            print(f"*** MAX_MODULUS = 10^{log10_max_modulus:.0f} is too close to "
                  f"or below the best-case subset modulus (10^{log10_M_best:.2f}) ***")
            print("search_lll/modularthread.py silently skips any CRT lift with")
            print("M > MAX_MODULUS, regardless of whether the completeness proof")
            print("would otherwise pass. RECOMMENDATION: raise MAX_MODULUS well")
            print(f"above 10^{log10_M_best:.2f} (e.g. 10^{int(log10_M_best) + 50}).")
        print("=" * 70)

    return {
        'pass': passes,
        'max_modulus_ok': max_modulus_ok,
        'log10_M_worst': log10_M_worst,
        'log10_M_best': log10_M_best,
        'log10_threshold': log10_threshold,
        'recommended_min_subset_size': recommended_min_subset_size,
        'worst_case_primes_used': worst_primes,
    }

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

def m_is_locally_allowed(m, precomputed_residues, prime_pool, v_tuple=None):
    """
    Given a rational m (QQ-coercible), test whether for each prime in prime_pool we
    can find that m (mod p) among precomputed residues (for some vector if v_tuple None).
    If any prime with numeric data rules out the residue, we report it as 'locally blocked'.

    Returns:
      (allowed_bool, details)
    where details is a dict with per-prime status values:
      {'p': {'residue': r or None, 'status': 'matched'|'unseen'|'denom_zero'|'no_data'}}
    Implementation re-uses the same expectations on precomputed_residues as search_lll.py.
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

def blocking_primes_for_m(m, precomputed_residues, prime_pool, v_tuple=None):
    """
    Returns list of primes that would block m (i.e., have status 'unseen' in m_is_locally_allowed).
    """
    allowed, details = m_is_locally_allowed(m, precomputed_residues, prime_pool, v_tuple=v_tuple)
    blocked = [p for p, d in details.items() if d['status'] == 'unseen']
    return blocked, details

# ---------------------------
# Heuristic "algebraic Brauer" probe

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

def compute_ramification_locus(cd, verbose=False):
    """
    Compute the ramification locus for an elliptic fibration:
    - primes dividing denominators of a4, a6 (in their coefficients)
    - primes where the Weierstrass discriminant Δ(m) has repeated roots mod p
    - primes where gcd(Δ, Δ') > 1 (fiber collisions)
    """

    ram_locus = set()

    # ------------------------------------------------------------
    # 1. primes dividing denominators of a4, a6
    # ------------------------------------------------------------
    def add_denominator_primes(x):
        # x is a rational function in m (element of QQ(m)) or polynomial in QQ[m]
        # We want primes dividing the denominators of the COEFFICIENTS of x.

        polys_to_check = []
        if hasattr(x, "numerator"):
            polys_to_check.append(x.numerator())
            polys_to_check.append(x.denominator())
        else:
            polys_to_check.append(x)

        for poly in polys_to_check:
            # If it's a polynomial, iterate coefficients. If scalar, check directly.
            if hasattr(poly, "coefficients"):
                coeffs = poly.coefficients()
            else:
                coeffs = [poly]

            for c in coeffs:
                # c is expected to be in QQ
                try:
                    val = QQ(c)
                    denom = val.denominator()
                    if denom != 1:
                        for p, _ in Integer(denom).factor():
                            ram_locus.add(int(p))
                except Exception:
                    pass

    add_denominator_primes(cd.a4)
    add_denominator_primes(cd.a6)

    # ------------------------------------------------------------
    # 2. build discriminant Δ(m) = -16 (4 a4^3 + 27 a6^2)
    #    as a polynomial in QQ[m]
    # ------------------------------------------------------------
    Delta = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)

    PRQ = PolynomialRing(QQ, 'm')
    PRZ = PolynomialRing(ZZ, 'm')

    # rational_fct.numerator() if available (Sage rational function)
    if hasattr(Delta, "numerator"):
        Delta_rat = Delta.numerator()
    else:
        Delta_rat = Delta

    try:
        Delta_Q = PRQ(Delta_rat)
    except Exception:
        if verbose:
            print("[ram_locus] Failed to coerce Δ into QQ[m]; skipping.")
        # Don't raise, just return what we have so far (denom primes)
        return ram_locus

    if Delta_Q.degree() <= 0:
        if verbose:
            print("[ram_locus] Δ is constant. Ram locus = denom primes only.")
        return ram_locus

    # ------------------------------------------------------------
    # 3. clear denominators → ZZ[m]
    # ------------------------------------------------------------
    coeffs = Delta_Q.coefficients()
    if not coeffs:
        if verbose:
            print("[ram_locus] Δ has no coefficients—degenerate.")
        return ram_locus

    denoms = [c.denominator() if hasattr(c, "denominator") else 1 for c in coeffs]
    common_den = int(lcm(denoms))

    # Δ_Z = common_den * Δ_Q  converted to ZZ[m]
    Delta_Z = PRZ((Delta_Q * common_den).change_ring(ZZ))

    # ------------------------------------------------------------
    # 4. extract content (integer factor) and primitive part
    # ------------------------------------------------------------
    content_int = Integer(Delta_Z.content())

    # Add primes from the content (and the cleared denominators)
    dencont = content_int * common_den
    if dencont != 0:
        for p, _ in Integer(dencont).factor():
            ram_locus.add(int(p))

    if content_int == 0:
         if verbose:
            print("[ram_locus] Δ becomes 0 polynomial after clearing denominators.")
         return ram_locus

    Delta_prim = Delta_Z // content_int

    if Delta_prim.degree() <= 0:
        if verbose:
            print("[ram_locus] primitive Δ is constant—no discriminant primes.")
        return ram_locus

    # ------------------------------------------------------------
    # 5. Compute gcd(Δ, Δ') to find repeated roots
    #    This catches fiber collisions!
    # ------------------------------------------------------------
    dDelta = Delta_prim.derivative()
    g = gcd(Delta_prim, dDelta)

    if g.degree() > 0:
        # g contains the repeated root factors
        if verbose:
            print(f"[ram_locus] Δ has repeated factors (gcd degree {g.degree()}).")

        # Factor g to extract primes where roots collide
        # First check if it's small enough to factor
        try:
            # Try to get integer content from g
            g_content = Integer(g.content())
            if g_content != 0 and g_content.abs().ndigits() < 30:
                for p, _ in g_content.factor():
                    ram_locus.add(int(p))
                    if verbose:
                        print(f"[ram_locus] Added collision prime from gcd content: {p}")
        except Exception:
            pass

        # Also try small primes by evaluation
        small_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
        for p in small_primes:
            try:
                # Check if g(m) ≡ 0 (mod p) for some m
                Fp = GF(p)
                PRp = PolynomialRing(Fp, 'm')
                g_modp = PRp([Fp(c) for c in g.coefficients(sparse=False)])

                if g_modp.degree() > 0:
                    roots_modp = g_modp.roots(multiplicities=False)
                    if roots_modp:
                        ram_locus.add(p)
                        if verbose:
                            print(f"[ram_locus] Added collision prime from gcd roots mod p: {p}")
            except Exception:
                pass

    # ------------------------------------------------------------
    # 6. Use discriminant of Δ for additional primes (optional, more comprehensive)
    # ------------------------------------------------------------
    try:
        disc_Delta = Delta_prim.discriminant()
        if disc_Delta != 0:
            disc_int = Integer(disc_Delta)
            if disc_int.abs().ndigits() < 100:  # Only if manageable size
                for p, _ in disc_int.factor():
                    ram_locus.add(int(p))
                    if verbose:
                        print(f"[ram_locus] Added prime from disc(Δ): {p}")
    except Exception:
        if verbose:
            print("[ram_locus] Could not compute discriminant of Δ")

    # ------------------------------------------------------------
    # 7. Explicitly scan small primes for mod-p collisions
    #    (Fixes AssertionError where search finds collisions missed by global analysis)
    # ------------------------------------------------------------
    # We check the primes typically used in the pool to ensure consistency with search_lll
    small_primes_scan = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113]

    for p in small_primes_scan:
        try:
            # If Delta_prim vanishes or drops degree significantly, it's ramified
            # (Use leading_coefficient() to be safe across Sage versions)
            lc = Delta_prim.leading_coefficient()
            if lc % p == 0:
                ram_locus.add(int(p))
                continue

            R_p = PolynomialRing(GF(p), 'm')
            Delta_p = R_p(Delta_prim)

            # Check for repeated roots: discriminant == 0 mod p
            if Delta_p.discriminant() == 0:
                ram_locus.add(int(p))
                if verbose:
                    print(f"[ram_locus] Added collision prime {p} (mod-p discriminant is 0)")
        except Exception:
            pass

    if verbose:
        print(f"[ram_locus] Final ramification locus: {sorted(ram_locus)}")

    return ram_locus
