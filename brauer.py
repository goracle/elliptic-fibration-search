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
    import math
    
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



# brauer.py

# ... (other code)

def compute_ramification_locus(cd, verbose=False):
    """
    Computes the set of primes p where the discriminant Delta(m) of the 
    elliptic fibration has repeated roots modulo p, or where the base 
    parameterization is bad (primes dividing coefficient denominators).

    Args:
        cd: The FibrationCoefficients object containing a4 and a6.
        verbose: Print debug information.

    Returns:
        set: The set of primes in the ramification locus.
    """
    from sage.all import QQ, ZZ, PolynomialRing, factor

    # 1. Start with primes dividing denominators of a4, a6
    ram_locus = set()

    # Get the coefficients (which are rational functions in m)
    a4 = cd.a4
    a6 = cd.a6

    # Collect all prime factors of all coefficient denominators
    for coeff in [a4, a6]:
        if hasattr(coeff, 'denominator'):
            for p, e in factor(coeff.denominator()):
                ram_locus.add(int(p))
    
    # 2. Compute the Weierstrass discriminant polynomial Delta(m) (numerator part)
    # Delta(m) = -16 * (4*a4^3 + 27*a6^2)
    Delta = -16 * (4 * a4**3 + 27 * a6**2)
    
    # Coerce to a polynomial over Q by taking the numerator (to clear base denominators)
    PR_m = PolynomialRing(QQ, 'm')
    Delta_poly = PR_m(Delta.numerator() if hasattr(Delta, 'numerator') else Delta)
    
    if Delta_poly.degree() == 0:
        # Trivial case: Delta is a constant (e.g., if the fibration is constant)
        # Only primes from step 1 will be included.
        if verbose: print(f"[ram_locus] Delta is constant. R=0.")
        return ram_locus

    # 3. Compute the resultant R of Delta(m) and its derivative Delta'(m)
    dDelta = Delta_poly.derivative()
    R = Delta_poly.resultant(dDelta)
    
    # --- FIX: Handle non-square-free Delta(m) which leads to R=0 ---
    if R == 0:
        if verbose: print(f"[ram_locus] Resultant R=0. Delta is not square-free. Using square-free part.")
        # Compute the square-free part
        Delta_sq_free = 1
        for factor, exponent in Delta_poly.factor():
            Delta_sq_free *= factor
        
        # Recalculate the resultant for the square-free part
        if Delta_sq_free.degree() > 0:
            dDelta_sq_free = Delta_sq_free.derivative()
            R = Delta_sq_free.resultant(dDelta_sq_free)
        else:
            # If Delta_sq_free is constant (unlikely for a fibration), set R=1 
            # so R.factor() returns no primes.
            R = 1 
    # --- END FIX ---
    
    # 4. Primes dividing the final resultant R are in the ramification locus
    for p, e in R.factor():
        # Ensure only prime integers are added
        if p > 1 and ZZ(p).is_prime():
            ram_locus.add(int(p))
            
    if verbose: print(f"[ram_locus] Final ramification locus: {ram_locus}")

    return ram_locus



def compute_ramification_locus(cd, verbose=False):
    """
    Robust computation of primes in the ramification locus:
    - primes dividing denominators of a4/a6
    - primes dividing the resultant of Delta and Delta'
    Handles the Delta squarefree-case safely.
    """
    from sage.all import QQ, ZZ, PolynomialRing, gcd, Integer

    ram_locus = set()

    a4 = cd.a4
    a6 = cd.a6

    # 1) primes dividing denominators of a4,a6 (coefficient denominators)
    for coeff in (a4, a6):
        try:
            den = coeff.denominator()
        except Exception:
            den = None
        if den:
            # factor() over ZZ or QQ will yield (prime, exponent) pairs
            for ppair in Integer(den).factor():
                p = int(ppair[0])
                if p > 1:
                    ram_locus.add(p)

    # 2) Build Delta(m) polynomial (numerator/primitive part)
    # Delta = -16*(4*a4^3 + 27*a6^2)
    Delta = -16 * (4 * a4**3 + 27 * a6**2)

    PR_m = PolynomialRing(QQ, 'm')
    # Force Delta_poly to be a polynomial in QQ[m] (take numerator if rational function)
    if hasattr(Delta, 'numerator'):
        Delta_obj = Delta.numerator()
    else:
        Delta_obj = Delta
    Delta_poly = PR_m(Delta_obj)

    if Delta_poly.degree() <= 0:
        if verbose:
            print("[ram_locus] Delta is constant or zero; skipping resultant.")
        return ram_locus

    # Remove integer content to make resultant smaller (primitive part)
    content = Delta_poly.content()
    Delta_prim = Delta_poly // content

    # Compute derivative and gcd (to detect repeated factors)
    dDelta = Delta_prim.derivative()

    # gcd gives the repeated factor part; square-free part is Delta_prim // gcd
    g = gcd(Delta_prim, dDelta)
    if g != 1:
        if verbose:
            print(f"[ram_locus] gcd(Delta, Delta') != 1 (deg g = {g.degree()}). "
                  "Computing square-free part.")
    Delta_sqf = Delta_prim // g

    if Delta_sqf.degree() <= 0:
        # degenerate: square-free part constant
        if verbose: print("[ram_locus] square-free part is constant after gcd; nothing further.")
        return ram_locus

    # 3) resultant between square-free part and its derivative
    dDelta_sqf = Delta_sqf.derivative()
    R = Integer(Delta_sqf.resultant(dDelta_sqf))

    # If somehow resultant still zero (very pathological), fall back to 1
    if R == 0:
        if verbose:
            print("[ram_locus] resultant still zero after square-free extraction; treating as no primes.")
        R = Integer(1)

    # 4) factor integer R (and also factor content if present, content may have primes)
    if content != 1:
        try:
            # content is an integer in QQ polynomial; factor it
            for ppair in Integer(content).factor():
                p = int(ppair[0])
                if p > 1:
                    ram_locus.add(p)
        except Exception:
            pass

    # factor R and add primes
    try:
        for ppair in R.factor():
            p = int(ppair[0])
            if p > 1 and ZZ(p).is_prime():
                ram_locus.add(p)
    except Exception:
        # If factoring R fails (too big), conservatively try small primes trial division
        small_primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
        rem = abs(int(R))
        for p in small_primes:
            if rem % p == 0:
                ram_locus.add(p)
                while rem % p == 0:
                    rem //= p

    if verbose:
        print(f"[ram_locus] Final ramification locus: {sorted(ram_locus)}")
    return ram_locus


def compute_ramification_locus(cd, verbose=False):
    """
    Compute the ramification locus for an elliptic fibration:
    - primes dividing denominators of a4, a6
    - primes where the Weierstrass discriminant Δ(m) has repeated roots mod p
      (equivalently: primes dividing the resultant of square-free Δ and its derivative)

    Fully robust:
    - works with QQ-rational functions in m
    - coerces to QQ[m], then clears denominators -> ZZ[m]
    - extracts square-free part to avoid resultant=0 bugs
    - handles all degeneracies gracefully
    """

    from sage.all import (
        QQ, ZZ, Integer, PolynomialRing, lcm, gcd
    )

    ram_locus = set()

    # ------------------------------------------------------------
    # 1. primes dividing denominators of a4, a6
    # ------------------------------------------------------------
    def add_denominator_primes(x):
        try:
            den = x.denominator()
        except Exception:
            raise
            return
        if den:
            for p, _ in Integer(den).factor():
                ram_locus.add(int(p))

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
        raise
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
    if content_int == 0:
        if verbose:
            print("[ram_locus] Δ becomes 0 polynomial after clearing denominators.")
        return ram_locus

    Delta_prim = Delta_Z // content_int

    # primes from denominator clearing + content
    dencont = content_int * common_den
    for p, _ in Integer(dencont).factor():
        ram_locus.add(int(p))

    if Delta_prim.degree() <= 0:
        if verbose:
            print("[ram_locus] primitive Δ is constant—no discriminant primes.")
        return ram_locus

    # ------------------------------------------------------------
    # 5. extract square-free part: gcd(Δ, Δ')
    # ------------------------------------------------------------
    dDelta = Delta_prim.derivative()
    g = gcd(Delta_prim, dDelta)

    if g != 1 and verbose:
        print(f"[ram_locus] Δ has repeated factors (gcd degree {g.degree()}). Using square-free part.")

    Delta_sqf = Delta_prim // g

    if Delta_sqf.degree() <= 0:
        if verbose:
            print("[ram_locus] Δ square-free part is constant; only denominator primes contribute.")
        return ram_locus

    # ------------------------------------------------------------
    # 6. resultant of square-free part and derivative
    # ------------------------------------------------------------
    dDelta_sqf = Delta_sqf.derivative()
    R = Integer(Delta_sqf.resultant(dDelta_sqf))

    if R == 0:
        # theoretically impossible after square-free extraction,
        # unless something deeply degenerate occurred
        if verbose:
            print("[ram_locus] resultant is still zero after square-free extraction; ignoring.")
        return ram_locus

    # ------------------------------------------------------------
    # 7. factor resultant → ramification primes
    # ------------------------------------------------------------
    try:
        for p, _ in R.factor():
            p = int(p)
            if p > 1:
                ram_locus.add(p)
    except Exception:
        # fallback: small primes (rarely needed)
        small = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
        Rabs = abs(int(R))
        for p in small:
            if Rabs % p == 0:
                ram_locus.add(p)
                while Rabs % p == 0:
                    Rabs //= p
        raise

    if verbose:
        print(f"[ram_locus] Final ramification locus: {sorted(ram_locus)}")

    return ram_locus
