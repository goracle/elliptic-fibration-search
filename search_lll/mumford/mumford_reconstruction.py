from functools import lru_cache
from sage.all import QQ, ZZ
#from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from ..rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .mumford_verification import verify_mumford_pair, canonicalize_and_dedup, discriminant_has_nonqr_s_p
from itertools import product, islice
from .mumford_timing import mumford_timer_add

import time

def reconstruct_and_verify_mumford(mumford_residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Compatible with branch-mode mumford_residues produced by
    mumford_precompute_residues_parallel (branch-format).

    Input format (mumford_residues):
      mumford_residues[p][v_tuple][x_res] -> list of entries
      where each entry is (r_mod_p, v_at_r_mod_p, s_mod_p, p_mod_p)
      - r_mod_p may be int in 0..p-1 or None (irreducible local u)
      - v_at_r_mod_p may be int or None
      - s_mod_p, p_mod_p are included for bookkeeping

    Output:
      found_xs, mumford_divisors (same as previous API)
    """
    t0 = time.time()
    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE (branch-mode)")
    print("="*70)

    mumford_timer_add("crt_reconstruction_loop", 0.0)  # ensure timer exists

    total_stats = {
        'attempted': 0,
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0,
        'prefilter_reject': 0,
        'skipped_2prime': 0,
        'skipped_high_density': 0
    }

    # results to return
    found_xs = set()
    mumford_divisors_raw = []

    # Basic guards
    if not mumford_residues:
        if debug:
            print("No mumford_residues provided (empty).")
        mumford_timer_add("crt_reconstruction_loop", time.time() - t0)
        return found_xs, []

    # iterate over vectors (v_tuple) present in residues
    for p in sorted(mumford_residues.keys()):
        # just to ensure primes are consistent
        if p not in prime_list:
            prime_list.append(p)
    prime_list = sorted(set(prime_list))

    # Build per-vector aggregated structure:
    # for each v_tuple we need an ordered list of primes that provide data and per-prime assigned residues
    for v_tuple in sorted({vt for pmap in mumford_residues.values() for vt in pmap.keys()}):
        # gather per-prime entries for this v_tuple
        primes_with_data = []
        perprime_entries = {}
        for p in prime_list:
            pmap = mumford_residues.get(p, {})
            if v_tuple not in pmap:
                continue
            xres_map = pmap[v_tuple]
            # xres_map: x_res -> list of entries
            # collect all r values and v_at per prime (flatten)
            r_vals = []
            v_at_map = {}  # r_val -> list of v_at values (take first if multiple)
            paired_sp = []  # store s_mod,p_mod occurrences for diagnostics
            for x_res_key, sols in xres_map.items():
                for entry in sols:
                    # entry is (r_mod_p, v_at_r_mod_p, s_mod_p, p_mod_p)
                    r_mod_p, v_at_r_mod_p, s_mod_p, p_mod_p = entry
                    paired_sp.append((s_mod_p, p_mod_p))
                    if r_mod_p is None:
                        # irreducible local u - record as special marker (no root in Fp)
                        continue
                    # add the residue
                    r_vals.append(int(r_mod_p))
                    # prefer first seen v_at
                    if int(r_mod_p) not in v_at_map:
                        v_at_map[int(r_mod_p)] = int(v_at_r_mod_p) if v_at_r_mod_p is not None else None

            if r_vals:
                r_vals = sorted(set(r_vals))
                # deterministic assignment: smaller -> root A, larger -> root B (if available)
                if len(r_vals) == 1:
                    rA = r_vals[0]
                    rB = None
                else:
                    rA = r_vals[0]
                    rB = r_vals[-1]
                perprime_entries[p] = {
                    'rA': rA,
                    'vA': v_at_map.get(rA, None),
                    'rB': rB,
                    'vB': v_at_map.get(rB, None),
                    'sps': paired_sp  # bookkeeping
                }
                primes_with_data.append(p)
            else:
                # if no r-values but paired_sp contains an irreducible marker, keep that info
                if paired_sp:
                    perprime_entries[p] = {
                        'rA': None,
                        'vA': None,
                        'rB': None,
                        'vB': None,
                        'sps': paired_sp
                    }
                    primes_with_data.append(p)
                # else no data for this v_tuple at this prime

        if not primes_with_data:
            continue

        # Deterministic prime ordering for CRT
        primes_order = sorted(primes_with_data)

        # Use the deterministic assignment above: reconstruct rA using all primes where rA not None,
        # reconstruct rB using all primes where rB not None.
        primes_for_A = [p for p in primes_order if perprime_entries[p]['rA'] is not None]
        primes_for_B = [p for p in primes_order if perprime_entries[p]['rB'] is not None]

        # Minimal size checks: need at least 2 primes to attempt CRT->rational (heuristic)
        if len(primes_for_A) < 2 and len(primes_for_B) < 2:
            # not enough information to reconstruct two rational roots; skip
            continue

        # Helper to attempt CRT+rational reconstruction for a list of primes and corresponding residues
        def try_reconstruct_root(prime_subset, residue_getter):
            """
            prime_subset: list of primes (ints)
            residue_getter: function(p) -> residue_mod_p (int 0..p-1)
            returns QQ rational or raises RationalReconstructionError / returns None on failure
            """
            if not prime_subset:
                return None
            M = 1
            vals = []
            for p in prime_subset:
                M *= int(p)
            for p in prime_subset:
                val_mod = int(residue_getter(p)) % int(p)
                vals.append(int(val_mod))
            # compute CRT integer
            crt_val = crt_cached(tuple(vals), tuple(prime_subset))
            try:
                num, den = rational_reconstruct(crt_val, M)
            except Exception:
                # reconstruction failed -> no rational from this subset
                return None
            # height sanity: keep as-is (do not over-filter here)
            return QQ(num) / QQ(den)

        # Attempt reconstruction for root A and root B
        rA_q = None
        rB_q = None
        # reconstruct rA
        if primes_for_A:
            def get_resA(p): return perprime_entries[p]['rA']
            rA_q = try_reconstruct_root(primes_for_A, get_resA)

        # reconstruct rB
        if primes_for_B:
            def get_resB(p): return perprime_entries[p]['rB']
            rB_q = try_reconstruct_root(primes_for_B, get_resB)

        # If one of the roots failed to reconstruct, we can still try a mixed strategy:
        # if rA_q exists and rB_q is None, attempt to reconstruct rB from primes where rB exists but not used in A
        if rA_q is None and rB_q is not None:
            # swap roles so rA_q becomes rB_q and vice versa for downstream consistency
            rA_q, rB_q = rB_q, rA_q

        if rA_q is None or rB_q is None:
            # try a fallback: split primes_order into two halves and reconstruct from halves
            k = len(primes_order) // 2
            if k >= 1:
                left = primes_order[:k]
                right = primes_order[k:]
                if rA_q is None:
                    # try reconstruct from left using rA residues if present, else try using rB residues
                    def get_res_left(p):
                        ent = perprime_entries.get(p, {})
                        return ent.get('rA') if ent.get('rA') is not None else ent.get('rB')
                    rA_q = try_reconstruct_root(left, get_res_left)
                if rB_q is None:
                    def get_res_right(p):
                        ent = perprime_entries.get(p, {})
                        return ent.get('rB') if ent.get('rB') is not None else ent.get('rA')
                    rB_q = try_reconstruct_root(right, get_res_right)

        # If still missing either root, skip (can't form quadratic)
        if rA_q is None or rB_q is None:
            continue

        # If reconstructed roots are equal, skip (double root; user can adjust if they want these)
        if rA_q == rB_q:
            continue

        # Reconstruct v_at_r1 and v_at_r2 using the same primes used for their roots (prefer matching sets)
        def try_reconstruct_v_at_root(prime_subset, v_getter):
            if not prime_subset:
                return None
            M = 1
            vals = []
            for p in prime_subset:
                M *= int(p)
            for p in prime_subset:
                vmod = v_getter(p)
                if vmod is None:
                    # cannot reconstruct v_at with missing residues at this prime -> fail
                    return None
                vals.append(int(vmod) % int(p))
            crt_val = crt_cached(tuple(vals), tuple(prime_subset))
            try:
                num, den = rational_reconstruct(crt_val, M)
            except Exception:
                return None
            return QQ(num) / QQ(den)

        vA_q = try_reconstruct_v_at_root(primes_for_A, lambda p: perprime_entries[p]['vA'] if p in perprime_entries else None)
        vB_q = try_reconstruct_v_at_root(primes_for_B, lambda p: perprime_entries[p]['vB'] if p in perprime_entries else None)

        # If v reconstructions fail, try cross-use primes: attempt to use primes_for_A for both or primes_for_B for both
        if vA_q is None:
            vA_q = try_reconstruct_v_at_root(primes_for_B, lambda p: perprime_entries[p]['vA'] if p in perprime_entries else None)
        if vB_q is None:
            vB_q = try_reconstruct_v_at_root(primes_for_A, lambda p: perprime_entries[p]['vB'] if p in perprime_entries else None)

        # if still missing, we cannot determine v coefficients; skip
        if vA_q is None or vB_q is None:
            continue

        # Now compute s and p and solve for v1,v0
        s_q = rA_q + rB_q
        p_q = rA_q * rB_q

        # solve linear system:
        # v1 * rA + v0 = vA_q
        # v1 * rB + v0 = vB_q
        denom = (rA_q - rB_q)
        if denom == 0:
            continue
        v1_q = (vA_q - vB_q) / denom
        v0_q = vA_q - v1_q * rA_q

        # Convert to QQ explicitly if not already
        s_q = QQ(s_q)
        p_q = QQ(p_q)
        v0_q = QQ(v0_q)
        v1_q = QQ(v1_q)

        total_stats['attempted'] += 1

        # Algebraic verification
        ok = verify_mumford_pair(f_coeffs, s_q, p_q, v0_q, v1_q, modulus=None, debug_first_failure=False)
        if not ok:
            total_stats['algebraic_reject'] += 1
            continue

        # Append in same record shape as before
        mumford_divisors_raw.append({
            'vector': v_tuple,
            's': s_q,
            'p': p_q,
            'v_0': v0_q,
            'v_1': v1_q
        })
        total_stats['success'] += 1

    # End per-vector loop

    # Summary & existing canonicalization / dedup pipeline
    mumford_timer_add("crt_reconstruction_loop", time.time() - t0)

    print(f"\n=== RECONSTRUCTION SUMMARY ===")
    print(f"  Attempted candidates: {total_stats['attempted']:,}")
    print(f"  Rejected by algebraic constraint: {total_stats['algebraic_reject']:,}")
    print(f"  Successful reconstructions: {total_stats['success']:,}")

    if not mumford_divisors_raw:
        print("  WARNING: No valid Mumford divisors reconstructed!")
        mumford_timers_print()
        return set(), []

    # canonicalize & dedup (use your existing helper)
    t0b = time.time()
    mumford_divisors_raw = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)
    mumford_timer_add("canonicalization", time.time() - t0b)

    # Now filter dependent divisors (same logic as previously)
    mumford_divisors = []
    for i, divi in enumerate(mumford_divisors_raw):
        is_dep = False
        for j, divj in enumerate(mumford_divisors_raw):
            if i <= j:
                continue
            if quick_dependence_check(divi, divj):
                is_dep = True
                break
        if not is_dep:
            mumford_divisors.append(divi)

    # Rational root check (same as before)
    t0c = time.time()
    for div in mumford_divisors:
        s = div['s']
        p_val = div['p']
        disc = s*s - 4*p_val
        if disc in QQ and disc >= 0 and disc.is_square():
            div['has_rational_roots'] = True
            r1 = (s + disc.sqrt())/2
            r2 = (s - disc.sqrt())/2
            for r in (r1, r2):
                x_cand = r - shift
                if rationality_test(x_cand) is not None:
                    found_xs.add(x_cand)
        else:
            div['has_rational_roots'] = False
    mumford_timer_add("rational_root_check", time.time() - t0c)

    print(f"  Unique Rational Points: {len(found_xs)}")

    # final sort / unique like original
    if mumford_divisors:
        unique = {frozenset(d.items()): d for d in mumford_divisors}
        mumford_divisors = list(unique.values())

        def naive_sort_key(d):
            return abs(QQ(d['s'])) + abs(QQ(d['p'])) + abs(QQ(d['v_0'])) + abs(QQ(d['v_1']))

        mumford_divisors.sort(key=naive_sort_key)
        mumford_divisors.reverse()

    return found_xs, mumford_divisors

def reconstruct_mumford_combo_fast(sol_combo, primes, M, max_height):
    """
    Fast reconstruction of a single combination with early rejection.
    
    Returns (s, p, v0, v1) or raises RationalReconstructionError early.
    """
    rec_vals = []
    
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        
        # Reconstruct with BOTH numerator and denominator bounds
        #num, den = rational_reconstruct(crt_val, M, max_den=max_height)
        num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
        
        rec_vals.append(QQ(num) / QQ(den))
    
    return rec_vals


def rational_reconstruct_fast(c, N, max_den=None, max_num=None):
    """
    Fast rational reconstruction with early height rejection.
    
    Fails fast if height bounds will be violated, avoiding unnecessary computation.
    Returns (num, den) or raises RationalReconstructionError.
    
    Args:
        c: integer to reconstruct
        N: modulus
        max_den: maximum allowed denominator (default: floor(sqrt(N/2)))
        max_num: maximum allowed numerator (if None, no numerator check)
    """
    if max_den is None:
        max_den = floor(sqrt(N / QQ(2)))
    
    c = c % N
    if c == 0:
        return 0, 1
    if c == 1 and max_den >= 1:
        return 1, 1
    
    # Early rejection: if c is too large, it's likely to produce large numerators
    # This is a heuristic but catches many cases
    if max_num is not None and c > max_num and (N - c) > max_num:
        raise RationalReconstructionError(f"CRT value too large: c={c}")
    
    # Extended Euclidean Algorithm with early termination
    r0, r1 = N, c
    t0, t1 = 0, 1
    
    while r1 != 0:
        # Check denominator bound BEFORE next iteration
        if abs(t1) > max_den:
            a, b = r0, t0
            break
        
        # Early numerator check during iteration
        if max_num is not None and abs(r1) > max_num:
            raise RationalReconstructionError(f"Numerator exceeds bound: {abs(r1)} > {max_num}")
        
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    else:
        # Loop finished because r1 == 0
        a, b = r0, t0
    
    # Final validation
    if abs(b) > max_den or b == 0:
        raise RationalReconstructionError(f"No reconstruction for c={c}, N={N}, max_den={max_den}")
    
    if b < 0:
        a, b = -a, -b
    
    # Final numerator check
    if max_num is not None and abs(a) > max_num:
        raise RationalReconstructionError(f"Numerator exceeds bound: {abs(a)} > {max_num}")
    
    if (a - c * b) % N != 0:
        raise RationalReconstructionError(f"Validation failed for c={c}, N={N}: got a={a}, b={b}")
    
    g = gcd(abs(a), abs(b))
    return int(a // g), int(b // g)


@lru_cache
def rational_reconstruct_with_height_check(crt_val, M, max_height):
    """
    Rational reconstruction with immediate height rejection.
    Returns (num, den) or raises RationalReconstructionError.
    """
    # Use standard denominator bound for reconstruction
    max_den = floor(sqrt(M / QQ(2)))
    num, den = rational_reconstruct(crt_val, M, max_den=max_den)
    
    # Then check BOTH against the actual height limit
    if abs(num) > max_height or abs(den) > max_height:
        raise RationalReconstructionError("Height too large")
    
    return num, den

def setup_crt_constants(primes):
    """
    Precompute weights for fast CRT: result = sum(val_i * w_i) % M.
    Returns (M, weights).
    """
    M = 1
    for p in primes:
        M *= p
    
    weights = []
    for p in primes:
        m_i = M // p
        # inverse of m_i mod p
        # use python int pow to ensure ValueError on failure, though m_i is coprime to p by definition
        inv = pow(int(m_i), -1, int(p))
        w_i = (m_i * inv)
        weights.append(w_i)
        
    return M, weights

def fast_rational_reconstruct_check(val, M, max_height):
    """
    Pure integer rational reconstruction check. 
    Returns (True, num, den) or (False, 0, 0).
    Optimized for tight loops: avoids object creation and returns early.
    """
    r0, r1 = M, val
    t0, t1 = 0, 1
    
    # Unrolled Euclidean Algorithm
    while r1 > max_height:
        if t1 > max_height or t1 < -max_height:
            return False, 0, 0
            
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
        
    if abs(t1) > max_height:
        return False, 0, 0
        
    if t1 < 0:
        t1 = -t1
        r1 = -r1
        
    if abs(r1) > max_height:
        return False, 0, 0
        
    return True, r1, t1

def batch_crt_for_combo(sol_combo, primes):
    """
    Compute CRT for all 4 coordinates at once.
    Returns list of 4 CRT values.
    """
    crt_vals = []
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        crt_vals.append(crt_val)
    return crt_vals

def prefilter_solutions_by_discriminant(sol_lists, primes):
    """
    Filter solution combinations that can't possibly satisfy s^2 - 4p >= 0 mod all primes.
    This eliminates many impossible combinations early.
    
    Returns: filtered generator of solution combinations
    """
    for sol_combo in product(*sol_lists):
        # Quick discriminant check mod each prime
        all_good = True
        for i, p in enumerate(primes):
            s_mod = sol_combo[i][0] % p
            p_mod = sol_combo[i][1] % p
            disc_mod = (s_mod * s_mod - 4 * p_mod) % p
            
            # If discriminant is negative mod p and p > 2, skip
            # (This is a quick heuristic, not perfect)
            if p > 2 and disc_mod != 0:
                # Check if disc_mod is a quadratic residue
                if pow(disc_mod, (p - 1) // 2, p) == p - 1:
                    all_good = False
                    break
        
        if all_good:
            yield sol_combo


#
