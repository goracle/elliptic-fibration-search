"""
search_analysis.py: Statistical analysis, auto-tuning, and diagnostics.
"""
from .search_config import (
    DEBUG, EXTRA_PRIME_MIN_R, EXTRA_PRIME_MAX_R, ROOTS_THRESHOLD,
    EXTRA_PRIME_TARGET_DENSITY, EXTRA_PRIME_MAX, EXTRA_PRIME_SKIP,
    TMAX, Counter
)
from stats import * # Project-level import
from itertools import combinations
from sage.all import *
from search_common import MIN_PRIME_SUBSET_SIZE, MIN_MAX_PRIME_SUBSET_SIZE

def estimate_prime_stats(prime_pool, precomputed_residues, sample_vecs, num_rhs=1):
    """Estimate average residue survival ratio r_p for each prime."""
    stats = {}
    for p in prime_pool:
        mapping = precomputed_residues.get(p, {})
        if not mapping:
            continue
        total = count = 0
        for v in sample_vecs:
            v_t = tuple(v)
            roots_list = mapping.get(v_t, [])
            if not roots_list:
                continue
            # combine across RHSs
            if num_rhs > 1:
                roots_union = set().union(*roots_list)
            else:
                roots_union = roots_list[0] if roots_list else set()
            total += len(roots_union)
            count += p
        stats[p] = (total / count) if count else 0.0
    return stats

def choose_extra_primes(stats, target_density=EXTRA_PRIME_TARGET_DENSITY, max_extra=EXTRA_PRIME_MAX, skip_small=EXTRA_PRIME_SKIP):
    """Select extra primes based on measured r_p values."""
    cand = [(p, r) for p, r in stats.items()
            if p not in skip_small and EXTRA_PRIME_MIN_R < r < EXTRA_PRIME_MAX_R]
    # sort by discriminatory power (entropy-like)
    cand.sort(key=lambda t: -(t[1] * (1 - t[1])))
    chosen, prod = [], 1.0
    for p, r in cand:
        if len(chosen) >= max_extra:
            break
        prod *= r
        chosen.append(p)
        if prod <= target_density:
            break
    if DEBUG:
        print(f"[auto-tune] selected extra primes {chosen} with expected density {prod:.2e}")
    return chosen

def expected_density(residue_sets, subset_size, prime_pool, max_samples=2000):
    """
    Estimate expected survivor density for subsets of given size.
    """
    all_subsets = list(combinations(prime_pool, subset_size))
    if len(all_subsets) > max_samples:
        import random
        all_subsets = random.sample(all_subsets, max_samples)

    densities = []
    for subset in all_subsets:
        d = 1.0
        for p in subset:
            d *= len(residue_sets[p]) / p
        densities.append(d)

    avg_density = sum(densities) / len(densities)
    return avg_density, min(densities), max(densities)

def _assert_rhs_consistency(precomputed_residues, prime_pool, vecs, num_rhs_fns, debug=DEBUG):
    """
    Validate that precomputed_residues has consistent structure.
    """
    errors = []

    # Check: every prime in prime_pool should be in precomputed_residues
    missing_primes = [p for p in prime_pool if p not in precomputed_residues]
    if missing_primes:
        errors.append(f"Missing primes in precomputed_residues: {missing_primes[:5]}{'...' if len(missing_primes) > 5 else ''}")

    # Check: for each prime p that exists, verify structure
    for p in precomputed_residues:
        p_data = precomputed_residues[p]

        if not isinstance(p_data, dict):
            errors.append(f"Prime p={p}: expected dict, got {type(p_data)}")
            continue

        sample_vecs = vecs[:min(5, len(vecs))]
        for v in sample_vecs:
            v_tuple = tuple(v)

            if v_tuple not in p_data:
                continue

            roots_list = p_data[v_tuple]

            if not isinstance(roots_list, (list, tuple)):
                errors.append(f"Prime p={p}, vector {v_tuple[:2]}...: expected list/tuple, got {type(roots_list)}")
                continue

            if len(roots_list) != num_rhs_fns:
                errors.append(
                    f"Prime p={p}, vector {v_tuple[:2]}...: "
                    f"expected {num_rhs_fns} RHS entries, got {len(roots_list)}"
                )
                continue

            for rhs_idx, roots_set in enumerate(roots_list):
                if not isinstance(roots_set, (set, frozenset)):
                    errors.append(
                        f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                        f"expected set, got {type(roots_set)}"
                    )
                    break

                for root in roots_set:
                    if not isinstance(root, (int, Integer)):
                        errors.append(
                            f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                            f"root {root} is not an integer (type {type(root)})"
                        )
                        break
                    if not (0 <= int(root) < p):
                        errors.append(
                            f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                            f"root {root} out of range [0, {p})"
                        )
                        break

    if errors:
        if debug:
            print("\n" + "="*70)
            print("RHS CONSISTENCY CHECK FAILED")
            print("="*70)
            for i, err in enumerate(errors[:10], 1):
                print(f"{i}. {err}")
            if len(errors) > 10:
                print(f"... and {len(errors) - 10} more errors")
            print("="*70 + "\n")

        raise AssertionError(
            f"precomputed_residues structure is malformed. "
            f"Found {len(errors)} error(s). See output above for details."
        )

def _print_subset_productivity_stats(productive, all_subsets):
    """Print quick stats on which prime subsets were productive"""
    total = len(all_subsets)
    productive_count = len(productive)
    total_candidates = sum(p['candidates'] for p in productive)

    print(f"\n[subset stats] {productive_count}/{total} subsets produced candidates "
          f"({100*productive_count/total:.1f}%)")
    print(f"[subset stats] {total_candidates} total candidates from productive subsets")

    by_size = Counter(p['size'] for p in productive)
    all_by_size = Counter(len(s) for s in all_subsets)

    print(f"[subset stats] Productivity by size:")
    for size in sorted(all_by_size.keys()):
        prod_count = by_size.get(size, 0)
        total_count = all_by_size[size]
        rate = 100 * prod_count / total_count if total_count > 0 else 0
        cands = sum(p['candidates'] for p in productive if p['size'] == size)
        print(f"  Size {size}: {prod_count}/{total_count} productive ({rate:.1f}%), "
              f"{cands} candidates")

    top = sorted(productive, key=lambda x: x['candidates'], reverse=True)[:5]
    print(f"[subset stats] Top 5 productive subsets:")
    for p in top:
        print(f"  {p['primes']}: {p['candidates']} candidates")

def _batch_check_rationality(candidates, r_m, shift, rationality_test_func, current_sections, stats):
    """
    Test a batch of (m, v_tuple) candidates for rationality in parallel.
    Returns set of (m, v_tuple) pairs that produced rational points.
    """
    rational_candidates = set()

    for m_val, v_tuple in candidates:
        stats.incr('rationality_tests_total')
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                stats.record_success(m_val, point=x_val)
                rational_candidates.add((m_val, v_tuple))
            else:
                stats.record_failure(m_val, reason='y_not_rational')
        except (TypeError, ZeroDivisionError, ArithmeticError):
            stats.record_failure(m_val, reason='rationality_test_error')
            continue

    return rational_candidates


def diagnose_missed_point(target_x, r_m_callable, shift, precomputed_residues, prime_pool, vecs, tmax=TMAX, debug=True):
    """
    Diagnose why a specific x-value wasn't found by the CRT search.
    
    Check if target_x is theoretically findable via CRT + rational reconstruction
    for any vector and prime subset combination.
    
    Args:
        target_x: The x-coordinate we're looking for (QQ or coercible)
        r_m_callable: Function to compute x from m (typically r_m from tower)
        shift: The shift applied to x-coordinates
        precomputed_residues: {p: {v_tuple: [roots_per_rhs]}} from workers
        prime_pool: List of primes used in search
        vecs: List of search vectors
        tmax: Maximum |t| to check in m = m0 + t*M
        debug: Print diagnostic info
    
    Returns:
        dict with diagnostic information
    """
    from sage.all import QQ, ZZ
    from itertools import combinations
    
    # Step 1: Solve for target m-value
    # x = r_m(m) - shift, so m = r_m^(-1)(x + shift)
    # For r_m(m) = -m - x1, we have: x = -m - x1 - shift
    # So: m = -x - x1 - shift = -(x + shift) - x1
    # But we need to be more careful. Let's solve symbolically.
    
    target_x_q = QQ(target_x)
    
    # For the linear case r_m(m) = -m - const, solve x = -m - const - shift
    # => m = -x - shift - const
    # We can get const by evaluating r_m at m=0
    try:
        const_term = r_m_callable(m=QQ(0))
        #target_m = -(target_x_q + shift + const_term) # wrong lol
        target_m = const_term - target_x_q - shift
    except Exception as e:
        if debug:
            print(f"[diagnose] Failed to compute target_m: {e}")
        return {'error': str(e)}
    
    if debug:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC: Checking if x = {target_x_q} is findable")
        print(f"{'='*70}")
        print(f"Target m-value: {target_m}")
        print(f"  (from x = r_m(m) - shift with shift={shift})")
    
    # Step 2: Express target_m = a/b and compute residues mod each prime
    a = ZZ(target_m.numerator())
    b = ZZ(target_m.denominator())
    
    residues_by_prime = {}
    matched_vectors_by_prime = {}  # {p: {v_tuple: [rhs_indices where m_p appears]}}
    
    if debug:
        print(f"\nComputing residues for m = {a}/{b} mod each prime...")
    
    for p in prime_pool:
        p_int = int(p)
        
        # Check if denominator is zero mod p
        if (b % p_int) == 0:
            residues_by_prime[p_int] = 'DENOM_ZERO'
            if debug:
                print(f"  p={p_int}: denominator zero mod p (skipping)")
            continue
        
        # Compute m_p = (a * b^(-1)) mod p
        try:
            b_inv = pow(int(b % p_int), -1, p_int)
            m_p = (int(a % p_int) * b_inv) % p_int
            residues_by_prime[p_int] = m_p
        except ValueError:
            residues_by_prime[p_int] = 'INV_FAIL'
            if debug:
                print(f"  p={p_int}: inverse computation failed")
            continue
        
        # Step 3: Check which vectors have this residue in precomputed data
        p_data = precomputed_residues.get(p_int, {})
        matched_vectors_by_prime[p_int] = {}
        
        for v in vecs:
            v_tuple = tuple(v)
            roots_list = p_data.get(v_tuple, [])
            
            if not roots_list:
                continue
            
            # roots_list is [roots_rhs0, roots_rhs1, ...]
            matching_rhs = []
            for rhs_idx, roots_set in enumerate(roots_list):
                if m_p in roots_set:
                    matching_rhs.append(rhs_idx)
            
            if matching_rhs:
                matched_vectors_by_prime[p_int][v_tuple] = matching_rhs
    
    # Step 4: Analyze coverage per vector
    if debug:
        print(f"\n{'='*70}")
        print("COVERAGE ANALYSIS BY VECTOR")
        print(f"{'='*70}")
    
    vector_coverage = {}
    for v in vecs:
        v_tuple = tuple(v)
        matched_primes = []
        
        for p_int in prime_pool:
            if p_int in matched_vectors_by_prime:
                if v_tuple in matched_vectors_by_prime[p_int]:
                    matched_primes.append(p_int)
        
        coverage_frac = len(matched_primes) / float(len(prime_pool)) if prime_pool else 0.0
        vector_coverage[v_tuple] = {
            'matched_primes': matched_primes,
            'coverage_fraction': coverage_frac,
            'num_matched': len(matched_primes)
        }
        
        if debug and coverage_frac > 0.0:
            print(f"\nVector {v_tuple[:3]}... :")
            print(f"  Matched primes: {matched_primes[:10]}{'...' if len(matched_primes) > 10 else ''}")
            print(f"  Coverage: {coverage_frac:.1%} ({len(matched_primes)}/{len(prime_pool)} primes)")
    
    # Step 5: Try CRT + rational reconstruction for promising vectors
    if debug:
        print(f"\n{'='*70}")
        print("TESTING CRT + RATIONAL RECONSTRUCTION")
        print(f"{'='*70}")
    
    viable_reconstructions = []
    
    # Sort vectors by coverage (best first)
    sorted_vectors = sorted(
        vector_coverage.items(),
        key=lambda x: x[1]['coverage_fraction'],
        reverse=True
    )
    
    for v_tuple, cov_info in sorted_vectors:
        if cov_info['num_matched'] < MIN_PRIME_SUBSET_SIZE:
            continue  # Not enough primes for a viable subset
        
        matched_primes = cov_info['matched_primes']
        
        if debug:
            print(f"\nTesting vector {v_tuple[:3]}... ({cov_info['num_matched']} matched primes)")
        
        # Try subsets of various sizes
        found_for_this_vector = False
        for subset_size in range(MIN_PRIME_SUBSET_SIZE, 
                                 min(MIN_MAX_PRIME_SUBSET_SIZE, len(matched_primes)) + 1):
            
            # Heuristic: try up to 100 random subsets of this size
            import random
            max_subsets_to_try = min(100, len(list(combinations(matched_primes, subset_size))))
            
            subsets_to_try = random.sample(
                list(combinations(matched_primes, subset_size)),
                min(max_subsets_to_try, len(list(combinations(matched_primes, subset_size))))
            )
            
            for subset in subsets_to_try:
                subset_list = list(subset)
                
                # Get residues for this subset
                residues = tuple(residues_by_prime[p] for p in subset_list)
                
                # CRT lift
                try:
                    m0 = crt_cached(residues, tuple(subset_list))
                    M = 1
                    for p in subset_list:
                        M *= int(p)
                except Exception:
                    continue
                
                # Check if target_m = m0 + t*M for some small |t|
                # target_m = a/b, so we need: a/b = m0 + t*M
                # => a = b*(m0 + t*M) = b*m0 + b*t*M
                # => t = (a - b*m0) / (b*M)
                
                numerator = a - b * m0
                denominator = b * M
                
                if numerator % denominator == 0:
                    t = numerator // denominator
                    
                    if abs(t) <= tmax:
                        m_reconstructed = QQ(m0 + t * M)
                        
                        if m_reconstructed == target_m:
                            viable_reconstructions.append({
                                'vector': v_tuple,
                                'subset': subset_list,
                                'subset_size': len(subset_list),
                                'm0': m0,
                                'M': M,
                                't': t,
                                'm_reconstructed': m_reconstructed
                            })
                            
                            if debug:
                                print(f"  ✓ FOUND via subset {subset_list}")
                                print(f"    m0={m0}, M={M}, t={t}")
                                print(f"    m = {m0} + {t}*{M} = {m_reconstructed}")
                            
                            found_for_this_vector = True
                            break  # Found one, that's enough for this subset size
                
                # Also try rational reconstruction
                try:
                    a_recon, b_recon = rational_reconstruct(m0 % M, M)
                    m_recon = QQ(a_recon) / QQ(b_recon)
                    
                    if m_recon == target_m:
                        viable_reconstructions.append({
                            'vector': v_tuple,
                            'subset': subset_list,
                            'subset_size': len(subset_list),
                            'm0': m0,
                            'M': M,
                            't': 'rational_recon',
                            'm_reconstructed': m_recon
                        })
                        
                        if debug:
                            print(f"  ✓ FOUND via rational reconstruction on subset {subset_list}")
                            print(f"    m0={m0}, M={M}")
                            print(f"    Reconstructed: {a_recon}/{b_recon} = {m_recon}")
                        
                        found_for_this_vector = True
                        break
                
                except RationalReconstructionError:
                    pass
            
            if found_for_this_vector:
                break  # Found it for this vector, move to next vector
    
    # Step 6: Summary
    if debug:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Target: x = {target_x_q}, m = {target_m}")
        print(f"Total vectors: {len(vecs)}")
        print(f"Vectors with any coverage: {sum(1 for v in vector_coverage.values() if v['num_matched'] > 0)}")
        print(f"Viable reconstructions found: {len(viable_reconstructions)}")
        #assert viable_reconstructions, ("no viable reconstructions found for target x:", target_x_q)
        
        if viable_reconstructions:
            print(f"\n✓ POINT IS FINDABLE")
            print(f"\nExample reconstructions:")
            for i, recon in enumerate(viable_reconstructions[:3]):
                print(f"\n  [{i+1}] Vector: {recon['vector'][:3]}...")
                print(f"      Subset size: {recon['subset_size']}")
                print(f"      Primes: {recon['subset']}")
                print(f"      t: {recon['t']}")
        else:
            print(f"\n✗ POINT NOT FINDABLE with current search parameters")
    
    return {
        'target_x': target_x_q,
        'target_m': target_m,
        'residues_by_prime': residues_by_prime,
        'vector_coverage': vector_coverage,
        'viable_reconstructions': viable_reconstructions,
        'is_findable': len(viable_reconstructions) > 0
    }



def compute_residue_coverage_for_m(m_value, precomputed_residues, prime_pool, v_tuple=None):
    """
    Compare a target rational m = a/b (in QQ) against the precomputed residue fingerprints.

    Args:
        m_value: rational number (QQ or coercible to QQ)
        precomputed_residues: dict mapping p -> { v_tuple : [ set(roots_rhs0), set(roots_rhs1), ... ] }
        prime_pool: iterable of primes to check
        v_tuple: optional key to restrict residue comparison to a specific vector tuple

    Returns:
        {
          'm': QQ rational value,
          'matched_primes': [p,...],
          'unseen_primes': [p,...],
          'denom_zero_primes': [p,...],
          'coverage_fraction': float between 0 and 1,
          'per_prime': { p: {'residue': r or None, 'status': 'matched'|'unseen'|'denom_zero'} }
        }
    """
    from sage.all import QQ, Mod

    # Coerce to QQ explicitly
    m_q = QQ(m_value)
    a = ZZ(m_q.numerator())
    b = ZZ(m_q.denominator())

    matched = []
    unseen = []
    denom_zero = []
    per_prime = {}

    for p in prime_pool:
        p = int(p)
        per_prime[p] = {'residue': None, 'status': 'unseen'}

        # If denominator is 0 mod p, cannot test modulo p
        if (b % p) == 0:
            denom_zero.append(p)
            per_prime[p]['status'] = 'denom_zero'
            continue

        # compute residue in GF(p)
        residue = int(Mod(a, p) * Mod(b, p)**(-1))
        per_prime[p]['residue'] = residue

        # check whether residue appears in precomputed_residues[p]
        p_map = precomputed_residues.get(p, {})
        if not p_map:
            unseen.append(p)
            per_prime[p]['status'] = 'unseen'
            continue

        # restrict to one v_tuple or scan all
        found = False
        if v_tuple is not None:
            sets_list = p_map.get(v_tuple, [])
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
            matched.append(p)
            per_prime[p]['status'] = 'matched'
        else:
            unseen.append(p)
            per_prime[p]['status'] = 'unseen'

    usable = max(1, len(prime_pool) - len(denom_zero))
    coverage = float(len(matched)) / float(usable) if usable > 0 else 0.0

    return {
        'm': m_q,
        'matched_primes': matched,
        'unseen_primes': unseen,
        'denom_zero_primes': denom_zero,
        'coverage_fraction': coverage,
        'per_prime': per_prime
    }
