"""
search_analysis.py: Statistical analysis, auto-tuning, and diagnostics.
"""
from .search_config import (
    DEBUG, EXTRA_PRIME_MIN_R, EXTRA_PRIME_MAX_R, ROOTS_THRESHOLD,
    EXTRA_PRIME_TARGET_DENSITY, EXTRA_PRIME_MAX, EXTRA_PRIME_SKIP,
    Counter
)
from stats import * # Project-level import
from itertools import combinations
from sage.all import *

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
