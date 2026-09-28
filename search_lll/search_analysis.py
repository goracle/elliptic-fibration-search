from .search_config import DEBUG, EXTRA_PRIME_MIN_R, EXTRA_PRIME_MAX_R, ROOTS_THRESHOLD, EXTRA_PRIME_TARGET_DENSITY, EXTRA_PRIME_MAX, EXTRA_PRIME_SKIP, TMAX, Counter
from stats import *
from itertools import combinations
from sage.all import *
from search_common import MIN_PRIME_SUBSET_SIZE, MIN_MAX_PRIME_SUBSET_SIZE

"""
search_analysis.py: Statistical analysis, auto-tuning, and diagnostics.
"""

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
    """Select extra primes based on measured r_p values.

    Kept in sync with ll_utilities.choose_extra_primes (which wins the
    actual import-shadow in search_lll/__init__.py and search_main.py --
    see the note there): sort primarily by prime size descending, with the
    entropy-like r*(1-r) term only as a tiebreaker, so this doesn't
    silently prefer small, weakly-discriminating primes the way the
    original size-blind sort did.
    """
    cand = [(p, r) for p, r in stats.items()
            if p not in skip_small and EXTRA_PRIME_MIN_R < r < EXTRA_PRIME_MAX_R]
    cand.sort(key=lambda t: (-t[0], -(t[1] * (1 - t[1]))))
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

"""
search_analysis.py: Statistical analysis, auto-tuning, and diagnostics.
"""

# ... (Functions estimate_prime_stats, choose_extra_primes, expected_density,
# _assert_rhs_consistency, _print_subset_productivity_stats, _batch_check_rationality
# remain largely the same, focusing on utility and structure validation) ...

def diagnose_missed_point(target_x, r_m_callable, shift, precomputed_residues, prime_pool, vecs, tmax=TMAX, debug=True):
    """
    Diagnose why a specific x-value wasn't found using Deterministic Capacity Check.
    Instead of random sampling, we calculate the total product of compatible primes (M_capacity).
    If M_capacity > M_required, the point is theoretically guaranteed to be found
    by rational reconstruction, assuming the search explores the prime combination.
    """
    from sage.all import QQ, ZZ, crt
    from search_lll.rational_arithmetic import rational_reconstruct

    # 1. Compute Target m
    try:
        const_term = r_m_callable(m=QQ(0))
        target_m = const_term - QQ(target_x) - shift
    except Exception as e:
        print(f"[diagnose] Error computing target m: {e}")
        return {}

    if debug:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC: Strict Capacity Check for x = {target_x}")
        print(f"{'='*70}")
        print(f"Target m: {target_m}")

    # 2. Compute Residues and Find Compatible Primes
    a = ZZ(target_m.numerator())
    b = ZZ(target_m.denominator())

    # Required modulus size for reconstruction: M > 2 * max(|a|, |b|)^2
    M_required = 2 * max(abs(a), abs(b))**2
    log_M_req = math.log10(M_required) if M_required > 0 else 0

    residues_by_prime = {}

    for p in prime_pool:
        p_int = int(p)
        if b % p_int == 0:
            residues_by_prime[p_int] = 'DENOM_ZERO'
            continue
        try:
            val = (a * pow(b, -1, p_int)) % p_int
            residues_by_prime[p_int] = val
        except:
            continue

    # 3. Check Vectors (Find best matching vector)
    vector_stats = []

    for v in vecs:
        v_tuple = tuple(v)
        compatible_primes = []
        M_capacity = 1

        for p in prime_pool:
            p_int = int(p)
            if p_int not in residues_by_prime or residues_by_prime[p_int] == 'DENOM_ZERO':
                continue

            expected_r = residues_by_prime[p_int]

            # Check if expected_r appears in ANY of the RHS sets for this vector
            p_data = precomputed_residues.get(p_int, {})
            roots_list = p_data.get(v_tuple, [])

            if any(expected_r in r_set for r_set in roots_list):
                compatible_primes.append(p_int)
                M_capacity *= p_int

        vector_stats.append({
            'vector': v_tuple,
            'compatible_count': len(compatible_primes),
            'primes': compatible_primes,
            'capacity': M_capacity,
            'log_capacity': math.log10(M_capacity) if M_capacity > 0 else 0
        })

    # Sort by capacity descending
    vector_stats.sort(key=lambda x: x['capacity'], reverse=True)

    best = vector_stats[0] if vector_stats else None

    if debug and best:
        print(f"Best matching vector: {best['vector'][:3]}...")
        print(f"  Compatible Primes: {best['compatible_count']} / {len(prime_pool)}")
        print(f"  Capacity (log10): {best['log_capacity']:.2f}")
        print(f"  Required (log10): {log_M_req:.2f}")

        if best['capacity'] > M_required:
            print("\n✓ STATUS: THEORETICALLY FINDABLE")
            print("  The residues for this vector contain the target point with sufficient capacity.")
            print("  If search failed, it likely didn't sample the specific subset of compatible primes.")

            # Proof of Concept Reconstruction attempt
            print("\n  [Proof of Concept Reconstruction]")
            proof_subset = []
            proof_prod = 1
            for p in best['primes']:
                proof_subset.append(p)
                proof_prod *= p
                if proof_prod > M_required * 10: # Safety margin
                    break

            print(f"  Using subset of size {len(proof_subset)}...")
            try:
                proof_residues = [residues_by_prime[p] for p in proof_subset]
                m0 = crt(proof_residues, proof_subset)
                M = proof_prod

                # Perform Rational Recon
                m_recon = rational_reconstruct(m0, M)
                m_recon = QQ(m_recon[0])/QQ(m_recon[1])

                if m_recon == target_m:
                    print("  ✓ MATCH! Target successfully reconstructed.")
                else:
                    print(f"  ✗ Mismatch (Got {m_recon}, Expected {target_m})")
            except Exception as e:
                print(f"  Reconstruction failed: {e}")

        else:
            print("\n✗ STATUS: NOT FINDABLE (Insufficient Capacity)")
            print("  The residues found do not carry enough information to reconstruct the point.")

    return {}

def discover_candidates_via_residue_graph(precomputed_residues, prime_pool, height_bound,
                                           v_tuple=None, top_k=10, debug=True,
                                           max_tuples=2_000_000, known_m=None,
                                           verbose_graph=False, max_chains=None):
    """
    Bottom-up candidate discovery via the cross-prime residue CRT graph
    (search_lll/residue_crt_graph.py), as an alternative/complement to
    diagnose_missed_point's top-down "given target x, is it findable"
    check and compute_residue_coverage_for_m's "given target m, which
    primes agree" check.

    Unlike both of those, this takes NO target m or x -- it builds the
    graph of which residues at DIFFERENT primes are pairwise/k-wise
    mutually CRT-consistent (see residue_crt_graph.build_residue_graph /
    build_residue_graph_ktuple), reads off the largest connected
    components as candidates, and attempts rational reconstruction on
    each. This is meant to be run instead of (or alongside) picking a
    random prime subset and hoping -- the graph tells you directly which
    residues could plausibly be the same underlying rational point.

    *** height_bound must be an actual usable bound for the primes in
    prime_pool, not the flat legacy constant, or min_tuple_size_for_margin
    may come back None (see its docstring: the smallest k primes' product
    has to clear a real margin over (2*height_bound+1)^2, and if the pool
    is too small / height_bound too large, no k works -- you'd need more
    or larger primes in the pool first). ***

    Returns:
        {
          'k_used': int or None,       # tuple size actually used for edges
          'graph': dict,                # raw build_residue_graph[_ktuple] result
          'candidates': [               # top_k components, each reconstructed
             {'primes': [...], 'num_nodes': int, 'reconstructions': [...]}
             , ...
          ]
        }
    If k_used is None, the pool/height_bound combination can't support a
    trustworthy edge test at all yet -- prints a message and returns
    candidates=[] rather than silently building an untrustworthy graph.
    """
    from . import residue_crt_graph as rcg

    H = int(height_bound)
    k_needed = rcg.min_tuple_size_for_margin(prime_pool, height_bound=H)

    if debug:
        box = (2 * H + 1) ** 2
        print(f"[residue_graph] height_bound={H}, box=(2H+1)^2={box}, "
              f"min tuple size for trustworthy edges: k={k_needed}")

    if k_needed is None:
        if debug:
            print("[residue_graph] prime_pool too small / height_bound too "
                  "large for ANY tuple size in this pool to clear the "
                  "margin -- add more/larger primes to the pool or tighten "
                  "height_bound before trusting this. Returning no candidates.")
        return {'k_used': None, 'graph': None, 'candidates': []}

    if k_needed == 2:
        graph = rcg.build_residue_graph(
            precomputed_residues, prime_pool, height_bound=H, v_tuple=v_tuple,
        )
    else:
        cost = rcg.estimate_ktuple_cost(prime_pool, k_needed)
        if debug:
            from .search_config import PARALLEL_PRIME_WORKERS
            print(f"[residue_graph] k={k_needed}: C(pool,k)={cost} tuples "
                  f"(max_tuples cap={max_tuples}, workers={PARALLEL_PRIME_WORKERS})")
        graph = rcg.build_residue_graph_ktuple(
            precomputed_residues, prime_pool, height_bound=H, k=k_needed, v_tuple=v_tuple,
            max_tuples=max_tuples, progress=verbose_graph, known_m=known_m,
            max_chains=max_chains, label=f"v={v_tuple}",
        )

    if debug:
        _keep = 100.0 * graph['edges_kept'] / graph['edges_tested'] if graph['edges_tested'] else 0.0
        _ctr = graph.get('counters', {})
        print(f"[residue_graph] v={v_tuple} k={k_needed}: nodes={graph['nodes']} "
              f"edges_tested={graph['edges_tested']} edges_kept={graph['edges_kept']} ({_keep:.1f}%) "
              f"confirmed_chains={graph.get('chains_confirmed', 'n/a')} "
              f"max_gen={graph.get('max_generation_reached', 'n/a')} "
              f"cap_dropped={_ctr.get('cap_dropped', 'n/a')} deferred={_ctr.get('deferred', 'n/a')} "
              f"dead_end_chains={_ctr.get('dead_end_dropped', 'n/a')}")
        for t in graph.get('trace', []):
            print(f"[residue_graph] v={v_tuple} trace m={t['m']}: "
                  + (f"CONFIRMED at gen {t['confirmed_gen']}" if t.get('confirmed_gen') is not None
                     else "alive at end but never confirmed" if t['lost_gen'] is None
                     else f"LOST at gen {t['lost_gen']} ({t['lost_why']})"))
        if 'confirmed_chains' not in graph:
            for i, row in enumerate(rcg.summarize_components(graph, top_k=top_k)):
                print(f"  component {i}: primes={row['primes']} "
                      f"(#primes={row['num_primes']}, #nodes={row['num_nodes']})")

    # Candidates come from confirmed_chains, not graph['components']: a
    # component is the union-find closure over EVERY chain that was ever
    # confirmed, so two unrelated local solutions that happen to share
    # one node (e.g. both touch prime 89) get merged into one blob, and
    # reconstructing "the" m from that blob is ambiguous -- there can
    # genuinely be several distinct local points tangled into a single
    # component. Real solutions here are local (a handful of primes
    # gluing together into one point) rather than global (one point
    # agreeing across most of the pool) -- trying too many primes at
    # once breaks the glue rather than reinforcing it -- so each
    # confirmed chain is already exactly one candidate m on its own,
    # with no ambiguity and no search needed to extract it.
    if 'confirmed_chains' in graph:
        # top_k is NOT applied here: unlike a component (one blob you'd
        # otherwise want to sample from), every confirmed chain is
        # already its own distinct candidate, cheap to reconstruct and
        # cheap to test for curve membership downstream (a few thousand
        # chains is instantaneous to check), so there's no reason to
        # throw most of them away before the caller even sees them.
        candidates = []
        for chain in graph['confirmed_chains']:
            recon = rcg.reconstruct_candidate_from_chain(chain, height_bound=H)
            if recon is None:
                continue
            candidates.append({
                'primes': chain['primes'],
                'num_nodes': len(chain['node_keys']),
                'reconstructions': [recon],
            })
    else:
        # build_residue_graph (k_needed == 2 path) has no chain concept,
        # only components -- fall back to the old component-based
        # reconstruction there.
        candidates = []
        for comp, comp_primes in zip(graph['components'][:top_k], graph['component_primes'][:top_k]):
            if len(comp_primes) < 2:
                continue
            reconstructions = rcg.reconstruct_candidates_from_component(comp, height_bound=H)
            candidates.append({
                'primes': sorted(comp_primes),
                'num_nodes': len(comp),
                'reconstructions': reconstructions,
            })

    return {'k_used': k_needed, 'graph': graph, 'candidates': candidates}


    """Wrapper for rational reconstruction."""
    from sage.all import QQ, ZZ
    try:
        ret = ZZ(m0).rational_reconstruction(ZZ(M))
        if ret is None:
            raise ValueError("Reconstruction failed")
        return ret
    except:
        raise ValueError("Reconstruction failed")
