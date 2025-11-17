
from search_common import *
# Add to tower.sage
def build_multiple_fibrations(fx_PR, pts_xy, num_fibrations, max_steps=3, 
                               base_seed=SEED_INT, verbose=DEBUG):
    """
    Build multiple independent fibration towers with different anchor points.
    Each fibration should find the same rational points (conjecturally).
    
    Args:
        fx_PR: Initial polynomial
        pts_xy: Base points
        num_fibrations: How many independent fibrations to construct
        max_steps: Tower depth
        base_seed: Base seed (each fibration gets base_seed + k)
        
    Returns:
        List of tower dictionaries, one per fibration
    """
    if not USE_ANCHOR_POINTS:
        raise RuntimeError("build_multiple_fibrations requires USE_ANCHOR_POINTS=True")
    
    fibrations = []
    for k in range(num_fibrations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Building Fibration {k+1}/{num_fibrations} (anchor seed={base_seed + k})")
            print(f"{'='*70}")
        
        # Each fibration uses a different seed for anchor point generation
        tower = iterate_tower(
            fx_PR=fx_PR,
            pts_xy=pts_xy,
            max_steps=max_steps,
            seed_int=base_seed + k,  # Different seed = different anchors
            verbose=verbose
        )
        
        fibrations.append({
            'tower': tower,
            'seed': base_seed + k,
            'id': k
        })
    
    return fibrations


def compute_consensus_residues(precomputed_residues_list, prime_pool, 
                                consensus_threshold=CONSENSUS_THRESHOLD,
                                debug=DEBUG):
    """
    Compute consensus residues across multiple fibrations.
    A residue is kept if it appears in >= consensus_threshold fraction of fibrations.
    
    Args:
        precomputed_residues_list: List of precomputed_residues dicts (one per fibration)
        prime_pool: List of primes
        consensus_threshold: Minimum fraction of fibrations that must agree
        
    Returns:
        consensus_residues: Dict in same format as precomputed_residues
        stats: Dict with filtering statistics
    """
    from collections import defaultdict, Counter
    
    num_fibrations = len(precomputed_residues_list)
    min_votes_needed = int(consensus_threshold * num_fibrations)
    
    if debug:
        print(f"\n{'='*70}")
        print(f"CONSENSUS FILTER: {num_fibrations} fibrations, "
              f"threshold={consensus_threshold:.1%} ({min_votes_needed} votes needed)")
        print(f"{'='*70}")
    
    # Track votes for each (prime, vector, rhs_idx, residue) tuple
    residue_votes = defaultdict(Counter)  # {(p, v_tuple, rhs_idx): Counter({residue: count})}
    
    # Count votes across all fibrations
    for fib_idx, precomp in enumerate(precomputed_residues_list):
        for p in prime_pool:
            if p not in precomp:
                continue
            
            mapping = precomp[p]
            for v_tuple, rhs_lists in mapping.items():
                for rhs_idx, residue_set in enumerate(rhs_lists):
                    key = (p, v_tuple, rhs_idx)
                    for r in residue_set:
                        if isinstance(r, int):
                            residue_votes[key][r] += 1
    
    # Build consensus: keep only residues with >= min_votes_needed
    consensus_residues = {}
    stats = {
        'total_before': 0,
        'total_after': 0,
        'per_prime_before': {},
        'per_prime_after': {},
        'reduction_ratio': 0.0
    }
    
    for p in prime_pool:
        consensus_residues[p] = {}
        
        # Collect all (v_tuple, rhs_idx) pairs for this prime
        keys_for_prime = {(v_tuple, rhs_idx) 
                          for (pp, v_tuple, rhs_idx) in residue_votes.keys() 
                          if pp == p}
        
        prime_before = 0
        prime_after = 0
        
        for v_tuple, rhs_idx in keys_for_prime:
            key = (p, v_tuple, rhs_idx)
            vote_counter = residue_votes.get(key, Counter())
            
            # Filter: keep residues with enough votes
            consensus_set = {r for r, votes in vote_counter.items() 
                            if votes >= min_votes_needed}
            
            # Track original size
            original_set = {r for r in vote_counter.keys()}
            prime_before += len(original_set)
            prime_after += len(consensus_set)
            
            # Store in output format
            if v_tuple not in consensus_residues[p]:
                # Initialize with empty sets for all RHS indices
                max_rhs = max(idx for (_, _, idx) in keys_for_prime if _ == v_tuple)
                consensus_residues[p][v_tuple] = [set() for _ in range(max_rhs + 1)]
            
            consensus_residues[p][v_tuple][rhs_idx] = consensus_set
        
        stats['per_prime_before'][p] = prime_before
        stats['per_prime_after'][p] = prime_after
        stats['total_before'] += prime_before
        stats['total_after'] += prime_after
    
    if stats['total_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_after'] / stats['total_before'])
    
    if debug:
        print(f"\nConsensus Filter Results:")
        print(f"  Total residues before: {stats['total_before']:,}")
        print(f"  Total residues after:  {stats['total_after']:,}")
        print(f"  Filtered out: {stats['total_before'] - stats['total_after']:,} "
              f"({100*stats['reduction_ratio']:.1f}%)")
        
        # Show per-prime breakdown for top primes
        sorted_primes = sorted(stats['per_prime_before'].items(), 
                               key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 primes by original residue count:")
        for p, before in sorted_primes:
            after = stats['per_prime_after'].get(p, 0)
            reduction = 1.0 - (after / before) if before > 0 else 0.0
            print(f"    p={p}: {before} → {after} ({100*reduction:.1f}% filtered)")
    
    return consensus_residues, stats


# Add to search7_genus2.sage in doloop_genus2() function
def doloop_genus2_with_consensus(data_pts, sextic_coeffs, all_known_x, cumulative_stats):
    """
    Modified doloop that uses multi-fibration consensus filter.
    """
    # ... [Initial setup code same as before] ...
    
    # Build MULTIPLE fibrations instead of one
    if USE_CONSENSUS_FILTER:
        print(f"\n{'='*70}")
        print(f"MULTI-FIBRATION CONSENSUS MODE")
        print(f"Building {NUM_CONSENSUS_FIBRATIONS} independent fibrations...")
        print(f"{'='*70}")
        
        fibrations = build_multiple_fibrations(
            fx_PR=shifted_G_poly,
            pts_xy=base_pts,
            num_fibrations=NUM_CONSENSUS_FIBRATIONS,
            max_steps=len(sextic_coeffs) - 5,
            base_seed=SEED_INT,
            verbose=DEBUG
        )
        
        # Use the FIRST fibration for the main search infrastructure
        # (cd, roots, etc. should be the same for all fibrations conjecturally)
        tower = fibrations[0]['tower']
        
    else:
        # Original single-fibration mode
        tower = iterate_tower(
            fx_PR=shifted_G_poly,
            pts_xy=base_pts,
            max_steps=len(sextic_coeffs) - 5,
            seed_int=SEED_INT,
            verbose=DEBUG
        )
        fibrations = None
    
    # ... [Extract roots and build cd as before] ...
    # All fibrations should have identical roots/cd structure
    
    # ... [Continue with search setup until precompute phase] ...
    
    # === MODIFIED: PRECOMPUTE FOR ALL FIBRATIONS ===
    if USE_CONSENSUS_FILTER and fibrations:
        print(f"\n{'='*70}")
        print(f"PRECOMPUTING RESIDUES FOR {len(fibrations)} FIBRATIONS")
        print(f"{'='*70}")
        
        all_precomputed_residues = []
        
        for fib_idx, fib_data in enumerate(fibrations):
            print(f"\nPrecomputing fibration {fib_idx + 1}/{len(fibrations)}...")
            
            # Each fibration has its own tower, extract its specific data
            fib_tower = fib_data['tower']
            fib_roots = [step['r_expr'] for step in fib_tower]
            fib_rhs = fib_tower[-1]['f_i']
            
            # Build search_rhs_list for this fibration
            # (same structure as main code)
            fib_search_rhs_list = [SR(cd.phi_x)]  # Top level same for all
            
            # Build cd for this specific fibration if needed
            # OR reuse main cd if structure is identical (they should be)
            
            # Precompute residues for this fibration
            # [Standard precompute code here, storing in fib_precomputed]
            
            # ... [Run the full precompute worker pool as in original code] ...
            # fib_precomputed = {p: {v_tuple: [sets]}}
            
            all_precomputed_residues.append(fib_precomputed)
        
        # === APPLY CONSENSUS FILTER ===
        consensus_residues, consensus_stats = compute_consensus_residues(
            all_precomputed_residues,
            prime_pool=prime_pool,
            consensus_threshold=CONSENSUS_THRESHOLD,
            debug=DEBUG
        )
        
        # Store consensus stats
        cumulative_stats.consensus_filter_stats = consensus_stats
        
        # Use consensus residues for the rest of the search
        precomputed_residues = consensus_residues
        
    else:
        # Original single-fibration precompute
        # ... [Standard precompute code] ...
        pass
    
    # === REST OF SEARCH UNCHANGED ===
    # The search now operates on consensus_residues instead of raw precomputed
    # Everything else (CRT, rationality checks, etc.) stays the same
    
    # ... [Continue with normal search flow] ...
    
    return all_known_x, cumulative_stats


# Utility: Print consensus effectiveness
def print_consensus_effectiveness(consensus_stats, cumulative_stats):
    """
    Print how effective the consensus filter was at reducing junk.
    """
    print(f"\n{'='*70}")
    print("CONSENSUS FILTER EFFECTIVENESS")
    print(f"{'='*70}")
    
    cs = consensus_stats
    print(f"\nResidues filtered: {cs['total_before'] - cs['total_after']:,} / {cs['total_before']:,}")
    print(f"Reduction: {100*cs['reduction_ratio']:.1f}%")
    
    # Compare to rationality test results
    if hasattr(cumulative_stats, 'counters'):
        total_tests = cumulative_stats.counters.get('rationality_tests_total', 0)
        successes = cumulative_stats.counters.get('rationality_tests_success', 0)
        
        if total_tests > 0:
            hit_rate = successes / total_tests
            print(f"\nRationality tests:")
            print(f"  Total: {total_tests:,}")
            print(f"  Successes: {successes:,}")
            print(f"  Hit rate: {100*hit_rate:.2f}%")
            
            # Estimate how many tests we saved
            tests_saved = int(cs['total_before'] - cs['total_after'])
            time_saved_est = tests_saved * (cumulative_stats.phases.get('search_subsets_and_check', 0) / max(1, total_tests))
            print(f"\nEstimated tests saved: ~{tests_saved:,}")
            print(f"Estimated time saved: ~{time_saved_est:.1f}s")
