from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict

@PROFILE
def evaluate_log_delta_ratio(cd_primary, cd_other, m_values):
    """
    Evaluate log(|Δ_other(m)| / |Δ_primary(m)|) for each m, then average.
    This is NOT the same as avg(log|Δ_other|) - avg(log|Δ_primary|)!
    """
    import math
    from sage.all import QQ
    
    Delta_primary = cd_primary.E_weier.discriminant()
    Delta_other = cd_other.E_weier.discriminant()
    
    if hasattr(Delta_primary, 'numerator'):
        Delta_primary = Delta_primary.numerator()
        Delta_other = Delta_other.numerator()
    
    m_sym = cd_primary.a4.parent().gen()
    log_ratios = []
    
    for m_val in m_values:
        try:
            Delta_p_val = Delta_primary.subs({m_sym: m_val})
            Delta_o_val = Delta_other.subs({m_sym: m_val})
            
            Delta_p_qq = QQ(Delta_p_val)
            Delta_o_qq = QQ(Delta_o_val)
            
            if Delta_p_qq != 0 and Delta_o_qq != 0:
                log_ratio = math.log(abs(float(Delta_o_qq))) - math.log(abs(float(Delta_p_qq)))
                log_ratios.append(log_ratio)
        except Exception:
            continue
    
    if not log_ratios:
        return 0.0
    
    return sum(log_ratios) / len(log_ratios)



@PROFILE
def sample_delta_ratios(cd, sections, num_samples=200, m_range=(-1000, 1000), seed=None):
    """
    Sample log|Δ| values at random rational m-values to estimate the 
    discriminant behavior of this fibration.
    """
    if seed is not None:
        set_random_seed(seed)
    
    samples = []
    m_sym = cd.a4.parent().gen()
    
    try:
        delta_m = cd.E_weier.discriminant()
    except Exception:
        try:
            delta_m = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
        except Exception:
            print(f"FATAL: Could not get discriminant from cd.E_weier or from cd.a4/a6 for {cd}")
            return [] 
    
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        num = ZZ.random_element(m_range[0], m_range[1])
        den = ZZ.random_element(1, 100)
        m_val = QQ(num) / QQ(den)
        
        if m_val in cd.bad_primes:
            continue
            
        try:
            delta_val = delta_m.subs({m_sym: m_val})
            delta_val = QQ(delta_val)
            
            if delta_val == 0:
                continue
                
            log_delta = float(math.log(abs(float(delta_val))))
            samples.append(log_delta)
            
        except Exception:
            continue
    
    if len(samples) < num_samples // 2:
        print(f"Warning: only got {len(samples)} valid samples out of {num_samples} requested")
    
    return samples


# DEPRECATED / COMPATIBILITY FUNCTIONS (Kept to avoid import errors)






# Deprecated function kept for compatibility
def compute_consensus_residues(precomputed_residues_list, prime_pool, consensus_threshold=0.7, debug=False):
    print("WARNING: calling deprecated compute_consensus_residues")
    return {}, {}






@PROFILE
def compute_consensus_residues_with_height_matching(
    all_precomputed_residues,
    fibration_geometries,
    prime_pool,
    consensus_threshold=0.2,  # RELAXED: 20% means ~2/10 agreement is enough
    height_tolerance_log=2.5,
    use_delta_scaling=True,
    debug=DEBUG
):
    """
    Height-aware consensus filter with RELAXED VOTING.
    
    Addresses the "Basis Span" problem: A fibration might find the correct vector (height match)
    but fail to find the residue because its section basis doesn't span that specific point 
    modulo p. 
    
    Strategy:
    1. Match vectors by height.
    2. If a fibration doesn't find a matching vector, it ABSTAINS.
    3. If a fibration finds a vector but has different/empty residues, it votes NO.
    4. We require (Primary + k others) to agree, where k is small (threshold ~0.2).
    """
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}

    # Policy: If only the primary sees the vector (e.g. difficult height), 
    # we trust it (min_participating=1).
    MIN_PARTICIPATING = 1 
    
    if debug:
        print(f"\n{'='*70}")
        print(f"HEIGHT-AWARE CONSENSUS: {num_fibs} fibrations")
        print(f"Metric: Normalized Height h_norm = h_canonical / deg(Delta)")
        print(f"Tolerance: |log(h_norm_A) - log(h_norm_B)| < {height_tolerance_log}")
        print(f"Policy: RELAXED VOTE (Threshold {consensus_threshold:.1%} of PARTICIPATING)")
        print(f"        (Primary + 1 confirmation is usually sufficient)")
        print(f"{'='*70}")
    
    # 1. Pre-calculate Discriminant Degrees
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        cd_obj = geom.get('cd')
        try:
            Delta = cd_obj.E_weier.discriminant()
            if hasattr(Delta, 'numerator'):
                d_deg = Delta.numerator().degree() 
            else:
                d_deg = Delta.degree()
        except:
            d_deg = max(1, 2 * int(cd_obj.a4.degree()))
            
        fib_meta.append({
            'H': geom['H'],
            'disc_deg': float(max(1, d_deg)),
            'name': geom.get('name', f"fib_{i}")
        })

    primary_deg = fib_meta[0]['disc_deg']
    primary_H = fib_meta[0]['H']

    # 2. Main Consensus Loop
    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'vectors_matched_all_fibs': 0,
        'vectors_with_consensus': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'per_prime_stats': {},
        'blame_height': defaultdict(int),
        'participation_dist': defaultdict(int)
    }
    
    for p in prime_pool:
        consensus_residues[p] = {}
        primary_map = all_precomputed_residues[0].get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {'vectors_primary': 0, 'residues_before': 0, 'residues_after': 0}
        
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            stats['total_vectors_primary'] += 1
            prime_stats['vectors_primary'] += 1
            
            # Count residues
            total_res_this_vec = sum(len(r_list) for r_list in rhs_lists_primary)
            prime_stats['residues_before'] += total_res_this_vec
            
            # --- Primary Height ---
            try:
                v_primary = vector(QQ, v_primary_tuple)
                hcanon_p = float(v_primary * primary_H * v_primary)
                hcanon_p = max(hcanon_p, 1e-20)
                log_h_primary_norm = math.log(hcanon_p) - math.log(primary_deg)
            except Exception:
                continue
            
            # Initialize VOTES
            num_rhs = len(rhs_lists_primary)
            rhs_votes = [Counter() for _ in range(num_rhs)]
            
            # Primary always votes
            for i in range(num_rhs):
                for r in rhs_lists_primary[i]:
                    if isinstance(r, int):
                        rhs_votes[i][r] += 1

            # Track who participated (found a matching vector)
            participating_fibs_count = 1 
            
            # --- Check against other fibrations ---
            for fib_idx in range(1, num_fibs):
                other_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_map: 
                    continue
                
                H_other = fib_meta[fib_idx]['H']
                deg_other = fib_meta[fib_idx]['disc_deg']
                
                best_match_residues = None
                best_diff = float('inf')
                
                # Search for height-compatible vector
                for v_o_t, rhs_o in other_map.items():
                    try:
                        v_o = vector(QQ, v_o_t)
                        h_o = max(float(v_o * H_other * v_o), 1e-20)
                        log_h_o = math.log(h_o) - math.log(deg_other)
                        diff = abs(log_h_primary_norm - log_h_o)
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_match_residues = rhs_o
                    except: pass
                
                # DECISION: Did this fibration find the point?
                if best_match_residues is not None and best_diff <= height_tolerance_log:
                    participating_fibs_count += 1
                    
                    # Cast Votes
                    for i in range(num_rhs):
                        if i < len(best_match_residues):
                            unique_res = set(r for r in best_match_residues[i] if isinstance(r, int))
                            for r in unique_res:
                                rhs_votes[i][r] += 1
                else:
                    # Log abstention reason
                    stats['blame_height'][fib_idx] += 1

            # --- DYNAMIC THRESHOLDING ---
            stats['participation_dist'][participating_fibs_count] += 1
            
            if participating_fibs_count < MIN_PARTICIPATING:
                continue

            # Calculate required votes. 
            # If threshold is 0.2 and participants is 10, requires 2 votes.
            # This ensures Primary + 1 Confirmation is enough.
            # If only Primary participates (1), requires 1 vote (Primary itself).
            min_votes_required = max(1, int(math.ceil(consensus_threshold * participating_fibs_count)))
            
            final_residue_sets = []
            has_consensus = False
            
            for i in range(num_rhs):
                kept = {r for r, count in rhs_votes[i].items() if count >= min_votes_required}
                final_residue_sets.append(kept)
                if kept:
                    has_consensus = True
                
            if has_consensus:
                consensus_residues[p][v_primary_tuple] = final_residue_sets
                stats['vectors_with_consensus'] += 1
                
                if participating_fibs_count == num_fibs:
                    stats['vectors_matched_all_fibs'] += 1
                
                prime_stats['residues_after'] += sum(len(s) for s in final_residue_sets)
        
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']

    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    
    if debug:
        print(f"\nConsensus Diagnostics:")
        print(f"  Participation Histogram (How many fibs saw a vector?):")
        for k in sorted(stats['participation_dist'].keys()):
            print(f"    Saw by {k} fibs: {stats['participation_dist'][k]} vectors")
            
        print(f"  Height Abstentions (Fibs that missed the vector):")
        for fid, count in sorted(stats['blame_height'].items()):
            name = fib_meta[fid]['name']
            print(f"    Fib {fid} ({name}): {count}")
        
        print(f"\nConsensus Statistics:")
        print(f"  Primary vectors: {stats['total_vectors_primary']}")
        print(f"  With consensus: {stats['vectors_with_consensus']}")
        print(f"  Residues: {stats['total_residues_before']:,} -> {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")

    return consensus_residues, stats



@PROFILE
def compute_consensus_residues_with_height_matching(
    all_precomputed_residues,
    fibration_geometries,
    prime_pool,
    consensus_threshold=0.3,  # 30% agreement on residues
    height_tolerance_log=2.5,
    use_delta_scaling=True, # dummy arg, backwards compat.
    debug=DEBUG
):
    """
    HYBRID consensus: Keep all PRIMARY vectors, but filter residues by cross-fibration agreement.
    
    Philosophy:
    - Don't lose vectors (primary knows best what heights to search)
    - Do filter junk residues (they won't be consistent across fibrations)
    - Abstentions don't vote NO, they just don't vote
    """
    residue_agreement_threshold= consensus_threshold 
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    primary_precomputed = all_precomputed_residues[0]
    primary_deg = fibration_geometries[0].get('disc_deg', 12)
    primary_H = fibration_geometries[0]['H']
    
    # Pre-calculate metadata
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        cd_obj = geom.get('cd')
        try:
            Delta = cd_obj.E_weier.discriminant()
            d_deg = Delta.numerator().degree() if hasattr(Delta, 'numerator') else Delta.degree()
        except:
            d_deg = max(1, 2 * int(cd_obj.a4.degree()))
        fib_meta.append({
            'H': geom['H'],
            'disc_deg': float(max(1, d_deg)),
            'name': geom.get('name', f"fib_{i}")
        })
    
    if debug:
        print(f"\n{'='*70}")
        print(f"HYBRID CONSENSUS: {num_fibs} fibrations")
        print(f"Strategy: Keep all primary vectors, filter residues by agreement")
        print(f"Residue threshold: {residue_agreement_threshold:.1%} of participants")
        print(f"{'='*70}")
    
    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'vectors_kept': 0,
        'vectors_matched_all_fibs': 0,  # Added for backwards compatibility
        'vectors_with_consensus': 0,     # Added for backwards compatibility
        'total_residues_before': 0,
        'total_residues_after': 0,
        'per_vector_participants': [],
        'per_prime_stats': {},
        'blame_height': defaultdict(int),  # Added for backwards compatibility
        'participation_dist': defaultdict(int)  # Added for backwards compatibility
    }
    
    for p in prime_pool:
        consensus_residues[p] = {}
        primary_map = primary_precomputed.get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {'vectors': 0, 'residues_before': 0, 'residues_after': 0}
        
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            stats['total_vectors_primary'] += 1
            prime_stats['vectors'] += 1
            
            # Count original residues
            total_res_before = sum(len(r_list) for r_list in rhs_lists_primary)
            prime_stats['residues_before'] += total_res_before
            
            # Calculate primary height
            try:
                v_primary = vector(QQ, v_primary_tuple)
                hcanon_p = float(v_primary * primary_H * v_primary)
                hcanon_p = max(hcanon_p, 1e-20)
                log_h_primary_norm = math.log(hcanon_p) - math.log(primary_deg)
            except:
                # Keep primary vector as-is if height calculation fails
                consensus_residues[p][v_primary_tuple] = rhs_lists_primary
                stats['vectors_kept'] += 1
                prime_stats['residues_after'] += total_res_before
                continue
            
            # Initialize vote counters for residues
            num_rhs = len(rhs_lists_primary)
            rhs_votes = [Counter() for _ in range(num_rhs)]
            
            # Primary always votes
            for i in range(num_rhs):
                for r in rhs_lists_primary[i]:
                    if isinstance(r, int):
                        rhs_votes[i][r] += 1
            
            participants = 1
            
            # Collect votes from other fibrations
            for fib_idx in range(1, num_fibs):
                other_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_map:
                    continue
                
                H_other = fib_meta[fib_idx]['H']
                deg_other = fib_meta[fib_idx]['disc_deg']
                
                # Find height-matching vector
                best_match_residues = None
                best_diff = float('inf')
                
                for v_o_t, rhs_o in other_map.items():
                    try:
                        v_o = vector(QQ, v_o_t)
                        h_o = max(float(v_o * H_other * v_o), 1e-20)
                        log_h_o = math.log(h_o) - math.log(deg_other)
                        diff = abs(log_h_primary_norm - log_h_o)
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_match_residues = rhs_o
                    except:
                        pass
                
                # If found matching vector, cast votes
                if best_match_residues and best_diff <= height_tolerance_log:
                    participants += 1
                    for i in range(num_rhs):
                        if i < len(best_match_residues):
                            for r in best_match_residues[i]:
                                if isinstance(r, int):
                                    rhs_votes[i][r] += 1
            
            stats['per_vector_participants'].append(participants)
            
            # Apply threshold: keep residues seen by >= threshold * participants
            min_votes = max(1, int(math.ceil(residue_agreement_threshold * participants)))
            
            filtered_rhs_lists = []
            total_res_after = 0
            
            for i in range(num_rhs):
                if participants == 1:
                    # Only primary saw this vector - trust it completely
                    kept = set(rhs_lists_primary[i])
                else:
                    # Multiple fibrations saw it - filter by agreement
                    kept = {r for r, count in rhs_votes[i].items() if count >= min_votes}
                
                filtered_rhs_lists.append(list(kept))
                total_res_after += len(kept)
            
            consensus_residues[p][v_primary_tuple] = filtered_rhs_lists
            stats['vectors_kept'] += 1
            prime_stats['residues_after'] += total_res_after
        
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']
    
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    
    if debug:
        print(f"\nHybrid Consensus Results:")
        print(f"  Vectors: {stats['total_vectors_primary']} primary → {stats['vectors_kept']} kept")
        print(f"  Residues: {stats['total_residues_before']:,} → {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
        
        part_dist = Counter(stats['per_vector_participants'])
        print(f"  Participation distribution:")
        for k in sorted(part_dist.keys()):
            print(f"    {k} fibrations: {part_dist[k]} vectors")
    
    return consensus_residues, stats


@PROFILE
def compute_consensus_residues_with_height_matching(
    all_precomputed_residues,
    fibration_geometries,
    prime_pool,
    consensus_threshold=0.5,
    height_tolerance_log=2.5,
    use_delta_scaling=True,
    debug=DEBUG
):
    """
    Simple height-aware consensus from the theory.
    
    Core idea:
    1. Match vectors by normalized height h_norm = h_canonical / deg(Delta)
    2. If heights match, residues SHOULD match (if section basis spans the point)
    3. Vote on residues - require threshold% agreement
    4. NOT strict intersection (some fibs might not span the point)
    """
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    primary_precomputed = all_precomputed_residues[0]
    
    # Extract geometry info for each fibration
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        cd_obj = geom.get('cd')
        try:
            Delta = cd_obj.E_weier.discriminant()
            d_deg = Delta.numerator().degree() if hasattr(Delta, 'numerator') else Delta.degree()
        except:
            d_deg = max(1, 2 * int(cd_obj.a4.degree()))
        
        fib_meta.append({
            'H': geom['H'],
            'disc_deg': float(max(1, d_deg)),
            'name': geom.get('name', f"fib_{i}"),
            'cd': cd_obj
        })
    
    primary_deg = fib_meta[0]['disc_deg']
    primary_H = fib_meta[0]['H']
    
    # Sample discriminant ratio correction (theory says this is ~1, but check)
    if use_delta_scaling and num_fibs > 1:
        # Sample a few m-values to estimate log|Delta_A(m)| / log|Delta_B(m)|
        # Theory suggests this is stable around 1.0-1.2
        m_samples = [QQ(k) for k in range(-5, 6) if k != 0][:20]
        
        # We'll just use degree ratio for now (sampling is expensive)
        # log_delta_correction = 1.0  # From theory: stable near 1
    
    if debug:
        print(f"\n{'='*70}")
        print(f"HEIGHT-AWARE CONSENSUS: {num_fibs} fibrations")
        print(f"Strategy: Match vectors by height, vote on residues")
        print(f"Height metric: h_norm = h_canonical / deg(Delta)")
        print(f"Tolerance: |log(h_A) - log(h_B)| < {height_tolerance_log}")
        print(f"Voting threshold: {consensus_threshold:.0%}")
        print(f"{'='*70}")
    
    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'vectors_kept': 0,
        'vectors_matched_all_fibs': 0,
        'vectors_with_consensus': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'per_vector_participants': [],
        'per_prime_stats': {},
        'blame_height': defaultdict(int),
        'participation_dist': defaultdict(int)
    }
    
    first = True
    for p in prime_pool:
        consensus_residues[p] = {}
        primary_map = primary_precomputed.get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {'vectors': 0, 'residues_before': 0, 'residues_after': 0}
        
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            if first:
                stats['total_vectors_primary'] += 1
                prime_stats['vectors'] += 1
            
            total_res_before = sum(len(r_list) for r_list in rhs_lists_primary)
            prime_stats['residues_before'] += total_res_before
            
            # Compute PRIMARY's normalized height
            try:
                v_primary = vector(QQ, v_primary_tuple)
                hcanon_p = float(v_primary * primary_H * v_primary)
                hcanon_p = max(hcanon_p, 1e-20)
                log_h_primary_norm = math.log(hcanon_p) - math.log(primary_deg)
            except:
                # Keep if height calc fails
                consensus_residues[p][v_primary_tuple] = rhs_lists_primary
                stats['vectors_kept'] += 1
                stats['vectors_with_consensus'] += 1
                prime_stats['residues_after'] += total_res_before
                stats['per_vector_participants'].append(1)
                stats['participation_dist'][1] += 1
                continue
            
            # Initialize vote counters for each RHS function
            num_rhs = len(rhs_lists_primary)
            rhs_votes = [Counter() for _ in range(num_rhs)]
            
            # Primary always votes for its residues
            for i in range(num_rhs):
                for r in rhs_lists_primary[i]:
                    if isinstance(r, int):
                        rhs_votes[i][r] += 1
            
            participants = 1
            
            # Check each other fibration
            for fib_idx in range(1, num_fibs):
                other_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_map:
                    continue
                
                H_other = fib_meta[fib_idx]['H']
                deg_other = fib_meta[fib_idx]['disc_deg']
                
                # Find vector in this fibration with matching height
                best_match_residues = None
                best_diff = float('inf')
                
                for v_o_t, rhs_o in other_map.items():
                    try:
                        v_o = vector(QQ, v_o_t)
                        h_o = max(float(v_o * H_other * v_o), 1e-20)
                        log_h_o = math.log(h_o) - math.log(deg_other)
                        
                        # Compare normalized heights in log-space
                        diff = abs(log_h_primary_norm - log_h_o)
                        
                        if diff < best_diff:
                            best_diff = diff
                            best_match_residues = rhs_o
                    except:
                        pass
                
                # If found matching vector by height, cast votes for its residues
                if best_match_residues and best_diff <= height_tolerance_log:
                    participants += 1
                    
                    for i in range(num_rhs):
                        if i < len(best_match_residues):
                            for r in best_match_residues[i]:
                                if isinstance(r, int):
                                    rhs_votes[i][r] += 1
                else:
                    # Track why this fibration didn't participate
                    stats['blame_height'][fib_idx] += 1
            
            stats['per_vector_participants'].append(participants)
            stats['participation_dist'][participants] += 1
            
            if participants == num_fibs:
                stats['vectors_matched_all_fibs'] += 1
            
            # Apply voting threshold to residues
            min_votes = max(1, int(math.ceil(consensus_threshold * participants)))
            
            filtered_rhs_lists = []
            total_kept = 0
            
            for i in range(num_rhs):
                kept = {r for r, count in rhs_votes[i].items() if count >= min_votes}
                filtered_rhs_lists.append(list(kept))
                total_kept += len(kept)
            
            if total_kept > 0:
                consensus_residues[p][v_primary_tuple] = filtered_rhs_lists
                stats['vectors_kept'] += 1
                stats['vectors_with_consensus'] += 1
                prime_stats['residues_after'] += total_kept
        
        first = False
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']
    
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    
    if debug:
        print(f"\nConsensus Results:")
        print(f"  Vectors: {stats['total_vectors_primary']} primary → {stats['vectors_kept']} kept")
        print(f"  Matched all fibs: {stats['vectors_matched_all_fibs']}")
        print(f"  Residues: {stats['total_residues_before']:,} → {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
        
        print(f"\n  Participation histogram:")
        for k in sorted(stats['participation_dist'].keys()):
            print(f"    {k}/{num_fibs} fibs saw vector: {stats['participation_dist'][k]} vectors")
        
        if stats['blame_height']:
            print(f"\n  Height mismatches (top 5):")
            for fid, count in sorted(stats['blame_height'].items(), key=lambda x: -x[1])[:5]:
                name = fib_meta[fid]['name']
                print(f"    Fib {fid} ({name}): {count} misses")
    
    return consensus_residues, stats
