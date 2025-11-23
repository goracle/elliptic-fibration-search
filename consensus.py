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
    
    # Track which vectors we've seen to count stats only once per vector
    seen_vectors = set()
    vector_max_participants = {}  # Track max participants seen for each vector
    
    for p in prime_pool:
        consensus_residues[p] = {}
        primary_map = primary_precomputed.get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {'vectors': 0, 'residues_before': 0, 'residues_after': 0}
        
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            # Count this vector only once across all primes
            if v_primary_tuple not in seen_vectors:
                stats['total_vectors_primary'] += 1
                seen_vectors.add(v_primary_tuple)
            
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
            
            # Track participation stats per vector (not per vector-prime pair)
            if v_primary_tuple not in vector_max_participants:
                vector_max_participants[v_primary_tuple] = participants
            else:
                # Update to max participants seen across all primes
                vector_max_participants[v_primary_tuple] = max(
                    vector_max_participants[v_primary_tuple], 
                    participants
                )
            
            # Count vectors matched by all fibs (check at each prime)
            if participants == num_fibs:
                stats['vectors_matched_all_fibs'] += 1
            
            # DECISION: Use PRIMARY + 1 confirmation (require 2 votes minimum)
            # This is lenient enough to catch rare residues where few fibs' sections span the point
            min_votes = 2
            
            filtered_rhs_lists = []
            total_kept = 0
            
            for i in range(num_rhs):
                kept = {r for r, count in rhs_votes[i].items() if count >= min_votes}
                filtered_rhs_lists.append(list(kept))
                total_kept += len(kept)
            
            # If NO residues survived voting, fall back to primary's residues
            if total_kept == 0:
                filtered_rhs_lists = [list(r_list) for r_list in rhs_lists_primary]
                total_kept = sum(len(r_list) for r_list in rhs_lists_primary)
            
            # Only add if we have residues
            if total_kept > 0:
                consensus_residues[p][v_primary_tuple] = filtered_rhs_lists
                prime_stats['residues_after'] += total_kept
                
                # Count this as a kept entry (one per vector-prime pair)
                stats['vectors_kept'] += 1
                stats['vectors_with_consensus'] += 1
        
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']
    
    # Build final participation distribution from per-vector max participants
    for v, max_part in vector_max_participants.items():
        stats['per_vector_participants'].append(max_part)
        stats['participation_dist'][max_part] += 1
    
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
    Height-aware consensus filter.
    
    1. For every vector v in the Primary Fibration:
       - Calculate its normalized canonical height h_norm.
    2. For every Other Fibration:
       - Find the single vector v' that minimizes |log(h_norm) - log(h_norm')|.
       - If error > tolerance, this fibration ABSTAINS.
    3. Collect residues from Primary v and all matching Other v'.
    4. Keep residues that appear in >= threshold fraction of participating fibrations.
    """
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    # 1. Metadata Setup
    primary_residues_map = all_precomputed_residues[0]
    primary_geom = fibration_geometries[0]
    
    # Pre-calculate geometry constants to speed up loop
    # Structure: [ {'H': matrix, 'deg': float, 'log_deg': float}, ... ]
    fib_constants = []
    for geom in fibration_geometries:
        d = float(max(1, geom['disc_deg']))
        fib_constants.append({
            'H': geom['H'],
            'deg': d,
            'log_deg': math.log(d)
        })

    # Helper to calc log normalized height
    def get_log_norm_height(v_tup, consts):
        try:
            # Check zero vector explicitly to avoid log(0) error
            if all(c == 0 for c in v_tup):
                return -9999.0 # Arbitrary small number representing 0 height
            
            v = vector(QQ, v_tup)
            h_can = float(v * consts['H'] * v)
            if h_can <= 1e-20: 
                return -9999.0
            return math.log(h_can) - consts['log_deg']
        except Exception:
            return None

    consensus_residues = {}
    
    # Stats containers
    unique_vectors_processed = set()
    vectors_with_consensus_kept = set()
    
    stats = {
        'total_vectors_primary': 0, # Unique vector tuples
        'total_residues_before': 0,
        'total_residues_after': 0,
        'vectors_kept': 0, # (prime, vector) pairs kept
        'participation_dist': defaultdict(int),
        'height_match_failures': defaultdict(int)
    }

    # 2. Main Loop over Primes
    for p in prime_pool:
        consensus_residues[p] = {}
        
        # If primary has no data for this prime, skip
        if p not in primary_residues_map:
            continue
            
        primary_vec_map = primary_residues_map[p]
        
        for v_prim_tuple, rhs_lists_primary in primary_vec_map.items():
            
            # --- Stats Counting ---
            if v_prim_tuple not in unique_vectors_processed:
                unique_vectors_processed.add(v_prim_tuple)
                stats['total_vectors_primary'] += 1
            
            # Calculate stats "Before"
            count_before = sum(len(s) for s in rhs_lists_primary)
            stats['total_residues_before'] += count_before
            
            # --- Identify Matching Vectors in Other Fibrations ---
            # We rely on the geometry (H), not the prime.
            log_h_prim = get_log_norm_height(v_prim_tuple, fib_constants[0])
            
            if log_h_prim is None:
                # If height calc fails, we can't match. Keep primary residues (conservative)
                consensus_residues[p][v_prim_tuple] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
                continue

            # We will collect sets of residues for each RHS index
            # distinct_rhs_votes[rhs_index] = Counter(residue -> count)
            num_rhs = len(rhs_lists_primary)
            distinct_rhs_votes = [Counter() for _ in range(num_rhs)]
            
            # Primary always votes
            participants = 1
            for i in range(num_rhs):
                for r in rhs_lists_primary[i]:
                    if isinstance(r, int):
                        distinct_rhs_votes[i][r] += 1

            # Check other fibrations
            for fib_idx in range(1, num_fibs):
                other_residues_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_residues_map:
                    continue # This fibration has no data for this prime, it abstains
                
                # --- Find Best Height Match ---
                # Note: Ideally we cache this mapping outside the prime loop, 
                # but vecs might change per prime if LLL was unstable. 
                # Assuming vectors are consistent enough or small enough to scan.
                
                best_v_match = None
                best_diff = float('inf')
                
                consts_other = fib_constants[fib_idx]
                
                # Scan all vectors in this fibration to find the height partner
                for v_other_tuple in other_residues_map.keys():
                    log_h_other = get_log_norm_height(v_other_tuple, consts_other)
                    if log_h_other is None: continue
                    
                    diff = abs(log_h_prim - log_h_other)
                    if diff < best_diff:
                        best_diff = diff
                        best_v_match = v_other_tuple
                
                # Did we find a valid geometric match?
                if best_v_match is not None and best_diff < height_tolerance_log:
                    # Valid match found, collect votes
                    participants += 1
                    matched_rhs_lists = other_residues_map[best_v_match]
                    
                    for i in range(min(num_rhs, len(matched_rhs_lists))):
                        for r in matched_rhs_lists[i]:
                            if isinstance(r, int):
                                distinct_rhs_votes[i][r] += 1
                else:
                    stats['height_match_failures'][fib_idx] += 1

            # --- Consensus Logic ---
            # Calculate dynamic threshold based on ACTUAL participants
            # (e.g. if 5 fibrations didn't have data for this prime, don't require 7 votes)
            req_votes = max(1, int(math.ceil(consensus_threshold * participants)))
            
            # If strict intersection is desired (threshold 1.0), ensure req_votes == participants
            if consensus_threshold >= 0.99:
                req_votes = participants

            final_rhs_lists = []
            has_any_residues = False
            
            for i in range(num_rhs):
                kept_residues = []
                for r, count in distinct_rhs_votes[i].items():
                    if count >= req_votes:
                        kept_residues.append(r)
                
                # Sort for determinism
                kept_residues.sort()
                final_rhs_lists.append(kept_residues)
                if kept_residues:
                    has_any_residues = True
            
            # Store results if anything survived
            if has_any_residues:
                consensus_residues[p][v_prim_tuple] = final_rhs_lists
                stats['total_residues_after'] += sum(len(x) for x in final_rhs_lists)
                stats['vectors_kept'] += 1
                
                if v_prim_tuple not in vectors_with_consensus_kept:
                    vectors_with_consensus_kept.add(v_prim_tuple)
                    stats['participation_dist'][participants] += 1

    # Final Stats Calculation
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    
    stats['vectors_matched_all_fibs'] = stats['participation_dist'].get(num_fibs, 0)
    stats['vectors_with_consensus'] = len(vectors_with_consensus_kept)

    if debug:
        print(f"\nConsensus Results:")
        print(f"  Vectors: {stats['total_vectors_primary']} primary -> {stats['vectors_with_consensus']} kept")
        print(f"  Matched all {num_fibs} fibs: {stats['vectors_matched_all_fibs']}")
        print(f"  Residues: {stats['total_residues_before']:,} -> {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}") # Positive % means reduction
        
    return consensus_residues, stats


