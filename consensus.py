from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict






# DEPRECATED / COMPATIBILITY FUNCTIONS (Kept to avoid import errors)






# Deprecated function kept for compatibility
def compute_consensus_residues(precomputed_residues_list, prime_pool, consensus_threshold=0.7, debug=False):
    print("WARNING: calling deprecated compute_consensus_residues")
    return {}, {}




























@PROFILE
def evaluate_log_delta_ratio(cd_primary, cd_other, m_values):
    """
    Evaluate log(|Δ_other(m)| / |Δ_primary(m)|) for each m, then average.
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
    Sample log|Δ| values at random rational m-values.
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
            print(f"FATAL: Could not get discriminant for {cd}")
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
    Height-aware consensus filter with ROBUST FINGERPRINTING and SAFETY FALLBACK.
    
    Key Updates:
    1. Fingerprinting: Pairs vectors based on Height AND Residue Jaccard Similarity.
    2. Voting: Partners only vote if they have data (empty = abstain).
    3. SAFETY FALLBACK: If consensus results in an empty set for a vector that
       had data in the Primary, we RESTORE the Primary's data.
       This prevents "veto by confusion" (false matches killing valid points).
    """
    import math
    from collections import Counter, defaultdict
    from sage.all import vector, QQ

    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    # 1. Metadata Setup
    primary_residues_map = all_precomputed_residues[0]
    
    # Pre-calculate geometry constants
    fib_constants = []
    for geom in fibration_geometries:
        d = float(max(1, geom['disc_deg']))
        fib_constants.append({
            'H': geom['H'],
            'deg': d,
            'log_deg': math.log(d)
        })

    def get_log_norm_height(v_tup, consts):
        try:
            if all(c == 0 for c in v_tup): return -9999.0
            v = vector(QQ, v_tup)
            h_can = float(v * consts['H'] * v)
            if h_can <= 1e-20: return -9999.0
            return math.log(h_can) - consts['log_deg']
        except Exception:
            return None

    # Define "Fingerprint Primes" dynamically - pick primes where Primary actually has data
    fingerprint_primes = []
    for p in prime_pool:
        if p in primary_residues_map and primary_residues_map[p]:
            fingerprint_primes.append(p)
        if len(fingerprint_primes) >= 5:
            break
            
    if not fingerprint_primes:
        fingerprint_primes = [p for p in prime_pool if p >= 5][:5]

    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'vectors_kept': 0,
        'partners_found': defaultdict(int),
        'false_matches_rejected': 0,
        'fallback_restorations': 0
    }

    # 2. Identify Matches (Fingerprinting)
    primary_vectors = set()
    for p in prime_pool:
        if p in primary_residues_map:
            primary_vectors.update(primary_residues_map[p].keys())
            
    primary_vectors = sorted(list(primary_vectors))
    vector_matches = defaultdict(list)
    
    if debug:
        print(f"  [Consensus] Fingerprinting {len(primary_vectors)} primary vectors using primes {fingerprint_primes}...")
    
    for v_prim in primary_vectors:
        stats['total_vectors_primary'] += 1
        log_h_prim = get_log_norm_height(v_prim, fib_constants[0])
        if log_h_prim is None: continue

        # For each other fibration, find the best PARTNER vector
        for fib_idx in range(1, num_fibs):
            best_match = None
            best_score = -1.0
            
            fib_res_map = all_precomputed_residues[fib_idx]
            
            # Find a sample prime where this fibration has data
            candidate_vectors = []
            for p_check in fingerprint_primes:
                if p_check in fib_res_map:
                    candidate_vectors = list(fib_res_map[p_check].keys())
                    break
            if not candidate_vectors and fib_res_map:
                 # Fallback to first available key
                 first_p = next(iter(fib_res_map))
                 candidate_vectors = list(fib_res_map[first_p].keys())

            for v_cand in candidate_vectors:
                # 1. Height Check (Loose Filter)
                log_h_cand = get_log_norm_height(v_cand, fib_constants[fib_idx])
                if log_h_cand is None: continue
                
                if abs(log_h_prim - log_h_cand) > height_tolerance_log:
                    continue
                
                # 2. Residue Correlation (Strict Filter)
                intersection_count = 0
                union_count = 0
                
                for fp in fingerprint_primes:
                    r_prim = set()
                    if fp in primary_residues_map and v_prim in primary_residues_map[fp]:
                        for rhs_l in primary_residues_map[fp][v_prim]:
                            r_prim.update(rhs_l)
                            
                    r_cand = set()
                    if fp in fib_res_map and v_cand in fib_res_map[fp]:
                        for rhs_l in fib_res_map[fp][v_cand]:
                            r_cand.update(rhs_l)
                    
                    if not r_prim and not r_cand: continue
                    
                    common = r_prim.intersection(r_cand)
                    total = r_prim.union(r_cand)
                    
                    intersection_count += len(common)
                    union_count += len(total)
                
                if union_count == 0:
                    score = 0.0
                else:
                    score = intersection_count / float(union_count)
                
                if score > best_score:
                    best_score = score
                    best_match = v_cand

            # THRESHOLD: STRICTER (0.6) to prevent noise-matching
            if best_match and best_score > 0.6:
                vector_matches[v_prim].append((fib_idx, best_match))
            elif best_match:
                stats['false_matches_rejected'] += 1

        stats['partners_found'][len(vector_matches[v_prim])] += 1

    # 3. Generate Consensus Residues
    for p in prime_pool:
        consensus_residues[p] = {}
        if p not in primary_residues_map: continue
        
        for v_prim, rhs_lists_primary in primary_residues_map[p].items():
            count_before = sum(len(s) for s in rhs_lists_primary)
            stats['total_residues_before'] += count_before
            
            partners = vector_matches.get(v_prim, [])
            
            # If no partners, keep primary (Abstention Logic)
            if not partners:
                consensus_residues[p][v_prim] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
                continue
                
            num_rhs = len(rhs_lists_primary)
            
            final_rhs_lists = []
            has_any_consensus = False
            
            # Process each RHS index
            for i in range(num_rhs):
                residue_votes = Counter()
                # Primary Vote
                current_valid_voters = 1
                for r in rhs_lists_primary[i]:
                    residue_votes[r] += 1
                
                # Partner Votes
                for fib_idx, v_match in partners:
                    fib_res = all_precomputed_residues[fib_idx]
                    if p in fib_res and v_match in fib_res[p]:
                        matched_rhs_lists = fib_res[p][v_match]
                        if i < len(matched_rhs_lists):
                            partner_list = matched_rhs_lists[i]
                            # An empty list means "I don't know" (blind), not "No" (veto)
                            if partner_list:
                                current_valid_voters += 1
                                for r in partner_list:
                                    residue_votes[r] += 1
                
                # Consensus Decision
                req_votes = max(1, int(math.ceil(consensus_threshold * current_valid_voters)))
                
                kept = []
                for r, count in residue_votes.items():
                    if count >= req_votes:
                        kept.append(r)
                kept.sort()
                final_rhs_lists.append(kept)
                if kept: has_any_consensus = True
            
            # SAFETY FALLBACK: 
            # If we have partners but the result is EMPTY, it likely means 
            # the partners were false matches or confused. 
            # We restore the Primary data to avoid deleting valid points.
            if not has_any_consensus and count_before > 0:
                consensus_residues[p][v_prim] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
                stats['fallback_restorations'] += 1
            elif has_any_consensus:
                consensus_residues[p][v_prim] = final_rhs_lists
                stats['total_residues_after'] += sum(len(x) for x in final_rhs_lists)
                stats['vectors_kept'] += 1

    # Final Stats
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
        
    stats['vectors_matched_all_fibs'] = stats['partners_found'].get(num_fibs-1, 0)
    stats['vectors_with_consensus'] = len(vector_matches)

    if debug:
        print(f"\nConsensus Stats:")
        print(f"  Fingerprinting rejected {stats['false_matches_rejected']} height-matches due to low correlation.")
        print(f"  Restored Primary data (fallback) for {stats['fallback_restorations']} vector-prime pairs.")
        print(f"  Vectors matched across ALL {num_fibs} fibs: {stats['vectors_matched_all_fibs']}")
        
    return consensus_residues, stats


from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict

@PROFILE
def evaluate_log_delta_ratio(cd_primary, cd_other, m_values):
    """
    Evaluate log(|Δ_other(m)| / |Δ_primary(m)|) for each m, then average.
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
    Sample log|Δ| values at random rational m-values.
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
            print(f"FATAL: Could not get discriminant for {cd}")
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
    Height-aware consensus filter with ROBUST FINGERPRINTING and SAFETY FALLBACK.
    
    Key Updates:
    1. Fingerprinting: Pairs vectors based on Height AND Residue Jaccard Similarity.
    2. Voting: Partners only vote if they have data (empty = abstain).
    3. SAFETY FALLBACK: If consensus results in an empty set for a vector that
       had data in the Primary, we RESTORE the Primary's data.
       This prevents "veto by confusion" (false matches killing valid points).
    """
    import math
    from collections import Counter, defaultdict
    from sage.all import vector, QQ

    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    # 1. Metadata Setup
    primary_residues_map = all_precomputed_residues[0]
    
    # Pre-calculate geometry constants
    fib_constants = []
    for geom in fibration_geometries:
        d = float(max(1, geom['disc_deg']))
        fib_constants.append({
            'H': geom['H'],
            'deg': d,
            'log_deg': math.log(d)
        })

    def get_log_norm_height(v_tup, consts):
        try:
            if all(c == 0 for c in v_tup): return -9999.0
            v = vector(QQ, v_tup)
            h_can = float(v * consts['H'] * v)
            if h_can <= 1e-20: return -9999.0
            return math.log(h_can) - consts['log_deg']
        except Exception:
            return None

    # Define "Fingerprint Primes" dynamically - pick primes where Primary actually has data
    fingerprint_primes = []
    for p in prime_pool:
        if p in primary_residues_map and primary_residues_map[p]:
            fingerprint_primes.append(p)
        if len(fingerprint_primes) >= 5:
            break
            
    if not fingerprint_primes:
        fingerprint_primes = [p for p in prime_pool if p >= 5][:5]

    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'vectors_kept': 0,
        'partners_found': defaultdict(int),
        'false_matches_rejected': 0,
        'fallback_restorations': 0
    }

    # 2. Identify Matches (Fingerprinting)
    primary_vectors = set()
    for p in prime_pool:
        if p in primary_residues_map:
            primary_vectors.update(primary_residues_map[p].keys())
            
    primary_vectors = sorted(list(primary_vectors))
    vector_matches = defaultdict(list)
    
    if debug:
        print(f"  [Consensus] Fingerprinting {len(primary_vectors)} primary vectors using primes {fingerprint_primes}...")
    
    for v_prim in primary_vectors:
        stats['total_vectors_primary'] += 1
        log_h_prim = get_log_norm_height(v_prim, fib_constants[0])
        if log_h_prim is None: continue

        # For each other fibration, find the best PARTNER vector
        for fib_idx in range(1, num_fibs):
            best_match = None
            best_score = -1.0
            
            fib_res_map = all_precomputed_residues[fib_idx]
            
            # Find a sample prime where this fibration has data
            candidate_vectors = []
            for p_check in fingerprint_primes:
                if p_check in fib_res_map:
                    candidate_vectors = list(fib_res_map[p_check].keys())
                    break
            if not candidate_vectors and fib_res_map:
                 # Fallback to first available key
                 first_p = next(iter(fib_res_map))
                 candidate_vectors = list(fib_res_map[first_p].keys())

            for v_cand in candidate_vectors:
                # 1. Height Check (Loose Filter)
                log_h_cand = get_log_norm_height(v_cand, fib_constants[fib_idx])
                if log_h_cand is None: continue
                
                if abs(log_h_prim - log_h_cand) > height_tolerance_log:
                    continue
                
                # 2. Residue Correlation (Strict Filter)
                intersection_count = 0
                union_count = 0
                
                for fp in fingerprint_primes:
                    r_prim = set()
                    if fp in primary_residues_map and v_prim in primary_residues_map[fp]:
                        for rhs_l in primary_residues_map[fp][v_prim]:
                            r_prim.update(rhs_l)
                            
                    r_cand = set()
                    if fp in fib_res_map and v_cand in fib_res_map[fp]:
                        for rhs_l in fib_res_map[fp][v_cand]:
                            r_cand.update(rhs_l)
                    
                    if not r_prim and not r_cand: continue
                    
                    common = r_prim.intersection(r_cand)
                    total = r_prim.union(r_cand)
                    
                    intersection_count += len(common)
                    union_count += len(total)
                
                if union_count == 0:
                    score = 0.0
                else:
                    score = intersection_count / float(union_count)
                
                if score > best_score:
                    best_score = score
                    best_match = v_cand

            # THRESHOLD: 0.3 to allow some noise but reject randoms
            if best_match and best_score > 0.3:
                vector_matches[v_prim].append((fib_idx, best_match))
            elif best_match:
                stats['false_matches_rejected'] += 1

        stats['partners_found'][len(vector_matches[v_prim])] += 1

    # 3. Generate Consensus Residues
    for p in prime_pool:
        consensus_residues[p] = {}
        if p not in primary_residues_map: continue
        
        for v_prim, rhs_lists_primary in primary_residues_map[p].items():
            count_before = sum(len(s) for s in rhs_lists_primary)
            stats['total_residues_before'] += count_before
            
            partners = vector_matches.get(v_prim, [])
            
            # If no partners, keep primary (Abstention Logic)
            if not partners:
                consensus_residues[p][v_prim] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
                continue
                
            num_rhs = len(rhs_lists_primary)
            
            final_rhs_lists = []
            has_any_consensus = False
            
            # Process each RHS index
            for i in range(num_rhs):
                residue_votes = Counter()
                # Primary Vote
                current_valid_voters = 1
                for r in rhs_lists_primary[i]:
                    residue_votes[r] += 1
                
                # Partner Votes
                for fib_idx, v_match in partners:
                    fib_res = all_precomputed_residues[fib_idx]
                    if p in fib_res and v_match in fib_res[p]:
                        matched_rhs_lists = fib_res[p][v_match]
                        if i < len(matched_rhs_lists):
                            partner_list = matched_rhs_lists[i]
                            # An empty list means "I don't know" (blind), not "No" (veto)
                            if partner_list:
                                current_valid_voters += 1
                                for r in partner_list:
                                    residue_votes[r] += 1
                
                # Consensus Decision
                req_votes = max(1, int(math.ceil(consensus_threshold * current_valid_voters)))
                
                kept = []
                for r, count in residue_votes.items():
                    if count >= req_votes:
                        kept.append(r)
                kept.sort()
                final_rhs_lists.append(kept)
                if kept: has_any_consensus = True
            
            # SAFETY FALLBACK: 
            # If we have partners but the result is EMPTY, it likely means 
            # the partners were false matches or confused. 
            # We restore the Primary data to avoid deleting valid points.
            if not has_any_consensus and count_before > 0:
                consensus_residues[p][v_prim] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
                stats['fallback_restorations'] += 1
            elif has_any_consensus:
                consensus_residues[p][v_prim] = final_rhs_lists
                stats['total_residues_after'] += sum(len(x) for x in final_rhs_lists)
                stats['vectors_kept'] += 1

    # Final Stats
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
        
    stats['vectors_matched_all_fibs'] = stats['partners_found'].get(num_fibs-1, 0)
    stats['vectors_with_consensus'] = len(vector_matches)

    if debug:
        print(f"\nConsensus Stats:")
        print(f"  Fingerprinting rejected {stats['false_matches_rejected']} height-matches due to low correlation.")
        print(f"  Restored Primary data (fallback) for {stats['fallback_restorations']} vector-prime pairs.")
        print(f"  Vectors matched across ALL {num_fibs} fibs: {stats['vectors_matched_all_fibs']}")
        
    return consensus_residues, stats
