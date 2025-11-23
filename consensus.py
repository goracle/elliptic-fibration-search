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
def compute_consensus_residues_with_height_matching( all_precomputed_residues,
                                                     fibration_geometries, prime_pool,
                                                     consensus_threshold=0.8,
                                                     height_tolerance_log=2.5,
                                                     use_delta_scaling=True,
                                                     debug=DEBUG, target_x=None, r_m=None, shift=QQ(0) ):
    """ Strict Height-Matching Consensus with Non-Empty Intersection Guard.

    Strategy:
    1. Iterate through vectors in the Primary Fibration.
    2. For each Partner Fibration, find the vector with the closest normalized canonical height.
    3. If a partner vector is found:
    - Compute Intersection(PrimaryResidues, PartnerResidues).
    - IF Intersection is NOT EMPTY: Update Primary with Intersection (Filtering).
    - IF Intersection IS EMPTY: Keep Primary (Assuming Partner is Blind/Inconsistent).

    This guarantees that we never ADD junk (unlike Union) and never DELETE a vector purely 
    due to a partner having disjoint residues (unlike Strict Intersection).
    """
    import math
    from collections import Counter, defaultdict
    from sage.all import vector, QQ

    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}

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

    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'vectors_matched_all_fibs': 0,
        'vectors_with_consensus': 0,
        'reduction_ratio': 0.0,
        'intersections_applied': 0,
        'empty_intersections_ignored': 0
    }

    # We iterate per prime, then per vector
    for p in prime_pool:
        consensus_residues[p] = {}
        if p not in primary_residues_map: 
            continue

        for v_prim, rhs_lists_primary in primary_residues_map[p].items():
            stats['total_vectors_primary'] += 1 if p == prime_pool[0] else 0 

            count_before = sum(len(s) for s in rhs_lists_primary)
            stats['total_residues_before'] += count_before

            # Calculate Height of Primary Vector
            log_h_prim = get_log_norm_height(v_prim, fib_constants[0])

            # Start with the Primary's residues
            current_consensus_lists = [set(s) for s in rhs_lists_primary]
            partners_found = 0

            # Attempt to match with other fibrations
            for fib_idx in range(1, num_fibs):
                fib_res_map = all_precomputed_residues[fib_idx]
                if p not in fib_res_map:
                    continue 

                # Find best height match in this fibration
                best_match_vec = None
                min_diff = float('inf')

                # Scan all vectors in this fibration for this prime
                for v_cand in fib_res_map[p].keys():
                    log_h_cand = get_log_norm_height(v_cand, fib_constants[fib_idx])
                    if log_h_prim is None or log_h_cand is None: 
                        continue

                    diff = abs(log_h_prim - log_h_cand)
                    if diff < min_diff:
                        min_diff = diff
                        best_match_vec = v_cand

                # If we found a valid match within tolerance
                if best_match_vec and min_diff < height_tolerance_log:
                    partners_found += 1
                    partner_rhs_lists = fib_res_map[p][best_match_vec]

                    # Guarded Intersection:
                    # For each polynomial root index
                    for i in range(len(current_consensus_lists)):
                        if i < len(partner_rhs_lists):
                            # Calculate intersection
                            common = current_consensus_lists[i].intersection(partner_rhs_lists[i])

                            # SAFETY CHECK: Only apply intersection if it is NOT empty
                            # If empty, we assume the partner is inconsistent/blind and ignore it
                            # to prevent killing the Primary data.
                            if len(common) > 0:
                                current_consensus_lists[i] = common
                                stats['intersections_applied'] += 1
                            else:
                                stats['empty_intersections_ignored'] += 1

            final_lists = [sorted(list(s)) for s in current_consensus_lists]
            consensus_residues[p][v_prim] = final_lists

            count_after = sum(len(s) for s in final_lists)
            stats['total_residues_after'] += count_after

            if partners_found == (num_fibs - 1):
                stats['vectors_matched_all_fibs'] += 1 if p == prime_pool[0] else 0

    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])

    stats['vectors_with_consensus'] = stats['total_vectors_primary']

    if debug:
        print(f"\nConsensus Stats:")
        print(f"  Applied Intersections: {stats['intersections_applied']}")
        print(f"  Ignored Empty Intersections: {stats['empty_intersections_ignored']}")
        print(f"  Residues: {stats['total_residues_before']} -> {stats['total_residues_after']}")

    return consensus_residues, stats
