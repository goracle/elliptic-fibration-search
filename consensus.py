from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict


# DEPRECATED / COMPATIBILITY FUNCTIONS
def compute_consensus_residues(precomputed_residues_list, prime_pool, consensus_threshold=0.7, debug=False):
    print("WARNING: calling deprecated compute_consensus_residues")
    return {}, {}

@PROFILE
def evaluate_log_delta_ratio(cd_primary, cd_other, m_values):
    """ Evaluate log(|Δ_other(m)| / |Δ_primary(m)|) for each m, then average. """
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
            if QQ(Delta_p_val) != 0 and QQ(Delta_o_val) != 0:
                log_ratio = math.log(abs(float(QQ(Delta_o_val)))) - math.log(abs(float(QQ(Delta_p_val))))
                log_ratios.append(log_ratio)
        except Exception:
            continue
    
    if not log_ratios: return 0.0
    return sum(log_ratios) / len(log_ratios)

@PROFILE
def sample_delta_ratios(cd, sections, num_samples=200, m_range=(-1000, 1000), seed=None):
    if seed is not None: set_random_seed(seed)
    samples = []
    m_sym = cd.a4.parent().gen()
    try:
        delta_m = cd.E_weier.discriminant()
    except Exception:
        try: delta_m = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
        except Exception: return [] 
    
    attempts = 0
    while len(samples) < num_samples and attempts < num_samples * 10:
        attempts += 1
        m_val = QQ(ZZ.random_element(m_range[0], m_range[1])) / QQ(ZZ.random_element(1, 100))
        if m_val in cd.bad_primes: continue
        try:
            delta_val = QQ(delta_m.subs({m_sym: m_val}))
            if delta_val != 0: samples.append(float(math.log(abs(float(delta_val)))))
        except Exception: continue
    return samples


def get_log_norm_height(v_tup, consts):
    """Helper to calculate log normalized height."""
    try:
        if all(c == 0 for c in v_tup): return -9999.0
        v = vector(QQ, v_tup)
        h_can = float(v * consts['H'] * v)
        if h_can <= 1e-20: return -9999.0
        return math.log(h_can) - consts['log_deg']
    except Exception:
        return None

@PROFILE
def compute_consensus_residues_with_height_matching(all_precomputed_residues,
                                                    fibration_geometries, prime_pool,
                                                    consensus_threshold=0.5,
                                                    height_tolerance_log=2.5,
                                                    use_delta_scaling=True,
                                                    debug=DEBUG, target_x=None, r_m=None, shift=QQ(0)):
    """ 
    Height-Matching Consensus: VETO MODE.

    Problem: Many fibrations are "blind" to the true point at a given prime p 
             (no vector found at height).
    
    Solution:
    1. Iterate vectors in Primary Fibration.
    2. Check Partner Fibrations for vectors at the matching normalized height.
    3. Three outcomes per partner:
       a. ABSTAIN (No vector at height): Ignore. (Assumed blind).
       b. CONFIRM (Vector at height + Residue Match): Good.
       c. VETO (Vector at height + Residue Mismatch): Bad. (Implies Primary vector is a ghost).
    
    Decision:
    - Keep vector IF (Vetos == 0) OR (Confirms > Vetos).
    - Default is KEEP (to fix "too aggressive" pruning of blind points).
    """
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0: return {}, {}

    primary_residues_map = all_precomputed_residues[0]

    # Pre-calculate geometry constants
    fib_constants = []
    for geom in fibration_geometries:
        d = float(max(1, geom['disc_deg']))
        fib_constants.append({'H': geom['H'], 'deg': d, 'log_deg': math.log(d)})

    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0, 'total_residues_before': 0, 'total_residues_after': 0,
        'vectors_kept': 0, 'vectors_vetoed': 0,
        'partners_abstain': 0, 'partners_confirm': 0, 'partners_veto': 0,
        'vectors_matched_all_fibs': 0
    }
    
    if debug:
        print(f"Consensus Config: VETO MODE (Filtering ghosts, preserving blind points)")

    for p in prime_pool:
        consensus_residues[p] = {}
        if p not in primary_residues_map: continue

        for v_prim, rhs_lists_primary in primary_residues_map[p].items():
            stats['total_vectors_primary'] += 1
            count_before = sum(len(s) for s in rhs_lists_primary)
            stats['total_residues_before'] += count_before

            log_h_prim = get_log_norm_height(v_prim, fib_constants[0])
            
            # Status counters for this vector
            confirmations = 0
            vetos = 0
            abstentions = 0

            # Check partners
            for fib_idx in range(1, num_fibs):
                fib_res_map = all_precomputed_residues[fib_idx]
                if p not in fib_res_map: 
                    abstentions += 1
                    continue

                # Scan vectors in partner fibration for a height match
                best_match_vecs = []
                
                # Check ALL vectors in partner for height match
                for v_cand in fib_res_map[p].keys():
                    log_h_cand = get_log_norm_height(v_cand, fib_constants[fib_idx])
                    if log_h_prim is None or log_h_cand is None: continue
                    
                    diff = abs(log_h_prim - log_h_cand)
                    if diff < height_tolerance_log:
                        best_match_vecs.append(v_cand)

                if not best_match_vecs:
                    # Case A: ABSTAIN (Partner found nothing at this height)
                    abstentions += 1
                else:
                    # Partner found something at this height. Does residue match?
                    # Since m -> x expansion is identical, residues must overlap.
                    
                    # Flatten primary residues for checking
                    prim_flat = set()
                    for s in rhs_lists_primary:
                        prim_flat.update(s)
                    
                    # Check if ANY candidate vector overlaps with ANY primary residue
                    overlap_found = False
                    for v_match in best_match_vecs:
                        cand_lists = fib_res_map[p][v_match]
                        cand_flat = set()
                        for s in cand_lists:
                            cand_flat.update(s)
                        
                        if not prim_flat.isdisjoint(cand_flat):
                            overlap_found = True
                            break
                    
                    if overlap_found:
                        # Case B: CONFIRM (Height match + Residue match)
                        confirmations += 1
                    else:
                        # Case C: VETO (Height match BUT Residue mismatch)
                        # This implies the Primary vector corresponds to a height 
                        # coincident with a Partner vector, but they map to different x.
                        vetos += 1

            stats['partners_abstain'] += abstentions
            stats['partners_confirm'] += confirmations
            stats['partners_veto'] += vetos

            # Track if confirmed by ALL partners (strict match)
            if num_fibs > 1 and confirmations == (num_fibs - 1):
                stats['vectors_matched_all_fibs'] += 1

            # --- DECISION LOGIC ---
            # If we have VETOS, it's likely a ghost/collision, unless we have overwhelming confirmation.
            # If we have only ABSTENTIONS, we MUST KEEP (blindness).
            
            is_good = True
            
            if vetos > 0:
                # If verified veto (partner explicitly disagrees), discard unless confirmed by others
                if vetos >= confirmations:
                    is_good = False
            
            if is_good:
                consensus_residues[p][v_prim] = rhs_lists_primary
                stats['total_residues_after'] += count_before
                stats['vectors_kept'] += 1
            else:
                stats['vectors_vetoed'] += 1

    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])

    stats['vectors_with_consensus'] = stats['vectors_kept']

    if debug:
        print(f"\nConsensus Stats (Veto Mode):")
        print(f"  Vectors Kept: {stats['vectors_kept']}")
        print(f"  Vectors Vetoed: {stats['vectors_vetoed']}")
        print(f"  Partners: {stats['partners_abstain']} abstain, {stats['partners_confirm']} confirm, {stats['partners_veto']} veto")
        print(f"  Residues: {stats['total_residues_before']} -> {stats['total_residues_after']}")

    return consensus_residues, stats



