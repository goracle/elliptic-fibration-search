from sage.all import *
from search_common import *

):
    """
    Height-aware consensus filter that tries to match residues across fibrations
    by comparing normalized canonical heights in log-space (and optionally scaling
    by a per-fibration log|Δ| factor estimated from samples).

    Raises informative exceptions on invalid inputs (no silent failures).
    Returns: consensus_residues: {p: {v_primary_tuple: [consensus_set_per_rhs]}}
    """

    from collections import Counter
    import math

    # --- input validation ---
    if not isinstance(all_precomputed_residues, (list, tuple)) or not all_precomputed_residues:
        raise ValueError("all_precomputed_residues must be a non-empty list of per-fibration residue maps")

    if not isinstance(fibration_geometries, (list, tuple)) or len(fibration_geometries) != len(all_precomputed_residues):
        raise ValueError("fibration_geometries must be a list of same length as all_precomputed_residues")

    num_fibs = len(all_precomputed_residues)
    if not 0 < consensus_threshold <= 1.0:
        raise ValueError("consensus_threshold must be in (0,1]")

    if height_tolerance_log <= 0:
        raise ValueError("height_tolerance_log must be positive")

    # quick helpers
    def safe_float(x):
        try:
            return float(x)
        except Exception as e:
            raise TypeError(f"expected numeric value, got {type(x)}: {x}") from e

    # --- build per-fibration scaling factor for heights ---
    # If delta_samples provided, use median(log|Δ|) ratio relative to primary; else fall back to deg ratio
    primary_geom = fibration_geometries[0]
    if 'deg' not in primary_geom or 'H' not in primary_geom:
        raise KeyError("primary fibration geometry must contain 'H' and 'deg'")

    # compute per-fibration log-Δ scaling factor s_f such that
    # adjusted_height_f = raw_height_f * s_f  (we will compare logs so add log(s_f))
    log_delta_multipliers = []
    try:
        # If user supplied delta_samples (list of log|Δ| values) use medians
        primary_ld = None
        if use_delta_scaling and primary_geom.get('delta_samples'):
            primary_ld = float(median(primary_geom['delta_samples']))
        per_fib_logmult = []
        for geom in fibration_geometries:
            if 'H' not in geom or 'deg' not in geom:
                raise KeyError("each fibration geometry dict must contain 'H' and 'deg'")
            ld = None
            if use_delta_scaling and geom.get('delta_samples'):
                ld = float(median(geom['delta_samples']))
            per_fib_logmult.append(ld)
        # if we have primary_ld and per-fib, convert to additive adjustments:
        if use_delta_scaling and primary_ld is not None:
            log_adjust = []
            for ld in per_fib_logmult:
                if ld is None:
                    # missing sample for this fibration -> fallback to deg ratio
                    log_adjust.append(math.log(float(geom['deg'])/float(primary_geom['deg'])))
                else:
                    log_adjust.append(ld - primary_ld + math.log(float(geom['deg'])/float(primary_geom['deg'])))
            delta_log_mult = log_adjust
        else:
            # fallback: use degree ratio only (log-space)
            delta_log_mult = [math.log(float(geom['deg'])/float(primary_geom['deg'])) for geom in fibration_geometries]

    except Exception:
        raise

    # Precompute needed metadata to avoid repeated work
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        H = geom['H']
        deg = int(geom['deg'])
        name = geom.get('name', f"fib_{i}")
        log_delta_adj = delta_log_mult[i]
        # Validate H shape vs vectors later (we can't do full check here)
        fib_meta.append({'H': H, 'deg': deg, 'name': name, 'log_delta_adj': log_delta_adj})

    consensus_residues = {}

    # Number of votes needed
    min_votes = max(1, int(math.ceil(consensus_threshold * num_fibs)))

    # iterate primes
    for p in prime_pool:
        consensus_residues[p] = {}

        primary_map = all_precomputed_residues[0].get(p, {})
        if not primary_map:
            # primary has no residues for this prime; nothing to do
            continue

        # loop over primary vectors
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            # compute canonical height proxy for primary vector
            try:
                v_primary = vector(QQ, v_primary_tuple)
            except Exception as e:
                raise TypeError(f"primary vector tuple not coercible to QQ vector: {v_primary_tuple}") from e

            H_primary = fib_meta[0]['H']
            deg_primary = fib_meta[0]['deg']
            # canonical height proxy (quadratic form)
            try:
                hcanon_p = float((v_primary.T * H_primary * v_primary)[0])
            except Exception as e:
                raise RuntimeError("error computing primary canonical height proxy") from e
            if hcanon_p <= 0:
                # small/zero heights: treat specially by adding a tiny floor to avoid log(0)
                hcanon_p = max(hcanon_p, 1e-30)

            # normalized log-canonical height for primary, with Δ adjustment
            log_h_primary_norm = math.log(hcanon_p) - math.log(float(deg_primary))  + fib_meta[0]['log_delta_adj']

            consensus_by_rhs = []

            # for each RHS index, collect votes
            for rhs_idx, residues_primary in enumerate(rhs_lists_primary):
                residues_primary = set(r for r in residues_primary if isinstance(r, int))
                residue_votes = Counter()
                for r in residues_primary:
                    residue_votes[r] += 1  # primary always votes

                # consult secondary fibrations
                for fib_idx in range(1, num_fibs):
                    other_map = all_precomputed_residues[fib_idx].get(p, {})
                    if not other_map:
                        continue
                    H_other = fib_meta[fib_idx]['H']
                    deg_other = fib_meta[fib_idx]['deg']
                    log_delta_adj_other = fib_meta[fib_idx]['log_delta_adj']

                    # find best matching vector in other_map by closeness of log-normalized heights
                    best_match_rhs_list = None
                    best_diff = float('inf')

                    for v_other_tuple, rhs_lists_other in other_map.items():
                        try:
                            v_other = vector(QQ, v_other_tuple)
                        except Exception:
                            continue
                        try:
                            hcanon_o = float((v_other.T * H_other * v_other)[0])
                        except Exception:
                            continue
                        if hcanon_o <= 0:
                            hcanon_o = max(hcanon_o, 1e-30)
                        log_h_other_norm = math.log(hcanon_o) - math.log(float(deg_other)) + log_delta_adj_other

                        diff = abs(log_h_primary_norm - log_h_other_norm)
                        if diff < best_diff:
                            best_diff = diff
                            best_match_rhs_list = rhs_lists_other

                    # accept the match only if within tolerance
                    if best_match_rhs_list is not None and best_diff <= height_tolerance_log:
                        # add votes from best_match for the same rhs_idx (if present)
                        if rhs_idx < len(best_match_rhs_list):
                            for r in best_match_rhs_list[rhs_idx]:
                                if isinstance(r, int):
                                    residue_votes[r] += 1

                # now finalize consensus for this RHS index
                consensus_set = {r for r, count in residue_votes.items() if count >= min_votes}
                consensus_by_rhs.append(consensus_set)

            # store if any consensus survived
            if any(len(s) for s in consensus_by_rhs):
                consensus_residues[p][v_primary_tuple] = consensus_by_rhs

    return consensus_residues

@PROFILE
from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict

):
    """
    Height-aware consensus filter with strict intersection.
    
    Key insight: Once vectors are height-matched across fibrations, their residues
    must be IDENTICAL along the rail (proven from genus 2 rigidity).
    
    Therefore, we use INTERSECTION of residue sets, not voting.
    
    Returns: 
        consensus_residues: {p: {v_primary_tuple: [consensus_set_per_rhs]}}
        stats: dict with filtering statistics
    """
    
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    if debug:
        print(f"\n{'='*70}")
        print(f"HEIGHT-AWARE STRICT CONSENSUS: {num_fibs} fibrations")
        print(f"Height tolerance: {height_tolerance_log} (log-space)")
        print(f"Δ-scaling: {'enabled' if use_delta_scaling else 'disabled'}")
        print(f"Residue policy: STRICT INTERSECTION (must match exactly)")
        print(f"{'='*70}")
    
    # --- Validate inputs ---
    if not isinstance(all_precomputed_residues, (list, tuple)):
        raise ValueError("all_precomputed_residues must be a list of per-fibration residue maps")
    
    if not isinstance(fibration_geometries, (list, tuple)) or len(fibration_geometries) != num_fibs:
        raise ValueError("fibration_geometries must match length of all_precomputed_residues")
    
    # --- Build per-fibration log-Δ scaling factors ---
    primary_geom = fibration_geometries[0]
    if 'deg' not in primary_geom or 'H' not in primary_geom:
        raise KeyError("primary fibration geometry must contain 'H' and 'deg'")
    
    # Compute log-delta adjustments for each fibration
    log_delta_adjustments = []
    
    if use_delta_scaling:
        # Sample Δ for primary fibration if not already provided
        if 'delta_samples' not in primary_geom or not primary_geom['delta_samples']:
            if debug:
                print("Sampling Δ for primary fibration...")
            if 'cd' in primary_geom and 'sections' in primary_geom:
                primary_geom['delta_samples'] = sample_delta_ratios(
                    primary_geom['cd'], 
                    primary_geom['sections'],
                    num_samples=100
                )
            else:
                print("Warning: Cannot sample Δ (missing cd/sections), falling back to degree ratio")
                use_delta_scaling = False
        
        if use_delta_scaling:
            primary_log_delta = float(median(primary_geom['delta_samples']))
            
            for i, geom in enumerate(fibration_geometries):
                # Sample if not already done
                if 'delta_samples' not in geom or not geom['delta_samples']:
                    if debug and i > 0:
                        print(f"Sampling Δ for fibration {i}...")
                    if 'cd' in geom and 'sections' in geom:
                        geom['delta_samples'] = sample_delta_ratios(
                            geom['cd'], 
                            geom['sections'],
                            num_samples=100
                        )
                
                if geom.get('delta_samples'):
                    log_delta_i = float(median(geom['delta_samples']))
                    # Adjustment = (log|Δ_i| - log|Δ_primary|) + log(deg_i/deg_primary)
                    adj = (log_delta_i - primary_log_delta) + math.log(float(geom['deg'])/float(primary_geom['deg']))
                else:
                    # Fallback to degree ratio only
                    adj = math.log(float(geom['deg'])/float(primary_geom['deg']))
                
                log_delta_adjustments.append(adj)
    else:
        # No Δ-scaling, just use degree ratios
        for geom in fibration_geometries:
            adj = math.log(float(geom['deg'])/float(primary_geom['deg']))
            log_delta_adjustments.append(adj)
    
    # --- Build metadata for each fibration ---
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        fib_meta.append({
            'H': geom['H'],
            'deg': int(geom['deg']),
            'name': geom.get('name', f"fib_{i}"),
            'log_delta_adj': log_delta_adjustments[i]
        })
    
    # --- Main consensus computation ---
    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'vectors_matched_all_fibs': 0,
        'vectors_with_consensus': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'per_prime_stats': {}
    }
    
    for p in prime_pool:
        consensus_residues[p] = {}
        
        primary_map = all_precomputed_residues[0].get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {
            'vectors_primary': len(primary_map),
            'vectors_matched': 0,
            'vectors_consensus': 0,
            'residues_before': 0,
            'residues_after': 0
        }
        
        # Loop over primary vectors
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            stats['total_vectors_primary'] += 1
            prime_stats['vectors_primary'] += 1
            
            # Count residues before filtering
            for r_list in rhs_lists_primary:
                prime_stats['residues_before'] += len(r_list)
            
            # Compute canonical height for primary vector
            try:
                v_primary = vector(QQ, v_primary_tuple)
            except Exception as e:
                if debug:
                    print(f"Warning: could not coerce primary vector {v_primary_tuple}: {e}")
                continue
            
            H_primary = fib_meta[0]['H']
            try:
                hcanon_p = float((v_primary.T * H_primary * v_primary)[0])
            except Exception as e:
                if debug:
                    print(f"Warning: height computation failed for primary vector: {e}")
                continue
            
            if hcanon_p <= 0:
                hcanon_p = max(hcanon_p, 1e-30)
            
            # Normalized log-height for primary (with Δ adjustment)
            log_h_primary_norm = math.log(hcanon_p) - math.log(float(fib_meta[0]['deg'])) + fib_meta[0]['log_delta_adj']
            
            # Find matching vectors in ALL other fibrations
            matched_rhs_lists = [rhs_lists_primary]  # Start with primary
            all_matched = True
            
            for fib_idx in range(1, num_fibs):
                other_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_map:
                    all_matched = False
                    break
                
                H_other = fib_meta[fib_idx]['H']
                log_delta_adj_other = fib_meta[fib_idx]['log_delta_adj']
                deg_other = fib_meta[fib_idx]['deg']
                
                # Find best matching vector by height
                best_match_rhs_list = None
                best_diff = float('inf')
                
                for v_other_tuple, rhs_lists_other in other_map.items():
                    try:
                        v_other = vector(QQ, v_other_tuple)
                    except Exception:
                        continue
                    
                    try:
                        hcanon_o = float((v_other.T * H_other * v_other)[0])
                    except Exception:
                        continue
                    
                    if hcanon_o <= 0:
                        hcanon_o = max(hcanon_o, 1e-30)
                    
                    log_h_other_norm = math.log(hcanon_o) - math.log(float(deg_other)) + log_delta_adj_other
                    
                    diff = abs(log_h_primary_norm - log_h_other_norm)
                    if diff < best_diff:
                        best_diff = diff
                        best_match_rhs_list = rhs_lists_other
                
                # Accept match only if within tolerance
                if best_match_rhs_list is not None and best_diff <= height_tolerance_log:
                    matched_rhs_lists.append(best_match_rhs_list)
                else:
                    all_matched = False
                    break
            
            # If we found matches in ALL fibrations, compute strict intersection
            if all_matched and len(matched_rhs_lists) == num_fibs:
                stats['vectors_matched_all_fibs'] += 1
                prime_stats['vectors_matched'] += 1
                
                consensus_by_rhs = []
                num_rhs = len(rhs_lists_primary)
                
                for rhs_idx in range(num_rhs):
                    # Collect residue sets from all matched fibrations
                    residue_sets = []
                    for matched_rhs in matched_rhs_lists:
                        if rhs_idx < len(matched_rhs):
                            r_set = {r for r in matched_rhs[rhs_idx] if isinstance(r, int)}
                            residue_sets.append(r_set)
                    
                    # STRICT INTERSECTION (must appear in ALL fibrations)
                    if residue_sets:
                        consensus_set = set.intersection(*residue_sets)
                        consensus_by_rhs.append(consensus_set)
                    else:
                        consensus_by_rhs.append(set())
                
                # Store if any consensus survived
                if any(len(s) for s in consensus_by_rhs):
                    consensus_residues[p][v_primary_tuple] = consensus_by_rhs
                    stats['vectors_with_consensus'] += 1
                    prime_stats['vectors_consensus'] += 1
                    
                    # Count residues after filtering
                    for c_set in consensus_by_rhs:
                        prime_stats['residues_after'] += len(c_set)
        
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']
    
    # Compute reduction ratio
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    else:
        stats['reduction_ratio'] = 0.0
    
    if debug:
        print(f"\nConsensus Statistics:")
        print(f"  Primary vectors: {stats['total_vectors_primary']}")
        print(f"  Matched all fibs: {stats['vectors_matched_all_fibs']}")
        print(f"  With consensus: {stats['vectors_with_consensus']}")
        print(f"  Residues: {stats['total_residues_before']:,} → {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
    
    return consensus_residues, stats


@PROFILE
from sage.all import *
from search_common import *
import math
from collections import Counter, defaultdict

def sample_delta_ratios(cd, sections, num_samples=100, m_range=(-1000, 1000), seed=None):
    """
    Sample log|Δ| values at random rational m-values to estimate the 
    discriminant behavior of this fibration.
    
    Args:
        cd: CurveData object for this fibration
        sections: List of sections (used to verify good reduction)
        num_samples: Number of random m-values to sample
        m_range: (min, max) range for sampling m-values
        seed: Random seed for reproducibility
        
    Returns:
        List of log|Δ(m)| values (floats)
    """
    if seed is not None:
        set_random_seed(seed)
    
    samples = []
    m_sym = cd.a4.parent().gen()
    
    # Get discriminant as a function of m
    try:
        delta_m = cd.curve.discriminant()
    except Exception:
        # Fallback: discriminant should be in cd somewhere
        delta_m = cd.disc_m
    
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample a random rational m-value
        num = ZZ.random_element(m_range[0], m_range[1])
        den = ZZ.random_element(1, 100)
        m_val = QQ(num) / QQ(den)
        
        # Skip bad primes and singular fibers
        if m_val in cd.bad_primes:
            continue
            
        try:
            # Evaluate discriminant at this m
            delta_val = delta_m.subs({m_sym: m_val})
            delta_val = QQ(delta_val)
            
            if delta_val == 0:
                continue
                
            # Compute log|Δ|
            log_delta = float(math.log(abs(float(delta_val))))
            samples.append(log_delta)
            
        except Exception:
            continue
    
    if len(samples) < num_samples // 2:
        print(f"Warning: only got {len(samples)} valid samples out of {num_samples} requested")
    
    return samples


def compute_consensus_residues_with_height_matching(
    all_precomputed_residues,   # List of dicts: per-fibration {p: {v_tuple: [rhs_sets]}}
    fibration_geometries,       # List of dicts per fibration: {'H': matrix, 'deg': int, 'name': str}
    prime_pool,                 # iterable of primes to consider
    consensus_threshold=0.7,    # Not used anymore (strict intersection), but kept for API compatibility
    height_tolerance_log=0.5,   # log tolerance (abs diff of logs)
    use_delta_scaling=True,     # whether to apply log|Δ| normalization
    debug=False
):
    """
    Height-aware consensus filter with strict intersection and robust abstention.
    
    Key insight: Once vectors are height-matched across fibrations, their residues
    must be IDENTICAL along the rail (proven from genus 2 rigidity).
    
    Therefore, we use INTERSECTION of residue sets from *participating* fibrations.
    A fibration that fails to find a height-match *abstains* and does not veto.
    
    Returns: 
        consensus_residues: {p: {v_primary_tuple: [consensus_set_per_rhs]}}
        stats: dict with filtering statistics
    """
    
    num_fibs = len(all_precomputed_residues)
    if num_fibs == 0:
        return {}, {}
    
    if debug:
        print(f"\n{'='*70}")
        print(f"HEIGHT-AWARE STRICT CONSENSUS: {num_fibs} fibrations")
        print(f"Height tolerance: {height_tolerance_log} (log-space)")
        print(f"Δ-scaling: {'enabled' if use_delta_scaling else 'disabled'}")
        print(f"Residue policy: STRICT INTERSECTION (of participating fibrations)")
        print(f"{'='*70}")
    
    # --- Validate inputs ---
    if not isinstance(all_precomputed_residues, (list, tuple)):
        raise ValueError("all_precomputed_residues must be a list of per-fibration residue maps")
    
    if not isinstance(fibration_geometries, (list, tuple)) or len(fibration_geometries) != num_fibs:
        raise ValueError("fibration_geometries must match length of all_precomputed_residues")
    
    # --- Build per-fibration log-Δ scaling factors ---
    primary_geom = fibration_geometries[0]
    if 'deg' not in primary_geom or 'H' not in primary_geom:
        raise KeyError("primary fibration geometry must contain 'H' and 'deg'")
    
    # Compute log-delta adjustments for each fibration
    log_delta_adjustments = []
    
    if use_delta_scaling:
        # Sample Δ for primary fibration if not already provided
        if 'delta_samples' not in primary_geom or not primary_geom['delta_samples']:
            if debug:
                print("Sampling Δ for primary fibration...")
            if 'cd' in primary_geom and 'sections' in primary_geom:
                primary_geom['delta_samples'] = sample_delta_ratios(
                    primary_geom['cd'], 
                    primary_geom['sections'],
                    num_samples=100
                )
            else:
                print("Warning: Cannot sample Δ (missing cd/sections), falling back to degree ratio")
                use_delta_scaling = False
        
        if use_delta_scaling:
            primary_log_delta = float(median(primary_geom['delta_samples']))
            
            for i, geom in enumerate(fibration_geometries):
                # Sample if not already done
                if 'delta_samples' not in geom or not geom['delta_samples']:
                    if debug and i > 0:
                        print(f"Sampling Δ for fibration {i}...")
                    if 'cd' in geom and 'sections' in geom:
                        geom['delta_samples'] = sample_delta_ratios(
                            geom['cd'], 
                            geom['sections'],
                            num_samples=100
                        )
                
                if geom.get('delta_samples'):
                    log_delta_i = float(median(geom['delta_samples']))
                    # Adjustment = (log|Δ_i| - log|Δ_primary|) + log(deg_i/deg_primary)
                    adj = (log_delta_i - primary_log_delta) + math.log(float(geom['deg'])/float(primary_geom['deg']))
                else:
                    # Fallback to degree ratio only
                    adj = math.log(float(geom['deg'])/float(primary_geom['deg']))
                
                log_delta_adjustments.append(adj)
    else:
        # No Δ-scaling, just use degree ratios
        for geom in fibration_geometries:
            adj = math.log(float(geom['deg'])/float(primary_geom['deg']))
            log_delta_adjustments.append(adj)
    
    # --- Build metadata for each fibration ---
    fib_meta = []
    for i, geom in enumerate(fibration_geometries):
        fib_meta.append({
            'H': geom['H'],
            'deg': int(geom['deg']),
            'name': geom.get('name', f"fib_{i}"),
            'log_delta_adj': log_delta_adjustments[i]
        })
    
    # --- Main consensus computation ---
    consensus_residues = {}
    stats = {
        'total_vectors_primary': 0,
        'vectors_matched_all_fibs': 0, # <-- This stat is now "matched by all *other* fibs"
        'vectors_with_consensus': 0,
        'total_residues_before': 0,
        'total_residues_after': 0,
        'per_prime_stats': {}
    }
    
    for p in prime_pool:
        consensus_residues[p] = {}
        
        primary_map = all_precomputed_residues[0].get(p, {})
        if not primary_map:
            continue
        
        prime_stats = {
            'vectors_primary': 0, # Corrected: initialize to 0
            'vectors_matched': 0,
            'vectors_consensus': 0,
            'residues_before': 0,
            'residues_after': 0
        }
        
        # Loop over primary vectors
        for v_primary_tuple, rhs_lists_primary in primary_map.items():
            stats['total_vectors_primary'] += 1
            prime_stats['vectors_primary'] += 1
            
            # Count residues before filtering (from primary only)
            for r_list in rhs_lists_primary:
                prime_stats['residues_before'] += len(r_list)
            
            # Compute canonical height for primary vector
            try:
                v_primary = vector(QQ, v_primary_tuple)
            except Exception as e:
                if debug:
                    print(f"Warning: could not coerce primary vector {v_primary_tuple}: {e}")
                continue
            
            H_primary = fib_meta[0]['H']
            try:
                hcanon_p = float((v_primary.T * H_primary * v_primary)[0])
            except Exception as e:
                if debug:
                    print(f"Warning: height computation failed for primary vector: {e}")
                continue
            
            if hcanon_p <= 0:
                hcanon_p = max(hcanon_p, 1e-30)
            
            # Normalized log-height for primary (with Δ adjustment)
            log_h_primary_norm = math.log(hcanon_p) - math.log(float(fib_meta[0]['deg'])) + fib_meta[0]['log_delta_adj']
            
            # --- START CONSENSUS FIX ---
            # Initialize consensus with the primary fibration's residues
            num_rhs = len(rhs_lists_primary)
            current_consensus_by_rhs = []
            for rhs_idx in range(num_rhs):
                r_set = {r for r in rhs_lists_primary[rhs_idx] if isinstance(r, int)}
                current_consensus_by_rhs.append(r_set)
                
            participating_fibs_count = 1 # Primary always participates

            # Find matching vectors in ALL other fibrations
            for fib_idx in range(1, num_fibs):
                other_map = all_precomputed_residues[fib_idx].get(p, {})
                if not other_map:
                    continue # This fibration abstains for this prime
                
                H_other = fib_meta[fib_idx]['H']
                log_delta_adj_other = fib_meta[fib_idx]['log_delta_adj']
                deg_other = fib_meta[fib_idx]['deg']
                
                # Find best matching vector by height
                best_match_rhs_list = None
                best_diff = float('inf')
                
                for v_other_tuple, rhs_lists_other in other_map.items():
                    try:
                        v_other = vector(QQ, v_other_tuple)
                    except Exception:
                        continue
                    
                    try:
                        hcanon_o = float((v_other.T * H_other * v_other)[0])
                    except Exception:
                        continue
                    
                    if hcanon_o <= 0:
                        hcanon_o = max(hcanon_o, 1e-30)
                    
                    log_h_other_norm = math.log(hcanon_o) - math.log(float(deg_other)) + log_delta_adj_other
                    
                    diff = abs(log_h_primary_norm - log_h_other_norm)
                    if diff < best_diff:
                        best_diff = diff
                        best_match_rhs_list = rhs_lists_other
                
                # Accept match only if within tolerance
                if best_match_rhs_list is not None and best_diff <= height_tolerance_log:
                    # This fibration is participating. Intersect its residues.
                    participating_fibs_count += 1
                    for rhs_idx in range(num_rhs):
                        if rhs_idx < len(best_match_rhs_list):
                            other_set = {r for r in best_match_rhs_list[rhs_idx] if isinstance(r, int)}
                            # Intersect with the current consensus
                            current_consensus_by_rhs[rhs_idx].intersection_update(other_set)
                        else:
                            # This fibration's list was too short, it didn't vote on this RHS
                            # We could treat this as an empty set intersection, but safer to
                            # assume it just didn't have data.
                            # For strictness, we'll intersect with empty.
                            current_consensus_by_rhs[rhs_idx].intersection_update(set())
                
                # If no match was found (best_diff > tolerance), this fibration
                # simply abstains. We do nothing and move to the next fibration.

            # --- END CONSENSUS FIX ---

            # Store if any consensus survived
            if any(len(s) for s in current_consensus_by_rhs):
                consensus_residues[p][v_primary_tuple] = current_consensus_by_rhs
                stats['vectors_with_consensus'] += 1
                prime_stats['vectors_consensus'] += 1
                
                if participating_fibs_count == num_fibs:
                    stats['vectors_matched_all_fibs'] += 1
                    prime_stats['vectors_matched'] += 1
                
                # Count residues after filtering
                for c_set in current_consensus_by_rhs:
                    prime_stats['residues_after'] += len(c_set)
        
        stats['per_prime_stats'][p] = prime_stats
        stats['total_residues_before'] += prime_stats['residues_before']
        stats['total_residues_after'] += prime_stats['residues_after']
    
    # Compute reduction ratio
    if stats['total_residues_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_residues_after'] / stats['total_residues_before'])
    else:
        stats['reduction_ratio'] = 0.0
    
    if debug:
        print(f"\nConsensus Statistics:")
        print(f"  Primary vectors: {stats['total_vectors_primary']}")
        print(f"  Matched all fibs: {stats['vectors_matched_all_fibs']}")
        print(f"  With consensus: {stats['vectors_with_consensus']}")
        print(f"  Residues: {stats['total_residues_before']:,} → {stats['total_residues_after']:,}")
        print(f"  Reduction: {stats['reduction_ratio']:.1%}")
    
    return consensus_residues, stats


def compute_consensus_residues(precomputed_residues_list, prime_pool, 
                                consensus_threshold=CONSENSUS_THRESHOLD,
                                debug=DEBUG):
    """
    DEPRECATED: Vector-Blind Consensus (kept for backwards compatibility)
    
    Use compute_consensus_residues_with_height_matching instead.
    """
    from collections import defaultdict, Counter
    import math

    num_fibrations = len(precomputed_residues_list)
    if num_fibrations == 0:
        return {}, {}

    if debug:
        print(f"\n{'='*70}")
        print(f"WARNING: Using deprecated vector-blind consensus")
        print(f"Consider switching to height-aware consensus")
        print(f"{'='*70}")

    # [Rest of old implementation kept for compatibility]
    valid_m_sets = defaultdict(lambda: defaultdict(set))
    participating_counts = defaultdict(int)

    for fib_idx, precomp in enumerate(precomputed_residues_list):
        for p in prime_pool:
            if p not in precomp or not precomp[p]:
                continue
            
            participating_counts[p] += 1
            mapping = precomp[p]
            
            for v_tuple, rhs_lists in mapping.items():
                for r_list in rhs_lists:
                    for r in r_list:
                        if isinstance(r, int):
                            valid_m_sets[p][fib_idx].add(r)

    consensus_m_values = {}
    
    for p in prime_pool:
        n_part = participating_counts[p]
        if n_part == 0:
            continue
            
        min_votes = max(1, int(math.ceil(consensus_threshold * n_part)))
        
        residue_counts = Counter()
        for fib_idx, r_set in valid_m_sets[p].items():
            for r in r_set:
                residue_counts[r] += 1
        
        allowed = {r for r, count in residue_counts.items() if count >= min_votes}
        if allowed:
            consensus_m_values[p] = allowed

    primary_precomp = precomputed_residues_list[-1]
    filtered_residues = {}
    
    dim = 0
    num_rhs = 1

    # Try to find dim and num_rhs from the first valid entry
    try:
        for p_map in precomputed_residues_list[0].values():
            if p_map:
                first_v = next(iter(p_map.keys()))
                dim = len(first_v)
                num_rhs = len(p_map[first_v])
                break
    except Exception:
        pass # Will use defaults dim=0, num_rhs=1

    if dim == 0:
        try:
            # Fallback to primary
            for p_map in primary_precomp.values():
                if p_map:
                    first_v = next(iter(p_map.keys()))
                    dim = len(first_v)
                    num_rhs = len(p_map[first_v])
                    break
        except Exception:
            pass # Still dim=0

    if dim == 0:
        print("Warning: Could not determine section dimension (dim=0). Consensus disabled.")
        return {}, {}
        
    DUMMY_VECTOR_KEY = tuple([0] * dim)

    stats = {
        'total_before': 0,
        'total_after': 0,
        'reduction_ratio': 0.0,
        'per_prime_before': {},
        'per_prime_after': {}
    }

    for p in prime_pool:
        count_before = 0
        if p in primary_precomp:
            for rhs_lists in primary_precomp[p].values():
                for r_list in rhs_lists:
                    count_before += len(r_list)
        
        consensus_set = consensus_m_values.get(p, set())
        
        if consensus_set:
            p_map_new = {}
            consensus_lists = [consensus_set for _ in range(num_rhs)]
            p_map_new[DUMMY_VECTOR_KEY] = consensus_lists
            filtered_residues[p] = p_map_new
            count_after = len(consensus_set) * num_rhs
        else:
            count_after = 0

        stats['per_prime_before'][p] = count_before
        stats['per_prime_after'][p] = count_after
        stats['total_before'] += count_before
        stats['total_after'] += count_after

    if stats['total_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_after'] / stats['total_before'])
        
    return filtered_residues, stats
