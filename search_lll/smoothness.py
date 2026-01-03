from math import log2
from collections import Counter
from sage.all import matrix, GF, vector
from collections import Counter, defaultdict
from sage.all import matrix, GF, vector, Integer, gcd, factor, ZZ, sqrt as sage_sqrt
from search_common import FINITE_FIELD, COEFFS_GENUS2

# ============================================================================
# ORIGINAL DIAGNOSTICS (kept for Q mode)
# ============================================================================

def diagnostic_section_collapse(divisors, tag_fn=lambda d: d.get('origin',None)):
    """
    Check whether most divisors come from few section multiples.
    """
    tags = [tag_fn(d) for d in divisors if tag_fn(d) is not None]
    if not tags:
        return

    ctr = Counter(tags)
    print("[diag:section-collapse]")
    for k,v in ctr.most_common(5):
        print(f"  source {k}: {v} divisors")
    print(f"  unique sources: {len(ctr)} / total {len(tags)}")


def diagnostic_x_root_distribution(divisors, p, label=""):
    """
    Measure how x-roots of u(x) distribute mod p.
    Looks for clustering / bias.
    """
    roots = []
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        if pow(disc, (p-1)//2, p) != 1:
            continue
        sqrt_disc = pow(disc, (p+1)//4, p) if p % 4 == 1 else None
        if sqrt_disc is None:
            continue
        r1 = (s + sqrt_disc) * pow(2, -1, p) % p
        r2 = (s - sqrt_disc) * pow(2, -1, p) % p
        roots.extend([r1, r2])

    if not roots:
        return

    ctr = Counter(roots)
    collisions = sum(v-1 for v in ctr.values() if v > 1)
    entropy = -sum((v/len(roots))*log2(v/len(roots)) for v in ctr.values())

    print(f"[diag:x-roots]{label}")
    print(f"  total roots: {len(roots)}")
    print(f"  unique roots: {len(ctr)}")
    print(f"  collision excess: {collisions}")
    print(f"  empirical entropy: {entropy} (max {log2(p)})")


def diagnostic_smoothness_proxy(divisors, p):
    """
    Measures whether u(x) tends to split into low-degree factors mod p.
    This is a proxy for index-calculus smoothness.
    """
    deg1 = 0
    deg2 = 0

    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        if disc == 0:
            deg1 += 1
        elif pow(disc, (p-1)//2, p) == 1:
            deg1 += 1
        else:
            deg2 += 1

    total = deg1 + deg2
    if total == 0:
        return

    print("[diag:smoothness]")
    print(f"  split u(x): {deg1}/{total} ({deg1/total:.3%})")
    print(f"  irreducible u(x): {deg2}/{total} ({deg2/total:.3%})")


def diagnostic_factor_base_saturation(divisors, p):
    """
    Checks if we are reusing the same x-coordinates (good for Index Calculus)
    or constantly introducing new ones (bad for Index Calculus).
    """
    all_roots = []
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        
        if pow(disc, (p-1)//2, p) != 1:
            if disc == 0:
                 r = (s * pow(2, -1, p)) % p
                 all_roots.extend([r, r])
            continue
            
        sqrt_disc = pow(disc, (p+1)//4, p) if p % 4 == 1 else pow(disc, (p+1)//4, p)
        
        r1 = (s + sqrt_disc) * pow(2, -1, p) % p
        r2 = (s - sqrt_disc) * pow(2, -1, p) % p
        all_roots.extend([r1, r2])

    if not all_roots:
        return

    unique_roots = set(all_roots)
    distinct_count = len(unique_roots)
    total_roots = len(all_roots)
    
    print("[diag:saturation]")
    print(f"  total root instances: {total_roots}")
    print(f"  unique roots (factor base size): {distinct_count}")
    print(f"  saturation ratio: {total_roots/max(1, distinct_count)} (higher is better for attacks)")


def diagnostic_mod_p_coverage(divisors, p, genus=2):
    """
    Checks the rank of the generated divisors modulo p.
    """
    vecs = []
    for d in divisors:
        try:
            row = [
                int(d['s']) % p, 
                int(d['p']) % p,
                int(d['v_0']) % p,
                int(d['v_1']) % p
            ]
            vecs.append(row)
        except (ValueError, KeyError):
            continue
            
    if not vecs:
        return

    M = matrix(GF(p), vecs)
    r = M.rank()
    
    print("[diag:coverage]")
    print(f"  generated divisors: {len(divisors)}")
    print(f"  linear rank (heuristic): {r}")
    if r < len(divisors):
        print(f"  (!) Dependencies found mod {p}. Potential relations for DLP.")
    else:
        print(f"  Independence held mod {p}. Basis is expanding.")
    return r


# ============================================================================
# NEW: FINITE FIELD INDEX CALCULUS DIAGNOSTICS
# ============================================================================

def index_calculus_factor_base_analysis(divisors, p, f_coeffs, verbose=True):
    """
    Complete factor base analysis for HECC index calculus over GF(p).
    
    This analyzes:
    1. Factor base quality (smoothness, coverage)
    2. Relation matrix structure
    3. Linear algebra attack feasibility
    4. Expected DLP complexity
    
    Args:
        divisors: List of Mumford divisor dicts with keys 's', 'p', 'v_0', 'v_1'
        p: Prime (field characteristic)
        f_coeffs: Curve coefficients (for genus verification)
        verbose: Print detailed diagnostics
    
    Returns:
        dict: Comprehensive analysis report
    """
    from sage.all import HyperellipticCurve, PolynomialRing, GF as SageGF
    
    # Build curve to get genus
    try:
        R = PolynomialRing(SageGF(p), 'x')
        x = R.gen()
        f_poly = sum(SageGF(p)(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly)
        g = C.genus()
    except Exception as e:
        if verbose:
            print(f"[IC Analysis] Could not build curve: {e}")
        g = 2  # Assume genus 2
    
    if verbose:
        print("\n" + "="*70)
        print("INDEX CALCULUS FEASIBILITY ANALYSIS")
        print("="*70)
        print(f"Field: GF({p})")
        print(f"Genus: {g}")
        print(f"Divisors collected: {len(divisors)}")
    
    # === 1. FACTOR BASE CONSTRUCTION ===
    factor_base = extract_factor_base(divisors, p, verbose=verbose)
    
    # === 2. SMOOTHNESS ANALYSIS ===
    smoothness_report = analyze_smoothness_distribution(divisors, p, factor_base, verbose=verbose)
    
    # === 3. RELATION MATRIX ===
    relation_matrix = build_relation_matrix(divisors, factor_base, p, verbose=verbose)
    
    # === 4. LINEAR ALGEBRA ATTACK ===
    attack_report = assess_linear_algebra_attack(relation_matrix, factor_base, p, g, verbose=verbose)
    
    # === 5. DLP COMPLEXITY ESTIMATE ===
    complexity_report = estimate_dlp_complexity(len(factor_base), len(divisors), p, g, verbose=verbose)
    
    if verbose:
        print("="*70 + "\n")
    
    return {
        'field_size': p,
        'genus': g,
        'num_divisors': len(divisors),
        'factor_base': factor_base,
        'smoothness': smoothness_report,
        'relation_matrix': relation_matrix,
        'attack_feasibility': attack_report,
        'complexity': complexity_report
    }


def analyze_smoothness_distribution(divisors, p, factor_base, verbose=True):
    """
    Analyze how 'smooth' the divisors are over the factor base.
    
    For HECC, a divisor D is 'smooth' if its Mumford polynomial u(x) 
    factors completely into degree-1 factors over GF(p).
    
    Returns:
        dict with smoothness statistics
    """
    smooth_count = 0
    partly_smooth_count = 0
    non_smooth_count = 0
    
    smoothness_by_vector = defaultdict(lambda: {'smooth': 0, 'total': 0})
    
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        
        vector = d.get('vector', None)
        smoothness_by_vector[vector]['total'] += 1
        
        if disc == 0 or pow(disc, (p-1)//2, p) == 1:
            # Completely smooth (splits over GF(p))
            smooth_count += 1
            smoothness_by_vector[vector]['smooth'] += 1
        else:
            # Not smooth (irreducible over GF(p))
            non_smooth_count += 1
    
    total = len(divisors)
    
    if verbose:
        print(f"\n[Smoothness Distribution]")
        print(f"  Smooth divisors: {smooth_count}/{total} ({100*smooth_count/total}%)")
        print(f"  Non-smooth: {non_smooth_count}/{total} ({100*non_smooth_count/total}%)")
        
        if len(smoothness_by_vector) > 1:
            print(f"  Smoothness by search vector:")
            for vec, stats in sorted(smoothness_by_vector.items(), key=lambda x: -x[1]['smooth'])[:10]:
                if stats['total'] > 0:
                    pct = 100 * stats['smooth'] / stats['total']
                    print(f"    Vector {vec}: {stats['smooth']}/{stats['total']} ({pct}%)")
    
    return {
        'smooth_count': smooth_count,
        'non_smooth_count': non_smooth_count,
        'smooth_fraction': smooth_count / max(1, total),
        'by_vector': dict(smoothness_by_vector)
    }


def assess_linear_algebra_attack(relation_matrix, factor_base, p, genus, verbose=True):
    """
    Assess feasibility of linear algebra phase of index calculus attack.
    
    Returns:
        dict with attack feasibility metrics
    """
    if relation_matrix is None:
        return {'feasible': False, 'reason': 'no_smooth_divisors'}
    
    M = relation_matrix['matrix']
    n_relations = M.nrows()
    n_unknowns = M.ncols()
    rank = relation_matrix['rank']
    
    # Expected Jacobian size for genus g curve over GF(p)
    # Hasse-Weil bounds: |#J - p^g| ≤ g * p^((g-1)/2)
    jacobian_size_estimate = p ** genus
    
    # For successful DLP attack, we need:
    # 1. Enough relations (n_relations >= n_unknowns)
    # 2. Full rank matrix
    # 3. Factor base not too large relative to Jacobian
    
    feasible = (n_relations >= n_unknowns) and (rank == n_unknowns)
    # In smoothness.py, assess_linear_algebra_attack:
    deficit_relations = max(0, n_unknowns - rank)  # Not n_unknowns - n_relations!
 
    if verbose:
        print(f"\n[Linear Algebra Attack Feasibility]")
        print(f"  Factor base size: {n_unknowns}")
        print(f"  Relations collected: {n_relations}")
        print(f"  Matrix rank: {rank}")
        print(f"  Estimated #J(GF({p})): ~{jacobian_size_estimate:.2e}")
        print(f"  Factor base coverage: {100*n_unknowns/jacobian_size_estimate}%")
        
        if feasible:
            print(f"  ✓ ATTACK FEASIBLE: Matrix is full-rank and over-determined")
            print(f"    Next step: Solve linear system mod #J to express arbitrary divisors")
        else:
            if n_relations < n_unknowns:
                print(f"  ✗ ATTACK NOT FEASIBLE: Need {deficit_relations} more relations")
            elif rank < n_unknowns:
                print(f"  ✗ ATTACK NOT FEASIBLE: Matrix is rank-deficient")
                print(f"    (Rank = {rank}, expected {n_unknowns})")
            
            print(f"    Recommendation: Expand search vectors or use smaller factor base")
    
    return {
        'feasible': feasible,
        'factor_base_size': n_unknowns,
        'relations_count': n_relations,
        'rank': rank,
        'jacobian_size_estimate': jacobian_size_estimate,
        'over_determined': n_relations >= n_unknowns,
        'full_rank': rank == n_unknowns,
        'deficit_relations': deficit_relations,  # <-- Use this
    }


def estimate_dlp_complexity(factor_base_size, n_relations, p, genus, verbose=True):
    """
    Estimate computational complexity of index calculus DLP attack.
    
    Index calculus complexity for HECC (Gaudry-Thomé-Thériault-Diem):
      O(p^(2 - 2/g)) for genus g
    
    With factor base size B and relation collection:
      - Relation collection: O(B) smooth divisor searches
      - Linear algebra: O(B^3) or O(B^2) with structured methods
      - Individual DLP: O(B) smooth decomposition
    
    Returns:
        dict with complexity estimates
    """
    import math
    
    # Theoretical complexity exponent for HECC index calculus
    if genus == 1:
        # Elliptic curves: O(√p) generic, O(p^(1/2)) with index calculus
        exponent = 0.5
    elif genus == 2:
        # Genus 2: O(p^(1.5))
        exponent = 1.5
    else:
        # General: O(p^(2 - 2/g))
        exponent = 2 - 2/genus
    
    theoretical_ops = p ** exponent
    
    # Practical complexity with collected data
    B = factor_base_size
    
    # Relation collection phase (already done)
    relation_ops = n_relations * B  # Rough estimate
    
    # Linear algebra phase: Solve B x B system
    # Dense: O(B^3), Sparse (Wiedemann): O(B^2)
    linalg_ops_dense = B ** 3
    linalg_ops_sparse = B ** 2
    
    # Individual DLP: decompose target divisor (same as relation collection)
    individual_dlp_ops = B
    
    total_ops_dense = relation_ops + linalg_ops_dense + individual_dlp_ops
    total_ops_sparse = relation_ops + linalg_ops_sparse + individual_dlp_ops
    
    if verbose:
        print(f"\n[DLP Complexity Estimate]")
        print(f"  Theoretical (genus {genus}): O(p^{exponent}) ≈ {theoretical_ops} ops")
        print(f"  Practical with B={B}:")
        print(f"    Relation collection: {relation_ops} ops (DONE)")
        print(f"    Linear algebra (dense): {linalg_ops_dense} ops")
        print(f"    Linear algebra (sparse): {linalg_ops_sparse} ops")
        print(f"    Individual DLP: {individual_dlp_ops} ops")
        print(f"  Total (dense): {total_ops_dense} ops")
        print(f"  Total (sparse): {total_ops_sparse} ops")
        
        # Comparison to generic attacks
        generic_baby_step = p ** (genus / 2)
        print(f"  Generic baby-step-giant-step: O(p^{genus/2}) ≈ {generic_baby_step} ops")
        
        if total_ops_sparse < generic_baby_step:
            speedup = generic_baby_step / total_ops_sparse
            print(f"  ✓ Index calculus is faster by ~{speedup}x")
        else:
            print(f"  ✗ Generic attack may be competitive")
    
    return {
        'theoretical_complexity': theoretical_ops,
        'theoretical_exponent': exponent,
        'practical_dense': total_ops_dense,
        'practical_sparse': total_ops_sparse,
        'factor_base_size': B,
        'generic_complexity': p ** (genus / 2),
        'speedup_vs_generic': (p ** (genus / 2)) / total_ops_sparse
    }


def tonelli_shanks(n, p):
    """
    Compute square root of n mod p using Tonelli-Shanks algorithm.
    Assumes n is a quadratic residue mod p.
    """
    if pow(n, (p-1)//2, p) != 1:
        raise ValueError(f"{n} is not a quadratic residue mod {p}")
    
    # Fast path for p ≡ 3 (mod 4)
    if p % 4 == 3:
        return pow(n, (p+1)//4, p)
    
    # General case: Tonelli-Shanks
    # Write p-1 = 2^s * q with q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    
    # Find a quadratic non-residue z
    z = 2
    while pow(z, (p-1)//2, p) != p - 1:
        z += 1
    
    m = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q+1)//2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) = 1
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    return r


def diagnose_finite_field_search(divisors, verbose=True):
    """
    Comprehensive diagnostic for finite field searches.
    Combines all analyses into a single report.
    
    Usage:
        from smoothness import diagnose_finite_field_search
        report = diagnose_finite_field_search(mumford_divisors, 997, f_coeffs)
    """
    p = FINITE_FIELD
    f_coeffs = COEFFS_GENUS2
    report = index_calculus_factor_base_analysis(divisors, p, f_coeffs, verbose=verbose)
    
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*70)
        
        if report['attack_feasibility']['feasible']:
            print("✓ INDEX CALCULUS ATTACK IS FEASIBLE")
            print(f"  - Factor base size: {report['factor_base']['size']}")
            print(f"  - Relations collected: {report['relation_matrix']['matrix'].nrows()}")
            print(f"  - Matrix is full-rank and over-determined")
            print(f"  - Estimated complexity: {report['complexity']['practical_sparse']} operations")
            print(f"\nNext steps:")
            print(f"  1. Solve linear system to express factor base in terms of relations")
            print(f"  2. For target divisor D, decompose D over factor base")
            print(f"  3. Express D = linear combination of known divisors")
        else:
            print("✗ INDEX CALCULUS ATTACK NOT YET FEASIBLE")
            deficit = report['attack_feasibility']['deficit_relations'] if report['relation_matrix'] else float('inf')
            print(f"  - Need {deficit} more relations to achieve full rank")
            print(f"  - Need {deficit} more smooth relations")
            print(f"  - Current factor base size: {report['factor_base']['size']}")
            print(f"\nRecommendations:")
            print(f"  1. Expand search vectors (try more multiples of sections)")
            print(f"  2. Use smaller factor base (B ≈ {int(p**0.5)})")
            print(f"  3. Collect more relations with current setup")
        
        print("="*70 + "\n")
    
    return report


def extract_factor_row(roots, fb_index):
    """
    roots: list of x-roots
    fb_index: dict x -> column index
    """
    row = {}
    for x in roots:
        j = fb_index.get(x)
        if j is None:
            raise ValueError(f"x={x} not in factor base")
        row[j] = row.get(j, 0) + 1
    return row

def divisor_support_key(div):
    """
    Canonical key for index calculus factor base.
    Ignores v-data, keeps only unordered u-roots.
    For two divisors to be linearly independent in the factor base,
    they must have different support.
    """
    roots = div.get('roots')
    if not roots:
        # Not smooth over base field
        return None
    return tuple(sorted(roots))

def extract_factor_base(divisors, p, verbose=True):
    """
    Extract the factor base with support-based deduplication.
    Only keep ONE divisor per unique support to avoid linear dependence.
    """
    all_roots = []
    factored_count = 0
    
    # Track which supports we've seen
    seen_supports = set()
    unique_divisors = []
    support_multiplicities = defaultdict(int)
    
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        disc = (s*s - 4*pp) % p
        
        # Check if u(x) = x² - sx + p splits over GF(p)
        if disc == 0:
            # Double root
            r = (s * pow(2, -1, p)) % p
            roots = [r, r]
            factored_count += 1
        elif pow(disc, (p-1)//2, p) == 1:
            # Two distinct roots
            sqrt_disc = tonelli_shanks(disc, p)
            r1 = (s + sqrt_disc) * pow(2, -1, p) % p
            r2 = (s - sqrt_disc) * pow(2, -1, p) % p
            roots = [r1, r2]
            factored_count += 1
        else:
            # Not smooth
            continue
        
        # Get support key
        support = tuple(sorted(roots))
        support_multiplicities[support] += 1
        
        # Only add to factor base if this is the FIRST time we see this support
        if support not in seen_supports:
            seen_supports.add(support)
            all_roots.extend(roots)
            unique_divisors.append(d)
    
    root_counts = Counter(all_roots)
    unique_roots = set(all_roots)
    
    if verbose:
        print(f"\n[Factor Base - Support Deduplicated]")
        print(f"  Unique supports: {len(seen_supports)}")
        print(f"  Distinct x-coordinates: {len(unique_roots)}")
        print(f"  Total x-instances (after dedup): {len(all_roots)}")
        print(f"  Original factored divisors: {factored_count}")
        print(f"  Kept after support dedup: {len(unique_divisors)}")
        print(f"  Average multiplicity per support: {factored_count/max(1, len(seen_supports)):.1f}")
        
        # Show most duplicated supports
        if support_multiplicities:
            print(f"  Top 5 most duplicated supports:")
            for support, count in sorted(support_multiplicities.items(), 
                                         key=lambda x: -x[1])[:5]:
                print(f"    {support}: {count} duplicates (kept 1)")
        
        # Show most common roots
        if len(root_counts) > 0:
            print(f"  Top 5 most frequent x-coordinates (after dedup):")
            for root, count in root_counts.most_common(5):
                print(f"    x={root}: appears {count} times")
    
    return {
        'roots': unique_roots,
        'multiplicities': root_counts,
        'size': len(unique_roots),
        'coverage': len(unique_divisors) / max(1, len(divisors)),
        'avg_reuse': len(all_roots) / max(1, len(unique_roots)),
        'unique_divisors': unique_divisors,  # NEW: return deduplicated list
        'duplicate_count': factored_count - len(unique_divisors)
    }


def build_relation_matrix(divisors, factor_base, p, verbose=True):
    """
    Build relation matrix using ONLY support-deduplicated divisors.
    """
    # Use the deduplicated divisor list from factor_base
    smooth_divisors_unique = factor_base.get('unique_divisors', [])
    
    if not smooth_divisors_unique:
        # Fallback: extract from divisors
        seen_supports = set()
        smooth_divisors_unique = []
        for d in divisors:
            support = divisor_support_key(d)
            if support and support not in seen_supports:
                seen_supports.add(support)
                smooth_divisors_unique.append(d)
    
    # Map roots to indices
    root_list = sorted(factor_base['roots'])
    root_to_idx = {r: i for i, r in enumerate(root_list)}
    
    # Build matrix rows (only from unique supports)
    matrix_rows = []
    
    for d in smooth_divisors_unique:
        roots = d.get('roots', [])
        if not roots:
            continue
        
        # Build exponent vector
        row = [0] * len(root_list)
        for r in roots:
            if r in root_to_idx:
                row[root_to_idx[r]] += 1
        
        matrix_rows.append(row)
    
    if not matrix_rows:
        if verbose:
            print(f"\n[Relation Matrix]")
            print(f"  ERROR: No smooth divisors found after deduplication!")
        return None
    
    # Build Sage matrix
    M = matrix(ZZ, matrix_rows)
    
    # Analyze matrix
    rank = M.rank()
    nullity = M.ncols() - rank
    
    if verbose:
        print(f"\n[Relation Matrix]")
        print(f"  Dimensions: {M.nrows()} relations × {M.ncols()} factor base elements")
        print(f"  (After support deduplication)")
        print(f"  Rank: {rank}")
        print(f"  Nullity: {nullity}")
        print(f"  Over-determined: {M.nrows() > M.ncols()}")
        
        if M.nrows() >= M.ncols():
            print(f"  ✓ Sufficient relations for linear algebra attack!")
        else:
            deficit = M.ncols() - M.nrows()
            print(f"  ✗ Need {deficit} more UNIQUE relations")
        
        # Row sparsity
        row_weights = [sum(1 for x in row if x != 0) for row in matrix_rows]
        avg_weight = sum(row_weights) / len(row_weights)
        max_weight = max(row_weights)
        min_weight = min(row_weights)
        
        print(f"  Row sparsity:")
        print(f"    Avg non-zeros per row: {avg_weight:.1f}")
        print(f"    Min: {min_weight}, Max: {max_weight}")
    
    return {
        'matrix': M,
        'smooth_divisors': smooth_divisors_unique,
        'root_list': root_list,
        'rank': rank,
        'nullity': nullity,
        'sufficient': M.nrows() >= M.ncols(),
        'deficit': max(0, M.ncols() - M.nrows())
    }
