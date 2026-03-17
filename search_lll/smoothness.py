from math import log2
from collections import Counter, defaultdict
from sage.all import matrix, GF, vector, Integer, gcd, factor, ZZ, sqrt as sage_sqrt, PolynomialRing
from search_common import *

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
            raise
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

def diagnose_vector_diversity(divisors, verbose=True):
    """
    Check if divisors come from diverse search vectors.
    Lack of diversity indicates geometric degeneracy.
    """

    vector_counts = Counter(d.get('vector') for d in divisors)

    if verbose:
        print(f"\n[Vector Diversity Analysis]")
        print(f"  Total divisors: {len(divisors)}")
        print(f"  Unique vectors used: {len(vector_counts)}")

        if len(vector_counts) == 1:
            print(f"  ⚠️  SEVERE: All divisors from single vector!")
            print(f"     This causes rank collapse in factor base")
        elif len(vector_counts) < 5:
            print(f"  ⚠️  LOW DIVERSITY: Only {len(vector_counts)} vectors active")

        print(f"  Divisors per vector:")
        for vec, count in sorted(vector_counts.items(), key=lambda x: -x[1])[:10]:
            pct = 100.0 * count / len(divisors)
            print(f"    Vector {vec}: {count} divisors ({pct:.1f}%)")

    return {
        'unique_vectors': len(vector_counts),
        'vector_counts': dict(vector_counts),
        'is_degenerate': len(vector_counts) <= 2
    }

def estimate_dlp_complexity(fb_size, rel_count, p, g, verbose=False):
    """
    Estimates the complexity of a DLP attack using Index Calculus.
    """
    if fb_size == 0 or rel_count == 0:
        if verbose:
            print("  [Complexity] Missing factor base or relations. Ops set to 0.")
        return {'total_ops_sparse': 0, 'speedup': 0}

    # Order of the Jacobian J(C) ~ p^g
    group_order = p**g
    generic_baby_step = sage_sqrt(QQ(group_order))

    # Simple complexity model for sparse linear algebra (e.g., Wiedemann)
    # Usually O(w * B^2) where w is weight per row
    avg_weight = 2 # In genus 2, usually 2 points per divisor
    total_ops_sparse = QQ(avg_weight * fb_size**2)

    # Prevent division by zero if ops are zero or negative
    if total_ops_sparse <= 0:
        speedup = 0
    else:
        speedup = generic_baby_step / total_ops_sparse

    if verbose:
        print(f"  Theoretical (genus {g}): O(p^{g/2}) ≈ {float(generic_baby_step):.2e} ops")
        print(f"  Practical (Sparse LA): {float(total_ops_sparse):.2e} ops")
        print(f"  Estimated Speedup: {float(speedup):.2e}x")

    return {
        'total_ops_sparse': total_ops_sparse,
        'generic_ops': generic_baby_step,
        'speedup': speedup
    }

def assess_linear_algebra_attack(relation_matrix, factor_base, p, genus, verbose=True):
    """
    Assess feasibility of linear algebra phase of index calculus attack.
    Always returns a dictionary with all keys to prevent KeyErrors in the UI.
    """
    n_unknowns = len(factor_base) if factor_base else 0
    jacobian_size_estimate = p ** genus if (p and genus) else 0

    # Initialize with default "not feasible" state
    report = {
        'feasible': False,
        'reason': 'no_smooth_divisors',
        'factor_base_size': n_unknowns,
        'relations_count': 0,
        'rank': 0,
        'jacobian_size_estimate': jacobian_size_estimate,
        'over_determined': False,
        'full_rank': False,
        'deficit_relations': n_unknowns
    }

    if relation_matrix is None or 'matrix' not in relation_matrix:
        if verbose:
            print("\n[Linear Algebra Attack Feasibility]")
            print("  ✗ ATTACK NOT FEASIBLE: No smooth relations found to build matrix.")
        return report

    M = relation_matrix['matrix']
    n_relations = M.nrows()
    rank = relation_matrix['rank']

    report.update({
        'relations_count': n_relations,
        'rank': rank,
        'over_determined': n_relations >= n_unknowns,
        'full_rank': rank == n_unknowns,
        'deficit_relations': max(0, n_unknowns - rank),
        'feasible': (n_relations >= n_unknowns) and (rank == n_unknowns)
    })

    if verbose:
        print(f"\n[Linear Algebra Attack Feasibility]")
        print(f"  Factor base size: {n_unknowns}")
        print(f"  Relations collected: {n_relations}")
        print(f"  Matrix rank: {rank}")
        print(f"  Estimated #J(GF({p})): ~{jacobian_size_estimate:.2e}")

        if report['feasible']:
            print(f"  ✓ ATTACK FEASIBLE: Matrix is full-rank and over-determined")
        else:
            print(f"  ✗ ATTACK NOT FEASIBLE: Need {report['deficit_relations']} more relations")

    return report

def index_calculus_factor_base_analysis(divisors, p, f_coeffs, verbose=True):
    """
    Complete factor base analysis for HECC index calculus over GF(p).
    """
    from sage.all import HyperellipticCurve, PolynomialRing, GF as SageGF

    try:
        R = PolynomialRing(SageGF(p), 'x')
        f_poly = sage_poly_from_coeffs(f_coeffs, R)
        x = R.gen()
        C = HyperellipticCurve(f_poly)
        g = C.genus()
    except Exception:
        g = 2  # Default to genus 2 if curve construction fails
        raise

    if verbose:
        print("\n" + "="*70)
        print("INDEX CALCULUS FEASIBILITY ANALYSIS")
        print("="*70)
        print(f"Field: GF({p})")
        print(f"Genus: {g}")
        print(f"Divisors collected: {len(divisors)}")

    factor_base = extract_factor_base(divisors, p, f_poly, verbose=verbose)
    smoothness_report = analyze_smoothness_distribution(divisors, p, factor_base, verbose=verbose)
    relation_matrix = build_relation_matrix(divisors, factor_base, p, verbose=verbose)
    attack_report = assess_linear_algebra_attack(relation_matrix, factor_base, p, g, verbose=verbose)
    complexity_report = estimate_dlp_complexity(len(factor_base), len(divisors), p, g, verbose=verbose)

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

def analyze_preferred_hit_rate(divisors, preferred_set, p=None):
    """
    Diagnostic to see how well we biased the factor base.
    """
    if not preferred_set:
        return

    pref = set(int(x) for x in preferred_set)
    hits = set()
    divisors_with_hit = 0

    for d in divisors:
        # Extract roots
        roots = []
        if 'roots' in d and d['roots']:
             roots = [int(r) for r in d['roots']]
        elif p:
             s, pp = int(d['s']), int(d['p'])
             disc = (s*s - 4*pp) % p
             if pow(disc, (p-1)//2, p) == 1:
                 delta = GF(p)(disc).sqrt()
                 inv2 = pow(2, -1, p)
                 roots = [int((s+delta)*inv2), int((s-delta)*inv2)]

        has_hit = False
        for r in roots:
            if r in pref:
                hits.add(r)
                has_hit = True

        if has_hit:
            divisors_with_hit += 1

    print(f"\n[Preferred Coordinates Hit Rate]")
    print(f"  Target coordinates: {len(pref)}")
    print(f"  Found in Factor Base: {len(hits)} / {len(pref)} ({100.0*len(hits)/max(1, len(pref)):.1f}%)")
    print(f"  Divisors containing target: {divisors_with_hit} / {len(divisors)} ({100.0*divisors_with_hit/max(1, len(divisors)):.1f}%)")

# unified extract_factor_base: backward-compatible

# In smoothness.py, REPLACE build_relation_matrix function (around line 257)
# with this corrected version that matches index_calculus.py sign convention:

# --- start patch (put in smoothness.py) ---

def _canonical_y_at_x(v_poly, x_elem, f_p, p):
    """
    Return canonical y = v(x) mod p normalized to the canonical representative.
    For y=0 returns 0. For nonzero return min(y, p-y).
    """
    K = GF(p)
    xK = K(x_elem)
    y_val = int(v_poly(xK)) % p
    if y_val == 0:
        return 0
    # canonical sqrt of f(x) (for debugging/consistency) -- but take v(x) as authoritative
    return int(min(y_val, p - y_val))

def atoms_from_mumford(div, f_p, p):
    """
    Given a Mumford divisor `div` (u_poly, v_poly) over GF(p),
    return a list of (atom_key, multiplicity) pairs where atom_key is canonical:
      - degree-1: ('d1', x_int, y_can)
      - degree-2: ('d2', tuple(u_coeffs), tuple(v_mod_u_coeffs))
    Returns None if u_poly does not factor (i.e., not smooth over GF(p)).
    """
    K = GF(p)
    R = f_p.parent()
    u_poly = div[0]
    v_poly = div[1]

    # Factor u over K
    try:
        facs = u_poly.factor()
    except Exception:
        raise
        return None

    # If factoring yields pieces whose degrees don't sum to deg(u), bail (not fully split)
    deg_sum = sum([fac.degree()*mult for fac, mult in facs])
    if deg_sum != u_poly.degree():
        return None

    atoms = []
    for fac, mult in facs:
        deg = fac.degree()
        if deg == 1:
            # get root
            roots = fac.roots(K)
            if not roots:
                return None
            x_elem = roots[0][0]
            x_int = int(x_elem)
            y_can = _canonical_y_at_x(v_poly, x_elem, f_p, p)
            atom = ('d1', x_int, int(y_can))
            atoms.append((atom, int(mult)))
        elif deg == 2:
            # Store canonical coefficients for u (monic) and v mod u
            u_coeffs = tuple(int(c) for c in fac.change_ring(K).list())
            v_mod = (v_poly % fac).change_ring(K)
            # normalize v_mod coeffs to length < 2
            v_coeffs = tuple(int(c) for c in v_mod.list())
            atom = ('d2', u_coeffs, v_coeffs)
            atoms.append((atom, int(mult)))
        else:
            # If support contains higher-degree prime, treat as not smooth
            return None

    return atoms

# --- end patch ---

# --- start patch (put in index_calculus.py) ---
# assumes: from smoothness import atoms_from_mumford, extract_factor_base

def extract_factor_base(sample_divisors, p, f_p=None, verbose=False):
    """
    Canonical prime-divisor factor base extractor.
    Returns:
        atom_to_idx : dict(atom -> col index)
        fb_y_cache  : dict(x -> canonical y)
    """
    if f_p is None:
        raise ValueError("extract_factor_base now REQUIRES f_p")

    atom_to_idx = {}
    fb_y_cache = {}
    next_idx = 0
    K = GF(p)
    R = PolynomialRing(K, 'x')

    for D in sample_divisors:
        # normalize divisor - handle multiple formats
        if isinstance(D, dict):
            if 'u_coeffs' in D and 'v_coeffs' in D:
                u = R(D['u_coeffs'])
                v = R(D['v_coeffs'])
            elif 's' in D and 'p' in D:
                # Build from (s, p, v_0, v_1) format
                x = R.gen()
                s = int(D['s'])
                pp = int(D['p'])
                v0 = int(D.get('v_0', 0))
                v1 = int(D.get('v_1', 0))
                u = x**2 - K(s)*x + K(pp)
                v = K(v1)*x + K(v0)
            else:
                raise ValueError(f"Divisor dict missing required keys: {D.keys()}")
        else:
            # Assume it's a Sage Jacobian element
            u, v = D[0], D[1]

        deg = u.degree()

        if deg == 1:
            # Degree-1 divisor: create d1 atom
            x_val = int(u.roots()[0][0])
            y_val = int(v(K(x_val)))
            y_val = min(y_val, p - y_val)
            atom = ('d1', x_val, y_val)
            fb_y_cache[x_val] = y_val

            if atom not in atom_to_idx:
                atom_to_idx[atom] = next_idx
                next_idx += 1

        elif deg == 2:
            # Check if u splits completely
            try:
                roots = u.roots(K)
                splits_completely = sum(m for _, m in roots) == 2
            except:
                splits_completely = False

            if splits_completely:
                # CRITICAL: Create BOTH d2 atom AND d1 atoms for roots

                # 1. Create d2 atom for the divisor itself
                u_coeffs = tuple(int(c) for c in u.list())
                v_coeffs = tuple(int(c) for c in v.list())
                atom_d2 = ('d2', u_coeffs, v_coeffs)

                if atom_d2 not in atom_to_idx:
                    atom_to_idx[atom_d2] = next_idx
                    next_idx += 1

                # 2. Also create d1 atoms for the individual roots
                for root, _ in roots:
                    x_val = int(root)
                    y_val = int(v(K(x_val)))
                    y_val = min(y_val, p - y_val)
                    atom_d1 = ('d1', x_val, y_val)
                    fb_y_cache[x_val] = y_val

                    if atom_d1 not in atom_to_idx:
                        atom_to_idx[atom_d1] = next_idx
                        next_idx += 1
            else:
                # Doesn't split: create only d2 atom
                u_coeffs = tuple(int(c) for c in u.list())
                v_coeffs = tuple(int(c) for c in v.list())
                atom_d2 = ('d2', u_coeffs, v_coeffs)

                if atom_d2 not in atom_to_idx:
                    atom_to_idx[atom_d2] = next_idx
                    next_idx += 1

    if verbose:
        d1_count = sum(1 for atom in atom_to_idx.keys() if atom[0] == 'd1')
        d2_count = sum(1 for atom in atom_to_idx.keys() if atom[0] == 'd2')
        print(f"[Factor Base] {len(atom_to_idx)} prime atoms ({d1_count} d1, {d2_count} d2)")

    return atom_to_idx, fb_y_cache

def build_relation_matrix(divisors, factor_base, p=None, f_p=None, verbose=False, debug=False):
    """
    Constructs the relation matrix for Index Calculus.
    Maps each divisor to a row representing its decomposition over the factor base.

    Args:
        divisors: List of smooth divisor dictionaries.
        factor_base: List of atoms (d1 or d2).
        p: The prime field characteristic.
        f_p: The curve polynomial mod p.
        verbose: Print progress and matrix stats.
        debug: If True, performs the expensive M.rank() check.
    """
    if verbose:
        print(f"  [Matrix] Building matrix from {len(divisors)} relations over {len(factor_base)} FB elements...")
        sys.stdout.flush()

    # 1. Map factor base atoms to column indices
    # We use the atom itself as the key.
    # d1 atoms: ('d1', x, y) where y = min(y, p-y)
    # d2 atoms: ('d2', u_coeffs, v_coeffs)
    atom_to_idx = {atom: i for i, atom in enumerate(factor_base)}
    num_columns = len(factor_base)
    matrix_rows = []

    # If we are in a Finite Field context, we need the ring for evaluations
    K = GF(p) if p else None

    # 2. Process each divisor to build a sparse row
    for d in divisors:
        row = [0] * num_columns

        # A divisor in this library typically has 'u' and 'v' polynomials (Mumford representation)
        u = d.get('u')
        v = d.get('v')

        if u is None or v is None:
            continue

        # Decomposition logic:
        # We look at the roots of u(x) to determine the points (d1) or
        # the irreducible factors (d2).

        if u.degree() == 0:
            continue # Zero divisor / Identity

        # Split u into its irreducible factors over GF(p)
        factors = u.factor()

        possible_relation = True
        current_row_data = Counter()

        for poly, mult in factors:
            if poly.degree() == 1:
                # Linear factor -> Point (d1)
                x_val = int(-poly.constant_coefficient())
                y_val = int(v(K(x_val))) if K else int(v(x_val))

                # Normalize y-coordinate to match the factor base's "min(y, p-y)" convention
                # If y > p-y, it means this is the negative of the FB point.
                if p and y_val > p // 2:
                    y_norm = p - y_val
                    atom = ('d1', x_val, y_norm)
                    weight = -1 * mult
                else:
                    atom = ('d1', x_val, y_val)
                    weight = 1 * mult

                if atom in atom_to_idx:
                    current_row_data[atom_to_idx[atom]] += weight
                else:
                    possible_relation = False
                    break

            elif poly.degree() == 2:
                # Quadratic factor -> (d2) atom
                # Normalize the quadratic poly so it's monic (already monic from .factor())
                u_coeffs = tuple(int(c) for c in poly.list())

                # For v(x), we often reduce it mod poly
                v_red = v % poly
                v_coeffs = tuple(int(c) for c in v_red.list())

                atom = ('d2', u_coeffs, v_coeffs)

                if atom in atom_to_idx:
                    current_row_data[atom_to_idx[atom]] += mult
                else:
                    # Check if the "negative" of the d2 atom is in the FB
                    # In Genus 2, -(u, v) = (u, -v)
                    v_neg_coeffs = tuple(int(-c % p) for c in v_red.list())
                    atom_neg = ('d2', u_coeffs, v_neg_coeffs)

                    if atom_neg in atom_to_idx:
                        current_row_data[atom_to_idx[atom_neg]] -= mult
                    else:
                        possible_relation = False
                        break
            else:
                # Higher degree irreducible factors are not smooth
                possible_relation = False
                break

        if possible_relation:
            # Convert Counter to a row list
            row_vec = [0] * num_columns
            for idx, val in current_row_data.items():
                row_vec[idx] = val
            matrix_rows.append(row_vec)

    # 3. Construct Matrix and Rank
    if not matrix_rows:
        M = matrix(ZZ, 0, num_columns)
        rank = 0
    else:
        # We use ZZ because relations can have negative coefficients
        M = matrix(ZZ, matrix_rows)

        # CRITICAL PERFORMANCE GATE:
        # Matrix rank is O(N^3) or O(N^2) depending on sparsity.
        # We skip it in production to avoid the "expensive rank check" bottleneck.
        if debug:
            if verbose:
                print(f"  [Debug] Computing rank of {M.nrows()}x{M.ncols()} matrix...")
            rank = M.rank()
        else:
            rank = None

    if verbose:
        rank_str = str(rank) if rank is not None else "(skipped)"
        print(f"  [Matrix] Final: {M.nrows()} rows, {M.ncols()} cols. Rank: {rank_str}")
        sys.stdout.flush()

    return {
        'matrix': M,
        'rank': rank,
        'atom_to_idx': atom_to_idx,
        'num_relations': len(matrix_rows)
    }

def diagnose_finite_field_search(divisors, f_p, verbose=True, debug=False):
    """
    Comprehensive diagnostic for the Index Calculus attack state.
    Checks factor base quality, key divisor smoothness (G and Q), and matrix rank.
    """
    p = f_p.base_ring().order()

    # 1. Extract the factor base from the discovered smooth divisors
    # This typically collects all unique d1 (points) and d2 (quadratic) atoms.
    fb_res = extract_factor_base(divisors, p, f_p, verbose=False)
    fb_roots = fb_res['factor_base']
    atom_to_idx = fb_res['atom_to_idx']
    fb_y_cache = fb_res['fb_y_cache']

    # 2. Build the relation matrix
    # The debug flag here controls whether the expensive M.rank() is performed.
    res = build_relation_matrix(
        divisors,
        fb_roots,
        p=p,
        f_p=f_p,
        verbose=False,
        debug=debug
    )

    rank = res['rank']
    needed = len(fb_roots)

    # 3. Check smoothness of the Base (G) and Target (Q) divisors
    # These are usually defined in search_common or passed via globals.
    G = globals().get('BASE_DIVISOR')
    Q = globals().get('TARGET_DIVISOR')

    def check_smoothness(D):
        if D is None:
            return False, []
        try:
            # A divisor is smooth if all its points (x-coordinates)
            # are present in the factor base.
            u_poly = D.u()
            roots = [int(r[0]) for r in u_poly.roots(GF(p))]
            # If the number of roots mod p matches the degree, it's d1-smooth
            if len(roots) < u_poly.degree():
                return False, roots

            is_smooth = all(r in fb_y_cache for r in roots)
            return is_smooth, roots
        except Exception:
            return False, []

    G_is_smooth, G_roots = check_smoothness(G)
    Q_is_smooth, Q_roots = check_smoothness(Q)

    # 4. Final Diagnostic Output
    if verbose:
        print("\n" + "="*70)
        print(" INDEX CALCULUS ATTACK DIAGNOSTIC")
        print("="*70)
        print(f" Target Field:   GF({p})")
        print(f" Relations:      {len(divisors)}")
        print(f" Factor Base:    {len(fb_roots)} atoms")

        # Display rank or notice of skip
        if rank is not None:
            deficit = max(0, needed - rank)
            rank_status = f"{rank} (Deficit: {deficit})"
        else:
            rank_status = "SKIPPED (debug=False)"

        print(f" Matrix Rank:    {rank_status}")
        print("-" * 70)

        # Smoothness status
        G_status = "SMOOTH" if G_is_smooth else "NOT SMOOTH"
        Q_status = "SMOOTH" if Q_is_smooth else "NOT SMOOTH"
        print(f" Base Divisor G: {G_status} (Roots: {G_roots})")
        print(f" Target Divisor Q: {Q_status} (Roots: {Q_roots})")
        print("-" * 70)

        # Verdict logic
        if rank is not None:
            if rank >= needed and G_is_smooth and Q_is_smooth:
                print(" SUCCESS: Matrix is full rank and keys are smooth.")
                print("          Linear algebra (Wiedemann/Lanczos) can now proceed.")
            else:
                if rank < needed:
                    print(f" FAILURE: Need {needed - rank} more independent relations.")
                if not G_is_smooth or not Q_is_smooth:
                    print(" FAILURE: One or both keys are not smooth over current factor base.")
                print(" Action: Increase TMAX or add more search vectors to find more relations.")
        else:
            print(" INFO: Matrix constructed. Check G/Q smoothness above.")
            if G_is_smooth and Q_is_smooth:
                print("       Keys are smooth. Attempt linear algebra if relation count is > FB size.")
            else:
                print("       Keys are not yet smooth. More relations required.")

        print("="*70 + "\n")
        sys.stdout.flush()

    return {
        'factor_base': fb_roots,
        'atom_to_idx': atom_to_idx,
        'fb_y_cache': fb_y_cache,
        'matrix': res['matrix'],
        'rank': rank,
        'G_is_smooth': G_is_smooth,
        'Q_is_smooth': Q_is_smooth,
        'G_roots': G_roots,
        'Q_roots': Q_roots,
        'num_relations': len(divisors)
    }
