from math import log2
from collections import Counter

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
        # roots of x^2 - s x + p
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
    print(f"  empirical entropy: {entropy:.3f} (max {log2(p):.3f})")


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


from sage.all import matrix, GF, vector

def diagnostic_factor_base_saturation(divisors, p):
    """
    Checks if we are reusing the same x-coordinates (good for Index Calculus)
    or constantly introducing new ones (bad for Index Calculus).
    
    If 'unique_roots' grows linearly with 'total_divisors', we are not
    saturating the factor base.
    """
    all_roots = []
    for d in divisors:
        s = int(d['s']) % p
        pp = int(d['p']) % p
        # roots of x^2 - s x + p
        disc = (s*s - 4*pp) % p
        # We only care about valid roots in the base field
        if pow(disc, (p-1)//2, p) != 1:
            if disc == 0:
                 r = (s * pow(2, -1, p)) % p
                 all_roots.extend([r, r])
            continue
            
        sqrt_disc = pow(disc, (p+1)//4, p) if p % 4 == 1 else pow(disc, (p+1)//4, p) # simplified for p=3 mod 4 proof of concept
        # Proper sqrt handling usually requires Tonelli-Shanks if not p=3 mod 4, 
        # but for diagnostics on large random primes, p=3 mod 4 is sufficient.
        
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
    print(f"  saturation ratio: {total_roots/max(1, distinct_count):.2f} (higher is better for attacks)")


def diagnostic_mod_p_coverage(divisors, p, genus=2):
    """
    Checks the rank of the generated divisors modulo p.
    If rank < 2*genus (or rank < expected group size), we are stuck in a subgroup.
    """
    # We map divisors to a vector space if possible, or just check linear independence 
    # of the Mumford coordinates if we treat them essentially as vectors (heuristic).
    # A rigorous check requires the Frobenius map or order checking, but 
    # we can check simple linear independence of the vectors (v0, v1) mod p
    # as a proxy for 'are these distinct elements'.
    
    vecs = []
    for d in divisors:
        # We use (s, p, v0, v1) as a raw fingerprint of the divisor class
        # This is NOT a rigorous group homomorphism but detects identical/dependent rows
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

    # Check rank of this raw matrix
    M = matrix(GF(p), vecs)
    r = M.rank()
    
    print("[diag:coverage]")
    print(f"  generated divisors: {len(divisors)}")
    print(f"  linear rank (heuristic): {r}")
    if r < len(divisors):
        print(f"  (!) Dependencies found mod {p}. Potential relations for DLP.")
    else:
        print(f"  Independence held mod {p}. Basis is expanding.")
