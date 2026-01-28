from sage.all import *
from collections import defaultdict
from search_common import DEBUG

# diagnostics_univariate.py
"""
Univariate search equation diagnostics.
Analyzes why symbolic search may fail while modular search succeeds.
Checks for Galois obstructions on the torsor equations x([n]P)(m) = -m + x1.
"""

"""
Univariate search equation diagnostics.
Analyzes why symbolic search may fail while modular search succeeds.
Checks for Galois obstructions on the torsor equations x([n]P)(m) = -m + x1.
"""

def extract_search_polynomials(cd, current_sections, rhs_list, vecs, max_n=12):
    """
    Extract the actual univariate search polynomials F_n(m) = x([n]P)(m) + m - x1
    for the first max_n multiples of each section.

    Returns: dict mapping (section_idx, n, rhs_idx) -> polynomial in QQ[m]
    """
    assert current_sections, "current_sections cannot be empty"

    PR_m = PolynomialRing(QQ, 'm')
    Fm = PR_m.fraction_field()
    m_var = PR_m.gen()
    SR_m = var('m')

    polys = {}

    for sec_idx, section in enumerate(current_sections):
        if section[2].is_zero():
            continue

        x_P = section[0] / section[2]
        x_P_Fm = Fm(SR(x_P))

        current_sec = section
        for n in range(1, max_n + 1):
            if current_sec[2].is_zero():
                break

            x_nP = current_sec[0] / current_sec[2]
            x_nP_Fm = Fm(SR(x_nP))

            for rhs_idx, rhs_func in enumerate(rhs_list):
                rhs_Fm = Fm(SR(rhs_func))
                F = x_nP_Fm - rhs_Fm

                num = F.numerator()
                den = F.denominator()

                num_pr = PR_m(num)
                den_pr = PR_m(den)

                g = gcd(num_pr, den_pr)
                if g.degree() > 0 or g != 1:
                    num_pr = num_pr // g
                    den_pr = den_pr // g

                content_num = gcd(num_pr.coefficients())
                if content_num != 1:
                    num_pr = PR_m([QQ(c) / QQ(content_num) for c in num_pr.list()])

                if num_pr.degree() > 0:
                    polys[(sec_idx, n, rhs_idx)] = num_pr

            current_sec = elliptic_add_generic(current_sec, section, cd)

    return polys

def elliptic_add_generic(P, Q, cd):
    """
    Generic elliptic curve addition for sections over QQ(m).
    Handles y^2 = x^3 + a4*x + a6 curves.
    """
    if P[2].is_zero():
        return Q
    if Q[2].is_zero():
        return P

    x1, y1, z1 = P[0], P[1], P[2]
    x2, y2, z2 = Q[0], Q[1], Q[2]

    x1_aff = x1 / z1
    y1_aff = y1 / z1
    x2_aff = x2 / z2
    y2_aff = y2 / z2

    if x1_aff == x2_aff:
        if y1_aff == -y2_aff:
            return (cd.a4.parent().zero(), cd.a4.parent().one(), cd.a4.parent().zero())
        s = (3 * x1_aff**2 + cd.a4) / (2 * y1_aff)
        x3 = s**2 - 2*x1_aff
        y3 = s*(x1_aff - x3) - y1_aff
        return (x3, y3, cd.a4.parent().one())
    else:
        s = (y2_aff - y1_aff) / (x2_aff - x1_aff)
        x3 = s**2 - x1_aff - x2_aff
        y3 = s*(x1_aff - x3) - y1_aff
        return (x3, y3, cd.a4.parent().one())

def analyze_polynomial(F, prime_bound=100):
    """
    Analyze a single polynomial F in QQ[m]:
    - degree
    - irreducibility
    - factorization
    - Galois group (when feasible)
    - rational roots
    - local solubility at primes up to prime_bound

    Returns: dict with analysis results
    """
    result = {
        'degree': F.degree(),
        'irreducible': None,
        'factorization': None,
        'galois_group': None,
        'rational_roots': [],
        'local_obstructions': [],
        'locally_soluble_primes': []
    }

    if F.degree() == 0:
        return result

    result['irreducible'] = F.is_irreducible()
    if not result['irreducible']:
        result['factorization'] = F.factor()

    roots = F.roots(QQ, multiplicities=False)
    result['rational_roots'] = [QQ(r) for r in roots]

    primes_to_check = [p for p in primes_first_n(prime_bound) if p < prime_bound]

    for p in primes_to_check:
        has_root = check_local_root(F, p, k=3)
        if has_root:
            result['locally_soluble_primes'].append(p)
        else:
            result['local_obstructions'].append(p)

    if result['irreducible'] and F.degree() <= 8:
        K = F.splitting_field('a')
        G = K.galois_group()
        result['galois_group'] = str(G.structure_description())
    elif result['irreducible'] and F.degree() > 8:
        result['galois_group'] = f"S{F.degree()} (assumed, deg > 8)"

    return result

def check_local_root(F, p, k=3):
    """
    Check if F has a root in QQ_p via Hensel lifting.
    First checks mod p, then lifts to p^k.

    Returns: True if locally soluble, False otherwise
    """
    Fp = GF(p)
    Fpx = PolynomialRing(Fp, 'x')

    coeffs_mod_p = []
    for c in F.list():
        c_qq = QQ(c)
        num = ZZ(c_qq.numerator())
        den = ZZ(c_qq.denominator())
        if den % p == 0:
            return False
        den_inv = inverse_mod(den, p)
        coeffs_mod_p.append((num * den_inv) % p)

    F_mod_p = Fpx(coeffs_mod_p)

    roots_mod_p = F_mod_p.roots(multiplicities=False)
    if not roots_mod_p:
        return False

    r0 = ZZ(roots_mod_p[0])

    current_r = r0
    current_mod = p

    for _ in range(k - 1):
        F_val = QQ(F(current_r))
        F_prime_val = QQ(F.derivative()(current_r))

        F_val_num = ZZ(F_val.numerator())
        F_val_den = ZZ(F_val.denominator())
        F_prime_num = ZZ(F_prime_val.numerator())
        F_prime_den = ZZ(F_prime_val.denominator())

        if F_prime_num % p != 0 and F_prime_den % p != 0:
            F_val_times_den = F_val_num * inverse_mod(F_val_den, current_mod)
            if F_val_times_den % current_mod != 0:
                F_prime_inv = (F_prime_num * inverse_mod(F_prime_den, p)) % p
                if F_prime_inv != 0:
                    lift = ((F_val_times_den // current_mod) * inverse_mod(F_prime_inv, p)) % p
                    current_r = current_r - lift * current_mod
                    current_mod = current_mod * p
                else:
                    return False
            else:
                current_mod = current_mod * p
        else:
            F_val_final = QQ(F(current_r))
            F_val_final_num = ZZ(F_val_final.numerator())
            F_val_final_den = ZZ(F_val_final.denominator())
            if F_val_final_den % (p**k) != 0:
                return (F_val_final_num * inverse_mod(F_val_final_den, p**k)) % (p**k) == 0
            return False

    F_final = QQ(F(current_r))
    F_final_num = ZZ(F_final.numerator())
    F_final_den = ZZ(F_final.denominator())
    if F_final_den % (p**k) != 0:
        return (F_final_num * inverse_mod(F_final_den, p**k)) % (p**k) == 0
    return False

def classify_obstruction(analysis):
    """
    Classify the obstruction type based on polynomial analysis.

    Returns: one of:
        "LOCAL_OBSTRUCTION" - fails locally at some prime
        "ADELIC_BUT_NO_Q" - locally soluble everywhere but no rational roots
        "DEFINITE_GALOIS_OBSTRUCTION" - transitive Galois group + no rational roots
        "HAS_RATIONAL_ROOTS" - has rational solutions
        "UNDECIDED" - cannot determine
    """
    if analysis['rational_roots']:
        return "HAS_RATIONAL_ROOTS"

    if analysis['local_obstructions']:
        return "LOCAL_OBSTRUCTION"

    if analysis['irreducible'] and analysis['galois_group']:
        if 'S' in str(analysis['galois_group']) or 'Symmetric' in str(analysis['galois_group']):
            return "DEFINITE_GALOIS_OBSTRUCTION"

    if not analysis['local_obstructions'] and not analysis['rational_roots']:
        return "ADELIC_BUT_NO_Q"

    return "UNDECIDED"

def print_polynomial_diagnostics(key, analysis):
    """
    Print diagnostics for a single polynomial.
    """
    sec_idx, n, rhs_idx = key

    print(f"\nSection {sec_idx}, n={n}, RHS {rhs_idx}:")
    print(f"  degree = {analysis['degree']}")
    print(f"  irreducible = {analysis['irreducible']}")

    if analysis['galois_group']:
        print(f"  Galois group = {analysis['galois_group']}")

    if analysis['rational_roots']:
        print(f"  rational roots = {analysis['rational_roots']}")
    else:
        print(f"  rational roots = NONE")

    obstruction = classify_obstruction(analysis)

    if obstruction == "LOCAL_OBSTRUCTION":
        print(f"  LOCAL OBSTRUCTION at primes: {analysis['local_obstructions'][:5]}")
    elif obstruction == "DEFINITE_GALOIS_OBSTRUCTION":
        print(f"  DEFINITE GALOIS OBSTRUCTION (no rational roots + high Galois group)")
    elif obstruction == "ADELIC_BUT_NO_Q":
        print(f"  ADELIC NO-Q ROOT (locally soluble everywhere, no global roots)")
        print(f"     Locally soluble at: {len(analysis['locally_soluble_primes'])} primes tested")
    elif obstruction == "HAS_RATIONAL_ROOTS":
        print(f"  HAS RATIONAL ROOTS: {analysis['rational_roots']}")
    else:
        print(f"  UNDECIDED")

def run_univariate_diagnostics(cd, current_sections, rhs_list, vecs, max_n=12):
    """
    Main entry point for univariate diagnostics.
    Analyzes search polynomials and prints summary.
    """
    print("\n" + "="*70)
    print("UNIVARIATE SEARCH EQUATION DIAGNOSTICS")
    print("="*70)

    assert current_sections, "No sections to analyze"

    print(f"Extracting search polynomials for {len(current_sections)} section(s)...")
    print(f"Computing up to [{max_n}]P for each section...")

    polys = extract_search_polynomials(cd, current_sections, rhs_list, vecs, max_n=max_n)

    assert polys, "No polynomials extracted"

    print(f"Extracted {len(polys)} search polynomial(s).")
    print("\nAnalyzing polynomials...")

    analyses = {}
    obstruction_counts = defaultdict(int)

    for key, F in polys.items():
        analysis = analyze_polynomial(F, prime_bound=100)
        analyses[key] = analysis
        obstruction = classify_obstruction(analysis)
        obstruction_counts[obstruction] += 1

    print("\n" + "-"*70)
    print("SUMMARY STATISTICS")
    print("-"*70)
    print(f"Total polynomials analyzed: {len(polys)}")
    print(f"\nObstruction types:")
    for obs_type, count in sorted(obstruction_counts.items()):
        print(f"  {obs_type}: {count}")

    print("\n" + "-"*70)
    print("DETAILED DIAGNOSTICS (sample)")
    print("-"*70)

    printed_per_type = defaultdict(int)
    #max_print_per_type = 3
    max_print_per_type = max_n

    for key in sorted(polys.keys()):
        analysis = analyses[key]
        obstruction = classify_obstruction(analysis)

        if printed_per_type[obstruction] < max_print_per_type:
            print_polynomial_diagnostics(key, analysis)

            sec_idx, n, rhs_idx = key
            F = polys[key]
            coeffs = F.list()
            content = gcd([abs(c.numerator()) for c in coeffs if c != 0])
            max_coeff = max([abs(c.numerator()) for c in coeffs] + [abs(c.denominator()) for c in coeffs])

            print(f"     Polynomial degree: {F.degree()}")
            print(f"     Content (gcd of coeffs): {content}")
            print(f"     Max coefficient size: {max_coeff}")
            print(f"     Leading coeff: {F.leading_coefficient()}")

            printed_per_type[obstruction] += 1

    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if obstruction_counts["HAS_RATIONAL_ROOTS"] > 0:
        print("Some search equations have rational roots.")
        print("Symbolic search should find points.")
    elif obstruction_counts["LOCAL_OBSTRUCTION"] > 0:
        print("Local obstructions detected at small primes.")
        print("This explains why symbolic search fails.")
        print("Mod-p search succeeds because it only checks individual primes.")
    elif obstruction_counts["DEFINITE_GALOIS_OBSTRUCTION"] > 0:
        print("Definite Galois obstructions detected.")
        print("Search equations are irreducible with large Galois groups.")
        print("No rational roots exist - this is a fundamental obstruction.")
    elif obstruction_counts["ADELIC_BUT_NO_Q"] > 0:
        print("Adelic obstruction detected.")
        print("Locally soluble everywhere but no global rational roots.")
        print("This is a subtle Galois/Brauer-Manin type obstruction.")
    else:
        print("Unable to determine obstruction type.")

    print("="*70 + "\n")

    return analyses
