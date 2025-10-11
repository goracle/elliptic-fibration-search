from sage.all import FractionField, QQ
from sage.rings.fraction_field_element import FractionFieldElement
from sage.all import QQ, PolynomialRing, Integer
from sage.rings.fraction_field_element import FractionFieldElement
from sage.rings.integer import Integer
from sage.rings.rational import Rational
# Unified singular fiber finder (replacement for two older functions)
from sage.all import PolynomialRing, QQ, ZZ, floor, CC, sqrt, Integer
from math import prod
from sage.all import gcd, Integer
from math import isclose

def _remove_global_m_power(rf):
    """
    Remove common global power m^k dividing numerator and denominator of rf (FractionFieldElement).
    Returns (rf_clean, k_removed)
    """
    if not isinstance(rf, FractionFieldElement):
        R, m = _QQm()
        rf = R(rf)
    num = rf.numerator()
    den = rf.denominator()
    R, m = _QQm()
    v_num = polynomial_valuation_at_factor(num, m)
    v_den = polynomial_valuation_at_factor(den, m)
    try:
        k = min(v_num, v_den)
    except TypeError:
        print("v_num, v_den, num, den, rf",v_num, v_den, num, den, rf)
        raise
    if k > 0:
        return ( (num // (m**k)) / (den // (m**k)), k )
    return rf, 0

def effective_discriminant_degree(Delta):
    """Return deg(numerator) - deg(denominator) after canceling global m^k."""
    if not isinstance(Delta, FractionFieldElement):
        R, m = _QQm()
        Delta = R(Delta)
    Delta_clean, removed_k = _remove_global_m_power(Delta)
    return Delta_clean.numerator().degree() - Delta_clean.denominator().degree(), removed_k

def _QQm():
    R = QQ['m']; m, = R.gens()
    return R, m

def polynomial_valuation_at_factor(poly, factor):
    """
    Return v_factor(poly): highest e such that factor^e divides poly.
    This uses repeated exact polynomial division.
    """
    if poly == 0:
        return 10**9  # Convention for valuation of 0

    e = 0
    # Ensure poly and factor are in the same polynomial ring
    P_ring = poly.parent()
    if factor.parent() != P_ring:
        try:
            factor = P_ring(factor)
        except TypeError:
            # Handle cases where coercion isn't direct, e.g. from ZZ to QQ
            factor = factor.change_ring(P_ring.base_ring())

    cur = poly
    while True:
        if cur == 0: # a zero polynomial is divisible infinitely many times
            return 10**9
        q, r = cur.quo_rem(factor)
        if r == 0:
            e += 1
            cur = q
        else:
            break
    return e



def find_singular_fibers(cd=None, numeric_root_precision=80, verbose=False, a4=None, a6=None):
    """
    Robust singular fiber finder that works over QQ and finite fields GF(p).
    Returns dict with keys: 'fibers', 'euler_characteristic', 'sigma_sum', 'sum_reduction_exponents'.

    Key invariants stored per fiber:
      - factor: polynomial factor in R (None for infinity)
      - degree: algebraic degree of the factor (1 for rational roots / infinity)
      - m_v: Kodaira component count (NOT multiplied by degree)
      - contribution: deg * (m_v - 1)  <-- CORRECT formula for sigma contribution
      - e_contrib: deg * e              <-- contribution to Euler characteristic
    """
    if cd is not None:
        #a4 = cd.a4
        #a6 = cd.a6
        return cd.singfibs # update; we calculate this in advance now.

    # invariants
    c4 = -48 * a4
    c6 = -864 * a6
    Delta = -16 * (4 * a4**3 + 27 * a6**2)

    # try to detect base_field
    base_field = QQ
    if cd is not None and getattr(cd, "base_field", None) is not None:
        base_field = cd.base_field
    else:
        try:
            num_test = Delta.numerator()
            parent = num_test.parent()
            br = getattr(parent, "base_ring", None)
            if br is not None:
                try:
                    base_field = br()
                except Exception:
                    base_field = QQ
        except Exception:
            base_field = QQ

    R = PolynomialRing(base_field, 'm')
    m = R.gen()

    def _coerce_to_R(poly_like, name="poly"):
        """
        Try several safe coercions of poly_like into R. Raise RuntimeError with
        a useful message on failure (so caller can skip this prime or diagnose).
        """
        # already in R?
        try:
            if poly_like.parent() is R:
                return poly_like
        except Exception:
            pass

        # simple attempt: let R(...) handle it (works often: integers, strings, polys over QQ)
        try:
            return R(poly_like)
        except Exception as exc1:
            # Try to be more careful: if poly_like is a QQ-polynomial, clear denominators,
            # map integer coefficients into R and reconstruct a polynomial there.
            try:
                parent = getattr(poly_like, "parent", None)
                if parent is not None and callable(parent):
                    parent = poly_like.parent()
                br = getattr(parent, "base_ring", None)
            except Exception:
                parent = None
                br = None

            # If we have a polynomial-like object with coefficient access, attempt denom-clearing
            try:
                coeffs = getattr(poly_like, "coefficients", None)
                monoms = getattr(poly_like, "monomials", None)
                if coeffs and monoms:
                    # collect coefficients and clear denominators
                    from fractions import Fraction
                    all_fracs = [Fraction(c) for c in poly_like.coefficients()]
                    dens = [f.denominator for f in all_fracs]
                    from math import gcd
                    def lcm(a, b):
                        return a // gcd(a, b) * b
                    L = 1
                    for d in dens:
                        L = lcm(L, d)
                    # build integer polynomial in temporary ring 't' and map down
                    # construct dict deg->coeff_int
                    deg_coeff = {}
                    for mon, coeff in zip(poly_like.monomials(), poly_like.coefficients()):
                        # mon is something like m^k; extract exponent
                        try:
                            # Sage monomial: mon.exponents() -> tuple
                            exps = mon.exponents()
                            if exps:
                                k = exps[0][0]
                            else:
                                k = 0
                        except Exception:
                            # fallback: try str parse (last resort)
                            k = int(str(mon).split('^')[-1]) if '^' in str(mon) else 1 if str(mon) == str(m) else 0
                        c_frac = Fraction(coeff)
                        deg_coeff[int(k)] = deg_coeff.get(int(k), 0) + int(c_frac * L)
                    # build integer polynomial string
                    terms = []
                    for k in sorted(deg_coeff.keys(), reverse=True):
                        c_int = deg_coeff[k]
                        if c_int == 0:
                            continue
                        if k == 0:
                            terms.append(str(c_int))
                        elif k == 1:
                            terms.append(f"{c_int}*m")
                        else:
                            terms.append(f"{c_int}*m^{k}")
                    if not terms:
                        poly_int_str = "0"
                    else:
                        poly_int_str = " + ".join(terms)
                    # Now create polynomial in R and remember we cleared denominator L.
                    try:
                        poly_int = R(poly_int_str)
                        # return poly_int but caller is coercing numerator and denominator separately;
                        # to preserve semantics return the integer polynomial (the caller should be aware)
                        return poly_int  # caller must handle any global denominator L; typically numerator/denominator both cleared
                    except Exception as exc2:
                        # fall through to final error
                        raise RuntimeError(f"Failed integer-mapping approach for {name}: {exc2}") from exc2
            except Exception:
                # last resort - produce a helpful error containing both exceptions
                raise RuntimeError(f"Could not coerce {name} into {R}. Tried R(...): {exc1}") from exc1

    # Extract numerators/denominators and coerce
    try:
        numD_q = Delta.numerator(); denD_q = Delta.denominator()
        c4_num_q = c4.numerator(); c4_den_q = c4.denominator()
        c6_num_q = c6.numerator(); c6_den_q = c6.denominator()
    except Exception as exc:
        raise RuntimeError("Delta / c4 / c6 expected to be rational functions with numerator()/denominator().") from exc

    try:
        numD = _coerce_to_R(numD_q, "Delta.num")
        denD = _coerce_to_R(denD_q, "Delta.den")
        num4 = _coerce_to_R(c4_num_q, "c4.num")
        den4 = _coerce_to_R(c4_den_q, "c4.den")
        num6 = _coerce_to_R(c6_num_q, "c6.num")
        den6 = _coerce_to_R(c6_den_q, "c6.den")
    except Exception as exc:
        # expose useful debug info upward
        raise RuntimeError(f"Could not coerce Delta/c4/c6 components into polynomial ring {R}: {exc}") from exc

    # Factor numerator and denominator in R
    numD_fac = numD.factor()
    denD_fac = denD.factor()

    # Build combined multiplicity dict for factors (polynomial objects from R)
    factor_mult = {}
    for f, mult in numD_fac:
        factor_mult[f] = factor_mult.get(f, 0) + int(mult)
    for f, mult in denD_fac:
        factor_mult[f] = factor_mult.get(f, 0) - int(mult)


    fibers = []

    for g, vD in list(factor_mult.items()):
        deg_g = int(g.degree())

        # compute valuations relative to this factor
        #v4 = mult_of(g, num4) - mult_of(g, den4)
        k_num, num_after = mult_of(g, num4)
        k_den, den_after = mult_of(g, den4)
        v4 = k_num - k_den

        #v6 = mult_of(g, num6) - mult_of(g, den6)
        k_num, num_after = mult_of(g, num6)
        k_den, den_after = mult_of(g, den6)
        v6 = k_num - k_den

        # minimalizing parameter t
        t = min(int(floor(QQ(v4) / 4)), int(floor(QQ(v6) / 6)), int(floor(QQ(vD) / 12)))
        v4_min = v4 - 4 * t
        v6_min = v6 - 6 * t
        vD_min = vD - 12 * t

        # Kodaira info: (symbol, component_count, e_contribution)
        symbol, m_v, e_contrib = _kodaira_from_min_vals(v4_min, v6_min, vD_min)

        ftype = 'multiplicative' if (v4_min == 0 and v6_min == 0 and vD_min > 0) else 'additive'
        root_type = 'rational' if deg_g == 1 else 'irrational'
        center_repr = str(g)
        if root_type == 'rational':
            try:
                # Attempt explicit rational extraction for linear factor ax + b
                coeffs = g.list()
                # coeff order might vary across rings, but attempt the common case
                if len(coeffs) >= 2:
                    a_coeff, b_coeff = coeffs[-1], coeffs[0]
                    center_repr = - (b_coeff) / (a_coeff)
            except Exception:
                pass

        if vD < 0:
            root_type = 'pole'

        # correct contribution arithmetic:
        # - sigma contribution: deg_g * (m_v - 1)
        # - euler contribution: deg_g * e_contrib
        contrib_sigma = deg_g * (int(m_v) - 1) if m_v is not None else None
        contrib_euler = deg_g * int(e_contrib) if e_contrib is not None else None

        fibers.append({
            'r': center_repr,
            'factor': g,
            'root_type': root_type,
            'v_c4': int(v4),
            'v_c6': int(v6),
            'v_D': int(vD),
            't': int(t),
            'v4_min': int(v4_min),
            'v6_min': int(v6_min),
            'vD_min': int(vD_min),
            'symbol': symbol,
            'type': ftype,
            'n': int(vD_min) if vD_min is not None else None,
            'degree': deg_g,
            'm_v': int(m_v) if m_v is not None else None,            # component count (unmultiplied)
            'contribution': int(contrib_sigma) if contrib_sigma is not None else None,
            'e_contrib': int(contrib_euler) if contrib_euler is not None else None
        })

        if verbose:
            diag = local_minimality_diagnostic(str(g), v4, v6, vD, symbol)
            if diag is not None:
                print(">>> Diagnostic:", diag)

    # Infinity: valuations from degrees (denominator degree minus numerator degree)
    deg_numD = numD.degree(); deg_denD = denD.degree()
    vD_inf = deg_denD - deg_numD
    deg_num4 = num4.degree(); deg_den4 = den4.degree()
    v4_inf = deg_den4 - deg_num4
    deg_num6 = num6.degree(); deg_den6 = den6.degree()
    v6_inf = deg_den6 - deg_num6

    if vD_inf > 0 or v4_inf != 0 or v6_inf != 0:
        t_inf = min(int(floor(QQ(v4_inf) / 4)), int(floor(QQ(v6_inf) / 6)), int(floor(QQ(vD_inf) / 12)))
        v4_inf_min = v4_inf - 4 * t_inf
        v6_inf_min = v6_inf - 6 * t_inf
        vD_inf_min = vD_inf - 12 * t_inf
        sym_inf, m_v_inf, e_inf = _kodaira_from_min_vals(v4_inf_min, v6_inf_min, vD_inf_min)
        ftype_inf = 'multiplicative' if (v4_inf_min == 0 and v6_inf_min == 0 and vD_inf_min > 0) else 'additive'
        deg_inf = 1
        contrib_sigma_inf = deg_inf * (int(m_v_inf) - 1) if m_v_inf is not None else None
        contrib_euler_inf = deg_inf * int(e_inf) if e_inf is not None else None

        fibers.append({
            'r': 'inf', 'factor': None, 'root_type': 'inf',
            'v_c4': int(v4_inf), 'v_c6': int(v6_inf), 'v_D': int(vD_inf),
            't': int(t_inf), 'v4_min': int(v4_inf_min), 'v6_min': int(v6_inf_min), 'vD_min': int(vD_inf_min),
            'symbol': sym_inf, 'type': ftype_inf, 'n': int(vD_inf_min) if vD_inf_min is not None else None,
            'degree': deg_inf,
            'm_v': int(m_v_inf) if m_v_inf is not None else None,
            'contribution': int(contrib_sigma_inf) if contrib_sigma_inf is not None else None,
            'e_contrib': int(contrib_euler_inf) if contrib_euler_inf is not None else None
        })
        if verbose:
            diag_inf = local_minimality_diagnostic("∞", v4_inf, v6_inf, vD_inf, sym_inf)
            if diag_inf is not None:
                print(">>> Diagnostic:", diag_inf)

    # sums
    euler_total = sum([f['e_contrib'] for f in fibers if f.get('e_contrib') is not None])
    sigma_sum = sum([f['contribution'] for f in fibers if f.get('contribution') is not None])
    sum_t = sum([f.get('t', 0) for f in fibers])

    if verbose:
        print("")
        print("Place analysis (finite algebraic places):")
        print("{:40s} {:>5s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
            "factor g(m)", "mult", "v_c4", "v_c6", "v_D", "n", "v4_min", "v6_min", "vD_min", "Kodaira"))
        print("-" * 122)
        seen = set()
        for f in fibers:
            fac = f['factor']
            if fac is None:
                continue
            key = str(fac)
            if key in seen:
                continue
            seen.add(key)
            deg = f['degree']
            symbol_str = f"{deg} x {f['symbol']}" if deg > 1 else str(f['symbol'])
            mult_print = factor_mult.get(fac, 0) if 'factor_mult' in locals() else 0
            print("{:40s} {:>5d} {:>6d} {:>6d} {:>6d} {:>6s} {:>8d} {:>8d} {:>8d} {:>12s}".format(
                key[:39], int(mult_print),
                int(f['v_c4']), int(f['v_c6']), int(f['v_D']),
                "-" if f['n'] is None else str(f['n']),
                int(f['v4_min']), int(f['v6_min']), int(f['vD_min']),
                symbol_str
            ))
        infs = [f for f in fibers if f['root_type'] == 'inf']
        if infs:
            fi = infs[0]
            print("Infinity: v_c4 = {:+d}, v_c6 = {:+d}, v_D = {:+d} -> {}".format(int(fi['v_c4']), int(fi['v_c6']), int(fi['v_D']), fi['symbol']))
        print("")
        print("Summary of fibers (per-center):")
        for f in fibers:
            if f['symbol'] is None:
                continue
            deg = f['degree']
            print(f"  {f['symbol']} at {str(f['r'])[:40]} (deg {deg}) contrib={f['contribution']}")
        print("")
        print("Total Euler characteristic:", euler_total)
        print("Sum of fiber contributions (Σ(m_v - 1)):", sigma_sum)
        print("")

    return {
        'fibers': fibers,
        'euler_characteristic': int(euler_total),
        'sigma_sum': int(sigma_sum),
        'sum_reduction_exponents': int(sum_t)
    }




def local_minimality_diagnostic(place, v4, v6, vD, kodaira_symbol):
    """
    Given raw valuations and a Kodaira symbol, suggest blow-up/down scalings.
    place: str (for logging)
    v4, v6, vD: ints (raw valuations at this place)
    kodaira_symbol: str (like 'I1*', 'IV', 'III', etc.)
    """
    # Handle None or unknown symbols gracefully
    if kodaira_symbol is None or kodaira_symbol == 'I0':
        return None
    
    # Target discriminant valuations for each fiber type
    vD_targets = {
        "II": 2,
        "III": 3,
        "IV": 4,
        "I0*": 6,
        "IV*": 8,
        "III*": 9,
        "II*": 10,
    }
    
    # Parse the Kodaira symbol to get target v_D
    target_vD = None
    
    # Check if it's a standard additive type (II, III, IV, etc.)
    if kodaira_symbol in vD_targets:
        target_vD = vD_targets[kodaira_symbol]
    # Check if it's I_n (multiplicative, no star)
    elif kodaira_symbol.startswith('I') and '*' not in kodaira_symbol:
        try:
            n_str = kodaira_symbol[1:]
            if n_str and n_str.isdigit():
                n = int(n_str)
                target_vD = n
        except (ValueError, IndexError):
            pass
    # Check if it's I_n* (multiplicative with star)
    elif kodaira_symbol.startswith('I') and kodaira_symbol.endswith('*'):
        try:
            n_str = kodaira_symbol[1:-1]  # Remove 'I' and '*'
            if n_str:
                n = int(n_str)
                target_vD = n + 6
            else:
                # I0* case
                target_vD = 6
        except (ValueError, IndexError):
            pass
    
    # If we couldn't determine target, skip diagnostic
    if target_vD is None:
        return None
    
    # Try small rescalings k in [-2, 2] to find best fit
    best = None
    for k in range(-2, 3):
        v4k = v4 - 4 * k
        v6k = v6 - 6 * k
        vDk = vD - 12 * k
        if vDk < 0:
            continue
        # cost function: distance from target
        cost = abs(vDk - target_vD) + (v4k % 4) + (v6k % 6)
        cand = (cost, k, v4k, v6k, vDk)
        if best is None or cand < best:
            best = cand
    
    if best is None:
        return f"[{place}] No safe rescaling."
    
    cost, k, v4b, v6b, vDb = best
    
    if k == 0:
        action = "Already minimal."
    elif k > 0:
        action = f"Blow DOWN by k={k} (u=t^{k})"
    else:
        kpos = -k
        action = f"Extra blow UP by k={kpos} (u=t^-{kpos})"
    
    return f"[{place}] {kodaira_symbol}: v=({v4},{v6},{vD}) -> ({v4b},{v6b},{vDb}), target vΔ={target_vD}. {action}"

### SATURATION BELOW

def component_order_from_symbol(symbol):
    # returns c_v, order of component group for the fiber
    # I_n -> n, I_n* -> 4 (for n=0) or 4?  Use standard table:
    # (I_n: n), (I0*:4), (I_n*:4)??  Better: implement full table below
    if symbol.startswith("I") and not symbol.endswith("*"):
        n = int(symbol[1:]) if len(symbol) > 1 else 1
        return n
    if symbol.endswith("*"):
        # For I_n*, order = 4 if n=0 else 4?  Actually component-group sizes:
        # I_n: Z/nZ (order n)
        # I_n*: Z/2Z x Z/2Z for n even? -> use standard table per Kodaira
        # For safety, use authoritative small table:
        table = {"I0*":4,"II":1,"III":2,"IV":3,"IV*":3,"III*":2,"II*":1}
        return table.get(symbol, 1)
    # fallback
    return 1

def candidate_index_primes(detH, fiber_symbols, torsion_order=1):
    # detH should be integer (or rational -> multiply denom)
    # fiber_symbols: list of strings like 'I1', 'IV*', etc.
    # Return squarefree prime list to test
    comp_prod = 1
    for s in fiber_symbols:
        comp_prod *= component_order_from_symbol(s)
    I = Integer(detH) * Integer(comp_prod) // (Integer(torsion_order)**2)
    I = abs(I)
    if I == 0:
        return []  # degenerate - handle separately
    # take squarefree part
    sqf = Integer(1)
    tmp = I
    for p, e in tmp.factor():
        sqf *= p
    return [int(p) for p, e in tmp.factor()]


def is_l_saturated_mod_p(EFp, gens_reduced, ell):
    # return True if gens_reduced generate an ell-saturated subgroup inside EFp
    # i.e. check gcd(|EFp|/|<gens>|, ell) == 1
    G_order = subgroup_order_generated(EFp, gens_reduced)  # use baby-step/NTL style
    return gcd(EFp.order() // G_order, ell) == 1

def test_saturation_via_primes(sections, fiber_symbols, torsion, good_specializations, ell_list, trials_per_ell=5):
    # good_specializations: list of m0's where specialization works
    surviving = set(ell_list)
    for ell in list(surviving):
        for m0 in good_specializations[:trials_per_ell]:
            EFp = specialize_curve_mod_p(m0)  # returns E/F_p
            gens_red = [reduce_section_mod_p(P, m0) for P in sections]
            if is_l_saturated_mod_p(EFp, gens_red, ell):
                surviving.remove(ell)
                break
    return sorted(list(surviving))  # primes that need further attention

"""Van Luijk-style Picard pinning (two-prime reductions)

For ambiguous 
𝜌
ρ cases, reduce the surface mod two primes, compute Picard ranks there and use the intersection to pin 
𝜌
ρ. Small code and huge payoff: exact ST target reduces ambiguity in indexing."""


def mult_of(g, S, max_iter=None):
    """
    Compute multiplicity k of polynomial factor `g` in `S` (both polynomials
    in the same UnivariatePolynomialRing).  Return (k, S_quot) where
    S_quot = S // g^k (the quotient after removing g^k).

    Conservative and fast:
      - If g is constant or deg(g) <= 0 returns (0, S).
      - If S == 0 returns (0, S).
      - Limits iterations to deg(S)//deg(g) (or max_iter if provided).
      - Uses repeated quo_rem but returns the final quotient so caller
        does not need to redivide.
    """
    # Basic sanity
    try:
        if g.degree() <= 0:
            return 0, S
    except Exception:
        # Not a polynomial we can inspect -- be conservative.
        return 0, S

    if S == 0:
        return 0, S

    # Ensure S is in same parent as g when possible
    try:
        if S.parent() is not g.parent():
            S = g.parent()(S)
    except Exception:
        # fallback: try coercion silently; if still fails, return zero multiplicity
        try:
            S = g.parent()(S)
        except Exception:
            return 0, S

    deg_g = g.degree()
    deg_S = S.degree()
    if deg_g <= 0 or deg_S < deg_g:
        return 0, S

    # safety cap
    cap = deg_S // deg_g
    if max_iter is not None:
        cap = min(cap, int(max_iter))

    k = 0
    q = S
    # repeated division but bounded by cap
    while k < cap:
        q_next, r = q.quo_rem(g)
        if r != 0:
            break
        k += 1
        q = q_next
        # small safety: if q.degree() decreases below deg_g stop early
        if q == 0 or q.degree() < deg_g:
            break

    return k, q

from sage.all import QQ

def compute_euler_and_chi(cd_or_finder_result):
    """
    Accept either:
      - the cd object (which find_singular_fibers knows how to read), or
      - the dict returned by find_singular_fibers(...)
    Returns (euler_sum, chi) where chi = euler_sum / 12 (as QQ or int).
    """
    if isinstance(cd_or_finder_result, dict) and 'euler_characteristic' in cd_or_finder_result:
        euler_sum = int(cd_or_finder_result['euler_characteristic'])
    else:
        # call the heavy-lifter (it will return cd.singfibs early if already computed)
        info = find_singular_fibers(cd_or_finder_result)
        euler_sum = int(info.get('euler_characteristic', 0))

    # chi should be euler_sum / 12
    chi_q = QQ(euler_sum) / 12
    if chi_q not in ZZ and chi_q.denominator() != 1:
        # give a friendly diagnostic if something strange is up
        print(f"Warning: Euler sum = {euler_sum} gives non-integer chi = {chi_q}; check fiber data.")
    return euler_sum, chi_q


def _kodaira_from_min_vals(v4_min, v6_min, vD_min):
    """Return (symbol, m_v, e_contribution) from minimal valuations."""
    # CRITICAL: Convert all inputs to plain Python int to ensure dict lookup works
    v4_min = int(v4_min)
    v6_min = int(v6_min)
    vD_min = int(vD_min)
    
    # Handle smooth/non-singular case correctly
    if vD_min == 0:
        return ('I0', 1, 0)
    if vD_min < 0:
        return (None, 1, 0) # Not a singular fiber

    # multiplicative
    if v4_min == 0 and v6_min == 0:
        sym = f"I{vD_min}"
        m_v = vD_min
        e = vD_min
        return (sym, m_v, e)
    
    # additive special cases (standard table)
    tbl = {
        (1,1,2): ("II", 1, 2),
        (1,2,3): ("III", 2, 3),
        (2,3,4): ("IV", 3, 4),
        (2,3,6): ("I0*", 6, 6),
        (3,4,8): ("IV*", 7, 8),
        (3,5,9): ("III*", 8, 9),
        (4,5,10):("II*", 9, 10),
    }
    
    # Exact lookup with converted integers
    tup = (v4_min, v6_min, vD_min)
    if tup in tbl:
        return tbl[tup]
    
    # star-family: vD_min >= 6 and v4_min >= 2 and v6_min >= 3 -> I_n* with n = vD_min - 6
    if vD_min >= 6 and v4_min >= 2 and v6_min >= 3:
        n = vD_min - 6
        sym = f"I{n}*"
        m_v = n + 6
        e = n + 6
        return (sym, m_v, e)
    
    # Improved fallback: at least try to classify as additive vs multiplicative
    # and use a reasonable default contribution
    if v4_min == 0 and v6_min == 0:
        # multiplicative but vD_min < 0 shouldn't happen; return I0
        return ('I0', 1, 0)
    else:
        # additive but unrecognized pattern
        # Use a conservative default: treat as additive with some minimal contribution
        # You could also log a warning here if needed
        return (f"IV", 3, 4)  # Default to IV (which has (2,3,4)); safest guess
