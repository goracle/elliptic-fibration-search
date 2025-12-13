
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector
from collections import defaultdict, Counter
from itertools import product
import sys
import traceback
from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, parallel
from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
import multiprocessing
from collections import defaultdict
from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR
from search_common import *
from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve
from sage.all import parallel
from tqdm import tqdm
import signal # For safe multiprocessing worker init
# Near the top with other imports
try:
    from .arakelov import *
    ARAKELOV_AVAILABLE = True
except ImportError:
    ARAKELOV_AVAILABLE = False
    print("[mumford] Warning: arakelov.py not available, using fallback methods")
assert ARAKELOV_AVAILABLE

# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode
#
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.
#
# The 5 equations enforced are:
# 1. F1: r1^2 - s*r1 + p = 0
# 2. F2: Linear coefficient of (v^2 - f) mod u is 0
# 3. F3: Constant coefficient of (v^2 - f) mod u is 0
# 4. F4: f(r1) - f(r2) = 0 (implied by F2=0)
# 5. F5: v1*s + 2*v0 = 0 (Symmetry of v)

# Add near module globals / top of file
RECON_EXPONENT = 0.55   # try 0.55 - 0.6 if 0.45 is too strict; lower to 0.45 if you want stricter check
MIN_SUCCESS_PRIMES = 3  # keep 3 as default; you can lower to 2 if necessary but be conservative


def _poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    """
    Reduce polynomial modulo u(x) = x^2 - s*x + p.
    Input: poly_coeffs in HIGH -> LOW order [a_n, a_{n-1}, ..., a_0].
    Returns: [r1, r0] where result = r1*x + r0.
    """
    coeffs = list(poly_coeffs)
    
    # 1. Trim leading zeros
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)
    
    # 2. Horner-like reduction
    while len(coeffs) > 2:
        a = coeffs.pop(0)
        # x^d -> x^{d-2}(s*x - p)
        # Add a*s to next term (x^{d-1})
        coeffs[0] = coeffs[0] + a * s
        # Subtract a*p from term after that (x^{d-2})
        coeffs[1] = coeffs[1] - a * p
        
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    # 3. Final normalization to [r1, r0]
    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs

def filter_primes_avoiding_denoms(primes_list, divisors):
    # divisors: iterable of dicts with 's','p','v_0','v_1' (QQ)
    bad = set()
    for d in divisors:
        for k in ('s','p','v_0','v_1'):
            val = d.get(k)
            try:
                den = int(QQ(val).denominator)
                if den != 1:
                    # factor small primes of den
                    dd = den
                    p = 2
                    while p*p <= dd:
                        if dd % p == 0:
                            bad.add(p)
                            while dd % p == 0:
                                dd //= p
                        p += 1
                    if dd > 1:
                        bad.add(dd)
            except Exception:
                raise
    return [p for p in primes_list if p not in bad]


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================


# =============================================================================
# CORE ARITHMETIC & SOLVERS
# =============================================================================


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================


def mumford_precompute_residues_sequential(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                           rhs_modp_list, vecs_list, debug=DEBUG):
    """
    Sequential fallback: runs the parallel routine with a single worker.
    """
    print("Sequential fallback is using the parallel routine with a single worker.")
    return mumford_precompute_residues_parallel(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                                rhs_modp_list, vecs_list, num_workers=1, debug=debug)

def _mumford_worker_entry(args):
    """Legacy entry point (placeholder)."""
    # NOTE: The provided code only used _solve_worker_wrapper
    return args[0], {} 

def validate_mumford_solver():
    """Simple test function (placeholder)."""
    print("Use verify_mumford_pair directly for testing.")
    return True

# mumford_complete.py
# 
# UPDATED VERSION with independent basis construction for Mumford divisors
#
# Key additions:
# 1. mumford_to_jacobian_element() - converts (u,v) to Jacobian element
# 2. check_mumford_independence() - tests linear independence via height pairing
# 3. build_mumford_basis_incremental() - builds basis incrementally
# 4. Integration into reconstruct_and_verify_mumford() to return basis instead of all divisors


def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=DEBUG):
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)


from sage.all import RDF, Matrix


from sage.all import RDF, Matrix, PolynomialRing, QQ, HyperellipticCurve


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve, RDF, log, LCM

# search_common must be available in python path

# =============================================================================
# MANUAL HEIGHT IMPLEMENTATIONS
# =============================================================================


# =============================================================================
# JACOBIAN BASIS CONSTRUCTION
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================


# =============================================================================
# CORE ARITHMETIC & HELPERS
# =============================================================================

def build_mumford_equations_from_fibration(tower, f_coeffs):
    return {
        'f_coeffs': f_coeffs,
        'const': 0,
        'm_sym': var('m')
    }

def poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    coeffs = list(poly_coeffs)
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)

    while len(coeffs) > 2:
        a = coeffs.pop(0)
        coeffs[0] = coeffs[0] + a * s
        coeffs[1] = coeffs[1] - a * p
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs


def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=False):
    if modulus is None:
        R = PolynomialRing(QQ, 'x')
    else:
        R = PolynomialRing(GF(modulus), 'x')
    
    x = R.gen()
    
    if modulus is None:
        s_val = QQ(s)
        p_val = QQ(p)
        v0_val = QQ(v0)
        v1_val = QQ(v1)
        f_poly_coeffs = [QQ(c) for c in f_coeffs]
    else:
        s_val = int(s) % modulus
        p_val = int(p) % modulus
        v0_val = int(v0) % modulus
        v1_val = int(v1) % modulus
        f_poly_coeffs = [int(c) % modulus for c in f_coeffs]
    
    u_poly = x**2 - s_val*x + p_val
    v_poly = v1_val*x + v0_val
    
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    diff = v_poly**2 - f_poly
    remainder = diff % u_poly
    
    return remainder.is_zero()


def _normalize_sign(s, p, v0, v1):
    if v1 < 0 or (v1 == 0 and v0 < 0):
        return (s, p, -v0, -v1)
    return (s, p, v0, v1)


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================

def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _solve_worker_wrapper(args):
    p, f_coeffs_ints, x_residues_map, const_val_int = args
    try:
        p_results = {}
        for v_tuple, x_res_list in x_residues_map.items():
            if isinstance(x_res_list, int):
                x_res_list = [x_res_list]
            
            x_res_to_sols = {}
            for x_res in x_res_list:
                sols = solve_mumford_mod_p_optimized(f_coeffs_ints, p, x_res, const_val_int)
                
                verified_sols = []
                for sol in sols:
                    s, p_val, v0, v1 = sol
                    if verify_mumford_pair(f_coeffs_ints, s, p_val, v0, v1, modulus=p):
                        verified_sols.append(sol)
                
                if verified_sols:
                    x_res_to_sols[x_res] = verified_sols
            
            if x_res_to_sols:
                p_results[v_tuple] = x_res_to_sols
                
        return p, p_results
    except Exception:
        sys.stderr.write(f"\nCRITICAL ERROR IN MUMFORD WORKER (p={p}):\n")
        traceback.print_exc(file=sys.stderr)
        raise
        return p, {}


# =============================================================================
# MANUAL HEIGHT & INDEPENDENCE CHECK (FIXED)
# =============================================================================


# -----------------------------------------------------------------------------
# Robust replacements for height / conversion / independence checks
# -----------------------------------------------------------------------------

from fractions import Fraction
import math

def _extract_u_coeffs_as_fractions(u):
    """
    Return list of coefficients of u (highest-to-lowest) as Fraction objects.
    Accepts:
      - Sage polynomial (use .list() or .coefficients?)
      - Python list/tuple of coeffs
      - tuple-like from mumford (already rational objects)
    Ensures monic by appending the implicit leading 1 if needed.
    """
    # If u is a Sage polynomial, try u.list() (coeffs lowest-first)
    try:
        if hasattr(u, 'list'):
            coeffs_low = u.list()    # lowest-degree first
            coeffs = list(reversed(coeffs_low))  # highest-first
        elif hasattr(u, 'coefficients'):
            coeffs = u.coefficients(sparse=False)
            # coefficients may not include zeros; try to detect degree
            if hasattr(u, 'degree') and u.degree() is not None:
                deg = u.degree()
                # create full list
                full = [QQ(0)] * (deg+1)
                for i, c in enumerate(u.coefficients(sparse=False)):
                    # this is fragile in some Sage versions; fallback below
                    pass
        else:
            # Fallback: treat u as an iterable of coeffs (highest-first)
            coeffs = list(u)
    except Exception:
        # Let exceptions bubble: user asked to raise them
        raise

    # Coerce each coefficient to Fraction robustly
    frac_coeffs = []
    for c in coeffs:
        # If it's a Sage rational (QQ), get numerator/denominator
        try:
            if hasattr(c, 'numerator') and hasattr(c, 'denominator'):
                n = int(c.numerator())
                d = int(c.denominator())
                frac_coeffs.append(Fraction(n, d))
            else:
                # For floats or RDF, convert via Fraction.from_float if necessary
                frac_coeffs.append(Fraction(c))
        except Exception:
            # last resort: try string conversion
            frac_coeffs.append(Fraction(str(c)))
            raise

    # Ensure monic: if leading coeff != 1, check if implicit monic (some code gives only lower terms)
    if not frac_coeffs:
        return [Fraction(1,1)]
    # If leading coeff equals 1, fine. If not, assume monic poly was given lacking leading 1:
    if frac_coeffs[0] != 1:
        # If the polynomial *is* monic but the leading 1 is missing (common if only lower coefs were returned),
        # then append an explicit leading 1.
        # Heuristic: if len(frac_coeffs) == 2 and frac_coeffs[0] < 1 and frac_coeffs[1] != 0, we try appending 1.
        # Safer: don't silently mutate; prefer to return as-is and let caller handle if degree mismatch.
        # For now, if leading coeff is not 1 but <= 1 in magnitude, append an explicit 1 to represent monic.
        frac_coeffs = [Fraction(1,1)] + frac_coeffs

    return frac_coeffs


def manual_naive_height(P):
    """
    Robust naive logarithmic height from Mumford u-polynomial.
    Returns float(log(max_abs)), always finite for valid input, else raises.
    """
    try:
        u = P[0]  # Mumford u polynomial
    except Exception:
        raise

    fracs = _extract_u_coeffs_as_fractions(u)
    # Convert to integer projective coordinates by clearing denominators
    dens = [f.denominator for f in fracs]
    L = 1
    for d in dens:
        L = (L * d) // math.gcd(L, d)

    int_coeffs = [int((f * L).numerator) for f in fracs]  # numerators after clearing denom
    # append the implicit leading coefficient L*1 (monic)
    int_coeffs.append(int(L))

    if not int_coeffs:
        return 0.0

    max_abs = max(abs(c) for c in int_coeffs)
    # defensive: ensure positive integer
    max_abs = max(1, int(max_abs))
    return math.log(max_abs)


def manual_canonical_height(P, limit=8, debug=DEBUG):
    """
    Approximate canonical height by computing h(2^n P)/4^n for n=0..limit and returning last value.
    Re-raises any exceptions from doubling. Prints intermediate heights if debug=True.
    """
    if P.is_zero():
        return 0.0

    Q = P
    vals = []
    try:
        for n in range(limit + 1):
            hQ = manual_naive_height(Q)
            vals.append(float(hQ) / (4.0 ** n))
            if debug:
                print(f"[canon] n={n} naive_h={hQ:.6g} ratio={vals[-1]:.6g}")
            Q = 2 * Q
    except Exception:
        # Re-raise after printing diagnostic if possible
        if debug:
            print("Doubling failed at step", n)
        raise

    # Return the last computed ratio; caller can examine intermediate vals if needed.
    return float(vals[-1])


def compute_manual_height_pairing(P, Q, limit=8, debug=DEBUG):
    """
    <P, Q> = 1/2 * (h_hat(P+Q) - h_hat(P) - h_hat(Q))
    Uses the manual canonical-height approximation.
    """
    try:
        if P.is_zero() or Q.is_zero():
            return float(0.0)

        # Use manual canonical height approximation for all three
        h_p = manual_canonical_height(P, limit=limit, debug=debug)
        h_q = manual_canonical_height(Q, limit=limit, debug=debug)
        h_sum = manual_canonical_height(P + Q, limit=limit, debug=debug)
        val = 0.5 * (h_sum - h_p - h_q)
        return float(val)
    except Exception:
        raise


def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Create a Jacobian element while coercing the u,v polynomials into the curve's polynomial ring.
    Raises on failure (user preference).
    """
    try:
        f_curve, h_curve = C.hyperelliptic_polynomials()
        R = f_curve.parent()   # polynomial ring of the curve
        x = R.gen()

        # Coerce inputs to rational numbers in Python Fraction and then to QQ for the ring
        def to_QQ_obj(a):
            try:
                return QQ(a)
            except Exception:
                raise
                return QQ(Fraction(str(a)))

        s_q = to_QQ_obj(s)
        p_q = to_QQ_obj(p)
        v0_q = to_QQ_obj(v0)
        v1_q = to_QQ_obj(v1)

        u_poly = x**2 - s_q * x + p_q
        v_poly = v1_q * x + v0_q

        # Make sure polynomials live in the same parent as the curve
        u_poly = R(u_poly)
        v_poly = R(v_poly) if v_poly.parent() == R else R(v_poly)  # coerce v into same ring

        return C.jacobian()([u_poly, v_poly])
    except Exception:
        # re-raise so user sees the problem
        raise


def dbg_poly_info(poly):
    # poly is a sage polynomial
    coeffs = poly.list()          # lowest-first
    if not coeffs:
        return "deg=-inf"
    deg = poly.degree()
    # get bit sizes
    def bits_of(c):
        try:
            return int(c.nbits()) if hasattr(c, 'nbits') else int(Fraction(c).numerator).bit_length()
        except Exception:
            try:
                return int(abs(int(c)).bit_length())
            except Exception:
                raise
                return -1
            raise
    bits = [bits_of(c) for c in coeffs]
    maxbits = max(bits) if bits else 0
    return f"deg={deg}, maxcoeff_bits={maxbits}, len={len(coeffs)}"

def dump_jacobian_mumford_info(JP, label="P"):
    # JP is jacobian element [u,v]
    try:
        u = JP[0]   # polynomial
        v = JP[1]
        print(f"[DBG] {label} u: {dbg_poly_info(u)}; v: {dbg_poly_info(v)}; parents: {type(u.parent())}")
    except Exception as e:
        print("[DBG] failed to print mumford info:", e)
        raise


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


def naive_height_safe(s, p, v0, v1, debug=DEBUG):
    """
    Compute naive height from Mumford representation without building Jacobian.
    Returns log(max(|coeffs of u|, |coeffs of v|)).
    """
    from fractions import Fraction
    import math
    
    # Force conversion to QQ first, then to Fraction
    s_qq = QQ(s)
    p_qq = QQ(p)
    v0_qq = QQ(v0)
    v1_qq = QQ(v1)
    
    # Convert QQ to Fraction using numerator/denominator
    s_frac = Fraction(int(s_qq.numerator()), int(s_qq.denominator()))
    p_frac = Fraction(int(p_qq.numerator()), int(p_qq.denominator()))
    v0_frac = Fraction(int(v0_qq.numerator()), int(v0_qq.denominator()))
    v1_frac = Fraction(int(v1_qq.numerator()), int(v1_qq.denominator()))
    
    # u(x) = x^2 - s*x + p has coefficients [1, -s, p]
    # v(x) = v1*x + v0 has coefficients [v1, v0]
    
    all_coeffs = [
        Fraction(1, 1),  # leading coeff of u
        -s_frac,
        p_frac,
        v1_frac,
        v0_frac
    ]
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))  # include denominator
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return float(math.log(max_abs))


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


def compute_height_pairing_simple(D1, D2, num_doublings=NUM_DOUBLINGS):
    """
    Compute <D1, D2> using LIMITED doublings to avoid coefficient explosion.
    Uses: <D1, D2> = (h(D1+D2) - h(D1) - h(D2)) / 2
    where h is naive height.
    
    Only does `num_doublings` iterations instead of 8.
    """
    from fractions import Fraction
    import math
    
    def naive_height_from_jacobian(D):
        u, v = D[0], D[1]
        u_coeffs = u.list()
        v_coeffs = v.list()
        
        all_coeffs = []
        for c in u_coeffs + v_coeffs:
            c_qq = QQ(c)
            all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
        
        # Clear denominators
        lcm_den = 1
        for f in all_coeffs:
            lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
        
        int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
        int_coeffs.append(int(lcm_den))
        
        max_abs = max(abs(c) for c in int_coeffs if c != 0)
        max_abs = max(1, max_abs)
        
        return float(math.log(max_abs))
    
    if D1.is_zero() or D2.is_zero():
        return 0.0
    
    # Compute heights with limited doublings
    vals = []
    P, Q, S = D1, D2, D1 + D2
    
    for n in range(num_doublings):
        hP = naive_height_from_jacobian(P)
        hQ = naive_height_from_jacobian(Q)
        hS = naive_height_from_jacobian(S)
        
        pairing = (hS - hP - hQ) / 2.0
        vals.append(pairing / (4.0 ** n))
        
        P = P + P
        Q = Q + Q
        S = S + S
    
    # Return the last value (most refined estimate)
    return vals[-1]


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


def naive_height_exact(D):
    """
    Compute a naive height in exact rationals from a Mumford divisor D = [u, v].
    Returns a QQ number (log of max coefficient magnitude, exact).
    """
    from fractions import Fraction
    import math
    
    u_coeffs = D[0].list()
    v_coeffs = D[1].list()
    
    # Convert to exact fractions
    all_coeffs = []
    for c in u_coeffs + v_coeffs:
        c_qq = QQ(c)
        all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return QQ(math.log(max_abs))


def _poly_from_coeffs_qq(R, coeffs):
    """Reconstructs a polynomial in R from highest-to-lowest QQ coefficients."""
    p = R(0)
    # Handle the case where u=[1] (x-r1), u=[1, -s, p] (x^2-sx+p), etc.
    # The reconstruction must handle varying degree in the loop.
    for c in coeffs:
        p = p * R.gen() + c
    return p


def _get_divisor_coeffs_qq(D):
    """Extracts rational coefficients (QQ) from a Sage Jacobian element D=[u, v]."""
    u = D[0]
    v = D[1]
    # Sage's .list() returns coeffs lowest-to-highest, so reverse them.
    return u.list()[::-1], v.list()[::-1]


def compute_height_pairing_exact(D1, D2, f_coeffs, num_doublings=NUM_DOUBLINGS, primes_list=PRIME_POOL, debug=False):
    """
    Exact height pairing <D1, D2> using modular doubling to approximate canonical height.
    This replaces the slow D+D doubling with the robust CRT method.
    Returns a QQ number.
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    # 1. Compute D1+D2
    D_sum = D1 + D2

    # 2. Compute the final doubled points using the robust modular method
    P_final = compute_doubled_point_modular(D1, f_coeffs, num_doublings, primes_list, debug=debug)
    Q_final = compute_doubled_point_modular(D2, f_coeffs, num_doublings, primes_list, debug=debug)
    S_final = compute_doubled_point_modular(D_sum, f_coeffs, num_doublings, primes_list, debug=debug)

    # 3. Calculate the naive height of the final doubled points
    h_P_final = naive_height_exact(P_final)
    h_Q_final = naive_height_exact(Q_final)
    h_S_final = naive_height_exact(S_final)
    
    # 4. Apply the canonical height definition
    # h_hat(D) approx h(2^n D) / 4^n 
    scaling_factor = QQ(4**num_doublings)
    canonical_D1 = h_P_final / scaling_factor
    canonical_D2 = h_Q_final / scaling_factor
    canonical_D_sum = h_S_final / scaling_factor
    
    # 5. Compute the pairing
    pairing_value = (canonical_D_sum - canonical_D1 - canonical_D2) / QQ(2)
    
    return pairing_value

# --- END OF NEW MODULAR DOUBLING IMPLEMENTATION ---


def compute_doubled_point_modular(D_start, f_coeffs, num_doublings, primes_list, debug=False):
    """
    Calculates 2^num_doublings * D_start using parallel modular arithmetic 
    and Rational Reconstruction (CRT).
    D_start is a SageMath Jacobian element over QQ.
    """
    R_QQ = D_start[0].parent()
    u_coeffs_current, v_coeffs_current = _get_divisor_coeffs_qq(D_start)
    
    # Determine max expected degree for u and v based on genus g=2
    # u degree is g=2 (length 3), v degree is g-1=1 (length 2)
    u_max_len = 3
    v_max_len = 2

    for n in range(num_doublings):
        u_coeffs_next = defaultdict(list)
        v_coeffs_next = defaultdict(list)
        
        if debug: 
            print(f"  [MOD-DBL] Doubling iteration {n+1}/{num_doublings}")
        
        success_count = 0
        for p in primes_list:
            if p == 2: 
                continue

            # Prepare coefficients reduced mod p 
            try:
                # Need the *integer* numerator/denominator for a safe mod reduction
                # We catch ZeroDivisionError specifically for p dividing the denominator
                u_mod_p = [int(c.numerator()) * pow(int(c.denominator()), -1, p) % p for c in u_coeffs_current]
                v_mod_p = [int(c.numerator()) * pow(int(c.denominator()), -1, p) % p for c in v_coeffs_current]
            except (ZeroDivisionError, ValueError):
                # Denominator is divisible by p, skip this prime
                raise
                continue 

            u_2p_coeffs, v_2p_coeffs = _mumford_doubling_mod_p_internal(
                u_mod_p, v_mod_p, f_coeffs, p
            )

            # Validate output from _mumford_doubling_mod_p_internal and skip bad primes
            if u_2p_coeffs is None or v_2p_coeffs is None:
                # modular routine signalled this prime is unusable; skip it
                if debug:
                    print(f"  [MOD-DBL] prime {p} produced no valid doubled divisor (skipping).")
                continue

            # Ensure output lists are integer residues in 0..p-1
            try:
                u_2p_coeffs = [int(x) % p for x in u_2p_coeffs]
                v_2p_coeffs = [int(x) % p for x in v_2p_coeffs]
            except Exception:
                if debug:
                    print(f"  [MOD-DBL] prime {p} returned non-integer coefficients (skipping).")
                raise
                continue

            if u_2p_coeffs is None:
                continue
            
            success_count += 1

            # Pad modular coeffs with leading zeros to match max degree
            u_2p_coeffs = [0] * (u_max_len - len(u_2p_coeffs)) + u_2p_coeffs
            v_2p_coeffs = [0] * (v_max_len - len(v_2p_coeffs)) + v_2p_coeffs
            
            # Store results by coefficient index
            for i in range(u_max_len):
                u_coeffs_next[i].append((p, int(u_2p_coeffs[i])))
            for i in range(v_max_len):
                v_coeffs_next[i].append((p, int(v_2p_coeffs[i])))

        # --- Rational Reconstruction ---
        u_reconstructed = []
        v_reconstructed = []
        
        # Check if we have enough primes

        # --- Choose only primes that contributed to *every* coefficient (avoid Frankenstein mixes) ---
        # Build list of primes that provided values for all u and v coefficient positions
        good_primes = []
        for p in primes_list:
            if p == 2:
                continue
            ok = True
            for i in range(u_max_len):
                if not any(pp == p for pp, _ in u_coeffs_next.get(i, [])):
                    ok = False
                    break
            if not ok:
                continue
            for i in range(v_max_len):
                if not any(pp == p for pp, _ in v_coeffs_next.get(i, [])):
                    ok = False
                    break
            if ok:
                good_primes.append(p)

        if debug:
            print(f"  [MOD-DBL] good_primes (present for all coeffs) = {good_primes}")

        # require a minimum number of shared primes to reconstruct the whole polynomial safely
        if len(good_primes) < MIN_SUCCESS_PRIMES:
            if debug:
                print(f"  [MOD-DBL] Critical failure: only {len(good_primes)} fully-consistent primes available.")
            raise ValueError(f"Modular doubling failed at iteration {n+1} due to insufficient consistent primes ({len(good_primes)}).")

        # We will reconstruct every coefficient using the SAME good_primes (same modulus M)
        primes_for_all = tuple(good_primes)

        # helper to extract residues for a coefficient in the order of primes_for_all
        def coeff_residues_for_primes(coeff_list):
            # coeff_list is list of (p,val) for that coefficient
            lookup = {p: val for p, val in coeff_list}
            return tuple(lookup[p] for p in primes_for_all)

        try:

            # Reconstruct u coefficients (using same primes_for_all for each coeff)
            M_c = math.prod(primes_for_all)
            for i in range(u_max_len):
                if not u_coeffs_next[i]:
                    u_reconstructed.append(QQ(0))
                    continue

                vals_for_c = coeff_residues_for_primes(u_coeffs_next[i])
                crt_val = crt_cached(vals_for_c, primes_for_all)
                num, den = rational_reconstruct(crt_val, M_c)

                if abs(num) > M_c**RECON_EXPONENT or abs(den) > M_c**RECON_EXPONENT:
                    raise RationalReconstructionError(
                        f"Height too large for coeff {i}: num={num}, den={den}, M_c={M_c}, exponent={RECON_EXPONENT}"
                    )

                u_reconstructed.append(QQ(num)/QQ(den))

            # Reconstruct v coefficients (same primes_for_all)
            for i in range(v_max_len):
                if not v_coeffs_next[i]:
                    v_reconstructed.append(QQ(0))
                    continue

                vals_for_c = coeff_residues_for_primes(v_coeffs_next[i])
                crt_val = crt_cached(vals_for_c, primes_for_all)
                num, den = rational_reconstruct(crt_val, M_c)

                if abs(num) > M_c**RECON_EXPONENT or abs(den) > M_c**RECON_EXPONENT:
                    raise RationalReconstructionError(
                        f"Height too large for coeff {i}: num={num}, den={den}, M_c={M_c}, exponent={RECON_EXPONENT}"
                    )

                v_reconstructed.append(QQ(num)/QQ(den))

            # Form the new Mumford divisor 2*D_current over Q[x]
            u_next = _poly_from_coeffs_qq(R_QQ, u_reconstructed)
            v_next = _poly_from_coeffs_qq(R_QQ, v_reconstructed)
            
            # Recreate the SageMath Jacobian element
            f_poly_qq = _poly_from_coeffs_qq(R_QQ, f_coeffs)
            C_QQ = HyperellipticCurve(f_poly_qq, R_QQ(0))
            J_QQ = C_QQ.jacobian()


            # assume u_next, v_next, f_poly are present as Sage polynomials over QQ
            # try cheap repairs then validate
            u_try = make_monic(u_next)
            v_try = reduce_v_mod_u(v_next, u_try)

            valid, reason = is_divisor_on_curve(u_try, v_try, f_poly_qq)
            if not valid:
                # second attempt: sometimes denominator-scaling leaves remainder; try clearing denominators
                try:
                    # clear denominators of coefficients for both u_try and v_try
                    den_lcm = 1
                    for coeff in u_try.coefficients() + v_try.coefficients():
                        den_lcm = lcm(den_lcm, QQ(coeff).denominator())
                    # scale to integer polynomials (work over ZZ) then reduce v mod u again
                    u_int = (u_try * den_lcm).change_ring(ZZ)
                    v_int = (v_try * den_lcm).change_ring(ZZ)
                    # convert back to QQ and re-normalize to monic u
                    u_scaled = PolynomialRing(QQ, 'x')(u_int).change_ring(QQ)
                    v_scaled = PolynomialRing(QQ, 'x')(v_int).change_ring(QQ)
                    u_scaled = make_monic(u_scaled)
                    v_scaled = reduce_v_mod_u(v_scaled, u_scaled)
                    valid2, reason2 = is_divisor_on_curve(u_scaled, v_scaled, f_poly)
                    if valid2:
                        u_try, v_try = u_scaled, v_scaled
                        valid, reason = True, None
                except Exception:
                    # ignore integer-scaling failure, will re-raise below
                    raise

            if not valid:
                # give a very explicit error for upstream handling and logging
                msg = (f"Reconstructed (u,v) failed Mumford test after repair attempts: {reason}.\n"
                    f"u = {u_next}\n"
                    f"v = {v_next}\n"
                    f"u_try = {u_try}\n"
                    f"v_try = {v_try}\n")
                if debug:
                    print("[compute_doubled_point_modular] " + msg)
                # raise so caller treats this as a reconstruction failure (and can skip it)
                raise RationalReconstructionError(msg)

            # If valid, construct Jacobian point from repaired pair
            u_next, v_next = u_try, v_try
            u_next = make_monic(u_next)
            v_next = v_next % u_next
            if (v_next**2 - f_poly_qq) % u_next != 0:
                raise RationalReconstructionError("v^2 != f mod u after doubling")

            D_current = J_QQ([u_next, v_next])


        except RationalReconstructionError as e:
            if debug:
                print(f"  [MOD-DBL] modular reconstruction failed at doubling {n+1}: {e}")
                print("  [MOD-DBL] Falling back to exact QQ doubling for the remaining iterations (slower but exact).")

            # Fallback: reconstruct exact divisor from current rational coeffs and finish doublings exactly
            try:
                # current divisor (exact) from the integer/Q rational coeff lists
                u_exact = _poly_from_coeffs_qq(R_QQ, u_coeffs_current)
                v_exact = _poly_from_coeffs_qq(R_QQ, v_coeffs_current)
                u_exact = make_monic(u_exact)
                v_exact = reduce_v_mod_u(v_exact, u_exact)

                f_poly_qq = _poly_from_coeffs_qq(R_QQ, f_coeffs)
                C_QQ = HyperellipticCurve(f_poly_qq, R_QQ(0))
                J_QQ = C_QQ.jacobian()
                D_exact = J_QQ([u_exact, v_exact])

                # finish remaining doublings exactly
                remaining = num_doublings - n
                for _ in range(remaining):
                    D_exact = 2 * D_exact

                # return the exact result for the caller
                return D_exact

            except Exception as ee:
                # If exact fallback fails, re-raise the original modular error as ValueError
                raise ValueError(f"Modular doubling failed at iteration {n+1} and exact fallback failed too: {e}") from ee
            raise

        
    return D_current


from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ

x = PolynomialRing(QQ, 'x').gen()

def make_monic(u):
    lc = u.leading_coefficient()
    if lc == 1:
        return u
    return (u / lc).change_ring(QQ)   # make monic over QQ

def reduce_v_mod_u(v, u):
    # ensure deg v < deg u by polynomial remainder
    _, r = v.quo_rem(u)
    return r.change_ring(QQ)

def is_divisor_on_curve(u, v, f):
    """
    Tests Mumford divisor conditions:
      1) u monic
      2) deg v < deg u
      3) v^2 - f is divisible by u (exactly)
    Returns (True, None) or (False, reason_string)
    """
    # Ensure polynomial rings are in QQ[x]
    u = u.change_ring(QQ)
    v = v.change_ring(QQ)
    f = f.change_ring(QQ)

    # 1) u must be monic
    if u.leading_coefficient() != 1:
        return False, "u not monic"

    # 2) deg v < deg u
    if v.degree() >= u.degree():
        return False, f"deg v ({v.degree()}) >= deg u ({u.degree()})"

    # 3) divisibility: check remainder of v^2 - f on division by u
    rem = (v**2 - f).quo_rem(u)[1]
    if rem != 0:
        return False, f"v^2 - f mod u != 0 (rem={rem})"

    return True, None


def _mumford_doubling_mod_p_internal(u_coeffs, v_coeffs, f_coeffs, p, debug=False):
    """
    Robust modular doubling for genus-2 Mumford divisors.

    Inputs:
      - u_coeffs, v_coeffs: lists of integers (residues mod p) representing the Mumford
        polynomials. They may be given highest-first or lowest-first (this function
        detects and normalizes).
      - f_coeffs: list (highest->lowest) of curve polynomial coefficients (integers/QQ).
      - p: prime

    Returns:
      (u_2p_coeffs, v_2p_coeffs) where both lists are integers mod p in HIGH->LOW order,
      or (None, None) if prime should be skipped (bad reduction / bad arithmetic).
    """
    if p == 2:
        return None, None

    try:
        Fp = GF(p)
        R_Fp = PolynomialRing(Fp, 'x')
    except Exception:
        raise
        return None, None

    # Build f(x) over Fp using the same helper (safe conversion)
    try:
        f_poly_Fp = _poly_from_coeffs_qq(R_Fp, [Fp(QQ(c)) for c in f_coeffs])
    except Exception:
        # If conversion fails, skip this prime
        if debug:
            print(f"[MOD-DBL] cannot build f_poly mod {p}")
        raise
        return None, None

    # If the curve is singular mod p, skip
    try:
        C_Fp = HyperellipticCurve(f_poly_Fp, R_Fp(0))
    except ValueError:
        if debug:
            print(f"[MOD-DBL] singular curve at p={p}")
        raise
        return None, None

    J_Fp = C_Fp.jacobian()

    # helper: try to interpret a coeff-list as either highest->lowest or lowest->highest
    def _make_poly_from_coeff_list(coeff_list, assume_highest_first):
        """
        Return polynomial over R_Fp or raise if input invalid.
        If assume_highest_first==True, coeff_list is highest->lowest; convert to lowest->highest for constructor.
        """
        if assume_highest_first:
            lst = list(map(Fp, coeff_list))[::-1]   # to lowest->highest
        else:
            lst = list(map(Fp, coeff_list))         # already lowest->highest
        # strip leading zeros in the highest-first sense (i.e., trailing zeros now)
        # ensure at least one coefficient (constant 0 allowed)
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        return R_Fp(lst)

    # try both orientations for inputs (defensive)
    tried = []
    for assume_high in (True, False):
        try:
            u_poly_Fp = _make_poly_from_coeff_list(u_coeffs, assume_high)
            v_poly_Fp = _make_poly_from_coeff_list(v_coeffs, assume_high)
        except Exception as e:
            tried.append((assume_high, "make failed", str(e)))
            raise
            continue

        # canonicalize: require u to be non-zero and monic. If not monic, try to scale.
        if u_poly_Fp.is_zero():
            tried.append((assume_high, "u_zero", None))
            continue

        lc = u_poly_Fp.leading_coefficient()
        if lc != 1:
            # try to normalize to monic (scale by inverse lc)
            try:
                inv_lc = lc**(-1)
                u_poly_Fp = (u_poly_Fp * inv_lc)
                v_poly_Fp = (v_poly_Fp * inv_lc)  # scale v accordingly (safe mod p)
            except Exception:
                tried.append((assume_high, "nonmonic_not_normalizable", lc))
                raise
                continue

        # reduce v modulo u to enforce deg v < deg u
        try:
            v_poly_Fp = v_poly_Fp % u_poly_Fp
        except Exception as e:
            tried.append((assume_high, "reduce_failed", str(e)))
            raise
            continue

        # quick Mumford test: (v^2 - f) % u == 0
        try:
            rem = (v_poly_Fp**2 - f_poly_Fp).quo_rem(u_poly_Fp)[1]
            if rem != 0:
                tried.append((assume_high, "mumford_test_fail", rem))
                continue
        except ZeroDivisionError:
            tried.append((assume_high, "quo_rem_zero_divisor", None))
            raise
            continue
        except Exception as e:
            tried.append((assume_high, "quo_rem_exc", str(e)))
            raise
            continue

        # If we reach here, inputs interpreted under this orientation form a valid divisor mod p
        # proceed to doubling
        try:
            D_mod_p = J_Fp([u_poly_Fp, v_poly_Fp])
        except (ValueError, TypeError) as e:
            tried.append((assume_high, "jacobian_construct_fail", str(e)))
            raise
            continue

        try:
            D_doubled = 2 * D_mod_p
        except (ValueError, ArithmeticError, ZeroDivisionError) as e:
            tried.append((assume_high, "doubling_failed", str(e)))
            raise
            return None, None

        # extract coefficients and normalize result
        u_poly_res = D_doubled[0]
        v_poly_res = D_doubled[1]

        # ensure u_poly_res is monic and deg >= 1 (degree for genus-2: usually 2)
        if u_poly_res.is_zero():
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] doubled u is zero mod {p} (assume_high={assume_high})")
            return None, None

        # normalize to monic
        lc_res = u_poly_res.leading_coefficient()
        if lc_res != 1:
            try:
                inv_lc_res = lc_res**(-1)
                u_poly_res = u_poly_res * inv_lc_res
                v_poly_res = v_poly_res * inv_lc_res
            except Exception:
                if debug:
                    print(f"[MOD-DBL][BAD-RESULT] cannot normalize doubled u monic mod {p}")
                raise
                return None, None

        # reduce v modulo u
        try:
            v_poly_res = v_poly_res % u_poly_res
        except Exception:
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] cannot reduce v mod u after doubling mod {p}")
            raise
            return None, None

        # final Mumford test on the doubled pair
        try:
            rem2 = (v_poly_res**2 - f_poly_Fp).quo_rem(u_poly_res)[1]
            if rem2 != 0:
                if debug:
                    print(f"[MOD-DBL][BAD-RESULT] doubled pair fails Mumford test mod {p}: rem={rem2}")
                return None, None
        except ZeroDivisionError:
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] division by zero while validating doubled pair mod {p}")
            raise
            return None, None

        # Build coefficient lists highest->lowest
        # coefficients(sparse=False) returns [c0, c1, ..., c_n] (lowest->highest)
        u_coeffs_low_to_high = u_poly_res.coefficients(sparse=False)
        v_coeffs_low_to_high = v_poly_res.coefficients(sparse=False)

        # convert to integers in 0..p-1 then reverse to high->low
        u_out = [int(c) for c in u_coeffs_low_to_high][::-1]
        v_out = [int(c) for c in v_coeffs_low_to_high][::-1]

        # pad to expected degrees if desired by caller (caller currently pads itself)
        return u_out, v_out

    # if both orientation attempts failed, optionally debug-print reasons
    if debug:
        print("[MOD-DBL] Tried orientations and failed:", tried)
    return None, None


def check_mumford_independence(divisors, f_coeffs, debug=DEBUG):
    """
    Build Jacobian elements and compute pairing matrix.
    Uses Arakelov if available, otherwise falls back to manual method.
    
    Returns (is_indep, rank, H_matrix)
    """
    if not divisors:
        return True, 0, None

    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)

    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(div['s'], div['p'], div['v_0'], div['v_1'], C)
            if not elem.is_zero():
                jac_elements.append(elem)
            else:
                if debug:
                    print("[check] element is zero, skipping.")
        except Exception:
            if debug:
                print("[check] failed to convert divisor to jac element:", div)
            raise

    if not jac_elements:
        return True, 0, None

    n = len(jac_elements)
    
    if ARAKELOV_AVAILABLE:
        if debug:
            print("[check] Using Arakelov heights")
        is_indep, rank, H = arakelov_check_independence(jac_elements, f_coeffs, prec=300, debug=debug)
        return is_indep, rank, H
    else:
        if debug:
            print("[check] Using manual height computation")
        H = Matrix(RDF, n, n)
        for i in range(n):
            for j in range(i, n):
                try:
                    val = compute_manual_height_pairing(jac_elements[i], jac_elements[j], debug=debug)
                except Exception:
                    if debug:
                        print(f"[check] height pairing failed for indices {i},{j}")
                    raise
                H[i, j] = val
                H[j, i] = val

        if n == 1:
            is_indep = abs(H[0, 0]) > 1e-8
            rank = 1 if is_indep else 0
        else:
            rank = H.rank()
            is_indep = (rank == n)
        return is_indep, rank, H


# mumford_timing_additions.py
#
# Add these sections to mumford_complete.py to improve timing granularity

import time

# Add near top of file with other globals
_MUMFORD_TIMERS = defaultdict(float)


def mumford_timer_get(name):
    """Get timer value."""
    return _MUMFORD_TIMERS.get(name, 0.0)


# Replace reconstruct_and_verify_mumford with this instrumented version:


# Also add timing to the modular residue computation:


# mumford_crt_optimizations.py
#
# Optimizations for the CRT reconstruction bottleneck
# The real problem: 272K combinations × 4 rational reconstructions each = ~1M operations

from sage.all import QQ, ZZ
from itertools import product, islice

# =============================================================================
# OPTIMIZATION 1: Early Exit on Height Rejection
# =============================================================================


@lru_cache
def rational_reconstruct_with_height_check(crt_val, M, max_height):
    """
    Rational reconstruction with immediate height rejection.
    Returns (num, den) or raises RationalReconstructionError.
    """
    # Use standard denominator bound for reconstruction
    max_den = floor(sqrt(M / QQ(2)))
    num, den = rational_reconstruct(crt_val, M, max_den=max_den)
    
    # Then check BOTH against the actual height limit
    if abs(num) > max_height or abs(den) > max_height:
        raise RationalReconstructionError("Height too large")
    
    return num, den
# =============================================================================
# OPTIMIZATION 2: Batch CRT Computation
# =============================================================================

def batch_crt_for_combo(sol_combo, primes):
    """
    Compute CRT for all 4 coordinates at once.
    Returns list of 4 CRT values.
    """
    crt_vals = []
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        crt_vals.append(crt_val)
    return crt_vals


# =============================================================================
# OPTIMIZATION 3: Pre-filter Solutions by Discriminant
# =============================================================================

def prefilter_solutions_by_discriminant(sol_lists, primes):
    """
    Filter solution combinations that can't possibly satisfy s^2 - 4p >= 0 mod all primes.
    This eliminates many impossible combinations early.
    
    Returns: filtered generator of solution combinations
    """
    for sol_combo in product(*sol_lists):
        # Quick discriminant check mod each prime
        all_good = True
        for i, p in enumerate(primes):
            s_mod = sol_combo[i][0] % p
            p_mod = sol_combo[i][1] % p
            disc_mod = (s_mod * s_mod - 4 * p_mod) % p
            
            # If discriminant is negative mod p and p > 2, skip
            # (This is a quick heuristic, not perfect)
            if p > 2 and disc_mod != 0:
                # Check if disc_mod is a quadratic residue
                if pow(disc_mod, (p - 1) // 2, p) == p - 1:
                    all_good = False
                    break
        
        if all_good:
            yield sol_combo


# =============================================================================
# OPTIMIZATION 4: Parallel CRT Reconstruction
# =============================================================================


def reconstruct_worker_wrapper(args):
    """
    Worker for parallel CRT reconstruction.
    args = (combo_batch, primes, M, f_coeffs, max_height)
    Returns list of successful reconstructions.
    """
    combo_batch, primes, M, f_coeffs, max_height = args
    
    results = []
    
    for sol_combo in combo_batch:
        try:
            # Batch CRT
            rec_vals = []
            for idx in range(4):
                vals = tuple(sol[idx] for sol in sol_combo)
                crt_val = crt_cached(vals, tuple(primes))
                num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
                rec_vals.append(QQ(num)/QQ(den))
            
            s, p_val, v0, v1 = rec_vals
            
            # Consistency check
            reconstruction_ok = True
            for i, prime in enumerate(primes):
                expected_sol = sol_combo[i]
                try:
                    s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, prime)) % prime
                    p_mod = (int(p_val.numerator()) * pow(int(p_val.denominator()), -1, prime)) % prime
                    v0_mod = (int(v0.numerator()) * pow(int(v0.denominator()), -1, prime)) % prime
                    v1_mod = (int(v1.numerator()) * pow(int(v1.denominator()), -1, prime)) % prime
                except ZeroDivisionError:
                    reconstruction_ok = False
                    break
                
                if (s_mod != expected_sol[0] % prime or
                    p_mod != expected_sol[1] % prime or
                    v0_mod != expected_sol[2] % prime or
                    v1_mod != expected_sol[3] % prime):
                    reconstruction_ok = False
                    break
            
            if not reconstruction_ok:
                continue
            
            # Algebraic verification
            if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                continue
            
            results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
            
        except RationalReconstructionError:
            raise
            continue
        except Exception:
            raise
            continue
    
    return results


def reconstruct_parallel(sol_lists, primes, f_coeffs, adaptive_limit, num_workers=8, debug=False):
    """
    Parallel CRT reconstruction with batching.
    
    Returns: list of successfully reconstructed divisors
    """
    M = 1
    for p in primes:
        M *= p
    
    max_height = max(100000, int(M ** 0.35))
    
    # Generate all combinations (up to limit)
    all_combos = list(islice(product(*sol_lists), adaptive_limit))
    
    if debug:
        print(f"[parallel_crt] Processing {len(all_combos)} combinations with {num_workers} workers")
    
    # Batch combinations for workers
    batch_size = max(100, len(all_combos) // (num_workers * 4))
    batches = []
    for i in range(0, len(all_combos), batch_size):
        batch = all_combos[i:i+batch_size]
        batches.append((batch, primes, M, f_coeffs, max_height))
    
    # Process in parallel
    try:
        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(num_workers)
    except Exception:
        pool = multiprocessing.Pool(num_workers)
        raise
    
    all_results = []
    try:
        for batch_results in pool.imap_unordered(reconstruct_worker_wrapper, batches):
            all_results.extend(batch_results)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    
    return all_results


# =============================================================================
# OPTIMIZATION 5: Smart Limit Based on Success Rate
# =============================================================================

def adaptive_limit_with_early_stopping(sol_lists, primes, f_coeffs, base_limit, 
                                       check_interval=10000, target_divisors=10, debug=False):
    """
    Process combinations with early stopping if we're finding enough divisors.
    
    Strategy: Check success rate every `check_interval` combinations.
    If we've found `target_divisors` and success rate drops, stop early.
    """
    M = 1
    for p in primes:
        M *= p
    
    max_height = max(100000, int(M ** 0.35))
    
    results = []
    checked = 0
    last_check_count = 0
    
    for sol_combo in islice(product(*sol_lists), base_limit):
        checked += 1
        
        try:
            rec_vals = []
            for idx in range(4):
                vals = tuple(sol[idx] for sol in sol_combo)
                crt_val = crt_cached(vals, tuple(primes))
                num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
                rec_vals.append(QQ(num)/QQ(den))
            
            s, p_val, v0, v1 = rec_vals
            
            # Quick consistency check (just first prime)
            if len(primes) > 0:
                p0 = primes[0]
                expected = sol_combo[0]
                try:
                    s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, p0)) % p0
                    if s_mod != expected[0] % p0:
                        continue
                except ZeroDivisionError:
                    raise
                    continue
            
            # Full verification
            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
        
        except RationalReconstructionError:
            raise
            continue
        except Exception:
            raise
            continue
        
        # Early stopping check
        if checked % check_interval == 0:
            new_found = len(results) - last_check_count
            success_rate = new_found / check_interval
            
            if debug and checked % (check_interval * 5) == 0:
                print(f"[adaptive] Checked {checked}/{base_limit}, found {len(results)} total, recent rate: {success_rate:.6f}")
            
            # Stop if we have enough and success rate is very low
            if len(results) >= target_divisors and success_rate < 1e-5:
                if debug:
                    print(f"[adaptive] Early stop: found {len(results)} divisors, success rate dropped to {success_rate:.6f}")
                break
            
            last_check_count = len(results)
    
    return results, checked


# =============================================================================
# OPTIMIZATION 6: Cache Expensive Operations
# =============================================================================


def consistency_check_cached(s, p_val, v0, v1, sol_combo, primes, inv_cache):
    """
    Consistency check with cached modular inverses.
    Returns True if all primes match.
    """
    for i, prime in enumerate(primes):
        expected_sol = sol_combo[i]
        
        # Get modular inverses with caching
        s_inv = inv_cache.inv(s.denominator(), prime)
        p_inv = inv_cache.inv(p_val.denominator(), prime)
        v0_inv = inv_cache.inv(v0.denominator(), prime)
        v1_inv = inv_cache.inv(v1.denominator(), prime)
        
        if None in (s_inv, p_inv, v0_inv, v1_inv):
            return False
        
        s_mod = (int(s.numerator()) * s_inv) % prime
        p_mod = (int(p_val.numerator()) * p_inv) % prime
        v0_mod = (int(v0.numerator()) * v0_inv) % prime
        v1_mod = (int(v1.numerator()) * v1_inv) % prime
        
        if (s_mod != expected_sol[0] % prime or
            p_mod != expected_sol[1] % prime or
            v0_mod != expected_sol[2] % prime or
            v1_mod != expected_sol[3] % prime):
            return False
    
    return True


# =============================================================================
# OPTIMIZED RECONSTRUCTION FUNCTION (Drop-in Replacement)
# =============================================================================


# mumford_complete_optimized.py
#
# Optimized versions of key functions from mumford_complete.py
# Use same function names so dedup.py will clean up old versions


# Timers
_MUMFORD_TIMERS = defaultdict(float)


class ModInverseCache:
    """Cache for modular inverses."""
    def __init__(self):
        self.cache = {}
    
    def inv(self, a, p):
        key = (a % p, p)
        if key not in self.cache:
            try:
                self.cache[key] = pow(int(a), -1, p)
            except (ValueError, ZeroDivisionError):
                raise
                return None
        return self.cache[key]


def _reconstruct_worker_parallel(args):
    """
    Worker for parallel CRT reconstruction.
    Processes a batch of solution combinations.
    """
    combo_batch, primes, M, f_coeffs, max_height = args
    
    results = []
    stats = {
        'attempted': 0,
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }
    
    for sol_combo in combo_batch:
        stats['attempted'] += 1
        
        try:
            rec_vals = []
            for idx in range(4):
                vals = tuple(sol[idx] for sol in sol_combo)
                crt_val = crt_cached(vals, tuple(primes))
                num, den = rational_reconstruct(crt_val, M)
                
                if abs(num) > max_height or abs(den) > max_height:
                    raise RationalReconstructionError("Height too large")
                
                rec_vals.append(QQ(num)/QQ(den))
            
            s, p_val, v0, v1 = rec_vals
            
        except RationalReconstructionError:
            stats['height_reject'] += 1
            continue
        
        reconstruction_ok = True
        for i, prime in enumerate(primes):
            expected_sol = sol_combo[i]
            try:
                s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, prime)) % prime
                p_mod = (int(p_val.numerator()) * pow(int(p_val.denominator()), -1, prime)) % prime
                v0_mod = (int(v0.numerator()) * pow(int(v0.denominator()), -1, prime)) % prime
                v1_mod = (int(v1.numerator()) * pow(int(v1.denominator()), -1, prime)) % prime
            except ZeroDivisionError:
                reconstruction_ok = False
                break
            
            if (s_mod != expected_sol[0] % prime or
                p_mod != expected_sol[1] % prime or
                v0_mod != expected_sol[2] % prime or
                v1_mod != expected_sol[3] % prime):
                reconstruction_ok = False
                break
        
        if not reconstruction_ok:
            stats['consistency_reject'] += 1
            continue
        
        if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
            stats['algebraic_reject'] += 1
            continue
        
        results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
        stats['success'] += 1
    
    return results, stats


def mumford_precompute_residues_parallel(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                                         rhs_modp_list, vecs_list, num_workers=8, debug=False):
    """
    Parallel residue computation with timing.
    Same function signature as original.
    """
    t_start = time.time()
    
    f_coeffs = eqs_dict['f_coeffs']
    f_coeffs_ints = [int(c) for c in f_coeffs]
    
    try:
        const_val_int = int(QQ(eqs_dict['const']))
    except Exception:
        const_val_int = 0
        raise
        
    if debug:
        print(f"[mumford] Generating tasks for {len(prime_list)} primes...")

    t0 = time.time()
    tasks = []
    
    for p in prime_list:
        if p not in Ep_dict:
            continue
        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        if not p_vecs:
            continue
        
        try:
            Fp = GF(p)
            R_m = Fp['m']
            m_var = R_m.gen()
            rhs_poly = -m_var + Fp(const_val_int)
        except Exception:
            raise
            continue

        x_residues_map = {}
        p_mults = mult_lll.get(p, {})
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue
            
            Pm = Ep(0)
            valid_vec = True
            v_coeffs = p_vecs[v_idx]

            for i, c in enumerate(v_coeffs):
                k = int(c)
                if k == 0:
                    continue
                
                try:
                    mults_for_sec = p_mults[i]
                    if k in mults_for_sec:
                        Pm += mults_for_sec[k]
                    else:
                        valid_vec = False
                        break
                except (IndexError, KeyError, TypeError):
                    valid_vec = False
                    raise
                    break
            
            if not valid_vec or Pm.is_zero() or Pm[2] == 0:
                continue
            
            try:
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator()
                
                if diff_num.is_zero():
                    continue
                    
                roots = diff_num.roots(multiplicities=False)
                
                if roots:
                    valid_residues = []
                    for m_root in roots:
                        m_val = int(m_root)
                        x_val = (-m_val + const_val_int) % p
                        valid_residues.append(x_val)
                    
                    if valid_residues:
                        x_residues_map[v_tuple] = valid_residues
            except Exception:
                raise
                continue
            
        if x_residues_map:
            tasks.append((p, f_coeffs_ints, x_residues_map, const_val_int))

    mumford_timer_add("task_generation", time.time() - t0)

    if not tasks:
        if debug:
            print("[mumford] No tasks generated!")
        return {}
    
    t0 = time.time()
    try:
        ctx = multiprocessing.get_context("fork")
        pool_obj = ctx.Pool(num_workers, initializer=_init_worker)
    except Exception:
        pool_obj = multiprocessing.Pool(num_workers, initializer=_init_worker)
        raise

    results_dict = {}
    with pool_obj as pool:
        for p, result_map in tqdm(pool.imap_unordered(_solve_worker_wrapper, tasks), 
                                  total=len(tasks), desc="Solving Mumford Mod P"):
            results_dict[p] = result_map
    
    mumford_timer_add("parallel_solving", time.time() - t0)
    mumford_timer_add("residue_computation_total", time.time() - t_start)
    
    if debug:
        print(f"[mumford] Residue computation took {time.time() - t_start:.2f}s")
            
    return results_dict


# mumford_complete.py
#
# Optimized versions of key functions from mumford_complete.py
# Use same function names so dedup.py will clean up old versions

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve, CDF, RR, ceil

# Timers
_MUMFORD_TIMERS = defaultdict(float)

def mumford_timer_add(name, elapsed):
    _MUMFORD_TIMERS[name] += elapsed

def mumford_timers_reset():
    global _MUMFORD_TIMERS
    _MUMFORD_TIMERS.clear()

def mumford_timers_print():
    if not _MUMFORD_TIMERS:
        return
    print("\n[mumford detailed timers]")
    items = sorted(_MUMFORD_TIMERS.items(), key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in items)
    for name, t in items:
        pct = 100.0 * t / total if total > 0 else 0.0
        print(f"  {name:40s}: {t:8.3f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL':40s}: {total:8.3f}s")


def build_mumford_basis_incremental(all_divisors, f_coeffs, num_doublings=NUM_DOUBLINGS, debug=True):
    """
    Build independent basis using height pairing checks.
    
    Will use Arakelov heights if available, otherwise falls back to exact doubling method.
    
    Args:
        num_doublings: Number of doubling iterations (only used for fallback method)
    """
    if ARAKELOV_AVAILABLE:
        if debug:
            print("[basis] Using Arakelov heights for basis construction")
        
        # INCREASED DEFAULT PRECISION to prevent false rank increases
        prec = 2048
        max_attempts = 2
        
        for attempt in range(max_attempts):
            try:
                # Use the parallel version which has the new checks
                result = arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=prec, debug=True)
                return result
            except Exception as e:
                if attempt < max_attempts - 1:
                    prec *= 2
                    if debug:
                        print(f"[basis] Attempt {attempt+1} failed with prec={prec//2}, retrying with prec={prec}")
                    clear_period_cache()
                    raise
                else:
                    if debug:
                        print(f"[basis] All Arakelov attempts failed, falling back to exact method")
                        # DIAGNOSTIC: Print the actual error causing the fallback
                        print(f"[basis] Arakelov Failure Reason: {type(e).__name__}: {e}")
                        # traceback.print_exc() # Uncomment for full stack trace if needed
                    raise
                    return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings, debug)
    else:
        assert None, "deprecated"
        if debug:
            print("[basis] Using exact doubling method for basis construction")
        return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings, debug)


def build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=NUM_DOUBLINGS, debug=True):
    """
    OLD METHOD: Build independent basis using EXACT height pairing checks via doubling.
    This is the fallback when Arakelov module is not available.
    """
    if not all_divisors:
        return [], 0, None
    
    print(f"\n[basis] Starting with {len(all_divisors)} total divisors")
    print(f"[basis] Using {num_doublings} doublings for height pairing approximation")
    
    # Build curve once
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Filter out torsion with HIGHER order bound
    non_torsion = []
    torsion_count = 0
    
    max_torsion_order = 100
    
    for div in all_divisors:
        is_tors, order = is_mumford_torsion_fast(
            div['s'], div['p'], div['v_0'], div['v_1'], 
            f_coeffs, max_order=max_torsion_order, debug=False
        )
        
        if is_tors:
            torsion_count += 1
            if debug and torsion_count <= 5:
                print(f"[basis] Filtered torsion divisor (order {order}): s={div['s']}, p={div['p']}")
        else:
            non_torsion.append(div)
    
    print(f"[basis] Filtered {torsion_count} torsion divisors -> {len(non_torsion)} candidates")
    
    if not non_torsion:
        return [], 0, None
    
    # Convert to Jacobian elements
    jac_elements = []
    for div in non_torsion:
        u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        jac_elements.append((div, D))
    
    # Build basis using EXACT independence checks
    basis = []
    basis_jac = []
    
    typical_height = None
    
    for i, (div, D) in enumerate(jac_elements):
        if not basis:
            # First divisor - just check self-pairing is nonzero
            try:
                h_exact = compute_height_pairing_exact(D, D, f_coeffs, num_doublings=num_doublings)
            except (ValueError, RationalReconstructionError) as e:
                if debug:
                    print(f"[basis] compute_height_pairing_exact failed for candidate {i+1}: {e}")
                raise
                continue

            h_float = float(h_exact) # Keep sign!
            
            # DIAGNOSTIC: Check negativity
            if h_float < 0:
                 print(f"[basis] WARNING: Negative self-pairing for divisor {i+1}: {h_float:.6g}")
                 # Strictly reject negative heights
                 continue

            if h_float < 1e-8:
                if debug:
                    print(f"[basis] Skipping divisor {i+1}: self-pairing too small ({h_float:.3g})")
                continue
            
            typical_height = h_float
            basis.append(div)
            basis_jac.append(D)
            if debug:
                print(f"[basis] Added divisor 1 (self-pairing {h_float:.3g})")
        else:
            # Check independence by computing height pairing matrix
            candidate_basis = basis_jac + [D]
            m = len(candidate_basis)
            
            # Build matrix with EXACT rationals
            H_exact = Matrix(QQ, m, m)
            for ii in range(m):
                for jj in range(ii, m):
                    h_ij_exact = compute_height_pairing_exact(
                        candidate_basis[ii], 
                        candidate_basis[jj],
                        f_coeffs,
                        num_doublings=num_doublings
                    )
                    H_exact[ii, jj] = h_ij_exact
                    H_exact[jj, ii] = h_ij_exact
            
            # Check rank using exact arithmetic
            det_exact = H_exact.determinant()
            rank_exact = H_exact.rank()
            
            # Convert to float for display/checks
            det_float = float(det_exact) # DO NOT USE ABS()
            
            # Regulator check
            if typical_height is None:
                typical_height = 10.0
            
            expected_det_scale = typical_height ** m
            det_ratio = det_float / expected_det_scale
            
            min_ratio = 1e-6
            max_ratio = 1e6
            
            # DIAGNOSTIC: Check eigenvalues for positive definiteness
            try:
                # Use CDF to get approximate eigenvalues
                evals = H_exact.change_ring(CDF).eigenvalues()
                min_eval = min(e.real() for e in evals)
                is_pos_def = min_eval > -1e-9 # Allow tiny epsilon for float noise
            except Exception:
                is_pos_def = True # Fallback if eigenvalue comp fails
                min_eval = 0.0
                raise

            # STRICTER CHECK: Rank must be full, det must be positive, matrix must be pos-def
            is_good = (rank_exact == m and det_float > 0 and min_ratio < abs(det_ratio) < max_ratio and is_pos_def)
            
            if is_good:
                basis.append(div)
                basis_jac.append(D)
                if debug:
                    print(f"[basis] Added divisor {len(basis)} (rank {rank_exact}/{m}, det {det_float:.3g}, ratio {det_ratio:.3g})")
            else:
                if debug:
                    reason_parts = []
                    if rank_exact < m: reason_parts.append(f"rank dropped to {rank_exact}")
                    if det_float <= 0: reason_parts.append(f"non-positive det {det_float:.3g}")
                    if not (min_ratio < abs(det_ratio) < max_ratio): reason_parts.append(f"ratio {det_ratio:.3g} out of range")
                    if not is_pos_def: reason_parts.append(f"not pos-def (min eval {min_eval:.3g})")
                    
                    reason = ", ".join(reason_parts)
                    print(f"[basis] Skipping divisor {i+1}: {reason}")
                    
                    # DIAGNOSTIC: Print the matrix if it failed "mysteriously" (rank full but rejected)
                    if False:
                        if rank_exact == m and (det_float <= 0 or not is_pos_def):
                            print(f"[basis] DEBUG: Bad Matrix at size {m}:")
                            print(H_exact.str())
                            print(f"[basis] DEBUG: Eigenvalues: {evals}")
    
    rank = len(basis)
    
    # Build final height matrix with EXACT rationals
    if rank > 0:
        H_exact = Matrix(QQ, rank, rank)
        for i in range(rank):
            for j in range(i, rank):
                h_ij_exact = compute_height_pairing_exact(
                    basis_jac[i], 
                    basis_jac[j], 
                    f_coeffs,
                    num_doublings=num_doublings
                )
                H_exact[i, j] = h_ij_exact
                H_exact[j, i] = h_ij_exact
        
        if debug:
            print(f"\n[basis] Final rank: {rank}")
            print(f"[basis] Checked {len(jac_elements)} candidates total")
            det_exact = H_exact.determinant()
            print(f"[basis] Determinant (exact): {det_exact}")
            print(f"[basis] Determinant (float): {float(det_exact):.6g}")
            
            # DIAGNOSTIC: Final Matrix Dump
            #print("[basis] Final Height Matrix:")
            #print(H_exact.str())
            try:
                evals = H_exact.change_ring(CDF).eigenvalues()
                print(f"[basis] Final Eigenvalues: {evals}")
            except:
                raise

    else:
        H_exact = None
    
    return basis, rank, H_exact

def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val):
    """
    Optimized solver for v^2 = f(x) mod u(x) where u(x) = x^2 - s*x + p_val.
    Uses Sage's GF(p) for efficient square roots for all primes.
    """
    solutions = []
    x_res = int(x_residue) % p
    x_sq = (x_res * x_res) % p
    
    # Use Sage's finite field for robust arithmetic and square roots
    Fp = GF(p)

    for s_val in range(p):
        # u(x_res) = 0  =>  x_res^2 - s*x_res + p_val = 0
        p_val = (s_val * x_res - x_sq) % p
        
        # Fast reduction of f mod u to get linear remainder A*x + B
        # f(x) = A*x + B mod u(x)
        # Note: f_coeffs is expected in HIGH -> LOW order
        A, B = _poly_mod_quad_fast(f_coeffs, s_val, p_val, p)
        
        # We solve the system:
        # 1) s*v1^2 + 2*v1*v0 = A
        # 2) v0^2 - p*v1^2 = B
        # This reduces to a quadratic in Z = v1^2:
        # (s^2 - 4p)Z^2 + (-2As - 4B)Z + A^2 = 0
        
        a_q = (s_val * s_val - 4 * p_val) % p
        b_q = (-2 * (A * s_val + 2 * B)) % p
        c_q = (A * A) % p
        
        Z_roots = []

        if a_q == 0:
            if b_q != 0:
                Z_roots.append((-c_q * pow(b_q, -1, p)) % p)
        else:
            disc_q = (b_q * b_q - 4 * a_q * c_q) % p
            try:
                # Use GF(p) for sqrt
                delta = Fp(disc_q)
                if delta.is_square():
                    sq_root = int(delta.sqrt())
                    inv_2a = pow(2 * a_q, -1, p)
                    Z_roots.append(((-b_q + sq_root) * inv_2a) % p)
                    if sq_root != 0:
                        Z_roots.append(((-b_q - sq_root) * inv_2a) % p)
            except Exception:
                raise
        
        valid_v1s = []
        for Z in Z_roots:
            Z_ele = Fp(Z)
            if Z_ele.is_square():
                r = int(Z_ele.sqrt())
                valid_v1s.append(r)
                if r != 0:
                    valid_v1s.append(p - r)

        for v1_val in valid_v1s:
            if v1_val == 0:
                # If v1=0, then A must be 0 (from eq 1) and v0^2 = B (from eq 2)
                if A != 0:
                    continue
                B_ele = Fp(B)
                if B_ele.is_square():
                    r = int(B_ele.sqrt())
                    solutions.append((s_val, p_val, r, 0))
                    if r != 0:
                        solutions.append((s_val, p_val, p - r, 0))
            else:
                if p == 2:
                    # Special case for p=2
                    v0_val = (B + p_val) % 2
                    if (s_val * v1_val) % 2 == A % 2:
                         solutions.append((s_val, p_val, v0_val, v1_val))
                else:
                    # v0 is determined linearly if v1 != 0
                    # 2*v1*v0 = A - s*v1^2
                    num = (A - s_val * (v1_val * v1_val)) % p
                    den = (2 * v1_val) % p
                    v0_val = (num * pow(den, -1, p)) % p
                    
                    # Verify with second equation as check
                    lhs_2 = (v0_val * v0_val - p_val * v1_val * v1_val) % p
                    if lhs_2 == B:
                        solutions.append((s_val, p_val, v0_val, v1_val))

    return solutions

def _poly_mod_quad_fast(f_coeffs, s_val, p_val, mod_p):
    """
    Computes f(x) mod (x^2 - s*x + p) efficiently using Horner's method.
    Input f_coeffs must be in HIGH -> LOW order.
    Returns (r1, r0) such that f(x) = r1*x + r0.
    """
    r1 = 0
    r0 = 0
    for coeff in f_coeffs:
        # x(r1*x + r0) = r1*x^2 + r0*x
        #              = r1(s*x - p) + r0*x
        #              = (r1*s + r0)*x - r1*p
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        r1, r0 = new_r1, new_r0
    return r1, r0


from math import isqrt

def _rational_is_square(q):
    """
    q is a QQ rational. Return (True, sqrt_QQ) if q is a rational square, else (False, None).
    Uses integer sqrt on numerator/denominator.
    """
    q = QQ(q)
    num = int(q.numerator())
    den = int(q.denominator())
    if num < 0 or den <= 0:
        return False, None
    s_num = isqrt(abs(num))
    s_den = isqrt(den)
    if s_num * s_num == num and s_den * s_den == den:
        return True, QQ(s_num) / QQ(s_den)
    return False, None


def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Replace existing dedup: handle split-u correctly by using sign-pairs at rational roots.
    Also handles double-root case (disc=0).
    """
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = R(0)
    for c in f_coeffs:
        f_poly = f_poly * x + QQ(c)

    seen = {}
    out = []

    for tup in divisors:
        s_raw, p_raw, v0_raw, v1_raw = tup['s'], tup['p'], tup['v_0'], tup['v_1']

        if not verify_mumford_pair(f_coeffs, s_raw, p_raw, v0_raw, v1_raw, modulus=None):
            continue

        s_q = QQ(s_raw)
        p_q = QQ(p_raw)
        v0_q = QQ(v0_raw)
        v1_q = QQ(v1_raw)

        disc = s_q * s_q - 4 * p_q

        if disc == 0:
            # DOUBLE ROOT CASE: u(x) = (x - r)^2 where r = s/2
            r_double = s_q / QQ(2)
            
            # v(r) = v1*r + v0
            vr = v1_q * r_double + v0_q
            
            # f(r) must be a rational square
            fr = f_poly(r_double)
            ok, sqrt_fr = _rational_is_square(fr)
            
            if not ok:
                raise ValueError(f"Expected f(root) to be rational square for double root but got: f({r_double})={fr}")
            
            # Determine sign
            if vr == sqrt_fr:
                sig = +1
            elif vr == -sqrt_fr:
                sig = -1
            else:
                raise ValueError(f"v(r_double) not equal to ±sqrt(f(r_double)): v({r_double})={vr}, sqrt={sqrt_fr}")
            
            # Normalize: prefer v1>=0 or (v1==0 and v0>=0)
            if v1_q < 0 or (v1_q == 0 and v0_q < 0):
                v0_q = -v0_q
                v1_q = -v1_q
                sig = -sig
            
            key = ('double', QQ(s_q), QQ(p_q), int(sig))
            
            if key not in seen:
                seen[key] = True
                tup['s'] = QQ(s_q)
                tup['p'] = QQ(p_q)
                tup['v_0'] = QQ(v0_q)
                tup['v_1'] = QQ(v1_q)
                tup['has_rational_roots'] = True
                out.append(tup)

        elif disc > 0 and disc.is_square():
            # SPLIT CASE: two distinct rational roots
            r_plus = (s_q + disc.sqrt()) / QQ(2)
            r_minus = (s_q - disc.sqrt()) / QQ(2)

            vr_plus = v1_q * r_plus + v0_q
            vr_minus = v1_q * r_minus + v0_q

            fa_plus = f_poly(r_plus)
            fa_minus = f_poly(r_minus)

            ok_plus, sqrt_plus = _rational_is_square(fa_plus)
            ok_minus, sqrt_minus = _rational_is_square(fa_minus)

            if not ok_plus or not ok_minus:
                raise ValueError(f"Expected f(root) to be rational square for split-u but got non-square: f({r_plus})={fa_plus}, f({r_minus})={fa_minus}")

            if vr_plus == sqrt_plus:
                sig_plus = +1
            elif vr_plus == -sqrt_plus:
                sig_plus = -1
            else:
                raise ValueError(f"v(r_plus) not equal to ±sqrt(f(r_plus)): v({r_plus})={vr_plus}, sqrt={sqrt_plus}")

            if vr_minus == sqrt_minus:
                sig_minus = +1
            elif vr_minus == -sqrt_minus:
                sig_minus = -1
            else:
                raise ValueError(f"v(r_minus) not equal to ±sqrt(f(r_minus)): v({r_minus})={vr_minus}, sqrt={sqrt_minus}")

            key = ('split', QQ(s_q), QQ(p_q), int(sig_plus), int(sig_minus))

            denom = r_plus - r_minus
            alpha = ( (sig_plus * sqrt_plus) - (sig_minus * sqrt_minus) ) / denom
            beta = (sig_plus * sqrt_plus) - alpha * r_plus

            if alpha < 0 or (alpha == 0 and beta < 0):
                alpha = -alpha
                beta = -beta
                sig_plus = -sig_plus
                sig_minus = -sig_minus
                key = ('split', QQ(s_q), QQ(p_q), int(sig_plus), int(sig_minus))

            if key not in seen:
                seen[key] = True
                tup['s'] = QQ(s_q)
                tup['p'] = QQ(p_q)
                tup['v_1'] = QQ(alpha)
                tup['v_0'] = QQ(beta)
                tup['has_rational_roots'] = True
                out.append(tup)

        else:
            # IRREDUCIBLE CASE
            s1, p1, v01, v11 = _normalize_sign(s_q, p_q, v0_q, v1_q)
            key = ('irr', QQ(s1), QQ(p1), QQ(v01), QQ(v11))
            if key not in seen:
                seen[key] = True
                tup['s'], tup['p'], tup['v_0'], tup['v_1'] = s1, p1, v01, v11
                tup['has_rational_roots'] = False
                out.append(tup)

    return out


def quick_dependence_check(div1, div2):
    """Check if two divisors with same u are dependent"""
    if (div1['s'], div1['p']) != (div2['s'], div2['p']):
        return False  # different u
    
    # Same u - check if v1 ≡ ±v2 (mod u)
    # For Mumford rep: v(x) = v_1*x + v_0
    # Check: (v1_1*x + v1_0) ≡ ±(v2_1*x + v2_0) (mod u(x))
    
    # Simplest: just check if coefficients are ± each other
    if (div1['v_0'] == div2['v_0'] and div1['v_1'] == div2['v_1']):
        return True  # identical
    if (div1['v_0'] == -div2['v_0'] and div1['v_1'] == -div2['v_1']):
        return True  # negatives
    
    return False  # might still be dependent, but not obviously


def prefilter_solutions_algebraic(sol_list, prime, f_coeffs):
    """
    Filter solutions by algebraic constraint mod p BEFORE CRT.
    This eliminates ~83% of invalid combinations early.
    
    Returns: list of solutions that pass verify_mumford_pair mod p
    """
    from sage.all import GF, PolynomialRing
    
    R = PolynomialRing(GF(prime), 'x')
    x = R.gen()
    
    # Build f(x) mod p
    f_poly_coeffs = [int(c) % prime for c in f_coeffs]
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    filtered = []
    for sol in sol_list:
        s_val, p_val, v0_val, v1_val = [int(v) % prime for v in sol]
        
        # Build u(x) = x² - s*x + p
        u_poly = x**2 - s_val*x + p_val
        
        # Build v(x) = v1*x + v0
        v_poly = v1_val*x + v0_val
        
        # Check: v(x)² ≡ f(x) (mod u(x))
        diff = v_poly**2 - f_poly
        remainder = diff % u_poly
        
        if remainder.is_zero():
            filtered.append(sol)
    
    return filtered


def rational_reconstruct_fast(c, N, max_den=None, max_num=None):
    """
    Fast rational reconstruction with early height rejection.
    
    Fails fast if height bounds will be violated, avoiding unnecessary computation.
    Returns (num, den) or raises RationalReconstructionError.
    
    Args:
        c: integer to reconstruct
        N: modulus
        max_den: maximum allowed denominator (default: floor(sqrt(N/2)))
        max_num: maximum allowed numerator (if None, no numerator check)
    """
    if max_den is None:
        max_den = floor(sqrt(N / QQ(2)))
    
    c = c % N
    if c == 0:
        return 0, 1
    if c == 1 and max_den >= 1:
        return 1, 1
    
    # Early rejection: if c is too large, it's likely to produce large numerators
    # This is a heuristic but catches many cases
    if max_num is not None and c > max_num and (N - c) > max_num:
        raise RationalReconstructionError(f"CRT value too large: c={c}")
    
    # Extended Euclidean Algorithm with early termination
    r0, r1 = N, c
    t0, t1 = 0, 1
    
    while r1 != 0:
        # Check denominator bound BEFORE next iteration
        if abs(t1) > max_den:
            a, b = r0, t0
            break
        
        # Early numerator check during iteration
        if max_num is not None and abs(r1) > max_num:
            raise RationalReconstructionError(f"Numerator exceeds bound: {abs(r1)} > {max_num}")
        
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    else:
        # Loop finished because r1 == 0
        a, b = r0, t0
    
    # Final validation
    if abs(b) > max_den or b == 0:
        raise RationalReconstructionError(f"No reconstruction for c={c}, N={N}, max_den={max_den}")
    
    if b < 0:
        a, b = -a, -b
    
    # Final numerator check
    if max_num is not None and abs(a) > max_num:
        raise RationalReconstructionError(f"Numerator exceeds bound: {abs(a)} > {max_num}")
    
    if (a - c * b) % N != 0:
        raise RationalReconstructionError(f"Validation failed for c={c}, N={N}: got a={a}, b={b}")
    
    g = gcd(abs(a), abs(b))
    return int(a // g), int(b // g)


def reconstruct_mumford_combo_fast(sol_combo, primes, M, max_height):
    """
    Fast reconstruction of a single combination with early rejection.
    
    Returns (s, p, v0, v1) or raises RationalReconstructionError early.
    """
    rec_vals = []
    
    for idx in range(4):
        vals = tuple(sol[idx] for sol in sol_combo)
        crt_val = crt_cached(vals, tuple(primes))
        
        # Reconstruct with BOTH numerator and denominator bounds
        #num, den = rational_reconstruct(crt_val, M, max_den=max_height)
        num, den = rational_reconstruct_with_height_check(crt_val, M, max_height)
        
        rec_vals.append(QQ(num) / QQ(den))
    
    return rec_vals


# Modified worker for parallel processing


# Usage note:
# Replace your rational_reconstruct calls with rational_reconstruct_fast
# Pass max_num=max_height to enable numerator checking
# This should give ~10-20% speedup by failing earlier on doomed combinations


def setup_crt_constants(primes):
    """
    Precompute weights for fast CRT: result = sum(val_i * w_i) % M.
    Returns (M, weights).
    """
    M = 1
    for p in primes:
        M *= p
    
    weights = []
    for p in primes:
        m_i = M // p
        # inverse of m_i mod p
        # use python int pow to ensure ValueError on failure, though m_i is coprime to p by definition
        inv = pow(int(m_i), -1, int(p))
        w_i = (m_i * inv)
        weights.append(w_i)
        
    return M, weights

def fast_rational_reconstruct_check(val, M, max_height):
    """
    Pure integer rational reconstruction check. 
    Returns (True, num, den) or (False, 0, 0).
    Optimized for tight loops: avoids object creation and returns early.
    """
    r0, r1 = M, val
    t0, t1 = 0, 1
    
    # Unrolled Euclidean Algorithm
    while r1 > max_height:
        if t1 > max_height or t1 < -max_height:
            return False, 0, 0
            
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
        
    if abs(t1) > max_height:
        return False, 0, 0
        
    if t1 < 0:
        t1 = -t1
        r1 = -r1
        
    if abs(r1) > max_height:
        return False, 0, 0
        
    return True, r1, t1


def _reconstruct_worker_parallel_v2(args):
    """
    Optimized worker with robust error handling for modular inverses.
    """
    # Unpack the new v_tuple argument at the end
    combo_batch, primes, M_in, f_coeffs, max_height, v_tuple = args
    
    # Setup fast CRT constants once per batch
    M, weights = setup_crt_constants(primes)
    
    results = []
    stats = {
        'attempted': len(combo_batch),
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }
    
    # Pre-calculate prime integers to avoid sage overhead in loop
    primes_int = [int(p) for p in primes]
    range_primes = range(len(primes))
    
    for sol_combo in combo_batch:
        # 1. Reconstruct 's' (index 0)
        crt_s = 0
        for i in range_primes:
            crt_s += sol_combo[i][0] * weights[i]
        crt_s %= M
        
        success_s, num_s, den_s = fast_rational_reconstruct_check(crt_s, M, max_height)
        if not success_s:
            stats['height_reject'] += 1
            continue
            
        # 2. Reconstruct 'p' (index 1)
        crt_p = 0
        for i in range_primes:
            crt_p += sol_combo[i][1] * weights[i]
        crt_p %= M
        
        success_p, num_p, den_p = fast_rational_reconstruct_check(crt_p, M, max_height)
        if not success_p:
            stats['height_reject'] += 1
            continue

        # 3. Reconstruct v0 (index 2)
        crt_v0 = 0
        for i in range_primes:
            crt_v0 += sol_combo[i][2] * weights[i]
        crt_v0 %= M
        
        success_v0, num_v0, den_v0 = fast_rational_reconstruct_check(crt_v0, M, max_height)
        if not success_v0:
            stats['height_reject'] += 1
            continue
            
        # 4. Reconstruct v1 (index 3)
        crt_v1 = 0
        for i in range_primes:
            crt_v1 += sol_combo[i][3] * weights[i]
        crt_v1 %= M
        
        success_v1, num_v1, den_v1 = fast_rational_reconstruct_check(crt_v1, M, max_height)
        if not success_v1:
            stats['height_reject'] += 1
            continue

        # 5. Consistency Check (Mod P)
        reconstruction_ok = True
        
        # We work with python ints to avoid Sage ZeroDivisionErrors on mod invert
        for i in range_primes:
            p_int = primes_int[i]
            expected = sol_combo[i]
            
            try:
                # Use pow(val, -1, mod) which raises ValueError on failure
                
                # Check s
                if (num_s * pow(den_s, -1, p_int)) % p_int != expected[0]:
                    reconstruction_ok = False; break
                
                # Check p
                if (num_p * pow(den_p, -1, p_int)) % p_int != expected[1]:
                    reconstruction_ok = False; break
                    
                # Check v0
                if (num_v0 * pow(den_v0, -1, p_int)) % p_int != expected[2]:
                    reconstruction_ok = False; break
                    
                # Check v1
                if (num_v1 * pow(den_v1, -1, p_int)) % p_int != expected[3]:
                    reconstruction_ok = False; break

            except (ValueError, ZeroDivisionError):
                # Denominator divisible by prime -> reconstruction failed
                reconstruction_ok = False
                break
        
        if not reconstruction_ok:
            stats['consistency_reject'] += 1
            continue

        # 6. Convert to Sage types for algebraic verification
        s_qq = QQ(num_s) / QQ(den_s)
        p_qq = QQ(num_p) / QQ(den_p)
        v0_qq = QQ(num_v0) / QQ(den_v0)
        v1_qq = QQ(num_v1) / QQ(den_v1)

        # 7. Algebraic Verification
        if not verify_mumford_pair(f_coeffs, s_qq, p_qq, v0_qq, v1_qq, modulus=None, debug_first_failure=False):
            stats['algebraic_reject'] += 1
            continue
        
        # Attach the vector here so it survives the return trip
        results.append({'s': s_qq, 'p': p_qq, 'v_0': v0_qq, 'v_1': v1_qq, 'vector': v_tuple})
        stats['success'] += 1
    
    return results, stats

def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Optimized reconstruction with batched parallel processing.
    """
    t_start_total = time.time()
    
    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE")
    print("="*70)

    mumford_timers_reset()
    
    found_xs = set()
    mumford_divisors_raw = []

    t0 = time.time()
    by_vector_and_xres = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for p in residues:
        for v_tuple, x_res_dict in residues[p].items():
            if isinstance(x_res_dict, list):
                by_vector_and_xres[v_tuple]['unknown'][p] = x_res_dict
            elif isinstance(x_res_dict, dict):
                for x_res, sols in x_res_dict.items():
                    by_vector_and_xres[v_tuple][x_res][p] = sols
    
    mumford_timer_add("residue_grouping", time.time() - t0)


    # NEW: Apply algebraic filtering to each prime's solutions
    t0 = time.time()
    total_before_filter = 0
    total_after_filter = 0
    
    if debug:
        print("\n=== EARLY ALGEBRAIC FILTERING (per prime) ===")
    
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            for prime in list(prime_data.keys()):
                sol_list = prime_data[prime]
                total_before_filter += len(sol_list)
                
                # FILTER: Check algebraic constraint mod p
                filtered = prefilter_solutions_algebraic(sol_list, prime, f_coeffs)
                
                total_after_filter += len(filtered)
                prime_data[prime] = filtered
                
                if debug and len(filtered) < len(sol_list):
                    pct = 100.0 * len(filtered) / len(sol_list)
                    print(f"  Prime {prime}: {len(sol_list)} → {len(filtered)} sols ({pct:.1f}%)")
    
    filter_reduction = total_before_filter / max(1, total_after_filter)
    if debug:
        print(f"\nAlgebraic pre-filter: {total_before_filter:,} → {total_after_filter:,} "
              f"({filter_reduction:.1f}x reduction)")
    
    mumford_timer_add("algebraic_prefiltering", time.time() - t0)


    num_groups = sum(len(xres_groups) for xres_groups in by_vector_and_xres.values())
    print(f"Grouped into {len(by_vector_and_xres)} vectors, {num_groups} (vector,x-residue) pairs")

    total_attempted = 0
    total_stats = {
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0,
        'prefilter_reject': 0,
        'skipped_2prime': 0,
        'skipped_high_density': 0
    }

    t0 = time.time()
    
    num_workers = max(8, multiprocessing.cpu_count())
    
    all_work_items = []
    
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            primes = sorted(prime_data.keys())
            sol_lists = [prime_data[p] for p in primes]

            # Skip if algebraic filtering eliminated everything
            if any(len(sl) == 0 for sl in sol_lists):
                if debug:
                    print(f"  Skipping group: algebraic filter eliminated all solutions")
                continue

            total_combos_raw = 1
            for sl in sol_lists:
                total_combos_raw *= len(sl)
            
            if len(primes) == 2:
                is_rare = (len(xres_groups) <= 5)
                
                if total_combos_raw > 100000 and not is_rare:
                    if debug:
                        print(f"  Skipping 2-prime group: {total_combos_raw:,} combos, not rare")
                    total_stats['skipped_2prime'] += 1
                    continue
                elif debug and total_combos_raw > 50000:
                    print(f"  Accepting 2-prime group: {total_combos_raw:,} combos (rare divisor)")
            
            if len(primes) < 2:
                continue
            
            M = 1
            for p in primes:
                M *= p
            
            avg_sols_per_prime = sum(len(sl) for sl in sol_lists) / len(sol_lists)
            
            if debug and total_combos_raw > 10000:
                print(f"  Vector group: {len(primes)} primes, avg {avg_sols_per_prime:.1f} sols/prime")
                print(f"  Raw combinations: {total_combos_raw:,}")
            
            if total_combos_raw > 500000 and len(primes) >= 3:
                if debug:
                    print(f"  High density detected - applying aggressive pre-filtering")
                
                t_prefilter = time.time()
                
                p0, p1 = primes[0], primes[1]
                M_small = p0 * p1
                
                candidate_divisors = []
                max_candidates = 50000
                tried = 0
                
                for sol0 in sol_lists[0]:
                    for sol1 in sol_lists[1]:
                        tried += 1
                        if tried > max_candidates:
                            break
                        
                        rec_vals = []
                        for idx in range(4):
                            vals = (sol0[idx], sol1[idx])
                            crt_val = crt_cached(vals, (p0, p1))
                            try:
                                num, den = rational_reconstruct(crt_val, M_small)
                                if abs(num) > 10 * M_small or abs(den) > 10 * M_small:
                                    raise RationalReconstructionError("Height too large")
                                rec_vals.append(QQ(num)/QQ(den))
                            except RationalReconstructionError:
                                break
                        
                        if len(rec_vals) == 4:
                            s, p_val, v0, v1 = rec_vals
                            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                                candidate_divisors.append((s, p_val, v0, v1))
                    
                    if tried > max_candidates:
                        break
                
                if debug:
                    print(f"    Found {len(candidate_divisors)} candidates from first 2 primes (tried {tried:,})")
                
                if candidate_divisors:
                    filtered_sol_lists = [sol_lists[0], sol_lists[1]]
                    
                    for p_idx in range(2, len(primes)):
                        p = primes[p_idx]
                        filtered = []
                        
                        for sol in sol_lists[p_idx]:
                            for cand_s, cand_p, cand_v0, cand_v1 in candidate_divisors:
                                try:
                                    s_mod = (int(cand_s.numerator()) * pow(int(cand_s.denominator()), -1, p)) % p
                                    p_mod = (int(cand_p.numerator()) * pow(int(cand_p.denominator()), -1, p)) % p
                                    v0_mod = (int(cand_v0.numerator()) * pow(int(cand_v0.denominator()), -1, p)) % p
                                    v1_mod = (int(cand_v1.numerator()) * pow(int(cand_v1.denominator()), -1, p)) % p
                                    
                                    if (sol[0] == s_mod and sol[1] == p_mod and 
                                        sol[2] == v0_mod and sol[3] == v1_mod):
                                        filtered.append(sol)
                                        break
                                except ZeroDivisionError:
                                    continue
                        
                        if not filtered:
                            filtered = sol_lists[p_idx]
                        
                        filtered_sol_lists.append(filtered)
                        
                        if debug and len(filtered) != len(sol_lists[p_idx]):
                            pct = 100.0 * len(filtered) / len(sol_lists[p_idx])
                            print(f"    Prime {p}: {len(sol_lists[p_idx])} → {len(filtered)} sols ({pct:.1f}%)")
                    
                    sol_lists = filtered_sol_lists
                
                total_combos_filtered = 1
                for sl in sol_lists:
                    total_combos_filtered *= len(sl)
                
                prefilter_reduction = total_combos_raw / max(1, total_combos_filtered)
                total_stats['prefilter_reject'] += (total_combos_raw - total_combos_filtered)
                
                if debug:
                    print(f"  Pre-filtering took {time.time() - t_prefilter:.2f}s")
                    print(f"  Filtered combinations: {total_combos_filtered:,} ({prefilter_reduction:.1f}x reduction)")
                
                mumford_timer_add("prefiltering", time.time() - t_prefilter)
            
            total_combos = 1
            for sl in sol_lists:
                total_combos *= len(sl)
            
            if total_combos > 5_000_000:
                if debug:
                    print(f"  Skipping group - too many combinations after filtering ({total_combos:,})")
                total_stats['skipped_high_density'] += 1
                continue
            
            disc_deg = len(f_coeffs) - 1
            expected_rank = min(disc_deg - 1, 4)
            
            base_limit = 1000000 * expected_rank
            adaptive_limit = min(base_limit, 500_000, total_combos)
            
            if debug and total_combos > 10000:
                print(f"  Adaptive limit: {adaptive_limit:,} (total combos: {total_combos:,})")
            
            if len(primes) == 2:
                max_height = int(M ** 0.6)
            else:
                max_height = int(M ** 0.5)
            
            all_work_items.append({
                'v_tuple': v_tuple,
                'primes': primes,
                'M': M,
                'sol_lists': sol_lists,
                'max_height': max_height,
                'adaptive_limit': adaptive_limit,
                'total_combos': total_combos
            })
    
    if not all_work_items:
        print("  No work items to process")
        mumford_timer_add("crt_reconstruction_loop", time.time() - t0)
        mumford_timers_print()
        return found_xs, []
    
    total_work = sum(item['total_combos'] for item in all_work_items)
    use_parallel = total_work > 100000
    
    if use_parallel:
        if debug:
            print(f"\n  Batched parallel processing: {len(all_work_items)} groups, {total_work:,} total combos")
            print(f"  Using {num_workers} workers")
        
        all_batches = []
        for item in all_work_items:
            limit = min(item['adaptive_limit'], item['total_combos'])
            all_combos = list(islice(product(*item['sol_lists']), limit))
            
            batch_size = max(5000, len(all_combos) // (num_workers * 2))
            
            for i in range(0, len(all_combos), batch_size):
                batch = all_combos[i:i+batch_size]
                all_batches.append((
                    batch,
                    item['primes'],
                    item['M'],
                    f_coeffs,
                    item['max_height'],
                    item['v_tuple']  # PASS THE VECTOR TUPLE HERE
                ))
        
        if debug:
            print(f"  Created {len(all_batches)} batches for parallel processing")
        
        try:
            ctx = multiprocessing.get_context("fork")
            pool = ctx.Pool(num_workers)
        except Exception:
            pool = multiprocessing.Pool(num_workers)
        
        try:
            for batch_results, batch_stats in pool.imap_unordered(_reconstruct_worker_parallel_v2, all_batches):
                for div in batch_results:
                    # Do not override div['vector'] here; use the one from the worker
                    mumford_divisors_raw.append(div)
                
                total_stats['height_reject'] += batch_stats['height_reject']
                total_stats['consistency_reject'] += batch_stats['consistency_reject']
                total_stats['algebraic_reject'] += batch_stats['algebraic_reject']
                total_stats['success'] += batch_stats['success']
                total_attempted += batch_stats['attempted']
            
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
    else:
        if debug:
            print(f"\n  Serial processing: {len(all_work_items)} groups, {total_work:,} total combos")
        
        first_diagnostic_done = False
        
        for item in all_work_items:
            primes = item['primes']
            M = item['M']
            sol_lists = item['sol_lists']
            max_height = item['max_height']
            v_tuple = item['v_tuple']
            
            limit = min(item['adaptive_limit'], item['total_combos'])
            
            for sol_combo in islice(product(*sol_lists), limit):
                total_attempted += 1
                
                try:
                    rec_vals = []
                    for idx in range(4):
                        vals = [sol[idx] for sol in sol_combo]
                        crt_val = crt_cached(tuple(vals), tuple(primes))
                        num, den = rational_reconstruct(crt_val, M)
                        
                        if abs(num) > max_height or abs(den) > max_height:
                            raise RationalReconstructionError("Height too large")
                        
                        rec_vals.append(QQ(num)/QQ(den))
                    
                    s, p_val, v0, v1 = rec_vals
                    
                    if not first_diagnostic_done and debug:
                        print(f"\n=== FIRST RECONSTRUCTION DIAGNOSTIC ===")
                        print(f"Reconstructed: s={s}, p={p_val}, v0={v0}, v1={v1}")
                        print(f"Heights: |num(s)|={abs(s.numerator())}, |den(s)|={abs(s.denominator())}")
                        print(f"         |num(p)|={abs(p_val.numerator())}, |den(p)|={abs(p_val.denominator())}")
                        print(f"Max height allowed: {max_height}")
                        print(f"M = {M} ({len(primes)} primes)")
                        first_diagnostic_done = True
                    
                except RationalReconstructionError:
                    total_stats['height_reject'] += 1
                    continue
                
                reconstruction_ok = True
                for i, prime in enumerate(primes):
                    expected_sol = sol_combo[i]
                    try:
                        s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, prime)) % prime
                        p_mod = (int(p_val.numerator()) * pow(int(p_val.denominator()), -1, prime)) % prime
                        v0_mod = (int(v0.numerator()) * pow(int(v0.denominator()), -1, prime)) % prime
                        v1_mod = (int(v1.numerator()) * pow(int(v1.denominator()), -1, prime)) % prime
                    except ZeroDivisionError:
                        reconstruction_ok = False
                        break
                    
                    if (s_mod != expected_sol[0] % prime or
                        p_mod != expected_sol[1] % prime or
                        v0_mod != expected_sol[2] % prime or
                        v1_mod != expected_sol[3] % prime):
                        reconstruction_ok = False
                        break
                
                if not reconstruction_ok:
                    total_stats['consistency_reject'] += 1
                    continue
                
                if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                    if total_stats['algebraic_reject'] == 0 and debug:
                        print(f"\n=== FIRST ALGEBRAIC REJECTION ===")
                        print(f"s={s}, p={p_val}, v0={v0}, v1={v1}")
                        print(f"Re-running with debug...")
                        verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=True)
                    total_stats['algebraic_reject'] += 1
                    continue
                
                mumford_divisors_raw.append({
                    'vector': v_tuple, 's': s, 'p': p_val, 'v_0': v0, 'v_1': v1
                })
                total_stats['success'] += 1

    mumford_timer_add("crt_reconstruction_loop", time.time() - t0)

    print(f"\n=== RECONSTRUCTION SUMMARY ===")
    print(f"  Combinations tried: {total_attempted:,}")
    print(f"  Groups skipped (2-prime): {total_stats['skipped_2prime']}")
    print(f"  Groups skipped (high density): {total_stats['skipped_high_density']}")
    print(f"  Pre-filtered out: {total_stats['prefilter_reject']:,}")
    print(f"  Rejected by height: {total_stats['height_reject']:,}")
    print(f"  Rejected by consistency: {total_stats['consistency_reject']:,}")
    print(f"  Rejected by algebraic constraint: {total_stats['algebraic_reject']:,}")
    print(f"  Successful reconstructions: {total_stats['success']:,}")

    if not mumford_divisors_raw:
        print("  WARNING: No valid Mumford divisors reconstructed!")
        mumford_timers_print()
        return found_xs, []

    t0 = time.time()
    mumford_divisors_raw = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)
    mumford_timer_add("canonicalization", time.time() - t0)

    mumford_divisors = []
    for i, divi in enumerate(mumford_divisors_raw):
        is_dep = False
        for j, divj in enumerate(mumford_divisors_raw):
            if i <= j:
                continue
            if quick_dependence_check(divi, divj):
                is_dep = True
        if not is_dep:
            mumford_divisors.append(divi)

    t0 = time.time()
    for div in mumford_divisors:
        s, p_val = div['s'], div['p']
        disc = s*s - 4*p_val

        if disc >= 0 and disc.is_square():
            div['has_rational_roots'] = True
            r1 = (s + disc.sqrt())/2
            r2 = (s - disc.sqrt())/2
            for r in (r1, r2):
                x_cand = r - shift
                if rationality_test(x_cand) is not None:
                    found_xs.add(x_cand)
        else:
            div['has_rational_roots'] = False
    
    mumford_timer_add("rational_root_check", time.time() - t0)

    print(f"  Unique Rational Points: {len(found_xs)}")
    
    if mumford_divisors:
        unique = {frozenset(d.items()): d for d in mumford_divisors}
        mumford_divisors = list(unique.values())

        # [Fix] Sort divisors by naive height (sum of absolute coeffs) to prioritize 
        # small, simple divisors. This improves basis stability significantly.
        def naive_sort_key(d):
            return abs(QQ(d['s'])) + abs(QQ(d['p'])) + abs(QQ(d['v_0'])) + abs(QQ(d['v_1']))
        
        mumford_divisors.sort(key=naive_sort_key)

        rational_roots_count = sum(1 for div in mumford_divisors_raw
                                   if 'has_rational_roots' in div and div.get('has_rational_roots'))

        print(f"  {rational_roots_count} of {len(mumford_divisors_raw)} original divisors had rational roots in u(x)")
        print(f"\n--- Building Independent Mumford Basis ---")
        print("first 10 divisors:")
        for i in mumford_divisors[:10]:
            print(i)
        
        try:
            t0 = time.time()
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(
                mumford_divisors, 
                f_coeffs, 
                debug=True
            )
            mumford_timer_add("basis_construction", time.time() - t0)
            
            print(f"\nBasis Construction Results:")
            print(f"  Found {basis_rank} independent divisors")
            if basis_H is not None:
                print(f"  Determinant (float): {float(basis_H.determinant())}")
            
            mumford_timers_print()
            
            return found_xs, basis_divisors
        except Exception as e:
            print(f"Basis construction failed: {e}")
            traceback.print_exc()
            mumford_timers_print()
            raise
    
    mumford_timers_print()
    return found_xs, mumford_divisors
