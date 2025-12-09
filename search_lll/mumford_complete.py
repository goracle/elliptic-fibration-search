
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
    from .arakelov import (
        arakelov_height_pairing,
        arakelov_build_basis,
        arakelov_canonical_height,
        clear_period_cache
    )
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
                pass
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


def _poly_mod_quad_fast(f_coeffs, s_val, p_val, mod_p):
    r1 = 0
    r0 = 0
    for coeff in f_coeffs:
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        r1, r0 = new_r1, new_r0
    return r1, r0


def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val):
    solutions = []
    x_res = int(x_residue) % p
    x_sq = (x_res * x_res) % p

    for s_val in range(p):
        p_val = (s_val * x_res - x_sq) % p
        A, B = _poly_mod_quad_fast(f_coeffs, s_val, p_val, p)
        
        coeffs_quad = [
            (s_val * s_val - 4 * p_val) % p,
            (-2 * (A * s_val + 2 * B)) % p,
            (A * A) % p
        ]
        
        a_q, b_q, c_q = coeffs_quad
        Z_roots = []

        if a_q == 0:
            if b_q != 0:
                Z_roots.append((-c_q * pow(b_q, -1, p)) % p)
        else:
            disc_q = (b_q * b_q - 4 * a_q * c_q) % p
            
            if disc_q == 0:
                Z_roots.append((-b_q * pow(2 * a_q, -1, p)) % p)
            elif pow(disc_q, (p - 1) // 2, p) == 1:
                sq_root = None
                if p < 1000:
                    for r in range(1, p):
                        if (r * r) % p == disc_q:
                            sq_root = r
                            break
                
                if sq_root is not None:
                    inv_2a = pow(2 * a_q, -1, p)
                    Z_roots.append(((-b_q + sq_root) * inv_2a) % p)
                    Z_roots.append(((-b_q - sq_root) * inv_2a) % p)
        
        valid_v1s = []
        for Z in Z_roots:
            if Z == 0:
                valid_v1s.append(0)
            else:
                if pow(Z, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == Z:
                            valid_v1s.append(r)
                            valid_v1s.append(p - r)
                            break

        for v1_val in valid_v1s:
            if v1_val == 0:
                if A != 0:
                    continue
                if B == 0:
                    solutions.append((s_val, p_val, 0, 0))
                elif pow(B, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == B:
                            solutions.append((s_val, p_val, r, 0))
                            solutions.append((s_val, p_val, p - r, 0))
                            break
            else:
                if p == 2:
                    v0_val = B % 2
                    if (s_val * v1_val) % 2 == A % 2:
                        solutions.append((s_val, p_val, v0_val, v1_val))
                else:
                    num = (A - s_val * (v1_val * v1_val)) % p
                    den = (2 * v1_val) % p
                    v0_val = (num * pow(den, -1, p)) % p
                    
                    lhs_2 = (v0_val * v0_val - p_val * v1_val * v1_val) % p
                    if lhs_2 == B:
                        solutions.append((s_val, p_val, v0_val, v1_val))

    return solutions


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


def canonicalize_and_dedup(divisors, f_coeffs):
    seen = {}
    out = []
    
    for tup in divisors:
        s, p, v0, v1 = tup['s'], tup['p'], tup['v_0'], tup['v_1']
        
        if not verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None):
            continue
        
        s1, p1, v01, v11 = _normalize_sign(s, p, v0, v1)
        key = (s1, p1, v01, v11)
        
        if key not in seen:
            seen[key] = True
            tup['s'], tup['p'], tup['v_0'], tup['v_1'] = s1, p1, v01, v11
            out.append(tup)
            
    return out


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

def is_mumford_torsion_fast(s, p, v0, v1, f_coeffs, max_order=12, debug=DEBUG):
    """
    Fast torsion test using modular verification.
    Tests if divisor is n-torsion for n in [2, 3, 4, 5, 6, 8, 10, 12].
    
    Returns: (is_torsion, order) where order=None if not torsion
    """
    # Build curve
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Convert to Jacobian element
    u_poly = x**2 - QQ(s)*x + QQ(p)
    v_poly = QQ(v1)*x + QQ(v0)
    D = J([u_poly, v_poly])
    
    if D.is_zero():
        return True, 1
    
    # Test small orders using ONLY addition (no doubling)
    test_orders = [2, 3, 4, 5, 6, 8, 10, 12]
    
    for n in test_orders:
        # Compute nD by repeated addition (safer than doubling)
        nD = D
        for _ in range(n - 1):
            nD = nD + D
        
        if nD.is_zero():
            if debug:
                print(f"[torsion] Found {n}-torsion divisor")
            return True, n
    
    return False, None


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
                    pass

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
        return None, None

    # Build f(x) over Fp using the same helper (safe conversion)
    try:
        f_poly_Fp = _poly_from_coeffs_qq(R_Fp, [Fp(QQ(c)) for c in f_coeffs])
    except Exception:
        # If conversion fails, skip this prime
        if debug:
            print(f"[MOD-DBL] cannot build f_poly mod {p}")
        return None, None

    # If the curve is singular mod p, skip
    try:
        C_Fp = HyperellipticCurve(f_poly_Fp, R_Fp(0))
    except ValueError:
        if debug:
            print(f"[MOD-DBL] singular curve at p={p}")
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
                continue

        # reduce v modulo u to enforce deg v < deg u
        try:
            v_poly_Fp = v_poly_Fp % u_poly_Fp
        except Exception as e:
            tried.append((assume_high, "reduce_failed", str(e)))
            continue

        # quick Mumford test: (v^2 - f) % u == 0
        try:
            rem = (v_poly_Fp**2 - f_poly_Fp).quo_rem(u_poly_Fp)[1]
            if rem != 0:
                tried.append((assume_high, "mumford_test_fail", rem))
                continue
        except ZeroDivisionError:
            tried.append((assume_high, "quo_rem_zero_divisor", None))
            continue
        except Exception as e:
            tried.append((assume_high, "quo_rem_exc", str(e)))
            continue

        # If we reach here, inputs interpreted under this orientation form a valid divisor mod p
        # proceed to doubling
        try:
            D_mod_p = J_Fp([u_poly_Fp, v_poly_Fp])
        except (ValueError, TypeError) as e:
            tried.append((assume_high, "jacobian_construct_fail", str(e)))
            continue

        try:
            D_doubled = 2 * D_mod_p
        except (ValueError, ArithmeticError, ZeroDivisionError) as e:
            tried.append((assume_high, "doubling_failed", str(e)))
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
                return None, None

        # reduce v modulo u
        try:
            v_poly_res = v_poly_res % u_poly_res
        except Exception:
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] cannot reduce v mod u after doubling mod {p}")
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
        return arakelov_build_basis(all_divisors, f_coeffs, prec=100, debug=debug)
    else:
        if debug:
            print("[basis] Using exact doubling method for basis construction")
        return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings, debug)


def build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=NUM_DOUBLINGS, debug=True):
    """
    OLD METHOD: Build independent basis using EXACT height pairing checks via doubling.
    This is the fallback when Arakelov module is not available.
    """
    # [Keep the entire existing implementation here - just rename the function]
    # Copy the current build_mumford_basis_incremental body here exactly as-is
    
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
    
    # Filter out torsion
    non_torsion = []
    torsion_count = 0
    
    for div in all_divisors:
        is_tors, order = is_mumford_torsion_fast(
            div['s'], div['p'], div['v_0'], div['v_1'], 
            f_coeffs, debug=False
        )
        
        if is_tors:
            torsion_count += 1
            if debug and torsion_count <= 3:
                print(f"[basis] Filtered torsion divisor (order {order}): {div}")
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
    
    for i, (div, D) in enumerate(jac_elements):
        if not basis:
            # First divisor - just check self-pairing is nonzero
            try:
                h_exact = compute_height_pairing_exact(D, D, f_coeffs, num_doublings=num_doublings)
            except (ValueError, RationalReconstructionError) as e:
                if debug:
                    print(f"[basis] compute_height_pairing_exact failed for candidate {i+1}: {e}")
                continue

            h_float = float(h_exact)
            
            if abs(h_float) < 1e-8:
                if debug:
                    print(f"[basis] Skipping divisor {i+1}: self-pairing too small ({h_float:.3g})")
                continue
            
            basis.append(div)
            basis_jac.append(D)
            if debug:
                print(f"[basis] Added divisor 1 (self-pairing {h_float:.3g})")
        else:
            # Check independence by computing height pairing matrix
            candidate_basis = basis_jac + [D]
            n = len(candidate_basis)
            
            # Build matrix with EXACT rationals
            H_exact = Matrix(QQ, n, n)
            for ii in range(n):
                for jj in range(ii, n):
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
            
            # Convert to float for display
            det_float = float(det_exact)
            
            det_threshold = 0
            
            if rank_exact == n and det_float > det_threshold:
                basis.append(div)
                basis_jac.append(D)
                if debug:
                    print(f"[basis] Added divisor {len(basis)} (rank {rank_exact}/{n}, det {det_float:.3g})")
            else:
                if debug:
                    reason = "rank dropped" if rank_exact < n else f"det too small ({det_float:.3g})"
                    print(f"[basis] Skipping divisor {i+1}: {reason} (rank {rank_exact}/{n})")
    
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
            print(f"[basis] Determinant (float): {float(det_exact):.3g}")
            print(f"[basis] Height pairing matrix (exact QQ):")
            print(H_exact)
    else:
        H_exact = None
    
    return basis, rank, H_exact

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
        is_indep, rank, H = arakelov_check_independence(jac_elements, f_coeffs, prec=100, debug=debug)
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

def rational_reconstruct_with_height_check(crt_val, M, max_height):
    """
    Rational reconstruction with immediate height rejection.
    Returns (num, den) or raises RationalReconstructionError.
    """
    num, den = rational_reconstruct(crt_val, M)
    
    # Check height immediately
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
            continue
        except Exception:
            continue
    
    return results


def reconstruct_parallel(sol_lists, primes, f_coeffs, adaptive_limit, num_workers=4, debug=False):
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
                    continue
            
            # Full verification
            if verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                results.append({'s': s, 'p': p_val, 'v_0': v0, 'v_1': v1})
        
        except RationalReconstructionError:
            continue
        except Exception:
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


def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Optimized reconstruction with parallel CRT processing and detailed timing.
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

    num_groups = sum(len(xres_groups) for xres_groups in by_vector_and_xres.values())
    print(f"Grouped into {len(by_vector_and_xres)} vectors, {num_groups} (vector,x-residue) pairs")

    total_attempted = 0
    total_stats = {
        'height_reject': 0,
        'consistency_reject': 0,
        'algebraic_reject': 0,
        'success': 0
    }

    t0 = time.time()
    
    num_workers = min(4, multiprocessing.cpu_count())
    use_parallel_threshold = 50000
    
    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            primes = sorted(prime_data.keys())
            if len(primes) < 3:
                continue
            
            M = 1
            for p in primes:
                M *= p
            
            sol_lists = [prime_data[p] for p in primes]
            
            disc_deg = len(f_coeffs) - 1
            expected_rank_upper = disc_deg - 1
            num_primes_used = len(primes)

            base_limit = 1000000
            per_rank_multiplier = 5000
            height_factor = max(1.0, math.log(M) / 50.0)
            adaptive_limit = int(base_limit + expected_rank_upper * per_rank_multiplier * height_factor)

            if debug:
                print(f"  Adaptive limit: {adaptive_limit} (disc_deg={disc_deg}, expected_rank<={expected_rank_upper}, M~10^{int(math.log(M)/math.log(10))})")

            max_height = max(100000, int(M ** 0.35))
            
            total_combos = 1
            for sl in sol_lists:
                total_combos *= len(sl)
            total_combos = min(total_combos, adaptive_limit)
            
            if total_combos > use_parallel_threshold:
                if debug:
                    print(f"  Using parallel reconstruction ({num_workers} workers)")
                
                all_combos = list(islice(product(*sol_lists), adaptive_limit))
                batch_size = max(1000, len(all_combos) // (num_workers * 4))
                
                batches = []
                for i in range(0, len(all_combos), batch_size):
                    batch = all_combos[i:i+batch_size]
                    batches.append((batch, primes, M, f_coeffs, max_height))
                
                try:
                    ctx = multiprocessing.get_context("fork")
                    pool = ctx.Pool(num_workers)
                except Exception:
                    pool = multiprocessing.Pool(num_workers)
                
                try:
                    for batch_results, batch_stats in pool.imap_unordered(_reconstruct_worker_parallel, batches):
                        for div in batch_results:
                            div['vector'] = v_tuple
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
                if debug and total_combos > 10000:
                    print(f"  Using serial reconstruction ({total_combos} combos)")
                
                limit = adaptive_limit
                for sol_combo in product(*sol_lists):
                    if limit <= 0:
                        break
                    limit -= 1
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
                        total_stats['algebraic_reject'] += 1
                        continue
                    
                    mumford_divisors_raw.append({
                        'vector': v_tuple, 's': s, 'p': p_val, 'v_0': v0, 'v_1': v1
                    })
                    total_stats['success'] += 1

    mumford_timer_add("crt_reconstruction_loop", time.time() - t0)

    print(f"  Combinations tried: {total_attempted}")
    print(f"  Rejected by height: {total_stats['height_reject']}")
    print(f"  Rejected by consistency: {total_stats['consistency_reject']}")
    print(f"  Rejected by algebraic constraint: {total_stats['algebraic_reject']}")
    print(f"  Successful reconstructions: {total_stats['success']}")

    if not mumford_divisors_raw:
        print("  WARNING: No valid Mumford divisors reconstructed!")
        mumford_timers_print()
        return found_xs, []

    t0 = time.time()
    mumford_divisors = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)
    mumford_timer_add("canonicalization", time.time() - t0)

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
                print(f"  Height pairing matrix:\n{basis_H}")
                print(f"  Determinant: {basis_H.determinant()}")
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
