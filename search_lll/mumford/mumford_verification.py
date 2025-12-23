from sage.all import QQ, PolynomialRing
from .mumford_core import make_monic, reduce_v_mod_u, is_divisor_on_curve
from math import isqrt
from sage.all import GF, QQ
from sage.all import QQ, PolynomialRing, HyperellipticCurve, Matrix, CDF, RealField


# canonicalize_and_dedup.py  -- replace your existing function with this
from sage.all import PolynomialRing, QQ, Integer
import math
import logging

logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.WARNING)


def _make_f_poly(R, f_coeffs):
    """
    Construct polynomial f in R (PolynomialRing(QQ,'x')) robustly.
    Accepts either coefficients high->low or low->high heuristically.
    """
    # Try treating f_coeffs as highest->lowest first
    try:
        f_try = R(list(f_coeffs))
        # degree guess: if many leading zeros then maybe reversed ordering
        if f_try.degree() >= 0:
            return f_try
    except Exception:
        raise
    # fallback: Horner building (assume high->low)
    f = R(0)
    for c in f_coeffs:
        f = f * R.gen() + QQ(c)
    return f


# patch: tolerant canonicalization for small rational scale factors
from sage.all import QQ, Integer
logger = logging.getLogger("canonicalize_and_dedup")

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4), QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# keep the same scale trials as before
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4),
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


from sage.all import QQ, PolynomialRing, GF, HyperellipticCurve, Matrix, CDF, RealField, Integer

logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4), 
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4), 
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4), 
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4), 
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


def validate_mumford_solver():
    """Simple test function (placeholder)."""
    print("Use verify_mumford_pair directly for testing.")
    return True


def quick_dependence_check(div1, div2):
    """Check if two divisors with same u are dependent"""
    if (div1['s'], div1['p']) != (div2['s'], div2['p']):
        return False  # different u
    
    # Same u - check if v1 ≡ ±v2 (mod u)
    if (div1['v_0'] == div2['v_0'] and div1['v_1'] == div2['v_1']):
        return True  # identical
    if (div1['v_0'] == -div2['v_0'] and div1['v_1'] == -div2['v_1']):
        return True  # negatives
    
    return False


def discriminant_has_nonqr_s_p(s, p, primes_nr):
    """
    Return True iff Delta = s^2 - 4*p is a quadratic NON-residue
    modulo at least one prime in primes_nr.
    """
    Delta = QQ(s) * QQ(s) - QQ(4) * QQ(p)

    if Delta == 0:
        return False  # double root => reducible

    num = int(Delta.numerator())
    den = int(Delta.denominator())

    tested_any = False

    for pr in primes_nr:
        assert pr > 2

        if den % pr == 0:
            continue

        tested_any = True

        # Check Legendre symbol
        num_mod = (num % pr) * pow(den % pr, -1, pr) % pr
        if pow(int(num_mod), (pr - 1)//2, pr) == pr - 1:
            return True

    if tested_any:
        return False

    return True

def _rational_is_square(q):
    """
    q is a QQ rational. Return (True, sqrt_QQ) if q is a rational square, else (False, None).
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


def _try_scale_and_accept(u, v, f_poly, s_q, p_q, orig_tup, seen, R, out):
    """
    Try small scale factors lam in _SCALE_TRIALS so that (lam*v)^2 - f is divisible by u.
    If successful, append canonicalized dict to out and return True.
    """
    for lam in _SCALE_TRIALS:
        try:
            v_candidate = (lam * v)
            try:
                v_candidate = v_candidate.change_ring(QQ)
            except Exception:
                pass
            
            # Note: v_candidate is already normalized for parity by caller before this check
            # if we are in the main path, but for fallback loops it might not be. 
            # However, logic in canonicalize_and_dedup applies normalize_infinity_parity 
            # usually before calling this or relies on this to find the match.
            
            if (v_candidate**2 - f_poly) % u == 0:
                key = _canon_key_from_polys(u, v_candidate)
                if key not in seen:
                    seen.add(key)
                    newt = dict(orig_tup)
                    newt['u_poly'] = u
                    newt['v_poly'] = v_candidate
                    coeffs = v_candidate.list()
                    if len(coeffs) == 0:
                        newt['v_0'] = QQ(0); newt['v_1'] = QQ(0)
                    elif len(coeffs) == 1:
                        newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(0)
                    else:
                        newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(coeffs[1])
                    newt['s'] = QQ(s_q)
                    newt['p'] = QQ(p_q)
                    newt['has_rational_roots'] = True if u.degree() == 2 and u.discriminant().is_square() else (u.degree() <= 1)
                    newt['scale_used'] = lam
                    out.append(newt)
                    logger.info("Accepted divisor s=%r p=%r with scale=%r", s_q, p_q, lam)
                return True
        except Exception:
            continue
    return False


# Set up logging once
logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4),
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


def build_curve_from_coeffs(f_coeffs):
    """Return (C, R, x) from f_coeffs."""
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    return C, R, x


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4),
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# candidate scales we try (including reciprocals)
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4),
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]

def build_f_poly(f_coeffs, R):
    """
    Build f(x) with f_coeffs given strictly highest-degree -> lowest-degree.
    Uses Horner's method to guarantee correct coefficient ordering.
    """
    x = R.gen()
    f = R(0)
    for c in f_coeffs:
        f = f * x + QQ(c)
    return f

def normalize_infinity_parity(v_poly, f_poly):
    """
    Enforce canonical infinity-parity.
    Standardizes v such that the leading coefficient is positive.
    This ensures (u, v) and (u, -v) map to the same canonical form.
    """
    if v_poly.is_zero():
        return v_poly
    
    try:
        lc = v_poly.leading_coefficient()
        if lc < 0:
            return -v_poly
        return v_poly
    except Exception:
        return v_poly

def _u_from_sp(s_q, p_q, R):
    x = R.gen()
    return x**2 - s_q * x + p_q

def _v_from_coeffs(v1_q, v0_q, R):
    x = R.gen()
    return v1_q * x + v0_q

def rational_pair_key(c):
    from sage.all import Integer
    q = QQ(c)
    a = Integer(q.numerator())
    b = Integer(q.denominator())
    return (a, b)

def _canon_key_from_polys(u, v):
    def coeff_pairs(poly):
        return tuple(rational_pair_key(c) for c in poly.list())
    return ("u", coeff_pairs(u), "v", coeff_pairs(v))

def same_v_up_to_sign_mod_u(v1, v2, u):
    diff = (v1 - v2) % u
    if diff.is_zero(): return True
    summ = (v1 + v2) % u
    if summ.is_zero(): return True
    return False

def rational_sqrt(q):
    try:
        if q < 0: return None
        a = Integer(q.numerator())
        b = Integer(q.denominator())
        if a.is_square() and b.is_square():
            return QQ(Integer(a.isqrt()) / Integer(b.isqrt()))
        return None
    except Exception:
        return None

def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=None):
    """
    Standalone verification of Mumford condition.
    Useful for unit testing specific divisors.
    """
    if modulus is None:
        R = PolynomialRing(QQ, 'x')
    else:
        R = PolynomialRing(GF(modulus), 'x')
    
    x = R.gen()
    
    if modulus is None:
        s_val = QQ(s); p_val = QQ(p); v0_val = QQ(v0); v1_val = QQ(v1)
        f_poly_coeffs = [QQ(c) for c in f_coeffs]
    else:
        s_val = int(s) % modulus; p_val = int(p) % modulus
        v0_val = int(v0) % modulus; v1_val = int(v1) % modulus
        f_poly_coeffs = [int(c) % modulus for c in f_coeffs]
    
    u_poly = x**2 - s_val*x + p_val
    v_poly = v1_val*x + v0_val
    
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    diff = v_poly**2 - f_poly
    remainder = diff % u_poly
    
    return remainder.is_zero()

def check_2torsion_difference(div1, div2, f_coeffs):
    try:
        R = PolynomialRing(QQ, 'x')
        f_poly = build_f_poly(f_coeffs, R)
        C = HyperellipticCurve(f_poly)
        J = C.jacobian()
        
        u1 = div1['u_poly']; v1 = div1['v_poly']
        D1 = J([u1, v1])
        u2 = div2['u_poly']; v2 = div2['v_poly']
        D2 = J([u2, v2])
        
        diff = D1 - D2
        if (2 * diff).is_zero():
            return True
        return False
    except Exception:
        return False

def _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, orig_tup, seen_keys, seen_u_map, out_list):
    """
    Tries scaling v_candidate. If (v_scaled^2 - f) % u == 0:
      1. Normalizes v_scaled (infinity parity)
      2. Deduplicates
      3. Saves
    """
    for lam in _SCALE_TRIALS:
        try:
            v_test = lam * v_candidate
            
            # --- CRITICAL FIX ---
            # Verify the Mumford condition BEFORE any normalization.
            # Normalization flips signs, which doesn't affect v^2 (the check),
            # but we want to ensure we found a valid root first.
            if (v_test**2 - f_poly) % u == 0:
                
                # Normalize AFTER acceptance to ensure canonical storage
                v_norm = normalize_infinity_parity(v_test, f_poly)
                
                key = _canon_key_from_polys(u, v_norm)
                if key not in seen_keys:
                    seen_keys.add(key)
                    
                    u_key = tuple(rational_pair_key(c) for c in u.list())
                    seen_u_map[u_key] = (u, v_norm)

                    newt = dict(orig_tup)
                    newt['u_poly'] = u
                    newt['v_poly'] = v_norm
                    coeffs = v_norm.list()
                    if len(coeffs) == 0:
                        newt['v_0'] = QQ(0); newt['v_1'] = QQ(0)
                    elif len(coeffs) == 1:
                        newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(0)
                    else:
                        newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(coeffs[1])
                        
                    newt['s'] = QQ(s_q)
                    newt['p'] = QQ(p_q)
                    u_disc = u.discriminant()
                    # Exact check for rational roots support
                    newt['has_rational_roots'] = True if (u.degree() <= 2 and u_disc.is_square()) else False
                    newt['scale_used'] = lam
                    
                    out_list.append(newt)
                    # Log explicitly for debugging
                    logger.info("Accepted divisor s=%r p=%r with scale=%r", s_q, p_q, lam)
                return True
        except Exception:
            continue
    return False

def canonicalize_and_dedup(divisors, f_coeffs):
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    
    # Strictly build f(x) from High->Low coefficients
    f_poly = build_f_poly(f_coeffs, R)
    
    # Sanity check for genus 2
    if f_poly.degree() not in [5, 6]:
        logger.warning(f"Warning: f_poly degree is {f_poly.degree()}, expected 5 or 6 for Genus 2.")

    seen = set()
    seen_u = dict()
    out = []
    skipped_count = 0
    accepted_count = 0

    def u_key_from_poly(u_poly):
        return tuple(rational_pair_key(c) for c in u_poly.list())

    for tup in divisors:
        try:
            s_q = QQ(tup['s'])
            p_q = QQ(tup['p'])
            v0_q = QQ(tup['v_0'])
            v1_q = QQ(tup['v_1'])
        except Exception:
            skipped_count += 1
            continue

        try:
            u = _u_from_sp(s_q, p_q, R)
            v = _v_from_coeffs(v1_q, v0_q, R)
        except Exception:
            skipped_count += 1
            continue

        # Start with reduced v, but do NOT normalize parity yet.
        try:
            v_red = v % u
        except Exception:
            skipped_count += 1
            continue

        u_k = u_key_from_poly(u)

        # 1. Check for Duplicate U
        if u_k in seen_u:
            stored_u, stored_v = seen_u[u_k]
            # If same v (up to sign), it's a duplicate.
            # If different v, it's a dependent divisor (same x-coords, different y).
            # We filter both to enforce one divisor per u support.
            if not same_v_up_to_sign_mod_u(v_red, stored_v, stored_u):
                 # Optional: Check 2-torsion difference for debug purposes only
                temp_div1 = {'u_poly': u, 'v_poly': normalize_infinity_parity(v_red, f_poly)}
                temp_div2 = {'u_poly': stored_u, 'v_poly': stored_v}
                if check_2torsion_difference(temp_div1, temp_div2, f_coeffs):
                     logger.debug("Skipping divisor differing by 2-torsion: s=%r p=%r", s_q, p_q)
            
            skipped_count += 1
            continue

        # 2. Strategy 1: Direct Scaling
        if _attempt_scale_and_save(u, v_red, f_poly, s_q, p_q, tup, seen, seen_u, out):
            accepted_count += 1
            continue

        # Discriminant for Root Strategies
        disc = s_q * s_q - 4 * p_q
        disc_sqrt = rational_sqrt(disc) if disc != 0 else QQ(0)

        # 3. Strategy 2: Double Root (u = (x-r)^2)
        if disc_sqrt is not None and disc_sqrt == 0:
            r_double = s_q / QQ(2)
            fr = f_poly(r_double)
            sqrt_fr = rational_sqrt(fr)
            
            if sqrt_fr is not None:
                vr = v_red(r_double)
                if vr != 0:
                    lam_candidate = QQ(sqrt_fr) / QQ(vr)
                    v_scaled = lam_candidate * v_red
                    if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out):
                        accepted_count += 1
                        continue
            continue

        # 4. Strategy 3: Split Roots (u = (x-r1)(x-r2))
        if disc_sqrt is not None:
            r_plus = (s_q + disc_sqrt) / QQ(2)
            r_minus = (s_q - disc_sqrt) / QQ(2)
            denom = r_plus - r_minus
            
            if denom != 0:
                fa_plus = f_poly(r_plus)
                fa_minus = f_poly(r_minus)
                sqrt_plus = rational_sqrt(fa_plus)
                sqrt_minus = rational_sqrt(fa_minus)
                
                if sqrt_plus is not None and sqrt_minus is not None:
                    # A. Try scaling existing v
                    vr_plus = v_red(r_plus)
                    if vr_plus != 0:
                        tried_scale = False
                        for target in (sqrt_plus, -sqrt_plus):
                            lam = QQ(target) / QQ(vr_plus)
                            v_scaled = lam * v_red
                            if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out):
                                accepted_count += 1
                                tried_scale = True
                                break
                        if tried_scale:
                            continue

                    # B. Interpolate new v
                    matched = False
                    for sig_plus in (+1, -1):
                        for sig_minus in (+1, -1):
                            y_plus = QQ(sig_plus) * sqrt_plus
                            y_minus = QQ(sig_minus) * sqrt_minus
                            
                            alpha = (y_plus - y_minus) / denom
                            beta = y_plus - alpha * r_plus
                            v_candidate = alpha * x + beta
                            
                            if _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, tup, seen, seen_u, out):
                                accepted_count += 1
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        continue

        skipped_count += 1

    logger.info("canonicalize_and_dedup: accepted=%d skipped=%d", len(out), skipped_count)
    return out
