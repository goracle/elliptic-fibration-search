from sage.all import QQ, PolynomialRing
from .mumford_core import make_monic, reduce_v_mod_u, is_divisor_on_curve
from math import isqrt
from sage.all import GF, QQ
from sage.all import QQ, PolynomialRing, HyperellipticCurve, Matrix, CDF, RealField

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

def validate_mumford_solver():
    """Simple test function (placeholder)."""
    print("Use verify_mumford_pair directly for testing.")
    return True


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


def discriminant_has_nonqr_s_p(s, p, primes_nr):
    """
    Return True iff Delta = s^2 - 4*p is a quadratic NON-residue
    modulo at least one prime in primes_nr.
    If no primes are testable (all divide denominator), do NOT reject.
    """
    Delta = QQ(s) * QQ(s) - QQ(4) * QQ(p)
    assert Delta in QQ

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

        num_mod = (num % pr) * pow(den % pr, -1, pr) % pr
        if pow(int(num_mod), (pr - 1)//2, pr) == pr - 1:
            return True

    # If we tested primes and found only residues → reject
    if tested_any:
        return False

    # If no primes were testable, do NOT reject
    return True

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
        pass
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

def rational_sqrt(q):
    """
    If q in QQ is an exact rational square, return its QQ square root.
    Otherwise return None.
    """
    try:
        if q < 0:
            return None
        a = Integer(q.numerator())
        b = Integer(q.denominator())
        if a.is_square() and b.is_square():
            # exact square roots
            return QQ(Integer(a.isqrt()) / Integer(b.isqrt()))
        return None
    except Exception:
        return None

def build_f_poly(f_coeffs, R):
    """
    Build f(x) with f_coeffs given highest-degree -> constant.
    """
    x = R.gen()
    f = R(0)
    for c in f_coeffs:
        f = f * x + QQ(c)
    return f

def _u_from_sp(s_q, p_q, R):
    x = R.gen()
    return x**2 - s_q * x + p_q

def _v_from_coeffs(v1_q, v0_q, R):
    x = R.gen()
    return v1_q * x + v0_q


def check_2torsion_difference(div1, div2, f_coeffs):
    """
    Check if div1 - div2 is a 2-torsion element.
    Returns True if they differ by 2-torsion (should be filtered).
    """
    try:
        # Build curve and Jacobian
        C, R, x = build_curve_from_coeffs(f_coeffs)
        J = C.jacobian()
        
        # Convert both to Jacobian elements
        u1 = div1['u_poly']
        v1 = div1['v_poly']
        D1 = J([u1, v1])
        
        u2 = div2['u_poly']
        v2 = div2['v_poly']
        D2 = J([u2, v2])
        
        # Compute difference
        diff = D1 - D2
        
        # Check if 2*diff = 0
        if (2 * diff).is_zero():
            return True
            
        return False
        
    except Exception:
        # If check fails, conservatively don't filter
        raise
        return False


def build_curve_from_coeffs(f_coeffs):
    """Return (C, R, x) from f_coeffs (list-like of coefficients highest->lowest)."""
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    return C, R, x


logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# keep the same scale trials as before
_SCALE_TRIALS = [QQ(1), QQ(2), QQ(4), QQ(-1), QQ(-2), QQ(-4),
                 QQ(1)/QQ(2), QQ(1)/QQ(4), QQ(-1)/QQ(2), QQ(-1)/QQ(4)]


def normalize_infinity_parity(v_poly, f_poly):
    """
    Enforce a canonical infinity-parity for even-degree hyperelliptic curves.

    For deg(f) even (e.g. 6), the expected "degree" of v in the local parameter at infinity
    is d/2 - 1. We use the coefficient of x^(d/2 - 1) in v(x) as the parity indicator.
    If that coefficient is negative, flip v -> -v. If it's zero, fall back deterministically
    to the next highest coefficient (and finally the constant term) so the choice is stable.

    This forces all accepted divisors into a single infinity parity coset.
    """
    try:
        d = int(f_poly.degree())
    except Exception:
        return v_poly

    if d % 2 != 0:
        return v_poly

    deg_v_expected = d // 2 - 1

    coeffs = list(v_poly.list()) if hasattr(v_poly, "list") else []
    
    def get_coeff_at(idx):
        if idx < 0:
            return QQ(0)
        if idx < len(coeffs):
            return QQ(coeffs[idx])
        return QQ(0)

    sign_coeff = None
    for idx in range(deg_v_expected, -1, -1):
        c = get_coeff_at(idx)
        if c != 0:
            sign_coeff = c
            break
    if sign_coeff is None:
        for idx in range(deg_v_expected + 1, deg_v_expected + 4):
            c = get_coeff_at(idx)
            if c != 0:
                sign_coeff = c
                break

    if sign_coeff is None:
        return v_poly

    if sign_coeff < 0:
        try:
            v_poly = -v_poly
        except Exception:
            pass

    return v_poly


def _try_scale_and_accept(u, v, f_poly, s_q, p_q, orig_tup, seen, R, out):
    """
    Try small scale factors lam in _SCALE_TRIALS so that (lam*v)^2 - f is divisible by u.
    If successful, append canonicalized dict to out and return True.
    """
    x = R.gen()
    for lam in _SCALE_TRIALS:
        try:
            v_candidate = (lam * v)
            try:
                v_candidate = v_candidate.change_ring(QQ)
            except Exception:
                pass
            
            try:
                v_candidate = normalize_infinity_parity(v_candidate, f_poly)
            except Exception:
                pass
            
            if (v_candidate**2 - f_poly) % u == 0:
                coeffs = v_candidate.list()
                v_lead = QQ(coeffs[-1]) if len(coeffs) > 0 else QQ(0)
                v_const = QQ(coeffs[0]) if len(coeffs) > 0 else QQ(0)
                if v_lead < 0 or (v_lead == 0 and v_const < 0):
                    v_candidate = -v_candidate
                    lam = -lam
                    coeffs = v_candidate.list()
                    v_lead = QQ(coeffs[-1]) if len(coeffs) > 0 else QQ(0)
                    v_const = QQ(coeffs[0]) if len(coeffs) > 0 else QQ(0)
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


def rational_pair_key(c):
    from sage.all import gcd, Integer
    q = QQ(c)
    a = Integer(q.numerator())
    b = Integer(q.denominator())
    g = gcd(a, b)
    return (a//g, b//g)


def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Canonicalize and deduplicate Mumford (s,p,v0,v1) reconstructions.
    Enforces: at most ONE divisor per unique u(x), modulo v ↦ ±v (mod u).
    Also applies infinity parity normalization for even-degree curves.
    """
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = build_f_poly(f_coeffs, R)

    seen = set()
    seen_u = dict()
    out = []
    skipped_examples = []
    accepted_count = 0
    skipped_count = 0

    def u_key_from_poly(u_poly):
        pairs = tuple(rational_pair_key(c) for c in u_poly.list())
        return pairs

    def same_v_up_to_sign_mod_u(v_a, v_b, u_poly):
        try:
            ra = v_a % u_poly
            rb = v_b % u_poly
            diff = (ra - rb) % u_poly
            ssum = (ra + rb) % u_poly
            return diff.is_zero() or ssum.is_zero()
        except Exception:
            return False

    def finalize_v_and_normalize(u_poly, v_poly):
        """
        Normalize sign of v_poly (assumes v_poly is already reduced mod u_poly).
        """
        v_red = v_poly
        try:
            v_red = v_red.change_ring(QQ)
        except Exception:
            pass
        coeffs = v_red.list()
        if len(coeffs) == 0:
            v_lead = QQ(0); v_const = QQ(0)
        else:
            v_lead = QQ(coeffs[-1])
            v_const = QQ(coeffs[0])
        if v_lead < 0 or (v_lead == 0 and v_const < 0):
            v_red = -v_red
        return v_red

    def local_try_accept(u_poly, v_poly, s_q, p_q, orig_tup):
        nonlocal accepted_count, skipped_count

        try:
            v_red = v_poly % u_poly
        except Exception:
            skipped_count += 1
            return False

        v_norm = finalize_v_and_normalize(u_poly, v_red)

        v_norm = normalize_infinity_parity(v_norm, f_poly)

        u_k = u_key_from_poly(u_poly)

        if u_k in seen_u:
            stored_u, stored_v = seen_u[u_k]
            if same_v_up_to_sign_mod_u(v_norm, stored_v, stored_u):
                return False
            else:
                skipped_count += 1
                return False

        accepted = _try_scale_and_accept(
            u_poly, v_norm, f_poly,
            s_q, p_q, orig_tup,
            seen, R, out
        )

        if accepted:
            seen_u[u_k] = (u_poly, v_norm)
            accepted_count += 1
            return True

        return False

    for tup in divisors:
        try:
            s_raw = tup['s']; p_raw = tup['p']; v0_raw = tup['v_0']; v1_raw = tup['v_1']
        except Exception:
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("malformed", tup))
            continue

        try:
            s_q = QQ(s_raw); p_q = QQ(p_raw); v0_q = QQ(v0_raw); v1_q = QQ(v1_raw)
        except Exception:
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("nonrational", (s_raw, p_raw, v0_raw, v1_raw)))
            continue

        try:
            u = _u_from_sp(s_q, p_q, R)
            v = _v_from_coeffs(v1_q, v0_q, R)
        except Exception:
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("poly_build_fail", (s_q, p_q, v1_q, v0_q)))
            continue

        if local_try_accept(u, v, s_q, p_q, tup):
            continue

        disc = s_q * s_q - 4 * p_q
        disc_sqrt = None
        if disc == 0:
            disc_sqrt = QQ(0)
        else:
            disc_sqrt = rational_sqrt(disc)

        if disc_sqrt is not None and disc_sqrt == 0:
            r_double = s_q / QQ(2)
            fr = f_poly(r_double)
            sqrt_fr = rational_sqrt(fr)
            if sqrt_fr is None:
                skipped_count += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(("double_non_square", (s_q, p_q, r_double, fr)))
                logger.debug("Double-root but f(r) not square: s=%r p=%r r=%r f(r)=%r; skipping", s_q, p_q, r_double, fr)
                continue

            v_temp = v % u
            v_temp = finalize_v_and_normalize(u, v_temp)
            vr = v_temp(r_double)

            if vr != 0:
                lam_candidate = QQ(sqrt_fr) / QQ(vr)
                if lam_candidate in _SCALE_TRIALS:
                    try:
                        v_scaled = lam_candidate * v
                        if local_try_accept(u, v_scaled, s_q, p_q, tup):
                            continue
                    except Exception:
                        pass

            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("double_unmatched", (s_q, p_q, vr, sqrt_fr)))
            logger.debug("Double-root and v(r) != ±sqrt(f(r)) after scaling attempts: s=%r p=%r vr=%r sqrt_fr=%r; skipping", s_q, p_q, vr, sqrt_fr)
            continue

        if disc_sqrt is not None:
            r_plus = (s_q + disc_sqrt) / QQ(2)
            r_minus = (s_q - disc_sqrt) / QQ(2)
            denom = r_plus - r_minus
            if denom == 0:
                skipped_count += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(("split_zero_denom", (s_q, p_q)))
                logger.debug("Denominator zero in split-case interpolation; skipping: s=%r p=%r", s_q, p_q)
                continue

            fa_plus = f_poly(r_plus)
            fa_minus = f_poly(r_minus)
            sqrt_plus = rational_sqrt(fa_plus)
            sqrt_minus = rational_sqrt(fa_minus)
            if sqrt_plus is None or sqrt_minus is None:
                skipped_count += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(("root_not_square", (s_q, p_q, fa_plus, fa_minus)))
                logger.debug("Root values not rational squares: f(r+)=%r f(r-)=%r ; skipping", fa_plus, fa_minus)
                continue

            v_temp = v % u
            v_temp = finalize_v_and_normalize(u, v_temp)
            vr_plus = v_temp(r_plus)
            vr_minus = v_temp(r_minus)

            tried_scale = False
            if vr_plus != 0:
                for target in (sqrt_plus, -sqrt_plus):
                    lam_candidate = QQ(target) / QQ(vr_plus)
                    if lam_candidate in _SCALE_TRIALS:
                        try:
                            if (lam_candidate * vr_minus == sqrt_minus) or (lam_candidate * vr_minus == -sqrt_minus):
                                v_scaled = lam_candidate * v
                                if local_try_accept(u, v_scaled, s_q, p_q, tup):
                                    tried_scale = True
                                    break
                        except Exception:
                            pass
                if tried_scale:
                    continue

            matched = False
            for sig_plus in (+1, -1):
                for sig_minus in (+1, -1):
                    num = (QQ(sig_plus) * sqrt_plus) - (QQ(sig_minus) * sqrt_minus)
                    alpha = num / denom
                    beta = (QQ(sig_plus) * sqrt_plus) - alpha * r_plus
                    v_candidate = alpha * x + beta
                    try:
                        if local_try_accept(u, v_candidate, s_q, p_q, tup):
                            matched = True
                            break
                    except Exception:
                        continue
                if matched:
                    break
            if matched:
                continue

            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("split_unmatched", (s_q, p_q, vr_plus, vr_minus)))
            logger.debug("Split-case canonicalization failed for s=%r p=%r; skipping", s_q, p_q)
            continue

        accepted = False
        for lam in _SCALE_TRIALS:
            try:
                v_scaled = lam * v
                if local_try_accept(u, v_scaled, s_q, p_q, tup):
                    accepted = True
                    break
            except Exception:
                continue
        if accepted:
            continue

        skipped_count += 1
        if len(skipped_examples) < 10:
            skipped_examples.append(("irr_unmatched", (s_q, p_q, v1_q, v0_q)))
        logger.debug("Irreducible-case failed for s=%r p=%r; skipping", s_q, p_q)

    logger.info("canonicalize_and_dedup: accepted=%d skipped=%d total_input=%d", len(out), skipped_count, len(divisors))
    if skipped_examples:
        logger.info("Sample skipped cases (up to 10): %r", skipped_examples)

    for i in range(len(out)):
        for j in range(i):
            assert out[i]['u_poly'] != out[j]['u_poly'] or not same_v_up_to_sign_mod_u(out[i]['v_poly'], out[j]['v_poly'], out[i]['u_poly']), "duplicate v for same u slipped through"

    if f_poly.degree() % 2 == 0:
        for d in out:
            vcheck = normalize_infinity_parity(d['v_poly'], f_poly)
            assert vcheck == d['v_poly'], "parity not normalized for accepted v"

    return out


def _canon_key_from_polys(u, v):
    def coeff_pairs(poly):
        return tuple(rational_pair_key(c) for c in poly.list())
    return ("u", coeff_pairs(u), "v", coeff_pairs(v))

