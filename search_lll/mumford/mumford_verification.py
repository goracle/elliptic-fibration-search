from sage.all import QQ, PolynomialRing
from .mumford_core import make_monic, reduce_v_mod_u, is_divisor_on_curve
from math import isqrt
from sage.all import GF, QQ

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

def _canon_key_from_polys(u, v):
    """
    Canonical key for deduplication: coefficient pairs (num,den) low->high for u and v.
    """
    def coeff_pairs(poly):
        # Sage poly.list() returns coefficients low->high
        pairs = tuple((int(QQ(c).numerator()), int(QQ(c).denominator())) for c in poly.list())
        return pairs
    return ("u", coeff_pairs(u), "v", coeff_pairs(v))

def _try_scale_and_accept(u, v, f_poly, s_q, p_q, orig_tup, seen, R, out):
    """
    Try small scale factors lam in _SCALE_TRIALS so that (lam*v)^2 - f is divisible by u.
    If successful, append canonicalized dict to out and return True.
    """
    x = R.gen()
    for lam in _SCALE_TRIALS:
        try:
            v_candidate = (lam * v)
            # Ensure coefficients in QQ
            try:
                v_candidate = v_candidate.change_ring(QQ)
            except Exception:
                pass
            # exact Mumford relation?
            if (v_candidate**2 - f_poly) % u == 0:
                # normalize sign: prefer leading coeff >= 0 (tie-break on const)
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
                    newt = dict(orig_tup)  # shallow copy original metadata
                    newt['u_poly'] = u
                    newt['v_poly'] = v_candidate
                    # Ensure v_0 and v_1 are stored as QQ
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
            # on any algebraic failure, continue trying other scales
            continue
    return False

def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Canonicalize and deduplicate Mumford (s,p,v0,v1) reconstructions.

    Returns list of canonicalized dicts with keys:
      'u_poly', 'v_poly', 's', 'p', 'v_0', 'v_1', 'has_rational_roots', and optional 'scale_used'
    """
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = build_f_poly(f_coeffs, R)

    seen = set()
    out = []
    skipped_examples = []
    accepted_count = 0
    skipped_count = 0

    for tup in divisors:
        # expected fields
        try:
            s_raw = tup['s']; p_raw = tup['p']; v0_raw = tup['v_0']; v1_raw = tup['v_1']
        except Exception:
            # malformed input: skip but record example
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("malformed", tup))
            continue

        # coerce
        try:
            s_q = QQ(s_raw); p_q = QQ(p_raw); v0_q = QQ(v0_raw); v1_q = QQ(v1_raw)
        except Exception:
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("nonrational", (s_raw, p_raw, v0_raw, v1_raw)))
            continue

        # build u, v polynomials
        try:
            u = _u_from_sp(s_q, p_q, R)
            v = _v_from_coeffs(v1_q, v0_q, R)
        except Exception:
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("poly_build_fail", (s_q, p_q, v1_q, v0_q)))
            continue

        # quick direct check: does current v already satisfy Mumford relation exactly?
        try:
            if (v**2 - f_poly) % u == 0:
                # accept with optional scale=1
                accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                if accepted:
                    accepted_count += 1
                    continue
                # if not accepted (shouldn't happen), fall through to attempts
        except Exception:
            # proceed to further logic
            pass

        # compute discriminant of u to decide branch
        disc = s_q * s_q - 4 * p_q
        disc_sqrt = None
        if disc == 0:
            disc_sqrt = QQ(0)
        else:
            disc_sqrt = rational_sqrt(disc)

        # HANDLE DOUBLE-ROOT (disc == 0)
        if disc_sqrt is not None and disc_sqrt == 0:
            r_double = s_q / QQ(2)
            vr = v1_q * r_double + v0_q
            fr = f_poly(r_double)
            sqrt_fr = rational_sqrt(fr)
            if sqrt_fr is None:
                skipped_count += 1
                if len(skipped_examples) < 10:
                    skipped_examples.append(("double_non_square", (s_q, p_q, r_double, fr)))
                logger.debug("Double-root but f(r) not square: s=%r p=%r r=%r f(r)=%r; skipping", s_q, p_q, r_double, fr)
                continue

            # try direct ± match or small scaling via _try_scale_and_accept
            # First see if current v matches ±sqrt_fr at root (possibly sign-flipped)
            if vr == sqrt_fr or vr == -sqrt_fr:
                accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                if accepted:
                    accepted_count += 1
                    continue

            # try scale lam = sqrt_fr / vr when vr != 0 and lam is small (in trials)
            if vr != 0:
                lam_candidate = QQ(sqrt_fr) / QQ(vr)
                if lam_candidate in _SCALE_TRIALS:
                    # attempt acceptance with lam_candidate
                    try:
                        v_scaled = lam_candidate * v
                        if (v_scaled**2 - f_poly) % u == 0:
                            accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                            if accepted:
                                accepted_count += 1
                                continue
                    except Exception:
                        pass

            # nothing worked
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("double_unmatched", (s_q, p_q, vr, sqrt_fr)))
            logger.debug("Double-root and v(r) != ±sqrt(f(r)) after scaling attempts: s=%r p=%r vr=%r sqrt_fr=%r; skipping", s_q, p_q, vr, sqrt_fr)
            continue

        # HANDLE SPLIT-ROOT (disc is a nonzero rational square)
        if disc_sqrt is not None:
            r_plus = (s_q + disc_sqrt) / QQ(2)
            r_minus = (s_q - disc_sqrt) / QQ(2)
            # avoid numerical denom=0 just in case (shouldn't happen when disc_sqrt != 0)
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

            vr_plus = v1_q * r_plus + v0_q
            vr_minus = v1_q * r_minus + v0_q

            # quick exact match check (±)
            if (vr_plus == sqrt_plus and vr_minus == sqrt_minus) or (vr_plus == -sqrt_plus and vr_minus == -sqrt_minus):
                accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                if accepted:
                    accepted_count += 1
                    continue

            # try small scaling candidates derived from first root and check second
            tried_scale = False
            if vr_plus != 0:
                for target in (sqrt_plus, -sqrt_plus):
                    lam_candidate = QQ(target) / QQ(vr_plus)
                    if lam_candidate in _SCALE_TRIALS:
                        # check if lam_candidate makes second root match ±sqrt
                        if (lam_candidate * vr_minus == sqrt_minus) or (lam_candidate * vr_minus == -sqrt_minus):
                            # test algebraic divisibility
                            try:
                                v_scaled = lam_candidate * v
                                if (v_scaled**2 - f_poly) % u == 0:
                                    accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                                    if accepted:
                                        accepted_count += 1
                                        tried_scale = True
                                        break
                            except Exception:
                                pass
                if tried_scale:
                    continue

            # fallback: attempt to interpolate linear v that matches ±sqrt values (all sign combos)
            matched = False
            for sig_plus in (+1, -1):
                for sig_minus in (+1, -1):
                    num = (QQ(sig_plus) * sqrt_plus) - (QQ(sig_minus) * sqrt_minus)
                    alpha = num / denom
                    beta = (QQ(sig_plus) * sqrt_plus) - alpha * r_plus
                    v_candidate = alpha * x + beta
                    try:
                        if (v_candidate**2 - f_poly) % u == 0:
                            # acceptance tries scales inside _try_scale_and_accept
                            accepted = _try_scale_and_accept(u, v_candidate, f_poly, s_q, p_q, tup, seen, R, out)
                            if accepted:
                                accepted_count += 1
                                matched = True
                                break
                    except Exception:
                        continue
                if matched:
                    break
            if matched:
                continue

            # nothing accepted
            skipped_count += 1
            if len(skipped_examples) < 10:
                skipped_examples.append(("split_unmatched", (s_q, p_q, vr_plus, vr_minus)))
            logger.debug("Split-case canonicalization failed for s=%r p=%r; skipping", s_q, p_q)
            continue

        # IRREDUCIBLE CASE (non-square discriminant)
        # Try negating v (global sign) and small scalings to see if we can match Mumford relation
        accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
        if accepted:
            accepted_count += 1
            continue
        # try negated v
        accepted = _try_scale_and_accept(u, -v, f_poly, s_q, p_q, tup, seen, R, out)
        if accepted:
            accepted_count += 1
            continue
        # finally try small lambda multiples if above didn't pick them up (redundant but safe)
        for lam in _SCALE_TRIALS:
            try:
                v_scaled = lam * v
                if (v_scaled**2 - f_poly) % u == 0:
                    accepted = _try_scale_and_accept(u, v, f_poly, s_q, p_q, tup, seen, R, out)
                    if accepted:
                        accepted_count += 1
                        break
            except Exception:
                continue
        if accepted:
            continue

        # nothing worked for irreducible: skip
        skipped_count += 1
        if len(skipped_examples) < 10:
            skipped_examples.append(("irr_unmatched", (s_q, p_q, v1_q, v0_q)))
        logger.debug("Irreducible-case failed for s=%r p=%r; skipping", s_q, p_q)
        continue

    # summary logging
    logger.info("canonicalize_and_dedup: accepted=%d skipped=%d total_input=%d", len(out), skipped_count, len(divisors))
    if skipped_examples:
        logger.info("Sample skipped cases (up to 10): %r", skipped_examples)

    return out
