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


# canonicalize_and_dedup.py  -- improved infinity-aware canonicalization

logger = logging.getLogger("canonicalize_and_dedup")
logger.setLevel(logging.INFO)

# small rational scales to try
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
    """
    Check whether v1 == +/- v2 (mod u). Works on reduced representatives.
    """
    diff = (v1 - v2) % u
    if diff.is_zero(): return True
    summ = (v1 + v2) % u
    if summ.is_zero(): return True
    return False

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

def rational_sqrt(q):
    try:
        q = QQ(q)
        if q < 0:
            return None
        a = int(q.numerator()); b = int(q.denominator())
        # integer square test
        sa = isqrt(a); sb = isqrt(b)
        if sa * sa == a and sb * sb == b:
            return QQ(sa) / QQ(sb)
        return None
    except Exception:
        return None

def normalize_infinity_parity(v_poly, f_poly):
    """
    Enforce canonical infinity-parity. For genus-2 (deg f 5 or 6) this
    chooses the sign so that the coefficient of x^(g-1) is nonnegative.

    We intentionally only flip sign here (i.e. v -> -v) so we don't destroy
    other structure. The deeper 'infinity translation' collapse is handled
    by the Jacobian-based dedup step below.
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

def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=None):
    """
    Standalone verification of Mumford condition using the same poly-build as
    canonicalize_and_dedup. Returns True iff v^2 - f is divisible by u.
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

def _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, orig_tup, seen_keys, seen_u_map, out_list, C=None, J=None, jac_points=None):
    """
    Try lam in _SCALE_TRIALS: if (lam*v)^2 - f is divisible by u,
    normalize and (heuristically) deduplicate using Mumford key + Jacobian heuristics.
    If C and J are provided, compute jacobian points to allow cross-u deduplication.
    """
    for lam in _SCALE_TRIALS:
        try:
            v_test = lam * v_candidate

            # verify Mumford condition BEFORE normalization
            if (v_test**2 - f_poly) % u == 0:
                # normalize parity for storage
                v_norm = normalize_infinity_parity(v_test, f_poly)

                # canonical key on polynomials
                key = _canon_key_from_polys(u, v_norm)
                if key in seen_keys:
                    return True  # already present

                # If we have a Jacobian available, try to deduplicate across u's
                if J is not None and jac_points is not None and C is not None:
                    try:
                        # Build jacobian point (use explicit constructor)
                        # The J constructor accepts a Mumford pair [u,v] in sage
                        newJ = J([u, v_norm])
                        # Try to match newJ against previous saved points up to small multiples
                        # of the "infinity generator" (heuristic).
                        # We approximate an 'infinity generator' D_inf once (see below).
                        D_inf = jac_points.get('__D_inf__', None)
                        if D_inf is None:
                            # Heuristic: try to create an obvious infinity-like element.
                            # For even-degree curves, try J([x,0]) as a generator probe.
                            try:
                                R = u.parent()
                                x = R.gen()
                                D_inf = J([x, R(0)])  # heuristic probe
                                jac_points['__D_inf__'] = D_inf
                            except Exception:
                                D_inf = None

                        # Try to match against stored points
                        for stored_key, stored_data in jac_points.items():
                            if stored_key == '__D_inf__':
                                continue
                            storedJ = stored_data['Jpt']
                            diff = newJ - storedJ
                            # test small multiples
                            matched = False
                            if D_inf is not None:
                                for k in range(-3, 4):
                                    try:
                                        if (diff - k * D_inf).is_zero():
                                            logger.info("Jacobian dedup: matched new divisor to stored key %s with k=%d", stored_key, k)
                                            matched = True
                                            break
                                    except Exception:
                                        # ignore arithmetic issues in heuristics
                                        continue
                            else:
                                # If no D_inf, still check exact equality
                                try:
                                    if diff.is_zero():
                                        logger.info("Jacobian dedup: exact equality match for stored key %s", stored_key)
                                        matched = True
                                except Exception:
                                    matched = False
                            if matched:
                                # mark canonical key seen and skip actual storage (we regard as duplicate)
                                seen_keys.add(key)
                                return True
                    except Exception:
                        # Fail softly; fallback to polynomial-only dedup
                        pass

                # store canonical representation
                seen_keys.add(key)
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
                newt['has_rational_roots'] = True if (u.degree() <= 2 and u_disc.is_square()) else False
                newt['scale_used'] = lam

                out_list.append(newt)

                # If we have a Jacobian store, also keep this point for cross-checks
                if J is not None and jac_points is not None and C is not None:
                    try:
                        newJ = J([u, v_norm])
                        # make a stable key from u,v for jac_points index
                        jkey = _canon_key_from_polys(u, v_norm)
                        jac_points[jkey] = {'Jpt': newJ, 'u': u, 'v': v_norm}
                    except Exception:
                        pass

                logger.info("Accepted divisor s=%r p=%r with scale=%r", s_q, p_q, lam)
                return True
        except Exception:
            continue
    return False

def check_2torsion_difference(div1, div2, f_coeffs):
    """
    Optional heavy check: see whether 2*(D1-D2) == 0 in J(Q).
    This is a diagnostic only and can be slow. Returns True if difference is 2-torsion.
    """
    try:
        R = PolynomialRing(QQ, 'x')
        f_poly = build_f_poly(f_coeffs, R)
        C = HyperellipticCurve(f_poly)
        J = C.jacobian()
        u1 = div1['u_poly']; v1 = div1['v_poly']
        u2 = div2['u_poly']; v2 = div2['v_poly']
        D1 = J([u1, v1])
        D2 = J([u2, v2])
        diff = D1 - D2
        if (2 * diff).is_zero():
            return True
        return False
    except Exception:
        return False

def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Main entry point: returns canonicalized, deduplicated divisors (list of dicts).
    Uses polynomial acceptance logic and a heuristic Jacobian-based deduplication
    step to collapse divisors differing by small multiples of an infinity-like generator.
    """
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = build_f_poly(f_coeffs, R)

    # Build curve and Jacobian for cross-u checks (heuristic)
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    seen = set()      # polynomial canon keys
    seen_u = dict()   # map u_key -> (u, v_stored)
    out = []
    skipped_count = 0
    accepted_count = 0

    # jac_points stores known J points keyed by polynomial key; includes '__D_inf__' probe
    jac_points = {}

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

        # reduced v modulo u (we keep reduced rep for checks)
        try:
            v_red = v % u
        except Exception:
            skipped_count += 1
            continue

        u_k = u_key_from_poly(u)

        # If we've seen same u already, collapse to single representative (one divisor per u)
        if u_k in seen_u:
            stored_u, stored_v = seen_u[u_k]
            # if same up to sign, it's duplicate; otherwise we regard it as dependent and skip
            if same_v_up_to_sign_mod_u(v_red, stored_v, stored_u):
                skipped_count += 1
                continue
            else:
                # Record that there was another v for same u but skip storing both to avoid
                # multiple representatives with identical support.
                skipped_count += 1
                continue

        # Strategy 1: direct scaling attempt with Jacobian-aware dedup
        if _attempt_scale_and_save(u, v_red, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
            # keep the canonical u->v mapping for quick same-u collapse
            if len(out) > 0:
                last = out[-1]
                seen_u[u_k] = (u, last['v_poly'])
            accepted_count += 1
            continue

        # If direct scaling failed, attempt other strategies (double root, split roots)
        disc = s_q * s_q - 4 * p_q
        disc_sqrt = rational_sqrt(disc) if disc != 0 else QQ(0)

        # Double root case: u = (x - r)^2
        if disc_sqrt is not None and disc_sqrt == 0:
            r_double = s_q / QQ(2)
            fr = f_poly(r_double)
            sqrt_fr = rational_sqrt(fr)
            if sqrt_fr is not None:
                vr = v_red(r_double)
                if vr != 0:
                    lam_candidate = QQ(sqrt_fr) / QQ(vr)
                    v_scaled = lam_candidate * v_red
                    if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                        if len(out) > 0:
                            last = out[-1]
                            seen_u[u_k] = (u, last['v_poly'])
                        accepted_count += 1
                        continue
            skipped_count += 1
            continue

        # Split roots: u = (x - r1)(x - r2)
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
                    # A. Try scaling existing v value to match y at roots
                    vr_plus = v_red(r_plus)
                    if vr_plus != 0:
                        tried_scale = False
                        for target in (sqrt_plus, -sqrt_plus):
                            lam = QQ(target) / QQ(vr_plus)
                            v_scaled = lam * v_red
                            if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                                if len(out) > 0:
                                    last = out[-1]
                                    seen_u[u_k] = (u, last['v_poly'])
                                accepted_count += 1
                                tried_scale = True
                                break
                        if tried_scale:
                            continue

                    # B. Interpolate a fresh v that matches signs of sqrt at roots
                    matched = False
                    for sig_plus in (+1, -1):
                        for sig_minus in (+1, -1):
                            y_plus = QQ(sig_plus) * sqrt_plus
                            y_minus = QQ(sig_minus) * sqrt_minus
                            alpha = (y_plus - y_minus) / denom
                            beta = y_plus - alpha * r_plus
                            v_candidate = alpha * x + beta
                            if _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                                if len(out) > 0:
                                    last = out[-1]
                                    seen_u[u_k] = (u, last['v_poly'])
                                accepted_count += 1
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        continue

        # Give up on this tuple
        skipped_count += 1

    logger.info("canonicalize_and_dedup: accepted=%d skipped=%d", len(out), skipped_count)
    return out


from sage.all import QQ, PolynomialRing, HyperellipticCurve, Integer
from .mumford_core import make_monic, reduce_v_mod_u, is_divisor_on_curve
from math import isqrt
import logging

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

def _u_from_sp(s_q, p_q, R):
    x = R.gen()
    return x**2 - s_q * x + p_q

def _v_from_coeffs(v1_q, v0_q, R):
    x = R.gen()
    return v1_q * x + v0_q

def rational_pair_key(c):
    q = QQ(c)
    a = Integer(q.numerator())
    b = Integer(q.denominator())
    return (a, b)

def _canon_key_from_polys(u, v):
    def coeff_pairs(poly):
        return tuple(rational_pair_key(c) for c in poly.list())
    return ("u", coeff_pairs(u), "v", coeff_pairs(v))

def same_v_up_to_sign_mod_u(v1, v2, u):
    """
    Check whether v1 == +/- v2 (mod u). Works on reduced representatives.
    """
    diff = (v1 - v2) % u
    if diff.is_zero(): return True
    summ = (v1 + v2) % u
    if summ.is_zero(): return True
    return False

def rational_sqrt(q):
    q = QQ(q)
    if q < 0:
        return None
    a = int(q.numerator())
    b = int(q.denominator())
    sa = isqrt(a)
    sb = isqrt(b)
    if sa * sa == a and sb * sb == b:
        return QQ(sa) / QQ(sb)
    return None

def normalize_infinity_parity(v_poly, f_poly):
    """
    Enforce canonical infinity-parity. For genus-2 (deg f 5 or 6) this
    chooses the sign so that the coefficient of x^(g-1) is nonnegative.
    """
    if v_poly.is_zero():
        return v_poly
    lc = v_poly.leading_coefficient()
    if lc < 0:
        return -v_poly
    return v_poly

def _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, orig_tup, seen_keys, seen_u_map, out_list, C=None, J=None, jac_points=None):
    """
    Try lam in _SCALE_TRIALS: if (lam*v)^2 - f is divisible by u,
    normalize and (heuristically) deduplicate using Mumford key + Jacobian heuristics.
    """
    for lam in _SCALE_TRIALS:
        v_test = lam * v_candidate

        if (v_test**2 - f_poly) % u == 0:
            v_norm = normalize_infinity_parity(v_test, f_poly)

            key = _canon_key_from_polys(u, v_norm)
            if key in seen_keys:
                return True

            if J is not None and jac_points is not None and C is not None:
                try:
                    newJ = J([u, v_norm])
                    D_inf = jac_points.get('__D_inf__', None)
                    if D_inf is None:
                        R = u.parent()
                        x = R.gen()
                        D_inf = J([x, R(0)])
                        jac_points['__D_inf__'] = D_inf

                    for stored_key, stored_data in jac_points.items():
                        if stored_key == '__D_inf__':
                            continue
                        storedJ = stored_data['Jpt']
                        diff = newJ - storedJ
                        matched = False
                        if D_inf is not None:
                            for k in range(-3, 4):
                                if (diff - k * D_inf).is_zero():
                                    logger.info("Jacobian dedup: matched new divisor to stored key %s with k=%d", stored_key, k)
                                    matched = True
                                    break
                        else:
                            if diff.is_zero():
                                logger.info("Jacobian dedup: exact equality match for stored key %s", stored_key)
                                matched = True
                        if matched:
                            seen_keys.add(key)
                            return True
                except Exception as e:
                    logger.warning("Jacobian comparison failed: %s", e)

            seen_keys.add(key)
            newt = dict(orig_tup)
            newt['u_poly'] = u
            newt['v_poly'] = v_norm
            coeffs = v_norm.list()
            # Careful extraction of v coeffs
            if len(coeffs) == 0:
                newt['v_0'] = QQ(0); newt['v_1'] = QQ(0)
            elif len(coeffs) == 1:
                newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(0)
            else:
                newt['v_0'] = QQ(coeffs[0]); newt['v_1'] = QQ(coeffs[1])
            
            # Use original s,p unless u changed (which is handled in caller)
            newt['s'] = QQ(s_q)
            newt['p'] = QQ(p_q)
            
            u_disc = u.discriminant()
            newt['has_rational_roots'] = True if (u.degree() <= 2 and u_disc.is_square()) else False
            newt['scale_used'] = lam

            out_list.append(newt)

            if J is not None and jac_points is not None and C is not None:
                try:
                    newJ = J([u, v_norm])
                    jkey = _canon_key_from_polys(u, v_norm)
                    jac_points[jkey] = {'Jpt': newJ, 'u': u, 'v': v_norm}
                except Exception:
                    pass

            logger.info("Accepted divisor s=%r p=%r with scale=%r", s_q, p_q, lam)
            return True
    return False

def canonicalize_and_dedup(divisors, f_coeffs, seed_x_coords=None):
    """
    Main entry point: returns canonicalized, deduplicated divisors (list of dicts).
    REMOVES divisors with Weierstrass point support by adding a seed divisor.
    """
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = build_f_poly(f_coeffs, R)

    weierstrass_points = []
    f_factors = f_poly.factor()
    for factor, mult in f_factors:
        if factor.degree() == 1:
            root = -factor[0] / factor[1]
            weierstrass_points.append(QQ(root))
    logger.info("Found %d rational Weierstrass points: %s", len(weierstrass_points), weierstrass_points)

    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    D_generic = None
    if seed_x_coords is not None and len(seed_x_coords) > 0:
        for seed_x in seed_x_coords:
            seed_x_q = QQ(seed_x)
            if seed_x_q in weierstrass_points:
                continue
            
            f_at_seed = f_poly(seed_x_q)
            sqrt_f_seed = rational_sqrt(f_at_seed)
            if sqrt_f_seed is not None:
                u_seed = x - seed_x_q
                v_seed = R(sqrt_f_seed)
                D_generic = J([u_seed, v_seed])
                logger.info("Created generic divisor from seed point x=%s, y=%s", seed_x_q, sqrt_f_seed)
                break
            else:
                logger.warning("f(%s) = %s is not a rational square, trying next seed", seed_x_q, f_at_seed)
    
    seen = set()
    seen_u = dict()
    out = []
    skipped_count = 0
    accepted_count = 0
    jac_points = {}

    def u_key_from_poly(u_poly):
        return tuple(rational_pair_key(c) for c in u_poly.list())
    
    def has_weierstrass_support(u_poly):
        for wp in weierstrass_points:
            if u_poly(wp) == 0:
                return True
        return False

    for tup in divisors:
        s_q = QQ(tup['s'])
        p_q = QQ(tup['p'])
        v0_q = QQ(tup['v_0'])
        v1_q = QQ(tup['v_1'])

        u = _u_from_sp(s_q, p_q, R)
        v = _v_from_coeffs(v1_q, v0_q, R)

        if has_weierstrass_support(u):
            if D_generic is None:
                logger.warning("Divisor has Weierstrass support but no valid D_generic found. Skipping.")
                skipped_count += 1
                continue

            logger.info("Divisor with Weierstrass support detected: s=%s, p=%s", s_q, p_q)
            try:
                D = J([u, v])
            except Exception as e:
                logger.warning("Failed to construct Jacobian point for Weierstrass shifting: %s", e)
                skipped_count += 1
                continue

            # Try shifting by multiples of D_generic to move off Weierstrass support
            shifted_success = False
            # Try a broader range of shifts to be robust against torsion/geometric collisions
            shift_attempts = []
            for k in range(1, 10):
                shift_attempts.append(k)
                shift_attempts.append(-k)
            
            for k in shift_attempts:
                try:
                    D_shifted = D + k * D_generic
                    u_new, v_new = D_shifted
                    
                    # CRITICAL: Ensure we have a valid generic Mumford divisor (degree 2 for genus 2)
                    # If degree is != 2, we can't represent it with (s, p) cleanly, so we skip it.
                    if u_new.degree() != 2:
                        logger.debug("Shift resulted in u_poly of degree %d (expected 2), trying next", u_new.degree())
                        continue

                    if not has_weierstrass_support(u_new):
                        u = u_new
                        v = v_new
                        
                        u_coeffs = u.list()
                        # u = x^2 - sx + p  =>  s = -coeff[1], p = coeff[0]
                        # We are guaranteed degree 2 here by the check above.
                        s_q = -u_coeffs[1]
                        p_q = u_coeffs[0]
                            
                        v_coeffs = v.list()
                        v0_q = v_coeffs[0] if len(v_coeffs) > 0 else QQ(0)
                        v1_q = v_coeffs[1] if len(v_coeffs) > 1 else QQ(0)
                        
                        logger.info("Shifted away from Weierstrass support using k=%d. New s=%s, p=%s", k, s_q, p_q)
                        shifted_success = True
                        break
                except Exception as e:
                    logger.debug("Shift attempt k=%d failed: %s", k, e)
                    continue
            
            if not shifted_success:
                logger.warning("Could not shift away from Weierstrass support for s=%s, p=%s after multiple attempts", s_q, p_q)
                skipped_count += 1
                continue

        v_red = v % u

        u_k = u_key_from_poly(u)

        if u_k in seen_u:
            stored_u, stored_v = seen_u[u_k]
            if same_v_up_to_sign_mod_u(v_red, stored_v, stored_u):
                skipped_count += 1
                continue
            else:
                skipped_count += 1
                continue

        if _attempt_scale_and_save(u, v_red, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
            if len(out) > 0:
                last = out[-1]
                seen_u[u_k] = (u, last['v_poly'])
            accepted_count += 1
            continue
        
        # ... [Logic for s^2-4p checks omitted for brevity, logic follows essentially the same flow] ...
        # If we reached here, try standard scaling heuristics based on discriminant...
        
        # Recalculate disc in case s,p changed
        disc = s_q * s_q - 4 * p_q
        disc_sqrt = rational_sqrt(disc) if disc != 0 else QQ(0)

        if disc_sqrt is not None and disc_sqrt == 0:
             # Repeated root logic
             r_double = s_q / QQ(2)
             fr = f_poly(r_double)
             sqrt_fr = rational_sqrt(fr)
             if sqrt_fr is not None:
                vr = v_red(r_double)
                if vr != 0:
                    lam_candidate = QQ(sqrt_fr) / QQ(vr)
                    v_scaled = lam_candidate * v_red
                    if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                         if len(out) > 0:
                            last = out[-1]
                            seen_u[u_k] = (u, last['v_poly'])
                         accepted_count += 1
                         continue
        
        elif disc_sqrt is not None:
            # Distinct roots logic
            r_plus = (s_q + disc_sqrt) / QQ(2)
            r_minus = (s_q - disc_sqrt) / QQ(2)
            denom = r_plus - r_minus
            if denom != 0:
                fa_plus = f_poly(r_plus)
                fa_minus = f_poly(r_minus)
                sqrt_plus = rational_sqrt(fa_plus)
                sqrt_minus = rational_sqrt(fa_minus)

                if sqrt_plus is not None and sqrt_minus is not None:
                    vr_plus = v_red(r_plus)
                    if vr_plus != 0:
                        tried_scale = False
                        for target in (sqrt_plus, -sqrt_plus):
                            lam = QQ(target) / QQ(vr_plus)
                            v_scaled = lam * v_red
                            if _attempt_scale_and_save(u, v_scaled, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                                if len(out) > 0:
                                    last = out[-1]
                                    seen_u[u_k] = (u, last['v_poly'])
                                accepted_count += 1
                                tried_scale = True
                                break
                        if tried_scale:
                            continue

                    # Try linear interpolation v(x) construction
                    matched = False
                    for sig_plus in (+1, -1):
                        for sig_minus in (+1, -1):
                            y_plus = QQ(sig_plus) * sqrt_plus
                            y_minus = QQ(sig_minus) * sqrt_minus
                            alpha = (y_plus - y_minus) / denom
                            beta = y_plus - alpha * r_plus
                            v_candidate = alpha * x + beta
                            if _attempt_scale_and_save(u, v_candidate, f_poly, s_q, p_q, tup, seen, seen_u, out, C=C, J=J, jac_points=jac_points):
                                if len(out) > 0:
                                    last = out[-1]
                                    seen_u[u_k] = (u, last['v_poly'])
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

def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=None):
    """
    Standalone verification of Mumford condition.
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

    # Reconstruct f using Horner's method (assuming High->Low coeffs)
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff

    diff = v_poly**2 - f_poly
    remainder = diff % u_poly

    return remainder.is_zero()

def quick_dependence_check(div1, div2):
    """Check if two divisors with same u are dependent"""
    if (div1['s'], div1['p']) != (div2['s'], div2['p']):
        return False
    
    if (div1['v_0'] == div2['v_0'] and div1['v_1'] == div2['v_1']):
        return True
    if (div1['v_0'] == -div2['v_0'] and div1['v_1'] == -div2['v_1']):
        return True
    
    return False
