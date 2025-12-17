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


