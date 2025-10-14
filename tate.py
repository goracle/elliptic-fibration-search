"""
Tate's Algorithm Implementation for Elliptic Curve Fibrations

This module implements Tate's algorithm for classifying singular fibers
of elliptic curves over function fields, along with height pairing computations.
"""

import sys
import random
from collections import namedtuple
from tqdm import tqdm
import re

# SageMath imports
from sage.all import (
    QQ, ZZ, GF, PolynomialRing, LaurentSeriesRing, PowerSeriesRing,
    matrix, Matrix, vector, sqrt, floor, var, lcm, gcd,
    EllipticCurve, Curve, Jacobian
)
from sage.rings.fraction_field_element import FractionFieldElement
from sage.all import QQbar, parent

# Profile decorator fallback for when line_profiler is not available
try:
    PROFILE = profile
except NameError:
    def profile(func):
        """Default profiler when line_profiler is not available."""
        return func
    PROFILE = profile

from diagnostics2 import *


# Kodaira symbol classification constants
KODAIRA_CLASSIFICATION = {
    # (v_D, min_v_c4, min_v_c6): symbol
    (0, 0, 0): 'I0',
    (1, 0, 0): 'I1',
    (2, 0, 0): 'II',
    (3, 0, 0): 'III',
    (4, 0, 0): 'IV',
    (6, 2, 3): 'I0*',
    (8, 3, 5): 'IV*',
    (9, 4, 6): 'III*',
    (10, 5, 7): 'II*',
}

# Inverse intersection matrices for additive fibers
_INV_INT_I1STAR = Matrix(QQ, [
    [-4,  2,  0,  0,  0,  0],
    [ 2, -2,  1,  0,  0,  0],
    [ 0,  1, -2,  1,  0,  0],
    [ 0,  0,  1, -2,  1,  0],
    [ 0,  0,  0,  1, -2,  1],
    [ 0,  0,  0,  0,  1, -2],
])

ADDITIVE_INVERSE_INTERSECTION = {
    'I1*': _INV_INT_I1STAR,
    # Add more as needed
}

ADDITIVE_DELTA = {
    'II': QQ(1)/6,
    'III': QQ(1)/3,
    'IV': QQ(1)/2,
    'I0*': QQ(1)/2,
    'I1*': QQ(3)/4,
    'IV*': QQ(1)/2,
    'III*': QQ(2)/3,
    'II*': QQ(5)/6,
}

USE_PRIME_VALS = True  # set default True


def _call_tate_for_root(a4, a6, var_sym, root, debug=False, minimal=True):
    """
    root may be:
      - a rational element of QQ (exact)
      - a Python float (approx) coming from numerical root-finding
      - an algebraic number (QQbar) already
    We decide whether to pass m0 (rational) or g (irreducible irreducible polynomial)
    """
    # exact rational
    if root is None:
        # fiber at infinity
        return tates_algorithm(a4, a6, var_sym, None, at_infinity=True, debug=debug, minimal=minimal)

    # If it's rational in QQ:
    try:
        # try to coerce into QQ element
        r_QQ = QQ(root)
        # QQ(...) raises if not exact/rational; if success, pass m0=r_QQ
        return tates_algorithm(a4, a6, var_sym, r_QQ, at_infinity=False, debug=debug, minimal=minimal)
    except Exception:
        pass

    # If it's already an algebraic element from QQbar
    if hasattr(root, 'minpoly'):
        g = root.minpoly()
        return tates_algorithm(a4, a6, var_sym, None, g=g, at_infinity=False, debug=debug, minimal=minimal)

    # If it's a numeric approximation (float or complex), make an algebraic number
    # Use QQbar to get an algebraic number from approximate root;
    # QQbar(root) will try to recognize algebraic numbers from high-precision approximations.
    try:
        alg = QQbar(root)   # QQbar tries to find minimal polynomial
        g = alg.minpoly()
        return tates_algorithm(a4, a6, var_sym, None, g=g, at_infinity=False, debug=debug, minimal=minimal)
    except Exception as e:
        if debug:
            print("[find_singular_fibers] Could not coerce root to algebraic number:", root, "err:", e)
        # fallback: call numerical tate (if you have such a function) or skip
        return numerical_tates_algorithm(a4, a6, var_sym, root, precision=200, debug=debug)


def local_vals_driver(a4, a6, center=None, g=None, at_infinity=False, minimal=True, prec=80):
    if USE_PRIME_VALS:
        return valuations_at_place(a4, a6, g=g, m0=center, at_infinity=at_infinity, minimal=minimal)
    else:
        # your previous laurent-series path
        return valuations_at_point(a4, a6, var=SR.var('m'), center=center, minimal=minimal, prec=prec)


# ===== Prime-ideal local valuations over Q(m) =================================
# Works at: rational m0 in QQ, algebraic points via an irreducible g(m) in QQ[m],
# and at infinity. No floating point, no series — just exact divisibility.

def _QQm():
    R = QQ['m']; m, = R.gens()
    return R, m

def _make_monic(p):
    # p in QQ[m]
    if p == 0: return p
    return p / p.leading_coefficient()

def _ord_poly_at_factor(h, g):
    # integer exponent of irreducible g dividing polynomial h (exact, monic)
    if h == 0:
        return 10**9  # treat identically zero as +infty
    v = 0
    while h % g == 0:
        h //= g
        v += 1
    return v

def _ord_via_irreducible_polynomial(poly, g):
    """Computes valuation of a rational function `poly` with respect to an irreducible polynomial `g`."""
    num = poly.numerator()
    den = poly.denominator()

    base_ring = num.parent()
    if g.parent() != base_ring:
        g = base_ring(g)

    # Valuation is defined with respect to a monic irreducible polynomial.
    if g != 0 and not g.is_monic():
        g = g.monic()

    v_num = polynomial_valuation_at_factor(num, g)
    v_den = polynomial_valuation_at_factor(den, g)

    return v_num - v_den

def _ord_via_rational_point(poly, m0):
    """Computes valuation of a rational function `poly` at a rational point `m0`."""
    # The uniformizer (prime ideal generator) is (m - m0).
    try:
        # Case 1: poly is a FractionFieldElement.
        poly_ring = poly.parent().ring()
    except AttributeError:
        # Case 2: poly is a Polynomial.
        poly_ring = poly.parent()

    m = poly_ring.gen()
    g = m - m0

    return _ord_via_irreducible_polynomial(poly, g)


def ord_at_prime(poly, g=None, m0=None, at_infinity=False, minimal=True, debug=False):
    """
    poly: polynomial (or rational function) in m (a4/a6/c4/c6 etc)
    Provide exactly one of:
      - m0 (rational)  -> local field Q_p where m = m0
      - g   (irreducible polynomial in QQ[m]) -> local extension defined by g
      - at_infinity=True -> check valuation at t = 1/m (place at infinity)
    """
    def _deg(p):
        """Robustly get the degree of a polynomial, returns 0 for constants."""
        try:
            # is_constant() is a good check before calling .degree()
            if p.is_constant():
                return 0
            return p.degree()
        except (AttributeError, TypeError):
            return 0

    if at_infinity:
        # For a rational function P(m)/Q(m), the valuation at infinity
        # is defined as deg(Q) - deg(P).
        num = poly.numerator()
        den = poly.denominator()
        return _deg(den) - _deg(num)

    # If g is a QQbar element, use its minimal polynomial.
    if g is not None and hasattr(g, 'minpoly'):
        g = g.minpoly()

    # Case 1: Valuation with respect to an irreducible polynomial g.
    if g is not None:
        return _ord_via_irreducible_polynomial(poly, g)

    # Case 2: Valuation at a rational point m0.
    if m0 is not None:
        return _ord_via_rational_point(poly, m0)

    raise ValueError("ord_at_prime: must provide m0, g, or at_infinity=True")


def valuations_at_place(a4, a6, g=None, m0=None, at_infinity=False, minimal=True):
    """
    Compute local valuations (v_c4, v_c6, v_D) at the place P defined by:
      - P = infinity             if at_infinity=True
      - P = (m - m0)             if m0 in QQ
      - P = (g(m)) irreducible   if g is given
    for the short Weierstrass model y^2 = x^3 + a4(m) x + a6(m).

    Returns: v_c4, v_c6, v_D, n, (v_c4_min, v_c6_min, v_D_min)
    """
    R, m = _QQm()
    # invariants
    c4 = -48 * a4
    c6 = -864 * a6
    Delta = -16 * (4*a4**3 + 27*a6**2)

    kv = dict(g=g, m0=m0, at_infinity=at_infinity)
    v_c4 = ord_at_prime(c4, **kv)
    v_c6 = ord_at_prime(c6, **kv)
    v_D  = ord_at_prime(Delta, **kv)

    if v_D <= 0:
        return v_c4, v_c6, v_D, 0, (v_c4, v_c6, v_D)

    # minimalization exponent (x,y) -> (u^2 x', u^3 y'), v's shift by (-4n,-6n,-12n)
    n = 0
    if minimal:
        n = min(v_c4 // 4, v_c6 // 6, v_D // 12)
        #if n < 0: n = 0

    return v_c4, v_c6, v_D, n, (v_c4 - 4*n, v_c6 - 6*n, v_D - 12*n)



# --- Robust local valuations via Laurent series ------------------------------

def _laurent_series(expr, var, center, prec=80):
    """
    Return expr as a Laurent series in t around m=center (or infinity if center=None).
    """
    from sage.rings.laurent_series_ring import LaurentSeriesRing
    R = LaurentSeriesRing(QQ, 't', default_prec=prec)
    t = R.gen()
    if center is None:
        # expansion at infinity: m = 1/t
        return R(expr.subs({var: 1/t}))
    else:
        return R(expr.subs({var: center + t}))

def local_valuation(expr, var, center, prec=80, max_refine=3):
    """
    v_center(expr): exponent of the first nonzero term of the Laurent expansion.
    Works at finite points and at infinity (center=None).
    Retries with higher precision if the leading coefficient is swallowed by precision.
    Returns an int (can be negative). On hard failures returns None.
    """
    # Fast path for 0
    try:
        if expr == 0:
            return +10**9  # treat identically zero as +infty valuation
    except Exception:
        pass

    pr = prec
    for _ in range(max_refine):
        try:
            ls = _laurent_series(expr, var, center, prec=pr)
            v = ls.valuation()             # integer valuation in t
            return int(v)
        except Exception:
            pr = int(1.5 * pr) + 8  # refine precision and retry
            continue
    return None

def valuations_at_point(a4, a6, var, center, minimal=True, prec=80):
    """
    Compute local valuations (v_c4, v_c6, v_D) at m=center for the short Weierstrass model
        y^2 = x^3 + a4(m) x + a6(m)
    using Laurent series. Also compute the minimalizing exponent n (if minimal=True).

    Returns:
      v_c4, v_c6, v_D, n, (v_c4_min, v_c6_min, v_D_min)
    """
    # Invariants (short Weierstrass):
    #   c4 = -48*a4
    #   c6 = -864*a6
    #   Δ  = -16*(4*a4^3 + 27*a6^2)
    c4 = -48 * a4
    c6 = -864 * a6
    Delta = -16 * (4*a4**3 + 27*a6**2)

    v_c4 = local_valuation(c4, var, center, prec=prec)
    v_c6 = local_valuation(c6, var, center, prec=prec)
    v_D  = local_valuation(Delta, var, center, prec=prec)

    # If any valuation failed, try a bigger hammer once
    if None in (v_c4, v_c6, v_D):
        v_c4 = local_valuation(c4, var, center, prec=2*prec) if v_c4 is None else v_c4
        v_c6 = local_valuation(c6, var, center, prec=2*prec) if v_c6 is None else v_c6
        v_D  = local_valuation(Delta, var, center, prec=2*prec) if v_D  is None else v_D

    # If still missing, give up gracefully
    if None in (v_c4, v_c6, v_D):
        return v_c4, v_c6, v_D, None, (None, None, None)

    # If Δ does not vanish (v_D <= 0), fiber is smooth; return raw vals.
    if v_D <= 0:
        return v_c4, v_c6, v_D, 0, (v_c4, v_c6, v_D)

    # Minimalization exponent n:
    # under (x,y) -> (u^2 x', u^3 y'), we have c4' = u^{-4} c4, c6' = u^{-6} c6, Δ' = u^{-12} Δ.
    # valuations shift by (-4n,-6n,-12n) if v(u)=n.
    n = 0
    if minimal:
        n = min(v_c4 // 4, v_c6 // 6, v_D // 12)  # integer floor mins
        # keep n >= 0
        if n < 0:
            n = 0

    v_c4_min = v_c4 - 4*n
    v_c6_min = v_c6 - 6*n
    v_D_min  = v_D  - 12*n

    return v_c4, v_c6, v_D, n, (v_c4_min, v_c6_min, v_D_min)



def _coerce_to_laurent(expr, var_sym, center, max_prec=80):
    """
    Try multiple coercions (Laurent series with increasing precision).
    Raise a clear error with the expression if coercion fails.
    """
    for prec in (30, 60, 120):
        try:
            R = LaurentSeriesRing(QQ, 't', default_prec=prec)
            t = R.gen()
            if center is None:
                expr_t = expr.subs({var_sym: 1/t})
            else:
                expr_t = expr.subs({var_sym: center + t})
            return R(expr_t)
        except Exception:
            continue
    raise ValueError(f"Could not coerce expression to Laurent series. Expr: {expr}")


def kodaira_symbol(v_c4, v_c6, v_D):
    """
    Robust (minimal) Kodaira detector for the common cases used in our pipeline.
    This is not a complete Tate implementation, but handles I_n, I_n*, II,III,IV families
    that we actually encounter.
    """
    # smooth
    if v_D == 0:
        return 'I0'
    # multiplicative (I_n)
    if v_c4 == 0 and v_c6 == 0:
        return f'I{v_D}'
    # additive small (II, III, IV)
    if v_D == 2:
        return 'II'
    if v_D == 3:
        return 'III'
    if v_D == 4:
        return 'IV'
    # I_n* pattern: in minimal model one expects v_D >= 6 and v_c4 >= 2, v_c6 >= 3
    if v_D >= 6 and v_c4 >= 2 and v_c6 >= 3:
        n = v_D - 6
        return f'I{n}*'
    # fallback
    return 'Unknown'

def kodaira_components_count(sym):
    s = str(sym).strip()
    if s.startswith('I') and not s.endswith('*'):
        try:
            n = int(s[1:])
            return max(1, n)
        except Exception:
            return 1
    if s.startswith('I') and s.endswith('*'):
        # I_n* has (n + 5) irreducible components (I0* has 5)
        try:
            n = int(s[1:-1]) if s[1:-1] else 0
            return n + 5
        except Exception:
            return 5
    mapping = {'II':1, 'III':2, 'IV':3, 'II*':9, 'III*':8, 'IV*':7}
    return mapping.get(s, 1)

def kodaira_euler_number(s):
    if s is None:
        return 0
    s = s.strip()
    m = re.match(r'^I(\d+)(\*)?$', s)
    if m:
        n = int(m.group(1))
        if m.group(2):  # I_n*
            return n + 6
        return n
    roman_map = {'II':2, 'III':3, 'IV':4, 'II*':10, 'III*':9, 'IV*':8}
    if s in roman_map:
        return roman_map[s]
    raise ValueError(f"Unknown Kodaira symbol: {s}")


@PROFILE
def taylor_valuation(expr, var_sym, center, prec=10, debug=False):
    """
    Compute valuation of expression at given point using Taylor expansion.
    
    Args:
        expr: Symbolic expression
        var_sym: Variable symbol
        center: Point to expand around
        prec: Precision for power series
        debug: Print debug information
        
    Returns:
        int: Valuation at the point
    """
    t = var('t')
    R = PowerSeriesRing(QQ, names=('t',), default_prec=prec)
    (t,) = R._first_ngens(1)
    
    # Substitute var = center + t
    expr_t = expr.subs({var_sym: center + t})
    ps = R(expr_t)
    
    if debug:
        print(f"Power series expansion at {center}: {ps}")
    
    return ps.valuation()

def is_split_multiplicative_fiber(a4p, a6p, n):
    # a4p, a6p are Laurent series with constant term accessible by a4p[0], a6p[0]
    #a4_0 = a4p.padded_list(1)[0]
    #a6_0 = a6p.padded_list(1)[0]
    a4_0 = a4p[0]  # constant term of Laurent series
    a6_0 = a6p[0]
    # polynomial over QQ
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    poly = x**3 + QQ(a4_0) * x + QQ(a6_0)
    fac = poly.factor()
    # if poly has a rational linear factor then the node is rational (split)
    for f, m in fac:
        if f.degree() == 1:
            return True
    return False



def kodaira_components_count(sym):
    """
    Return the number of irreducible components of the Kodaira fiber 'sym'.
    """
    if sym is None:
        return 1

    s = str(sym).strip()
    # Multiplicative I_n
    if s.startswith('I') and '*' not in s:
        try:
            n = int(s[1:])
            # I0 and I1 are smooth/irreducible, I_n for n>1 has n components.
            return max(1, n)
        except (ValueError, IndexError):
            return 1 # Fallback for malformed symbol like 'I'

    # Starred types (I_n*)
    if s.startswith('I') and s.endswith('*'):
        try:
            n_str = s[1:-1]
            n = int(n_str) if n_str else 0
            if n == 0:
                return 4  # I0* has 4 components
            else:
                return n + 5  # In* for n>0 has n+5 components
        except (ValueError, IndexError):
            # Fallback for malformed symbol like 'I*'
            return 4

    # Other additive types
    mapping = {
        'II': 1, 'III': 2, 'IV': 3,
        'II*': 9, 'III*': 8, 'IV*': 7
    }
    return mapping.get(s, 1)

def safe_substitution(expr, var_sym, center, t):
    """
    Safely substitute variable with error handling.
    
    Args:
        expr: Expression to substitute into
        var_sym: Variable symbol
        center: Center point
        t: Parameter variable
        
    Returns:
        Substituted expression or original if substitution fails
    """
    if expr is None:
        return expr
    try:
        return expr.subs({var_sym: center + t})
    except (AttributeError, TypeError):
        return expr



def _coerce_laurent_with_precision(expr, var_sym, center, prec_list=(30, 60, 120)):
    """
    Try coercion to a LaurentSeriesRing at several precisions.
    Returns the Laurent series object on success or raises a ValueError with
    detailed diagnostics on failure.
    """
    last_exc = None
    for prec in prec_list:
        try:
            R = LaurentSeriesRing(QQ, 't', default_prec=prec)
            t = R.gen()
            if center is None:
                expr_t = expr.subs({var_sym: 1/t})
            else:
                expr_t = expr.subs({var_sym: center + t})
            return R(expr_t)  # will raise on bad coercion
        except Exception as e:
            last_exc = e
            continue
    # If we get here, coercion failed at all precisions
    raise ValueError(f"Could not coerce expression to Laurent series at centers {center}. "
                     f"Last exception: {last_exc}\nExpression was: {expr}")



def get_section_specialization_additive(curve_data, section, center, symbol, var_sym):
    """
    Determine component specialization for additive fibers.
    
    Args:
        curve_data: CurveData object
        section: Point coordinates
        center: Singular point location
        symbol: Kodaira symbol
        var_sym: Parameter variable
        
    Returns:
        int: Component index
    """
    t = var('t')
    R = LaurentSeriesRing(QQ, names=('t',), default_prec=10)
    (t,) = R._first_ngens(1)
    
    X = safe_substitution(section[0], var_sym, center, t)
    Y = safe_substitution(section[1], var_sym, center, t)
    
    X_laurent = R(X.parent().coerce(X))
    Y_laurent = R(Y.parent().coerce(Y))
    
    vX, vY = X_laurent.valuation(), Y_laurent.valuation()
    
    # Heuristic: if Y has much higher valuation than X, meets identity
    if vY > 1.5 * vX:
        return 0
    else:
        return 1


# Replace your ADDITIVE_CONTRIBUTIONS / ADDITIVE_DELTA with this single dict.


# Local correction (height pairing) contributions for additive fibers.
# Values are the rational numbers one typically adds to the naive height matrix
# at an additive fiber (these are the usual local correction contributions).
# If you later need other fiber-specific metadata, add nested dict entries.
# old
#FIBER_LOCAL_CORRECTION = {
    #'I0':     {'local_height': QQ(0)},      # Good reduction
    #'I1':     {'local_height': QQ(0)},      # Add this line
    #'II':     {'local_height': QQ(1)/6},
    #'III':    {'local_height': QQ(1)/3},
    #'IV':     {'local_height': QQ(1)/2},
    #'I0*':    {'local_height': QQ(1)/2},
    #'I1*':    {'local_height': QQ(7)/12},   # matches your earlier run's pattern
    #'I2*':    {'local_height': QQ(2)/3},    # important: I2* -> 2/3
    #'I3*':    {'local_height': QQ(3)/4},
    #'I6*':    {'local_height': QQ(1)}, # needs to be checked?
    #'I4*':    {'local_height': QQ(5)/6},
    #'IV*':    {'local_height': QQ(1)/2},
    #'III*':   {'local_height': QQ(2)/3},
    #'II*':    {'local_height': QQ(5)/6},
    #}
####

# keep existing explicit table for common non-starred/additive types
# --- robust local correction resolver ---
# minimal canonical table for non-star / small additive types
FIBER_LOCAL_CORRECTION = {
    'I0':   {'local_height': QQ(0)},
    'I1':   {'local_height': QQ(0)},
    'II':   {'local_height': QQ(1)/6},
    'III':  {'local_height': QQ(1)/3},
    'IV':   {'local_height': QQ(1)/2},
    'I0*':  {'local_height': QQ(1)/2},
    'IV*':  {'local_height': QQ(1)/2},
    'III*': {'local_height': QQ(2)/3},
    'II*':  {'local_height': QQ(5)/6},
    # do not enumerate every I_n* here; handle them generically below
}

def local_correction_value(symbol):
    """
    Resolve local height correction for Kodaira symbol `symbol`.
    - Handles I_n* generically via formula (n + 6) / 12
    - Returns QQ rational; multiplicative I_n -> 0
    - Unknown additive symbols produce a warning and a conservative default.
    """
    if symbol is None:
        return QQ(0)

    sym = str(symbol).strip()

    # direct explicit table lookup first
    meta = FIBER_LOCAL_CORRECTION.get(sym)
    if meta is not None:
        return meta['local_height']

    # generic I_n* handling: (n + 6)/12
    m = re.match(r'^I(\d+)\*$', sym)
    if m:
        n = int(m.group(1))
        return QQ(n + 6) / QQ(12)

    # multiplicative I_n (no star) have zero additive correction
    if re.match(r'^I\d+$', sym):
        return QQ(0)

    # last resort: warn and pick a conservative default (1/3)
    warnings.warn(f"local_correction_value: unknown Kodaira symbol '{sym}' -- using fallback correction 1/3")
    return QQ(1) / QQ(3)



@PROFILE
def get_section_specialization_In(cd, section, fiber_data, var_sym):
    """
    Determine which component of an I_n fiber a section specializes to.

    Args:
        cd: CurveData object.
        section: A point on the Weierstrass model, e.g., P.
        fiber_data: Dictionary for the singular fiber.
        var_sym: The symbolic variable for the fibration parameter, e.g., m.

    Returns:
        int: The index of the component (0 to n-1).
    """
    center = fiber_data.get('center')
    n = fiber_data.get('n')

    # This calculation is only for multiplicative fibers of type I_n with n > 0.
    if not (fiber_data.get('type') == 'multiplicative' and n > 0):
        return 0

    try:
        # 1. Set up the Laurent series ring with a local parameter 't'.
        R = LaurentSeriesRing(QQ, 't', default_prec=40)
        t = R.gen()

        # 2. Get the section's projective coordinates.
        X_expr, Y_expr, Z_expr = section[0], section[1], section[2]

        # 3. Substitute m = center + t (or m = 1/t for infinity) and create series.
        if center is None:  # Fiber at infinity
            X_s = R(X_expr.subs({var_sym: 1 / t}))
            Y_s = R(Y_expr.subs({var_sym: 1 / t}))
            Z_s = R(Z_expr.subs({var_sym: 1 / t}))
        else:  # Finite fiber
            X_s = R(X_expr.subs({var_sym: center + t}))
            Y_s = R(Y_expr.subs({var_sym: center + t}))
            Z_s = R(Z_expr.subs({var_sym: center + t}))

        # 4. Determine component index based on valuations.
        v_X = X_s.valuation()
        v_Y = Y_s.valuation()
        v_Z = Z_s.valuation()

        if v_X > v_Z:
            # Specializes to the identity component.
            return 0
        elif v_X < v_Z:
            # Specializes to a non-identity component. Index is based on Y's valuation.
            return (v_Y - 3 * v_X) % n
        else: # v_X == v_Z
            # This case requires analyzing leading coefficients. For now, we assume
            # a generic position which maps to the identity component.
            # A more advanced implementation would handle non-generic specializations here.
            return 0

    except Exception as e:
        # If any step fails (e.g., substitution, coercion to series),
        # we cannot determine the component, so we default to the identity component (0).
        print(f"Warning: Could not determine component for fiber at {center}. Defaulting to 0. Error: {e}")
        return 0


@PROFILE
def local_pairing_contribution(P, Q, fiber_data, curve_data, var_sym):
    """
    Compute local height pairing contribution at a singular fiber.
    """
    center = fiber_data.get('center') or fiber_data.get('r')
    symbol = fiber_data['symbol']
    
    is_split_multiplicative = (
        fiber_data.get('type') == 'multiplicative' and
        symbol.startswith('I') and
        symbol[1:].isdigit()
    )

    # Multiplicative fibers (I_n for n > 1)
    if is_split_multiplicative:
        n = fiber_data.get('n', 0)
        if n <= 1:
            return 0
        
        # Get component indices using the corrected function call
        iP = get_section_specialization_In(curve_data, P, fiber_data, var_sym)
        iQ = get_section_specialization_In(curve_data, Q, fiber_data, var_sym)
        iPQ = get_section_specialization_In(curve_data, P + Q, fiber_data, var_sym)
        
        # Second Bernoulli polynomial B_2(x) = x^2 - x + 1/6
        def B2(x):
            return x**2 - x + QQ(1)/6
        
        term_p = B2(QQ(iP) / n)
        term_q = B2(QQ(iQ) / n)
        term_pq = B2(QQ(iPQ) / n)
        
        correction = -n/2 * (term_pq - term_p - term_q)
        return correction
    
    # Additive fibers
    if fiber_data.get('type') == 'additive':
        try:
            return local_correction_value(symbol)
        except KeyError:
            raise RuntimeError(f"Missing local correction for additive symbol '{symbol}' at center {center}")
    
    return 0

def validate_tates_algorithm():
    """
    Validate Tate's algorithm against known results for Legendre fibration.
    
    The Legendre family y^2 = x(x-1)(x-m) transforms to Weierstrass form
    y^2 = x^3 + A(m)x + B(m) where:
    - A(m) = -(m^2 - m + 1)/3  
    - B(m) = (-2m^3 + 3m^2 + 3m - 2)/27
    
    This has I2 fibers at m=0 and m=1.
    
    Returns:
        bool: True if all tests pass
    """
    print("\n--- Validating Tate's Algorithm Implementation ---")
    
    R_m = PolynomialRing(QQ, 'm')
    m_sym = R_m.gen()
    
    A = -(m_sym**2 - m_sym + 1) / 3
    B = (-2*m_sym**3 + 3*m_sym**2 + 3*m_sym - 2) / 27
    
    known_fibers = {0: 'I2', 1: 'I2'}
    
    print("Test fibration: Legendre Family y^2 = x(x-1)(x-m)")
    print(f"A(m) = {A}")
    print(f"B(m) = {B}")
    
    all_passed = True
    for center, expected_symbol in known_fibers.items():
        print(f"\nChecking fiber at m = {center}...")
        
        fiber_data = tates_algorithm(A, B, m_sym, center, debug=True)
        computed_symbol = fiber_data.get('symbol', 'Error')
        
        print(f"  Expected: {expected_symbol}")
        print(f"  Computed: {computed_symbol}")
        
        if computed_symbol == expected_symbol:
            print("  ✅ PASS")
        else:
            print("  ❌ FAIL")
            all_passed = False
    
    return all_passed


def main():
    """Main function for testing and validation."""
    print("Tate's Algorithm for Elliptic Fibrations")
    print("=" * 50)
    
    validation_result = validate_tates_algorithm()
    
    if validation_result:
        print("\n🎉 All validation tests passed!")
    else:
        print("\n❌ Some validation tests failed.")
    
    return validation_result


### claude's opinions:
"""
Complete singular fiber detection for elliptic surfaces.
Finds all singular fibers including those with irrational/complex centers.
"""

import numpy as np
from sage.all import (
    QQ, CC, AA, QQbar, PolynomialRing, 
    ComplexField, RealField, AlgebraicField,
    numerical_approx, factor, gcd
)


def compute_euler_characteristic(fibers):
    """
    Compute total Euler characteristic from fiber list.
    
    Args:
        fibers: List of fiber data
    
    Returns:
        int: Total Euler characteristic
    """
    euler_map = {
        'I0': 0, 'I1': 1, 'I2': 2, 'I3': 3, 'I4': 4, 'I5': 5, 'I6': 6,
        'I7': 7, 'I8': 8, 'I9': 9, 'I10': 10, 'I11': 11, 'I12': 12,
        'II': 2, 'III': 3, 'IV': 4,
        'I0*': 6, 'I1*': 7, 'I2*': 8, 'I3*': 9, 'I4*': 10, 'I5*': 11, 'I6*': 12,
        'II*': 10, 'III*': 9, 'IV*': 8
    }
    
    total_euler = 0
    for fiber in fibers:
        symbol = fiber.get('symbol', 'Unknown')
        euler_contrib = euler_map.get(symbol, 0)
        total_euler += euler_contrib
    
    print(f"\nEuler characteristic calculation:")
    for fiber in fibers:
        symbol = fiber.get('symbol', 'Unknown')
        center = fiber.get('center', 'Unknown')
        euler_contrib = euler_map.get(symbol, 0)
        print(f"  {symbol} at {center}: +{euler_contrib}")
    
    print(f"Total Euler characteristic: {total_euler}")
    
    # Determine surface type
    if abs(total_euler - 12) < 0.1:
        print("*** RATIONAL ELLIPTIC SURFACE (χ = 12) ***")
    elif abs(total_euler - 24) < 0.1:
        print("*** K3 SURFACE (χ = 24) ***")
    else:
        print(f"*** UNUSUAL SURFACE TYPE (χ = {total_euler}) ***")
    
    return total_euler

# Import the tates_algorithm function (assuming it's available)
# from your_tate_module import tates_algorithm

# Example usage:

def find_all_discriminant_roots(discriminant, var_sym, precision=100, debug=False):
    """
    Find ALL roots of the discriminant, including irrational and complex ones.
    This version correctly finds rational roots from the numerator and denominator.
    """
    num = discriminant.numerator()
    den = discriminant.denominator()

    if debug:
        print(f"Numerator degree: {num.degree()}")
        print(f"Denominator degree: {den.degree()}")

    roots = {'rational': [], 'irrational': [], 'complex': []}

    # 1. Find rational roots using exact methods from both numerator and denominator
    print("Finding rational roots...")
    rational_roots = []
    
    # Process numerator
    for factor, mult in num.factor():
        try:
            factor_roots = factor.roots(ring=QQ, multiplicities=False)
            rational_roots.extend(factor_roots)
        except Exception:
            # a factor might not have rational roots
            continue
            
    # Process denominator
    if not den.is_constant():
        for factor, mult in den.factor():
            try:
                factor_roots = factor.roots(ring=QQ, multiplicities=False)
                rational_roots.extend(factor_roots)
            except Exception:
                continue

    roots['rational'] = sorted(list(set(rational_roots)))
    print(f"Found {len(roots['rational'])} rational roots: {roots['rational']}")

    # 2. Find all roots numerically using numpy (from numerator)
    print("Finding all roots numerically...")
    try:
        max_deg = num.degree()
        coeffs_list = num.coefficients(sparse=False)
        # Sage returns coeffs lowest to highest, numpy wants highest to lowest
        coeffs_list.reverse()
        numerical_roots = np.roots(coeffs_list)
        
        if debug:
            print(f"Found {len(numerical_roots)} numerical roots")

        tolerance = 10**(-precision//2)
        for root in numerical_roots:
            if abs(root.imag) < tolerance:
                real_root = float(root.real)
                is_rational = any(abs(real_root - float(r)) < tolerance for r in roots['rational'])
                if not is_rational:
                    roots['irrational'].append(real_root)
            else:
                roots['complex'].append(complex(root))
    except Exception as e:
        print(f"Error in numerical root finding: {e}")

    # 3. Clean up the lists
    roots['irrational'] = sorted(list(set(roots['irrational'])))
    
    complex_cleaned = []
    if roots['complex']:
        tolerance = 10**(-precision//3)
        # Remove conjugate duplicates
        for root in roots['complex']:
            is_duplicate = any(abs(root - existing) < tolerance or abs(root.conjugate() - existing) < tolerance for existing in complex_cleaned)
            if not is_duplicate:
                complex_cleaned.append(root)
    roots['complex'] = complex_cleaned

    print("Summary:")
    print(f"  Rational roots: {len(roots['rational'])}")
    print(f"  Irrational roots: {len(roots['irrational'])}")
    print(f"  Complex roots: {len(roots['complex'])}")
    print(f"  Total: {sum(len(v) for v in roots.values())}")

    return roots

def numerical_tates_algorithm(a4, a6, var_sym, center, precision=100, debug=False):
    """
    Numerical version of Tate's algorithm - RAISES exceptions instead of hiding them.
    """
    # Evaluate a4, a6 at the center
    a4_val = complex(a4.subs({var_sym: center}))
    a6_val = complex(a6.subs({var_sym: center}))
    
    # Compute discriminant and c-invariants
    Delta_val = -16 * (4*a4_val**3 + 27*a6_val**2)
    c4_val = -48 * a4_val
    c6_val = -864 * a6_val
    
    # Proper numerical valuation estimation
    def estimate_valuation(z, tol=10**(-precision//4)):
        """
        Estimate valuation - simple but honest version.
        """
        if abs(z) > tol:
            return 0  # Non-zero, valuation 0
        
        # For very small values, try to estimate order
        # This is heuristic but better than nothing
        if abs(z) < tol**2:
            return 2  # Very small, probably order 2 or higher
        else:
            return 1  # Small, probably order 1
    
    v_Delta = estimate_valuation(Delta_val)
    v_c4 = estimate_valuation(c4_val)  
    v_c6 = estimate_valuation(c6_val)
    
    if debug:
        print(f"At center {center}: Delta_val={Delta_val}, v_Delta={v_Delta}, v_c4={v_c4}, v_c6={v_c6}")
    
    # Classify using similar logic to exact case
    if v_Delta == 0:
        symbol = 'I0'
    elif v_c4 == 0 and v_c6 == 0:
        # Multiplicative case
        symbol = f'I{v_Delta}'
    else:
        # Additive case - simple heuristics
        if v_Delta == 2:
            symbol = 'II'
        elif v_Delta == 3:
            symbol = 'III'
        elif v_Delta == 4:
            symbol = 'IV'
        elif v_Delta >= 6:
            symbol = f'I{v_Delta-6}*'
        else:
            symbol = 'Unknown'
    
    fiber_type = 'multiplicative' if symbol.startswith('I') and '*' not in symbol else 'additive'
    
    fiber = {
        'symbol': symbol,
        'v_c4': v_c4,
        'v_c6': v_c6,
        'v_D': v_Delta,
        'center': center,
        'r': center,
        'type': fiber_type
    }
    
    # Add multiplicative-specific data
    if fiber_type == 'multiplicative' and symbol != 'I0':
        if symbol.endswith('*'):
            n = int(symbol[1:-1]) if symbol[1:-1].isdigit() else 0
        else:
            n = int(symbol[1:]) if symbol[1:].isdigit() else 1
        fiber['n'] = n
        fiber['split'] = True  # Assume split for numerical case
    
    return fiber




# ---------- PATCH for tate.py ----------


# Put near top of tate.py (imports already in your file assumed)
# add near top of file
from functools import reduce
from math import gcd as _gcd
from sage.all import ZZ, PolynomialRing

def _lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a // _gcd(a, b) * b)

def _make_integer_primitive(poly_q):
    # poly_q could be in QQ(m) or fraction field over QQ
    # get numerator and denominator as polynomials in QQ[m]
    num = poly_q.numerator()    # element of QQ[m]
    den = poly_q.denominator()  # element of QQ[m]
    # clear coefficient denominators separately:
    # convert num, den to polynomial over ZZ by multiplying by common denom of coefficients
    num_int = num * lcm([c.denominator() for c in num.coefficients()])
    den_int = den * lcm([c.denominator() for c in den.coefficients()])

    # now we can change ring to ZZ safely:
    num_Z = num_int.change_ring(ZZ)
    den_Z = den_int.change_ring(ZZ)

    # primitive part: remove content gcd of coefficients
    g = gcd([abs(int(c)) for c in num_Z.coefficients()])
    num_prim = num_Z // g

    return num_prim.monic()  # or num_prim.primitive_part() if available for that ring

# ---------- END PATCH ----------

# --- in tate.py ---

def kodaira_components_count(sym):
    s = str(sym).strip()
    if s.startswith('I') and '*' not in s:
        try:
            n = int(s[1:])
            return max(1, n)  # I0=1, I1=1, I2=2, ...
        except Exception:
            return 1
    if s.startswith('I') and s.endswith('*'):
        try:
            n = int(s[1:-1])
            return n + 6      # I_n* has n+6 components
        except Exception:
            return 7
    mapping = {'II': 1, 'III': 2, 'IV': 3, 'II*': 9, 'III*': 8, 'IV*': 7}
    return mapping.get(s, 1)



def kodaira_euler_number(s):
    if s is None:
        return 0
    s = s.strip()
    # I_n or I_n*
    m = re.match(r'^I(\d+)(\*)?$', s)
    if m:
        n = int(m.group(1))
        if m.group(2):
            return n + 6   # I_n* -> n + 6
        return n
    # direct roman names
    roman_map = {
        'II':2, 'III':3, 'IV':4,
        'II*':10, 'III*':9, 'IV*':8
    }
    if s in roman_map:
        return roman_map[s]
    # unknown fallback
    raise ValueError(f"Unknown Kodaira symbol: {s}")


from math import floor
from sage.all import QQ, PolynomialRing

def shioda_tate_from_fiber_list(fibers, rho_geom=None, debug=False, return_diagnostics=False, clamp_negative=False, allow_auto_rho=False):
    """
    Compute Shioda-Tate rank estimate from a fiber list.
    Expects each fiber dict to contain:
      - 'degree': algebraic degree of factor
      - 'symbol': Kodaira symbol (or None)
      - 'm_v': Kodaira component count
      - 'e': Euler number per fiber (or 'e_contrib' was already stored); but we will recompute e via helper.
    """
    total_contrib = 0
    euler_sum = 0
    fiber_info = []
    for f in fibers:
        sym = f.get('symbol')
        if sym is None or sym == 'I0':
            continue

        # degree of algebraic factor
        deg = int(f.get('degree', 1))

        # component count m_v (unmultiplied)
        mv = f.get('m_v', None)
        if mv is None:
            # if not present, get it from helper
            mv = kodaira_components_count(sym)

        # contribution to sigma: deg * (m_v - 1)
        contrib = max(0, int(mv) - 1)
        # e (Euler number for one copy of that Kodaira fiber)
        e = kodaira_euler_number(sym)

        # aggregate
        total_contrib += deg * contrib
        euler_sum += deg * int(e)

        fiber_info.append({
            'center': f.get('r'),
            'symbol': sym,
            'm_v': int(mv),
            'm_v-1': int(contrib),
            'e': int(e),
            'degree': deg
        })

    diagnostics = {
        'sum_contributions': int(total_contrib),
        'euler_characteristic': int(euler_sum),
        'num_fibers': len(fiber_info),
        'fibers': fiber_info
    }

    rho_min = 2 + total_contrib
    diagnostics['rho_min_for_nonnegative_MW_rank'] = int(rho_min)
    rank = None

    if rho_geom is not None:
        try:
            rho_in = int(rho_geom)
            diagnostics['rho_used_initial'] = rho_in
            rank_calc = rho_in - 2 - total_contrib
            diagnostics['rank_raw'] = float(rank_calc)

            if rank_calc < 0:
                diagnostics['inconsistent_rho'] = True
                if allow_auto_rho:
                    diagnostics['note'] = f"Provided rho_geom={rho_in} is too small (yields rank {rank_calc}). Automatically using rho_min={rho_min}."
                    rho_used = rho_min
                    rank = rho_used - 2 - total_contrib
                    diagnostics['rho_used_after_auto'] = int(rho_used)
                elif clamp_negative:
                    diagnostics['note'] = f"Calculated rank is negative ({rank_calc}), clamping to 0."
                    rank = 0
                else:
                    diagnostics['note'] = f"Provided rho_geom={rho_in} is inconsistent with fiber data (yields rank {rank_calc}). Rank is undefined."
                    rank = None
            else:
                rank = int(rank_calc)
        except (ValueError, TypeError):
            diagnostics['rho_input_error'] = f"non-integer rho_geom provided: {rho_geom}"
            rank = None
    else:
        diagnostics['note'] = "No rho_geom provided; cannot compute rank."

    if return_diagnostics:
        return rank, {'sum_contributions': total_contrib, 'euler_characteristic': euler_sum, 'fibers': fiber_info}, diagnostics

    return rank, {'sum_contributions': total_contrib, 'euler_characteristic': euler_sum, 'fibers': fiber_info}




def classify_from_minimal_vals(v4, v6, vD):
    """
    Classify fiber type from minimal valuations (v_c4, v_c6, v_D).
    This uses the standard Kodaira classification table.
    """
    # Smooth fiber
    if vD <= 0:
        return 'I0', 'smooth'
    
    # Multiplicative I_n: v_c4 = v_c6 = 0, v_D >= 1
    if v4 == 0 and v6 == 0 and vD >= 1:
        return f'I{vD}', 'multiplicative'
    
    # Additive fibers - match against standard table
    # Key: (v_c4, v_c6, v_D) -> symbol
    additive_table = {
        (1, 1, 2): 'II',
        (1, 2, 3): 'III',
        (2, 3, 4): 'IV',
        (2, 3, 6): 'I0*',
        (3, 4, 8): 'IV*',
        (3, 5, 9): 'III*',
        (4, 5, 10): 'II*',
    }
    
    key = (v4, v6, vD)
    if key in additive_table:
        return additive_table[key], 'additive'
    
    # I_n* pattern: v_c4 >= 2, v_c6 >= 3, v_D >= 6
    # General I_n* has v_D = n + 6
    if v4 >= 2 and v6 >= 3 and vD >= 6:
        n = vD - 6
        return f'I{n}*', 'additive'
    
    # Fallback for unrecognized additive pattern
    # Return a placeholder that will be caught during height pairing
    return f'AdditiveFiber({v4},{v6},{vD})', 'additive'


# Then in tates_algorithm, use it like this:

def tates_algorithm(a4, a6, var_sym, center=None, debug=False, minimal=True, g=None, at_infinity=False):
    """
    Core Tate's algorithm for fiber classification.
    """
    # 1) Compute local valuations via exact prime-ideal method
    if center is None:
        assert at_infinity

    if g is not None:
        v4, v6, vD, n, (v4m, v6m, vDm) = local_vals_driver(
            a4, a6, at_infinity=at_infinity, g=g, minimal=minimal
        )
        root_type = 'irrational'
        center_display = 'root(g)'
    elif at_infinity:
        v4, v6, vD, n, (v4m, v6m, vDm) = local_vals_driver(
            a4, a6, at_infinity=at_infinity, minimal=minimal
        )
        root_type = 'infinity'
        center_display = None
    else:
        v4, v6, vD, n, (v4m, v6m, vDm) = local_vals_driver(
            a4, a6, at_infinity=at_infinity, center=center, minimal=minimal
        )
        root_type = 'rational'
        center_display = center

    if debug:
        print(f"[tate] P=({('∞' if at_infinity else ('g' if g is not None else center))}) "
              f"v=(c4 {v4}, c6 {v6}, Δ {vD}) -> n={n} ; minimal ( {v4m}, {v6m}, {vDm} )")

    # 2) Quick exit for smooth fibers
    if vDm <= 0:
        return {
            'symbol': 'I0',
            'v_c4': v4m, 'v_c6': v6m, 'v_D': vDm,
            'center': center_display, 'r': center_display,
            'minimal_used': bool(minimal),
            'type': 'smooth' if vDm < 0 else 'multiplicative',
            'n': n,
            'split': None,
            'root_type': root_type
        }

    # 3) Classify from minimal valuations
    symbol, typ = classify_from_minimal_vals(v4m, v6m, vDm)

    result = {
        'symbol': symbol,
        'v_c4': v4m, 'v_c6': v6m, 'v_D': vDm,
        'center': center_display, 'r': center_display,
        'minimal_used': bool(minimal),
        'type': typ,
        'n': n,
        'split': (typ == 'multiplicative'),
        'root_type': root_type
    }

    if debug:
        print(f"[tate] -> classified as: {symbol} ({typ})")

    return result


if __name__ == '__main__':
    main()
