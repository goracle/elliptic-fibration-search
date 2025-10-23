# === imports ===
import sys
import os
import subprocess
import warnings
#from collections import namedtuple
from typing import NamedTuple
from functools import lru_cache
import itertools
import multiprocessing
from multiprocessing import TimeoutError
# Sage imports (explicit, minimal)
from sage.all import (
    QQ, ZZ, RR, GF, SR, var, PolynomialRing, Matrix, matrix, vector, diff, floor,
    Curve, Jacobian, EllipticCurve, sqrt, CRT, lcm, primes, QuadraticForm, ceil,
    is_prime, Integer, log
)
from math import gcd, log

def parse_hyperelliptic_db_entry(db_string):
    """
    Parse a hyperelliptic curve entry from the MIT database and extract coefficients.
    https://math.mit.edu/~drew/gce_genus3_hyperelliptic.txt    
    Input format: D:N:[f(x),h(x)]
    where the curve is: y^2 + h(x)*y = f(x)
    
    We transform this to: Y^2 = h(x)^2 + 4*f(x)
    where Y = 2*y + h(x)
    
    Returns a coefficient vector [c_0, c_1, ..., c_n] where
    the right-hand side polynomial is c_0 + c_1*x + c_2*x^2 + ...
    
    Args:
        db_string: String like "10000000:2000000:[-5*x^7-4*x^6-3*x^5-2*x^4,x^3+x^2+x+1]"
    
    Returns:
        list of QQ coefficients (low to high degree)
    """
    
    # Parse the database format: D:N:[f(x),h(x)]
    # Extract the part inside the brackets
    match = re.search(r'\[(.*?)\]$', db_string)
    if not match:
        raise ValueError(f"Could not parse database string: {db_string}")
    
    poly_part = match.group(1)
    
    # Split by comma at the top level (not inside nested parens)
    parts = []
    depth = 0
    current = ""
    for char in poly_part:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    
    if len(parts) != 2:
        raise ValueError(f"Expected 2 polynomials (f and h), got {len(parts)}: {parts}")
    
    f_str, h_str = parts
    
    # Create polynomial ring for parsing
    PR = PolynomialRing(QQ, 'x')
    x = PR.gen()
    
    # Replace ^ with ** for Python exponentiation
    f_str = f_str.replace('^', '**')
    h_str = h_str.replace('^', '**')
    
    # Create a safe namespace with the polynomial variable
    namespace = {'x': x}
    
    # Parse polynomials
    try:
        f = eval(f_str, {"__builtins__": {}}, namespace)
        h = eval(h_str, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(f"Could not parse polynomials: f={f_str}, h={h_str}. Error: {e}")
    
    # Compute the transformed RHS: h(x)^2 + 4*f(x)
    rhs_poly = h**2 + 4*f
    
    # Polynomials in Sage's FLINT ring are already expanded, no need to call .expand()
    
    # Extract coefficients (low to high degree)
    coeffs = rhs_poly.coefficients(sparse=False)
    
    # Convert to QQ-wrapped integers
    coeffs = [QQ(int(c)) for c in coeffs]
    
    return coeffs




# Add these with your other Sage imports
#from sage.libs.pari.pari_error import PariError
from cysignals.signals import SignalError
import random

from sage.all import sage_eval, SR, PolynomialRing, QQ, EllipticCurve, RR
import traceback
import math

# Local modules
from tate import *


#### BEGIN USER CONFIG

# Input curve coefficients (starting curve coefficients)
A1 = QQ(8)
A2 = QQ(-3)
A3 = QQ(-14)
A4 = QQ(3)
A5 = QQ(6)
A6 = QQ(1)

# Starting rational data points (starting rational point list)
DATA_PTS = [(QQ(1)/QQ(2), QQ(7)/QQ(4)), (QQ(3), QQ(37)), (QQ(-1), QQ(1))]

# TEST CURVE 1
# --- Configuration, deg x = 5---
A1 = 4
A2 = 8
A3 = 20
A4 = -4
A5 = -4
A6 = 1
COEFFS = [A1, A2, A3, A4, A5, A6]
DATA_PTS = [(QQ(0), QQ(1))] # finds all known rational points
TERMINATE_WHEN = 4


##### TEST CURVES (from lmfdb.org) ######


# --- Configuration, deg x = 6---
# y^2 = a0*x^6 + a1*x^5 + ... + a6
# old curves
COEFFS_GENUS2 = [QQ(4), QQ(-8), QQ(-20), QQ(0), QQ(16), QQ(8), QQ(1)]
COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(5), QQ(6), QQ(5), QQ(2), QQ(1)]
COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(7), QQ(6), QQ(-3), QQ(-8), QQ(-4)]

# old curve, the OG
#x^7 - 10 x^5 + 15 x + 5
COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(-2), QQ(-18), QQ(1), QQ(38), QQ(25)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 11

# # doesn't find y=0 point... added a special function to find these...maybe ok...
COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(-4), QQ(10), QQ(-24), QQ(24), QQ(-7)]
DATA_PTS_GENUS2 = [QQ(2)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(12), QQ(16), QQ(-12), QQ(-20), QQ(12)]
DATA_PTS_GENUS2 = [QQ(-2)] # just the x values lol
TERMINATE_WHEN_6 = 2


COEFFS_GENUS2 = [QQ(4), QQ(0), QQ(-12), QQ(-4), QQ(12), QQ(8), QQ(-7)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(-11), QQ(-12), QQ(56), QQ(16), QQ(-116)]
DATA_PTS_GENUS2 = [QQ(-3)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(1), QQ(-6), QQ(2), QQ(8), QQ(-7)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 2

COEFFS_GENUS2 = [QQ(4), QQ(-4), QQ(-36), QQ(5), QQ(96), QQ(64)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 4

COEFFS_GENUS2 = [QQ(4), QQ(0), QQ(-16), QQ(24), QQ(-16), QQ(5)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 2

COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(2), QQ(-18), QQ(21), QQ(-10), QQ(1)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 4

COEFFS_GENUS2 = [QQ(1), QQ(6), QQ(10), QQ(7), QQ(1), QQ(0)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(4), QQ(4), QQ(8), QQ(-8), QQ(-12)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(3), QQ(2), QQ(5), QQ(8), QQ(-4)]
DATA_PTS_GENUS2 = [QQ(-5)/QQ(3)] # just the x values lol
TERMINATE_WHEN_6 = 3

# deg 5
COEFFS_GENUS2 = [QQ(4), QQ(4), QQ(-16), QQ(-19), QQ(16), QQ(20)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 2


# genus 3 test curve
COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(-4), QQ(0), QQ(1)]
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4 # only 3 points, but set to 4 to demonstrate the search

#Y² = -20x^7 - 15x^6 - 10x^5 - 5x^4 + 4x^3 + 3x^2 + 2x + 1
COEFFS_GENUS2 = [QQ(-20), QQ(-15), QQ(-10), QQ(-5), QQ(4), QQ(3), QQ(2), QQ(1)]
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4 # only 3 points, but set to 4 to demonstrate the search

#db_entry = '9995456:2498864:[2*x^7-4*x^6-5*x^5+10*x^4+5*x^3-8*x^2-3*x+1,x^2+x]'
db_entry = '9995408:2498852:[x^8-x^6+x^3+2*x^2+x,x^2+x+1]' # first number is disc, second number is conductor
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 5

db_entry='10000000:2000000:[-5*x^7-4*x^6-3*x^5-2*x^4,x^3+x^2+x+1]'
db_entry='9999936:1249992:[x^6+3*x^5+5*x^4+5*x^3+4*x^2+2*x,x^4+x^3+x^2+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9999899:769223:[-2*x^8-3*x^7-x^6-5*x^5-2*x^4-x^3-3*x^2-1,x+1]'
db_entry = '9999875:9999875:[x^8+3*x^7-6*x^5-4*x^4,x^4+x^3+x+1]'
db_entry = '9999872:4999936:[x^7-x^4+x^3-x^2,x^2+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9999868:4999934:[2*x^5+6*x^4+5*x^3+x^2+x+1,x^4+x^3+x]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4

db_entry = '9999609:9999609:[-3*x^6-6*x^5-8*x^4-4*x^3-x^2+x,x^4+x^2+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 1

db_entry = '9999469:9999469:[-x^7+2*x^6+x^5-5*x^4+x^3+2*x^2-2*x,x^3+x^2+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(1)] # just the x values
TERMINATE_WHEN_6 = 1

db_entry = '9998993:9998993:[x^7+x^6-4*x^5+x^4+4*x^3-3*x^2-x+1,x^2]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9998809:9998809:[x^7-3*x^6-3*x^5+5*x^4-2*x^3-4*x^2+2*x-1,x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry='9998659:9998659:[-x^6+3*x^4-7*x^2-12*x-9,x^4+x^3+x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

db_entry = '9998263:9998263:[3*x^7+x^6-3*x^5-2*x^4+10*x^3-12*x^2+5*x-1,x^4+x^2+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9998039:9998039:[x^4+2*x^3+x^2+x+1,x^4+x^3+x^2]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9997263:3332421:[x^7+x^6-4*x^5-2*x^4+x^3-x,x^4+x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4

db_entry = '9997256:9997256:[x^7+x^6-2*x^5-5*x^4-x^3+2*x^2-1,x^4+x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9997199:9997199:[3*x^3+x^2-2*x,x^4+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

db_entry = '9996912:3332304:[x^5+2*x^4+x^3-x^2-2*x-1,x^4+x^3+x^2]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

db_entry = '9996680:2499170:[-x^7-x^6+8*x^5-13*x^4+12*x^3-6*x^2+x,x^4+x]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9996392:2499098:[x^8+3*x^7-2*x^6-8*x^5+3*x^4+7*x^3-5*x^2-2*x+1,x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9995673:9995673:[-x^7+4*x^6-7*x^5+4*x^4-x^3-2*x^2,x^3+x+1]'
db_entry = '9996294:9996294:[2*x^8+x^6-6*x^5+2*x^2-2*x,x^3+x+1]'
db_entry = '9995549:9995549:[x^8+3*x^7+2*x^6+x^5+3*x^4+x^3+x,x]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3


db_entry = '9995167:9995167:[-x^7+5*x^6-4*x^5-12*x^4+6*x^3+8*x^2+2*x,x^3+x+1]'
db_entry = '9995087:9995087:[-x^7-x^6-2*x^5+x^2,x^4+x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

db_entry = '9995008:4997504:[-x^8+5*x^6-x^5-8*x^4+4*x^3+4*x^2-4*x,x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 1

db_entry = '9995008:624688:[x^7-x^6-3*x^5+x^4-x^2,x^3+x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9994635:3331545:[x^7+2*x^6-x^5+8*x^3+3*x^2-5*x-2,x^4+x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4

db_entry = '9996352:312386:[-2*x^6-6*x^5+x^4+18*x^3+10*x^2-17*x-15,x^4+x^3+x]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(2), QQ(-30), QQ(33), QQ(-10), QQ(1)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 4

# prestige curve lol
COEFFS_GENUS2 = [QQ(1), QQ(8), QQ(10), QQ(-10), QQ(-11), QQ(2), QQ(1)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 11


COEFFS_GENUS2 = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 12

db_entry = '9995456:2498864:[2*x^7-4*x^6-5*x^5+10*x^4+5*x^3-8*x^2-3*x+1,x^2+x]'
db_entry = '9995408:2498852:[x^8-x^6+x^3+2*x^2+x,x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 5


##### END TEST CURVES ######


# BEGIN STATIC CONFIG (default config; mostly deprecated)


HEIGHT_BOUND = 370 # not that important, mostly, it seems

# prime config
# magic prime settings, chosen empirically.
#PRIME_POOL = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
#PRIME_POOL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
PRIME_POOL = list(primes(90))   # All primes less than N, excluding 2,3; >=50 should be good... might need more for high height points!
MIN_PRIME_SUBSET_SIZE = 3 # just keep this at 3
MIN_MAX_PRIME_SUBSET_SIZE = 9 # safe is 7-9; above 15 is too stringent
NUM_PRIME_SUBSETS = 1000 # important for stability under different seeds, must be large enough >= 250 should be good...
MAX_MODULUS = 10**9 # idk
NUM_SAMPLES_HEIGHT_MAT = 10 # seems not important
HEIGHT_BOUND_NON_MINIMAL = 2*HEIGHT_BOUND # New bound for non-minimal models, just double the minimal one lol  # 420 blaze it
###### END STATIC CONFIG


# random seed for reproducibility.
SEED_INT = random.randint(-10**6, 10**6)

DEBUG = False
DEBUG = True
USE_MINIMAL_MODEL = False # uses the generic fiber
USE_MINIMAL_MODEL = True # more correct, and more slow
SYMBOLIC_SEARCH = True   # the search over Q (often slower, usually doesn't find anything)
SYMBOLIC_SEARCH = False   # mod p search (usually faster; the default)



try:
    PROFILE = profile
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


# This defines the data structure used to pass around curve information.
#CurveData = namedtuple('CurveData', ['E_curve', 'E_weier', 'E_rhs', 'a4', 'a6', 'phi_x', 'blowup_factor',
#                                     'quartic_rhs', 'tate_exponent', 'k_base_change', 'bad_primes',
#                                     'morphs', 'use_minimal', 'singfibs'])


class CurveDataExt(NamedTuple):
    E_curve: object
    E_weier: object
    E_rhs: object
    a4: object
    a6: object
    phi_x: object
    quartic_rhs: object
    tate_exponent: int
    k_base_change: int
    bad_primes: list
    morphs: tuple
    use_minimal: bool
    blowup_factor: int
    singfibs: list
    # New SR-coerced versions
    SR_a4: object
    SR_a6: object
    SR_phi_x: object
    SR_m: object



# --- START: Modular Reduction Helpers (centralized from picard.py) ---

def to_mod_poly(poly_q, R, debug=False):
    """
    Coerce `poly_q` (polynomial-like over QQ or FractionField) into R = PolynomialRing(GF(ell), 'm').
    """
    try:
        if poly_q.parent() is R:
            return poly_q
    except Exception:
        pass
    try:
        return R(poly_q)
    except Exception as e_direct:
        if debug:
            print(f"[debug to_mod_poly] direct coercion failed: {e_direct}")
    try:
        PQ = PolynomialRing(QQ, 'm')
        poly_QQ = PQ(poly_q)
    except Exception as e_pq:
        raise RuntimeError(f"Cannot coerce to QQ polynomial: {e_pq}")

    coeffs = list(poly_QQ.list())
    dens = [int(QQ(c).denominator()) for c in coeffs]
    lcm_val = 1
    for d in dens:
        lcm_val = lcm_val * d // gcd(lcm_val, d)

    B = R.base_ring()
    char = int(B.characteristic())
    if char != 0 and (lcm_val % char == 0):
        raise RuntimeError(f"Cannot clear rational denominators: lcm({set(dens)}) = {lcm_val} is NOT invertible mod {char}.")

    mF = R.gen()
    res = R(0)
    for i, c in enumerate(coeffs):
        int_coeff = int(QQ(c) * lcm_val)
        res += B(int_coeff) * (mF**i)

    if char != 0:
        inv_lcm = B(lcm_val).inverse()
        res *= inv_lcm

    return res



def reduce_cd_mod_ell(cd, ell, debug=False):
    """
    Robust reduction of cd.a4, cd.a6 to GF(ell)(m) rational functions.
    """
    ell_int = int(ell)
    if ell_int < 2 or not is_prime(ell_int):
        raise ValueError(f"ell must be a prime; got {ell_int}")

    F = GF(ell_int)
    R = PolynomialRing(F, 'm')
    R_frac = R.fraction_field()
    mF = R.gen()

    try:
        Delta = -16 * (4 * cd.a4**3 + 27 * cd.a6**2)
    except Exception as exc:
        raise RuntimeError("cd.a4 / cd.a6 not usable to build discriminant.") from exc

    try:
        a4_num, a4_den = cd.a4.numerator(), cd.a4.denominator()
        a6_num, a6_den = cd.a6.numerator(), cd.a6.denominator()
        Delta_num, Delta_den = Delta.numerator(), Delta.denominator()
    except Exception as exc:
        raise RuntimeError("Could not access numerator()/denominator() on cd.a4/a6/Delta.") from exc

    try:
        a4_num_mod = to_mod_poly(a4_num, R, debug=debug)
        a4_den_mod = to_mod_poly(a4_den, R, debug=debug)
        a6_num_mod = to_mod_poly(a6_num, R, debug=debug)
        a6_den_mod = to_mod_poly(a6_den, R, debug=debug)
        Delta_num_mod = to_mod_poly(Delta_num, R, debug=debug)
        Delta_den_mod = to_mod_poly(Delta_den, R, debug=debug)
    except Exception as exc:
        raise RuntimeError(f"Polynomial coercion to GF({ell_int})[m] failed: {exc}") from exc

    if a4_den_mod == 0 or a6_den_mod == 0 or Delta_den_mod == 0 or Delta_num_mod == 0:
        raise RuntimeError(f"Denominator or discriminant reduces to zero mod {ell_int}; bad prime.")

    a4_mod = R_frac(a4_num_mod) / R_frac(a4_den_mod)
    a6_mod = R_frac(a6_num_mod) / R_frac(a6_den_mod)

    class CDMod:
        pass

    cd_ell = CDMod()
    cd_ell.a4 = a4_mod
    cd_ell.a6 = a6_mod
    cd_ell.base_field = F
    cd_ell.m_symbol = mF
    return cd_ell


def is_good_prime_for_surface(cd, ell):
    """
    Check whether `ell` is a usable prime for reduction of the surface `cd`.
    A prime is bad if reduction fails or the discriminant collapses.
    """
    try:
        ell_int = int(ell)
    except Exception:
        return False

    if ell_int < 2 or not is_prime(ell_int):
        return False
        
    try:
        for ff in (cd.a4, cd.a6):
            den = ff.denominator()
            for c in den.coefficients():
                if QQ(c).denominator() % ell_int == 0:
                    return False
    except Exception:
        return False
        
    try:
        cd_ell = reduce_cd_mod_ell(cd, ell_int, debug=False)
    except Exception:
        return False

    try:
        Delta = -16 * (4 * cd_ell.a4**3 + 27 * cd_ell.a6**2)
        if Delta.numerator() == 0:
            return False
    except Exception:
        return False

    return True

# --- END: Modular Reduction Helpers ---


# Put these functions near the top-level of your script (so they are picklable).

def _worker_build_spec_from_serial(a4_str, a6_str, sect_triples_strs, m_val, precision_levels, factor):
    """
    Reconstruct small objects inside worker: a4(m), a6(m), and section coordinates.
    Returns a rational height matrix or raises.
    """
    try:
        # Reconstruct symbolic environment
        Fm = PolynomialRing(QQ, 'm')
        m = Fm.gen()
        # Convert strings back to expressions in the Sage SR environment
        # Use sage_eval to ensure we get QQ polynomials / rational functions
        a4_sym = sage_eval(a4_str, locals={'m': m, 'QQ': QQ, 'SR': SR})
        a6_sym = sage_eval(a6_str, locals={'m': m, 'QQ': QQ, 'SR': SR})

        # Evaluate coefficients at this m_val
        a4_spec = QQ(a4_sym.subs({m: m_val}))
        a6_spec = QQ(a6_sym.subs({m: m_val}))

        # Skip huge denominators
        if abs(int(a4_spec.denominator())) > 10**500 or abs(int(a6_spec.denominator())) > 10**500:
            raise ValueError("Coefficients too large for this m")

        E_spec = EllipticCurve(QQ, [QQ(0), QQ(0), QQ(0), a4_spec, a6_spec])

        if E_spec.discriminant() == 0:
            raise ValueError("Singular fiber at m")

        # build specialized points
        specialized_points = []
        for triple in sect_triples_strs:
            # triple is (X_str, Y_str, Z_str)
            X_expr = sage_eval(triple[0], locals={'m': m, 'QQ': QQ, 'SR': SR})
            Y_expr = sage_eval(triple[1], locals={'m': m, 'QQ': QQ, 'SR': SR})
            Z_expr = sage_eval(triple[2], locals={'m': m, 'QQ': QQ, 'SR': SR})

            X_val = X_expr.subs({m: m_val})
            Y_val = Y_expr.subs({m: m_val})
            Z_val = Z_expr.subs({m: m_val})

            # Convert to rationals
            Xq = QQ(X_val)
            Yq = QQ(Y_val)
            Zq = QQ(Z_val)

            if Zq == 0:
                # projective point at infinity or invalid -- use E_spec(0) (neutral)
                specialized_points.append(E_spec(0))
            else:
                try:
                    P = E_spec([Xq, Yq, Zq])
                except Exception:
                    # Try affine normalization (X/Z^2, Y/Z^3) if construction failed
                    X_aff = Xq / (Zq**2)
                    Y_aff = Yq / (Zq**3)
                    specialized_points.append(E_spec([X_aff, Y_aff]))
                else:
                    specialized_points.append(P)

        # Now compute a numeric height matrix using increasing precision levels
        n = len(specialized_points)
        H_spec_real = None
        for prec in precision_levels:
            try:
                # compute canonical heights (numerical) for each point
                h_list = [specialized_points[i].height(precision=prec, normalised=True) for i in range(n)]
                M = matrix(RR, n)
                for i in range(n):
                    for j in range(i, n):
                        hpq = (specialized_points[i] + specialized_points[j]).height(precision=prec, normalised=True)
                        val = 0.5 * (hpq - h_list[i] - h_list[j])
                        M[i, j] = M[j, i] = val
                H_spec_real = M
                break
            except Exception:
                # try next precision
                continue

        if H_spec_real is None:
            raise RuntimeError("Failed to compute numeric height matrix for m=%s" % str(m_val))

        # Convert to rational matrix with factor
        H_spec_rational = matrix(QQ, n)
        for i in range(n):
            for j in range(n):
                x = float(H_spec_real[i, j])
                n_int = int(round(x * factor))
                H_spec_rational[i, j] = QQ(n_int) / QQ(factor)

        return H_spec_rational

    except Exception as e:
        # Give a full traceback to stdout so you can debug in process logs
        print("Exception in _worker_build_spec_from_serial for m_val =", m_val)
        traceback.print_exc()
        raise


def compute_coarse_height_matrix_serializable(cd, sections,
                                              num_samples=NUM_SAMPLES_HEIGHT_MAT, max_coord=200,
                                              decimal_places=2, min_integer_samples=3):
    """
    A safe, multiprocessing-friendly replacement for compute_coarse_height_matrix.
    It serializes only small strings (a4, a6, and section expressions) and reconstructs
    them inside worker processes to avoid pickling heavy Sage objects.
    Returns an average rational matrix built from successful samples.
    """
    # Build serializable representations
    a4_str = str(cd.a4)   # polynomial/rational function of m
    a6_str = str(cd.a6)
    # sections: each P is a Sage point with projective coords P[0], P[1], P[2]
    sect_triples_strs = []
    for P in sections:
        # P elements may be callable morphism components or a proper point in cd.E_weier
        # If P is a Sage point (iterable), stringify coords; if tuple-like of callables, we assume they are expressions and convert str()
        try:
            # If P is an EllipticCurve point object (special projective coords)
            Xs = str(P[0])
            Ys = str(P[1])
            Zs = str(P[2])
        except Exception:
            # Otherwise assume it's a triple of expressions or callables already stringifiable
            Xs = str(P[0])
            Ys = str(P[1])
            Zs = str(P[2])
        sect_triples_strs.append((Xs, Ys, Zs))

    factor = QQ(10**decimal_places)
    precision_levels = [80, 120, 200, 400]
    TIMEOUT_SECONDS = 12

    # Build m candidates (deterministic + random)
    m_candidates = set()
    # deterministic integers
    for m_int in range(-max_coord, max_coord + 1):
        m_candidates.add(QQ(m_int))
        if len(m_candidates) >= num_samples + min_integer_samples:
            break

    # add some random rationals if needed
    import random
    while len(m_candidates) < (num_samples + min_integer_samples):
        rn = random.randint(-max_coord, max_coord)
        rd = random.randint(1, max(1, max_coord))
        m_candidates.add(QQ(rn) / QQ(rd))

    m_candidates = list(m_candidates)

    total_H = matrix(QQ, len(sections))
    valid_samples = 0

    # We'll spawn a pool and run tasks. The worker is _worker_build_spec_from_serial.
    with multiprocessing.Pool(processes=min(len(m_candidates), multiprocessing.cpu_count())) as pool:
        # Kick off tasks
        async_results = []
        for m_val in m_candidates:
            async_results.append(pool.apply_async(_worker_build_spec_from_serial,
                                                 (a4_str, a6_str, sect_triples_strs, m_val, precision_levels, int(factor))))

        # iterate with a progress bar and collect
        for ar in async_results:
            try:
                H_spec = ar.get(timeout=TIMEOUT_SECONDS)
                if H_spec is not None and H_spec.nrows() == len(sections):
                    total_H += H_spec
                    valid_samples += 1
            except multiprocessing.TimeoutError:
                print("Timeout computing height sample.")
            except Exception:
                # Show full traceback from the worker (should already have printed it)
                print("Worker failed for one m candidate. See above trace.")
                continue

    if valid_samples < min(len(sections), 3):
        raise RuntimeError(f"compute_coarse_height_matrix_serializable: insufficient valid samples ({valid_samples})")

    H_avg = total_H / QQ(valid_samples)
    if DEBUG:
        print(f"compute_coarse_height_matrix_serializable: built coarse H from {valid_samples} samples")
    return H_avg


@PROFILE
def lll_reduce_mw_basis(cd, P_list):
    r = len(P_list)
    if r == 0:
        return []
    
    # Use the same logic as check_independence to get the right height matrix
    is_independent, H = check_independence(P_list, cd.E_curve, cd)

    if not is_independent:
        print("Warning: height matrix not full rank. Skipping LLL.")
        return P_list

    denoms = [H[i,j].denominator() for i in range(r) for j in range(r)]
    D = lcm(denoms) if denoms else 1
    H_int = (H * D).change_ring(ZZ)

    try:
        U = H_int.LLL_gram()
    except ValueError:
        print("H_int not LLL-compatible:", H_int)
        return P_list

    new_Ps = []
    for i in range(r):
        comb = sum(U[j,i] * P_list[j] for j in range(r))
        new_Ps.append(comb)
    return new_Ps

@PROFILE
def deg_height(P):
    """
    Return max(degree of num(m), degree of den(m)) for x-coordinate of P.
    """
    f = P[0] / P[2]
    num = f.numerator()
    den = f.denominator()
    return max(num.degree(), den.degree())

@PROFILE
def naive_pairing(P, Q):
    H_P  = deg_height(P)
    H_Q  = deg_height(Q)
    H_PQ = deg_height(P + Q)
    return (H_PQ - H_P - H_Q) // 2


def canonicalize_by_sign(vecs):
    seen = set()
    out = []
    for v in vecs:
        vt = tuple(int(x) for x in v)
        if all(x == 0 for x in vt):
            continue
        neg = tuple(-x for x in vt)
        can = vt if vt <= neg else neg
        if can not in seen:
            seen.add(can)
            out.append(vt)
    return out



def compute_search_vectors(H, height_bound):
    print("height bound:", height_bound)

    H_matrix = matrix(H)
    denominators = [H_matrix[i,j].denominator()
                    for i in range(H_matrix.nrows())
                    for j in range(H_matrix.ncols())]
    lcm_denom = lcm(denominators) if denominators else 1

    H_scaled = lcm_denom * H_matrix        # now integer entries
    H_even = 2 * H_scaled                  # make diagonal even initially

    # Ensure exact symmetry (defensive)
    H_even = (H_even + H_even.transpose())  # still exact; should remain integer

    # Quick diagnostics
    try:
        is_pd = H_even.is_positive_definite()
    except Exception:
        is_pd = False

    if not is_pd:
        det = H_even.det()
        print("Gram initially not PD (det,rank) =", det, H_even.rank())
        R = RealField(80)
        print("approx min eigenvalue (before shift) =",
              min([RR(ev) for ev in H_even.eigenvalues()]))

        # Find an integer eps by doubling that makes H_even + eps*I PD
        eps = Integer(1)
        H_try = H_even + eps * Matrix.identity(H_even.nrows())
        attempts = 0
        while not H_try.is_positive_definite():
            eps *= 2
            H_try = H_even + eps * Matrix.identity(H_even.nrows())
            attempts += 1
            if attempts > 60:
                raise RuntimeError("Could not regularize Gram matrix after many attempts")

        # Make eps even (QuadraticForm wants even diagonal). If odd, make it eps+1 (even).
        if eps % 2 == 1:
            eps += 1
            H_try = H_even + eps * Matrix.identity(H_even.nrows())

        # If that even eps is still not PD (rare), step by +2 until PD
        while not H_try.is_positive_definite():
            eps += 2
            H_try = H_even + eps * Matrix.identity(H_even.nrows())

        # Try to minimize eps (binary search over even values between 0 and current eps)
        lo = Integer(0)
        hi = Integer(eps)
        while hi - lo > 2:
            mid = lo + (hi - lo) // 2
            if mid % 2 == 1:
                mid += 1
            if (H_even + mid * Matrix.identity(H_even.nrows())).is_positive_definite():
                hi = mid
            else:
                lo = mid
        eps = hi

        # apply final even shift
        H_even = H_even + eps * Matrix.identity(H_even.nrows())
        print("Added eps*I with eps =", eps, "to make Gram PD")
        # final eigen diag
        print("new approx min eigenvalue =",
              min([RR(ev) for ev in H_even.eigenvalues()]))

    # ensure integer matrix over ZZ with even diagonal
    H_even = matrix(ZZ, H_even)   # cast to integer matrix (should be integral now)
    # last sanity: assert diagonal even
    if any([int(H_even[i,i]) % 2 != 0 for i in range(H_even.nrows())]):
        raise RuntimeError("Diagonal still not even after adjustments; unexpected")

    # build quadratic form and enumerate
    Q = QuadraticForm(ZZ, H_even)
    scaled_height_bound = 2 * lcm_denom * height_bound

    vecs = Q.short_vector_list_up_to_length(scaled_height_bound)
    vecs = [v for sublist in vecs for v in sublist]
    return vecs

@PROFILE
def check_independence(sections, curve, cd):
    """
    Check linear independence of a list of sections on the Mordell-Weil group.
    Returns (independent_bool, height_matrix_H).

    For minimal models, use the canonical height calculation.
    For non-minimal models, use the coarse sampled height matrix; if that
    fails, fall back to naive_pairing.
    """
    n = len(sections)
    if n == 0:
        return False, matrix(QQ, 0)

    H = None

    if USE_MINIMAL_MODEL:
        # canonical, exact route for minimal models
        H = compute_canonical_height_matrix(sections, cd)
    else:
        # coarse sampled approximation for non-minimal models
        print("--- Estimating non-minimal height matrix via sampling ---")
        H = compute_coarse_height_matrix_serializable(cd, sections)

        if H is None:
            # coarse sampling failed — fall back to naive pairing (last resort)
            print("Coarse sampling produced no valid samples. Falling back to naive pairing.")
            H_naive = matrix(QQ, n)
            for i in range(n):
                for j in range(i, n):
                    val = naive_pairing(sections[i], sections[j])
                    H_naive[i, j] = val
                    H_naive[j, i] = val
            H = H_naive

    if H is None or H.nrows() == 0:
        return False, matrix(QQ, 0)

    det = H.det()
    print(f"Canonical height pairing matrix determinant = {det}")
    independent = (det != 0)
    print(f"Sections are {'independent' if independent else 'dependent'}.")
    return independent, H



@PROFILE
def summarize_fibration_info(cd, data_pts, sections):
    print("=== Fibration Summary ===")
    print(f"Data points: {data_pts}")
    print(f"Genus 1 quartic: {cd.E_curve}")

    polys = cd.E_curve.defining_polynomials()
    if not polys:
        raise ValueError("No defining polynomial found.")

    f = polys[0]
    print("Defining polynomial f:", f)

    vars = f.parent().gens()
    disc_var = vars[-1] if len(vars) > 1 else vars[0]
    disc = f.discriminant(disc_var)
    print("Discriminant of f:", disc)

    disc2 = cd.E_weier.discriminant()
    print("Weierstrass discriminant:", disc2)

    a4 = cd.a4
    a6 = cd.a6
    print("Weierstrass model coefficients:")
    print("  a4(m):", a4)
    print("  a6(m):", a6)

    for i, P in enumerate(sections, 1):
        print(f"P{i}:", P)

    if DEBUG:
        pass
        #estimate_generic_rank_bound(cd)

    validate_fibration_geometry(cd)
    print("=========================")


@profile
def get_phi_x(one, two, three, x_coord_func, quartic_rhs):
    """
    Compute phi_x = X_sub / Z_sub without global simplification.
    Assumes X_sub, Z_sub live in the same parent (symbolic or rational function field).
    Raises on Z_sub == 0 to avoid silent garbage.
    """
    Z_sub = three.subs(x=x_coord_func, y=sqrt(quartic_rhs))

    if Z_sub == 0:
        raise ZeroDivisionError("get_phi_x: Z_sub is zero")
    X_sub = one.subs(x=x_coord_func, y=sqrt(quartic_rhs))

    phi_x = X_sub / Z_sub
    phi_x = phi_x.simplify_rational()

    return phi_x


@PROFILE
def _effective_degree(rational_expr, m):
    """
    Robust effective degree of a rational function: deg(numerator) - deg(denominator).
    """
    num = rational_expr.numerator()
    den = rational_expr.denominator()
    def _deg(poly):
        try:
            return int(poly.degree())
        except Exception:
            pass
        try:
            fac = poly.factor()
            deg = 0
            for base, exp in fac:
                try:
                    if base == m:
                        deg += int(exp)
                except Exception:
                    continue
            if deg:
                return deg
        except Exception:
            pass
        try:
            R = PolynomialRing(QQ, str(m))
            p = R(poly)
            return int(p.degree())
        except Exception:
            return 0
    return _deg(num) - _deg(den)

def _refresh_state(a4_final, a6_final, Fm):
    var('m')
    E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
    Delta_final = E_weier_final.discriminant()
    # raw degree (no cancellation) used for minimality check
    deg_delta_raw = _effective_degree(Delta_final, m)
    # effective_discriminant_degree returns (deg_after_cancel, removed_k)
    deg_delta_effective, _removed_k = effective_discriminant_degree(Delta_final)
    fiber_info = find_singular_fibers(a4=a4_final, a6=a6_final, verbose=True)
    euler_sum = int(fiber_info['euler_characteristic'])
    sigma_sum = int(fiber_info.get('sigma_sum', 0))
    # return values for convenience
    ret = [E_weier_final, Delta_final, deg_delta_raw, deg_delta_effective, fiber_info, euler_sum, sigma_sum]
    return deg_delta_raw, deg_delta_effective, euler_sum, sigma_sum, ret


# Small helper: raw effective degree (no global cancellation)
def _effective_degree_raw(rational_expr, m_sym):
    """
    Degree = deg(numerator) - deg(denominator) without canceling global m^k.
    Returns integer (may be negative if there is a pole).
    """
    num = rational_expr.numerator()
    den = rational_expr.denominator()
    try:
        return int(num.degree()) - int(den.degree())
    except Exception as exc:
        raise RuntimeError("_effective_degree_raw: degree extraction failed") from exc


# At the end of buildcd, just before returning cd:
# -------------------------------------------------
# 5) Compute global bad primes
def get_primes_from_poly(ff):
    """
    Return a set of integer primes that divide any numerator or denominator
    of the rational coefficients appearing in `ff` (which is expected to be
    a FractionField element a4/a6 or similar).

    - Only integer primes are returned.
    - Symbolic polynomial factors or 'm' are ignored.
    - Robust against unexpected types; returns empty set on failure.
    """
    primes = set()

    # helper: add primes from a (possibly rational) coefficient c
    def add_primes_from_coeff(c):
        try:
            q = QQ(c)            # try to coerce coefficient to a rational
        except Exception:
            return
        try:
            N = Integer(q.numerator())
            D = Integer(q.denominator())
        except Exception:
            return
        if abs(N) > 1:
            for p, _ in N.factor():
                primes.add(int(p))
        if D > 1:
            for p, _ in D.factor():
                primes.add(int(p))

    # 1) Try numerator()/denominator() API (works for FractionField elements)
    try:
        num = ff.numerator()
        den = ff.denominator()
    except Exception:
        num = ff
        den = None

    # 2) For each (num, den) gather rational coefficient primes.
    for poly in (num, den):
        if poly is None:
            continue
        # If poly provides coefficients (typical for polynomial numerators/denoms)
        if hasattr(poly, "coefficients"):
            try:
                coeffs = list(poly.coefficients())
            except Exception:
                coeffs = [poly]
        else:
            coeffs = [poly]

        for c in coeffs:
            add_primes_from_coeff(c)

    # tidy: remove 0/1 if any sneaked in
    primes = {p for p in primes if isinstance(p, int) and p > 1}
    return primes


# ---- buildcd replacement ----
@PROFILE
def buildcd(E_curve, phi_x, quartic_rhs, E_rhs, morph_triplet,
            verify=True, compute_minimal=USE_MINIMAL_MODEL):
    """
    Builds the CurveDataExt object for the fibration.

    If compute_minimal is True, it computes the minimal Weierstrass model and applies
    the necessary transformations to the coordinate maps.

    If compute_minimal is False, it uses the raw, non-minimal model directly from the
    Jacobian, which is often faster for searching.
    """
    print("--- Entering buildcd ---")
    E_weier_raw = Jacobian(E_curve)
    a4_raw = E_weier_raw.a4()
    a6_raw = E_weier_raw.a6()
    Fm = a4_raw.parent()
    m = Fm.gen()
    y = var('y'); x = var('x')

    # Initialize variables
    one_s, two_s, three_s = None, None, None
    one, two, three = morph_triplet

    if compute_minimal:
        print("--- Computing Minimal Model ---")
        a4_final, a6_final = a4_raw, a6_raw
        phi_x_final = phi_x
        blowup_factor = 0
        blowdown_0 = 0

        # Step 1: Handle poles at m=0 (blow-up)
        v4 = min_order_in_m(a4_raw, m)
        v6 = min_order_in_m(a6_raw, m)
        if v4 < 0 or v6 < 0:
            k_for_a4 = ceil(-v4 / 4)
            k_for_a6 = ceil(-v6 / 6)
            blowup_factor = int(max(k_for_a4, k_for_a6))
            if blowup_factor > 0:
                print(f"Applying blow-up with k={blowup_factor} to handle poles at m=0")
                a4_final = a4_raw * m**(4 * blowup_factor)
                a6_final = a6_raw * m**(6 * blowup_factor)
                phi_x_final = phi_x / (m**(2 * blowup_factor))

        # Step 2: Handle common zeros at m=0 (blow-down)
        while True:
            v4_0 = min_order_in_m(a4_final, m)
            v6_0 = min_order_in_m(a6_final, m)
            k0 = int(min(v4_0 // 4, v6_0 // 6)) if (v4_0 > 0 and v6_0 > 0) else 0

            if k0 <= 0:
                break
            print(f"Applying blow-down with k={k0} to handle zeros at m=0")
            a4_final = a4_final / (m**(4 * k0))
            a6_final = a6_final / (m**(6 * k0))
            phi_x_final = phi_x_final * (m**(2 * k0))
            blowdown_0 += k0

        E_weier_final = EllipticCurve(Fm, [0,0,0, a4_final, a6_final])

        # Step 3: Apply net scaling transformation to morphisms
        net_k = int(blowup_factor - blowdown_0)
        x_morphism_scale = m**(2 * net_k)
        y_morphism_scale = m**(3 * net_k)

        one_s = MorphismWrapper(one, 1, x_morphism_scale, a4_final)
        two_s = MorphismWrapper(two, 1, y_morphism_scale, a4_final)
        three_s = MorphismWrapper(three, 1, 1, a4_final) # Z is unscaled

    else: # Use the raw, non-minimal model
        print("--- Using non-minimal model, skipping all rescaling ---")
        a4_final = a4_raw
        a6_final = a6_raw
        phi_x_final = phi_x
        blowup_factor = 0
        E_weier_final = E_weier_raw

        # Morphisms are used without any scaling
        one_s = MorphismWrapper(one, 1, 1, a4_final)
        two_s = MorphismWrapper(two, 1, 1, a4_final)
        three_s = MorphismWrapper(three, 1, 1, a4_final)

    # --- FORCE canonical SR forms for diagnostics & downstream substitution ---
    # canonical symbolic var for substitutions
    SR_m = var('m')

    # try to coerce the key returned objects into SR for consistent .subs({SR_m: ...})
    try:
        SR_a4 = SR(a4_final)
        SR_a6 = SR(a6_final)
        SR_phi_x = SR(phi_x_final)
    except Exception:
        raise AssertionError("buildcd: failed to coerce a4/a6/phi_x to SR; check types.")

    # quick consistency check: ensure same symbolic var name appears where expected
    try:
        a4_vars = [str(v) for v in SR_a4.variables()]
        phi_vars = [str(v) for v in SR_phi_x.variables()]
    except Exception:
        a4_vars = []
        phi_vars = []

    # record SR objects on CurveDataExt for downstream use
    cd_extra = {}
    cd_extra['SR_a4'] = SR_a4
    cd_extra['SR_a6'] = SR_a6
    cd_extra['SR_phi_x'] = SR_phi_x
    cd_extra['SR_m'] = SR_m

    # attach to cd later (after cd constructed)


    # --- Common logic for both models ---

    # Compute global bad primes
    print("\n--- Identifying Globally Bad Primes ---")
    class TempCD:
        def __init__(self, a4, a6): self.a4, self.a6 = a4, a6
    temp_cd = TempCD(a4_final, a6_final)
    bad_primes = [p for p in PRIME_POOL if not is_good_prime_for_surface(temp_cd, p)]
    print(f"Identified {len(bad_primes)} globally bad prime(s) from the pool: {sorted(bad_primes)}")

    try:
        E_rhs_final = y**2 - x**3 - a4_final * x - a6_final
    except Exception:
        E_rhs_final = E_rhs

    # Get singular fiber info for diagnostics
    singfibs = find_singular_fibers(a4=a4_final, a6=a6_final, verbose=True)

    # --- Package and return the final data ---

    # Promote to symbolic ring for consistency
    SR_a4 = SR(a4_final)
    SR_a6 = SR(a6_final)
    SR_phi_x = SR(phi_x_final)
    SR_m = SR(m)

    cd = CurveDataExt(
        E_curve=E_curve,
        E_weier=E_weier_final,
        E_rhs=E_rhs_final,
        a4=a4_final,
        a6=a6_final,
        phi_x=phi_x_final,
        quartic_rhs=quartic_rhs,
        tate_exponent=0,     # legacy, kept for compatibility
        k_base_change=1,
        bad_primes=bad_primes,
        morphs=(one_s, two_s, three_s),
        use_minimal=compute_minimal,
        blowup_factor=int(blowup_factor),
        singfibs=singfibs,
        SR_a4=SR_a4,
        SR_a6=SR_a6,
        SR_phi_x=SR_phi_x,
        SR_m=SR_m,
    )


    # attach SR copies for downstream code to use
    assert cd.SR_a4 == cd_extra['SR_a4']
    assert cd.SR_a6 == cd_extra['SR_a6']
    assert cd.SR_phi_x == cd_extra['SR_phi_x']
    assert cd.SR_m == cd_extra['SR_m']

    # Strict sanity: phi_x should mention the parameter SR_m (unless degenerate)
    try:
        phi_vars = [str(v) for v in SR(cd.SR_phi_x).variables()]
    except Exception:
        phi_vars = []
    assert (phi_vars == [] or str(cd.SR_m) in phi_vars), "buildcd: phi_x seems not to contain 'm' (vars=%s). This will break symbolic substitution." % (repr(phi_vars))

    print("--- Exiting buildcd ---")
    if verify and DEBUG:
        validate_fibration_geometry(cd)
        if cd.E_weier.discriminant().is_zero():
            raise ValueError("buildcd: Resulting discriminant is identically zero.")
    return cd

def to_rational(c):
    if c == 0:
        return QQ(0)
    if isinstance(c, (list, tuple)) and len(c) == 2:
        return QQ(c[0]) / QQ(c[1])
    return QQ(c)



def min_order_in_m(expr, m):
    """
    Find the minimum order of m in an expression using Sage's valuation.
    """
    if expr.is_zero():
        return float('inf')
    
    try:
        return expr.valuation(m)
    except:
        try:
            if hasattr(expr, 'numerator') and hasattr(expr, 'denominator'):
                num_val = expr.numerator().valuation(m) if not expr.numerator().is_zero() else float('inf')
                den_val = expr.denominator().valuation(m) if not expr.denominator().is_zero() else float('inf')
                return num_val - den_val
            else:
                return expr.valuation(m)
        except:
            print(f"WARNING: Could not compute valuation of {expr}")
            return 0


# The rationality test stays the same (cached)
@lru_cache(maxsize=None)
def get_y_unshifted_genus2(x):
    """
    Test if x gives a rational y on the genus-2 curve y^2 = G(x).
    Returns y if rational, None otherwise.
    """
    x = QQ(x)
    
    # Evaluate G(x) = sum of coeffs * x^i
    # Horner's method is faster than repeated exponentiation
    rhs = COEFFS_GENUS2[0]
    for coeff in COEFFS_GENUS2[1:]:
        rhs = rhs * x + coeff
    
    # Quick checks before expensive square root test
    num = ZZ(rhs.numerator())
    den = ZZ(rhs.denominator())
    
    if num < 0 or den <= 0:
        return None
    
    # Check if num and den are both perfect squares
    # Use Sage's is_square() which is optimized
    if not num.is_square() or not den.is_square():
        return None
    
    return QQ(num.sqrt()) / QQ(den.sqrt())

@PROFILE
def compute_morphism(E_rhs):
    # E_rhs_serialized should be a reproducible string key for E_rhs, e.g. str(E_rhs)
    #E_rhs = parse_E_rhs_from_string(E_rhs_serialized)  # adapt to your environment
    R = PolynomialRing(E_rhs.base_ring(), 2, names=('x', 'y'))
    x, y = R.gens()
    E_curve = Curve(R(y**2 - E_rhs))
    try:
        one, two, three = Jacobian(E_curve, morphism=True)
    except Exception:
        print("E_curve which is giving problem:", E_curve)
        raise
    return E_curve, one, two, three


class MorphismWrapper:
    """
    A picklable wrapper class to apply a base change (m -> m^k) and scaling
    to a morphism component. Replaces the un-picklable `_wrap` closure.
    """
    def __init__(self, callable_obj, k, scale, a4_min):
        self.callable_obj = callable_obj
        self.k = k
        self.scale = scale
        # Store the parent ring to reconstruct the generator 'm'
        self.parent_ring = a4_min.parent()

    def __call__(self, **kwargs):
        val = self.callable_obj(**kwargs)
        if self.k == 1:
            return val * self.scale

        # Reconstruct the variable 'm'
        m = self.parent_ring.gen()
        val_sym = SR(val)
        val_bc = val_sym.subs({m: m**self.k})
        return val_bc * self.scale


@PROFILE
def validate_fibration_geometry(cd):
    """
    Robust geometric validation of the fibration.
    """
    print("\n--- Geometric Validation ---")
    try:
        a4 = cd.a4
        a6 = cd.a6
        m_var = a4.parent().gen() if hasattr(a4, 'parent') else None

        Delta = -16 * (4 * a4**3 + 27 * a6**2)
        if Delta.is_zero():
            print("❌ Validation FAIL: Discriminant is identically zero.")
            print("--------------------------")
            return

        effective_degree = _effective_degree(Delta, m_var)

        print("Delta:", Delta)
        print(f"  Effective Discriminant Degree: {effective_degree}")
        stored_n = getattr(cd, 'tate_exponent', '<missing>')
        print(f"  Weierstrass scaling exponent (n) used: {stored_n}")

        if effective_degree == 12:
            print("  ✅ PASS: Discriminant degree is 12. Standard geometric checks should apply.")
        else:
            print(f"  ⚠ NOTICE: Discriminant degree is {effective_degree}, not 12.")
            print("    This fibration may be non-minimal or use a different base parameterization.")
    except Exception as e:
        print(f"An error occurred during geometric validation: {e}")
        raise
    print("--------------------------")

@PROFILE
def find_cm_fibers(cd):
    """
    Finds fibers with potential Complex Multiplication by finding rational
    roots of a4(m) and a6(m), which correspond to j=1728 and j=0 respectively.
    """
    m = cd.a4.parent().gen()
    fibers = set()
    print("\n--- Searching for CM Fibers ---")

    a4_num = cd.a4.numerator()
    if not a4_num.is_constant():
        for f, _ in a4_num.factor():
            roots = f.roots(ring=QQ, multiplicities=False)
            if roots:
                print(f"Found roots for a4(m)=0 (potential j=1728): {roots}")
                fibers.update(roots)

    a6_num = cd.a6.numerator()
    if not a6_num.is_constant():
        for f, _ in a6_num.factor():
            roots = f.roots(ring=QQ, multiplicities=False)
            if roots:
                print(f"Found roots for a6(m)=0 (potential j=0): {roots}")
                fibers.update(roots)

    return list(fibers)

@PROFILE
def find_special_j_invariant_fibers(cd, j_invariants_to_check):
    """
    Finds m-values where the fibration's j-invariant matches a target value.

    This is useful for finding fibers with Complex Multiplication (CM) or other
    arithmetically significant properties.

    Args:
        cd (CurveData): The curve data object containing a4 and a6.
        j_invariants_to_check (list): A list of QQ rational numbers representing
                                      the j-invariants to check for.

    Returns:
        set: A set of all rational m-values found.
    """
    print(f"\n--- Searching for special fibers via j-invariants: {j_invariants_to_check} ---")
    m = cd.a4.parent().gen()
    a4, a6 = cd.a4, cd.a6

    # The j-invariant is j = 1728 * (4*a4^3) / (4*a4^3 + 27*a6^2)
    # Let D = 4*a4^3 + 27*a6^2. The equation is j_target = 1728 * (4*a4^3) / D
    # j_target * D = 1728 * (4*a4^3)
    # j_target * (4*a4^3 + 27*a6^2) = 1728 * (4*a4^3)
    # j_target * 27*a6^2 = (1728 - j_target) * 4*a4^3

    found_m_values = set()

    for j_target in j_invariants_to_check:
        j_target = QQ(j_target)

        # Handle the special, simpler cases first.
        if j_target == 1728: # Equation simplifies to a6^2 = 0
            poly_to_solve = cd.a6.numerator()
        elif j_target == 0: # Equation simplifies to a4^3 = 0
            poly_to_solve = cd.a4.numerator()
        else:
            # General case
            lhs = j_target * 27 * a6**2
            rhs = (1728 - j_target) * 4 * a4**3
            poly_to_solve = (lhs - rhs).numerator()

        if poly_to_solve.is_constant() and not poly_to_solve.is_zero():
            continue

        try:
            roots = poly_to_solve.roots(ring=QQ, multiplicities=False)
            if roots:
                print(f"Found roots for j(m) = {j_target}: {roots}")
                found_m_values.update(roots)
        except Exception as e:
            print(f"Could not solve for j(m) = {j_target}: {e}")

    return found_m_values

@profile
def test_y_rationality_genus2(m_candidates, r_m, shift):
    """Tests if m values lead to rational points on the original sextic."""
    found = set()
    for m_val in set(m_candidates):
        try:
            x = r_m(m=m_val) - shift
            y = get_y_unshifted_genus2(x)
            if y is not None:
                found.add(x)
                print(f"Found rational point from fiber m={m_val}: (x,y) = ({x}, {y})")
        except (TypeError, ZeroDivisionError):
            continue
    return found




@PROFILE
def compute_base_sections_m(cd, base_pts):
    """
    Compute base sections for the fibration.
    This function uses the morphisms stored in the CurveData object.
    """
    one_use, two_use, three_use = cd.morphs

    ret = []
    seen = set()
    for xi, yi in base_pts:
        if (xi, yi) in seen or xi is None:
            continue

        X_aff = one_use(x=xi, y=yi)
        Y_aff = two_use(x=xi, y=yi)
        Z_aff = three_use(x=xi, y=yi)

        if DEBUG:
            print("\n--- DEBUGGING POINT CONSTRUCTION ---")
            print(f"Attempting to create point for (x,y) = ({xi}, {yi})")
            print("Curve:", cd.E_weier)
            print("\nPoint Coordinates (X_aff, Y_aff, Z_aff):")
            print("X_aff:", X_aff)
            print("Y_aff:", Y_aff)
            print("Z_aff:", Z_aff)
            try:
                LHS = Y_aff**2 * Z_aff
                RHS = X_aff**3 + cd.a4 * X_aff * Z_aff**2 + cd.a6 * Z_aff**3
                print("\nEquation Check:")
                print("LHS (Y^2*Z):", LHS)
                print("RHS (X^3 + a4*X*Z^2 + a6*Z^3):", RHS)

                diff = (LHS - RHS)
                print("\nDifference (LHS - RHS):", diff)
                if diff.is_zero():
                    print("✅ Algebraic identity holds.")
                else:
                    print("❌ Algebraic identity DOES NOT hold.")
            except Exception as e:
                print(f"An error occurred during manual check: {e}")
            print("--- END DEBUGGING ---")

        P = cd.E_weier([X_aff, Y_aff, Z_aff])
        print("section weierstrass coordinates:", P)
        ret.append(P)
        seen.add((xi, yi))
    return ret

@PROFILE
def verify_morphism_on_samples(cd, base_pts):
    """
    Verify images of base_pts under cd.morphs lie on cd.E_weier.
    """
    one_s, two_s, three_s = cd.morphs
    E_min = cd.E_weier
    for xi, yi in base_pts:
        try:
            X = one_s(x=xi, y=yi)
            Y = two_s(x=xi, y=yi)
            Z = three_s(x=xi, y=yi)
            P = E_min([X, Y, Z])
        except Exception as e:
            print(f"Error verifying morphism on sample point ({xi}, {yi}): {e}")
            raise
    return True


@PROFILE
def suggest_height_bound(H_ref, H_used, base_bound, safety=1.10, method='det'):
    n = H_used.nrows()
    try:
        if method == 'det':
            det_ref = float(H_ref.det())
            det_used = float(H_used.det())
            if det_ref > 0 and det_used > 0:
                alpha = (det_used / det_ref) ** (1.0 / float(n))
            else:
                alpha = 1.0
        else:  # 'trace' fallback
            trace_ref = sum(float(H_ref[i,i]) for i in range(n))
            trace_used = sum(float(H_used[i,i]) for i in range(n))
            alpha = (trace_used / trace_ref) if trace_ref != 0 else 1.0
    except Exception:
        alpha = 1.0

    used_bound = int(ceil(base_bound * alpha * safety))
    return used_bound, alpha

@PROFILE
def augment_known(known_pts, found, deg6=False):
    if DEBUG:
        print("known_pts", known_pts)
        print("found", found)
    ret = set(known_pts)
    known_x = set([i for i, _ in known_pts])
    for i in found:
        if i in known_x:
            continue
        print(f"new x: {i}")
        if deg6:
            rhsy = get_y_unshifted_genus2(i)
        else:
            rhsy = get_unshifted_y(i)
        ret.add((i, rhsy))
    return ret


def sections_to_ns_vectors(cd, sections, rho, mw_rank, chi):
    """
    Convert a list of Weierstrass points (sections) into NS lattice coordinates.

    Parameters
    ----------
    cd : CurveData object
        Must have `singfibs` dict with 'fibers' list from find_singular_fibers().
    sections : list
        List of Weierstrass points (sage points) representing sections.

    Returns
    -------
    list of sage vectors over QQ
        Each vector corresponds to NS coordinates in the order of cd.basis_labels.
    """
    from sage.all import vector, QQ

    basis_labels, Q, _ = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    r = Q.nrows()
    ns_vectors = []

    for P in sections:
        v = [0] * r
        # Zero section S has index 0
        if 'S' in basis_labels:
            v[basis_labels.index('S')] = 1

        # Fiber F has index 1  
        if 'F' in basis_labels:
            v[basis_labels.index('F')] = 1  # sections intersect fiber once

        # Process singular fibers from cd.singfibs - only for reducible fibers
        fibers = cd.singfibs.get('fibers', [])
        for i, fiber_data in enumerate(fibers):
            m_v = fiber_data.get('m_v', 1)  # number of components
            if m_v is None or m_v <= 1:
                continue  # smooth fiber, no components to process
                
            # For reducible fibers with multiple components (m_v > 1)
            for j in range(m_v):
                comp_label = f"fib{i}_c{j}"
                if comp_label in basis_labels:
                    comp_idx = basis_labels.index(comp_label)
                    # Default intersection pattern: sections typically avoid 
                    # the zero component (j=0) and may intersect others
                    # This is a geometric placeholder - actual logic depends on
                    # the specific section and fiber geometry
                    if j == 0:  # zero/identity component
                        v[comp_idx] = 0
                    else:  # non-identity components  
                        v[comp_idx] = 0  # conservatively assume no intersection

        ns_vectors.append(vector(QQ, v))

    return ns_vectors


def solve_shioda_image(sect_vec, Q, S_vec, F_vec, Theta_vecs):
    """
    Compute the Shioda-map image φ(P) as an NS vector:
      φ = sect_vec - S_vec - alpha*F_vec - sum beta_j * Theta_j,
    where the coefficients (alpha, beta_j) are chosen so that φ·F = 0
    and φ·Theta_i = 0 for all fiber components Theta_i.

    Returns φ as a Matrix(QQ, n, 1).

    Raises ValueError if the linear system determining the coefficients is singular
    (which means the trivial-lattice projection cannot be computed from the data).
    """
    # inputs are Matrix/Vector-like over QQ or ZZ
    n = Q.nrows()
    # Ensure shapes - all column vectors n×1
    sect = Matrix(QQ, n, 1, sect_vec)
    S = Matrix(QQ, n, 1, S_vec)
    F = Matrix(QQ, n, 1, F_vec)
    Thetas = [Matrix(QQ, n, 1, t) for t in Theta_vecs]

    print(f"DEBUG solve_shioda_image: n={n}, len(Thetas)={len(Thetas)}")
    print(f"DEBUG sect shape: {sect.dimensions()}")
    print(f"DEBUG S shape: {S.dimensions()}")
    print(f"DEBUG F shape: {F.dimensions()}")

    # if no reducible fiber components, trivial solve:
    if len(Thetas) == 0:
        print("DEBUG: No reducible fiber components, using simple φ = sect - S")
        # For a section class sect, (sect - S)·F should be 0 (sections intersect fiber once)
        # Then φ = sect - S is already orthogonal to F (and there are no Theta constraints).
        phi = sect - S
        
        # Verify that φ·F = 0 (should be true for sections)
        dot_product = (phi.transpose() * Q * F)[0,0]
        print(f"DEBUG: φ·F = {dot_product} (should be 0 for sections)")
        
        return phi

    # Build constraint vectors W = [F] + Theta_vecs
    W = [F] + Thetas
    m = len(W)             # m = 1 + #components

    # Unknowns are coefficients for [F] and each Theta -> same length m
    unknowns = [F] + Thetas

    print(f"DEBUG: Building {m}×{m} system with W={len(W)} constraints")

    # Build exact QQ linear system A * coeffs = b
    A = Matrix(QQ, m, m, lambda i, j: (unknowns[j].transpose() * Q * W[i])[0,0])
    b = Matrix(QQ, m, 1, lambda i, j: ((sect - S).transpose() * Q * W[i])[0,0])

    print(f"DEBUG: A matrix:")
    print(A)
    print(f"DEBUG: b vector: {b}")
    print(f"DEBUG: det(A) = {A.det()}")

    # Check invertibility (exact)
    if A.det() == 0:
        raise ValueError("solve_shioda_image: trivial-lattice Gram matrix is singular; cannot compute unique Shioda projection.")

    coeffs = A.solve_right(b)   # exact QQ solution (column vector length m)

    # Form φ = sect - S - Σ coeffs_j * unknowns[j]
    phi = sect - S
    for j in range(m):
        phi = phi - coeffs[j,0] * unknowns[j]

    # Verify orthogonality exactly
    for W_i in W:
        val = (phi.transpose() * Q * W_i)[0,0]
        if val != 0:
            raise AssertionError("Shioda projection failed orthogonality check (nonzero dot).")

    return phi


def construct_NS_from_cd(cd, current_sections, rho, mw_rank, chi, max_search_degree=4,
                         height_bound=20, max_coord=3):
    basis_labels, Q, h_vec = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    n = len(basis_labels)
    basis_unit_vectors = [Matrix(ZZ, n, 1, [1 if i == j else 0 for i in range(n)]) for j in range(n)]
    gen_labels = list(basis_labels)
    gen_vectors = list(basis_unit_vectors)

    counts, reps = staged_rational_curve_search(cd, current_sections, rho, mw_rank, chi,
                                                height_bounds=(height_bound,),
                                                max_coords=(max_coord,),
                                                return_reps=True)
    added = 0
    for d in sorted(reps.keys()):
        if d > max_search_degree: continue
        for v in reps[d]:
            assert v.nrows() == n
            arr = [int(c) for c in v.list()]
            sign = 1
            for a in arr:
                if a != 0:
                    sign = -1 if a < 0 else 1
                    break
            if sign == -1: arr = [-a for a in arr]
            col = Matrix(ZZ, n, 1, arr)
            gen_labels.append(f"rep_deg{d}_{added}")
            gen_vectors.append(col)
            added += 1

    m = len(gen_vectors)
    Gram = Matrix(ZZ, m, m)
    for i in range(m):
        vi = gen_vectors[i]
        for j in range(i, m):
            vj = gen_vectors[j]
            val = int((vi.transpose() * Q * vj)[0, 0])
            Gram[i, j] = val
            Gram[j, i] = val
    return basis_labels, Q, h_vec, gen_labels, gen_vectors, Gram


def build_ns_basis_and_Q(cd, rho, mw_rank, chi):
    """
    Build NS basis labels, intersection matrix Q, and height vector h_vec for an elliptic surface.

    Args:
        cd: dict-like with cd.singfibs['fibers'] (each fiber has 'symbol' and 'm_v')
        rho: target Picard number
        mw_rank: Mordell-Weil free rank
        chi: Euler characteristic chi(O_X)
        sum_fiber_contrib: sum_v (m_v - 1), already computed

    Returns:
        basis_labels (list of str), Q (Matrix(ZZ)), h_vec (Matrix(ZZ) column)
    """
    fibers = cd.singfibs.get('fibers', [])

    fiber_data = find_singular_fibers(cd)
    sum_fiber_contrib = fiber_data['sigma_sum']


    # basic feasibility
    min_possible = 2 + mw_rank
    max_possible = min_possible + sum(m.get('m_v', 1) - 1 for m in fibers if m.get('m_v',1) > 1)
    assert min_possible <= rho <= max_possible, "rho out of feasible range"

    # Initialize S and F
    basis_labels = ['S', 'F']
    Q = Matrix(ZZ, 2, 2, [0, 1, 1, 0])
    Q[0, 0] = -chi
    h_vec = Matrix(ZZ, 2, 1, [1, 1])

    target_extra = rho - 2
    if target_extra <= 0:
        return basis_labels, Q, h_vec

    # Add fiber root lattices (largest fibers first)
    fib_list = []
    for i, f in enumerate(fibers):
        sym = f.get('symbol', None)
        mv = int(f.get('m_v', 1))
        adj, comps = _kodaira_adjacency_and_mv(sym, mv)
        fib_list.append((i, sym, mv, comps, adj))

    fib_list.sort(key=lambda t: -t[3])
    added = 0
    for idx, sym, mv, comps, adj in fib_list:
        if comps <= 1 or added >= target_extra:
            continue
        start_index = Q.nrows()
        for comp_idx in range(1, comps):
            if added >= target_extra:
                break
            label = f"fib{idx}_c{comp_idx}"
            basis_labels.append(label)
            n = Q.nrows()
            Q = Q.stack(Matrix(ZZ, 1, n, [0]*n))
            #Q = Q.column_stack(Matrix(ZZ, n+1, 1, [0]*(n+1)))
            Q = Q.augment(Matrix(ZZ, n+1, 1, [0]*(n+1)))
            Q[n, n] = -2
            Q[0, n] = Q[n, 0] = 0
            Q[1, n] = Q[n, 1] = 0
            h_vec = h_vec.stack(Matrix(ZZ,1,1,[0]))
            added += 1
        n_total = Q.nrows()
        for a, neighs in adj.items():
            if a == 0:
                continue
            for b in neighs:
                if b == 0:
                    continue
                ia = start_index + (a - 1)
                ib = start_index + (b - 1)
                if 0 <= ia < n_total and 0 <= ib < n_total:
                    Q[ia, ib] = 1
                    Q[ib, ia] = 1

    remaining = target_extra - added
    for mi in range(remaining):
        label = f"MW{mi}"
        basis_labels.append(label)
        n = Q.nrows()
        Q = Q.stack(Matrix(ZZ, 1, n, [0]*n))
        #Q = Q.column_stack(Matrix(ZZ, n+1, 1, [0]*(n+1)))
        Q = Q.augment(Matrix(ZZ, n+1, 1, [0]*(n+1)))
        Q[n, n] = 2*chi  # placeholder canonical height (adjust later if needed)
        Q[0, n] = Q[n, 0] = 0
        Q[1, n] = Q[n, 1] = 1
        h_vec = h_vec.stack(Matrix(ZZ,1,1,[0]))

    return basis_labels, Q, h_vec





@PROFILE
def compute_canonical_height_matrix(sections, cd):
    """
    Compute the canonical height pairing matrix <P_i, P_j> using the
    explicit Shioda-Tate formula:
    <P,Q> = chi + (P.O) + (Q.O) - (P.Q) - sum_v contr_v(P,Q)
    """
    n = len(sections)
    if n == 0:
        return matrix(QQ, 0)

    # 1. Compute naive intersection matrix for the (P.Q) term
    H_naive = matrix(QQ, n)
    for i in range(n):
        for j in range(i, n):
            val = QQ(naive_pairing(sections[i], sections[j]))
            H_naive[i, j] = val
            H_naive[j, i] = val

    # 2. Get Euler characteristic (chi) and singular fiber data
    fibers_data = find_singular_fibers(cd)
    fibers = fibers_data.get('fibers', [])
    euler_total = fibers_data.get('euler_characteristic', None)

    if euler_total is None:
        raise ValueError("Could not determine total Euler characteristic from find_singular_fibers.")
    # chi = e/12, where e is the sum of Euler numbers of singular fibers
    chi = QQ(euler_total) / QQ(12)

    # 3. Compute the local contributions matrix C = sum_v contr_v(P,Q)
    C = matrix(QQ, n)
    try:
        m_sym = cd.a4.parent().gen()
    except Exception:
        raise AttributeError("Could not get generator 'm' from cd.a4.parent()")

    for i in range(n):
        for j in range(i, n):
            total_corr = QQ(0)
            for fiber in fibers:
                # local_pairing_contribution should be defined elsewhere
                total_corr += QQ(local_pairing_contribution(sections[i], sections[j], fiber, cd, m_sym))
            C[i, j] = total_corr
            C[j, i] = total_corr

    # 4. Compute intersection with the zero section, (P.O)
    PO = [None] * n
    O2 = -chi  # Self-intersection of the zero section is -chi

    # Fallback heuristic: (P.O) = (P^2 - O^2) / 2 = ( (P.P) - (-chi) ) / 2
    # This is standard when an explicit zero section object isn't available.
    for i in range(n):
        P2 = H_naive[i, i]  # This is (P_i . P_i)
        PO[i] = (P2 - O2) / QQ(2)

    # 5. Assemble the final height matrix using the formula
    H = matrix(QQ, n)
    for i in range(n):
        for j in range(i, n):
            val = chi + PO[i] + PO[j] - H_naive[i, j] - C[i, j]
            H[i, j] = val
            H[j, i] = val

    return H

# ==============================================================================
# === Internal Implementation ==================================================
# ==============================================================================


# --- Kodaira Adjacency Builders ---
def _adjacency_In(n):
    if n <= 1: return {i: [] for i in range(n)}
    adj = {i: [] for i in range(n)}
    for i in range(n):
        j = (i + 1) % n
        adj[i].append(j)
        adj[j].append(i)
    return adj

def _adjacency_I0star():
    return {0:[1], 1:[0,2,3,5], 2:[1], 3:[1,4], 4:[3], 5:[1]}

def _adjacency_Instar(n):
    total_nodes = n + 6
    if total_nodes < 6: return {}
    if total_nodes == 6: return _adjacency_I0star()
    adj = {i: [] for i in range(total_nodes)}
    adj[2].extend([0, 1]); adj[0].append(2); adj[1].append(2)
    adj[n+3].extend([n+4, n+5]); adj[n+4].append(n+3); adj[n+5].append(n+3)
    for i in range(2, n + 3):
        adj[i].append(i + 1); adj[i + 1].append(i)
    return adj

def _adjacency_IVstar():
    return {0:[3], 1:[3], 2:[3], 3:[0,1,2,4], 4:[3,5], 5:[4]}

def _adjacency_IIIstar():
    return {0:[4], 1:[3], 2:[3], 3:[1,2,4], 4:[0,3,5], 5:[4,6], 6:[5,7], 7:[6]}
    
def _adjacency_IIstar():
    return {0:[6], 1:[2], 2:[1,3], 3:[2,4], 4:[3,5], 5:[4,6], 6:[0,5,7], 7:[6,8], 8:[7]}

_KODAIRA_DISPATCH = {
    "II":   lambda n: ({0:[]}, 1),
    "III":  lambda n: ({0:[1], 1:[0]}, 2),
    "IV":   lambda n: (_adjacency_In(3), 3),
    "I0*":  lambda n: (_adjacency_I0star(), 6),
    "IV*":  lambda n: (_adjacency_IVstar(), 7),
    "III*": lambda n: (_adjacency_IIIstar(), 8),
    "II*":  lambda n: (_adjacency_IIstar(), 9),
}

def _kodaira_adjacency_and_mv(symbol, m_v):
    if symbol is None: return {}, 1
    s = symbol.strip()
    if s in _KODAIRA_DISPATCH:
        return _KODAIRA_DISPATCH[s](m_v)
    if s.startswith("I"):
        if s.endswith("*"):
            try: n = int(s[1:-1])
            except ValueError: n = m_v - 6 if m_v else 0
            return _adjacency_Instar(n), n + 6
        else:
            try: n = int(s[1:])
            except ValueError: n = m_v if m_v else 1
            return _adjacency_In(n), n
    return {}, m_v if m_v is not None else 1

# Add this helper function near the top of search_common.py
# (Ensure necessary imports like QQ, Integer, log, math are present)
import math

def point_height(pt):
    """Calculates a simple height for a point (x, y). Uses x-height."""
    x, y = pt
    try:
        # Ensure x is QQ before accessing numerator/denominator
        x_qq = QQ(x)
        num = abs(Integer(x_qq.numerator()))
        den = abs(Integer(x_qq.denominator()))
        # Use log(max(1, |num|, |den|)) for stability at (0, y) or (1, y) etc.
        h = float(log(max(1, num, den)))
        return h
    except Exception as e:
        # Handle potential errors during conversion or calculation
        # Assign effectively infinite height to prioritize valid points
        print(f"Warning: Could not compute height for point {pt}: {e}")
        raise
        return float('inf')

# Replace the existing get_data_pts function with this one:
@PROFILE
def get_data_pts(known_pts, excluded):
    """
    Gets the next combination of 1, 2, or 3 points for a fibration.
    Prioritizes combinations made from lower height points first.
    """
    # Convert set to list and sort known_pts by height (ascending)
    # Points with calculation errors will be pushed to the end
    sorted_pts = sorted(list(known_pts), key=point_height)

    # Iterate through r (number of points in combination: 1, 2, 3)
    for r in range(1, 4):
        # Generate combinations from the sorted list.
        # itertools.combinations preserves the input order, so combinations
        # using points earlier in the sorted list (lower height) are yielded first.
        for combo in itertools.combinations(sorted_pts, r):
            # Check if this combination has already been excluded
            if frozenset(combo) not in excluded:
                # Return the first valid combination found
                return combo

    # If all combinations have been checked and excluded
    return None


