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
    is_prime, Integer, log, next_prime, HyperellipticCurve
)
from math import gcd, log

def get_random_x_on_hyperelliptic(coeffs, p):
    """
    Finds a random x-coordinate such that f(x) is a quadratic residue mod p.
    Assumes coeffs are [a_n, ..., a_0].
    """
    Fp = GF(p)
    # We use a loop with a high safety limit
    for _ in range(1000):
        try_x = Fp.random_element()
        
        # Manually evaluate the polynomial to avoid symbolic overhead
        # f(x) = sum(a_i * x^(deg-i))
        val = Fp(0)
        deg = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            val += Fp(c) * (try_x**(deg - i))
        
        if val.is_square() and val:
            return QQ(int(try_x))
            
    raise ValueError(f"Failed to find a valid point on the curve after 1000 random attempts mod {p}.")

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
COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(5), QQ(6), QQ(5), QQ(2), QQ(1)]
COEFFS_GENUS2 = [QQ(1), QQ(2), QQ(7), QQ(6), QQ(-3), QQ(-8), QQ(-4)]

# old curve, the OG
#x^7 - 10 x^5 + 15 x + 5
COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(-2), QQ(-18), QQ(1), QQ(38), QQ(25)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 11

# # doesn't find y=0 point... added a special function to find these...maybe ok...
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

COEFFS_GENUS2 = [QQ(4), QQ(0), QQ(-16), QQ(24), QQ(-16), QQ(5)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 2

COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(2), QQ(-18), QQ(21), QQ(-10), QQ(1)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 4

COEFFS_GENUS2 = [QQ(1), QQ(6), QQ(10), QQ(7), QQ(1), QQ(0)]
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

db_entry = '9997256:9997256:[x^7+x^6-2*x^5-5*x^4-x^3+2*x^2-1,x^4+x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 2

db_entry = '9997199:9997199:[3*x^3+x^2-2*x,x^4+x+1]'
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


db_entry = '9997263:3332421:[x^7+x^6-4*x^5-2*x^4+x^3-x,x^4+x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4

db_entry = '9994635:3331545:[x^7+2*x^6-x^5+8*x^3+3*x^2-5*x-2,x^4+x^3+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 4


db_entry = '9996912:3332304:[x^5+2*x^4+x^3-x^2-2*x-1,x^4+x^3+x^2]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3

db_entry = '9995456:2498864:[2*x^7-4*x^6-5*x^5+10*x^4+5*x^3-8*x^2-3*x+1,x^2+x]'
db_entry = '9995408:2498852:[x^8-x^6+x^3+2*x^2+x,x^2+x+1]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 5

db_entry = '9996352:312386:[-2*x^6-6*x^5+x^4+18*x^3+10*x^2-17*x-15,x^4+x^3+x]'
COEFFS_GENUS2 = parse_hyperelliptic_db_entry(db_entry)
DATA_PTS_GENUS2 = [QQ(0)] # just the x values
TERMINATE_WHEN_6 = 3


COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(2), QQ(-30), QQ(33), QQ(-10), QQ(1)]
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 4

COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(-4), QQ(10), QQ(-24), QQ(24), QQ(-7)]
DATA_PTS_GENUS2 = [QQ(2)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(4), QQ(-8), QQ(-20), QQ(0), QQ(16), QQ(8), QQ(1)]
DATA_PTS_GENUS2 = [QQ(0)] # just the x values lol


COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(4), QQ(4), QQ(8), QQ(-8), QQ(-12)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(4), QQ(-4), QQ(-36), QQ(5), QQ(96), QQ(64)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 4

# $y^2 = 4x^6 + 9x^4 - 4x^3 + 2x^2 - 4x + 1$ # rank 2
COEFFS_GENUS2 = [QQ(4), QQ(0), QQ(9), QQ(-4), QQ(2), QQ(-4), QQ(1)] # rank 2
DATA_PTS_GENUS2 = [QQ(0)] # just the x values lol
TERMINATE_WHEN_6 = 3

# $y^2 = 4x^6 - 12x^5 + 16x^4 - 8x^3 - 3x^2 + 4x$ # rank 2
COEFFS_GENUS2 = [QQ(4), QQ(-12), QQ(16), QQ(-8), QQ(-3), QQ(4), QQ(0)] # rank 2
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 3

COEFFS_GENUS2 = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)] # rank 4
DATA_PTS_GENUS2 = [QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 12

# prestige curve lol, rank 4
COEFFS_GENUS2 = [QQ(1), QQ(8), QQ(10), QQ(-10), QQ(-11), QQ(2), QQ(1)]
DATA_PTS_GENUS2 = [QQ(-1)] # just the x values lol
TERMINATE_WHEN_6 = 11

# attack curve, i guess
#y² = 8x⁵ + 16x⁴ - 60x³ + 69x² - 36x + 8
COEFFS_GENUS2 = [QQ(8), QQ(16), QQ(-60), QQ(69), QQ(-36), QQ(8)]
DATA_PTS_GENUS2 = [QQ(1)/QQ(2)] # just the x values lol
TERMINATE_WHEN_6 = 2

# claude generated this curve, not in the lmfdb as of Jan 3 2026
# y² = -3x⁶ + 11x⁵ + 6x⁴ - 9x³ + 2x² + x + 25
COEFFS_GENUS2 = [QQ(-3), QQ(11), QQ(6), QQ(-9), QQ(2), QQ(1), QQ(25)]
DATA_PTS_GENUS2 = [QQ(0)/QQ(1)] # just the x values lol
TERMINATE_WHEN_6 = 2

# y^2 = x^5 + 3x^3 + 2x^2 + 5x + 4
COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(3), QQ(2), QQ(5), QQ(4)]
DATA_PTS_GENUS2 = [QQ(0)/QQ(1)] 
DATA_PTS_GENUS2 = [457208] 
TERMINATE_WHEN_6 = 3

# $$y^2 = x^5 + x + 2$$
COEFFS_GENUS2 = [QQ(1), QQ(0),QQ(0),QQ(0),QQ(1),QQ(2)]
#DATA_PTS_GENUS2 = [QQ(1)/QQ(1)] # just the x values lol
DATA_PTS_GENUS2 = None # placeholder for random.
DATA_PTS_GENUS2 = [QQ(12630360)]
TERMINATE_WHEN_6 = 3

##### END TEST CURVES ######


# BEGIN STATIC CONFIG (default config; mostly deprecated)

NUM_DOUBLINGS = 20 # for mumford height pairing independence test
HEIGHT_BOUND = 6*370 # not that important, mostly, it seems
# prime config
# magic prime settings, chosen empirically.
#PRIME_POOL = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
#PRIME_POOL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
PRIME_POOL = list(primes(590))  # All primes less than N, excluding 2,3; >=50 should be good... might need more for high height points!

# CRYPTOGRAPHY RELATED PARAMS
FINITE_FIELD = None
FINITE_FIELD = next_prime(2**25)
MAXN = 80 # since there is no notion of height on finite field mode, this serves as the max n for section multiple [n]P
SECRET_KEY = 800 # how many multiples of base genus 2 divisor to use to obtain the target starting from the base divisor from DATA_PTS_GENUS2[0]
BASE_DIVISOR, TARGET_DIVISOR, PREFERRED_X_COORDS = None, None, None # constructed below, here for reference

# 1) Generate the random point if requested
if DATA_PTS_GENUS2 is None:
    # Ensure we use the prime currently active in your pool
    _p_init = FINITE_FIELD
    DATA_PTS_GENUS2 = [get_random_x_on_hyperelliptic(COEFFS_GENUS2, _p_init)]


if FINITE_FIELD:
    # Use only the field characteristic as our "prime"
    PRIME_POOL = [FINITE_FIELD]

NUM_PRIME_SUBSETS = 500 # important for stability under different seeds, must be large enough >= 250 should be good...

VERIFY_INDEPENDENCE_MOD_P = False # verify mumford_search divisors mod a prime of good reduction
VERIFY_INDEPENDENCE_MOD_P = True # verify mumford_search divisors mod a prime of good reduction

MIN_PRIME_SUBSET_SIZE = 3 # just keep this at 3
MIN_MAX_PRIME_SUBSET_SIZE = 9 # safe is 7-9; above 15 is too stringent
MAX_MODULUS = 10**9# idk
NUM_SAMPLES_HEIGHT_MAT = 10 # seems not important
HEIGHT_BOUND_NON_MINIMAL = 2*HEIGHT_BOUND # New bound for non-minimal models, just double the minimal one lol  # 420 blaze it
HENSEL_SLOPPY = False
HENSEL_SLOPPY = True # goes fast, but hensel filtering is really only saying we only expect solutions at simple roots, which may not always be true, but this rarely loses information.
TORSION_SLOPPY = True # an even more unmotivated filter; filter out small ord_p residues for some reason.
MAX_TORSION_ORDER_TO_FILTER = -1 # what ord_p max to filter out.  -1 means only filter out singularity specialization. (N.B. does not turn TORSION_SLOPPY off!)
TORSION_SLOPPY = False
###### END STATIC CONFIG


# random seed for reproducibility.
SEED_INT = random.randint(-10**6, 10**6)
ANCHOR_SEED = SEED_INT           # Seed for reproducible anchor point generation

DEBUG = False
DEBUG = True
TARGETED_X = QQ(182)/QQ(141) # sample value used to debug
TARGETED_X = None # only set to numeric value to debug; None by default

USE_MINIMAL_MODEL = False # uses the generic fiber
USE_MINIMAL_MODEL = True # more correct, and more slow
SYMBOLIC_SEARCH = True   # the search over Q (often slower, usually doesn't find anything)
SYMBOLIC_SEARCH = False   # mod p search (usually faster; the default)
MOBIUS_TRANS = True # search after applying a mobius transformation to x to attempt to improve the prime content.  generally worse.
MOBIUS_TRANS = False
MUMFORD_SEARCH = False # look for elements of J(C); only supports genus 2 right now.
MUMFORD_SEARCH = True # look for elements of J(C); only supports genus 2 right now.

AVOID = {2,3,5,7,11,13,17,19}   # tweak as you like, for MOBIUS_TRANS mode, avoid primes
PREFER =  {31,37,41,43,47,53}   # or {23,29} if you want to force primes upward, for MOBIUS_TRANS


# Add to search_common.py or search_config.py
USE_CONSENSUS_FILTER = True  # Toggle for multi-fibration consensus
USE_CONSENSUS_FILTER = False  # Toggle for multi-fibration consensus
NUM_CONSENSUS_FIBRATIONS = 4  # How many independent fibrations to use
CONSENSUS_THRESHOLD = 0.5     # Fraction of fibrations that must agree (0.8 = 80%)

# Add these constants near the top of tower.sage (with other config constants)

# === ANCHOR POINT MODE CONFIGURATION ===
USE_ANCHOR_POINTS = True  # Toggle: True = use random anchor points, False = use tangency
USE_ANCHOR_POINTS = False  # Toggle: True = use random anchor points, False = use tangency
USE_ANCHOR_POINTS = USE_CONSENSUS_FILTER
NUM_ANCHOR_POINTS = 1      # How many anchor points to use (only when USE_ANCHOR_POINTS=True)


def generate_random_curve_point(f_poly, p):
    """
    Generates a random point P on C(F_p) by finding x with f(x) a square.
    Returns the Jacobian element J(C(x, y)).
    """
    R = PolynomialRing(K, 'x')
    f = R(f_poly)
    C = HyperellipticCurve(f)
    J = C.jacobian()
    
    max_attempts = 1000
    for _ in range(max_attempts):
        x_coord = K.random_element()
        y2 = f(x_coord)
        if y2.is_square():
            y_coord = y2.sqrt()
            P = J(C((x_coord, y_coord)))
            if 2 * P != J(0):
                return P
    
    raise ValueError("Failed to generate random curve point after max attempts")


def generate_keypair_from_secret(coeffs_genus2, p, secret_key):
    """
    Generates keypair (G, Q) where G is constructed from the first data point
    and Q = [secret_key]G.
    
    Args:
        coeffs_genus2: Coefficients of f(x) in hyperelliptic curve y^2 = f(x)
        p: Prime field size
        secret_key: Integer secret key from search_common.SECRET_KEY
    
    Returns:
        (G, Q, preferred_x_values) where G is base divisor, Q is target divisor,
        and preferred_x_values is a set of x-coords to bias toward
    """
    key = (tuple(coeffs_genus2), p, secret_key)
    if key in generate_keypair_from_secret.cache:
        return generate_keypair_from_secret.cache[key]
    K = GF(p)
    R = PolynomialRing(K, 'x')
    # Build f(x) - coeffs are highest degree first
    f_poly = R([K(c) for c in reversed(coeffs_genus2)])
    
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Get base x-coordinate from our fibration data
    x_base_qq = DATA_PTS_GENUS2[0]  # This should be a rational x-coordinate
    x_base = K(x_base_qq)
    
    # Construct a divisor from this point
    # For genus 2: D = (x_base, y_base) - ∞
    y2_val = f_poly(x_base)
    
    if not y2_val.is_square():
        raise ValueError(f"Base point x={x_base_qq} does not give a quadratic residue: f(x)={y2_val}")
    
    y_base = y2_val.sqrt()
    
    # Create point on curve
    P_base = C((x_base, y_base))
    
    # Convert to Jacobian element (divisor)
    G = J(P_base)
    
    # Compute target divisor
    Q = Integer(secret_key) * G
    
    # Extract x-coordinates from both divisors to use as preferred values
    preferred_x_values = set()
    
    # Extract from G
    u_G = G[0]  # u(x) polynomial
    if u_G.degree() > 0:
        for root, _ in u_G.roots():
            preferred_x_values.add(int(root))
    
    # Extract from Q
    u_Q = Q[0]
    if u_Q.degree() > 0:
        for root, _ in u_Q.roots():
            preferred_x_values.add(int(root))
    
    print(f"Generated keypair:")
    print(f"  Base point x-coord: {x_base_qq}")
    print(f"  Base divisor G: {G}")
    print(f"  Secret key d: {secret_key}")
    print(f"  Target Q = [d]G: {Q}")
    print(f"  Preferred x-values for smoothness: {sorted(preferred_x_values)}")
    ret = G, Q, preferred_x_values
    generate_keypair_from_secret.cache[key] = ret
    return ret
generate_keypair_from_secret.cache = {}

# Initialize PREFERRED_X_COORDS for finite field mode
# This must happen at module import time, not just in main
if FINITE_FIELD is not None:
    try:
        BASE_DIVISOR, TARGET_DIVISOR, PREFERRED_X_COORDS = generate_keypair_from_secret(
            COEFFS_GENUS2, 
            FINITE_FIELD, 
            SECRET_KEY
        )
        print(f"Initialized PREFERRED_X_COORDS: {PREFERRED_X_COORDS}")
    except Exception as e:
        print(f"Warning: Keypair generation failed: {e}")
        # Fallback: use the base point x-coordinate
        PREFERRED_X_COORDS = [DATA_PTS_GENUS2[0]] if DATA_PTS_GENUS2 else [0]
        BASE_DIVISOR, TARGET_DIVISOR = None, None
else:
    # Not in finite field mode - these aren't needed
    BASE_DIVISOR, TARGET_DIVISOR, PREFERRED_X_COORDS = None, None, None


try:
    PROFILE = profile
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


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
    base_field: object


# --- START: Modular Reduction Helpers (centralized from picard.py) ---

def to_mod_poly(poly_q, R, debug=False):
    """
    Coerce `poly_q` (polynomial-like over QQ or FractionField) into R = PolynomialRing(GF(ell), 'm').
    """
    try:
        if poly_q.parent() is R:
            return poly_q
    except Exception:
        raise
    try:
        return R(poly_q)
    except Exception as e_direct:
        if debug:
            print(f"[debug to_mod_poly] direct coercion failed: {e_direct}")
        raise
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
            raise
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
                raise
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
    """
    Canonicalize vectors by sign: make first non-zero element positive.
    This ensures (4,) and (-4,) map to (4,), not (-4,).
    """
    seen = set()
    out = []
    for v in vecs:
        vt = tuple(int(x) for x in v)
        if all(x == 0 for x in vt):
            continue
        
        # Find first non-zero element
        first_nonzero_idx = None
        for i, x in enumerate(vt):
            if x != 0:
                first_nonzero_idx = i
                break
        
        if first_nonzero_idx is None:
            continue  # All zeros
        
        # Canonicalize: make first non-zero positive
        if vt[first_nonzero_idx] < 0:
            can = tuple(-x for x in vt)
        else:
            can = vt
        
        if can not in seen:
            seen.add(can)
            out.append(can)  # <-- Return canonical form, not original vt
    
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
        raise

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


@PROFILE
def effective_degree(rational_expr, m):
    """
    Robust effective degree of a rational function: deg(numerator) - deg(denominator).
    """
    num = rational_expr.numerator()
    den = rational_expr.denominator()
    def _deg(poly):
        try:
            return int(poly.degree())
        except Exception:
            raise
        try:
            fac = poly.factor()
            deg = 0
            for base, exp in fac:
                try:
                    if base == m:
                        deg += int(exp)
                except Exception:
                    raise
                    continue
            if deg:
                return deg
        except Exception:
            raise
        try:
            R = PolynomialRing(QQ, str(m))
            p = R(poly)
            return int(p.degree())
        except Exception:
            raise
            return 0
    return _deg(num) - _deg(den)

def _refresh_state(a4_final, a6_final, Fm):
    var('m')
    E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
    Delta_final = E_weier_final.discriminant()
    # raw degree (no cancellation) used for minimality check
    deg_delta_raw = effective_degree(Delta_final, m)
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
            raise
            return
        try:
            N = Integer(q.numerator())
            D = Integer(q.denominator())
        except Exception:
            raise
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
        raise

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
                raise
        else:
            coeffs = [poly]

        for c in coeffs:
            add_primes_from_coeff(c)

    # tidy: remove 0/1 if any sneaked in
    primes = {p for p in primes if isinstance(p, int) and p > 1}
    return primes


# ---- buildcd replacement ----

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

        effective_degree2 = effective_degree(Delta, m_var)

        print("Delta:", Delta)
        print(f"  Effective Discriminant Degree: {effective_degree2}")
        stored_n = getattr(cd, 'tate_exponent', '<missing>')
        print(f"  Weierstrass scaling exponent (n) used: {stored_n}")

        if effective_degree2 == 12:
            print("  ✅ PASS: Discriminant degree is 12. Standard geometric checks should apply.")
        else:
            print(f"  ⚠ NOTICE: Discriminant degree is {effective_degree2}, not 12.")
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
            raise

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
            raise
            continue
    return found


# pseudo-code sketch (raise on unexpected failure)
from sage.all import valuation, gcd


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
        raise

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


def sample_rationals_by_height_random(N, B):
    """
    Return list of N rationals QQ(a)/QQ(b) with gcd(a,b)=1, 1 <= b <= B, |a| <= B.
    Height proxy: max(|a|,|b|) <= B.
    Diagnostic-only; does not alter search.
    """
    assert int(N) > 0
    assert int(B) >= 1
    out = []
    tries = 0
    while len(out) < int(N):
        a = random.randint(-B, B)
        b = random.randint(1, B)
        if gcd(a, b) != 1:
            tries += 1
            continue
        out.append(QQ(a) / QQ(b))
        tries += 1
        # allow loop to raise naturally if something extremely odd happens
    return out


def enumerate_rationals_height_bound(B):
    """
    Deterministic list of rationals QQ(a)/QQ(b) with 1 <= b <= B and |a| <= B,
    returned in lexicographic order (a then b). Use for reproducible diagnostics.
    """
    assert int(B) >= 1
    out = []
    for b in range(1, B + 1):
        for a in range(-B, B + 1):
            if gcd(a, b) != 1:
                continue
            out.append(QQ(a) / QQ(b))
    return out


def get_sections_for_fibration(cd, base_pts):
    """
    Compute and reduce base sections for a specific fibration geometry.
    """
    # Compute raw sections from base points
    raw_sections = compute_base_sections_m(cd, base_pts)
    
    if not raw_sections:
        return []

    # Reduce them (LLL) to get a nice basis
    reduced_sections = lll_reduce_mw_basis(cd, raw_sections)
    
    # Ensure uniqueness
    unique_sections = list(set(reduced_sections))
    return unique_sections


def try_scale_out_power_of_two(cd, max_t=2, debug=False):
    """
    Robust version that avoids coercion errors when testing Δ' mod 2.

    - For each candidate u = 2^t, builds a4', a6', Δ' (rational polynomial).
    - Converts Δ' to an integer polynomial by multiplying by the LCM of denominators.
    - Divides out any overall 2-power from the integer polynomial (but only for
      the purpose of the mod-2 test — we don't mutate a4'/a6' here).
    - Tests whether the resulting polynomial is identically zero in GF(2).
    - On success returns a new CurveDataExt with scaled a4/a6; on failure raises.
    """
    from sage.all import ZZ, GF, lcm as sage_lcm, gcd as sage_gcd, valuation

    a4 = cd.a4
    a6 = cd.a6

    # Compute original Δ and quick-check (guard clause)
    Delta = -16 * (4 * a4**3 + 27 * a6**2)
    Delta_num = Delta.numerator()

    # Helper: safe polynomial -> GF(2) zero-check
    def poly_is_zero_mod2_safe(poly_q):
        """
        poly_q: polynomial over QQ (rational coefficients)
        Returns True if poly_q is zero in GF(2), False otherwise.
        """
        # Extract coefficients as rationals
        coeffs = poly_q.coefficients()
        if not coeffs:
            # zero polynomial already
            return True

        # compute LCM of denominators
        dens = [c.denominator() for c in coeffs]
        try:
            D = ZZ(1)
            for d in dens:
                D = sage_lcm(D, ZZ(d))
        except Exception as e:
            raise RuntimeError(f"Failed to compute denominator LCM for Δ: {e}")

        # Multiply to get integer polynomial
        try:
            poly_int = (poly_q * D).change_ring(ZZ)   # now polynomial over ZZ
        except Exception as e:
            raise RuntimeError(f"Failed to coerce Δ*LCM to ZZ polynomial: {e}")

        # If all integer coefficients share a power of 2, divide it out for mod-2 test.
        int_coeffs = [int(c) for c in poly_int.coefficients()]
        # gcd could be 0 if polynomial is zero, so guard
        if all(c == 0 for c in int_coeffs):
            return True
        common_g = abs(int(int(sage_gcd(int_coeffs))))
        # compute v2 of common_g
        v2 = 0
        if common_g != 0:
            while common_g % 2 == 0:
                common_g //= 2
                v2 += 1

        if v2 > 0:
            poly_int_trim = poly_int // (ZZ(2)**v2)
        else:
            poly_int_trim = poly_int

        # Now safe to change ring to GF(2).
        try:
            poly_gf2 = poly_int_trim.change_ring(GF(2))
        except Exception as e:
            # This should not normally happen, but surface it with context if it does.
            raise RuntimeError(f"Failed converting Δ to GF(2): {e}")

        return poly_gf2.is_zero()

    # If already OK w/o scaling, return original
    try:
        if not poly_is_zero_mod2_safe(Delta_num):
            if debug:
                print("[try_scale_out_power_of_two] Δ already nonzero mod 2; no scaling.")
            return cd
    except Exception as e:
        # Surface a friendly error rather than letting Sage raise raw low-level exceptions.
        raise RuntimeError(f"Pre-scale Δ mod 2 test failed: {e}")

    # Try scale factors u = 2^t
    for t in range(1, max_t + 1):
        u = ZZ(2) ** t
        try:
            a4_new = a4 / (u**4)
            a6_new = a6 / (u**6)
        except Exception as e:
            raise RuntimeError(f"Scaling a4/a6 by u=2^{t} failed: {e}")

        try:
            Delta_new = -16 * (4 * a4_new**3 + 27 * a6_new**2)
            Delta_new_num = Delta_new.numerator()
        except Exception as e:
            raise RuntimeError(f"Failed to form Δ' for u=2^{t}: {e}")

        # Use safe conversion/test
        try:
            is_zero_mod2 = poly_is_zero_mod2_safe(Delta_new_num)
        except Exception as e:
            raise RuntimeError(f"Δ mod 2 test failed for u=2^{t}: {e}")

        if not is_zero_mod2:
            # Success: rebuild cd with the new coefficients
            if debug:
                print(f"[try_scale_out_power_of_two] Success with u=2^{t}.")
            return CurveDataExt(
                E_curve     = cd.E_curve,
                E_weier     = cd.E_weier,
                E_rhs       = cd.E_rhs,
                a4          = a4_new,
                a6          = a6_new,
                phi_x       = cd.phi_x,
                quartic_rhs = cd.quartic_rhs,
                tate_exponent = cd.tate_exponent,
                k_base_change = cd.k_base_change,
                bad_primes    = cd.bad_primes,
                morphs        = cd.morphs,
                use_minimal   = cd.use_minimal,
                blowup_factor = cd.blowup_factor,
                singfibs      = cd.singfibs,
                SR_a4         = cd.SR_a4,
                SR_a6         = cd.SR_a6,
                SR_phi_x      = cd.SR_phi_x,
                SR_m          = cd.SR_m,
            )

    # If reached, scaling didn't remove the mod-2 collapse up to max_t
    raise RuntimeError(f"Failed to remove 2-adic global factor up to u=2^{max_t}.")


# ---------------------------
# Finite-field aware section helpers
# ---------------------------


@PROFILE
def verify_morphism_on_samples(cd, base_pts):
    """
    Verify images of base_pts under cd.morphs lie on cd.E_weier.

    Accepts the same base_pts format as compute_base_sections_m and will coerce
    coordinates into cd.base_field when present.
    """
    one_s, two_s, three_s = cd.morphs
    E_min = cd.E_weier
    base_field = getattr(cd, 'base_field', None)

    for xi, yi in base_pts:
        try:
            if base_field is not None:
                F = base_field
                xi_f = F(xi)
                yi_f = F(yi)
                X = one_s(x=xi_f, y=yi_f)
                Y = two_s(x=xi_f, y=yi_f)
                Z = three_s(x=xi_f, y=yi_f)
                P = E_min([X, Y, Z])
            else:
                X = one_s(x=xi, y=yi)
                Y = two_s(x=xi, y=yi)
                Z = three_s(x=xi, y=yi)
                P = E_min([X, Y, Z])
        except Exception as e:
            print(f"Error verifying morphism on sample point ({xi}, {yi}): {e}")
            raise
    return True


def compute_base_sections_m_direct(cd, quartic_pts):
    """Apply Weierstrass morphism to points already on the quartic."""
    one_use, two_use, three_use = cd.morphs
    ret = []
    seen = set()
    base_field = getattr(cd, 'base_field', None)
    
    for xi, yi in quartic_pts:
        if (xi, yi) in seen or xi is None:
            continue
            
        if base_field is not None:
            F = base_field
            xi_f = F(xi)
            yi_f = F(yi)
            X_aff = one_use(x=xi_f, y=yi_f)
            Y_aff = two_use(x=xi_f, y=yi_f)
            Z_aff = three_use(x=xi_f, y=yi_f)
            P = cd.E_weier([X_aff, Y_aff, Z_aff])
        else:
            X_aff = one_use(x=xi, y=yi)
            Y_aff = two_use(x=xi, y=yi)
            Z_aff = three_use(x=xi, y=yi)
            
            if DEBUG:
                print(f"\n--- DEBUGGING POINT CONSTRUCTION ---")
                print(f"Quartic point: ({xi}, {yi})")
                print(f"Weierstrass coords: X={X_aff}, Y={Y_aff}, Z={Z_aff}")
                try:
                    LHS = Y_aff**2 * Z_aff
                    RHS = X_aff**3 + cd.a4 * X_aff * Z_aff**2 + cd.a6 * Z_aff**3
                    print(f"LHS - RHS = {(LHS - RHS).simplify_rational()}")
                except Exception as e:
                    print(f"Verification failed: {e}")
                    raise
            
            P = cd.E_weier([X_aff, Y_aff, Z_aff])
        
        ret.append(P)
        seen.add((xi, yi))
    
    return ret


@PROFILE
def buildcd(E_curve, phi_x, quartic_rhs, E_rhs, morph_triplet,
            verify=True, compute_minimal=USE_MINIMAL_MODEL):
    """
    Builds the CurveDataExt object for the fibration.

    If compute_minimal is True, it computes the minimal Weierstrass model and applies
    the necessary transformations to the coordinate maps.

    If compute_minimal is False, it uses the raw, non-minimal model directly from the
    Jacobian, which is often faster for searching.
    
    If FINITE_FIELD is set, builds everything over GF(FINITE_FIELD) instead of QQ.
    """
    print("--- Entering buildcd ---")
    
    # Determine base field
    # Determine base field and function field
    if FINITE_FIELD:
        base_field = GF(FINITE_FIELD)
        print(f"Building over finite field GF({FINITE_FIELD})")
        # Explicitly construct the function field over GF(p)
        Pm_base = PolynomialRing(base_field, 'm')
        Fm = FractionField(Pm_base)
        m = Fm.gen()
    else:
        base_field = QQ
        print("Building over QQ")
        Fm = a4_raw.parent()
        m = Fm.gen()

    E_weier_raw = Jacobian(E_curve)
    a4_raw = E_weier_raw.a4()
    a6_raw = E_weier_raw.a6()
    # Now coerce a4_raw and a6_raw into this field
    a4_raw = Fm(a4_raw)
    a6_raw = Fm(a6_raw)
    
    # Get the parent ring - over finite field or QQ
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

        # Build Weierstrass model over appropriate base field
        if FINITE_FIELD:
            # For finite fields, build curve over function field GF(p)(m)
            E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
        else:
            E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])

        # Step 3: Apply net scaling transformation to morphisms
        net_k = int(blowup_factor - blowdown_0)
        x_morphism_scale = m**(2 * net_k)
        y_morphism_scale = m**(3 * net_k)

        one_s = MorphismWrapper(one, 1, x_morphism_scale, a4_final)
        two_s = MorphismWrapper(two, 1, y_morphism_scale, a4_final)
        three_s = MorphismWrapper(three, 1, 1, a4_final)

    else:
        print("--- Using non-minimal model, skipping all rescaling ---")
        a4_final = a4_raw
        a6_final = a6_raw
        phi_x_final = phi_x
        blowup_factor = 0
        
        if FINITE_FIELD:
            E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
        else:
            E_weier_final = E_weier_raw

        one_s = MorphismWrapper(one, 1, 1, a4_final)
        two_s = MorphismWrapper(two, 1, 1, a4_final)
        three_s = MorphismWrapper(three, 1, 1, a4_final)

    # --- Handle SR coercion based on field ---
    if FINITE_FIELD:
        # Over finite fields, do NOT coerce to SR
        SR_a4 = a4_final
        SR_a6 = a6_final
        SR_phi_x = phi_x_final
        SR_m = m  # Keep as finite field generator
    else:
        # Over QQ, coerce to SR for symbolic substitution
        SR_m = var('m')
        try:
            SR_a4 = SR(a4_final)
            SR_a6 = SR(a6_final)
            SR_phi_x = SR(phi_x_final)
        except Exception:
            raise AssertionError("buildcd: failed to coerce a4/a6/phi_x to SR; check types.")

    # --- Common logic for both models ---

    # Compute global bad primes
    print("\n--- Identifying Globally Bad Primes ---")
    class TempCD:
        def __init__(self, a4, a6): 
            self.a4, self.a6 = a4, a6
    
    temp_cd = TempCD(a4_final, a6_final)
    
    if FINITE_FIELD:
        # In finite field mode, only the characteristic is bad
        bad_primes = [FINITE_FIELD]
        print(f"Finite field mode: characteristic {FINITE_FIELD} is the only bad prime")
    else:
        bad_primes = [p for p in PRIME_POOL if not is_good_prime_for_surface(temp_cd, p)]
        print(f"Identified {len(bad_primes)} globally bad prime(s) from the pool: {sorted(bad_primes)}")

    try:
        if FINITE_FIELD:
            E_rhs_final = E_rhs  # Keep original for finite field
        else:
            E_rhs_final = y**2 - x**3 - a4_final * x - a6_final
    except Exception:
        E_rhs_final = E_rhs
        raise

    # Get singular fiber info for diagnostics
    if FINITE_FIELD:
        # Skip singular fiber analysis over finite fields (needs QQ)
        singfibs = {'fibers': [], 'euler_characteristic': 0, 'sigma_sum': 0}
        print("Skipping singular fiber analysis over finite field")
    else:
        singfibs = find_singular_fibers(a4=a4_final, a6=a6_final, verbose=True)

    # --- Package and return the final data ---
    cd = CurveDataExt(
        E_curve=E_curve,
        E_weier=E_weier_final,
        E_rhs=E_rhs_final,
        a4=a4_final,
        a6=a6_final,
        phi_x=phi_x_final,
        quartic_rhs=quartic_rhs,
        tate_exponent=0,
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
        base_field=base_field if FINITE_FIELD else None
    )

    if not FINITE_FIELD:
        cd = try_scale_out_power_of_two(cd)

    print("--- Exiting buildcd ---")
    if verify and DEBUG and not FINITE_FIELD:
        validate_fibration_geometry(cd)
        if cd.E_weier.discriminant().is_zero():
            raise ValueError("buildcd: Resulting discriminant is identically zero.")
    
    return cd


def compute_base_sections_m(cd, base_pts, tower=None):
    """
    Maps points from the hyperelliptic curve to sections on the Weierstrass fibration E_m.
    Ensures that for index calculus, evaluation stays within GF(p)(m).
    """
    if not base_pts:
        return []
    one_use, two_use, three_use = cd.morphs

    # In Finite Field mode, we must ensure the morphism polynomials are mapped to GF(p)(m)
    if FINITE_FIELD is not None:
        Fp = GF(FINITE_FIELD)
        Pm_p = PolynomialRing(Fp, 'm')
        Km_p = FractionField(Pm_p)
        
        # Helper to map QQ(m) polynomials to Fp(m) polynomials
        def map_to_fp(poly):
            new_parent = PolynomialRing(Km_p, poly.parent().variable_names())
            d = poly.dict()
            new_d = {}
            for mon, coeff in d.items():
                # coeff is in QQ(m), map num/den to Fp(m)
                num_p = Pm_p(coeff.numerator())
                den_p = Pm_p(coeff.denominator())
                new_d[mon] = Km_p(num_p) / Km_p(den_p)
            return new_parent(new_d)

        # Extract the underlying polynomials from the MorphismWrappers
        one_poly = one_use.callable_obj
        two_poly = two_use.callable_obj
        three_poly = three_use.callable_obj
        
        # Map them to Fp(m)
        one_mapped = map_to_fp(one_poly)
        two_mapped = map_to_fp(two_poly)
        three_mapped = map_to_fp(three_poly)
        
        # Create new MorphismWrappers with the mapped polynomials
        # Need to map the scaling factors too
        m_new = Km_p.gen()
        
        def map_scale(scale_expr):
            """Map a scaling expression from QQ(m) to Fp(m)"""
            if scale_expr == 1:
                return Km_p(1)
            # scale_expr is likely a power of m
            num = Pm_p(scale_expr.numerator())
            den = Pm_p(scale_expr.denominator())
            return Km_p(num) / Km_p(den)
        
        one_scale = map_scale(one_use.scale)
        two_scale = map_scale(two_use.scale)
        three_scale = map_scale(three_use.scale)
        
        # Create new wrappers over Fp(m)
        one_use = MorphismWrapper(one_mapped, one_use.k, one_scale, Pm_p(cd.a4))
        two_use = MorphismWrapper(two_mapped, two_use.k, two_scale, Pm_p(cd.a4))
        three_use = MorphismWrapper(three_mapped, three_use.k, three_scale, Pm_p(cd.a4))
        
        # Ensure the Weierstrass model itself is over the finite field
        if hasattr(cd, 'E_weier') and cd.E_weier.base_ring() != Km_p:
            cd.E_weier = cd.E_weier.change_ring(Km_p)

    ret = []
    seen = set()

    for pt in base_pts:
        xi_raw, yi_raw = pt[0], pt[1]
        
        # In index calculus mode, we treat these as native elements of Fp
        if FINITE_FIELD is not None:
            Fp = GF(FINITE_FIELD)
            xi = Fp(xi_raw)
            yi = Fp(yi_raw)
        else:
            xi, yi = xi_raw, yi_raw

        if (xi, yi) in seen:
            continue

        try:
            # Now evaluation happens entirely within Fp(m)
            X_aff = one_use(x=xi, y=yi)
            Y_aff = two_use(x=xi, y=yi)
            Z_aff = three_use(x=xi, y=yi)
            
            P = cd.E_weier([X_aff, Y_aff, Z_aff])
            ret.append(P)
            seen.add((xi, yi))
        except Exception as e:
            raise RuntimeError(f"Morphism evaluation failed for point ({xi}, {yi}) mod {FINITE_FIELD}: {e}")

    return ret


@PROFILE
def check_independence(sections, curve, cd):
    """
    Check linear independence of a list of sections.

    Returns (independent_bool, height_matrix_H).

    - In characteristic 0: use canonical/sample heights (existing behavior).
    - In finite-field mode: test independence in the group by randomized
      linear-combination testing (no heights, no SR).
    """

    n = len(sections)
    if n == 0:
        # no sections => not independent; return empty matrix for compatibility
        return False, matrix(QQ, 0)

    ff_mode = (FINITE_FIELD is not None)

    # ----------------------
    # Helper: robust identity test for a point object P
    # ----------------------
    def point_is_identity(P):
        # Try known point methods
        try:
            if hasattr(P, "is_zero"):
                return bool(P.is_zero())
        except Exception:
            raise
        try:
            if hasattr(P, "is_infinite"):
                return bool(P.is_infinite())
        except Exception:
            raise

        # Try equality with curve(0) only as a last resort (guarded)
        try:
            O = curve(0)
            try:
                return P == O
            except Exception:
                raise
        except Exception:
            # constructor not supported; skip
            raise

        # If the point exposes affine/projective coordinates, try heuristics
        try:
            if hasattr(P, "is_identity") and callable(P.is_identity):
                return bool(P.is_identity())
        except Exception:
            raise

        # Last-ditch: try to inspect coords (some point objects return None for infinity)
        try:
            coords = P.coordinates() if hasattr(P, "coordinates") else None
            if coords is None:
                return True
            # coords maybe tuple of length 2 or 3; if any entry is None, treat as infinity
            if isinstance(coords, (tuple, list)):
                return any(c is None for c in coords)
        except Exception:
            raise

        # Unknown; assume not identity (conservative for tests)
        return False

    # ----------------------
    # FINITE FIELD MODE: group-law independence (probabilistic)
    # ----------------------
    if ff_mode:
        print("--- Checking independence in finite-field mode (group law) ---")

        # Quick sanity: ensure none of the base sections are the identity
        for i, P in enumerate(sections):
            if point_is_identity(P):
                print(f"Section {i} is the identity (zero) section -> dependent")
                return False, None

        # Randomized linear-combination testing:
        # If we find a nontrivial integer relation sum a_i * P_i = O, declare dependent.
        # Parameters below trade speed vs confidence.
        import random
        NUM_TRIALS = 60         # more trials -> smaller false-positive prob
        MAX_ABS_COEFF = 15      # coefficients sampled uniformly from [-MAX..MAX]

        for trial in range(NUM_TRIALS):
            coeffs = [random.randint(-MAX_ABS_COEFF, MAX_ABS_COEFF) for _ in range(n)]
            if all(c == 0 for c in coeffs):
                continue

            # compute the linear combination S = sum c_i * P_i
            # use repeated addition / scalar multiplication; rely on Sage point arithmetic
            S = None
            for c, P in zip(coeffs, sections):
                if c == 0:
                    continue
                try:
                    term = c * P
                except Exception:
                    # try P * c as alternate multiplication order
                    term = P * c
                    raise
                if S is None:
                    S = term
                else:
                    S = S + term

            # If S is None (all coeffs 0) it's not interesting; otherwise test identity
            if S is None:
                continue

            if point_is_identity(S):
                print(f"Dependent: found relation {coeffs}")
                return False, None

        # no relation found (probabilistic evidence of independence)
        print("No nontrivial relations found after randomized tests — treating as independent.")
        return True, None

    # ----------------------
    # CHARACTERISTIC 0: original height-based logic unchanged
    # ----------------------
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
def lll_reduce_mw_basis(cd, P_list):
    """
    Reduce a Mordell–Weil basis.

    - In characteristic 0: perform true LLL reduction using the height pairing.
    - In finite-field mode: LLL is undefined; return a cleaned, deterministic basis.
    """

    r = len(P_list)
    if r == 0:
        return []

    ff_mode = (FINITE_FIELD is not None)

    # ------------------------------------------------------------
    # Finite field mode: NO LLL (no heights exist)
    # ------------------------------------------------------------
    if ff_mode:
        print("--- Finite-field mode: skipping LLL (no height lattice) ---")

        # Optional light normalization to keep things stable:
        # 1) remove identity sections
        # 2) remove obvious duplicates
        cleaned = []
        for P in P_list:
            try:
                if hasattr(P, "is_zero") and P.is_zero():
                    continue
            except Exception:
                raise

            if P not in cleaned:
                cleaned.append(P)

        return cleaned

    # ------------------------------------------------------------
    # Characteristic 0: real LLL on height lattice
    # ------------------------------------------------------------
    is_independent, H = check_independence(P_list, cd.E_curve, cd)

    if not is_independent:
        print("Warning: height matrix not full rank. Skipping LLL.")
        return P_list

    if H is None or H.nrows() != r:
        print("Warning: invalid height matrix. Skipping LLL.")
        return P_list

    # Clear denominators to get an integral Gram matrix
    try:
        denoms = [H[i, j].denominator() for i in range(r) for j in range(r)]
        D = lcm(denoms) if denoms else 1
        H_int = (H * D).change_ring(ZZ)
    except Exception as e:
        print("Failed to clear denominators in height matrix:", e)
        raise
        return P_list

    # Perform LLL on Gram matrix
    try:
        U = H_int.LLL_gram()
    except Exception as e:
        print("Height matrix not LLL-compatible. Skipping LLL.")
        print("Reason:", e)
        raise
        return P_list

    # Apply unimodular change of basis
    new_Ps = []
    for i in range(r):
        comb = None
        for j in range(r):
            c = U[j, i]
            if c == 0:
                continue
            term = c * P_list[j]
            comb = term if comb is None else comb + term
        new_Ps.append(comb)

    return new_Ps


# Finite-field compatible rationality test
@lru_cache(maxsize=None)
def get_y_unshifted_genus2(x):
    """
    Test if x gives a y-coordinate on the genus-2 curve y^2 = G(x).
    
    - In QQ mode: Returns y if rational, None otherwise.
    - In finite field mode: Returns y in GF(p) if it exists, None otherwise.
    """
    if FINITE_FIELD is not None:
        # Finite field mode
        F = GF(FINITE_FIELD)
        x_f = F(x)
        
        # Evaluate G(x) in F using Horner's method
        rhs = F(COEFFS_GENUS2[0])
        for coeff in COEFFS_GENUS2[1:]:
            rhs = rhs * x_f + F(coeff)
        
        # Check if rhs is a perfect square in F
        if not rhs.is_square():
            return None
        
        return rhs.sqrt()
    
    else:
        # Rational (QQ) mode - original logic
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
def get_phi_x(one, two, three, x_coord_func, quartic_rhs):
    """
    Compute phi_x = X_sub / Z_sub without global simplification.
    Supports QQ/SR mode and finite-field mode (GF(p)).

    This version fixes the coercion errors by substituting x first,
    coercing into the fraction field in 'm', then handling sqrt / rationalization.
    """
    ff_mode = FINITE_FIELD is not None

    if ff_mode:
        F = GF(FINITE_FIELD)
        PR_m = PolynomialRing(F, 'm')
        K = PR_m.fraction_field()

        # --- Coerce x_coord_func into K ---
        try:
            xK = K(x_coord_func)
        except Exception:
            try:
                if hasattr(x_coord_func, 'parent') and x_coord_func.parent() is PR_m:
                    xK = K(x_coord_func)
                else:
                    xK = K( QQ(str(x_coord_func)) )
            except Exception as e:
                raise RuntimeError(f"get_phi_x: cannot coerce x_coord_func to fraction field in m: {e}")

        # --- Substitute x := xK into quartic_rhs ---
        try:
            quartic_at_x = quartic_rhs.subs(x=xK)
        except Exception:
            try:
                quartic_at_x = quartic_rhs(x=xK)
            except Exception as e:
                raise RuntimeError(f"get_phi_x: failed to substitute x into quartic_rhs: {e}")

        # Coerce quartic_at_x into the fraction field K
        try:
            y_poly = K(quartic_at_x)
        except Exception:
            try:
                y_poly = PR_m(quartic_at_x)
                y_poly = K(y_poly)
            except Exception as e:
                raise RuntimeError(f"get_phi_x: failed to coerce quartic_at_x into fraction field K: {e}")

        # --- Attempt to get y_val_sqrt inside K ---
        y_val_sqrt = None
        try:
            is_const = False
            if hasattr(y_poly, 'is_constant'):
                is_const = y_poly.is_constant()
            elif hasattr(y_poly, 'numerator') and hasattr(y_poly, 'denominator'):
                 is_const = (y_poly.numerator().degree() <= 0 and y_poly.denominator().degree() <= 0)
            
            if is_const:
                const = F(y_poly)
                if const.is_square():
                    y_val_sqrt = const.sqrt()
            else:
                if y_poly.is_square():
                    y_val_sqrt = y_poly.sqrt()
        except Exception:
            y_val_sqrt = None

        # --- If we found a sqrt inside K ---
        if y_val_sqrt is not None:
            try:
                Z_sub = three.subs(x=xK, y=y_val_sqrt)
                X_sub = one.subs(x=xK, y=y_val_sqrt)
            except Exception as e:
                raise RuntimeError(f"get_phi_x: failed to substitute into X/Z with y_val_sqrt: {e}")

            if Z_sub == 0:
                return "INF"
            return X_sub / Z_sub

        # --- Rationalize phi = X_sub/Z_sub ---
        try:
            PR_Y = PolynomialRing(K, 'Y')
            Y = PR_Y.gen()

            X_as_poly = PR_Y(one.subs(x=xK, y=Y))
            Z_as_poly = PR_Y(three.subs(x=xK, y=Y))

            # FIX: Use .coefficient(n) for univariate polynomials in Sage
            a0 = X_as_poly.coefficient(0)
            a1 = X_as_poly.coefficient(1)
            b0 = Z_as_poly.coefficient(0)
            b1 = Z_as_poly.coefficient(1)

            numerator_0 = a0 * b0 - a1 * b1 * y_poly
            numerator_1 = (a1 * b0 - a0 * b1)
            denominator = b0 * b0 - b1 * b1 * y_poly

            if denominator == 0:
                return "INF"

            if numerator_1 == 0:
                return K(numerator_0) / K(denominator)

        except Exception:
            pass # Fall through to extension

        # --- Last resort: Quadratic extension ---
        try:
            T = PolynomialRing(K, 'T').gen()
            minimal = T**2 - K(y_poly)
            L = K.extension(minimal, 'Y')
            Y_L = L.gen()

            X_sub_L = L(one.subs(x=xK, y=Y_L))
            Z_sub_L = L(three.subs(x=xK, y=Y_L))

            if Z_sub_L == 0:
                return "INF"

            phiL = X_sub_L / Z_sub_L

            try:
                return K(phiL)
            except Exception:
                return phiL

        except Exception as e:
            raise RuntimeError(f"get_phi_x: failed to construct quadratic extension: {e}")

    else:
        y_val_sqrt = sqrt(quartic_rhs)
        Z_sub = three.subs(x=x_coord_func, y=y_val_sqrt)
        X_sub = one.subs(x=x_coord_func, y=y_val_sqrt)

        if Z_sub == 0:
            return "INF"

        return X_sub / Z_sub
