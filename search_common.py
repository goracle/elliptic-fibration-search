import sys, os, subprocess, warnings, itertools, multiprocessing, random, traceback, math
from typing import NamedTuple
from functools import lru_cache
from multiprocessing import TimeoutError
from sage.all import QQ, ZZ, RR, GF, SR, var, PolynomialRing, Matrix, matrix, vector, diff, floor, Curve, Jacobian, sqrt, CRT, lcm, primes, QuadraticForm, ceil, is_prime, Integer, log, next_prime, HyperellipticCurve, sage_eval, EllipticCurve, set_random_seed, valuation, gcd
from math import gcd, log
from cysignals.signals import SignalError
from prime_subgroup_projection import *
from parse_genus3 import *
from tate import *

_IS_MAIN_PROCESS = multiprocessing.current_process().name == 'MainProcess'
# === imports ===

# local modules

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
DATA_PTS_GENUS2 = [QQ(10598399)]
DATA_PTS_GENUS2 = [QQ(15998132)] # 1
DATA_PTS_GENUS2 = [QQ(12862063)] # 2
DATA_PTS_GENUS2 = [QQ(1)]
TERMINATE_WHEN_6 = 30

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
FINITE_FIELD = next_prime(2**16)
MAXN = 20 # since there is no notion of height on finite field mode, this serves as the max n for section multiple [n]P
SECRET_KEY = 800 # how many multiples of base genus 2 divisor to use to obtain the target starting from the base divisor from DATA_PTS_GENUS2[0]
BASE_DIVISOR, TARGET_DIVISOR, PREFERRED_X_COORDS = None, None, None # constructed below, here for reference
BLOCK_WIEDEMANN = False   # set True to always use block Wiedemann in the final solve
BLOCK_WIEDEMANN = True   # set True to always use block Wiedemann in the final solve

# 1) Generate the random point if requested
if DATA_PTS_GENUS2 is None:
    # Ensure we use the prime currently active in your pool
    _p_init = FINITE_FIELD
    DATA_PTS_GENUS2 = [get_random_x_on_hyperelliptic(COEFFS_GENUS2, _p_init)]
    if _IS_MAIN_PROCESS:
        print("after random:", DATA_PTS_GENUS2)

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

# project into mod ell subgroup to remove torsion/cofactor complications
# project into mod ell subgroup to remove torsion/cofactor complications
if FINITE_FIELD is not None:

    # Fix: Force deterministic generation of the cryptosystem parameters (G, Q)
    # so they are identical across all worker processes.
    set_random_seed(12345)

    GROUP_MODULUS, DATA_PTS_GENUS2, BASE_DIVISOR, TARGET_DIVISOR, PREFERRED_X_COORDS, SECRET_KEY = \
        setup_prime_subgroup_cryptosystem(
            FINITE_FIELD,
            COEFFS_GENUS2,
            DATA_PTS_GENUS2,
            SECRET_KEY,
            verbose=_IS_MAIN_PROCESS
        )

    assert len(PREFERRED_X_COORDS) == 4, PREFERRED_X_COORDS
    assert BASE_DIVISOR*SECRET_KEY == TARGET_DIVISOR, (BASE_DIVISOR*SECRET_KEY, TARGET_DIVISOR)
    assert (GROUP_MODULUS * BASE_DIVISOR).is_zero()
    assert (GROUP_MODULUS * TARGET_DIVISOR).is_zero()

    # Re-randomize the seed so that subsequent operations (like random walks in workers)
    # are not identical across processes.

    set_random_seed()
else:
    BASE_DIVISOR = TARGET_DIVISOR = PREFERRED_X_COORDS = None

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
    #R = PolynomialRing(E_rhs.base_ring(), 2, names=('x', 'y'))
    #R = PolynomialRing(E_rhs.base_ring(), 2, names=('x','y'), implementation='generic')
    R = PolynomialRing(E_rhs.base_ring(), ['x','y'], order='lex', implementation='generic')

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
        # Accept either a ring or a ring element
        if hasattr(a4_min, 'parent'):
            self.parent_ring = a4_min.parent()
        else:
            self.parent_ring = a4_min  # already a ring

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

# === Refactored Functions for search_common.py ===

@PROFILE
def buildcd(E_curve, phi_x, quartic_rhs, E_rhs, morph_triplet,
            verify=True, compute_minimal=USE_MINIMAL_MODEL):
    """
    Builds the CurveDataExt object for the fibration.

    Bimodal operation:
    - FINITE_FIELD mode: builds over GF(p)(m)
    - QQ mode: builds over QQ(m) with minimal model computation

    Args:
        E_curve: Quartic curve object
        phi_x: x-coordinate morphism
        quartic_rhs: RHS of quartic equation
        E_rhs: Weierstrass RHS
        morph_triplet: (one, two, three) coordinate maps
        verify: Run geometric validation
        compute_minimal: Compute minimal Weierstrass model

    Returns:
        CurveDataExt object with all fibration data
    """
    print("="*70)
    print("BUILDCD: Constructing Fibration Data")
    print("="*70)
    sys.stdout.flush()

    # ========================================================================
    # MODE DETECTION AND FIELD SETUP
    # ========================================================================

    ff_mode = (FINITE_FIELD is not None)

    if ff_mode:
        base_field = GF(FINITE_FIELD)
        print(f"[buildcd] Mode: FINITE_FIELD GF({FINITE_FIELD})")

        # Build function field over finite field
        Pm_base = PolynomialRing(base_field, 'm')
        Fm = FractionField(Pm_base)
        m = Fm.gen()
    else:
        base_field = QQ
        print("[buildcd] Mode: RATIONAL (QQ)")

        # Build function field over QQ
        Pm_base = PolynomialRing(QQ, 'm')
        Fm = FractionField(Pm_base)
        m = Fm.gen()

    sys.stdout.flush()

    # ========================================================================
    # EXTRACT RAW WEIERSTRASS MODEL
    # ========================================================================

    print("[buildcd] Extracting Weierstrass coefficients...")
    sys.stdout.flush()

    try:
        E_weier_raw = Jacobian(E_curve)
        a4_raw = E_weier_raw.a4()
        a6_raw = E_weier_raw.a6()
    except Exception as e:
        raise RuntimeError(f"buildcd: failed to extract Jacobian coefficients: {e}")

    # Coerce into function field
    try:
        a4_raw = Fm(a4_raw)
        a6_raw = Fm(a6_raw)
    except Exception as e:
        raise RuntimeError(f"buildcd: failed to coerce a4/a6 into Fm: {e}")

    print(f"[buildcd] Raw a4 degree: {a4_raw.numerator().degree() if hasattr(a4_raw.numerator(), 'degree') else 'unknown'}")
    print(f"[buildcd] Raw a6 degree: {a6_raw.numerator().degree() if hasattr(a6_raw.numerator(), 'degree') else 'unknown'}")
    sys.stdout.flush()

    # ========================================================================
    # MINIMAL MODEL COMPUTATION (QQ mode only)
    # ========================================================================

    one, two, three = morph_triplet

    if compute_minimal and not ff_mode:
        print("\n" + "="*70)
        print("MINIMAL MODEL COMPUTATION")
        print("="*70)
        sys.stdout.flush()

        a4_final = a4_raw
        a6_final = a6_raw
        phi_x_final = phi_x
        blowup_factor = 0
        blowdown_0 = 0

        # --------------------------------------------------------------------
        # STEP 1: Handle poles at m=0 (blow-up)
        # --------------------------------------------------------------------

        print("\n[Step 1] Checking for poles at m=0...")
        sys.stdout.flush()

        try:
            v4 = min_order_in_m(a4_raw, m)
            v6 = min_order_in_m(a6_raw, m)
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to compute m-valuations: {e}")

        print(f"  v_m(a4) = {v4}")
        print(f"  v_m(a6) = {v6}")
        sys.stdout.flush()

        if v4 < 0 or v6 < 0:
            k_for_a4 = ceil(-v4 / 4) if v4 < 0 else 0
            k_for_a6 = ceil(-v6 / 6) if v6 < 0 else 0
            blowup_factor = int(max(k_for_a4, k_for_a6))

            assert blowup_factor > 0, "buildcd: computed blowup_factor <= 0 despite negative valuations"

            print(f"Poles detected! Applying blow-up with k={blowup_factor}")
            sys.stdout.flush()

            try:
                a4_final = a4_raw * m**(4 * blowup_factor)
                a6_final = a6_raw * m**(6 * blowup_factor)
                phi_x_final = phi_x / (m**(2 * blowup_factor))
            except Exception as e:
                raise RuntimeError(f"buildcd: blow-up transformation failed: {e}")

            # Verify poles removed
            v4_new = min_order_in_m(a4_final, m)
            v6_new = min_order_in_m(a6_final, m)
            assert v4_new >= 0 and v6_new >= 0, \
                f"buildcd: blow-up failed to remove poles (v4={v4_new}, v6={v6_new})"

            print(f"Blow-up complete: v_m(a4')={v4_new}, v_m(a6')={v6_new}")
            sys.stdout.flush()
        else:
            print("No poles at m=0")
            sys.stdout.flush()

        # --------------------------------------------------------------------
        # STEP 2: Handle common zeros at m=0 (blow-down)
        # --------------------------------------------------------------------

        print("\n[Step 2] Checking for common zeros at m=0...")
        sys.stdout.flush()

        blowdown_rounds = 0
        while True:
            try:
                v4_0 = min_order_in_m(a4_final, m)
                v6_0 = min_order_in_m(a6_final, m)
            except Exception as e:
                raise RuntimeError(f"buildcd: failed to compute valuations in blow-down: {e}")

            k0 = int(min(v4_0 // 4, v6_0 // 6)) if (v4_0 > 0 and v6_0 > 0) else 0

            if k0 <= 0:
                break

            print(f"  Round {blowdown_rounds + 1}: Applying blow-down with k={k0}")
            sys.stdout.flush()

            try:
                a4_final = a4_final / (m**(4 * k0))
                a6_final = a6_final / (m**(6 * k0))
                phi_x_final = phi_x_final * (m**(2 * k0))
            except Exception as e:
                raise RuntimeError(f"buildcd: blow-down transformation failed: {e}")

            blowdown_0 += k0
            blowdown_rounds += 1

            # Safety check: prevent infinite loops
            assert blowdown_rounds < 100, \
                "buildcd: blow-down exceeded 100 iterations (likely infinite loop)"

        if blowdown_0 > 0:
            print(f"Blow-down complete: removed m^({4*blowdown_0}) from a4, m^({6*blowdown_0}) from a6")
        else:
            print("No common zeros at m=0")
        sys.stdout.flush()

        # --------------------------------------------------------------------
        # STEP 3: Build minimal Weierstrass model and wrap morphisms
        # --------------------------------------------------------------------

        print("\n[Step 3] Building minimal Weierstrass model...")
        sys.stdout.flush()

        try:
            E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to construct minimal Weierstrass curve: {e}")

        # Verify discriminant non-zero
        try:
            Delta_final = E_weier_final.discriminant()
            assert Delta_final.numerator() != 0, \
                "buildcd: minimal model has zero discriminant"
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to verify discriminant: {e}")

        print(f"Minimal model constructed")
        sys.stdout.flush()

        # Compute net scaling for morphisms
        net_k = int(blowup_factor - blowdown_0)
        print(f"  Net scaling exponent: {net_k} (blow-up: {blowup_factor}, blow-down: {blowdown_0})")
        sys.stdout.flush()

        x_morphism_scale = m**(2 * net_k)
        y_morphism_scale = m**(3 * net_k)

        # Wrap morphisms with scaling
        try:
            one_s = MorphismWrapper(one, 1, x_morphism_scale, a4_final)
            two_s = MorphismWrapper(two, 1, y_morphism_scale, a4_final)
            three_s = MorphismWrapper(three, 1, 1, a4_final)
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to create morphism wrappers: {e}")

        print("="*70)
        print("MINIMAL MODEL SUMMARY")
        print("="*70)
        print(f"Blow-up factor:   {blowup_factor}")
        print(f"Blow-down factor: {blowdown_0}")
        print(f"Net scaling:      {net_k}")
        print(f"X-coord scale:    m^{2*net_k}")
        print(f"Y-coord scale:    m^{3*net_k}")
        print("="*70)
        sys.stdout.flush()

    else:
        # ====================================================================
        # NON-MINIMAL OR FINITE FIELD MODE
        # ====================================================================

        if ff_mode:
            print("\n[buildcd] Finite field mode: using raw model (no minimization)")
        else:
            print("\n[buildcd] Non-minimal mode: using raw model")
        sys.stdout.flush()

        a4_final = a4_raw
        a6_final = a6_raw
        phi_x_final = phi_x
        blowup_factor = 0

        try:
            E_weier_final = EllipticCurve(Fm, [0, 0, 0, a4_final, a6_final])
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to construct Weierstrass curve: {e}")

        # No scaling - identity wrappers
        try:
            one_s = MorphismWrapper(one, 1, 1, a4_final)
            two_s = MorphismWrapper(two, 1, 1, a4_final)
            three_s = MorphismWrapper(three, 1, 1, a4_final)
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to create morphism wrappers: {e}")

    # ========================================================================
    # SR COERCION (QQ mode only)
    # ========================================================================

    if ff_mode:
        # No SR in finite field mode
        SR_a4 = a4_final
        SR_a6 = a6_final
        SR_phi_x = phi_x_final
        SR_m = m
        print("[buildcd] Skipping SR coercion (finite field mode)")
    else:
        # Coerce to SR for symbolic manipulation
        print("[buildcd] Coercing to SR for symbolic operations...")
        sys.stdout.flush()

        try:
            SR_m = var('m')
            SR_a4 = SR(a4_final)
            SR_a6 = SR(a6_final)
            SR_phi_x = SR(phi_x_final)
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to coerce to SR: {e}")

        print("SR coercion complete")

    sys.stdout.flush()

    # ========================================================================
    # COMPUTE BAD PRIMES
    # ========================================================================

    print("\n[buildcd] Computing bad primes...")
    sys.stdout.flush()

    if ff_mode:
        # Only characteristic is bad in finite field mode
        bad_primes = [FINITE_FIELD]
        print(f"  Finite field mode: p={FINITE_FIELD} is the only bad prime")
    else:
        # Test each prime in pool
        class TempCD:
            def __init__(self, a4, a6):
                self.a4, self.a6 = a4, a6

        temp_cd = TempCD(a4_final, a6_final)

        try:
            bad_primes = [p for p in PRIME_POOL if not is_good_prime_for_surface(temp_cd, p)]
        except Exception as e:
            raise RuntimeError(f"buildcd: failed to compute bad primes: {e}")

        print(f"Found {len(bad_primes)} bad primes: {sorted(bad_primes)[:10]}{'...' if len(bad_primes) > 10 else ''}")

    sys.stdout.flush()

    # ========================================================================
    # COMPUTE SINGULAR FIBERS (QQ mode only)
    # ========================================================================

    if ff_mode:
        print("\n[buildcd] Skipping singular fiber analysis (finite field mode)")
        singfibs = {'fibers': [], 'euler_characteristic': 0, 'sigma_sum': 0}
    else:
        print("\n[buildcd] Computing singular fibers...")
        sys.stdout.flush()

        try:
            singfibs = find_singular_fibers(a4=a4_final, a6=a6_final, verbose=DEBUG)
        except Exception as e:
            raise RuntimeError(f"buildcd: singular fiber computation failed: {e}")

        print(f"  Found {len(singfibs.get('fibers', []))} singular fibers")
        print(f"  Euler characteristic: {singfibs.get('euler_characteristic', 'unknown')}")
        print(f"  Sigma sum: {singfibs.get('sigma_sum', 'unknown')}")

    sys.stdout.flush()

    # ========================================================================
    # BUILD E_RHS (for compatibility)
    # ========================================================================

    if ff_mode:
        E_rhs_final = E_rhs
    else:
        try:
            y, x = var('y x')
            E_rhs_final = y**2 - x**3 - a4_final * x - a6_final
        except Exception:
            E_rhs_final = E_rhs

    # ========================================================================
    # PACKAGE CURVE DATA
    # ========================================================================

    print("\n[buildcd] Packaging CurveDataExt...")
    sys.stdout.flush()

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
        base_field=base_field if ff_mode else None
    )

    # ========================================================================
    # POST-PROCESSING: 2-adic scaling (QQ mode only)
    # ========================================================================

    if not ff_mode:
        print("\n[buildcd] Attempting 2-adic scaling...")
        sys.stdout.flush()

        try:
            cd = try_scale_out_power_of_two(cd, max_t=2, debug=DEBUG)
            print("2-adic scaling complete")
        except Exception as e:
            print(f"2-adic scaling failed (continuing anyway): {e}")

    sys.stdout.flush()

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    if verify and DEBUG and not ff_mode:
        print("\n[buildcd] Running geometric validation...")
        sys.stdout.flush()

        try:
            validate_fibration_geometry(cd)
        except Exception as e:
            raise RuntimeError(f"buildcd: geometric validation failed: {e}")

        # Final discriminant check
        try:
            Delta_check = cd.E_weier.discriminant()
            assert Delta_check.numerator() != 0, \
                "buildcd: final discriminant is zero"
        except Exception as e:
            raise RuntimeError(f"buildcd: final discriminant check failed: {e}")

    print("\n" + "="*70)
    print("BUILDCD COMPLETE")
    print("="*70)
    sys.stdout.flush()

    return cd

def reduce_mod_quartic(expr, y, y2):
    """
    Reduce an expression modulo y^2 = y2 so that result is A + B*y.
    Works in polynomial ring.
    """
    R = expr.parent()
    PR = PolynomialRing(R.base_ring(), 'Y')
    Y = PR.gen()

    f = PR(expr.subs({y: Y}))

    A = PR(0)
    B = PR(0)

    for i in range(f.degree() + 1):
        coeff = f.coefficient(i)
        if i % 2 == 0:
            A += coeff * (y2 ** (i // 2))
        else:
            B += coeff * (y2 ** ((i - 1) // 2))

    return A, B

def reduce_Y_powers(poly, y_poly):
    """
    Reduce powers of Y using Y^2 = y_poly.
    Returns polynomial of degree ≤ 1 in Y.
    """
    result = 0
    for exp in range(poly.degree() + 1):
        coeff = poly.coefficient(exp)
        if coeff == 0:
            continue

        if exp == 0:
            result += coeff
        elif exp == 1:
            result += coeff * Y
        else:
            # Y^n = Y^(n % 2) * (y_poly)^(n // 2)
            k = exp // 2
            r = exp % 2
            term = coeff * (y_poly ** k)
            if r == 1:
                term *= Y
            result += term

    return result

@PROFILE
def check_independence(sections, curve, cd):
    """
    Check linear independence of sections.

    Bimodal operation:
    - FINITE_FIELD mode: probabilistic group-law test
    - QQ mode: height pairing matrix determinant

    Returns:
        (bool, Matrix or None): (independent, height_matrix)
    """
    n = len(sections)
    if n == 0:
        return False, matrix(QQ, 0)

    if n == 1:
        return True, None

    ff_mode = (FINITE_FIELD is not None)

    print(f"\n[check_independence] Testing {n} sections (mode={'FF' if ff_mode else 'QQ'})")
    sys.stdout.flush()

    # ========================================================================
    # Helper: robust identity test
    # ========================================================================

    def point_is_identity(P):
        """Test if P is the identity/zero element."""
        # Try is_zero method
        try:
            if hasattr(P, "is_zero"):
                return bool(P.is_zero())
        except Exception:
            raise

        # Try is_infinite method
        try:
            if hasattr(P, "is_infinite"):
                return bool(P.is_infinite())
        except Exception:
            raise

        # Try equality with curve(0)
        try:
            O = curve(0)
            return P == O
        except Exception:
            raise

        # Try is_identity method
        try:
            if hasattr(P, "is_identity") and callable(P.is_identity):
                return bool(P.is_identity())
        except Exception:
            raise

        # Check coordinates for None (infinity convention)
        try:
            coords = P.coordinates() if hasattr(P, "coordinates") else None
            if coords is None:
                return True
            if isinstance(coords, (tuple, list)):
                return any(c is None for c in coords)
        except Exception:
            raise

        # Conservative: assume not identity
        return False

    # ========================================================================
    # FINITE FIELD MODE: Probabilistic Group-Law Test
    # ========================================================================

    if ff_mode:
        print("[check_independence FF] Using probabilistic linear combination test")
        sys.stdout.flush()

        # Quick check: no identity sections
        for i, P in enumerate(sections):
            if point_is_identity(P):
                print(f"  Section {i} is identity -> dependent")
                sys.stdout.flush()
                return False, None

        # Probabilistic test: find nontrivial relation
        NUM_TRIALS = 60 if n > 1 else 0
        MAX_ABS_COEFF = 15

        if n > 1:
            print(f"  Running {NUM_TRIALS} random linear combination tests...")
        sys.stdout.flush()

        for trial in range(NUM_TRIALS):
            coeffs = [random.randint(-MAX_ABS_COEFF, MAX_ABS_COEFF) for _ in range(n)]

            if all(c == 0 for c in coeffs):
                continue

            # Compute S = sum(c_i * P_i)
            S = None
            for c, P in zip(coeffs, sections):
                if c == 0:
                    continue

                try:
                    term = c * P
                except Exception:
                    try:
                        term = P * c
                    except Exception as e:
                        raise RuntimeError(f"check_independence (FF): scalar multiplication failed: {e}")
                    raise

                if S is None:
                    S = term
                else:
                    S = S + term

            if S is None:
                continue

            if point_is_identity(S):
                print(f"  âŒ Dependent: found relation {coeffs}")
                sys.stdout.flush()
                return False, None

        if n > 1:
            print(f"No relations found in {NUM_TRIALS} trials -> independent (probabilistic)")
        sys.stdout.flush()
        return True, None

    # ========================================================================
    # QQ MODE: Height Pairing Matrix
    # ========================================================================

    print("[check_independence QQ] Computing height pairing matrix")
    sys.stdout.flush()

    H = None

    if USE_MINIMAL_MODEL:
        print("Using canonical height pairing (minimal model)")
        sys.stdout.flush()

        try:
            H = compute_canonical_height_matrix(sections, cd)
        except Exception as e:
            raise RuntimeError(f"check_independence (QQ): canonical height computation failed: {e}")

    else:
        print("  Using coarse height sampling (non-minimal model)")
        sys.stdout.flush()

        try:
            H = compute_coarse_height_matrix_serializable(cd, sections)
        except Exception as e:
            print(f"  Coarse sampling failed: {e}")
            print("  Falling back to naive pairing...")
            sys.stdout.flush()

            try:
                H = matrix(QQ, n)
                for i in range(n):
                    for j in range(i, n):
                        val = naive_pairing(sections[i], sections[j])
                        H[i, j] = val
                        H[j, i] = val
            except Exception as e2:
                raise RuntimeError(f"check_independence (QQ): all height methods failed: {e2}")

    if H is None or H.nrows() == 0:
        print("Failed to compute height matrix")
        sys.stdout.flush()
        return False, matrix(QQ, 0)

    # Check determinant
    try:
        det = H.det()
    except Exception as e:
        raise RuntimeError(f"check_independence (QQ): determinant computation failed: {e}")

    independent = (det != 0)

    print(f"  Height matrix determinant: {det}")
    print(f"  Result: {'independent' if independent else 'DEPENDENT'}")
    sys.stdout.flush()

    return independent, H

@PROFILE
def lll_reduce_mw_basis(cd, P_list):
    """
    Reduce a Mordell-Weil basis using LLL.

    Bimodal operation:
    - FINITE_FIELD mode: No LLL (no heights), just clean duplicates
    - QQ mode: True LLL reduction on height lattice

    Args:
        cd: CurveDataExt object
        P_list: List of section points

    Returns:
        List of reduced sections
    """
    r = len(P_list)
    if r == 0:
        return []

    ff_mode = (FINITE_FIELD is not None)

    print(f"\n[lll_reduce] Processing {r} sections (mode={'FF' if ff_mode else 'QQ'})")
    sys.stdout.flush()

    # ========================================================================
    # FINITE FIELD MODE: No LLL available
    # ========================================================================

    if ff_mode:
        print("[lll_reduce FF] Skipping LLL (no height structure)")
        sys.stdout.flush()

        # Light cleaning: remove identity and duplicates
        cleaned = []
        for P in P_list:
            # Check if identity
            try:
                if hasattr(P, "is_zero") and P.is_zero():
                    continue
            except Exception:
                pass

            # Check if already present
            if P not in cleaned:
                cleaned.append(P)

        print(f"[lll_reduce FF] Cleaned: {r} -> {len(cleaned)} sections")
        sys.stdout.flush()
        return cleaned

    # ========================================================================
    # QQ MODE: True LLL Reduction
    # ========================================================================

    print("[lll_reduce QQ] Computing height matrix for LLL")
    sys.stdout.flush()

    # Check independence and get height matrix
    try:
        is_independent, H = check_independence(P_list, cd.E_curve, cd)
    except Exception as e:
        raise RuntimeError(f"lll_reduce: independence check failed: {e}")

    if not is_independent:
        print("[lll_reduce QQ] WARNING: sections not independent, skipping LLL")
        sys.stdout.flush()
        return P_list

    if H is None or H.nrows() != r:
        print("[lll_reduce QQ] WARNING: invalid height matrix, skipping LLL")
        sys.stdout.flush()
        return P_list

    # Clear denominators to get integer Gram matrix
    print("[lll_reduce QQ] Clearing denominators...")
    sys.stdout.flush()

    try:
        denoms = [H[i, j].denominator() for i in range(r) for j in range(r)]
        D = lcm(denoms) if denoms else 1
        H_int = (H * D).change_ring(ZZ)
    except Exception as e:
        raise RuntimeError(f"lll_reduce: failed to clear denominators: {e}")

    print(f"[lll_reduce QQ] Cleared with LCM = {D}")
    sys.stdout.flush()

    # Perform LLL
    print("[lll_reduce QQ] Running LLL algorithm...")
    sys.stdout.flush()

    try:
        U = H_int.LLL_gram()
    except Exception as e:
        print(f"[lll_reduce QQ] LLL failed: {e}")
        print("[lll_reduce QQ] Returning unreduced basis")
        sys.stdout.flush()
        return P_list

    print("[lll_reduce QQ] LLL complete, applying transformation...")
    sys.stdout.flush()

    # Apply unimodular transformation
    new_Ps = []
    for i in range(r):
        comb = None
        for j in range(r):
            c = U[j, i]
            if c == 0:
                continue

            try:
                term = c * P_list[j]
            except Exception as e:
                raise RuntimeError(f"lll_reduce: scalar multiplication failed: {e}")

            if comb is None:
                comb = term
            else:
                comb = comb + term

        assert comb is not None, "lll_reduce: transformation produced null combination"
        new_Ps.append(comb)

    print(f"[lll_reduce QQ] Reduced basis has {len(new_Ps)} sections")
    sys.stdout.flush()

    return new_Ps

@PROFILE
def compute_base_sections_m(cd, base_pts, tower=None):
    """
    Map hyperelliptic points to Weierstrass sections.

    Bimodal operation:
    - FINITE_FIELD mode: evaluation stays in GF(p)(m)
    - QQ mode: evaluation in QQ(m)

    Args:
        cd: CurveDataExt object with morphisms
        base_pts: List of (x, y) points on hyperelliptic curve
        tower: Optional tower data (unused currently)

    Returns:
        List of Weierstrass points (sections)
    """
    if not base_pts:
        return []

    ff_mode = (FINITE_FIELD is not None)

    print(f"\n[compute_base_sections] Mapping {len(base_pts)} points (mode={'FF' if ff_mode else 'QQ'})")
    sys.stdout.flush()

    one_use, two_use, three_use = cd.morphs

    # ========================================================================
    # FINITE FIELD MODE: Ensure morphisms work in GF(p)(m)
    # ========================================================================

    if ff_mode:
        print("[compute_base_sections FF] Setting up GF(p)(m) arithmetic")
        sys.stdout.flush()

        Fp = GF(FINITE_FIELD)
        Pm_p = PolynomialRing(Fp, 'm')
        Km_p = FractionField(Pm_p)

        # Helper to map QQ(m) polynomials to Fp(m)
        def map_to_fp(poly):
            """Map polynomial from QQ(m) to Fp(m)."""
            try:
                new_parent = PolynomialRing(Km_p, poly.parent().variable_names())
                d = poly.dict()
                new_d = {}
                for mon, coeff in d.items():
                    # coeff is in QQ(m)
                    num_p = Pm_p(coeff.numerator())
                    den_p = Pm_p(coeff.denominator())
                    new_d[mon] = Km_p(num_p) / Km_p(den_p)
                return new_parent(new_d)
            except Exception as e:
                raise RuntimeError(f"map_to_fp failed: {e}")

        def map_scale(scale_expr):
            """Map scaling expression from QQ(m) to Fp(m)."""
            if scale_expr == 1:
                return Km_p(1)
            try:
                num = Pm_p(scale_expr.numerator())
                den = Pm_p(scale_expr.denominator())
                return Km_p(num) / Km_p(den)
            except Exception as e:
                raise RuntimeError(f"map_scale failed: {e}")

        # Extract and map the morphism polynomials
        try:
            one_poly = one_use.callable_obj
            two_poly = two_use.callable_obj
            three_poly = three_use.callable_obj

            one_mapped = map_to_fp(one_poly)
            two_mapped = map_to_fp(two_poly)
            three_mapped = map_to_fp(three_poly)

            one_scale = map_scale(one_use.scale)
            two_scale = map_scale(two_use.scale)
            three_scale = map_scale(three_use.scale)

            # Create new wrappers over Fp(m)
            one_use   = MorphismWrapper(one_mapped,   one_use.k,   one_scale,   Pm_p)
            two_use   = MorphismWrapper(two_mapped,   two_use.k,   two_scale,   Pm_p)
            three_use = MorphismWrapper(three_mapped, three_use.k, three_scale, Pm_p)
        except Exception as e:
            raise RuntimeError(f"compute_base_sections (FF): morphism mapping failed: {e}")

        # Ensure Weierstrass curve is over correct field
        try:
            if hasattr(cd, 'E_weier') and cd.E_weier.base_ring() != Km_p:
                cd.E_weier = cd.E_weier.change_ring(Km_p)
        except Exception as e:
            raise RuntimeError(f"compute_base_sections (FF): curve base change failed: {e}")

        print("[compute_base_sections FF] Morphisms mapped to Fp(m)")
        sys.stdout.flush()

    # ========================================================================
    # COMMON: Apply morphisms to all points
    # ========================================================================

    ret = []
    seen = set()

    for idx, pt in enumerate(base_pts):
        xi_raw, yi_raw = pt[0], pt[1]

        # Coerce to appropriate field
        if ff_mode:
            Fp = GF(FINITE_FIELD)
            try:
                xi = Fp(xi_raw)
                yi = Fp(yi_raw)
            except Exception as e:
                raise RuntimeError(f"compute_base_sections (FF): point {idx} coercion failed: {e}")
        else:
            xi, yi = xi_raw, yi_raw

        if (xi, yi) in seen:
            continue

        # Apply morphism
        try:
            X_aff = one_use(x=xi, y=yi)
            Y_aff = two_use(x=xi, y=yi)
            Z_aff = three_use(x=xi, y=yi)
        except Exception as e:
            raise RuntimeError(f"compute_base_sections: morphism evaluation failed at point {idx}: {e}")

        # Construct Weierstrass point
        try:
            P = cd.E_weier([X_aff, Y_aff, Z_aff])
        except Exception as e:
            raise RuntimeError(f"compute_base_sections: point construction failed at point {idx}: {e}")

        ret.append(P)
        seen.add((xi, yi))

    print(f"[compute_base_sections] Mapped to {len(ret)} sections")
    sys.stdout.flush()

    return ret

@PROFILE
def verify_morphism_on_samples(cd, base_pts):
    """
    Verify that morphism images lie on Weierstrass curve.

    Tests each base point to ensure X,Y,Z satisfy the Weierstrass equation.

    Args:
        cd: CurveDataExt with morphs and E_weier
        base_pts: List of (x,y) points to test

    Returns:
        True if all points verify (raises on failure)
    """
    one_s, two_s, three_s = cd.morphs
    E_min = cd.E_weier
    base_field = getattr(cd, 'base_field', None)

    print(f"\n[verify_morphism] Testing {len(base_pts)} sample points")
    sys.stdout.flush()

    for idx, (xi, yi) in enumerate(base_pts):
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
            raise RuntimeError(f"verify_morphism: verification failed at point {idx} ({xi}, {yi}): {e}")

    print("[verify_morphism] All sample points verified")
    sys.stdout.flush()

    return True

@PROFILE
def compute_search_vectors(H, height_bound):
    """
    Enumerate short vectors in the height lattice.

    Uses LLL on the Gram matrix to find all vectors with height <= bound.

    Args:
        H: Height pairing matrix (symmetric, positive definite)
        height_bound: Maximum height for enumeration

    Returns:
        List of tuples representing short vectors
    """
    print(f"\n[compute_search_vectors] Enumerating vectors with height <= {height_bound}")
    sys.stdout.flush()

    H_matrix = matrix(H)
    n = H_matrix.nrows()

    assert n > 0, "compute_search_vectors: empty height matrix"

    # Clear denominators
    print("[compute_search_vectors] Clearing denominators...")
    sys.stdout.flush()

    try:
        denominators = [H_matrix[i,j].denominator()
                       for i in range(n) for j in range(n)]
        lcm_denom = lcm(denominators) if denominators else 1
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: denominator extraction failed: {e}")

    print(f"[compute_search_vectors] LCM of denominators: {lcm_denom}")
    sys.stdout.flush()

    # Scale to integer matrix with even diagonal
    try:
        H_scaled = lcm_denom * H_matrix
        H_even = 2 * H_scaled

        # Ensure exact symmetry
        H_even = (H_even + H_even.transpose())
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: scaling failed: {e}")

    # Check positive definiteness
    print("[compute_search_vectors] Checking positive definiteness...")
    sys.stdout.flush()

    try:
        is_pd = H_even.is_positive_definite()
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: PD check failed: {e}")

    if not is_pd:
        det = H_even.det()
        print(f"[compute_search_vectors] WARNING: Not PD (det={det}, rank={H_even.rank()})")

        try:
            evals = H_even.eigenvalues()
            min_ev = min([RR(ev) for ev in evals])
            print(f"[compute_search_vectors] Min eigenvalue (approx): {min_ev}")
        except Exception:
            print("[compute_search_vectors] Could not compute eigenvalues")

        sys.stdout.flush()

        # Regularize by adding eps*I
        print("[compute_search_vectors] Regularizing with diagonal shift...")
        sys.stdout.flush()

        eps = Integer(1)
        H_try = H_even + eps * Matrix.identity(n)
        attempts = 0

        while not H_try.is_positive_definite():
            eps *= 2
            H_try = H_even + eps * Matrix.identity(n)
            attempts += 1

            assert attempts < 60, "compute_search_vectors: regularization exceeded 60 iterations"

        # Make eps even (QuadraticForm requires even diagonal)
        if eps % 2 == 1:
            eps += 1
            H_try = H_even + eps * Matrix.identity(n)

        # Ensure still PD after rounding
        while not H_try.is_positive_definite():
            eps += 2
            H_try = H_even + eps * Matrix.identity(n)

        # Binary search to minimize eps
        lo = Integer(0)
        hi = Integer(eps)
        while hi - lo > 2:
            mid = lo + (hi - lo) // 2
            if mid % 2 == 1:
                mid += 1
            if (H_even + mid * Matrix.identity(n)).is_positive_definite():
                hi = mid
            else:
                lo = mid
        eps = hi

        H_even = H_even + eps * Matrix.identity(n)
        print(f"[compute_search_vectors] Added eps*I with eps={eps}")

        try:
            evals_new = H_even.eigenvalues()
            min_ev_new = min([RR(ev) for ev in evals_new])
            print(f"[compute_search_vectors] New min eigenvalue (approx): {min_ev_new}")
        except Exception:
            pass

        sys.stdout.flush()

    # Convert to integer matrix over ZZ
    try:
        H_even = matrix(ZZ, H_even)
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: ZZ conversion failed: {e}")

    # Verify diagonal is even
    for i in range(n):
        assert int(H_even[i,i]) % 2 == 0, \
            f"compute_search_vectors: diagonal entry H[{i},{i}]={H_even[i,i]} is odd"

    # Build quadratic form and enumerate
    print("[compute_search_vectors] Building quadratic form...")
    sys.stdout.flush()

    try:
        Q = QuadraticForm(ZZ, H_even)
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: QuadraticForm construction failed: {e}")

    scaled_height_bound = 2 * lcm_denom * height_bound
    print(f"[compute_search_vectors] Enumerating up to scaled bound {scaled_height_bound}")
    sys.stdout.flush()

    try:
        vecs = Q.short_vector_list_up_to_length(scaled_height_bound)
        vecs = [v for sublist in vecs for v in sublist]
    except Exception as e:
        raise RuntimeError(f"compute_search_vectors: enumeration failed: {e}")

    print(f"[compute_search_vectors] Found {len(vecs)} vectors")
    sys.stdout.flush()

    return vecs

@PROFILE
def canonicalize_by_sign(vecs):
    """
    Canonicalize vectors by making first non-zero element positive.

    Removes duplicates modulo sign: (4,) and (-4,) both map to (4,).

    Args:
        vecs: List of vectors (tuples or lists)

    Returns:
        List of canonical tuples (first nonzero positive)
    """
    seen = set()
    out = []

    for v in vecs:
        vt = tuple(int(x) for x in v)

        # Skip zero vector
        if all(x == 0 for x in vt):
            continue

        # Find first non-zero element
        first_nonzero_idx = None
        for i, x in enumerate(vt):
            if x != 0:
                first_nonzero_idx = i
                break

        if first_nonzero_idx is None:
            continue  # All zeros (shouldn't happen due to check above)

        # Canonicalize: make first non-zero positive
        if vt[first_nonzero_idx] < 0:
            can = tuple(-x for x in vt)
        else:
            can = vt

        if can not in seen:
            seen.add(can)
            out.append(can)

    return out

@PROFILE
def validate_fibration_geometry(cd):
    """
    Run geometric validation checks on the fibration.

    Verifies:
    - Discriminant is non-zero
    - Discriminant degree matches expectations
    - Weierstrass scaling is consistent

    Args:
        cd: CurveDataExt object

    Returns:
        True (raises on validation failure)
    """
    print("\n" + "="*70)
    print("GEOMETRIC VALIDATION")
    print("="*70)
    sys.stdout.flush()

    try:
        a4 = cd.a4
        a6 = cd.a6
        m_var = a4.parent().gen() if hasattr(a4, 'parent') else None
    except Exception as e:
        raise RuntimeError(f"validate_fibration: failed to extract coefficients: {e}")

    # Compute discriminant
    try:
        Delta = -16 * (4 * a4**3 + 27 * a6**2)
    except Exception as e:
        raise RuntimeError(f"validate_fibration: discriminant computation failed: {e}")

    # Check non-zero
    if Delta.is_zero():
        print("[validate_fibration] FAIL: Discriminant is identically zero")
        sys.stdout.flush()
        raise ValueError("validate_fibration: discriminant is zero")

    print("[validate_fibration] Discriminant is non-zero")
    sys.stdout.flush()

    # Compute effective degree
    try:
        effective_degree_val = effective_degree(Delta, m_var)
    except Exception as e:
        print(f"[validate_fibration] WARNING: Could not compute effective degree: {e}")
        effective_degree_val = "unknown"

    print(f"[validate_fibration] Discriminant: {Delta}")
    print(f"[validate_fibration] Effective degree: {effective_degree_val}")

    stored_n = getattr(cd, 'tate_exponent', 'unknown')
    print(f"[validate_fibration] Stored Weierstrass exponent: {stored_n}")

    if effective_degree_val == 12:
        print("[validate_fibration] PASS: Discriminant degree is 12 (standard)")
    else:
        print(f"[validate_fibration] NOTICE: Discriminant degree is {effective_degree_val}, not 12")
        print("[validate_fibration]         (non-minimal or different parameterization)")

    print("="*70)
    sys.stdout.flush()

    return True

@PROFILE
def summarize_fibration_info(cd, data_pts, base_pts):
    """
    Print diagnostic summary of the fibration.

    Shows:
    - Data points used
    - Quartic and Weierstrass equations
    - Discriminants
    - Section information

    Args:
        cd: CurveDataExt object
        data_pts: Original input points
        base_pts: Transformed base points
    """
    print("\n" + "="*70)
    print("FIBRATION SUMMARY")
    print("="*70)
    sys.stdout.flush()

    print(f"Data points: {data_pts}")
    print(f"Base points: {base_pts}")
    sys.stdout.flush()

    # Quartic curve
    print(f"\nQuartic curve: {cd.E_curve}")
    sys.stdout.flush()

    # Get defining polynomial
    try:
        polys = cd.E_curve.defining_polynomials()
        assert polys, "No defining polynomial found"
        f = polys[0]
    except Exception as e:
        raise RuntimeError(f"summarize_fibration: failed to extract defining polynomial: {e}")

    print(f"Defining polynomial: {f}")
    sys.stdout.flush()

    # Compute quartic discriminant
    try:
        vars_list = f.parent().gens()
        disc_var = vars_list[-1] if len(vars_list) > 1 else vars_list[0]
        disc = f.discriminant(disc_var)
    except Exception as e:
        print(f"WARNING: Could not compute quartic discriminant: {e}")
        disc = "unknown"

    print(f"Quartic discriminant: {disc}")
    sys.stdout.flush()

    # Weierstrass discriminant
    try:
        disc2 = cd.E_weier.discriminant()
    except Exception as e:
        print(f"WARNING: Could not compute Weierstrass discriminant: {e}")
        disc2 = "unknown"

    print(f"\nWeierstrass discriminant: {disc2}")
    sys.stdout.flush()

    # Weierstrass coefficients
    print("\nWeierstrass model coefficients:")
    print(f"  a4(m): {cd.a4}")
    print(f"  a6(m): {cd.a6}")
    sys.stdout.flush()

    # Sections (if provided in base_pts)
    if hasattr(base_pts, '__iter__'):
        print(f"\nBase sections ({len(base_pts)} total):")
        for i, P in enumerate(base_pts, 1):
            print(f"  P{i}: {P}")
        sys.stdout.flush()

    print("="*70)
    sys.stdout.flush()

@PROFILE
def augment_known(known_pts, found, deg6=False):
    """
    Augment known points with newly found x-coordinates.

    Computes y-coordinates for new x values and adds (x,±y) to known set.

    Args:
        known_pts: Set of (x,y) tuples
        found: Set of x-coordinates
        deg6: If True, use genus 2 rationality test

    Returns:
        Updated set of (x,y) tuples
    """
    if DEBUG:
        print(f"\n[augment_known] Starting with {len(known_pts)} known points")
        print(f"[augment_known] Processing {len(found)} found x-coordinates")
        sys.stdout.flush()

    ret = set(known_pts)
    known_x = set([i for i, _ in known_pts])

    for x_val in found:
        if x_val in known_x:
            continue

        print(f"[augment_known] New x: {x_val}")
        sys.stdout.flush()

        # Compute y
        try:
            if deg6:
                rhsy = get_y_unshifted_genus2(x_val)
            else:
                rhsy = get_unshifted_y(x_val)
        except Exception as e:
            print(f"[augment_known] WARNING: Failed to compute y for x={x_val}: {e}")
            sys.stdout.flush()
            continue

        if rhsy is None:
            print(f"[augment_known] WARNING: x={x_val} does not give rational y")
            sys.stdout.flush()
            continue

        ret.add((x_val, rhsy))
        if rhsy != 0:
            ret.add((x_val, -rhsy))

    print(f"[augment_known] Result: {len(ret)} total points")
    sys.stdout.flush()

    return ret

if _IS_MAIN_PROCESS:
    print("DATA_PTS_GENUS2 =", DATA_PTS_GENUS2)

@PROFILE
def get_phi_x(one, two, three, x_coord_func, quartic_rhs):
    """
    Compute phi_x = X/Z on the Weierstrass model.

    Bimodal operation:
    - FINITE_FIELD mode: works in GF(p)(m), rationalizes via y^2 = quartic_rhs
    - QQ mode: symbolic computation with SR

    Returns:
        phi_x = X/Z as an element of the base fraction field when possible,
        or "INF" if the denominator vanishes.
    """
    ff_mode = (FINITE_FIELD is not None)

    def _reduce_to_linear_in_Y(expr, K, y2):
        """
        Given expr in a polynomial ring over K in variable Y,
        reduce modulo Y^2 = y2 and return A + B*Y.
        """
        PRY = expr.parent()
        Y = PRY.gen()
        poly = PRY(expr)

        A = K(0)
        B = K(0)

        deg = int(poly.degree())
        if deg < 0:
            return A, B

        for i in range(deg + 1):
            c = poly.coefficient(i)
            if c == 0:
                continue
            if i % 2 == 0:
                A += c * (y2 ** (i // 2))
            else:
                B += c * (y2 ** (i // 2))

        return A, B

    def _rationalize_linear_ratio(Ax, Bx, Az, Bz, K, y2):
        """
        Rationalize (Ax + Bx*Y)/(Az + Bz*Y) using Y^2 = y2.
        Returns an element of K when the Y-term cancels, otherwise None.
        """
        denom = Az * Az - (Bz * Bz) * y2
        if denom == 0:
            return "INF"

        num0 = Ax * Az - (Bx * Bz) * y2
        num1 = Bx * Az - Ax * Bz

        if num1 == 0:
            return K(num0 / denom)

        return None

    if ff_mode:
        F = GF(FINITE_FIELD)
        PR_m = PolynomialRing(F, 'm')
        K = PR_m.fraction_field()

        # Coerce x into K
        try:
            xK = K(x_coord_func)
        except Exception:
            try:
                xK = K(QQ(str(x_coord_func)))
            except Exception as e:
                raise RuntimeError(f"get_phi_x (FF): cannot coerce x to K: {e}")

        # Substitute x into quartic_rhs and coerce to K
        try:
            quartic_at_x = quartic_rhs.subs(x=xK)
        except Exception:
            try:
                quartic_at_x = quartic_rhs(x=xK)
            except Exception as e:
                raise RuntimeError(f"get_phi_x (FF): substitution failed: {e}")

        try:
            y2 = K(quartic_at_x)
        except Exception:
            try:
                y2 = K(PR_m(quartic_at_x))
            except Exception as e:
                raise RuntimeError(f"get_phi_x (FF): coercion to K failed: {e}")

        # Fast path: if sqrt exists in K, use it directly
        y_val_sqrt = None
        try:
            is_const = False
            if hasattr(y2, 'is_constant'):
                is_const = y2.is_constant()
            elif hasattr(y2, 'numerator') and hasattr(y2, 'denominator'):
                is_const = (y2.numerator().degree() <= 0 and y2.denominator().degree() <= 0)

            if is_const:
                const = F(y2)
                if const.is_square():
                    y_val_sqrt = const.sqrt()
            elif hasattr(y2, 'is_square') and y2.is_square():
                y_val_sqrt = y2.sqrt()
        except Exception:
            y_val_sqrt = None

        if y_val_sqrt is not None:
            try:
                Z_sub = three.subs(x=xK, y=y_val_sqrt)
                X_sub = one.subs(x=xK, y=y_val_sqrt)
            except Exception as e:
                raise RuntimeError(f"get_phi_x (FF): morphism evaluation failed: {e}")

            if Z_sub == 0:
                return "INF"
            return X_sub / Z_sub

        # Rationalize in the quadratic basis 1, Y
        try:
            PRY = PolynomialRing(K, 'Y')
            Y = PRY.gen()

            X_as = PRY(one.subs(x=xK, y=Y))
            Z_as = PRY(three.subs(x=xK, y=Y))

            Ax, Bx = _reduce_to_linear_in_Y(X_as, K, y2)
            Az, Bz = _reduce_to_linear_in_Y(Z_as, K, y2)

            rational = _rationalize_linear_ratio(Ax, Bx, Az, Bz, K, y2)
            if rational is not None:
                return rational

            # Last resort: keep the quadratic extension, but still use the same Y-ring
            T = PolynomialRing(K, 'T').gen()
            L = K.extension(T**2 - y2, 'Y')
            YL = L.gen()

            X_L = L(Ax) + L(Bx) * YL
            Z_L = L(Az) + L(Bz) * YL

            if Z_L == 0:
                return "INF"

            phiL = X_L / Z_L
            try:
                return K(phiL)
            except Exception:
                return phiL

        except Exception as e:
            raise RuntimeError(f"get_phi_x (FF): rationalization failed: {e}")

    # QQ / SR mode
    try:
        y_val_sqrt = sqrt(quartic_rhs)
    except Exception as e:
        raise RuntimeError(f"get_phi_x (QQ): sqrt(quartic_rhs) failed: {e}")

    try:
        Z_sub = three.subs(x=x_coord_func, y=y_val_sqrt)
        X_sub = one.subs(x=x_coord_func, y=y_val_sqrt)
    except Exception as e:
        raise RuntimeError(f"get_phi_x (QQ): symbolic computation failed: {e}")

    if Z_sub == 0:
        return "INF"

    return X_sub / Z_sub
