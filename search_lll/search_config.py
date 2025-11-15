"""
search_config.py: Central config for the search_lll package.

Imports global run constants (DEBUG, PRIME_POOL, etc.) from search_common.py
and defines LLL-specific algorithmic constants (LLL_DELTA, TMAX, etc.).
"""

# === 1. Standard library imports ===
import sys
import random
import itertools
import multiprocessing
import math
from math import floor, sqrt, gcd, ceil, log
from fractions import Fraction
from functools import reduce, lru_cache, partial
from operator import mul
from collections import namedtuple, Counter 
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# === 2. Third-party imports ===
from tqdm import tqdm
from colorama import Fore, Style

# === 3. SageMath imports ===
from sage.all import (
    QQ, ZZ, GF, PolynomialRing, EllipticCurve,
    matrix, vector, identity_matrix, zero_matrix, diagonal_matrix,
    crt, lcm, sqrt, polygen, Integer, ceil, SR, var
)
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement

# === 4. Global Config Import ===
# Import global run constants from search_common.py in the parent directory
# This assumes the main script is run from the parent directory.
try:
    from search_common import (
        DEBUG, PROFILE, HENSEL_SLOPPY, TORSION_SLOPPY, TARGETED_X, PRIME_POOL,
        SEED_INT, MAX_TORSION_ORDER_TO_FILTER, MIN_PRIME_SUBSET_SIZE,
        MIN_MAX_PRIME_SUBSET_SIZE, MAX_MODULUS
    )
except ImportError:
    print("CRITICAL: search_lll/search_config.py could not import from search_common.")
    # Define fallbacks to prevent total crash, though this indicates a path issue
    DEBUG = False
    PROFILE = lambda f: f
    HENSEL_SLOPPY = True
    TORSION_SLOPPY = True
    TARGETED_X = None
    PRIME_POOL = [5, 7, 11, 13, 17, 19, 23]
    SEED_INT = 42
    MAX_TORSION_ORDER_TO_FILTER = 12
    MIN_PRIME_SUBSET_SIZE = 3
    MIN_MAX_PRIME_SUBSET_SIZE = 7
    MAX_MODULUS = 10**30
    raise

from stats import *
from brauer import *


# === 5. LLL-Package Specific Constants ===
# These are the algorithmic constants from search_lll.py.bak

# Core Limits and Defaults
DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_MAX_DENOMINATOR_BOUND = None
FALLBACK_MATRIX_WARNING = "WARNING: LLL reduction failed, falling back to identity matrix"
ROOTS_THRESHOLD = 12 # only multiply primes' root counts into the estimate when the total roots for that prime exceed this threshold
TMAX = 500

# LLL/BKZ Tuning Parameters
LLL_DELTA = 0.98           # strong LLL reduction; reduce if it slows too much (0.9--0.98 recommended)
BKZ_BLOCK = 12             # try BKZ with this block; lower for speed, larger for quality
MAX_COL_SCALE = 10**6      # don't scale any column by more than this (keeps integers reasonable)
TARGET_COLUMN_NORM = 1e6   # target column norm after scaling (heuristic)
MAX_K_ABS = 500            # ignore multiplier indices |k| > MAX_K_ABS when building mults
TRUNCATE_MAX_DEG = 30      # truncate polynomial coefficients at this degree to limit dimension
PARALLEL_PRIME_WORKERS = min(8, max(1, multiprocessing.cpu_count() // 2))

# Auto-Tune / Residue Filter Parameters
EXTRA_PRIME_TARGET_DENSITY = 1e-5   # desired survivor fraction after extras
EXTRA_PRIME_MAX = 6                 # cap on number of extra primes
EXTRA_PRIME_SKIP = {2, 3}        # avoid small degenerates
EXTRA_PRIME_SAMPLE_SIZE = 300       # sample vectors for stats
EXTRA_PRIME_MIN_R = 1e-4            # ignore primes with r_p < this
EXTRA_PRIME_MAX_R = 0.9             # ignore primes with r_p > this

# === 6. Custom Exception Classes ===
class EllipticCurveSearchError(Exception):
    """Base exception for errors in the search process."""
    pass

class RationalReconstructionError(EllipticCurveSearchError):
    """Raised when rational reconstruction fails."""
    pass






