"""
search_config.py: Global constants, parameters, and all necessary imports.
"""
# Standard library imports
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

# Third-party imports
from tqdm import tqdm
from colorama import Fore, Style

# SageMath imports (Centralize these, as they are used across the project)
from sage.all import (
    QQ, ZZ, GF, PolynomialRing, EllipticCurve,
    matrix, vector, identity_matrix, zero_matrix, diagonal_matrix,
    crt, lcm, sqrt, polygen, Integer, ceil, SR, var
)
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement

# Local project imports
# Assuming these are in the parent directory or installed
# from search_common import * # from stats import *
# from brauer import *

# --- Debug Flags ---
# (Import these from search_common or define them here)
DEBUG = globals().get('DEBUG', False)
PROFILE = globals().get('PROFILE', False)
HENSEL_SLOPPY = globals().get('HENSEL_SLOPPY', True)
TORSION_SLOPPY = globals().get('TORSION_SLOPPY', True)
TARGETED_X = globals().get('TARGETED_X', None)
PRIME_POOL = globals().get('PRIME_POOL', [])
SEED_INT = globals().get('SEED_INT', 42)
MAX_TORSION_ORDER_TO_FILTER = globals().get('MAX_TORSION_ORDER_TO_FILTER', 12)
MIN_PRIME_SUBSET_SIZE = globals().get('MIN_PRIME_SUBSET_SIZE', 3)
MIN_MAX_PRIME_SUBSET_SIZE = globals().get('MIN_MAX_PRIME_SUBSET_SIZE', 7)
MAX_MODULUS = globals().get('MAX_MODULUS', 10**30)


# ==============================================================================
# Constants & Tuning Knobs
# ==============================================================================

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

# ==============================================================
# === Auto-Tune / Residue Filter Parameters ====================
# ==============================================================

EXTRA_PRIME_TARGET_DENSITY = 1e-5   # desired survivor fraction after extras
EXTRA_PRIME_MAX = 6                 # cap on number of extra primes
EXTRA_PRIME_SKIP = {2, 3}        # avoid small degenerates
EXTRA_PRIME_SAMPLE_SIZE = 300       # sample vectors for stats
EXTRA_PRIME_MIN_R = 1e-4            # ignore primes with r_p < this
EXTRA_PRIME_MAX_R = 0.9             # ignore primes with r_p > this


# Custom Exception Classes
class EllipticCurveSearchError(Exception):
    """Base exception for errors in the search process."""
    pass

class RationalReconstructionError(EllipticCurveSearchError):
    """Raised when rational reconstruction fails."""
    pass
