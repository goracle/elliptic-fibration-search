"""
search_config.py: Global constants, parameters, and all necessary imports.
"""
# Standard library imports (Keep a core set needed by all/most modules)
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
    crt, lcm, sqrt, polygen, Integer, ceil
)
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement

# Local project imports (Assuming these are outside the search_lll folder)
# These will now be imported here and re-exported via __init__.py or used directly by workers
from search_common import DEBUG, PROFILE
from stats import *
from brauer import *

# ==============================================================================
# Constants & Tuning Knobs
# ==============================================================================

# Core Limits and Defaults
DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_MAX_DENOMINATOR_BOUND = None
FALLBACK_MATRIX_WARNING = "WARNING: LLL reduction failed, falling back to identity matrix"
ROOTS_THRESHOLD = 12 
HENSEL_STRICT = True # Strict Hensel filtering by default
HENSEL_ALLOW_WEAK = False # Don't allow weak roots by default

# LLL/BKZ Tuning Parameters
LLL_DELTA = 0.99 
BKZ_BLOCK = 20 
MAX_COL_SCALE = 1e6 # Heuristic max scale factor for columns

# Extra Prime Filtering Parameters
EXTRA_PRIME_COUNT = 3
EXTRA_PRIME_MAX_SIZE = 100000
EXTRA_PRIME_EFFICIENCY_LIMIT = 0.001
EXTRA_PRIME_TARGET_DENSITY = 0.5

# Custom Exception Classes
class EllipticCurveSearchError(Exception):
    """Base exception for errors in the search process."""
    pass

class RationalReconstructionError(EllipticCurveSearchError):
    """Raised when rational reconstruction fails."""
    pass
