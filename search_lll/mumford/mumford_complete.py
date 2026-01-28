from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
from .mumford_core import *
from .mumford_solver import solve_mumford_mod_p, solve_mumford_mod_p_optimized
from .mumford_verification import verify_mumford_pair, canonicalize_and_dedup
from .mumford_height import compute_height_pairing_exact
from .mumford_basis import build_mumford_basis_incremental, check_mumford_independence
from .mumford_reconstruction import reconstruct_and_verify_mumford
from .mumford_parallel import mumford_precompute_residues_parallel
from .mumford_timing import mumford_timers_print, mumford_timers_reset
from .mumford_equations import build_mumford_equations_from_fibration
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL
from .smoothness import *

"""
Main entry point for Mumford divisor search.
Re-exports key functions for backward compatibility.
"""

# Core imports

# Constants

# Try to import Arakelov
try:
    from .arakelov import *
    ARAKELOV_AVAILABLE = True
except ImportError:
    ARAKELOV_AVAILABLE = False
    print("[mumford] Warning: arakelov.py not available, using fallback methods")

# Module constants
RECON_EXPONENT = 0.55
MIN_SUCCESS_PRIMES = 3
PRIMES_NR = (3, 5, 7, 11, 13, 17, 19, 23)
MAX_BASIS_CANDIDATES = 6

__all__ = [
    'solve_mumford_mod_p',
    'solve_mumford_mod_p_optimized',
    'verify_mumford_pair',
    'build_mumford_basis_incremental',
    'check_mumford_independence',
    'reconstruct_and_verify_mumford',
    'mumford_precompute_residues_parallel',
    'build_mumford_equations_from_fibration',
    'ARAKELOV_AVAILABLE',
]
