from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
from .mumford_core import _poly_reduce_mod_u, poly_reduce_mod_u, _poly_mod_quad_fast, _normalize_sign, _poly_from_coeffs_qq, _get_divisor_coeffs_qq, make_monic, reduce_v_mod_u, is_divisor_on_curve
from .mumford_solver import solve_mumford_mod_p, solve_mumford_mod_p_optimized, filter_primes_avoiding_denoms
from .mumford_verification import verify_mumford_pair, validate_mumford_solver, canonicalize_and_dedup, quick_dependence_check, discriminant_has_nonqr_s_p
from .mumford_height import naive_height_safe, naive_height_exact, manual_naive_height, manual_canonical_height, compute_manual_height_pairing, compute_height_pairing_simple, compute_height_pairing_exact
from .mumford_doubling import compute_doubled_point_modular
from .mumford_basis import build_mumford_basis_incremental, build_mumford_basis_incremental_exact, check_mumford_independence, mumford_to_jacobian_element
from .mumford_reconstruction import reconstruct_and_verify_mumford, reconstruct_mumford_combo_fast, rational_reconstruct_fast, setup_crt_constants
from .mumford_parallel import *
from .mumford_timing import mumford_timer_add, mumford_timer_get, mumford_timers_reset, mumford_timers_print
from .mumford_equations import build_mumford_equations_from_fibration

"""
Mumford divisor search module.
Provides functionality for genus-2 Jacobian point search via Mumford coordinates.
"""

# Core functionality

# Solver

# Verification

# Height computations

# Doubling

# Basis construction

# Reconstruction

# Parallel processing
# Timing

# Equations

# Try to import Arakelov
try:
    from ..arakelov import *
    ARAKELOV_AVAILABLE = True
except ImportError:
    ARAKELOV_AVAILABLE = False
    print("[mumford] Warning: arakelov.py not available, using fallback methods")

# Import smoothness diagnostics from parent
try:
    from ..smoothness import *
except ImportError:
    print("[mumford] Warning: smoothness.py not available")

# Module constants (keep these here for now, or move to mumford_config.py later)
RECON_EXPONENT = 0.55
MIN_SUCCESS_PRIMES = 3
PRIMES_NR = (3, 5, 7, 11, 13, 17, 19, 23)

# Public API
__all__ = [
    # Core
    'poly_reduce_mod_u',
    'verify_mumford_pair',
    'make_monic',
    'reduce_v_mod_u',
    'is_divisor_on_curve',

    # Solver
    'solve_mumford_mod_p',
    'solve_mumford_mod_p_optimized',

    # Main operations
    'build_mumford_basis_incremental',
    'check_mumford_independence',
    'reconstruct_and_verify_mumford',
    'mumford_precompute_residues_parallel',
    'build_mumford_equations_from_fibration',

    # Height
    'compute_height_pairing_exact',
    'naive_height_exact',

    # Timing
    'mumford_timers_print',
    'mumford_timers_reset',

    # Constants
    'ARAKELOV_AVAILABLE',
    'RECON_EXPONENT',
    'MIN_SUCCESS_PRIMES',
    'PRIMES_NR',
]
