from .core import arakelov_build_basis_with_heights, dedupe_basis, select_independent_indices_from_gram, is_independent_by_projection_log
from .heights import arakelov_canonical_height, arakelov_quasi_height, naive_height_qq
from .pairings import get_pairing, neron_tate_height_pairing, precompute_pairings_parallel, gram_logdet_and_cond
from .periods import choose_numerical_base_point, abel_jacobi_mumford, normalize_periods_and_z
from .theta import compute_theta_high_prec, theta_direct
from .local import get_bad_primes, local_naive_height_p, local_height_correction_finite
from .archimedean import archimedean_height_correction, reduce_z_arakelov, print_archimedean_diagnostics
from .integration import *
from .utilities import sanity_check_pairings, robust_eig_clip, make_matrix_numerically_positive_definite
from .parallel import compute_height_worker, compute_pairing_worker

"""
Jacobian Basis Module
=====================

A comprehensive toolkit for computing bases of Jacobian groups over hyperelliptic curves
using Arakelov theory, canonical heights, and NĂ©ron-Tate pairings.
"""

# Core functionality

# Height computations

# Pairing computations

# Period matrix and Abel-Jacobi

# Theta functions

# Local (p-adic) functions

# Archimedean corrections

# Integration
# Utilities

# Parallel workers (typically not needed by end users)

__all__ = [
    # Core
    'arakelov_build_basis_with_heights',
    'dedupe_basis',
    'select_independent_indices_from_gram',
    'is_independent_by_projection_log',
    # Heights
    'arakelov_canonical_height',
    'arakelov_quasi_height',
    'naive_height_qq',
    # Pairings
    'get_pairing',
    'neron_tate_height_pairing',
    'precompute_pairings_parallel',
    'gram_logdet_and_cond',
    # Periods
    'choose_numerical_base_point',
    'abel_jacobi_mumford',
    'normalize_periods_and_z',
    # Theta
    'compute_theta_high_prec',
    'theta_direct',
    # Local
    'get_bad_primes',
    'local_naive_height_p',
    'local_height_correction_finite',
    # Archimedean
    'archimedean_height_correction',
    'reduce_z_arakelov',
    'print_archimedean_diagnostics',
    # Integration
    'integrate_differential_path_with_branch',
    # Utilities
    'sanity_check_pairings',
    'robust_eig_clip',
    'make_matrix_numerically_positive_definite',
]
