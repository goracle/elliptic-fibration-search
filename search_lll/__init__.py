"""
__init__.py: Exposes key functions from the submodules.
"""
# Expose core configuration and exceptions
from .search_config import (
    DEFAULT_MAX_CACHE_SIZE, ROOTS_THRESHOLD, LLL_DELTA, 
    EllipticCurveSearchError, RationalReconstructionError, DEBUG, PROFILE
)

# Expose main execution functions
from .modularthread import (
    _process_prime_subset, _process_prime_subset_precomputed, 
    prepare_modular_data_lll, check_specific_t_value, process_candidate_numeric
)

# Expose main utilities (as needed by the parent scripts)
from .rational_arithmetic import crt_cached, rational_reconstruct
from .ll_utilities import _scale_matrix_columns_int
from .search_analysis import estimate_prime_stats, choose_extra_primes
