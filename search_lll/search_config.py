import sys, random, itertools, multiprocessing, math
from math import floor, sqrt, gcd, ceil, log
from fractions import Fraction
from functools import reduce, lru_cache, partial
from operator import mul
from collections import namedtuple, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from colorama import Fore, Style
from sage.all import QQ, ZZ, GF, PolynomialRing, EllipticCurve, matrix, vector, identity_matrix, zero_matrix, diagonal_matrix, crt, lcm, sqrt, polygen, Integer, ceil, SR, var
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement
from stats import *
from brauer import *

"""
search_config.py: Central config for the search_lll package.

Imports global run constants (DEBUG, PRIME_POOL, etc.) from search_common.py
and defines LLL-specific algorithmic constants (LLL_DELTA, TMAX, etc.).
"""

# === 1. Standard library imports ===

# === 2. Third-party imports ===

# === 3. SageMath imports ===

# === 4. Global Config Import ===
# Import global run constants from search_common.py in the parent directory
# This assumes the main script is run from the parent directory.
try:
    from search_common import (
        DEBUG, PROFILE, HENSEL_SLOPPY, TORSION_SLOPPY, TARGETED_X, PRIME_POOL,
        SEED_INT, MAX_TORSION_ORDER_TO_FILTER, MIN_PRIME_SUBSET_SIZE,
        MIN_MAX_PRIME_SUBSET_SIZE, MAX_MODULUS, MAX_COMBOS_PER_SUBSET,
        M_HEIGHT_BOUND, PER_VECTOR_M_BOUND_C
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
    MAX_COMBOS_PER_SUBSET = 5000
    M_HEIGHT_BOUND = 37000
    PER_VECTOR_M_BOUND_C = None
    raise

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

# Anomalous-residue sweep (run_standard_lattice_search inner loop)
MAX_ANOMALOUS_SWEEP_ROUNDS = 8       # hard cap on rounds re-running the full pool with explained residues pruned, before giving up

# Per-subset CRT combo cap (search_lll/modularthread.py: process_prime_subset_precomputed)
#
# NOTE: this is deliberately a constant separate from both MAX_MODULUS and
# the run_standard_lattice_search-local `combo_cap` (search_lll/search_main.py,
# computed as 50000**min(40, 7*subset_size//3)). Those two already exist to
# bound "the estimated number of CRT residue combinations for a subset" up
# front, at subset-generation time (generate_biased_prime_subsets_by_coverage_v2
# and the filtered_subsets loop in search_main.py) -- but at
# MIN_PRIME_SUBSET_SIZE=65+ (see search_common.py) the exponent clamps to 40,
# giving combo_cap = 50000**40 ~= 10**188. That's headroom sized to "never
# overflow / never spuriously reject", not "bound actual iteration cost" --
# real per-subset combo counts here are ~avg_roots**subset_size, i.e.
# ~10**3-10**5 for the observed avg_roots ~1.1-1.5, which sails under a
# 10**188 cap without being remotely cheap to actually iterate. So subsets
# pass the up-front estimate check (correctly, by that check's own math) and
# then process_prime_subset_precomputed iterates the FULL
# itertools.product(*lists) for every one of those tens of thousands of
# combos -- full-precision CRT, rational reconstruction, and a
# Kronecker/modular filter per combo -- with nothing capping how much of that
# work is actually worth doing. That per-subset iteration cost, not the
# subset-count or the up-front estimate, is the actual runtime/memory sink.
#
# MAX_COMBOS_PER_SUBSET bounds real iterated work directly, independent of
# whatever the caller's own combo_cap estimate was tuned for: subsets whose
# estimated combo count exceeds this are skipped up front (mirroring the
# existing checks, just against a threshold sized to iteration cost), AND the
# itertools.product loop itself now bails out once it has processed this many
# combos, so nothing that reaches the loop can run unbounded regardless of
# what any upstream estimate said.
#
# User-overridable in search_common.py (imported above, same pattern as
# MIN_PRIME_SUBSET_SIZE); this 5000 is only the fallback used if that import
# fails. Tune down if subsets are still slow; tune up only if you confirm
# (via the crt_lift_skipped_combo_cap / crt_lift_truncated_mid_loop stats
# counters) that truncation is discarding combos that would have mattered.
if 'MAX_COMBOS_PER_SUBSET' not in dir():
    MAX_COMBOS_PER_SUBSET = 5000

# Arc-consistency prefilter (search_lll/modularthread.py:
# process_prime_subset_precomputed, _kronecker_prefilter_domain,
# _pairwise_crt_survivors; search_lll/rational_arithmetic.py:
# lattice_rational_lift_exists, modulus_is_informative)
#
# Before the itertools.product(*lists) CRT sweep above ever runs, each
# prime's residue domain is pruned in two cheap passes, run per (vector,
# rhs) group:
#   1. Kronecker/QR prefilter -- kills a residue `a` mod p outright if the
#      induced y-coordinate condition (G(x) a quadratic non-residue mod p)
#      fails, using only p itself. No CRT involved.
#   2. CRT-then-lattice-reduction prefilter -- picks the surviving-domain's
#      smallest prime as an anchor, greedily folds in enough of the
#      remaining (smallest-domain-first) partner primes to push the
#      combined modulus (anchor * partner primes) past 2*HEIGHT_BOUND^2,
#      then drops anchor residues that have no partner-residue combination
#      whose CRT class admits a lattice point (r,s) with |r|,|s| <=
#      HEIGHT_BOUND (lattice_rational_lift_exists). If the subset's primes
#      can't reach that combined-modulus threshold at all, Stage 2 is
#      skipped for that anchor rather than trusting an uninformative
#      result.
#
#      *** This replaced an earlier version that called
#      rational_reconstruct(c, p*q, max_den=HEIGHT_BOUND) pairwise at a
#      fixed 2-prime modulus. That was a no-op in practice: bounding only
#      the denominator (not the numerator) against HEIGHT_BOUND, at a
#      modulus p*q of only a few thousand, let essentially every residue
#      "reconstruct" trivially (verified: 8633/8633 at p=97,q=89,
#      HEIGHT_BOUND=37000) -- arc_consistency_pairwise_pruned sat at 0 for
#      exactly this reason. lattice_rational_lift_exists fixes this by
#      bounding both numerator and denominator via the full continued-
#      fraction/Euclidean chain, and modulus_is_informative(M, H) = M >
#      2*H^2 gates when the modulus is actually large enough (measured:
#      survival rate is ~100% up to M=2*H^2, falling off roughly like
#      1/M beyond it) for a rejection to mean anything, rather than
#      pruning on pigeonhole noise. ***
#
# Only survivors feed the existing combo-count guard and itertools.product
# below -- this doesn't change what a "combo" is, it just shrinks the
# per-prime domains before they're multiplied together.
#
# Stats: arc_consistency_kronecker_pruned, arc_consistency_pairwise_pruned
# (residues dropped by each stage), arc_consistency_pairwise_informative_hits
# (anchors where the combined modulus cleared the threshold and Stage 2 ran
# for real), arc_consistency_pairwise_uninformative_skip (anchors where it
# couldn't, so Stage 2 was skipped), and arc_consistency_subset_emptied (a
# (vector, rhs) group whose domains hit empty after pruning -- would have
# produced zero combos anyway, just discovered earlier and cheaper). If
# arc_consistency_pairwise_uninformative_skip dominates
# arc_consistency_pairwise_informative_hits across a run, the subset's
# primes are too small/few to ever reach the 2*H^2 threshold and Stage 2
# is contributing little -- consider using larger primes in the pool or a
# smaller effective HEIGHT_BOUND for the sieve step.

# === 6. Custom Exception Classes ===
class EllipticCurveSearchError(Exception):
    """Base exception for errors in the search process."""
    pass

class RationalReconstructionError(EllipticCurveSearchError):
    """Raised when rational reconstruction fails."""
    pass

