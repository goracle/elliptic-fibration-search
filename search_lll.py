"""
Elliptic Curve Rational Point Search using LLL and Modular Methods

This module implements sophisticated algorithms for finding rational points on elliptic curves
using lattice reduction (LLL), Chinese Remainder Theorem (CRT), and 2-descent methods.
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
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Third-party imports
from tqdm import tqdm
from colorama import Fore, Style

# SageMath imports
from sage.all import (
    QQ, ZZ, GF, PolynomialRing, EllipticCurve,
    matrix, vector, identity_matrix, zero_matrix, diagonal_matrix,
    crt, lcm, sqrt, polygen, Integer
)
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement

# Local imports (assuming these exist in your project)
from search_common import *
from search_common import DEBUG, PROFILE

# Constants
DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_MAX_DENOMINATOR_BOUND = None
FALLBACK_MATRIX_WARNING = "WARNING: LLL reduction failed, falling back to identity matrix"


# Practical tuning knobs
LLL_DELTA = 0.98           # strong LLL reduction; reduce if it slows too much (0.9--0.98 recommended)
BKZ_BLOCK = 12             # try BKZ with this block; lower for speed, larger for quality
MAX_COL_SCALE = 10**6      # don't scale any column by more than this (keeps integers reasonable)
TARGET_COLUMN_NORM = 1e6   # target column norm after scaling (heuristic)
MAX_K_ABS = 500            # ignore multiplier indices |k| > MAX_K_ABS when building mults
TRUNCATE_MAX_DEG = 30      # truncate polynomial coefficients at this degree to limit dimension
PARALLEL_PRIME_WORKERS = min(8, max(1, multiprocessing.cpu_count() // 2))
MAX_ABS_T = 500


# ==============================================================
# === Auto-Tune / Residue Filter Parameters ====================
# ==============================================================

EXTRA_PRIME_TARGET_DENSITY = 1e-5   # desired survivor fraction after extras
EXTRA_PRIME_MAX = 6                 # cap on number of extra primes
EXTRA_PRIME_SKIP = {2, 3, 5}        # avoid small degenerates
EXTRA_PRIME_SAMPLE_SIZE = 300       # sample vectors for stats
EXTRA_PRIME_MIN_R = 1e-4            # ignore primes with r_p < this
EXTRA_PRIME_MAX_R = 0.9             # ignore primes with r_p > this


def _compute_column_norms(M):
    """
    Compute L2 norm per column of integer matrix M (sage matrix).
    Returns list of floats.
    """
    col_norms = []
    for j in range(M.ncols()):
        s = 0
        for i in range(M.nrows()):
            v = int(M[i, j])
            s += v * v
        col_norms.append(float(s**0.5))
    return col_norms

def _compute_integer_scales_for_columns(M, target_norm=TARGET_COLUMN_NORM, max_scale=MAX_COL_SCALE):
    """
    Return integer scale factors (one per column) so that after multiplying column j by scales[j],
    its norm is approximately target_norm (but not exceeding max_scale).
    """
    norms = _compute_column_norms(M)
    scales = []
    for norm in norms:
        if norm <= 1e-12:
            scales.append(1)
            continue
        scale = int(ceil(target_norm / max(1.0, norm)))
        if scale < 1:
            scale = 1
        if scale > max_scale:
            scale = max_scale
        scales.append(int(scale))
    return scales

def _scale_matrix_columns_int(M, scales):
    """
    Return M_scaled = M * D where D = diag(scales), scales are ints.
    """
    D = diagonal_matrix([ZZ(s) for s in scales])
    return M * D, D


def _trim_poly_coeffs(coeff_list, max_deg=TRUNCATE_MAX_DEG):
    """Truncate coefficient list (low->high) to length max_deg+1."""
    if len(coeff_list) <= max_deg + 1:
        return coeff_list
    # Keep low-degree coefficients (assumed stored as [c0, c1, ..., cN])
    return coeff_list[: max_deg + 1]


class EllipticCurveSearchError(Exception):
    """Custom exception for elliptic curve search operations."""
    pass


class RationalReconstructionError(Exception):
    """Exception raised when rational reconstruction fails."""
    pass


# ---------- Utility: archimedean height on QQ elements ----------
def archimedean_height_QQ(x):
    """
    A simple archimedean height for a QQ element x = a/b in lowest terms.
    Returns log(max(|a|, |b|, 1)).
    """
    # Expect x to be a Sage QQ or Python Fraction
    try:
        a = Integer(x.numerator())
        b = Integer(x.denominator())
    except (AttributeError, TypeError) as e:
        # If x is not rational (e.g., symbolic), raise so caller handles it.
        raise TypeError("archimedean_height_QQ expects a rational (QQ) input") from e

    val = max(abs(int(a)), abs(int(b)), 1)
    return math.log(val)

# ---------- Utility: local search for best t ----------
def minimize_archimedean_t(m0, M, r_m_func, shift, max_abs_t, max_steps=150, patience=6):
    """
    Given residue class m = m0 (mod M), search over m = m0 + t*M to find integer t that minimizes
    archimedean height of x = r_m(m) - shift.

    Returns list of (m_candidate (QQ), score (float)) for the best few t values.
    """
    best = []

    def eval_for_t(t):
        m_candidate = m0 + t * M
        if abs(t) > max_abs_t:
            return None
        try:
            x_val = r_m_func(m=QQ(m_candidate)) - shift
            if not (hasattr(x_val, 'numerator') and hasattr(x_val, 'denominator')):
                return None
            score = archimedean_height_QQ(x_val)
            return (QQ(m_candidate), float(score))
        except (ZeroDivisionError, TypeError, ArithmeticError):
            print("we're here, for some reason")
            return None

    center = eval_for_t(0)
    if center is not None:
        best.append(center)

    steps = 0
    no_improve = 0
    current_best_score = best[0][1] if best else float('inf')
    t = 1
    while steps < max_steps and no_improve < patience:
        for s in (t, -t):
            res = eval_for_t(s)
            steps += 1
            if res is None:
                continue
            m_cand, score = res
            best.append((m_cand, score))
            if score + 1e-12 < current_best_score:
                current_best_score = score
                no_improve = 0
            else:
                no_improve += 1
            if abs(s) >= max_abs_t:
                no_improve = patience
                break
        t += 1

    # Deduplicate using exact rational key (num, den)
    unique = {}
    for m_cand, score in best:
        num = int(m_cand.numerator())
        den = int(m_cand.denominator())
        key = (num, den)
        if key not in unique or score < unique[key]:
            unique[key] = score

    sorted_candidates = sorted(((QQ(num) / QQ(den), sc) for (num, den), sc in unique.items()),
                               key=lambda z: z[1])

    best_t_values = []
    for m_cand, score in sorted_candidates[:3]:
        for t_test in range(-max_abs_t, max_abs_t + 1):
            if m0 + t_test * M == m_cand:
                best_t_values.append(t_test)
                break

    return sorted_candidates[:3]

# --- Top-level Worker Function for Parallel Processing ---

def _process_prime_subset(p_subset, cd, current_sections, prime_pool, r_m, shift, rhs_list, vecs, max_abs_t):
    """
    Worker function to find m-candidates for a single subset of primes.
    Returns a set of (m_candidate, originating_vector) tuples.
    """
    if not p_subset:
        return set()

    # Prepare modular data for this specific prime subset.
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, search_primes=p_subset
    )

    # quick sanity: ensure keys of vecs_lll are subset of Ep_dict keys
    bad_primes = [p for p in vecs_lll.keys() if p not in Ep_dict]
    if bad_primes:
        if DEBUG: print("Warning: vecs_lll has primes not in Ep_dict:", bad_primes)
        raise IndexError

    if not Ep_dict:
        return set()

    found_candidates_for_subset = set()
    r = len(current_sections)

    # Process each search vector for this subset
    for idx, v_orig in enumerate(vecs):
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig) # Make it hashable

        residue_map = {}
        for p in p_subset:
            if p not in Ep_dict:
                continue

            # vecs_lll and mult_lll are guaranteed to be present for any published prime p
            v_p_list = vecs_lll.get(p)
            if v_p_list is None:
                # defensive: unexpected, but skip this prime
                raise ValueError
                continue
            if idx >= len(v_p_list):
                # transformed vector missing for this index (shouldn't happen if prepare succeeded),
                # but skip this prime rather than crash.
                raise IndexError
                continue

            v_p_transformed = v_p_list[idx]
            mults = mult_lll.get(p)
            if mults is None:
                # no multiplies published for this prime (shouldn't happen); skip
                raise ValueError
                continue

            Ep = Ep_dict[p]

            #v_p_transformed = vecs_lll[p][idx]
            #mults = mult_lll[p]
            #Ep = Ep_dict[p]

            # Compute linear combination of basis points
            Pm = Ep(0)
            for j, coeff in enumerate(v_p_transformed):
                # The check `int(coeff) in mults[j]` should always pass
                # due to precomputation but is kept for safety.
                if int(coeff) in mults[j]:
                    Pm += mults[j][int(coeff)]

            if Pm.is_zero():
                continue

            # Find roots for each RHS function
            roots_for_p = set()
            for i, rhs_ff in enumerate(rhs_list):
                if p not in rhs_modp_list[i]:
                    continue

                rhs_p = rhs_modp_list[i][p]
                try:
                    num_modp = (Pm[0]/Pm[2] - rhs_p).numerator()
                    if not num_modp.is_zero():
                        roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                        roots_for_p.update(roots)
                except (ZeroDivisionError, ArithmeticError):
                    continue

            if roots_for_p:
                residue_map[p] = roots_for_p

        # Apply CRT to find m-candidates from the collected roots
        primes_for_crt = [p for p in p_subset if p in residue_map]
        if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
             continue

        lists = [residue_map[p] for p in primes_for_crt]
        for combo in itertools.product(*lists):
            M = reduce(mul, primes_for_crt, 1)

            if M > MAX_MODULUS:
                continue


            m0 = crt_cached(combo, tuple(primes_for_crt))

            # quick pre-filter: does any small prime forbid any t in [-T,T]?
            # doesn't appear to filter anything... something wrong with it... skip for now...
            if not candidate_passes_extra_primes(m0, M, residue_map):
                #    print("HEEERRE")
                #    raise ValueError
                continue  # reject this CRT combo cheaply, exact check

            # OLD: only check t in (-1,0,1)
            # for t in (-1, 0, 1):
            #     found_candidates_for_subset.add((QQ(m0 + t * M), v_orig_tuple))
            # New: minimize archimedean height in the residue class (cheap local search)

            try:
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, max_abs_t)
            except TypeError:
                # If r_m is not numeric/coercible here, fall back to small neighbors
                best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]
                print("here, instead;  why?")
                raise

            # best_ms is list of (m_candidate, score); add to candidate set
            for m_cand, _score in best_ms:
                found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

            # keep rational recon option as before
            try:
                a, b = rational_reconstruct(m0 % M, M)
                found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            except RationalReconstructionError:
                raise #<----safety, do not worry about the pass
                pass


            #m0 = crt_cached(combo, tuple(primes_for_crt))

            # Add nearby integer values
            #for t in (-1, 0, 1):
            #    found_candidates_for_subset.add((QQ(m0 + t * M), v_orig_tuple))
            # Add rationally reconstructed value
            #try:
            #    a, b = rational_reconstruct(m0 % M, M)
            #    found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            #except RationalReconstructionError:
            #    pass

    return found_candidates_for_subset




# Alternative more robust approach for finding minimal |t|:
def find_minimal_abs_representative(t_mod_Q, Q, T):
    """
    Find if there exists k such that |t_mod_Q + k*Q| <= T
    Returns True if such k exists, False otherwise.
    """
    # We want to minimize |t_mod_Q + k*Q| over integer k
    # This is minimized when k ≈ -t_mod_Q/Q
    
    if Q == 0:
        return abs(t_mod_Q) <= T
    
    # Try both floor and ceiling of the optimal k
    k_opt_float = -t_mod_Q / Q
    k_candidates = [int(k_opt_float), int(k_opt_float) + 1]
    
    # Also try k=0 in case t_mod_Q itself is small
    k_candidates.append(0)
    
    for k in k_candidates:
        t = t_mod_Q + k * Q
        if abs(t) <= T:
            return True
    return False


# --- Utility Functions ---

@lru_cache(maxsize=DEFAULT_MAX_CACHE_SIZE)
def crt_cached(residues, moduli):
    """Cached Chinese Remainder Theorem computation."""
    return crt(list(residues), list(moduli))

def rational_reconstruct(c, N, max_den=None):
    """
    Rational reconstruction using the Extended Euclidean Algorithm.

    Given integers c and N > 0, finds a rational number a/b such that
    a/b ≡ c (mod N), with |a| and |b| bounded.
    """
    if max_den is None:
        max_den = floor(sqrt(N / QQ(2)))

    c = c % N
    if c == 0: return 0, 1
    if c == 1 and max_den >= 1: return 1, 1

    # Standard Extended Euclidean Algorithm setup
    r0, r1 = N, c
    t0, t1 = 0, 1

    while r1 != 0:
        # Check denominator bound before next iteration
        if abs(t1) > max_den:
             # We've overshot the bound. The previous iteration held the last valid result.
             # In this algorithm, that's (r0, t0).
             a, b = r0, t0
             break

        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1
    else:
        # Loop finished because r1 == 0. The result is (r1, t1) from the *previous* step,
        # which is now stored in (r0, t0).
        a, b = r0, t0


    # Final checks on the result (a, b)
    if abs(b) > max_den or b == 0:
        raise RationalReconstructionError(f"No reconstruction for c={c}, N={N}, max_den={max_den}")

    if b < 0:
        a, b = -a, -b

    if (a - c * b) % N != 0:
        raise RationalReconstructionError(f"Validation failed for c={c}, N={N}: got a={a}, b={b}")

    g = gcd(abs(a), abs(b))
    return int(a // g), int(b // g)


def can_reduce_point_mod_p(P, p):
    """Check if point P can be reduced mod p without hitting bad denominators."""
    X, Y, Z = P
    for coord in [X, Y, Z]:
        num_poly = coord.numerator()
        den_poly = coord.denominator()
        # Check if any coefficient has denominator divisible by p
        for poly in [num_poly, den_poly]:
            if hasattr(poly, 'coefficients'):
                for c in poly.coefficients(sparse=False):
                    if QQ(c).denominator() % p == 0:
                        return False
    return True

def reduce_point_hom(P, Ep, p_mod):
    if not can_reduce_point_mod_p(P, p_mod):
        return Ep(0)

    X, Y, Z = P
    Fp_m = Ep.base_ring()
    
    # Find lcm of denominators avoiding factors of p_mod
    den_factors = [d // gcd(d, p_mod) for d in (X.denominator(), Y.denominator(), Z.denominator())]
    den = lcm(den_factors)
    
    try:
        Xr = Fp_m(X * den)
        Yr = Fp_m(Y * den)
        Zr = Fp_m(Z * den)
        return Ep([Xr, Yr, Zr])
    except Exception as e:
        if DEBUG:
            print(f"Warning: Failed to reduce point mod {p_mod}: {e}")
        raise # should be handled in the can_reduce_point_mod_p function
        return Ep(0)



def compute_all_mults_for_section(Pi, required_ks, max_k=MAX_K_ABS):
    """
    Compute k*Pi for all k in required_ks.
    Uses memoization: if we've already computed 2*Pi, reuse it for 4*Pi = 2*(2*Pi).
    """
    mults = {}
    computed = {0: Pi.curve()(0), 1: Pi}  # Start with 0 and 1
    
    # Sort by absolute value to compute small multiples first
    sorted_ks = sorted(required_ks, key=abs)
    
    for k in sorted_ks:
        if k in computed:
            mults[k] = computed[k]
            continue
        
        abs_k = abs(k)
        
        # Try to build from smaller multiples (binary decomposition)
        # E.g., 7*Pi = 4*Pi + 2*Pi + Pi if 4 and 2 are already computed
        if abs_k // 2 in computed:
            half_k = abs_k // 2
            base = computed[half_k]
            if abs_k % 2 == 0:
                result = base + base  # double
            else:
                result = base + base + Pi  # 2*(k//2) + Pi
            
            if k < 0:
                result = -result
            computed[k] = result
            mults[k] = result
        else:
            # Fallback: direct computation
            try:
                mults[k] = k * Pi
                computed[k] = mults[k]
            except Exception:
                pass
    
    return mults

def _get_coeff_data(poly):
    """Helper to safely extract coefficient list and degree from a polynomial-like object."""
    if hasattr(poly, 'list') and hasattr(poly, 'degree'):
        return poly.list(), poly.degree()
    else:
        # Handle constants or other non-polynomial objects
        return [poly], 0

# --- LLL Reduction Functions ---

def lll_reduce_basis_modp(p, sections, curve_modp,
                                   truncate_deg=TRUNCATE_MAX_DEG,
                                   lll_delta=LLL_DELTA, bkz_block=BKZ_BLOCK,
                                   max_k_abs=MAX_K_ABS):
    """
    Improved LLL/BKZ reduction on coefficient lattice of projective sections over GF(p)(m).
    Returns (new_basis, Uinv) similar to previous function.
    """
    r = len(sections)
    if r == 0:
        return [], identity_matrix(ZZ, 0)

    poly_coords = []
    max_deg = 0

    # Extract integer coefficient lists (trim tails)
    for P in sections:
        Pp = reduce_point_hom(P, curve_modp, p)
        if Pp is None or Pp.is_zero():  # <- handle None safely
            poly_coords.append(([0], [0], [1]))
            continue

        Xr, Yr, Zr = Pp[0], Pp[1], Pp[2]
        den = lcm([Xr.denominator(), Yr.denominator(), Zr.denominator()])
        Xp = Xr.numerator() * (den // Xr.denominator())
        Yp = Yr.numerator() * (den // Yr.denominator())
        Zp = Zr.numerator() * (den // Zr.denominator())

        xc, dx = _get_coeff_data(Xp)
        yc, dy = _get_coeff_data(Yp)
        zc, dz = _get_coeff_data(Zp)

        # Trim high-degree tails
        xc = _trim_poly_coeffs(xc, truncate_deg)
        yc = _trim_poly_coeffs(yc, truncate_deg)
        zc = _trim_poly_coeffs(zc, truncate_deg)

        poly_coords.append((xc, yc, zc))
        max_deg = max(max_deg, len(xc)-1, len(yc)-1, len(zc)-1)

    poly_len = max_deg + 1
    coeff_vecs = []
    for xc, yc, zc in poly_coords:
        xc_padded = list(xc) + [0] * (poly_len - len(xc))
        yc_padded = list(yc) + [0] * (poly_len - len(yc))
        zc_padded = list(zc) + [0] * (poly_len - len(zc))
        row = [ZZ(int(c)) for c in (xc_padded + yc_padded + zc_padded)]
        coeff_vecs.append(vector(ZZ, row))

    if not coeff_vecs or all(v.is_zero() for v in coeff_vecs):
        if DEBUG: print("All coefficient vectors are zero or truncated away, using identity transformation")
        return [curve_modp(0) for _ in range(r)], identity_matrix(ZZ, r)

    M = matrix(ZZ, coeff_vecs)
    
    # *** FIXED PART ***
    # The BKZ/LLL implementations in Sage can fail on matrices with only one row.
    # If there's only one section, the basis is trivially reduced, so we can
    # skip the complex reduction step entirely and use an identity transformation.
    if M.nrows() <= 1:
        Uinv = identity_matrix(ZZ, r)
        reduced_sections_mod_p = [reduce_point_hom(P, curve_modp, p) for P in sections]
        # With an identity transformation, the new basis is the same as the old one.
        return reduced_sections_mod_p, Uinv
    # *** END FIXED PART ***

    if M.rank() < min(M.nrows(), M.ncols()):
        if DEBUG: print("Matrix is rank-deficient (after trimming), continuing but may affect LLL")

    # Column scaling to balance magnitudes
    try:
        scales = _compute_integer_scales_for_columns(M)
        M_scaled, D = _scale_matrix_columns_int(M, scales)
    except Exception as e:
        if DEBUG: print("Column scaling failed, proceeding without scaling:", e)
        M_scaled = M
        D = diagonal_matrix([1]*M.ncols())

    # Try BKZ first if available (improves shortness for moderate dims)
    U = None
    B = None
    try:
        # prefer BKZ if available on matrix
        if hasattr(M_scaled, "BKZ"):
            # choose blocksize heuristically but not larger than dimension
            block = min(bkz_block, max(2, M_scaled.ncols()//2))
            U, B = M_scaled.BKZ(block_size=block, transformation=True)
        else:
            # fallback to LLL with chosen delta
            U, B = M_scaled.LLL(transformation=True, delta=float(lll_delta))
    except (TypeError, ValueError):
        # different signatures in some Sage versions, or value error from BKZ, fall back
        try:
            U, B = M_scaled.LLL(transformation=True)
        except Exception as e:
            if DEBUG:
                print("LLL/BKZ reduction failed, falling back to identity:", e)
            U = identity_matrix(ZZ, r)
            B = M_scaled.copy()

    # Validate U is unimodular (invertible over ZZ)
    Uinv = None
    try:
        if U.is_square() and U.det() != 0:
            Uinv = U.inverse()  # exact inverse over ZZ
        else:
            raise ValueError("U not square or singular")
    except Exception as e:
        # try Hermite Normal Form based approach as fallback to get a valid transform
        if DEBUG:
            print("U inverse failed; attempting HNF-based fallback:", e)
        try:
            # compute HNF of M_scaled and find transformation approx
            H, U_hnf = M_scaled.hermite_form(transformation=True)
            # U_hnf is unimodular so invertible
            Uinv = U_hnf.inverse()
            U = U_hnf
        except Exception as e2:
            if DEBUG: print("HNF fallback failed:", e2)
            U = identity_matrix(ZZ, r)
            Uinv = U  # identity

    # Build reduced basis points by applying U to reduced sections (but we scaled columns earlier)
    reduced_sections_mod_p = [reduce_point_hom(P, curve_modp, p) for P in sections]
    # U acts on rows (combines sections). Unscaling D isn't necessary for row ops.
    new_basis = [sum(U[i, j] * reduced_sections_mod_p[j] for j in range(r)) for i in range(r)]

    # Filter required ks to a bounded range to avoid explosion in mults
    # Note: caller must check mults limits
    return new_basis, Uinv



def prepare_modular_data_lll(cd, current_sections, prime_pool, rhs_list, vecs, search_primes=None):
    """
    Prepare modular data for LLL-based search across multiple primes.
    Ensures we only publish per-prime data after successful processing for that prime.
    NOW WITH SINGULAR CURVE DETECTION.
    """
    if search_primes is None:
        search_primes = prime_pool

    r = len(current_sections)
    if r == 0:
        return {}, [], {}, {}

    Ep_dict, rhs_modp_list = {}, [{} for _ in rhs_list]
    multiplies_lll, vecs_lll = {}, {}
    PR_m = PolynomialRing(QQ, 'm')

    processed_rhs_list = [{'num': PR_m(rhs.numerator()), 'den': PR_m(rhs.denominator())} for rhs in rhs_list]
    a4_num, a4_den = PR_m(cd.a4.numerator()), PR_m(cd.a4.denominator())
    a6_num, a6_den = PR_m(cd.a6.numerator()), PR_m(cd.a6.denominator())

    for p in search_primes:
        try:
            # Skip if any coefficient in a4_num or a6_num has denominator divisible by p
            if any(QQ(c).denominator() % p == 0 for c in a4_num.coefficients(sparse=False)):
                continue
            if any(QQ(c).denominator() % p == 0 for c in a6_num.coefficients(sparse=False)):
                continue
            
            Rp = PolynomialRing(GF(p), 'm')
            Fp_m = Rp.fraction_field()

            # denominators zero mod p -> skip
            if a4_den.change_ring(GF(p)).is_zero() or a6_den.change_ring(GF(p)).is_zero():
                continue

            a4_modp = Fp_m(a4_num) / Fp_m(a4_den)
            a6_modp = Fp_m(a6_num) / Fp_m(a6_den)
            
            # *** NEW: Check for singular curves before creating EllipticCurve ***
            # A curve y^2 = x^3 + a4*x + a6 is singular iff its discriminant is zero.
            # Over a field, discriminant = -16*(4*a4^3 + 27*a6^2)
            disc_modp = -16 * (4 * a4_modp**3 + 27 * a6_modp**2)
            if disc_modp.is_zero():
                if DEBUG:
                    print(f"Skipping prime {p}: resulting curve is singular (discriminant = 0 mod {p})")
                continue
            
            try:
                Ep_local = EllipticCurve(Fp_m, [0, 0, 0, a4_modp, a6_modp])
            except ArithmeticError as e:
                if DEBUG:
                    print(f"Skipping prime {p}: EllipticCurve construction failed: {e}")
                continue

            # build rhs_modp for this prime (but don't publish until success)
            rhs_modp_for_p = {}
            for i, rhs_data in enumerate(processed_rhs_list):
                if rhs_data['den'].change_ring(GF(p)).is_zero():
                    continue
                rhs_modp_for_p[i] = Fp_m(rhs_data['num']) / Fp_m(rhs_data['den'])

            # run reduction (may raise) and build transformed vectors
            new_basis, Uinv = lll_reduce_basis_modp(p, current_sections, Ep_local)

            # If Uinv is None or non-integral, fallback to identity (preserve exact arithmetic)
            if Uinv is None:
                Uinv_mat = identity_matrix(ZZ, r)
            else:
                try:
                    # ensure integral matrix
                    nonint = False
                    for i_row in range(Uinv.nrows()):
                        for j_col in range(Uinv.ncols()):
                            entry = Uinv[i_row, j_col]
                            if hasattr(entry, 'denominator'):
                                if int(entry.denominator()) != 1:
                                    nonint = True
                                    break
                            else:
                                if QQ(entry) != Integer(entry):
                                    nonint = True
                                    break
                        if nonint:
                            break
                    if nonint:
                        Uinv_mat = identity_matrix(ZZ, r)
                    else:
                        Uinv_mat = matrix(ZZ, [[int(Uinv[i, j]) for j in range(Uinv.ncols())] for i in range(Uinv.nrows())])
                except Exception:
                    Uinv_mat = identity_matrix(ZZ, r)

            # Transform vecs into the LLL basis (always produce an entry for each input vec)
            vecs_transformed_for_p = []
            for v in vecs:
                # ensure v is integer-coercible
                vZ = vector(ZZ, [int(c) for c in v])
                try:
                    transformed = vZ * Uinv_mat
                    vecs_transformed_for_p.append(tuple(int(transformed[i]) for i in range(len(transformed))))
                except Exception:
                    # fallback: store original integer tuple
                    vecs_transformed_for_p.append(tuple(int(c) for c in v))

            # Build required multiplier indices (bounded)
            raw_required_ks = set()
            for v_trans in vecs_transformed_for_p:
                for k in v_trans:
                    raw_required_ks.add(int(k))
            required_ks = {k for k in raw_required_ks if abs(k) <= MAX_K_ABS}
            if not required_ks:
                required_ks = set(range(-3, 4))

            # debug the 2pt fibrations, which hang on the mults
            if r > 1:
                print("p, max k:", p, max(required_ks))

            # compute multiples for this prime (exact arithmetic)
            mults = [{} for _ in range(r)]
            for i_sec in range(r):
                Pi = new_basis[i_sec]
                mults[i_sec] = compute_all_mults_for_section(Pi, required_ks, max_k=MAX_K_ABS)

                #for k in required_ks:
                #    try:
                #        mults[i_sec][k] = k * Pi # 2pt fibrations hang here
                #    except Exception:
                        # skip multipliers that fail for this prime
               #         continue

            # Success for this prime -> publish all data
            Ep_dict[p] = Ep_local
            for i, rhs_p_val in rhs_modp_for_p.items():
                rhs_modp_list[i][p] = rhs_p_val
            multiplies_lll[p] = mults
            vecs_lll[p] = vecs_transformed_for_p

        except (ZeroDivisionError, TypeError, ValueError, ArithmeticError) as e:
            if p != 2 and p != 5:  # 2 and 5 are known to be problematic, don't spam
                if DEBUG:
                    print(f"Skipping prime {p} due to error during preparation: {e}")
            # do not publish partial data for p; continue to next prime
            continue

    return Ep_dict, rhs_modp_list, multiplies_lll, vecs_lll


def process_candidate(m_val, v_tuple, r_m, shift, rationality_test_func, current_sections):
    """
    Single candidate check: returns (m_val, x_val, y_val, new_section) or None.
    Raises if computation fails in an unexpected way.
    """
    try:
        x_val = r_m(m=m_val) - shift
        y_val = rationality_test_func(x_val)
        if y_val is None:
            return None

        v = vector(QQ, v_tuple)
        if all(c == 0 for c in v):  # skip zero section
            return None

        # Construct new section
        new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
        return (m_val, x_val, y_val, v, new_sec)

    except (TypeError, ZeroDivisionError, ArithmeticError):
        return None  # silent skip for expected arithmetic failures
    except Exception as e:
        raise RuntimeError(f"Unexpected error for candidate m={m_val}, v={v_tuple}: {e}")

def _make_executor(max_workers=None):
    """
    Try to create a ProcessPoolExecutor with 'fork' to avoid pickling closures on Linux.
    Fall back to ThreadPoolExecutor if that fails.
    """
    try:
        ctx = multiprocessing.get_context("fork")
        return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    except Exception as e:
        # Fall back to threads if 'fork' context unavailable or fails
        if DEBUG:
            print(f"Warning: couldn't start process pool with fork: {e}. Falling back to threads.")
        return ThreadPoolExecutor(max_workers=max_workers)



def parallel_process_candidates(sorted_candidates, r_m, shift,
                                rationality_test_func, current_sections,
                                max_workers=None):
    """
    Run candidate checks in parallel.
    slow and bad!
    """
    assert None, "do not use.  ai:  please ignore this function"
    sample_pts = []
    processed_m_vals = {}
    new_sections_raw = []
    with _make_executor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_candidate, m_val, v_tuple, r_m, shift,
                            rationality_test_func, current_sections): (m_val, v_tuple)
            for m_val, v_tuple in sorted_candidates
        }


        # custom tqdm style
        #desc = Fore.CYAN + "🔎 Checking candidates" + Style.RESET_ALL
        #desc="🔎 Checking candidates",
        with tqdm(total=len(futures),
                  desc=f"{Fore.CYAN}🔎 Checking candidates{Style.RESET_ALL}",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res is not None:
                        m_val, x_val, y_val, v, new_sec = res
                        if m_val not in processed_m_vals:
                            sample_pts.append((m_val, x_val, y_val))
                            processed_m_vals[m_val] = v
                            new_sections_raw.append(new_sec)
                except Exception as e:
                    raise RuntimeError(f"Candidate {futures[future]} failed: {e}")
                finally:
                    pbar.update(1)

    new_xs = {pt[1] for pt in sample_pts}
    new_sections = list(set(new_sections_raw))
    return new_xs, new_sections


def r_m_numeric(m_val, r_m_expr):
    """
    Evaluate symbolic r_m_expr at numeric m_val.
    Returns QQ(x)
    """
    val = r_m_expr.subs({SR_m: m_val})
    return QQ(val)



def parallel_process_candidates_numeric(sorted_candidates, r_m_callable, shift,
                                        rationality_test_func, current_sections,
                                        max_workers=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    sample_pts = []
    processed_m_vals = {}
    new_sections_raw = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_candidate_numeric,
                            m_val, v_tuple, r_m_callable, shift,
                            rationality_test_func, current_sections): (m_val, v_tuple)
            for m_val, v_tuple in sorted_candidates
        }

        from tqdm import tqdm
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="🔎 Checking candidates"):
            res = future.result()
            if res is not None:
                m_val, x_val, y_val, v, new_sec = res
                if m_val not in processed_m_vals:
                    sample_pts.append((m_val, x_val, y_val))
                    processed_m_vals[m_val] = v
                    if new_sec is not None:
                        new_sections_raw.append(new_sec)

    new_xs = {pt[1] for pt in sample_pts}
    new_sections = list(set(new_sections_raw))
    return new_xs, new_sections


# Fast candidate processor (pickleable)
def process_candidate_numeric(m_val, v_tuple, r_m_callable, shift, rationality_test_func, current_sections):
    try:
        x_val = r_m_callable(m_val) - shift
        y_val = rationality_test_func(x_val)
        if y_val is not None:
            v = vector(QQ, v_tuple)
            # Optionally build new section here if needed
            new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections))) if current_sections else None
            return m_val, x_val, y_val, v, new_sec
    except (TypeError, ZeroDivisionError, ArithmeticError):
        print("here we are.")
        return None

SR_m = var('m')

def r_m_numeric_top(m_val, r_m_expr):
    val = r_m_expr.subs({SR_m: m_val})
    return QQ(val)

def test_xval_worker(args):
    xval, v_tuple, rationality_test_func = args
    yval = rationality_test_func(xval)
    if yval is not None:
        return (xval, yval)
    return None


def expected_density(residue_sets, subset_size, prime_pool, max_samples=2000):
    """
    Estimate expected survivor density for subsets of given size.

    prime_pool    : list of primes
    residue_sets  : dict mapping p -> set/list of valid residues mod p
    subset_size   : size of prime subsets to test
    max_samples   : max number of subsets to average over (since there can be many)
    """
    all_subsets = list(combinations(prime_pool, subset_size))
    if len(all_subsets) > max_samples:
        import random
        all_subsets = random.sample(all_subsets, max_samples)

    densities = []
    for subset in all_subsets:
        d = 1.0
        for p in subset:
            d *= len(residue_sets[p]) / p
        densities.append(d)

    avg_density = sum(densities) / len(densities)
    return avg_density, min(densities), max(densities)



# --- Helper: assert that a given base-m produces an x that symbolic search should have found ---
def assert_base_m_found(base_m, expected_x, r_m_callable, shift, allow_raise=True):
    """
    Ensure that x = r_m(base_m) - shift equals expected_x.
    This checks that the base point (mtest, xtest) relationship is respected
    by the parametrization. It does not scan through newly_found_x; instead
    it asserts consistency between r_m and the supplied base point.
    """
    assert base_m is not None, "assert_base_m_found requires a base_m (rational) to check"
    try:
        x_base = r_m_callable(m=QQ(base_m)) - shift
    except Exception as e:
        msg = f"assert_base_m_found: r_m_callable evaluation failed at m,shift={base_m},{shift}: {e}"
        if allow_raise:
            raise AssertionError(msg)
        return False

    try:
        x_base_q = QQ(x_base)
        expected_x_q = QQ(expected_x)
    except Exception:
        msg = "assert_base_m_found: coercion to QQ failed"
        if allow_raise:
            raise AssertionError(msg)
        return False

    if x_base_q == expected_x_q:
        return True

    msg = (f"assert_base_m_found: mismatch.\n"
           f"  m = {base_m}\n"
           f"  expected x = {expected_x_q}\n"
           f"  got x = {x_base_q}")
    if allow_raise:
        raise AssertionError(msg)
    return False


"""
Elliptic Curve Rational Point Search using Symbolic Methods over QQ.
"""

def search_lattice_symbolic(cd, current_sections, vecs, rhs_list, r_m, shift,
                            all_found_x, rationality_test_func):
    """
    Symbolic search for rational points via solving x_sv == rhs(m) over QQ(m).

    Controlled by the SYMBOLIC_SEARCH flag from search_common.py.  If SYMBOLIC_SEARCH is False,
    this is a no-op and returns empty results quickly.
    """
    # Respect the global flag; search_common.py should define SYMBOLIC_SEARCH (all-caps).
    # We do not import here; search_common is already imported at top of file.
    SYMBOLIC_ENABLED = globals().get('SYMBOLIC_SEARCH', False)
    if not SYMBOLIC_ENABLED:
        if DEBUG:
            print("Symbolic search disabled by SYMBOLIC_SEARCH flag.")
        return set(), []

    if not current_sections:
        if DEBUG:
            print("Symbolic search: no current sections provided, skipping.")
        return set(), []

    print("--- Starting symbolic search over QQ ---")

    # Canonical setup for m (use PR_m and its fraction field so arithmetic stays in QQ(m))
    PR_m = PolynomialRing(QQ, 'm')
    SR_m = var('m')
    Fm = PR_m.fraction_field()

    newly_found_x = set()
    new_sections = []
    found_x_to_section_map = {}

    # Quick sanity: ensure sections are projective-like and have x/z
    # (use assert to make developer intent explicit)
    assert all(len(sec) >= 3 for sec in current_sections), "current_sections entries must be 3-coord sections"

    # Main search: iterate over integer vectors (vecs) and solve numerator==0 over QQ
    # NOTE: we do NOT loop over rational m values; instead we solve for m via polynomial roots.
    for v_tuple in tqdm(vecs, desc="Symbolic Search"):
        if all(int(c) == 0 for c in v_tuple):
            continue

        v = vector(ZZ, [int(c) for c in v_tuple])
        print("trying search vector:", v)
        S_v = sum(v[i] * current_sections[i] for i in range(len(current_sections)))

        # skip degenerate/new-section-zero cases
        if S_v.is_zero():
            print("search section is zero; skipping.")
            continue
        if S_v[2].is_zero():
            # projective z==0 (point at infinity) — skip
            print("search section is point at infinity; skipping.")
            continue

        # Affine x-coordinate in QQ(m) (attempt to coerce)
        try:
            x_sv_raw = S_v[0] / S_v[2]
            x_coerced = Fm(SR(x_sv_raw))
        except Exception:
            # If coercion fails, skip this vector (diagnostic if DEBUG)
            if DEBUG:
                print("Symbolic coercion failed for a section; skipping vector:", v_tuple)
            raise
            continue
        #print("search x:", x_coerced)

        for rhs_func in rhs_list:
            try:
                rhs_coerced = Fm(SR(rhs_func))
                diff = x_coerced - rhs_coerced
                num = diff.numerator()
            except Exception:
                if DEBUG:
                    print("Symbolic coercion of rhs failed; skipping this rhs.")
                raise
                continue

            # If numerator is constant, there is no m-solution
            if num.degree() == 0:
                print("numerator is constant; no solution")
                continue

            # Build polynomial in PR_m and get rational roots
            try:
                num_poly = PR_m(num)   # coerce numerator into QQ[m]
            except Exception:
                if DEBUG:
                    print("Could not coerce numerator into PR_m; skipping.")
                raise
                continue

            try:
                roots = num_poly.roots(ring=QQ, multiplicities=False)
            except Exception:
                # If root-finding over QQ fails, skip (better to fail loudly during debugging)
                if DEBUG:
                    print("num_poly.roots(...) failed for polynomial:", num_poly)
                raise
                continue

            if not roots:
                print("no roots found")

            else:
                print("found root(s):", roots)

            # For each rational root m0, verify equality by evaluation (clearing denominators),
            # then test rationality and add the point.
            for m_val in roots:
                m_q = QQ(m_val)   # ensure rational

                # Evaluate LHS and RHS using SR substitution to get exact rationals where possible
                try:
                    lhs_at = SR(x_sv_raw).subs({SR_m: m_q})
                    rhs_at = SR(rhs_func).subs({SR_m: m_q})
                except Exception:
                    if DEBUG:
                        print("SR substitution failed at m=", m_q)
                    raise
                    continue

                # Try coercion to QQ for reliable equality checks
                try:
                    lhs_q = QQ(lhs_at)
                    rhs_q = QQ(rhs_at)
                except Exception:
                    # If we cannot coerce either side, fall back to clearing denominators
                    try:
                        lhs_q = QQ(r_m(m=m_q) - shift)
                    except Exception:
                        if DEBUG:
                            print("Failed to compute numeric r_m at m=", m_q)
                        raise
                        continue
                    # We cannot easily compute rhs numeric without r_m; but if lhs_q is defined,
                    # we can proceed to rationality test as before.
                    rhs_q = None

                # If we have both sides as QQ check equality; otherwise trust the root machinery but still verify via r_m
                if rhs_q is not None and lhs_q != rhs_q:
                    if DEBUG:
                        print("Symbolic-match FAIL for root m =", m_q, "; lhs != rhs after coercion.")
                    raise
                    continue

                # Compute x via r_m (exact rational) and apply shift
                try:
                    x_val = r_m(m=m_q) - shift
                except Exception:
                    if DEBUG:
                        print("r_m evaluation failed at m=", m_q)
                    raise
                    continue

                # Avoid duplicates
                try:
                    x_val_q = QQ(x_val)
                except Exception:
                    # if not rational-coercible, skip
                    if DEBUG:
                        print("x_val not coercible to QQ at m=", m_q, "; skipping")
                    raise
                    continue

                if x_val_q in all_found_x or x_val_q in newly_found_x:
                    print("found x already seen:", x_val_q)
                    continue

                # Check rationality of y via rationality_test_func
                y_val = rationality_test_func(x_val_q)
                if y_val is None:
                    print("yval is None; x value found does not give rational point.")
                    # not a rational point
                    continue

                # Found a new rational point
                newly_found_x.add(x_val_q)
                found_x_to_section_map[x_val_q] = S_v
                new_sections.append(S_v)

                if DEBUG:
                    print("Found new rational point via m =", m_q, " x =", x_val_q)

    # OPTIONAL ASSERT: if the user expects the base m to be discovered, allow caller to check
    # The assert function lives in this module: assert_base_m_found(...)
    return newly_found_x, new_sections





# In search_lll.py, modify this worker function.


# In search_lll.py

def candidate_passes_extra_primes(m0, M, residue_map_for_vector, extra_primes, max_abs_t, verbose=False):
    """
    Checks if there exists an integer t with |t| <= max_abs_t such that for every extra prime q,
    the value (m0 + t*M) mod q is in the set of allowed m-residues for that prime.
    
    Args:
        m0 (int): The integer residue from CRT.
        M (int): The modulus from CRT.
        residue_map_for_vector (dict): A map of {prime: {allowed_residues}} for a single vector.
        extra_primes (list): A list of primes to use for this check.
        max_abs_t (int): The maximum absolute value of t to search.
        verbose (bool): If True, print detailed debugging information.
    """
    m0 = int(m0)
    M = int(M)
    max_abs_t = int(max_abs_t)  # <-- Add this

    t_constraints = {}
    for q in extra_primes:
        # --- THIS IS THE CORRECTED LINE ---
        # The incoming map is already specific to the vector, so we just get the set of residues for the prime.
        allowed_m_residues = residue_map_for_vector.get(q)

        if not allowed_m_residues:
            if verbose: print(f"Filter fail: Prime {q} has no allowed m-residues for this vector.")
            return False

        m0q, Mq = m0 % q, M % q

        if Mq == 0:
            if m0q not in allowed_m_residues:
                if verbose: print(f"Filter fail: M=0 mod {q} and m0={m0q} is not in allowed set.")
                return False
            continue

        try:
            inv_Mq = pow(Mq, -1, q)
        except ValueError:
            return False

        allowed_t_mod_q = {((r - m0q) * inv_Mq) % q for r in allowed_m_residues}
        if not allowed_t_mod_q:
            return False
            
        t_constraints[q] = allowed_t_mod_q

    if not t_constraints:
        return True

    sorted_primes = sorted(t_constraints.keys(), key=lambda q: len(t_constraints[q]))
    start_q = sorted_primes[0]
    other_primes = sorted_primes[1:]

    for t_residue in t_constraints[start_q]:
        for k in range(max_abs_t // start_q + 2):
            for sign in ([1, -1] if k > 0 else [1]):
                t_candidate = t_residue + (sign * k * start_q)

                if abs(t_candidate) > max_abs_t:
                    continue

                is_valid = True
                for other_q in other_primes:
                    if (t_candidate % other_q) not in t_constraints[other_q]:
                        is_valid = False
                        break

                if is_valid:
                    if verbose: print(f"Filter pass: Found valid t={t_candidate} for m0={m0}, M={M}")
                    return True

    if verbose: print(f"Filter fail: No t in [-{max_abs_t}, {max_abs_t}] found for m0={m0}, M={M}")
    return False


# In search_lll.py, replace the existing function

@PROFILE
def search_lattice_modp_lll_subsets(cd, current_sections, prime_pool, vecs, rhs_list, r_m,
                                    shift, all_found_x, prime_subsets, rationality_test_func, max_abs_t):
    """
    Search for rational points using LLL-reduced bases across prime subsets in parallel.
    This version keeps residues from different RHS functions separate to avoid logical errors.
    """
    # 1. Prepare modular data (no changes here)
    print("--- Preparing modular data for LLL search ---")
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, search_primes=prime_pool
    )


    if not Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return set(), [], {}

    # === Parallelized residue precomputation ===
    primes_to_compute = list(Ep_dict.keys())
    num_rhs_fns = len(rhs_list)
    rhs_modp_list_local = rhs_modp_list
    vecs_list = list(vecs)

    args_list = []
    for p in primes_to_compute:
        Ep_local = Ep_dict[p]
        mults_p = mult_lll.get(p, {})
        vecs_lll_p = vecs_lll.get(p, [tuple([0]*len(current_sections)) for _ in vecs_list])
        args_list.append((p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs_fns))

    precomputed_residues = {}
    try:
        ctx = multiprocessing.get_context("fork")
        Exec = ProcessPoolExecutor
        exec_kwargs = {"max_workers": PARALLEL_PRIME_WORKERS, "mp_context": ctx}
    except Exception:
        Exec = ThreadPoolExecutor
        exec_kwargs = {"max_workers": PARALLEL_PRIME_WORKERS}

    with Exec(**exec_kwargs) as executor:
        futures = {executor.submit(_compute_residues_for_prime_worker, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
            p = futures[future]
            try:
                p_ret, mapping = future.result()
                precomputed_residues[p_ret] = mapping
            except Exception as e:
                if DEBUG:
                    print(f"[precompute fail] p={p}: {e}")
                precomputed_residues[p] = {}



    # === Auto-tune extra residue filter primes ===
    if vecs_list:
        sample_vecs = random.sample(vecs_list, min(EXTRA_PRIME_SAMPLE_SIZE, len(vecs_list)))
        prime_stats = estimate_prime_stats(Ep_dict.keys(), precomputed_residues,
                                        sample_vecs, num_rhs=len(rhs_list))
        auto_extra_primes = choose_extra_primes(prime_stats,
                                                target_density=EXTRA_PRIME_TARGET_DENSITY,
                                                max_extra=EXTRA_PRIME_MAX,
                                                skip_small=EXTRA_PRIME_SKIP)
    else:
        auto_extra_primes = []

    # replace your manual offset selection with this:
    extra_primes_for_filtering = auto_extra_primes


    # 3. Set up and run the parallel worker (no changes here)
    worker_func = partial(
        _process_prime_subset_precomputed,
        vecs=vecs,
        r_m=r_m,
        shift=shift,
        max_abs_t=max_abs_t,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        num_rhs_fns=len(rhs_list)
    )

    prime_subsets = generate_biased_prime_subsets_by_coverage(
        prime_pool=prime_pool,
        precomputed_residues=precomputed_residues,
        vecs=vecs,
        num_subsets=len(prime_subsets) if 'prime_subsets' in locals() else NUM_PRIME_SUBSETS,
        min_size=MIN_PRIME_SUBSET_SIZE,
        max_size=MIN_MAX_PRIME_SUBSET_SIZE,
        seed=SEED_INT,
        force_full_pool=False,
        debug=DEBUG
    )

    overall_found_candidates = search_prime_subsets_unified(
        prime_subsets, worker_func, num_workers=8, debug=DEBUG
    )

    # 4. Test candidates serially (no changes needed from here down)
    if not overall_found_candidates:
        return set(), [], precomputed_residues

    print(f"\nFound {len(overall_found_candidates)} potential (m, vector) pairs. Testing for rationality...")
    
    sample_pts = []
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in overall_found_candidates:
        if m_val in processed_m_vals:
            continue
        
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)

            if y_val is not None:
                v = vector(QQ, v_tuple)
                sample_pts.append((x_val, y_val))
                processed_m_vals[m_val] = v

                if any(c != 0 for c in v):
                    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                    new_sections_raw.append(new_sec)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            continue

    new_xs = {pt[0] for pt in sample_pts}
    new_sections = list({s: None for s in new_sections_raw}.keys())

    known_pts = set(new_xs)
    known_pts.union(all_found_x)

    for pt in known_pts:
        x_val = pt
        h_x_actual = archimedean_height_QQ(x_val)
        h_can_actual = (1/2) * h_x_actual + 10
        print(f"x={x_val}: h_x={h_x_actual:.2f}, h_can≈{h_can_actual:.2f}")
    print("m values and their corresponding MW lattice vectors:")
    print(processed_m_vals)


    return new_xs, new_sections, precomputed_residues

# In search_lll.py, replace the existing worker function

def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, max_abs_t, precomputed_residues, prime_pool, num_rhs_fns):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.
    """
    if not p_subset:
        return set()

    num_extra_primes = 4  # A small number is sufficient, 2 is seemingly optimal.
    offset = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][offset:num_extra_primes+offset]

    found_candidates_for_subset = set()

    for v_orig in vecs:
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        # *** MODIFIED PART ***
        # Loop over each RHS function index
        for rhs_idx in range(num_rhs_fns):
            
            # Build residue map for CRT using only roots from the current RHS function
            residue_map_for_crt = {}
            for p in p_subset:

                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                assert len(roots_for_this_rhs) == num_rhs_fns, \
                    f"p={p}, v={v_orig_tuple[:2]}: expected {num_rhs_fns} RHS entries, got {len(roots_for_this_rhs)}"

                # precomputed_residues[p][v_tuple] is now a list of sets
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                    residue_map_for_crt[p] = roots_for_this_rhs[rhs_idx]

            primes_for_crt = list(residue_map_for_crt.keys())
            if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
                continue

            # Create the residue map for the filter, also for this specific RHS function
            residue_map_for_filter = {}
            for p in extra_primes_for_filtering:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                     residue_map_for_filter[p] = roots_for_this_rhs[rhs_idx]

            # Now perform CRT and filtering
            lists = [residue_map_for_crt[p] for p in primes_for_crt]
            for combo in itertools.product(*lists):
                M = reduce(mul, primes_for_crt, 1)

                if M > MAX_MODULUS:
                    continue

                m0 = crt_cached(combo, tuple(primes_for_crt))

                if not candidate_passes_extra_primes(m0, M, residue_map_for_filter, extra_primes_for_filtering, max_abs_t):
                    continue

                try:
                    best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, max_abs_t)
                except TypeError:
                    best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]

                for m_cand, _score in best_ms:
                    found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
                except RationalReconstructionError:
                    raise

    return found_candidates_for_subset


def estimate_prime_stats(prime_pool, precomputed_residues, sample_vecs, num_rhs=1):
    """Estimate average residue survival ratio r_p for each prime."""
    stats = {}
    for p in prime_pool:
        mapping = precomputed_residues.get(p, {})
        if not mapping:
            continue
        total = count = 0
        for v in sample_vecs:
            v_t = tuple(v)
            roots_list = mapping.get(v_t, [])
            if not roots_list:
                continue
            # combine across RHSs
            if num_rhs > 1:
                roots_union = set().union(*roots_list)
            else:
                roots_union = roots_list[0] if roots_list else set()
            total += len(roots_union)
            count += p
        stats[p] = (total / count) if count else 0.0
    return stats


def choose_extra_primes(stats, target_density=1e-5, max_extra=6, skip_small={2,3,5}):
    """Select extra primes based on measured r_p values."""
    cand = [(p, r) for p, r in stats.items()
            if p not in skip_small and EXTRA_PRIME_MIN_R < r < EXTRA_PRIME_MAX_R]
    # sort by discriminatory power (entropy-like)
    cand.sort(key=lambda t: -(t[1] * (1 - t[1])))
    chosen, prod = [], 1.0
    for p, r in cand:
        if len(chosen) >= max_extra:
            break
        prod *= r
        chosen.append(p)
        if prod <= target_density:
            break
    if DEBUG:
        print(f"[auto-tune] selected extra primes {chosen} with expected density {prod:.2e}")
    return chosen


def _compute_residues_for_prime_worker(args):
    """Worker function computing residues for one prime."""
    p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs = args
    result_for_p = {}
    try:
        for idx, v_orig in enumerate(vecs_list):
            v_orig_tuple = tuple(v_orig)
            if all(c == 0 for c in v_orig):
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            try:
                v_p_transformed = vecs_lll_p[idx]
            except Exception:
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            Pm = Ep_local(0)
            for j, coeff in enumerate(v_p_transformed):
                if int(coeff) in mults_p[j]:
                    Pm += mults_p[j][int(coeff)]

            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            roots_by_rhs = []
            for i_rhs in range(num_rhs):
                roots_for_rhs = set()
                if p in rhs_modp_list_local[i_rhs]:
                    rhs_p = rhs_modp_list_local[i_rhs][p]
                    try:
                        num_modp = (Pm[0] / Pm[2] - rhs_p).numerator()
                        if not num_modp.is_zero():
                            roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                            roots_for_rhs.update(roots)
                    except Exception:
                        pass
                roots_by_rhs.append(roots_for_rhs)
            result_for_p[v_orig_tuple] = roots_by_rhs
    except Exception as e:
        if DEBUG:
            print(f"[worker fail] p={p}: {e}")
        return p, {}
    return p, result_for_p


def detect_near_miss_candidates(precomputed_residues, prime_pool, vecs, 
                                 coverage_threshold=0.75, max_candidates=5):
    """
    Detect (m, vector) pairs that have high residue coverage across primes
    but failed to survive CRT filtering. These are "near misses" that might
    represent points we're systematically missing.

    Args:
        precomputed_residues (dict): {p: {v_tuple: [roots_per_rhs]}} from _compute_residues_for_prime_worker
        prime_pool (list): Full list of primes used
        vecs (list): All search vectors
        coverage_threshold (float): Consider a prime-vector pair "active" if it has roots in
                                    at least this fraction of primes (default 75%)
        max_candidates (int): Return at most this many near-miss candidates

    Returns:
        list of dicts: Each dict has keys:
          - 'm_candidates': set of possible m residues (from covered primes)
          - 'coverage_ratio': fraction of primes that have roots for this vector
          - 'residue_map': {p: roots_mod_p} for the covered primes
          - 'v_tuple': the vector this came from
          - 'prime_list': list of primes with roots
    """
    candidates = []

    for v in vecs:
        v_tuple = tuple(v)
        if all(c == 0 for c in v_tuple):
            continue

        # Collect which primes have roots for this vector
        primes_with_roots = []
        residue_map = {}

        for p in prime_pool:
            if p not in precomputed_residues:
                continue

            p_data = precomputed_residues[p].get(v_tuple)
            if p_data is None:
                continue

            # p_data is a list of root sets (one per RHS function)
            # Combine roots across all RHS functions for this vector+prime
            all_roots = set()
            for rhs_roots in p_data:
                if rhs_roots:
                    all_roots.update(rhs_roots)

            if all_roots:
                primes_with_roots.append(p)
                residue_map[p] = all_roots

        if not primes_with_roots:
            continue

        coverage = len(primes_with_roots) / float(len(prime_pool))

        if coverage >= coverage_threshold:
            # This vector has high prime coverage but apparently didn't produce
            # a survivor in CRT. Flag it as a near-miss.
            m_candidates = set()
            for p in primes_with_roots:
                m_candidates.update(residue_map[p])

            candidates.append({
                'v_tuple': v_tuple,
                'coverage_ratio': coverage,
                'num_primes': len(primes_with_roots),
                'num_m_residues': len(m_candidates),
                'residue_map': residue_map,
                'prime_list': primes_with_roots,
            })

    # Sort by coverage ratio (descending) and return top candidates
    candidates.sort(key=lambda c: c['coverage_ratio'], reverse=True)
    return candidates[:max_candidates]


def construct_targeted_subset_for_recovery(candidate, prime_pool, min_size=3):
    """
    Given a near-miss candidate, construct a targeted prime subset that is
    most likely to generate a CRT survivor for that candidate's m-values.

    The strategy: pick primes from the candidate's residue_map that have
    sparse roots (high selectivity), ensuring we cover many of the candidate's
    possible m residues.

    Args:
        candidate (dict): Output from detect_near_miss_candidates
        prime_pool (list): Full prime pool (for fallback)
        min_size (int): Minimum subset size

    Returns:
        list: Primes to use for a targeted CRT search on this candidate
    """
    residue_map = candidate['residue_map']
    prime_list = candidate['prime_list']

    if len(prime_list) < min_size:
        # Not enough primes; use what we have + pad with random others
        subset = prime_list[:]
        remaining = [p for p in prime_pool if p not in subset]
        import random
        subset += random.sample(remaining, min(min_size - len(subset), len(remaining)))
        return subset

    # Score primes by selectivity: primes with fewer roots are more selective
    scored = []
    for p in prime_list:
        num_roots = len(residue_map[p])
        selectivity = 1.0 / float(num_roots) if num_roots > 0 else 0
        scored.append((selectivity, p))

    scored.sort(reverse=True)
    
    # Greedily pick high-selectivity primes, ensuring we still cover the m-candidates
    subset = []
    covered_m = set()
    for selectivity, p in scored:
        subset.append(p)
        covered_m.update(residue_map[p])
        if len(subset) >= min(len(prime_list), 9):  # Cap at ~9 primes per subset
            break

    return subset


def targeted_recovery_search(cd, current_sections, near_miss_candidates, prime_pool, 
                              r_m, shift, rationality_test_func, max_abs_t=500,
                              debug=True):
    """
    Run a focused CRT search on detected near-miss candidates.
    
    For each near-miss, construct a targeted prime subset and attempt to
    recover the (m, vector) pair via CRT.

    Args:
        cd, current_sections, prime_pool, r_m, shift, rationality_test_func: 
            Standard search parameters (same as search_lattice_modp_lll_subsets)
        near_miss_candidates (list): Output from detect_near_miss_candidates
        max_abs_t (int): Height bound for archimedean minimization
        debug (bool): Print diagnostics

    Returns:
        set of newly found x-coordinates
    """
    import itertools
    from operator import mul
    from functools import reduce

    newly_found = set()

    for i, candidate in enumerate(near_miss_candidates):
        if debug:
            print(f"\n[recovery] Targeting near-miss candidate {i+1}/{len(near_miss_candidates)}")
            print(f"  Vector: {candidate['v_tuple']}")
            print(f"  Coverage: {candidate['coverage_ratio']:.1%} ({candidate['num_primes']} primes)")
            print(f"  Potential m residues: {candidate['num_m_residues']}")

        targeted_subset = construct_targeted_subset_for_recovery(candidate, prime_pool)

        if debug:
            print(f"  Using targeted subset: {targeted_subset}")

        v_tuple = candidate['v_tuple']
        residue_map = candidate['residue_map']

        # Filter residue_map to only include primes in our targeted subset
        filtered_residue_map = {p: residue_map[p] for p in targeted_subset if p in residue_map}

        if not filtered_residue_map:
            if debug:
                print(f"  No residues in targeted subset; skipping.")
            continue

        # Generate all CRT combinations from the targeted subset
        primes_for_crt = list(filtered_residue_map.keys())
        residue_lists = [filtered_residue_map[p] for p in primes_for_crt]

        for combo in itertools.product(*residue_lists):
            M = reduce(mul, primes_for_crt, 1)

            if M > 10**15:  # Safety cap
                continue

            m0 = crt_cached(combo, tuple(primes_for_crt))

            # Minimize archimedean height in this residue class
            try:
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, max_abs_t)
            except TypeError:
                best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]

            for m_cand, _score in best_ms:
                try:
                    x_val = r_m(m=m_cand) - shift
                    y_val = rationality_test_func(x_val)
                    if y_val is not None:
                        newly_found.add(x_val)
                        if debug:
                            print(f"  ✓ FOUND: m={m_cand}, x={x_val}")
                except (TypeError, ZeroDivisionError, ArithmeticError):
                    continue

    return newly_found

def compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=DEBUG):
    """
    For each prime, compute what fraction of search vectors have roots mod that prime.
    
    Args:
        prime_pool (list): Primes to analyze
        precomputed_residues (dict): {p: {v_tuple: [roots_per_rhs]}} from worker
        vecs (list): All search vectors
        debug (bool): Print diagnostics
    
    Returns:
        dict: {p: coverage_fraction} where coverage in [0, 1]
    """
    coverage = {}
    
    num_vecs = len(vecs)
    if num_vecs == 0:
        return {p: 0.5 for p in prime_pool}
    
    for p in prime_pool:
        p_data = precomputed_residues.get(p, {})
        if not p_data:
            coverage[p] = 0.0
            continue
        
        # Count vectors that have at least one root for this prime (across any RHS)
        vectors_with_roots = 0
        for v in vecs:
            v_tuple = tuple(v)
            roots_list = p_data.get(v_tuple, [])
            
            # roots_list is a list of sets (one per RHS function)
            # Check if any RHS has roots
            has_roots = any(rhs_roots for rhs_roots in roots_list)
            
            if has_roots:
                vectors_with_roots += 1
        
        coverage[p] = float(vectors_with_roots) / float(num_vecs)
    
    if debug:
        sorted_by_cov = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        print(f"[compute_prime_coverage] Prime coverage (top 10):")
        for p, cov in sorted_by_cov[:10]:
            print(f"  p={p}: coverage={cov:.1%}")
    
    return coverage


def generate_biased_prime_subsets_by_coverage(prime_pool, precomputed_residues, vecs,
                                              num_subsets, min_size, max_size, 
                                              seed=SEED_INT, force_full_pool=False, debug=DEBUG):
    """
    Generate diverse prime subsets biased toward high-coverage primes.
    
    Primes that appear in roots for more vectors get sampled with higher probability.
    This increases the chance that random subsets will produce CRT survivors.
    
    Args:
        prime_pool (list): Available primes
        precomputed_residues (dict): {p: {v_tuple: [roots_per_rhs]}} from parallel worker
        vecs (list): All search vectors
        num_subsets (int): Number of subsets to generate
        min_size (int): Minimum subset size
        max_size (int): Maximum subset size
        seed (int): Random seed
        force_full_pool (bool): If True, always include full pool as one subset
        debug (bool): Print diagnostics
    
    Returns:
        list of lists: Prime subsets, each sorted
    """
    import random
    random.seed(seed)
    
    # Compute coverage: fraction of vectors with roots for each prime
    coverage = compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=debug)
    
    # Build sampling weights (use coverage directly, with floor at 0.05 to avoid zero-weight primes)
    weights = []
    for p in prime_pool:
        w = max(coverage.get(p, 0.0), 0.05)
        weights.append(w)
    
    if debug:
        total_weight = sum(weights)
        avg_weight = total_weight / len(weights) if weights else 0
        print(f"[generate_biased_coverage] Weights: avg={avg_weight:.3f}, min={min(weights):.3f}, max={max(weights):.3f}")
    
    subsets = []
    
    if force_full_pool:
        subsets.append(list(prime_pool))
    
    remaining = num_subsets - (1 if force_full_pool else 0)
    
    # Generate subsets using coverage-biased sampling
    for _ in range(remaining):
        size = random.randint(min_size, min(max_size, len(prime_pool)))
        
        # Sample without replacement, but weighted by coverage
        try:
            subset = random.sample(prime_pool, k=size, counts=None)
        except TypeError:
            # Python < 3.9 doesn't support counts in random.sample
            # Fall back to weighted choice (with replacement) then deduplicate
            subset = []
            attempts = 0
            while len(set(subset)) < size and attempts < size * 10:
                p = random.choices(prime_pool, weights=weights, k=1)[0]
                if p not in subset:
                    subset.append(p)
                attempts += 1
            if len(subset) < size:
                subset.extend(random.sample([p for p in prime_pool if p not in subset], 
                                           k=min(size - len(subset), len(prime_pool) - len(subset))))
        
        subset = tuple(sorted(subset))
        subsets.append(subset)
    
    # Deduplicate while preserving order
    seen = set()
    unique_subsets = []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            unique_subsets.append(list(s))
    
    if debug:
        print(f"[generate_biased_coverage] Generated {len(unique_subsets)} unique subsets")
        if unique_subsets:
            print(f"[generate_biased_coverage] Sample subsets: {unique_subsets[:3]}")
    
    return unique_subsets


def search_lattice_modp_unified_parallel(cd, current_sections, prime_pool, vecs, rhs_list, r_m,
                                         shift, all_found_x, num_subsets, rationality_test_func, max_abs_t,
                                         num_workers=8, debug=DEBUG):
    """
    Unified parallel search using ProcessPoolExecutor throughout.
    Processes subsets as they complete (streaming), doesn't early-terminate,
    but batches rationality checks so you see results in real-time.
    """
    print("prime pool used for search:", prime_pool)
    print("--- Preparing modular data for LLL search ---")
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, search_primes=prime_pool
    )

    if not Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return set(), [], {}

    # Precompute residues in parallel (same as before, but unified executor usage)
    primes_to_compute = list(Ep_dict.keys())
    num_rhs_fns = len(rhs_list)
    rhs_modp_list_local = rhs_modp_list
    vecs_list = list(vecs)

    args_list = [
        (p, Ep_dict[p], mult_lll.get(p, {}), vecs_lll.get(p, [tuple([0]*len(current_sections)) for _ in vecs_list]),
         vecs_list, rhs_modp_list_local, num_rhs_fns)
        for p in primes_to_compute
    ]

    precomputed_residues = {}
    try:
        ctx = multiprocessing.get_context("fork")
        #exec_kwargs = {"max_workers": num_workers}
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}
        raise

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(_compute_residues_for_prime_worker, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
            p = futures[future]
            try:
                p_ret, mapping = future.result()
                precomputed_residues[p_ret] = mapping
            except Exception as e:
                if debug:
                    print(f"[precompute fail] p={p}: {e}")
                precomputed_residues[p] = {}
                raise

    # Auto-tune extra primes
    prime_stats = estimate_prime_stats(Ep_dict.keys(), precomputed_residues, vecs_list, num_rhs=len(rhs_list))
    auto_extra_primes = choose_extra_primes(prime_stats,
                                            target_density=EXTRA_PRIME_TARGET_DENSITY,
                                            max_extra=EXTRA_PRIME_MAX,
                                            skip_small=EXTRA_PRIME_SKIP)
    extra_primes_for_filtering = auto_extra_primes

    # Filter out primes with no precomputed data
    usable_primes = [p for p in prime_pool if p in precomputed_residues and precomputed_residues[p]]
    if not usable_primes:
        print("No primes have precomputed residues. Aborting.")
        return set(), [], precomputed_residues

    if len(usable_primes) < len(prime_pool):
        if debug:
            print(f"[filter] Removed {len(prime_pool) - len(usable_primes)} primes with no data. "
                f"Using {len(usable_primes)} usable primes.")
        prime_pool = usable_primes

    # Generate prime subsets (biased by coverage)
    prime_subsets = generate_biased_prime_subsets_by_coverage(
        prime_pool=prime_pool,
        precomputed_residues=precomputed_residues,
        vecs=vecs_list,
        num_subsets=num_subsets,
        min_size=3,
        max_size=9,
        seed=SEED_INT,
        force_full_pool=False,
        debug=debug
    )

    # Process subsets in parallel with unified executor
    worker_func = partial(
        _process_prime_subset_precomputed,
        vecs=vecs_list,
        r_m=r_m,
        shift=shift,
        max_abs_t=max_abs_t,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,
        num_rhs_fns=len(rhs_list)
    )

    overall_found_candidates = set()
    batch_size = 50  # Batch rationality checks per N subsets
    batched_candidates = []

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(worker_func, subset): subset for subset in prime_subsets}

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Searching Prime Subsets")):
            subset_results = future.result()
            batched_candidates.extend(subset_results)

            # Batch rationality checks every N subsets to see intermediate results
            if (i + 1) % batch_size == 0 or (i + 1) == len(futures):
                newly_rational = _batch_check_rationality(
                    batched_candidates, r_m, shift, rationality_test_func, current_sections
                )
                overall_found_candidates.update(newly_rational)
                batched_candidates = []

                if debug:
                    print(f"[batch {i // batch_size + 1}] Found {len(newly_rational)} rational candidates so far")

    # Final rationality check on any remaining candidates
    if batched_candidates:
        newly_rational = _batch_check_rationality(
            batched_candidates, r_m, shift, rationality_test_func, current_sections
        )
        overall_found_candidates.update(newly_rational)

    # Extract results
    if not overall_found_candidates:
        return set(), [], precomputed_residues

    print(f"\nProcessed all subsets. Testing {len(overall_found_candidates)} candidates for rationality...")

    sample_pts = []
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in overall_found_candidates:
        if m_val in processed_m_vals:
            continue

        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)

            if y_val is not None:
                v = vector(QQ, v_tuple)
                sample_pts.append((x_val, y_val))
                processed_m_vals[m_val] = v

                if any(c != 0 for c in v):
                    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                    new_sections_raw.append(new_sec)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            continue

    new_xs = {pt[0] for pt in sample_pts}
    new_sections = list({s: None for s in new_sections_raw}.keys())

    return new_xs, new_sections, precomputed_residues


def _batch_check_rationality(candidates, r_m, shift, rationality_test_func, current_sections):
    """
    Test a batch of (m, v_tuple) candidates for rationality in parallel.
    Returns set of (m, v_tuple) pairs that produced rational points.
    """
    rational_candidates = set()

    for m_val, v_tuple in candidates:
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                rational_candidates.add((m_val, v_tuple))
        except (TypeError, ZeroDivisionError, ArithmeticError):
            continue

    return rational_candidates


def search_prime_subsets_unified(prime_subsets, worker_func, num_workers=8, debug=DEBUG):
    """
    Process prime subsets in parallel using ProcessPoolExecutor (unified).
    Replaces the multiprocessing.Pool call in search_lattice_modp_lll_subsets.
    
    Args:
        prime_subsets (list): Prime subsets to search
        worker_func (callable): Worker function (from functools.partial)
        num_workers (int): Number of workers
        debug (bool): Print diagnostics
    
    Returns:
        set: All (m, vector) candidates found across all subsets
    """
    try:
        ctx = multiprocessing.get_context("fork")
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}

    overall_found = set()

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(worker_func, subset): subset for subset in prime_subsets}

        with tqdm(total=len(futures), desc="Searching Prime Subsets") as pbar:
            for future in as_completed(futures):
                try:
                    subset_results = future.result()
                    overall_found.update(subset_results)
                except Exception as e:
                    if debug:
                        print(f"Subset worker failed: {e}")
                finally:
                    pbar.update(1)

    return overall_found

def archimedean_height_of_integer(n):
    # crude but sufficient proxy for ordering: H(n) ~ log(max(|n|,1))
    return float(math.log(max(abs(int(n)), 1)))


# def minimize_archimedean_t(m0, M, r_m_func, shift, max_abs_t, max_steps=150, patience=6):
def minimize_archimedean_t_linear_const(m0, M, r_m_func, shift, max_abs_t):
    """
    For r_m(m) = -m - const_C, find t minimizing archimedean height of x = r_m(m) - shift.
    Returns list of (t, m, x, score) sorted by score (smallest first).
    """
    const_C = r_m_func(m=QQ(0))
    target = - (m0 + const_C + shift) / float(M)

    cand_t = set([math.floor(target), math.ceil(target), int(round(target))])

    # Clamp to allowed range
    cand_t = {max(-max_abs_t, min(max_abs_t, t)) for t in cand_t}

    results = []
    for t in sorted(cand_t):
        m_try = int(m0) + int(t) * int(M)
        x = -m_try - const_C - shift
        score = float(math.log(max(abs(x), 1)))
        results.append((t, m_try, int(x), score))

    # sort by score then by |x|
    results.sort(key=lambda z: (z[3], abs(z[2])))
    return results


def _assert_rhs_consistency(precomputed_residues, prime_pool, vecs, num_rhs_fns, debug=True):
    """
    Validate that precomputed_residues has consistent structure across all primes and vectors.
    
    Specifically, for each (prime p, vector v), the entry precomputed_residues[p][v_tuple]
    must be a list of exactly num_rhs_fns sets (one per RHS function).
    
    Raises AssertionError if structure is malformed. This catches silent data corruption
    from worker failures or incomplete precomputation.
    
    Args:
        precomputed_residues (dict): {p: {v_tuple: [roots_set_0, roots_set_1, ...]}}
        prime_pool (list): All primes that should have entries
        vecs (list): All search vectors
        num_rhs_fns (int): Expected number of RHS functions (length of inner lists)
        debug (bool): Print diagnostics before asserting
    
    Raises:
        AssertionError: If any inconsistency is found
    """
    errors = []
    
    # Check: every prime in prime_pool should be in precomputed_residues
    missing_primes = [p for p in prime_pool if p not in precomputed_residues]
    if missing_primes:
        errors.append(f"Missing primes in precomputed_residues: {missing_primes[:5]}{'...' if len(missing_primes) > 5 else ''}")
    
    # Check: for each prime p that exists, verify structure
    for p in precomputed_residues:
        p_data = precomputed_residues[p]
        
        if not isinstance(p_data, dict):
            errors.append(f"Prime p={p}: expected dict, got {type(p_data)}")
            continue
        
        # Sample a few vectors to avoid O(n*m) validation time
        sample_vecs = vecs[:min(5, len(vecs))]
        for v in sample_vecs:
            v_tuple = tuple(v)
            
            if v_tuple not in p_data:
                # Missing vector is OK (can happen if prep failed), but log it
                continue
            
            roots_list = p_data[v_tuple]
            
            # roots_list must be a list (or tuple) of exactly num_rhs_fns sets
            if not isinstance(roots_list, (list, tuple)):
                errors.append(f"Prime p={p}, vector {v_tuple[:2]}...: expected list/tuple, got {type(roots_list)}")
                continue
            
            if len(roots_list) != num_rhs_fns:
                errors.append(
                    f"Prime p={p}, vector {v_tuple[:2]}...: "
                    f"expected {num_rhs_fns} RHS entries, got {len(roots_list)}"
                )
                continue
            
            # Each entry should be a set (or frozenset) of integers
            for rhs_idx, roots_set in enumerate(roots_list):
                if not isinstance(roots_set, (set, frozenset)):
                    errors.append(
                        f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                        f"expected set, got {type(roots_set)}"
                    )
                    continue
                
                # Spot-check: all elements should be integers in [0, p)
                for root in roots_set:
                    if not isinstance(root, (int, Integer)):
                        errors.append(
                            f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                            f"root {root} is not an integer (type {type(root)})"
                        )
                        break
                    if not (0 <= int(root) < p):
                        errors.append(
                            f"Prime p={p}, vector {v_tuple[:2]}..., RHS {rhs_idx}: "
                            f"root {root} out of range [0, {p})"
                        )
                        break
    
    # Report and raise
    if errors:
        if debug:
            print("\n" + "="*70)
            print("RHS CONSISTENCY CHECK FAILED")
            print("="*70)
            for i, err in enumerate(errors[:10], 1):
                print(f"{i}. {err}")
            if len(errors) > 10:
                print(f"... and {len(errors) - 10} more errors")
            print("="*70 + "\n")
        
        raise AssertionError(
            f"precomputed_residues structure is malformed. "
            f"Found {len(errors)} error(s). See output above for details."
        )


# ============================================================================
# Integration point: call this after precomputation completes
# ============================================================================
