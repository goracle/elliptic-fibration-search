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
from collections import namedtuple, Counter # <-- IMPORTED COUNTER
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Third-party imports
from tqdm import tqdm
from colorama import Fore, Style

# SageMath imports
from sage.all import (
    QQ, ZZ, GF, PolynomialRing, EllipticCurve,
    matrix, vector, identity_matrix, zero_matrix, diagonal_matrix,
    crt, lcm, sqrt, polygen, Integer, ceil
)
from sage.rings.rational import Rational
from sage.rings.fraction_field_element import FractionFieldElement

# Local imports (assuming these exist in your project)
from search_common import *
from search_common import DEBUG, PROFILE
from stats import * # <-- ADDED IMPORT

# Constants
DEFAULT_MAX_CACHE_SIZE = 10000
DEFAULT_MAX_DENOMINATOR_BOUND = None
FALLBACK_MATRIX_WARNING = "WARNING: LLL reduction failed, falling back to identity matrix"
ROOTS_THRESHOLD = 12 # only multiply primes' root counts into the estimate when the total roots for that prime exceed this threshold


# Practical tuning knobs
LLL_DELTA = 0.98           # strong LLL reduction; reduce if it slows too much (0.9--0.98 recommended)
BKZ_BLOCK = 12             # try BKZ with this block; lower for speed, larger for quality
MAX_COL_SCALE = 10**6      # don't scale any column by more than this (keeps integers reasonable)
TARGET_COLUMN_NORM = 1e6   # target column norm after scaling (heuristic)
MAX_K_ABS = 500            # ignore multiplier indices |k| > MAX_K_ABS when building mults
TRUNCATE_MAX_DEG = 30      # truncate polynomial coefficients at this degree to limit dimension
PARALLEL_PRIME_WORKERS = min(8, max(1, multiprocessing.cpu_count() // 2))
TMAX = 500

# ==============================================================
# === Auto-Tune / Residue Filter Parameters ====================
# ==============================================================

EXTRA_PRIME_TARGET_DENSITY = 1e-5   # desired survivor fraction after extras
EXTRA_PRIME_MAX = 6                 # cap on number of extra primes
EXTRA_PRIME_SKIP = {2, 3}        # avoid small degenerates
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
def minimize_archimedean_t(m0, M, r_m_func, shift, tmax, max_steps=150, patience=6):
    """
    Given residue class m = m0 (mod M), search over m = m0 + t*M to find integer t that minimizes
    archimedean height of x = r_m(m) - shift.

    Returns list of (m_candidate (QQ), score (float)) for the best few t values.
    """
    best = []

    def eval_for_t(t):
        m_candidate = m0 + t * M
        if abs(t) > tmax:
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
            if abs(s) >= tmax:
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
        for t_test in range(-tmax, tmax + 1):
            if m0 + t_test * M == m_cand:
                best_t_values.append(t_test)
                break

    return sorted_candidates[:3]

# --- Top-level Worker Function for Parallel Processing ---

def _process_prime_subset(p_subset, cd, current_sections, prime_pool, r_m, shift, rhs_list, vecs, tmax):
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
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
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

    if M.ncols() > 5 * M.nrows():
        if DEBUG:
            print(f"[LLL] Matrix too wide ({M.nrows()}x{M.ncols()}), skipping LLL for this prime")
        return [reduce_point_hom(P, curve_modp, p) for P in sections], identity_matrix(ZZ, r)

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
    


    try:
        if hasattr(M_scaled, "BKZ"):
            block = min(bkz_block, max(2, M_scaled.ncols()//2))
            U, B = M_scaled.BKZ(block_size=block, transformation=True)
        else:
            U, B = M_scaled.LLL(transformation=True, delta=float(lll_delta))
    except (TypeError, ValueError):
        try:
            U, B = M_scaled.LLL(transformation=True)
        except Exception as e:
            if DEBUG:
                print("LLL/BKZ reduction failed, falling back to identity:", e)
            U = identity_matrix(ZZ, r)
            B = M_scaled.copy()

    # DEBUG OUTPUT
    print(f"[LLL_DEBUG] p={p}, r={r}, M_scaled shape: {M_scaled.nrows()}x{M_scaled.ncols()}")
    print(f"[LLL_DEBUG] U type: {type(U).__name__}, has nrows: {hasattr(U, 'nrows')}")
    if hasattr(U, 'nrows'):
        print(f"[LLL_DEBUG] U shape: {U.nrows()}x{U.ncols()}")
    else:
        print(f"[LLL_DEBUG] U is not a matrix. repr (first 100 chars): {repr(U)[:100]}")
    print(f"[LLL_DEBUG] B type: {type(B).__name__}")

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
            print("U=", U)
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

def candidate_passes_extra_primes(m0, M, residue_map_for_vector, extra_primes, tmax, verbose=False):
    """
    Checks if there exists an integer t with |t| <= tmax such that for every extra prime q,
    the value (m0 + t*M) mod q is in the set of allowed m-residues for that prime.
    
    Args:
        m0 (int): The integer residue from CRT.
        M (int): The modulus from CRT.
        residue_map_for_vector (dict): A map of {prime: {allowed_residues}} for a single vector.
        extra_primes (list): A list of primes to use for this check.
        tmax (int): The maximum absolute value of t to search.
        verbose (bool): If True, print detailed debugging information.
    """
    m0 = int(m0)
    M = int(M)
    tmax = int(tmax)  # <-- Add this

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
        for k in range(tmax // start_q + 2):
            for sign in ([1, -1] if k > 0 else [1]):
                t_candidate = t_residue + (sign * k * start_q)

                if abs(t_candidate) > tmax:
                    continue

                is_valid = True
                for other_q in other_primes:
                    if (t_candidate % other_q) not in t_constraints[other_q]:
                        is_valid = False
                        break

                if is_valid:
                    if verbose: print(f"Filter pass: Found valid t={t_candidate} for m0={m0}, M={M}")
                    return True

    if verbose: print(f"Filter fail: No t in [-{tmax}, {tmax}] found for m0={m0}, M={M}")
    return False


# In search_lll.py, replace the existing function

# In search_lll.py, replace the existing worker function

def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, tmax, combo_cap, precomputed_residues, prime_pool, num_rhs_fns):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.
    """
    if not p_subset:
        return set()


    found_candidates_for_subset = set()
    stats_counter = Counter()
    tested_crt_classes = set()  # <-- NEW

    # these are now skipped!  this shouldn't print anymore!
    if len(p_subset) > 1 and all(p in precomputed_residues for p in p_subset):
        est = 1
        for p in p_subset:
            vks = precomputed_residues[p]
            for roots_list in vks.values():
                # roots_list is a list of sets per RHS function
                if any(len(roots) > ROOTS_THRESHOLD for roots in roots_list):
                    est *= sum(len(roots) for roots in roots_list)
        if est > combo_cap and DEBUG:
            print("[heavy subset]", p_subset, "estimated combos:", est)


    num_extra_primes = 4  # A small number is sufficient, 2 is seemingly optimal.
    offset = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][offset:num_extra_primes+offset]

    found_candidates_for_subset = set()
    stats_counter = Counter() # <-- Stats collector for this worker

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
                stats_counter['crt_lift_attempts'] += 1 # <-- STATS
                M = 1
                for p in primes_for_crt:
                    M *= int(p)

                #M = reduce(mul, primes_for_crt, 1)

                if M > MAX_MODULUS:
                    continue

                m0 = crt_cached(combo, tuple(primes_for_crt))
                tested_crt_classes.add((int(m0) % int(M), int(M)))  # <-- NEW


                if not candidate_passes_extra_primes(m0, M, residue_map_for_filter, extra_primes_for_filtering, tmax):
                    continue

                try:
                    best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
                except TypeError:
                    best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]

                for _, m_cand, _, _ in best_ms:  # Unpack all 4 values, ignore t, x, score
                    found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                #for m_cand, _score in best_ms:
                #    found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                stats_counter['rational_recon_attempts_worker'] += 1 # <-- STATS
                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
                    stats_counter['rational_recon_success_worker'] += 1 # <-- STATS
                    #print("prime subset:", combo, "found a candidate!:", (QQ(a) / QQ(b)), m)
                except RationalReconstructionError:
                    stats_counter['rational_recon_failure_worker'] += 1 # <-- STATS
                    raise

    return found_candidates_for_subset, stats_counter, tested_crt_classes  # <-- Return classes

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





def _batch_check_rationality(candidates, r_m, shift, rationality_test_func, current_sections, stats):
    """
    Test a batch of (m, v_tuple) candidates for rationality in parallel.
    Returns set of (m, v_tuple) pairs that produced rational points.
    UPDATED to accept and use a stats object with new counter names.
    """
    rational_candidates = set()

    for m_val, v_tuple in candidates:
        stats.incr('rationality_tests_total') # <-- STATS
        try:
            x_val = r_m(m=m_val) - shift
            y_val = rationality_test_func(x_val)
            if y_val is not None:
                stats.record_success(m_val, point=x_val) # <-- STATS (increments rationality_tests_success)
                rational_candidates.add((m_val, v_tuple))
            else:
                stats.record_failure(m_val, reason='y_not_rational') # <-- STATS (increments rationality_tests_failure)
        except (TypeError, ZeroDivisionError, ArithmeticError):
            stats.record_failure(m_val, reason='rationality_test_error') # <-- STATS (increments rationality_tests_failure)
            continue

    return rational_candidates




def archimedean_height_of_integer(n):
    # crude but sufficient proxy for ordering: H(n) ~ log(max(|n|,1))
    return float(math.log(max(abs(int(n)), 1)))


# def minimize_archimedean_t(m0, M, r_m_func, shift, tmax, max_steps=150, patience=6):
def minimize_archimedean_t_linear_const(m0, M, r_m_func, shift, tmax):
    """
    For r_m(m) = -m - const_C, find t minimizing archimedean height of x = r_m(m) - shift.
    Returns list of (t, m, x, score) sorted by score (smallest first).
    """
    const_C = r_m_func(m=QQ(0))
    target = - (m0 + const_C + shift) / float(M)

    cand_t = set([math.floor(target), math.ceil(target), int(round(target))])

    # Clamp to allowed range
    cand_t = {max(-tmax, min(tmax, t)) for t in cand_t}

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


# Add to search_lll.py (after the main function)

def _print_subset_productivity_stats(productive, all_subsets):
    """Print quick stats on which prime subsets were productive"""
    from collections import Counter
    
    total = len(all_subsets)
    productive_count = len(productive)
    total_candidates = sum(p['candidates'] for p in productive)
    
    print(f"\n[subset stats] {productive_count}/{total} subsets produced candidates "
          f"({100*productive_count/total:.1f}%)")
    print(f"[subset stats] {total_candidates} total candidates from productive subsets")
    
    # By size
    by_size = Counter(p['size'] for p in productive)
    all_by_size = Counter(len(s) for s in all_subsets)
    
    print(f"[subset stats] Productivity by size:")
    for size in sorted(all_by_size.keys()):
        prod_count = by_size.get(size, 0)
        total_count = all_by_size[size]
        rate = 100 * prod_count / total_count if total_count > 0 else 0
        cands = sum(p['candidates'] for p in productive if p['size'] == size)
        print(f"  Size {size}: {prod_count}/{total_count} productive ({rate:.1f}%), "
              f"{cands} candidates")
    
    # Top productive subsets
    top = sorted(productive, key=lambda x: x['candidates'], reverse=True)[:5]
    print(f"[subset stats] Top 5 productive subsets:")
    for p in top:
        print(f"  {p['primes']}: {p['candidates']} candidates")


def prepare_modular_data_lll(cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=None):
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

            # Build required multiplier indices: union across ALL vectors for this section
            required_ks_per_section = [set() for _ in range(r)]
            for v_trans in vecs_transformed_for_p:
                for j, coeff in enumerate(v_trans):
                    required_ks_per_section[j].add(int(coeff))

            # Now compute mults per section with only what's needed
            mults = [{} for _ in range(r)]
            for i_sec in range(r):
                Pi = new_basis[i_sec]
                required_ks = required_ks_per_section[i_sec]
                if not required_ks:
                    required_ks = {-1, 0, 1}

                # Pass 'stats' object here
                mults[i_sec] = compute_all_mults_for_section(
                    Pi, required_ks, stats, # <-- Pass stats
                    max_k=max((abs(k) for k in required_ks), default=1),
                    debug=(r>1)
                )


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

# In search_lll.py

# Add 'stats' to the signature
def compute_all_mults_for_section(Pi, required_ks, stats, max_k=MAX_K_ABS, debug=False):
    mults = {}
    computed = {0: Pi.curve()(0), 1: Pi}

    sorted_ks = sorted(required_ks, key=abs)

    for i, k in enumerate(sorted_ks):
        if debug and i % 10 == 0:
            print(f"  [mults] Computing k={k} ({i}/{len(sorted_ks)})", flush=True)

        if k in computed:
            mults[k] = computed[k]
            continue

        abs_k = abs(k)

        if abs_k // 2 in computed:
            half_k = abs_k // 2
            base = computed[half_k]
            if debug:
                print(f"    [mults] k={k}: building from {half_k}*Pi", flush=True)

            if abs_k % 2 == 0:
                if debug:
                    print(f"      [mults] Doubling {half_k}*Pi...", flush=True)
                result = base + base
                stats.incr('multiply_ops') # <-- STATS (doubling)
            else:
                if debug:
                    print(f"      [mults] Adding {half_k}*Pi + {half_k}*Pi + Pi...", flush=True)
                result = base + base + Pi
                stats.incr('multiply_ops', n=2) # <-- STATS (doubling + addition)

            if k < 0:
                result = -result
                # We don't count negation as an op here, could add if needed
            computed[k] = result
            mults[k] = result
        else:
            # Fallback to direct multiplication (less efficient, counts as one op)
            if debug:
                print(f"    [mults] k={k}: direct multiplication", flush=True)
            try:
                mults[k] = k * Pi
                stats.incr('multiply_ops') # <-- STATS (direct multiplication)
                computed[k] = mults[k]
            except Exception as e:
                if debug:
                    print(f"      [mults] Failed: {e}", flush=True)

    return mults

# In search_lll.py

# Add 'stats' to the signature
def search_lattice_symbolic(cd, current_sections, vecs, rhs_list, r_m, shift,
                            all_found_x, rationality_test_func, stats):
    """
    Symbolic search for rational points via solving x_sv == rhs(m) over QQ(m).

    Controlled by the SYMBOLIC_SEARCH flag from search_common.py. If SYMBOLIC_SEARCH is False,
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
    stats.start_phase('symbolic_search') # <-- STATS

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
        #print("trying search vector:", v) # Reduced verbosity
        S_v = sum(v[i] * current_sections[i] for i in range(len(current_sections)))

        # skip degenerate/new-section-zero cases
        if S_v.is_zero():
            #print("search section is zero; skipping.")
            continue
        if S_v[2].is_zero():
            # projective z==0 (point at infinity) — skip
            #print("search section is point at infinity; skipping.")
            continue

        # Affine x-coordinate in QQ(m) (attempt to coerce)
        try:
            x_sv_raw = S_v[0] / S_v[2]
            x_coerced = Fm(SR(x_sv_raw))
        except Exception:
            # If coercion fails, skip this vector (diagnostic if DEBUG)
            if DEBUG:
                print("Symbolic coercion failed for a section; skipping vector:", v_tuple)
            # raise # Let's not raise here unless debugging is critical
            continue
        #print("search x:", x_coerced)

        for rhs_func in rhs_list:
            stats.incr('symbolic_solves_attempted') # <-- STATS
            try:
                rhs_coerced = Fm(SR(rhs_func))
                diff = x_coerced - rhs_coerced
                num = diff.numerator()
            except Exception:
                if DEBUG:
                    print("Symbolic coercion of rhs failed; skipping this rhs.")
                # raise
                continue

            # If numerator is constant, there is no m-solution
            if num.degree() == 0:
                #print("numerator is constant; no solution")
                continue

            # Build polynomial in PR_m and get rational roots
            try:
                num_poly = PR_m(num)   # coerce numerator into QQ[m]
            except Exception:
                if DEBUG:
                    print("Could not coerce numerator into PR_m; skipping.")
                # raise
                continue

            try:
                roots = num_poly.roots(ring=QQ, multiplicities=False)
            except Exception:
                # If root-finding over QQ fails, skip (better to fail loudly during debugging)
                if DEBUG:
                    print("num_poly.roots(...) failed for polynomial:", num_poly)
                # raise
                continue

            if not roots:
                #print("no roots found")
                pass # This happens often, no need to print
            else:
                stats.incr('symbolic_solves_success', n=len(roots)) # <-- STATS
                if DEBUG: print("Symbolic solve success! Found root(s):", roots)

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
                    # raise
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
                        # raise
                        continue
                    # We cannot easily compute rhs numeric without r_m; but if lhs_q is defined,
                    # we can proceed to rationality test as before.
                    rhs_q = None

                # If we have both sides as QQ check equality; otherwise trust the root machinery but still verify via r_m
                if rhs_q is not None and lhs_q != rhs_q:
                    if DEBUG:
                        print("Symbolic-match FAIL for root m =", m_q, "; lhs != rhs after coercion.")
                    # raise
                    continue

                # Compute x via r_m (exact rational) and apply shift
                try:
                    x_val = r_m(m=m_q) - shift
                except Exception:
                    if DEBUG:
                        print("r_m evaluation failed at m=", m_q)
                    # raise
                    continue

                # Avoid duplicates
                try:
                    x_val_q = QQ(x_val)
                except Exception:
                    # if not rational-coercible, skip
                    if DEBUG:
                        print("x_val not coercible to QQ at m=", m_q, "; skipping")
                    # raise
                    continue

                if x_val_q in all_found_x or x_val_q in newly_found_x:
                    #print("found x already seen:", x_val_q)
                    continue

                # Check rationality of y via rationality_test_func
                stats.incr('rationality_tests_total') # <-- STATS (Symbolic path)
                y_val = rationality_test_func(x_val_q)
                if y_val is None:
                    stats.record_failure(m_q, reason='y_not_rational_symbolic') # <-- STATS
                    #print("yval is None; x value found does not give rational point.")
                    # not a rational point
                    continue

                # Found a new rational point
                stats.record_success(m_q, point=x_val_q) # <-- STATS (Symbolic path)
                newly_found_x.add(x_val_q)
                found_x_to_section_map[x_val_q] = S_v
                new_sections.append(S_v)

                if DEBUG:
                    print("Found new rational point via symbolic m =", m_q, " x =", x_val_q)

    # OPTIONAL ASSERT: if the user expects the base m to be discovered, allow caller to check
    # The assert function lives in this module: assert_base_m_found(...)
    stats.end_phase('symbolic_search') # <-- STATS
    return newly_found_x, new_sections, stats

# In search_lll.py

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
        list: A list of tuples, one for each subset processed:
              [(subset, candidates_set, worker_stats_dict), ...]
        Counter: Merged stats_counter dict from all workers (Redundant, can be rebuilt from list)
    """
    try:
        ctx = multiprocessing.get_context("fork")
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}

    # List to store results per subset
    subset_results_list = []
    merged_stats = Counter() # Keep merging stats here too for now
    all_crt_classes = set()  # <-- NEW

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(worker_func, subset): subset for subset in prime_subsets}

        with tqdm(total=len(futures), desc="Searching Prime Subsets") as pbar:
            for future in as_completed(futures):
                original_subset = futures[future]
                try:
                    # Worker now returns three items
                    candidates_set, stats_dict, crt_classes  = future.result()
                    # Append the result tuple to the list
                    subset_results_list.append((original_subset, candidates_set, stats_dict))
                    merged_stats.update(stats_dict) # Keep merging here
                    all_crt_classes.update(crt_classes)  # <-- Collect
                except Exception as e:
                    if debug:
                        print(f"Subset worker failed for subset {original_subset}: {e}")
                    # Append a failure placeholder if needed, or just skip
                    subset_results_list.append((original_subset, set(), Counter()))
                    raise
                finally:
                    pbar.update(1)

    # Return the list of per-subset results and the merged stats
    return subset_results_list, merged_stats, all_crt_classes  # <-- Return classes


# In search_lll.py

# ============================================================================
# Integration point: call this after precomputation completes
# ============================================================================
def search_lattice_modp_unified_parallel(cd, current_sections, prime_pool, height_bound,
                                         vecs, rhs_list, r_m, shift,
                                         all_found_x, num_subsets, rationality_test_func,
                                         sconf, num_workers=8, debug=DEBUG):
    """
    Unified parallel search using ProcessPoolExecutor throughout.
    Hardened against the "filtered to 0 subsets" failure:
      - require primes to have actual residues (not just empty mappings)
      - compute numeric residue sets per-prime and use those counts for combo estimates
      - fall back deterministically if coverage-based generator returns nothing
    Returns: new_xs, new_sections, precomputed_residues, stats
    """
    # === UNPACK: SCONF ===
    min_prime_subset_size = sconf['MIN_PRIME_SUBSET_SIZE']
    min_max_prime_subset_size = sconf['MIN_MAX_PRIME_SUBSET_SIZE']
    max_modulus = sconf['MAX_MODULUS']
    tmax = sconf['TMAX']

    # === STATS: INIT ===
    stats = SearchStats()

    from bounds import compute_residue_counts_for_primes  # if not already imported
    residue_counts = compute_residue_counts_for_primes(cd, rhs_list, prime_pool, max_primes=30)
    coverage_estimator = CoverageEstimator(prime_pool, residue_counts)

    print("prime pool used for search:", prime_pool)

    # === PHASE: PREP MOD DATA ===
    stats.start_phase('prep_mod_data')
    print("--- Preparing modular data for LLL search ---")
    # Pass stats down
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=prime_pool
    )
    stats.end_phase('prep_mod_data')

    if not Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return set(), [], {}, stats  # <-- Return stats

    # === PHASE: PRECOMPUTE RESIDUES ===
    stats.start_phase('precompute_residues')
    primes_to_compute = list(Ep_dict.keys())
    num_rhs_fns = len(rhs_list)
    vecs_list = list(vecs)

    args_list = [
        (
            p,
            Ep_dict[p],
            mult_lll.get(p, {}),
            vecs_lll.get(p, [tuple([0] * len(current_sections)) for _ in vecs_list]),
            vecs_list,
            rhs_modp_list,
            num_rhs_fns,
            stats  # pass the stats object (worker ignores if not used)
        )
        for p in primes_to_compute
    ]

    precomputed_residues = {}
    total_modular_checks = 0

    try:
        ctx = multiprocessing.get_context("fork")
        exec_kwargs = {"max_workers": num_workers, "mp_context": ctx}
    except Exception:
        exec_kwargs = {"max_workers": num_workers}

    with ProcessPoolExecutor(**exec_kwargs) as executor:
        futures = {executor.submit(_compute_residues_for_prime_worker, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-computing residues"):
            p = futures[future]
            try:
                p_ret, mapping, local_modular_checks = future.result()
                mapping = mapping or {}
                precomputed_residues[p_ret] = mapping
                total_modular_checks += int(local_modular_checks or 0)

                # Now compute the union of numeric residues (ignore non-int markers)
                residues_union = set()
                for vtuple, rhs_lists in mapping.items():
                    for rl in rhs_lists:
                        # ignore sentinel markers like "DEN_ZERO" or other non-integer entries
                        for r in rl:
                            if isinstance(r, int):
                                residues_union.add(r)

                stats.residues_by_prime[p_ret].update(residues_union)

                # update main counters per-prime
                stats.counters['modular_checks'] += int(local_modular_checks or 0)
                stats.counters[f'modular_checks_p_{p_ret}'] += int(local_modular_checks or 0)
                stats.counters[f'residues_seen_p_{p_ret}'] = len(stats.residues_by_prime[p_ret])

            except Exception as e:
                if debug:
                    print(f"[precompute fail] p={p}: {e}")
                precomputed_residues[p] = {}
                stats.residues_by_prime[p].update(set())
                stats.counters[f'modular_checks_p_{p}'] = 0
                stats.counters[f'residues_seen_p_{p}'] = 0

    if debug:
        print(f"[precompute] total_modular_checks={total_modular_checks}, primes precomputed={len(precomputed_residues)}")

    stats.end_phase('precompute_residues')


    ##### WHY ISN'T A PARTICULAR FIBRATION FINDING A POINT?  FIND OUT HERE!

    if False: # comment out when not in use

        ret = diagnose_missed_point(182/141, r_m, shift, precomputed_residues, prime_pool, vecs)
        #print("ret=", ret)
        matched_subset = None
        if 'matched_primes' in ret:
            matched_subset = ret['matched_primes']

        cov1 = compute_residue_coverage_for_m(QQ(-323)/QQ(141), precomputed_residues, PRIME_POOL)
        print("cov1: m = -323/141 coverage:", cov1['coverage_fraction'])
        print("cov1: matched primes:", cov1['matched_primes'])

        cov2 = compute_residue_coverage_for_m(QQ(-41)/QQ(141), precomputed_residues, PRIME_POOL)
        print("cov2: m = -41/141 coverage:", cov2['coverage_fraction'])
        print("cov2: matched primes:", cov2['matched_primes'])

        cov3 = compute_residue_coverage_for_m(QQ(41)/QQ(141), precomputed_residues, PRIME_POOL)
        print("cov3: m = 41/141 coverage:", cov3['coverage_fraction'])
        print("cov3: matched primes:", cov3['matched_primes'])


    # Build a per-prime numeric residue set for later use (and require non-empty)
    residues_by_prime_numeric = {}
    for p, mapping in precomputed_residues.items():
        residues_set = set()
        for vtuple, rhs_lists in mapping.items():
            for rl in rhs_lists:
                for r in rl:
                    if isinstance(r, int):
                        residues_set.add(r)
        residues_by_prime_numeric[p] = residues_set

    # Only keep primes that actually gave numeric residues (not merely empty mappings)
    usable_primes = [p for p in prime_pool if p in residues_by_prime_numeric and residues_by_prime_numeric[p]]
    if not usable_primes:
        print("No primes have numeric precomputed residues. Aborting.")
        return set(), [], precomputed_residues, stats
    if len(usable_primes) < len(prime_pool):
        if debug:
            print(f"[filter] Removed {len(prime_pool) - len(usable_primes)} primes with no numeric data. Using {len(usable_primes)} usable primes.")
        prime_pool = usable_primes

    # === PHASE: AUTOTUNE PRIMES ===
    stats.start_phase('autotune_primes')
    prime_stats = estimate_prime_stats(prime_pool, precomputed_residues, vecs_list, num_rhs=len(rhs_list))
    auto_extra_primes = choose_extra_primes(prime_stats,
                                            target_density=EXTRA_PRIME_TARGET_DENSITY,
                                            max_extra=EXTRA_PRIME_MAX,
                                            skip_small=EXTRA_PRIME_SKIP)
    extra_primes_for_filtering = auto_extra_primes
    stats.end_phase('autotune_primes')

    # Filtering stage: compute product estimate using distinct numeric residues per prime
    combo_cap = ceil(50000**(7*min_prime_subset_size/3)) # too many residues for this prime subset, too many possibilities, modular constraints are too loose
    roots_threshold = ROOTS_THRESHOLD
    if debug:
        print("combo_cap:", combo_cap, "roots_threshold:", roots_threshold)

    # === PHASE: GEN SUBSETS ===
    stats.start_phase('gen_subsets')
    prime_subsets_initial = generate_biased_prime_subsets_by_coverage(
        prime_pool=prime_pool,
        precomputed_residues=precomputed_residues,
        vecs=vecs_list,
        num_subsets=num_subsets,
        min_size=min_prime_subset_size,
        max_size=min_max_prime_subset_size,
        combo_cap=combo_cap,
        seed=SEED_INT,
        force_full_pool=False,
        debug=debug
    )
    stats.incr('subsets_generated_initial', n=len(prime_subsets_initial))

    filtered_subsets = []
    for subset in prime_subsets_initial:
        est = 1
        is_viable = True
        for p in subset:
            residues_set = residues_by_prime_numeric.get(p, set())
            roots_count = len(residues_set)
            if roots_count == 0:
                is_viable = False
                break
            # if any single prime has more residues than the threshold, it's likely to explode
            if roots_count > roots_threshold:
                est *= roots_count
                if est > combo_cap:
                    is_viable = False
                    break
            else:
                est *= max(1, roots_count)
                if est > combo_cap:
                    is_viable = False
                    break
        if is_viable and est <= combo_cap:
            filtered_subsets.append(subset)

    filtered_out_count = len(prime_subsets_initial) - len(filtered_subsets)
    stats.incr('subsets_filtered_out_combo', n=filtered_out_count)
    if debug:
        print("Generated", len(prime_subsets_initial), "prime_subsets -> filtered to", len(filtered_subsets))
    prime_subsets_to_process = filtered_subsets
    assert matched_subset is None or matched_subset in prime_subsets_to_process, (prime_subsets_to_process, matched_subset)
    count_subsets = {}
    for subset in prime_subsets_to_process:
        key = len(subset)
        if key in count_subsets:
            count_subsets[key] += 1
        else:
            count_subsets[key] = 0

    for key in count_subsets:
        print("using", count_subsets[key], "subsets of len =", key)

    # If filtering removed everything, build a deterministic fallback pool of small subsets.
    if not prime_subsets_to_process:
        if debug:
            print("[fallback] coverage-based filtering removed all subsets. Building deterministic fallback subsets.")
        from itertools import combinations
        fallback = []
        max_k = min(6, len(prime_pool))
        # prefer sizes 3..max_k
        for k in range(3, max_k + 1):
            for comb in combinations(prime_pool, k):
                # only keep combos with at least one residue per prime
                good = True
                for p in comb:
                    if not residues_by_prime_numeric.get(p):
                        good = False
                        break
                if not good:
                    continue
                # estimate as above
                est = 1
                for p in comb:
                    est *= max(1, len(residues_by_prime_numeric[p]))
                    if est > combo_cap:
                        good = False
                        break
                if good:
                    fallback.append(list(comb))
                if len(fallback) >= max(1, num_subsets):
                    break
            if len(fallback) >= max(1, num_subsets):
                break
        if fallback:
            prime_subsets_to_process = fallback[:num_subsets]
            if debug:
                print(f"[fallback] Using {len(prime_subsets_to_process)} deterministic fallback subsets.")
        else:
            # give up cleanly
            print("No viable prime subsets generated or remaining after filtering. Aborting.")
            stats.end_phase('gen_subsets')
            print("\n--- Search Statistics (No Subsets) ---")
            print(stats.summary_string())
            return set(), [], precomputed_residues, stats

    stats.end_phase('gen_subsets')

    # === PHASE: SEARCH & CHECK ===
    stats.start_phase('search_subsets_and_check')
    worker_func = partial(
        _process_prime_subset_precomputed,
        vecs=vecs_list,
        r_m=r_m,
        shift=shift,
        tmax=tmax,
        combo_cap=combo_cap,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool,  # current (filtered) prime_pool
        num_rhs_fns=len(rhs_list)
    )

    subset_results_list, worker_stats_dict, all_crt_classes = search_prime_subsets_unified(
        prime_subsets_to_process, worker_func, num_workers=num_workers, debug=debug
    )

    # update coverage estimator
    coverage_estimator.tested_classes = all_crt_classes
    coverage_report = coverage_estimator.estimate_coverage(prime_subsets_to_process)

    if debug:
        print("\n--- Coverage Estimate ---")
        if coverage_report.get('direct_coverage') is not None:
            print(f"  Direct coverage: {100 * coverage_report['direct_coverage']:.2f}%")
        if coverage_report.get('birthday_coverage') is not None:
            print(f"  Birthday estimate: {100 * coverage_report['birthday_coverage']:.2f}%")
        print(f"  Heuristic (density): {100 * coverage_report.get('heuristic_coverage', 0):.4f}%")
        print(f"  CRT classes tested: {coverage_report.get('classes_tested', 0):,}")
        print(f"  Search space size: ~{coverage_report.get('space_size_estimate', 0):.2e}")
        additional_runs = coverage_estimator.recommend_additional_runs(prime_subsets_to_process, target_coverage=0.95)
        if additional_runs > 0:
            print(f"  ⚠️  Recommend {additional_runs} more run(s) to reach 95% coverage")

    # Merge worker stats collected by the manager
    stats.merge_dict(worker_stats_dict)
    stats.incr('subsets_processed', n=len(subset_results_list))

    # aggregate worker candidates
    overall_found_candidates_from_workers = set()
    productive_subsets_data = []
    for subset, candidates_set, _ in subset_results_list:
        overall_found_candidates_from_workers.update(candidates_set)
        if candidates_set:
            productive_subsets_data.append({
                'primes': subset,
                'size': len(subset),
                'candidates': len(candidates_set)
            })

    stats.incr('crt_candidates_found', n=len(overall_found_candidates_from_workers))

    # Batch check rationality
    print(f"\nChecking rationality for {len(overall_found_candidates_from_workers)} unique candidates...")
    final_rational_candidates = set()
    candidate_list = list(overall_found_candidates_from_workers)
    if not candidate_list:
        stats.end_phase('search_subsets_and_check')
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return set(), [], precomputed_residues, stats

    batch_size = max(1, floor(0.05 * len(candidate_list)))
    for i in range(0, len(candidate_list), batch_size):
        batch = candidate_list[i:i + batch_size]
        newly_rational = _batch_check_rationality(
            batch, r_m, shift, rationality_test_func, current_sections, stats
        )
        final_rational_candidates.update(newly_rational)
        if debug:
            print(f"[batch check] processed {min(i + batch_size, len(candidate_list))}/{len(candidate_list)}, found {len(final_rational_candidates)} rational so far")

    stats.end_phase('search_subsets_and_check')

    # print productivity stats
    try:
        _print_subset_productivity_stats(productive_subsets_data, prime_subsets_to_process)
    except Exception as e:
        if debug:
            print(f"Failed to print productivity stats: {e}")

    if not final_rational_candidates:
        print("\n--- Search Statistics (No Points Found) ---")
        print(stats.summary_string())
        return set(), [], precomputed_residues, stats

    print(f"\nFound {len(final_rational_candidates)} rational (m, vector) pairs after checking.")

    # === PHASE: POST PROCESS ===
    stats.start_phase('post_process')
    sample_pts = []
    new_sections_raw = []
    processed_m_vals = {}

    for m_val, v_tuple in final_rational_candidates:
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
    stats.incr('rational_points_unique', n=len(new_xs))
    stats.incr('new_sections_unique', n=len(new_sections))
    stats.end_phase('post_process')

    print("\n--- Search Statistics ---")
    print(stats.summary_string())

    return new_xs, new_sections, precomputed_residues, stats


def _compute_residues_for_prime_worker(args):
    """
    Worker computing residues for one prime.
    Returns (p, result_for_p, local_modular_checks)
    - result_for_p: { v_orig_tuple : [set(roots_for_rhs0), set(roots_for_rhs1), ...] }
    - local_modular_checks: integer count of attempted modular RHS checks
    """
    # Unpack args (compatible with call-sites that append stats as final arg)
    # Expected tuple:
    # (p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs, _stats)
    try:
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs, _stats = args
    except Exception:
        # Backwards compatibility: if stats not provided
        p, Ep_local, mults_p, vecs_lll_p, vecs_list, rhs_modp_list_local, num_rhs = args
        _stats = None

    result_for_p = {}
    local_modular_checks = 0

    try:
        for idx, v_orig in enumerate(vecs_list):
            v_orig_tuple = tuple(v_orig)

            # zero-vector shortcut
            if all(c == 0 for c in v_orig):
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            # transformed vector at this prime (if not present, skip)
            try:
                v_p_transformed = vecs_lll_p[idx]
            except Exception:
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            # Build Pm safely using mults_p which may be dict or list
            Pm = Ep_local(0)
            for j, coeff in enumerate(v_p_transformed):
                try:
                    mpj = mults_p[j]                       # may raise KeyError or IndexError
                except Exception:
                    mpj = None

                if mpj is None:
                    continue

                # mpj may be dict-like or list-like
                try:
                    key = int(coeff)
                    if hasattr(mpj, 'get'):               # dict-like
                        if key in mpj:
                            Pm += mpj[key]
                    else:                                 # list/tuple-like
                        if 0 <= key < len(mpj):
                            Pm += mpj[key]
                except Exception:
                    # be conservative: skip this coefficient if anything goes wrong
                    continue

            # If the point sum is the identity, no residues for this vector
            if Pm.is_zero():
                result_for_p[v_orig_tuple] = [set() for _ in range(num_rhs)]
                continue

            # For each RHS function, compute roots modulo p (if data exists)
            roots_by_rhs = []
            for i_rhs in range(num_rhs):
                roots_for_rhs = set()
                # rhs_modp_list_local[i_rhs] expected to be a dict mapping p -> rhs_mod_p
                rhs_map = rhs_modp_list_local[i_rhs]
                if p in rhs_map:
                    rhs_p = rhs_map[p]
                    try:
                        # If the denominator is zero mod p, treat as "no numeric root"
                        den = Pm[2]
                        # Attempt to get integer modulo p; some ring elements might not support %
                        try:
                            den_modp = int(den) % int(p)
                        except Exception:
                            # fallback: coerce numerator/denominator via .numerator/.denominator if available
                            try:
                                den_modp = int(den.numerator()) % int(p)
                            except Exception:
                                den_modp = 0  # be conservative

                        if den_modp == 0:
                            # Can't test this RHS modulo p (denominator zero) -> no numeric root recorded
                            # (do NOT insert non-numeric sentinels; just append empty set)
                            pass
                        else:
                            # compute the polynomial numerator for (X - rhs_p) mod p
                            num_modp = (Pm[0] / Pm[2] - rhs_p).numerator()
                            if not num_modp.is_zero():
                                local_modular_checks += 1
                                # compute roots in GF(p)
                                roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                                roots_for_rhs.update(roots)
                    except Exception:
                        # any algebraic failure just yields no roots for this RHS
                        pass

                roots_by_rhs.append(roots_for_rhs)

            # final assignment for this vector
            result_for_p[v_orig_tuple] = roots_by_rhs

    except Exception as e:
        # safe fail: return empty mapping + zero checks
        if DEBUG:
            print(f"[worker fail] p={p}: {e}")
        return p, {}, 0

    return p, result_for_p, local_modular_checks


def generate_biased_prime_subsets_by_coverage(prime_pool, precomputed_residues, vecs,
                                              num_subsets, min_size, max_size, combo_cap,
                                              seed=SEED_INT, force_full_pool=False, debug=DEBUG,
                                              roots_threshold=ROOTS_THRESHOLD):
    """
    Generate diverse prime subsets biased toward high-coverage primes, but skip
    combinatorially-pathological subsets whose estimated Cartesian-product of
    roots would exceed combo_cap.
    """
    # basic assertions
    assert isinstance(prime_pool, (list, tuple))
    assert isinstance(vecs, (list, tuple))
    assert num_subsets >= 1
    assert 1 <= min_size <= max_size
    assert max_size <= len(prime_pool)

    # random must be imported at module scope; fail loudly if not
    if 'random' not in globals():
        raise ImportError("module 'random' must be imported at module scope (add 'import random' at file top)")

    # compute coverage per prime (fraction of vectors with at least one root)
    coverage = compute_prime_coverage(prime_pool, precomputed_residues, vecs, debug=debug)

    # build sampling weights from coverage (floor at 0.05)
    weights = []
    for p in prime_pool:
        w = coverage.get(p, 0.0)
        if w < 0.05:
            w = 0.05
        weights.append(w)

    if debug:
        total_weight = sum(weights)
        avg_weight = total_weight / len(weights) if weights else 0
        print("generate_biased_coverage: avg weight =", avg_weight, "min =", min(weights), "max =", max(weights))

    # Identify top coverage primes (e.g., top 30-40%)
    sorted_primes = sorted(prime_pool, key=lambda p: coverage.get(p, 0.0), reverse=True)
    top_k = max(3, len(sorted_primes) // 3)  # top third of primes
    top_primes = sorted_primes[:top_k]

    if debug:
        print(f"generate_biased_coverage: Top {len(top_primes)} primes by coverage: {top_primes[:10]}")

    subsets = []
    if force_full_pool:
        subsets.append(list(prime_pool))

    # Generate some subsets that include combinations of top primes
    forced_subsets = []
    num_forced = min(20, max(2, num_subsets // 10))  # Increased from 10 to 20
    if len(top_primes) >= 2:
        import random
        for i in range(num_forced):
            # Vary the size: alternate between small (min_size) and larger
            if i % 3 == 0:
                # Small subsets focusing on top primes (1/3 of forced subsets)
                size = min_size
                num_top = min(size, len(top_primes))
                subset = random.sample(top_primes, k=num_top)
            else:
                # Larger subsets mixing top primes with others (2/3 of forced subsets)
                size = random.randint(min_size, min(max_size, len(prime_pool)))
                num_top = min(2, size // 2, len(top_primes))
                subset = random.sample(top_primes, k=num_top)
                remaining_slots = size - len(subset)
                if remaining_slots > 0:
                    other_primes = [p for p in prime_pool if p not in subset]
                    if other_primes:
                        subset.extend(random.sample(other_primes, k=min(remaining_slots, len(other_primes))))
            forced_subsets.append(tuple(sorted(subset)))

    subsets.extend(forced_subsets)

    # Calculate how many more subsets we need
    remaining = num_subsets - len(subsets)
    if remaining <= 0:
        # dedupe and return early
        seen = set()
        unique = []
        for s in subsets:
            t = tuple(sorted(s))
            if t not in seen:
                seen.add(t)
                unique.append(list(t))
        if debug:
            print("generate_biased_coverage: Generated", len(unique), "unique subsets")
        return unique

    # helper to estimate root-product for a candidate subset
    def _estimate_subset_explosion(subset):
        est = 1
        for p in subset:
            mapping = precomputed_residues.get(p)
            if not mapping:
                return 0
            roots_total = 0
            for roots_lists in mapping.values():
                for rl in roots_lists:
                    roots_total += len(rl)
            if roots_total == 0:
                return 0
            if roots_total > roots_threshold:
                est *= roots_total
                if est > combo_cap:
                    return est
        return est

    # produce remaining subsets by weighted sampling, skipping heavy ones
    max_attempts_per_subset = 200
    import random
    for _ in range(remaining):
        attempts = 0
        chosen = None
        while attempts < max_attempts_per_subset:
            attempts += 1
            size = random.randint(min_size, min(max_size, len(prime_pool)))

            subset = None
            try:
                subset = random.sample(prime_pool, k=size)
            except TypeError:
                subset = []
                tries_inner = 0
                while len(subset) < size and tries_inner < size * 20:
                    p = random.choices(prime_pool, weights=weights, k=1)[0]
                    if p not in subset:
                        subset.append(p)
                    tries_inner += 1
                if len(subset) < size:
                    remaining_primes = [p for p in prime_pool if p not in subset]
                    need = min(size - len(subset), len(remaining_primes))
                    if need > 0:
                        subset.extend(random.sample(remaining_primes, k=need))

            subset = tuple(sorted(subset))

            est = _estimate_subset_explosion(subset)
            if est == 0:
                continue
            if est > combo_cap:
                continue

            chosen = list(subset)
            break

        if chosen is None:
            chosen = list(random.sample(prime_pool, k=min(min_size, len(prime_pool))))
            if debug:
                print("generate_biased_coverage: fallback subset used after attempts")

        subsets.append(tuple(sorted(chosen)))

    # deduplicate preserving order
    seen = set()
    unique_subsets = []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            unique_subsets.append(list(s))

    if debug:
        print("generate_biased_coverage: Generated", len(unique_subsets), "unique subsets")
        if unique_subsets:
            sample_show = unique_subsets[:3]
            print("generate_biased_coverage: Sample subsets:", sample_show)

    #print("subsets used:", unique_subsets)
    return unique_subsets

def compute_residues_for_m(m, prime_pool):
    """
    Compute residue fingerprint of rational m = a/b modulo each prime in prime_pool.

    Args:
        m: rational (QQ, Fraction, or (a,b) tuple). If tuple (a,b) is provided, will use that.
        prime_pool: iterable of primes (ints).

    Returns:
        dict mapping p -> int residue in [0,p) OR the string 'DENOM_ZERO' when
        the denominator is 0 mod p (so residue info is unreliable), OR None for skipped primes.
    """
    # Coerce to (a,b)
    try:
        if isinstance(m, tuple) and len(m) == 2:
            a, b = int(m[0]), int(m[1])
        else:
            # accept Sage QQ or Python Fraction
            a = int(QQ(m).numerator())
            b = int(QQ(m).denominator())
    except Exception as e:
        raise ValueError(f"compute_residues_for_m: could not coerce m={m} to rational: {e}")

    res = {}
    for p in prime_pool:
        try:
            p = int(p)
        except Exception:
            res[p] = None
            continue

        if b % p == 0:
            # Denominator zero mod p -> no reliable residue information
            res[p] = 'DENOM_ZERO'
            continue

        try:
            inv_b = pow(b % p, -1, p)
        except ValueError:
            # modular inverse doesn't exist (should be captured by b % p == 0 above)
            res[p] = 'DENOM_ZERO'
            continue

        residue = (a * inv_b) % p
        res[p] = int(residue)

    return res


def compute_residue_coverage_for_m(m_value, precomputed_residues, prime_pool, v_tuple=None):
    """
    Compare a target rational m = a/b (in QQ) against the precomputed residue fingerprints.

    Args:
        m_value: rational number (QQ or coercible to QQ)
        precomputed_residues: dict mapping p -> { v_tuple : [ set(roots_rhs0), set(roots_rhs1), ... ] }
        prime_pool: iterable of primes to check
        v_tuple: optional key to restrict residue comparison to a specific vector tuple

    Returns:
        {
          'm': QQ rational value,
          'matched_primes': [p,...],
          'unseen_primes': [p,...],
          'denom_zero_primes': [p,...],
          'coverage_fraction': float between 0 and 1,
          'per_prime': { p: {'residue': r or None, 'status': 'matched'|'unseen'|'denom_zero'} }
        }
    """
    from sage.all import QQ, Mod

    # Coerce to QQ explicitly
    m_q = QQ(m_value)
    a = ZZ(m_q.numerator())
    b = ZZ(m_q.denominator())

    matched = []
    unseen = []
    denom_zero = []
    per_prime = {}

    for p in prime_pool:
        p = int(p)
        per_prime[p] = {'residue': None, 'status': 'unseen'}

        # If denominator is 0 mod p, cannot test modulo p
        if (b % p) == 0:
            denom_zero.append(p)
            per_prime[p]['status'] = 'denom_zero'
            continue

        # compute residue in GF(p)
        residue = int(Mod(a, p) * Mod(b, p)**(-1))
        per_prime[p]['residue'] = residue

        # check whether residue appears in precomputed_residues[p]
        p_map = precomputed_residues.get(p, {})
        if not p_map:
            unseen.append(p)
            per_prime[p]['status'] = 'unseen'
            continue

        # restrict to one v_tuple or scan all
        found = False
        if v_tuple is not None:
            sets_list = p_map.get(v_tuple, [])
            for s in sets_list:
                if residue in s:
                    found = True
                    break
        else:
            for sets_list in p_map.values():
                for s in sets_list:
                    if residue in s:
                        found = True
                        break
                if found:
                    break

        if found:
            matched.append(p)
            per_prime[p]['status'] = 'matched'
        else:
            unseen.append(p)
            per_prime[p]['status'] = 'unseen'

    usable = max(1, len(prime_pool) - len(denom_zero))
    coverage = float(len(matched)) / float(usable) if usable > 0 else 0.0

    return {
        'm': m_q,
        'matched_primes': matched,
        'unseen_primes': unseen,
        'denom_zero_primes': denom_zero,
        'coverage_fraction': coverage,
        'per_prime': per_prime
    }


def build_targeted_subset(m_value, precomputed_residues, prime_pool,
                          v_tuple=None, min_size=4, max_size=8, prefer_matched_only=False):
    """
    Build a targeted prime subset optimized to reconstruct a specific rational m.

    Args:
        m_value: QQ-coercible rational (target m)
        precomputed_residues: dict mapping p -> { v_tuple : [ set(...) , ... ] }
        prime_pool: iterable/list of primes (ints)
        v_tuple: optional vector-tuple key used in precomputed_residues to restrict which residue-sets to consult
        min_size: minimum subset size to return
        max_size: maximum subset size to return
        prefer_matched_only: if True, only use primes where the residue was observed;
                             otherwise allow padding from high-coverage primes

    Returns:
        list of primes (ints) of length between min_size and max_size (or fewer if pool limited)
    """
    from sage.all import QQ, ZZ

    # Coerce m and extract numerator/denominator
    m_q = QQ(m_value)
    a = ZZ(m_q.numerator()); b = ZZ(m_q.denominator())

    # Step 1: compute coverage report (fast inline to avoid external dependency)
    matched_primes = []
    denom_zero_primes = []
    per_prime_residue = {}

    for p in prime_pool:
        p = int(p)
        if (b % p) == 0:
            denom_zero_primes.append(p)
            per_prime_residue[p] = None
            continue
        residue = int((a % p) * pow(int(b % p), -1, p))
        per_prime_residue[p] = residue

        p_map = precomputed_residues.get(p, {})
        found = False
        if v_tuple is not None:
            sets_list = p_map.get(v_tuple, [])
            for s in sets_list:
                if residue in s:
                    found = True
                    break
        else:
            for sets_list in p_map.values():
                for s in sets_list:
                    if residue in s:
                        found = True
                        break
                if found:
                    break
        if found:
            matched_primes.append(p)

    # Step 2: Build subset preferring matched primes, avoiding denom-zero primes
    avoid = set(denom_zero_primes)
    targeted = []

    # Primary fill: matched primes (sorted by small->large to prefer small primes first)
    # but we can also sort matched primes by "quality": how many sets recorded at that prime (more is better)
    def prime_quality(p):
        # number of vector-keys present for p (proxy for how often p was used)
        return len(precomputed_residues.get(p, {})) if precomputed_residues.get(p, {}) else 0

    matched_sorted = sorted(matched_primes, key=lambda q: (-prime_quality(q), q))
    for p in matched_sorted:
        if p in avoid:
            continue
        targeted.append(p)
        if len(targeted) >= min_size:
            break

    # If prefer_matched_only and we have enough matched primes, truncate there
    if prefer_matched_only:
        # if we have fewer than min_size matched primes, we still fall back below
        if len(targeted) >= min_size:
            # optionally shrink to min_size exactly
            return targeted[:max_size]

    # Step 3: Pad with high-quality primes (not in avoid, not already chosen)
    if len(targeted) < min_size:
        # build a ranked list of candidate primes (exclude denom-zero and already chosen)
        candidates = [p for p in prime_pool if p not in avoid and p not in targeted]
        # sort by quality (more precomputed residue-keys first), tie-break by prime size (smaller primes first)
        candidates_sorted = sorted(candidates, key=lambda q: (-prime_quality(q), q))
        for p in candidates_sorted:
            targeted.append(p)
            if len(targeted) >= min_size:
                break

    # Step 4: Optionally add additional primes up to max_size to increase CRT modulus
    if len(targeted) < max_size:
        # continue with same candidate list
        for p in (candidates_sorted if 'candidates_sorted' in locals() else []):
            if len(targeted) >= max_size:
                break
            if p not in targeted:
                targeted.append(p)

    # Final safety: if targeted is still empty (exotic), just return small slice of prime_pool avoiding denom-zero
    if not targeted:
        fallback = [p for p in prime_pool if p not in avoid][:min_size]
        return fallback

    return targeted[:max_size]


def targeted_recovery_search(cd, current_sections, near_miss_candidates,
                              prime_pool, precomputed_residues,
                              r_m, shift, rationality_test_func,
                              tmax=TMAX, debug=True):
    """
    Run a focused CRT search on detected near-miss candidates.
    """

    import itertools
    from operator import mul
    from functools import reduce
    from sage.all import QQ

    newly_found = set()

    for i, candidate in enumerate(near_miss_candidates):
        if debug:
            print(f"\n[recovery] Targeting near-miss candidate {i+1}/{len(near_miss_candidates)}")
            print(f"  Vector: {candidate['v_tuple']}")
            print(f"  Coverage: {candidate['coverage_ratio']:.1%} ({candidate['num_primes']} primes)")
            print(f"  Potential m residues: {candidate['num_m_residues']}")

        # --- NEW: compute coverage report for debug/diagnostics ---
        cov_report = compute_residue_coverage_for_m(
            candidate,
            precomputed_residues,
            prime_pool,
            v_tuple=candidate.get('v_tuple', None)
        )

        targeted_subset = build_targeted_subset(
            candidate,
            precomputed_residues,
            prime_pool,
            v_tuple=candidate.get('v_tuple', None),
            min_size=4,
            max_size=8,
            prefer_matched_only=False
        )

        if debug:
            print(f"[targeted] m={candidate.get('m_pair')} -> subset={targeted_subset} "
                  f"matched_count={len([p for p in targeted_subset if p in cov_report.get('matched_primes', [])])}")

        if debug:
            print(f"  Using targeted subset: {targeted_subset}")

        v_tuple = candidate['v_tuple']
        residue_map = candidate['residue_map']

        # Filter residue_map to only include primes in our targeted subset
        filtered_residue_map = {p: residue_map[p] for p in targeted_subset if p in residue_map}

        if not filtered_residue_map:
            if debug:
                print("  No residues in targeted subset; skipping.")
            continue

        # Generate all CRT combinations from the targeted subset
        primes_for_crt = list(filtered_residue_map.keys())
        residue_lists = [filtered_residue_map[p] for p in primes_for_crt]

        for combo in itertools.product(*residue_lists):
            M = reduce(mul, primes_for_crt, 1)
            if M > MAX_MODULUS:
                continue

            m0 = crt_cached(combo, tuple(primes_for_crt))

            try:
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
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
        print(f"[compute_prime_coverage] Prime coverage (top 20):")
        for p, cov in sorted_by_cov[:20]:
            print(f"  p={p}: coverage={cov:.1%}")
    
    return coverage


# Add to search_lll.py

def diagnose_missed_point(target_x, r_m_callable, shift, precomputed_residues, prime_pool, vecs, tmax=TMAX, debug=True):
    """
    Diagnose why a specific x-value wasn't found by the CRT search.
    
    Check if target_x is theoretically findable via CRT + rational reconstruction
    for any vector and prime subset combination.
    
    Args:
        target_x: The x-coordinate we're looking for (QQ or coercible)
        r_m_callable: Function to compute x from m (typically r_m from tower)
        shift: The shift applied to x-coordinates
        precomputed_residues: {p: {v_tuple: [roots_per_rhs]}} from workers
        prime_pool: List of primes used in search
        vecs: List of search vectors
        tmax: Maximum |t| to check in m = m0 + t*M
        debug: Print diagnostic info
    
    Returns:
        dict with diagnostic information
    """
    from sage.all import QQ, ZZ
    from itertools import combinations
    
    # Step 1: Solve for target m-value
    # x = r_m(m) - shift, so m = r_m^(-1)(x + shift)
    # For r_m(m) = -m - x1, we have: x = -m - x1 - shift
    # So: m = -x - x1 - shift = -(x + shift) - x1
    # But we need to be more careful. Let's solve symbolically.
    
    target_x_q = QQ(target_x)
    
    # For the linear case r_m(m) = -m - const, solve x = -m - const - shift
    # => m = -x - shift - const
    # We can get const by evaluating r_m at m=0
    try:
        const_term = r_m_callable(m=QQ(0))
        target_m = -(target_x_q + shift + const_term)
    except Exception as e:
        if debug:
            print(f"[diagnose] Failed to compute target_m: {e}")
        return {'error': str(e)}
    
    if debug:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTIC: Checking if x = {target_x_q} is findable")
        print(f"{'='*70}")
        print(f"Target m-value: {target_m}")
        print(f"  (from x = r_m(m) - shift with shift={shift})")
    
    # Step 2: Express target_m = a/b and compute residues mod each prime
    a = ZZ(target_m.numerator())
    b = ZZ(target_m.denominator())
    
    residues_by_prime = {}
    matched_vectors_by_prime = {}  # {p: {v_tuple: [rhs_indices where m_p appears]}}
    
    if debug:
        print(f"\nComputing residues for m = {a}/{b} mod each prime...")
    
    for p in prime_pool:
        p_int = int(p)
        
        # Check if denominator is zero mod p
        if (b % p_int) == 0:
            residues_by_prime[p_int] = 'DENOM_ZERO'
            if debug:
                print(f"  p={p_int}: denominator zero mod p (skipping)")
            continue
        
        # Compute m_p = (a * b^(-1)) mod p
        try:
            b_inv = pow(int(b % p_int), -1, p_int)
            m_p = (int(a % p_int) * b_inv) % p_int
            residues_by_prime[p_int] = m_p
        except ValueError:
            residues_by_prime[p_int] = 'INV_FAIL'
            if debug:
                print(f"  p={p_int}: inverse computation failed")
            continue
        
        # Step 3: Check which vectors have this residue in precomputed data
        p_data = precomputed_residues.get(p_int, {})
        matched_vectors_by_prime[p_int] = {}
        
        for v in vecs:
            v_tuple = tuple(v)
            roots_list = p_data.get(v_tuple, [])
            
            if not roots_list:
                continue
            
            # roots_list is [roots_rhs0, roots_rhs1, ...]
            matching_rhs = []
            for rhs_idx, roots_set in enumerate(roots_list):
                if m_p in roots_set:
                    matching_rhs.append(rhs_idx)
            
            if matching_rhs:
                matched_vectors_by_prime[p_int][v_tuple] = matching_rhs
    
    # Step 4: Analyze coverage per vector
    if debug:
        print(f"\n{'='*70}")
        print("COVERAGE ANALYSIS BY VECTOR")
        print(f"{'='*70}")
    
    vector_coverage = {}
    for v in vecs:
        v_tuple = tuple(v)
        matched_primes = []
        
        for p_int in prime_pool:
            if p_int in matched_vectors_by_prime:
                if v_tuple in matched_vectors_by_prime[p_int]:
                    matched_primes.append(p_int)
        
        coverage_frac = len(matched_primes) / float(len(prime_pool)) if prime_pool else 0.0
        vector_coverage[v_tuple] = {
            'matched_primes': matched_primes,
            'coverage_fraction': coverage_frac,
            'num_matched': len(matched_primes)
        }
        
        if debug and coverage_frac > 0.0:
            print(f"\nVector {v_tuple[:3]}... :")
            print(f"  Matched primes: {matched_primes[:10]}{'...' if len(matched_primes) > 10 else ''}")
            print(f"  Coverage: {coverage_frac:.1%} ({len(matched_primes)}/{len(prime_pool)} primes)")
    
    # Step 5: Try CRT + rational reconstruction for promising vectors
    if debug:
        print(f"\n{'='*70}")
        print("TESTING CRT + RATIONAL RECONSTRUCTION")
        print(f"{'='*70}")
    
    viable_reconstructions = []
    
    # Sort vectors by coverage (best first)
    sorted_vectors = sorted(
        vector_coverage.items(),
        key=lambda x: x[1]['coverage_fraction'],
        reverse=True
    )
    
    for v_tuple, cov_info in sorted_vectors:
        if cov_info['num_matched'] < MIN_PRIME_SUBSET_SIZE:
            continue  # Not enough primes for a viable subset
        
        matched_primes = cov_info['matched_primes']
        
        if debug:
            print(f"\nTesting vector {v_tuple[:3]}... ({cov_info['num_matched']} matched primes)")
        
        # Try subsets of various sizes
        found_for_this_vector = False
        for subset_size in range(MIN_PRIME_SUBSET_SIZE, 
                                 min(MIN_MAX_PRIME_SUBSET_SIZE, len(matched_primes)) + 1):
            
            # Heuristic: try up to 100 random subsets of this size
            import random
            max_subsets_to_try = min(100, len(list(combinations(matched_primes, subset_size))))
            
            subsets_to_try = random.sample(
                list(combinations(matched_primes, subset_size)),
                min(max_subsets_to_try, len(list(combinations(matched_primes, subset_size))))
            )
            
            for subset in subsets_to_try:
                subset_list = list(subset)
                
                # Get residues for this subset
                residues = tuple(residues_by_prime[p] for p in subset_list)
                
                # CRT lift
                try:
                    m0 = crt_cached(residues, tuple(subset_list))
                    M = 1
                    for p in subset_list:
                        M *= int(p)
                except Exception:
                    continue
                
                # Check if target_m = m0 + t*M for some small |t|
                # target_m = a/b, so we need: a/b = m0 + t*M
                # => a = b*(m0 + t*M) = b*m0 + b*t*M
                # => t = (a - b*m0) / (b*M)
                
                numerator = a - b * m0
                denominator = b * M
                
                if numerator % denominator == 0:
                    t = numerator // denominator
                    
                    if abs(t) <= tmax:
                        m_reconstructed = QQ(m0 + t * M)
                        
                        if m_reconstructed == target_m:
                            viable_reconstructions.append({
                                'vector': v_tuple,
                                'subset': subset_list,
                                'subset_size': len(subset_list),
                                'm0': m0,
                                'M': M,
                                't': t,
                                'm_reconstructed': m_reconstructed
                            })
                            
                            if debug:
                                print(f"  ✓ FOUND via subset {subset_list}")
                                print(f"    m0={m0}, M={M}, t={t}")
                                print(f"    m = {m0} + {t}*{M} = {m_reconstructed}")
                            
                            found_for_this_vector = True
                            break  # Found one, that's enough for this subset size
                
                # Also try rational reconstruction
                try:
                    a_recon, b_recon = rational_reconstruct(m0 % M, M)
                    m_recon = QQ(a_recon) / QQ(b_recon)
                    
                    if m_recon == target_m:
                        viable_reconstructions.append({
                            'vector': v_tuple,
                            'subset': subset_list,
                            'subset_size': len(subset_list),
                            'm0': m0,
                            'M': M,
                            't': 'rational_recon',
                            'm_reconstructed': m_recon
                        })
                        
                        if debug:
                            print(f"  ✓ FOUND via rational reconstruction on subset {subset_list}")
                            print(f"    m0={m0}, M={M}")
                            print(f"    Reconstructed: {a_recon}/{b_recon} = {m_recon}")
                        
                        found_for_this_vector = True
                        break
                
                except RationalReconstructionError:
                    pass
            
            if found_for_this_vector:
                break  # Found it for this vector, move to next vector
    
    # Step 6: Summary
    if debug:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Target: x = {target_x_q}, m = {target_m}")
        print(f"Total vectors: {len(vecs)}")
        print(f"Vectors with any coverage: {sum(1 for v in vector_coverage.values() if v['num_matched'] > 0)}")
        print(f"Viable reconstructions found: {len(viable_reconstructions)}")
        
        if viable_reconstructions:
            print(f"\n✓ POINT IS FINDABLE")
            print(f"\nExample reconstructions:")
            for i, recon in enumerate(viable_reconstructions[:3]):
                print(f"\n  [{i+1}] Vector: {recon['vector'][:3]}...")
                print(f"      Subset size: {recon['subset_size']}")
                print(f"      Primes: {recon['subset']}")
                print(f"      t: {recon['t']}")
        else:
            print(f"\n✗ POINT NOT FINDABLE with current search parameters")
    
    return {
        'target_x': target_x_q,
        'target_m': target_m,
        'residues_by_prime': residues_by_prime,
        'vector_coverage': vector_coverage,
        'viable_reconstructions': viable_reconstructions,
        'is_findable': len(viable_reconstructions) > 0
    }


# In search_lll.py, replace the existing _process_prime_subset_precomputed function with this:

def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, tmax, combo_cap, precomputed_residues, prime_pool, num_rhs_fns):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.
    
    *** MODIFIED to decouple t-search filter from rational reconstruction ***
    """
    if not p_subset:
        return set()

    found_candidates_for_subset = set()
    stats_counter = Counter()
    tested_crt_classes = set()

    # these are now skipped!  this shouldn't print anymore!
    if len(p_subset) > 1 and all(p in precomputed_residues for p in p_subset):
        est = 1
        for p in p_subset:
            vks = precomputed_residues[p]
            for roots_list in vks.values():
                # roots_list is a list of sets per RHS function
                if any(len(roots) > ROOTS_THRESHOLD for roots in roots_list):
                    est *= sum(len(roots) for roots in roots_list)
        if est > combo_cap and DEBUG:
            print("[heavy subset]", p_subset, "estimated combos:", est)

    num_extra_primes = 4
    offset = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][offset:num_extra_primes+offset]

    for v_orig in vecs:
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        for rhs_idx in range(num_rhs_fns):
            
            residue_map_for_crt = {}
            for p in p_subset:
                # precomputed_residues[p][v_tuple] is now a list of sets
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                    residue_map_for_crt[p] = roots_for_this_rhs[rhs_idx]

            primes_for_crt = list(residue_map_for_crt.keys())
            if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
                continue

            residue_map_for_filter = {}
            for p in extra_primes_for_filtering:
                roots_for_this_rhs = precomputed_residues.get(p, {}).get(v_orig_tuple, [])
                if rhs_idx < len(roots_for_this_rhs) and roots_for_this_rhs[rhs_idx]:
                     residue_map_for_filter[p] = roots_for_this_rhs[rhs_idx]

            lists = [residue_map_for_crt[p] for p in primes_for_crt]
            
            for combo in itertools.product(*lists):
                stats_counter['crt_lift_attempts'] += 1
                M = 1
                for p in primes_for_crt:
                    M *= int(p)

                if M > MAX_MODULUS:
                    continue

                m0 = crt_cached(combo, tuple(primes_for_crt))
                tested_crt_classes.add((int(m0) % int(M), int(M)))

                # --- START MODIFICATION ---
                # Path 1: t-search (guarded by the filter)
                if candidate_passes_extra_primes(m0, M, residue_map_for_filter, extra_primes_for_filtering, tmax):
                    try:
                        best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
                    except TypeError:
                        best_ms = [(0, QQ(m0 + t * M), 0, 0.0) for t in (-1, 0, 1)] # t, m, x, score

                    for _, m_cand, _, _ in best_ms:
                        found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                # Path 2: Rational Reconstruction (run *unconditionally*)
                stats_counter['rational_recon_attempts_worker'] += 1
                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
                    stats_counter['rational_recon_success_worker'] += 1
                except RationalReconstructionError:
                    stats_counter['rational_recon_failure_worker'] += 1
                    # Do not raise here, just fail to add
                    pass
                # --- END MODIFICATION ---

    return found_candidates_for_subset, stats_counter, tested_crt_classes
