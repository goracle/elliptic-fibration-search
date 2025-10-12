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
from math import floor, sqrt, gcd, ceil
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
                best_ms = minimize_archimedean_t(int(m0), int(M), r_m, shift, max_abs_t)
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


#EXTRA_PRIMES = sorted(list(PRIME_POOL[:-5])) # small primes
def candidate_passes_extra_primes(m0, M, extra_residue_map, small_primes, max_abs_t, verbose=False):
    """
    CORRECTED VERSION: The issue was misunderstanding the problem.
    
    For each prime q, extra_residue_map[q] is a SET of acceptable residues.
    We need to find ONE t such that for ALL primes q, the value (m0 + t*M) mod q 
    is in the acceptable set for that prime.
    
    This is NOT a CRT problem! We need to find the intersection of allowed t values
    across all primes.
    """
    if verbose:
        print(f"=== FIXED Debug: m0={m0}, M={M}, T={max_abs_t} ===")
        print(f"small_primes: {list(small_primes)}")
        print(f"extra_residue_map: {extra_residue_map}")
    
    m0 = int(m0)
    M = int(M)
    
    # For each prime, find ALL t values (mod q) that work
    t_constraints = []  # List of (modulus, allowed_t_set) pairs
    
    for q in small_primes:
        if q not in extra_residue_map:
            if verbose:
                print(f"Prime {q} not in extra_residue_map, skipping")
            continue
            
        Rq = extra_residue_map[q]
        if verbose:
            print(f"Prime {q}: allowed residues for m = {Rq}")
            
        if not Rq:
            if verbose:
                print(f"Prime {q}: no allowed residues, returning False")
            return False

        m0q = m0 % q
        Mq = M % q
        
        if verbose:
            print(f"Prime {q}: m0 ≡ {m0q} (mod {q}), M ≡ {Mq} (mod {q})")

        if Mq == 0:
            # t doesn't change m mod q; requirement is m0q in Rq
            if m0q not in Rq:
                if verbose:
                    print(f"Prime {q}: M≡0 but m0≡{m0q} ∉ {Rq}, returning False")
                return False
            if verbose:
                print(f"Prime {q}: M≡0 and m0≡{m0q} ∈ {Rq}, any t works for this prime")
            # This prime puts no constraint on t
            continue

        # Mq is invertible mod q
        try:
            inv_Mq = pow(Mq, -1, q)
        except:
            if verbose:
                print(f"Prime {q}: M≡{Mq} not invertible mod {q}, returning False")
            return False
            
        if verbose:
            print(f"Prime {q}: M≡{Mq}, inv(M)≡{inv_Mq} (mod {q})")

        # Find all t values mod q that make m = m0 + t*M have an acceptable residue
        allowed_t_mod_q = set()
        for r in Rq:
            # We want m0 + t*M ≡ r (mod q)
            # So t*M ≡ r - m0 (mod q)  
            # So t ≡ (r - m0) * inv(M) (mod q)
            t0 = ((r - m0q) * inv_Mq) % q
            allowed_t_mod_q.add(int(t0))
            if verbose:
                print(f"  For m≡{r} (mod {q}): need t≡{t0} (mod {q})")

        if not allowed_t_mod_q:
            if verbose:
                print(f"Prime {q}: no valid t values, returning False")
            return False

        t_constraints.append((q, allowed_t_mod_q))
        if verbose:
            print(f"Prime {q}: allowed t values ≡ {sorted(allowed_t_mod_q)} (mod {q})")

    if verbose:
        print(f"Constraints: {t_constraints}")

    # If no constraints, any t works
    if not t_constraints:
        if verbose:
            print("No constraints on t, returning True")
        return True

    # Now we need to find a t that satisfies ALL constraints simultaneously
    # This means: t ≡ a1 (mod q1) AND t ≡ a2 (mod q2) AND ... for SOME choice of a1, a2, ...
    
    # We'll use CRT to combine choices, but now we're looking for ANY valid combination
    moduli = [q for q, _ in t_constraints]
    t_sets = [allowed_set for _, allowed_set in t_constraints]
    
    # Calculate combinations
    count_combinations = 1
    for s in t_sets:
        count_combinations *= len(s)
    
    if verbose:
        print(f"Total combinations to check: {count_combinations}")
    
    # If too many combinations, we could use a more sophisticated approach,
    # but for now let's be conservative
    max_combinations = 50000
    if count_combinations > max_combinations:
        if verbose:
            print(f"Too many combinations ({count_combinations} > {max_combinations})")
        # Could implement more sophisticated algorithm here
        # For now, fall back to checking individual primes
        return False

    # Check all combinations of t residue choices
    from itertools import product
    
    Q = 1
    for q in moduli:
        Q *= q
    
    for i, combo in enumerate(product(*t_sets)):
        if verbose and i < 10:
            print(f"Checking combination {i}: {combo}")
        
        try:
            # Use CRT to find t ≡ combo[0] (mod moduli[0]), t ≡ combo[1] (mod moduli[1]), etc.
            t_mod_Q = crt(list(combo), list(moduli))
        except Exception as e:
            if verbose and i < 10:
                print(f"  CRT failed: {e}")
            continue
        
        if verbose and i < 10:
            print(f"  CRT gives t ≡ {t_mod_Q} (mod {Q})")

        # Find the representative of t ≡ t_mod_Q (mod Q) with smallest |t|
        # t = t_mod_Q + k*Q, we want to minimize |t|
        
        best_t = None
        best_abs_t = float('inf')
        
        # The optimal k is approximately -t_mod_Q/Q
        if Q > 0:
            k_float = -t_mod_Q / Q
            k_candidates = [int(k_float) - 1, int(k_float), int(k_float) + 1]
        else:
            k_candidates = [0]
        
        # Also try k=0 case
        k_candidates.append(0)
        
        for k in set(k_candidates):  # remove duplicates
            t_candidate = t_mod_Q + k * Q
            if abs(t_candidate) < best_abs_t:
                best_abs_t = abs(t_candidate)
                best_t = t_candidate
        
        if verbose and i < 10:
            print(f"  Best representative: t = {best_t}, |t| = {best_abs_t}")
        
        if best_abs_t <= max_abs_t:
            if verbose:
                print(f"*** SOLUTION FOUND: t = {best_t} ***")
                # Verify the solution
                for q, allowed_set in t_constraints:
                    m_val = (m0 + best_t * M) % q
                    expected_t_mod_q = best_t % q  
                    print(f"  Verification: t≡{expected_t_mod_q} (mod {q}), m≡{m_val} (mod {q}), allowed: {extra_residue_map[q]}")
            return True
    
    if verbose:
        print(f"Checked {count_combinations} combinations, no solution with |t| ≤ {max_abs_t}")
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
            Ep_local = EllipticCurve(Fp_m, [0, 0, 0, a4_modp, a6_modp])

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
                    raise

            # Build required multiplier indices (bounded)
            raw_required_ks = set()
            for v_trans in vecs_transformed_for_p:
                for k in v_trans:
                    raw_required_ks.add(int(k))
            required_ks = {k for k in raw_required_ks if abs(k) <= MAX_K_ABS}
            if not required_ks:
                required_ks = set(range(-3, 4))

            # compute multiples for this prime (exact arithmetic)
            mults = [{} for _ in range(r)]
            for i_sec in range(r):
                Pi = new_basis[i_sec]
                for k in required_ks:
                    try:
                        mults[i_sec][k] = k * Pi
                    except Exception:
                        # skip multipliers that fail for this prime
                        continue

            # Success for this prime -> publish all data
            Ep_dict[p] = Ep_local
            for i, rhs_p_val in rhs_modp_for_p.items():
                rhs_modp_list[i][p] = rhs_p_val
            multiplies_lll[p] = mults
            vecs_lll[p] = vecs_transformed_for_p

        except (ZeroDivisionError, TypeError, ValueError, ArithmeticError) as e:
            if p != 2:
                print(f"Skipping prime {p} due to error during preparation: {e}")
                raise
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


# Place this new worker function before the main search_lattice_modp_lll_subsets function.
def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, max_abs_t, precomputed_residues, prime_pool):
    """
    Worker function to find m-candidates for a single subset of primes using precomputed residues.
    Returns a set of (m_candidate, originating_vector) tuples.
    """
    if not p_subset:
        return set()

    found_candidates_for_subset = set()

    # Process each search vector for this subset
    for v_orig in vecs:
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        # Build the residue map for this vector from the precomputed data
        residue_map = {}
        for p in p_subset:
            # Look up the precomputed roots for this prime and vector
            roots_for_p = precomputed_residues.get(p, {}).get(v_orig_tuple)
            if roots_for_p: # Ensure the set is not None or empty
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
            if not candidate_passes_extra_primes(m0, M, residue_map, prime_pool[:floor(len(prime_pool)/2)], max_abs_t):
                #    print("HEEERRE")
                #    raise ValueError
                continue  # reject this CRT combo cheaply, exact check

            try:
                best_ms = minimize_archimedean_t(int(m0), int(M), r_m, shift, max_abs_t)
            except TypeError:
                best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]
                raise

            for m_cand, _score in best_ms:
                found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

            try:
                a, b = rational_reconstruct(m0 % M, M)
                found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            except RationalReconstructionError:
                raise

    return found_candidates_for_subset



@PROFILE
def search_lattice_modp_lll_subsets(cd, current_sections, prime_pool, vecs, rhs_list, r_m,
                                    shift, all_found_x, prime_subsets, rationality_test_func, max_abs_t):
    """
    Search for rational points using LLL-reduced bases across prime subsets in parallel.
    This version pre-computes modular residues and serially checks rational candidates.
    """
    # 1. Prepare modular data for all primes (curve, RHS functions, basis points)
    print("--- Preparing modular data for LLL search ---")
    Ep_dict, rhs_modp_list, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, search_primes=prime_pool
    )

    if not Ep_dict:
        print("No valid primes found for modular search. Aborting.")
        return set(), []

    # 2. Pre-compute all modular residues for each (prime, vector) pair
    print(f"--- Pre-computing residues for {len(Ep_dict)} primes and {len(vecs)} vectors ---")
    precomputed_residues = {p: {} for p in Ep_dict}

    for p in tqdm(Ep_dict.keys(), desc="Pre-computing residues"):
        Ep = Ep_dict[p]
        mults_p = mult_lll[p]
        vecs_lll_p = vecs_lll[p]

        for idx, v_orig in enumerate(vecs):
            v_orig_tuple = tuple(v_orig)
            if all(c == 0 for c in v_orig):
                precomputed_residues[p][v_orig_tuple] = set()
                continue

            # Compute the linear combination of basis points mod p
            try:
                v_p_transformed = vecs_lll_p[idx]
                Pm = Ep(0)
                for j, coeff in enumerate(v_p_transformed):
                    if int(coeff) in mults_p[j]:
                        Pm += mults_p[j][int(coeff)]
            except (KeyError, IndexError):
                continue # Basis or vector data missing for this prime, skip

            if Pm.is_zero():
                precomputed_residues[p][v_orig_tuple] = set()
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

            precomputed_residues[p][v_orig_tuple] = roots_for_p

    # 3. Set up the parallel worker with precomputed data
    worker_func = partial(
        _process_prime_subset_precomputed,
        vecs=vecs,
        r_m=r_m,
        shift=shift,
        max_abs_t=max_abs_t,
        precomputed_residues=precomputed_residues,
        prime_pool=prime_pool
    )

    # 4. Run the parallel search over prime subsets
    overall_found_candidates = set()
    with multiprocessing.Pool() as pool:
        pbar = tqdm(
            pool.imap(worker_func, prime_subsets),
            total=len(prime_subsets),
            desc="Searching Prime Subsets"
        )
        for subset_results in pbar:
            overall_found_candidates.update(subset_results)

    # 5. Test candidates for rationality and construct new sections SERIALLY
    if not overall_found_candidates:
        return set(), []

    sample_pts = []
    new_sections_raw = []
    processed_m_vals = {} # Use a dictionary to process each m_val only once

    #sorted_candidates = sorted(list(overall_found_candidates), key=lambda x: (x[0], x[1]))
    #sorted_candidates = list(overall_found_candidates)

    print(f"\nFound {len(overall_found_candidates)} potential (m, vector) pairs. Testing for rationality...")

    const = DATA_PTS_GENUS2[0]-shift
    #xvals = [(r_m(m=mval)-shift, v_tuple, rationality_test_func) for mval, v_tuple in overall_found_candidates]
    xvals = [QQ(-1)*mval+const for mval, _ in overall_found_candidates]

    #with Pool(processes=cpu_count()) as pool:
    with multiprocessing.Pool() as pool:
        results = pool.map(rationality_test_func, xvals) # use a larger chunksize for 1000 items
        sample_pts = [(xvals[i], yval) for i, yval in enumerate(results) if yval is not None]

    # Call parallel search
    for xval in xvals:
        break # uncomment if using the parallel version

        try:
            #x_val = r_m(m=m_val) - shift
            yval = rationality_test_func(xval)

            if yval is not None:
                #v = vector(QQ, v_tuple)
                sample_pts.append((xval, yval))
                #processed_m_vals[m_val] = v

                continue

                # Construct the potential new section from the successful vector
                #if any(c != 0 for c in v): # Ensure it's not the zero section
                #    new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections)))
                #    new_sections_raw.append(new_sec)

        except (TypeError, ZeroDivisionError, ArithmeticError):
            continue

    new_xs = {pt[0] for pt in sample_pts}
    # dedupe sections
    new_sections = list({s: None for s in new_sections_raw}.keys())



    return new_xs, new_sections


def candidate_passes_extra_primes(m0, M, extra_residue_map, small_primes, max_abs_t, verbose=False):
    """
    Checks if there exists an integer t with |t| <= max_abs_t such that for every prime q,
    (m0 + t*M) mod q is in the set of allowed residues extra_residue_map[q].

    This is an efficient, iterative implementation that avoids combinatorial explosion.
    """
    m0 = int(m0)
    M = int(M)

    # 1. Pre-calculate the allowed 't' residues for each prime
    t_constraints = {}
    for q in small_primes:
        if q not in extra_residue_map or not extra_residue_map[q]:
            # If a prime has no allowed residues for m, no solution is possible.
            return False

        Rq = extra_residue_map[q]
        m0q, Mq = m0 % q, M % q

        if Mq == 0:
            # If M is 0 mod q, t has no effect. The condition is just m0q in Rq.
            if m0q not in Rq:
                return False
            # This prime imposes no constraint on t, so we can ignore it for filtering.
            continue

        try:
            inv_Mq = pow(Mq, -1, q)
        except ValueError:
            # Should not happen for a prime q unless Mq is 0, which is handled above.
            return False

        # We need m0 + t*M ≡ r (mod q) => t ≡ (r - m0) * inv(M) (mod q)
        allowed_t_mod_q = {((r - m0q) * inv_Mq) % q for r in Rq}
        t_constraints[q] = allowed_t_mod_q

    if not t_constraints:
        # No primes imposed any constraints on t, so any t works.
        # Since we are checking for |t|<=T, and t=0 is a candidate, this passes.
        return True

    # 2. Start with the most restrictive prime (fewest allowed t residues)
    # This minimizes the number of t values we need to test.
    sorted_primes = sorted(t_constraints.keys(), key=lambda q: len(t_constraints[q]))
    start_q = sorted_primes[0]
    other_primes = sorted_primes[1:]

    # 3. Iterate through candidate t values generated from the most restrictive prime
    for t_residue in t_constraints[start_q]:
        # Test t values in the arithmetic progression: t_residue, t_residue + q, t_residue - q, ...
        # up to the bound max_abs_t.
        for k in range(max_abs_t // start_q + 2):
            for t_candidate in {t_residue + k * start_q, t_residue - k * start_q}:
                if abs(t_candidate) > max_abs_t:
                    continue

                # 4. Check this t_candidate against all other prime constraints
                is_valid = True
                for other_q in other_primes:
                    if (t_candidate % other_q) not in t_constraints[other_q]:
                        is_valid = False
                        break # Fails condition for other_q, try next t

                if is_valid:
                    # Found a t that satisfies all constraints
                    if verbose:
                        print(f"Found valid t = {t_candidate} for m0={m0}, M={M}")
                    return True

    # If we exhaust all possibilities without finding a valid t
    return False

# Place this corrected version in search_lll.py, replacing both existing definitions.

def candidate_passes_extra_primes(m0, M, full_residue_map, extra_primes, max_abs_t, verbose=False):
    """
    Checks if there exists an integer t with |t| <= max_abs_t such that for every extra prime q,
    the value (m0 + t*M) mod q is in the set of allowed m-residues for that prime.

    Args:
        m0 (int): The integer residue from CRT.
        M (int): The modulus from CRT.
        full_residue_map (dict): A map from {prime: {vector: {residues...}}}.
                                 This should be the complete precomputed residue map.
        extra_primes (list): A list of primes to use for this check. These should NOT
                             be the primes used to construct m0 and M.
        max_abs_t (int): The maximum absolute value of t to search.
        verbose (bool): If True, print detailed debugging information.
    """
    m0 = int(m0)
    M = int(M)

    # 1. For each extra prime, determine the constraints it imposes on t.
    t_constraints = {}
    for q in extra_primes:
        # We need to know the allowed residues for 'm' for the specific vector that produced m0.
        # Since this check is generic, we find the union of all allowed residues for prime q.
        # This is a slight simplification but is much faster and still a very effective filter.
        allowed_m_residues = full_residue_map.get(q, {}).get(v_orig_tuple) # ASSUMES v_orig_tuple is in scope
        
        if not allowed_m_residues:
            # If this extra prime forbids all m-values for this vector, then no solution is possible.
            if verbose: print(f"Filter fail: Prime {q} has no allowed m-residues for this vector.")
            return False

        m0q, Mq = m0 % q, M % q

        if Mq == 0:
            # If M is 0 mod q, t has no effect. The condition is just m0q being in the allowed set.
            if m0q not in allowed_m_residues:
                if verbose: print(f"Filter fail: M=0 mod {q} and m0={m0q} is not in allowed set {allowed_m_residues}.")
                return False
            # This prime imposes no constraint on t, so we skip it.
            continue

        try:
            inv_Mq = pow(Mq, -1, q)
        except ValueError:
            # Should not happen for a prime q if Mq != 0.
            return False

        # We need m0 + t*M ≡ r (mod q)  =>  t ≡ (r - m0) * inv(M) (mod q)
        allowed_t_mod_q = {((r - m0q) * inv_Mq) % q for r in allowed_m_residues}
        if not allowed_t_mod_q:
            return False # Should not happen if allowed_m_residues is non-empty
            
        t_constraints[q] = allowed_t_mod_q

    if not t_constraints:
        # No extra primes imposed any constraints on t, so the candidate passes.
        return True

    # 2. Sort primes by how restrictive they are to check the hardest constraints first.
    sorted_primes = sorted(t_constraints.keys(), key=lambda q: len(t_constraints[q]))
    start_q = sorted_primes[0]
    other_primes = sorted_primes[1:]

    # 3. Iterate through candidate t values generated from the most restrictive prime.
    for t_residue in t_constraints[start_q]:
        # Test t values in the arithmetic progression: t_residue, t_residue ± q, t_residue ± 2q, ...
        # up to the bound max_abs_t. Check t=0 and expand outwards.
        for k in range(max_abs_t // start_q + 2):
            for sign in ([1, -1] if k > 0 else [1]):
                t_candidate = t_residue + (sign * k * start_q)

                if abs(t_candidate) > max_abs_t:
                    continue

                # 4. Check this t_candidate against all other prime constraints.
                is_valid = True
                for other_q in other_primes:
                    if (t_candidate % other_q) not in t_constraints[other_q]:
                        is_valid = False
                        break  # Fails constraint for other_q, try next t

                if is_valid:
                    # Found a 't' that satisfies all constraints.
                    if verbose: print(f"Filter pass: Found valid t={t_candidate} for m0={m0}, M={M}")
                    return True

    # If we exhaust all possibilities without finding a valid t, the candidate fails.
    if verbose: print(f"Filter fail: No t in [-{max_abs_t}, {max_abs_t}] found for m0={m0}, M={M}")
    return False


# In search_lll.py, modify this worker function.

def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, max_abs_t, precomputed_residues, prime_pool):
    """
    Worker function to find m-candidates for a single subset of primes using precomputed residues.
    Returns a set of (m_candidate, originating_vector) tuples.
    """
    if not p_subset:
        return set()

    # Define a small set of extra primes for cheap filtering.
    # Choose the first few primes from the main pool that are NOT in our current CRT subset.
    num_extra_primes = 2
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][:num_extra_primes]

    found_candidates_for_subset = set()

    # Process each search vector for this subset
    for v_orig in vecs:
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        # Build the residue map for this vector from the precomputed data
        residue_map_for_crt = {}
        for p in p_subset:
            roots_for_p = precomputed_residues.get(p, {}).get(v_orig_tuple)
            if roots_for_p:
                residue_map_for_crt[p] = roots_for_p

        # Apply CRT to find m-candidates from the collected roots
        primes_for_crt = list(residue_map_for_crt.keys())
        if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
             continue

        lists = [residue_map_for_crt[p] for p in primes_for_crt]
        for combo in itertools.product(*lists):
            M = reduce(mul, primes_for_crt, 1)

            if M > MAX_MODULUS:
                continue

            m0 = crt_cached(combo, tuple(primes_for_crt))

            # *** CORRECTED FILTERING CALL ***
            # Pass the FULL precomputed_residues map and the chosen extra_primes.
            # The filter also needs the vector to look up the correct residues.
            # We will pass it via a partial or lambda if needed, but for simplicity, we assume
            # the corrected filter can access `v_orig_tuple` from its scope or is modified to accept it.
            # Let's adapt the call to pass the vector-specific map.
            
            # Build the map needed for the filter
            full_residue_map_for_vector = {p: precomputed_residues.get(p, {}).get(v_orig_tuple, set()) for p in extra_primes_for_filtering}

            if not candidate_passes_extra_primes(m0, M, full_residue_map_for_vector, extra_primes_for_filtering, max_abs_t):
                continue  # This CRT combo was successfully filtered out.

            try:
                best_ms = minimize_archimedean_t(int(m0), int(M), r_m, shift, max_abs_t)
            except TypeError:
                best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]
                # This fallback might indicate a deeper issue if it happens frequently.

            for m_cand, _score in best_ms:
                found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

            try:
                a, b = rational_reconstruct(m0 % M, M)
                found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            except RationalReconstructionError:
                pass # This is an expected failure, not an error.

    return found_candidates_for_subset


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
        return set(), []

    # 2. Pre-compute modular residues, keeping RHS functions separate
    print(f"--- Pre-computing residues for {len(Ep_dict)} primes and {len(vecs)} vectors ---")
    precomputed_residues = {p: {} for p in Ep_dict}

    for p in tqdm(Ep_dict.keys(), desc="Pre-computing residues"):
        Ep = Ep_dict[p]
        mults_p = mult_lll[p]
        vecs_lll_p = vecs_lll[p]

        for idx, v_orig in enumerate(vecs):
            v_orig_tuple = tuple(v_orig)
            if all(c == 0 for c in v_orig):
                precomputed_residues[p][v_orig_tuple] = [set() for _ in rhs_list]
                continue

            try:
                v_p_transformed = vecs_lll_p[idx]
                Pm = Ep(0)
                for j, coeff in enumerate(v_p_transformed):
                    if int(coeff) in mults_p[j]:
                        Pm += mults_p[j][int(coeff)]
            except (KeyError, IndexError):
                precomputed_residues[p][v_orig_tuple] = [set() for _ in rhs_list]
                continue

            if Pm.is_zero():
                precomputed_residues[p][v_orig_tuple] = [set() for _ in rhs_list]
                continue

            # *** MODIFIED PART ***
            # Store roots in a list, one entry per RHS function.
            roots_by_rhs = []
            for i, rhs_ff in enumerate(rhs_list):
                roots_for_p_and_rhs = set()
                if p in rhs_modp_list[i]:
                    rhs_p = rhs_modp_list[i][p]
                    try:
                        num_modp = (Pm[0]/Pm[2] - rhs_p).numerator()
                        if not num_modp.is_zero():
                            roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                            roots_for_p_and_rhs.update(roots)
                    except (ZeroDivisionError, ArithmeticError):
                        pass
                roots_by_rhs.append(roots_for_p_and_rhs)
            
            precomputed_residues[p][v_orig_tuple] = roots_by_rhs

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

    overall_found_candidates = set()
    with multiprocessing.Pool() as pool:
        pbar = tqdm(
            pool.imap(worker_func, prime_subsets),
            total=len(prime_subsets),
            desc="Searching Prime Subsets"
        )
        for subset_results in pbar:
            overall_found_candidates.update(subset_results)

    # 4. Test candidates serially (no changes needed from here down)
    if not overall_found_candidates:
        return set(), []

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

    return new_xs, new_sections

# In search_lll.py, replace the existing worker function

def _process_prime_subset_precomputed(p_subset, vecs, r_m, shift, max_abs_t, precomputed_residues, prime_pool, num_rhs_fns):
    """
    Worker function to find m-candidates for a single subset of primes.
    This version processes each RHS function independently.
    """
    if not p_subset:
        return set()

    num_extra_primes = 5  # A small number is sufficient, 2 is seemingly optimal.
    extra_primes_for_filtering = [p for p in prime_pool if p not in p_subset][2:num_extra_primes]

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
                    best_ms = minimize_archimedean_t(int(m0), int(M), r_m, shift, max_abs_t)
                except TypeError:
                    best_ms = [(QQ(m0 + t * M), 0.0) for t in (-1, 0, 1)]

                for m_cand, _score in best_ms:
                    found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

                try:
                    a, b = rational_reconstruct(m0 % M, M)
                    found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
                except RationalReconstructionError:
                    pass

    return found_candidates_for_subset
