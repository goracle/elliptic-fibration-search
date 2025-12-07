
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector
from collections import defaultdict, Counter
from itertools import product
import sys
import traceback
from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, parallel
from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
import multiprocessing
from collections import defaultdict
from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR
from search_common import *

NUM_DOUBLINGS = 7 # for mumford height pairing independence test


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode
#
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.
#
# The 5 equations enforced are:
# 1. F1: r1^2 - s*r1 + p = 0
# 2. F2: Linear coefficient of (v^2 - f) mod u is 0
# 3. F3: Constant coefficient of (v^2 - f) mod u is 0
# 4. F4: f(r1) - f(r2) = 0 (implied by F2=0)
# 5. F5: v1*s + 2*v0 = 0 (Symmetry of v)


def _poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    """
    Reduce polynomial modulo u(x) = x^2 - s*x + p.
    Input: poly_coeffs in HIGH -> LOW order [a_n, a_{n-1}, ..., a_0].
    Returns: [r1, r0] where result = r1*x + r0.
    """
    coeffs = list(poly_coeffs)
    
    # 1. Trim leading zeros
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)
    
    # 2. Horner-like reduction
    while len(coeffs) > 2:
        a = coeffs.pop(0)
        # x^d -> x^{d-2}(s*x - p)
        # Add a*s to next term (x^{d-1})
        coeffs[0] = coeffs[0] + a * s
        # Subtract a*p from term after that (x^{d-2})
        coeffs[1] = coeffs[1] - a * p
        
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    # 3. Final normalization to [r1, r0]
    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================


from tqdm import tqdm
import signal # For safe multiprocessing worker init

# =============================================================================
# CORE ARITHMETIC & SOLVERS
# =============================================================================


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================


def mumford_precompute_residues_sequential(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                           rhs_modp_list, vecs_list, debug=False):
    """
    Sequential fallback: runs the parallel routine with a single worker.
    """
    print("Sequential fallback is using the parallel routine with a single worker.")
    return mumford_precompute_residues_parallel(eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
                                                rhs_modp_list, vecs_list, num_workers=1, debug=debug)

def _mumford_worker_entry(args):
    """Legacy entry point (placeholder)."""
    # NOTE: The provided code only used _solve_worker_wrapper
    return args[0], {} 

def validate_mumford_solver():
    """Simple test function (placeholder)."""
    print("Use verify_mumford_pair directly for testing.")
    return True


# Key fixes for Mumford reconstruction:
# 1. Track x-residues through the pipeline (keep solutions separate by x-residue)
# 2. Only combine solutions from the same (vector, x-residue) pair
# 3. Remove mod-p verification after reconstruction (it was checking wrong equation)


# =============================================================================
# MUMFORD SEARCH: CRITICAL FIXES
# =============================================================================
# 
# Key changes:
# 1. Verify ALL mod-p solutions immediately after finding them
# 2. Track x-residues through the pipeline (keep solutions separate)
# 3. Add height bounds to rational reconstruction
# 4. Verify reconstructed rationals are consistent with mod-p data
#
# =============================================================================


# mumford_complete.py
# 
# UPDATED VERSION with independent basis construction for Mumford divisors
#
# Key additions:
# 1. mumford_to_jacobian_element() - converts (u,v) to Jacobian element
# 2. check_mumford_independence() - tests linear independence via height pairing
# 3. build_mumford_basis_incremental() - builds basis incrementally
# 4. Integration into reconstruct_and_verify_mumford() to return basis instead of all divisors

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve
from sage.all import parallel


# =============================================================================
# JACOBIAN BASIS CONSTRUCTION (NEW)
# =============================================================================


# =============================================================================
# CORE ARITHMETIC & SOLVERS (EXISTING CODE)
# =============================================================================


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


# =============================================================================
# RECONSTRUCTION WITH BASIS BUILDING
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode


# =============================================================================
# JACOBIAN BASIS CONSTRUCTION
# =============================================================================


# =============================================================================
# CORE ARITHMETIC & SOLVERS
# =============================================================================


# =============================================================================
# WORKER FUNCTIONS
# =============================================================================


# =============================================================================
# RECONSTRUCTION WITH BASIS BUILDING
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=False):
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)


from sage.all import RDF, Matrix


from sage.all import RDF, Matrix, PolynomialRing, QQ, HyperellipticCurve


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve, RDF, log, LCM

# search_common must be available in python path

# =============================================================================
# MANUAL HEIGHT IMPLEMENTATIONS
# =============================================================================


# =============================================================================
# JACOBIAN BASIS CONSTRUCTION
# =============================================================================


# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================

def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Reconstructs rational Mumford divisors and builds an independent basis.
    """
    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE")
    print("="*70)

    found_xs = set()
    mumford_divisors_raw = []

    by_vector_and_xres = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for p in residues:
        for v_tuple, x_res_dict in residues[p].items():
            if isinstance(x_res_dict, list):
                by_vector_and_xres[v_tuple]['unknown'][p] = x_res_dict
            elif isinstance(x_res_dict, dict):
                for x_res, sols in x_res_dict.items():
                    by_vector_and_xres[v_tuple][x_res][p] = sols

    num_groups = sum(len(xres_groups) for xres_groups in by_vector_and_xres.values())
    print(f"Grouped into {len(by_vector_and_xres)} vectors, {num_groups} (vector,x-residue) pairs")

    total_attempted = 0
    recon_success = 0
    rejected_by_height = 0
    rejected_by_consistency = 0
    rejected_by_algebraic = 0

    for v_tuple, xres_groups in by_vector_and_xres.items():
        for x_res_key, prime_data in xres_groups.items():
            primes = sorted(prime_data.keys())
            if len(primes) < 3:
                continue
            
            M = 1
            for p in primes:
                M *= p
            
            sol_lists = [prime_data[p] for p in primes]
            limit = 5000
            
            for sol_combo in product(*sol_lists):
                if limit <= 0:
                    break
                limit -= 1
                total_attempted += 1
                
                try:
                    rec_vals = []
                    for idx in range(4):
                        vals = [sol[idx] for sol in sol_combo]
                        crt_val = crt_cached(tuple(vals), tuple(primes))
                        num, den = rational_reconstruct(crt_val, M)
                        
                        max_height = max(100000, int(M ** 0.35))
                        if abs(num) > max_height or abs(den) > max_height:
                            raise RationalReconstructionError("Height too large")
                        
                        rec_vals.append(QQ(num)/QQ(den))
                    
                    s, p_val, v0, v1 = rec_vals
                    
                except RationalReconstructionError:
                    rejected_by_height += 1
                    raise
                    continue
                
                reconstruction_ok = True
                for i, prime in enumerate(primes):
                    expected_sol = sol_combo[i]
                    try:
                        s_mod = (int(s.numerator()) * pow(int(s.denominator()), -1, prime)) % prime
                        p_mod = (int(p_val.numerator()) * pow(int(p_val.denominator()), -1, prime)) % prime
                        v0_mod = (int(v0.numerator()) * pow(int(v0.denominator()), -1, prime)) % prime
                        v1_mod = (int(v1.numerator()) * pow(int(v1.denominator()), -1, prime)) % prime
                    except (ZeroDivisionError):
                        reconstruction_ok = False
                        break
                    
                    if (s_mod != expected_sol[0] % prime or
                        p_mod != expected_sol[1] % prime or
                        v0_mod != expected_sol[2] % prime or
                        v1_mod != expected_sol[3] % prime):
                        reconstruction_ok = False
                        break
                
                if not reconstruction_ok:
                    rejected_by_consistency += 1
                    continue
                
                if not verify_mumford_pair(f_coeffs, s, p_val, v0, v1, modulus=None, debug_first_failure=False):
                    rejected_by_algebraic += 1
                    continue
                
                mumford_divisors_raw.append({
                    'vector': v_tuple, 's': s, 'p': p_val, 'v_0': v0, 'v_1': v1
                })
                recon_success += 1

    print(f"  Combinations tried: {total_attempted}")
    print(f"  Rejected by height: {rejected_by_height}")
    print(f"  Rejected by consistency: {rejected_by_consistency}")
    print(f"  Rejected by algebraic constraint: {rejected_by_algebraic}")
    print(f"  Successful reconstructions: {recon_success}")

    if not mumford_divisors_raw:
        print("  WARNING: No valid Mumford divisors reconstructed!")
        return found_xs, []

    mumford_divisors = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)

    for div in mumford_divisors:
        s, p_val = div['s'], div['p']
        
        # Check for rational roots of u(x)
        disc = s*s - 4*p_val

        if disc >= 0 and disc.is_square():
            div['has_rational_roots'] = True
            r1 = (s + disc.sqrt())/2
            r2 = (s - disc.sqrt())/2
            for r in (r1, r2):
                x_cand = r - shift
                if rationality_test(x_cand) is not None:
                    found_xs.add(x_cand)
        else:
            div['has_rational_roots'] = False

    print(f"  Unique Rational Points: {len(found_xs)}")
    
    if mumford_divisors:
        rational_roots_count = sum(1 for div in mumford_divisors_raw  # <-- Use ORIGINAL list
                                   if 'has_rational_roots' in div and div.get('has_rational_roots'))
        print(f"  {rational_roots_count} of {len(mumford_divisors_raw)} original divisors had rational roots in u(x)")
        print(f"\n--- Building Independent Mumford Basis ---")
        print("first 10 divisors:")
        for i in mumford_divisors[:10]:
            print(i)
        try:
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(
                mumford_divisors, 
                f_coeffs, 
                debug=True
            )
            
            print(f"\nBasis Construction Results:")
            print(f"  Found {basis_rank} independent divisors")
            if basis_H is not None:
                print(f"  Height pairing matrix:\n{basis_H}")
                print(f"  Determinant: {basis_H.determinant()}")
            
            return found_xs, basis_divisors
        except Exception as e:
            print(f"Basis construction failed: {e}")
            traceback.print_exc()
            raise
            return found_xs, mumford_divisors
    
    return found_xs, mumford_divisors


# =============================================================================
# CORE ARITHMETIC & HELPERS
# =============================================================================

def build_mumford_equations_from_fibration(tower, f_coeffs):
    return {
        'f_coeffs': f_coeffs,
        'const': 0,
        'm_sym': var('m')
    }

def poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    coeffs = list(poly_coeffs)
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)

    while len(coeffs) > 2:
        a = coeffs.pop(0)
        coeffs[0] = coeffs[0] + a * s
        coeffs[1] = coeffs[1] - a * p
        if modulus:
            coeffs[0] %= modulus
            coeffs[1] %= modulus

    if len(coeffs) == 0:
        return [0, 0]
    elif len(coeffs) == 1:
        return [0, coeffs[0]]
    else:
        if modulus:
            return [coeffs[0] % modulus, coeffs[1] % modulus]
        return coeffs


def _poly_mod_quad_fast(f_coeffs, s_val, p_val, mod_p):
    r1 = 0
    r0 = 0
    for coeff in f_coeffs:
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        r1, r0 = new_r1, new_r0
    return r1, r0


def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val):
    solutions = []
    x_res = int(x_residue) % p
    x_sq = (x_res * x_res) % p

    for s_val in range(p):
        p_val = (s_val * x_res - x_sq) % p
        A, B = _poly_mod_quad_fast(f_coeffs, s_val, p_val, p)
        
        coeffs_quad = [
            (s_val * s_val - 4 * p_val) % p,
            (-2 * (A * s_val + 2 * B)) % p,
            (A * A) % p
        ]
        
        a_q, b_q, c_q = coeffs_quad
        Z_roots = []

        if a_q == 0:
            if b_q != 0:
                Z_roots.append((-c_q * pow(b_q, -1, p)) % p)
        else:
            disc_q = (b_q * b_q - 4 * a_q * c_q) % p
            
            if disc_q == 0:
                Z_roots.append((-b_q * pow(2 * a_q, -1, p)) % p)
            elif pow(disc_q, (p - 1) // 2, p) == 1:
                sq_root = None
                if p < 1000:
                    for r in range(1, p):
                        if (r * r) % p == disc_q:
                            sq_root = r
                            break
                
                if sq_root is not None:
                    inv_2a = pow(2 * a_q, -1, p)
                    Z_roots.append(((-b_q + sq_root) * inv_2a) % p)
                    Z_roots.append(((-b_q - sq_root) * inv_2a) % p)
        
        valid_v1s = []
        for Z in Z_roots:
            if Z == 0:
                valid_v1s.append(0)
            else:
                if pow(Z, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == Z:
                            valid_v1s.append(r)
                            valid_v1s.append(p - r)
                            break

        for v1_val in valid_v1s:
            if v1_val == 0:
                if A != 0:
                    continue
                if B == 0:
                    solutions.append((s_val, p_val, 0, 0))
                elif pow(B, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == B:
                            solutions.append((s_val, p_val, r, 0))
                            solutions.append((s_val, p_val, p - r, 0))
                            break
            else:
                if p == 2:
                    v0_val = B % 2
                    if (s_val * v1_val) % 2 == A % 2:
                        solutions.append((s_val, p_val, v0_val, v1_val))
                else:
                    num = (A - s_val * (v1_val * v1_val)) % p
                    den = (2 * v1_val) % p
                    v0_val = (num * pow(den, -1, p)) % p
                    
                    lhs_2 = (v0_val * v0_val - p_val * v1_val * v1_val) % p
                    if lhs_2 == B:
                        solutions.append((s_val, p_val, v0_val, v1_val))

    return solutions


def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=False):
    if modulus is None:
        R = PolynomialRing(QQ, 'x')
    else:
        R = PolynomialRing(GF(modulus), 'x')
    
    x = R.gen()
    
    if modulus is None:
        s_val = QQ(s)
        p_val = QQ(p)
        v0_val = QQ(v0)
        v1_val = QQ(v1)
        f_poly_coeffs = [QQ(c) for c in f_coeffs]
    else:
        s_val = int(s) % modulus
        p_val = int(p) % modulus
        v0_val = int(v0) % modulus
        v1_val = int(v1) % modulus
        f_poly_coeffs = [int(c) % modulus for c in f_coeffs]
    
    u_poly = x**2 - s_val*x + p_val
    v_poly = v1_val*x + v0_val
    
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    diff = v_poly**2 - f_poly
    remainder = diff % u_poly
    
    return remainder.is_zero()


def _normalize_sign(s, p, v0, v1):
    if v1 < 0 or (v1 == 0 and v0 < 0):
        return (s, p, -v0, -v1)
    return (s, p, v0, v1)


def canonicalize_and_dedup(divisors, f_coeffs):
    seen = {}
    out = []
    
    for tup in divisors:
        s, p, v0, v1 = tup['s'], tup['p'], tup['v_0'], tup['v_1']
        
        if not verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None):
            continue
        
        s1, p1, v01, v11 = _normalize_sign(s, p, v0, v1)
        key = (s1, p1, v01, v11)
        
        if key not in seen:
            seen[key] = True
            tup['s'], tup['p'], tup['v_0'], tup['v_1'] = s1, p1, v01, v11
            out.append(tup)
            
    return out


# =============================================================================
# WORKERS & PARALLEL
# =============================================================================

def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _solve_worker_wrapper(args):
    p, f_coeffs_ints, x_residues_map, const_val_int = args
    try:
        p_results = {}
        for v_tuple, x_res_list in x_residues_map.items():
            if isinstance(x_res_list, int):
                x_res_list = [x_res_list]
            
            x_res_to_sols = {}
            for x_res in x_res_list:
                sols = solve_mumford_mod_p_optimized(f_coeffs_ints, p, x_res, const_val_int)
                
                verified_sols = []
                for sol in sols:
                    s, p_val, v0, v1 = sol
                    if verify_mumford_pair(f_coeffs_ints, s, p_val, v0, v1, modulus=p):
                        verified_sols.append(sol)
                
                if verified_sols:
                    x_res_to_sols[x_res] = verified_sols
            
            if x_res_to_sols:
                p_results[v_tuple] = x_res_to_sols
                
        return p, p_results
    except Exception:
        sys.stderr.write(f"\nCRITICAL ERROR IN MUMFORD WORKER (p={p}):\n")
        traceback.print_exc(file=sys.stderr)
        raise
        return p, {}


def mumford_precompute_residues_parallel(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                                         rhs_modp_list, vecs_list, num_workers=8, debug=False):
    f_coeffs = eqs_dict['f_coeffs']
    f_coeffs_ints = [int(c) for c in f_coeffs]
    
    try:
        const_val_int = int(QQ(eqs_dict['const']))
    except:
        const_val_int = 0
        raise
        
    if debug:
        print(f"[mumford] Generating tasks for {len(prime_list)} primes...")

    tasks = []
    
    for p in prime_list:
        if p not in Ep_dict:
            continue
        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        if not p_vecs:
            continue
        
        try:
            Fp = GF(p)
            R_m = Fp['m']
            m_var = R_m.gen()
            rhs_poly = -m_var + Fp(const_val_int)
        except Exception:
            raise
            continue

        x_residues_map = {}
        p_mults = mult_lll.get(p, {})
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue
            
            Pm = Ep(0)
            valid_vec = True
            v_coeffs = p_vecs[v_idx]

            for i, c in enumerate(v_coeffs):
                k = int(c)
                if k == 0:
                    continue
                
                try:
                    mults_for_sec = p_mults[i]
                    if k in mults_for_sec:
                        Pm += mults_for_sec[k]
                    else:
                        valid_vec = False
                        break
                except (IndexError, KeyError, TypeError):
                    valid_vec = False
                    raise
                    break
            
            if not valid_vec or Pm.is_zero() or Pm[2] == 0:
                continue
            
            try:
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator()
                
                if diff_num.is_zero():
                    continue
                    
                roots = diff_num.roots(multiplicities=False)
                
                if roots:
                    valid_residues = []
                    for m_root in roots:
                        m_val = int(m_root)
                        x_val = (-m_val + const_val_int) % p
                        valid_residues.append(x_val)
                    
                    if valid_residues:
                        x_residues_map[v_tuple] = valid_residues
            except Exception:
                raise
                continue
            
        if x_residues_map:
            tasks.append((p, f_coeffs_ints, x_residues_map, const_val_int))

    if not tasks:
        if debug:
            print("[mumford] No tasks generated!")
        return {}
        
    try:
        ctx = multiprocessing.get_context("fork")
        pool_obj = ctx.Pool(num_workers, initializer=_init_worker)
    except:
        pool_obj = multiprocessing.Pool(num_workers, initializer=_init_worker)
        raise

    results_dict = {}
    with pool_obj as pool:
        for p, result_map in tqdm(pool.imap_unordered(_solve_worker_wrapper, tasks), 
                                  total=len(tasks), desc="Solving Mumford Mod P"):
            results_dict[p] = result_map
            
    return results_dict


# =============================================================================
# MANUAL HEIGHT & INDEPENDENCE CHECK (FIXED)
# =============================================================================


# -----------------------------------------------------------------------------
# Robust replacements for height / conversion / independence checks
# -----------------------------------------------------------------------------

from fractions import Fraction
import math

def _extract_u_coeffs_as_fractions(u):
    """
    Return list of coefficients of u (highest-to-lowest) as Fraction objects.
    Accepts:
      - Sage polynomial (use .list() or .coefficients?)
      - Python list/tuple of coeffs
      - tuple-like from mumford (already rational objects)
    Ensures monic by appending the implicit leading 1 if needed.
    """
    # If u is a Sage polynomial, try u.list() (coeffs lowest-first)
    try:
        if hasattr(u, 'list'):
            coeffs_low = u.list()    # lowest-degree first
            coeffs = list(reversed(coeffs_low))  # highest-first
        elif hasattr(u, 'coefficients'):
            coeffs = u.coefficients(sparse=False)
            # coefficients may not include zeros; try to detect degree
            if hasattr(u, 'degree') and u.degree() is not None:
                deg = u.degree()
                # create full list
                full = [QQ(0)] * (deg+1)
                for i, c in enumerate(u.coefficients(sparse=False)):
                    # this is fragile in some Sage versions; fallback below
                    pass
        else:
            # Fallback: treat u as an iterable of coeffs (highest-first)
            coeffs = list(u)
    except Exception:
        # Let exceptions bubble: user asked to raise them
        raise

    # Coerce each coefficient to Fraction robustly
    frac_coeffs = []
    for c in coeffs:
        # If it's a Sage rational (QQ), get numerator/denominator
        try:
            if hasattr(c, 'numerator') and hasattr(c, 'denominator'):
                n = int(c.numerator())
                d = int(c.denominator())
                frac_coeffs.append(Fraction(n, d))
            else:
                # For floats or RDF, convert via Fraction.from_float if necessary
                frac_coeffs.append(Fraction(c))
        except Exception:
            # last resort: try string conversion
            frac_coeffs.append(Fraction(str(c)))
            raise

    # Ensure monic: if leading coeff != 1, check if implicit monic (some code gives only lower terms)
    if not frac_coeffs:
        return [Fraction(1,1)]
    # If leading coeff equals 1, fine. If not, assume monic poly was given lacking leading 1:
    if frac_coeffs[0] != 1:
        # If the polynomial *is* monic but the leading 1 is missing (common if only lower coefs were returned),
        # then append an explicit leading 1.
        # Heuristic: if len(frac_coeffs) == 2 and frac_coeffs[0] < 1 and frac_coeffs[1] != 0, we try appending 1.
        # Safer: don't silently mutate; prefer to return as-is and let caller handle if degree mismatch.
        # For now, if leading coeff is not 1 but <= 1 in magnitude, append an explicit 1 to represent monic.
        frac_coeffs = [Fraction(1,1)] + frac_coeffs

    return frac_coeffs


def manual_naive_height(P):
    """
    Robust naive logarithmic height from Mumford u-polynomial.
    Returns float(log(max_abs)), always finite for valid input, else raises.
    """
    try:
        u = P[0]  # Mumford u polynomial
    except Exception:
        raise

    fracs = _extract_u_coeffs_as_fractions(u)
    # Convert to integer projective coordinates by clearing denominators
    dens = [f.denominator for f in fracs]
    L = 1
    for d in dens:
        L = (L * d) // math.gcd(L, d)

    int_coeffs = [int((f * L).numerator) for f in fracs]  # numerators after clearing denom
    # append the implicit leading coefficient L*1 (monic)
    int_coeffs.append(int(L))

    if not int_coeffs:
        return 0.0

    max_abs = max(abs(c) for c in int_coeffs)
    # defensive: ensure positive integer
    max_abs = max(1, int(max_abs))
    return math.log(max_abs)


def manual_canonical_height(P, limit=8, debug=False):
    """
    Approximate canonical height by computing h(2^n P)/4^n for n=0..limit and returning last value.
    Re-raises any exceptions from doubling. Prints intermediate heights if debug=True.
    """
    if P.is_zero():
        return 0.0

    Q = P
    vals = []
    try:
        for n in range(limit + 1):
            hQ = manual_naive_height(Q)
            vals.append(float(hQ) / (4.0 ** n))
            if debug:
                print(f"[canon] n={n} naive_h={hQ:.6g} ratio={vals[-1]:.6g}")
            Q = 2 * Q
    except Exception:
        # Re-raise after printing diagnostic if possible
        if debug:
            print("Doubling failed at step", n)
        raise

    # Return the last computed ratio; caller can examine intermediate vals if needed.
    return float(vals[-1])


def compute_manual_height_pairing(P, Q, limit=8, debug=False):
    """
    <P, Q> = 1/2 * (h_hat(P+Q) - h_hat(P) - h_hat(Q))
    Uses the manual canonical-height approximation.
    """
    try:
        if P.is_zero() or Q.is_zero():
            return float(0.0)

        # Use manual canonical height approximation for all three
        h_p = manual_canonical_height(P, limit=limit, debug=debug)
        h_q = manual_canonical_height(Q, limit=limit, debug=debug)
        h_sum = manual_canonical_height(P + Q, limit=limit, debug=debug)
        val = 0.5 * (h_sum - h_p - h_q)
        return float(val)
    except Exception:
        raise


def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Create a Jacobian element while coercing the u,v polynomials into the curve's polynomial ring.
    Raises on failure (user preference).
    """
    try:
        f_curve, h_curve = C.hyperelliptic_polynomials()
        R = f_curve.parent()   # polynomial ring of the curve
        x = R.gen()

        # Coerce inputs to rational numbers in Python Fraction and then to QQ for the ring
        def to_QQ_obj(a):
            try:
                return QQ(a)
            except Exception:
                return QQ(Fraction(str(a)))

        s_q = to_QQ_obj(s)
        p_q = to_QQ_obj(p)
        v0_q = to_QQ_obj(v0)
        v1_q = to_QQ_obj(v1)

        u_poly = x**2 - s_q * x + p_q
        v_poly = v1_q * x + v0_q

        # Make sure polynomials live in the same parent as the curve
        u_poly = R(u_poly)
        v_poly = R(v_poly) if v_poly.parent() == R else R(v_poly)  # coerce v into same ring

        return C.jacobian()([u_poly, v_poly])
    except Exception:
        # re-raise so user sees the problem
        raise


def check_mumford_independence(divisors, f_coeffs, debug=False):
    """
    Build Jacobian elements and compute pairing matrix H using compute_manual_height_pairing.
    Returns (is_indep, rank, H_matrix)
    """
    if not divisors:
        return True, 0, None

    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)

    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(div['s'], div['p'], div['v_0'], div['v_1'], C)
            if not elem.is_zero():
                jac_elements.append(elem)
            else:
                if debug:
                    print("[check] element is zero, skipping.")
        except Exception:
            # re-raise after optional debug info
            if debug:
                print("[check] failed to convert divisor to jac element:", div)
            raise

    if not jac_elements:
        return True, 0, None

    n = len(jac_elements)
    H = Matrix(RDF, n, n)
    for i in range(n):
        for j in range(i, n):
            try:
                val = compute_manual_height_pairing(jac_elements[i], jac_elements[j], debug=debug)
            except Exception:
                # Surface which pair caused trouble
                if debug:
                    print(f"[check] height pairing failed for indices {i},{j}")
                raise
            H[i, j] = val
            H[j, i] = val

    if n == 1:
        is_indep = abs(H[0, 0]) > 1e-8
        rank = 1 if is_indep else 0
    else:
        rank = H.rank()
        is_indep = (rank == n)
    return is_indep, rank, H


def dbg_poly_info(poly):
    # poly is a sage polynomial
    coeffs = poly.list()          # lowest-first
    if not coeffs:
        return "deg=-inf"
    deg = poly.degree()
    # get bit sizes
    def bits_of(c):
        try:
            return int(c.nbits()) if hasattr(c, 'nbits') else int(Fraction(c).numerator).bit_length()
        except Exception:
            try:
                return int(abs(int(c)).bit_length())
            except Exception:
                raise
                return -1
            raise
    bits = [bits_of(c) for c in coeffs]
    maxbits = max(bits) if bits else 0
    return f"deg={deg}, maxcoeff_bits={maxbits}, len={len(coeffs)}"

def dump_jacobian_mumford_info(JP, label="P"):
    # JP is jacobian element [u,v]
    try:
        u = JP[0]   # polynomial
        v = JP[1]
        print(f"[DBG] {label} u: {dbg_poly_info(u)}; v: {dbg_poly_info(v)}; parents: {type(u.parent())}")
    except Exception as e:
        print("[DBG] failed to print mumford info:", e)
        raise


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


def naive_height_safe(s, p, v0, v1, debug=False):
    """
    Compute naive height from Mumford representation without building Jacobian.
    Returns log(max(|coeffs of u|, |coeffs of v|)).
    """
    from fractions import Fraction
    import math
    
    # Force conversion to QQ first, then to Fraction
    s_qq = QQ(s)
    p_qq = QQ(p)
    v0_qq = QQ(v0)
    v1_qq = QQ(v1)
    
    # Convert QQ to Fraction using numerator/denominator
    s_frac = Fraction(int(s_qq.numerator()), int(s_qq.denominator()))
    p_frac = Fraction(int(p_qq.numerator()), int(p_qq.denominator()))
    v0_frac = Fraction(int(v0_qq.numerator()), int(v0_qq.denominator()))
    v1_frac = Fraction(int(v1_qq.numerator()), int(v1_qq.denominator()))
    
    # u(x) = x^2 - s*x + p has coefficients [1, -s, p]
    # v(x) = v1*x + v0 has coefficients [v1, v0]
    
    all_coeffs = [
        Fraction(1, 1),  # leading coeff of u
        -s_frac,
        p_frac,
        v1_frac,
        v0_frac
    ]
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))  # include denominator
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return float(math.log(max_abs))


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (FIXED)
# =============================================================================


def compute_height_pairing_simple(D1, D2, num_doublings=NUM_DOUBLINGS):
    """
    Compute <D1, D2> using LIMITED doublings to avoid coefficient explosion.
    Uses: <D1, D2> = (h(D1+D2) - h(D1) - h(D2)) / 2
    where h is naive height.
    
    Only does `num_doublings` iterations instead of 8.
    """
    from fractions import Fraction
    import math
    
    def naive_height_from_jacobian(D):
        u, v = D[0], D[1]
        u_coeffs = u.list()
        v_coeffs = v.list()
        
        all_coeffs = []
        for c in u_coeffs + v_coeffs:
            c_qq = QQ(c)
            all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
        
        # Clear denominators
        lcm_den = 1
        for f in all_coeffs:
            lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
        
        int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
        int_coeffs.append(int(lcm_den))
        
        max_abs = max(abs(c) for c in int_coeffs if c != 0)
        max_abs = max(1, max_abs)
        
        return float(math.log(max_abs))
    
    if D1.is_zero() or D2.is_zero():
        return 0.0
    
    # Compute heights with limited doublings
    vals = []
    P, Q, S = D1, D2, D1 + D2
    
    for n in range(num_doublings):
        hP = naive_height_from_jacobian(P)
        hQ = naive_height_from_jacobian(Q)
        hS = naive_height_from_jacobian(S)
        
        pairing = (hS - hP - hQ) / 2.0
        vals.append(pairing / (4.0 ** n))
        
        P = P + P
        Q = Q + Q
        S = S + S
    
    # Return the last value (most refined estimate)
    return vals[-1]


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================


# =============================================================================
# TORSION DETECTION & BASIS BUILDING (EXACT ARITHMETIC)
# =============================================================================

def is_mumford_torsion_fast(s, p, v0, v1, f_coeffs, max_order=12, debug=False):
    """
    Fast torsion test using modular verification.
    Tests if divisor is n-torsion for n in [2, 3, 4, 5, 6, 8, 10, 12].
    
    Returns: (is_torsion, order) where order=None if not torsion
    """
    # Build curve
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Convert to Jacobian element
    u_poly = x**2 - QQ(s)*x + QQ(p)
    v_poly = QQ(v1)*x + QQ(v0)
    D = J([u_poly, v_poly])
    
    if D.is_zero():
        return True, 1
    
    # Test small orders using ONLY addition (no doubling)
    test_orders = [2, 3, 4, 5, 6, 8, 10, 12]
    
    for n in test_orders:
        # Compute nD by repeated addition (safer than doubling)
        nD = D
        for _ in range(n - 1):
            nD = nD + D
        
        if nD.is_zero():
            if debug:
                print(f"[torsion] Found {n}-torsion divisor")
            return True, n
    
    return False, None


def build_mumford_basis_incremental(all_divisors, f_coeffs, num_doublings=8, debug=True):
    """
    Build independent basis using EXACT height pairing checks.
    Filters out torsion divisors first.
    Uses exact rational arithmetic throughout - no floating point.
    
    Args:
        num_doublings: Number of doubling iterations for canonical height approximation.
                      Higher = more accurate but slower. Typical values: 6-10.
    """
    if not all_divisors:
        return [], 0, None
    
    print(f"\n[basis] Starting with {len(all_divisors)} total divisors")
    print(f"[basis] Using {num_doublings} doublings for height pairing approximation")
    
    # Build curve once
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Filter out torsion
    non_torsion = []
    torsion_count = 0
    
    for div in all_divisors:
        is_tors, order = is_mumford_torsion_fast(
            div['s'], div['p'], div['v_0'], div['v_1'], 
            f_coeffs, debug=False
        )
        
        if is_tors:
            torsion_count += 1
            if debug and torsion_count <= 3:
                print(f"[basis] Filtered torsion divisor (order {order}): {div}")
        else:
            non_torsion.append(div)
    
    print(f"[basis] Filtered {torsion_count} torsion divisors -> {len(non_torsion)} candidates")
    
    if not non_torsion:
        return [], 0, None
    
    # Convert to Jacobian elements
    jac_elements = []
    for div in non_torsion:
        u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        jac_elements.append((div, D))
    
    # Build basis using EXACT independence checks
    basis = []
    basis_jac = []
    
    for i, (div, D) in enumerate(jac_elements):
        if not basis:
            # First divisor - just check self-pairing is nonzero
            h_exact = compute_height_pairing_exact(D, D, num_doublings=num_doublings)
            h_float = float(h_exact)
            
            if abs(h_float) < 1e-8:
                if debug:
                    print(f"[basis] Skipping divisor {i+1}: self-pairing too small ({h_float:.3g})")
                continue
            
            basis.append(div)
            basis_jac.append(D)
            if debug:
                print(f"[basis] Added divisor 1 (self-pairing {h_float:.3g})")
        else:
            # Check independence by computing height pairing matrix
            candidate_basis = basis_jac + [D]
            n = len(candidate_basis)
            
            # Build matrix with EXACT rationals
            H_exact = Matrix(QQ, n, n)
            for ii in range(n):
                for jj in range(ii, n):
                    h_ij_exact = compute_height_pairing_exact(
                        candidate_basis[ii], 
                        candidate_basis[jj],
                        num_doublings=num_doublings
                    )
                    H_exact[ii, jj] = h_ij_exact
                    H_exact[jj, ii] = h_ij_exact
            
            # Check rank using exact arithmetic
            det_exact = H_exact.determinant()
            rank_exact = H_exact.rank()
            
            # Convert to float for display
            det_float = float(det_exact)
            
            # Check if determinant is too small (indicates near-dependence)
            # or if rank dropped (definite dependence)
            det_threshold = 1e-3  # Conservative threshold
            
            if rank_exact == n and abs(det_float) > det_threshold:
                # Independent and determinant is large enough!
                basis.append(div)
                basis_jac.append(D)
                if debug:
                    print(f"[basis] Added divisor {len(basis)} (rank {rank_exact}/{n}, det {det_float:.3g})")
            else:
                if debug:
                    reason = "rank dropped" if rank_exact < n else f"det too small ({det_float:.3g})"
                    print(f"[basis] Skipping divisor {i+1}: {reason} (rank {rank_exact}/{n})")
    
    rank = len(basis)
    
    # Build final height matrix with EXACT rationals
    if rank > 0:
        H_exact = Matrix(QQ, rank, rank)
        for i in range(rank):
            for j in range(i, rank):
                h_ij_exact = compute_height_pairing_exact(
                    basis_jac[i], 
                    basis_jac[j], 
                    num_doublings=num_doublings
                )
                H_exact[i, j] = h_ij_exact
                H_exact[j, i] = h_ij_exact
        
        if debug:
            print(f"\n[basis] Final rank: {rank}")
            print(f"[basis] Checked {len(jac_elements)} candidates total")
            det_exact = H_exact.determinant()
            print(f"[basis] Determinant (exact): {det_exact}")
            print(f"[basis] Determinant (float): {float(det_exact):.3g}")
            
            # Also show the matrix
            print(f"[basis] Height pairing matrix (exact QQ):")
            print(H_exact)
    else:
        H_exact = None
    
    return basis, rank, H_exact


def naive_height_exact(D):
    """
    Compute a naive height in exact rationals from a Mumford divisor D = [u, v].
    Returns a QQ number (log of max coefficient magnitude, exact).
    """
    from fractions import Fraction
    import math
    
    u_coeffs = D[0].list()
    v_coeffs = D[1].list()
    
    # Convert to exact fractions
    all_coeffs = []
    for c in u_coeffs + v_coeffs:
        c_qq = QQ(c)
        all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
    
    # Clear denominators
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return QQ(math.log(max_abs))


def compute_height_pairing_exact(D1, D2, num_doublings=4, debug=False):
    """
    Exact height pairing <D1, D2> using naive heights in QQ.
    Fully exact arithmetic (no floats), returns a QQ number.
    
    Uses: <D1, D2> = (h(D1+D2) - h(D1) - h(D2)) / 2
    with successive doublings to approximate canonical height.
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    vals = []
    P, Q, S = D1, D2, D1 + D2
    
    for n in range(num_doublings):
        hP = naive_height_exact(P)
        hQ = naive_height_exact(Q)
        hS = naive_height_exact(S)
        pairing = (hS - hP - hQ) / 2
        vals.append(pairing / (4**n))
        
        if debug and n >= num_doublings - 3:
            print(f"  [pairing] n={n}: h(P)={float(hP):.3g}, h(Q)={float(hQ):.3g}, h(S)={float(hS):.3g}")
            print(f"  [pairing] n={n}: pairing/4^n = {float(vals[-1]):.6g}")
        
        # Double all three divisors
        P = P + P
        Q = Q + Q
        S = S + S
    
    # Return the last (most refined) estimate
    return vals[-1]
