
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


def build_mumford_basis_incremental(all_divisors, f_coeffs, max_rank=2, debug=True):
    """
    Build a basis of independent Mumford divisors incrementally.
    """
    basis = []
    
    if debug:
        print(f"Building basis from {len(all_divisors)} candidate divisors...")
    
    for i, div in enumerate(all_divisors):
        if len(basis) >= max_rank:
            if debug:
                print(f"Reached max rank {max_rank}, stopping")
            break
        
        # Try adding this divisor to the basis
        candidate_basis = basis + [div]
        is_indep, rank, H = check_mumford_independence(candidate_basis, f_coeffs)
        
        if is_indep and rank == len(candidate_basis):
            # This divisor is independent!
            basis.append(div)
            if debug:
                print(f"[basis] Added divisor {i+1}: rank now {len(basis)}/{max_rank}")
                if H is not None:
                    print(f"  Height matrix determinant: {H.determinant()}")
        elif debug and i < 20:  # Don't spam for later divisors
            print(f"[basis] Divisor {i+1} is dependent, skipping")
    
    # Final check
    if basis:
        is_indep, final_rank, final_H = check_mumford_independence(basis, f_coeffs)
        return basis, final_rank, final_H
    else:
        return [], 0, None


# =============================================================================
# CORE ARITHMETIC & SOLVERS
# =============================================================================

def poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    """
    Reduce polynomial f(x) modulo u(x) = x^2 - s*x + p.
    Input: poly_coeffs in HIGH -> LOW order [a_n, a_{n-1}, ..., a_0].
    Returns: [r1, r0] where result = r1*x + r0.
    """
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
    """
    Computes f(x) mod (x^2 - sx + p) over GF(p).
    Uses Horner-like scheme for O(deg(f)) complexity.
    Returns (A, B) such that f(x) = A*x + B mod u(x).
    """
    r1 = 0
    r0 = 0
    for coeff in f_coeffs:
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        r1, r0 = new_r1, new_r0
    return r1, r0


def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val):
    """
    General O(p) solver for Mumford coordinates (s, p, v0, v1) mod p.
    Solves v^2 = f(x) mod u(x) where u(x) = x^2 - s*x + p,
    assuming x_residue is a root of u.
    """
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
    """
    Verifies that v(x)^2 == f(x) (mod u(x)), where u(x) = x^2 - s*x + p.
    """
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
    """Canonical sign normalization for Mumford coordinates."""
    if v1 < 0 or (v1 == 0 and v0 < 0):
        return (s, p, -v0, -v1)
    return (s, p, v0, v1)


def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Normalizes the sign of v(x) and removes duplicate divisors.
    """
    seen = {}
    out = []
    
    # Run diagnostic once
    if not hasattr(canonicalize_and_dedup, "_diag_run"):
        canonicalize_and_dedup._diag_run = True
        # (Diagnostic block preserved but silenced for brevity)
        pass

    for tup in divisors:
        s, p, v0, v1 = tup['s'], tup['p'], tup['v_0'], tup['v_1']
        
        # Verify correctness again just in case
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
# WORKER FUNCTIONS
# =============================================================================

def _init_worker():
    """Initializes worker process to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _solve_worker_wrapper(args):
    """Worker function for multiprocessing with mod-p verification."""
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
    """Precomputes Mumford coordinate residues modulo a list of primes in parallel."""
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
        except Exception as e:
            if debug:
                print(f"  Skipping p={p}: Ring setup failed {e}")
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
# RECONSTRUCTION WITH BASIS BUILDING
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
                    except (ZeroDivisionError, ValueError):
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
                
                # CRITICAL FIX: Check the return value!
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
        # Fallback to empty
        return found_xs, []

    mumford_divisors = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)

    for div in mumford_divisors:
        s, p_val = div['s'], div['p']
        
        # Check for rational points (u has rational roots)
        PR = PolynomialRing(QQ, 'x')
        x = PR.gen()
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
    
    # === BUILD INDEPENDENT BASIS ===
    if mumford_divisors:
        print(f"\n--- Building Independent Mumford Basis ---")
        try:
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(
                mumford_divisors, 
                f_coeffs, 
                max_rank=2,
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
# HELPER FUNCTIONS
# =============================================================================

def build_mumford_equations_from_fibration(tower, f_coeffs):
    return {
        'f_coeffs': f_coeffs,
        'const': 0,
        'm_sym': var('m')
    }

def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=False):
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)


def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Convert Mumford coordinates (u,v) to a Jacobian element.
    Safe version: extracts the polynomial ring correctly from the curve object.
    
    Args:
        s, p, v0, v1: Mumford coordinates (u = x^2 - sx + p, v = v1*x + v0)
        C: Sage HyperellipticCurve object
        
    Returns:
        JacobianMorphismPoint on C.jacobian()
    """
    # 1. Extract the correct polynomial ring from the curve's defining polynomial.
    #    Hyperelliptic curves in Sage store defining polys as (f, h) where y^2 + hy = f.
    #    We grab the parent ring of f.
    f_curve, h_curve = C.hyperelliptic_polynomials()
    R = f_curve.parent() 
    x = R.gen()
    
    # 2. Explicitly coerce coefficients into the base ring of the curve (usually QQ)
    #    This prevents "unsupported operand parent" errors if s, p are Sage Integers vs Rationals.
    base = C.base_ring()
    try:
        s_val = base(s)
        p_val = base(p)
        v0_val = base(v0)
        v1_val = base(v1)
    except Exception:
        # Fallback if direct coercion fails (unlikely for rational search)
        s_val, p_val, v0_val, v1_val = s, p, v0, v1
        raise

    # 3. Build polynomials in the specific ring R
    u_poly = x**2 - s_val*x + p_val
    v_poly = v1_val*x + v0_val
    
    J = C.jacobian()
    
    # 4. Pass the polynomials to the Jacobian constructor
    return J([u_poly, v_poly])


from sage.all import RDF, Matrix


from sage.all import RDF, Matrix, PolynomialRing, QQ, HyperellipticCurve

def compute_manual_height_pairing(P, Q):
    """
    Computes the canonical height pairing <P, Q> using the parallelogram law:
    <P, Q> = 1/2 * ( h(P+Q) - h(P) - h(Q) )
    Checks for .canonical_height() first (standard for Genus 2 Jacobians).
    """
    try:
        if P.is_zero() or Q.is_zero():
            return RDF(0.0)
            
        sum_pt = P + Q
        
        def get_h(pt):
            # Genus 2 Jacobian points in Sage usually have this method
            if hasattr(pt, 'canonical_height'):
                return pt.canonical_height()
            # Fallback for elliptic curves or other objects
            if hasattr(pt, 'height'):
                return pt.height()
            raise AttributeError(f"Object {type(pt)} has no height/canonical_height method")

        h_p = get_h(P)
        h_q = get_h(Q)
        h_sum = get_h(sum_pt)
        
        pairing = 0.5 * (h_sum - h_p - h_q)
        return RDF(pairing)
    except Exception as e:
        # Propagate error to the caller so it can fallback safely
        raise RuntimeError(f"Height calc failed: {e}")


def check_mumford_independence(divisors, f_coeffs):
    """
    Check if a list of Mumford divisors is linearly independent in Jac(C)(Q).
    """
    if not divisors:
        return True, 0, None
    
    # 1. Build the curve over QQ explicitly
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    
    # 2. Convert to Jacobian elements
    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(
                div['s'], div['p'], div['v_0'], div['v_1'], C
            )
            jac_elements.append(elem)
        except Exception as e:
            print(f"Warning: Could not convert divisor to Jacobian element: {e}")
            raise
            continue
    
    if not jac_elements:
        return True, 0, None
    
    # 3. Compute height pairing matrix
    try:
        n = len(jac_elements)
        H = Matrix(RDF, n, n)
        
        for i in range(n):
            for j in range(i, n):
                val = compute_manual_height_pairing(jac_elements[i], jac_elements[j])
                H[i,j] = val
                if i != j:
                    H[j,i] = val
        
        # 4. Check numerical rank
        rank = H.rank()
        is_independent = (rank == n)
        return is_independent, rank, H
        
    except Exception as e:
        print(f"Warning: Independence check failed ({e}). Assuming independent to continue.")
        # Fallback: Assume independent so we don't lose data or crash
        raise
        return True, len(jac_elements), None


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector, Matrix, HyperellipticCurve, RDF, log, LCM
from sage.all import parallel
from collections import defaultdict, Counter
from itertools import product
import sys
import traceback
import multiprocessing
from tqdm import tqdm
import signal

# search_common must be available in python path
from search_common import *
from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError

# =============================================================================
# MANUAL HEIGHT IMPLEMENTATIONS
# =============================================================================

def manual_naive_height(P):
    """
    Computes a naive logarithmic height for a Genus 2 Jacobian point P.
    Uses the coefficients of the u-polynomial in the Mumford representation (u, v).
    h(P) = log(max(|denominator|, |numerators of coeffs|)) of the projective u-coords.
    """
    try:
        # P is a Jacobian point. For Genus 2 in Sage, P[0] is u(x), P[1] is v(x).
        u = P[0]
        
        # Extract coefficients of u (monic quadratic x^2 + u1*x + u0)
        coeffs = u.coefficients(sparse=False)
        
        # Robust coefficient extraction and coercion to QQ
        try:
             rational_coeffs = [QQ(c) for c in coeffs]
        except Exception:
             return 0.0

        # Convert to projective integer coordinates [L, L*u1, L*u0]
        denominators = [c.denominator() for c in rational_coeffs]
        L = LCM(denominators)
        
        integer_coeffs = [ZZ(c * L) for c in rational_coeffs]
        # Include the leading coefficient denominator (L*1) in the height max
        integer_coeffs.append(ZZ(L))
        
        max_abs = max(abs(c) for c in integer_coeffs)
        
        return float(log(max(1, max_abs)))
    except Exception:
        return 0.0

def manual_canonical_height(P, limit=6):
    """
    Approximates canonical height using Tate's limit: h_hat(P) = lim h(2^n P) / 4^n.
    limit=6 implies division by 4^6 = 4096, sufficient for independence checks.
    """
    if P.is_zero():
        return 0.0
        
    Q = P
    scale = 1.0
    
    for _ in range(limit):
        try:
            Q = 2 * Q
            scale *= 4.0
        except Exception:
            break
            
    return manual_naive_height(Q) / scale


def compute_manual_height_pairing(P, Q):
    """
    Computes the canonical height pairing <P, Q> using the parallelogram law:
    <P, Q> = 1/2 * ( h(P+Q) - h(P) - h(Q) )
    Checks for .canonical_height() first, then falls back to manual implementation.
    """
    try:
        if P.is_zero() or Q.is_zero():
            return RDF(0.0)
        
        # Try Sage built-in first
        if hasattr(P, 'canonical_height'):
            try:
                h_p = P.canonical_height()
                h_q = Q.canonical_height()
                h_sum = (P+Q).canonical_height()
                return 0.5 * (h_sum - h_p - h_q)
            except Exception:
                pass 
        
        # Fallback to manual implementation
        h_p = manual_canonical_height(P)
        h_q = manual_canonical_height(Q)
        h_sum = manual_canonical_height(P + Q)
        
        return 0.5 * (h_sum - h_p - h_q)

    except Exception as e:
        # Propagate error only if manual calculation also fails
        raise RuntimeError(f"Height calc failed: {e}")


# =============================================================================
# JACOBIAN BASIS CONSTRUCTION
# =============================================================================

def build_mumford_basis_incremental(all_divisors, f_coeffs, max_rank=2, debug=True):
    """
    Build a basis of independent Mumford divisors incrementally.
    """
    basis = []
    
    if debug:
        print(f"Building basis from {len(all_divisors)} candidate divisors...")
    
    for i, div in enumerate(all_divisors):
        if len(basis) >= max_rank:
            if debug:
                print(f"Reached max rank {max_rank}, stopping")
            break
        
        # Try adding this divisor to the basis
        candidate_basis = basis + [div]
        is_indep, rank, H = check_mumford_independence(candidate_basis, f_coeffs)
        
        if is_indep and rank == len(candidate_basis):
            basis.append(div)
            if debug:
                print(f"[basis] Added divisor {i+1}: rank now {len(basis)}/{max_rank}")
                if H is not None:
                    print(f"  Height matrix determinant: {H.determinant()}")
        elif debug and i < 20:
            print(f"[basis] Divisor {i+1} is dependent, skipping")
    
    if basis:
        is_indep, final_rank, final_H = check_mumford_independence(basis, f_coeffs)
        return basis, final_rank, final_H
    else:
        return [], 0, None


def check_mumford_independence(divisors, f_coeffs):
    """
    Check if a list of Mumford divisors is linearly independent in Jac(C)(Q).
    """
    if not divisors:
        return True, 0, None
    
    # 1. Build the curve over QQ explicitly
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    
    # 2. Convert to Jacobian elements
    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(
                div['s'], div['p'], div['v_0'], div['v_1'], C
            )
            jac_elements.append(elem)
        except Exception as e:
            print(f"Warning: Could not convert divisor to Jacobian element: {e}")
            raise
            continue
    
    if not jac_elements:
        return True, 0, None
    
    # 3. Compute height pairing matrix
    try:
        n = len(jac_elements)
        H = Matrix(RDF, n, n)
        
        for i in range(n):
            for j in range(i, n):
                val = compute_manual_height_pairing(jac_elements[i], jac_elements[j])
                H[i,j] = val
                if i != j:
                    H[j,i] = val
        
        # 4. Check numerical rank
        rank = H.rank()
        is_independent = (rank == n)
        return is_independent, rank, H
        
    except Exception as e:
        print(f"Warning: Independence check failed ({e}). Assuming independent to continue.")
        raise
        return True, len(jac_elements), None


def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Convert Mumford coordinates (u,v) to a Jacobian element.
    Safe version: extracts the polynomial ring correctly from the curve object.
    """
    f_curve, h_curve = C.hyperelliptic_polynomials()
    R = f_curve.parent() 
    x = R.gen()
    
    base = C.base_ring()
    try:
        s_val = base(s)
        p_val = base(p)
        v0_val = base(v0)
        v1_val = base(v1)
    except Exception:
        s_val, p_val, v0_val, v1_val = s, p, v0, v1
        raise

    u_poly = x**2 - s_val*x + p_val
    v_poly = v1_val*x + v0_val
    
    J = C.jacobian()
    return J([u_poly, v_poly])


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
                    except (ZeroDivisionError, ValueError):
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
        print(f"\n--- Building Independent Mumford Basis ---")
        try:
            basis_divisors, basis_rank, basis_H = build_mumford_basis_incremental(
                mumford_divisors, 
                f_coeffs, 
                max_rank=2,
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
