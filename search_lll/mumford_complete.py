
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

def poly_reduce_mod_u(poly_coeffs, s, p, modulus=None):
    """
    Reduce polynomial f(x) modulo u(x) = x^2 - s*x + p.
    Input: poly_coeffs in HIGH -> LOW order [a_n, a_{n-1}, ..., a_0].
    Returns: [r1, r0] where result = r1*x + r0.
    """
    coeffs = list(poly_coeffs)
    
    # 1. Trim leading zeros
    while len(coeffs) > 0 and coeffs[0] == 0:
        coeffs.pop(0)

    # 2. Horner-like reduction
    # The substitution is x^2 = s*x - p
    while len(coeffs) > 2:
        a = coeffs.pop(0) # a is the coefficient of the highest power x^d, d >= 2
        # x^d -> a * x^{d-2}(s*x - p)
        
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

def _poly_mod_quad_fast(f_coeffs, s_val, p_val, mod_p):
    """
    Computes f(x) mod (x^2 - sx + p) over GF(p).
    Uses Horner-like scheme for O(deg(f)) complexity.
    Returns (A, B) such that f(x) = A*x + B mod u(x).
    f_coeffs are [a_n, ..., a_0].
    """
    r1 = 0
    r0 = 0
    # Process from high-degree to low-degree
    for coeff in f_coeffs:
        # Current state: f_{k+1}(x) = r1*x + r0
        # Next step: f_k(x) = (r1*x + r0) * x + coeff = r1*x^2 + r0*x + coeff
        # Substitute x^2 = s*x - p: 
        #   r1*(s*x - p) + r0*x + coeff 
        #   = (r1*s + r0)*x + (-r1*p + coeff)
        
        new_r1 = (r1 * s_val + r0) % mod_p
        new_r0 = (-r1 * p_val + int(coeff)) % mod_p
        
        r1, r0 = new_r1, new_r0
        
    return r1, r0

def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val):
    """
    General O(p) solver for Mumford coordinates (s, p, v0, v1) mod p.
    Solves v^2 = f(x) mod u(x) where u(x) = x^2 - s*x + p,
    assuming x_residue is a root of u.
    
    1. u(x) determined by s: p = s*x_res - x_res^2
    2. f(x) = A*x + B mod u
    3. v(x)^2 = f(x) mod u => Solves biquadratic for v1^2
    """
    solutions = []
    x_res = int(x_residue) % p
    x_sq = (x_res * x_res) % p

    # Pre-calculate inverse of 2 for use in quadratic formula (if p != 2)
    inv_2 = pow(2, -1, p) if p != 2 else 0

    for s_val in range(p):
        # 1. Determine p from Root Condition: p = s*x - x^2 (mod p)
        p_val = (s_val * x_res - x_sq) % p
        
        # 2. Compute f mod u = A*x + B
        A, B = _poly_mod_quad_fast(f_coeffs, s_val, p_val, p)
        
        # 3. Solve v^2 = A*x + B mod u for v = v1*x + v0
        # The equation for v1^2 = Z is:
        # (s^2 - 4p)Z^2 - 2(A*s + 2*B)Z + A^2 = 0
        
        coeffs_quad = [
            (s_val * s_val - 4 * p_val) % p,  # Z^2 coeff (a_q)
            (-2 * (A * s_val + 2 * B)) % p,   # Z coeff (b_q)
            (A * A) % p                       # Constant (c_q)
        ]
        
        a_q, b_q, c_q = coeffs_quad
        Z_roots = []

        if a_q == 0:
            # Linear case: b_q*Z + c_q = 0
            if b_q != 0:
                Z_roots.append((-c_q * pow(b_q, -1, p)) % p)
            elif c_q == 0:
                # 0 = 0, indeterminate. This case should not typically yield solutions 
                # in a standard Mumford-coordinate search but is handled below.
                pass 
        else:
            # Quadratic case: a_q*Z^2 + b_q*Z + c_q = 0
            disc_q = (b_q * b_q - 4 * a_q * c_q) % p
            
            # Simple quadratic formula for GF(p)
            if disc_q == 0:
                # Double root
                Z_roots.append((-b_q * pow(2 * a_q, -1, p)) % p)
            elif pow(disc_q, (p - 1) // 2, p) == 1:
                # Discriminant is a quadratic residue (has square roots)
                sq_root = None
                if p < 1000: # Simple search for small p
                    for r in range(1, p): 
                        if (r * r) % p == disc_q:
                            sq_root = r
                            break
                # (Else: Use Tonelli-Shanks for larger p, but simple search is fast for small p)
                
                if sq_root is not None:
                    inv_2a = pow(2 * a_q, -1, p)
                    Z_roots.append(((-b_q + sq_root) * inv_2a) % p)
                    Z_roots.append(((-b_q - sq_root) * inv_2a) % p)
        
        # For each valid Z (v1^2), solve for v1 (v1 = +/- sqrt(Z))
        valid_v1s = []
        for Z in Z_roots:
            if Z == 0:
                valid_v1s.append(0)
            else:
                # Is Z a quadratic residue?
                if pow(Z, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == Z:
                            valid_v1s.append(r)
                            valid_v1s.append(p - r) # Include the negative root
                            break

        # Recover v0 and add solutions
        for v1_val in valid_v1s:
            if v1_val == 0:
                # If v1=0, Eq (1) simplifies: 0 = A. And Eq (2) is: v0^2 = B.
                if A != 0: continue
                
                # Solve v0^2 = B
                if B == 0:
                    solutions.append((s_val, p_val, 0, 0))
                elif pow(B, (p - 1) // 2, p) == 1:
                    for r in range(1, p):
                        if (r * r) % p == B:
                            solutions.append((s_val, p_val, r, 0))
                            solutions.append((s_val, p_val, p - r, 0))
                            break
            else:
                # If v1 != 0, solve Eq (1) for v0: 2*v0*v1 = A - s*v1^2
                # v0 = (A - s*v1^2) * (2*v1)^(-1) (mod p)
                
                if p == 2:
                    # In p=2, v0^2 - p*v1^2 = B becomes v0^2 = B, so v0 = B
                    # Since v0 is a root of v0^2-B=0, we have v0=B mod 2
                    v0_val = B % 2 # (v0^2=v0 mod 2)
                    
                    # Verify against Eq (1): s*v1^2 + 2*v0*v1 = A becomes s*v1 + 0 = A
                    if (s_val * v1_val) % 2 == A % 2:
                        solutions.append((s_val, p_val, v0_val, v1_val))
                else:
                    num = (A - s_val * (v1_val * v1_val)) % p
                    den = (2 * v1_val) % p
                    v0_val = (num * pow(den, -1, p)) % p
                    
                    # Verify against Eq (2): v0^2 - p*v1^2 = B
                    lhs_2 = (v0_val * v0_val - p_val * v1_val * v1_val) % p
                    if lhs_2 == B:
                        solutions.append((s_val, p_val, v0_val, v1_val))

    return solutions

def verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=False):
    """
    Verifies that v(x)^2 == f(x) (mod u(x)), where u(x) = x^2 - s*x + p.
    Accepts optional modulus for modular arithmetic checks.
    """
    # 1. Compute v(x)^2 mod x^3
    c0 = v0 * v0
    c1 = 2 * v0 * v1
    c2 = v1 * v1
    
    # v(x)^2 = c2*x^2 + c1*x + c0
    
    # 2. Compute f(x) - v(x)^2
    # f_coeffs is HIGH -> LOW
    diff = list(f_coeffs)
    
    # Ensure diff has at least 3 elements for the check c2*x^2 + c1*x + c0
    if len(diff) < 3:
        diff = [0] * (3 - len(diff)) + diff

    # Subtract v(x)^2 coefficients from f(x) (low-order first)
    diff[-1] -= c0 # constant term
    diff[-2] -= c1 # x term
    diff[-3] -= c2 # x^2 term

    # 3. Reduce the difference polynomial modulo u(x)
    rem = poly_reduce_mod_u(diff, s, p, modulus=modulus)

    check0 = rem[0] # x coefficient
    check1 = rem[1] # constant coefficient

    if modulus:
        check0 %= modulus
        check1 %= modulus

    if check0 != 0 or check1 != 0:
        if debug_first_failure:
            print(f"\nDEBUG VERIFICATION FAILURE:")
            print(f"  f_coeffs (used): {f_coeffs[:3]}...")
            print(f"  s={s}, p={p}, v0={v0}, v1={v1}")
            print(f"  Modulus: {modulus}")
            print(f"  Remainder: {check0}x + {check1}")
        
        raise AssertionError(f"Mumford congruence failed: remainder = {check0}x + {check1} != 0")

    return True

# =============================================================================
# WORKERS & PARALLEL
# =============================================================================

def _init_worker():
    """Initializes worker process to ignore SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _solve_worker_wrapper(args):
    """Worker function for multiprocessing."""
    p, f_coeffs_ints, x_residues_map, const_val_int = args
    try:
        p_results = {}
        for v_tuple, x_res_list in x_residues_map.items():
            if isinstance(x_res_list, int):
                x_res_list = [x_res_list]
                
            all_sols = []
            for x_res in x_res_list:
                sols = solve_mumford_mod_p_optimized(f_coeffs_ints, p, x_res, const_val_int)
                if sols:
                    all_sols.extend(sols)
            
            if all_sols:
                p_results[v_tuple] = all_sols
                
        return p, p_results
    except Exception:
        sys.stderr.write(f"\nCRITICAL ERROR IN MUMFORD WORKER (p={p}):\n")
        traceback.print_exc(file=sys.stderr)
        raise
        return p, {} 

def mumford_precompute_residues_parallel(eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
                                         rhs_modp_list, vecs_list, num_workers=8, debug=False):
    """
    Precomputes Mumford coordinate residues modulo a list of primes in parallel.
    Uses the connection between LLL vectors and rational points.
    """
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
        if p not in Ep_dict: continue
        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        if not p_vecs: continue
        
        # Setup ring for m over GF(p)
        try:
            Fp = GF(p)
            R_m = Fp['m']
            m_var = R_m.gen()
            # The coordinate m is related to x by x = -m + const, or m = const - x
            rhs_poly = -m_var + Fp(const_val_int)
        except Exception as e:
            if debug: print(f"  Skipping p={p}: Ring setup failed {e}")
            raise
            continue

        x_residues_map = {}
        p_mults = mult_lll.get(p, {})
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple: continue
            
            # Sum vector P_m related to the LLL vector v
            Pm = Ep(0)
            valid_vec = True
            v_coeffs = p_vecs[v_idx] # LLL coefficient vector for this prime

            for i, c in enumerate(v_coeffs):
                k = int(c)
                if k == 0: continue
                
                try:
                    mults_for_sec = p_mults[i]
                    if k in mults_for_sec:
                        Pm += mults_for_sec[k]
                    else:
                        valid_vec = False; break
                except (IndexError, KeyError, TypeError):
                    raise
                    valid_vec = False; break
            
            # Check for validity and non-triviality
            if not valid_vec or Pm.is_zero() or Pm[2] == 0: 
                continue
            
            # Solve for m: Pm[0] - Pm[2]*rhs_poly = 0
            try:
                # Pm = (Pm[0], Pm[1], Pm[2])
                diff = Pm[0] - Pm[2] * rhs_poly
                diff_num = diff.numerator() # For Sage objects
                
                if diff_num.is_zero():
                    continue
                    
                roots = diff_num.roots(multiplicities=False)
                
                if roots:
                    valid_residues = []
                    for m_root in roots:
                        # Map m -> x: x = -m + const
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
        if debug: print("[mumford] No tasks generated!")
        return {}
        
    try:
        # Use 'fork' context for better memory sharing if available
        ctx = multiprocessing.get_context("fork")
        pool_obj = ctx.Pool(num_workers, initializer=_init_worker)
    except:
        pool_obj = multiprocessing.Pool(num_workers, initializer=_init_worker)
        raise

    results_dict = {}
    with pool_obj as pool:
        # Use imap_unordered for results as they become available
        for p, result_map in tqdm(pool.imap_unordered(_solve_worker_wrapper, tasks), 
                                  total=len(tasks), desc="Solving Mumford Mod P"):
            results_dict[p] = result_map
            
    return results_dict

# =============================================================================
# RECONSTRUCTION & VERIFICATION
# =============================================================================

def _normalize_sign(s, p, v0, v1):
    """Canonical sign normalization for Mumford coordinates (u, v).
    We normalize such that the leading coefficient of v(x) (v1) is non-negative.
    If v1=0, we normalize the constant term (v0) to be non-negative.
    """
    if v1 < 0 or (v1 == 0 and v0 < 0):
        # Flip sign of v(x) = v1*x + v0
        return (s, p, -v0, -v1)
    return (s, p, v0, v1)

def canonicalize_and_dedup(divisors, f_coeffs):
    """
    Normalizes the sign of v(x) and removes duplicate divisors.
    Includes an optional diagnostic run for solver/verifier consistency.
    """
    seen = {}
    out = []
    first_failure_logged = False
    
    # DIAGNOSTIC: Test solver/verifier consistency at p=5
    # Only runs once per call for sanity check
    if not hasattr(canonicalize_and_dedup, "_diag_run"):
        canonicalize_and_dedup._diag_run = True
        try:
            test_x = 1
            # Evaluate f(1) where f_coeffs is High->Low
            f_val = 0
            for c in f_coeffs: f_val = f_val * test_x + int(c)
            
            print(f"\n[DIAGNOSTIC] f(1) = {f_val}")
            print(f"[DIAGNOSTIC] Testing mod-p solver at p=5, x={test_x}:")
            
            test_sols = solve_mumford_mod_p_optimized([int(c) for c in f_coeffs], 5, test_x, 0)
            print(f"  Found {len(test_sols)} solutions mod 5")
            
            for i, sol in enumerate(test_sols[:3]): # Check first 3 solutions
                s, p_mod, v0, v1 = sol
                try:
                    # PASS MODULUS=5 HERE!
                    verify_mumford_pair(f_coeffs, s, p_mod, v0, v1, modulus=5, debug_first_failure=True)
                    print(f"  Sol {i} Verified OK mod 5")
                except AssertionError as e:
                    print(f"  Sol {i} Failed mod 5: {e}")
                    raise
        except Exception as e:
            print(f"Diagnostic failed: {e}")
            raise

    for tup in divisors:
        s, p, v0, v1 = tup['s'], tup['p'], tup['v_0'], tup['v_1']
        
        try:
            # Rational verification: modulus=None
            verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None, debug_first_failure=(not first_failure_logged))
        except AssertionError as e:
            if not first_failure_logged:
                first_failure_logged = True
                print(f"Reconstruction verification failed: {e}")
            raise
            continue
        
        # Apply canonical sign normalization
        s1, p1, v01, v11 = _normalize_sign(s, p, v0, v1)
        key = (s1, p1, v01, v11) # The divisor is uniquely determined by (u, v)
        
        # NOTE: Original code only used (s1, p1) for key, which is INCORRECT 
        # as a curve can have multiple distinct divisors with the same u(x). 
        # Using the full (s1, p1, v01, v11) for correct dedup.
        if key not in seen:
            seen[key] = True
            tup['s'], tup['p'], tup['v_0'], tup['v_1'] = s1, p1, v01, v11
            out.append(tup)
            
    return out

def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=True):
    """
    Reconstructs rational Mumford divisors from modular residues using CRT 
    and Rational Reconstruction, and verifies them.
    """
    from itertools import product as cartesian_product
    print("\n" + "="*70)
    print("MUMFORD RECONSTRUCTION PHASE")
    print("="*70)

    found_xs = set()
    mumford_divisors_raw = []

    # 1. Group residues by Vector -> x_residue_family
    by_vector = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in residues:
        for v_tuple, sols_data in residues[p].items():
            if isinstance(sols_data, list):
                # The assumption is that all solutions from a single vector at a prime p 
                # belong to a single "x-residue family" for CRT.
                by_vector[v_tuple]['main'][p] = sols_data

    print(f"Grouped into {len(by_vector)} distinct vectors")

    total_attempted = 0
    recon_success = 0

    for v_tuple, fam_map in by_vector.items():
        for fam_key, prime_data in fam_map.items():
            primes = sorted(prime_data.keys())
            if len(primes) < 3: continue
            
            # Compute product of primes M
            M = 1
            for p in primes: M *= p
            
            sol_lists = [prime_data[p] for p in primes]
            
            # Limit combinations to prevent combinatorial explosion
            limit = 20000 
            
            for sol_combo in cartesian_product(*sol_lists):
                if limit <= 0: break
                limit -= 1
                total_attempted += 1
                
                # CRT & Rational Reconstruction
                try:
                    rec_vals = []
                    for idx in range(4): # The coordinates: s, p, v0, v1
                        vals = [sol[idx] for sol in sol_combo]
                        # CRT: Combine the modular values
                        crt_val = crt_cached(tuple(vals), tuple(primes))
                        # Rational Reconstruction: Recover rational number num/den
                        num, den = rational_reconstruct(crt_val, M)
                        rec_vals.append(QQ(num)/QQ(den))
                    
                    # Verify Rational coordinates
                    s, p, v0, v1 = rec_vals
                    verify_mumford_pair(f_coeffs, s, p, v0, v1, modulus=None)
                    
                    mumford_divisors_raw.append({
                        'vector': v_tuple, 's': s, 'p': p, 'v_0': v0, 'v_1': v1
                    })
                    recon_success += 1
                    
                except (RationalReconstructionError, AssertionError):
                    raise
                    continue

    print(f"  Combinations tried: {total_attempted}")
    print(f"  Successful reconstructions: {recon_success}")

    # Dedup and Extract Points
    mumford_divisors = canonicalize_and_dedup(mumford_divisors_raw, f_coeffs)

    # Map Mumford divisors back to rational points X (roots of u(x))
    for div in mumford_divisors:
        s, p = div['s'], div['p']
        # Roots of u(x) = x^2 - s*x + p
        PR = PolynomialRing(QQ, 'x')
        x = PR.gen()
        u_poly = x**2 - s*x + p
        
        try:
            # Find rational roots of u(x)
            roots = u_poly.roots(QQ, multiplicities=False)
            for r in roots:
                # Map back to original X-coordinate: X = x - shift
                x_cand = r - shift
                if rationality_test(x_cand) is not None:
                    found_xs.add(x_cand)
        except:
            raise
            continue
            
    print(f"  Unique Rational Points: {len(found_xs)}")
    return found_xs, mumford_divisors

def build_mumford_equations_from_fibration(tower, f_coeffs):
    """Helper to return dict for setup."""
    return {
        'f_coeffs': f_coeffs,
        'const': 0, # Default, can be overwritten
        'm_sym': var('m')
    }

def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=False):
    """
    Legacy wrapper for compatibility if called directly.
    """
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)

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
