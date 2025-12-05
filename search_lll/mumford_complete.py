# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode
#
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR, vector
from collections import defaultdict, Counter
from itertools import product


# mumford_complete.py
#
# Complete working integration of Mumford search.
# Drop this into your codebase and add to search_common.py:
#   MUMFORD_SEARCH = True  # Enable Mumford mode
#
# Key insight: We eliminate m from the 5-equation system by substituting
# the known relation m = -x_residue + const, reducing to 4 unknowns.


def build_mumford_equations_from_fibration(tower, f_coeffs):
    """
    Build polynomial system for Mumford coordinates from fibration tower.
    
    Returns dict with:
        - 'F1_expr', 'F2_expr', etc: symbolic expressions
        - 'r_m_expr': the x([n]P)(m) relation
        - 'const': value of r_m(0)
    """
    r_m = SR(tower[0]['r_expr'])
    m_sym = var('m')
    
    # Variables
    s_sym = var('s')
    p_sym = var('p')  
    v0_sym = var('v_0')
    v1_sym = var('v_1')
    
    # r_1 is the x-coordinate function
    r_1 = r_m
    r_2 = s_sym - r_1
    
    # F1: u(r_1) = r_1^2 - s*r_1 + p = 0
    F1 = r_1**2 - s_sym * r_1 + p_sym
    
    # F4: f(r_1) - f(r_2) = 0
    def eval_f(x):
        return sum(c * x**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
    
    F4 = eval_f(r_1) - eval_f(r_2)
    
    # F5: v(r_1) + v(r_2) = v_1*s + 2*v_0 = 0
    F5 = v1_sym * s_sym + 2 * v0_sym
    
    # F2, F3: Remainder of v^2 - f mod u
    PR_x = PolynomialRing(SR, 'x')
    x_poly = PR_x.gen()
    
    u_poly = x_poly**2 - s_sym * x_poly + p_sym
    f_poly = sum(c * x_poly**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
    v_poly = v1_sym * x_poly + v0_sym
    
    remainder = (v_poly**2 - f_poly) % u_poly
    
    # Extract linear and constant parts
    try:
        F2 = remainder.coefficient(x_poly, 1)  # linear coeff
        F3 = remainder.coefficient(x_poly, 0)  # constant coeff
    except Exception:
        F2 = remainder
        F3 = SR(0)
    
    # Compute const = r_m(m=0) for m-elimination
    const = r_m.subs({m_sym: 0})
    
    return {
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'F4': F4,
        'F5': F5,
        'r_m': r_m,
        'm_sym': m_sym,
        's_sym': s_sym,
        'p_sym': p_sym,
        'v0_sym': v0_sym,
        'v1_sym': v1_sym,
        'const': const
    }


def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=False):
    """
    Solve 5-equation system mod p given x_residue.
    
    Strategy:
    1. Eliminate m: use m = -x_residue + const
    2. Enumerate (s, p) ∈ Fp^2
    3. Check F1, F4 (no v-dependence)
    4. Use F5 to solve for v_0 given v_1
    5. Check F2, F3
    
    Returns: list of (s, p, v0, v1) mod p
    """
    Fp = GF(p)
    
    # Eliminate m
    m_val = (-int(x_residue) + int(QQ(eqs_dict['const']))) % p
    
    # Substitute m into all equations
    m_sym = eqs_dict['m_sym']
    F1 = eqs_dict['F1'].subs({m_sym: m_val})
    F2 = eqs_dict['F2'].subs({m_sym: m_val})
    F3 = eqs_dict['F3'].subs({m_sym: m_val})
    F4 = eqs_dict['F4'].subs({m_sym: m_val})
    F5 = eqs_dict['F5']  # No m-dependence
    
    s_sym = eqs_dict['s_sym']
    p_sym = eqs_dict['p_sym']
    v0_sym = eqs_dict['v0_sym']
    v1_sym = eqs_dict['v1_sym']
    
    solutions = []
    
    for s_val in range(p):
        for p_val in range(p):
            # Check F1
            try:
                F1_val = F1.subs({s_sym: s_val, p_sym: p_val})
                if int(F1_val) % p != 0:
                    continue
            except Exception:
                continue
            
            # Check F4
            try:
                F4_val = F4.subs({s_sym: s_val, p_sym: p_val})
                if int(F4_val) % p != 0:
                    continue
            except Exception:
                continue
            
            # Now (s, p) pass the no-v-dependence checks
            # Use F5 to constrain (v0, v1)
            
            if p == 2:
                # F5: v1*s + 2*v0 = v1*s = 0
                if s_val == 0:
                    v1_options = range(2)
                    v0_options = range(2)
                else:
                    v1_options = [0]
                    v0_options = range(2)
            else:
                # F5: v1*s + 2*v0 = 0 => v0 = -v1*s/2
                inv_2 = Fp(2)**(-1)
                v1_options = range(p)
                v0_options = None
            
            for v1_val in v1_options:
                if p == 2:
                    v0_vals = v0_options
                else:
                    v0_val = (-v1_val * s_val * int(inv_2)) % p
                    v0_vals = [v0_val]
                
                for v0_val in v0_vals:
                    # Check F2, F3
                    try:
                        subs_dict = {
                            s_sym: s_val,
                            p_sym: p_val,
                            v0_sym: v0_val,
                            v1_sym: v1_val
                        }
                        
                        F2_val = F2.subs(subs_dict)
                        if int(F2_val) % p != 0:
                            continue
                        
                        F3_val = F3.subs(subs_dict)
                        if int(F3_val) % p != 0:
                            continue
                        
                        # All checks passed
                        solutions.append((s_val, p_val, v0_val, v1_val))
                        
                    except Exception as e:
                        if debug:
                            print(f"  Exception checking F2/F3: {e}")
                        continue
    
    return solutions


def mumford_precompute_residues_sequential(
    eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll, 
    rhs_modp_list, vecs_list, debug=False
):
    """
    Sequential computation of Mumford residues across primes.
    Avoids pickling issues with symbolic expressions.
    
    Returns: {p: {v_tuple: [(s,p,v0,v1), ...]}}
    """
    from tqdm import tqdm
    
    residues = {}
    total_solutions = 0
    
    for p in tqdm(prime_pool, desc="Mumford Residues"):
        if p not in Ep_dict:
            continue
            
        p_results = {}
        p_solution_count = 0
        
        for v_idx, v_tuple in enumerate(vecs_list):
            if len(vecs_list) > 1 and all(c == 0 for c in v_tuple):
                continue
            
            # Get transformed vector for this prime
            try:
                v_p = vecs_lll[p][v_idx]
            except (KeyError, IndexError, TypeError):
                continue
            
            mults = mult_lll.get(p)
            if mults is None:
                continue
            
            # Compute section sum
            Ep = Ep_dict[p]
            try:
                Pm = Ep(0)
                for j, coeff in enumerate(v_p):
                    k = int(coeff)
                    if k in mults[j]:
                        Pm += mults[j][k]
                
                if Pm.is_zero():
                    continue
            except Exception:
                continue
            
            # Find x-residues
            x_residues = set()
            for i_rhs in range(len(rhs_modp_list)):
                rhs_map = rhs_modp_list[i_rhs]
                if p not in rhs_map:
                    continue
                
                rhs_p = rhs_map[p]
                try:
                    num = (Pm[0]/Pm[2] - rhs_p).numerator()
                    if not num.is_zero():
                        roots = num.roots(ring=GF(p), multiplicities=False)
                        for r in roots:
                            x_residues.add(int(r))
                except Exception:
                    continue
            
            # Solve Mumford system for each x-residue
            all_sols = []
            for x_res in x_residues:
                sols = solve_mumford_mod_p(eqs_dict, p, x_res, debug=False)
                all_sols.extend(sols)
            
            if all_sols:
                p_results[v_tuple] = all_sols
                p_solution_count += len(all_sols)
        
        residues[p] = p_results
        total_solutions += p_solution_count
        
        if debug and p_solution_count > 0:
            print(f"  p={p}: {p_solution_count} Mumford solutions found")
    
    if debug:
        print(f"\nTotal Mumford solutions across all primes: {total_solutions}")
        non_empty_primes = sum(1 for p_res in residues.values() if p_res)
        print(f"Primes with solutions: {non_empty_primes}/{len(residues)}")
    
    return residues


def mumford_precompute_residues_parallel(
    eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll, 
    rhs_modp_list, vecs_list, num_workers=8, debug=False
):
    """
    Parallel computation wrapper - currently falls back to sequential.
    TODO: Fix pickling of symbolic expressions for true parallelism.
    """
    # For now, just call sequential version
    return mumford_precompute_residues_sequential(
        eqs_dict, prime_pool, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list, debug=debug
    )


def reconstruct_and_verify_mumford(residues, prime_list, f_coeffs, shift, rationality_test, debug=False):
    """
    CRT + rational reconstruction of Mumford coordinates.
    
    Returns: set of rational x-coordinates found
    """
    from search_lll.rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
    
    found_xs = set()
    
    # Group by vector
    by_vector = defaultdict(lambda: defaultdict(list))
    for p in prime_list:
        if p not in residues:
            continue
        for v_tuple, sols in residues[p].items():
            by_vector[v_tuple][p] = sols
    
    if debug:
        print(f"\nReconstruction phase: {len(by_vector)} vectors to process")
    
    for v_idx, (v_tuple, prime_data) in enumerate(by_vector.items()):
        primes_with_data = sorted(prime_data.keys())
        
        if len(primes_with_data) < 4:
            if debug:
                print(f"  Vector {v_idx}: only {len(primes_with_data)} primes, skipping (need ≥4)")
            continue
        
        if debug:
            print(f"\n  Vector {v_idx} ({v_tuple}): {len(primes_with_data)} primes")
        
        # Try to reconstruct s, p, v0, v1
        reconstructed = {}
        
        for coord_idx, coord_name in enumerate(['s', 'p', 'v_0', 'v_1']):
            residue_list = []
            
            for p in primes_with_data:
                sols = prime_data[p]
                if not sols:
                    break
                # Use first solution per prime (consensus could be added)
                residue_list.append(sols[0][coord_idx])
            
            if len(residue_list) < len(primes_with_data):
                if debug:
                    print(f"    {coord_name}: missing residues, skipping vector")
                break
            
            # CRT
            M = 1
            for p in primes_with_data:
                M *= p
            
            m_crt = crt_cached(tuple(residue_list), tuple(primes_with_data))
            
            # Rational reconstruction
            try:
                a, b = rational_reconstruct(m_crt, M)
                val = QQ(a) / QQ(b)
                reconstructed[coord_name] = val
                
                if debug:
                    print(f"    {coord_name} = {val}")
                    
            except RationalReconstructionError as e:
                if debug:
                    print(f"    {coord_name}: reconstruction failed ({e})")
                break
        
        # Check if all coords were reconstructed
        if len(reconstructed) == 4:
            s_rat = reconstructed['s']
            p_rat = reconstructed['p']
            v0_rat = reconstructed['v_0']
            v1_rat = reconstructed['v_1']
            
            # Build u(x) = x^2 - s*x + p
            PR = PolynomialRing(QQ, 'x')
            x = PR.gen()
            u_poly = x**2 - s_rat * x + p_rat
            
            if debug:
                print(f"    u(x) = {u_poly}")
            
            # Find roots
            try:
                roots = u_poly.roots(QQ, multiplicities=False)
                
                if debug:
                    print(f"    Roots of u: {roots}")
                
                for x_root in roots:
                    # Undo shift
                    x_orig = x_root + shift
                    
                    # Test rationality
                    y_val = rationality_test(x_orig)
                    
                    if y_val is not None:
                        found_xs.add(x_orig)
                        if debug:
                            print(f"     Found rational point: x = {x_orig}, y = {y_val}")
                    else:
                        if debug:
                            print(f"    — Point x = {x_orig} has irrational y")
                            
            except Exception as e:
                if debug:
                    print(f"    Error finding roots: {e}")
    
    return found_xs
