# mumford_integration.py
#
# Integration layer between mumford_search.py and the main search pipeline.
# Wires up the Mumford coordinate search for genus-2 Jacobians.

from sage.all import QQ, ZZ, GF, PolynomialRing, var, SR
from collections import defaultdict
import mumford_search

def build_mumford_system_from_tower(tower, f_coeffs, shift):
    """
    Build the 5-equation polynomial system for Mumford search.
    
    Args:
        tower: fibration tower data (list of dicts with 'r_expr', 'f_i')
        f_coeffs: coefficients of original genus-2 curve y^2 = f(x)
        shift: the shift applied to x-coordinates
    
    Returns:
        dict with polynomial equations F1...F5 and variable info
    """
    # Extract r_m (the intersection locus)
    r_m = SR(tower[0]['r_expr'])
    m_sym = var('m')
    
    # r_1 = x([n]P)(m) = -m + const (generically)
    # We need the explicit form
    r_1 = r_m  # This is x([n]P)(m) as a function of m
    
    # Unknowns: m, s, p, v_0, v_1
    # where u(x) = x^2 - s*x + p
    # and v(x) = v_1*x + v_0
    
    s_sym = var('s')
    p_sym = var('p')
    v0_sym = var('v_0')
    v1_sym = var('v_1')
    
    # F1: r_1^2 - s*r_1 + p = 0 (r_1 is root of u)
    F1 = r_1**2 - s_sym*r_1 + p_sym
    
    # r_2 = s - r_1 (the conjugate root)
    r_2 = s_sym - r_1
    
    # F4: f(r_1) - f(r_2) = 0 (hyperelliptic involution)
    def eval_poly(coeffs, x):
        result = 0
        for i, c in enumerate(reversed(coeffs)):
            result = result * x + c
        return result
    
    f_r1 = eval_poly(f_coeffs, r_1)
    f_r2 = eval_poly(f_coeffs, r_2)
    F4 = f_r1 - f_r2
    
    # F5: v(r_1) + v(r_2) = 0
    # v(x) = v_1*x + v_0, so v(r_1) + v(r_2) = v_1*(r_1+r_2) + 2*v_0 = v_1*s + 2*v_0
    F5 = v1_sym * s_sym + 2 * v0_sym
    
    # F2, F3: remainder of v^2 - f(x) mod u(x)
    # v^2 = (v_1*x + v_0)^2 = v_1^2*x^2 + 2*v_1*v_0*x + v_0^2
    # We need to reduce x^2 using u(x) = x^2 - s*x + p
    # So x^2 = s*x - p
    
    # v^2 mod u = v_1^2*(s*x - p) + 2*v_1*v_0*x + v_0^2
    #           = (v_1^2*s + 2*v_1*v_0)*x + (v_0^2 - v_1^2*p)
    
    # f(x) mod u: We need to reduce f(x) (degree 5 or 6) mod (x^2 - s*x + p)
    # This is more complex. Let's use Sage's polynomial division.
    
    PR_x = PolynomialRing(SR, 'x')
    x_var = PR_x.gen()
    
    # Build u(x) in this ring
    u_poly = x_var**2 - s_sym*x_var + p_sym
    
    # Build f(x)
    f_poly = sum(c * x_var**i for i, c in enumerate(reversed(f_coeffs)))
    
    # Build v^2
    v_squared = (v1_sym * x_var + v0_sym)**2
    
    # Compute (v^2 - f) mod u
    diff = v_squared - f_poly
    remainder = diff % u_poly
    
    # Extract coefficients (should be linear: a*x + b)
    try:
        remainder_coeffs = [remainder.coefficient(x_var, i) for i in range(2)]
        F2 = remainder_coeffs[1]  # coeff of x
        F3 = remainder_coeffs[0]  # constant term
    except Exception as e:
        print(f"ERROR: Could not extract remainder coefficients: {e}")
        # Fallback: just use the remainder itself
        F2 = remainder
        F3 = 0
    
    return {
        'equations': {
            'F1': F1,
            'F2': F2,
            'F3': F3,
            'F4': F4,
            'F5': F5
        },
        'variables': {
            'm': m_sym,
            's': s_sym,
            'p': p_sym,
            'v_0': v0_sym,
            'v_1': v1_sym
        },
        'r_m': r_m,
        'shift': shift
    }


def solve_mumford_system_modp(system_dict, prime, x_residue):
    """
    Solve the 5-equation Mumford system mod p for a given x_residue.
    
    Args:
        system_dict: output from build_mumford_system_from_tower
        prime: prime modulus
        x_residue: the value of x([n]P)(m) mod p (determines m mod p)
    
    Returns:
        list of solutions [(s, p, v_0, v_1), ...] mod prime
    """
    p = int(prime)
    Fp = GF(p)
    
    # Extract equations
    eqs = system_dict['equations']
    vars_dict = system_dict['variables']
    r_m = system_dict['r_m']
    
    # We need to eliminate m using r_1 = x_residue
    # From r_m expression, solve for m in terms of the residue
    # Typically r_m = -m + const, so m = -x_residue + const
    
    m_sym = vars_dict['m']
    
    # Evaluate const = r_m(m=0)
    try:
        const = r_m.subs({m_sym: 0})
        const_mod_p = int(QQ(const)) % p
    except Exception:
        const_mod_p = 0
    
    # m = -x_residue + const (mod p)
    m_val = (-int(x_residue) + const_mod_p) % p
    
    # Now substitute m into all equations
    substituted_eqs = {}
    for name, eq in eqs.items():
        try:
            eq_sub = eq.subs({m_sym: m_val})
            substituted_eqs[name] = eq_sub
        except Exception as e:
            print(f"ERROR substituting into {name}: {e}")
            return []
    
    # Now we have 5 equations in 4 unknowns: s, p, v_0, v_1
    # Use the simple brute-force solver from mumford_search.py
    
    solutions = []
    
    # Strategy: iterate over s, p, then solve for v_0, v_1
    for s_val in range(p):
        for p_val in range(p):
            # Check F1: should be 0
            try:
                F1_val = substituted_eqs['F1'].subs({
                    vars_dict['s']: s_val,
                    vars_dict['p']: p_val
                })
                if int(F1_val) % p != 0:
                    continue
            except Exception:
                continue
            
            # Check F4: should be 0
            try:
                F4_val = substituted_eqs['F4'].subs({
                    vars_dict['s']: s_val,
                    vars_dict['p']: p_val
                })
                if int(F4_val) % p != 0:
                    continue
            except Exception:
                continue
            
            # Now solve F5 for v_0 given s
            # F5: v_1*s + 2*v_0 = 0
            # So v_0 = -v_1*s/2 (if p != 2)
            
            if p == 2:
                # Special handling for p=2
                # F5 becomes: v_1*s = 0 (since 2=0 mod 2)
                # So either s=0 or v_1=0
                v1_vals = range(p)
                if s_val == 0:
                    v0_vals = range(p)
                else:
                    v1_vals = [0]
                    v0_vals = range(p)
            else:
                inv_2 = Fp(2)**(-1)
                v1_vals = range(p)
                v0_vals = None  # computed per v1
            
            for v1_val in v1_vals:
                if p == 2:
                    v0_candidates = v0_vals
                else:
                    # v_0 = -v_1*s/2 mod p
                    v0_val = (- v1_val * s_val * int(inv_2)) % p
                    v0_candidates = [v0_val]
                
                for v0_val in v0_candidates:
                    # Check F2 and F3
                    try:
                        F2_val = substituted_eqs['F2'].subs({
                            vars_dict['s']: s_val,
                            vars_dict['p']: p_val,
                            vars_dict['v_0']: v0_val,
                            vars_dict['v_1']: v1_val
                        })
                        if int(F2_val) % p != 0:
                            continue
                        
                        F3_val = substituted_eqs['F3'].subs({
                            vars_dict['s']: s_val,
                            vars_dict['p']: p_val,
                            vars_dict['v_0']: v0_val,
                            vars_dict['v_1']: v1_val
                        })
                        if int(F3_val) % p != 0:
                            continue
                        
                        # All checks passed!
                        solutions.append((s_val, p_val, v0_val, v1_val))
                    except Exception:
                        continue
    
    return solutions


def mumford_search_worker(p, system_dict, vecs, rhs_modp_list, mult_lll):
    """
    Worker function for parallel Mumford search at prime p.
    
    Replaces the standard residue computation worker.
    """
    result_for_p = {}
    
    # For each search vector, compute x-coordinate residues
    for v_tuple in vecs:
        # Compute x([n]P) mod p using the section multiples
        # This is the same as before
        
        # Then for each x_residue, solve the Mumford system
        x_residues = []  # TODO: compute from mult_lll
        
        mumford_solutions = []
        for x_res in x_residues:
            sols = solve_mumford_system_modp(system_dict, p, x_res)
            mumford_solutions.extend(sols)
        
        result_for_p[v_tuple] = mumford_solutions
    
    return p, result_for_p


def reconstruct_mumford_element(residues_by_prime, prime_list, f_coeffs):
    """
    Use CRT + rational reconstruction to lift Mumford coordinates.
    
    Args:
        residues_by_prime: dict {p: [(s, p, v0, v1), ...]}
        prime_list: list of primes used
        f_coeffs: curve coefficients for verification
    
    Returns:
        list of rational Mumford elements [(u, v), ...] that passed verification
    """
    from sage.all import PolynomialRing
    from .rational_arithmetic import crt_cached, rational_reconstruct
    
    verified_elements = []
    
    # Need to match up residues across primes
    # This is complex - for now, try all CRT combinations
    
    # Simpler approach: assume we have consensus residues per prime
    # and just reconstruct each coordinate independently
    
    for coord_idx in range(4):  # s, p, v0, v1
        residues = []
        for p in prime_list:
            sols = residues_by_prime.get(p, [])
            if sols:
                # Take first solution (or use consensus)
                residues.append(sols[0][coord_idx])
            else:
                break
        
        if len(residues) != len(prime_list):
            continue
        
        # CRT
        M = 1
        for p in prime_list:
            M *= p
        
        val_mod_M = crt_cached(tuple(residues), tuple(prime_list))
        
        # Rational reconstruction
        try:
            num, den = rational_reconstruct(val_mod_M, M)
            rational_val = QQ(num) / QQ(den)
            # Store this coordinate
        except Exception:
            continue
    
    # TODO: Build u(x), v(x) polynomials and verify
    # For now, return placeholder
    
    return verified_elements


def mumford_to_rational_point(u_poly, v_poly, f_coeffs, shift):
    """
    Convert Mumford coordinates (u, v) to rational point on original curve.
    
    Args:
        u_poly: monic quadratic u(x) = x^2 - s*x + p
        v_poly: linear v(x) = v_1*x + v_0
        f_coeffs: curve coefficients
        shift: x-coordinate shift
    
    Returns:
        (x, y) rational point or None
    """
    from sage.all import PolynomialRing
    
    PR = PolynomialRing(QQ, 'x')
    x = PR.gen()
    
    # Find roots of u(x)
    roots = u_poly.roots(QQ, multiplicities=False)
    
    if not roots:
        return None
    
    # Take first root (both should give points)
    x_val = roots[0]
    
    # Evaluate v at this root
    y_val = v_poly(x_val)
    
    # Undo shift
    x_orig = x_val + shift
    
    # Verify on original curve
    f_val = sum(c * x_orig**i for i, c in enumerate(reversed(f_coeffs)))
    
    if y_val**2 == f_val:
        return (x_orig, y_val)
    else:
        return None
