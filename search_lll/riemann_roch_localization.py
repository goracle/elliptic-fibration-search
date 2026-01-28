from sage.all import Matrix, vector, ZZ, QQ, GF, PolynomialRing
import sys
import itertools
from sage.all import Matrix, vector, ZZ, GF, PolynomialRing

# search_lll/riemann_roch_localization.py








def get_rr_basis(n_pole, f_poly, p):
    if f_poly.degree() != 5:
        raise ValueError("RR Localizer requires odd-degree genus 2 model (deg f=5).")
    basis = []
    for i in range(n_pole // 2 + 1):
        basis.append({'pwr': i, 'has_y': False, 'pole': 2*i})
    for i in range((n_pole - 5) // 2 + 1):
        basis.append({'pwr': i, 'has_y': True, 'pole': 2*i + 5})
    
    g = 2
    if n_pole >= 2*g - 1:
        expected_dim = n_pole - g + 1
        if len(basis) != expected_dim:
            raise ArithmeticError(f"RR Basis Error: Got dim {len(basis)}, expected {expected_dim}")
    return basis

def resolve_log_from_rr_decomposition(roots_data, root_to_idx, fb_y_cache, poly_a, poly_b, p):
    """
    Resolves the coefficients b_j such that [Q] = sum b_j [P_j] in the Jacobian.
    Uses the fact that the principal divisor (h) = [Q] + sum m_i [P_i] - n[inf] = 0.
    """
    K = GF(p)
    # The relation from the principal divisor h is: Q + sum(m_i * P_i) = 0
    # Therefore [Q] = -sum(m_i * P_i).
    relation_vec = [0] * len(root_to_idx)
    
    for r, mult in roots_data:
        r_val = int(r)
        idx = root_to_idx[r_val]
        b_val = poly_b(r_val)
        
        # Calculate y-coordinate from the function h(x,y) = A(x) + B(x)y = 0
        if b_val == 0:
            # Weierstrass point (y=0). P = -P. 
            # Contribution to [Q] is -mult * [P_idx]
            relation_vec[idx] -= mult
        else:
            y_calc = (-(poly_a(r_val)) * b_val**-1)
            y_fb = K(fb_y_cache[r_val])
            
            if y_calc == y_fb:
                # The point in the divisor is P_idx. [Q] += -mult * [P_idx]
                relation_vec[idx] -= mult
            else:
                # The point in the divisor is -P_idx. [Q] += -mult * [-P_idx] = mult * [P_idx]
                relation_vec[idx] += mult
                
    return relation_vec

def localize_target_via_rr(target_div, factor_base_roots, f_poly, p, n_pole=7):
    p_int = int(p)
    K = GF(p_int)
    R = PolynomialRing(K, 'x')
    u_q, v_q = target_div[0], target_div[1]
    basis = get_rr_basis(n_pole, f_poly, p_int)
    
    constraints = []
    for b in basis:
        func_mod = (R.gen()**b['pwr'] * (v_q if b['has_y'] else 1)) % u_q
        constraints.append([func_mod[0], func_mod[1]])
        
    M = Matrix(K, constraints).transpose()
    kernel = M.right_kernel().basis()
    k_dim = len(kernel)
    
    if k_dim == 0: return None, None, None, None

    candidates = []
    # Enumeration logic with ChatGPT's entropy-steering recommendations
    if k_dim <= 3:
        for coeffs in itertools.product([-2, -1, 0, 1, 2], repeat=k_dim):
            if all(c == 0 for c in coeffs): continue
            candidates.append(sum(coeffs[i] * vector(ZZ, kernel[i]) for i in range(k_dim)))
    elif k_dim <= 5:
        for coeffs in itertools.product([-1, 0, 1], repeat=k_dim):
            if all(c == 0 for c in coeffs): continue
            candidates.append(sum(coeffs[i] * vector(ZZ, kernel[i]) for i in range(k_dim)))
    else:
        weights = Matrix.diagonal(ZZ, [b['pole'] + 1 for b in basis])
        lifted = Matrix(ZZ, [[int(c) for c in b] for b in kernel])
        lll_res = (lifted * weights).LLL() * weights.inverse()
        candidates = [lll_res.row(i) for i in range(lll_res.nrows())]

    for vec in candidates:
        poly_a, poly_b = R(0), R(0)
        for i, b in enumerate(basis):
            coeff = K(vec[i])
            if b['has_y']: poly_b += coeff * R.gen()**b['pwr']
            else:          poly_a += coeff * R.gen()**b['pwr']
            
        norm_poly = poly_a**2 - poly_b**2 * f_poly
        if (norm_poly % u_q) != 0: continue 
            
        final_poly = (norm_poly // u_q).monic()
        # FIXED: multiplicity -> multiplicities
        roots_data = final_poly.roots(multiplicities=True)
        
        if sum(m for r, m in roots_data) == final_poly.degree():
            if all(r in factor_base_roots for r, m in roots_data):
                return roots_data, poly_a, poly_b, vec
                    
    return None, None, None, None

def localize_wrapper(args):
    """
    Wrapper for parallel execution of localize_target_via_rr.
    args: (target_div, factor_base_roots, f_poly, p, n_pole)
    """
    try:
        target_div, factor_base_roots, f_poly, p, n_pole = args
        return localize_target_via_rr(target_div, factor_base_roots, f_poly, p, n_pole=n_pole)
    except Exception as e:
        # Allow exception to propagate to parent
        raise e
