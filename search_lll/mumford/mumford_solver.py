# Needs imports:
from sage.all import QQ, ZZ, GF, PolynomialRing
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL
from .mumford_core import _poly_mod_quad_fast

def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=DEBUG):
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)


def _mumford_doubling_mod_p_internal(u_coeffs, v_coeffs, f_coeffs, p, debug=False):
    """
    Robust modular doubling for genus-2 Mumford divisors.

    Inputs:
      - u_coeffs, v_coeffs: lists of integers (residues mod p) representing the Mumford
        polynomials. They may be given highest-first or lowest-first (this function
        detects and normalizes).
      - f_coeffs: list (highest->lowest) of curve polynomial coefficients (integers/QQ).
      - p: prime

    Returns:
      (u_2p_coeffs, v_2p_coeffs) where both lists are integers mod p in HIGH->LOW order,
      or (None, None) if prime should be skipped (bad reduction / bad arithmetic).
    """
    if p == 2:
        return None, None

    try:
        Fp = GF(p)
        R_Fp = PolynomialRing(Fp, 'x')
    except Exception:
        raise
        return None, None

    # Build f(x) over Fp using the same helper (safe conversion)
    try:
        f_poly_Fp = _poly_from_coeffs_qq(R_Fp, [Fp(QQ(c)) for c in f_coeffs])
    except Exception:
        # If conversion fails, skip this prime
        if debug:
            print(f"[MOD-DBL] cannot build f_poly mod {p}")
        raise
        return None, None

    # If the curve is singular mod p, skip
    try:
        C_Fp = HyperellipticCurve(f_poly_Fp, R_Fp(0))
    except ValueError:
        if debug:
            print(f"[MOD-DBL] singular curve at p={p}")
        raise
        return None, None

    J_Fp = C_Fp.jacobian()

    # helper: try to interpret a coeff-list as either highest->lowest or lowest->highest
    def _make_poly_from_coeff_list(coeff_list, assume_highest_first):
        """
        Return polynomial over R_Fp or raise if input invalid.
        If assume_highest_first==True, coeff_list is highest->lowest; convert to lowest->highest for constructor.
        """
        if assume_highest_first:
            lst = list(map(Fp, coeff_list))[::-1]   # to lowest->highest
        else:
            lst = list(map(Fp, coeff_list))         # already lowest->highest
        # strip leading zeros in the highest-first sense (i.e., trailing zeros now)
        # ensure at least one coefficient (constant 0 allowed)
        while len(lst) > 1 and lst[-1] == 0:
            lst.pop()
        return R_Fp(lst)

    # try both orientations for inputs (defensive)
    tried = []
    for assume_high in (True, False):
        try:
            u_poly_Fp = _make_poly_from_coeff_list(u_coeffs, assume_high)
            v_poly_Fp = _make_poly_from_coeff_list(v_coeffs, assume_high)
        except Exception as e:
            tried.append((assume_high, "make failed", str(e)))
            raise
            continue

        # canonicalize: require u to be non-zero and monic. If not monic, try to scale.
        if u_poly_Fp.is_zero():
            tried.append((assume_high, "u_zero", None))
            continue

        lc = u_poly_Fp.leading_coefficient()
        if lc != 1:
            # try to normalize to monic (scale by inverse lc)
            try:
                inv_lc = lc**(-1)
                u_poly_Fp = (u_poly_Fp * inv_lc)
                v_poly_Fp = (v_poly_Fp * inv_lc)  # scale v accordingly (safe mod p)
            except Exception:
                tried.append((assume_high, "nonmonic_not_normalizable", lc))
                raise
                continue

        # reduce v modulo u to enforce deg v < deg u
        try:
            v_poly_Fp = v_poly_Fp % u_poly_Fp
        except Exception as e:
            tried.append((assume_high, "reduce_failed", str(e)))
            raise
            continue

        # quick Mumford test: (v^2 - f) % u == 0
        try:
            rem = (v_poly_Fp**2 - f_poly_Fp).quo_rem(u_poly_Fp)[1]
            if rem != 0:
                tried.append((assume_high, "mumford_test_fail", rem))
                continue
        except ZeroDivisionError:
            tried.append((assume_high, "quo_rem_zero_divisor", None))
            raise
            continue
        except Exception as e:
            tried.append((assume_high, "quo_rem_exc", str(e)))
            raise
            continue

        # If we reach here, inputs interpreted under this orientation form a valid divisor mod p
        # proceed to doubling
        try:
            D_mod_p = J_Fp([u_poly_Fp, v_poly_Fp])
        except (ValueError, TypeError) as e:
            tried.append((assume_high, "jacobian_construct_fail", str(e)))
            raise
            continue

        try:
            D_doubled = 2 * D_mod_p
        except (ValueError, ArithmeticError, ZeroDivisionError) as e:
            tried.append((assume_high, "doubling_failed", str(e)))
            raise
            return None, None

        # extract coefficients and normalize result
        u_poly_res = D_doubled[0]
        v_poly_res = D_doubled[1]

        # ensure u_poly_res is monic and deg >= 1 (degree for genus-2: usually 2)
        if u_poly_res.is_zero():
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] doubled u is zero mod {p} (assume_high={assume_high})")
            return None, None

        # normalize to monic
        lc_res = u_poly_res.leading_coefficient()
        if lc_res != 1:
            try:
                inv_lc_res = lc_res**(-1)
                u_poly_res = u_poly_res * inv_lc_res
                v_poly_res = v_poly_res * inv_lc_res
            except Exception:
                if debug:
                    print(f"[MOD-DBL][BAD-RESULT] cannot normalize doubled u monic mod {p}")
                raise
                return None, None

        # reduce v modulo u
        try:
            v_poly_res = v_poly_res % u_poly_res
        except Exception:
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] cannot reduce v mod u after doubling mod {p}")
            raise
            return None, None

        # final Mumford test on the doubled pair
        try:
            rem2 = (v_poly_res**2 - f_poly_Fp).quo_rem(u_poly_res)[1]
            if rem2 != 0:
                if debug:
                    print(f"[MOD-DBL][BAD-RESULT] doubled pair fails Mumford test mod {p}: rem={rem2}")
                return None, None
        except ZeroDivisionError:
            if debug:
                print(f"[MOD-DBL][BAD-RESULT] division by zero while validating doubled pair mod {p}")
            raise
            return None, None

        # Build coefficient lists highest->lowest
        # coefficients(sparse=False) returns [c0, c1, ..., c_n] (lowest->highest)
        u_coeffs_low_to_high = u_poly_res.coefficients(sparse=False)
        v_coeffs_low_to_high = v_poly_res.coefficients(sparse=False)

        # convert to integers in 0..p-1 then reverse to high->low
        u_out = [int(c) for c in u_coeffs_low_to_high][::-1]
        v_out = [int(c) for c in v_coeffs_low_to_high][::-1]

        # pad to expected degrees if desired by caller (caller currently pads itself)
        return u_out, v_out

    # if both orientation attempts failed, optionally debug-print reasons
    if debug:
        print("[MOD-DBL] Tried orientations and failed:", tried)
    return None, None

def prefilter_solutions_algebraic(sol_list, prime, f_coeffs):
    """
    Filter solutions by algebraic constraint mod p BEFORE CRT.
    This eliminates ~83% of invalid combinations early.
    
    Returns: list of solutions that pass verify_mumford_pair mod p
    """
    
    R = PolynomialRing(GF(prime), 'x')
    x = R.gen()
    
    # Build f(x) mod p
    f_poly_coeffs = [int(c) % prime for c in f_coeffs]
    f_poly = R(0)
    for coeff in f_poly_coeffs:
        f_poly = f_poly * x + coeff
    
    filtered = []
    for sol in sol_list:
        s_val, p_val, v0_val, v1_val = [int(v) % prime for v in sol]

        # local discriminant mod pr
        #Delta_p = (s_val*s_val - 4*p_val) % prime

        # reject this prime's contribution if it splits
        #if pow(int(Delta_p), (prime - 1)//2, prime) != prime - 1:
        #    continue
        
        # Build u(x) = x² - s*x + p
        u_poly = x**2 - s_val*x + p_val
        
        # Build v(x) = v1*x + v0
        v_poly = v1_val*x + v0_val
        
        # Check: v(x)² ≡ f(x) (mod u(x))
        diff = v_poly**2 - f_poly
        remainder = diff % u_poly
        
        if remainder.is_zero():
            filtered.append(sol)
    
    return filtered

def filter_primes_avoiding_denoms(primes_list, divisors):
    # divisors: iterable of dicts with 's','p','v_0','v_1' (QQ)
    bad = set()
    for d in divisors:
        for k in ('s','p','v_0','v_1'):
            val = d.get(k)
            try:
                den = int(QQ(val).denominator)
                if den != 1:
                    # factor small primes of den
                    dd = den
                    p = 2
                    while p*p <= dd:
                        if dd % p == 0:
                            bad.add(p)
                            while dd % p == 0:
                                dd //= p
                        p += 1
                    if dd > 1:
                        bad.add(dd)
            except Exception:
                raise
    return [p for p in primes_list if p not in bad]


def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val, max_solutions=500):
    """
    Optimized solver for Index Calculus with early termination and smoothness filter.
    """
    solutions = []
    x_res = int(x_residue) % p
    x_sq = (x_res * x_res) % p
    
    Fp = GF(p)
    
    # Precompute inverse of 2 (used frequently)
    inv_2 = pow(2, -1, p) if p > 2 else None
    
    for s_val in range(p):
        # Early termination for Index Calculus
        if len(solutions) >= max_solutions:
            break
            
        p_val = (s_val * x_res - x_sq) % p
        
        # CRITICAL: Smoothness filter - only keep divisors where u(x) splits
        disc = (s_val * s_val - 4 * p_val) % p
        if disc != 0:
            disc_elem = Fp(disc)
            if not disc_elem.is_square():
                continue  # Skip non-smooth divisors
        
        A, B = _poly_mod_quad_fast(f_coeffs, s_val, p_val, p)
        
        # Reuse disc computation
        a_q = disc
        b_q = (-2 * (A * s_val + 2 * B)) % p
        c_q = (A * A) % p
        
        Z_roots = []
        
        if a_q == 0:
            if b_q != 0:
                try:
                    Z_roots.append((-c_q * pow(b_q, -1, p)) % p)
                except (ValueError, ZeroDivisionError):
                    continue
        else:
            disc_q = (b_q * b_q - 4 * a_q * c_q) % p
            try:
                delta = Fp(disc_q)
                if delta.is_square():
                    sq_root = int(delta.sqrt())
                    try:
                        inv_a = pow(a_q, -1, p)
                        inv_2a = (inv_2 * inv_a) % p
                    except (ValueError, ZeroDivisionError):
                        continue
                    Z_roots.append(((-b_q + sq_root) * inv_2a) % p)
                    if sq_root != 0:
                        Z_roots.append(((-b_q - sq_root) * inv_2a) % p)
            except Exception:
                continue
        
        valid_v1s = []
        for Z in Z_roots:
            Z_ele = Fp(Z)
            if Z_ele.is_square():
                r = int(Z_ele.sqrt())
                valid_v1s.append(r)
                if r != 0:
                    valid_v1s.append(p - r)
        
        for v1_val in valid_v1s:
            if v1_val == 0:
                if A != 0:
                    continue
                B_ele = Fp(B)
                if B_ele.is_square():
                    r = int(B_ele.sqrt())
                    solutions.append((s_val, p_val, r, 0))
                    if r != 0:
                        solutions.append((s_val, p_val, p - r, 0))
            else:
                if p == 2:
                    v0_val = (B + p_val) % 2
                    if (s_val * v1_val) % 2 == A % 2:
                         solutions.append((s_val, p_val, v0_val, v1_val))
                else:
                    num = (A - s_val * (v1_val * v1_val)) % p
                    den = (2 * v1_val) % p
                    try:
                        v0_val = (num * pow(den, -1, p)) % p
                    except (ValueError, ZeroDivisionError):
                        continue
                    
                    lhs_2 = (v0_val * v0_val - p_val * v1_val * v1_val) % p
                    if lhs_2 == B:
                        solutions.append((s_val, p_val, v0_val, v1_val))
    
    return solutions
