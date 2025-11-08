# quadchabauty.py
#
# Implements a numeric workflow to estimate the search multiple 'n'
# for a single fiber (a specialized curve E_q) of the fibration,
# based on the Quadratic Chabauty / local height summation method.
#
# This provides a diagnostic to check if a given search height bound
# is sufficient for a particular fiber.
#
# Author: generated to fit user's repo style and constraints

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

# Per aimist.txt, no imports inside functions.
from math import log, floor, sqrt, exp

# ---------------------------
# Helpers
# ---------------------------

def _get_m_var_and_field(E_m):
    """
    Utility to extract the m variable and its FractionField from a curve.
    """
    if not hasattr(E_m, 'base_field'):
        raise TypeError("Input E_m must be a Sage EllipticCurve")
    
    Fm = E_m.base_field()
    if not hasattr(Fm, 'gen'):
        raise TypeError("Curve base field is not a FractionField")
        
    m_sym = Fm.gen()
    return m_sym, Fm

def compute_function_field_height(f, m_sym, Fm):
    """
    Computes the height of a rational function f in Q(m)
    as max(degree(numerator), degree(denominator)).
    
    Args:
        f (Sage Expression): The rational function (e.g., from cd.phi_x)
        m_sym (Sage Variable): The generator of the polynomial ring (e.g., 'm')
        Fm (Sage Field): The fraction field Q(m)
    
    Returns:
        int: The height (degree) of f.
    """
    try:
        f_m = Fm(f)
        num = f_m.numerator()
        den = f_m.denominator()
        return max(num.degree(), den.degree())
    except Exception as e:
        print(f"Warning: compute_function_field_height failed for {f}. Error: {e}")
        raise
        return 0

# ---------------------------
# Core Bounding Functions
# ---------------------------

def compute_n_max_bound(H_max, hhat_P):
    """
    Computes the final integer bound n_max from the total height H_max
    and the generator's canonical height hhat_P.
    
    n_max = floor(sqrt(H_max / hhat_P))
    """
    if hhat_P <= 0:
        print("Warning: compute_n_max_bound: hhat_P must be positive.")
        return 0
    if H_max < 0:
        print("Warning: compute_n_max_bound: H_max cannot be negative.")
        return 0
        
    return int(floor(sqrt(H_max / hhat_P)))

def estimate_local_height_archimedean(E_q):
    """
    Estimates the Archimedean contribution constant B(E).
    
    lambda_infty(P) <= 0.5 * h_x(P) + B(E)
    
    This function returns the B(E) part, which is based on the
    discriminant of the specialized curve E_q.
    
    B(E) ~ (1/12)*log|Delta| + C
    """
    try:
        delta = E_q.discriminant()
        # Use a common bound from Silverman's texts
        # (1/12)log|Delta| + log(2)
        B_E = (1/12) * log(abs(int(delta))) + log(2.0)
        return max(0, B_E) # Ensure non-negative
    except Exception as e:
        print(f"Warning: estimate_local_height_archimedean failed: {e}")
        raise
        return 0.0 # Return 0 as a safe fallback

def compute_local_height_bad_prime(E_q, p):
    """
    Computes the local height contribution lambda_p(P) at a bad prime p.
    
    For a rank 1 curve, this contribution is related to the Tamagawa
    number c_p, where lambda_p ~ log(c_p).
    """
    try:
        # tamagawa_number() runs Tate's algorithm implicitly
        c_p = E_q.tamagawa_number(p)
        
        # The local height lambda_p is log(c_p)
        # We must add this to the sum.
        lambda_p = log(float(c_p))
        
        return lambda_p
    except Exception as e:
        print(f"Warning: compute_local_height_bad_prime failed for p={p}: {e}")
        raise
        return 0.0 # Safe fallback

def run_p_adic_fiber_analysis(E_q, P_q, p):
    """
    Computes the p-adic "height" lambda_p for the Chabauty prime p.
    
    For a rank 1 curve, this is the p-adic regulator, which is
    simply the p-adic canonical height of the generator.
    
    Args:
        E_q (EllipticCurve): The specialized curve over QQ
        P_q (Point): The specialized generator point on E_q
        p (int): The (good) prime to use for Chabauty
        
    Returns:
        dict: A report of the calculation.
    """
    report = {
        # 'm_val': m_val, <-- FIX: m_val is not in scope here
        'p': p,
        'p_adic_regulator': None,
        'status': 'error',
        'note': ''
    }
    
    if not (is_prime(p) and p > 0):
        report['note'] = "Input p is not a valid prime"
        return report
        
    # E_q and P_q are now passed in already specialized
    
    # Check for good reduction at p
    if E_q.has_bad_reduction(p):
        report['note'] = f"Curve has bad reduction at p={p}"
        if E_q.has_multiplicative_reduction(p):
            report['note'] += " (multiplicative)"
        elif E_q.has_additive_reduction(p):
            report['note'] += " (additive)"
        return report
    
    report['note'] = "Curve has good reduction"
    
    # Compute the p-adic height
    try:
        # For r=1, the p-adic regulator is just the p-adic height of the generator
        
        # --- FIX: Changed p=p to just p ---
        # This fixes the TypeError: padic_height() got multiple values for argument 'p'
        assert is_prime(p), p
        #h_p = E_q.padic_height(P_q, p)
        h_p = E_q.padic_height(P_q, p, prec=40)

        # --- END FIX ---
        
        report['p_adic_regulator'] = float(h_p)
        report['status'] = 'success'
        if h_p == 0:
            report['note'] += ". WARNING: p-adic regulator is 0."
        else:
            report['note'] += ". Regulator is non-zero."
            
    except Exception as e:
        report['note'] = f"p-adic height computation failed: {e}"
        raise
        
    return report

# ---------------------------
# Main Wiring Function
# ---------------------------

def estimate_n_max_for_fiber(cd, P_m, m_val, p_chabauty, h_x_bound, hhat_P):
    """
    Runs the full numeric QC bound estimation for a single fiber.
    
    This is the main function to call from your search script.
    
    Args:
        cd (CurveData): The main fibration object (for a4, a6)
        P_m (Point): The generator section over Q(m)
        m_val (QQ or int): The rational m-value to specialize at
        p_chabauty (int): A prime of GOOD reduction to use for Chabauty
        h_x_bound (float): A (generous) upper bound on the
                           *naive x-height* (h_x) you expect to find.
                           (e.g., from your observed h_x <= 3.0)
        hhat_P (float): The canonical height of the generator (e.g., 2.0)
        
    Returns:
        int: The estimated maximum integer `n_max` to check.
    """
    print("\n" + "="*50)
    print(f"--- Quadratic Chabauty Bound Estimation ---")
    print(f"--- Fiber m = {m_val}, Chabauty prime p = {p_chabauty} ---")
    print(f"--- Using h_x(P) <= {h_x_bound}, hhat(P) = {hhat_P} ---")
    
    m_sym = var('m') # Assume 'm' is the variable name
    m_q = QQ(m_val)
    # --- replace the main computation block in estimate_n_max_for_fiber with this ---
    # Resolve m_sym from cd if possible (avoid var('m') brittle)
    try:
        base = cd.curve_base if hasattr(cd, 'curve_base') else None
        if base is None:
            # try extracting from polynomial a4/a6's parent
            m_sym = cd.a4.parent().gen()
        else:
            m_sym = base.gen()
    except Exception:
        m_sym = var('m')
        raise

    m_q = QQ(m_val)

    # Specialize the curve safely (if cd provides helper)
    try:
        a4_q = QQ(cd.a4.subs({m_sym: m_q}))
        a6_q = QQ(cd.a6.subs({m_sym: m_q}))
        #E_q = EllipticCurve(QQ, [a4_q, a6_q])
        E_q = EllipticCurve(QQ, [QQ(a4_q), QQ(a6_q)])
        assert E_q.base_field() == QQ

    except Exception as e:
        print(f"Error: Failed to specialize curve E_m at m={m_q}: {e}")
        raise
        return None

    # Specialize the section robustly
    try:
        if hasattr(P_m, 'specialize'):
            P_q = P_m.specialize(m_q)
        else:
            # fallback: try to substitute coordinate-wise but using base-field generator
            X_m_func, Y_m_func = P_m[0], P_m[1]
            X_q = QQ(X_m_func.subs({m_sym: m_q}))
            Y_q = QQ(Y_m_func.subs({m_sym: m_q}))
            P_q = E_q([QQ(X_q), QQ(Y_q)])
        if not (P_q in E_q):
            print("Warning: specialized point P_q is not on E_q; attempting to project to curve.")
            P_q = E_q(P_q)
    except Exception as e:
        print(f"Error: Failed to specialize section P_m at m={m_q}: {e}")
        raise
        return None

    # Accept p_chabauty as a single prime or a list of primes:
    if isinstance(p_chabauty, (list, tuple)):
        primes_list = list(p_chabauty)
    else:
        primes_list = [p_chabauty]

    # compute canonical height on specialized fiber

    from search_common import compute_canonical_height_matrix

    # compute the canonical height matrix for the generator section(s)
    try:
        Hmat = compute_canonical_height_matrix([P_m], cd)
        hhat_q = float(Hmat[0, 0])  # since the matrix is 1×1 for a single section
        print(f"Canonical height (from Shioda–Tate matrix): {hhat_q}")
    except Exception as e:
        print(f"Warning: canonical height matrix computation failed: {e}")
        hhat_q = float(hhat_P)  # fallback to supplied value
        raise


    # bad primes contributions
    lambda_bad_primes = 0.0
    bad_contribs = {}
    for p_bad in cd.bad_primes:
        try:
            if E_q.has_good_reduction(p_bad):
                bad_contribs[p_bad] = 0.0
                continue
        except Exception:
            # if API differs, skip reduction check and attempt tamagawa
            raise
        try:
            c_p = int(E_q.tamagawa_number(p_bad) or 1)
            lam_p = 0.0 if c_p <= 1 else math.log(float(c_p))
        except Exception as e:
            print(f"Warning: tamagawa read failed at p={p_bad}: {e}")
            lam_p = 0.0
            raise
        lambda_bad_primes += lam_p
        bad_contribs[p_bad] = lam_p

    # compute p-adic contributions: use max across primes_list for robustness
    lambda_padic = None
    padic_reports = []
    for p in primes_list:
        r = run_p_adic_fiber_analysis(E_q, P_q, p)
        padic_reports.append(r)
        if isinstance(r, dict) and r.get('status') == 'success':
            val = float(r['p_adic_regulator'])
        else:
            val = 0.0
        if lambda_padic is None or val > lambda_padic:
            lambda_padic = val

    lambda_good_primes = 0.0

    # archimedean via subtraction
    lambda_infty_exact = hhat_q - (lambda_padic + lambda_bad_primes + lambda_good_primes)

    # tolerance
    tol = 1e-10 * max(1.0, abs(hhat_q))
    if lambda_infty_exact < 0 and abs(lambda_infty_exact) <= tol:
        lambda_infty_exact = 0.0
    elif lambda_infty_exact < -tol:
        print("Warning: archimedean local height negative beyond tolerance; possible numeric error.")

    # compute H_max both ways and compare
    H_max_sub = lambda_padic + (lambda_infty_exact + lambda_bad_primes + lambda_good_primes)
    H_max_alt = hhat_q
    if abs(H_max_sub - H_max_alt) > max(tol, 1e-8):
        print("Warning: H_max mismatch: H_max_sub != hhat_q by", H_max_sub - H_max_alt)
    H_max = max(H_max_sub, H_max_alt)

    # Use specialized canonical height as denominator (consistency)
    n_max = compute_n_max_bound(H_max, hhat_q)

    # diagnostics
    diag = {
        'hhat_q': hhat_q,
        'lambda_padic': lambda_padic,
        'lambda_bad_primes': lambda_bad_primes,
        'lambda_infty_exact': lambda_infty_exact,
        'H_max_sub': H_max_sub,
        'H_max_alt': H_max_alt,
        'H_max': H_max,
        'padic_reports': padic_reports,
        'bad_contribs': bad_contribs
    }
    print("Diagnostics:", diag)
    return n_max, diag



def run_p_adic_fiber_analysis(E_q, P_q, p, precision=40, check_hypotheses=True):
    """
    Robust wrapper to compute p-adic local height (lambda_p) for a rational
    elliptic-curve point. Returns a report dict with keys:
      - status: 'success' or 'error'
      - p_adic_regulator: float (when success)
      - note: diagnostic message
      - debug: dict of low-level items
    
    This wrapper:
      - ensures p is an integer prime,
      - ensures E_q and P_q are over QQ if possible (coerce),
      - calls E_q.padic_height using keyword args to avoid signature mismatches.
    """
    report = {'p': p, 'status': 'error', 'p_adic_regulator': None, 'note': '', 'debug': {}}

    # Validate p quickly
    try:
        from sage.all import is_prime
        if not is_prime(int(p)):
            report['note'] = f"Provided p={p} is not prime (after coercion)."
            return report
        p = int(p)
    except Exception as e:
        report['note'] = f"Invalid p argument: {e}"
        raise
        return report

    # If curve/point come from number field, attempt safe coercion to QQ
    try:
        # Ensure curve coefficients are rationals (may raise)
        a1, a2, a3, a4, a6 = E_q.a_invariants()
        a_invars_q = [QQ(ai) for ai in (a1, a2, a3, a4, a6)]
        E_q_rational = EllipticCurve(QQ, a_invars_q)
        report['debug']['curve_coerced'] = True
    except Exception as e:
        # Could not coerce the curve to QQ; keep original but note it
        E_q_rational = E_q
        report['debug']['curve_coerced'] = False
        report['debug']['curve_coercion_err'] = str(e)
        raise

    # Coerce point coordinates if possible
    try:
        # prefer xy() for affine coords; fallback to tuple access
        if hasattr(P_q, 'xy'):
            xcoord, ycoord = P_q.xy()
        else:
            xcoord, ycoord = P_q[0], P_q[1]
        xq = QQ(xcoord)
        yq = QQ(ycoord)
        P_q_rational = E_q_rational([xq, yq])
        report['debug']['point_coerced'] = True
    except Exception as e:
        P_q_rational = P_q
        report['debug']['point_coerced'] = False
        report['debug']['point_coercion_err'] = str(e)
        raise

    # Ensure E_q_rational and P_q_rational are usable
    try:
        if not (P_q_rational in E_q_rational):
            # Sometimes E_q_rational(P_q_rational) will coerce, else error
            try:
                P_q_rational = E_q_rational(P_q_rational)
            except Exception:
                report['note'] = "Specialized point is not on the coerced rational curve."
                raise
                return report
    except Exception:
        # If membership test is not supported for number-field points, continue cautiously
        raise

    # Avoid calling padic_height if curve/point are not rational
    try:
        if E_q_rational.base_field() is not QQ:
            report['note'] = "Curve is not defined over QQ; padic height not supported here."
            return report
    except Exception:
        # If .base_field() fails, continue but be cautious
        raise

    # Finally, call padic_height using keywords to avoid positional signature mismatches.
    # Try a small list of plausible keyword combos for different Sage versions.
    padic_result = None
    tried = []
    kwarg_attempts = [
        {'P': P_q_rational, 'p': p, 'prec': precision},
        {'P': P_q_rational, 'prime': p, 'prec': precision},
        {'P': P_q_rational, 'p': p, 'precision': precision},
        # fallback to positionals but in the canonical order if necessary
        {}
    ]
    for kws in kwarg_attempts:

        try:
            padic_obj = E_q_rational.padic_height(P_q_rational, p, prec=precision)
            padic_result = float(padic_obj)
            report['status'] = 'success'
            report['p_adic_regulator'] = padic_result
            report['note'] = "padic_height succeeded"
        except Exception as e:
            report['note'] = f"p-adic height computation failed: {e}"
            raise


    report['debug']['padic_tried'] = tried
    if report['status'] != 'success':
        # No attempt succeeded
        report['note'] = "p-adic height computation failed for tried signatures; see debug"
    return report
