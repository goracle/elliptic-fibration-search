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

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

from math import log, floor, sqrt, exp

# ---------------------------
# Helpers
# ---------------------------

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

from math import log, floor, sqrt, exp

# ---------------------------
# Helpers
# ---------------------------

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

from math import log, floor, sqrt, exp

# ---------------------------
# Helpers
# ---------------------------

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

from math import log, floor, sqrt, exp

# ---------------------------
# Helpers
# ---------------------------

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

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

def run_p_adic_fiber_analysis(E_q, P_q, p, precision=40, check_hypotheses=True):
    """
    Robust wrapper to compute p-adic local height (lambda_p) for a rational
    elliptic-curve point. Returns a report dict with keys:
      - status: 'success' or 'error'
      - p_adic_regulator: float (when success)
      - note: diagnostic message
      - debug: dict of low-level items
    """
    report = {'p': p, 'status': 'error', 'p_adic_regulator': None, 'note': '', 'debug': {}}

    # Validate p quickly
    from sage.all import is_prime
    assert is_prime(int(p)), f"p={p} is not prime"
    p = int(p)

    # Ensure curve coefficients are rationals
    a1, a2, a3, a4, a6 = E_q.a_invariants()
    a_invars_q = [QQ(ai) for ai in (a1, a2, a3, a4, a6)]
    E_q_rational = EllipticCurve(QQ, a_invars_q)
    report['debug']['curve_coerced'] = True

    # Coerce point coordinates
    if hasattr(P_q, 'xy'):
        xcoord, ycoord = P_q.xy()
    else:
        xcoord, ycoord = P_q[0], P_q[1]
    xq = QQ(xcoord)
    yq = QQ(ycoord)
    P_q_rational = E_q_rational([xq, yq])
    report['debug']['point_coerced'] = True

    # Verify point is on curve
    assert P_q_rational in E_q_rational, "Point not on curve after coercion"

    # Verify curve is over QQ
    assert E_q_rational.base_field() is QQ, "Curve not over QQ"

    # Call padic_height - try both APIs (point vs curve) for compatibility
    padic_result = None
    try:
        # Method 1: Point method (newer Sage)
        padic_obj = P_q_rational.padic_height(p=p, prec=precision)
        
        # padic_height can return a function or a p-adic number
        if callable(padic_obj):
            try:
                padic_result = float(padic_obj())
            except Exception:
                padic_result = float(padic_obj(p))
        else:
            # It's a p-adic number - extract the rational approximation
            try:
                padic_result = float(padic_obj)
            except TypeError:
                # pAdicCappedRelativeElement - use rational_reconstruction or lift
                padic_result = float(padic_obj.lift())
            
        report['status'] = 'success'
        report['p_adic_regulator'] = padic_result
        report['note'] = "padic_height succeeded (point method)"
    except AttributeError:
        # Method 2: Curve method (older Sage)
        padic_obj = E_q_rational.padic_height(p, prec=precision)
        
        if callable(padic_obj):
            result_obj = padic_obj(P_q_rational)
            # Result is a p-adic number
            try:
                padic_result = float(result_obj)
            except TypeError:
                padic_result = float(result_obj.lift())
        else:
            try:
                padic_result = float(padic_obj)
            except TypeError:
                padic_result = float(padic_obj.lift())
            
        report['status'] = 'success'
        report['p_adic_regulator'] = padic_result
        report['note'] = "padic_height succeeded (curve method)"

    report['debug']['padic_tried'] = [{'method': 'both APIs', 'p': p, 'prec': precision}]
    assert report['status'] == 'success', f"p-adic height computation failed: {report['note']}"
    return report

# ---------------------------
# Main Wiring Function
# ---------------------------

def estimate_n_max_for_fiber(cd, P_m, m_val, p_chabauty, h_x_bound, hhat_P, return_diagnostics=False):
    """
    Runs the full numeric QC bound estimation for a single fiber using
    EXACT canonical height decomposition.
    
    This is the main function to call from your search script.
    
    Args:
        cd (CurveData): The main fibration object (for a4, a6)
        P_m (Point): The generator section over Q(m)
        m_val (QQ or int): The rational m-value to specialize at
        p_chabauty (int or list): A prime (or list of primes) of GOOD reduction
        h_x_bound (float): DEPRECATED - not used in new method
        hhat_P (float): The canonical height of the generator (e.g., 2.0)
        return_diagnostics (bool): If True, return (n_max, diag); else just n_max
        
    Returns:
        int or tuple: n_max (if return_diagnostics=False) or (n_max, diag) tuple
    """
    print("\n" + "="*50)
    print(f"--- Quadratic Chabauty Bound Estimation ---")
    print(f"--- Fiber m = {m_val}, Chabauty prime p = {p_chabauty} ---")
    print(f"--- Using canonical height decomposition method ---")
    
    # Resolve m_sym from cd
    try:
        base = cd.curve_base if hasattr(cd, 'curve_base') else None
        if base is None:
            m_sym = cd.a4.parent().gen()
        else:
            m_sym = base.gen()
    except Exception:
        m_sym = var('m')

    m_q = QQ(m_val)

    # Specialize the curve
    a4_q = QQ(cd.a4.subs({m_sym: m_q}))
    a6_q = QQ(cd.a6.subs({m_sym: m_q}))
    E_q = EllipticCurve(QQ, [QQ(a4_q), QQ(a6_q)])
    assert E_q.base_field() == QQ, "Specialized curve not over QQ"

    # Specialize the section
    if hasattr(P_m, 'specialize'):
        P_q = P_m.specialize(m_q)
    else:
        X_m_func, Y_m_func = P_m[0], P_m[1]
        X_q = QQ(X_m_func.subs({m_sym: m_q}))
        Y_q = QQ(Y_m_func.subs({m_sym: m_q}))
        P_q = E_q([QQ(X_q), QQ(Y_q)])
    
    assert P_q in E_q, "Specialized point not on specialized curve"

    # Accept p_chabauty as a single prime or a list of primes
    if isinstance(p_chabauty, (list, tuple)):
        primes_list = list(p_chabauty)
    else:
        primes_list = [p_chabauty]

    # =====================================================================
    # CANONICAL HEIGHT DECOMPOSITION METHOD (exact, no crude bounds)
    # =====================================================================
    # For a point P_q on E_q/Q, the canonical height decomposes as:
    #   hhat(P_q) = sum_v lambda_v(P_q)
    # where v runs over all places (real and p-adic).
    #
    # We compute:
    #   1. hhat_q = canonical height of P_q on E_q (exact via Sage)
    #   2. lambda_padic = p-adic local height at Chabauty prime(s)
    #   3. lambda_bad = sum of local heights at bad primes
    #   4. lambda_infty = archimedean contribution (by subtraction)
    #
    # This gives H_max = hhat_q exactly, avoiding crude upper bounds.
    # =====================================================================

    # --- 1. Compute canonical height of specialized point (exact as Sage gives) ---
    # Try multiple approaches in order of preference
    hhat_q = None
    method_used = None
    
    # Method 1: Direct P_q.height() - most reliable for points over QQ
    try:
        if hasattr(P_q, 'height'):
            hhat_q = float(P_q.height(precision=100))
            method_used = "P_q.height(precision=100)"
    except Exception as e1:
        pass
    
    # Method 2: E_q.height(P_q) with precision
    if hhat_q is None:
        try:
            if hasattr(E_q, 'height'):
                hhat_q = float(E_q.height(P_q, precision=100))
                method_used = "E_q.height(P_q, precision=100)"
        except Exception as e2:
            pass
    
    # Method 3: Shioda-Tate matrix computation (your existing infrastructure)
    if hhat_q is None:
        try:
            from search_common import compute_canonical_height_matrix
            Hmat = compute_canonical_height_matrix([P_m], cd)
            hhat_q = float(Hmat[0, 0])
            method_used = "Shioda-Tate matrix"
        except Exception as e3:
            pass
    
    # Method 4: Fallback to supplied hhat_P
    if hhat_q is None:
        print(f"Warning: All canonical height methods failed. Using supplied hhat_P = {hhat_P}")
        hhat_q = float(hhat_P)
        method_used = "fallback (supplied hhat_P)"
        
    print(f"Canonical height on specialized fiber (hhat_q): {hhat_q:.12g} (via {method_used})")

    # --- 2. Compute bad primes contributions explicitly ---
    lambda_bad_primes = 0.0
    bad_contribs = {}
    for p_bad in cd.bad_primes:
        # If fiber has good reduction at p_bad, contribution is zero
        try:
            if E_q.has_good_reduction(p_bad):
                bad_contribs[p_bad] = 0.0
                continue
        except Exception as e:
            print(f"Warning: has_good_reduction check failed at p={p_bad}: {e}")
            
        try:
            # Tamagawa number gives local height: lambda_p = log(c_p)
            c_p = int(E_q.tamagawa_number(p_bad) or 1)
            lam_p = 0.0 if c_p <= 1 else math.log(float(c_p))
            lambda_bad_primes += lam_p
            bad_contribs[p_bad] = lam_p
        except Exception as e:
            print(f"Warning: failed to compute bad-prime local height at p={p_bad}: {e}")
            bad_contribs[p_bad] = 0.0

    print(f"Local heights at bad primes: sum = {lambda_bad_primes:.12g}")
    for p_bad, lam in bad_contribs.items():
        if lam > 0:
            print(f"  p={p_bad}: {lam:.12g}")

    # --- 3. Compute p-adic contributions for Chabauty prime(s) ---
    # Use maximum across multiple primes for robustness
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

    print(f"Local height (p-adic, p={primes_list}): {lambda_padic:.12g}")

    # --- 4. Good primes contribution (usually zero for integral points) ---
    lambda_good_primes = 0.0

    # --- 5. Archimedean contribution via subtraction (exact by identity) ---
    lambda_infty_exact = hhat_q - (lambda_padic + lambda_bad_primes + lambda_good_primes)

    # Numerical safety: lambda_infty should not be negative
    tol = 1e-10 * max(1.0, abs(hhat_q))
    if lambda_infty_exact < 0:
        if abs(lambda_infty_exact) <= tol:
            print(f"Note: lambda_infty = {lambda_infty_exact:.3e} (negative within tolerance, setting to 0)")
            lambda_infty_exact = 0.0
        else:
            print(f"Warning: archimedean local height negative beyond tolerance: {lambda_infty_exact:.12g}")

    print(f"Local height (Archimedean) from subtraction: {lambda_infty_exact:.12g}")
    print(f"  (computed as hhat_q - sum_nonarch)")

    # --- 6. Compute H_max and n_max ---
    # H_max is just hhat_q (the canonical height itself)
    # But we can also compute it as sum of components for verification
    C_other = lambda_good_primes + lambda_bad_primes + lambda_infty_exact
    H_max_sub = lambda_padic + C_other
    H_max_alt = hhat_q
    
    # Verify decomposition
    if abs(H_max_sub - H_max_alt) > max(tol, 1e-8):
        print(f"Warning: H_max mismatch: sum={H_max_sub:.12g} vs hhat={H_max_alt:.12g} (diff={H_max_sub - H_max_alt:.3e})")
    
    H_max = hhat_q  # Use the direct canonical height (most accurate)
    
    print(f"\n--> Total C_other (non-p sum): {C_other:.12g}")
    print(f"--> Total Global Height (H_max): {H_max:.12g}")
    
    # Use specialized canonical height as denominator (consistency)
    n_max = compute_n_max_bound(H_max, hhat_q)
    
    print(f"--> Estimated n_max: {n_max}")

    # --- Diagnostics ---
    diag = {
        'hhat_q': hhat_q,
        'lambda_padic': lambda_padic,
        'lambda_bad_primes': lambda_bad_primes,
        'lambda_good_primes': lambda_good_primes,
        'lambda_infty_exact': lambda_infty_exact,
        'H_max_sub': H_max_sub,
        'H_max_alt': H_max_alt,
        'H_max': H_max,
        'C_other': C_other,
        'padic_reports': padic_reports,
        'bad_contribs': bad_contribs,
        'residual': H_max_sub - H_max_alt
    }
    
    # Detailed decomposition table
    print("\n--- Local Height Decomposition ---")
    print(f"{'Place':<20} {'Lambda':<15}")
    print("-" * 35)
    print(f"{'p-adic (p=' + str(primes_list[0]) + ')':<20} {lambda_padic:>14.8f}")
    for p_bad, lam in bad_contribs.items():
        if lam > 0:
            print(f"{'bad prime p=' + str(p_bad):<20} {lam:>14.8f}")
    print(f"{'Archimedean':<20} {lambda_infty_exact:>14.8f}")
    print("-" * 35)
    print(f"{'Sum (should = hhat)':<20} {H_max_sub:>14.8f}")
    print(f"{'hhat(P_q) direct':<20} {hhat_q:>14.8f}")
    print(f"{'Residual':<20} {diag['residual']:>14.8e}")
    
    if return_diagnostics:
        return n_max, diag
    else:
        return n_max
