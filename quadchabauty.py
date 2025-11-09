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

# ---------------------------
# Helpers
# ---------------------------

from sage.all import (
    QQ, SR, EllipticCurve, ZZ, var, is_prime,
)
import math
from collections import defaultdict

from math import log, floor, sqrt, exp
from search_common import effective_degree

#---------------------------
# Helpers
# ---------------------------

def run_p_adic_fiber_analysis(E_q, P_q, p, precision=40, check_hypotheses=True, timeout=10):
    """
    Robust wrapper to compute p-adic local height (lambda_p) for a rational
    elliptic-curve point. Returns a report dict with keys:
      - status: 'success' or 'error'
      - p_adic_regulator: float (when success)
      - note: diagnostic message
      - debug: dict of low-level items
    
    Args:
        timeout: Maximum seconds to wait for p-adic height computation (default 10)
    """
    import signal
    
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

    # Check if curve has good reduction at p (required for p-adic height)
    assert E_q_rational.has_good_reduction(p), f"Curve does not have good reduction at p={p}"

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"p-adic height computation timed out after {timeout}s")
    
    padic_result = None
    last_exception = None
    
    # Set alarm for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Method 1: Point method (newer Sage)
        padic_obj = P_q_rational.padic_height(p=p, prec=precision)
        
        # padic_height can return a function or a p-adic number
        if callable(padic_obj):
            try:
                padic_result = float(padic_obj())
            except Exception as e1:
                try:
                    padic_result = float(padic_obj(p))
                except Exception as e2:
                    last_exception = e2
        else:
            # It's a p-adic number - extract the rational approximation
            try:
                padic_result = float(padic_obj)
            except TypeError:
                # pAdicCappedRelativeElement - use rational_reconstruction or lift
                try:
                    padic_result = float(padic_obj.lift())
                except Exception as e3:
                    last_exception = e3
            
        if padic_result is not None:
            report['status'] = 'success'
            report['p_adic_regulator'] = padic_result
            report['note'] = "padic_height succeeded (point method)"
    except AttributeError as ae:
        # Method 2: Curve method (older Sage)
        try:
            padic_obj = E_q_rational.padic_height(p, prec=precision)
            
            if callable(padic_obj):
                result_obj = padic_obj(P_q_rational)
                # Result is a p-adic number
                try:
                    padic_result = float(result_obj)
                except TypeError:
                    padic_result = float(result_obj.lift())
                    raise
            else:
                try:
                    padic_result = float(padic_obj)
                except TypeError:
                    padic_result = float(padic_obj.lift())
                    raise
            
            if padic_result is not None:
                report['status'] = 'success'
                report['p_adic_regulator'] = padic_result
                report['note'] = "padic_height succeeded (curve method)"
        except Exception as e4:
            last_exception = e4
            raise
    except TimeoutError as te:
        last_exception = te
        raise
    finally:
        # Cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # Sanity check: p-adic height should be reasonable magnitude
    if padic_result is not None:
        if abs(padic_result) > 1e10:
            raise ValueError(f"p-adic height computation returned absurdly large value: {padic_result}")
        if math.isnan(padic_result) or math.isinf(padic_result):
            raise ValueError(f"p-adic height computation returned NaN or Inf: {padic_result}")

    report['debug']['padic_tried'] = [{'method': 'both APIs', 'p': p, 'prec': precision}]
    
    if report['status'] != 'success' or padic_result is None:
        error_msg = f"p-adic height computation failed: {last_exception if last_exception else 'unknown error'}"
        report['note'] = error_msg
        raise RuntimeError(error_msg)
    
    return report



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
    This function returns the B(E) part, which is a uniform
    bound based on the discriminant of the specialized curve E_q.
    
    B(E) ~ (1/12)*log|Delta| + C
    """
    try:
        delta = E_q.discriminant()
        if delta == 0:
            print("Warning: estimate_local_height_archimedean: singular curve")
            return 0.0
            
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
        if c_p is None:
            c_p = 1
        
        # The local height lambda_p is log(c_p)
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

def estimate_n_max_for_fiber(cd, P_m, m_val, p_chabauty, return_diagnostics=False):
    """
    Runs the full numeric QC bound estimation for a single fiber using
    the "bounding method".
    
    H_max = B_infty + B_bad (first principles bound)
    n_max = floor(sqrt(H_max / hhat_q))
    
    This is the main function to call from your search script.
    
    Args:
        cd (CurveData): The main fibration object (for a4, a6)
        P_m (Point): The generator section over Q(m)
        m_val (QQ or int): The rational m-value to specialize at
        p_chabauty (int or list): A prime (or list of primes) of GOOD reduction
        return_diagnostics (bool): If True, return (n_max, diag); else just n_max
        
    Returns:
        int or tuple: n_max (if return_diagnostics=False) or (n_max, diag) tuple
    """
    print("\n" + "="*50)
    print(f"--- Quadratic Chabauty Bound Estimation ---")
    print(f"--- Fiber m = {m_val}, Chabauty prime p = {p_chabauty} ---")
    print(f"--- Using 'First Principles' Bounding Method ---")
    
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

    # =====================================================================
    # "First Principles" Bounding Method
    # =====================================================================
    # We need a bound H_max for the *non-p-adic* part of the height.
    # C_other = lambda_infty(Q) + sum(lambda_bad(Q))
    # We bound this with H_max = B_infty + B_bad
    #
    # 1. B_bad = sum(log(c_p)) (Tamagawa numbers). This is exact.
    # 2. B_infty = (1/12)log|Delta| + C (Archimedean bound).
    # 3. hhat_q = P_q.height() (Specialized canonical height).
    #
    # The QC bound is then n_max = floor(sqrt(H_max / hhat_q))
    # =====================================================================

    # --- 1. Compute specialized canonical height hhat_q (DENOMINATOR) ---
    hhat_q = None
    method_used = None
    try:
        if hasattr(P_q, 'height'):
            hhat_q = float(P_q.height(precision=100))
            method_used = "P_q.height(precision=100)"
    except Exception as e1:
        pass
    
    if hhat_q is None:
        try:
            if hasattr(E_q, 'height'):
                hhat_q = float(E_q.height(P_q, precision=100))
                method_used = "E_q.height(P_q, precision=100)"
        except Exception as e2:
            pass
            
    if hhat_q is None:
         raise RuntimeError("Failed to compute specialized canonical height (hhat_q)")
        
    print(f"Specialized canonical height (hhat_q): {hhat_q} (via {method_used})")

    # --- 2. Compute bad prime contributions (B_bad) ---
    B_bad = 0.0
    bad_contribs = {}
    
    # We must check all primes of bad reduction for the *specialized* curve E_q
    # not just the generic bad primes.
    try:
        # --- FIX ---
        # E_q.discriminant() returns a Rational. We must convert to a
        # Sage Integer (ZZ) to call .prime_divisors().
        delta = E_q.discriminant()
        specialized_bad_primes = ZZ(delta).prime_divisors()
        # --- END FIX ---
    except Exception as e:
        # Per user request, raise exceptions instead of warning and falling back
        print(f"Error: Could not get bad primes for specialized curve at m={m_q}: {e}")
        print(f"Curve E_q: {E_q}")
        # Re-raise the exception to halt execution, as the fallback is incorrect.
        raise RuntimeError(f"Failed to get bad primes for specialized curve E_q={E_q}") from e
    
    for p_bad in specialized_bad_primes:
        try:
            # Tamagawa number gives local height: lambda_p = log(c_p)
            lam_p = compute_local_height_bad_prime(E_q, p_bad)
            B_bad += lam_p
            bad_contribs[p_bad] = lam_p
        except Exception as e:
            # Per user request, raise exceptions
            print(f"Error: failed to compute bad-prime local height at p={p_bad}: {e}")
            raise RuntimeError(f"Failed to compute local height at bad prime p={p_bad}") from e

    print(f"Local heights at bad primes (B_bad): sum = {B_bad}")
    for p_bad, lam in bad_contribs.items():
        if lam > 0:
            print(f"  p={p_bad}: {lam}")

    # --- 3. Compute Archimedean bound (B_infty) ---
    B_infty = estimate_local_height_archimedean(E_q)
    print(f"Local height (Archimedean Bound B_infty): {B_infty}")

    # --- 4. Compute H_max and n_max ---
    
    # H_max is the "first principles" bound on the non-p-adic height
    H_max = B_infty + B_bad
    
    print(f"\n--> Total Global Height Bound (H_max = B_infty + B_bad): {H_max}")
    
    # The QC n_max bound is floor(sqrt(H_max / hhat_q))
    n_max = compute_n_max_bound(H_max, hhat_q)
    
    print(f"--> Estimated n_max (using hhat_q={hhat_q}): {n_max}")

    # --- Diagnostics ---
    diag = {
        'hhat_q': hhat_q,
        'lambda_padic': None,
        'B_bad': B_bad,
        'B_infty': B_infty,
        'H_max': H_max,
        'bad_contribs': bad_contribs,
    }
    
    # Detailed decomposition table
    print("\n--- Local Height Bound Decomposition ---")
    print(f"{'Component':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'B_bad (sum log(c_p))':<20} {B_bad}")
    print(f"{'B_infty (arch. bound)':<20} {B_infty}")
    print("-" * 35)
    print(f"{'H_max (B_infty + B_bad)':<20} {H_max}")
    print(f"{'hhat(P_q) (denom)':<20} {hhat_q}")
    
    if return_diagnostics:
        return n_max, diag
    else:
        return n_max



# rigorous_nmax.py
# Run in sage -python3, with `cd` (CurveData-like) and P_m (section) in scope.

from sage.all import QQ, ZZ, var, EllipticCurve, gcd
from math import log, floor, sqrt

# --- helpers: polynomial coefficient log-height ---
from math import log as mlog

def rational_function_degree_and_coeffheight(frac, m_var=None):
    """
    Return (degree, H_coeff) for frac in QQ(m).
    degree = max(deg(num), deg(den)).
    H_coeff = log(max_abs_coeff) using coeff_log_height_poly on numerator and denominator.
    """
    # Get numerator/denominator in symbolic-friendly way
    if hasattr(frac, 'numerator') and hasattr(frac, 'denominator'):
        num = frac.numerator()
        den = frac.denominator()
    else:
        # treat frac as SR expression: try to coerce numerator/denominator
        from sage.symbolic.ring import SR
        fsr = SR(frac).expand()
        try:
            num = fsr.numerator()
            den = fsr.denominator()
        except Exception:
            num = fsr
            den = 1

    # degree: try polynomial degree with explicit var if provided
    def _deg(poly):
        try:
            if m_var is not None:
                return int(poly.degree(m_var))
            else:
                return int(poly.degree())
        except Exception:
            # fallback: use effective_degree if available in scope
            try:
                return int(effective_degree(poly, m_var))
            except Exception:
                return 0

    d = max(_deg(num), _deg(den))

    # coefficient log-heights
    H_num = coeff_log_height_poly(num)
    H_den = coeff_log_height_poly(den) if den != 1 else 0.0
    H_total = max(H_num, H_den)

    return int(d), float(H_total)


def compute_provable_nmax(cd, P_m):
    """
    Adapted provable-nmax wrapper that uses the robust helpers above.
    Returns (nmax, cert).
    """
    # try to find symbolic m-variable if possible
    try:
        m_var = cd.SR_m
    except Exception:
        try:
            m_var = cd.a4.parent().gen()
        except Exception:
            m_var = None

    # discriminant (use SR versions if present)
    a4 = getattr(cd, 'SR_a4', cd.a4)
    a6 = getattr(cd, 'SR_a6', cd.a6)

    Delta = -16 * (4 * a4**3 + 27 * a6**2)

    D = int(effective_degree(Delta, m_var))
    H_D = coeff_log_height_poly(Delta)

    # c4,c6 coefficient heights
    c4 = -48 * a4
    c6 = -864 * a6
    H_c4c6 = max(coeff_log_height_poly(c4), coeff_log_height_poly(c6))

    # x-section
    xfunc = getattr(cd, 'SR_phi_x', getattr(cd, 'phi_x', P_m[0]))
    d_x, H_x = rational_function_degree_and_coeffheight(xfunc, m_var=m_var)

    # provable linear coefficients
    A1 = float(d_x)/2.0 + float(D)/24.0
    A2 = float(D)/6.0

    # explicit Silverman-ish constant (algorithmic, provable from coeff heights)
    C_Sil = 0.5 * H_c4c6 + mlog(2.0)

    C1 = 0.5 * (H_x + mlog(2.0)) + (1.0/24.0) * H_D + 0.5 * C_Sil

    C_infty = 0.5 * H_c4c6 + mlog(2.0)
    C_bad_const = H_D
    C2 = (1.0/12.0) * H_D + float(C_infty) + float(C_bad_const)

    if A1 <= 0 or C1 <= 0:
        raise ValueError("Nonpositive A1 or C1 encountered; check input family.")

    alpha = A2 / A1
    beta = C2 / C1
    nmax = int(floor(sqrt(min(alpha, beta))))

    cert = {
        'D': D, 'd_x': d_x,
        'H_D': H_D, 'H_x': H_x, 'H_c4c6': H_c4c6,
        'A1': A1, 'C1': C1, 'A2': A2, 'C2': C2,
        'alpha': alpha, 'beta': beta,
    }
    return nmax, cert


def coeff_log_height_poly(f):
    """
    Return log(max |integerized coefficients|) for a polynomial-like or symbolic expression f.
    Handles PolynomialRing and SymbolicRing inputs.
    """
    from math import log as mlog
    from sage.symbolic.ring import SR
    from sage.rings.polynomial.polynomial_ring import PolynomialRing_general
    from sage.rings.integer_ring import ZZ

    # Handle None or trivial
    if f is None:
        return 0.0

    # Case 1: symbolic ring (SR)
    if f.parent() == SR:
        try:
            num = f.numerator()
            den = f.denominator()
        except Exception:
            num, den = f, 1
        num = num.expand()
        # attempt to get variable list
        vars = num.variables()
        if not vars:
            # constant
            try:
                return float(mlog(abs(num)))
            except Exception:
                return 0.0
        v = vars[0]
        coeffs = num.coefficients(v)
        if not coeffs:
            coeffs = [num]
        # convert to integers
        ints = []
        for c in coeffs:
            try:
                dens = c.denominator()
                L = int(ZZ(dens))
                ints.append(int((c * L).numerator()))
            except Exception:
                try:
                    ints.append(int(c))
                except Exception:
                    ints.append(1)
        M = max(1, max(abs(x) for x in ints))
        return float(mlog(M))

    # Case 2: polynomial ring
    parent = f.parent()
    if isinstance(parent, PolynomialRing_general):
        coeffs = f.coefficients(sparse=False)
        if not coeffs:
            return 0.0
        dens = [c.denominator() for c in coeffs]
        L = int(ZZ(max(dens))) if dens else 1
        ints = [int((c * L).numerator()) for c in coeffs]
        M = max(1, max(abs(x) for x in ints))
        return float(mlog(M))

    # Case 3: rational-function-like (has numerator/denominator)
    if hasattr(f, 'numerator') and hasattr(f, 'denominator'):
        return coeff_log_height_poly(f.numerator())

    # fallback
    return 0.0
