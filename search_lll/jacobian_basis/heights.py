"""Height computation functions."""

import warnings
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve

from .archimedean import archimedean_height_correction
from .local import get_bad_primes, local_height_correction_finite
from .periods import choose_numerical_base_point
from search_lll.homology import *

def archimedean_naive_height(div):
    u, v = div
    coeffs = u.list() + v.list()
    vals = [abs(float(c)) for c in coeffs if c != 0]
    if not vals:
        return 0.0
    return math.log(max(vals))

# Functions: naive_height_qq, arakelov_quasi_height, arakelov_canonical_height

def naive_height_qq(div, prec=53):
    """
    Compute naive (logarithmic) height of Mumford polynomials, including denominators.
    Corresponds to the Weil height on the coefficients.
    """
    from sage.all import QQ, RealField
    
    # Extract coefficients from Sage Jacobian element or list
    # div[0] is u(x), div[1] is v(x)
    u_coeffs = [QQ(c) for c in div[0].list()]
    v_coeffs = [QQ(c) for c in div[1].list()]
    
    # Filter out zeros
    all_coeffs = [c for c in (u_coeffs + v_coeffs) if c != 0]
    
    if not all_coeffs:
        return QQ(0)
        
    # Clear denominators
    dens = [c.denominator() for c in all_coeffs]
    
    from math import gcd
    from functools import reduce
    lcm_den = reduce(lambda a, b: (a * b) // gcd(a, b), dens, 1)
    
    # Scale coefficients to integers
    int_coeffs = [int((c * lcm_den).numerator()) for c in all_coeffs]
    # Include the denominator itself in the height calculation (Projective height)
    # The height of [x0 : ... : xn] is max(|x_i|).
    # If coords are rational c_i, we map to [c_0*L : ... : c_n*L : L] where L is LCM.
    # So we must include lcm_den in the max set.
    int_coeffs.append(lcm_den)
    
    max_abs = max(abs(c) for c in int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log()


def arakelov_quasi_height(div, f_coeffs, period_matrix=None, prec=300,
                          use_finite_places=True, arch_override=None):
    """
    Compute a quasi-canonical Arakelov height for a Mumford divisor `div` on
    the genus-2 Jacobian of y^2 = f(x).
    
    Uses the full Weil height (naive_height_qq) as the base, plus Arakelov corrections.
    """
    key = (str(div), tuple(f_coeffs), prec, use_finite_places, arch_override)
    if key in arakelov_quasi_height.cache:
        return arakelov_quasi_height.cache[key]
    
    from sage.all import RealField

    # fast path
    if getattr(div, "is_zero", lambda: False)():
        return RealField(prec)(0)

    RF = RealField(prec)

    # 1) Naive Weil Height (Global, includes denominators)
    # CRITICAL FIX: Use naive_height_qq instead of archimedean_naive_height
    h_naive = naive_height_qq(div, prec=prec)
    h_total = RF(h_naive)

    # 2) Archimedean quadratic (theta / Green) term
    if arch_override is not None:
        h_arch = RF(arch_override)
    else:
        if period_matrix is None:
            period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
        h_arch = RF(archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec))

    h_total += h_arch

    # 3) Finite local corrections (Neron functions at bad primes)
    # Note: These corrections adjust the naive height to the canonical local height.
    h_finite = RF(0)
    
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        
        for p in bad_primes:
            val = local_height_correction_finite(div, p, f_coeffs)
            
            if val is None:
                raise RuntimeError(f"[arakelov_quasi_height] local correction returned None for p={p}")
            
            h_finite += RF(val)

    h_total += h_finite

    # Final sanity: ensure non-negative-ish (small negative numerical noise -> clamp to 0)
    if float(h_total) < -1e-9:
         # Warn but don't crash yet, let canonical height logic handle it if it persists
         pass

    ret = h_total
    arakelov_quasi_height.cache[key] = ret
    return ret
arakelov_quasi_height.cache = {}


def arakelov_canonical_height(div, f_coeffs, prec=2048, max_prec=8192, debug=False, period_matrix=None):
    """
    Compute the Arakelov canonical height of `div`.
    Accepts optional `period_matrix` to avoid recomputing expensive periods.
    """
    key = (str(div), tuple(f_coeffs), prec, max_prec)
    if key in arakelov_canonical_height.cache:
        val = arakelov_canonical_height.cache[key]
        if val < -1e-9:
             # If cached value is bad, ignore cache and recompute
             pass
        else:
             return val

    TOL_NEG = 1e-9
    PREC_INCR_FACTOR = 2

    def _compute_quasi_heights(use_finite, use_period_prec, provided_PM=None):
        PM = provided_PM if provided_PM is not None else get_period_matrix_auto_B(f_coeffs, prec=use_period_prec)
        D1 = div
        D2 = div + div
        D3 = D2 + div
        h1 = arakelov_quasi_height(D1, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h2 = arakelov_quasi_height(D2, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h3 = arakelov_quasi_height(D3, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        return float(h1), float(h2), float(h3)

    # First attempt
    try:
        h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=prec, provided_PM=period_matrix)
        h_can = (h3 + h1 - 2.0 * h2) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] attempt prec={prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
    except (ZeroDivisionError, RuntimeError, ValueError) as e:
        warnings.warn(f"[arakelov_canonical_height] initial full computation failed: {e}. Attempting retries.", RuntimeWarning)
        h_can = -1.0
    except Exception as e:
        warnings.warn(f"[arakelov_canonical_height] initial computation failed with unexpected error: {e}", RuntimeWarning)
        raise

    if h_can >= -TOL_NEG:
        ret = float(max(h_can, 0.0))
        arakelov_canonical_height.cache[key] = ret
        return ret

    # Retry loop with increasing precision
    cur_prec = prec
    while cur_prec < max_prec:
        cur_prec = min(max_prec, int(cur_prec * PREC_INCR_FACTOR))
        try:
            h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=cur_prec, provided_PM=period_matrix)
            h_can = (h3 + h1 - 2.0 * h2) / 2.0
            if debug:
                print(f"[arakelov_canonical_height] retry prec={cur_prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
            if h_can >= -TOL_NEG:
                warnings.warn(f"[arakelov_canonical_height] resolved negativity after increasing precision to {cur_prec}.", RuntimeWarning)
                ret = float(max(h_can, 0.0))
                arakelov_canonical_height.cache[key] = ret
                return ret
        except (ZeroDivisionError, RuntimeError, ValueError) as e:
             if cur_prec >= max_prec:
                raise ValueError(f"Canonical height failed after retries: {e}")
             continue
        except Exception:
            raise

    raise ValueError(f"Canonical height computation failed or remained negative ({h_can}) after retries for divisor {div}")

arakelov_canonical_height.cache = {}
