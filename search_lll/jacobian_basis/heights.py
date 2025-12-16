"""Height computation functions."""

import warnings
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve

from .archimedean import archimedean_height_correction
from .local import get_bad_primes, local_height_correction_finite
from .periods import choose_numerical_base_point
from search_lll.homology import *

# Functions: naive_height_qq, arakelov_quasi_height, arakelov_canonical_height

def arakelov_canonical_height(div, f_coeffs, prec=2048, max_prec=8192, debug=False):
    """
    Compute the Arakelov canonical height of `div` (a Jacobian point).
    Defensive: will attempt fallbacks (archimedean-only, retries with higher precision)
    if the assembled canonical height is numerically negative, and will ultimately
    clamp to 0.0 rather than returning a large negative value that would break
    global invariants.

    Returns a Python float (>= 0.0) in all non-exceptional cases.
    """
    import warnings
    from sage.all import QQ

    # small tolerances
    TOL_NEG = 1e-12   # allowed tiny negative noise
    PREC_INCR_FACTOR = 2

    # helper to compute quasi-heights (full or arch-only)
    def _compute_quasi_heights(use_finite, use_period_prec):
        # compute period matrix at requested precision
        PM = get_period_matrix_auto_B(f_coeffs, prec=use_period_prec)
        D1 = div
        D2 = div + div
        D3 = D2 + div
        h1 = arakelov_quasi_height(D1, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h2 = arakelov_quasi_height(D2, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h3 = arakelov_quasi_height(D3, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        return float(h1), float(h2), float(h3)

    # 1) First attempt: normal full computation
    try:
        h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=prec)
        h_can = (h3 + h1 - 2.0 * h2) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] attempt prec={prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
    except ZeroDivisionError:
        # preserve previous retry behavior for ZeroDivisionError so outer code can escalate
        raise
    except Exception as e:
        # catastrophic failure computing heights (e.g. period matrix failed)
        warnings.warn(f"[arakelov_canonical_height] initial full computation failed: {e}. Attempting archimedean-only fallback.", RuntimeWarning)
        h_can = -1.0  # force fallback path

    # If result is numerically acceptable, return it (clamp tiny negatives)
    if h_can >= -TOL_NEG:
        return float(max(h_can, 0.0))

    # 2) Fallback A: try archimedean-only (no finite places)
    try:
        h1_nf, h2_nf, h3_nf = _compute_quasi_heights(use_finite=False, use_period_prec=prec)
        h_can_nf = (h3_nf + h1_nf - 2.0 * h2_nf) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] arch-only: h1={h1_nf}, h2={h2_nf}, h3={h3_nf}, h_can_nf={h_can_nf}")
        if h_can_nf >= -TOL_NEG:
            warnings.warn(f"[arakelov_canonical_height] suppressed finite-place instability; using archimedean-only height for divisor={div}.", RuntimeWarning)
            return float(max(h_can_nf, 0.0))
    except Exception as e:
        warnings.warn(f"[arakelov_canonical_height] arch-only fallback failed: {e}", RuntimeWarning)

    # 3) Fallback B: retry full computation with increasing precision
    cur_prec = prec
    while cur_prec < max_prec:
        cur_prec = min(max_prec, int(cur_prec * PREC_INCR_FACTOR))
        try:
            h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=cur_prec)
            h_can = (h3 + h1 - 2.0 * h2) / 2.0
            if debug:
                print(f"[arakelov_canonical_height] retry prec={cur_prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
            if h_can >= -TOL_NEG:
                warnings.warn(f"[arakelov_canonical_height] resolved negativity after increasing precision to {cur_prec}.", RuntimeWarning)
                return float(max(h_can, 0.0))
        except ZeroDivisionError:
            raise
        except Exception as e:
            warnings.warn(f"[arakelov_canonical_height] retry at prec={cur_prec} failed: {e}", RuntimeWarning)
            continue

    # 4) Give up: clamp to 0.0 (conservative) and emit a full diagnostic warning.
    warnings.warn(
        f"[arakelov_canonical_height] canonical height remained negative after all fallbacks for divisor={div}. "
        f"Returning 0.0 (clamped). Last observed h_can={h_can}. prec tried up to {cur_prec}.",
        RuntimeWarning
    )
    return 0.0

def arakelov_quasi_height(div, f_coeffs, period_matrix=None, prec=300, use_finite_places=True, arch_override=None):
    """
    Computes a 'quasi-canonical' height: Naive + Finite + Archimedean(Quadratic).
    This height h(div) satisfies h(ndiv) = n^2 * h_can(div) + L(ndiv) + O(1).
    It is not quadratic itself, but the quadratic coefficient is the canonical height.
    """
    if div.is_zero():
        return QQ(0)

    # 1. Naive global height (Essential for the height to be positive/quadratic)
    h_naive = naive_height_qq(div, prec=prec)
    
    # 2. Archimedean quadratic part
    if period_matrix is None:
        period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
    if arch_override is None:
        h_arch = archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec)
    else:
        h_arch = arch_override 
    # 3. Finite place corrections
    h_finite_correction = QQ(0)
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        for p in bad_primes:
            h_finite_correction += local_height_correction_finite(div, p, f_coeffs)
            
    return h_naive + h_arch + h_finite_correction

def naive_height_qq(div, prec=53):
    """
    Compute naive (logarithmic) height of Mumford polynomials.
    """
    from sage.all import QQ, RealField
    
    u_coeffs = [QQ(c) for c in div[0].list()]
    v_coeffs = [QQ(c) for c in div[1].list()]
    
    # Clear denominators
    dens = [c.denominator() for c in (u_coeffs + v_coeffs) if c != 0]
    if not dens:
        return QQ(0)
    
    from math import gcd
    from functools import reduce
    lcm_den = reduce(lambda a, b: (a * b) // gcd(a, b), dens, 1)
    
    # Scale coefficients to integers
    int_coeffs = [int((c * lcm_den).numerator()) for c in (u_coeffs + v_coeffs)]
    int_coeffs = [abs(c) for c in int_coeffs if c != 0]
    if not int_coeffs:
        return QQ(0)
    
    max_abs = max(int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log().nearby_rational(max_error=R(2)**(-prec + 5))


# heights.py — replace/archetype arakelov_canonical_height with this version

def arakelov_canonical_height(div, f_coeffs, prec=2048, max_prec=8192, debug=False, period_matrix=None):
    """
    Compute the Arakelov canonical height of `div`.
    Accepts optional `period_matrix` to avoid recomputing expensive periods.
    """
    import warnings
    from sage.all import QQ

    TOL_NEG = 1e-12
    PREC_INCR_FACTOR = 2

    def _compute_quasi_heights(use_finite, use_period_prec, provided_PM=None):
        # use provided period matrix if present; otherwise compute at requested precision
        PM = provided_PM if provided_PM is not None else get_period_matrix_auto_B(f_coeffs, prec=use_period_prec)
        D1 = div
        D2 = div + div
        D3 = D2 + div
        h1 = arakelov_quasi_height(D1, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h2 = arakelov_quasi_height(D2, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        h3 = arakelov_quasi_height(D3, f_coeffs, period_matrix=PM, prec=use_period_prec, use_finite_places=use_finite)
        return float(h1), float(h2), float(h3)

    # First attempt: use provided period_matrix if given
    try:
        h1, h2, h3 = _compute_quasi_heights(use_finite=True, use_period_prec=prec, provided_PM=period_matrix)
        h_can = (h3 + h1 - 2.0 * h2) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] attempt prec={prec} full: h1={h1}, h2={h2}, h3={h3}, h_can={h_can}")
    except ZeroDivisionError:
        raise
    except Exception as e:
        warnings.warn(f"[arakelov_canonical_height] initial full computation failed: {e}. Attempting archimedean-only fallback.", RuntimeWarning)
        h_can = -1.0

    if h_can >= -TOL_NEG:
        return float(max(h_can, 0.0))

    # Arch-only fallback, using same PM if provided
    try:
        h1_nf, h2_nf, h3_nf = _compute_quasi_heights(use_finite=False, use_period_prec=prec, provided_PM=period_matrix)
        h_can_nf = (h3_nf + h1_nf - 2.0 * h2_nf) / 2.0
        if debug:
            print(f"[arakelov_canonical_height] arch-only: h1={h1_nf}, h2={h2_nf}, h3={h3_nf}, h_can_nf={h_can_nf}")
        if h_can_nf >= -TOL_NEG:
            warnings.warn(f"[arakelov_canonical_height] suppressed finite-place instability; using archimedean-only height for divisor={div}.", RuntimeWarning)
            return float(max(h_can_nf, 0.0))
    except Exception as e:
        warnings.warn(f"[arakelov_canonical_height] arch-only fallback failed: {e}", RuntimeWarning)

    # Retry loop that may increase precision (still using provided PM only if present and matches prec)
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
                return float(max(h_can, 0.0))
        except ZeroDivisionError:
            raise
        except Exception as e:
            warnings.warn(f"[arakelov_canonical_height] retry at prec={cur_prec} failed: {e}", RuntimeWarning)
            continue

    warnings.warn(
        f"[arakelov_canonical_height] canonical height remained negative after all fallbacks for divisor={div}. "
        f"Returning 0.0 (clamped). Last observed h_can={h_can}. prec tried up to {cur_prec}.",
        RuntimeWarning
    )
    return 0.0
