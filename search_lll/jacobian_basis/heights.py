"""Height computation functions."""

import warnings
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve

from .archimedean import archimedean_height_correction
from .local import get_bad_primes, local_height_correction_finite
from .periods import choose_numerical_base_point
from search_lll.homology import *

# Functions: naive_height_qq, arakelov_quasi_height, arakelov_canonical_height
def archimedean_naive_height(div):
    u, v = div
    coeffs = u.list() + v.list()
    vals = [abs(float(c)) for c in coeffs if c != 0]
    if not vals:
        return 0.0
    return math.log(max(vals))


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


def arakelov_quasi_height(div, f_coeffs, period_matrix=None, prec=300,
                          use_finite_places=True, arch_override=None):
    """
    Compute a quasi-canonical Arakelov height for a Mumford divisor `div` on
    the genus-2 Jacobian of y^2 = f(x).

    Returned value is a RealField(prec) element (a real numeric approximation).
    Decomposition: naive_arch + archimedean_quadratic + sum(local_finite_corrections).

    Behavior / robustness:
      - If div.is_zero() -> returns 0.0 in RealField(prec).
      - If period_matrix is None, it will be computed (may raise on PM failures).
      - If many finite-place corrections fail (cached failures in local_height_correction_finite),
        we fall back to archimedean-only and warn.
      - All numeric contributions are accumulated in a RealField to avoid mistaken QQ casts.
    """
    key = (str(div), tuple(f_coeffs), prec, use_finite_places, arch_override)
    if key in arakelov_quasi_height.cache:
        return arakelov_quasi_height.cache[key]
    import warnings
    from sage.all import RealField, RR as sage_RR
    # functions assumed available in module scope:
    #   archimedean_naive_height(div)
    #   archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec)
    #   get_period_matrix_auto_B(f_coeffs, prec=prec)
    #   get_bad_primes(f_coeffs)
    #   local_height_correction_finite(div, p, f_coeffs, ...)
    #
    # Note: local_height_correction_finite maintains a set `failed_pairs` of (div_id,p)
    # when it gives up; we respect that cache here.

    # fast path
    if getattr(div, "is_zero", lambda: False)():
        return RealField(prec)(0)

    RF = RealField(prec)

    # 1) naive archimedean/global naive anchor (keeps heights positive in practice)
    try:
        h_inf_naive = archimedean_naive_height(div)
    except Exception as e:
        warnings.warn(f"[arakelov_quasi_height] archimedean_naive_height failed: {e}. Using 0.", RuntimeWarning)
        h_inf_naive = 0.0
        raise

    # ensure we work in RF
    h_total = RF(h_inf_naive)

    # 2) archimedean quadratic (theta / Green) term
    if arch_override is not None:
        try:
            h_arch = RF(arch_override)
        except Exception:
            h_arch = RF(0)
            raise
    else:
        try:
            if period_matrix is None:
                period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
            h_arch = RF(archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec))
        except Exception as e:
            warnings.warn(f"[arakelov_quasi_height] archimedean_height_correction failed: {e}. Using 0.", RuntimeWarning)
            h_arch = RF(0)

    h_total += h_arch

    # 3) finite local corrections (only at bad primes)
    h_finite = RF(0)
    failed_count = 0
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        # get the failed_pairs cache correctly (it's a set)
        failed_pairs = getattr(local_height_correction_finite, "failed_pairs", set())

        for p in bad_primes:
            try:
                val = local_height_correction_finite(div, p, f_coeffs)
            except Exception as e:
                # If the worker raises, treat as failure (and let the function's own
                # caching have recorded it).
                val = None
                warnings.warn(f"[arakelov_quasi_height] local_height_correction_finite raised for p={p}: {e}", RuntimeWarning)
                raise

            # if val is None treat as failure, else add numeric contribution
            try:
                div_id = (str(div[0]), str(div[1]))
            except Exception:
                div_id = (repr(div),)
                raise

            if val is None:
                # count as failure
                failed_count += 1
            else:
                # add contribution (cast into RF)
                try:
                    h_finite += RF(val)
                except Exception:
                    # fallback: try float -> RF
                    try:
                        h_finite += RF(float(val))
                    except Exception:
                        warnings.warn(f"[arakelov_quasi_height] could not cast finite correction for p={p}; skipping", RuntimeWarning)
                        raise
                    raise

            # account for cached failed_pairs recorded by the local routine
            if (div_id, p) in failed_pairs:
                failed_count += 1

        # If many finite corrections failed, fall back to arch-only
        if len(bad_primes) > 0 and failed_count >= max(1, len(bad_primes) // 2):
            warnings.warn(
                f"[arakelov_quasi_height] many finite-place corrections failed ({failed_count}/{len(bad_primes)}) "
                f"for divisor={div}. Falling back to archimedean-only for stability.",
                RuntimeWarning
            )
            h_finite = RF(0)

    h_total += h_finite

    # Final sanity: ensure non-negative-ish (small negative numerical noise -> clamp to 0)
    # but do not aggressively change large negatives (those indicate deeper problems)
    if float(h_total) < -1e-12:
        warnings.warn(f"[arakelov_quasi_height] height is slightly negative ({float(h_total)}). Returning as-is.", RuntimeWarning)

    ret = h_total
    arakelov_quasi_height.cache[key] = ret
    return ret
arakelov_quasi_height.cache = {}

# In heights.py

def arakelov_canonical_height(div, f_coeffs, prec=2048, max_prec=8192, debug=False, period_matrix=None):
    """
    Compute the Arakelov canonical height of `div`.
    Accepts optional `period_matrix` to avoid recomputing expensive periods.
    
    Raises ValueError if the computed height is negative (unstable/invalid).
    """
    key = (str(div), tuple(f_coeffs), prec, max_prec)
    if key in arakelov_canonical_height.cache:
        val = arakelov_canonical_height.cache[key]
        if val < -1e-9:
            # If a cached value is negative, it's poison; raise to force drop.
            raise ValueError(f"Cached canonical height is negative: {val}")
        return val

    import warnings
    from sage.all import QQ

    TOL_NEG = 1e-9  # Stricter tolerance
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
        # Don't return -1.0 yet; let the logic below decide to retry or fail
        
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
        except ZeroDivisionError:
            raise
        except Exception as e:
            warnings.warn(f"[arakelov_canonical_height] retry at prec={cur_prec} failed: {e}", RuntimeWarning)
            # Do not suppress exception if it's the last try
            if cur_prec >= max_prec:
                raise
            continue

    # If we reach here, h_can is still negative or unset.
    # CRITICAL CHANGE: Raise error instead of returning negative garbage.
    raise ValueError(f"Canonical height computation failed or remained negative ({h_can}) after retries for divisor {div}")

arakelov_canonical_height.cache = {}
