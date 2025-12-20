"""Height computation functions."""

import warnings
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve

from .archimedean import archimedean_height_correction
from .local import get_bad_primes, local_height_correction_finite
from .periods import choose_numerical_base_point
from search_lll.homology import *
import multiprocessing
from .local import get_bad_primes, local_height_correction_finite, local_correction_worker

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
    from math import gcd
    
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
    
    # Compute LCM of denominators without reduce()
    lcm_den = 1
    for d in dens:
        if d == 1:
            continue
        lcm_den = (lcm_den * d) // gcd(lcm_den, d)
    
    # Scale coefficients to integers
    int_coeffs = [int((c * lcm_den).numerator()) for c in all_coeffs]
    # Include the denominator itself in the height calculation (Projective height)
    # The height of [x0 : ... : xn] is max(|x_i|).
    int_coeffs.append(lcm_den)
    
    max_abs = max(abs(c) for c in int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log()


def arakelov_quasi_height(div, f_coeffs, period_matrix=None, prec=300,
                          use_finite_places=True, arch_override=None):
    """
    Compute a quasi-canonical Arakelov height for a Mumford divisor `div`.
    This version is SERIAL for safety. Parallelism is handled in canonical height.
    """
    key = (str(div), tuple(f_coeffs), prec, use_finite_places, arch_override)
    if key in arakelov_quasi_height.cache:
        return arakelov_quasi_height.cache[key]
    
    from sage.all import RealField

    # fast path
    if getattr(div, "is_zero", lambda: False)():
        return RealField(prec)(0)

    RF = RealField(prec)

    # 1) Naive Weil Height
    h_naive = naive_height_qq(div, prec=prec)
    h_total = RF(h_naive)

    # 2) Archimedean
    if arch_override is not None:
        h_arch = RF(arch_override)
    else:
        if period_matrix is None:
            period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
        h_arch = RF(archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec))

    h_total += h_arch

    # 3) Finite local corrections
    # Serial execution here - batching is done in canonical_height
    h_finite = RF(0)
    
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        for p in bad_primes:
            val = local_height_correction_finite(div, p, f_coeffs)
            if val is None:
                raise RuntimeError(f"[arakelov_quasi_height] local correction returned None for p={p}")
            h_finite += RF(val)

    h_total += h_finite

    if float(h_total) < -1e-9:
         pass

    ret = h_total
    arakelov_quasi_height.cache[key] = ret
    return ret
arakelov_quasi_height.cache = {}


def _div_to_coeff_tuple(sage_div):
    # sage_div is expected to be a Jacobian element or [u, v]
    try:
        u, v = sage_div[0], sage_div[1]
        # convert coefficients to rational pairs (num, den) so they are simple Python types
        def coeffs_to_pairs(poly):
            pairs = []
            for c in poly.list():
                cQQ = QQ(c)
                pairs.append((int(cQQ.numerator()), int(cQQ.denominator())))
            return tuple(pairs)
        return (coeffs_to_pairs(u), coeffs_to_pairs(v))
    except Exception:
        # fallback: stringify (cheap) — but prefer the normal path
        raise
        return str(sage_div)


# helper_jacobian_adapter.py (paste into heights.py or import)
from sage.all import PolynomialRing, QQ, HyperellipticCurve


# replace previous helper with this version
from sage.all import PolynomialRing, QQ, HyperellipticCurve, Integer
import logging

logger = logging.getLogger("mumford_adapter")

def _rational_sqrt_if_exact(q):
    """Return QQ(s) if q is exact rational square, else None."""
    try:
        qqq = QQ(q)
        if qqq < 0:
            return None
        a = Integer(qqq.numerator()); b = Integer(qqq.denominator())
        if a.is_square() and b.is_square():
            return QQ(Integer(a.isqrt()) / Integer(b.isqrt()))
        return None
    except Exception:
        raise
        return None

def build_f_poly_from_coeffs(f_coeffs, R=None):
    """
    Build f(x) from f_coeffs highest->constant into polynomial ring R.
    """
    if R is None:
        R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f = R(0)
    for c in f_coeffs:
        f = f * x + QQ(c)
    return f

def _scale_f_coeffs_by(f_coeffs, denom):
    """
    Return new list of coefficients f_coeffs / denom (denom in QQ).
    Assumes denom != 0.
    """
    return [ QQ(c) / QQ(denom) for c in f_coeffs ]

def mumford_dict_to_jacobian_element(div_dict, f_coeffs):
    """
    Convert canonicalized dict {'u_poly','v_poly',...} into a Sage Jacobian element,
    but first normalize the hyperelliptic model when possible to a monic sextic by
    dividing f by its leading coefficient LC and scaling v by 1/sqrt(LC) when LC is
    an exact rational square.

    Returns Sage Jacobian element J_elem. Raises on failure.
    """
    if not isinstance(div_dict, dict) or 'u_poly' not in div_dict or 'v_poly' not in div_dict:
        raise TypeError("Expected canonicalized dict with 'u_poly' and 'v_poly'")

    # Determine leading coefficient (first element of f_coeffs list)
    LC = QQ(f_coeffs[0])
    lc_sqrt = _rational_sqrt_if_exact(LC)

    # If LC is exact rational square, switch to monic model by dividing coeffs by LC
    if lc_sqrt is not None and LC != 1:
        logger.info("Scaling model to monic by dividing f by LC=%s and scaling v by 1/%s", LC, lc_sqrt)
        f_coeffs_scaled = _scale_f_coeffs_by(f_coeffs, LC)
        # We will scale v by 1 / lc_sqrt
        v_scale = QQ(1) / QQ(lc_sqrt)
    else:
        # no scaling
        f_coeffs_scaled = list(map(QQ, f_coeffs))
        v_scale = QQ(1)

    # build polynomial ring and monic f
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = build_f_poly_from_coeffs(f_coeffs_scaled, R)

    # Build u and v from div_dict, but convert to polynomials of R (QQ)
    u = div_dict['u_poly']
    v = div_dict['v_poly']

    # Ensure u,v are in the same ring R and over QQ
    try:
        if u.parent().gen() != R.gen() or u.parent().base_ring() != QQ:
            # change ring if necessary
            u = R(u.list())
    except Exception:
        # fallback: coerce via coefficients
        u = R(u.list())
        raise

    try:
        if v.parent().gen() != R.gen() or v.parent().base_ring() != QQ:
            v = R(v.list())
    except Exception:
        v = R(v.list())
        raise

    # apply global v scaling needed for monic model and also combine with any recorded scale_used
    total_scale = QQ(1)
    if 'scale_used' in div_dict:
        try:
            total_scale = QQ(div_dict['scale_used'])
        except Exception:
            total_scale = QQ(1)
    # first apply recorded scale (this was used to recover algebraic relation earlier)
    # then apply model v_scale (1/sqrt(LC)) to match monic model
    combined_scale = total_scale * v_scale

    if combined_scale != 1:
        try:
            v = combined_scale * v
        except Exception:
            # fallback: rebuild coefficients multiplied by combined_scale
            v = R([ QQ(c) * combined_scale for c in v.list() ])
            raise

    # Now ensure u is monic and v reduced mod u
    u = u.monic()
    v = v % u

    # Build curve and Jacobian for the monic model
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    # Final safety: ensure deg v < deg u (reduction)
    if v.degree() >= u.degree():
        v = v % u

    # Construct Jacobian element (this gives the Pic^0 class)
    try:
        J_elem = J(u, v)
    except Exception as e:
        # Try alternative constructor: J._from_mumford if available (version differences)
        try:
            J_elem = J._from_mumford(u, v)
        except Exception:
            logger.exception("Failed to construct Jacobian element from (u,v) after normalization")
            raise
        raise

    return J_elem


# Replace existing arakelov_canonical_height with this debug-friendly version.
# Requires: mumford_dict_to_jacobian_element, arakelov_quasi_height, get_bad_primes,
# and local_correction_worker (or a per-prime local correction API) to be present in the module.
from sage.all import QQ

logger = logging.getLogger("arakelov_debug")
logger.setLevel(logging.INFO)

def arakelov_canonical_height(div, f_coeffs, prec=1024, max_prec=8192, debug=True, period_matrix=None):
    """
    Debug-friendly canonical height builder.
    Computes:
       h_arch = arakelov_quasi_height(J_elem, use_finite_places=False)
       s_p for p in bad_primes (computed one-by-one)
       h_can = h_arch + sum_p s_p

    Accepts either:
      - a Sage Jacobian element, or
      - a canonicalized dict with 'u_poly'/'v_poly' fields.
    Returns float h_can. Does not raise immediately on negative result; logs full breakdown.
    """
    # 1) ensure we have a Sage Jacobian element
    try:
        # if it's a dict from canonicalizer, convert
        if isinstance(div, dict) and 'u_poly' in div and 'v_poly' in div:
            # use the normalized conversion helper (monic model scaling inside)
            J_elem = mumford_dict_to_jacobian_element(div, f_coeffs)
        else:
            J_elem = div  # assume it's already a Jacobian element
    except Exception as e:
        logger.exception("Failed to convert input to Jacobian element: %s", e)
        raise

    # 2) compute archimedean (analytic) piece
    try:
        # arakelov_quasi_height should accept a Jacobian element and return a real value,
        # with option to skip finite places (we use false to compute only analytic part)
        # many codebases name it differently; adapt to your code if needed.
        h_arch = arakelov_quasi_height(J_elem, f_coeffs, period_matrix=period_matrix, prec=prec, use_finite_places=False)
        h_arch = float(h_arch)
    except Exception as e:
        logger.exception("archimedean contribution failed: %s", e)
        # still continue to compute finite corrections for diagnostics
        h_arch = None
        raise

    # 3) find bad primes (same code your logging shows)
    try:
        bad_primes = get_bad_primes(f_coeffs)
    except Exception:
        # fallback: if you have cached list or computed earlier, adapt accordingly
        bad_primes = []
        logger.warning("get_bad_primes unavailable; proceeding with empty list of bad primes")
        raise

    # 4) compute finite local corrections one-by-one (s_p)
    s_list = []
    per_prime = {}
    for p in bad_primes:
        try:
            # if you already have a worker function that returns (idx, s_p) for a task,
            # use local_naive_height_p or local_correction_worker adapted for single prime.
            # Example: local_naive_height_p(J_elem, f_coeffs, p, prec) -> s_p (maybe returns list)
            # Use whichever function in your code computes the finite correction for a single p.
            # I'm using a conservative call to local_correction_worker(task) expecting it to accept a single-task tuple.
            task = (0, _div_to_coeff_tuple_for_worker(J_elem), p, tuple(f_coeffs))
            # NOTE: if your local_correction_worker expects a different format, replace accordingly.
            # Use direct call (no multiprocessing) for deterministic logging:
            idx, s_p = local_correction_worker(task)
            s_val = float(s_p)
            s_list.append(s_val)
            per_prime[p] = s_val
        except Exception as e:
            # log exception for that prime; append None so indices line up
            logger.exception("Local correction failed for p=%r: %s", p, e)
            per_prime[p] = None
            s_list.append(None)
            raise

    # 5) combine using additive formula: h_can = h_arch + sum(s_p)
    # Only sum non-None s_p
    sum_s = 0.0
    for s in s_list:
        if s is None:
            raise RuntimeError("Critical failure: Local height correction is missing. Result is invalid.")
        sum_s += float(s)

    h_can = None
    if h_arch is None:
        # if archimedean failed, try fallback: maybe arakelov_quasi_height with use_finite_places=True returns total
        try:
            h_try = arakelov_quasi_height(J_elem, f_coeffs, period_matrix=period_matrix, prec=prec, use_finite_places=True)
            h_can = float(h_try)
            logger.info("[fallback] arakelov_quasi_height returned full h_can=%r", h_can)
        except Exception:
            logger.error("archimedean contribution missing and fallback failed.")
            h_can = None
            raise
    else:
        h_can = float(h_arch + sum_s)

    # 6) debugging output
    logger.info("=== ARakelov height debug ===")
    logger.info("Div (repr): %s", getattr(J_elem, "__repr__", lambda: "<repr-failed>")()[:200] if hasattr(J_elem, "__repr__") else str(J_elem))
    logger.info("archimedean h_arch = %r (prec=%d)", h_arch, prec)
    logger.info("bad primes = %r", bad_primes)
    for p in bad_primes:
        logger.info("  p=%r -> s_p = %r", p, per_prime.get(p))
    logger.info("sum finite corrections = %r", sum_s)
    logger.info("combined h_can = %r", h_can)

    # 7) If h_can negative, don't raise here: return the value but also dump extra diagnostics
    if h_can is None:
        logger.error("h_can is None (archimedean and fallback failed). returning None.")
        return None

    if h_can < -1e-12:
        # extra diagnosis: compute naive height or check torsion
        try:
            # If Sage exposes canonical_height for Jacobian elements, use that
            if hasattr(J_elem, "height") or hasattr(J_elem, "canonical_height"):
                try:
                    naive = J_elem.canonical_height() if hasattr(J_elem, "canonical_height") else J_elem.height()
                    logger.info("Sage-reported canonical/naive height: %r", naive)
                except Exception:
                    raise
        except Exception:
            raise

        logger.warning("Canonical height negative: h_can=%r for divisor; returning value (not raising) for debug.", h_can)
        # return the negative value so caller can decide; but do not raise
        return float(h_can)

    # Otherwise return positive height
    return float(h_can)
arakelov_canonical_height.cache = {}


# Helper to adapt a Jacobian element for the local worker tuple format.
# Replace this with the exact function you use to build the worker tuple from a Mumford pair.


def _div_to_coeff_tuple_for_worker(J_elem):
    """
    Robustly extract (u,v) Mumford data from a Sage Jacobian element
    and convert to the format expected by local_correction_worker:
    ((u_pairs), (v_pairs)) where each pair is (numerator, denominator).
    """
    from sage.all import QQ
    
    # Try direct access to Mumford representation first
    try:
        # For JacobianMorphism_divisor_class_field, access the internal _data attribute
        # which contains the Mumford representation (u, v)
        if hasattr(J_elem, '_data'):
            u, v = J_elem._data
        else:
            # Fallback: try accessing as tuple
            u, v = J_elem[0], J_elem[1]
    except Exception as e1:
        # Try alternative methods
        try:
            # Some Sage versions support direct indexing
            u = J_elem[0]
            v = J_elem[1]
        except Exception as e2:
            # Last resort: try getting through divisor and reduction
            try:
                D = J_elem.divisor()
                Dred = D.reduced()
                u, v = Dred.mumford_representation()
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to extract Mumford (u,v) from Jacobian element of type {type(J_elem)}. "
                    f"Tried _data access: {e1}, indexing: {e2}, divisor method: {e3}"
                )
            raise
        raise
    
    # Convert to coefficient pairs (numerator, denominator) as expected by worker
    def coeffs_to_pairs(poly):
        """Convert polynomial coefficients to (num, den) pairs."""
        pairs = []
        for c in poly.list():
            cQQ = QQ(c)
            pairs.append((int(cQQ.numerator()), int(cQQ.denominator())))
        return tuple(pairs)
    
    try:
        u_pairs = coeffs_to_pairs(u)
        v_pairs = coeffs_to_pairs(v)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert Mumford polynomials to coefficient pairs: {e}"
        )
    
    return (u_pairs, v_pairs)
