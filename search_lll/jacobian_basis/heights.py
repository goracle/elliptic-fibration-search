"""Height computation functions."""

import warnings
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve

from .archimedean import archimedean_height_correction
from .periods import choose_numerical_base_point
from search_lll.homology import *
import multiprocessing
from .local import get_bad_primes, local_height_correction_finite, local_correction_worker
from sage.all import PolynomialRing, QQ, HyperellipticCurve
from sage.all import PolynomialRing, QQ, HyperellipticCurve, Integer
from sage.all import QQ
import logging

logger = logging.getLogger("mumford_adapter")


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


"""Height computation functions."""

from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve, QQbar


logger = logging.getLogger("mumford_adapter")

# =========================================================================
# NEW: Weierstrass Point Detection and Handling
# =========================================================================

def get_weierstrass_points(f_coeffs, prec=300):
    """
    Find all rational Weierstrass points (roots of f(x)).
    For y^2 = f(x), Weierstrass points are where f(x) = 0.
    """
    key = tuple(f_coeffs)
    if key in get_weierstrass_points.cache:
        return get_weierstrass_points.cache[key]
    
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    weier_pts = []
    try:
        # Get all rational roots
        roots = f_poly.roots(QQ, multiplicities=False)
        weier_pts.extend(roots)
    except Exception as e:
        logger.warning(f"Could not compute rational Weierstrass points: {e}")
    
    result = sorted(weier_pts)
    get_weierstrass_points.cache[key] = result
    return result

get_weierstrass_points.cache = {}


def count_weierstrass_in_support(div, f_coeffs, tolerance=1e-10):
    """
    Count how many Weierstrass points appear in the support of divisor div.
    
    Args:
        div: Sage Jacobian element with Mumford representation [u(x), v(x)]
        f_coeffs: Coefficients of the curve polynomial (high to low degree)
        tolerance: Numerical tolerance for root matching
    
    Returns:
        int: Number of Weierstrass points in the divisor's support
    """
    weier_pts = get_weierstrass_points(f_coeffs)
    if not weier_pts:
        return 0
    
    # Extract u(x) from Mumford representation
    try:
        u_poly = div[0]
    except Exception:
        try:
            u_poly, _ = div.mumford_representation()
        except Exception as e:
            logger.warning(f"Could not extract u_poly from divisor: {e}")
            return 0
    
    # Get roots of u(x) - these are the x-coordinates in the divisor's support
    try:
        # Try rational roots first
        div_roots = u_poly.roots(QQ, multiplicities=False)
        if not div_roots:
            # Fall back to algebraic closure if needed
            div_roots = u_poly.roots(QQbar, multiplicities=False)
    except Exception as e:
        logger.warning(f"Could not compute roots of u_poly: {e}")
        return 0
    
    # Count matches between divisor roots and Weierstrass points
    count = 0
    for w in weier_pts:
        w_alg = QQbar(w)
        for r in div_roots:
            r_alg = QQbar(r)
            if abs(w_alg - r_alg) < tolerance:
                count += 1
                break  # Don't double-count the same Weierstrass point
    
    return count


def local_height_at_weierstrass_points(div, f_coeffs, debug=False):
    """
    Compute the additional local height correction at Weierstrass points.
    
    For each Weierstrass point in the support, we need -log(2) correction
    due to ramification at these points.
    
    Returns:
        float: Total correction (typically negative)
    """
    num_weier = count_weierstrass_in_support(div, f_coeffs)
    
    if num_weier == 0:
        return 0.0
    
    import math
    correction = -float(num_weier) * math.log(2)
    
    if debug:
        print(f"  Weierstrass correction: {num_weier} points → {correction:.5f}")
    
    return correction

# =========================================================================
# Modified Functions
# =========================================================================

def naive_height_qq(div, prec=53):
    """
    Compute naive (logarithmic) height of Mumford u-polynomial coefficients.
    Strictly excludes v(x) to match Kummer surface height definitions.
    """
    from sage.all import QQ, RealField
    from math import gcd
    
    # Extract coefficients from u(x) only
    # div[0] is u(x), div[1] is v(x)
    u_coeffs = [QQ(c) for c in div[0].list()]
    
    all_coeffs = [c for c in u_coeffs if c != 0]
    
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
    # Include the denominator itself (Projective height of [coeffs : 1])
    int_coeffs.append(lcm_den)
    
    max_abs = max(abs(c) for c in int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log()


def archimedean_naive_height(div):
    """
    Float version of naive height for diagnostics.
    """
    import math
    u, _ = div
    coeffs = u.list()
    vals = [abs(float(c)) for c in coeffs if c != 0]
    if not vals:
        return 0.0
    return math.log(max(vals))


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def _div_to_coeff_tuple(sage_div):
    try:
        u, v = sage_div[0], sage_div[1]
        def coeffs_to_pairs(poly):
            pairs = []
            for c in poly.list():
                cQQ = QQ(c)
                pairs.append((int(cQQ.numerator()), int(cQQ.denominator())))
            return tuple(pairs)
        return (coeffs_to_pairs(u), coeffs_to_pairs(v))
    except Exception:
        raise

def _div_to_coeff_tuple_for_worker(J_elem):
    """
    Robustly extract (u,v) Mumford data.
    """
    from sage.all import QQ
    try:
        if hasattr(J_elem, '_data'): 
            u, v = J_elem._data
        else:
            u, v = J_elem[0], J_elem[1]
    except Exception:
        try:
            u, v = J_elem.divisor().reduced().mumford_representation()
        except Exception as e:
            raise RuntimeError(f"Could not extract uv: {e}")
        raise

    def coeffs_to_pairs(poly):
        pairs = []
        for c in poly.list():
            cQQ = QQ(c)
            pairs.append((int(cQQ.numerator()), int(cQQ.denominator())))
        return tuple(pairs)
    
    return (coeffs_to_pairs(u), coeffs_to_pairs(v))

def _rational_sqrt_if_exact(q):
    try:
        qqq = QQ(q)
        if qqq < 0:
            return None
        a = qqq.numerator(); b = qqq.denominator()
        if a.is_square() and b.is_square():
            return QQ(a.isqrt()) / QQ(b.isqrt())
        return None
    except Exception:
        raise

def build_f_poly_from_coeffs(f_coeffs, R=None):
    if R is None:
        R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f = R(0)
    for c in f_coeffs:
        f = f * x + QQ(c)
    return f

def _scale_f_coeffs_by(f_coeffs, denom):
    return [ QQ(c) / QQ(denom) for c in f_coeffs ]

def mumford_dict_to_jacobian_element(div_dict, f_coeffs):
    """
    Convert canonicalized dict into a Sage Jacobian element.
    Normalizes to monic sextic if LC is square.
    """
    if not isinstance(div_dict, dict) or 'u_poly' not in div_dict or 'v_poly' not in div_dict:
        raise TypeError("Expected canonicalized dict with 'u_poly' and 'v_poly'")

    LC = QQ(f_coeffs[0])
    lc_sqrt = _rational_sqrt_if_exact(LC)

    if lc_sqrt is not None and LC != 1:
        f_coeffs_scaled = _scale_f_coeffs_by(f_coeffs, LC)
        v_scale = QQ(1) / QQ(lc_sqrt)
    else:
        f_coeffs_scaled = list(map(QQ, f_coeffs))
        v_scale = QQ(1)

    R = PolynomialRing(QQ, 'x')
    f_poly = build_f_poly_from_coeffs(f_coeffs_scaled, R)

    u = div_dict['u_poly']
    v = div_dict['v_poly']

    try:
        if u.parent().gen() != R.gen() or u.parent().base_ring() != QQ:
            u = R(u.list())
    except Exception:
        u = R(u.list())
        raise

    try:
        if v.parent().gen() != R.gen() or v.parent().base_ring() != QQ:
            v = R(v.list())
    except Exception:
        v = R(v.list())
        raise

    total_scale = QQ(1)
    if 'scale_used' in div_dict:
        try:
            total_scale = QQ(div_dict['scale_used'])
        except Exception:
            total_scale = QQ(1)
            raise
            
    combined_scale = total_scale * v_scale

    if combined_scale != 1:
        try:
            v = combined_scale * v
        except Exception:
            v = R([ QQ(c) * combined_scale for c in v.list() ])
            raise

    u = u.monic()
    v = v % u

    C = HyperellipticCurve(f_poly)
    J = C.jacobian()

    if v.degree() >= u.degree():
        v = v % u

    try:
        J_elem = J(u, v)
    except Exception:
        try:
            J_elem = J._from_mumford(u, v)
        except Exception:
            raise
        raise

    return J_elem


def arakelov_quasi_height(div, f_coeffs, period_matrix, prec=300,
                          use_finite_places=True, arch_override=None):
    """
    Compute a quasi-canonical Arakelov height for a Mumford divisor `div`.
    """
    key = (str(div), tuple(f_coeffs), prec, use_finite_places, arch_override)
    if key in arakelov_quasi_height.cache:
        return arakelov_quasi_height.cache[key]
    assert period_matrix is not None, "period_matrix must be provided"
    
    from sage.all import RealField

    # fast path
    if getattr(div, "is_zero", lambda: False)():
        return RealField(prec)(0)

    RF = RealField(prec)

    # 1) Naive Weil Height (u-only)
    h_naive = naive_height_qq(div, prec=prec)
    h_total = RF(h_naive)

    # 2) Archimedean (now includes Weierstrass corrections internally)
    if arch_override is not None:
        h_arch = RF(arch_override)
    else:
        # Note: archimedean_height_correction handles the difference 
        # between the full analytic height and the naive height
        h_arch = RF(archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec))

    h_total += h_arch

    # 3) Finite local corrections
    h_finite = RF(0)
    
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        for p in bad_primes:
            val = local_height_correction_finite(div, p, f_coeffs)
            if val is None:
                raise RuntimeError(f"[arakelov_quasi_height] local correction returned None for p={p}")
            h_finite += RF(val)

    h_total += h_finite

    ret = h_total
    arakelov_quasi_height.cache[key] = ret
    return ret
arakelov_quasi_height.cache = {}


"""
CRITICAL FIX: Weierstrass point corrections must NOT be included in 
height pairing computations, only in standalone canonical heights.

The polarization formula h(<D1,D2>) = (h(D1+D2) - h(D1) - h(D2))/2
requires that h(·) be a quadratic form. Adding Weierstrass corrections
breaks this property because the correction depends on which Weierstrass 
points appear in the support, not on the divisor class itself.
"""

# In heights.py, modify arakelov_canonical_height():

def arakelov_canonical_height(div, f_coeffs, period_matrix, prec=1024, 
                              max_prec=8192, debug=True, 
                              include_weierstrass=False):  # NEW FLAG
    """
    Computes canonical height h(D) = h_arch(D) + sum(local_corrections).
    
    Args:
        include_weierstrass: If False, omit Weierstrass corrections.
                            Use False when computing heights for pairings,
                            True for standalone divisor heights.
    """
    from sage.all import QQ
    
    # 1) ensure we have a Sage Jacobian element
    if isinstance(div, dict) and 'u_poly' in div:
        J_elem = mumford_dict_to_jacobian_element(div, f_coeffs)
    else:
        J_elem = div

    # CRITICAL: Access the curve via the parent Jacobian group.
    jac = J_elem.parent()
    curve = jac.curve()
    
    f_poly, _ = curve.hyperelliptic_polynomials()
    consistent_coeffs = [QQ(c) for c in f_poly.list()[::-1]]
    
    if period_matrix is not None:
        if list(consistent_coeffs) != list(map(QQ, f_coeffs)):
            period_matrix = None

    # 2) compute archimedean (analytic) piece
    h_arch_total = arakelov_quasi_height(J_elem, consistent_coeffs, period_matrix, 
                                         prec=prec, use_finite_places=False)
    h_arch_total = float(h_arch_total)

    # 3) find bad primes
    bad_primes = get_bad_primes(consistent_coeffs)

    # 4) compute finite local corrections
    s_list = []
    per_prime = {}
    
    for p in bad_primes:
        task = (0, _div_to_coeff_tuple_for_worker(J_elem), p, tuple(consistent_coeffs))
        idx, result = local_correction_worker(task)
        
        if isinstance(result, Exception):
            raise result
        
        s_val = float(result)
        s_list.append(s_val)
        per_prime[p] = s_val

    sum_s = sum(s_list)
    
    # 5) FIXED: Only add Weierstrass correction if requested
    weier_correction = 0.0
    assert not include_weierstrass, "mathematically mistaken, weierstrass correction not possible"
    if include_weierstrass:
        weier_correction = local_height_at_weierstrass_points(J_elem, consistent_coeffs, debug=debug)
    
    h_can = float(h_arch_total + sum_s + weier_correction)

    # 6) debugging output
    if debug:
        h_naive_val = float(naive_height_qq(J_elem, prec=prec))
        h_arch_correction = h_arch_total - h_naive_val
        num_weier = count_weierstrass_in_support(J_elem, consistent_coeffs)
        
        print(f"Height breakdown (Model LC={consistent_coeffs[0]}):")
        print(f"  Naive(u-only) = {h_naive_val:.5f}")
        print(f"  ArchCorr      = {h_arch_correction:.5f}")
        print(f"  Locals        = {sum_s:.5f} (p=2: {per_prime.get(2, 'N/A')})")
        print(f"  Total         = {h_can:.5f}")

    if h_can < -1e-9 and include_weierstrass:
        print(f"WARNING: Canonical height negative: {h_can}. Matrix may not be positive definite.")
    
    return float(h_can)


# In your arakelov pairing code (wherever arakelov_height_pairing is defined):

def arakelov_height_pairing(D1, D2, f_coeffs, prec=1024, debug=False):
    """
    Compute <D1, D2> via polarization: (h(D1+D2) - h(D1) - h(D2))/2
    
    CRITICAL: Must use include_weierstrass=False to maintain bilinearity!
    """
    from sage.all import QQ
    
    # Get period matrix once
    PM = get_period_matrix_auto_B(f_coeffs, prec=prec)
    
    # Compute heights WITHOUT Weierstrass corrections
    h1 = arakelov_canonical_height(D1, f_coeffs, PM, prec=prec, debug=False,
                                   include_weierstrass=False)
    h2 = arakelov_canonical_height(D2, f_coeffs, PM, prec=prec, debug=False,
                                   include_weierstrass=False)
    
    # Compute sum
    D_sum = D1 + D2
    h_sum = arakelov_canonical_height(D_sum, f_coeffs, PM, prec=prec, debug=False,
                                     include_weierstrass=False)
    
    # Polarization formula
    pairing = (h_sum - h1 - h2) / 2.0
    
    if debug:
        print(f"  Pair: h1={h1:.5f}, h2={h2:.5f}, h_sum={h_sum:.5f}, <D1,D2>={pairing:.5f}")
    
    return float(pairing)


