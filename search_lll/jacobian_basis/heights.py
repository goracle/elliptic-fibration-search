import warnings, multiprocessing, logging
from sage.all import QQ, RealField, PolynomialRing, HyperellipticCurve, Integer, QQbar
from .archimedean import archimedean_height_correction
from .periods import choose_numerical_base_point
from .theta import *
from search_lll.homology import *
from .local import get_bad_primes, local_height_correction_finite, local_correction_worker

"""Height computation functions."""

logger = logging.getLogger("mumford_adapter")

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

"""Height computation functions."""

logger = logging.getLogger("mumford_adapter")

# =========================================================================
# NEW: Weierstrass Point Detection and Handling
# =========================================================================

# =========================================================================
# Modified Functions
# =========================================================================

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

def _div_to_coeff_tuple(sage_div):
    try:
        u, v = sage_div[0], sage_div[1]
        return (coeffs_to_pairs(u), coeffs_to_pairs(v))
    except Exception:
        raise

def _div_to_coeff_tuple_for_worker(J_elem):
    """
    Robustly extract (u,v) Mumford data.
    """
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

"""
CRITICAL FIX: Weierstrass point corrections must NOT be included in
height pairing computations, only in standalone canonical heights.

The polarization formula h(<D1,D2>) = (h(D1+D2) - h(D1) - h(D2))/2
requires that h(·) be a quadratic form. Adding Weierstrass corrections
breaks this property because the correction depends on which Weierstrass
points appear in the support, not on the divisor class itself.
"""

# In heights.py, modify arakelov_canonical_height():

# In your arakelov pairing code (wherever arakelov_height_pairing is defined):

"""Height computation functions."""

logger = logging.getLogger("mumford_adapter")

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
            return 0

    # Get roots of u(x)
    try:
        # Try rational roots first
        div_roots = u_poly.roots(QQ, multiplicities=False)
    except Exception:
        return 0

    count = 0
    for w in weier_pts:
        if w in div_roots:
            count += 1

    return count

def local_height_at_weierstrass_points(div, f_coeffs, debug=False):
    """
    Compute the additional local height correction at Weierstrass points.
    For each Weierstrass point in the support, we need -log(2) correction.
    """
    num_weier = count_weierstrass_in_support(div, f_coeffs)

    if num_weier == 0:
        return 0.0

    correction = -float(num_weier) * math.log(2)

    if debug:
        print(f"  Weierstrass correction: {num_weier} points → {correction:.5f}")

    return correction

def naive_height_qq(div, prec=53):
    """
    Compute naive (logarithmic) height of Mumford u-polynomial coefficients.
    Strictly excludes v(x).
    """
    from sage.all import QQ, RealField
    from math import gcd

    u_coeffs = [QQ(c) for c in div[0].list()]
    all_coeffs = [c for c in u_coeffs if c != 0]

    if not all_coeffs:
        return QQ(0)

    dens = [c.denominator() for c in all_coeffs]
    lcm_den = 1
    for d in dens:
        if d == 1: continue
        lcm_den = (lcm_den * d) // gcd(lcm_den, d)

    int_coeffs = [int((c * lcm_den).numerator()) for c in all_coeffs]
    int_coeffs.append(lcm_den)

    max_abs = max(abs(c) for c in int_coeffs)
    R = RealField(prec)
    return R(max_abs).log()

def arakelov_canonical_height(div, f_coeffs, period_matrix, prec=1024,
                              max_prec=8192, debug=True,
                              include_weierstrass=False):
    """
    Computes canonical height h(D).

    Args:
        include_weierstrass: If False, omit Weierstrass corrections.
                            Use False when computing heights for pairings,
                            True for standalone divisor heights.
    """

    if isinstance(div, dict) and 'u_poly' in div:
        # Helper to convert dict back to element, assuming existence in module
        # (Using minimal reconstruction logic here if helper missing)
        try:
            from .mumford_basis import mumford_dict_to_jacobian_element
            J_elem = mumford_dict_to_jacobian_element(div, f_coeffs)
        except ImportError:
            # Fallback if circular import prevention needed
            # (Assume caller usually passes Jacobian element)
            raise RuntimeError("Cannot convert dict to Jacobian element without helper")
    else:
        J_elem = div

    jac = J_elem.parent()
    curve = jac.curve()
    f_poly, _ = curve.hyperelliptic_polynomials()
    consistent_coeffs = [QQ(c) for c in f_poly.list()[::-1]]

    if period_matrix is not None:
        if list(consistent_coeffs) != list(map(QQ, f_coeffs)):
            period_matrix = None

    # 2) compute archimedean (analytic) piece (without Weierstrass)
    h_arch_total = arakelov_quasi_height(J_elem, consistent_coeffs, period_matrix,
                                         prec=prec, use_finite_places=False)
    h_arch_total = float(h_arch_total)

    # 3) find bad primes
    bad_primes = get_bad_primes(consistent_coeffs)

    # 4) compute finite local corrections
    s_list = []

    # We must construct a clean tuple for the worker to avoid pickling the Jacobian element
    # if using multiprocessing, but here we run serial or lightweight
    from .local import _pairs_to_qq_poly
    # Helper extract
    try:
        u, v = J_elem[0], J_elem[1]
    except Exception:
        u, v = J_elem.divisor().reduced().mumford_representation()

    div_for_local = [u, v]

    for p in bad_primes:
        val = local_height_correction_finite(div_for_local, p, consistent_coeffs)
        s_list.append(float(val))

    sum_s = sum(s_list)

    # 5) Add Weierstrass correction ONLY if requested
    weier_correction = 0.0
    if include_weierstrass:
        weier_correction = local_height_at_weierstrass_points(J_elem, consistent_coeffs, debug=debug)

    h_can = float(h_arch_total + sum_s + weier_correction)

    return float(h_can)

def arakelov_height_pairing(D1, D2, f_coeffs, prec=1024, debug=False):
    """
    Compute <D1, D2> via polarization: (h(D1+D2) - h(D1) - h(D2))/2
    CRITICAL: Must use include_weierstrass=False to maintain bilinearity!
    """

    # Assumption: get_period_matrix_auto_B exists in scope or imported
    from .utilities import get_period_matrix_auto_B
    PM = get_period_matrix_auto_B(f_coeffs, prec=prec)

    h1 = arakelov_canonical_height(D1, f_coeffs, PM, prec=prec, debug=False,
                                   include_weierstrass=False)
    h2 = arakelov_canonical_height(D2, f_coeffs, PM, prec=prec, debug=False,
                                   include_weierstrass=False)

    D_sum = D1 + D2
    h_sum = arakelov_canonical_height(D_sum, f_coeffs, PM, prec=prec, debug=False,
                                     include_weierstrass=False)

    pairing = (h_sum - h1 - h2) / 2.0
    return float(pairing)

# ============================================================================
# FOR heights.py - add this import at top:
# from .archimedean import ThetaComputationError
# ============================================================================

def arakelov_quasi_height(div, f_coeffs, period_matrix, prec=300,
                          use_finite_places=True, arch_override=None):
    """
    Compute a quasi-canonical Arakelov height for a Mumford divisor `div`.
    """
    key = (str(div), tuple(f_coeffs), prec, use_finite_places, arch_override)
    if key in arakelov_quasi_height.cache:
        return arakelov_quasi_height.cache[key]
    assert period_matrix is not None, "period_matrix must be provided"

    if getattr(div, "is_zero", lambda: False)():
        return RealField(prec)(0)

    RF = RealField(prec)

    h_naive = naive_height_qq(div, prec=prec)
    h_total = RF(h_naive)

    if arch_override is not None:
        h_arch = RF(arch_override)
    else:
        try:
            h_arch = RF(archimedean_height_correction(div, f_coeffs, period_matrix, prec=prec))
        except (RuntimeError, ThetaComputationError) as e:
            raise RuntimeError(f"Archimedean height failed: {e}")

    h_total += h_arch

    h_finite = RF(0)
    if use_finite_places:
        bad_primes = get_bad_primes(f_coeffs)
        for p in bad_primes:
            val = local_height_correction_finite(div, p, f_coeffs)
            if val is None:
                raise RuntimeError(f"local correction returned None for p={p}")
            h_finite += RF(val)

    h_total += h_finite

    ret = h_total
    arakelov_quasi_height.cache[key] = ret
    return ret
arakelov_quasi_height.cache = {}
