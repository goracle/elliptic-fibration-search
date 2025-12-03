# mobius.py
#
# Möbius transformation utilities for elliptic-fibration experiments.
# Everything is strict: symbolic, exact, and must raise on failure.
#
# Provides:
#   * MobiusTransform(a,b,c,d)  -- represents T(x) = (a x + b)/(c x + d)
#   * apply_to_poly(f, T)       -- substitute T(x) into a polynomial f(x)
#   * apply_to_points(points,T) -- transform (x,y) pairs (y unchanged)
#   * prime_support(expr)       -- primes dividing num/den of a QQ expression
#   * choose_transform(...)     -- automated search over candidate Möbius transforms
#
# This module gives you the promised "prime moving knob":
#   choose_transform(f, pts, avoid={2,3,5}, search_range=5)
#
# The returned transform can be used to drive tower.sage’s iterate_tower.
#

from sage.all import QQ, PolynomialRing, factor, SR, var, Integer
import itertools


class MobiusTransform:
    """
    Represents T(x) = (a x + b) / (c x + d), with ad - bc != 0.
    a,b,c,d must be QQ.
    """
    def __init__(self, a, b, c, d):
        try:
            self.a = QQ(a)
            self.b = QQ(b)
            self.c = QQ(c)
            self.d = QQ(d)
        except Exception as e:
            raise RuntimeError(f"Mobius coeffs not coercible to QQ: {e}")

        det = self.a*self.d - self.b*self.c
        if det == 0:
            raise RuntimeError("Invalid Möbius transform: ad - bc = 0")

        self.det = det

    def __repr__(self):
        return f"MobiusTransform(a={self.a}, b={self.b}, c={self.c}, d={self.d})"

    def apply(self, xexpr):
        """
        Return T(xexpr) = (a*xexpr + b)/(c*xexpr + d)
        Raises if denominator vanishes symbolically.
        """
        num = self.a * xexpr + self.b
        den = self.c * xexpr + self.d
        if den == 0:
            raise RuntimeError("Möbius transform produced zero denominator.")
        return num/den

    def inverse_transform(self, x):
        """
        Apply the inverse transform T^-1(x) = (dx - b) / (-cx + a)
        """
        num = self.d * x - self.b
        den = -self.c * x + self.a
        if den == 0:
            raise RuntimeError("Inverse transform produced zero denominator.")
        return num / den


def apply_to_points(points, T):
    """
    Transform a list/set of points (x,y)   →   (T(x), y).
    Does not modify y because you are using transformed x in fibration construction.
    """
    out = []
    for x in points:
        try:
            xx = T.apply(QQ(x))
        except Exception:
            raise
        out.append(xx)
        #out.append((xx, QQ(y)))
    return out


def total_prime_support_poly(poly):
    """
    Union of prime supports of all coefficients of a polynomial.
    """
    ps = set()
    for c in poly.list():
        ps |= prime_support(c)
    return ps


# =====================================================================
#       Automatic transform selector (smart knob)
# =====================================================================


def prime_support(expr):
    """
    Return a Python set of primes dividing numerator or denominator of a QQ/SR expression.
    """
    try:
        q = QQ(expr)
    except Exception:
        q = QQ(SR(expr))

    num = Integer(q.numerator())
    den = Integer(q.denominator())
    ps = set()

    for s in (num, den):
        s_abs = abs(Integer(s))
        if s_abs in (0, 1):
            continue
        fac = factor(s_abs)          # Factorization-like object
        for p, exp in fac:          # iterate (prime, exponent) pairs
            ps.add(Integer(p))

    return ps


# Replace apply_to_poly with this exact function


# Replace choose_transform with this exact function body (only changed to handle the new error)


# Replace apply_to_poly with this exact function


# Replace choose_transform with this exact function body (only changed to handle the new error)


def test_transform_on_points(fx, T, test_points, verbose=False):
    """
    Test if applying transform T to fx preserves rationality when evaluated at test_points.
    Returns True if transform is safe, False otherwise.
    """
    a, b, c, d = T.a, T.b, T.c, T.d
    
    # For each test point, compute T(x) and check if we can evaluate fx(T(x)) rationally
    for pt in test_points:
        try:
            if isinstance(pt, tuple):
                x_val = QQ(pt[0])
            else:
                x_val = QQ(pt)
                
            # Compute T(x_val)
            Tx_num = a * x_val + b
            Tx_den = c * x_val + d
            
            if Tx_den == 0:
                if verbose:
                    print(f"  Transform {T} makes denominator zero at x={x_val}")
                return False
                
            Tx_val = Tx_num / Tx_den
            
            # Try to evaluate fx at T(x_val)
            result = fx(Tx_val)
            
            # Make sure result is rational
            _ = QQ(result)
            
        except (TypeError, ValueError, ZeroDivisionError) as e:
            if verbose:
                print(f"  Transform {T} failed at x={x_val}: {e}")
            return False
    
    return True


def choose_transform(
    fx,
    base_points,
    avoid_primes=None,
    prefer_primes=None,
    search_range=3,
    allow_c_nonzero=False
):
    if avoid_primes is None:
        avoid_primes = set()
    if prefer_primes is None:
        prefer_primes = set()

    best = None
    best_score = None
    best_primes = None

    vals = list(range(-search_range, search_range+1))
    
    # Convert base_points to a list for testing
    test_pts = []
    for xx in base_points:
        try:
            test_pts.append(QQ(xx))
        except:
            try:
                test_pts.append(QQ(xx[0]))
            except:
                pass

    print(f"[mobius] Searching {len(vals)**4} transforms with range={search_range}")
    print(f"[mobius] Test points: {test_pts}")
    print(f"[mobius] Avoiding primes: {sorted(avoid_primes)}")
    print(f"[mobius] Preferring primes: {sorted(prefer_primes)}")
    
    candidates_tested = 0
    candidates_rejected_det = 0
    candidates_rejected_point_test = 0
    candidates_rejected_poly_transform = 0
    candidates_accepted = 0

    for a,b,c,d in itertools.product(vals, vals, vals, vals):
        if a*d - b*c == 0:
            candidates_rejected_det += 1
            continue
        if not allow_c_nonzero and c != 0:
            continue
        if all(v == 0 for v in [a,b,c,d]):
            continue

        candidates_tested += 1

        try:
            T = MobiusTransform(a,b,c,d)
        except Exception:
            continue

        # FIRST: Test on actual points (this is fast)
        if test_pts and not test_transform_on_points(fx, T, test_pts):
            candidates_rejected_point_test += 1
            continue

        # SECOND: Try the full polynomial transformation (this is slow)
        try:
            fT = apply_to_poly(fx, T)
        except RuntimeError:
            candidates_rejected_poly_transform += 1
            continue
        except Exception:
            candidates_rejected_poly_transform += 1
            continue

        # Transform base x-values
        try:
            transformed_pts = []
            for xx in base_points:
                try:
                    X = QQ(xx)
                except Exception:
                    X = QQ(xx[0])
                num = T.a * X + T.b
                den = T.c * X + T.d
                if den == 0:
                    raise RuntimeError("apply_to_points: denominator zero for base point")
                transformed_pts.append(num/den)
        except Exception:
            continue

        ps = total_prime_support_poly(fT)

        bad = ps & avoid_primes
        good = ps & prefer_primes

        # More aggressive scoring: 
        # - Heavily penalize ANY bad prime
        # - Prefer transforms that completely avoid bad primes
        # - Secondary: minimize total number of primes
        # - Tertiary: reward good primes
        if bad:
            score = 10000 * len(bad) + 100 * len(ps) - 10 * len(good)
        else:
            # No bad primes! This is great - just minimize total primes
            score = len(ps) - 10 * len(good)

        if (best_score is None) or (score < best_score):
            best = T
            best_score = score
            best_primes = ps
            candidates_accepted += 1
            
            # Print when we find a better candidate
            if candidates_accepted <= 10:  # Only print first 10
                print(f"  New best: T={T}, score={score}, primes={sorted(ps)}, bad={sorted(bad)}")

        # Early exit if we find a perfect transform (no bad primes, minimal total)
        if not bad and len(ps) <= 3:
            print(f"  Found perfect transform early!")
            break

    print(f"[mobius] Search complete:")
    print(f"  Tested: {candidates_tested}")
    print(f"  Rejected (det=0): {candidates_rejected_det}")
    print(f"  Rejected (point test): {candidates_rejected_point_test}")
    print(f"  Rejected (poly transform): {candidates_rejected_poly_transform}")
    print(f"  Candidates evaluated for score: {candidates_accepted}")
    
    if best is None:
        raise RuntimeError("No acceptable Möbius transform found in search window.")
    
    print(f"\n[mobius] Selected transform: {best}")
    print(f"  Score: {best_score}")
    print(f"  Prime support: {sorted(best_primes)}")
    print(f"  Bad primes hit: {sorted(best_primes & avoid_primes)}")
    print(f"  Good primes hit: {sorted(best_primes & prefer_primes)}")

    return best


def apply_to_poly(fx, T):
    """
    Given fx in QQ[x] and MobiusTransform T, return fx(T(x)) as a QQ[x] polynomial.
    Clear denominators by multiplying by denominator^deg.
    Strictly raise RuntimeError if any coefficient is not rational.
    """
    from sage.rings.rational import Rational
    from sage.rings.integer import Integer
    
    R = fx.parent()
    x = R.gen()
    a, b, c, d = T.a, T.b, T.c, T.d
    deg = fx.degree()
    
    # Build the transformation in a fraction field
    R2 = PolynomialRing(QQ, 'x')
    x2 = R2.gen()
    F2 = R2.fraction_field()
    
    fx2 = R2([QQ(coeff) for coeff in fx.list()])
    
    Tx_num = a * x2 + b
    Tx_den = c * x2 + d
    
    # Substitute: fx(T(x)) where T(x) = Tx_num/Tx_den
    try:
        result_frac = fx2(x2 = Tx_num / Tx_den)
    except Exception as e:
        raise RuntimeError(f"apply_to_poly: substitution failed: {e}") from e
    
    # Multiply by denominator^deg to clear
    # This ensures we get a polynomial with degree at most deg * deg
    cleared = result_frac * (Tx_den ** deg)
    
    # Try to convert to polynomial
    try:
        result_poly = R2(cleared)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"apply_to_poly: result not a polynomial after clearing: {e}") from e
    
    # Verify all coefficients are rational
    for coeff in result_poly.list():
        if not isinstance(coeff, (Rational, Integer, int)):
            raise RuntimeError(f"apply_to_poly: non-rational coefficient: {type(coeff)}")
    
    return result_poly



def apply_to_poly(fx, T):
    """
    Given fx in QQ[x] and MobiusTransform T, return fx(T(x)) * (Tx_den)^deg.
    Uses homogeneous substitution to avoid fraction field artifacts.
    This guarantees the result is a polynomial of degree <= deg.
    """
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    from sage.rings.rational_field import QQ
    
    deg = fx.degree()
    # Create target ring (ensure it's over QQ)
    R2 = PolynomialRing(QQ, 'x')
    x = R2.gen()
    
    a, b, c, d = T.a, T.b, T.c, T.d
    
    # Numerator and Denominator of T(x)
    num = a * x + b
    den = c * x + d
    
    # Compute sum( coeff_i * num^i * den^(deg-i) )
    # This corresponds to homogenizing f(x) -> F(X,Z) and evaluating F(num, den)
    result = R2(0)
    coeffs = fx.list()
    
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        # We need explicit casting to avoid potential coercion issues
        term = R2(coeff) * (num ** i) * (den ** (deg - i))
        result += term
        
    return result
