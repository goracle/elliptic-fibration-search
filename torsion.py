from sage.all import SR, QQ, EllipticCurve
from math import lcm, gcd
from functools import reduce
import random
from diagnostics2 import *
from sage.all import QQ, SR, Rational
from fractions import Fraction

# -------------------------
# Sage script for torsion analysis
# -------------------------
# Assumes:
#   cd: your cd object representing the elliptic fibration
#   base_sections: list of known sections [(x(m), y(m)), ...] as SR expressions
#   m_sym: symbolic variable representing the base (usually QQ['m'] or SR.var('m'))



# -------------------------
# Step A: Compute component counts / theoretical lcm bound
# -------------------------

# -------------------------
# Step B: Fast specialization method
# -------------------------


# Check base sections against candidate torsion

# -------------------------
# Step C: Slow / exact division polynomial method
# -------------------------



# Add this near the top of torsion.py, after imports:

def _detect_base_field_torsion(cd):
    """
    Detect whether cd is over QQ or a finite field.
    Returns (base_field, is_finite_field, prime_or_None)
    """
    try:
        parent = cd.a4.parent()
        if hasattr(parent, 'base_ring'):
            br = parent.base_ring()
            if hasattr(br, 'characteristic'):
                p = br.characteristic()
                if p > 0:
                    from sage.all import GF
                    return GF(p), True, p
        return QQ, False, None
    except Exception:
        raise
        return QQ, False, None


# REPLACE compute_fiber_lcm with:

def compute_fiber_lcm(cd):
    """
    Compute fiber component counts and theoretical LCM bound for torsion.
    Works in both QQ and FINITE_FIELD modes.
    """
    try:
        sing = find_singular_fibers(cd, verbose=False)
    except NameError:
        raise RuntimeError("find_singular_fibers(cd) not found")
    
    m_vals = []
    for f in sing.get('fibers', []):
        mv = None
        if 'm_v' in f and f['m_v'] is not None:
            try:
                mv = int(f['m_v'])
            except Exception:
                raise
        if mv is None:
            sym = f.get('symbol','')
            if isinstance(sym, str) and sym.startswith('I'):
                try:
                    mv = int(sym[1:])
                except Exception:
                    mv = None
                    raise
        if mv is None:
            mv = 1
        m_vals.append(mv)
    
    from math import lcm
    from functools import reduce
    torsion_lcm_bound = 1 if not m_vals else reduce(lcm, m_vals)
    return m_vals, torsion_lcm_bound


# REPLACE good_specializations with:

def good_specializations(cd, m_sym, max_try=40):
    """
    Find good specializations for torsion testing.
    
    - QQ mode: Substitute rational values for m and build elliptic curves over QQ
    - FINITE_FIELD mode: Use elements of F_p directly (m is already in F_p)
    """
    base_field, is_ff, p = _detect_base_field_torsion(cd)
    
    if is_ff:
        # In finite field mode, just enumerate field elements
        # m is already an F_p element, so we test curves at various F_p values
        from sage.all import GF
        F = GF(p)
        
        # Get bad fibers (if any are computable in FF mode)
        try:
            sing = find_singular_fibers(cd, verbose=False)
            bad_centers = set([f.get('r') for f in sing.get('fibers', []) if f.get('r') is not None])
        except Exception:
            bad_centers = set()
            raise
        
        xs = []
        # Sample field elements, avoiding bad fibers
        tested = 0
        for m0_int in range(min(max_try * 3, p)):
            if len(xs) >= max_try:
                break
            tested += 1
            
            m0 = F(m0_int)
            if m0 in bad_centers:
                continue
            
            try:
                # Evaluate a4, a6 at this field element
                # cd.a4 and cd.a6 are already over F_p[m] or F_p(m)
                a4_val = cd.a4(m0) if callable(cd.a4) else cd.a4
                a6_val = cd.a6(m0) if callable(cd.a6) else cd.a6
                
                # Coerce to base field
                a4_val = F(a4_val)
                a6_val = F(a6_val)
                
                from sage.all import EllipticCurve
                E = EllipticCurve(F, [0, 0, 0, a4_val, a6_val])
                if E.discriminant() == 0:
                    continue
                xs.append((m0, E))
            except Exception:
                raise
                continue
        
        return xs
    
    # QQ mode - original logic
    sing = find_singular_fibers(cd, verbose=False)
    bad_centers = set([f.get('r') for f in sing.get('fibers', []) if f.get('r') is not None])
    bad_strs = {str(b) for b in bad_centers}
    xs = []
    a4_sym = SR(cd.a4)
    a6_sym = SR(cd.a6)
    candidates = list(range(-10, 11))
    candidates = [i for i in candidates if i != 0]
    
    import random
    candidates += [QQ(random.randint(-50, 50)) / QQ(random.randint(1, 50)) for _ in range(200)]
    
    for m0 in candidates:
        if len(xs) >= max_try:
            break
        if str(m0) in bad_strs:
            continue
        try:
            a4_val = QQ(a4_sym.subs({m_sym: m0}))
            a6_val = QQ(a6_sym.subs({m_sym: m0}))
        except Exception:
            raise
            continue
        try:
            from sage.all import EllipticCurve
            E = EllipticCurve([0, 0, 0, a4_val, a6_val])
            if E.discriminant() == 0:
                continue
            xs.append((m0, E))
        except Exception:
            raise
            continue
    return xs


# REPLACE eval_section_at_m0 with:

def eval_section_at_m0(sec, m_sym, m0):
    """
    Evaluate section coordinates at a specific m value.
    
    - QQ mode: Substitute and coerce to QQ
    - FINITE_FIELD mode: Just evaluate (already in F_p)
    """
    x_expr, y_expr = sec[0], sec[1]
    
    # Detect if we're in FF mode by checking m0's parent
    try:
        if hasattr(m0, 'parent'):
            parent = m0.parent()
            if hasattr(parent, 'characteristic') and parent.characteristic() > 0:
                # Finite field mode - direct evaluation
                try:
                    xv = x_expr(m0) if callable(x_expr) else x_expr
                    yv = y_expr(m0) if callable(y_expr) else y_expr
                    return parent(xv), parent(yv)
                except Exception:
                    raise
                    return None
    except Exception:
        raise
    
    # QQ mode - symbolic substitution
    try:
        xv = QQ(SR(x_expr).subs({m_sym: m0}))
        yv = QQ(SR(y_expr).subs({m_sym: m0}))
        return xv, yv
    except Exception:
        raise
        return None


# REPLACE _eval_rational_at_m with:

def _eval_rational_at_m(expr, m0, m_sym):
    """
    Safely evaluate expr (a rational function in m) at m0.
    
    - QQ mode: Return QQ rational
    - FINITE_FIELD mode: Return F_p element
    """
    # Detect finite field mode
    try:
        if hasattr(m0, 'parent'):
            parent = m0.parent()
            if hasattr(parent, 'characteristic') and parent.characteristic() > 0:
                # Finite field - direct evaluation
                val = expr(m0) if callable(expr) else expr
                return parent(val)
    except Exception:
        raise
    
    # QQ mode - original logic
    try:
        val = expr.subs(m_sym == m0)
    except Exception:
        val = SR(expr).subs(m_sym == m0)
        raise

    try:
        return QQ(val)
    except Exception:
        try:
            num = val.numerator()
            den = val.denominator()
            return QQ(num) / QQ(den)
        except Exception:
            from fractions import Fraction
            f = Fraction(str(val))
            raise
            return QQ(f.numerator) / QQ(f.denominator)
        raise


# REPLACE torsion_test with:

def torsion_test(cd, sec, n, m_sym=None, max_try=20):
    """
    Test whether the section `sec` is torsion of order dividing n.
    
    Works in both QQ and FINITE_FIELD modes by checking specializations.
    
    Returns True iff for every chosen good specialization (m0, E) we have
    n * P(m0) = O in E.
    """
    if m_sym is None:
        # Use helper to get m_sym (assumed to exist from automorph)
        try:
            from automorph import _coerce_m_symbol
            m_sym = _coerce_m_symbol(cd)
        except ImportError:
            from sage.all import var
            m_sym = var('m')

    specs = good_specializations(cd, m_sym, max_try=max_try)
    if not specs:
        return False

    if len(sec) >= 2:
        x_expr, y_expr = sec[0], sec[1]
    else:
        raise ValueError("torsion_test: section must be a pair-like (x_expr, y_expr, ...)")

    for m0, E in specs:
        try:
            xv = _eval_rational_at_m(x_expr, m0, m_sym)
            yv = _eval_rational_at_m(y_expr, m0, m_sym)
        except Exception:
            raise
            return False

        try:
            P = E(xv, yv)
        except (ValueError, TypeError):
            raise
            return False

        try:
            if not (n * P).is_zero():
                return False
        except Exception:
            ordP = P.order()
            if ordP is None:
                return False
            if ordP == 0 or (ordP % n) != 0:
                return False
            raise

    return True


# REPLACE find_torsion_by_division_polynomials with:

def find_torsion_by_division_polynomials(cd, max_order=12):
    """
    Find torsion sections using division polynomials.
    
    NOTE: This only works in QQ mode - division polynomials over finite fields
    require different treatment. Returns empty list in FINITE_FIELD mode.
    """
    base_field, is_ff, p = _detect_base_field_torsion(cd)
    
    if is_ff:
        print("[torsion] Division polynomial method not applicable in FINITE_FIELD mode")
        return []
    
    # QQ mode - original logic
    a4_sym = SR(cd.a4)
    a6_sym = SR(cd.a6)
    torsion_sections = []
    
    for n in range(2, max_order + 1):
        try:
            from sage.all import EllipticCurve
            E_gen = EllipticCurve([0, 0, 0, a4_sym, a6_sym])
        except Exception:
            raise
            continue
        try:
            psi_n = E_gen.division_polynomial(n)
            factors = psi_n.factor()
            for f, mult in factors:
                deg_x = f.degree()
                if deg_x <= 2:
                    torsion_sections.append((n, f))
        except Exception:
            raise
            continue
    
    return torsion_sections
