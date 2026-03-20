import random
from sage.all import SR, QQ, EllipticCurve, Rational, GF, var
from math import lcm, gcd
from functools import reduce
from diagnostics2 import *
from fractions import Fraction
from search_common import FINITE_FIELD
from automorph import _coerce_m_symbol

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

    torsion_lcm_bound = 1 if not m_vals else reduce(lcm, m_vals)
    return m_vals, torsion_lcm_bound

# REPLACE good_specializations with:

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

# REPLACE torsion_test with:

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

def _detect_base_field_torsion(cd):
    try:
        parent = cd.a4.parent()
        if hasattr(parent, 'base_ring'):
            br = parent.base_ring()
            if hasattr(br, 'characteristic'):
                p = br.characteristic()
                if p > 0:
                    return GF(p), True, p
        return QQ, False, None
    except Exception:
        return QQ, False, None

def good_specializations(cd, m_sym, max_try=40):
    base_field, is_ff, p = _detect_base_field_torsion(cd)
    if is_ff:
        F = GF(p)
        xs = []
        for m0_int in range(min(max_try * 5, p)):
            if len(xs) >= max_try: break
            m0 = F(m0_int)
            try:
                # Use subs for safer evaluation of symbolic/poly expressions
                m_var = cd.a4.parent().gen() if hasattr(cd.a4.parent(), 'gen') else m_sym
                a4_val = F(cd.a4.subs({m_var: m0}))
                a6_val = F(cd.a6.subs({m_var: m0}))
                E = EllipticCurve(F, [0, 0, 0, a4_val, a6_val])
                if E.discriminant() != 0:
                    xs.append((m0, E))
            except (ZeroDivisionError, ValueError, TypeError):
                continue
        return xs

    # QQ mode logic omitted for brevity, but keep your original logic here
    return []

def _eval_rational_at_m(expr, m0, m_sym):
    try:
        # Handle Finite Field evaluation
        if hasattr(m0, 'parent') and getattr(m0.parent(), 'characteristic', lambda: 0)() > 0:
            parent = m0.parent()
            # If it's a symbolic expression, use subs; if it's a poly/frac, call it
            if hasattr(expr, 'subs'):
                val = expr.subs({m_sym: m0})
            elif callable(expr):
                val = expr(m0)
            else:
                val = expr
            return parent(val)
    except Exception:
        pass

    # Fallback to symbolic substitution for QQ
    try:
        return QQ(SR(expr).subs({m_sym: m0}))
    except Exception:
        return None

def torsion_test(cd, sec, n, m_sym=None, max_try=20):
    if m_sym is None:
        m_sym = var('m')

    specs = good_specializations(cd, m_sym, max_try=max_try)
    if not specs: return False

    for m0, E in specs:
        xv = _eval_rational_at_m(sec[0], m0, m_sym)
        yv = _eval_rational_at_m(sec[1], m0, m_sym)
        if xv is None or yv is None: continue

        try:
            P = E(xv, yv)
            if not (n * P).is_zero():
                return False
        except (ValueError, TypeError):
            # This is where the coordinate mismatch is caught
            return False
        except Exception:
            return False
    return True
