# -------------------------
# Sage script for torsion analysis
# -------------------------
# Assumes:
#   cd: your cd object representing the elliptic fibration
#   base_sections: list of known sections [(x(m), y(m)), ...] as SR expressions
#   m_sym: symbolic variable representing the base (usually QQ['m'] or SR.var('m'))

from sage.all import SR, QQ, EllipticCurve
from math import lcm, gcd
from functools import reduce
import random

from diagnostics2 import *

# -------------------------
# Step A: Compute component counts / theoretical lcm bound
# -------------------------
def compute_fiber_lcm(cd):
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
                pass
        if mv is None:
            sym = f.get('symbol','')
            if isinstance(sym, str) and sym.startswith('I'):
                try:
                    mv = int(sym[1:])
                except Exception:
                    mv = None
        if mv is None:
            mv = 1
        m_vals.append(mv)
    torsion_lcm_bound = 1 if not m_vals else lcm(*m_vals)
    return m_vals, torsion_lcm_bound

# -------------------------
# Step B: Fast specialization method
# -------------------------
def good_specializations(cd, m_sym, max_try=40):
    sing = find_singular_fibers(cd, verbose=False)
    bad_centers = set([f.get('r') for f in sing.get('fibers', []) if f.get('r') is not None])
    bad_strs = {str(b) for b in bad_centers}
    xs = []
    a4_sym = SR(cd.a4)
    a6_sym = SR(cd.a6)
    candidates = list(range(-10,11))
    candidates = [i for i in candidates if i != 0]  # avoid obvious bad points
    # add some random rationals
    candidates += [QQ(random.randint(-50,50))/QQ(random.randint(1,50)) for _ in range(200)]
    for m0 in candidates:
        if len(xs) >= max_try:
            break
        if str(m0) in bad_strs:
            continue
        try:
            a4_val = QQ(a4_sym.subs({m_sym: m0}))
            a6_val = QQ(a6_sym.subs({m_sym: m0}))
        except Exception:
            continue
        try:
            E = EllipticCurve([0,0,0,a4_val,a6_val])
            if E.discriminant() == 0:
                continue
            xs.append((m0,E))
        except Exception:
            continue
    return xs


# Check base sections against candidate torsion
def eval_section_at_m0(sec, m_sym, m0):
    x_expr, y_expr = sec[0], sec[1]
    try:
        xv = QQ(SR(x_expr).subs({m_sym: m0}))
        yv = QQ(SR(y_expr).subs({m_sym: m0}))
        return xv, yv
    except Exception:
        return None

# -------------------------
# Step C: Slow / exact division polynomial method
# -------------------------
def find_torsion_by_division_polynomials(cd, max_order=12):
    a4_sym = SR(cd.a4)
    a6_sym = SR(cd.a6)
    torsion_sections = []
    for n in range(2, max_order+1):
        try:
            E_gen = EllipticCurve([0,0,0,a4_sym,a6_sym])
        except Exception:
            continue
        try:
            psi_n = E_gen.division_polynomial(n)
            # attempt to factor psi_n over SR/QQ(m)
            factors = psi_n.factor()
            for f, mult in factors:
                # any linear or low-degree factor in x(m) is a candidate torsion section
                deg_x = f.degree()
                if deg_x <= 2:  # you can adjust this cutoff
                    torsion_sections.append((n,f))
        except Exception:
            continue
    return torsion_sections

from sage.all import QQ, SR, Rational
from fractions import Fraction

def _eval_rational_at_m(expr, m0, m_sym):
    """
    Safely evaluate expr (a rational function in m) at rational m0.
    Return a QQ rational.
    """
    # try direct substitution on algebraic/FR elements first
    try:
        val = expr.subs(m_sym == m0)
    except Exception:
        # fallback to SR substitution (less desirable but sometimes necessary)
        val = SR(expr).subs(m_sym == m0)

    # If it's already a Sage rational number or Python rational-like, coerce to QQ
    try:
        return QQ(val)
    except Exception:
        # try to coerce via numerator/denominator if available
        try:
            num = val.numerator()
            den = val.denominator()
            return QQ(num) / QQ(den)
        except Exception:
            # final fallback: stringify and use Python Fraction then QQ
            f = Fraction(str(val))
            return QQ(f.numerator) / QQ(f.denominator)


def torsion_test(cd, sec, n, m_sym=None, max_try=20):
    """
    Test whether the section `sec` (given as (x_expr, y_expr, 1) or similar)
    is torsion of order dividing n by checking several good specializations.

    Returns True iff for every chosen good specialization (m0, E) we have
    n * P(m0) = O in E(Q).
    """
    if m_sym is None:
        m_sym = _coerce_m_symbol(cd)  # use your existing helper

    # get specializations (m0, E) where curve is defined and not singular
    specs = good_specializations(cd, m_sym, max_try=max_try)
    if not specs:
        return False   # no usable specializations -> cannot certify torsion

    # canonical format: sec is something like (x_expr, y_expr, 1) or (x_expr, y_expr)
    if len(sec) >= 2:
        x_expr, y_expr = sec[0], sec[1]
    else:
        raise ValueError("torsion_test: section must be a pair-like (x_expr,y_expr,...)")

    for m0, E in specs:
        # Evaluate the section coordinates at m0
        try:
            xv = _eval_rational_at_m(x_expr, m0, m_sym)
            yv = _eval_rational_at_m(y_expr, m0, m_sym)
        except Exception:
            # if evaluation fails, this specialization is unusable — treat as failure
            return False

        # try to build point on specialized curve
        try:
            P = E(xv, yv)
        except (ValueError, TypeError):
            # not on curve (or bad input) -> not torsion
            return False

        # check n*P == O
        try:
            if not (n * P).is_zero():
                return False
        except Exception:
            # fallback: check order if available
            ordP = P.order()
            if ordP is None:
                return False
            if ordP == 0 or (ordP % n) != 0:
                return False

    # all tested specializations passed -> likely torsion of order dividing n
    return True
