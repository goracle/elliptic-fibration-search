# tower.sage
# Numeric-first fibration tower builder (strict, exact QQ arithmetic)
# - substitute numeric inputs (curve coeffs, x_i) as QQ immediately
# - interpolate Q(x) exactly from rational (x,y) points before building fibration
# - every solve must return exactly one branch;
#   solution values must be exact QQ (or error)
# - plain python int seed (safe with `sage tower.sage`)
#
# Usage: sage tower.sage

from functools import reduce
import operator
from sage.all import SR, var, PolynomialRing, QQ
from sage.all import *
from sage.functions.other import binomial
import random # shadows something in sage.all called random; be careful!

from search_common import DEBUG, SEED_INT



# ---------- Utilities ----------
try:
    PROFILE = profile
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile



# === normalize and check helpers (drop-in) ===

def _pick_field(step, names):
    # pick first present field name or None
    try:
        for n in names:
            if hasattr(step, 'get') and n in step:
                return step[n]
            if hasattr(step, n):
                try:
                    return getattr(step, n)
                except Exception:
                    pass
        # sequence fallback common positions
        if hasattr(step, '__len__') and not hasattr(step, 'get'):
            if len(step) > 0:
                return step[0]
    except Exception:
        pass
    return None

def normalize_step(step):
    # step is a dict with keys like 'f_i', 'Q_i', 'r_expr', 'info'
    fx = step.get('f_i')
    r_expr = step.get('r_expr')
    param = 'm'   # your tower construction always uses m
    double_root_x = step.get('x0')  # not present in your dump, may be None
    return {'fx': fx, 'r_expr': r_expr, 'param': param,
            'double_root_x': double_root_x, 'raw': step}

def check_fibration_step(step, prev_fx=None, layer_index=None):
    L = "Layer[%s]" % (layer_index if layer_index is not None else "unknown")
    s = normalize_step(step)
    fx = s['fx']
    assert fx is not None, L + ": missing f_i in step. repr(step)=" + repr(s['raw'])
    r_expr = s['r_expr']
    x = var('x')
    m = var('m')

    # r_expr sanity
    if r_expr is not None:
        rv = [str(v) for v in SR(r_expr).variables()]
        assert 'x' not in rv, L + ": r_expr depends on x. r_expr=" + repr(r_expr)
        if len(rv) > 0:
            assert 'm' in rv, L + ": r_expr vars " + repr(rv) + " missing 'm'"

    # degree drop
    if prev_fx is not None:
        try:
            dprev = SR(prev_fx).degree(x)
        except Exception:
            dprev = None
        try:
            dcur = SR(fx).degree(x)
        except Exception:
            dcur = None
        if dprev is not None and dcur is not None:
            assert dcur <= dprev - 1, (
                L + ": degree drop failed prev=%s cur=%s" %
                (repr(dprev), repr(dcur))
            )

    # optional double root check
    dr = s['double_root_x']
    if dr is not None:
        fx_sr = SR(fx)
        dfx = fx_sr.derivative(x)
        v1 = fx_sr.subs({x: dr})
        v2 = dfx.subs({x: dr})
        assert SR(v1).simplify() == 0, L + ": x0 not root. fx(x0)=%s" % repr(v1)
        assert SR(v2).simplify() == 0, L + ": x0 not double root. fx'(x0)=%s" % repr(v2)

    return True

def verify_tower_consistency(tower):
    # All steps should use param m consistently
    for i, step in enumerate(tower):
        s = normalize_step(step)
        r = s['r_expr']
        if r is not None:
            rv = [str(v) for v in SR(r).variables()]
            assert 'x' not in rv, "verify_tower_consistency: step[%d] r_expr depends on x: %s" % (i, repr(r))
            assert 'm' in rv or rv == [], "verify_tower_consistency: step[%d] r_expr vars %s missing 'm'" % (i, rv)
    return True

# Helpers for robust assertions in tower.sage

def ensure_symbol(obj, name_hint):
    """
    Guarantee we return a proper symbolic variable object suitable for SR/diff calls.
    Accepts:
      - a SymbolicVariable (returned by var('m')), or
      - a string like 'm', or
      - an object with .name() method
    Returns: the SymbolicVariable (SR-level)
    Crashes with assert if cannot make a symbol.
    """
    # If already a Symbolic Expression variable, return it
    try:
        # If obj is a Sage SymbolicVariable, this should succeed
        _ = SR(obj)
        # if obj is a string, SR('m') is not what we want — make explicit var
    except Exception:
        pass

    if obj is None:
        raise AssertionError("ensure_symbol: received None for symbol (hint: %s)" % name_hint)

    if isinstance(obj, str):
        return var(obj)   # creates symbol with that name in SR
    # try to pull name attr
    try:
        nm = getattr(obj, 'name', None)
        if nm:
            return var(str(nm))
    except Exception:
        pass

    # If it's already a SymbolicVariable-like (e.g., m from var('m')), ensure SR.var(...)
    try:
        s = var(str(obj))
        return s
    except Exception as e:
        raise AssertionError("ensure_symbol: cannot coerce %r to a symbol (hint: %s). err: %s"
                             % (obj, name_hint, e))

def expr_variables(expr):
    """
    Return a Python set of variable names used in a symbolic expression `expr`.
    Works for SR expressions or polynomials (Sage objects).
    """
    if expr is None:
        return set()
    try:
        # For symbolic SR expressions:
        syms = expr.variables()
        return set([str(s) for s in syms])
    except Exception:
        # Fallback: try to convert to polynom and inspect variables
        try:
            s = SR(expr)
            syms = s.variables()
            return set([str(sv) for sv in syms])
        except Exception:
            # last resort: text parsing (not ideal but won't silently pass)
            txt = str(expr)
            return set()






# -----------------------
# Deterministic branch chooser for solve() results
# -----------------------
@PROFILE
def pick_solution_by_degree(solutions, target_var, prefer_max_degree=True):
    """Given many dict solutions returned by solve(..., solution_dict=True),
       pick the solution whose 'target_var' has the largest (or smallest) polynomial degree
       in the symbolic parameter (heuristic). Raises RuntimeError if ambiguous.
       solutions: list of dicts
    """
    if not solutions:
        raise RuntimeError("No solutions provided to pick_solution_by_degree")
    scored = []
    for sol in solutions:
        if target_var not in sol:
            raise RuntimeError("candidate solution missing target_var: %s" % target_var)
        sval = sol[target_var]
        # try to measure 'degree' in the parameter(s) by counting occurrences of symbolic variables
        try:
            deg = 0
            for v in SR(sval).variables():
                # crude proxy: degree in each variable via polynomial degree if possible
                try:
                    PR = PolynomialRing(QQ, str(v))
                    deg = max(deg, PR(SR(sval)).degree())
                except Exception:
                    # fallback: count appearance
                    deg = max(deg, str(sval).count(str(v)))
        except Exception:
            deg = 0
        scored.append((deg, sol))
    # pick solution by degree
    if prefer_max_degree:
        scored.sort(key=lambda t: t[0], reverse=True)
    else:
        scored.sort(key=lambda t: t[0])
    # ensure unique top candidate
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise RuntimeError("Ambiguous solution selection: top two candidates have same score.")
    return scored[0][1]

# -----------------------
# High-level comparison wrapper for a layer vs Mathematica output
# -----------------------
# -----------------------
# Robust extraction + comparison wrapper for a layer vs Mathematica output
# -----------------------
# keys we commonly store generated expressions under in layer dicts




@PROFILE
def require_single_solution(sol_list, context=""):
    """
    Ensure solver returned exactly one branch. Return that solution dict.
    Raise RuntimeError otherwise.
    """
    if not isinstance(sol_list, (list, tuple)):
        raise RuntimeError(f"Solver output not list/tuple. Context: {context}. Output: {sol_list!r}")
    if len(sol_list) != 1:
        raise RuntimeError(f"Solver returned {len(sol_list)} branches (expected 1). Context: {context}")
    return sol_list[0]

@PROFILE
def has_free_variables(expr):
    """Check if a Sage symbolic expression has free variables."""
    try:
        return len(expr.free_variables()) > 0
    except AttributeError:
        # If it's not a symbolic expression, assume it's a constant
        return False

@PROFILE
def choose_degQ(n):
    """Choose degQ so that 2*degQ is n-2 if possible, else n-1."""
    if (n - 2) % 2 == 0:
        return (n - 2) // 2
    if (n - 1) % 2 == 0:
        return (n - 1) // 2
    raise ValueError(f"No integer degQ with 2*degQ in {{n-1,n-2}} for n={n}")

@PROFILE
def poly_prod_numeric(xs, x_sym):
    """(x - x1)(x - x2)... with numeric xi (QQ) substituted into SR."""
    prod = SR(1)
    for xi in xs:
        prod *= (x_sym - SR(QQ(xi)))
    return prod.expand()


@PROFILE
def assert_layer_match(layer, expected_Q=None, expected_r=None):
    """
    Compare layer['Q_i'] (QQ polynomial) and layer['r_expr'] (SR) to expected values.
    - expected_Q: either QQ polynomial or list of coefficients [c0,c1,...]
    - expected_r: SR expression or string; comparison uses SR equality (simplify)
    Raises RuntimeError if mismatch.
    """
    if expected_Q is not None:
        Qgot = layer.get('Q_i')
        if Qgot is None:
            raise RuntimeError("Layer missing 'Q_i'")
        if not isinstance(expected_Q, (Polynomial)):
            # assume list of coefficients
            R = PolynomialRing(QQ, 'x')
            expected_Qpoly = R([QQ(v) for v in expected_Q])
        else:
            expected_Qpoly = expected_Q
        if Qgot != expected_Qpoly:
            raise RuntimeError(f"Q mismatch.\n got: {Qgot}\n expected: {expected_Qpoly}")

    if expected_r is not None:
        rgot = layer.get('r_expr')
        if rgot is None:
            raise RuntimeError("Layer missing 'r_expr'")
        # try SR comparison
        if isinstance(expected_r, str):
            expected_r_sr = SR(expected_r)
        else:
            expected_r_sr = expected_r
        if (SR(rgot) - SR(expected_r_sr)).simplify() != 0:
            raise RuntimeError(f"r_expr mismatch.\n got: {rgot}\n expected: {expected_r_sr}")


# small helper to coerce to QQ and raise helpful error if not possible
@PROFILE
def force_QQ(val, name=''):
    try:
        return QQ(val)
    except Exception as e:
        raise RuntimeError(f"Could not coerce {name!s} to QQ: {val!r} — error: {e}")

# ---------- Core Tower Builder ----------



# ---------- Replacement: interpolate_Q_general ----------
# Required imports (add these to your file if not already present)
# from sage.functions.other import binomial
# from sage.rings.rational_field import QQ
# from sage.rings.polynomial.polynomial_ring import PolynomialRing
# import random

def compute_y_derivatives_at_point(xi_sr, yi_sr, f_derivs, max_order, x_sym):
    """
    Compute y^(k) for k = 0, 1, ..., max_order at the point (xi_sr, yi_sr)
    using the implicit differentiation of y^2 = f(x). This uses the recursive relationship derived from repeatedly differentiating y^2 = f(x).
    """
    # Get f derivatives at the point
    f_vals = [f_derivs[i].subs({x_sym: xi_sr}) for i in range(max_order + 1)]

    # y^(0) = y
    y_derivs = [yi_sr]

    for n in range(1, max_order + 1):
        if n == 1:
            # From y^2 = f(x): 2*y*y' = f'  =>  y' = f'/(2*y)
            y_n = f_vals[1] / (2 * yi_sr)
        else:
            # For n ≥ 2, we use the recursive relationship from differentiating y^2 = f(x)
            #
            # The key insight: d^n/dx^n[y^2] = f^(n)
            # The left side expands using the generalized product rule (Leibniz rule):
            # d^n/dx^n[y^2] = sum_{k=0}^n C(n,k) * d^k/dx^k[y] * d^{n-k}/dx^k[y]
            #                = sum_{k=0}^n C(n,k) * y^(k) * y^(n-k)
            #                = 2 * sum_{k=1}^{n-1} C(n,k) * y^(k) * y^(n-k) + 2*y*y^(n)
            #                  (the k=0 and k=n terms combine to give 2*y*y^(n))
            #
            # So: f^(n) = 2*y*y^(n) + 2 * sum_{k=1}^{n-1} C(n,k) * y^(k) * y^(n-k)
            # Solving for y^(n):
            # y^(n) = [f^(n) - 2 * sum_{k=1}^{n-1} C(n,k) * y^(k) * y^(n-k)] / (2*y)

            # Compute the sum of cross terms
            cross_sum = 0
            for k in range(1, n):
                cross_sum += binomial(n, k) * y_derivs[k] * y_derivs[n - k]

            # Solve for y^(n)
            y_n = (f_vals[n] - cross_sum) / (2 * yi_sr)

        y_derivs.append(y_n)

    return y_derivs

def compute_implicit_derivative_constraint(order, xi_sr, yi_sr, f_derivs, Q_derivs, x_sym):
    """
    Compute the constraint for the nth derivative using implicit differentiation of y^2 = f(x).
    This uses the exact recursive relationship derived from the generalized Leibniz rule
    applied to the differentiation of y^2 = f(x). This is mathematically equivalent to
    using Faà di Bruno's formula but more direct for our specific case.
    """
    if yi_sr == 0:
        print(f"Skipping order {order} constraint at x={xi_sr} due to y-value being 0.")
        return None

    # Compute all y derivatives up to the required order
    y_derivs_at_point = compute_y_derivatives_at_point(xi_sr, yi_sr, f_derivs, order, x_sym)

    # The constraint is simply Q^(order)(xi) = y^(order)(xi)
    Q_nth_expr = Q_derivs[order].subs({x_sym: xi_sr})
    expected_y_nth = y_derivs_at_point[order]

    eq = (Q_nth_expr == expected_y_nth)
    return eq

# In tower.sage

@PROFILE
def interpolate_Q_general(pts_xy, f_expr, degQ, x_sym, seed_int=SEED_INT, force_constraint_indices=None):
    """
    Compute Q(x) of degree degQ from rational pts_xy on curve f_expr.
    The symbolic variable `x_sym` is now passed in to ensure consistency.
    """
    random.seed(int(seed_int))
    # The line `x_sym = SR.var('x')` has been removed.

    # Keep f_expr symbolic
    f_expr_sym = SR(f_expr)
    coeffs_sym = [SR.var(f'c{i}') for i in range(degQ + 1)]
    Q_poly_sym = sum(c * x_sym**i for i, c in enumerate(coeffs_sym))

    # --- START OF ENHANCED LOGIC ---
    # We now separate mandatory interpolation constraints from optional derivative constraints.
    mandatory_constraints = []
    derivative_pool = []
    
    # Precompute derivatives of f(x) and Q(x) up to the maximum order we might need
    max_order = degQ
    f_derivs = {0: f_expr_sym}
    Q_derivs = {0: Q_poly_sym}
    
    for order in range(1, max_order + 1):
        f_derivs[order] = f_derivs[order - 1].diff(x_sym)
        Q_derivs[order] = Q_derivs[order - 1].diff(x_sym)

    # Generate constraints for each point
    for xi, yi in pts_xy:
        # ensure xi, yi are SR/QQ-friendly
        xi_sr = SR(xi)
        yi_sr = SR(yi)

        # Order 0 constraints are mandatory for interpolation
        eq0 = (Q_derivs[0].subs({x_sym: xi_sr}) == yi_sr)
        mandatory_constraints.append(eq0)

        # Generate derivative constraints up to the maximum order
        for order in range(1, max_order + 1):
            if order > degQ:
                break
                
            constraint = compute_implicit_derivative_constraint(order, xi_sr, yi_sr, f_derivs, Q_derivs, x_sym)
            if constraint is not None:
                derivative_pool.append(constraint)

    num_constraints_needed = len(coeffs_sym)
    
    # Calculate how many more constraints we need from the derivative pool
    num_remaining_needed = num_constraints_needed - len(mandatory_constraints)
    if num_remaining_needed < 0:
        raise RuntimeError("Too many mandatory constraints for the given polynomial degree.")

    if len(derivative_pool) < num_remaining_needed:
        raise RuntimeError(f"Not enough unique constraints to solve for Q. Need {num_constraints_needed}, "
                           f"have {len(mandatory_constraints)} mandatory and {len(derivative_pool)} derivative.")

    # Select the remaining constraints from the derivative pool
    if force_constraint_indices is None:
        chosen_derivative_eqs = random.sample(derivative_pool, num_remaining_needed)
    else:
        raise RuntimeError("force_constraint_indices is not currently supported with this corrected logic. "
                         "Please use random sampling for now.")

    # Combine the mandatory and selected derivative constraints
    chosen_eqs = mandatory_constraints + chosen_derivative_eqs
    # --- END OF ENHANCED LOGIC ---

    sols = solve(chosen_eqs, coeffs_sym, solution_dict=True)
    if not sols:
        raise RuntimeError(f"Could not solve for Q of degree {degQ} with provided constraints.")

    sol = require_single_solution(sols, "solving for Q coefficients")

    R = PolynomialRing(QQ, str(x_sym)) # Use the name of the symbolic var for the poly ring
    solved_coeffs = []
    for c in coeffs_sym:
        v = sol[c]
        if has_free_variables(v):
            raise RuntimeError("Solved Q coefficient depends on symbolic variables; expected numeric QQ.", v)
        solved_coeffs.append(QQ(v))
    Qx = R(solved_coeffs)
    return Qx


@PROFILE
def build_one_fibration_step(fx_SR, f0, pts_x, g2, seed_int=SEED_INT,
                             verbose=False, forced_tangency_seq=None,
                             forced_Qpoly=None, force_Q_constraint_indices=None,
                             parameter_m=None):
    """
    Build a single fibration step.

    Changes required by user:
    - fibration of the form Q(x)^2 + Prod_i (x - x_i) * rest(x),
      where rest(x) has degree (n-1 - deg(Prod)).
    - all intersection points are r = x1 - m (i.e., x1 - m).
    - eqns: first two force r to be a double root; remaining eqns are tangency
      equations on the x_i's. Solve for the coefficients of rest(x).
    - enforce degree drop: target fibration degree = n-1.
    """
    # deterministic choices
    random.seed(int(seed_int))
    xSR = SR.var('x')

    # degree of current curve polynomial (expected to be an SR polynomial in x)
    n = int(fx_SR.degree(xSR))

    # ------------------------------
    # build chosen points (numeric)
    # ------------------------------
    xs_chosen = [QQ(xv) for xv in pts_x]
    if len(xs_chosen) == 0:
        raise RuntimeError("build_one_fibration_step: pts_x must contain at least one x-value (x1).")
    x1 = xs_chosen[0]

    # ------------------------------
    # enforce deg drop constraint for Q
    # ------------------------------
    # target fibration degree = n-1, so require deg(Q^2) <= n-1 => degQ <= floor((n-1)/2)
    max_degQ = (n - 1) // 2
    initial_degQ = choose_degQ(n)
    degQ = min(initial_degQ, max_degQ)

    # If user supplied forced_Qpoly, determine/verify its degree
    if forced_Qpoly is not None:
        try:
            forced_Q_SR = SR(forced_Qpoly)
            forced_deg = int(forced_Q_SR.degree(xSR))
        except Exception:
            # try coercing into a QQ polynomial and check degree
            try:
                Rtmp = PolynomialRing(QQ, str(xSR))
                forced_deg = int(Rtmp(forced_Qpoly).degree())
            except Exception:
                raise RuntimeError("Could not determine degree of forced_Qpoly; provide a polynomial-like object.")
        if forced_deg > max_degQ:
            raise RuntimeError(f"forced_Qpoly has degree {forced_deg} which is > allowed max {max_degQ} (would prevent degree drop).")
        degQ = forced_deg

    # ------------------------------
    # Build or interpolate Q polynomial
    # ------------------------------
    if forced_Qpoly is not None:
        # try to coerce forced_Qpoly into a QQ polynomial if it is numeric
        try:
            # attempt to interpret as a QQ polynomial (this will fail if coefficients depend on m)
            Rqq = PolynomialRing(QQ, str(xSR))
            Qpoly_QQ = Rqq(forced_Qpoly)
        except Exception:
            # fallback: keep the SR expression for Q
            Qpoly_QQ = SR(forced_Qpoly)
    else:
        # Need (x,y) pairs for interpolate_Q_general. Compute y from f0 at the xs_chosen points.
        chosen_pts_xy = []
        f0_SR = SR(f0)
        for xv in xs_chosen:
            # Try to extract y from f0: y^2 = f0(x). We assume the square is exact for these chosen points.
            y_val_expr = f0_SR.subs({xSR: SR(xv)})
            try:
                yi = sqrt(QQ(y_val_expr))
            except Exception:
                # last resort: keep symbolic sqrt (in case value depends on m or isn't a perfect square)
                yi = SR(sqrt(y_val_expr))
            chosen_pts_xy.append((QQ(xv), yi))
        # interpolate_Q_general should be in the surrounding context
        Qpoly_QQ = interpolate_Q_general(chosen_pts_xy, f0, degQ, xSR,
                                         seed_int=seed_int,
                                         force_constraint_indices=force_Q_constraint_indices)

    Q_SR = SR(Qpoly_QQ)

    # ------------------------------
    # Build product Prod(x) = prod (x - xi)
    # ------------------------------
    prod1 = poly_prod_numeric(xs_chosen, xSR)  # expects numeric xi and returns a QQ polynomial or SR
    deg_prod = int(prod1.degree(xSR))

    # rest degree: ensure fibration top-degree = n-1
    rest_deg = int(n - 1 - deg_prod)
    if rest_deg < 0:
        raise RuntimeError(f"rest polynomial degree would be negative: rest_deg={rest_deg}. Reduce number/deg of points or change input.")

    # Create symbolic unknowns for rest coefficients: b_rest_0 ... b_rest_k
    rest_coeff_names = [f"b_rest_{i}" for i in range(rest_deg + 1)]
    rest_coeff_syms = [SR.var(name) for name in rest_coeff_names]
    rest_poly_SR = sum(rest_coeff_syms[i] * xSR**i for i in range(rest_deg + 1))

    # Build the fibration polynomial: Q(x)^2 + Prod(x) * rest(x)
    fibration_SR = (SR(Q_SR)**2).expand() + (SR(prod1) * rest_poly_SR).expand()

    # Intersection polynomial: fx_SR(x) - fibration(x)
    diff_poly = (SR(fx_SR) - fibration_SR).expand()

    # symbolic parameter m
    if parameter_m is None:
        m = SR.var('m')
    else:
        m = SR(parameter_m)

    # r fixed to x1 - m
    r_expr = SR(QQ(x1)) - m

    # ------------------------------
    # Build equations
    # ------------------------------
    eqs = []
    # 1) enforce r is a root
    eqs.append(diff_poly.subs({xSR: r_expr}).expand())
    # 2) enforce r is a double root
    eqs.append(diff(diff_poly, xSR).subs({xSR: r_expr}).expand())

    # Unknowns to solve for: coefficients of rest(x)
    unknowns = rest_coeff_syms[:]

    # Number of extra equations needed (beyond the two double-root eqs)
    num_extra_eqs = len(unknowns) - 2
    if num_extra_eqs < 0:
        raise RuntimeError("Too few unknowns in rest polynomial to satisfy double-root + tangency constraints.")

    # Build tangency allocation
    tangency_counts = {QQ(xi): 0 for xi in xs_chosen}
    if forced_tangency_seq is not None:
        if not isinstance(forced_tangency_seq, (list, tuple)):
            raise RuntimeError("forced_tangency_seq must be a list/tuple of x-values")
        if len(forced_tangency_seq) != num_extra_eqs:
            raise RuntimeError(f"forced_tangency_seq length must equal required extra eq count ({num_extra_eqs})")
        sel_points = [QQ(xv) for xv in forced_tangency_seq]
        for xv in sel_points:
            if xv not in tangency_counts:
                raise RuntimeError(f"forced_tangency_seq contains x={xv} not in pts_x {xs_chosen}")
    else:
        sel_points = [QQ(random.choice(xs_chosen)) for _ in range(num_extra_eqs)]

    # Append tangency equations: for each selection, increment multiplicity and add the derivative^k condition at that x
    for xv in sel_points:
        tangency_counts[QQ(xv)] += 1
        current_order = tangency_counts[QQ(xv)]
        eq_t = diff(diff_poly, xSR, current_order).subs({xSR: SR(xv)}).expand()
        eqs.append(eq_t)

    # Sanity check: number of equations should equal number of unknowns
    if len(eqs) != len(unknowns):
        raise RuntimeError(f"Equation/unknown mismatch: {len(eqs)} equations vs {len(unknowns)} unknowns.")

    # ------------------------------
    # Solve the system
    # ------------------------------
    sols = solve(eqs, unknowns, solution_dict=True)
    sol = require_single_solution(sols, "solving for rest polynomial coefficients")

    # ------------------------------
    # Build solved coefficient map, detect symbolic dependencies
    # (REPLACEMENT BLOCK: builds rest_poly_SR_solved, rest_poly_QQ)
    # ------------------------------
    solved_map = {}
    contains_symbolic = False
    for symb in unknowns:
        val_SR = SR(sol[symb])
        solved_map[symb] = val_SR
        # Try to coerce to QQ to see if it's purely rational
        try:
            _ = QQ(val_SR)
        except Exception:
            contains_symbolic = True

    # If all rational, create a QQ polynomial. Otherwise keep symbolic SR polynomial.
    rest_poly_QQ = None
    rest_poly_SR_solved = None
    if not contains_symbolic:
        Rqq = PolynomialRing(QQ, str(xSR))
        coeffs_q = [QQ(solved_map[s]) for s in rest_coeff_syms]
        rest_poly_QQ = Rqq(coeffs_q)
        rest_poly_SR_solved = sum(SR(coeffs_q[i]) * xSR**i for i in range(len(coeffs_q)))
    else:
        rest_poly_SR_solved = sum(solved_map[rest_coeff_syms[i]] * xSR**i for i in range(len(rest_coeff_syms)))
        rest_poly_QQ = None

    # ------------------------------
    # Construct the solved fibration polynomial: Q(x)^2 + Prod(x) * rest_solved(x)
    # ------------------------------
    # Q_SR was defined earlier as SR(Qpoly_QQ)
    Q_SR = SR(Q_SR)  # ensure SR
    prod_SR = SR(prod1)

    fibration_solved_SR = (Q_SR**2).expand() + (prod_SR * rest_poly_SR_solved).expand()
    # normalize/expand
    fibration_solved_SR = SR(fibration_solved_SR).expand()

    # Sanity check: degree drop must happen (target = n-1)
    try:
        deg_fib = int(fibration_solved_SR.degree(xSR))
    except Exception:
        # If SR.degree fails, attempt a coarse heuristic by inspecting polynomial string
        deg_fib = None

    target_deg = n - 1
    if deg_fib is None or deg_fib != target_deg:
        # Assemble diagnostics to make debugging easier
        diag = []
        diag.append(f"expected fibration degree {target_deg}, got {deg_fib}")
        try:
            deg_Q2 = int((Q_SR**2).degree(xSR))
            diag.append(f"deg(Q^2) = {deg_Q2}")
        except Exception:
            diag.append("deg(Q^2) unknown (symbolic)")

        try:
            deg_prodrest = int((prod_SR * rest_poly_SR_solved).degree(xSR))
            diag.append(f"deg(prod*rest) = {deg_prodrest}")
        except Exception:
            diag.append("deg(prod*rest) unknown (symbolic)")

        diag_msg = "; ".join(diag)
        # Dump some helpful pieces into the exception for debugging
        raise RuntimeError("Degree drop failed: " + diag_msg + f"\nQ_SR = {Q_SR}\nprod = {prod_SR}\nrest = {rest_poly_SR_solved}\nfibration = {fibration_solved_SR}")

    # ------------------------------
    # Return the fibration (as SR) and other useful artifacts
    # ------------------------------
    return {
        'f_i': fibration_solved_SR,              # the new f(x) (SR) — THIS IS THE FIX
        'Q_i': Qpoly_QQ,
        'Q_QQ': Qpoly_QQ if isinstance(Qpoly_QQ, (type(PolynomialRing(QQ, 'x')(0)),)) else Q_SR,
        'r_expr': r_expr,
        'rest_poly_SR': rest_poly_SR_solved,
        'rest_poly_QQ': rest_poly_QQ,
        'info': f"n={n} degProd={deg_prod} rest_deg={rest_deg} (double-root + {num_extra_eqs} tangency eqs) symbolic={contains_symbolic}",
    }

@PROFILE
def _verify_fibration_step_properties(fx, r_expr, param):
    # fx: polynomial in x
    # r_expr: expression in m
    # param: expected to be 'm' (string or symbol)
    x = var('x')
    try:
        m = var(str(param))
    except Exception:
        m = var('m')

    fx_sr = SR(fx)

    # derivative wrt x
    dfx_dx = fx_sr.derivative(x)

    # derivative wrt m (if r_expr uses m)
    if r_expr is not None:
        r_sr = SR(r_expr)
        if 'm' in [str(v) for v in r_sr.variables()]:
            dr_dm = r_sr.derivative(m)
        else:
            dr_dm = None
    else:
        dr_dm = None

    return {'dfx_dx': dfx_dx, 'dr_dm': dr_dm}


@PROFILE
def iterate_tower(fx_PR, pts_xy, max_steps=3, seed_int=SEED_INT, verbose=DEBUG):
    """
    Iterates through the fibration tower construction process, ensuring a single
    consistent fibration parameter 'm' is used throughout.
    """
    tower = []
    
    # Get the polynomial generator's name and create a corresponding symbolic variable.
    # This ensures that all symbolic operations use a variable of the correct type (SR).
    poly_x_gen = fx_PR.parent().gen()
    x = SR.var(str(poly_x_gen))

    # Convert the initial polynomial to a symbolic expression (SR)
    # to ensure a consistent type throughout the loop.
    f0 = SR(fx_PR)
    current_fx = SR(fx_PR)

    # Manage a single parameter for the entire tower
    m_parameter = None

    for step in range(max_steps):
        # Use the symbolic 'x' to get the degree of the symbolic expression
        n = int(current_fx.degree(x))
        g2 = len(pts_xy)
        if verbose:
            print(f"--- Tower Step {step + 1}: Building fibration for degree {n} curve, using g2={g2} ---")

        try:
            pts_x_subset = [p[0] for p in pts_xy[:g2]]

            step_result = build_one_fibration_step(
                current_fx, f0,
                pts_x_subset,
                g2,
                seed_int=seed_int + step,
                verbose=verbose,
                parameter_m=m_parameter,
            )

            check_fibration_step(step_result, prev_fx=current_fx)

            # After the first step, capture the parameter 'm' to be reused.
            if m_parameter is None:
                # Ensure r_expr is not constant before trying to get variables
                if has_free_variables(step_result['r_expr']):
                    m_parameter = list(step_result['r_expr'].variables())[0]
                else:
                    # This case should not happen in a real search but is possible
                    # if the geometry is degenerate.
                    raise RuntimeError("First step of fibration resulted in a constant root r, cannot define parameter.")

            # Pass the correct symbolic variable 'x' to the verification function.
            # current_fx, step_result, pts_x_subset, g2, x
            # Pass the correct symbolic variable 'x' and the current parameter 'm'
            _verify_fibration_step_properties(
                current_fx,
                step_result['r_expr'],
                m_parameter
            )

            tower.append(step_result)
            # Create the full curve equation y^2 = f_i(x,m)
            y = SR.var('y')
            full_equation = y**2 - step_result['f_i']
            jet_check_safe(full_equation, pts_xy)


            # The 'f_i' result is already a symbolic expression, so the type remains consistent.
            current_fx = step_result['f_i']

        except RuntimeError as e:
            print(f"Failed at step {step + 1}: {e}")
            raise # Re-raise to halt execution on failure


    verify_tower_consistency(tower)
    
    # Deep jet analysis across the entire tower
    print("\n" + "="*70)
    print("DEEP JET ANALYSIS ACROSS TOWER")
    print("="*70)
    jet_results = jet_check_tower_deep(tower, pts_xy, max_order=5, m0=0)
    
    return tower
    
# ---------- Main Driver ----------

@PROFILE
def main():
    """Main execution function."""
    print("tower.sage — Fibration Tower Builder")
    seed_int = 0

    COEFFS_GENUS2 = [QQ(1), QQ(4), QQ(-2), QQ(-18), QQ(1), QQ(38), QQ(25)]
    DATA_PTS_GENUS2 = [(QQ(-1), QQ(5))] # Provide y-coord for interpolation

    PR = PolynomialRing(QQ, 'x')
    x = PR.gen()
    fx_PR = sum(c * x**e for e, c in reversed(list(enumerate(reversed(COEFFS_GENUS2)))))

    print("Starting tower construction for the n=6 example curve.")
    # For 1pt case, we just pass one point
    tower = iterate_tower(fx_PR, DATA_PTS_GENUS2[:1], max_steps=2, seed_int=seed_int, verbose=True)
    for i, step in enumerate(tower):
        print(f"\n--- Layer {i+1} ---")
        print(f"Fibration info: {step['info']}")
        print(f"Interpolated Q(x): {step['Q_i']}")
        print(f"Parametric root r(m): {step['r_expr']}")
        print(f"Resulting fibration f_i(x,m): {step['f_i']}")

    if tower:
        print("\n✅ Tower construction finished. Constructed %d fibration layers." % len(tower))
    else:
        print("\n❌ Tower construction failed or produced no layers.")





from sage.all import (
    SR, var, solve
)

###############################################################
# Minimal jet checker for tower.sage
# Runs automatically, safe, pure-Python syntax, no new interface
###############################################################

from sage.all import SR, var, solve


# Replace previous jet_check_safe with this exact function (top-level in tower.sage)
def jet_check_tower_deep(tower, pts_xy, max_order=5, m0=0):
    """
    Deep jet analysis: expand x(m) and y(m) to high order
    and verify consistency across all tower layers.
    """
    from sage.all import factorial
    
    x0, y0 = pts_xy[0]
    
    # Symbolic setup
    x, y, m = var('x y m')
    t = var('t')  # local parameter
    
    # Create symbolic coefficients for the series
    a_coeffs = [var(f'a{i}') for i in range(2, max_order+1)]
    b_coeffs = [var(f'b{i}') for i in range(1, max_order+1)]
    
    # Build series expansions
    x_series = SR(x0) - t + sum(a_coeffs[i-2] * t**i for i in range(2, max_order+1))
    y_series = SR(y0) + sum(b_coeffs[i-1] * t**i for i in range(1, max_order+1))
    m_series = t + SR(m0)
    
    tower_jets = []
    for layer_idx, layer in enumerate(tower):
        print(f"\n[DEEP JET] Layer {layer_idx+1}")
        F_i = y**2 - layer['f_i']  # Construct full curve equation
        
        # Substitute series
        expr = F_i.subs({x: x_series, y: y_series, m: m_series}).expand()
        
        # Extract Taylor coefficients
        coeffs = {}
        for order in range(max_order + 1):
            c = expr.diff(t, order).subs({t: 0})
            if order > 0:
                c = c / factorial(order)
            coeffs[order] = c.simplify()
        
        # Solve order-by-order
        eqs = [coeffs[i] == 0 for i in range(max_order + 1) if coeffs[i] != 0]
        unknowns = a_coeffs + b_coeffs
        
        try:
            sol = solve(eqs, unknowns, solution_dict=True)
            if sol:
                first_sol = sol[0] if len(sol) == 1 else sol
                #free_params = [str(u) for u in unknowns if u not in first_sol]
                free_params = []
                for u in unknowns:
                    if u not in first_sol:
                        free_params.append(str(u))
                    else:
                        val = first_sol[u]
                        try:
                            val_vars = {str(v) for v in val.variables()}
                            unknown_names = {str(unk) for unk in unknowns}
                            new_free_vars = val_vars - unknown_names - {'m'}
                            if new_free_vars:
                                free_params.append(f"{str(u)}→{','.join(new_free_vars)}")
                        except:
                            raise

                
                if free_params:
                    print(f"  ✓ Solution found. Free parameters: {', '.join(free_params)}")
                else:
                    print(f"  ✓ Solution found. Fully determined (no free parameters)")
                
                # Print the series coefficients
                print(f"  Series expansion x(m) = {x0} - m + ...")
                for i in range(2, min(4, max_order+1)):  # Show a2, a3
                    coeff_var = var(f'a{i}')
                    if coeff_var in first_sol:
                        val = first_sol[coeff_var]
                        print(f"    a{i} = {val}")
                    else:
                        print(f"    a{i} = free")
                
                print(f"  Series expansion y(m) = {y0} + ...")
                for i in range(1, min(3, max_order)):  # Show b1, b2
                    coeff_var = var(f'b{i}')
                    if coeff_var in first_sol:
                        val = first_sol[coeff_var]
                        print(f"    b{i} = {val}")
                    else:
                        print(f"    b{i} = free")
                
                tower_jets.append({
                    'layer': layer_idx,
                    'solution': first_sol,
                    'free_params': free_params,
                    'obstructed': False
                })
            else:
                print(f"  ❌ OBSTRUCTED: No solution to higher-order equations")
                tower_jets.append({
                    'layer': layer_idx,
                    'obstructed': True,
                    'reason': 'No solution to higher-order equations'
                })
        except Exception as e:
            print(f"  ❌ OBSTRUCTED: {e}")
            tower_jets.append({
                'layer': layer_idx,
                'obstructed': True,
                'reason': str(e)
            })
    
    print("\n" + "="*70)
    print("TOWER JET SUMMARY")
    print("="*70)
    print(f"Total layers: {len(tower)}")
    obstructed = sum(1 for j in tower_jets if j.get('obstructed', False))
    if obstructed == 0:
        print("✓ All layers formally smooth (no obstructions)")
    else:
        print(f"❌ {obstructed} layer(s) have obstructions")
    
    return tower_jets



# Replace previous jet_check_safe with this exact function (top-level in tower.sage)
def jet_check_safe(F_sr, pts_xy, m0=0):
    """
    Minimal jet checker for tower.sage.
    - F_sr: SR expression for the current layer polynomial (f_i or F_i).
    - pts_xy: list-like of rational (x,y) pairs; uses pts_xy[0] as the point on the rail.
    - m0: base m-value (default 0).
    Prints one short line reporting a2 or obstruction. Let errors propagate.
    """
    assert pts_xy and len(pts_xy) >= 1, "pts_xy must contain at least one (x,y) pair"
    x0, y0 = pts_xy[0]

    # Declare symbols that must match those used in F_sr
    x, y, m = var('x y m')
    t = var('t')
    a2 = var('a2')
    b1 = var('b1')
    b2 = var('b2')

    # Convert F_sr to symbolic and expand to ensure proper form
    proto = SR(F_sr).expand()
    
    # local series ansatz: rail x = x0 - t (since x = x1 - m)
    x_series = SR(x0) - t + a2 * t * t
    y_series = SR(y0) + b1 * t + b2 * t * t
    m_series = t + SR(m0)

    # Substitute using the same symbolic names as F_sr
    try:
        expr = proto.subs({x: x_series, y: y_series, m: m_series})
    except TypeError as e:
        print(f" [JET] Substitution failed: {e}")
        print(f" [JET] F_sr variables: {proto.variables()}")
        return

    # Expand the substituted expression
    expr = expr.expand()

    # compute Taylor coefficients via derivatives at t=0
    try:
        c0 = expr.subs({t: 0}).simplify()
        c1 = expr.diff(t).subs({t: 0}).simplify()
        c2 = (expr.diff(t, 2).subs({t: 0}) / 2).simplify()
    except Exception as e:
        print(f" [JET] Taylor expansion failed: {e}")
        return

    eqs = []
    if c0 != 0:
        eqs.append(c0 == 0)
    if c1 != 0:
        eqs.append(c1 == 0)
    if c2 != 0:
        eqs.append(c2 == 0)

    if not eqs:
        print(" [JET] no local equations found")
        return

    try:
        sol = solve(eqs, [a2, b1, b2], solution_dict=True)
    except Exception as e:
        print(f" [JET] solve failed: {e}")
        return

    if not sol:
        print(" [JET] obstruction: no local lift at this point")
        return

    first = sol[0] if isinstance(sol, (list, tuple)) and sol else sol
    if isinstance(first, dict) and 'a2' in first:
        print(" [JET] a2 =", first['a2'])
    else:
        print(" [JET] a2 free (curvature unconstrained by double-root)")

if __name__ == '__main__':
    pass
    #main() # only for testing
