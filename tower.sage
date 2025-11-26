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
from sage.all import QQ, ZZ, gcd, factor, primes, SR, PolynomialRing, Integer, cached_function
import random, math


from search_common import DEBUG, SEED_INT, PRIME_POOL


# CONFIG: tune these to trade runtime vs accuracy
_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37,41]   # primes to test for rejections and collisions
_SAMPLE_M_VALUES = [ -7, -3, -1, 0, 1, 2, 3, 7 ]      # small integer m samples to probe modular behaviour
_NUM_RANDOM_M = 5                                      # additional random m samples (drawn small)
_MAX_DEGREE_PENALTY = 5.0
_WEIGHT_HEIGHT = 1.0
_WEIGHT_DEG = 1.2
_WEIGHT_DISC = 1.5
_WEIGHT_BADPRIME = 6.0    # each small bad prime is expensive -> large penalty
_WEIGHT_COLLISION = 8.0   # collisions are deadly for consensus; heavy penalty


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


from stats import *
# Utility: Print consensus effectiveness
@PROFILE
def build_multiple_fibrations(fx_PR, pts_xy, num_fibrations, max_steps=3, 
                               base_seed=SEED_INT, verbose=DEBUG):
    """
    Build multiple independent fibrations with different anchor points.
    Each fibration should find the same rational points (conjecturally).
    
    The number of anchor points is automatically determined to maximize
    diversity while maintaining degree drop constraints.
    """
    if not USE_ANCHOR_POINTS:
        raise RuntimeError("build_multiple_fibrations requires USE_ANCHOR_POINTS=True")
    
    # Calculate optimal number of anchor points
    # For a degree n curve reducing to degree 4, we have multiple steps
    # At each step, we can vary anchor points to create diversity
    from sage.all import PolynomialRing, QQ
    R_x = PolynomialRing(QQ, 'x')
    x = R_x.gen()
    #n = int(fx_PR.degree(x))
    n = int(fx_PR.degree())
 
    # For the first step (n -> n-1), degQ ~ (n-2)/2
    # We need degQ+1 constraints total, have 1 base point
    # So we have degQ degrees of freedom to distribute between tangency and anchors
    initial_degQ = (n - 2) // 2 if (n - 2) % 2 == 0 else (n - 1) // 2
    available_dof = initial_degQ  # After using 1 base point
    
    # Use 50% of DOF as anchor points (rest for tangency)
    # This balances diversity with computational stability
    num_anchors = available_dof
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-FIBRATION CONSENSUS MODE")
        print(f"Building {num_fibrations} independent fibrations")
        print(f"Using {num_anchors} anchor points per fibration")
        print(f"{'='*70}")
    
    # Temporarily override the global settings
    global NUM_ANCHOR_POINTS
    original_num_anchors = NUM_ANCHOR_POINTS
    NUM_ANCHOR_POINTS = num_anchors
    
    try:
        fibrations = []
        for k in range(num_fibrations):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Building Fibration {k+1}/{num_fibrations} (seed={base_seed + k})")
                print(f"{'='*70}")
            
            tower = iterate_tower(
                fx_PR=fx_PR,
                pts_xy=pts_xy,
                max_steps=max_steps,
                seed_int=base_seed + k,
                verbose=verbose,
                use_anchor_points=USE_ANCHOR_POINTS
            )

            import copy

            # inside build_multiple_fibrations(...) loop, replace the append with:
            # old:
            # fibrations.append({
            #     'tower': tower,
            #     'seed': base_seed + k,
            #     'id': k
            # })
            #
            # new (deepcopy snapshot + debug id print):
            _fib_snapshot = {
                'tower': copy.deepcopy(tower),
                'seed': base_seed + k,
                'id': k
            }
            fibrations.append(_fib_snapshot)

            # quick debug check (temporary; remove once verified)
            try:
                print("DEBUG appended tower id:", id(_fib_snapshot['tower']), "seed:", _fib_snapshot['seed'])
                # also show Q_i repr of last tower step if available (safe to call repr)
                if isinstance(_fib_snapshot['tower'], (list, tuple)) and _fib_snapshot['tower']:
                    last_step = _fib_snapshot['tower'][-1]
                    print("DEBUG last step Q_i repr:", repr(last_step.get('Q_i', '<no Q_i>')))
            except Exception:
                raise


        return fibrations
    finally:
        # Restore original setting
        NUM_ANCHOR_POINTS = original_num_anchors


if __name__ == '__main__':
    pass
    #main() # only for testing


def interpolate_Q_with_anchors(base_pts, degQ, x_sym, anchor_pts, seed_int=SEED_INT):
    """
    Compute Q(x) using base interpolation points plus anchor points (no tangency).
    Anchor points are just arbitrary rational (x,y) pairs that pin down Q's free DOF.
    
    Args:
        base_pts: list of (x,y) from the fibration base point
        degQ: degree of Q
        x_sym: symbolic variable
        anchor_pts: additional arbitrary rational (x,y) pairs
        seed_int: random seed
        
    Returns:
        Q polynomial over QQ
    """
    # Combine base points and anchor points
    all_pts = list(base_pts) + list(anchor_pts)
    
    # We need exactly degQ+1 points for interpolation
    if len(all_pts) != degQ + 1:
        raise RuntimeError(f"Need exactly {degQ+1} points for degree {degQ} Q, but have {len(all_pts)} (base: {len(base_pts)}, anchors: {len(anchor_pts)})")
    
    # Extract x and y coordinates
    xs = [QQ(pt[0]) for pt in all_pts]
    ys = [QQ(pt[1]) for pt in all_pts]
    
    # Check for duplicate x-coordinates
    if len(set(xs)) != len(xs):
        raise RuntimeError(f"Duplicate x-coordinates in interpolation points: {xs}")
    
    # Build polynomial using Lagrange interpolation
    R = PolynomialRing(QQ, str(x_sym))
    
    # Lagrange interpolation: Q(x) = Σ y_i * L_i(x)
    # where L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
    Qx = R(0)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        # Build Lagrange basis polynomial L_i(x)
        Li = R(1)
        for j, xj in enumerate(xs):
            if i != j:
                Li *= (R.gen() - xj) / (xi - xj)
        Qx += yi * Li
    
    return Qx


# Utility: Print consensus effectiveness
def print_consensus_effectiveness(consensus_stats, cumulative_stats):
    """
    Print how effective the consensus filter was at reducing junk.
    """
    print(f"\n{'='*70}")
    print("CONSENSUS FILTER EFFECTIVENESS")
    print(f"{'='*70}")
    
    cs = consensus_stats
    print(f"\nResidues filtered: {cs['total_before'] - cs['total_after']:,} / {cs['total_before']:,}")
    print(f"Reduction: {100*cs['reduction_ratio']:.1f}%")
    
    # Compare to rationality test results
    total_tests = cumulative_stats.counters.get('rationality_tests_total', 0)
    successes = cumulative_stats.counters.get('rationality_tests_success', 0)
    
    if total_tests > 0:
        hit_rate = successes / total_tests
        print(f"\nRationality tests:")
        print(f"  Total: {total_tests:,}")
        print(f"  Successes: {successes:,}")
        print(f"  Hit rate: {100*hit_rate:.2f}%")
        
        # Estimate how many tests we saved
        tests_saved = int(cs['total_before'] - cs['total_after'])
        
        # Calculate average time per test from search phase
        search_time = cumulative_stats.phase_times.get('search_subsets_and_check', 0)
        if total_tests > 0:
            time_per_test = search_time / total_tests
            time_saved_est = tests_saved * time_per_test
            
            print(f"\nEstimated tests saved: ~{tests_saved:,}")
            print(f"Estimated time saved: ~{time_saved_est:.1f}s")


@PROFILE
def compute_consensus_residues(precomputed_residues_list, prime_pool, 
                                consensus_threshold=CONSENSUS_THRESHOLD,
                                debug=DEBUG):
    """
    Compute consensus residues across multiple fibrations.
    A residue is kept if it appears in >= consensus_threshold fraction of 
    *participating* fibrations (those that successfully computed residues for that prime).
    
    Args:
        precomputed_residues_list: List of precomputed_residues dicts (one per fibration)
        prime_pool: List of primes
        consensus_threshold: Minimum fraction of participating fibrations that must agree
        
    Returns:
        consensus_residues: Dict in same format as precomputed_residues
        stats: Dict with filtering statistics
    """
    from collections import defaultdict, Counter
    import math
    
    num_fibrations = len(precomputed_residues_list)
    
    if debug:
        print(f"\n{'='*70}")
        print(f"CONSENSUS FILTER: {num_fibrations} fibrations, threshold={consensus_threshold:.1%}")
        print(f"Policy: Primes with no data in a fibration abstain from voting.")
        print(f"{'='*70}")
    
    # 1. Determine participation per prime
    # participating_counts[p] = number of fibrations that have non-empty data for p
    participating_counts = defaultdict(int)
    
    for precomp in precomputed_residues_list:
        for p in prime_pool:
            # Check if p exists and has any vectors/residues
            if p in precomp and precomp[p]:
                participating_counts[p] += 1
                
    # 2. Track votes
    # residue_votes[(p, v_tuple, rhs_idx)][residue] = count
    residue_votes = defaultdict(Counter)
    
    # Track max RHS index per (p, v_tuple) to properly initialize lists later
    max_rhs_indices = defaultdict(int)
    
    for precomp in precomputed_residues_list:
        for p in prime_pool:
            if p not in precomp or not precomp[p]:
                continue
                
            mapping = precomp[p]
            for v_tuple, rhs_lists in mapping.items():
                # Track max rhs index
                if len(rhs_lists) - 1 > max_rhs_indices[(p, v_tuple)]:
                    max_rhs_indices[(p, v_tuple)] = len(rhs_lists) - 1
                    
                for rhs_idx, residue_set in enumerate(rhs_lists):
                    key = (p, v_tuple, rhs_idx)
                    for r in residue_set:
                        if isinstance(r, int):
                            residue_votes[key][r] += 1

    # 3. Filter
    consensus_residues = {}
    stats = {
        'total_before': 0,
        'total_after': 0,
        'per_prime_before': {},
        'per_prime_after': {},
        'reduction_ratio': 0.0,
        'participation': {}
    }
    
    # Helper to organize keys by prime for efficiency
    keys_by_prime = defaultdict(list)
    for k in residue_votes.keys():
        keys_by_prime[k[0]].append(k)
        
    for p in prime_pool:
        n_participating = participating_counts[p]
        stats['participation'][p] = n_participating
        
        if n_participating == 0:
            continue
            
        # Calculate votes needed for THIS prime
        # Use ceil to ensure we don't accept 0 votes, and strictness matches intent
        # e.g. 0.8 * 2 = 1.6 -> 2. 0.8 * 1 = 0.8 -> 1.
        min_votes_needed = int(math.ceil(consensus_threshold * n_participating))
        # Ensure at least 1 vote is needed if anyone participated
        min_votes_needed = max(1, min_votes_needed)
        
        consensus_residues[p] = {}
        
        prime_before = 0
        prime_after = 0
        
        p_keys = keys_by_prime[p]
        
        for key in p_keys:
            _, v_tuple, rhs_idx = key
            vote_counter = residue_votes[key]
            
            original_set = set(vote_counter.keys())
            consensus_set = {r for r, c in vote_counter.items() if c >= min_votes_needed}
            
            prime_before += len(original_set)
            prime_after += len(consensus_set)
            
            if consensus_set:
                if v_tuple not in consensus_residues[p]:
                    # Initialize list with empty sets up to max needed
                    needed = max_rhs_indices[(p, v_tuple)] + 1
                    consensus_residues[p][v_tuple] = [set() for _ in range(needed)]
                
                # Just in case logic implies we need to extend (should be covered by max_rhs_indices)
                while len(consensus_residues[p][v_tuple]) <= rhs_idx:
                    consensus_residues[p][v_tuple].append(set())
                
                consensus_residues[p][v_tuple][rhs_idx] = consensus_set

        stats['per_prime_before'][p] = prime_before
        stats['per_prime_after'][p] = prime_after
        stats['total_before'] += prime_before
        stats['total_after'] += prime_after

    if stats['total_before'] > 0:
        stats['reduction_ratio'] = 1.0 - (stats['total_after'] / stats['total_before'])

    if debug:
        print(f"\nConsensus Filter Results:")
        print(f"  Total residues before: {stats['total_before']:,}")
        print(f"  Total residues after:  {stats['total_after']:,}")
        print(f"  Filtered out: {stats['total_before'] - stats['total_after']:,} "
              f"({100*stats['reduction_ratio']:.1f}%)")
        
        # Show per-prime breakdown for top primes
        sorted_primes = sorted(stats['per_prime_before'].items(), 
                              key=lambda x: -x[1])[:10]
        print(f"\n  Top 10 primes by original residue count:")
        for p, before in sorted_primes:
            after = stats['per_prime_after'].get(p, 0)
            part = stats['participation'].get(p, 0)
            reduction = 1.0 - (after / before) if before > 0 else 0.0
            print(f"  p={p}: {before} -> {after} ({100*reduction:.1f}% filtered) [Participating: {part}/{num_fibrations}]")

    return consensus_residues, stats


def generate_anchor_points(num_points, seed=ANCHOR_SEED, exclude_x=None):
    """
    Generate anchor points with SMALL denominators to minimize coefficient blowup.
    Strategy: Use x,y in QQ with denominators that are powers of a SINGLE small prime.
    """
    random.seed(int(seed))
    anchor_pts = []
    
    if exclude_x is None:
        used_x = set()
    else:
        used_x = set(QQ(x) for x in exclude_x)
    
    # Use only denominators that are powers of 2 (or 3, but 2 is safest)
    #allowed_denoms = [1, 2, 4, 8]  # Powers of 2
    # Use larger primes to push singularities away from simple integers
    #allowed_denoms = [2, 3, 5, 7, 11, 13] 
    #allowed_denoms = PRIME_POOL[3:]
    allowed_denoms = [2, 4, 8, 16, 32]

    max_attempts = 100
    attempts = 0
    
    while len(anchor_pts) < num_points and attempts < max_attempts:
        attempts += 1
        
        # Generate x with small denominator
        num_x = random.randint(-10, 10)
        den_x = random.choice(allowed_denoms)
        x_val = QQ(num_x) / QQ(den_x)
        
        if x_val in used_x:
            continue
        
        # Generate y with small denominator
        num_y = random.randint(-10, 10)
        den_y = random.choice(allowed_denoms)
        y_val = QQ(num_y) / QQ(den_y)
        
        anchor_pts.append((x_val, y_val))
        used_x.add(x_val)
    
    if len(anchor_pts) < num_points:
        raise RuntimeError(f"Could not generate {num_points} unique anchor points")
    
    return anchor_pts


def verify_y2_consistency_on_rail(tower, x1, m_vals):
    """Verify y²_i = y²_{i+1} along x = x₁ - m for each layer."""
    x, m = var('x m')
    
    for i in range(len(tower) - 1):
        # Force conversion to symbolic ring
        f_i = SR(tower[i]['f_i'])
        f_i_plus_1 = SR(tower[i+1]['f_i'])
        
        # Evaluate along the rail x = x₁ - m
        r_expr = SR(x1) - m
        
        for m_val in m_vals:
            # Substitute in two steps to ensure proper evaluation
            y2_i = f_i.subs({x: r_expr}).subs({m: m_val})
            y2_i_plus_1 = f_i_plus_1.subs({x: r_expr}).subs({m: m_val})
            
            diff = (y2_i - y2_i_plus_1).expand().simplify()
            
            if diff != 0:
                print(f"diff = {diff}, m_val = {m_val}")
                print(f"f_i type: {type(tower[i]['f_i'])}")
                print(f"f_i parent: {tower[i]['f_i'].parent() if hasattr(tower[i]['f_i'], 'parent') else 'no parent'}")
                raise RuntimeError(
                    f"y² consistency violated at layer {i}->{i+1}, m={m_val}"
                )


@PROFILE
def iterate_tower(fx_PR, pts_xy, max_steps=3, seed_int=SEED_INT, verbose=DEBUG, use_anchor_points=USE_ANCHOR_POINTS):
    """
    Iterates through the fibration tower construction process.
    Now uses a "Best-of-N" strategy to minimize coefficient height and stabilize Capacity.
    """
    tower = []
    
    poly_x_gen = fx_PR.parent().gen()
    x = SR.var(str(poly_x_gen))
    f0 = SR(fx_PR)
    current_fx = SR(fx_PR)
    m_parameter = None

    # --- CONFIG: STABILITY SETTINGS ---
    # Number of random geometries to try per step. 
    # Higher = more stable capacity, slightly slower build.
    CANDIDATES_PER_STEP = 40
    # ----------------------------------

    for step in range(max_steps):
        n = int(current_fx.degree(x))
        g2 = len(pts_xy)
        if verbose:
            print(f"--- Tower Step {step + 1}: Building fibration for degree {n} curve (Best-of-{CANDIDATES_PER_STEP}) ---")

        best_step_result = None
        best_score = float('inf')
        
        pts_x_subset = [p[0] for p in pts_xy[:g2]]

        # Try multiple seeds to find the "cleanest" geometry
        for attempt in range(CANDIDATES_PER_STEP):
            try:
                # Diverge the seed for each attempt
                # We use a deterministic offset so the "best" result is reproducible
                attempt_seed = seed_int * 1000 + step * 100 + attempt
                
                step_result = build_one_fibration_step(
                    current_fx, f0,
                    pts_x_subset,
                    g2,
                    seed_int=attempt_seed,
                    verbose=False, # Silence inner prints
                    parameter_m=m_parameter,
                    use_anchor_points=use_anchor_points
                )

                # Validate
                check_fibration_step(step_result, prev_fx=current_fx)
                
                # Verify properties (derivs, etc)
                temp_m = m_parameter
                if temp_m is None and has_free_variables(step_result['r_expr']):
                     temp_m = list(step_result['r_expr'].variables())[0]
                
                _verify_fibration_step_properties(current_fx, step_result['r_expr'], temp_m)

                # Score it
                score = measure_poly_complexity(step_result['f_i'])
                
                if score < best_score:
                    best_score = score
                    best_step_result = step_result
                    # If we set parameter_m for the first time, keep it consistent
                    if m_parameter is None:
                        m_parameter = temp_m

            except RuntimeError:
                raise
                continue # Skip failed attempts
        
        if best_step_result is None:
            raise RuntimeError(f"Failed to build valid geometry for step {step+1} after {CANDIDATES_PER_STEP} attempts")

        if verbose:
            print(f"  [Step {step+1}] Selected best geometry (Score={best_score:.1f})")

        tower.append(best_step_result)
        
        # Create the full curve equation y^2 = f_i(x,m)
        y = SR.var('y')
        full_equation = y**2 - best_step_result['f_i']
        #jet_check_safe(full_equation, pts_xy) # only for debugging

        current_fx = best_step_result['f_i']

    verify_tower_consistency(tower)
    verify_y2_consistency_on_rail(tower, x1=pts_xy[0][0], m_vals=[0, 1, -1, QQ(1/2)]) 
    
    if False: # only for debugging
        print("\n" + "="*70)
        print("DEEP JET ANALYSIS ACROSS TOWER")
        print("="*70)
        jet_results = jet_check_tower_deep(tower, pts_xy, max_order=5, m0=0)
    
    return tower


# Replace measure_poly_complexity with this more robust geometry scorer.
# Uses Sage objects but written in plain Python style.

def _int_log(x):
    if x <= 1:
        return 0.0
    return math.log(float(x))


def measure_poly_complexity(expr_sr):
    """
    Improved scoring: lower is better. Raises exceptions on failure.
    Input: expr_sr is an SR polynomial-like object representing the fibration polynomial f_i(x,m).
    """
    if expr_sr is None:
        raise ValueError("measure_poly_complexity: expr_sr is None")
    
    fx_sr = SR(expr_sr)
    x_var = SR.var('x')
    
    # 1. Coefficient Extraction
    coeffs = []
    raw_coeffs = None
    try:
        raw_coeffs = fx_sr.coefficients(x_var)
        coeffs = [c[0] for c in raw_coeffs]
    except Exception:
        coeffs = [fx_sr]
        raw_coeffs = None

    # 1. Height Score
    height_score = 0.0
    for c in coeffs:
        try:
            q = QQ(c)
            h = max(abs(int(q.numerator())), abs(int(q.denominator())))
            height_score += _int_log(h + 1)
        except (TypeError, ValueError):
            # Symbolic coefficient: penalize existence + complexity proxy (nops)
            # CRITICAL FIX: Do NOT use str(c) or repr(c) here, it causes segfaults/hangs on huge expressions.
            try:
                # nops() counts top-level operands; decent proxy for tree width
                n = c.nops()
            except:
                n = 1
            height_score += 1.0 + 0.1 * n
            
    # 2. Degree Score
    deg_x = 0
    deg_m = 0
    m_var = None
    
    try:
        # variables() is usually safe even for large expressions
        vars_list = fx_sr.variables()
        for v in vars_list:
            if str(v) == 'm':
                m_var = v
                break
                
        try:
            deg_x = int(fx_sr.degree(x_var))
        except Exception:
            deg_x = 0
            
        if m_var is not None:
            try:
                deg_m = int(fx_sr.degree(m_var))
            except Exception:
                deg_m = 0
    except Exception as e:
        raise RuntimeError(f"Failed to extract degree info: {e}")

    degree_penalty = _int_log(1 + deg_x) * 0.7 + _int_log(1 + deg_m) * 0.5
    
    # 3. Discriminant / Size Score
    # CRITICAL FIX: Do NOT use len(str(fx_sr))
    try:
        disc_score = 0.1 * fx_sr.nops()
    except:
        disc_score = 1.0
    
    # 4. Bad Prime / Collision Score
    bad_prime_count = 0
    collision_count = 0
    
    for p in _SMALL_PRIMES:
        p_bad = False
        
        # Check coefficients (numeric only)
        for c in coeffs:
            try:
                q = QQ(c)
                if int(q.denominator()) % p == 0:
                    p_bad = True
                    break
            except (TypeError, ValueError):
                # Skip symbolic check to avoid string conversion
                pass
        
        if p_bad:
            bad_prime_count += 1
            continue
            
        # Collision Check (Heuristic)
        # Safe construction of GF(p) polynomial
        for mval in (_SAMPLE_M_VALUES + [random.randint(-17,17) for _ in range(_NUM_RANDOM_M)]):
            collision_detected = False
            try:
                if raw_coeffs is not None:
                    poly_dict = {}
                    for c_expr, expon in raw_coeffs:
                        if m_var is not None:
                            val_sr = c_expr.subs({m_var: Integer(mval)})
                        else:
                            val_sr = c_expr
                        
                        val_qq = QQ(val_sr)
                        if Integer(val_qq.numerator()) % p != 0:
                            poly_dict[int(expon)] = GF(p)(val_qq)

                    if poly_dict:
                        R_p = PolynomialRing(GF(p), 'x')
                        poly_x = R_p(poly_dict)
                        
                        if poly_x.degree() > 0:
                            if poly_x.gcd(poly_x.derivative()).degree() > 0:
                                collision_detected = True
            

            except (ZeroDivisionError, ValueError, TypeError):
                # Denominator issue -> bad prime
                collision_detected = True
            except Exception:
                raise
            
            if collision_detected:
                collision_count += 1
                break
    
    max_checks = len(_SMALL_PRIMES) * (_NUM_RANDOM_M + len(_SAMPLE_M_VALUES))
    collision_frac = float(collision_count) / max(1, max_checks)
    
    total_score = (
        _WEIGHT_HEIGHT * height_score +
        _WEIGHT_DEG * degree_penalty +
        _WEIGHT_DISC * disc_score +
        _WEIGHT_BADPRIME * bad_prime_count +
        _WEIGHT_COLLISION * collision_frac * 10.0
    )
    
    return float(total_score)


    
    # 0. Prepare RHS (constant terms) and Matrix rows
    # Since equations are linear, eq = c0*

from sage.rings.rational_field import QQ


@PROFILE
def build_one_fibration_step(fx_SR, f0, pts_x, g2, seed_int=SEED_INT,
                             verbose=False, forced_tangency_seq=None,
                             forced_Qpoly=None, force_Q_constraint_indices=None,
                             parameter_m=None, use_anchor_points=USE_ANCHOR_POINTS):
    """
    Modified version: Reduces tangency constraints by 1 to impose a Q-dependence 
    mixing constraint. Uses linear algebra for solving to avoid Maxima hangs.
    Now clamps anchor usage to available degrees of freedom to prevent tower crashes.
    """
    random.seed(int(seed_int))
    xSR = SR.var('x')
    
    n = int(fx_SR.degree(xSR))
    xs_chosen = [QQ(xv) for xv in pts_x]
    if len(xs_chosen) == 0:
        raise RuntimeError("build_one_fibration_step: pts_x must contain at least one x-value (x1).")
    x1 = xs_chosen[0]
    
    # Degree drop constraint
    max_degQ = (n - 1) // 2
    initial_degQ = choose_degQ(n)
    degQ = min(initial_degQ, max_degQ)
    
    if forced_Qpoly is not None:
        try:
            forced_Q_SR = SR(forced_Qpoly)
            forced_deg = int(forced_Q_SR.degree(xSR))
        except Exception:
            try:
                Rtmp = PolynomialRing(QQ, str(xSR))
                forced_deg = int(Rtmp(forced_Qpoly).degree())
            except Exception:
                raise RuntimeError("Could not determine degree of forced_Qpoly")
        if forced_deg > max_degQ:
            raise RuntimeError(f"forced_Qpoly has degree {forced_deg} > allowed max {max_degQ}")
        degQ = forced_deg
    
    # Build Q polynomial
    if forced_Qpoly is not None:
        try:
            Rqq = PolynomialRing(QQ, str(xSR))
            Qpoly_QQ = Rqq(forced_Qpoly)
        except Exception:
            Qpoly_QQ = SR(forced_Qpoly)
    else:
        # Check if we should use anchor points
        if use_anchor_points:
            # We need degQ+1 total points. We have 1 base point.
            total_needed = degQ + 1
            base_pts_count = 1
            remaining_dof = total_needed - base_pts_count
            
            # Fix: Clamp the requested anchors to the actual available DOF
            # This prevents crashes when the tower gets deeper and degQ drops (e.g. n=6->5)
            num_anchors_needed = min(NUM_ANCHOR_POINTS, remaining_dof)
            
            # Generate anchor points
            base_x_coords = [QQ(xv) for xv in xs_chosen]
            anchor_pts = generate_anchor_points(num_anchors_needed, seed=seed_int, exclude_x=base_x_coords)
            
            if verbose:
                print(f"[ANCHOR MODE] Using {len(anchor_pts)} anchor points: {anchor_pts}")
            
            # Build base point
            chosen_pts_xy = []
            f0_SR = SR(f0)
            for xv in xs_chosen:
                y_val_expr = f0_SR.subs({xSR: SR(xv)})
                try:
                    yi = sqrt(QQ(y_val_expr))
                except Exception:
                    yi = SR(sqrt(y_val_expr))
                chosen_pts_xy.append((QQ(xv), yi))
            
            # If we still need tangency conditions after anchors
            num_tangency_needed = remaining_dof - num_anchors_needed
            
            if num_tangency_needed == 0:
                # Pure anchor mode: no tangency conditions
                Qpoly_QQ = interpolate_Q_with_anchors(chosen_pts_xy, degQ, xSR, anchor_pts, seed_int=seed_int)
            else:
                raise RuntimeError("Hybrid anchor+tangency mode not yet implemented. Set NUM_ANCHOR_POINTS to use all DOF or 0.")
        else:
            # Original tangency-based mode
            chosen_pts_xy = []
            f0_SR = SR(f0)
            for xv in xs_chosen:
                y_val_expr = f0_SR.subs({xSR: SR(xv)})
                try:
                    yi = sqrt(QQ(y_val_expr))
                except Exception:
                    yi = SR(sqrt(y_val_expr))
                chosen_pts_xy.append((QQ(xv), yi))
            
            Qpoly_QQ = interpolate_Q_general(chosen_pts_xy, f0, degQ, xSR,
                                            seed_int=seed_int,
                                            force_constraint_indices=force_Q_constraint_indices)
    
    import copy
    Q_SR = copy.deepcopy(SR(Qpoly_QQ))
    
    prod1 = poly_prod_numeric(xs_chosen, xSR)
    deg_prod = int(prod1.degree(xSR))
    rest_deg = int(n - 1 - deg_prod)
    
    if rest_deg < 0:
        raise RuntimeError(f"rest polynomial degree would be negative: rest_deg={rest_deg}")
    
    rest_coeff_names = [f"b_rest_{i}" for i in range(rest_deg + 1)]
    rest_coeff_syms = [SR.var(name) for name in rest_coeff_names]
    rest_poly_SR = sum(rest_coeff_syms[i] * xSR**i for i in range(rest_deg + 1))
    
    fibration_SR = (SR(Q_SR)**2).expand() + (SR(prod1) * rest_poly_SR).expand()
    diff_poly = (SR(fx_SR) - fibration_SR).expand()
    
    if parameter_m is None:
        m = SR.var('m')
    else:
        m = SR(parameter_m)
    
    r_expr = SR(QQ(x1)) - m
    
    eqs = []
    # 1. Root condition: f(r) = f0(r)
    eqs.append(diff_poly.subs({xSR: r_expr}))
    # 2. Derivative condition at r
    eqs.append(diff(diff_poly, xSR).subs({xSR: r_expr}))
    
    unknowns = rest_coeff_syms[:]
    
    # We reserve equations based on strategy
    num_tangency_eqs = len(unknowns) - 2 - 1 if use_anchor_points else len(unknowns) - 2
    
    if num_tangency_eqs < 0 or not use_anchor_points:
        if use_anchor_points:
            print("Warning: Not enough DOF for Q-mixing strategy, reverting to full tangency.")
        num_tangency_eqs = len(unknowns) - 2
        use_mixing = False
    else:
        use_mixing = True

    assert use_mixing == use_anchor_points, use_mixing

    tangency_counts = {QQ(xi): 0 for xi in xs_chosen}
    
    # Select tangency points
    sel_points = []
    if num_tangency_eqs > 0:
        if forced_tangency_seq is not None:
            if len(forced_tangency_seq) >= num_tangency_eqs:
                sel_points = [QQ(xv) for xv in forced_tangency_seq[:num_tangency_eqs]]
            else:
                 raise RuntimeError("forced_tangency_seq too short for requested tangency count")
        else:
            sel_points = [QQ(random.choice(xs_chosen)) for _ in range(num_tangency_eqs)]

    # Add tangency equations
    for xv in sel_points:
        tangency_counts[QQ(xv)] += 1
        current_order = tangency_counts[QQ(xv)]
        eq_t = diff(diff_poly, xSR, current_order).subs({xSR: SR(xv)}).expand()
        eqs.append(eq_t)
    
    # Add Q-Mixing Constraint
    if use_mixing:
        x_mix_num = 2 * int(QQ(x1).numerator()) + 3
        x_mix_den = QQ(x1).denominator()
        while x_mix_den % 2 == 0:
            x_mix_den //= 2
        x_mix_den = 2
        x_mix = QQ(x_mix_num) / QQ(x_mix_den)

        val_Q = Q_SR.subs({xSR: x_mix})
        val_R = rest_poly_SR.subs({xSR: x_mix})
        eq_mix = (val_R - val_Q).expand()
        eqs.append(eq_mix)

    if len(eqs) != len(unknowns):
        raise RuntimeError(f"Equation/unknown mismatch: {len(eqs)} equations vs {len(unknowns)} unknowns")
    
    # --- REPLACEMENT: Linear Algebra Solver instead of Maxima ---
    # The system is linear in `unknowns`.
    # Convert to Ax=b and solve using matrix(SR).
    
    from sage.matrix.constructor import matrix
    
    # 0. Prepare RHS (constant terms) and Matrix rows
    # Since equations are linear, eq = c0*b0 + c1*b1 + ... + const
    # const = eq.subs({all_unknowns: 0})
    # coeff_i = eq.coefficient(bi)
    
    zero_sub = {u: 0 for u in unknowns}
    rhs_vec = []
    rows = []
    
    for eq in eqs:
        # constant term is eq evaluated at all unknowns=0
        c_term = eq.subs(zero_sub)
        rhs_vec.append(-c_term) # Move constant to RHS
        
        row = []
        for u in unknowns:
            # Efficiently extract coefficient in SR
            row.append(eq.coefficient(u))
        rows.append(row)


    # ... inside build_one_fibration_step ...

    # Define Fraction Field for m to solve system exactly and quickly

    # We need to detect the variable name used for m
    m_name = 'm'
    if parameter_m is not None:
        m_name = str(parameter_m)

    R_m = PolynomialRing(QQ, m_name)
    Fm = R_m.fraction_field()

    # Helper to coerce SR expression to Fm
    def to_Fm(expr):
        try:
            return Fm(expr) 
        except:
            # Fallback: conversion via numerator/denominator polynomials
            try:
                # This handles cases where direct conversion fails but it is rational
                ex_sr = SR(expr)
                numer = ex_sr.numerator()
                denom = ex_sr.denominator()
                return R_m(SR(numer)) / R_m(SR(denom))
            except Exception:
                raise ValueError(f"Cannot coerce {expr} to Fm")

    # STRATEGY: Solve over FractionField(QQ['m']) - Fast & Exact
    # We skip trying QQ because we know 'm' is involved in the equations (via r_expr).
    try:
        rows_Fm = []
        rhs_Fm = []
        for r_idx, row in enumerate(rows):
            rows_Fm.append([to_Fm(c) for c in row])
            rhs_Fm.append(to_Fm(rhs_vec[r_idx]))

        M_Fm = matrix(Fm, rows_Fm)
        b_vec_Fm = vector(Fm, rhs_Fm)

        # This solve is fast (Gaussian elimination on rational functions)
        sol_vec = M_Fm.solve_right(b_vec_Fm)
        sol = {u: sol_vec[i] for i, u in enumerate(unknowns)}

    except Exception as e:
        # Fallback to SR (Slow, Maxima) only if Fm fails (e.g. sqrt(2) in coeffs)
        if verbose: print(f"Matrix QQ(m) solve failed ({e}), falling back to SR (slow)...")
        M = matrix(SR, rows)
        b_vec = vector(SR, rhs_vec)
        sol_vec = M.solve_right(b_vec)
        sol = {u: sol_vec[i] for i, u in enumerate(unknowns)}

    # -----------------------------------------------------------
    
    solved_map = {}
    contains_symbolic = False
    for symb in unknowns:
        val_SR = SR(sol[symb])
        solved_map[symb] = val_SR
        try:
            _ = QQ(val_SR)
        except Exception:
            contains_symbolic = True
    
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
    
    Q_SR = SR(Q_SR)
    prod_SR = SR(prod1)
    
    Q_SR_symbolic = SR(Q_SR)
    prod_SR_symbolic = SR(prod_SR) 
    rest_SR_symbolic = SR(rest_poly_SR_solved)

    fibration_solved_SR = (Q_SR_symbolic**2).expand() + (prod_SR_symbolic * rest_SR_symbolic).expand()
    fibration_solved_SR = SR(fibration_solved_SR).expand()

    # Extract all denominators from coefficients
    PR_m = PolynomialRing(QQ, 'm')
    Fm = PR_m.fraction_field()
    m_poly = PR_m.gen()

    coeffs_in_x = [fibration_solved_SR.coefficient(xSR, i) for i in range(n)]
    all_denoms = []

    for c in coeffs_in_x:
        try:
            c_poly = Fm(c)
            for coef in c_poly.list():
                if coef != 0:
                    all_denoms.append(QQ(coef).denominator())
        except:
            pass

    if all_denoms:
        from sage.arith.misc import lcm as sage_lcm
        denom_lcm = sage_lcm(all_denoms)
        fibration_solved_SR = (fibration_solved_SR * denom_lcm).expand()
        if verbose:
            print(f"[denom_clear] Cleared denominators by multiplying by {denom_lcm}")

    test_r = r_expr.subs({m: 0})
    lhs = SR(fibration_solved_SR).subs({xSR: test_r, m: 0})
    rhs = SR(fx_SR).subs({xSR: test_r})
    diff_check = (lhs - rhs).expand()
    assert diff_check.simplify() == 0, f"Rail consistency violated at m=0: diff = {diff_check}"

    try:
        deg_fib = int(fibration_solved_SR.degree(xSR))
    except Exception:
        deg_fib = None
    
    target_deg = n - 1
    if deg_fib is None or deg_fib != target_deg:
        diag = []
        diag.append(f"expected fibration degree {target_deg}, got {deg_fib}")
        try:
            deg_Q2 = int((Q_SR**2).degree(xSR))
            diag.append(f"deg(Q^2) = {deg_Q2}")
        except Exception:
            diag.append("deg(Q^2) unknown")
        try:
            deg_prodrest = int((prod_SR * rest_poly_SR_solved).degree(xSR))
            diag.append(f"deg(prod*rest) = {deg_prodrest}")
        except Exception:
            diag.append("deg(prod*rest) unknown")
        diag_msg = "; ".join(diag)
        raise RuntimeError("Degree drop failed: " + diag_msg)
    
    return {
        'f_i': SR(fibration_solved_SR),
        'Q_i': Qpoly_QQ,
        'Q_QQ': Qpoly_QQ if isinstance(Qpoly_QQ, type(PolynomialRing(QQ, 'x')(0))) else Q_SR,
        'r_expr': SR(r_expr),
        'rest_poly_SR': SR(rest_poly_SR_solved),
        'rest_poly_QQ': rest_poly_QQ,
        'info': f"n={n} degProd={deg_prod} rest_deg={rest_deg} anchor_mode={use_anchor_points} num_anchors={NUM_ANCHOR_POINTS if use_anchor_points else 0} mixed={use_mixing}",
    }



from sage.all import Integer, PolynomialRing, GF, inverse_mod, ZZ

def measure_poly_complexity(expr_sr):
    """
    Improved scoring: lower is better. Raises exceptions on unexpected failure.
    Input: expr_sr is an SR polynomial-like object representing f_i(x,m).
    This version avoids heavy QQ(...) conversions in the inner loops and
    reuses per-prime rings. If a coefficient cannot be cheaply reduced to
    numerator/denominator integers we treat that (p,m) check as failing
    (counts as a bad prime / collision as configured).
    """
    if expr_sr is None:
        raise ValueError("measure_poly_complexity: expr_sr is None")

    fx_sr = SR(expr_sr)
    x_var = SR.var('x')

    # 1. Coefficient extraction (best-effort, fall back to whole expr)
    try:
        raw_coeffs = fx_sr.coefficients(x_var)  # list of (coeff_expr, exponent)
    except Exception as e:
        # If coefficient extraction itself fails, escalate (this should be rare).
        raise RuntimeError(f"Failed to extract coefficients: {e}")

    if not raw_coeffs:
        # treat as constant
        raw_coeffs = [(fx_sr, 0)]

    # Helper: cheap conversion of a coefficient expression (after m-substitution)
    def _cheap_num_den_from_sr(val_sr):
        """
        Try to obtain (num, den) as plain Python ints from val_sr without triggering
        heavy symbolic evaluation. Returns (num, den) or raises ValueError if not possible.
        """
        # Fast path: already a Sage Integer or Python int
        if isinstance(val_sr, Integer) or isinstance(val_sr, int):
            return int(val_sr), 1

        # Some SR results expose numerator()/denominator() cheaply
        try:
            num_obj = val_sr.numerator()
            den_obj = val_sr.denominator()
            # if these are Sage ints or Python ints, convert without QQ()
            if (isinstance(num_obj, Integer) or isinstance(num_obj, int)) and \
               (isinstance(den_obj, Integer) or isinstance(den_obj, int)):
                return int(num_obj), int(den_obj)
        except Exception:
            # numerator/denominator might call Maxima for complicated SR; bail out
            raise ValueError("Cannot cheaply extract numerator/denominator")

        # Last-chance cheap rationalization: some SR objects support .is_rational() -> then cast
        try:
            # do not use QQ(val_sr) here (heavy). Instead try int() promotion if it is exact
            if hasattr(val_sr, 'is_integer') and val_sr.is_integer():
                return int(val_sr), 1
        except Exception:
            pass

        raise ValueError("Cannot cheaply extract numerator/denominator (fallback)")

    # Helper: produce modular integer 0..p-1 for a coefficient after substituting m
    def _coeff_mod_p(c_expr, m_var, mval, p):
        """
        Return integer in [0, p-1] representing c_expr(m:=mval) mod p.
        If the denominator is divisible by p (bad prime), raise ZeroDivisionError.
        If coefficient cannot be cheaply reduced, raise ValueError.
        """
        # Substitute m value if present; keep this substitution minimal
        if m_var is not None:
            try:
                val_sr = c_expr.subs({m_var: Integer(mval)})
            except Exception:
                # heavy substitution triggered; treat as non-evaluable cheaply
                raise ValueError("Substitution too heavy")
        else:
            val_sr = c_expr

        # Try to get numerator/denominator without QQ(...)
        num, den = _cheap_num_den_from_sr(val_sr)

        den_mod_p = int(den) % p
        if den_mod_p == 0:
            # denominator zero mod p => bad prime for this coefficient
            raise ZeroDivisionError("denominator divisible by p")

        # compute modular integer without constructing GF(p) element
        inv = inverse_mod(den_mod_p, p)
        val_mod = (int(num) % p) * inv % p
        return int(val_mod)

    # 1. Height score (keeps your original heuristic but cheaper on symbolic coeffs)
    height_score = 0.0
    for c_expr, _ in raw_coeffs:
        try:
            # cheap numeric check first
            if isinstance(c_expr, (Integer, int)):
                h = abs(int(c_expr))
                height_score += _int_log(h + 1)
                continue
            # try to extract numerator/denominator cheaply
            num, den = _cheap_num_den_from_sr(c_expr)
            h = max(abs(int(num)), abs(int(den)))
            height_score += _int_log(h + 1)
        except ValueError:
            # symbolic / complex coefficient: use operand count proxy
            try:
                n = c_expr.nops()
            except Exception:
                n = 1
            height_score += 1.0 + 0.1 * n
        except Exception as e:
            # unexpected problem: escalate
            raise RuntimeError(f"Unexpected error when computing height score: {e}")

    # 2. Degree info
    try:
        vars_list = fx_sr.variables()
        m_var = None
        for v in vars_list:
            if str(v) == 'm':
                m_var = v
                break

        try:
            deg_x = int(fx_sr.degree(x_var))
        except Exception:
            deg_x = 0

        if m_var is not None:
            try:
                deg_m = int(fx_sr.degree(m_var))
            except Exception:
                deg_m = 0
        else:
            deg_m = 0
    except Exception as e:
        raise RuntimeError(f"Failed to extract degree info: {e}")

    degree_penalty = _int_log(1 + deg_x) * 0.7 + _int_log(1 + deg_m) * 0.5

    # 3. Discriminant / size proxy (cheap)
    try:
        disc_score = 0.1 * fx_sr.nops()
    except Exception:
        disc_score = 1.0

    # 4. Bad prime / collision checks (optimized)
    bad_prime_count = 0
    collision_count = 0

    # Pre-generate m samples once
    sample_m_list = list(_SAMPLE_M_VALUES) + [random.randint(-17, 17) for _ in range(_NUM_RANDOM_M)]

    # Cache PolynomialRing per p
    ring_cache = {}

    # For each small prime -- try a limited number of (m) checks; cheap-fail if coefficients cannot be reduced
    for p in _SMALL_PRIMES:
        try:
            if p not in ring_cache:
                GFp = GF(p)
                R_p = PolynomialRing(GFp, 'x')
                ring_cache[p] = (GFp, R_p)
            else:
                GFp, R_p = ring_cache[p]
        except Exception as e:
            # If a prime fails to build (shouldn't happen), count as bad and continue
            bad_prime_count += 1
            continue

        p_bad = False
        collision_for_this_p = False

        # For each sampled m value, try to build fast integer-list coefficients mod p
        for mval in sample_m_list:
            # Build dense coefficient list as integers modulo p (fast Python ints)
            try:
                # determine maximum exponent to allocate list
                max_expon = max(int(exp) for (_, exp) in raw_coeffs)
            except Exception:
                max_expon = 0

            # initialize dense list of zeros
            coeffs_modp = [0] * (max_expon + 1)
            any_nonzero = False

            try:
                for c_expr, expon in raw_coeffs:
                    try:
                        val_mod = _coeff_mod_p(c_expr, m_var, mval, p)
                    except ZeroDivisionError:
                        # denominator divisible by p => prime is bad for this polynomial
                        p_bad = True
                        break
                    except ValueError:
                        # this coefficient couldn't be cheaply reduced -> treat this (p,m) check as uninterpretable
                        # Mark prime as bad-ish and break (cheap fallback). This avoids heavy QQ calls.
                        p_bad = True
                        break

                    if val_mod != 0:
                        any_nonzero = True
                        coeffs_modp[int(expon)] = int(val_mod)

                if p_bad:
                    break  # stop m-loop for this prime

                if not any_nonzero:
                    # zero polynomial mod p for this mval -> treat as collision/bad
                    collision_for_this_p = True
                    break

                # Construct polynomial in GF(p) ring (we pass list of coefficients)
                poly_x = R_p(coeffs_modp)

                # If polynomial degree 0 or negative, skip
                if poly_x.degree() <= 0:
                    # no nontrivial polynomial -> treat as no collision for this mval and continue
                    continue

                # gcd with derivative: use ring's gcd (cheap)
                try:
                    gcd_poly = poly_x.gcd(poly_x.derivative())
                    if gcd_poly.degree() > 0:
                        collision_for_this_p = True
                        break
                except Exception as e:
                    # if gcd computation fails unexpectedly, mark prime as bad and continue
                    p_bad = True
                    break

            except Exception as e:
                # Any unexpected problem in inner loop: escalate
                raise RuntimeError(f"Unexpected error during collision check for p={p}, m={mval}: {e}")

        if p_bad:
            bad_prime_count += 1
            continue

        if collision_for_this_p:
            collision_count += 1
            # continue to next prime

    max_checks = len(_SMALL_PRIMES) * max(1, len(sample_m_list))
    collision_frac = float(collision_count) / max(1, max_checks)

    total_score = (
        _WEIGHT_HEIGHT * height_score +
        _WEIGHT_DEG * degree_penalty +
        _WEIGHT_DISC * disc_score +
        _WEIGHT_BADPRIME * bad_prime_count +
        _WEIGHT_COLLISION * collision_frac * 10.0
    )

    return float(total_score)
