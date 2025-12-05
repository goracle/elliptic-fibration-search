# mumford_search.py
#
# Draft module for MUMFORD_SEARCH mode.
# This constructs local congruence conditions for the Mumford
# u,v representation on a genus-2 curve:
#
#     y^2 = f(x)
#
# with u(x) = x^2 - s x + p and
# v(x) defined by v^2 = f(x) mod u(x).
#
# The main caller will feed in (prime, residue data,
# x([n]P)(m) mod p, etc.) and we generate the small congruence
# system for s, p, and the "value of x([n]P)(m)" under the
# Mumford hypothesis.
#
# NOTHING about the CRT, reconstruction, height filtering,
# or point verification is done here.
#
# This file intentionally contains no imports.
#
# Public API:
#    build_local_mumford_equations(...)
#    reduce_mumford_system_mod_p(...)
#    solve_local_mumford_system(...)
#    assemble_mumford_residue(...)
#    mumford_local_search(...)
#
# One top-level call:
#    mumford_local_search(...)  — returns a dict (consistent or empty)

#######################################################################
# Basic helpers
#######################################################################

def _assert_is_rr(x):
    # small helper to ensure we did not get weird data
    if not (hasattr(x, "is_rational") or hasattr(x, "parent")):
        raise ValueError("invalid numeric input")
    return True


def _simple_poly_eval(coeffs, x):
    # coeffs = [c0, c1, c2, ...] meaning c0 + c1*x + c2*x^2 ...
    total = 0
    xx = 1
    for c in coeffs:
        total += c * xx
        xx = xx * x
    return total


#######################################################################
# Core local Mumford equations
#######################################################################

def build_local_mumford_equations(f_coeffs, x_residue, modulus):
    """
    Build the three local equations:

        (1) matched_x ≡ x_residue   (where matched_x = -m + x1),
            but caller will supply x_residue already reduced mod p.

        (2) u(matched_x) = 0  modulo p
            => matched_x^2 - s*matched_x + p = 0

        (3) v^2 ≡ f(matched_x) mod p, but reduced via u(x):
            f(matched_x) reduced mod (x^2 - s x + p)

    The caller provides:
        f_coeffs    list of coefficients of f(x)
        x_residue   the (mod p) value of x([n]P)(m)
        modulus     p

    Returns a dict with symbolic placeholders:
        {
            "x_residue": x_residue,
            "equation_u":  (s, p) in a polynomial relation mod p,
            "equation_v":  (s, p, v_val) in a polynomial relation mod p
        }

    No solving is done here.
    The dict is simple enough to be read by the local solver.
    """

    _assert_is_rr(modulus)
    _assert_is_rr(x_residue)

    # Equation u(x) = x^2 - s x + p = 0  (to be interpreted mod p)
    # Represent it in dictionary form.
    eq_u = ("u", x_residue)

    # Evaluate f(x_residue) mod p.
    f_at_x = _simple_poly_eval(f_coeffs, x_residue)

    # Equation v^2 = f(x_residue)  (mod p) reduced in residue class.
    # The local solver will decide how to handle sign choices.
    eq_v = ("v2", f_at_x)

    return {
        "equation_u": eq_u,
        "equation_v": eq_v,
        "x_residue": x_residue,
        "modulus": modulus
    }


#######################################################################
# Reduce system mod p
#######################################################################

def reduce_mumford_system_mod_p(system_dict):
    """
    Normalize the system_dict (returned by build_local_mumford_equations)
    by reducing each coefficient mod p.

    Returns a new dict.
    """

    p = system_dict["modulus"]
    x_res = system_dict["x_residue"]

    eq_u = system_dict["equation_u"]
    eq_v = system_dict["equation_v"]

    # eq_u = ("u", x_res)
    # eq_v = ("v2", f_at_x)

    # Reduce f(x_res) mod p
    v_rhs = eq_v[1] % p

    return {
        "modulus": p,
        "x_residue": x_res % p,
        "equation_u": ("u", x_res % p),
        "equation_v": ("v2", v_rhs)
    }


#######################################################################
# Solve the small local system
#######################################################################

def solve_local_mumford_system(reduced_system):
    """
    Very small local solver.
    We solve:

        matched_x = x_residue
        matched_x^2 - s*matched_x + p == 0   (mod p)
        v^2 == f(matched_x)                  (mod p)

    This yields congruence classes for (s, p, v).
    There can be 0, 1, or 2 solutions mod p.

    Returns a list of solutions:
        [ (s_mod_p, p_mod_p, v_mod_p), ... ]

    The calling code will feed these into the CRT layer.
    """

    p = reduced_system["modulus"]
    x0 = reduced_system["x_residue"]

    eq_v_rhs = reduced_system["equation_v"][1]

    sols = []

    # Solve s from the quadratic relation:
    #    x0^2 - s*x0 + p ≡ 0 (mod p)
    #
    # We treat p_mod_p as a free variable in F_p.
    # For each p_val in 0..p-1, solve for s_val.
    #
    # This is modest, p is small. No loops over m.

    for p_guess in range(p):
        # x0^2 - s*x0 + p_guess ≡ 0 => s ≡ (x0^2 + p_guess) * (x0)^(-1)
        if x0 % p == 0:
            # degenerate; skip
            continue
        inv_x0 = ~x0 % p
        s_val = ((x0 * x0 + p_guess) % p) * inv_x0 % p

        # Now handle v^2 ≡ eq_v_rhs mod p
        rhs = eq_v_rhs % p

        # Try v=0..p-1 such that v^2=r
        # (Small p so this is cheap.)
        for v_val in range(p):
            if (v_val * v_val) % p == rhs:
                sols.append((s_val, p_guess, v_val % p))

    return sols


#######################################################################
# Assemble residue for CRT layer
#######################################################################

def assemble_mumford_residue(prime, local_solutions):
    """
    Convert the list of local solutions into a structure suitable for
    the prime-layer CRT aggregator.

    local_solutions: list of (s_mod_p, p_mod_p, v_mod_p) for this prime.

    Returns:
        {
            prime: prime,
            "solutions": local_solutions
        }
    """

    _assert_is_rr(prime)

    return {
        "prime": prime,
        "solutions": local_solutions
    }


#######################################################################
# One top-level per-prime entry point
#######################################################################

def mumford_local_search(prime, f_coeffs, x_residue):
    """
    Single per-prime hook the main search(\*) code can call.

    Inputs:
        prime       small rational prime p
        f_coeffs    list of f(x) coefficients for y^2=f(x)
        x_residue   x([n]P)(m) mod p (already computed by the usual pipeline)

    Outputs:
        Either {} (no solutions) or a dict from assemble_mumford_residue().

    No interaction with the global MUMFORD_SEARCH here.
    The main code decides whether to call this.
    """

    _assert_is_rr(prime)
    _assert_is_rr(x_residue)

    local_system = build_local_mumford_equations(f_coeffs, x_residue, prime)
    reduced = reduce_mumford_system_mod_p(local_system)
    sols = solve_local_mumford_system(reduced)

    if len(sols) == 0:
        return {}

    return assemble_mumford_residue(prime, sols)
