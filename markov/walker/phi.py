"""phi.py  –  markov/walker/phi.py

Construct the rational function

    φ(x, y) = A(x) + c·y,   A(x) = a₀ + a₁x + a₂x²,   c ∈ F_p

on the genus-2 hyperelliptic curve  C: y² = f(x)  (deg f = 5)
that is "adapted" to a fiber  F: y² = g(x)  (deg g = 4)  at two
intersection points P (double tangency) and Q (simple tangency).

Expected principal divisor
--------------------------
    div(φ) = 2·P + 2·Q + R − 5·∞

which has degree 0 and is therefore principal.  R is a third F_p-rational
point, recovered from the remaining root of c²f(x) − A(x)² via Vieta.

The four interpolation conditions
-----------------------------------
Let φ'|_C denote the derivative of φ along C under  2y dy = f'(x) dx:

    φ'|_C = A'(x) + c · f'(x) / (2y)

and let  y'_F(x) = g'(x) / (2y)  be the slope of the fiber branch.

  (1)  φ(x_P, y_P) = 0        ← φ vanishes at P
  (2)  φ(x_Q, y_Q) = 0        ← φ vanishes at Q
  (3)  φ'|_C (P) = y'_F(P)    ← zero-locus of φ is tangent to F at P
  (4)  φ'|_C (Q) = y'_F(Q)    ← zero-locus of φ is tangent to F at Q

These are 4 linear equations in (a₀, a₁, a₂, c) ∈ F_p⁴.

Coefficient convention
-----------------------
Polynomials are represented as lists of coefficients **low-degree first**:

    f_coeffs[i]  =  coefficient of x^i

so  f_coeffs = [f0, f1, f2, f3, f4, f5]  for a degree-5 poly.

Usage
-----
    from markov.walker.phi import compute_phi, verify_phi

    A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    checks = verify_phi(p, f_coeffs, g_coeffs, A_coeffs, c, P, Q, R)
    assert all(checks.values()), checks
"""

from __future__ import annotations
from typing import Sequence, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Fp      = int
Point   = Tuple[Fp, Fp]


# ---------------------------------------------------------------------------
# Low-level modular helpers
# ---------------------------------------------------------------------------

def _poly_eval(coeffs: Sequence[int], x: int, p: int) -> int:
    """Horner evaluation of polynomial (low-degree first) at x mod p."""
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % p
    return result


def _poly_deriv(coeffs: Sequence[int], p: int) -> list[int]:
    """Formal derivative of a polynomial given coefficient-low-first."""
    return [(i * coeffs[i]) % p for i in range(1, len(coeffs))]


def _modinv(a: int, p: int) -> int:
    """Modular inverse via Fermat (p must be prime)."""
    a = a % p
    if a == 0:
        raise ZeroDivisionError(f"modinv: zero argument mod {p}")
    return pow(a, p - 2, p)


def _solve_4x4(M: list[list[int]], b: list[int], p: int) -> list[int]:
    """
    Solve M·x = b over F_p by Gauss-Jordan elimination.
    M is a 4×4 list-of-lists; b is length-4.  Returns [x0, x1, x2, x3].
    Raises ValueError if the system is singular.
    """
    # Work on augmented matrix [M | b], all entries mod p.
    aug = [[int(M[i][j]) % p for j in range(4)] + [int(b[i]) % p]
           for i in range(4)]

    for col in range(4):
        # Partial pivot.
        pivot = next(
            (row for row in range(col, 4) if aug[row][col] % p != 0),
            None,
        )
        if pivot is None:
            raise ValueError(
                f"compute_phi: singular linear system at column {col}; "
                "P and Q may coincide or be degenerate."
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]

        # Scale pivot row so diagonal is 1.
        inv = _modinv(aug[col][col], p)
        aug[col] = [(v * inv) % p for v in aug[col]]

        # Eliminate column in all other rows.
        for row in range(4):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col]
                aug[row] = [(aug[row][j] - factor * aug[col][j]) % p
                            for j in range(5)]

    return [aug[i][4] for i in range(4)]


# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

def compute_phi(
    p: int,
    f_coeffs: Sequence[int],   # y² = f(x), len 6, degree-5 poly  (low first)
    g_coeffs: Sequence[int],   # fiber y² = g(x), len 5, degree-4 (low first)
    P: Point,                  # (x_P, y_P) — double-tangency intersection
    Q: Point,                  # (x_Q, y_Q) — single-tangency intersection
) -> tuple[list[int], int, Point]:
    """
    Compute the rational function φ(x,y) = A(x) + c·y adapted to the fiber
    at P (double tangency) and Q (single tangency).

    Parameters
    ----------
    p         : prime characteristic
    f_coeffs  : coefficients of the curve polynomial f (low first), deg 5
    g_coeffs  : coefficients of the fiber polynomial g (low first), deg ≤ 4
    P         : (x_P, y_P)  — double-tangency point in C ∩ F
    Q         : (x_Q, y_Q)  — single-tangency point in C ∩ F

    Returns
    -------
    A_coeffs  : [a0, a1, a2]  coefficients of A(x) = a0 + a1·x + a2·x²
    c         : scalar in F_p
    R         : (x_R, y_R)   third zero of φ on C, from Vieta + φ=0
    """
    f  = [int(v) % p for v in f_coeffs]
    g  = [int(v) % p for v in g_coeffs]
    fp = _poly_deriv(f, p)
    gp = _poly_deriv(g, p)

    xP, yP = int(P[0]) % p, int(P[1]) % p
    xQ, yQ = int(Q[0]) % p, int(Q[1]) % p

    if yP == 0 or yQ == 0:
        raise ValueError(
            "compute_phi: P or Q is a Weierstrass point (y=0); "
            "the fiber-slope formula 1/(2y) is undefined there."
        )

    inv2  = _modinv(2, p)
    invyP = _modinv(yP, p)
    invyQ = _modinv(yQ, p)

    # RHS of conditions (3) and (4): g'(x)/(2y)  at P and Q.
    rhs_P = _poly_eval(gp, xP, p) * invyP % p * inv2 % p
    rhs_Q = _poly_eval(gp, xQ, p) * invyQ % p * inv2 % p

    # f'(x)/(2y) at P and Q: the f-slope contribution in φ'|_C.
    fslope_P = _poly_eval(fp, xP, p) * invyP % p * inv2 % p
    fslope_Q = _poly_eval(fp, xQ, p) * invyQ % p * inv2 % p

    # Linear system in unknowns [a0, a1, a2, c]:
    #
    #  (1) a0 + a1·xP + a2·xP² + c·yP = 0
    #  (2) a0 + a1·xQ + a2·xQ² + c·yQ = 0
    #  (3)      a1 + 2·a2·xP  + c·fslope_P = rhs_P
    #  (4)      a1 + 2·a2·xQ  + c·fslope_Q = rhs_Q

    xP2 = xP * xP % p
    xQ2 = xQ * xQ % p

    M = [
        [1,  xP,  xP2,           yP      ],
        [1,  xQ,  xQ2,           yQ      ],
        [0,  1,   2 * xP % p,    fslope_P],
        [0,  1,   2 * xQ % p,    fslope_Q],
    ]
    b = [0, 0, rhs_P, rhs_Q]

    a0, a1, a2, c = _solve_4x4(M, b, p)
    A_coeffs = [a0, a1, a2]

    if c == 0:
        raise ValueError(
            "compute_phi: solved c = 0, φ degenerates to a polynomial in x only. "
            "Try different P, Q or check that g ≠ f."
        )

    # -----------------------------------------------------------------------
    # Recover R via Vieta on  h(x) = c²·f(x) − A(x)²
    #
    # h has degree 5:
    #   coeff(x⁵) = c²·f[5]      (leading coeff of f, usually 1 if monic)
    #   coeff(x⁴) = c²·f[4] − a2²
    #
    # Sum of all 5 roots = −coeff(x⁴) / coeff(x⁵)
    # Under the claimed divisor 2P + 2Q + R: sum = 2·xP + 2·xQ + xR
    # -----------------------------------------------------------------------
    c2   = c * c % p
    f4   = f[4] if len(f) > 4 else 0
    f5   = f[5] if len(f) > 5 else 1    # leading coeff (1 for monic)
    a2sq = a2 * a2 % p

    sum_roots = (-(c2 * f4 - a2sq) % p) * _modinv(c2 * f5 % p, p) % p
    xR = (sum_roots - 2 * xP - 2 * xQ) % p

    # y_R from φ(R) = 0:  y_R = −A(x_R) / c
    yR = (-_poly_eval(A_coeffs, xR, p) % p) * _modinv(c, p) % p

    # Quick on-curve check — this also validates the Vieta step.
    yR_sq = yR * yR % p
    fR    = _poly_eval(f, xR, p)
    if yR_sq != fR:
        raise ArithmeticError(
            f"compute_phi: Vieta candidate R = ({xR}, {yR}) is not on the curve "
            f"(y²={yR_sq}, f(x_R)={fR}).  "
            "The claimed double-zero structure at P and Q may not hold — "
            "check the fiber/curve intersection multiplicities."
        )

    return A_coeffs, c, (xR, yR)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_phi(
    p: int,
    f_coeffs: Sequence[int],
    g_coeffs: Sequence[int],
    A_coeffs: Sequence[int],
    c: int,
    P: Point,
    Q: Point,
    R: Point,
) -> dict[str, bool]:
    """
    Sanity checks for the constructed φ and the claimed divisor 2P+2Q+R−5∞.

    Returns a dict of {check_name: bool}.  All should be True.

    Checks
    ------
    phi_P_zero, phi_Q_zero, phi_R_zero
        φ vanishes at all three points.

    P_on_curve, Q_on_curve, R_on_curve
        All three points lie on  y² = f(x).

    P_on_fiber, Q_on_fiber
        P and Q lie on  y² = g(x).

    double_zero_P, double_zero_Q
        x_P (resp. x_Q) is a double root of  h(x) = c²f(x) − A(x)².
        This is the condition that enforces multiplicity-2 zeros.
        NOTE: the interpolation conditions guarantee φ(P)=φ(Q)=0 but
        do NOT automatically enforce h'(x_P)=0 unless the fiber slope
        happens to coincide with the curve slope at P.  Check this to
        confirm the divisor structure.

    dphi_curve_P_eq_fiber_slope_P, dphi_curve_Q_eq_fiber_slope_Q
        The φ'|_C interpolation conditions were met.
    """
    f  = [int(v) % p for v in f_coeffs]
    g  = [int(v) % p for v in g_coeffs]
    A  = [int(v) % p for v in A_coeffs]
    fp = _poly_deriv(f, p)
    gp = _poly_deriv(g, p)
    Ap = _poly_deriv(A, p)
    inv2 = _modinv(2, p)

    def phi(pt: Point) -> int:
        x, y = int(pt[0]) % p, int(pt[1]) % p
        return (_poly_eval(A, x, p) + c * y) % p

    def dphi_curve(pt: Point) -> int:
        """φ'|_C = A'(x) + c·f'(x)/(2y)."""
        x, y = int(pt[0]) % p, int(pt[1]) % p
        return (_poly_eval(Ap, x, p) + c * _poly_eval(fp, x, p) % p * _modinv(y, p) % p * inv2) % p

    def fiber_slope(pt: Point) -> int:
        """y'_F = g'(x)/(2y)."""
        x, y = int(pt[0]) % p, int(pt[1]) % p
        return _poly_eval(gp, x, p) * _modinv(y, p) % p * inv2 % p

    def h_val(x: int) -> int:
        """h(x) = c²f(x) − A(x)²."""
        c2 = c * c % p
        return (c2 * _poly_eval(f, x, p) - _poly_eval(A, x, p) ** 2) % p

    def h_deriv_val(x: int) -> int:
        """h'(x) = c²f'(x) − 2A(x)A'(x)."""
        c2  = c * c % p
        Ax  = _poly_eval(A, x, p)
        Apx = _poly_eval(Ap, x, p)
        fpx = _poly_eval(fp, x, p)
        return (c2 * fpx - 2 * Ax * Apx) % p

    xP, yP = int(P[0]) % p, int(P[1]) % p
    xQ, yQ = int(Q[0]) % p, int(Q[1]) % p

    return {
        # φ vanishes at all three zeros.
        "phi_P_zero":  phi(P) == 0,
        "phi_Q_zero":  phi(Q) == 0,
        "phi_R_zero":  phi(R) == 0,
        # All points on the curve.
        "P_on_curve":  pow(yP, 2, p) == _poly_eval(f, xP, p),
        "Q_on_curve":  pow(yQ, 2, p) == _poly_eval(f, xQ, p),
        "R_on_curve":  pow(int(R[1]), 2, p) == _poly_eval(f, int(R[0]) % p, p),
        # P and Q on the fiber.
        "P_on_fiber":  pow(yP, 2, p) == _poly_eval(g, xP, p),
        "Q_on_fiber":  pow(yQ, 2, p) == _poly_eval(g, xQ, p),
        # Double-zero structure — h(x_P) = h'(x_P) = 0, same for Q.
        "double_zero_P":  h_val(xP) == 0 and h_deriv_val(xP) == 0,
        "double_zero_Q":  h_val(xQ) == 0 and h_deriv_val(xQ) == 0,
        # Interpolation conditions were met.
        "dphi_curve_P_eq_fiber_slope_P":
            dphi_curve(P) == fiber_slope(P),
        "dphi_curve_Q_eq_fiber_slope_Q":
            dphi_curve(Q) == fiber_slope(Q),
    }


# ---------------------------------------------------------------------------
# Convenience: evaluate φ at a point
# ---------------------------------------------------------------------------

def phi_eval(A_coeffs: Sequence[int], c: int, pt: Point, p: int) -> int:
    """Evaluate φ(x, y) = A(x) + c·y at pt over F_p."""
    x, y = int(pt[0]) % p, int(pt[1]) % p
    return (_poly_eval([int(a) % p for a in A_coeffs], x, p) + c * y) % p


# ---------------------------------------------------------------------------
# Convenience: recover the full quintic h(x) = c²f(x) − A(x)²
# ---------------------------------------------------------------------------

def phi_quintic(
    p: int,
    f_coeffs: Sequence[int],
    A_coeffs: Sequence[int],
    c: int,
) -> list[int]:
    """
    Return the coefficients of  h(x) = c²·f(x) − A(x)²  (low-degree first).

    The zeros of h on F_p are the x-coordinates of the zeros of φ on the curve
    (each zero of φ appears with the same multiplicity as the corresponding
    root of h).  Should factor as  (x−xP)²·(x−xQ)²·(x−xR) · c² (up to leading
    coefficient) under the claimed divisor 2P+2Q+R−5∞.
    """
    f = [int(v) % p for v in f_coeffs]
    A = [int(v) % p for v in A_coeffs]
    c2 = c * c % p

    # c²·f(x): multiply each coefficient by c².
    cf = [c2 * fi % p for fi in f]

    # A(x)²: convolve A with itself.
    deg_A = len(A) - 1
    A2 = [0] * (2 * deg_A + 1)
    for i, ai in enumerate(A):
        for j, aj in enumerate(A):
            A2[i + j] = (A2[i + j] + ai * aj) % p

    # h = c²f − A²: zero-pad to the same length.
    n = max(len(cf), len(A2))
    cf  = cf  + [0] * (n - len(cf))
    A2  = A2  + [0] * (n - len(A2))
    h   = [(cf[i] - A2[i]) % p for i in range(n)]

    return h
