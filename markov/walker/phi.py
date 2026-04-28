from __future__ import annotations
from typing import Sequence, Tuple

"""phi.py  –  markov/walker/phi.py

Construct the rational function

    φ(x, y) = A(x) + c·y,   A(x) = a₀ + a₁x + a₂x²,   c ∈ F_p

on the genus-2 hyperelliptic curve  C: y² = f(x)  (deg f = 5)
with principal divisor

    div(φ) = 2·P + 2·Q + R − 5·∞

P and Q are two F_p-rational points on C, not Weierstrass points.
R is a third F_p-rational point recovered via Vieta.

Why h(x) = f(x) − A(x)² has double roots at xP, xQ
-----------------------------------------------------
With c normalised to 1, φ(x, y) = A(x) + y and h(x) = f(x) − A(x)².

    h(xP) = 0   ←→   f(xP) = A(xP)²   ←→   yP² = A(xP)²
                ←→   A(xP) = ±yP       ←→   φ(P) = 0   [sign chosen: A(xP)=−yP]

    h'(xP) = f'(xP) − 2A(xP)A'(xP)
           = f'(xP) − 2(−yP)A'(xP)
           = f'(xP) + 2yP · A'(xP)

    Setting h'(xP) = 0:

        A'(xP) = −f'(xP) / (2yP)

    which is exactly  φ'|_C(P) = 0,  where  φ'|_C = A'(x) + f'(x)/(2y)
    is the derivative of φ along the curve  2y dy = f'(x) dx.

The four conditions (with c = 1)
---------------------------------
  (1) a₀ + a₁xP + a₂xP² + yP = 0          [φ(P) = 0]
  (2) a₁ + 2a₂xP + f'(xP)/(2yP) = 0       [φ'|_C(P) = 0  →  double root at xP]
  (3) a₀ + a₁xQ + a₂xQ² + yQ = 0          [φ(Q) = 0]
  (4) a₁ + 2a₂xQ + f'(xQ)/(2yQ) = 0       [φ'|_C(Q) = 0  →  double root at xQ]

(2) and (4) are a 2×2 system in (a₁, a₂) — solved first.
(1) then gives a₀.  (3) is a consistency check that constrains the valid
y-sign for Q: only one of (xQ, yQ) or (xQ, −yQ) will satisfy it.

Note on g_coeffs
-----------------
g_coeffs (a fiber polynomial) appears in the signature for backward
compatibility with callers that supply it, but it is NOT used in the
computation.  The correct double-zero conditions depend only on f.

Coefficient convention
-----------------------
Polynomials are lists of coefficients **low-degree first**:

    f_coeffs[i]  =  coefficient of x^i

Usage
-----
    from markov.walker.phi import compute_phi, verify_phi

    A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    checks = verify_phi(p, f_coeffs, A_coeffs, c, P, Q, R)
    assert all(checks.values()), checks
"""

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

# ---------------------------------------------------------------------------
# Main construction
# ---------------------------------------------------------------------------

def compute_phi(
    p: int,
    f_coeffs: Sequence[int],   # y² = f(x), len 6, degree-5 poly  (low first)
    g_coeffs: Sequence[int],   # UNUSED – kept for backward-compat with callers
    P: Point,                  # (x_P, y_P) – one point; double root enforced here
    Q: Point,                  # (x_Q, y_Q) – other point; double root enforced here
) -> tuple[list[int], int, Point]:
    """
    Compute φ(x,y) = A(x) + y  (c normalised to 1) such that
    div(φ) = 2P + 2Q + R − 5∞.

    Parameters
    ----------
    p         : prime characteristic
    f_coeffs  : coefficients of the curve polynomial f (low first), deg 5
    g_coeffs  : ignored (retained for API compatibility)
    P         : (x_P, y_P)  in C(F_p), not a Weierstrass point
    Q         : (x_Q, y_Q)  in C(F_p), not a Weierstrass point
                Only one y-sign for Q will be consistent; the caller is
                responsible for passing the correct branch (phi_search.py
                tries both).

    Returns
    -------
    A_coeffs  : [a0, a1, a2]  coefficients of A(x) = a0 + a1·x + a2·x²
    c         : 1  (always, by normalisation)
    R         : (x_R, y_R)   third zero of φ on C, from Vieta

    Raises
    ------
    ValueError        – degenerate geometry (same x-coord, Weierstrass pts,
                        or consistency check failed for chosen y-sign of Q)
    ArithmeticError   – Vieta candidate R is not on the curve (should not
                        happen when the consistency check passes)
    """
    f  = [int(v) % p for v in f_coeffs]
    fp = _poly_deriv(f, p)

    xP, yP = int(P[0]) % p, int(P[1]) % p
    xQ, yQ = int(Q[0]) % p, int(Q[1]) % p

    if yP == 0 or yQ == 0:
        raise ValueError(
            "compute_phi: P or Q is a Weierstrass point (y=0); "
            "φ'|_C = A'(x) + f'(x)/(2y) is undefined there."
        )

    if xP == xQ:
        raise ValueError(
            "compute_phi: P and Q share an x-coordinate; the 2×2 system "
            "for (a₁, a₂) is singular."
        )

    inv2  = _modinv(2, p)
    invyP = _modinv(yP, p)
    invyQ = _modinv(yQ, p)

    # φ'|_C at P and Q (must equal zero for double roots).
    # fslope_X  =  f'(xX) / (2·yX)
    fslope_P = _poly_eval(fp, xP, p) * invyP % p * inv2 % p
    fslope_Q = _poly_eval(fp, xQ, p) * invyQ % p * inv2 % p

    # -----------------------------------------------------------------------
    # Step 1: solve 2×2 for (a₁, a₂) from conditions (2) and (4).
    #
    #   a₁ + 2a₂xP = −fslope_P
    #   a₁ + 2a₂xQ = −fslope_Q
    #
    # Subtract → 2a₂(xP − xQ) = −fslope_P + fslope_Q
    # -----------------------------------------------------------------------
    two_xdiff = (2 * (xP - xQ)) % p
    a2 = (fslope_Q - fslope_P) % p * _modinv(two_xdiff, p) % p
    a1 = (-fslope_P - 2 * a2 * xP) % p

    # Step 2: a₀ from condition (1).
    xP2 = xP * xP % p
    a0  = (-yP - a1 * xP - a2 * xP2) % p

    # c is normalised to 1.
    c = 1
    A_coeffs = [a0, a1, a2]

    # Step 3: consistency check — condition (3) must hold for the chosen
    # y-sign of Q.  If it fails the caller should try Q = (xQ, p−yQ).
    xQ2   = xQ * xQ % p
    check = (a0 + a1 * xQ + a2 * xQ2 + yQ) % p
    if check != 0:
        raise ValueError(
            f"compute_phi: consistency check φ(Q)=0 failed (residue={check}). "
            "The y-sign of Q is wrong — try Q = (x_Q, p − y_Q)."
        )

    # -----------------------------------------------------------------------
    # Step 4: recover R via Vieta on  h(x) = f(x) − A(x)²  (c²=1).
    #
    # h has degree 5:
    #   leading coeff    = f[5]       (from f, degree 5)
    #   coeff of x⁴     = f[4] − a2²
    #
    # With divisor 2P + 2Q + R, the five roots are xP, xP, xQ, xQ, xR:
    #   sum of roots  = 2xP + 2xQ + xR  =  −coeff(x⁴) / coeff(x⁵)
    # -----------------------------------------------------------------------
    f4   = f[4] if len(f) > 4 else 0
    f5   = f[5] if len(f) > 5 else 1    # leading coeff (1 for monic)
    a2sq = a2 * a2 % p

    # sum_roots = -(f4 - a2sq) / f5  =  (a2sq - f4) / f5
    sum_roots = (a2sq - f4) % p * _modinv(f5, p) % p
    xR = (sum_roots - 2 * xP - 2 * xQ) % p

    # y_R from φ(R) = 0:  A(xR) + yR = 0  →  yR = −A(xR)
    yR = (-_poly_eval(A_coeffs, xR, p)) % p

    # On-curve check — validates the entire construction.
    yR_sq = yR * yR % p
    fR    = _poly_eval(f, xR, p)
    if yR_sq != fR:
        raise ArithmeticError(
            f"compute_phi: Vieta candidate R = ({xR}, {yR}) is not on the curve "
            f"(y²={yR_sq}, f(x_R)={fR}).  "
            "This should not happen when the consistency check passed; "
            "verify that P and Q are distinct F_p-points on C with the correct y-signs."
        )

    return A_coeffs, c, (xR, yR)

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def verify_phi(
    p: int,
    f_coeffs: Sequence[int],
    A_coeffs: Sequence[int],
    c: int,
    P: Point,
    Q: Point,
    R: Point,
    # g_coeffs is accepted but ignored (backward compat).
    g_coeffs: Sequence[int] | None = None,
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

    double_zero_P, double_zero_Q
        h(x) = c²f(x) − A(x)² has h(xP)=h'(xP)=0 and similarly for Q.

    dphi_curve_P_zero, dphi_curve_Q_zero
        φ'|_C = A'(x) + c·f'(x)/(2y) vanishes at P and Q.
        This is the condition that forces the double zeros.
    """
    f  = [int(v) % p for v in f_coeffs]
    A  = [int(v) % p for v in A_coeffs]
    fp = _poly_deriv(f, p)
    Ap = _poly_deriv(A, p)
    inv2 = _modinv(2, p)

    def phi(pt: Point) -> int:
        x, y = int(pt[0]) % p, int(pt[1]) % p
        return (_poly_eval(A, x, p) + c * y) % p

    def dphi_curve(pt: Point) -> int:
        """φ'|_C = A'(x) + c·f'(x)/(2y)."""
        x, y = int(pt[0]) % p, int(pt[1]) % p
        return (_poly_eval(Ap, x, p) + c * _poly_eval(fp, x, p) * _modinv(y, p) % p * inv2) % p

    def h_val(x: int) -> int:
        """h(x) = c²f(x) − A(x)²."""
        c2 = c * c % p
        return (c2 * _poly_eval(f, x, p) - pow(_poly_eval(A, x, p), 2, p)) % p

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
        # Double-zero structure: h(xP)=h'(xP)=0, same for Q.
        "double_zero_P":  h_val(xP) == 0 and h_deriv_val(xP) == 0,
        "double_zero_Q":  h_val(xQ) == 0 and h_deriv_val(xQ) == 0,
        # φ'|_C = 0 at P and Q — the direct condition for double zeros.
        "dphi_curve_P_zero":  dphi_curve(P) == 0,
        "dphi_curve_Q_zero":  dphi_curve(Q) == 0,
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

    With c=1 this is simply  f(x) − A(x)².  The zeros of h on F_p are the
    x-coordinates of the zeros of φ on the curve.  Under the claimed divisor
    2P+2Q+R−5∞, h factors as  (x−xP)²·(x−xQ)²·(x−xR)·f[5]  over F_p.
    """
    f = [int(v) % p for v in f_coeffs]
    A = [int(v) % p for v in A_coeffs]
    c2 = c * c % p

    # c²·f(x)
    cf = [c2 * fi % p for fi in f]

    # A(x)²: convolve A with itself.
    deg_A = len(A) - 1
    A2 = [0] * (2 * deg_A + 1)
    for i, ai in enumerate(A):
        for j, aj in enumerate(A):
            A2[i + j] = (A2[i + j] + ai * aj) % p

    # h = c²f − A²: zero-pad to the same length.
    n = max(len(cf), len(A2))
    cf = cf + [0] * (n - len(cf))
    A2 = A2 + [0] * (n - len(A2))
    h  = [(cf[i] - A2[i]) % p for i in range(n)]

    return h
