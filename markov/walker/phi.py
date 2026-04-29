"""phi.py  –  markov/walker/phi.py

Construct the rational function

    φ(x, y) = A(x) + c·y,   c ∈ F_p

on the genus-2 hyperelliptic curve  C: y² = f(x)  (deg f = 5).

Four geometries are supported, dispatched automatically by compute_phi:

┌──────────────────────────┬──────────┬──────────────────────────────┬──────────────┐
│ Geometry                 │ deg A(x) │ div(φ)                       │ Extra zeros  │
├──────────────────────────┼──────────┼──────────────────────────────┼──────────────┤
│ Generic  P ≠ Q           │    2     │ 2P + 2Q + R   − 5∞           │ R (1 point)  │
│ Conjugate  Q = (xP,−yP)  │    2     │ 4P            + R   − 5∞     │ R (1 point)  │
│ Self  P = Q              │    3     │ 4P + R + S    − 6∞           │ R,S (Mumford)│
│ Self-conjugate (y=0)     │    –     │ (degenerate; rejected)       │ –            │
└──────────────────────────┴──────────┴──────────────────────────────┴──────────────┘

Generic case (P ≠ Q, xP ≠ xQ)
-------------------------------
A(x) = a₀ + a₁x + a₂x², c = 1.
h(x) = f(x) − A(x)² has degree 5 and roots xP(×2), xQ(×2), xR(×1).
Four conditions:

  (1) A(xP) = −yP                     [φ(P) = 0]
  (2) A'(xP) = −f'(xP)/(2yP)         [double root at xP]
  (3) A(xQ) = −yQ                     [φ(Q) = 0, consistency check for y-sign]
  (4) A'(xQ) = −f'(xQ)/(2yQ)         [double root at xQ]

(2)+(4) → 2×2 system for (a₁,a₂); (1) → a₀; (3) → sign check on yQ.
Vieta: xR = −(f[4]−a₂²)/f[5] − 2xP − 2xQ.

Conjugate case (xP = xQ, yQ = −yP)
-------------------------------------
A(x) quadratic, c = 1.  div(φ) = 4P + R − 5∞.
Uses h=h'=h''=0 at xP (3 conditions → a₀,a₁,a₂).
Vieta: xR = −(f[4]−a₂²)/f[5] − 4xP.
(Implemented in _compute_phi_conjugate.)

Self case (P = Q exactly, same x and y)
-----------------------------------------
A(x) = a₀ + a₁x + a₂x² + a₃x³, c = 1.
h(x) = f(x) − A(x)² has degree 6 and roots xP(×4), xR, xS.
div(φ) = 4P + R + S − 6∞.
Four conditions (h=h'=h''=h'''=0 at xP):

  A(xP)   = −yP
  A'(xP)  = s  := −f'(xP)/(2yP)
  A''(xP) = d2 := (2s² − f''(xP)) / (2yP)
  A'''(xP)= d3 := (6s·d2 − f'''(xP)) / (2yP)

Then:
  a₃ = d3/6
  a₂ = (A''(xP) − 6a₃xP) / 2 = (d2 − 6a₃xP) / 2
  a₁ = s − 2a₂xP − 3a₃xP²
  a₀ = −yP − a₁xP − a₂xP² − a₃xP³

Vieta for the degree-6 h (leading coeff −a₃²):
  σ₁ = (f[5] − 2a₂a₃) / a₃²          [sum of all 6 roots]
  σ₂ = −(f[4] − a₂² − 2a₁a₃) / a₃²  [sum of all pairs]

  xR + xS = σ₁ − 4xP
  xR · xS = σ₂ − 6xP² − 4xP(xR+xS)

Returns R = (xR+xS, xR*xS) as a Mumford u-polynomial pair (not a single point).
The caller must factor u(x) = x² − (xR+xS)x + xR*xS over F_p to get individual
x-coordinates, which may or may not split.

Note on g_coeffs
-----------------
g_coeffs appears in the signature for backward compatibility but is NOT used.

Coefficient convention
-----------------------
Polynomials are lists of coefficients **low-degree first**:

    f_coeffs[i]  =  coefficient of x^i

Usage
-----
    from markov.walker.phi import compute_phi, verify_phi

    A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)

    # Generic/conjugate: R is a single (xR, yR) point.
    # Self (P=Q):        R is a Mumford pair ((sum, prod), None) — check with
    #                    isinstance(R[0], tuple).

    checks = verify_phi(p, f_coeffs, A_coeffs, c, P, Q, R)
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
        if yP == yQ:
            # P = Q exactly: degree-3 A, div(φ) = 4P + R + S − 6∞.
            # R is returned as a Mumford pair ((xR+xS, xR*xS), None).
            return _compute_phi_self(p, f, fp, xP, yP)
        # Q is the hyperelliptic conjugate of P: Q = (xP, −yP).
        # The standard 2×2 system is singular; use the conjugate branch instead.
        return _compute_phi_conjugate(p, f, fp, xP, yP)

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


def _compute_phi_self(
    p: int,
    f: list[int],    # already reduced mod p, low-degree first, deg 5
    fp: list[int],   # formal derivative of f
    xP: int,
    yP: int,
) -> tuple[list[int], int, tuple]:
    """Self case of compute_phi: P = Q (identical points).

    div(φ) = 4·P + R + S − 6·∞,  where R, S are the two additional zeros.

    A(x) = a₀ + a₁x + a₂x² + a₃x³  (degree 3, c = 1).
    h(x) = f(x) − A(x)² has degree 6 with a degree-4 zero at xP.

    The four conditions h(xP) = h'(xP) = h''(xP) = h'''(xP) = 0 determine
    all four coefficients of A uniquely.

    Returns
    -------
    A_coeffs : [a₀, a₁, a₂, a₃]
    c        : 1
    R        : ((sum_RS, prod_RS), None)
        A Mumford-style pair encoding the quadratic factor
            u(x) = x² − sum_RS·x + prod_RS
        whose roots are xR and xS (the two extra zeros of φ).
        The caller is responsible for factoring u over F_p to obtain
        individual x-coordinates (they may not be rational).

    Raises
    ------
    ZeroDivisionError  – yP = 0 (Weierstrass point) or a₃ = 0 (degenerate)
    ArithmeticError    – h does not actually have a degree-4 zero at xP
                         (indicates a bug; should not occur)
    """
    inv2  = _modinv(2, p)
    inv4  = _modinv(4, p)   # used as 1/4 = 1/2 · 1/2
    inv6  = _modinv(6, p)
    invyP = _modinv(yP, p)  # raises ZeroDivisionError if yP == 0

    fpp  = _poly_deriv(fp, p)   # f''
    fppp = _poly_deriv(fpp, p)  # f'''

    # s  = A'(xP)  = −f'(xP)/(2yP)
    s = (-_poly_eval(fp, xP, p)) % p * invyP % p * inv2 % p

    # d2 = A''(xP)
    #   h''(xP) = f''(xP) − 2(s² + A(xP)·A''(xP)) = 0
    #   with A(xP) = −yP:
    #     f''(xP) − 2s² + 2yP·A''(xP) = 0
    #     A''(xP) = (2s² − f''(xP)) / (2yP)
    f2xP = _poly_eval(fpp, xP, p)
    d2   = (2 * s % p * s - f2xP) % p * invyP % p * inv2 % p

    # d3 = A'''(xP)
    #   h'''(xP) = f'''(xP) − 2(3·A'·A'' + A·A''') = 0
    #   with A(xP) = −yP:
    #     f'''(xP) − 6s·d2 + 2yP·A'''(xP) = 0
    #     A'''(xP) = (6s·d2 − f'''(xP)) / (2yP)
    f3xP = _poly_eval(fppp, xP, p)
    d3   = (6 * s % p * d2 - f3xP) % p * invyP % p * inv2 % p

    # Recover A coefficients from Taylor data at xP.
    # A(x) = a₀ + a₁x + a₂x² + a₃x³
    # A'''(x) = 6a₃  →  a₃ = d3/6
    a3 = d3 * inv6 % p

    if a3 == 0:
        raise ZeroDivisionError(
            "compute_phi self: a₃ = 0 (degenerate self-divisor; "
            "φ collapses to the conjugate geometry)."
        )

    # A''(x) = 2a₂ + 6a₃x  →  A''(xP) = 2a₂ + 6a₃xP = d2
    #   a₂ = (d2 − 6a₃xP) / 2
    a2 = (d2 - 6 * a3 % p * xP) % p * inv2 % p

    # A'(xP) = a₁ + 2a₂xP + 3a₃xP² = s
    xP2 = xP * xP % p
    a1  = (s - 2 * a2 * xP - 3 * a3 % p * xP2) % p

    # A(xP) = a₀ + a₁xP + a₂xP² + a₃xP³ = −yP
    xP3 = xP2 * xP % p
    a0  = (-yP - a1 * xP - a2 * xP2 - a3 * xP3) % p

    A_coeffs = [a0, a1, a2, a3]

    # -----------------------------------------------------------------------
    # Vieta for h(x) = f(x) − A(x)²  (degree 6, leading coeff = −a₃²).
    #
    # Coefficients of A²:
    #   x⁶: a₃²
    #   x⁵: 2·a₂·a₃
    #   x⁴: a₂² + 2·a₁·a₃
    #
    # h = f − A²:
    #   coeff(x⁶) = −a₃²
    #   coeff(x⁵) = f[5] − 2·a₂·a₃
    #   coeff(x⁴) = f[4] − a₂² − 2·a₁·a₃
    #
    # Vieta (roots r₁…r₆, leading coeff L = −a₃²):
    #   Σrᵢ        = −coeff(x⁵)/L = (f[5] − 2a₂a₃) / a₃²        [= σ₁]
    #   Σᵢ<ⱼ rᵢrⱼ = coeff(x⁴)/L  = −(f[4] − a₂² − 2a₁a₃) / a₃² [= σ₂]
    #
    # The six roots are xP(×4), xR, xS, so:
    #   σ₁ = 4xP + (xR+xS)          →  xR+xS = σ₁ − 4xP
    #   σ₂ = 6xP² + 4xP(xR+xS) + xR·xS  →  xR·xS = σ₂ − 6xP² − 4xP(xR+xS)
    # -----------------------------------------------------------------------
    f5 = f[5] if len(f) > 5 else 1
    f4 = f[4] if len(f) > 4 else 0

    a3sq = a3 * a3 % p
    inv_a3sq = _modinv(a3sq, p)

    sigma1 = (f5 - 2 * a2 % p * a3) % p * inv_a3sq % p
    sigma2 = (-(f4 - a2 * a2 % p - 2 * a1 % p * a3)) % p * inv_a3sq % p

    sum_RS  = (sigma1 - 4 * xP) % p
    prod_RS = (sigma2 - 6 * xP2 - 4 * xP % p * sum_RS) % p

    # Sanity: verify h has a degree-4 zero at xP by checking h=h'=h''=h'''=0.
    def _h_derivs_at_xP(A, f_poly, x, mod):
        """Returns (h, h', h'', h''') at x for h = f − A²."""
        Ax   = _poly_eval(A, x, mod)
        Apx  = _poly_eval(_poly_deriv(A, mod), x, mod)
        fx   = _poly_eval(f_poly, x, mod)
        fpx  = _poly_eval(_poly_deriv(f_poly, mod), x, mod)
        h0   = (fx - Ax * Ax) % mod
        h1   = (fpx - 2 * Ax % mod * Apx) % mod
        A2   = _poly_deriv(A, mod)
        A3   = _poly_deriv(A2, mod)
        A4   = _poly_deriv(A3, mod)
        f2   = _poly_deriv(_poly_deriv(f_poly, mod), mod)
        f3   = _poly_deriv(f2, mod)
        Appx = _poly_eval(A2, x, mod)
        Apppx= _poly_eval(A3, x, mod)
        f2x  = _poly_eval(f2, x, mod)
        f3x  = _poly_eval(f3, x, mod)
        h2   = (f2x - 2 * (Apx * Apx % mod + Ax * Appx % mod)) % mod
        h3   = (f3x - 2 * (3 * Apx % mod * Appx % mod + Ax * Apppx % mod)) % mod
        return h0, h1, h2, h3

    h0, h1, h2, h3 = _h_derivs_at_xP(A_coeffs, f, xP, p)
    if any(v != 0 for v in (h0, h1, h2, h3)):
        raise ArithmeticError(
            f"compute_phi self: degree-4 zero verification failed at xP={xP}: "
            f"h={h0}, h'={h1}, h''={h2}, h'''={h3}.  "
            "This is a bug in _compute_phi_self."
        )

    return A_coeffs, 1, ((sum_RS, prod_RS), None)


def _compute_phi_conjugate(
    p: int,
    f: list[int],    # already reduced mod p, low-degree first
    fp: list[int],   # formal derivative of f, low-degree first
    xP: int,
    yP: int,
) -> tuple[list[int], int, tuple[int, int]]:
    """Conjugate branch of compute_phi: Q = (xP, −yP).

    div(φ) = 4·P_eff + R − 5·∞  where P_eff is one of the two points above xP.

    h(x) = f(x) − A(x)² has a degree-4 root at xP, so we need:
        h(xP)   = 0   →  A(xP)   = −yP           (φ(P) = 0, sign chosen)
        h'(xP)  = 0   →  A'(xP)  = −f'(xP)/(2yP)
        h''(xP) = 0   →  solves for a₂

    h''(x) = f''(x) − 2(A'(x)² + A(x)·A''(x))

    At xP with A(xP) = −yP, A'(xP) = s := −f'(xP)/(2yP), A''(xP) = 2a₂:

        h''(xP) = f''(xP) − 2(s² + (−yP)·2a₂) = 0
        f''(xP) − 2s² + 4yP·a₂ = 0
        a₂ = (2s² − f''(xP)) / (4yP)

    Then:
        a₁ = s − 2a₂xP       [from A'(xP) = a₁ + 2a₂xP = s]
        a₀ = −yP − a₁xP − a₂xP²

    Vieta: five roots are xP×4 + xR, sum = −(f[4]−a₂²)/f[5]:
        xR = −(f[4]−a₂²)/f[5] − 4xP
    """
    inv2 = _modinv(2, p)
    inv4 = _modinv(4, p)
    invyP = _modinv(yP, p)

    fpp = _poly_deriv(fp, p)   # second derivative of f

    # s = A'(xP) = −f'(xP)/(2yP)
    s = (-_poly_eval(fp, xP, p) % p) * invyP % p * inv2 % p

    # a₂ from h''(xP) = 0
    f2xP = _poly_eval(fpp, xP, p)
    # a₂ = (2s² − f''(xP)) / (4yP)
    a2 = (2 * s % p * s % p - f2xP) % p * inv4 % p * invyP % p

    # a₁ from A'(xP) = s
    a1 = (s - 2 * a2 * xP) % p

    # a₀ from A(xP) = −yP
    xP2 = xP * xP % p
    a0  = (-yP - a1 * xP - a2 * xP2) % p

    c = 1
    A_coeffs = [a0, a1, a2]

    # Vieta: roots are xP, xP, xP, xP, xR
    f4   = f[4] if len(f) > 4 else 0
    f5   = f[5] if len(f) > 5 else 1
    a2sq = a2 * a2 % p
    sum_roots = (a2sq - f4) % p * _modinv(f5, p) % p
    xR = (sum_roots - 4 * xP) % p

    yR = (-_poly_eval(A_coeffs, xR, p)) % p

    yR_sq = yR * yR % p
    fR    = _poly_eval(f, xR, p)
    if yR_sq != fR:
        raise ArithmeticError(
            f"compute_phi conjugate: R = ({xR}, {yR}) is not on the curve "
            f"(y²={yR_sq}, f(x_R)={fR})."
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
    R,   # (xR, yR) for generic/conjugate, or ((sum_RS, prod_RS), None) for self
    # g_coeffs is accepted but ignored (backward compat).
    g_coeffs: Sequence[int] | None = None,
) -> dict[str, bool]:
    """
    Sanity checks for the constructed φ and the claimed divisor.

    For generic/conjugate geometry (R is a single point):
        Checks phi_P_zero, phi_Q_zero, phi_R_zero, P/Q/R_on_curve,
        double_zero_P, double_zero_Q, dphi_curve_P_zero, dphi_curve_Q_zero.

    For self geometry (P=Q, R is a Mumford pair ((sum_RS, prod_RS), None)):
        Checks phi_P_zero, P_on_curve, double_zero_P,
        quad_zero_P (h'''(xP)=0), dphi_curve_P_zero, and
        mumford_u_correct (u(x)=x²−sum·x+prod divides h(x) over F_p via
        checking that h evaluated at any root of u is zero when that root
        exists in F_p, or that u² | h symbolically by checking
        the resultant vanishes — here we just verify Vieta coefficients
        match the trailing h coefficients).

    Returns a dict of {check_name: bool}.  All should be True.
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
        c2 = c * c % p
        return (c2 * _poly_eval(f, x, p) - pow(_poly_eval(A, x, p), 2, p)) % p

    def h_deriv_val(x: int) -> int:
        c2  = c * c % p
        Ax  = _poly_eval(A, x, p)
        Apx = _poly_eval(Ap, x, p)
        fpx = _poly_eval(fp, x, p)
        return (c2 * fpx - 2 * Ax * Apx) % p

    xP, yP = int(P[0]) % p, int(P[1]) % p
    xQ, yQ = int(Q[0]) % p, int(Q[1]) % p

    # Detect self geometry.
    _self_geo = (isinstance(R, tuple) and len(R) == 2
                 and isinstance(R[0], tuple) and R[1] is None)

    if _self_geo:
        sum_RS, prod_RS = R[0]

        # h''' at xP.
        fpp  = _poly_deriv(fp, p)
        fppp = _poly_deriv(fpp, p)
        App  = _poly_deriv(Ap, p)
        Appp = _poly_deriv(App, p)
        def h_triple_deriv(x: int) -> int:
            c2   = c * c % p
            Ax   = _poly_eval(A, x, p)
            Apx  = _poly_eval(Ap, x, p)
            Appx = _poly_eval(App, x, p)
            Apppx= _poly_eval(Appp, x, p)
            f3x  = _poly_eval(fppp, x, p)
            return (c2 * f3x - 2 * (3 * Apx % p * Appx + Ax * Apppx)) % p

        # Verify Mumford u-polynomial matches Vieta from h's leading coefficients.
        # h coefficients: leading = −a₃², x⁵ coeff = f[5] − 2a₂a₃.
        # We re-derive sum/prod from A and check against returned values.
        a3 = A[3] if len(A) > 3 else 0
        a2 = A[2] if len(A) > 2 else 0
        a1 = A[1] if len(A) > 1 else 0
        f5 = f[5] if len(f) > 5 else 1
        f4 = f[4] if len(f) > 4 else 0
        a3sq = a3 * a3 % p
        inv_a3sq = pow(a3sq, p - 2, p) if a3sq != 0 else None
        xP2 = xP * xP % p

        if inv_a3sq is not None:
            sigma1 = (f5 - 2 * a2 % p * a3) % p * inv_a3sq % p
            sigma2 = (-(f4 - a2 * a2 % p - 2 * a1 % p * a3)) % p * inv_a3sq % p
            expected_sum  = (sigma1 - 4 * xP) % p
            expected_prod = (sigma2 - 6 * xP2 - 4 * xP % p * expected_sum) % p
            mumford_ok = (int(sum_RS) % p == expected_sum
                          and int(prod_RS) % p == expected_prod)
        else:
            mumford_ok = False

        h2 = _poly_deriv(fp, p)
        def h_double_deriv(x: int) -> int:
            c2   = c * c % p
            Ax   = _poly_eval(A, x, p)
            Apx  = _poly_eval(Ap, x, p)
            Appx = _poly_eval(_poly_deriv(Ap, p), x, p)
            f2x  = _poly_eval(h2, x, p)
            return (c2 * f2x - 2 * (Apx * Apx + Ax * Appx)) % p

        return {
            "phi_P_zero":         phi(P) == 0,
            "P_on_curve":         pow(yP, 2, p) == _poly_eval(f, xP, p),
            "double_zero_P":      h_val(xP) == 0 and h_deriv_val(xP) == 0,
            "quad_zero_P":        h_double_deriv(xP) == 0 and h_triple_deriv(xP) == 0,
            "dphi_curve_P_zero":  dphi_curve(P) == 0,
            "mumford_u_correct":  mumford_ok,
        }

    # Generic / conjugate geometry.
    xR, yR = int(R[0]) % p, int(R[1]) % p

    return {
        # φ vanishes at all three zeros.
        "phi_P_zero":  phi(P) == 0,
        "phi_Q_zero":  phi(Q) == 0,
        "phi_R_zero":  phi(R) == 0,
        # All points on the curve.
        "P_on_curve":  pow(yP, 2, p) == _poly_eval(f, xP, p),
        "Q_on_curve":  pow(yQ, 2, p) == _poly_eval(f, xQ, p),
        "R_on_curve":  pow(int(R[1]), 2, p) == _poly_eval(f, int(R[0]) % p, p),
        # Double-zero structure.
        "double_zero_P":  h_val(xP) == 0 and h_deriv_val(xP) == 0,
        "double_zero_Q":  h_val(xQ) == 0 and h_deriv_val(xQ) == 0,
        # φ'|_C = 0 at P and Q.
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

    Works for any degree of A(x):
    - deg A = 2 (generic / conjugate): h has degree 5, roots xP(×2),xQ(×2),xR.
    - deg A = 3 (self, P=Q):           h has degree 6, roots xP(×4),xR,xS.

    Under the generic divisor 2P+2Q+R−5∞, h factors as
        f[5]·(x−xP)²·(x−xQ)²·(x−xR)  over F_p.
    Under the self divisor 4P+R+S−6∞, h factors as
        −a₃²·(x−xP)⁴·(x−xR)·(x−xS)  over F_p.
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
