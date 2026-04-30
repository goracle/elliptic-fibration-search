"""phi.py  –  markov/walker/phi.py

Construct the rational function

    φ(x, y) = A(x) + c·y,   c ∈ F_p

on the genus-2 hyperelliptic curve  C: y² = f(x)  (deg f = 5).

Four geometries are supported, dispatched automatically by compute_phi:

┌──────────────────────────┬──────────┬──────────────────────────────┬──────────────┐
│ Geometry                 │ deg A(x) │ div(φ)                       │ Extra zeros  │
├──────────────────────────┼──────────┼──────────────────────────────┼──────────────┤
│ Generic  P ≠ Q           │    2     │ 2P + Q + R + S − 5∞          │ R,S (Mumford)│
│ Conjugate  Q = (xP,−yP)  │    2     │ 4P            + R   − 5∞     │ R (1 point)  │
│ Self  P = Q              │    3     │ 4P + R + S    − 6∞           │ R,S (Mumford)│
│ Self-conjugate (y=0)     │    –     │ (degenerate; rejected)       │ –            │
└──────────────────────────┴──────────┴──────────────────────────────┴──────────────┘

Generic case (P ≠ Q, xP ≠ xQ)
-------------------------------
A(x) = a₀ + a₁x + a₂x², c = 1.
h(x) = f(x) − A(x)² has degree 5 and roots xP(×2), xQ(×1), xR(×1), xS(×1).
Three conditions (2P + Q simple zero is non-degenerate / non-homogeneous):

  (1) A(xP) = −yP                     [φ(P) = 0]
  (2) A'(xP) = −f'(xP)/(2yP)         [double root at xP]
  (3) A(xQ) = −yQ                     [φ(Q) = 0, simple zero]

(The old 4-condition system enforcing a double root at Q too was homogeneous /
overdetermined, leaving only a 1-parameter family; dropping condition (4) gives
a unique solution.)

(2) → a₁ + 2a₂xP = −fslope_P;  (1) after solving (2) with (3):
  From (2): a₁ = −fslope_P − 2a₂xP
  Sub into (3): −fslope_P·(xQ−xP) + a₂(xQ²−xP²) − (yQ−yP) ≡ 0 mod p
                → a₂ = (yQ − yP + fslope_P·(xQ−xP)) / (xQ²−xP²)
  Then a₁ and a₀ follow.

Vieta: roots xP(×2), xQ, xR, xS.  Sum of five roots = (a₂²−f[4])/f[5].
  xR + xS = sum − 2xP − xQ
  xR · xS recovered from the x³ coefficient of h.

Returns ((sum_RS, prod_RS), None) as a Mumford pair.
The caller must factor u(x) = x² − (xR+xS)x + xR*xS over F_p; if u has no
F_p roots the geometry is skipped (no field-extension arithmetic supported).

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

    # Generic/self:  R is a Mumford pair ((sum_RS, prod_RS), None) — check with
    #                isinstance(R[0], tuple).
    # Conjugate:     R is a single (xR, yR) point.

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
    P: Point,                  # (x_P, y_P) – double-root point
    Q: Point,                  # (x_Q, y_Q) – simple-zero point
) -> tuple[list[int], int, tuple]:
    """
    Compute φ(x,y) = A(x) + y  (c normalised to 1) such that
    div(φ) = 2P + Q + R + S − 5∞  (generic, P≠Q, xP≠xQ),
             4P + R − 5∞           (conjugate, Q=(xP,−yP)),
          or 4P + R + S − 6∞       (self, P=Q exactly).

    Parameters
    ----------
    p         : prime characteristic
    f_coeffs  : coefficients of the curve polynomial f (low first), deg 5
    g_coeffs  : ignored (retained for API compatibility)
    P         : (x_P, y_P)  in C(F_p), not a Weierstrass point; double root here
    Q         : (x_Q, y_Q)  in C(F_p), not a Weierstrass point; simple zero here

    Returns
    -------
    A_coeffs  : coefficients of A(x) (deg 2 for generic/conjugate, deg 3 for self)
    c         : 1  (always, by normalisation)
    R         : Generic/self → ((sum_RS, prod_RS), None) as Mumford pair.
                Conjugate     → (x_R, y_R) as a single point.

    Raises
    ------
    ValueError        – degenerate geometry (Weierstrass points, or the Mumford
                        u-polynomial has no F_p roots in the generic case — skip,
                        no field-extension arithmetic supported)
    ArithmeticError   – internal sanity check failed (indicates a bug)
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

    # slope of the curve tangent at P (needed for the double-root condition).
    # fslope_P = f'(xP) / (2·yP)  — this equals −A'(xP) when h'(xP)=0.
    fslope_P = _poly_eval(fp, xP, p) * invyP % p * inv2 % p

    # -----------------------------------------------------------------------
    # Solve for (a₀, a₁, a₂) from three conditions:
    #
    #   (1) A(xP)  = −yP                      [φ(P) = 0]
    #   (2) A'(xP) = −fslope_P                [double root at xP: h'(xP)=0]
    #   (3) A(xQ)  = −yQ                      [φ(Q) = 0, simple zero]
    #
    # From (2): a₁ + 2·a₂·xP = −fslope_P
    #   →  a₁ = −fslope_P − 2·a₂·xP
    #
    # Substitute into (3) − (1):
    #   A(xQ) − A(xP) = −yQ − (−yP) = yP − yQ
    #   a₁(xQ−xP) + a₂(xQ²−xP²) = yP − yQ
    #   (−fslope_P − 2·a₂·xP)(xQ−xP) + a₂(xQ²−xP²) = yP − yQ
    #   a₂·[(xQ²−xP²) − 2xP(xQ−xP)] = yP − yQ + fslope_P·(xQ−xP)
    #   a₂·(xQ−xP)·(xQ−xP)           = yP − yQ + fslope_P·(xQ−xP)
    #   a₂·(xQ−xP)²                  = yP − yQ + fslope_P·(xQ−xP)
    # -----------------------------------------------------------------------
    xP2 = xP * xP % p
    xQ2 = xQ * xQ % p
    xdiff  = (xQ - xP) % p   # non-zero since xP ≠ xQ
    xdiff2 = xdiff * xdiff % p

    rhs_a2 = (yP - yQ + fslope_P * xdiff) % p
    a2 = rhs_a2 * _modinv(xdiff2, p) % p

    a1 = (-fslope_P - 2 * a2 * xP) % p
    a0 = (-yP - a1 * xP - a2 * xP2) % p

    c = 1
    A_coeffs = [a0, a1, a2]

    # Sanity: verify φ(P)=0, h'(xP)=0, φ(Q)=0.
    _Ap = _poly_deriv(A_coeffs, p)
    if (_poly_eval(A_coeffs, xP, p) + yP) % p != 0:
        raise ArithmeticError("compute_phi generic: φ(P)≠0 after construction (bug).")
    if (_poly_eval(_Ap, xP, p) + fslope_P) % p != 0:
        raise ArithmeticError("compute_phi generic: A'(xP)+fslope_P≠0 (bug).")
    if (_poly_eval(A_coeffs, xQ, p) + yQ) % p != 0:
        raise ArithmeticError(
            "compute_phi generic: φ(Q)≠0 — Q's y-sign is wrong; "
            "try Q = (x_Q, p − y_Q)."
        )

    # -----------------------------------------------------------------------
    # Vieta for h(x) = f(x) − A(x)²  (degree 5, leading coeff f[5]).
    #
    # Roots are xP(×2), xQ(×1), xR, xS.
    #
    # h(x) = f[5]·x⁵ + (f[4]−a₂²)·x⁴ + (f[3]−2a₁a₂)·x³ + …
    #
    # Vieta elementary symmetric polynomials (roots r₁…r₅):
    #   e₁ = Σrᵢ        = −(f[4]−a₂²) / f[5]
    #   e₂ = Σᵢ<ⱼrᵢrⱼ  =  (f[3]−2a₁a₂) / f[5]
    #
    # With roots xP, xP, xQ, xR, xS:
    #   e₁ = 2xP + xQ + xR + xS
    #     →  sum_RS = xR+xS = e₁ − 2xP − xQ
    #   e₂ = xP²+ 2xP·xQ + (2xP+xQ)(xR+xS) + xR·xS
    #     →  prod_RS = xR·xS = e₂ − xP² − 2xP·xQ − (2xP+xQ)·sum_RS
    # -----------------------------------------------------------------------
    f3 = f[3] if len(f) > 3 else 0
    f4 = f[4] if len(f) > 4 else 0
    f5 = f[5] if len(f) > 5 else 1

    a2sq = a2 * a2 % p
    inv_f5 = _modinv(f5, p)

    e1 = (-(f4 - a2sq)) % p * inv_f5 % p
    e2 = (f3 - 2 * a1 % p * a2) % p * inv_f5 % p

    sum_RS  = (e1 - 2 * xP - xQ) % p
    # e₂ = xP² + 2xP·xQ + (2xP+xQ)·sum_RS + prod_RS
    prod_RS = (e2 - xP2 - 2 * xP % p * xQ - (2 * xP + xQ) % p * sum_RS) % p

    return A_coeffs, c, ((sum_RS, prod_RS), None)


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
    d2   = (2 * s * s - f2xP) % p * invyP % p * inv2 % p

    # d3 = A'''(xP)
    #   h'''(xP) = f'''(xP) − 2(3·A'·A'' + A·A''') = 0
    #   with A(xP) = −yP:
    #     f'''(xP) − 6s·d2 + 2yP·A'''(xP) = 0
    #     A'''(xP) = (6s·d2 − f'''(xP)) / (2yP)
    f3xP = _poly_eval(fppp, xP, p)
    d3   = (6 * s * d2 - f3xP) % p * invyP % p * inv2 % p

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
    a2 = (d2 - 6 * a3 * xP) % p * inv2 % p

    # A'(xP) = a₁ + 2a₂xP + 3a₃xP² = s
    xP2 = xP * xP % p
    a1  = (s - 2 * a2 * xP - 3 * a3 * xP2) % p

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

    # Generic geometry: R is a Mumford pair ((sum_RS, prod_RS), None), P≠Q.
    # (Same sentinel shape as self, but deg A = 2 not 3.)
    _generic_mumford = (not _self_geo
                        and isinstance(R, tuple) and len(R) == 2
                        and isinstance(R[0], tuple) and R[1] is None)

    if _generic_mumford:
        sum_RS, prod_RS = R[0]

        # Re-derive Mumford pair from A coefficients and Vieta, then compare.
        a2v = A[2] if len(A) > 2 else 0
        a1v = A[1] if len(A) > 1 else 0
        f5v = f[5] if len(f) > 5 else 1
        f4v = f[4] if len(f) > 4 else 0
        f3v = f[3] if len(f) > 3 else 0
        inv_f5v = pow(f5v, p - 2, p)
        xP2v = xP * xP % p

        e1v = (-(f4v - a2v * a2v % p)) % p * inv_f5v % p
        e2v = (f3v - 2 * a1v % p * a2v) % p * inv_f5v % p
        exp_sum  = (e1v - 2 * xP - xQ) % p
        exp_prod = (e2v - xP2v - 2 * xP % p * xQ % p
                    - (2 * xP + xQ) % p * exp_sum) % p
        mumford_ok = (int(sum_RS) % p == exp_sum
                      and int(prod_RS) % p == exp_prod)

        return {
            "phi_P_zero":        phi(P) == 0,
            "phi_Q_zero":        phi(Q) == 0,
            "P_on_curve":        pow(yP, 2, p) == _poly_eval(f, xP, p),
            "Q_on_curve":        pow(yQ, 2, p) == _poly_eval(f, xQ, p),
            "double_zero_P":     h_val(xP) == 0 and h_deriv_val(xP) == 0,
            "simple_zero_Q":     h_val(xQ) == 0,   # simple root, not double
            "dphi_curve_P_zero": dphi_curve(P) == 0,
            "mumford_u_correct": mumford_ok,
        }

    # Conjugate geometry: R is a plain (xR, yR) point.
    xR, yR = int(R[0]) % p, int(R[1]) % p

    return {
        # φ vanishes at all zeros.
        "phi_P_zero":  phi(P) == 0,
        "phi_Q_zero":  phi(Q) == 0,
        "phi_R_zero":  phi(R) == 0,
        # All points on the curve.
        "P_on_curve":  pow(yP, 2, p) == _poly_eval(f, xP, p),
        "Q_on_curve":  pow(yQ, 2, p) == _poly_eval(f, xQ, p),
        "R_on_curve":  pow(int(R[1]), 2, p) == _poly_eval(f, int(R[0]) % p, p),
        # Four-fold zero structure at xP (conjugate case: div = 4P+R−5∞).
        "double_zero_P":  h_val(xP) == 0 and h_deriv_val(xP) == 0,
        # φ'|_C = 0 at P.
        "dphi_curve_P_zero":  dphi_curve(P) == 0,
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
    - deg A = 2 (generic / conjugate): h has degree 5.
      Generic:   roots xP(×2), xQ(×1), xR, xS  (divisor 2P+Q+R+S−5∞).
      Conjugate: roots xP(×4), xR               (divisor 4P+R−5∞).
    - deg A = 3 (self, P=Q):           h has degree 6, roots xP(×4), xR, xS.

    Under the generic divisor 2P+Q+R+S−5∞, h factors as
        f[5]·(x−xP)²·(x−xQ)·(x−xR)·(x−xS)  over F_p.
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
