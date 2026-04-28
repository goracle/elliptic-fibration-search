"""phi_search.py  –  markov/walker/phi_search.py

Post-process one search-fn result dict by attempting a φ-step.

The Mumford/LLL search (markov_search_fn.py) already builds the fiber
    g(x)  over  F_p[m]  (stored as ``fi`` in the result dict)
and already finds m-roots that give F_p-rational x-coordinates.

This module takes that result and, for each candidate (x_step, m_val),
evaluates g at the concrete m_val to get a quartic g_n over F_p, then
calls ``compute_phi`` to obtain the rational function

    φ(x,y) = A(x) + c·y

adapted to the fiber at the current point (P) and the candidate point
(Q).  The quintic  h(x) = c²f(x) − A(x)²  is the intersection_poly
that the walker's ``_make_relation`` machinery already knows how to read.

Public API
----------
    from .phi_search import augment_with_phi

    result = search_fn(...)          # existing Mumford/LLL result dict
    result = augment_with_phi(
        result,
        f_coeffs  = f_coeffs,        # curve y²=f(x), list[int] low-first
        p         = p,
        x_src     = x_here,          # GF(p) element or int
        y_src     = y_here,          # GF(p) element or int
        sage_ring = R,               # PolynomialRing(GF(p), 'x')
    )

``augment_with_phi`` returns the same dict, with every candidate record
that successfully completes the φ construction having its
``intersection_poly`` replaced by h_sage (the authoritative Sage poly
over GF(p)).  Records for which φ fails are left unchanged so the
existing fallback path in ``step_from_candidate_search`` continues to
work.

If the result dict has no usable ``fi`` or no candidates with a concrete
m-value, the function is a no-op and returns the dict unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .phi import compute_phi, phi_quintic


# ---------------------------------------------------------------------------
# Helpers to evaluate the symbolic fiber poly at a concrete m value
# ---------------------------------------------------------------------------

def _eval_fi_at_m(fi, m_val, p: int) -> Optional[list[int]]:
    """Evaluate the symbolic fiber polynomial fi(x; m) at m = m_val.

    ``fi`` is a Sage polynomial in x whose coefficients live in
    Frac(GF(p)[m])  (the fraction field of the polynomial ring in m).
    Evaluating each coefficient at m = m_val gives a plain poly in x
    over GF(p).

    Returns a list of ints (low-degree first) of length deg(fi)+1,
    or None if the evaluation fails for any reason.
    """
    if fi is None:
        return None
    try:
        # fi.coefficients(sparse=False) gives [c0, c1, ...] low-first,
        # each c_i being a rational function of m.
        coeffs_raw = fi.coefficients(sparse=False)
        g = []
        for coeff in coeffs_raw:
            # Substitute m = m_val.  Works whether the coeff is a constant,
            # a polynomial in m, or a rational function in m.
            try:
                val = coeff(m_val)
            except TypeError:
                # coeff might not be callable; try direct coercion.
                val = coeff
            g.append(int(val) % p)
        return g
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: attempt φ on one (P, Q) pair
# ---------------------------------------------------------------------------

def _phi_for_pair(
    f_coeffs: list[int],
    g_coeffs: list[int],
    P: tuple[int, int],
    Q: tuple[int, int],
    p: int,
    sage_ring,
) -> Optional[Any]:
    """Try compute_phi(P, Q) and return the Sage intersection poly h, or None."""
    try:
        A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None

    h_coeffs = phi_quintic(p, f_coeffs, A_coeffs, c)
    return sage_ring(h_coeffs)


# ---------------------------------------------------------------------------
# y-recovery helper (avoids importing walker internals)
# ---------------------------------------------------------------------------

def _recover_y(x_int: int, f_coeffs: list[int], p: int) -> Optional[int]:
    """Return the canonical (smaller) square root of f(x) mod p, or None."""
    val = 0
    for i, c in enumerate(f_coeffs):
        val = (val + c * pow(x_int, i, p)) % p
    if val == 0:
        return None
    if pow(val, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        y = pow(val, (p + 1) // 4, p)
    else:
        y = _tonelli_shanks(val, p)
    return min(y, p - y) or None   # canonical branch; None if 0


def _tonelli_shanks(n: int, p: int) -> int:
    Q, S = p - 1, 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    M, c, t, R_ = S, pow(z, Q, p), pow(n, Q, p), pow(n, (Q + 1) // 2, p)
    while True:
        if t == 1:
            return R_
        i, tmp = 1, (t * t) % p
        while tmp != 1:
            tmp = (tmp * tmp) % p
            i += 1
        b  = pow(c, 1 << (M - i - 1), p)
        M  = i
        c  = (b * b) % p
        t  = (t * c) % p
        R_ = (R_ * b) % p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def augment_with_phi(
    result: Dict[str, Any],
    *,
    f_coeffs: Sequence[int],
    p: int,
    x_src,          # GF(p) element or int – current walk position
    y_src,          # GF(p) element or int – current y
    sage_ring,      # PolynomialRing(GF(p), 'x')
) -> Dict[str, Any]:
    """Attempt to replace intersection_poly in each candidate record with the
    φ-derived quintic h(x) = c²f(x) − A(x)².

    Works in-place on the candidate_records list inside ``result`` and also
    sets the top-level ``intersection_poly`` key if at least one record
    succeeds.  Returns ``result`` (same object).

    Does nothing and returns unchanged if:
    - result has no ``fi`` (symbolic fiber poly), or
    - no candidate record has a usable ``m`` value, or
    - every φ call fails (singular system, wrong intersection structure, …).
    """
    fi = result.get("fi")
    if fi is None:
        return result

    f_list    = [int(c) % p for c in f_coeffs]
    x_src_int = int(x_src) % p
    y_src_int = int(y_src) % p

    P = (x_src_int, y_src_int)   # current point is always one intersection

    candidates: List[Dict[str, Any]] = list(
        result.get("candidate_records") or result.get("candidates") or []
    )

    any_succeeded = False

    for rec in candidates:
        if not isinstance(rec, dict):
            continue

        m_val = rec.get("m")
        if m_val is None:
            continue

        # Evaluate the fiber at this concrete m.
        g_coeffs = _eval_fi_at_m(fi, m_val, p)
        if g_coeffs is None:
            continue

        # The candidate's x_step is the other intersection point Q.
        x_step = rec.get("x_step")
        if x_step is None:
            continue
        x_step_int = int(x_step) % p

        y_step_int = _recover_y(x_step_int, f_list, p)
        if y_step_int is None:
            continue

        Q = (x_step_int, y_step_int)

        # Try both orderings of (P, Q) — the 4×4 system may be singular for
        # one but not the other.
        h_sage = _phi_for_pair(f_list, g_coeffs, P, Q, p, sage_ring)
        if h_sage is None:
            h_sage = _phi_for_pair(f_list, g_coeffs, Q, P, p, sage_ring)
        if h_sage is None:
            continue

        # Success: stamp the intersection poly onto this record.
        rec["intersection_poly"] = h_sage
        rec["phi_P"] = list(P)
        rec["phi_Q"] = list(Q)
        any_succeeded = True

        # Also promote to top-level if not already set.
        if result.get("intersection_poly") is None:
            result["intersection_poly"] = h_sage

    return result
