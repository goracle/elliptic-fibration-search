from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
from .phi import compute_phi, phi_quintic

"""phi_search.py  –  markov/walker/phi_search.py

Post-process one search-fn result dict by attempting a φ-step.

The Mumford/LLL search (markov_search_fn.py) already builds the fiber
    g(x)  over  F_p[m]  (stored as ``fi`` in the result dict)
and already finds m-roots that give F_p-rational x-coordinates.

This module takes that result and, for each candidate (x_step, m_val),
evaluates g at the concrete m_val to get a quartic g_n over F_p, then
calls ``compute_phi`` to obtain the rational function

    φ(x,y) = A(x) + y   (c normalised to 1)

such that  div(φ) = 2P + 2Q + R − 5∞,  where P is the current walk
point and Q is the candidate.  The quintic  h(x) = f(x) − A(x)²  is
the intersection_poly that the walker's ``_make_relation`` machinery
already knows how to read.

y-sign handling
---------------
``_recover_y`` returns the canonical (smaller) square root.  The correct
branch for Q is not known a priori: the consistency condition φ(Q)=0
holds for exactly one of the two y-signs.  ``augment_with_phi`` therefore
tries both  (x_step, y_canonical)  and  (x_step, p − y_canonical)
before giving up on a record.  Swapping the roles of P and Q is NOT used
as a fallback — the divisor is symmetric only in the sense that either
ordering gives a valid (but different) φ, and a wrong y-sign on Q will
cause compute_phi to raise ValueError with a clear message.

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
        coeffs_raw = fi.coefficients(sparse=False)
        g = []
        for coeff in coeffs_raw:
            try:
                val = coeff(m_val)
            except TypeError:
                return None
            g.append(int(val) % p)
        return g
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Core: attempt φ on one (P, Q) pair
# ---------------------------------------------------------------------------

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
    φ-derived polynomial h(x) = f(x) − A(x)²  (c normalised to 1).

    Three geometries are handled:

    Generic (x_step ≠ x_src):
        A(x) quadratic, div(φ) = 2P+2Q+R−5∞, h has degree 5.
        Both y-signs of Q are tried; the one satisfying φ(Q)=0 is kept.

    Conjugate (x_step == x_src, y_step == −y_src):
        compute_phi dispatches automatically to _compute_phi_conjugate.
        A(x) quadratic, div(φ) = 4P+R−5∞, h has degree 5.

    Self (x_step == x_src, y_step == y_src  i.e. P = Q exactly):
        compute_phi dispatches to _compute_phi_self.
        A(x) cubic, div(φ) = 4P+R+S−6∞, h has degree 6.
        R is returned as a Mumford u-polynomial pair ((sum_RS, prod_RS), None).
        u(x) = x²−sum_RS·x+prod_RS is factored over F_p; for each rational
        root xrs a new synthetic candidate record with source='phi_self_rs'
        is appended to the candidates list so the Metropolis chooser sees it.

    Records for which φ fails (wrong y-sign / degenerate) and whose source is
    ``pure_fiber_intersection`` are removed from the candidate list — they have
    no ``x_res`` and would be unconditionally skipped by the walker anyway.
    Records from other sources (e.g. ``phi_self_rs``, legacy stubs) are left
    unchanged so existing fallback paths continue to work.

    The local ``candidates`` list (which may have ``phi_self_rs`` records
    appended during the self-geometry branch) is written back to
    ``result["candidate_records"]`` and ``result["candidates"]`` before
    returning, so those appended records are not silently dropped.

    Does nothing and returns unchanged if:
    - result has no ``fi`` (symbolic fiber poly), or
    - no candidate record has a usable ``m`` value, or
    - every φ call fails.
    """
    fi = result.get("fi")
    if fi is None:
        return result

    f_list    = [int(c) % p for c in f_coeffs]
    x_src_int = int(x_src) % p
    y_src_int = int(y_src) % p

    P = (x_src_int, y_src_int)   # current point — double zero enforced here too
    print(f"[phi_dbg] P={P} f(xP)={sum(c*pow(x_src_int,i,p)%p for i,c in enumerate(f_list))%p} yP^2={y_src_int*y_src_int%p}")

    candidates: List[Dict[str, Any]] = list(
        result.get("candidate_records") or result.get("candidates") or []
    )

    any_succeeded = False

    for i, _dbg_rec in enumerate(candidates[:3]):
        print(f"[phi_aug_dbg] rec#{i} "
              f"keys={list(_dbg_rec.keys()) if isinstance(_dbg_rec, dict) else type(_dbg_rec)} "
              f"m={_dbg_rec.get('m') if isinstance(_dbg_rec, dict) else '?'} "
              f"x_step={_dbg_rec.get('x_step') if isinstance(_dbg_rec, dict) else '?'}")

    for rec in candidates:
        if not isinstance(rec, dict):
            continue

        m_val = rec.get("m")
        if m_val is None:
            continue

        # Evaluate the fiber at this concrete m (passed to compute_phi for
        # API compatibility, not used in the actual computation).
        g_coeffs = _eval_fi_at_m(fi, m_val, p)
        if g_coeffs is None:
            continue

        x_step = rec.get("x_step")
        if x_step is None:
            continue
        x_step_int = int(x_step) % p

        # ---------------------------------------------------------------
        # Self geometry: P = Q (x_step == x_src, same y).
        # compute_phi uses a degree-3 A, div(φ) = 4P + R + S − 6∞.
        # R is returned as a Mumford pair ((sum_RS, prod_RS), None).
        # We factor u(x) = x²−sum·x+prod over F_p and stamp each rational
        # root as a separate candidate record.
        # ---------------------------------------------------------------
        if x_step_int == x_src_int:
            Q_self = P   # P = Q
            try:
                A_coeffs, c, R_mumford = compute_phi(
                    p, f_list, g_coeffs, P, Q_self
                )
            except (ValueError, ZeroDivisionError, ArithmeticError):
                continue

            # R_mumford is ((sum_RS, prod_RS), None) for the self geometry.
            if not (isinstance(R_mumford, tuple) and len(R_mumford) == 2
                    and isinstance(R_mumford[0], tuple) and R_mumford[1] is None):
                continue

            sum_RS, prod_RS = R_mumford[0]
            rs_roots = _mumford_roots(int(sum_RS) % p, int(prod_RS) % p, p)

            if not rs_roots:
                # u(x) doesn't split over F_p — no rational next point.
                continue

            h_coeffs = phi_quintic(p, f_list, A_coeffs, c)
            h_sage   = sage_ring(h_coeffs)

            # Stamp the self-geometry result onto the original record and
            # inject extra records for the additional root(s).
            rec["intersection_poly"] = h_sage
            rec["phi_P"] = list(P)
            rec["phi_Q"] = list(P)   # Q = P in self geometry
            rec["phi_mumford_RS"] = [int(sum_RS) % p, int(prod_RS) % p]
            rec["phi_geo"] = "self"
            any_succeeded = True

            if result.get("intersection_poly") is None:
                result["intersection_poly"] = h_sage

            # For each rational RS root, inject a synthetic candidate record
            # so the Metropolis chooser sees it as a possible next step.
            for xrs in rs_roots:
                y_canonical = _recover_y(xrs, f_list, p)
                if y_canonical is None:
                    continue
                y_neg_rs = (p - y_canonical) % p
                new_rec = dict(rec)
                new_rec["x_step"]            = xrs
                new_rec["x_res"]             = xrs   # self geometry: x_res == x_step
                new_rec["yj_sign"]           = 1
                new_rec["yk_sign"]           = 1
                new_rec["intersection_poly"] = h_sage
                new_rec["phi_geo"]           = "self_rs"
                new_rec["source"]            = "phi_self_rs"
                new_rec["phi_mumford_RS"]    = [int(sum_RS) % p, int(prod_RS) % p]
                candidates.append(new_rec)

            continue

        # ---------------------------------------------------------------
        # Generic / conjugate geometry: x_step != x_src (or same x, neg y).
        # ---------------------------------------------------------------
        y_canonical = _recover_y(x_step_int, f_list, p)
        if y_canonical is None:
            continue

        # Try both y-signs of Q.  Only one satisfies the φ(Q)=0 consistency
        # condition; compute_phi raises ValueError for the wrong sign.
        y_neg = (p - y_canonical) % p
        h_sage = None
        chosen_y = None
        for y_try in (y_canonical, y_neg):
            if y_try == 0:
                continue
            Q = (x_step_int, y_try)
            h_sage = _phi_for_pair(f_list, g_coeffs, P, Q, p, sage_ring)
            if h_sage is not None:
                chosen_y = y_try
                break

        if h_sage is None:
            continue

        # Recover R so we can populate x_res.  Re-run compute_phi with the
        # winning y-sign — _phi_for_pair discards R, so we need it again.
        Q_win = (x_step_int, chosen_y)
        try:
            A_coeffs_win, c_win, R_win = compute_phi(p, f_list, g_coeffs, P, Q_win)
            xR_int = int(R_win[0]) % p
            yR_int = int(R_win[1]) % p
            # Canonical y-sign for x_res: 1 if yR is the smaller root, -1 otherwise
            yR_neg = (p - yR_int) % p
            yk_sign = 1 if yR_int <= yR_neg else -1
        except Exception:
            xR_int  = None
            yk_sign = 1

        # Success: stamp intersection poly, x_res, and sign metadata.
        rec["intersection_poly"] = h_sage
        rec["x_res"]    = xR_int
        rec["yk_sign"]  = yk_sign
        rec["yj_sign"]  = 1 if chosen_y == y_canonical else -1
        rec["phi_P"]    = list(P)
        rec["phi_Q"]    = [x_step_int, chosen_y]
        rec["phi_geo"]  = "generic"
        any_succeeded = True

        # Also promote to top-level if not already set.
        if result.get("intersection_poly") is None:
            result["intersection_poly"] = h_sage

    # Write the (possibly appended) candidates list back to result so that
    # phi_self_rs records injected during the self-geometry branch are not
    # silently dropped, and filter out pure_fiber_intersection records that
    # phi couldn't complete (x_res still None) — the walker would skip them
    # unconditionally anyway, so dropping them here gives a clean candidate
    # list and avoids the [cand_skip] missing x_step or x_res wall.
    candidates = [
        c for c in candidates
        if not (
            isinstance(c, dict)
            and c.get("source") == "pure_fiber_intersection"
            and c.get("x_res") is None
        )
    ]
    result["candidate_records"] = candidates
    result["candidates"]        = candidates

    return result

def _phi_for_pair(
    f_coeffs: list[int],
    g_coeffs: list[int],
    P: tuple[int, int],
    Q: tuple[int, int],
    p: int,
    sage_ring,
) -> Optional[Any]:
    """Attempt φ construction for one (P, Q) pair.

    Returns the Sage polynomial h(x) = f(x) − A(x)² on success, or None on
    a consistency failure (wrong y-branch for Q).

    For the generic/conjugate geometry (P ≠ Q or Q = conjugate of P) h has
    degree 5 and is returned directly.

    For the self geometry (P = Q exactly) h has degree 6.  R from compute_phi
    is a Mumford pair ((sum_RS, prod_RS), None) rather than a single point;
    phi_quintic is called the same way because it only needs A_coeffs.

    Raises all non-consistency arithmetic errors.
    """
    try:
        A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    except ValueError:
        return None
    except (ZeroDivisionError, ArithmeticError):
        raise

    h_coeffs = phi_quintic(p, f_coeffs, A_coeffs, c)
    return sage_ring(h_coeffs)


def _mumford_roots(sum_RS: int, prod_RS: int, p: int) -> list[int]:
    """Factor u(x) = x² − sum_RS·x + prod_RS over F_p.

    Returns a list of 0, 1, or 2 distinct roots (x-coordinates of the
    extra zeros R, S from the self-geometry φ).
    """
    # Discriminant = sum² − 4·prod
    disc = (sum_RS * sum_RS - 4 * prod_RS) % p
    if disc == 0:
        # Double root: x = sum/2
        return [sum_RS * pow(2, p - 2, p) % p]
    if pow(disc, (p - 1) // 2, p) != 1:
        # No roots over F_p.
        return []
    # Two distinct roots.
    if p % 4 == 3:
        sqrt_disc = pow(disc, (p + 1) // 4, p)
    else:
        # Tonelli-Shanks (reuse from module level)
        sqrt_disc = _tonelli_shanks(disc, p)
    inv2 = pow(2, p - 2, p)
    r1 = (sum_RS + sqrt_disc) * inv2 % p
    r2 = (sum_RS - sqrt_disc) * inv2 % p
    return sorted(set([r1, r2]))
