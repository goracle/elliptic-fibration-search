from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
from .phi import compute_phi, phi_quintic
from search_common import *

"""phi_search.py  –  markov/walker/phi_search.py

Post-process one search-fn result dict by attempting a φ-step.

The Mumford/LLL search (markov_search_fn.py) already builds the fiber
    g(x)  over  F_p[m]  (stored as ``fi`` in the result dict)
and already finds m-roots that give F_p-rational x-coordinates.

This module takes that result and, for each candidate (x_step, m_val),
evaluates g at the concrete m_val to get a quartic g_n over F_p, then
calls ``compute_phi`` to obtain the rational function

    φ(x,y) = A(x) + y   (c normalised to 1)

such that  div(φ) = 2P + Q + R + S − 5∞,  where P is the current walk
point and Q is the candidate.  A(x) is a degree-2 polynomial whose
coefficients are determined by the three conditions P (order-2 tangency),
Q, and one of the two free zeros R or S; the system is generically
non-degenerate (the earlier 2P+2Q+R−5∞ attempt produced a homogeneous
system and was dropped).  The quintic  h(x) = f(x) − A(x)²  is the
intersection_poly that the walker's ``_make_relation`` machinery already
knows how to read.

y-sign handling
---------------
``_recover_y`` returns the canonical (smaller) square root.  The correct
branch for Q is not known a priori: the consistency condition φ(Q)=0
holds for exactly one of the two y-signs.  ``augment_with_phi`` tries
both  (x_step, y_canonical)  and  (x_step, p − y_canonical)  before
giving up on a record.  A ``ValueError`` from ``compute_phi`` signals a
consistency failure for that branch (expected; retry the other sign).
A ``ZeroDivisionError`` or ``ArithmeticError`` signals degenerate
geometry (retrying the other sign won't help; give up immediately).
Swapping the roles of P and Q is NOT used as a fallback.

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
                raise
                return None
            g.append(int(val) % p)
        return g
    except Exception:
        raise
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
        A(x) quadratic, div(φ) = 2P+Q+R+S−5∞, h has degree 5.
        Both y-signs of Q are tried; the one satisfying φ(Q)=0 is kept.
        (The earlier 2P+2Q+R−5∞ form produced a homogeneous/overdetermined
        linear system for A's coefficients and was abandoned.)

    Conjugate (x_step == x_src, y_step == −y_src):
        compute_phi dispatches automatically to _compute_phi_conjugate.
        A(x) quadratic, div(φ) = 4P+R+S−5∞, h has degree 5.

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

    P = (x_src_int, y_src_int)
    fxP = sum(c * pow(x_src_int, i, p) % p for i, c in enumerate(f_list)) % p
    yP_sq = y_src_int * y_src_int % p
    if fxP != yP_sq:
        raise ArithmeticError(
            f"augment_with_phi: P=({x_src_int},{y_src_int}) is not on curve "
            f"(f(xP)={fxP}, yP²={yP_sq})"
        )

    candidates: List[Dict[str, Any]] = list(
        result.get("candidate_records") or result.get("candidates") or []
    )
    snapshot = list(candidates)
    derived_sources = {"phi_generic_r", "phi_self_rs"}

    any_succeeded = False

    # Failure counters — classified by root cause.
    _fail_not_dict     = 0  # rec not a dict
    _fail_no_m         = 0  # m_val missing
    _fail_fi_eval      = 0  # fiber eval failed at m
    _fail_no_xstep     = 0  # x_step missing
    _fail_self_phi     = 0  # self-geo: compute_phi raised
    _fail_self_mumford = 0  # self-geo: bad Mumford return shape
    _fail_self_nosplit = 0  # self-geo: u(x) doesn't split over F_p
    _fail_not_on_curve = 0  # generic: x_step not on curve
    _fail_both_signs   = 0  # generic: both y-signs failed consistency
    _fail_xres_recover = 0  # generic: succeeded but second compute_phi for xR threw
    _n_success_generic = 0
    _n_success_self    = 0

    for rec in snapshot:
        if not isinstance(rec, dict):
            _fail_not_dict += 1
            continue

        if rec.get("source") in derived_sources:
            continue

        m_val = rec.get("m")
        if m_val is None:
            _fail_no_m += 1
            continue

        g_coeffs_eval = _eval_fi_at_m(fi, m_val, p)
        if g_coeffs_eval is None:
            _fail_fi_eval += 1
            continue

        x_step = rec.get("x_step")
        if x_step is None:
            _fail_no_xstep += 1
            continue
        x_step_int = int(x_step) % p

        # ---------------------------------------------------------------
        # Self geometry: P = Q (x_step == x_src, same y).
        # ---------------------------------------------------------------
        if x_step_int == x_src_int:
            Q_self = P
            try:
                A_coeffs, c, R_mumford = compute_phi(
                    p, f_list, g_coeffs_eval, P, Q_self
                )
            except (ValueError, ZeroDivisionError, ArithmeticError) as e:
                _fail_self_phi += 1
                print(f"  [phi_fail:self_phi] x_step={x_step_int} m={m_val} err={e}")
                continue

            if not (isinstance(R_mumford, tuple) and len(R_mumford) == 2
                    and isinstance(R_mumford[0], tuple) and R_mumford[1] is None):
                _fail_self_mumford += 1
                print(f"  [phi_fail:self_mumford] x_step={x_step_int} m={m_val} "
                      f"R_mumford={R_mumford!r}")
                continue

            sum_RS, prod_RS = R_mumford[0]
            rs_roots = _mumford_roots(int(sum_RS) % p, int(prod_RS) % p, p)

            if not rs_roots:
                _fail_self_nosplit += 1
                disc = (sum_RS * sum_RS - 4 * prod_RS) % p
                print(f"  [phi_fail:self_nosplit] x_step={x_step_int} m={m_val} "
                      f"sum_RS={int(sum_RS)%p} prod_RS={int(prod_RS)%p} disc={disc}")
                continue

            h_coeffs = phi_quintic(p, f_list, A_coeffs, c)
            h_sage   = sage_ring(h_coeffs)

            rec["intersection_poly"] = h_sage
            rec["phi_P"] = list(P)
            rec["phi_Q"] = list(P)
            rec["phi_mumford_RS"] = [int(sum_RS) % p, int(prod_RS) % p]
            rec["phi_geo"] = "self"
            any_succeeded = True
            _n_success_self += 1

            if result.get("intersection_poly") is None:
                result["intersection_poly"] = h_sage

            for xrs in rs_roots:
                y_canonical = _recover_y(xrs, f_list, p)
                if y_canonical is None:
                    continue
                new_rec = dict(rec)
                new_rec["x_step"]            = xrs
                new_rec["x_res"]             = xrs
                new_rec["yj_sign"]           = 1
                new_rec["yk_sign"]           = 1
                new_rec["intersection_poly"] = h_sage
                new_rec["phi_geo"]           = "self_rs"
                new_rec["source"]            = "phi_self_rs"
                new_rec["phi_mumford_RS"]    = [int(sum_RS) % p, int(prod_RS) % p]
                candidates.append(new_rec)

            continue

        # ---------------------------------------------------------------
        # Generic / conjugate geometry.
        # div(φ) = 2P + Q + R + S − 5∞.
        # A(x) is quadratic; the three interpolation conditions are:
        #   - order-2 tangency at P  (2 conditions)
        #   - simple zero at Q       (1 condition)
        # R and S are the two remaining free zeros of h(x) = f(x)−A(x)².
        # Try both y-signs for Q; the correct branch is the one where
        # φ(Q) = 0 passes compute_phi's consistency check.
        # ValueError means wrong y-sign (expected, retry the other).
        # ZeroDivisionError/ArithmeticError means degenerate geometry
        # (don't bother retrying the other sign).
        # ---------------------------------------------------------------
        y_canonical = _recover_y(x_step_int, f_list, p)
        if y_canonical is None:
            _fail_not_on_curve += 1
            continue

        A_coeffs = c = R_mumford = None
        y_used = None
        _degenerate = False
        for y_try in (y_canonical, (p - y_canonical) % p):
            Q_try = (x_step_int, y_try)
            try:
                A_coeffs, c, R_mumford = compute_phi(p, f_list, g_coeffs_eval, P, Q_try)
                y_used = y_try
                break
            except ValueError as e:
                # Consistency check failed for this y-sign; try the other.
                print(f"  [phi_fail:generic] x_step={x_step_int} m={m_val} err={e}")
                continue
            except (ZeroDivisionError, ArithmeticError) as e:
                # Degenerate geometry — retrying the other sign won't help.
                print(f"  [phi_fail:generic] x_step={x_step_int} m={m_val} err={e}")
                _degenerate = True
                break

        if y_used is None:
            _fail_both_signs += 1
            continue

        # compute_phi generic/conjugate now returns a Mumford pair
        # ((sum_RS, prod_RS), None) for the two free zeros R and S of
        # div(φ) = 2P+Q+R+S−5∞.  Validate the shape.
        RS_mumford = R_mumford
        if not (isinstance(RS_mumford, tuple) and len(RS_mumford) == 2
                and isinstance(RS_mumford[0], tuple) and RS_mumford[1] is None):
            _fail_both_signs += 1
            print(f"  [phi_fail:generic_mumford] x_step={x_step_int} m={m_val} "
                  f"RS={RS_mumford!r}")
            continue

        sum_RS, prod_RS = RS_mumford[0]
        rs_roots = _mumford_roots(int(sum_RS) % p, int(prod_RS) % p, p)

        # x_res is the first rational root of u(x) = x^2-sum_RS*x+prod_RS,
        # or sum_RS if u doesn't split (walker will handle the miss).
        xR_int = rs_roots[0] if rs_roots else int(sum_RS) % p

        h_coeffs = phi_quintic(p, f_list, A_coeffs, c)
        h_sage   = sage_ring(h_coeffs)

        rec["intersection_poly"]  = h_sage
        rec["x_res"]              = xR_int
        rec["yk_sign"]            = 1
        rec["yj_sign"]            = 1
        rec["phi_P"]              = list(P)
        rec["phi_Q"]              = [x_step_int, y_used]
        rec["phi_mumford_RS"]     = [int(sum_RS) % p, int(prod_RS) % p]
        rec["phi_geo"]            = "generic"
        any_succeeded = True
        _n_success_generic += 1

        if result.get("intersection_poly") is None:
            result["intersection_poly"] = h_sage

        # Inject a synthetic candidate for each rational root of u(x)
        # so the walker can also step to R and/or S directly.
        for xrs in rs_roots:
            y_rs = _recover_y(xrs, f_list, p)
            if y_rs is None:
                continue
            new_rec = dict(rec)
            new_rec["x_step"]            = xrs
            new_rec["x_res"]             = xrs
            new_rec["yj_sign"]           = 1
            new_rec["yk_sign"]           = 1
            new_rec["intersection_poly"] = h_sage
            new_rec["phi_geo"]           = "generic_rs"
            new_rec["source"]            = "phi_generic_r"
            new_rec["phi_mumford_RS"]    = [int(sum_RS) % p, int(prod_RS) % p]
            candidates.append(new_rec)

    deduped_candidates: List[Dict[str, Any]] = []
    seen = set()
    for rec in candidates:
        if isinstance(rec, dict):
            key = (
                rec.get("source"),
                rec.get("m"),
                rec.get("x_step"),
                rec.get("x_res"),
                rec.get("phi_geo"),
                tuple(rec.get("phi_mumford_RS") or ()),
            )
            if key in seen:
                continue
            seen.add(key)
        deduped_candidates.append(rec)

    n_in = len([c for c in deduped_candidates if isinstance(c, dict)
                and c.get("source") not in ("phi_self_rs", "phi_generic_r")])
    n_out = _n_success_generic + _n_success_self
    print(
        f"[phi_aug] P=({x_src_int},{y_src_int})  in={n_in}  "
        f"ok={n_out} (generic={_n_success_generic} self={_n_success_self})  "
        f"fail: not_dict={_fail_not_dict} no_m={_fail_no_m} "
        f"fi_eval={_fail_fi_eval} no_xstep={_fail_no_xstep} "
        f"not_on_curve={_fail_not_on_curve} both_signs={_fail_both_signs} "
        f"xres_recover={_fail_xres_recover} "
        f"self_phi={_fail_self_phi} self_mumford={_fail_self_mumford} "
        f"self_nosplit={_fail_self_nosplit}"
    )

    # Drop dead-end pure_fiber_intersection records with no x_res.
    deduped_candidates = [
        c for c in deduped_candidates
        if not (
            isinstance(c, dict)
            and c.get("source") == "pure_fiber_intersection"
            and c.get("x_res") is None
        )
    ]
    result["candidate_records"] = deduped_candidates
    result["candidates"]        = deduped_candidates

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
    degree 5 and div(φ) = 2P+Q+R+S−5∞.  compute_phi returns a Mumford pair
    ((sum_RS, prod_RS), None) for the two free zeros.

    For the self geometry (P = Q exactly) h has degree 6 and div(φ) = 4P+R+S−6∞.
    R from compute_phi is also a Mumford pair ((sum_RS, prod_RS), None).
    phi_quintic is called the same way in both cases because it only needs A_coeffs.

    Raises all non-consistency arithmetic errors.
    """
    try:
        A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    except ValueError:
        raise
        return None
    except (ZeroDivisionError, ArithmeticError):
        raise

    h_coeffs = phi_quintic(p, f_coeffs, A_coeffs, c)
    return sage_ring(h_coeffs)


def _phi_for_pair_dbg(
    f_coeffs: list[int],
    g_coeffs: list[int],
    P: tuple[int, int],
    Q: tuple[int, int],
    p: int,
    sage_ring,
) -> tuple[Optional[Any], Optional[int]]:
    """Like _phi_for_pair but returns (h_sage, residue) so the caller can log
    the consistency residue on failure instead of just getting None.

    residue is 0 on success, the nonzero φ(Q) value on consistency failure,
    or None if a non-consistency exception was raised.
    """
    try:
        A_coeffs, c, R = compute_phi(p, f_coeffs, g_coeffs, P, Q)
    except ValueError as e:
        # Extract residue from the error message if present.
        msg = str(e)
        residue = None
        if "residue=" in msg:
            try:
                residue = int(msg.split("residue=")[1].rstrip(")").strip())
            except Exception:
                raise
        return None, residue
    except (ZeroDivisionError, ArithmeticError):
        raise

    h_coeffs = phi_quintic(p, f_coeffs, A_coeffs, c)
    return sage_ring(h_coeffs), 0


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
