from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence
from .phi import compute_phi, phi_quintic
from search_common import *
from .candidate_utils import *

"""phi_search.py  –  markov/walker/phi_search.py

Post-process one search-fn result dict by attempting a φ-step.

The Mumford/LLL search (markov_search_fn.py) already builds the fiber
    g(x)  over  F_p[m]  (stored as ``fi`` in the result dict)
and already finds m-roots that give F_p-rational x-coordinates.

This module takes that result and, for each candidate (pt_step, m_val),
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
both  (pt_step, y_canonical)  and  (pt_step, p − y_canonical)  before
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
        pt_src     = x_here,          # GF(p) element or int
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
    if isinstance(x_int, tuple):
        x_int = x_int[0]
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

def augment_with_phi(
    result: Dict[str, Any],
    *,
    f_coeffs: Sequence[int],
    p: int,
    pt_src,          # Expecting (x, y) tuple or point object
    sage_ring,
) -> Dict[str, Any]:
    """
    Attempt to replace intersection_poly in each candidate record using point-based
    geometry logic. Updates records to use 'pt' objects for P and Q.
    """
    fi = result.get("fi")
    if fi is None:
        return result

    f_list = [int(c) % p for c in f_coeffs]

    # 1. Normalize source point P
    P = (int(pt_src[0]) % p, int(pt_src[1]) % p)

    # Verify P is on the curve using index access
    fxP = sum(c * pow(P[0], i, p) % p for i, c in enumerate(f_list)) % p
    if fxP != (P[1] * P[1] % p):
        raise ArithmeticError(f"augment_with_phi: P={P} is not on curve")

    candidates: List[Dict[str, Any]] = list(
        result.get("candidate_records") or result.get("candidates") or []
    )
    snapshot = list(candidates)

    for rec in snapshot:
        if not isinstance(rec, dict) or rec.get("source") in {"phi_generic_r", "phi_self_rs"}:
            print("what up")
            continue

        m_val = rec.get("m")
        g_coeffs_eval = _eval_fi_at_m(fi, m_val, p)
        if g_coeffs_eval is None:
            print("here2")
            continue

        # 2. Extract and normalize candidate x-coordinate
        pt_step = rec.get("pt_step")
        if pt_step is None:
            print(rec)
            continue

        # ---------------------------------------------------------------
        # Self geometry: P = Q[cite: 15]
        # ---------------------------------------------------------------
        if pt_step == P:
            print("here")
            Q_self = P
            try:
                A_coeffs, c, R_mumford = compute_phi(p, f_list, g_coeffs_eval, P, Q_self)
                _apply_phi_to_record(rec, A_coeffs, c, R_mumford, P, Q_self, "self", sage_ring, f_list, p)

                # Inject synthetic RS points as point objects
                _inject_rs_candidates(candidates, rec, R_mumford, f_list, p, sage_ring, "self_rs")
            except (ValueError, ZeroDivisionError, ArithmeticError):
                raise
                continue
            continue

        # ---------------------------------------------------------------
        # Generic / conjugate geometry: Recover y and build Q[cite: 15]
        # ---------------------------------------------------------------
        y_canonical = _recover_y(pt_step, f_list, p)
        if y_canonical is None:
            continue

        y_used = None
        for y_try in (y_canonical, (p - y_canonical) % p):
            Q_try = (pt_step[0], y_try)
            try:
                A_coeffs, c, R_mumford = compute_phi(p, f_list, g_coeffs_eval, P, Q_try)
                y_used = y_try
                _apply_phi_to_record(rec, A_coeffs, c, R_mumford, P, Q_try, "generic", sage_ring, f_list, p)
                _inject_rs_candidates(candidates, rec, R_mumford, f_list, p, sage_ring, "generic_rs")
                break
            except ValueError:
                raise
                continue
            except (ZeroDivisionError, ArithmeticError):
                raise
                break

    # Final cleanup and write-back[cite: 15]
    result["candidate_records"] = _dedupe_records(candidates)
    result["candidates"] = result["candidate_records"]
    return result

def _apply_phi_to_record(rec, A_coeffs, c, R_m, P, Q, geo, ring, f_list, p):
    """Updates a record with phi metadata using point tuples."""
    h_coeffs = phi_quintic(p, f_list, A_coeffs, c)
    rec["intersection_poly"] = ring(h_coeffs)
    rec["phi_P"] = P  # Now a tuple
    rec["phi_Q"] = Q  # Now a tuple
    rec["phi_geo"] = geo
    if R_m and isinstance(R_m[0], tuple):
        rec["phi_mumford_RS"] = [int(R_m[0][0]) % p, int(R_m[0][1]) % p]

def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupes candidate records to prevent the walker from evaluating
    the same point multiple times in a single step.
    """
    seen_points = set()
    unique_records = []

    for rec in records:
        # Prioritize the full 'pt' tuple if available, fallback to 'pt_step'
        pt = rec.get("pt")
        if pt is None:
            pt = rec.get("pt_step")

        if pt not in seen_points:
            seen_points.add(pt)
            unique_records.append(rec)

    return unique_records

def _inject_rs_candidates(candidates, base_rec, R_m, f_list, p, ring, source_tag):
    """
    Calculates roots for R and S and adds them as full point records.

    DEFINITIVE BUG FIX: Strips all phi-related metadata inherited from base_rec
    to prevent the "double stuffer" bug (8 atoms) in walker_step_search.py.
    """
    if not (R_m and isinstance(R_m[0], tuple)):
        return

    sum_RS, prod_RS = R_m[0]
    for xrs in _mumford_roots(int(sum_RS) % p, int(prod_RS) % p, p):
        y_rs_canonical = _recover_y(xrs, f_list, p)
        if y_rs_canonical is None:
            continue

        # Inject both y-branches so the walker can find the valid direction
        for y_rs in (y_rs_canonical, (p - y_rs_canonical) % p):
            full_pt = (int(xrs), int(y_rs))

            # Create a shallow copy of the base record metadata
            new_rec = dict(base_rec)

            # --- THE FIX: STRIP ALL PHI GEOMETRY ---
            # We must remove everything that would trigger _get_geometry_metadata
            # into thinking this is still a 5-point phi-step.
            phi_keys = [
                "intersection_poly",
                "phi_P",
                "phi_Q",
                "phi_geo",
                "phi_mumford_RS",
                "_validated_atoms" # Clear any previous validation cache
            ]
            for key in phi_keys:
                if key in new_rec:
                    del new_rec[key]

            # Set the authoritative point fields for a standard step
            new_rec["pt_step"] = full_pt
            new_rec["pt"] = full_pt
            new_rec["pt_res"] = full_pt

            # Label the source for debugging
            new_rec["source"] = "phi_self_rs" if "self" in source_tag else "phi_generic_r"

            candidates.append(new_rec)
