"""dlp_anchor_injection.py  –  markov/walker/dlp_anchor_injection.py

Injects synthetic phi-relations that tie the DLP base (G) and target (Q)
divisor atoms into the relation matrix on every committed Markov step.

Background
----------
The Markov walk generates relations of the form

    2·walk_pt + candidate + R + S − 5·∞ = 0

The walk-head (walk_pt) rotates through the factor base, but G and Q
never appear as *atoms* unless explicitly injected.  Without their columns
in the relation matrix the nullspace carries no information about the DLP
exponent n in n·G = Q, so the solve always fails.

The fix: after every committed step, attempt to build

    phi with  P = G_atom_i  (double root),  Q = current_walk_pt  (simple zero)

and the analogous relation with P = Q_atom_j.  Each success stores a
RelationRecord whose atoms include one G or Q x-coordinate, giving those
coordinates columns in the matrix.

Y-branch correctness
--------------------
The Mumford representation D = (u(x), v(x)) encodes the divisor as the
formal sum of points (r, v(r)) for each root r of u(x).  This is the
*only* geometrically correct y for each atom — picking the canonical
(smaller) square-root branch is generally wrong.

set_dlp_points() therefore requires the caller to pass the Sage Jacobian
elements BASE_DIVISOR and TARGET_DIVISOR (not just the x-set), extracts
atoms as (x, v(x)) pairs, and stores them on the walker.

API
---
    # Once, after constructing the walker:
    walker.set_dlp_points(BASE_DIVISOR, TARGET_DIVISOR)

    # That's it.  _inject_dlp_relations() is called automatically from
    # _commit_step() in walker_step_search.py.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..walkerclass import Genus2MetropolisWalker, RelationRecord


# ---------------------------------------------------------------------------
# Public: attach DLP points to walker
# ---------------------------------------------------------------------------

def set_dlp_points(walker, base_divisor, target_divisor) -> None:
    """
    Parse BASE_DIVISOR and TARGET_DIVISOR (Sage Jacobian elements) into
    concrete (x, y) atom tuples using the Mumford v-polynomial for y, then
    store them on the walker.

    Parameters
    ----------
    walker          : Genus2MetropolisWalker
    base_divisor    : Sage element of J(C)(F_p), the G in n·G = T
    target_divisor  : Sage element of J(C)(F_p), the T in n·G = T

    Sets
    ----
    walker._dlp_base_atoms   : list of (x, y) tuples from G's Mumford decomposition
    walker._dlp_target_atoms : list of (x, y) tuples from T's Mumford decomposition
    walker._dlp_base_div     : base_divisor  (kept for diagnostics)
    walker._dlp_target_div   : target_divisor
    """
    Fp = walker.base_ring

    def _mumford_atoms(D, label: str) -> List[Tuple]:
        """Extract (x, v(x)) pairs from a Sage Jacobian element D."""
        if D is None or D.is_zero():
            raise ValueError(
                f"set_dlp_points: {label} divisor is zero or None — "
                "cannot extract atoms."
            )
        u_poly = D[0]   # Mumford u(x): product of (x - x_i)
        v_poly = D[1]   # Mumford v(x): y = v(x) at each root of u

        roots_with_mult = u_poly.roots()  # list of (root, multiplicity)
        if not roots_with_mult:
            raise ValueError(
                f"set_dlp_points: {label} u-polynomial has no F_p roots — "
                f"u(x) = {u_poly}.  The divisor does not have split reduction."
            )

        atoms = []
        for x_root, mult in roots_with_mult:
            x_fp = Fp(x_root)
            # v(x_root) is the correct y-coordinate for this atom.
            # Do NOT use sqrt(f(x)) — that picks the canonical branch which
            # may be the wrong sign.
            y_fp = Fp(v_poly(x_root))

            # Sanity: point must be on the curve.
            rhs = walker.curve_poly(x_fp)
            if rhs != y_fp * y_fp:
                raise ArithmeticError(
                    f"set_dlp_points: {label} atom ({x_fp}, {y_fp}) "
                    f"is not on the curve (f(x)={rhs}, y²={y_fp*y_fp}).  "
                    "This is a bug in the Mumford v-polynomial evaluation."
                )
            if y_fp == Fp(0):
                raise ValueError(
                    f"set_dlp_points: {label} atom ({x_fp}, 0) is a "
                    "Weierstrass point — phi is undefined there."
                )

            for _ in range(int(mult)):
                atoms.append((x_fp, y_fp))

        if not atoms:
            raise ValueError(
                f"set_dlp_points: {label} produced zero atoms — "
                "check that the divisor has degree ≥ 1."
            )

        return atoms

    walker._dlp_base_atoms   = _mumford_atoms(base_divisor,   "BASE_DIVISOR")
    walker._dlp_target_atoms = _mumford_atoms(target_divisor, "TARGET_DIVISOR")
    walker._dlp_base_div     = base_divisor
    walker._dlp_target_div   = target_divisor

    if getattr(walker.config, 'verbose', True):
        print(
            f"[dlp_anchor] Registered DLP points.\n"
            f"  G atoms: {[(int(x), int(y)) for x, y in walker._dlp_base_atoms]}\n"
            f"  T atoms: {[(int(x), int(y)) for x, y in walker._dlp_target_atoms]}"
        )


# ---------------------------------------------------------------------------
# Internal: attempt one synthetic phi-relation for a single anchor atom
# ---------------------------------------------------------------------------

def _try_phi_for_anchor(
    walker,
    anchor: Tuple,       # (x, y) — the DLP atom acting as P (double root)
    walk_pt: Tuple,      # (x, y) — current walk point acting as Q (simple zero)
    n: int,
    anchor_label: str,   # "G" or "T" for logging
) -> bool:
    """
    Attempt to build phi with P=anchor (double root) and Q=walk_pt (simple zero).

    Tries both y-branches for walk_pt (the phi consistency equation φ(Q)=0
    fixes the branch; one will succeed, one will raise ValueError).

    Returns True if a relation was stored, False otherwise.
    """
    # Lazy import to avoid circular deps.
    try:
        from .phi import compute_phi, phi_quintic
    except ImportError:
        raise

    # Also need _build_atoms from walker_step_search — import lazily.
    try:
        from .walker_step_search import _build_atoms
    except ImportError:
        raise

    p        = int(walker.p)
    f_coeffs = [int(c) % p for c in walker.curve_poly.list()]
    g_coeffs = f_coeffs   # unused by compute_phi, kept for API compat
    Fp       = walker.base_ring

    P = (int(anchor[0]) % p, int(anchor[1]) % p)

    # Guard: anchor must not be a Weierstrass point (y=0).
    if P[1] == 0:
        return False

    Q_x = int(walk_pt[0]) % p

    # Guard: walk_pt must not coincide with the anchor's x (would be conjugate
    # or self geometry — valid, but we handle it generically; compute_phi
    # dispatches internally).

    # Recover both y-branches for walk_pt at Q_x.
    # We try the branch stored in walk_pt first (it came from the walker's
    # current state and is correct for the walk), then the conjugate branch.
    walk_y = int(walk_pt[1]) % p
    branches = [walk_y]
    conj = (p - walk_y) % p
    if conj != walk_y:
        branches.append(conj)

    ring = walker.curve_poly.parent()

    for y_try in branches:
        Q = (Q_x, y_try)

        try:
            A_coeffs, c_phi, R_mumford = compute_phi(p, f_coeffs, g_coeffs, P, Q)
        except ValueError:
            # Wrong y-branch for Q, or degenerate geometry — try the other branch.
            continue
        except (ZeroDivisionError, ArithmeticError):
            # Degenerate (Weierstrass point etc.) — not recoverable by branch flip.
            return False

        h_coeffs = phi_quintic(p, f_coeffs, A_coeffs, c_phi)

        try:
            h_sage = ring(h_coeffs)
        except Exception:
            raise

        anchor_fp  = (Fp(P[0]), Fp(P[1]))
        walk_pt_fp = (Fp(Q[0]), Fp(Q[1]))

        # _build_atoms handles the branch-enumeration for the residual roots
        # (R, S) by trying all 2^k combinations and checking principality.
        atoms = _build_atoms(
            walker,
            anchor_fp,    # pt_src = DLP anchor (double root)
            walk_pt_fp,   # pt_step = current walk point (simple zero)
            None,         # pt_res = unknown; recovered from poly
            h_sage,       # authoritative intersection poly
            src_mult=None,
            extra_roots=None,
        )
        if not atoms:
            continue

        if not walker._verify_atoms_principal(atoms):
            continue

        # Build and store the RelationRecord.
        step_meta = {
            "source":           "dlp_anchor_injection",
            "anchor_label":     anchor_label,
            "anchor_pt":        (int(P[0]), int(P[1])),
            "walk_pt":          (int(Q[0]), int(Q[1])),
            "_validated_atoms": atoms,
        }

        rec = walker._make_relation(
            step_index=len(walker.history),
            n=n,
            pt_src=anchor_fp,
            m_val=None,
            pt_step=walk_pt_fp,
            pt_res=None,
            step_metadata=step_meta,
            accepted=True,
            restart=False,
            src_mult=2,
        )
        walker._store_record(rec)

        if getattr(walker.config, 'verbose', True):
            print(
                f"  [dlp_inject] {anchor_label}=({int(P[0])},{int(P[1])})  "
                f"walk=({int(Q[0])},{int(Q[1])})  "
                f"n_atoms={len(atoms)}"
            )
        return True

    return False


# ---------------------------------------------------------------------------
# Public: called from _commit_step in walker_step_search.py
# ---------------------------------------------------------------------------

def inject_dlp_relations(walker, committed_rec: "RelationRecord", n: int) -> None:
    """
    After committing a normal Markov step, attempt synthetic phi-relations
    that tie each G-atom and each T-atom into the factor base.

    Called unconditionally from _commit_step; exits immediately if
    set_dlp_points() has not been called.

    For each anchor in (G_atom_0, G_atom_1, T_atom_0, T_atom_1):
        - Try phi with P=anchor (double root), Q=current walk point.
        - Store the relation if phi succeeds and the divisor is principal.
        - Failures are silent (not every walk point produces a valid phi
          with every anchor).

    The injected records carry source="dlp_anchor_injection" and are stored
    in walker.history exactly like normal relations.  The relation matrix
    will therefore have columns for the G and T x-coordinates.
    """
    if not getattr(walker, '_dlp_base_atoms', None):
        return
    if not getattr(walker, '_dlp_target_atoms', None):
        return

    walk_pt = (walker.current_x, walker.current_y)

    anchors = (
        [("G", a) for a in walker._dlp_base_atoms] +
        [("T", a) for a in walker._dlp_target_atoms]
    )

    for label, anchor in anchors:
        # Skip if the anchor IS the walk point — the self geometry produces
        # a degree-6 poly (4P+R+S−6∞) which violates the curve_degree=5
        # invariant checked in _make_relation.  We only want generic geometry
        # (2P+Q+R+S−5∞, degree 5).
        if anchor[0] == walk_pt[0]:
            continue

        _try_phi_for_anchor(walker, anchor, walk_pt, n, label)
