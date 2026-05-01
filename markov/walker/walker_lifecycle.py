from search_common import *
from collections import Counter
from .candidate_utils import _get_fiber_context_for_rec, _get_S_of_m_for_rec

class _FiberPoleError(Exception):
    """Raised when S(m) has a pole at a specific m value.

    A zero denominator in S(m) means the secant-line fiber is degenerate
    (the rational function parametrisation breaks down) at that particular m.
    This is not a 2-cycle violation; the (x_src, x_step) pair is simply unevaluable
    and should be skipped rather than treated as an error.
    """

def close_under_involution2(walker) -> int:
    """Verifies T(T(x_step)) == x_step using canonical atom multiplicities."""
    if not RLINEAR:
        return 0

    deg = walker.config.curve_degree
    n_checked = 0
    seen_pairs = set()

    for rec in walker.history:
        # 1. Validation & Atom extraction
        if not rec.accepted or not getattr(rec, 'atoms', None):
            continue
        if _is_involution(rec): # Skip closure sentinels
            continue

        x_src_fp = walker.base_ring(rec.x_src)
        # Canonical multiplicity from atoms
        src_mult = sum(1 for a in rec.atoms if walker.base_ring(a) == x_src_fp)

        # Involution check only defined for (deg-2) non-src roots
        if src_mult != deg - 2:
            continue

        S_sym = _get_S_of_m_for_rec(rec)
        if S_sym is None:
            continue

        # 2. Check each candidate in the pool
        for cand in (rec.candidate_pool or []):
            xj_val = cand.get('x_step') if isinstance(cand, dict) else cand
            if xj_val is None or xj_val == rec.x_src:
                continue

            key = (rec.x_src, xj_val)
            if key in seen_pairs: continue
            seen_pairs.add(key)

            try:
                # Perform the T(T(x)) check
                partner = _eval_T(walker, S_sym, rec.x_src, xj_val, src_mult)
                roundtrip = _eval_T(walker, S_sym, rec.x_src, partner, src_mult)

                if roundtrip != walker.base_ring(xj_val):
                    raise AssertionError(f"Involution violation at x_src={rec.x_src}")
                n_checked += 1
            except _FiberPoleError:
                continue

    return n_checked

def generate_mixed_relations2(walker, atoms_to_inject: List[Any], label: str = "mixed") -> int:
    """Injects new atoms while maintaining the degree-sum invariant."""
    Fp = walker.base_ring
    deg = walker.config.curve_degree
    n_added = 0

    for rec in list(walker.history):
        if not rec.accepted or not getattr(rec, 'atoms', None):
            continue

        x_src = Fp(rec.x_src)
        src_mult = sum(1 for a in rec.atoms if Fp(a) == x_src)

        # Fiber must be exactly (deg-2)*x_src + x_step + x_res
        if src_mult != deg - 2:
            continue

        fi, G_poly_rec = _get_fiber_context_for_rec(rec)
        for xj_val in [Fp(a) for a in atoms_to_inject]:
            if xj_val == x_src: continue

            try:
                # Calculate x_res from the fiber intersection
                xk_val, _ = compute_xk_from_fiber(x_src, x_src - xj_val, xj_val, fi, G_poly_rec, deg)
                if xk_val is None or not walker.curve_poly(xk_val).is_square():
                    continue

                # Update bookkeeping and build record
                new_atoms = [x_src] * src_mult + [xj_val, xk_val]
                injected_rec = _create_injected_record(walker, rec, xj_val, xk_val, new_atoms, label)
                walker._store_record(injected_rec)
                n_added += 1
            except Exception:
                continue

    return n_added

def try_partial_cantor_reduction(walker, rec: RelationRecord) -> bool:
    """Mutates rec.atoms via Jacobian addition and splits roots over F_p."""
    Fp = walker.base_ring
    if len(rec.atoms or []) < 2: return False

    # 1. Randomly sample 2 atoms and their y-coordinates
    idx1, idx2 = walker.rng.sample(range(len(rec.atoms)), 2)
    a1, a2 = Fp(rec.atoms[idx1]), Fp(rec.atoms[idx2])
    y1, y2 = walker._recover_y(a1), walker._recover_y(a2)

    if y1 is None or y2 is None: return False

    # 2. Add on Jacobian: (a1, y1) + (a2, y2)
    try:
        J = HyperellipticCurve(walker.curve_poly).jacobian()(Fp)
        u_poly = (J(C([a1, y1])) + J(C([a2, y2])))[0]
        roots = u_poly.roots()

        if sum(m for _, m in roots) != u_poly.degree():
            return False # Doesn't split over Fp

        # 3. Reconstruct atoms list
        new_atoms = _replace_atoms(rec.atoms, [a1, a2], roots)
        rec.atoms = new_atoms
        _sync_navigation_fields(rec) # Updates x_step/x_res
        return True
    except Exception:
        raise
        return False
