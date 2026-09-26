"""
height_bound.py: Derive a per-vector rational-reconstruction height bound
from the Mordell-Weil canonical height pairing on E, replacing the flat
HEIGHT_BOUND = 100*370 constant currently used everywhere (search_common.py
line 54, "not that important, mostly, it seems").

--- Why this exists ---

The old HEIGHT_BOUND is a single number used for every vector `v_orig`
regardless of how large the section-multiple n = v_orig actually is. It's
also disconnected from the two places that are supposed to agree on a
bound for m's numerator/denominator:

  1. Stage 2 CRT prefilter (modularthread.py, currently `if False`-disabled):
     _pairwise_crt_survivors / lattice_rational_lift_exists, which enforces
     |r| <= H and |s| <= H on the CRT-lifted class.
  2. The acceptance path's rational_reconstruct(m0 % M, M) call, which
     currently passes no max_den at all (defaults to sqrt(M/2)).

Because these two bounds didn't agree, turning Stage 2 back on silently
dropped real points (see the block comment above the `if False` in
modularthread.py). Reconciling them against the *same* flat constant would
just make them wrong together. This module computes a bound that is
actually derived from the curve's arithmetic, so reconciling the two call
sites against *this* value is a real fix rather than a coincidence.

--- The math ---

For P in E(QQ), the canonical (Neron-Tate) height satisfies
    h_hat([n]P) = n^2 * h_hat(P)
and more generally, for a lattice vector v = (n_1, ..., n_k) combining
independent generators P_1, ..., P_k,
    h_hat(sum n_i P_i) = v^T H v
where H is the height-pairing matrix (E.height_pairing_matrix(sections)).

The canonical height differs from the naive (Weil) height on the
x-coordinate by a bounded, effectively computable amount:
    |h_hat(Q) - h(Q)| <= c
where c is a constant depending only on the curve (Silverman's bound,
E.silverman_height_bound() in Sage). So:
    h(x([v]P)) <= v^T H v + c

The naive height h(x) here is log(max(|numerator|, |denominator|)) of x
in lowest terms (Sage's convention: RationalField element .height() uses
log of the max of |num|,|den|; confirm this matches whatever normalization
current_sections/x-coordinates use before trusting the constant below).
Exponentiating gives a bound on both |numerator| and |denominator| of the
section's x-coordinate:
    max(|num|, |den|) <= exp(v^T H v + c)

This is NOT yet a bound on m itself -- m is related to x([v]P) by the
fibration's parametrization (r_m / shift / any Mobius transform T), which
is generically a low-degree (often linear or Mobius) map. For a linear or
Mobius map the numerator/denominator growth is bounded by a further
constant multiplicative factor tied to the map's own coefficient heights;
that additional factor is NOT computed here and needs to be folded in at
the call site if r_m is not literally the identity. See
`vector_height_bound_for_m` below for where that hook goes.

--- Validation requirement, not optional ---

Before this bound is used to reject anything (i.e. before Stage 2 in
modularthread.py is flipped back on, and before rational_reconstruct's
max_den is wired to this value), it MUST be checked against every known
rational point in the test curves (DATA_PTS_GENUS2 / example_runs) to
confirm the computed bound is never smaller than what those known points
actually require. This mirrors the validation discipline already noted in
search_config.py for the old flat constant -- a wrong (too-tight) bound
here silently drops real points exactly like the old bug did, just via a
new, curve-derived route instead of an arbitrary one.
"""

from sage.all import log, exp, ceil, QQ, ZZ


def height_pairing_matrix(E, sections):
    """
    Thin wrapper around E.height_pairing_matrix(sections), so the rest of
    this module (and callers) don't need to import/construct it directly.

    Args:
        E: EllipticCurve over QQ (or a number field -- untested here,
           this module assumes QQ throughout).
        sections: list of independent rational points on E (your
                  current_sections).

    Returns:
        A symmetric matrix H (Sage matrix over RR/QQ, whichever
        height_pairing_matrix returns) with H[i][j] = <P_i, P_j> under the
        canonical height pairing. H[i][i] = h_hat(P_i).

    Caches nothing -- callers should compute this once per curve/section
    set and reuse it across every vector, since it's the expensive part
    (each pairing entry requires a canonical height computation).
    """
    return E.height_pairing_matrix(sections)


def silverman_bound(E):
    """
    Thin wrapper around E.silverman_height_bound(): the constant c such
    that |h_hat(Q) - h(Q)| <= c for every Q in E(QQ), where h is the naive
    (Weil) height on the x-coordinate.

    Computed once per curve; cheap relative to height_pairing_matrix.
    """
    return E.silverman_height_bound()


def canonical_height_of_vector(v, H):
    """
    h_hat(sum v_i P_i) = v^T H v, given the precomputed height pairing
    matrix H for the generators P_i.

    Args:
        v: tuple/list of integers (the lattice vector, i.e. v_orig).
        H: height pairing matrix from height_pairing_matrix(E, sections),
           same generator ordering as v.

    Returns:
        Non-negative real number (canonical height is positive-definite
        on E(QQ)/torsion, so this should never come out negative for a
        nonzero v modulo floating-point noise from Sage's numerical
        height computation -- callers should not rely on it being an
        exact rational).
    """
    n = len(v)
    total = 0
    for i in range(n):
        for j in range(n):
            total += v[i] * H[i][j] * v[j]
    return total


def naive_height_bound_for_point(v, H, c):
    """
    h(x([v]P)) <= v^T H v + c

    Returns the bound on the naive height of x([v]P), i.e. an upper bound
    on log(max(|numerator|, |denominator|)) of that x-coordinate in
    lowest terms. This is NOT yet a numerator/denominator bound on m --
    see vector_height_bound_for_m.
    """
    return canonical_height_of_vector(v, H) + c


def max_num_den_bound_for_point(v, H, c):
    """
    Exponentiates naive_height_bound_for_point to get an actual integer
    bound: max(|numerator|, |denominator|) <= this value, for the
    x-coordinate of [v]P itself (not yet m -- see below).

    Rounds up (ceil) since this must be a valid upper bound, not an
    approximation that could round down below the true requirement.
    """
    return int(ceil(exp(naive_height_bound_for_point(v, H, c))))


def vector_height_bound_for_m(v, H, c, m_map_height_factor=None):
    """
    The bound that should actually be passed as HEIGHT_BOUND /
    height_bound / max_den at both Stage-2 (lattice_rational_lift_exists)
    and the acceptance path (rational_reconstruct's max_den) for this
    specific vector v.

    Args:
        v: the lattice vector (v_orig).
        H: height_pairing_matrix(E, current_sections).
        c: silverman_bound(E).
        m_map_height_factor: REQUIRED unless r_m (the map from m to the
            section's x-coordinate, composed with `shift` and any Mobius
            transform T) is literally the identity. This must be an
            upper bound on how much the numerator/denominator of x can
            grow when inverting r_m to recover m -- i.e. if
                x = r_m(m) - shift   (optionally further composed with T)
            then this factor bounds
                max(|num(m)|, |den(m)|) <= m_map_height_factor *
                                            max(|num(x)|, |den(x)|)
            (or whatever the correct combination law is for your specific
            r_m -- for a linear map m -> a*m+b this is elementary; for a
            Mobius transform T it's the standard height bound for Mobius
            maps in terms of T's coefficients; for anything of higher
            degree this whole approach needs re-deriving per fibration,
            since the bound derived here is only proven for the
            x-coordinate of the section itself, not for m unless the
            m -> x map has bounded height distortion).

            Passing None is a deliberate hard stop rather than a silent
            wrong answer: silently assuming m_map_height_factor = 1 (i.e.
            that m and x([v]P) have identical numerator/denominator size)
            is exactly the kind of unvalidated assumption that dropped
            real points before. This function refuses to guess it.

    Returns:
        Integer bound suitable for both call sites, once
        m_map_height_factor is supplied and validated.
    """
    if m_map_height_factor is None:
        raise ValueError(
            "vector_height_bound_for_m: m_map_height_factor must be "
            "supplied -- see docstring. Do not default this to 1 without "
            "deriving it for your actual r_m/shift/T; that assumption is "
            "exactly what needs validating against known points before "
            "this bound is used to reject anything."
        )
    x_bound = max_num_den_bound_for_point(v, H, c)
    return int(ceil(m_map_height_factor * x_bound))


def build_vector_height_bounds(E, sections, vecs, m_map_height_factor):
    """
    Convenience entry point: compute H and c once, then return a dict
    {v_orig_tuple: bound} for every vector in vecs.

    Args:
        E: EllipticCurve over QQ.
        sections: current_sections (list of independent rational points).
        vecs: iterable of lattice vectors (each a tuple/list of ints,
              same length as sections).
        m_map_height_factor: see vector_height_bound_for_m -- passed
              through unchanged, still mandatory, still not defaulted.

    Returns:
        dict mapping tuple(v) -> int bound, for use as a drop-in
        per-vector replacement for the flat HEIGHT_BOUND constant at
        both modularthread.py call sites (Stage 2 prefilter and the
        acceptance-path rational_reconstruct max_den).

    NOT validated here -- see module docstring. Callers must check this
    against known points before wiring it into anything that rejects
    candidates.
    """
    H = height_pairing_matrix(E, sections)
    c = silverman_bound(E)
    bounds = {}
    for v in vecs:
        v_tuple = tuple(int(x) for x in v)
        bounds[v_tuple] = vector_height_bound_for_m(
            v_tuple, H, c, m_map_height_factor=m_map_height_factor
        )
    return bounds


def canonical_height_of_vector_matrix(v, H):
    """
    Same as canonical_height_of_vector, but takes H as anything indexable
    H[i][j] (a Sage matrix, as produced by compute_canonical_height_matrix
    in search_common.py for an elliptic *surface* / fibration). Kept as a
    separate name so it's clear this is the entry point that does NOT
    assume E.height_pairing_matrix()'s number-field convention.
    """
    return canonical_height_of_vector(v, H)


def empirical_c_from_known_points(H, known_vectors_and_m):
    """
    Derive the additive constant c bridging canonical height (v^T H v) and
    naive height (log(max(|num|,|den|)) of the numeric value of m) for this
    specific fibration, EMPIRICALLY from points already found -- rather
    than guessing a number, or assuming Silverman's number-field bound
    applies (it doesn't: cd.E_weier is a curve over the function field
    Frac(QQ[m]), not over QQ, so E.silverman_height_bound() doesn't exist
    and wouldn't mean the same thing here if it did -- that constant is
    specific to a fixed elliptic curve over a number field, not to a
    varying fiber of a surface).

    For an elliptic surface the naive-vs-canonical height discrepancy is
    still bounded (this is the content of the relevant height-comparison
    theorems for elliptic surfaces), but the bound depends on the surface's
    bad-fiber data in a way this module does not attempt to derive in
    closed form. Instead: take every known (v_orig, m_value) pair already
    found by the search, compute naive_height(m_value) - v^T H v for each,
    and use the max observed value (plus a safety margin) as c. This is
    honest about being an empirical lower bound on the true c, not a proof
    -- exactly like validate_against_known_points below, a clean result
    here is "no known counterexample", not "proven correct". It also means
    this function is USELESS until you have at least one or two known
    points to calibrate against (raises if given none), which is the
    correct failure mode: refusing to guess c out of thin air rather than
    silently defaulting to 0 (which would UNDER-estimate the true bound and
    risk rejecting real points, the same failure class this whole module
    exists to avoid).

    Args:
        H: height_pairing matrix (Shioda-Tate, from
           compute_canonical_height_matrix), same ordering as the vectors
           in known_vectors_and_m.
        known_vectors_and_m: list of (v_orig_tuple, m_value) pairs, each a
            point this search has already found and confirmed rational.

    Returns:
        (c, margin_used): c is the constant to pass to
        naive_height_bound_for_point / vector_height_bound_for_m; margin
        used is reported for logging/debugging.

    Raises:
        ValueError if known_vectors_and_m is empty -- there is nothing to
        calibrate against, so refuse rather than default to 0.
    """
    if not known_vectors_and_m:
        raise ValueError(
            "empirical_c_from_known_points: no known points supplied -- "
            "cannot calibrate c without at least one confirmed rational "
            "point for this fibration. Do not default this to 0; that "
            "would silently produce a too-tight bound (c=0 assumes the "
            "canonical height already dominates the naive height, which "
            "is not proven here) and risks rejecting real points, exactly "
            "the failure this module exists to prevent."
        )

    from sage.all import QQ as _QQ

    discrepancies = []
    for v_tuple, m_value in known_vectors_and_m:
        m_q = _QQ(m_value)
        if m_q == 0:
            naive_h = 0.0
        else:
            naive_h = float(log(max(abs(int(m_q.numerator())), abs(int(m_q.denominator())))))
        can_h = float(canonical_height_of_vector(v_tuple, H))
        discrepancies.append(naive_h - can_h)

    max_discrepancy = max(discrepancies)
    # Safety margin: known points are the only calibration data we have:
    # pad generously so a slightly-larger not-yet-found point doesn't get
    # rejected by an underestimated c. This is still not a proof (see
    # docstring above) -- it just makes an accidental too-tight bound less
    # likely for the *next* point of similar size, not for arbitrarily
    # larger ones.
    margin = max(1.0, 0.5 * abs(max_discrepancy))
    c = max_discrepancy + margin
    return c, margin


def build_vector_height_bounds_from_matrix(H, vecs, m_map_height_factor,
                                            known_vectors_and_m=None, c=None):
    """
    Like build_vector_height_bounds, but takes H directly (e.g. the
    Shioda-Tate matrix from compute_canonical_height_matrix) instead of an
    EllipticCurve to call .height_pairing_matrix()/.silverman_height_bound()
    on -- those Sage methods don't exist for a curve over a function field,
    which is what cd.E_weier actually is in this codebase's fibration
    search (see search_lll/search_main.py's _resolve_height_bound).

    Exactly one of `c` or `known_vectors_and_m` must be supplied:
      - Pass `c` directly if you've already derived/validated it.
      - Pass `known_vectors_and_m` to derive c empirically via
        empirical_c_from_known_points (see that function's docstring for
        why this refuses to run with zero known points rather than
        defaulting c to 0).

    Returns:
        dict {v_orig_tuple: bound}, same shape as build_vector_height_bounds.
    """
    if (c is None) == (known_vectors_and_m is None):
        raise ValueError(
            "build_vector_height_bounds_from_matrix: pass exactly one of "
            "c= or known_vectors_and_m= (not both, not neither) -- see "
            "docstring."
        )
    if c is None:
        c, _margin = empirical_c_from_known_points(H, known_vectors_and_m)

    bounds = {}
    for v in vecs:
        v_tuple = tuple(int(x) for x in v)
        bounds[v_tuple] = vector_height_bound_for_m(
            v_tuple, H, c, m_map_height_factor=m_map_height_factor
        )
    return bounds


def validate_against_known_points(bounds_dict, known_m_values):
    """
    Sanity check to run BEFORE trusting bounds_dict for anything that
    rejects candidates (i.e. before flipping Stage 2 back on, before
    passing max_den= this value in the acceptance path).

    Args:
        bounds_dict: {v_orig_tuple: bound} from build_vector_height_bounds.
        known_m_values: list of (v_orig_tuple, m_value) pairs -- known
            rational points already found by this pipeline (e.g. from
            DATA_PTS_GENUS2 / example_runs), each with the v_orig vector
            that produced them.

    Returns:
        list of failures: (v_orig_tuple, m_value, required, bound) for
        any known point whose actual numerator/denominator exceeds the
        computed bound for its vector. An empty list means every known
        point fits inside its bound -- necessary but not sufficient
        evidence the bound is safe (it only checks the points you already
        know about), so still treat a clean pass as "no known
        counterexample" rather than "proven safe".
    """
    failures = []
    for v_tuple, m_value in known_m_values:
        if v_tuple not in bounds_dict:
            continue
        m_q = QQ(m_value)
        required = max(abs(int(m_q.numerator())), abs(int(m_q.denominator())))
        bound = bounds_dict[v_tuple]
        if required > bound:
            failures.append((v_tuple, m_value, required, bound))
    return failures
