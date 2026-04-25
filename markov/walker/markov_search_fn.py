from search_common import *
from .tower_context import *
from search_lll.ll_utilities import *
from search_lll.search_main import *
from .candidate_utils import *
from .candidate_utils import _normalize_markov_mumford_result
from .candidate_utils import _candidates_from_residues
from .enrichment import *

def make_project_markov_search_fn(
    *,
    coeffs_genus2=None,
    base_points=None,
    p: Optional[int] = None,
    prime_pool=None,
    sconf=None,
    num_subsets: Optional[int] = None,
    num_workers: Optional[int] = None,
    debug: bool = False,
    max_n: int = 80,
    precomputed_residues=None,
    all_found_x=None, pool=None, chunk_size=8
):
    """Build a search function that rebuilds the tower for every current x."""
    coeffs_genus2 = coeffs_genus2 if coeffs_genus2 is not None else resolve_project_symbol('COEFFS_GENUS2', required=True)
    p = FINITE_FIELD
    prime_pool = PRIME_POOL
    num_workers = 20
    all_found_x = set()

    def search_fn(x_src=None, current_x=None, n=None, seed=None, current_point=None, walker=None, **kwargs):
        yfun = get_y_unshifted_genus2
        if current_point is not None and isinstance(current_point, (tuple, list)) and len(current_point) >= 2:
            x_here, y_here = current_point[0], current_point[1]
        else:
            x_here = x_src if x_src is not None else current_x
            y_here = None
            if walker is not None and hasattr(walker, 'current_y'):
                y_here = walker.current_y
            if y_here is None:
                if yfun is not None:
                    y_here = yfun(x_here)
        assert yfun(x_here) == y_here, (yfun(x_here), y_here, current_point, current_x)

        ctx = build_project_tower_context_for_point(
            x_here,
            y_here,
            coeffs_genus2=coeffs_genus2,
            base_points=base_points if base_points is not None else getattr(walker, 'base_points', None),
            p=p,
            debug=debug,
        )

        assert ctx['search_rhs_list'] != ['INF'], ctx

        if debug:
            for k in ctx:
                print(k, ctx[k])

        n0 = int(n or 1)
        vecs = generate_ff_search_vectors(1)
        if not vecs:
            vecs = [tuple([n0])]

        raw = _run_markov_mumford_search_for_point(
            cd=ctx['cd'],
            current_sections=ctx['current_sections'],
            prime_pool=prime_pool,
            vecs=vecs,
            rhs_list=ctx['search_rhs_list'],
            shift=ctx['shift'],
            rationality_test_func=ctx['testfunc'],
            coeffs_genus2=coeffs_genus2,
            tower_data=ctx['tower_for_mumford'],
            num_workers=num_workers,
            debug=debug,
            x_b=None,
            shifted_coeffs=None, pool=pool, chunk_size=chunk_size
        )

        norm = _normalize_markov_mumford_result(raw, fallback_step=ctx)

        # Override candidate_records with sign-aware records extracted directly
        # from the raw residues dict, where each solution now carries yj_sign
        # and the v-polynomial coefficients needed to compute yk_sign.
        _raw_residues = norm.get('residues') if isinstance(norm, dict) else None
        if _raw_residues:
            sign_records = _candidates_from_residues(_raw_residues, p)
            if sign_records:
                norm['candidate_records'] = sign_records
                norm['candidates'] = sign_records
                norm['candidate_xs'] = {r['x_step'] for r in sign_records}

        # Fertility: fraction of n-values (vecs) that had at least one F_p root across any prime.
        # precomputed_residues[p] is keyed only by v_tuples that had roots, so union of keys = fertile set.
        # We try two possible shapes: {prime: {vtup: solutions}} and the flat {vtup: solutions} fallback.
        _precomp = norm.get('precomputed_residues') or (raw.get('precomputed_residues') if isinstance(raw, dict) else None)
        if _precomp and vecs:
            fertile_vtups = set()
            per_n_root_counts: Dict[str, int] = {}  # vtup_str -> number of F_p roots for that n
            # Shape A: {prime: {vtup: solutions}}  (expected from markov_mode search)
            for p_entry in _precomp.values():
                if isinstance(p_entry, dict):
                    fertile_vtups.update(p_entry.keys())
                    for vtup, solutions in p_entry.items():
                        k = str(vtup)
                        if isinstance(solutions, (list, tuple, set)):
                            cnt = len(solutions)
                        elif solutions is not None:
                            cnt = 1
                        else:
                            cnt = 0
                        # Take max across primes (FF mode has 1 prime; guard for multi-prime callers)
                        if cnt > 0:
                            per_n_root_counts[k] = max(per_n_root_counts.get(k, 0), cnt)
            # Shape B: flat {vtup: solutions} — if Shape A found nothing
            if not fertile_vtups:
                for k, v in _precomp.items():
                    if not isinstance(v, dict):
                        fertile_vtups.add(k)
                        per_n_root_counts[str(k)] = 1
            norm['n_with_roots'] = len(per_n_root_counts)
            norm['n_total'] = len(vecs)
            norm['total_roots'] = sum(per_n_root_counts.values())
            norm['per_n_roots'] = per_n_root_counts  # {vtup_str: root_count}
        else:
            # Fallback filled in below, after enriched_candidates is assembled.
            norm['n_with_roots'] = None
            norm['n_total'] = len(vecs) if vecs else None
            norm['total_roots'] = None

        # Grab ingredients for x_res computation.
        # f_i is in R_xm = PolynomialRing(Frac(GF(p)['m']), 'x') — poly in x with rational-function-in-m coefficients.
        # shifted_G_poly is the curve poly in x over GF(p).
        # At a specific m_val: evaluate each coeff of f_i at m=m_val -> univariate poly in x over GF(p).
        # Then G(x) - f_i(x, m_val) = 0 has roots x_src(x3), x_step, x_res.
        _G_poly = ctx.get('shifted_G_poly')
        _tower = ctx.get('primary_tower')
        _fi = None
        if _tower and isinstance(_tower, (list, tuple)) and len(_tower) > 0:
            last = _tower[-1]
            if isinstance(last, dict):
                _fi = last.get('f_i')

        # Curve degree from project globals, defaulting to 5.
        _curve_degree = int(resolve_project_symbol('CURVE_DEGREE', default=5))

        # enrich_candidates handles: degenerate-x_step skip, m recovery, x_res computation
        # via compute_xk_from_fiber, yk_sign computation from v(x_res), yj_sign/yk_sign
        # defaults, and x_res_head injection with roles swapped.  It replaces the old
        # inline loop which computed x_res but never computed yk_sign.
        enriched_candidates = enrich_candidates(
            norm,
            x_here=x_here,
            y_here=y_here,
            n0=n0,
            fi=_fi,
            G_poly=_G_poly,
            curve_degree=_curve_degree,
            p=p,
            shift=ctx.get('shift', 0),
            T=ctx.get('T'),
            T_inv=ctx.get('T_inv'),
        )

        # Attach S_of_m and inter_sym to every record while the tower context is
        # still available.  This is a fibration property of x_src (not of any x_step), so
        # we compute once and stamp it on all records.
        if _fi is not None and _G_poly is not None:
            _S_of_m_rec, _inter_sym_rec = compute_S_of_m(_fi, _G_poly, _curve_degree)
            for rec in enriched_candidates:
                if isinstance(rec, dict):
                    rec.setdefault('S_of_m', _S_of_m_rec)
                    rec.setdefault('inter_sym', _inter_sym_rec)
        # candidate_xs is the set of x_step values derived from actual m-roots only.
        # x_res_head records must be excluded here so the leaf-tracking in
        # _step_from_candidate_search can separate xj_set (m-root derived) from
        # xk_set (Vieta derived).  x_res_head entries are still in enriched_candidates
        # and eligible for the Metropolis chooser.
        candidate_xs = {
            c.get('x_step') for c in enriched_candidates
            if isinstance(c, dict)
            and c.get('x_step') is not None
            and c.get('source') != 'x_res_head'
        }

        # Dead-end reason classification — emitted into the result dict so the
        # walker can log *why* a step produced no candidates rather than silently
        # restarting.  Distinguishes failure modes:
        #   no_roots    — Mumford search found zero F_p roots for all vecs
        #   torsion     — roots found but all equal x_src (x_src is torsion / Weierstrass pt)
        #   ok          — candidates available
        # Note: all_inf_xk is no longer a reachable case — compute_xk_from_fiber now
        # raises AssertionError on a fiber pole rather than returning "∞".
        _dead_end_reason = 'ok'
        if not enriched_candidates:
            raw_candidate_records = norm.get('candidate_records', []) or []
            if not raw_candidate_records:
                _dead_end_reason = 'no_roots'
            else:
                xjs = [r.get('x_step') for r in raw_candidate_records if isinstance(r, dict)]
                xjs_nondegenerate = [x_step for x_step in xjs if x_step is not None and x_step != x_here]
                if not xjs_nondegenerate:
                    _dead_end_reason = 'torsion'
                else:
                    _dead_end_reason = 'no_roots'  # shouldn't happen; fallback

        # Deferred fertility fallback: if precomputed_residues wasn't available, use
        # the m-root-derived candidate count as a lower-bound proxy.
        # Do NOT include x_res_head records in this count — they are not m-roots.
        if norm.get('n_with_roots') is None and vecs:
            n_mroot_cands = sum(
                1 for c in enriched_candidates
                if isinstance(c, dict) and c.get('source') != 'x_res_head'
            )
            if n_mroot_cands > 0:
                norm['n_with_roots'] = min(n_mroot_cands, len(vecs))
                norm['n_total'] = len(vecs)
                norm['total_roots'] = n_mroot_cands
                norm['per_n_roots'] = {}  # per-vec provenance not available without precomputed_residues

        n_xk_head = sum(1 for c in enriched_candidates
                        if isinstance(c, dict) and c.get('source') == 'x_res_head')

        S_of_m_step, _ = compute_S_of_m(_fi, _G_poly, _curve_degree) if _fi is not None else (None, None)
        result = {
            'candidates': enriched_candidates,
            'candidate_records': enriched_candidates,
            'candidate_xs': candidate_xs,       # x_step-only (m-root derived), for leaf tracking
            'n_xk_head': n_xk_head,             # how many x_res_head alternatives were injected
            'stats': norm.get('stats', None),
            'found_xs': norm.get('found_xs', set()),
            'input_n': n0,
            'S_of_m': S_of_m_step,   # fibration property of this x_src, not of any x_step
            'fi': _fi,               # symbolic fiber poly in x over Frac(Fp[m]); needed for synthetic injection
            'G_poly': _G_poly,       # curve poly in x over Fp; needed for synthetic injection
            'vecs': vecs,
            'n_with_roots': norm.get('n_with_roots', None),
            'n_total': norm.get('n_total', None),
            'total_roots': norm.get('total_roots', None),
            'per_n_roots': norm.get('per_n_roots', None),
            'dead_end_reason': _dead_end_reason,
            # Memory Fix: Omit 'context', 'raw_mumford_residues', 'new_sections', 'precomputed_residues'
            # which hold uncollectable SageMath Rings and Ideals.
        }

        return result

    return search_fn


def _run_markov_mumford_search_for_point(
    *,
    cd,
    current_sections,
    prime_pool,
    vecs,
    rhs_list,
    shift,
    rationality_test_func,
    coeffs_genus2,
    tower_data,
    num_workers,
    debug=False,
    x_b=None,
    shifted_coeffs=None,
    pool=None, chunk_size=8
):
    """Call the legacy Mumford search and normalize the result for Markov use.

    The important thing here is to keep the raw payload available, because the
    smarter chooser wants provenance such as which n/vector produced each x_step.
    """
    raw = run_mumford_search(
        cd, current_sections, prime_pool, vecs, rhs_list, shift,
        rationality_test_func, coeffs_genus2, tower_data,
        num_workers, debug, x_b, shifted_coeffs, markov_mode=True, pool=pool, chunk_size=chunk_size
    )
    return _normalize_markov_mumford_result(raw)
