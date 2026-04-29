from search_common import *
from .tower_context import *
from search_lll.ll_utilities import *
from search_lll.search_main import *
from .candidate_utils import *
from .enrichment import *
from .phi_search import augment_with_phi as _augment_with_phi
from .candidate_utils import _normalize_markov_mumford_result, _candidates_from_residues

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

    # Build once at factory time for augment_with_phi.
    # coeffs_genus2 is high-degree first (descending); phi.py wants low-degree first.
    from sage.all import GF as _GF, PolynomialRing as _PR
    _phi_Fp       = _GF(int(p))
    _phi_ring     = _PR(_phi_Fp, 'x')
    _phi_f_coeffs = list(reversed(coeffs_genus2))

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
        _y_canonical = yfun(x_here)
        assert _y_canonical in (int(y_here) % p, (-int(y_here)) % p), (
            _y_canonical, y_here, current_point, current_x
        )

        # ------------------------------------------------------------------ #
        # 1. Build tower context for this point.                              #
        # ------------------------------------------------------------------ #
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

        # Extract fiber poly and curve poly from context up front so we can
        # diagnose None early rather than silently failing inside enrich_candidates.
        _G_poly = ctx.get('shifted_G_poly')
        _tower  = ctx.get('primary_tower')
        _fi     = None
        if _tower and isinstance(_tower, (list, tuple)) and len(_tower) > 0:
            last = _tower[-1]
            if isinstance(last, dict):
                _fi = last.get('f_i')

        if _fi is None:
            print(f"[search_fn] WARNING: _fi is None for x_here={x_here}  "
                  f"tower keys={list(last.keys()) if isinstance(last, dict) else 'not a dict'}  "
                  f"tower len={len(_tower) if _tower else 0}")
        if _G_poly is None:
            print(f"[search_fn] WARNING: _G_poly is None for x_here={x_here}  "
                  f"ctx keys={list(ctx.keys())}")

        _curve_degree = int(resolve_project_symbol('CURVE_DEGREE', default=5))

        # ------------------------------------------------------------------ #
        # 2. Run Mumford/Julia search.                                        #
        # ------------------------------------------------------------------ #
        n0   = int(n or 1)
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

        # ------------------------------------------------------------------ #
        # 3. Override candidates with sign-aware records from raw residues.   #
        # ------------------------------------------------------------------ #
        _raw_residues = norm.get('residues') if isinstance(norm, dict) else None
        if _raw_residues:
            sign_records = _candidates_from_residues(_raw_residues, p)
            if sign_records:
                norm['candidate_records'] = sign_records
                norm['candidates']        = sign_records
                norm['candidate_xs']      = {r['x_step'] for r in sign_records}

        # ------------------------------------------------------------------ #
        # 4. Compute fertility (n-values that produced at least one F_p root).#
        # ------------------------------------------------------------------ #
        _precomp = norm.get('precomputed_residues') or (
            raw.get('precomputed_residues') if isinstance(raw, dict) else None
        )
        if _precomp and vecs:
            per_n_root_counts: Dict[str, int] = {}
            # Shape A: {prime: {vtup: solutions}}  (markov_mode)
            for p_entry in _precomp.values():
                if isinstance(p_entry, dict):
                    for vtup, solutions in p_entry.items():
                        k = str(vtup)
                        cnt = (len(solutions) if isinstance(solutions, (list, tuple, set))
                               else (1 if solutions is not None else 0))
                        if cnt > 0:
                            per_n_root_counts[k] = max(per_n_root_counts.get(k, 0), cnt)
            # Shape B: flat {vtup: solutions} fallback
            if not per_n_root_counts:
                for k, v in _precomp.items():
                    if not isinstance(v, dict):
                        per_n_root_counts[str(k)] = 1
            norm['n_with_roots'] = len(per_n_root_counts)
            norm['n_total']      = len(vecs)
            norm['total_roots']  = sum(per_n_root_counts.values())
            norm['per_n_roots']  = per_n_root_counts
        else:
            norm['n_with_roots'] = None
            norm['n_total']      = len(vecs) if vecs else None
            norm['total_roots']  = None
            norm['per_n_roots']  = None

        # Snapshot pre-enrichment stub count for dead-end classification below.
        # This is the only reliable signal that Mumford/Julia found roots when
        # precomputed_residues is absent.
        _n_pre_enrich_stubs = len(norm.get('candidate_records') or norm.get('candidates') or [])

        # ------------------------------------------------------------------ #
        # 5. Enrich candidates: fiber intersection → (x_step, x_res, signs). #
        # ------------------------------------------------------------------ #
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

        print(f"[search_fn] x_here={x_here}  stubs={_n_pre_enrich_stubs}  "
              f"enriched={len(enriched_candidates)}  _fi={'ok' if _fi is not None else 'NONE'}  "
              f"_G_poly={'ok' if _G_poly is not None else 'NONE'}")

        # ------------------------------------------------------------------ #
        # 6. Attach S_of_m / inter_sym to every enriched record.             #
        # ------------------------------------------------------------------ #
        if _fi is not None and _G_poly is not None:
            _S_of_m_rec, _inter_sym_rec = compute_S_of_m(_fi, _G_poly, _curve_degree)
            for rec in enriched_candidates:
                if isinstance(rec, dict):
                    rec.setdefault('S_of_m',    _S_of_m_rec)
                    rec.setdefault('inter_sym', _inter_sym_rec)

        # ------------------------------------------------------------------ #
        # 7. candidate_xs: m-root-derived steps only (no x_res_head).        #
        # ------------------------------------------------------------------ #
        candidate_xs = {
            c.get('x_step') for c in enriched_candidates
            if isinstance(c, dict)
            and c.get('x_step') is not None
            and c.get('source') != 'x_res_head'
        }

        # ------------------------------------------------------------------ #
        # 8. Deferred fertility fallback using enriched m-root candidates.   #
        # ------------------------------------------------------------------ #
        if norm.get('n_with_roots') is None and vecs:
            n_mroot_cands = sum(
                1 for c in enriched_candidates
                if isinstance(c, dict) and c.get('source') != 'x_res_head'
            )
            if n_mroot_cands > 0:
                norm['n_with_roots'] = min(n_mroot_cands, len(vecs))
                norm['n_total']      = len(vecs)
                norm['total_roots']  = n_mroot_cands
                norm['per_n_roots']  = {}

        n_xk_head = sum(
            1 for c in enriched_candidates
            if isinstance(c, dict) and c.get('source') == 'x_res_head'
        )

        S_of_m_step, _ = (compute_S_of_m(_fi, _G_poly, _curve_degree)
                          if _fi is not None else (None, None))

        # ------------------------------------------------------------------ #
        # 9. Assemble result dict.                                            #
        # ------------------------------------------------------------------ #
        result = {
            'candidates':        enriched_candidates,
            'candidate_records': enriched_candidates,
            'candidate_xs':      candidate_xs,
            'n_xk_head':         n_xk_head,
            'stats':             norm.get('stats', None),
            'found_xs':          norm.get('found_xs', set()),
            'input_n':           n0,
            'S_of_m':            S_of_m_step,
            'fi':                _fi,
            'G_poly':            _G_poly,
            'vecs':              vecs,
            'n_with_roots':      norm.get('n_with_roots', None),
            'n_total':           norm.get('n_total', None),
            'total_roots':       norm.get('total_roots', None),
            'per_n_roots':       norm.get('per_n_roots', None),
            # Omit: 'context', 'raw_mumford_residues', 'new_sections',
            # 'precomputed_residues' — hold uncollectable SageMath objects.
        }

        # ------------------------------------------------------------------ #
        # 10. Augment with phi-derived intersection polys.                   #
        # ------------------------------------------------------------------ #
        result = _augment_with_phi(
            result,
            f_coeffs  = _phi_f_coeffs,
            p         = p,
            x_src     = x_here,
            y_src     = y_here,
            sage_ring = _phi_ring,
        )

        # ------------------------------------------------------------------ #
        # 11. Dead-end reason classification.                                 #
        #                                                                     #
        # Must run AFTER _augment_with_phi (phi-derived candidates counted)  #
        # and AFTER enrich_candidates (x_step populated).                    #
        #                                                                     #
        # Priority order:                                                     #
        #   ok                  — ≥1 candidate with x_step ≠ x_src           #
        #   torsion             — enrichment produced records but all have    #
        #                         x_step == x_src (Weierstrass/torsion pt)   #
        #   no_valid_candidates — Mumford/Julia found roots (stubs existed or #
        #                         n_with_roots set) but none survived         #
        #                         enrichment/phi (geometry failure)           #
        #   no_roots            — Mumford/Julia found zero F_p roots          #
        # ------------------------------------------------------------------ #
        final_candidates = result.get('candidate_records') or result.get('candidates') or []
        valid_final = [
            c for c in final_candidates
            if isinstance(c, dict)
            and c.get('x_step') is not None
            and c.get('x_step') != x_here
        ]

        if valid_final:
            _dead_end_reason = 'ok'
        elif final_candidates:
            # Post-phi candidates exist but every x_step == x_src — genuine torsion.
            _dead_end_reason = 'torsion'
            print(f"  [torsion_dbg] x_here={x_here}  "
                  f"enriched={len(enriched_candidates)}  "
                  f"final={len(final_candidates)}  valid_final=0")
            for _r in final_candidates[:5]:
                if not isinstance(_r, dict):
                    continue
                print(f"  [torsion_dbg]   source={_r.get('source')}  "
                      f"x_step={_r.get('x_step')}  x_res={_r.get('x_res')}  "
                      f"phi_geo={_r.get('phi_geo')}  m={_r.get('m')}")
        elif enriched_candidates or (norm.get('n_with_roots') or norm.get('total_roots') or _n_pre_enrich_stubs > 0):
            # Enrichment produced records but phi dropped them all (x_res never
            # populated), OR Julia found roots but enrichment itself dropped them.
            # Typical causes: both y-signs failed in phi, degenerate fiber,
            # _fi/_G_poly missing, or sign mismatches.
            _dead_end_reason = 'no_valid_candidates'
        else:
            _dead_end_reason = 'no_roots'

        result['dead_end_reason'] = _dead_end_reason
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
