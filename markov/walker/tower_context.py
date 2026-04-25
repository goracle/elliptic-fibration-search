from .fiber_geometry import *
from .project_loader import *
from search_common import *

def build_project_tower_context_for_point(
    xi,
    yi=None,
    *,
    coeffs_genus2=None,
    base_points=None,
    p: Optional[int] = None,
    debug: bool = False,
):
    """Rebuild the tower / fibration context for a single current point.

    This mirrors the search7_genus2.doloop_genus2 setup path but keeps only the
    ingredients needed by the Markov candidate-search branch.
    """
    # Tower construction requires xi to be an F_p point (no field extensions supported).
    # This is satisfied because xi comes from a prior m-root search over F_p, guaranteeing
    # it is a point on C(F_p).
    setup_field_and_rings = resolve_project_symbol('setup_field_and_rings', required=True)
    apply_shift_transformation = resolve_project_symbol('apply_shift_transformation', required=True)
    apply_mobius_transformation = resolve_project_symbol('apply_mobius_transformation', required=True)
    build_tower_and_fibrations = resolve_project_symbol('build_tower_and_fibrations', required=True)
    extract_geometry_from_tower = resolve_project_symbol('extract_geometry_from_tower', required=True)
    build_curve_data = resolve_project_symbol('build_curve_data', required=True)
    configure_search_parameters = resolve_project_symbol('configure_search_parameters', required=True)
    build_search_rhs_list = resolve_project_symbol('build_search_rhs_list', required=True)
    setup_rationality_test_function = resolve_project_symbol('setup_rationality_test_function', required=True)
    compute_base_sections_m = resolve_project_symbol('compute_base_sections_m', required=True)
    lll_reduce_mw_basis = resolve_project_symbol('lll_reduce_mw_basis', required=True)

    coeffs_genus2 = coeffs_genus2 if coeffs_genus2 is not None else COEFFS_GENUS2
    print("building tower search for point:", (xi, yi))

    #base_points = list(base_points or _project_base_points_from_globals(xi, yi, p=p))
    base_points = [(xi, yi)]
    assert xi is not None, xi
    assert yi is not None, yi
    if yi is None:
        yfun = resolve_project_symbol('get_y_unshifted_genus2', default=None)
        if yfun is not None:
            try:
                yi = yfun(xi)
            except Exception:
                yi = None
                raise

    if yi is None:
        raise ValueError(f"Could not recover y-value for xi={xi!r}; please supply base_points or yi.")

    data_pts = [(xi, yi)]
    for pt in base_points:
        if pt is None:
            continue
        if len(pt) >= 2 and pt[0] is not None and pt[1] is not None:
            data_pts.append((pt[0], pt[1]))

    # Deduplicate while preserving order
    seen = set()
    uniq_data_pts = []
    for pt in data_pts:
        key = (str(pt[0]), str(pt[1]))
        if key in seen:
            continue
        seen.add(key)
        uniq_data_pts.append(pt)

    field_data = setup_field_and_rings(coeffs_genus2, uniq_data_pts)
    shifted_G_poly, base_pts, shift = apply_shift_transformation(
        field_data['G'], field_data['real_pts'], field_data['base_field']
    )
    assert len(base_pts) == 1, base_pts
    shifted_G_poly, base_pts, T, T_inv, _all_known_x = apply_mobius_transformation(
        shifted_G_poly, {xi}, base_pts
    )

    #print("base_pts, non-legacy", base_pts)
    primary_tower, fibrations, tower_for_mumford = build_tower_and_fibrations(
        shifted_G_poly, base_pts
    )
    #print("primary_tower, fibrations, tower_for_mumford")
    #print(primary_tower, fibrations, tower_for_mumford)

    E_rhs_m, r_m, roots = extract_geometry_from_tower(primary_tower, field_data['Fm'])

    cd, morphism_data = build_curve_data(E_rhs_m, roots, base_pts)
    one, two, three = morphism_data

    if False:
        print("E_rhs_m, r_m, roots")
        print(E_rhs_m, r_m, roots)
        print("cd, morphism_data")
        print(cd, morphism_data)
        sys.exit()

    sconf, prime_pool = configure_search_parameters(cd, {xi}, base_pts, field_data['base_field'])
    E_rhs_m_symbolic = primary_tower[-1]['f_i'] if primary_tower else None
    search_rhs_list = build_search_rhs_list(cd, roots, E_rhs_m_symbolic, one, two, three)

    # Add xk(m) as second RHS via Vieta: xk = S(m) - (d-1)*xi - xj(m).
    # S(m) is the negated x^(d-1) coefficient of the monic fiber intersection poly,
    # which equals xi + xj + xk for a degree-5 curve (d-1 = 4 roots sum to S).
    # We use the actual xj(m) RHS from the search rather than the RLINEAR=True
    # shortcut xi-m, so this is valid regardless of RLINEAR.
    _fi_for_xk = primary_tower[-1].get('f_i') if primary_tower else None
    _curve_degree = int(resolve_project_symbol('CURVE_DEGREE', default=5))
    if _fi_for_xk is not None and shifted_G_poly is not None and len(search_rhs_list) == 1:
        S_of_m, _ = compute_S_of_m(_fi_for_xk, shifted_G_poly, _curve_degree)
        if S_of_m is not None:
            try:
                _base = S_of_m.parent()           # Frac(GF(p)[m])
                _xj_rhs = _base(r_m)
                _xi_lifted = _base(xi)
                xk = S_of_m - (_curve_degree - 1) * _xi_lifted - _xj_rhs
                lastrhs = E_rhs_m(x=xk)
                last_phi_x = get_phi_x(one, two, three, xk, lastrhs)
                search_rhs_list = list(search_rhs_list) + [last_phi_x]
            except Exception as e:
                print(f"[build_project_tower_context] warning: could not build xk RHS: {e}")
                raise

    assert len(search_rhs_list) > 1, search_rhs_list

    testfunc, shift = setup_rationality_test_function(shift, T, T_inv)

    base_sections = compute_base_sections_m(cd, base_pts, tower=primary_tower)
    if not base_sections:
        raise RuntimeError('compute_base_sections_m returned no sections for the rebuilt tower')
    if len(base_sections) > 1:
        base_sections = lll_reduce_mw_basis(cd, base_sections)
    current_sections = list(set(base_sections))
    if not current_sections:
        raise RuntimeError('No usable current sections after LLL reduction')

    # Markov mode: keep a single section so vecs can remain one-dimensional.
    current_sections = [current_sections[0]]

    if debug:
        print(f"[tower] rebuilt for xi={xi}; sections={len(current_sections)}; primes={len(prime_pool)}")

    return {
        'cd': cd,
        'current_sections': current_sections,
        'prime_pool': prime_pool,
        'r_m': r_m,
        'shift': shift,
        'search_rhs_list': search_rhs_list,
        'testfunc': testfunc,
        'field_data': field_data,
        'shifted_G_poly': shifted_G_poly,
        'base_pts': base_pts,
        'T': T,
        'T_inv': T_inv,
        'primary_tower': primary_tower,
        'fibrations': fibrations,
        'tower_for_mumford': tower_for_mumford,
        'roots': roots,
        'morphism_data': morphism_data,
        'sconf': sconf,
        'xi': xi,
        'yi': yi,
    }
