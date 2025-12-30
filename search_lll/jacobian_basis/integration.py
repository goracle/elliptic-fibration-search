"""Integration of differentials along paths."""

import math
from sage.all import ComplexField, PolynomialRing


"""Integration of differentials along paths. — more robust branch handling"""

from sage.all import ComplexField, RealField


# initialize cache


"""Integration of differentials along paths."""

from sage.all import ComplexField, PolynomialRing, RealField

def _round_for_cache(x, prec):
    try:
        z = complex(x)
        return (round(z.real, 12), round(z.imag, 12))
    except Exception:
        raise
        return str(x)


def integrate_differential_path_with_branch(x_start, x_end, y_start, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False,
                                            perp_offset=True, subdivide_thresh=1e-9, _depth=0):
    """
    Robust integration of differentials along a straight path from (x_start,y_start) to (x_end,y_end).
    - uses tanh-sinh nodes with propagation of sqrt branches
    - recursive subdivision when path passes too close to branch points
    - added recursion depth guard to avoid runaway recursion
    """
    import math
    from sage.all import ComplexField
    CC = ComplexField(prec)

    # recursion depth guard
    if _depth > 30:
        raise RuntimeError("Maximum subdivision depth reached in integrate_differential_path_with_branch")

    # safer cache key: round complex endpoints and f_coeffs moderate precision
    def _round_for_cache(x):
        try:
            z = complex(x)
            return (round(z.real, 12), round(z.imag, 12))
        except Exception:
            raise
            return str(x)

    key = (_round_for_cache(x_start), _round_for_cache(x_end),
           _round_for_cache(y_start), _round_for_cache(y_end),
           tuple([float(c) for c in f_coeffs]), use_x_weight, int(prec), bool(perp_offset))
    if key in integrate_differential_path_with_branch.cache:
        return integrate_differential_path_with_branch.cache[key]

    p0 = CC(x_start)
    p1 = CC(x_end)
    y0_target = CC(y_start)
    y1_target = CC(y_end)

    vec = p1 - p0
    nvec_abs = abs(vec)

    # perpendicular offset sizing: not too tiny relative to precision, but small vs geometry
    if perp_offset:
        # if the segment has length, offset perpendicular with magnitude that depends on precision and geometry
        if nvec_abs != 0:
            perp = CC(0, 1) * vec
            # choose offset magnitude respecting both precision and segment length
            offset_mag = max(nvec_abs * CC(1e-12), CC(2) ** (-(prec // 4)))
            off = perp / abs(perp) * offset_mag
        else:
            off = CC(0)
    else:
        off = CC(0)

    # tanh-sinh nodes generator (maps t in [-inf,inf] -> x_mapped in [-1,1])
    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0 / float(N)
        pi = math.pi
        for k in range(-N, N + 1):
            t = k * h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi / 2.0) * sx)
            # derivative of mapping factor wrt t (used as integration weight)
            dx_dt = (pi / 2.0) * math.cosh(t) / (math.cosh((pi / 2.0) * sx) ** 2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes

    # choose node count based on precision (bounded)
    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)

    # compute sample x-values along the segment and corresponding quadrature weights
    xvals = []
    ws = []
    for (_t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)   # mapping to [0,1]
        xval = p0 + s * vec + off
        xvals.append(xval)
        ws.append(CC(w))

    # polynomial evaluation for f(x) using Horner (descending coefficients)
    def f_at(z):
        res = CC(f_coeffs[0])
        for c in f_coeffs[1:]:
            res = res * z + CC(c)
        return res

    fvals = [f_at(xv) for xv in xvals]

    tiny = CC(2) ** (-prec // 2)
    # if any sample gets too close to a branch point (fval small), subdivide
    min_abs = min([abs(fv) for fv in fvals] + [CC(1e99)])
    if min_abs < CC(subdivide_thresh):
        mid_x = (p0 + p1) / CC(2)
        # for mid y estimate, prefer averaging endpoints (sheet sign may be corrected in subcalls)
        mid_y = (y0_target + y1_target) / CC(2)
        left = integrate_differential_path_with_branch(x_start, mid_x, y_start, mid_y, f_coeffs,
                                                      use_x_weight, prec, debug, perp_offset, subdivide_thresh, _depth=_depth+1)
        right = integrate_differential_path_with_branch(mid_x, x_end, mid_y, y_end, f_coeffs,
                                                       use_x_weight, prec, debug, perp_offset, subdivide_thresh, _depth=_depth+1)
        val = left + right
        integrate_differential_path_with_branch.cache[key] = val
        return val

    # find starting index with non-tiny fval to seed sqrt propagation
    start_idx = 0
    for i in range(len(xvals)):
        if abs(fvals[i]) >= tiny:
            start_idx = i
            break

    def choose_sqrt_match(prev_y, fval):
        # choose sqrt branch that minimizes distance to prev_y
        s = fval.sqrt()
        if abs(s - prev_y) <= abs(-s - prev_y):
            return s
        else:
            return -s

    def try_propagation(initial_sign_choice):
        n = len(xvals)
        yvals = [None] * n
        sqrt_start = fvals[start_idx].sqrt()
        yvals[start_idx] = sqrt_start if initial_sign_choice >= 0 else -sqrt_start

        # forward propagate
        for i in range(start_idx + 1, n):
            # guard against extremely small fval
            if abs(fvals[i]) < tiny:
                # use previous value's sign
                yvals[i] = CC(0) if abs(fvals[i]) < (tiny * CC(1e-3)) else choose_sqrt_match(yvals[i-1], fvals[i])
            else:
                yvals[i] = choose_sqrt_match(yvals[i - 1], fvals[i])
        # backward propagate
        for i in range(start_idx - 1, -1, -1):
            if abs(fvals[i]) < tiny:
                yvals[i] = CC(0) if abs(fvals[i]) < (tiny * CC(1e-3)) else choose_sqrt_match(yvals[i+1], fvals[i])
            else:
                yvals[i] = choose_sqrt_match(yvals[i + 1], fvals[i])

        # perform quadrature sum
        integral = CC(0)
        dx_factor = vec / CC(2)
        for i in range(n):
            y_cur = yvals[i]
            if y_cur is None or abs(y_cur) == 0:
                # avoid division by zero: treat as very small (shouldn't happen after subdivision)
                if debug:
                    print("[integrator] encountered near-zero y_cur at node", i)
                y_cur = tiny
            if use_x_weight:
                integrand = xvals[i] / (CC(2) * y_cur)
            else:
                integrand = CC(1) / (CC(2) * y_cur)
            dxd = dx_factor * ws[i]
            integral += integrand * dxd

        # compute mismatch vs requested endpoint y1_target to decide sign correction
        end_mismatch_plus = abs(yvals[-1] - y1_target)
        end_mismatch_minus = abs(-yvals[-1] - y1_target)
        mismatch = float(min(end_mismatch_plus, end_mismatch_minus))

        return integral, yvals, mismatch, end_mismatch_plus, end_mismatch_minus

    try0, yvals0, mismatch0, end0p, end0m = try_propagation(+1)
    try1, yvals1, mismatch1, end1p, end1m = try_propagation(-1)

    if mismatch0 <= mismatch1:
        integral = try0
        chosen_yvals = yvals0
        end0p_val = end0p
        end0m_val = end0m
    else:
        integral = try1
        chosen_yvals = yvals1
        end0p_val = end1p
        end0m_val = end1m

    # If the propagated endpoint is closer to -y_end than +y_end, flip the sign of the integral
    if end0p_val > end0m_val:
        integral = -integral

    integrate_differential_path_with_branch.cache[key] = integral
    return integral

# initialize cache
integrate_differential_path_with_branch.cache = {}


def integrate_differential_path_joint(x0, x1, y0, y1, f_coeffs,
                                      *, prec=200, debug=False):
    """
    Follow one path, one sheet history, and return:
        ( ∫ dx/(2y),  ∫ x dx/(2y) )
    FIXES Problem #3: both coordinates share the same branch + path.
    """
    from sage.all import ComplexField
    CC = ComplexField(prec)

    I0 = CC(0)
    I1 = CC(0)

    # reuse your existing quadrature machinery — but once — and
    # propagate sqrt signs only once for both integrands
    #
    # [minimal change: call the old path integrator in "vector mode"]
    #
    vals = _integrate_with_tracked_sheet(
        x0, x1, y0, y1, f_coeffs, prec=prec, debug=debug
    )

    for (xv, yv, w) in vals:
        I0 += w * (1 / (2 * yv))
        I1 += w * (xv / (2 * yv))

    return I0, I1
