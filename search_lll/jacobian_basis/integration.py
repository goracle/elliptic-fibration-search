"""Integration of differentials along paths."""

import math
from sage.all import ComplexField, PolynomialRing




"""Integration of differentials along paths. — more robust branch handling"""

from sage.all import ComplexField, RealField

def _round_for_cache(x, prec):
    # convert complex-like to rounded tuple for stable cache keys
    try:
        z = complex(x)
        scale = max(1, int(prec // 10))
        return (round(z.real, 12), round(z.imag, 12))
    except Exception:
        try:
            return (round(float(x), 12), 0.0)
        except Exception:
            return str(x)

def integrate_differential_path_with_branch(x_start, x_end, y_start, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False,
                                            perp_offset=True, subdivide_thresh=1e-8):
    """
    Robust integration from (x_start, y_start) to (x_end, y_end).
    BOTH y coordinates indicate sheet choice. This routine:
      - picks an initial sign at the start,
      - propagates continuously along the straight line,
      - if the endpoint sign disagrees, it tries the opposite start sign,
      - if the path gets too close to branch locus it subdivides and recurses.

    Arguments:
      perp_offset: if True, apply a tiny perpendicular offset to avoid branch nodes exactly on the straight line
      subdivide_thresh: relative threshold to decide when to subdivide (compared to |f| values)
    """
    key = (_round_for_cache(x_start, prec), _round_for_cache(x_end, prec),
           _round_for_cache(y_start, prec), _round_for_cache(y_end, prec),
           tuple([float(c) for c in f_coeffs]), use_x_weight, int(prec), bool(perp_offset))
    if key in integrate_differential_path_with_branch.cache:
        return integrate_differential_path_with_branch.cache[key]

    CC = ComplexField(prec)
    RR = RealField(max(53, min(prec, 200)))

    p0 = CC(x_start)
    p1 = CC(x_end)
    y0_target = CC(y_start)
    y1_target = CC(y_end)

    vec = p1 - p0

    # optional perpendicular tiny offset to avoid landing exactly on branch points
    off = CC(0)
    if perp_offset:
        # tiny offset perpendicular to vec
        if abs(vec) != 0:
            perp = CC(0, 1) * vec
            off = perp / abs(perp) * (abs(vec) * CC(1e-12))
        else:
            off = CC(0)

    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0 / float(N)
        pi = math.pi
        for k in range(-N, N + 1):
            t = k * h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi / 2.0) * sx)
            dx_dt = (pi / 2.0) * math.cosh(t) / (math.cosh((pi / 2.0) * sx) ** 2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes

    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)

    # build xvals and weights on straight line (with offset)
    xvals = []
    ws = []
    for (_t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        xvals.append(xval)
        ws.append(CC(w))

    def f_at(z):
        # Horner
        res = CC(f_coeffs[0])
        for c in f_coeffs[1:]:
            res = res * z + CC(c)
        return res

    fvals = [f_at(xv) for xv in xvals]
    tiny = CC(2) ** (-prec // 2)

    # If too close to branch points anywhere, subdivide recursively
    min_abs = min([abs(fv) for fv in fvals if fv is not None] + [CC(1e99)])
    if min_abs < CC(subdivide_thresh):
        # subdivide at midpoint and recurse
        mid_x = complex((p0 + p1) / CC(2))
        # compute approximate y at midpoint with naive sqrt (choose sheet by interpolation)
        try:
            mid_f = f_at((p0 + p1) / CC(2))
            mid_y = mid_f.sqrt()
        except Exception:
            mid_y = (y0_target + y1_target) / CC(2)
        left = integrate_differential_path_with_branch(x_start, mid_x, y_start, mid_y, f_coeffs,
                                                      use_x_weight=use_x_weight, prec=prec,
                                                      debug=debug, perp_offset=perp_offset,
                                                      subdivide_thresh=subdivide_thresh)
        right = integrate_differential_path_with_branch(mid_x, x_end, mid_y, y_end, f_coeffs,
                                                       use_x_weight=use_x_weight, prec=prec,
                                                       debug=debug, perp_offset=perp_offset,
                                                       subdivide_thresh=subdivide_thresh)
        val = left + right
        integrate_differential_path_with_branch.cache[key] = val
        return val

    # find a non-degenerate seed near the start (index)
    n = len(xvals)
    start_idx = None
    for i in range(0, max(3, n // 10)):
        if abs(fvals[i]) >= tiny:
            start_idx = i
            break
    if start_idx is None:
        # fallback: pick first index
        start_idx = 0

    # helper to choose sign by continuity: dot-product criterion
    def choose_sqrt_match(prev_y, fval):
        s = fval.sqrt()
        # choose sign so that Re(s * conj(prev_y)) >= 0  (equivalently dot-product >= 0)
        if (s.real() * prev_y.real() + s.imag() * prev_y.imag()) >= 0:
            return s
        else:
            return -s

    # primary propagation procedure: attempt starting sign choices and pick the one that matches endpoint
    def try_propagation(initial_sign_choice):
        # initial_sign_choice: +1 or -1 relative to principal sqrt at start_idx
        yvals = [None] * n
        sqrt_start = fvals[start_idx].sqrt()
        if initial_sign_choice >= 0:
            yvals[start_idx] = sqrt_start
        else:
            yvals[start_idx] = -sqrt_start

        # propagate forward from start_idx to end
        for i in range(start_idx + 1, n):
            s = fvals[i].sqrt()
            # continuity against previous
            yvals[i] = choose_sqrt_match(yvals[i - 1], fvals[i])

        # propagate backward from start_idx down to 0
        for i in range(start_idx - 1, -1, -1):
            s = fvals[i].sqrt()
            # choose so close to next
            yvals[i] = choose_sqrt_match(yvals[i + 1], fvals[i])

        # build integral
        integral = CC(0)
        dx_factor = vec / CC(2)
        for i in range(n):
            y_cur = yvals[i]
            if y_cur is None or abs(y_cur) == 0:
                continue
            if use_x_weight:
                integrand = xvals[i] / (CC(2) * y_cur)
            else:
                integrand = CC(1) / (CC(2) * y_cur)
            dxd = dx_factor * ws[i]
            integral += integrand * dxd

        # compute end mismatch metric to target end sheet
        end_mismatch = min(abs(yvals[-1] - y1_target), abs(-yvals[-1] - y1_target))
        return integral, yvals, float(end_mismatch)

    # try both initial choices and pick the one with smaller end mismatch
    try0, yvals0, mismatch0 = try_propagation(+1)
    try1, yvals1, mismatch1 = try_propagation(-1)

    if debug:
        print(f"[int_branch] mismatch0={mismatch0:.3e}, mismatch1={mismatch1:.3e}")

    if mismatch0 <= mismatch1:
        integral = try0
        chosen_yvals = yvals0
    else:
        integral = try1
        chosen_yvals = yvals1

    # final verification: if chosen_yvals[-1] still far from requested y_end, try global flip
    if abs(chosen_yvals[-1] - y1_target) > abs(-chosen_yvals[-1] - y1_target):
        # OK matches preferred sign
        pass
    else:
        # flip sign of integral and yvals
        integral = -integral
        chosen_yvals = [-yv for yv in chosen_yvals]

    integrate_differential_path_with_branch.cache[key] = integral
    return integral

# initialize cache
integrate_differential_path_with_branch.cache = {}
