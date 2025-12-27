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
        return str(x)

def integrate_differential_path_with_branch(x_start, x_end, y_start, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False,
                                            perp_offset=True, subdivide_thresh=1e-9): # TIGHTER THRESHOLD
    """
    Robust integration from (x_start, y_start) to (x_end, y_end).
    """
    key = (_round_for_cache(x_start, prec), _round_for_cache(x_end, prec),
           _round_for_cache(y_start, prec), _round_for_cache(y_end, prec),
           tuple([float(c) for c in f_coeffs]), use_x_weight, int(prec), bool(perp_offset))
    if key in integrate_differential_path_with_branch.cache:
        return integrate_differential_path_with_branch.cache[key]

    CC = ComplexField(prec)

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
            # Ensure offset is small but nonzero to avoid branch cuts
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

    xvals = []
    ws = []
    for (_t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        xvals.append(xval)
        ws.append(CC(w))

    def f_at(z):
        res = CC(f_coeffs[0])
        for c in f_coeffs[1:]:
            res = res * z + CC(c)
        return res

    fvals = [f_at(xv) for xv in xvals]
    tiny = CC(2) ** (-prec // 2)

    # RECURSIVE SUBDIVISION:
    # If any point on the path is too close to a root (fval ~ 0), 
    # we subdivide to maintain precision.
    min_abs = min([abs(fv) for fv in fvals if fv is not None] + [CC(1e99)])
    
    # Also check if endpoint signs are ambiguous
    if min_abs < CC(subdivide_thresh):
        mid_x = (p0 + p1) / CC(2)
        # Approximate mid_y linearly (sheet choice handled by recursion)
        mid_y = (y0_target + y1_target) / CC(2) 
        
        left = integrate_differential_path_with_branch(x_start, mid_x, y_start, mid_y, f_coeffs,
                                                      use_x_weight, prec, debug, perp_offset, subdivide_thresh)
        right = integrate_differential_path_with_branch(mid_x, x_end, mid_y, y_end, f_coeffs,
                                                       use_x_weight, prec, debug, perp_offset, subdivide_thresh)
        val = left + right
        integrate_differential_path_with_branch.cache[key] = val
        return val

    # find a non-degenerate seed near the start
    n = len(xvals)
    start_idx = 0
    for i in range(n):
        if abs(fvals[i]) >= tiny:
            start_idx = i
            break

    def choose_sqrt_match(prev_y, fval):
        s = fval.sqrt()
        # Choose sign to minimize distance to previous (continuity)
        if abs(s - prev_y) <= abs(-s - prev_y):
            return s
        else:
            return -s

    def try_propagation(initial_sign_choice):
        yvals = [None] * n
        sqrt_start = fvals[start_idx].sqrt()
        yvals[start_idx] = sqrt_start if initial_sign_choice >= 0 else -sqrt_start

        # forward
        for i in range(start_idx + 1, n):
            yvals[i] = choose_sqrt_match(yvals[i - 1], fvals[i])
        # backward
        for i in range(start_idx - 1, -1, -1):
            yvals[i] = choose_sqrt_match(yvals[i + 1], fvals[i])

        # Integrate
        integral = CC(0)
        dx_factor = vec / CC(2)
        for i in range(n):
            y_cur = yvals[i]
            if use_x_weight:
                integrand = xvals[i] / (CC(2) * y_cur)
            else:
                integrand = CC(1) / (CC(2) * y_cur)
            dxd = dx_factor * ws[i]
            integral += integrand * dxd

        # mismatch at END vs target
        end_mismatch = min(abs(yvals[-1] - y1_target), abs(-yvals[-1] - y1_target))
        return integral, yvals, float(end_mismatch)

    try0, yvals0, mismatch0 = try_propagation(+1)
    try1, yvals1, mismatch1 = try_propagation(-1)

    if mismatch0 <= mismatch1:
        integral = try0
        chosen_yvals = yvals0
    else:
        integral = try1
        chosen_yvals = yvals1

    # Fix final sign based on endpoint target
    # If the computed path arrived at -y_end instead of y_end, the integral 
    # computed was for the negative sheet. Flip it.
    if abs(chosen_yvals[-1] - y1_target) > abs(-chosen_yvals[-1] - y1_target):
        integral = -integral

    integrate_differential_path_with_branch.cache[key] = integral
    return integral

integrate_differential_path_with_branch.cache = {}
