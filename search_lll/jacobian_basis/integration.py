"""Integration of differentials along paths."""

import math
from sage.all import ComplexField, PolynomialRing


def integrate_differential_path_with_branch(x_start, x_end, y_start, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False):
    """
    Integrate from (x_start, y_start) to (x_end, y_end).
    BOTH y coordinates specify which sheet we're on.
    """
    import math
    from sage.all import ComplexField

    CC = ComplexField(prec)

    if debug:
        print(f"[integrate] from ({x_start}, {y_start}) to ({x_end}, {y_end})")

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

    p0 = CC(x_start)
    p1 = CC(x_end)
    y0 = CC(y_start)
    y1 = CC(y_end)
    
    vec = p1 - p0
    
    # NO PERPENDICULAR OFFSET - integrate on the straight line!
    # The branch tracking handles sheet selection
    
    dx_factor = vec / CC(2)

    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs) - 1 - i)) for i, c in enumerate(f_coeffs))

    # Build x-values along the straight line
    xvals = []
    ws = []
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)  # maps (-1,1) -> (0,1)
        xval = p0 + s * vec  # STRAIGHT LINE, no offset
        xvals.append(xval)
        ws.append(CC(w))

    n = len(xvals)
    fvals = [f_at(xv) for xv in xvals]

    tiny = CC(2) ** (-prec // 2)

    # CRITICAL: Establish branches at BOTH ends
    # At s=0 (start): we should get y_start
    # At s=1 (end): we should get y_end
    
    # Find good seed indices at both ends
    start_idx = None
    end_idx = None
    
    for i in range(n // 4):  # first quarter
        if abs(fvals[i]) >= tiny:
            start_idx = i
            break
    
    for i in range(n - 1, 3 * n // 4, -1):  # last quarter
        if abs(fvals[i]) >= tiny:
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        raise ValueError("Path too close to branch locus")

    # Assign branches at seed points
    yvals = [None] * n
    
    # Start: choose sqrt that matches y_start
    sqrt_start = fvals[start_idx].sqrt()
    if abs(sqrt_start - y0) <= abs(-sqrt_start - y0):
        yvals[start_idx] = sqrt_start
    else:
        yvals[start_idx] = -sqrt_start
    
    # End: choose sqrt that matches y_end  
    sqrt_end = fvals[end_idx].sqrt()
    if abs(sqrt_end - y1) <= abs(-sqrt_end - y1):
        yvals[end_idx] = sqrt_end
    else:
        yvals[end_idx] = -sqrt_end

    # Propagate from start_idx backward to 0
    for i in range(start_idx - 1, -1, -1):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i + 1]) <= abs(y_m - yvals[i + 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Propagate from start_idx forward to end_idx
    for i in range(start_idx + 1, end_idx + 1):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Propagate from end_idx forward to n-1
    for i in range(end_idx + 1, n):
        y_p = fvals[i].sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Integrate
    integral = CC(0)
    for i in range(n):
        y_cur = yvals[i]
        if abs(y_cur) == 0:
            continue
        
        if use_x_weight:
            integrand = xvals[i] / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * ws[i]
        integral += integrand * dxd

    return integral


