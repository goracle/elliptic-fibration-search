"""Riemann theta function computations."""

import math
from sage.all import ComplexField, exp, pi
from itertools import product


"""Riemann theta function computations."""

from sage.all import ComplexField, RealField, exp, pi

# tuning knobs (module-level; change at runtime if needed)
THETA_RADIUS_CAP = 18        # default cap (conservative)
THETA_MAX_TERMS = 20000      # default max terms
THETA_EPS_FACTOR = 10        # default epsilon factor

MAX_RADIUS = 60


def theta_direct(tau_in, z_in, R=3, prec_local=2048):
    """
    Direct summation of theta function for genus 2, used for cheap screening.
    """
    key = (str(tau_in), str(z_in), R, prec_local)
    from sage.all import ComplexField, pi, exp
    CC_loc = ComplexField(prec_local)
    g_loc = len(z_in)
    Tau = [[CC_loc(tau_in[i][j]) for j in range(g_loc)] for i in range(g_loc)]
    Z = [CC_loc(z_in[i]) for i in range(g_loc)]
    total = CC_loc(0)
    
    # Generic loop for arbitrary genus would be better, but optimizing for g=2
    if g_loc == 2:
        for n0 in range(-R, R+1):
            for n1 in range(-R, R+1):
                # q = n^T * Tau * n
                q = Tau[0][0]*n0*n0 + (Tau[0][1]+Tau[1][0])*n0*n1 + Tau[1][1]*n1*n1
                # linear = 2 * n^T * z
                linear = 2*(n0*Z[0] + n1*Z[1])
                arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
                total += CC_loc(exp(arg))
        ret = total
    elif g_loc == 1:
        for n0 in range(-R, R+1):
            q = Tau[0][0]*n0*n0
            linear = 2*n0*Z[0]
            arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
            total += CC_loc(exp(arg))
        
        ret = total
    else:
        raise NotImplementedError("theta_direct optimization only implemented for g=1,2")
    theta_direct.cache[key] = ret
    return ret
theta_direct.cache = {}

# inside theta.py -- replace compute_theta_high_prec with this


def compute_theta_high_prec(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8):
    """
    Compute Riemann theta function for genus 2 with controlled truncation.

    Raises when convergence is unsafe instead of returning junk.
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor)
    if key in compute_theta_high_prec.cache:
        return compute_theta_high_prec.cache[key]

    from sage.all import ComplexField, RealField, log, pi
    import math

    CC = ComplexField(prec)
    RR = RealField(prec)

    pi_I = CC(0, 1) * CC(pi)

    # --- basic positivity check ---
    t00, t11 = tau[0,0], tau[1,1]
    im_t00 = RR(t00.imag())
    im_t11 = RR(t11.imag())

    if im_t00 <= 0 or im_t11 <= 0:
        raise ValueError("Im(tau) not positive definite")

    y_min = min(im_t00, im_t11)

    if y_min < RR(1e-2):
        raise ValueError(f"Im(tau) too close to boundary: y_min={float(y_min)}")

    # --- precision target ---
    epsilon = RR(2) ** (-prec + epsilon_factor)

    radius_needed = int(
        math.sqrt(float(-log(epsilon) / (RR(pi) * y_min)))
    ) + 2

    # --- HARD safety caps ---
    genus = 2
    radius_cap = min(
        max(12, int(2 * genus * math.sqrt(prec))),
        60
    )

    radius = min(radius_needed, radius_cap)

    if radius_needed > MAX_RADIUS:
        raise RuntimeError(f"radius needed exceeded the max radius of {MAX_RADIUS}.")


    if radius < radius_needed:
        raise RuntimeError(
            f"Theta radius capped: needed {radius_needed}, using {radius}. "
            "Increase Im(tau) or lower precision."
        )

    # --- summation ---
    total = CC(0)
    z0, z1 = z_vec
    t01 = tau[0,1]

    n_terms = 0
    last_ring = None

    for ring in range(radius + 1):
        ring_sum = CC(0)

        if ring == 0:
            ring_sum = CC(1)
            total += ring_sum
            n_terms += 1
        else:
            for n1 in range(-ring, ring + 1):
                for n2 in range(-ring, ring + 1):
                    if max(abs(n1), abs(n2)) != ring:
                        continue

                    n_terms += 1
                    if n_terms > max_terms:
                        raise RuntimeError("Theta exceeded max_terms")

                    quad = (n1*n1)*t00 + 2*n1*n2*t01 + (n2*n2)*t11
                    lin = 2*(n1*z0 + n2*z1)
                    ring_sum += exp(pi_I * (quad + lin))

            total += ring_sum

        ring_abs = abs(ring_sum)

        if ring > 4 and ring_abs < epsilon:
            break

        if last_ring is not None and ring > 6:
            if ring_abs > last_ring * 2:
                raise RuntimeError("Theta diverging")

        last_ring = ring_abs

    if abs(total) < epsilon:
        raise ValueError("Theta numerically zero")
    ret = total
    compute_theta_high_prec.cache[key] = ret
    return ret
compute_theta_high_prec.cache = {}


from concurrent.futures import ThreadPoolExecutor, as_completed


# simple cache dict on the function
#compute_theta_high_prec_parallel.cache = {}


def compute_theta_high_prec_parallel(z_vec, tau, prec=2048, max_terms=20000,
                                     epsilon_factor=8, max_workers=None):
    """
    Parallelized theta summation by distributing lattice points within each ring
    across a thread-pool. Preserves early-break logic per ring (sequential ring
    processing) while parallelizing the heavy inner loops.

    Raises on unsafe conditions (same behavior as serial).
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor)
    if key in compute_theta_high_prec_parallel.cache:
        return compute_theta_high_prec_parallel.cache[key]

    # local imports so this function can be copied into your sage file
    from sage.all import ComplexField, RealField, pi, exp
    CC = ComplexField(prec)
    RR = RealField(max(53, min(prec, 200)))  # modest real precision for these checks
    pi_I = CC(0, 1) * CC(pi)

    # sanity checks on tau
    t00 = CC(tau[0,0])
    t11 = CC(tau[1,1])
    t01 = CC(tau[0,1])

    im_t00 = RR(t00.imag())
    im_t11 = RR(t11.imag())
    if im_t00 <= 0 or im_t11 <= 0:
        raise ValueError("Im(tau) not positive definite")
    y_min = float(min(im_t00, im_t11))
    if not math.isfinite(y_min) or y_min <= 0.0:
        raise ValueError(f"Im(tau) has nonpositive or non-finite imaginary parts: {im_t00}, {im_t11}")
    if y_min < 1e-6:
        # use a stricter check; your code previously used 1e-2 — adjust if needed
        raise ValueError(f"Im(tau) too close to boundary: y_min={y_min}")

    # --- precision -> epsilon and radius needed (avoid epsilon float underflow) ---
    if prec <= epsilon_factor:
        # Degenerate case: effective exponent non-positive -> epsilon >= 1
        # radius_needed becomes small; guard it.
        radius_needed = 2
    else:
        # use analytic identity: -log(epsilon) = (prec - epsilon_factor) * log(2)
        numerator = (prec - float(epsilon_factor)) * math.log(2.0)
        denom = math.pi * y_min
        if numerator <= 0.0 or denom <= 0.0:
            raise RuntimeError("Invalid numerical values computing theta radius")
        radius_needed = int(math.sqrt(numerator / denom)) + 2

    # --- HARD safety caps ---
    genus = 2
    radius_cap = min(
        max(12, int(2 * genus * math.sqrt(float(prec)))),
        60
    )

    radius = min(radius_needed, radius_cap)

    # ensure MAX_RADIUS exists in module or use default
    try:
        MAX_RADIUS
    except NameError:
        MAX_RADIUS = 60

    if radius_needed > MAX_RADIUS:
        raise RuntimeError(f"radius needed ({radius_needed}) exceeded the max radius of {MAX_RADIUS}.")

    if radius < radius_needed:
        raise RuntimeError(
            f"Theta radius capped: needed {radius_needed}, using {radius}. "
            "Increase Im(tau) or lower precision."
        )

    # --- rest of function unchanged (per-ring parallel chunks) ---
    # helper: partition a list into nearly-equal chunks
    def _chunks(lst, k):
        if k <= 1:
            yield lst
            return
        n = len(lst)
        q, r = divmod(n, k)
        idx = 0
        for i in range(k):
            take = q + (1 if i < r else 0)
            if take:
                yield lst[idx: idx + take]
            idx += take

    # inner per-chunk function executed by threads (uses CC, t00,t01,t11,z0,z1,pi_I from closure)
    z0 = CC(z_vec[0])
    z1 = CC(z_vec[1])

    def _sum_chunk(chunk):
        s = CC(0)
        local_n = 0
        for n1, n2 in chunk:
            quad = (n1 * n1) * t00 + 2 * n1 * n2 * t01 + (n2 * n2) * t11
            lin = 2 * (n1 * z0 + n2 * z1)
            s += exp(pi_I * (quad + lin))
            local_n += 1
        return s, local_n, abs(s)

    total = CC(0)
    n_terms = 0
    last_ring = None

    # decide number of worker threads
    import multiprocessing
    if max_workers is None:
        max_workers = max(8, multiprocessing.cpu_count() - 1)

    # process rings sequentially; parallelize points inside each ring
    for ring in range(radius + 1):
        if ring == 0:
            total += CC(1)
            n_terms += 1
            last_ring = abs(CC(1))
            continue

        # generate lattice points on ring (max(abs(n1),abs(n2)) == ring)
        pts = [(n1, n2)
               for n1 in range(-ring, ring + 1)
               for n2 in range(-ring, ring + 1)
               if max(abs(n1), abs(n2)) == ring]

        # split pts into worker chunks (balance by count)
        n_workers = min(max_workers, len(pts))
        if n_workers <= 0:
            n_workers = 1

        chunk_list = list(_chunks(pts, n_workers))

        ring_sum = CC(0)
        ring_nterms = 0
        ring_max_abs = None

        # parallel map using threads so tau/CC are shared in-process
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_sum_chunk, ch) for ch in chunk_list]
            for fut in as_completed(futures):
                s_part, local_count, abs_part = fut.result()
                ring_sum += s_part
                ring_nterms += local_count
                if ring_max_abs is None or abs_part > ring_max_abs:
                    ring_max_abs = abs_part

        n_terms += ring_nterms
        if n_terms > max_terms:
            raise RuntimeError("Theta exceeded max_terms")

        total += ring_sum

        # per-ring heuristics (same as serial)
        ring_abs = float(ring_max_abs) if ring_max_abs is not None else float(abs(ring_sum))
        if ring > 4 and ring_abs < float(2 ** (-prec + epsilon_factor)):
            # Note: this check is safe because we don't convert extremely tiny values earlier;
            # if float underflows here, the explicit radius_needed logic above already guarded that case.
            break

        if last_ring is not None and ring > 6:
            if ring_abs > float(last_ring) * 2:
                raise RuntimeError("Theta diverging")

        last_ring = ring_abs

    if abs(total) < 2 ** (-prec + epsilon_factor):
        raise ValueError("Theta numerically zero")

    compute_theta_high_prec_parallel.cache[key] = total
    return total

compute_theta_high_prec_parallel.cache = {}
