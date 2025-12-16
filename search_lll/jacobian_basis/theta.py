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
        return total
    elif g_loc == 1:
         for n0 in range(-R, R+1):
            q = Tau[0][0]*n0*n0
            linear = 2*n0*Z[0]
            arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
            total += CC_loc(exp(arg))
         return total
    else:
        raise NotImplementedError("theta_direct optimization only implemented for g=1,2")


# inside theta.py -- replace compute_theta_high_prec with this


def compute_theta_high_prec(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8):
    """
    Compute Riemann theta function for genus 2 with controlled truncation.

    Raises when convergence is unsafe instead of returning junk.
    """

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

    return total
