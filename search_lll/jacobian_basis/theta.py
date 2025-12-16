"""Riemann theta function computations."""

import math
from sage.all import ComplexField, exp, pi
from itertools import product


"""Riemann theta function computations."""

from sage.all import ComplexField, RealField, exp, pi


def compute_theta_high_prec(z_vec, tau, prec=300, max_terms=50000, epsilon_factor=10):
    """
    Computes Riemann Theta function theta(z, tau) at high precision with convergence checks.
    z_vec: vector of length 2 (Complex)
    tau: 2x2 symmetric matrix (Complex)
    max_terms: maximum number of terms to sum before raising error
    epsilon_factor: terms smaller than 2^(-prec + epsilon_factor) are negligible
    """
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Pre-compute constants
    pi_I = CC(0, 1) * CC(pi)
    
    # Determine summation radius for precision
    # Start conservative but not excessive
    radius = min(int(math.sqrt(prec * 0.25)) + 2, 30)  # Cap at 30
    
    # Convergence threshold
    epsilon = RR(2) ** (-prec + epsilon_factor)
    
    total = CC(0)
    n_terms = 0
    
    # Extract components for speed
    z0, z1 = z_vec[0], z_vec[1]
    t00, t01, t11 = tau[0,0], tau[0,1], tau[1,1]
    
    # Check Im(tau) for convergence estimate
    im_t00 = RR(t00.imag())
    im_t11 = RR(t11.imag())
    
    if im_t00 <= 0 or im_t11 <= 0:
        raise ValueError(f"Im(tau) not positive definite: diag = [{im_t00}, {im_t11}]")
    
    # Minimum eigenvalue estimate (lower bound)
    y_min = min(im_t00, im_t11)
    
    if y_min < 0.01:
        raise ValueError(f"Im(tau) eigenvalue too small: y_min ~ {float(y_min)} - theta won't converge")
    
    # Adaptive radius: if y_min is large, we can use smaller radius
    # e^(-pi * n^2 * y_min) < epsilon requires n^2 > -log(epsilon) / (pi * y_min)
    from sage.all import log
    radius_needed = int(math.sqrt(float(-log(epsilon) / (RR(pi) * y_min)))) + 2
    radius = min(radius, radius_needed)
    
    r_range = range(-radius, radius + 1)
    max_possible_terms = len(r_range) ** 2
    
    if max_possible_terms > max_terms:
        raise RuntimeError(
            f"Theta summation would require {max_possible_terms} terms "
            f"(radius={radius}) which exceeds max_terms={max_terms}. "
            f"Im(tau) eigenvalue may be too small (y_min ~ {float(y_min)})"
        )
    
    # Track last ring contribution for convergence
    last_ring_contribution = None
    
    # Sum in expanding rings for better convergence detection
    for ring in range(radius + 1):
        ring_sum = CC(0)
        ring_count = 0
        
        if ring == 0:
            # Center term
            n1, n2 = 0, 0
            quad = 0
            lin = 0
            term_exponent = pi_I * (quad + lin)
            term = exp(term_exponent)
            total += term
            ring_sum = term
            ring_count = 1
            n_terms += 1
        else:
            # Terms at Chebyshev distance exactly 'ring' from origin
            for n1 in range(-ring, ring + 1):
                for n2 in range(-ring, ring + 1):
                    # Only include boundary of ring
                    if max(abs(n1), abs(n2)) != ring:
                        continue
                    
                    n_terms += 1
                    if n_terms > max_terms:
                        raise RuntimeError(
                            f"Theta summation exceeded {max_terms} terms without converging. "
                            f"Last ring {ring} contribution: {float(abs(ring_sum))}"
                        )
                    
                    quad = (n1*n1)*t00 + (2*n1*n2)*t01 + (n2*n2)*t11
                    lin = 2 * (n1*z0 + n2*z1)
                    
                    term_exponent = pi_I * (quad + lin)
                    term = exp(term_exponent)
                    ring_sum += term
                    ring_count += 1
            
            total += ring_sum
        
        # Check convergence
        ring_contribution = abs(ring_sum)
        
        if ring > 3 and ring_contribution < epsilon:
            # Converged! Can stop early
            break
        
        # Check if we're diverging (shouldn't happen with positive definite Im(tau))
        if last_ring_contribution is not None and ring > 5:
            if ring_contribution > last_ring_contribution * 1.5:
                raise RuntimeError(
                    f"Theta summation diverging at ring {ring}: "
                    f"contribution {float(ring_contribution)} > previous {float(last_ring_contribution)}"
                )
        
        last_ring_contribution = ring_contribution
    
    if abs(total) < epsilon:
        raise ValueError(f"Theta function computed to be essentially zero: |theta| = {float(abs(total))}")
    
    return total


def theta_direct(tau_in, z_in, R=3, prec_local=256):
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
