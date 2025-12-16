"""Riemann theta function computations."""

import math
from sage.all import ComplexField, exp, pi
from itertools import product


def compute_theta_high_prec(z_vec, tau, prec=300):
    """
    Computes Riemann Theta function theta(z, tau) at high precision.
    z_vec: vector of length 2 (Complex)
    tau: 2x2 symmetric matrix (Complex)
    """
    from sage.all import ComplexField, exp, pi
    import math
    
    CC = ComplexField(prec)
    
    # Pre-compute constants
    pi_I = CC(0, 1) * CC(pi)
    
    # Determine summation radius for precision
    # e^(-pi * n^2 * y_min) < 2^-prec
    # n^2 > prec * log(2) / (pi * y_min)
    # Assuming y_min ~ 0.3 (from your log), n ~ 25 is safe for 2048 bits
    radius = int(math.sqrt(prec * 0.25)) + 2 # Conservative estimate
    
    total = CC(0)
    
    # Naive summation over Z^2 (fast enough for genus 2)
    # Iterating -R to R
    r_range = range(-radius, radius + 1)
    
    # Extract components for speed
    z0, z1 = z_vec[0], z_vec[1]
    t00, t01, t11 = tau[0,0], tau[0,1], tau[1,1]
    
    for n1 in r_range:
        for n2 in r_range:
            # exponent = i*pi * (n^T * tau * n + 2 * n^T * z)
            # n^T tau n = n1^2 t00 + 2 n1 n2 t01 + n2^2 t11
            quad = (n1*n1)*t00 + (2*n1*n2)*t01 + (n2*n2)*t11
            lin = 2 * (n1*z0 + n2*z1)
            
            term_exponent = pi_I * (quad + lin)
            total += exp(term_exponent)
            
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

