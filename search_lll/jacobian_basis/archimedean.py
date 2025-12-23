"""Archimedean height corrections."""

from sage.all import (
    QQ, ZZ, RealField, ComplexField,
    Matrix, vector, pi
)

from .theta import *
from .periods import (
    choose_numerical_base_point,
    abel_jacobi_mumford,
    normalize_periods_and_z
)

import math
# Suppress Sage's inexact ring eigenvalue warnings - we know what we're doing
import warnings
warnings.filterwarnings('ignore', message='Using generic algorithm for an inexact ring')


def print_archimedean_diagnostics(tau, z, quad_val, log_theta, prec, debug=False):
    """
    tau : g x g complex matrix or nested list
    z   : length-g complex vector or list
    quad_val, log_theta : the scalar values used in archimedean height
    prec : bits of precision used (int)
    Prints diagnostics and returns a dict with values.
    """
    # create high-precision fields for robust numeric conversion
    CC = ComplexField(prec)
    RR = RealField(prec)
    # convert tau / z into CC objects and build Im(tau)
    g = len(z)
    # make tau a Matrix(CC)
    try:
        Tau = Matrix(CC, g, g, [[CC(tau[i][j]) for j in range(g)] for i in range(g)])
    except Exception:
        # maybe tau is a Sage Matrix already with complex entries
        Tau = Matrix(CC, g, g, [[CC(tau[i,j]) for j in range(g)] for i in range(g)])
    Z = vector(CC, [CC(z[i]) for i in range(g)])
    # Imaginary part matrix
    ImTau = Matrix(RR, g, g, [[RR((Tau[i,j]).imag) for j in range(g)] for i in range(g)])
    eigs = [float(e) for e in ImTau.eigenvalues()]
    # Norms
    z_norm = float(sum(abs(Z[i])**2 for i in range(g)))

    print("\n[ARCH DIAG] precision:", prec, "bits")
    print("[ARCH DIAG] tau (approx):")
    for i in range(g):
        print("  ",[complex(Tau[i,j]) for j in range(g)])
    print("[ARCH DIAG] Im(tau) eigenvalues:", eigs)
    print("[ARCH DIAG] z (approx):", [complex(Z[i]) for i in range(g)])
    print("[ARCH DIAG] ||z||^2:", z_norm)
    print("[ARCH DIAG] quad_val:", float(quad_val))
    print("[ARCH DIAG] log|theta| (value used):", float(log_theta))
    print("[ARCH DIAG] quad - logtheta:", float(quad_val - log_theta))
    # return a dict if caller wants to inspect
    return dict(prec=prec, tau=Tau, ImTau=ImTau, ImTau_eigs=eigs, z=Z, z_norm=z_norm,
                quad_val=CC(quad_val), log_theta=CC(log_theta),
                arch = CC(quad_val - log_theta))


def reduce_z_arakelov(z_list, tau, prec=300, debug=False, max_attempts=8, y_min_threshold=0.25):
    """
    Reduce z modulo lattice selecting representative maximizing quad - log|theta|.
    If Im(tau) eigenvalue(s) are small (slow convergence) we *fallback* to a conservative
    reduction (return z_base) to avoid repeated heavy theta computations.
    """
    key = (tuple(z_list), prec, max_attempts, y_min_threshold)
    if key in reduce_z_arakelov.cache:
        return reduce_z_arakelov.cache[key]
    RR = RealField(prec)
    CC = ComplexField(prec)

    g = tau.nrows()
    if g == 0:
        raise ValueError("Cannot reduce z on genus 0 surface (tau is 0x0)")
    
    assert len(z_list) == g

    # ensure CC matrix for tau
    Tau = Matrix(CC, tau)
    z0 = vector(CC, [CC(z) for z in z_list])

    Im_tau = Matrix(RR, g, g, [[RR(Tau[i, j].imag()) for j in range(g)] for i in range(g)])
    Im_tau = 0.5 * (Im_tau + Im_tau.transpose())
    det_im = Im_tau.det()
    if det_im <= 0:
        raise ValueError(f"Im(tau) not positive definite: det = {float(det_im)}")

    eigs = [float(e) for e in Im_tau.eigenvalues()]
    if not eigs:
        raise ValueError(f"No eigenvalues found for {g}x{g} matrix (Im(tau))")
    y_min = min(eigs)

    # Full lattice reduction to base representative
    y = vector(RR, [RR(z.imag()) for z in z0])
    c = Im_tau.solve_right(y)
    n = vector(ZZ, [ZZ(int(round(float(ci)))) * (-1) for ci in c])
    z1 = z0 + Tau * n
    m = vector(ZZ, [ZZ(int(round(float(z1[i].real())))) * (-1) for i in range(g)])
    z_base = z1 + m

    # If the imaginary part is too small (theta converges slowly), avoid the theta search:
    if y_min < y_min_threshold:
        if debug:
            print(f"[reduce_z_arakelov] Im(tau) y_min={y_min:.4g} < threshold={y_min_threshold}; skipping theta search.")
        return [CC(z_base[i]) for i in range(g)]

    # helper: quadratic form
    def quad_val(zvec):
        y_im = vector(RR, [RR(zvec[i].imag()) for i in range(g)])
        v = Im_tau.solve_right(y_im)
        return RR(pi) * y_im.dot_product(v)

    best_score = None
    best_z = None
    successful_evals = 0
    failed_evals = 0

    # enumerate half-period shifts (limited)
    limit = 1 << g
    for a_mask in range(limit):
        if failed_evals > max_attempts:
            if debug:
                print(f"[reduce_z_arakelov] Too many theta failures ({failed_evals}), stopping search")
            break

        a = vector(CC, [CC((a_mask >> i) & 1) for i in range(g)])
        for b_mask in range(limit):
            b = vector(CC, [CC((b_mask >> i) & 1) for i in range(g)])
            shift = CC(1)/CC(2) * (a + Tau * b)
            z_candidate = z_base + shift

            # Reduce real parts to fundamental interval [0,1)
            z_candidate = vector(CC, [z_candidate[i] - CC(int(round(float(z_candidate[i].real())))) for i in range(g)])

            try:
                from sage.all import log as sage_log, RR as sage_RR
                epsilon = sage_RR(2) ** (-(prec) + 8)
                radius_needed = int(math.sqrt(float(-sage_log(epsilon) / (RR(pi) * RR(y_min))))) + 2
            except Exception:
                radius_needed = 10

            try:
                if radius_needed > 18:
                    theta = theta_direct(tau, list(z_candidate), R=4, prec_local=max(128, prec//2))
                else:
                    theta = compute_theta_high_prec_parallel(list(z_candidate), tau, prec=prec, max_terms=20000)
                abs_theta = abs(CC(theta))
                if abs_theta <= 0:
                    failed_evals += 1
                    continue

                score = quad_val(z_candidate) - RR(abs_theta).log()
                successful_evals += 1

                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_z = z_candidate

            except (ValueError, RuntimeError) as e:
                failed_evals += 1
                continue

    if best_z is None:
        if debug:
            print(f"[reduce_z_arakelov] All shifts failed; returning lattice base.")
        return [CC(z_base[i]) for i in range(g)]

    ret = [CC(best_z[i]) for i in range(g)]
    reduce_z_arakelov.cache[key] = ret
    return ret
reduce_z_arakelov.cache = {}


def archimedean_height_correction(div, f_coeffs, period_matrix, prec=300, debug=False):
    """
    Exact archimedean height correction.
    """
    key = (str(div), tuple(f_coeffs), prec)
    if key in archimedean_height_correction.cache:
        return archimedean_height_correction.cache[key]

    if div.is_zero():
        return QQ(0)

    RR = RealField(prec)
    CC = ComplexField(prec)

    base_point = choose_numerical_base_point(f_coeffs, prec=prec)
    z_vec = abel_jacobi_mumford(div, f_coeffs, base_point=base_point, prec=prec)

    tau, z_norm_mat = normalize_periods_and_z(period_matrix, z_vec)
    g = tau.nrows()
    
    if g == 0:
        raise ValueError(f"Normalized period matrix has dimension 0. Inputs: period_matrix {period_matrix.nrows()}x{period_matrix.ncols()}, z_vec {len(z_vec)}")

    z_raw = [CC(z_norm_mat[i,0]) for i in range(g)]
    z = reduce_z_arakelov(z_raw, tau, prec=prec, debug=debug)

    # Im(tau)
    Im_tau = Matrix(RR, g, g, [[RR(CC(tau[i,j]).imag()) for j in range(g)] for i in range(g)])
    Im_tau = 0.5 * (Im_tau + Im_tau.transpose())
    det_im = Im_tau.det()
    assert det_im > 0

    y_im = vector(RR, [RR(zi.imag()) for zi in z])
    v = Im_tau.solve_right(y_im)
    quad = RR(pi) * y_im.dot_product(v)

    theta = compute_theta_high_prec_parallel(z, tau, prec=prec)
    abs_theta = abs(CC(theta))
    assert abs_theta > 0

    log_theta = RR(abs_theta).log()
    corr = QQ(1)/QQ(2) * RR(det_im).log()

    arch = quad - log_theta + corr

    if arch < -RR(1e-12):
        if debug:
            print(f"[ARCH] Warning: negative correction {float(arch)}")

    ret = QQ(arch)
    archimedean_height_correction.cache[key] = ret
    return ret
archimedean_height_correction.cache = {}
