"""Archimedean height corrections."""

from sage.all import (
    QQ, ZZ, RealField, ComplexField,
    Matrix, vector, pi
)

from .theta import compute_theta_high_prec, theta_direct
from .periods import (
    choose_numerical_base_point,
    abel_jacobi_mumford,
    normalize_periods_and_z
)


def archimedean_height_correction(div, f_coeffs, period_matrix, prec=300, debug=False):
    """
    Exact archimedean height correction.
    Crashes on any inconsistency.
    """
    if div.is_zero():
        return QQ(0)

    RR = RealField(prec)
    CC = ComplexField(prec)

    base_point = choose_numerical_base_point(f_coeffs, prec=prec)
    z_vec = abel_jacobi_mumford(div, f_coeffs, base_point=base_point, prec=prec)

    tau, z_norm_mat = normalize_periods_and_z(period_matrix, z_vec)
    g = tau.nrows()

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

    theta = compute_theta_high_prec(z, tau, prec=prec)
    abs_theta = abs(CC(theta))
    assert abs_theta > 0

    log_theta = RR(abs_theta).log()
    corr = QQ(1)/QQ(2) * RR(det_im).log()

    arch = quad - log_theta + corr

    if arch < -RR(1e-12):
        print("\n[ARCHIMEDEAN HEIGHT FAILURE]")
        print("quad =", float(quad))
        print("log|theta| =", float(log_theta))
        print("det(Im tau) =", float(det_im))
        print("z =", [complex(zi) for zi in z])
        raise RuntimeError("Archimedean height negative")

    return QQ(arch)


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


# put this near the TOP of archimedean_height_correction (or module-level)

def reduce_z_arakelov(z_list, tau, prec=300, debug=False, max_attempts=16):
    """
    Reduce z modulo Z^g + tau Z^g by testing all half-period shifts
    and choosing the representative maximizing quad(z) - log|theta(z)|.
    With fallback for theta computation failures.
    """
    RR = RealField(prec)
    CC = ComplexField(prec)

    g = tau.nrows()
    assert len(z_list) == g

    # Ensure tau is a CC matrix
    Tau = Matrix(CC, tau)

    # Work with CC vector
    z0 = vector(CC, [CC(z) for z in z_list])

    # Build Im(tau) as an RR matrix and require positive-definiteness
    Im_tau = Matrix(RR, g, g, [[RR(Tau[i, j].imag()) for j in range(g)] for i in range(g)])
    Im_tau = 0.5 * (Im_tau + Im_tau.transpose())
    det_im = Im_tau.det()
    
    if det_im <= 0:
        raise ValueError(f"Im(tau) not positive definite: det = {float(det_im)}")

    # First reduce by full lattice
    y = vector(RR, [RR(z.imag()) for z in z0])
    c = Im_tau.solve_right(y)
    n = vector(ZZ, [ZZ(int(round(float(ci)))) * (-1) for ci in c])
    z1 = z0 + Tau * n

    # Reduce by integer real translations
    m = vector(ZZ, [ZZ(int(round(float(z1[i].real())))) * (-1) for i in range(g)])
    z_base = z1 + m

    # Helper: quadratic form pi * y^T Im_tau^{-1} y
    def quad_val(zvec):
        y_im = vector(RR, [RR(zvec[i].imag()) for i in range(g)])
        v = Im_tau.solve_right(y_im)
        return RR(pi) * y_im.dot_product(v)

    best_score = None
    best_z = None
    successful_evals = 0
    failed_evals = 0

    # Enumerate all half-period shifts
    limit = 1 << g
    for a_mask in range(limit):
        if failed_evals > max_attempts:
            if debug:
                print(f"[reduce_z_arakelov] Too many theta failures ({failed_evals}), stopping search")
            break
        
        a = vector(CC, [CC((a_mask >> i) & 1) for i in range(g)])
        for b_mask in range(limit):
            b = vector(CC, [CC((b_mask >> i) & 1) for i in range(g)])
            shift = CC(1) / CC(2) * (a + Tau * b)
            z_candidate = z_base + shift

            # Reduce reals to fundamental interval
            z_candidate = vector(CC, [
                z_candidate[i] - CC(int(round(float(z_candidate[i].real())))) 
                for i in range(g)
            ])

            try:
                theta = compute_theta_high_prec(
                    list(z_candidate), 
                    tau, 
                    prec=prec,
                    max_terms=50000
                )
                abs_theta = abs(CC(theta))
                
                if abs_theta <= 0:
                    failed_evals += 1
                    if debug:
                        print(f"[reduce_z_arakelov] Zero theta at shift a={a_mask}, b={b_mask}")
                    continue

                score = quad_val(z_candidate) - RR(abs_theta).log()
                successful_evals += 1

                if debug:
                    print(f"[reduce_z_arakelov] shift ({a_mask},{b_mask}): score = {float(score):.6f}")

                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_z = z_candidate
                    
            except (ValueError, RuntimeError) as e:
                failed_evals += 1
                if debug:
                    print(f"[reduce_z_arakelov] Theta failed at shift a={a_mask}, b={b_mask}: {e}")
                continue

    if best_z is None:
        raise RuntimeError(
            f"reduce_z_arakelov: All {limit**2} half-period shifts failed. "
            f"Successful: {successful_evals}, Failed: {failed_evals}. "
            f"Im(tau) det = {float(det_im)}"
        )
    
    if debug:
        print(f"[reduce_z_arakelov] Final: {successful_evals} successful, {failed_evals} failed")
    
    return [CC(best_z[i]) for i in range(g)]
