"""Hardened Riemann theta function with multiple convergence strategies."""

import math
from sage.all import ComplexField, RealField, exp, pi, log, Matrix, vector, IntegerRing
from concurrent.futures import ThreadPoolExecutor, as_completed

# tuning knobs (module-level; change at runtime if needed)
THETA_RADIUS_CAP = 18        # default cap (conservative)
THETA_MAX_TERMS = 200000      # default max terms
MAX_RADIUS = 260              # absolute max
THETA_EPS_FACTOR = 64        # default epsilon factor
MIN_EIGENVALUE_THRESHOLD = 0.0001  # Skip if any Im(tau) eigenvalue below this


class ThetaComputationError(Exception):
    """Raised when theta computation is unsafe or fails."""
    pass


def _eigenvalues_im_tau(tau, prec=53):
    """
    Compute eigenvalues of Im(tau) to detect flat directions.
    """
    RR = RealField(prec)
    
    # Handle both matrix and list-of-lists inputs
    if hasattr(tau, 'nrows'):
        g = tau.nrows()
    else:
        g = len(tau)
    
    # Extract imaginary part
    im_tau = Matrix(RR, 2, 2, [[RR(tau[i][j].imag()) for j in range(2)] for i in range(2)])

    # Check symmetry
    if not im_tau.is_symmetric():
        # Force symmetry if close
        im_tau = (im_tau + im_tau.transpose()) / 2

    try:
        eigvals = im_tau.eigenvalues()
    except Exception as e:
        raise ValueError(f"Failed to compute eigenvalues: {e}")
    
    eigvals_float = [float(ev) for ev in eigvals]
    return eigvals_float


def theta_direct(tau_in, z_in, R=3, prec_local=2048):
    """
    Direct summation of theta function for genus 2, used for cheap screening.
    """
    key = (str(tau_in), str(z_in), R, prec_local)
    if key in theta_direct.cache:
        return theta_direct.cache[key]
    
    CC_loc = ComplexField(prec_local)
    
    if hasattr(tau_in, 'nrows'):
        g_loc = tau_in.nrows()
        Tau = [[CC_loc(tau_in[i, j]) for j in range(g_loc)] for i in range(g_loc)]
    else:
        g_loc = len(tau_in)
        Tau = [[CC_loc(tau_in[i][j]) for j in range(g_loc)] for i in range(g_loc)]
    
    Z = [CC_loc(z_in[i]) for i in range(g_loc)]
    total = CC_loc(0)
    pi_I = CC_loc(0, 1) * CC_loc(pi)
    
    if g_loc == 2:
        for n0 in range(-R, R+1):
            for n1 in range(-R, R+1):
                q = Tau[0][0]*n0*n0 + (Tau[0][1]+Tau[1][0])*n0*n1 + Tau[1][1]*n1*n1
                linear = 2*(n0*Z[0] + n1*Z[1])
                arg = pi_I * (q + linear)
                total += exp(arg)
    elif g_loc == 1:
        for n0 in range(-R, R+1):
            q = Tau[0][0]*n0*n0
            linear = 2*n0*Z[0]
            arg = pi_I * (q + linear)
            total += exp(arg)
    else:
        raise NotImplementedError("theta_direct optimization only implemented for g=1,2")
    
    theta_direct.cache[key] = total
    return total

theta_direct.cache = {}


def _apply_functional_equation(z_vec, tau, prec):
    """
    Apply the modular functional equation: tau -> -1/tau.
    Maps small eigenvalues of Im(tau) to large eigenvalues.
    theta(z, tau) = exp(pi*i * z^T * tau^-1 * z) * det(-i*tau)^(-1/2) * theta(tau^-1*z, -tau^-1)
    """
    CC = ComplexField(prec)
    pi_I = CC(0, 1) * CC(pi)

    try:
        # 1. Convert inputs to Sage objects
        if hasattr(tau, 'inverse'):
            Tau = tau
        else:
            g = len(tau)
            Tau = Matrix(CC, g, g, tau)
            
        g = Tau.nrows()
        Z = vector(CC, z_vec)

        # 2. Compute inverted parameters
        # tau_prime = -tau^(-1)
        Tau_inv = Tau.inverse()
        Tau_prime = -Tau_inv
        
        # 3. Compute transformed arguments
        # z_prime = tau^(-1) * z
        Z_prime = Tau_inv * Z
        
        # 4. Compute pre-factors
        # Factor 1: exp(pi * i * (z . tau_inv . z))
        quad_term = Z * (Tau_inv * Z) # scalar
        factor_exp = exp(pi_I * quad_term)
        
        # Factor 2: det(-i * tau)^(-1/2)
        # Note: -i * tau = -i * (Re + i*Im) = Im - i*Re
        M_det = (-CC(0, 1) * Tau).determinant()
        factor_det = 1.0 / M_det.sqrt() 
        
        # 5. Recursive call
        z_prime_list = [Z_prime[i] for i in range(g)]
        
        # Pass check_functional_eq=False to prevent infinite recursion
        theta_inner = compute_theta_high_prec_parallel(
            z_prime_list, Tau_prime, prec=prec, 
            check_functional_eq=False 
        )
        
        res = factor_exp * factor_det * theta_inner
        return True, res

    except Exception:
        # If inversion fails or singularity occurs, we return False
        # and let the caller crash/raise naturally.
        return False, None


def compute_theta_high_prec(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8, check_functional_eq=True):
    """
    Compute Riemann theta function for genus 2 with controlled truncation.
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor, check_functional_eq)
    if key in compute_theta_high_prec.cache:
        return compute_theta_high_prec.cache[key]

    max_terms = max(THETA_MAX_TERMS, max_terms)

    CC = ComplexField(prec)
    RR = RealField(prec)
    pi_I = CC(0, 1) * CC(pi)

    # Check eigenvalues
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=min(prec, 200))
    except ValueError as e:
        raise ThetaComputationError(f"Im(tau) check failed: {e}")
    
    y_min = min(eigvals)
    
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        # Extremely small eigenvalues -> dangerous, but we try func eq below
        pass

    # Precision target
    epsilon = RR(2) ** (-prec + epsilon_factor)
    
    try:
        if y_min > 0:
            radius_needed = int(math.sqrt(float(-log(epsilon) / (RR(pi) * y_min)))) + 2
        else:
            radius_needed = MAX_RADIUS + 100
    except ValueError:
        radius_needed = MAX_RADIUS + 100

    # --- Functional Equation Fallback ---
    # Trigger if eigenvalues are small OR if we are about to crash due to radius
    if check_functional_eq:
        if y_min < 0.5 or radius_needed > MAX_RADIUS:
            success, res = _apply_functional_equation(z_vec, tau, prec)
            if success:
                compute_theta_high_prec.cache[key] = res
                return res
    # ------------------------------------

    # If we are here, either func eq failed or wasn't needed.
    # Check strict constraints now.
    
    if y_min < MIN_EIGENVALUE_THRESHOLD:
         raise ThetaComputationError(
            f"Im(tau) eigenvalue too small: {y_min:.6e} < {MIN_EIGENVALUE_THRESHOLD}. "
        )

    # Safety caps
    genus = 2
    if y_min < 0.01:
        radius_cap = min(max(25, int(3 * genus * math.sqrt(prec))), MAX_RADIUS)
    else:
        radius_cap = min(max(12, int(2 * genus * math.sqrt(prec))), MAX_RADIUS)

    radius = min(radius_needed, radius_cap)

    if radius_needed > MAX_RADIUS:
        raise ThetaComputationError(
            f"Radius needed ({radius_needed}) exceeds MAX_RADIUS ({MAX_RADIUS}). "
            f"Im(tau) eigenvalues: {eigvals}"
        )

    # Summation
    total = CC(0)
    z0, z1 = CC(z_vec[0]), CC(z_vec[1])
    
    if hasattr(tau, 'nrows'):
        t00, t01, t11 = CC(tau[0, 0]), CC(tau[0, 1]), CC(tau[1, 1])
    else:
        t00, t01, t11 = CC(tau[0][0]), CC(tau[0][1]), CC(tau[1][1])

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
                        raise ThetaComputationError(f"Exceeded max_terms ({max_terms})")

                    quad = (n1*n1)*t00 + 2*n1*n2*t01 + (n2*n2)*t11
                    lin = 2*(n1*z0 + n2*z1)
                    ring_sum += exp(pi_I * (quad + lin))

            total += ring_sum

        ring_abs = abs(ring_sum)
        if ring > 4 and ring_abs < epsilon:
            break
            
        if last_ring is not None and ring > 6:
            if ring_abs > last_ring * 2:
                raise ThetaComputationError(f"Theta diverging at ring {ring}")
        last_ring = ring_abs

    if abs(total) < epsilon:
        raise ThetaComputationError(f"Theta numerically zero: |total|={float(abs(total)):.6e}")

    compute_theta_high_prec.cache[key] = total
    return total

compute_theta_high_prec.cache = {}


def compute_theta_safe(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8,
                       parallel=True, max_workers=None):
    """
    Wrapper that attempts theta computation and returns None on failure.
    """
    try:
        if parallel:
            return compute_theta_high_prec_parallel(
                z_vec, tau, prec=prec, max_terms=max_terms,
                epsilon_factor=epsilon_factor, max_workers=max_workers
            )
        else:
            return compute_theta_high_prec(
                z_vec, tau, prec=prec, max_terms=max_terms,
                epsilon_factor=epsilon_factor
            )
    except ThetaComputationError:
        return None
    except Exception:
        raise

def check_tau_conditioning(tau, prec=53):
    """
    Check if tau is well-conditioned for theta computation.
    """
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=prec)
    except ValueError as e:
        return False, f"Im(tau) invalid: {e}", None
    
    y_min = min(eigvals)
    y_max = max(eigvals)
    condition = y_max / y_min if y_min > 0 else float('inf')
    
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        return False, f"Eigenvalue too small: {y_min:.6e}", eigvals
    
    return True, "OK", eigvals


def siegel_reduce_genus2(tau, prec=53):
    """
    Apply Siegel reduction to get tau into fundamental domain.
    This ensures Im(tau) eigenvalues are not too small or too large.
    """
    CC = ComplexField(prec)
    RR = RealField(prec)
    ZZ = IntegerRing()
    
    # Convert to matrix
    if not hasattr(tau, 'nrows'):
        g = len(tau)
        tau = Matrix(CC, g, g, tau)
    else:
        # Make a mutable copy
        tau = Matrix(CC, tau)
    
    max_iterations = 50
    
    for iteration in range(max_iterations):
        # Step 1: Reduce real parts mod 1 (bring to [-1/2, 1/2))
        for i in range(2):
            for j in range(2):
                re_part = tau[i,j].real()
                # Use Sage's round method or convert to int
                shift = ZZ(re_part.round()) if hasattr(re_part, 'round') else int(round(float(re_part)))
                tau[i,j] -= CC(shift, 0)
        
        # Step 2: Check if Im(tau) eigenvalues are in good range
        try:
            eigvals = _eigenvalues_im_tau(tau, prec=min(prec, 200))
        except ValueError:
            # If eigenvalue computation fails, return what we have
            return tau, iteration
            
        y_min, y_max = min(eigvals), max(eigvals)
        
        # Success criteria: eigenvalues in [0.1, 10]
        if y_min >= 0.1 and y_max <= 10:
            return tau, iteration
        
        # Step 3: Apply symplectic transformation if needed
        
        # If smallest eigenvalue is too small, invert
        if y_min < 0.5:
            try:
                tau_inv = tau.inverse()
                tau = -tau_inv
                continue
            except (ZeroDivisionError, ValueError):
                # Matrix is singular or nearly singular
                return tau, iteration
        
        # If largest eigenvalue is too large, also try inverting
        if y_max > 5:
            try:
                tau_inv = tau.inverse()
                tau = -tau_inv
                continue
            except (ZeroDivisionError, ValueError):
                return tau, iteration
        
        # If we get here, eigenvalues are in [0.5, 5] but not [0.1, 10]
        # This is "good enough" - don't loop forever
        if 0.2 <= y_min <= 3:
            return tau, iteration
        
        # Otherwise we're stuck, give up
        break
    
    return tau, max_iterations


def compute_theta_high_prec_parallel(z_vec, tau, prec=2048, max_terms=20000,
                                     epsilon_factor=8, max_workers=None, check_functional_eq=True):
    """
    Parallelized theta summation with hardened error checking.
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor, check_functional_eq)
    if key in compute_theta_high_prec_parallel.cache:
        return compute_theta_high_prec_parallel.cache[key]


    tau, _ = siegel_reduce_genus2(tau, prec=53)
    max_terms = max(THETA_MAX_TERMS, max_terms)

    CC = ComplexField(prec)
    RR = RealField(max(53, min(prec, 200)))
    pi_I = CC(0, 1) * CC(pi)

    # Check eigenvalues
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=min(prec, 200))
    except ValueError as e:
        raise ThetaComputationError(f"Im(tau) check failed: {e}")
    
    y_min = min(eigvals)
    
    # Precision and radius preliminary check
    epsilon = RR(2) ** (-prec + epsilon_factor)
    try:
        if y_min > 0:
            numerator = (prec - float(epsilon_factor)) * math.log(2.0)
            denom = math.pi * y_min
            radius_needed = int(math.sqrt(numerator / denom)) + 2
        else:
            radius_needed = MAX_RADIUS + 100
    except Exception:
        radius_needed = MAX_RADIUS + 100

    # Functional Equation Fallback
    if check_functional_eq:
        if y_min < 0.5 or radius_needed > MAX_RADIUS:
            success, res = _apply_functional_equation(z_vec, tau, prec)
            if success:
                compute_theta_high_prec_parallel.cache[key] = res
                return res
    
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        raise ThetaComputationError(
            f"Im(tau) eigenvalue too small: {y_min:.6e} < {MIN_EIGENVALUE_THRESHOLD}."
        )

    genus = 2
    if y_min < 0.01:
        radius_cap = min(max(25, int(3 * genus * math.sqrt(float(prec)))), MAX_RADIUS)
    else:
        radius_cap = min(max(12, int(2 * genus * math.sqrt(float(prec)))), MAX_RADIUS)

    radius = min(radius_needed, radius_cap)

    if radius_needed > MAX_RADIUS:
        raise ThetaComputationError(
            f"Radius needed ({radius_needed}) exceeds MAX_RADIUS ({MAX_RADIUS})."
        )

    # Chunk helper
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

    # Precompute tau elements
    if hasattr(tau, 'nrows'):
        t00, t11, t01 = CC(tau[0, 0]), CC(tau[1, 1]), CC(tau[0, 1])
    else:
        t00, t11, t01 = CC(tau[0][0]), CC(tau[1][1]), CC(tau[0][1])
    
    z0, z1 = CC(z_vec[0]), CC(z_vec[1])

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

    import multiprocessing
    if max_workers is None:
        max_workers = max(8, multiprocessing.cpu_count() - 1)

    for ring in range(radius + 1):
        if ring == 0:
            total += CC(1)
            n_terms += 1
            last_ring = abs(CC(1))
            continue

        pts = [(n1, n2)
               for n1 in range(-ring, ring + 1)
               for n2 in range(-ring, ring + 1)
               if max(abs(n1), abs(n2)) == ring]

        n_workers = min(max_workers, len(pts))
        if n_workers <= 0: 
            n_workers = 1

        chunk_list = list(_chunks(pts, n_workers))
        ring_sum = CC(0)
        ring_nterms = 0
        ring_max_abs = None

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
            raise ThetaComputationError(f"Exceeded max_terms ({max_terms})")

        total += ring_sum
        ring_abs = float(ring_max_abs) if ring_max_abs is not None else float(abs(ring_sum))
        
        if ring > 4 and ring_abs < epsilon:
            break

        if last_ring is not None and ring > 6:
            if ring_abs > float(last_ring) * 2:
                raise ThetaComputationError(f"Theta diverging at ring {ring}")
        last_ring = ring_abs

    if abs(total) < epsilon:
        raise ThetaComputationError(f"Theta numerically zero: |total|={float(abs(total)):.6e}")

    compute_theta_high_prec_parallel.cache[key] = total
    return total

compute_theta_high_prec_parallel.cache = {}
