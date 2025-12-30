"""Hardened Riemann theta function with multiple convergence strategies."""

import math
from sage.all import ComplexField, exp, pi
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from sage.all import ComplexField, RealField, exp, pi, log, Matrix
from sage.all import ComplexField, RealField, exp, pi

# tuning knobs (module-level; change at runtime if needed)
THETA_RADIUS_CAP = 18        # default cap (conservative)
THETA_MAX_TERMS = 20000      # default max terms
MAX_RADIUS = 60 # absolute max, do not raise it beyond this!  it is not combinatorially reasonable to do so!
THETA_EPS_FACTOR = 64        # default epsilon factor
MIN_EIGENVALUE_THRESHOLD = 0.01  # Skip if any Im(tau) eigenvalue below this


# Global parameters


theta_direct.cache = {}


compute_theta_high_prec.cache = {}


compute_theta_high_prec_parallel.cache = {}


"""Hardened Riemann theta function with multiple convergence strategies."""


# tuning knobs (module-level; change at runtime if needed)
THETA_RADIUS_CAP = 18        # default cap (conservative)
THETA_MAX_TERMS = 20000      # default max terms
MAX_RADIUS = 60 # absolute max, do not raise it beyond this!  it is not combinatorially reasonable to do so!
THETA_EPS_FACTOR = 64        # default epsilon factor
MIN_EIGENVALUE_THRESHOLD = 0.01  # Skip if any Im(tau) eigenvalue below this


# Global parameters

class ThetaComputationError(Exception):
    """Raised when theta computation is unsafe or fails."""
    pass


def _eigenvalues_im_tau(tau, prec=53):
    """
    Compute eigenvalues of Im(tau) to detect flat directions.
    
    Raises:
        ValueError: If tau is not positive definite
    """
    from sage.all import matrix, RealField
    RR = RealField(prec)
    
    # Handle both matrix and list-of-lists inputs
    if hasattr(tau, 'nrows'):  # It's a Sage matrix
        g = tau.nrows()
    else:  # It's a list of lists
        g = len(tau)
    
    # Extract imaginary part
    im_tau = matrix(RR, g, g)
    for i in range(g):
        for j in range(g):
            if hasattr(tau, 'nrows'):  # Sage matrix
                im_tau[i, j] = RR(tau[i, j].imag())
            else:  # List of lists
                im_tau[i, j] = RR(tau[i][j].imag())
    
    # Check symmetry
    if not im_tau.is_symmetric():
        raise ValueError("Im(tau) is not symmetric")
    
    # Compute eigenvalues
    try:
        eigvals = im_tau.eigenvalues()
    except Exception as e:
        raise ValueError(f"Failed to compute eigenvalues: {e}")
    
    eigvals_float = [float(ev) for ev in eigvals]
    
    # Check positivity
    if any(ev <= 0 for ev in eigvals_float):
        raise ValueError(f"Im(tau) not positive definite: eigenvalues = {eigvals_float}")
    
    return eigvals_float


def _reduce_tau_lll(tau, prec=2048):
    """
    Apply LLL reduction to tau to improve conditioning.
    
    For genus 2, attempts to find a basis where Im(tau) has larger eigenvalues.
    This doesn't always help with very flat curves, but worth trying.
    
    Returns:
        reduced_tau: Better-conditioned period matrix
        transform: GL(2,Z) transformation applied
    """
    from sage.all import matrix, ZZ
    
    # Handle both matrix and list-of-lists inputs
    if hasattr(tau, 'nrows'):
        g = tau.nrows()
    else:
        g = len(tau)
    
    CC = ComplexField(prec)
    
    # For genus 2, try simple transformations
    # More sophisticated: implement full Siegel reduction
    # For now, just return original
    # TODO: Implement proper Siegel fundamental domain reduction
    
    return tau, matrix(ZZ, g, g, 1)  # Identity transform


def theta_direct(tau_in, z_in, R=3, prec_local=2048):
    """
    Direct summation of theta function for genus 2, used for cheap screening.
    """
    key = (str(tau_in), str(z_in), R, prec_local)
    if key in theta_direct.cache:
        return theta_direct.cache[key]
    
    CC_loc = ComplexField(prec_local)
    
    # Handle both matrix and list-of-lists inputs for tau
    if hasattr(tau_in, 'nrows'):
        g_loc = tau_in.nrows()
        Tau = [[CC_loc(tau_in[i, j]) for j in range(g_loc)] for i in range(g_loc)]
    else:
        g_loc = len(tau_in)
        Tau = [[CC_loc(tau_in[i][j]) for j in range(g_loc)] for i in range(g_loc)]
    
    Z = [CC_loc(z_in[i]) for i in range(g_loc)]
    total = CC_loc(0)
    
    if g_loc == 2:
        for n0 in range(-R, R+1):
            for n1 in range(-R, R+1):
                q = Tau[0][0]*n0*n0 + (Tau[0][1]+Tau[1][0])*n0*n1 + Tau[1][1]*n1*n1
                linear = 2*(n0*Z[0] + n1*Z[1])
                arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
                total += CC_loc(exp(arg))
    elif g_loc == 1:
        for n0 in range(-R, R+1):
            q = Tau[0][0]*n0*n0
            linear = 2*n0*Z[0]
            arg = CC_loc(pi*1j) * q + CC_loc(pi*1j) * linear
            total += CC_loc(exp(arg))
    else:
        raise NotImplementedError("theta_direct optimization only implemented for g=1,2")
    
    theta_direct.cache[key] = total
    return total

theta_direct.cache = {}


def compute_theta_high_prec(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8):
    """
    Compute Riemann theta function for genus 2 with controlled truncation.
    
    Raises:
        ThetaComputationError: When convergence is unsafe or eigenvalues too small
        ValueError: For invalid inputs
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor)
    if key in compute_theta_high_prec.cache:
        return compute_theta_high_prec.cache[key]

    CC = ComplexField(prec)
    RR = RealField(prec)
    pi_I = CC(0, 1) * CC(pi)

    # Check eigenvalues first
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=min(prec, 200))
    except ValueError as e:
        raise ThetaComputationError(f"Im(tau) check failed: {e}")
    
    y_min = min(eigvals)
    
    # Hard cutoff for flat directions
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        raise ThetaComputationError(
            f"Im(tau) eigenvalue too small: {y_min:.6e} < {MIN_EIGENVALUE_THRESHOLD}. "
            "Use algebraic height instead or skip this divisor."
        )
    
    if y_min < RR(1e-2):
        raise ThetaComputationError(
            f"Im(tau) too close to boundary: y_min={float(y_min):.6e}"
        )

    # Precision target
    epsilon = RR(2) ** (-prec + epsilon_factor)

    radius_needed = int(
        math.sqrt(float(-log(epsilon) / (RR(pi) * y_min)))
    ) + 2

    # Safety caps
    genus = 2
    radius_cap = min(
        max(12, int(2 * genus * math.sqrt(prec))),
        MAX_RADIUS
    )

    radius = min(radius_needed, radius_cap)

    if radius_needed > MAX_RADIUS:
        raise ThetaComputationError(
            f"Radius needed ({radius_needed}) exceeds MAX_RADIUS ({MAX_RADIUS}). "
            f"Im(tau) eigenvalues: {eigvals}"
        )

    if radius < radius_needed:
        raise ThetaComputationError(
            f"Theta radius capped: needed {radius_needed}, using {radius}. "
            f"Im(tau) eigenvalues: {eigvals}. Increase Im(tau) or lower precision."
        )

    # Summation - handle both matrix and list-of-lists
    total = CC(0)
    z0, z1 = CC(z_vec[0]), CC(z_vec[1])
    
    if hasattr(tau, 'nrows'):  # Sage matrix
        t00, t01, t11 = CC(tau[0, 0]), CC(tau[0, 1]), CC(tau[1, 1])
    else:  # List of lists
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

        # Early termination check
        if ring > 4 and ring_abs < epsilon:
            break

        # Divergence detection
        if last_ring is not None and ring > 6:
            if ring_abs > last_ring * 2:
                raise ThetaComputationError(
                    f"Theta diverging at ring {ring}: "
                    f"|ring_sum|={float(ring_abs):.6e}, last={float(last_ring):.6e}"
                )

        last_ring = ring_abs

    if abs(total) < epsilon:
        raise ThetaComputationError(f"Theta numerically zero: |total|={float(abs(total)):.6e}")

    compute_theta_high_prec.cache[key] = total
    return total

compute_theta_high_prec.cache = {}


def compute_theta_high_prec_parallel(z_vec, tau, prec=2048, max_terms=20000,
                                     epsilon_factor=8, max_workers=None):
    """
    Parallelized theta summation with hardened error checking.
    
    Raises:
        ThetaComputationError: When convergence is unsafe
    """
    key = (tuple(z_vec), prec, max_terms, epsilon_factor)
    if key in compute_theta_high_prec_parallel.cache:
        return compute_theta_high_prec_parallel.cache[key]

    CC = ComplexField(prec)
    RR = RealField(max(53, min(prec, 200)))
    pi_I = CC(0, 1) * CC(pi)

    # Check eigenvalues
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=min(prec, 200))
    except ValueError as e:
        raise ThetaComputationError(f"Im(tau) check failed: {e}")
    
    y_min = min(eigvals)
    
    # Hard cutoff for flat directions
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        raise ThetaComputationError(
            f"Im(tau) eigenvalue too small: {y_min:.6e} < {MIN_EIGENVALUE_THRESHOLD}. "
            "Use algebraic height instead or skip this divisor."
        )
    
    if y_min < 1e-6:
        raise ThetaComputationError(f"Im(tau) too close to boundary: y_min={y_min:.6e}")

    # Precision and radius
    if prec <= epsilon_factor:
        radius_needed = 2
    else:
        numerator = (prec - float(epsilon_factor)) * math.log(2.0)
        denom = math.pi * y_min
        if numerator <= 0.0 or denom <= 0.0:
            raise ThetaComputationError("Invalid numerical values computing theta radius")
        radius_needed = int(math.sqrt(numerator / denom)) + 2

    # Safety caps
    genus = 2
    radius_cap = min(
        max(12, int(2 * genus * math.sqrt(float(prec)))),
        MAX_RADIUS
    )

    radius = min(radius_needed, radius_cap)

    if radius_needed > MAX_RADIUS:
        raise ThetaComputationError(
            f"Radius needed ({radius_needed}) exceeds MAX_RADIUS ({MAX_RADIUS}). "
            f"Im(tau) eigenvalues: {eigvals}"
        )

    if radius < radius_needed:
        raise ThetaComputationError(
            f"Theta radius capped: needed {radius_needed}, using {radius}. "
            f"Im(tau) eigenvalues: {eigvals}"
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

    # Precompute tau elements - handle both matrix and list-of-lists
    if hasattr(tau, 'nrows'):  # Sage matrix
        t00 = CC(tau[0, 0])
        t11 = CC(tau[1, 1])
        t01 = CC(tau[0, 1])
    else:  # List of lists
        t00 = CC(tau[0][0])
        t11 = CC(tau[1][1])
        t01 = CC(tau[0][1])
    
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

    import multiprocessing
    if max_workers is None:
        max_workers = max(8, multiprocessing.cpu_count() - 1)

    epsilon = 2 ** (-prec + epsilon_factor)

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
                raise ThetaComputationError(
                    f"Theta diverging at ring {ring}: "
                    f"ring_abs={ring_abs:.6e}, last={last_ring:.6e}"
                )

        last_ring = ring_abs

    if abs(total) < epsilon:
        raise ThetaComputationError(f"Theta numerically zero: |total|={float(abs(total)):.6e}")

    compute_theta_high_prec_parallel.cache[key] = total
    return total

compute_theta_high_prec_parallel.cache = {}


def compute_theta_safe(z_vec, tau, prec=2048, max_terms=20000, epsilon_factor=8,
                       parallel=True, max_workers=None):
    """
    Wrapper that attempts theta computation and returns None on failure.
    
    Use this in your basis construction loop to skip problematic divisors.
    
    Returns:
        theta value if successful, None if computation failed
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
        raise  # Re-raise unexpected errors


def check_tau_conditioning(tau, prec=53):
    """
    Check if tau is well-conditioned for theta computation.
    
    Returns:
        (is_safe, message, eigenvalues)
    """
    try:
        eigvals = _eigenvalues_im_tau(tau, prec=prec)
    except ValueError as e:
        return False, f"Im(tau) invalid: {e}", None
    
    y_min = min(eigvals)
    y_max = max(eigvals)
    condition = y_max / y_min if y_min > 0 else float('inf')
    
    if y_min < MIN_EIGENVALUE_THRESHOLD:
        return False, f"Eigenvalue too small: {y_min:.6e} < {MIN_EIGENVALUE_THRESHOLD}", eigvals
    
    if condition > 1000:
        return False, f"Poor conditioning: {condition:.2f}", eigvals
    
    return True, "OK", eigvals
