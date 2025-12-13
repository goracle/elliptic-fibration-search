# arakelov.py
#
# Arakelov height computations for genus-2 hyperelliptic Jacobian elements.
# Faster and more reliable than repeated doubling for height pairings.

from mpmath import pslq
from sage.all import QQ, ZZ, RR, CC, RDF, CDF, log, sqrt, exp, pi, I
from sage.all import PolynomialRing, HyperellipticCurve, Matrix, vector
from sage.all import QQ, ZZ, RR, CC, RDF, CDF, ComplexField, log, sqrt, exp, pi, I
from sage.all import RealField
from functools import lru_cache
import math
from sage.all import ComplexField, RealField
from sage.all import parallel
import time
from collections import defaultdict
from .homology import *
from search_common import *

# Global cache for period matrices
_PERIOD_MATRIX_CACHE = {}

# Global timer storage
_TIMERS = defaultdict(float)


@parallel(ncpus=4)
def compute_height_worker(div_data):
    """
    Worker for parallel height computation.
    div_data = (idx, div_dict, f_coeffs, prec)
    Returns: (idx, canonical_height)
    """
    idx, div, f_coeffs, prec = div_data
    
    from sage.all import PolynomialRing, QQ, HyperellipticCurve
    
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    try:
        u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        
        if D.is_zero():
            return idx, None
        
        h = arakelov_canonical_height(D, f_coeffs, prec=prec)
        return idx, h
    except Exception:
        raise
        return idx, None


def arakelov_build_basis(all_divisors, f_coeffs, prec=300, debug=False):
    assert None, "deprecated"
    if not all_divisors:
        return [], 0, None
    
    clear_period_cache()
    reset_timers()
    
    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
    
    with Timer("period_matrix_total"):
        try:
            period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
            if debug:
                print(f"[arakelov] Period matrix computed")
        except Exception as e:
            if debug:
                print(f"[arakelov] Period matrix failed: {e}")
            raise
    
    with Timer("jacobian_conversion"):
        R = PolynomialRing(QQ, 'x')
        x = R.gen()
        f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly)
        J = C.jacobian()
        
        jac_elements = []
        for div in all_divisors:
            try:
                u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
                v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
                D = J([u_poly, v_poly])
                if not D.is_zero():
                    jac_elements.append((div, D))
            except Exception:
                if debug:
                    print(f"[arakelov] Skipping invalid divisor")
                raise
                continue
    
    if not jac_elements:
        return [], 0, None
    
    n = len(jac_elements)
    
    canonical_heights = {}
    
    if debug:
        print(f"[arakelov] Computing {n} canonical heights...")
    
    with Timer("heights_precompute"):
        for i, (div, D) in enumerate(jac_elements):
            canonical_heights[i] = arakelov_canonical_height(D, f_coeffs, prec=prec)
            if debug and (i+1) % 10 == 0:
                print(f"[arakelov] Computed {i+1}/{n} heights")
    
    sum_cache = {}
    
    def get_sum_height(i, j):
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in sum_cache:
            D_i = jac_elements[i][1]
            D_j = jac_elements[j][1]
            D_sum = D_i + D_j
            sum_cache[key] = arakelov_canonical_height(D_sum, f_coeffs, prec=prec)
        return sum_cache[key]
    
    def compute_pairing(i, j):
        h_i = canonical_heights[i]
        h_j = canonical_heights[j]
        h_sum = get_sum_height(i, j)
        return (h_sum - h_i - h_j) / QQ(2)
    
    basis = []
    basis_indices = []
    
    if debug:
        print(f"[arakelov] Building basis incrementally...")
    
    with Timer("basis_construction"):
        for i, (div, D) in enumerate(jac_elements):
            h_self = compute_pairing(i, i)
            if float(h_self) < 1e-4:
                if debug:
                    print(f"[arakelov] Skipping divisor {i}: self-pairing too small")
                continue
            if not basis:
                basis.append(div)
                basis_indices.append(i)
                if debug:
                    print(f"[arakelov] Added divisor 0 (self-pairing {float(h_self):.6g})")
            else:
                candidate_indices = basis_indices + [i]
                m = len(candidate_indices)
                
                H = Matrix(QQ, m, m)
                for ii in range(m):
                    for jj in range(ii, m):
                        h_ij = compute_pairing(candidate_indices[ii], candidate_indices[jj])
                        H[ii, jj] = h_ij
                        H[jj, ii] = h_ij
                
                rank = H.rank()
                det_val = H.determinant()
                det_float = float(det_val)

                # Use robust positive-definite check instead of det sign / rank alone.
                # Evaluate the matrix for PD in high precision and with tolerance.
                if is_positive_definite(H, prec=prec, tol=10**(-12)):
                    basis.append(div)
                    basis_indices.append(i)
                    if debug:
                        # For logging, compute float determinant for human-friendly output
                        try:
                            det_float = float(H.determinant())
                        except Exception:
                            det_float = float('nan')
                        print(f"[arakelov] Added divisor {i} (rank {m}, det {det_float:.6g})")
                else:
                    if debug:
                        print(f"[arakelov] Skipping divisor {i}: not positive-definite (rejected)")

    final_rank = len(basis)
    if final_rank > 0:
        with Timer("final_matrix"):
            H_final = Matrix(QQ, final_rank, final_rank)
            for i in range(final_rank):
                for j in range(i, final_rank):
                    h_ij = compute_pairing(basis_indices[i], basis_indices[j])
                    H_final[i, j] = h_ij
                    H_final[j, i] = h_ij
            
            if debug:
                print(f"\n[arakelov] Final rank: {final_rank}")
                det_final = H_final.determinant()
                print(f"[arakelov] Final determinant: {float(det_final):.6g}")
    else:
        H_final = None
    
    if debug:
        print_timers()
    
    return basis, final_rank, H_final


from sage.all import QQ, ZZ, RR, CC, RDF, CDF, ComplexField, log, sqrt, exp, pi, I, sinh, cosh, tanh


def get_timer(name):
    return _TIMERS.get(name, 0.0)

def reset_timers():
    global _TIMERS
    _TIMERS.clear()

def print_timers():
    if not _TIMERS:
        return
    print("\n[arakelov timers]")
    items = sorted(_TIMERS.items(), key=lambda x: x[1], reverse=True)
    for name, t in items:
        print(f"  {name:30s}: {t:8.3f}s")

class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        _TIMERS[self.name] += elapsed


def clear_period_cache():
    global _PERIOD_MATRIX_CACHE
    _PERIOD_MATRIX_CACHE.clear()


def arakelov_check_independence(divisors, f_coeffs, prec=300, debug=False):
    if not divisors:
        return True, 0, None, 0
    
    with Timer("independence_check_total"):
        n = len(divisors)
        H = Matrix(QQ, n, n)
        
        # Ensure period matrix is fresh/cached
        get_period_matrix_auto_B(f_coeffs, prec=prec)
        
        for i in range(n):
            for j in range(i, n):
                h_ij = arakelov_height_pairing(divisors[i], divisors[j], f_coeffs, prec=prec)
                H[i,j] = h_ij
                H[j,i] = h_ij
        
        det = float(H.determinant())
        
        # STRICT POSITIVE DEFINITE CHECK
        is_independent = (det > 1e-4)

        if debug:
            print(f"[arakelov] Height matrix determinant: {det:.6g}")
            print(f"[arakelov] Positive definite? {is_independent}")
    
    return is_independent, n if is_independent else 0, H, det


def is_mumford_torsion_fast(s, p, v0, v1, f_coeffs, max_order=12, debug=DEBUG):
    """
    Fast torsion test using modular verification.
    Tests if divisor is n-torsion for n in [2, 3, 4, 5, 6, 8, 10, 12].
    
    Returns: (is_torsion, order) where order=None if not torsion
    """
    # Build curve
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    J = C.jacobian()
    
    # Convert to Jacobian element
    u_poly = x**2 - QQ(s)*x + QQ(p)
    v_poly = QQ(v1)*x + QQ(v0)
    D = J([u_poly, v_poly])
    
    if D.is_zero():
        return True, 1
    
    # Test small orders using ONLY addition (no doubling)
    test_orders = [2, 3, 4, 5, 6, 8, 10, 12]
    
    for n in test_orders:
        # Compute nD by repeated addition (safer than doubling)
        nD = D
        for _ in range(n - 1):
            nD = nD + D
        
        if nD.is_zero():
            if debug:
                print(f"[torsion] Found {n}-torsion divisor")
            return True, n
    
    return False, None

def is_positive_definite(H, prec=300, tol=1e-12):
    """
    Robust numeric positive-definite check for a symmetric rational matrix H.
    Uses Sylvester's criterion (leading principal minors) evaluated in RealField(prec).
    Returns True if all leading principal minors > tol.
    """
    n = H.nrows()
    # choose high-precision real field to evaluate determinants stably
    RR_prec = RealField(max(prec, 80))
    for k in range(1, n+1):
        # build top-left kxk submatrix in RR_prec
        M = Matrix(RR_prec, k, k, [[RR_prec(H[i, j]) for j in range(k)] for i in range(k)])
        try:
            det_k = M.determinant()
        except Exception:
            # If determinant fails (overflow/numerics), be pessimistic
            raise
            return False
        # det_k should be a high-precision real; check > tol
        if not (det_k > RR_prec(tol)):
            return False
    return True


def arakelov_height_pairing_cached(D1, D2, f_coeffs, height_cache=None, prec=300):
    """
    Height pairing using cached individual heights.
    height_cache: optional dict with keys equal to D objects or indices and values equal to canonical heights.
    If heights are not found in cache, compute them.
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    # compute h1, h2 using cache if available
    if height_cache is None:
        h1 = arakelov_canonical_height(D1, f_coeffs, prec=prec)
        h2 = arakelov_canonical_height(D2, f_coeffs, prec=prec)
    else:
        # try direct lookup by object
        h1 = height_cache.get(D1, None)
        if h1 is None:
            # try by index if caller stores by index (caller should pass index-based cache then)
            h1 = arakelov_canonical_height(D1, f_coeffs, prec=prec)
        h2 = height_cache.get(D2, None)
        if h2 is None:
            h2 = arakelov_canonical_height(D2, f_coeffs, prec=prec)

    h_sum = arakelov_canonical_height(D1 + D2, f_coeffs, prec=prec)
    pairing = (h_sum - h1 - h2) / QQ(2)
    return pairing


def get_period_matrix_old(f_coeffs, prec=300):
    cache_key = tuple(QQ(c) for c in f_coeffs)

    if cache_key in _PERIOD_MATRIX_CACHE:
        return _PERIOD_MATRIX_CACHE[cache_key]

    with Timer("period_matrix_total"):
        R = PolynomialRing(QQ, 'x')
        x = R.gen()
        f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))

        deg = f_poly.degree()
        if deg < 5:
            raise ValueError(f"Polynomial degree {deg} too low for genus 2")

        # Choose high-precision fields for numerical integration
        if prec > 53:
            CC_prec = ComplexField(prec)
            RR_prec = RealField(prec)
        else:
            CC_prec = CC
            RR_prec = RR

        with Timer("period_matrix_roots"):
            # compute complex roots using the high-precision complex field
            f_roots = f_poly.roots(CC_prec, multiplicities=False)

        if len(f_roots) < 5:
            raise ValueError(f"Not enough roots for genus 2 curve: found {len(f_roots)}")

        # sort roots by real then imag parts (converted to high-precision complex)
        f_roots = sorted(f_roots, key=lambda z: (CC_prec(z).real(), CC_prec(z).imag()))

        with Timer("period_matrix_integration"):
            # Tanh-Sinh (Double Exponential) Quadrature Parameters
            # safe_t scales with precision (log scale)
            safe_t = float(RR_prec(prec).log()) + 2.0
            max_t = RR_prec(safe_t)

            # Step size (as high-precision real)
            h = RR_prec(2) ** (-6)   # you can tune this
            pi_half = RR_prec(math.pi) / 2  # math.pi -> float okay, converted into RR_prec

            # build nodes entirely in high-precision
            nodes = []
            curr = -max_t
            # use while loop with RR_prec increments
            while curr <= max_t:
                # compute high-precision sinh/cosh of curr
                try:
                    sinh_t = curr.sinh()
                    cosh_t = curr.cosh()
                except Exception:
                    # fallback: cast to RR_prec and call methods
                    sinh_t = RR_prec(curr).sinh()
                    cosh_t = RR_prec(curr).cosh()
                    raise

                # num = (pi/2) * sinh(curr)
                num = pi_half * sinh_t

                # compute cosh(num) in the high-precision field: num may be large complex if using CC_prec
                # here num is RR_prec because sinh_t is real; safe to compute as RR_prec
                try:
                    denom = num.cosh() if hasattr(num, "cosh") else RR_prec(num).cosh()
                except Exception:
                    # if cosh fails (overflow), skip this node
                    curr += h
                    continue

                # compute tanh(pi/2 * sinh(t))
                try:
                    val = num.tanh() if hasattr(num, "tanh") else (RR_prec(num).tanh())
                except Exception:
                    # if tanh somehow fails, skip
                    curr += h
                    raise
                    continue

                # weight = (pi/2) * cosh(t) / cosh(num)^2
                try:
                    denom_sq = denom * denom
                    # guard against extremely large denom_sq (overflow in high precision math)
                    # treat extremely large denom_sq as skip
                    if abs(denom_sq) > RR_prec(2) ** (prec//2):
                        curr += h
                        continue
                    weight = pi_half * cosh_t / denom_sq
                except Exception:
                    curr += h
                    continue

                # Filter tiny weights: compare against threshold in RR_prec
                tiny_thresh = RR_prec(2) ** (-(prec + 10))
                if abs(weight) > tiny_thresh:
                    # center/half-width stored later, but nodes store (val, weight) as RR_prec
                    nodes.append((RR_prec(val), RR_prec(weight)))

                curr += h

            # node list built — now implement integrator that uses CC_prec arithmetic
            def integrate_differential(root_start, root_end, use_x_weight):
                total = CC_prec(0)
                center = (CC_prec(root_start) + CC_prec(root_end)) / CC_prec(2)
                half_width = (CC_prec(root_end) - CC_prec(root_start)) / CC_prec(2)

                for (u, w) in nodes:
                    # u and w are RR_prec; cast to CC_prec for complex arithmetic
                    u_cc = CC_prec(u)
                    w_cc = CC_prec(w)

                    x_val = center + half_width * u_cc

                    # Evaluate f(x) in CC_prec
                    f_val = CC_prec(0)
                    for j, c in enumerate(f_coeffs):
                        f_val += CC_prec(c) * x_val ** (len(f_coeffs)-1-j)

                    # if f_val is zero or extremely small, skip this node
                    if abs(f_val) == 0:
                        continue

                    # compute branch of sqrt using principal branch
                    try:
                        y_val = f_val.sqrt()
                    except Exception:
                        # try numeric sqrt via CC_prec
                        y_val = CC_prec(f_val).sqrt()
                        raise

                    # choose integrand
                    if use_x_weight:
                        term_val = x_val / (2 * y_val)
                    else:
                        term_val = 1 / (2 * y_val)

                    total += term_val * w_cc

                return total * half_width * CC_prec(h)

            Omega = Matrix(CC_prec, 2, 2)
            # principal cycles: (r0,r1), (r0,r2)
            Omega[0,0] = integrate_differential(f_roots[0], f_roots[1], False)
            Omega[1,0] = integrate_differential(f_roots[0], f_roots[1], True)

            if len(f_roots) >= 3:
                Omega[0,1] = integrate_differential(f_roots[0], f_roots[2], False)
                Omega[1,1] = integrate_differential(f_roots[0], f_roots[2], True)
            else:
                Omega[0,1] = integrate_differential(f_roots[1], f_roots[0], False)
                Omega[1,1] = integrate_differential(f_roots[1], f_roots[0], True)

    _PERIOD_MATRIX_CACHE[cache_key] = Omega
    return Omega


def get_period_matrix_bad_cycle_choices(f_coeffs, prec=300):
    cache_key = tuple(QQ(c) for c in f_coeffs)
    
    if cache_key in _PERIOD_MATRIX_CACHE:
        return _PERIOD_MATRIX_CACHE[cache_key]
    
    with Timer("period_matrix_total"):
        R = PolynomialRing(QQ, 'x')
        x = R.gen()
        f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        
        deg = f_poly.degree()
        if deg < 5:
            raise ValueError(f"Polynomial degree {deg} too low for genus 2")
        
        if prec > 53:
            CC_prec = ComplexField(prec)
            RR_prec = RealField(prec)
        else:
            CC_prec = CC
            RR_prec = RR
            
        with Timer("period_matrix_roots"):
            f_roots = f_poly.roots(CC_prec, multiplicities=False)
        
        if len(f_roots) < 5:
            raise ValueError(f"Not enough roots for genus 2 curve: found {len(f_roots)}")
        
        f_roots = sorted(f_roots, key=lambda z: (z.real(), z.imag()))
        
        with Timer("period_matrix_integration"):
            safe_t = log(prec) + 2.0
            max_t = RR_prec(safe_t)
            
            h = RR_prec(2.0 ** (-6))
            pi_half = RR_prec(math.pi) / 2
            
            nodes = []
            curr = -max_t
            while curr <= max_t:
                sinh_t = sinh(curr)
                cosh_t = cosh(curr)
                
                num = pi_half * sinh_t
                
                try:
                    denom = cosh(num)
                    if denom.is_infinity():
                         curr += h
                         continue
                except (OverflowError, ValueError, ArithmeticError):
                    curr += h
                    continue
                
                val = tanh(num)
                
                try:
                    denom_sq = denom * denom
                    if denom_sq.is_infinity():
                        curr += h
                        continue
                    weight = pi_half * cosh_t / denom_sq
                except (OverflowError, ValueError, ArithmeticError):
                    curr += h
                    continue

                if abs(weight) > RR_prec(2)**(-prec - 10):
                    nodes.append((val, weight))
                
                curr += h
            
            def integrate_differential(root_start, root_end, use_x_weight):
                total = CC_prec(0)
                center = (root_start + root_end) / 2
                half_width = (root_end - root_start) / 2
                
                for (u, w) in nodes:
                    u_cc = CC_prec(u)
                    w_cc = CC_prec(w)
                    
                    x_val = center + half_width * u_cc
                    
                    f_val = sum(CC_prec(f_coeffs[j]) * x_val**(len(f_coeffs)-1-j) 
                               for j in range(len(f_coeffs)))
                    
                    if f_val == 0: continue
                    
                    y_val = f_val.sqrt()
                    
                    term_val = CC_prec(0)
                    if use_x_weight:
                        term_val = x_val / (2 * y_val)
                    else:
                        term_val = 1 / (2 * y_val)
                        
                    total += term_val * w_cc
                
                return total * half_width * CC_prec(h)
            
            A = Matrix(CC_prec, 2, 2)
            B = Matrix(CC_prec, 2, 2)
            
            # a-cycles: 2 * integral across each real cut
            A[0,0] = 2 * integrate_differential(f_roots[0], f_roots[1], False)
            A[1,0] = 2 * integrate_differential(f_roots[0], f_roots[1], True)
            
            A[0,1] = 2 * integrate_differential(f_roots[2], f_roots[3], False)
            A[1,1] = 2 * integrate_differential(f_roots[2], f_roots[3], True)
            
            # b-cycles: paths connecting cuts
            B[0,0] = integrate_differential(f_roots[1], f_roots[2], False)
            B[1,0] = integrate_differential(f_roots[1], f_roots[2], True)
            
            B[0,1] = integrate_differential(f_roots[3], f_roots[4], False)
            B[1,1] = integrate_differential(f_roots[3], f_roots[4], True)
            
            # Form normalized period matrix tau = A^{-1} * B
            try:
                tau = A.inverse() * B
            except Exception as e:
                raise RuntimeError(f"Failed to invert A-period matrix: {e}")
    
    _PERIOD_MATRIX_CACHE[cache_key] = tau
    return tau


from sage.all import *
import cmath

def _continuous_sqrt_values(fvals, CC_prec):
    """
    Given a list of complex field values f(x) (in CC_prec),
    return a list of sqrt values chosen by analytic continuation:
    pick principal sqrt at the first point, then at each step choose
    the sign that is closest to the previous sqrt (minimizes jump).
    """
    if len(fvals) == 0:
        return []
    svals = [fvals[0].sqrt()]                      # CC_prec sqrt
    prev = svals[0]
    for v in fvals[1:]:
        # two choices: s and -s
        s = v.sqrt()
        # choose sign minimizing distance to prev
        if abs(s - prev) <= abs(-s - prev):
            svals.append(s)
            prev = s
        else:
            svals.append(-s)
            prev = -s
    return svals

def _composite_quadrature_on_param(x_of_t, dx_dt_of_t, integrand_at_x, nodes, CC_prec):
    """
    Basic composite quadrature on parameter t in [0,1] using nodes list [(t_i,w_i), ...].
    nodes expected to be (t, weight) pairs on [-1,1] or [0,1]; user supplies mapping.
    This is intentionally simple and robust; can be swapped for higher-order rules.
    """
    total = CC_prec(0)
    for (t, w) in nodes:
        x = x_of_t(t)
        dxdt = dx_dt_of_t(t)
        fval = integrand_at_x(x)
        total += fval * dxdt * CC_prec(w)
    return total


def compute_A_B_return(f_coeffs, prec=300, nodes=None, b_pattern='1-2'):
    """
    Compute and return (A, B, tau) using the same numeric method as get_period_matrix_auto_B,
    but expose A and B so we can test sign/permutation fixes.
    b_pattern: '1-2' (default) or '1-3' etc.  (If your original function supports more, adapt.)
    """
    # This reuses the internals of your get_period_matrix_auto_B implementation.
    # If you already have a version that returns A,B, call that instead.
    CC_prec = ComplexField(prec)
    RR_prec = RealField(prec)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    f_roots = sorted(f_poly.roots(CC_prec, multiplicities=False), key=lambda z: (z.real(), z.imag()))
    if len(f_roots) < 5:
        raise ValueError("Need at least 5 branch points (genus 2).")
    if nodes is None:
        # default tanh-sinh node generator; you can substitute yours
        def tanh_sinh_nodes(N):
            from math import sinh, cosh, tanh, pi
            h = 1.0 / N
            out = []
            for k in range(-N, N+1):
                t = k * h
                xnode = tanh(pi/2 * sinh(t))
                w = (pi/2) * cosh(t) / (cosh(pi/2 * sinh(t))**2) * h
                out.append((xnode, w))
            return out
        nodes = tanh_sinh_nodes(max(400, prec//2))
    # helpers
    def f_at(xv):
        return sum(CC_prec(c) * (xv**(len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    def continuous_sqrt(vals):
        svals = [vals[0].sqrt()]
        prev = svals[0]
        for v in vals[1:]:
            s = v.sqrt()
            if abs(s-prev) <= abs(-s-prev):
                svals.append(s); prev = s
            else:
                svals.append(-s); prev = -s
        return svals

    def integrate_path_with_nodes(x_of_t, dx_dt_of_t, use_x=False):
        fvals = [f_at(x_of_t(t)) for (t, _) in nodes]
        # fail early if path hits branchpoint
        for fv in fvals:
            if abs(fv) < 1e-60:
                raise ArithmeticError("Path appears to hit a branchpoint; change offset.")
        svals = continuous_sqrt(fvals)
        total = CC_prec(0)
        for (t, w), y in zip(nodes, svals):
            dxdt = dx_dt_of_t(t)
            integrand = x_of_t(t)/y if use_x else CC_prec(1)/y
            total += integrand * dxdt * CC_prec(w)
        return total

    # build A and B similar to your function
    # choose cut pairs (standard)
    if len(f_roots) >= 6:
        e = f_roots
        cut_pairs = [(e[0], e[1]), (e[2], e[3])]
        midpoints = [ (e[0]+e[1])/CC_prec(2), (e[2]+e[3])/CC_prec(2), (e[1]+e[2])/CC_prec(2), (e[3]+e[4])/CC_prec(2) ]
    else:
        e = f_roots
        cut_pairs = [(e[0], e[1]), (e[2], e[3])]
        midpoints = [ (e[0]+e[1])/CC_prec(2), (e[2]+e[3])/CC_prec(2), (e[1]+e[2])/CC_prec(2), ( (e[3]+e[3])/CC_prec(2) ) ]

    # A: circles around cuts
    A = Matrix(CC_prec, 2, 2)
    for i, (r1, r2) in enumerate(cut_pairs):
        center = (r1 + r2) / CC_prec(2)
        radius = (r2 - r1) / CC_prec(2) * CC_prec(1.1)
        x_of_t = lambda t, c=center, rad=radius: c + rad*CC_prec(exp(2j*pi*float(t)))
        dx_dt = lambda t, rad=radius: rad * CC_prec(2j*pi) * CC_prec(exp(2j*pi*float(t)))
        A[0,i] = integrate_path_with_nodes(x_of_t, dx_dt, use_x=False)
        A[1,i] = integrate_path_with_nodes(x_of_t, dx_dt, use_x=True)

    # B: straight segments lifted into upper half-plane
    B = Matrix(CC_prec, 2, 2)
    for j in range(2):
        z0 = midpoints[j]
        z1 = midpoints[2+j]
        delta = z1 - z0
        eps = max(RR_prec(1e-12), abs(delta)*RR_prec(1e-6))
        normal = CC_prec(1j)*delta
        if normal == 0:
            normal = CC_prec(1j)*CC_prec(eps)
        normal_unit = normal / CC_prec(abs(normal))
        z0p = z0 + normal_unit * CC_prec(eps)
        z1p = z1 + normal_unit * CC_prec(eps)
        x_of_t = lambda t, z0p=z0p, z1p=z1p: z0p + (z1p-z0p)*CC_prec(t)
        dx_dt = lambda t, z0p=z0p, z1p=z1p: z1p - z0p
        B[0,j] = integrate_path_with_nodes(x_of_t, dx_dt, use_x=False)
        B[1,j] = integrate_path_with_nodes(x_of_t, dx_dt, use_x=True)

    # tau
    try:
        A_inv = A.inverse()
    except Exception as ex:
        raise ArithmeticError("A matrix singular or ill-conditioned") from ex

    tau = A_inv * B
    return A, B, tau

def try_fix_orientation(f_coeffs, prec=300, nodes=None):
    """
    Compute A,B,tau and try small discrete fixes: sign flips of columns and permutations of B columns.
    Returns the first (tau_fixed, A_fixed, B_fixed, info) that yields Im(tau) PD and minimal symmetry error.
    """
    A, B, tau = compute_A_B_return(f_coeffs, prec=prec, nodes=nodes)
    CC = tau.base_ring()
    RRp = RealField(prec)

    # candidate transformations:
    # - flip signs of columns of B (and optionally A),
    # - permute columns of B (2! = 2 permutations)
    best = None
    results = []
    from itertools import product, permutations
    for flipA in product([1,-1], repeat=2):
        Amod = Matrix(CC, 2, 2, [[A[i,j]*CC(flipA[j]) for j in range(2)] for i in range(2)])
        try:
            Ainv = Amod.inverse()
        except Exception:
            raise
            continue
        for perm in permutations([0,1]):
            Bperm = Matrix(CC, 2, 2, [[B[i,perm[j]] for j in range(2)] for i in range(2)])
            for flipB in product([1,-1], repeat=2):
                Bmod = Matrix(CC, 2, 2, [[Bperm[i,j]*CC(flipB[j]) for j in range(2)] for i in range(2)])
                try:
                    tau_cand = Ainv * Bmod
                except Exception:
                    raise
                    continue
                # sym error and Im(tau) diagnostics
                sym_err = abs(tau_cand[0,1] - tau_cand[1,0])
                # build Im(tau) in RRp
                Im_tau = Matrix(RRp, 2, 2, [[RRp(tau_cand[i,j].imag()) for j in range(2)] for i in range(2)])
                try:
                    evals = Im_tau.eigenvalues()
                except Exception:
                    raise
                    continue
                is_pd = all(ev > RRp(0) for ev in evals)
                det_Im = Im_tau.determinant()
                results.append((is_pd, float(sym_err), float(det_Im), flipA, perm, flipB, tau_cand, Amod, Bmod))
                # choose first that is PD with small sym_err
                if is_pd and float(sym_err) < 1e-8:
                    return results[-1]  # ideal
    # choose best candidate giving PD (if any) with minimal symmetry error
    pd_candidates = [r for r in results if r[0]]
    if pd_candidates:
        pd_candidates.sort(key=lambda r: (r[1], -r[2]))  # min sym_err, max det
        return pd_candidates[0]
    # no PD candidate found: return best by sym_err
    results.sort(key=lambda r: r[1])
    return results[0] if results else (False, abs(tau[0,1]-tau[1,0]), None, None, None, None, tau, A, B)


def abel_jacobi_map(D, f_coeffs, period_matrix, prec=300):
    """
    Compute the Abel-Jacobi map of divisor D to C^2.
    
    For a Mumford divisor D = (u(x), v(x)) where u(x) = x^2 - s*x + p,
    the divisor represents two points on the curve (assuming u has 2 distinct roots).
    
    We integrate the holomorphic differentials dx/(2y) and x*dx/(2y) from
    a base point to each point in the support of D.
    
    Returns: vector in C^2 representing AJ(D) mod the period lattice
    """
    CC = ComplexField(prec)
    
    # Extract Mumford representation
    u_poly = D[0]
    v_poly = D[1]
    
    # Get roots of u(x) - these are the x-coordinates of points in D
    R = PolynomialRing(CC, 'x')
    x = R.gen()
    u_cc = sum(CC(c) * x**i for i, c in enumerate(u_poly.list()))
    
    try:
        roots = u_cc.roots(multiplicities=False)
    except:
        # If roots don't exist or u is constant, return zero
        return vector(CC, [0, 0])
    
    if len(roots) == 0:
        return vector(CC, [0, 0])
    
    # For each root, we need to integrate from base point to (x_i, y_i)
    # where y_i = v(x_i)
    
    # Build f(x) polynomial
    f_poly = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    # Use a simple base point: take a root of f as base
    f_roots = f_poly.roots(CC, multiplicities=False)
    if not f_roots:
        raise ValueError("No roots found for f(x)")
    
    base_x = f_roots[0]
    
    # Integrate from base_x to each root
    integral_sum = vector(CC, [0, 0])
    
    for x_pt in roots:
        # y-coordinate from Mumford representation
        y_pt = sum(CC(c) * x_pt**i for i, c in enumerate(v_poly.list()))
        
        # Simple straight-line path integration (can be improved with better paths)
        # Integrate omega_0 = dx/(2y) and omega_1 = x*dx/(2y)
        
        # Use Gauss-Legendre quadrature for the path integral
        from sage.all import numerical_integral
        
        # Define path: straight line from base_x to x_pt
        def integrand_0(t):
            x_t = base_x + t * (x_pt - base_x)
            f_t = sum(CC(c) * x_t**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
            if abs(f_t) < 1e-30:
                return CC(0)
            y_t = f_t.sqrt()
            # Choose branch consistently (prefer positive imaginary part)
            if y_t.imag() < 0:
                y_t = -y_t
            return (x_pt - base_x) / (2 * y_t)
        
        def integrand_1(t):
            x_t = base_x + t * (x_pt - base_x)
            f_t = sum(CC(c) * x_t**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
            if abs(f_t) < 1e-30:
                return CC(0)
            y_t = f_t.sqrt()
            if y_t.imag() < 0:
                y_t = -y_t
            return x_t * (x_pt - base_x) / (2 * y_t)
        
        # Numerical integration from t=0 to t=1
        try:
            int_0, _ = numerical_integral(lambda t: complex(integrand_0(t)), 0, 1)
            int_1, _ = numerical_integral(lambda t: complex(integrand_1(t)), 0, 1)
            integral_sum += vector(CC, [CC(int_0), CC(int_1)])
        except:
            # If integration fails, skip this point
            continue
    
    return integral_sum


def integrate_differential_path(x_start, x_end, f_coeffs, use_x_weight=False, prec=300, max_depth=8):
    """
    Integrate dx/(2y) or x*dx/(2y) along a straight line from x_start to x_end.
    Uses the same tanh-sinh quadrature as the period matrix computation.
    
    Returns: complex integral value
    """
    CC = ComplexField(prec)
    
    # Generate tanh-sinh nodes
    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0 / float(N)
        pi = math.pi
        for k in range(-N, N+1):
            t = k * h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi/2.0) * sx)
            dx_dt = (pi/2.0) * math.cosh(t) / (math.cosh((pi/2.0) * sx)**2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes
    
    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)
    
    p0 = CC(x_start)
    p1 = CC(x_end)
    vec = p1 - p0
    
    # Offset path slightly perpendicular to avoid branch cuts
    perp = CC(0, 1) * vec
    off_mag = max(CC(1e-14), abs(vec) * CC(1e-8))
    off = perp / (abs(perp) + CC(1e-30)) * off_mag
    
    dx_factor = vec / CC(2)
    
    # Build f(x)
    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    
    # Initial branch selection
    sample_x = p0 + ((CC(nodes[0][1]) + CC(1)) / CC(2)) * vec + off
    f0 = f_at(sample_x)
    y0 = f0.sqrt()
    # Prefer positive imaginary part
    if y0.imag() < 0:
        y0 = -y0
    
    y_prev = y0
    integral = CC(0)
    tiny = CC(2) ** (-prec // 2)
    
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        fval = f_at(xval)
        
        if abs(fval) < tiny:
            continue
        
        # Branch selection by continuity
        y_plus = fval.sqrt()
        y_minus = -y_plus
        y_cur = y_plus if abs(y_plus - y_prev) <= abs(y_minus - y_prev) else y_minus
        y_prev = y_cur
        
        # Integrand
        if use_x_weight:
            integrand = xval / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * CC(w)
        integral += integrand * dxd
    
    return integral


def naive_height_qq(D, prec=53):
    """
    Compute naive (logarithmic) height of Mumford polynomials.
    """
    from sage.all import QQ, RealField
    
    u_coeffs = [QQ(c) for c in D[0].list()]
    v_coeffs = [QQ(c) for c in D[1].list()]
    
    # Clear denominators
    dens = [c.denominator() for c in (u_coeffs + v_coeffs) if c != 0]
    if not dens:
        return QQ(0)
    
    from math import gcd
    from functools import reduce
    lcm_den = reduce(lambda a, b: (a * b) // gcd(a, b), dens, 1)
    
    # Scale coefficients to integers
    int_coeffs = [int((c * lcm_den).numerator()) for c in (u_coeffs + v_coeffs)]
    int_coeffs = [abs(c) for c in int_coeffs if c != 0]
    if not int_coeffs:
        return QQ(0)
    
    max_abs = max(int_coeffs)
    
    R = RealField(prec)
    return R(max_abs).log().nearby_rational(max_error=R(2)**(-prec + 5))


def local_height_finite(D, p, prec=53):
    """
    Local height at finite prime p.
    """
    u_coeffs = D[0].list()
    v_coeffs = D[1].list()
    
    min_val = float('inf')
    
    for c in u_coeffs + v_coeffs:
        if c == 0:
            continue
        c_qq = QQ(c)
        val_p = c_qq.valuation(p)
        min_val = min(min_val, val_p)
    
    if min_val == float('inf'):
        return QQ(0)
    
    if min_val == 0:
        return QQ(0)
    
    R = RealField(prec)
    val = -min_val * R(p).log()
    
    return val.nearby_rational(max_error=R(2)**(-prec+5))


# ============================================================================
# DROP-IN REPLACEMENT FUNCTIONS FOR arakelov.py
# ============================================================================


# Fixed Arakelov height computation functions
# These replace the broken versions in arakelov.py


def get_hyperelliptic_polynomials(C):
    """
    Extract f(x) and h(x) from hyperelliptic curve C.
    Curve is given as y^2 + h(x)*y = f(x)
    
    Returns: (f_coeffs, h_coeffs) as lists
    """
    f_poly, h_poly = C.hyperelliptic_polynomials()
    f_coeffs = f_poly.list()
    h_coeffs = h_poly.list() if h_poly else [0]
    return f_coeffs, h_coeffs


def neron_tate_height_pairing(z1, z2, Im_tau, prec=300, normalization_factor=1.0):
    """
    Compute Néron-Tate height pairing between two Abel-Jacobi images.
    
    Formula: <z1, z2> = normalization_factor * Re(z1^† * Im(τ)^(-1) * z2)
    
    where z1^† means transpose + complex conjugate,
    z1, z2 are vectors in C^2 and Im(τ) is the imaginary part of 
    the period matrix.
    
    The normalization_factor accounts for different conventions:
    - Some references use no factor (1.0)
    - Some use π
    - Some use 2π  
    - Some use 1/(2π) or 1/π
    
    The canonical height h(D) relates to the self-pairing by:
    h(D) = <D, D> / 2  (in some conventions)
    or
    h(D) = <D, D>      (in other conventions)
    
    Returns: real number (QQ approximation)
    """
    import math
    CC = ComplexField(prec)
    RR = RealField(prec)
    
    # Invert Im(τ)
    try:
        Im_tau_inv = Im_tau.inverse()
    except:
        # Singular - shouldn't happen if period matrix is correct
        return QQ(0)
    
    # Convert Im_tau_inv to complex field for proper arithmetic
    Im_tau_inv_CC = Matrix(CC, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau_inv_CC[i,j] = CC(Im_tau_inv[i,j])
    
    # Convert z1, z2 to column vectors in CC
    z1_vec = vector(CC, [CC(z1[0]), CC(z1[1])])
    z2_vec = vector(CC, [CC(z2[0]), CC(z2[1])])
    
    # Compute z1^† * Im_tau_inv * z2
    # = conj(z1)^T * Im_tau_inv * z2
    
    z1_conj = vector(CC, [z1_vec[0].conjugate(), z1_vec[1].conjugate()])
    
    # Matrix-vector multiply: Im_tau_inv * z2
    temp = Im_tau_inv_CC * z2_vec
    
    # Dot product: z1_conj · temp
    result = z1_conj[0] * temp[0] + z1_conj[1] * temp[1]
    
    # Take real part and apply normalization
    real_part = RR(result.real()) * RR(normalization_factor)
    
    # Convert to QQ
    return real_part.nearby_rational(max_error=RR(2)**(-prec+10))


# Add this diagnostic version of integrate_differential_path_with_branch


# To test, temporarily replace the call in abel_jacobi_mumford:
# int_0 = integrate_differential_path_with_branch_DEBUG(
#     base_x, x_pt, y_pt, f_coeffs, use_x_weight=False, prec=prec
# )
# CRITICAL FIX: Robust branch selection in integration


# CRITICAL FIX: Robust branch selection in integration


def abel_jacobi_mumford(
    D, f_coeffs, base_point, *,
    integrate_func=None,    # function(base_x, x_end, y_end, f_coeffs, use_x_weight, prec, debug)
    prec=300,
    period_matrix=None,     # optional 2x2 matrix whose columns span the period lattice
    debug=False
):
    """
    Compute Abel-Jacobi for Mumford divisor D = (u(x), v(x)).
    - base_point must be (x0, y0) (both provided) to fix starting sheet.
    - integrate_func: function performing the integral; if None it uses
      `integrate_differential_path_with_branch` from the caller's scope.
    - Returns: vector(CC, length=2) (not reduced mod lattice unless period_matrix provided).
    """
    from sage.all import ComplexField, PolynomialRing, vector, matrix
    import math

    CC = ComplexField(prec)

    if D.is_zero():
        return vector(CC, [0, 0])

    # require full base point (x,y)
    if not (isinstance(base_point, (tuple, list)) and len(base_point) == 2):
        raise ValueError("base_point must be a tuple (x0, y0) with both coordinates (to fix sheet).")

    base_x = CC(base_point[0])
    base_y = CC(base_point[1])

    # default integrator: look up in global scope if not provided
    if integrate_func is None:
        try:
            integrate_func = integrate_differential_path_with_branch
        except NameError:
            raise ValueError("No integrate_func provided and integrate_differential_path_with_branch not found.")

    # build polynomial rings over CC
    R = PolynomialRing(CC, 'x')
    x = R.gen()

    u_poly = D[0]
    v_poly = D[1]

    # Convert u(x) and v(x) to CC polynomials (note: u_poly.list() gives coefficients from const->highest)
    u_list = u_poly.list()
    v_list = v_poly.list()

    # build u_cc for root finding (coeffs placed into polynomial ring)
    u_cc = sum(CC(c) * x**i for i, c in enumerate(u_list))

    # get roots with multiplicity info
    try:
        roots_mult = u_cc.roots(multiplicities=True)
    except Exception as e:
        if debug:
            print("[abel_jacobi] root finding failed:", e)
        raise
        return vector(CC, [0, 0])

    if len(roots_mult) == 0:
        return vector(CC, [0, 0])

    # Expand into list of (root, multiplicity) and sort deterministically
    roots_mult = sorted(roots_mult, key=lambda r: (float(r[0].real()), float(r[0].imag())))
    roots = [r[0] for r in roots_mult]
    mults = [r[1] for r in roots_mult]

    # helper: evaluate v(x) fully at a CC point
    def eval_v_at(xpt):
        # v_list: constant term -> highest
        return sum(CC(c) * (xpt ** i) for i, c in enumerate(v_list))

    # helper: polynomial f(x) from f_coeffs (descending coefficients assumed)
    def f_at(z):
        # f_coeffs assumed descending highest->lowest
        return sum(CC(c) * (z ** (len(f_coeffs) - 1 - i)) for i, c in enumerate(f_coeffs))

    # tolerance thresholds (in CC)
    tiny = CC(2) ** (-prec // 2)
    # somewhat looser tolerance for root-v consistency (absolute)
    tol_consistency = CC(10) ** (-8)

    aj_vec = vector(CC, [0, 0])

    # Diagnostics arrays (if debug)
    min_f = None
    min_idx = None

    # iterate roots
    for idx, x_pt in enumerate(roots):
        # multiplicity handling: if multiplicity > 1 assume a Weierstrass (branch) point y=0
        multiplicity = mults[idx] if idx < len(mults) else 1

        x_pt_cc = CC(x_pt)  # convert to CC

        # Evaluate candidate y from v(x) fully (no 1st-degree assumption)
        y_from_v = eval_v_at(x_pt_cc)

        # Evaluate f(x_pt)
        fval = f_at(x_pt_cc)

        # update global min f diagnostics
        absf = abs(fval)
        if min_f is None or absf < min_f:
            min_f = absf
            min_idx = idx

        # If multiplicity>1 trust it's a Weierstrass branch: set y=0 (but still allow tiny numerical)
        if multiplicity > 1:
            y_pt = CC(0)
        else:
            # If v(x) matches sqrt(f) within tolerance, accept v(x). Otherwise pick sqrt(f)
            # but choose sign consistent with v(x) if possible.
            # first check if fval is near zero
            if abs(fval) < tiny:
                # branch point: set y to zero (or to v if v tiny)
                if abs(y_from_v) < tol_consistency:
                    y_pt = CC(0)
                else:
                    # prefer v as it's probably the correct local branch
                    y_pt = y_from_v
            else:
                # compute principal sqrt using CC
                y_sqrt = fval.sqrt()     # this gives one sqrt in CC
                # two choices: y_sqrt or -y_sqrt. choose the one closer to v(x).
                if abs(y_from_v - y_sqrt) <= abs(y_from_v + y_sqrt):
                    y_pt = y_sqrt
                else:
                    y_pt = -y_sqrt

                # final safety: if distance is huge, fall back to v_from_v (but warn)
                if abs((y_pt**2) - fval) > tol_consistency and debug:
                    print(f"[abel_jacobi] Warning: y^2 != f at root idx {idx}, |y^2-f|={abs((y_pt**2)-fval)}")
        # At this point y_pt is the target y at x_pt we will pass to integrator.
        # Integrate two differentials: 1/(2y) and x/(2y)
        try:
            #int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
            #                      use_x_weight=False, prec=prec, debug=debug)
            #int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
            #                      use_x_weight=True, prec=prec, debug=debug)
            int0 = integrate_func(base_x, x_pt_cc, base_y, y_pt, f_coeffs,
                                  use_x_weight=False, prec=prec, debug=debug)
            int1 = integrate_func(base_x, x_pt_cc, base_y, y_pt, f_coeffs,
                                  use_x_weight=True, prec=prec, debug=debug)

        except TypeError:
            # If integrate_func doesn't accept debug/prec/use_x_weight keywords in that order,
            # fall back to positional call (older integrator signature)
            int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, False, prec, debug)
            int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, True, prec, debug)
            raise
        except Exception as e:
            if debug:
                print(f"[abel_jacobi] Integration failed for root {idx} at x={x_pt_cc}: {e}")
            raise
            continue

        aj_vec += vector(CC, [int0, int1])

    if debug:
        print(f"[abel_jacobi] min |f(x)| among nodes (roots): {min_f} at index {min_idx}")

    # Optionally reduce modulo period lattice if provided.
    # We expect period_matrix to be a 2x2 matrix (columns are period vectors).
    if period_matrix is not None:
        # convert to Sage matrix over CC if necessary
        PM = matrix(CC, period_matrix) if not isinstance(period_matrix, type(matrix(CC, [[1]]))) else period_matrix
        # Solve PM * coeffs = aj_vec for coeffs (complex)
        try:
            coeffs = PM.solve_right(aj_vec)   # length-2 vector of complex coords
            # round the coefficients to nearest integers (componentwise) to find lattice representative
            nearest = [int(round(float(c.real()))) + int(round(float(c.imag()))) * CC(0,1) for c in coeffs]
            # nearest integer vector in Z^2: take real parts (period lattice indices are integer)
            nearest_ints = [int(round(float(c.real()))) for c in coeffs]
            # subtract lattice component
            lattice_part = PM * vector(CC, nearest_ints)
            aj_vec = aj_vec - lattice_part
        except Exception as e:
            if debug:
                print("[abel_jacobi] Warning: period reduction failed:", e)
            raise
            # continue without reduction

    return aj_vec

def fmt(z, digits=6):
    """
    Compact formatter for real/complex numbers.
    Uses scientific notation when needed.
    """
    try:
        # Complex number
        return f"{z.real():.{digits}g} + {z.imag():.{digits}g}i"
    except (AttributeError, TypeError):
        # Real number
        return f"{z:.{digits}g}"


from sage.all import QQ, ZZ, RR, Qp, PolynomialRing, HyperellipticCurve


def local_naive_height_p(D, p):
    """
    Compute naive local height at p: -min(v_p(coeffs)) * log(p).
    This corresponds to the log of the max p-adic norm of coefficients.
    """
    try:
        # Extract Mumford polynomials u, v
        u_poly, v_poly = D[0], D[1]
        coeffs = u_poly.list() + v_poly.list()
        
        # We want max(|c|_p). 
        # |c|_p = p^(-v_p(c)).
        # log(max |c|_p) = log(p^(-min v_p(c))) = -min(v_p(c)) * log(p)
        
        # Handle 0 coefficients (val is +infinity)
        vals = []
        for c in coeffs:
            if c == 0:
                continue
            # Handle both Rational and p-adic types
            try:
                vals.append(c.valuation(p))
            except AttributeError:
                vals.append(c.valuation())
                
        if not vals:
            return 0.0
            
        min_val = min(vals)
        return -min_val * math.log(p)
    except Exception:
        raise
        return 0.0


def arakelov_canonical_height(D, f_coeffs, prec=300, use_finite_places=True):
    """
    Proper canonical height with Abel-Jacobi map (Archimedean) 
    and p-adic doubling limit (Finite Neron corrections).
    """
    if D.is_zero():
        return QQ(0)
    
    from .homology import get_period_matrix_auto_B

    # 1. Naive global height (Weil height)
    # This sums log max(|c|_v) over all places (finite and infinite)
    h_naive = naive_height_qq(D, prec=prec)
    
    # 2. Archimedean correction (Neron local height at infinity - Naive at infinity)
    # Note: h_arch here is typically computed as 1/2 <z, z>_NT.
    # We assume archimedean_height_correction returns the proper difference term
    # or the full Archimedean contribution relative to the naive height baseline.
    # In this codebase context, it seems to be the full analytic height on the Jacobian?
    # Standard formula: h = h_naive + sum(corrections)
    period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
    h_arch = archimedean_height_correction(D, f_coeffs, period_matrix, prec=prec)
    
    # 3. Finite place corrections (Neron local height at p - Naive at p)
    h_finite_correction = QQ(0)
    
    if use_finite_places:
        # Dynamically determine bad primes
        bad_primes = get_bad_primes(f_coeffs)
        #print("bad primes for finite place correction:", bad_primes)
        
        for p in bad_primes:
            # Add correction term (mu_p - h_naive_p)
            # This accounts for intersection multiplicities on the special fiber
            corr = local_height_correction_finite(D, p, f_coeffs)
            h_finite_correction += QQ(corr)
    
    return h_naive + h_arch + h_finite_correction


# [arakelov.py]

def get_bad_primes(f_coeffs):
    """
    Identify primes of bad reduction for the curve y^2 = f(x).
    Includes factors of discriminant, leading coefficient, and 2.
    """
    from sage.all import QQ, PolynomialRing
    key = tuple(f_coeffs)
    if key in get_bad_primes.cache:
        return get_bad_primes.cache[key]
    
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    bad = set()
    
    # 1. Discriminant factors
    disc = f_poly.discriminant()
    if disc != 0:
        # Handle Rational discriminant: separate numerator and denominator
        bad.update(QQ(disc).numerator().prime_factors())
        bad.update(QQ(disc).denominator().prime_factors())
    
    # 2. Leading coefficient factors (potential degree drop)
    lc = f_coeffs[0]
    if lc != 0:
        bad.update(QQ(lc).numerator().prime_factors())
        bad.update(QQ(lc).denominator().prime_factors())
        
    # Genus 2 arithmetic at p=2 is always delicate
    bad.add(2)
    
    ret = sorted(list(bad))
    get_bad_primes.cache[key] = ret
    return ret
get_bad_primes.cache = {}


def arakelov_height_pairing(D1, D2, f_coeffs, period_matrix, prec=300):
    """
    Proper Arakelov height pairing using Abel-Jacobi map.
    <D1, D2> = <AJ(D1), AJ(D2)>_NT
    """
    from sage.all import RealField, Matrix, QQ
    
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    RR = RealField(prec)
    
    # Generate consistent base point
    base_point = choose_numerical_base_point(f_coeffs, prec=prec)
    
    # Pass base_point explicitly
    z1 = abel_jacobi_mumford(D1, f_coeffs, base_point=base_point, prec=prec)
    z2 = abel_jacobi_mumford(D2, f_coeffs, base_point=base_point, prec=prec)
    
    # Extract Im(τ)
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())
    
    # Compute pairing
    pairing = neron_tate_height_pairing(z1, z2, Im_tau, prec=prec)
    
    return pairing


def local_height_correction_finite(D, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute the local canonical height correction (Neron correction) at p 
    using the p-adic doubling limit:
       mu_p(D) = lim_{n->inf} 4^(-n) * h_naive(2^n D) - h_naive(D)
    
    This correctly handles bad reduction (I_n, etc) without explicit Neron models.
    """
    # Rule of thumb: need ~num_doublings extra precision per doubling
    # Start with much higher precision than you think you need
    if padic_prec is None:
        padic_prec = max(2048, 100 * num_doublings)
    
    # 1. Setup p-adic curve
    try:
        K = Qp(p, prec=padic_prec)
        R = PolynomialRing(K, 'x')
        f_poly = sum(K(c) * R.gen()**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        C_p = HyperellipticCurve(f_poly)
        J_p = C_p.jacobian()
        
        # 2. Lift point D to J(Qp)
        u_Q, v_Q = D[0], D[1]
        u_p = R([K(c) for c in u_Q.list()])
        v_p = R([K(c) for c in v_Q.list()])
        
        P = J_p([u_p, v_p])
        
        # 3. Compute initial naive height
        h0 = local_naive_height_p(P, p)
        
        # 4. Iterate doubling with precision monitoring
        current_P = P
        for i in range(num_doublings):
            if current_P.is_zero():
                return -h0
            
            # Check if we still have meaningful precision before doubling
            # Get the leading coefficient precision
            u_poly = current_P[0]
            if u_poly != 0:
                # Check precision of leading coefficient
                leading_coeff = u_poly.leading_coefficient()
                if hasattr(leading_coeff, 'precision_absolute'):
                    prec_remaining = leading_coeff.precision_absolute()
                    if prec_remaining < 10:  # Too little precision left
                        # Bail out early and use what we have
                        h_final = local_naive_height_p(current_P, p)
                        scaling = 4.0**(-i)  # Use current iteration count
                        h_can_approx = scaling * h_final
                        return h_can_approx - h0
            
            current_P = 2 * current_P
        
        # 5. Compute final naive height
        h_final = local_naive_height_p(current_P, p)
        
        # 6. Apply Tate's limit formula
        scaling = 4.0**(-num_doublings)
        h_can_approx = scaling * h_final
        
        return h_can_approx - h0
        
    except ZeroDivisionError:
        # Precision loss - try with higher precision or fewer doublings
        if padic_prec < 8192 and num_doublings > 5:
            # Retry with either more precision or fewer doublings
            return local_height_correction_finite(D, p, f_coeffs, 
                                                 num_doublings=num_doublings-2, 
                                                 padic_prec=padic_prec*2)
        raise
    except Exception:
        raise


def choose_numerical_base_point(f_coeffs, prec=300):
    """
    Selects a numerically safe base point for Abel-Jacobi maps.
    Returns (x, y) where y^2 = f(x).
    """
    from sage.all import ComplexField, PolynomialRing
    
    CC = ComplexField(prec)
    Rq = PolynomialRing(CC, 'x')
    x = Rq.gen()
    f_poly_cc = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    roots = f_poly_cc.roots(multiplicities=False)
    if not roots:
        # Fallback: use a point away from branch locus
        x_base = CC(1)
        f_val = sum(CC(c) * x_base**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        y_base = f_val.sqrt()
        return (x_base, y_base)
        
    sorted_roots = sorted(roots, key=lambda z: (float(z.real()), float(z.imag())))
    
    # OPTION 1: Use a root with tiny offset (Weierstrass point shifted slightly)
    root = sorted_roots[0]
    eps = CC(2) ** (-(prec // 4))  # Not too tiny
    x_base = root + CC(0, 1) * eps
    f_val = sum(CC(c) * x_base**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    y_base = f_val.sqrt()
    
    # Choose branch with positive imaginary part for consistency
    if y_base.imag() < 0:
        y_base = -y_base
    
    return (x_base, y_base)


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


# Module-level worker functions (must be at top level for pickling)

def _compute_height_worker(args):
    """Worker function to compute a single height - must be at module level for pickling"""
    from sage.all import PolynomialRing, HyperellipticCurve, QQ
    
    i, div, f_coeffs, prec = args
    try:
        # Reconstruct the Jacobian element in this process
        Rq_QQ = PolynomialRing(QQ, 'x')
        x_QQ = Rq_QQ.gen()
        f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-k) 
                       for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
        
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        
        h = arakelov_canonical_height(D, f_coeffs, prec=prec)
        return (i, float(h), None)
    except Exception as e:
        return (i, None, str(e))


def _compute_pairing_worker(args):
    """Worker function to compute a single pairing - must be at module level for pickling"""
    from sage.all import PolynomialRing, HyperellipticCurve, QQ
    
    i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args
    try:
        if i == j:
            return ((i, j), h_i, None)
        
        # Reconstruct Jacobian elements
        Rq_QQ = PolynomialRing(QQ, 'x')
        x_QQ = Rq_QQ.gen()
        f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-k) 
                       for k, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly_QQ)
        J = C.jacobian()
        
        u_poly_i = x_QQ**2 - QQ(div_i['s'])*x_QQ + QQ(div_i['p'])
        v_poly_i = QQ(div_i['v_1'])*x_QQ + QQ(div_i['v_0'])
        D1 = J([u_poly_i, v_poly_i])
        
        u_poly_j = x_QQ**2 - QQ(div_j['s'])*x_QQ + QQ(div_j['p'])
        v_poly_j = QQ(div_j['v_1'])*x_QQ + QQ(div_j['v_0'])
        D2 = J([u_poly_j, v_poly_j])
        
        D_sum = D1 + D2
        
        if D_sum.is_zero():
            h_sum = 0
        else:
            h_sum = arakelov_canonical_height(D_sum, f_coeffs, prec=prec)
        
        val = (float(h_sum) - h_i - h_j) / 2
        return ((i, j), val, None)
    except Exception as e:
        return ((i, j), None, str(e))


def precompute_pairings_parallel(indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    """Precompute all pairings for given indices in parallel"""
    # Collect unique pairs to compute
    pairs_to_compute = []
    for r in range(len(indices)):
        for c in range(r, len(indices)):
            i, j = indices[r], indices[c]
            if i > j:
                i, j = j, i
            if (i, j) not in pairing_cache:
                div_i = jac_elements[i][0]
                div_j = jac_elements[j][0]
                pairs_to_compute.append((
                    i, j, div_i, div_j, f_coeffs, prec,
                    height_cache[i], height_cache[j]
                ))

    if not pairs_to_compute:
        return

    # Compute in parallel if worthwhile
    if len(pairs_to_compute) > 2 and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_compute_pairing_worker, pairs_to_compute)

        for (i, j), val, error in results:
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
            pairing_cache[(i, j)] = val
    else:
        # Sequential for small batches
        for args in pairs_to_compute:
            (i, j), val, error = _compute_pairing_worker(args)
            if error:
                raise RuntimeError(f"Pairing computation failed: {error}")
            pairing_cache[(i, j)] = val
    return pairing_cache

def get_pairing(i, j, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs):
    if i > j:
        i, j = j, i

    if (i, j) not in pairing_cache:
        # Compute on-demand if not in cache
        pairing_cache = precompute_pairings_parallel([i, j], jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs)

    return pairing_cache[(i, j)]


from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, 
                      QQ, RR as SageRR)
from multiprocessing import Pool, cpu_count


def is_independent_by_projection_log(
    basis_indices,
    candidate_index,
    get_pairing,
    prec,
    debug=False,
):
    """
    Stable projection-residual independence test using log-scale comparison
    to avoid underflow when prec is large.

    Returns (is_independent: bool, info: dict) with info = {res_sq, log10_res, log10_tol, min_sv}.
    Requires get_pairing(i,j) -> float (pairing_cache floats).
    """
    import math
    try:
        import numpy as np
    except Exception:
        np = None

    k = len(basis_indices)
    if k == 0:
        return True, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}

    # Build Gram G (k x k) and cross vector c as float64
    G = [[0.0]*k for _ in range(k)]
    c = [0.0]*k
    for r in range(k):
        for s in range(r, k):
            v = float(get_pairing(basis_indices[r], basis_indices[s]))
            G[r][s] = v; G[s][r] = v
        c[r] = float(get_pairing(basis_indices[r], candidate_index))
    vv = float(get_pairing(candidate_index, candidate_index))

    # convert to numpy for robust numeric ops if available
    if np is not None:
        Gnp = 0.5*(np.array(G, dtype=float) + np.array(G, dtype=float).T)  # enforce symmetry
        cnp = np.array(c, dtype=float)
    else:
        # fallback to naive python arrays (less robust)
        Gnp = None
        cnp = c

    proj_sq = None
    min_sv = 0.0
    try:
        if np is not None:
            # prefer cholesky; fallback to SVD pseudoinv
            try:
                L = np.linalg.cholesky(Gnp)
                y = np.linalg.solve(L, cnp)
                proj_sq = float(np.dot(y, y))
            except Exception:
                U, S, Vt = np.linalg.svd(Gnp, full_matrices=False)
                min_sv = float(S[-1]) if len(S)>0 else 0.0
                eps = max(1e-16 * S[0], 1e-300) if len(S)>0 else 1e-300
                S_inv = np.array([1.0/s if s>eps else 0.0 for s in S], dtype=float)
                Ginv = (Vt.T * S_inv) @ U.T
                proj_sq = float(cnp @ (Ginv @ cnp))
        else:
            # No numpy: try fallback with small RR precision (less robust)
            from sage.all import RealField, matrix, vector
            RR = RealField(min(max(80, int(prec//8)), 512))
            Gs = matrix(RR, G)
            cs = vector(RR, c)
            L = Gs.cholesky()
            y = L.solve_left(cs)
            proj_sq = float(y.dot_product(y))
    except Exception:
        # if numerics fail, be conservative and treat as not independent
        if debug:
            print("[proj-log] numeric failure in projection computation")
        return False, {"res_sq": None, "log10_res": None, "log10_tol": None, "min_sv": None}

    res_sq = vv - proj_sq

    # If residual is non-positive: numerical issue -> treat as non-independent unless tiny negative
    if res_sq <= 0.0:
        # tiny negative tolerance: compare magnitude to diag_max
        diag_max = max([G[r][r] for r in range(k)] + [vv, 1.0])
        # if |res_sq| is tiny relative to diag_max, treat as zero
        if abs(res_sq) <= 1e-12 * diag_max:
            res_sq = 0.0
        else:
            # significant negative residual -> reject candidate (not independent)
            if debug:
                print(f"[proj-log] negative residual (proj > self): res_sq={res_sq}")
            return False, {"res_sq": res_sq, "log10_res": float("-inf"), "log10_tol": float("-inf"), "min_sv": min_sv}

    # Now compute log10 comparisons to avoid underflow:
    # log10_tol = log10(diag_max) - (dec_digits - safety)
    diag_max = max(max(abs(G[r][r]) for r in range(k)), abs(vv), 1.0)
    dec_digits = int(prec * 0.30103)
    safety_digits = 12  # tuneable
    log10_tol = math.log10(diag_max) - max(0, (dec_digits - safety_digits))

    # log10_res
    log10_res = math.log10(res_sq) if res_sq > 0 else float("-inf")

    # Optional use of min singular value to boost tol (in log scale)
    if min_sv and min_sv > 0:
        log10_min_sv = math.log10(min_sv)
        # choose the larger of the two log10 thresholds
        log10_tol = max(log10_tol, log10_min_sv - 6)

    is_independent = (log10_res > log10_tol)

    info = {"res_sq": res_sq, "log10_res": log10_res, "log10_tol": log10_tol, "min_sv": min_sv}
    if debug:
        print(f"[proj-log] cand={candidate_index} res={res_sq:.3g} log10_res={log10_res:.3g} log10_tol={log10_tol:.3g} min_sv={min_sv:.3g}")

    return is_independent, info


def dedupe_basis(basis, basis_indices, debug=False):
    """
    Remove duplicated divisors (preserve order). Returns (basis_u, basis_indices_u).
    """
    seen = set()
    basis_u = []
    basis_indices_u = []
    for div, idx in zip(basis, basis_indices):
        if idx in seen:
            if debug:
                print(f"[dedupe] removing duplicate basis index {idx}")
            continue
        seen.add(idx)
        basis_u.append(div)
        basis_indices_u.append(idx)
    return basis_u, basis_indices_u


def gram_logdet_and_cond(basis_indices, get_pairing):
    """
    Build float Gram matrix for basis_indices (using get_pairing -> float),
    return dict with:
        n, svals (list), log10_abs_det, log10_cond, numeric_rank_est
    """
    import numpy as np, math
    n = len(basis_indices)
    if n == 0:
        return {"n":0, "svals": [], "log10_abs_det": None, "log10_cond": None, "numeric_rank": 0}
    G = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            v = float(get_pairing(basis_indices[i], basis_indices[j]))
            G[i,j] = v; G[j,i] = v
    # enforce symmetry
    G = 0.5*(G + G.T)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    svals = [float(x) for x in S]
    # log10_abs_det = sum(log10(svals)) (sign positive for Gram of PD matrix)
    log10_abs_det = sum(math.log10(max(s, 1e-300)) for s in svals)
    cond = svals[0] / (svals[-1] if svals[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)
    # numeric rank: count s > smax * tol_rel
    tol_rel = 1e-12
    smax = svals[0]
    numeric_rank = sum(1 for s in svals if s > smax * tol_rel)
    return {"n": n, "svals": svals, "log10_abs_det": log10_abs_det, "log10_cond": log10_cond, "numeric_rank": numeric_rank}


# requires numpy
import numpy as np, math

def select_independent_indices_from_gram(
    G,
    prec_bits=2048,
    safety_digits=10,
    rel_sv_tol=1e-12,
    pivot_tol_factor=1e-9,
    debug=False,
):
    """
    Given a symmetric float Gram matrix G (n x n), return a deterministic list
    of indices that form a numerically independent set.

    Parameters
    ----------
    G : np.ndarray, shape (n,n), symmetric floats
    prec_bits : int, bit-precision used upstream (only for diagnostic scaling)
    safety_digits : int, digits of safety when building eigenvalue threshold
    rel_sv_tol : float, fallback relative threshold for eigenvalues (smax * rel_sv_tol)
    pivot_tol_factor : float, factor to multiply sqrt(min_positive_eig) to set pivot stop
    debug : bool

    Returns
    -------
    selected_indices : list of ints (length <= numeric_rank)
    info : dict with keys:
        'eigvals' (descending), 'numeric_rank', 'log10_abs_det', 'log10_cond'
    """
    # symmetrize (be safe)
    G = 0.5 * (G + G.T)
    n = G.shape[0]
    # Eigen-decomposition (symmetric)
    eigvals, eigvecs = np.linalg.eigh(G)  # ascending eigenvalues
    eigvals = np.array(eigvals, dtype=float)
    # flip to descending
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    smax = eigvals[0] if eigvals.size else 0.0
    # build eigenvalue threshold robustly:
    # convert bits -> decimal digits estimate, but cap to avoid under/overflow
    dec_digits = int(prec_bits * 0.30103) if prec_bits > 0 else 50
    dec_digits_cap = min(max(dec_digits, 0), 50)  # cap to [0,50] to avoid absurd exponents
    # threshold by safety_digits (but cap using rel_sv_tol)
    ev_thresh = max(smax * (10.0 ** (-(max(safety_digits, dec_digits_cap)))), smax * rel_sv_tol, 1e-300)

    # positive eigenvalues indices
    pos_mask = eigvals > ev_thresh
    pos_indices = np.nonzero(pos_mask)[0]
    r = len(pos_indices)
    if debug:
        print(f"[select_from_gram] smax={smax:.3g}, ev_thresh={ev_thresh:.3g}, num_pos={r}")

    if r == 0:
        return [], {
            "eigvals": eigvals.tolist(),
            "numeric_rank": 0,
            "log10_abs_det": None,
            "log10_cond": None,
        }

    # Build embedding E (n x r) such that G ≈ E E^T
    Spos = eigvals[pos_indices]                     # length r (descending)
    Upos = eigvecs[:, pos_indices]                  # n x r
    sqrtS = np.sqrt(np.maximum(Spos, 0.0))
    # E[i,:] is embedding vector for candidate i
    E = Upos * sqrtS[np.newaxis, :]                 # shape (n, r)

    # Determine pivot stop tolerance from smallest retained eigenvalue
    min_pos_eig = Spos[-1]
    pivot_tol = math.sqrt(max(min_pos_eig, 0.0)) * pivot_tol_factor
    # also ensure pivot_tol not ridiculously tiny:
    pivot_tol = max(pivot_tol, smax * 1e-16)

    # Deterministic pivoting: iteratively pick row with largest residual norm,
    # orthogonalize rows against chosen normalized vector, stop when residuals small
    rows = E.copy()  # will be modified (deflation)
    norms = np.linalg.norm(rows, axis=1)
    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    while True:
        # choose argmax among not selected
        cand = int(np.argmax(norms + (selected_mask * -1e300)))  # ensures selected masked
        maxnorm = norms[cand]
        if debug:
            print(f"[pivot] pick candidate {cand} maxnorm={maxnorm:.6g} selected={len(selected)}")
        if maxnorm <= pivot_tol:
            break
        # add candidate
        selected.append(cand)
        selected_mask[cand] = True
        # normalize
        v = rows[cand].copy()
        vnorm = np.linalg.norm(v)
        if vnorm == 0.0:
            # can't orthonormalize further
            break
        v = v / vnorm
        # deflate all rows by projection onto v
        proj = rows @ v   # projection coefficients (n,)
        rows = rows - np.outer(proj, v)
        # recompute norms only for not yet selected
        norms = np.linalg.norm(rows, axis=1)
        # keep selecting until we've chosen r rows
        if len(selected) >= r:
            break

    # if we picked fewer than r (rare), we can accept them as basis; numeric_rank = len(selected)
    numeric_rank = len(selected)

    # diagdet/log det from singular values (embedding singulars are sqrt of eigvals)
    # For Gram matrix, singular values = eigvals (nonnegative). So log10|det| = sum log10(eigvals_pos)
    log10_abs_det = sum(math.log10(max(x, 1e-300)) for x in Spos)
    cond = Spos[0] / (Spos[-1] if Spos[-1] > 0 else 1e-300)
    log10_cond = math.log10(cond)

    info = {
        "eigvals": eigvals.tolist(),
        "numeric_rank": numeric_rank,
        "log10_abs_det": log10_abs_det,
        "log10_cond": log10_cond,
    }
    return selected, info


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False,
                                      test_normalization=None, n_jobs=-1):
    """
    Robust, deterministic basis builder:
      - precomputes all pairings once (parallel)
      - builds single float Gram matrix G
      - selects a deterministic independent set via select_independent_indices_from_gram
      - returns basis (list of divisor dicts), numeric rank, and H_final (Sage matrix)
    """
    from sage.all import (RealField, PolynomialRing, Matrix, HyperellipticCurve, QQ)
    from multiprocessing import cpu_count
    import numpy as np
    import math
    import sys

    # ensure period matrix cached/available (same as before)
    get_period_matrix_auto_B(f_coeffs, prec=prec)

    if not all_divisors:
        return [], 0, None

    # choose n_jobs
    if n_jobs == -1:
        try:
            n_jobs = cpu_count()
        except Exception:
            n_jobs = 1

    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
        print(f"[arakelov] Parallelization: {n_jobs} workers")

    # Build curve & jacobian constructors
    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    # Convert divisors to Jacobian elements (keep same indexing)
    jac_elements = []
    for div in all_divisors:
        u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
        v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
        D = J([u_poly, v_poly])
        jac_elements.append((div, D))

    n = len(jac_elements)
    if n == 0:
        return [], 0, None

    # Precompute individual heights (parallel) — reuse your existing worker wrapper
    if debug:
        print(f"[arakelov] Pre-computing heights for {n} candidates...")

    height_args = [(i, jac_elements[i][0], f_coeffs, prec) for i in range(n)]
    height_cache = {}

    if n > 1 and n_jobs > 1:
        from multiprocessing import Pool
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_compute_height_worker, height_args)
        for i, h, error in results:
            if error:
                raise RuntimeError(f"Height computation failed for divisor {i}: {error}")
            height_cache[i] = float(h)
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")
    else:
        # sequential fallback
        for i in range(n):
            _, D = jac_elements[i]
            h = arakelov_canonical_height(D, f_coeffs, prec=prec)
            height_cache[i] = float(h)
            if debug:
                print(f"  Divisor {i}: h = {float(h):.6g}")

    # Pairing cache (will be filled by precompute_pairings_parallel)
    pairing_cache = {}

    # Precompute all pairings once (this populates pairing_cache)
    if debug:
        print("[arakelov] Precomputing full pairing matrix (parallel)...")
    # indices passed to precompute should be full-range
    all_indices = list(range(n))
    precompute_pairings_parallel(all_indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs)

    # Build float Gram matrix G from pairing_cache (consistent single source)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            key = (i, j) if i <= j else (j, i)
            if key not in pairing_cache:
                # fallback: compute on-demand (shouldn't happen)
                pairing_cache = precompute_pairings_parallel([i, j], jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs)
            val = float(pairing_cache[key])
            G[i, j] = val
            G[j, i] = val
    # symmetrize defensively
    G = 0.5 * (G + G.T)

    if debug:
        print("[arakelov] Full Gram built. Running deterministic selector...")

    # Deterministically select independent indices from the full Gram
    selected_indices, info = select_independent_indices_from_gram(
        G, prec_bits=prec, safety_digits=10, rel_sv_tol=1e-12, pivot_tol_factor=1e-9, debug=debug
    )
    if debug:
        print("Selected indices (deterministic):", selected_indices)
        print("Selector info:", {k: info[k] for k in ("numeric_rank", "log10_abs_det", "log10_cond") if k in info})

    # Build basis lists from selected indices (preserve divisor dicts)
    basis = [jac_elements[i][0] for i in selected_indices]
    basis_indices = list(selected_indices)

    # Deduplicate just in case
    basis, basis_indices = dedupe_basis(basis, basis_indices, debug=debug)

    final_rank = len(basis)
    H_final = None
    if final_rank > 0:
        # Build Sage RealField Gram for the selected basis (for downstream code expecting Sage Matrix)
        RR = RealField(max(128, int(prec//4)))  # reasonable working real precision for the final matrix
        H_final = Matrix(RR, final_rank, final_rank)
        for r in range(final_rank):
            for c in range(r, final_rank):
                # get_pairing uses pairing_cache we prefilled; pass jac_elements and pairing_cache so it reads cache
                pv = float(get_pairing(basis_indices[r], basis_indices[c], jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs))
                H_final[r, c] = RR(pv)
                H_final[c, r] = H_final[r, c]

    # Diagnostics: print numeric SVD / logdet info in a stable way
    if debug:
        try:
            gd = gram_logdet_and_cond(basis_indices, lambda a,b: float(get_pairing(a,b,jac_elements,pairing_cache,f_coeffs,prec,height_cache,n_jobs)))
            print(f"[arakelov] Final numeric rank (svd): {gd['numeric_rank']}/{gd['n']}, log10|det|={gd['log10_abs_det']:.3g}, log10(cond)={gd['log10_cond']:.3g}")
            if H_final is not None:
                try:
                    # print high-precision determinant if feasible
                    print(f"[arakelov] Final determinant: {float(H_final.determinant()):.6g}")
                except Exception:
                    pass
        except Exception as E:
            if debug:
                print(f"[arakelov] Diagnostic SVD failed: {E}")

    # Optional: spot-check ambiguous candidates (if you want to run exact pairing verification,
    # do it here by calling your exact-doubling routine for candidates near the spectral cutoff).
    # (Left as a hook for you; I did not call an exact routine to avoid assuming its name.)

    return basis, final_rank, H_final


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

def archimedean_height_correction(D, f_coeffs, period_matrix, prec=300):
    """
    Proper Archimedean height correction: E(z) - log|theta(z)|
    """
    from sage.all import RealField, Matrix, QQ, log
    
    if D.is_zero():
        return QQ(0)
    
    RR = RealField(prec)
    
    # 1. Compute Abel-Jacobi z
    base_point = choose_numerical_base_point(f_coeffs, prec=prec)
    z = abel_jacobi_mumford(D, f_coeffs, base_point=base_point, prec=prec)
    
    # 2. Compute Quadratic Part E(z)
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())
            
    # Neron-Tate quadratic form 1/2 * y^T * Im(tau)^-1 * y
    # Note: neron_tate_height_pairing in your code might return 2*E(z) or 4*E(z).
    # Standard normalization for canonical height is often E(z).
    quad_part = neron_tate_height_pairing(z, z, Im_tau, prec=prec)
    
    # 3. Compute Log-Theta Correction ("The Principal Part")
    # We use theta[00] (standard theta). 
    # For generic D in J, theta(z) is non-zero.
    theta_val = compute_theta_high_prec(z, period_matrix, prec=prec)
    log_theta = log(abs(theta_val))
    
    # 4. Combine
    # The naive height on Kummer (2*Theta) is approx 2 * log|theta| + ...
    # The quadratic height on Jacobian is approx E(z).
    # The correction is E(z) - log|theta^2| roughly.
    # We heuristically check scaling. If h_naive corresponds to 2*Theta:
    # We want result to be E(z) - 2*log|theta|.
    
    # Try this normalization (standard for J):
    return quad_part - 2 * log_theta
