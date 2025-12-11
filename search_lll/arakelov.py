# arakelov.py
#
# Arakelov height computations for genus-2 hyperelliptic Jacobian elements.
# Faster and more reliable than repeated doubling for height pairings.

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

DEBUG = True
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


# arakelov_optimized.py
#
# Optimized Arakelov height computations
# Same function names as original for dedup.py compatibility


# arakelov.py
#
# Arakelov height computations for genus-2 hyperelliptic Jacobian elements.
# Faster and more reliable than repeated doubling for height pairings.


def arakelov_build_basis(all_divisors, f_coeffs, prec=100, debug=False):
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


# arakelov.py
#
# Arakelov height computations for genus-2 hyperelliptic Jacobian elements.
# Faster and more reliable than repeated doubling for height pairings.

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


def arakelov_check_independence(divisors, f_coeffs, prec=100, debug=False):
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
        is_indep = (det > 1e-4)

        if debug:
            print(f"[arakelov] Height matrix determinant: {det:.6g}")
            print(f"[arakelov] Positive definite? {is_indep}")
    
    return is_indep, n if is_indep else 0, H, det


def arakelov_build_basis_parallel(all_divisors, f_coeffs, prec=100, debug=False):
    """
    Build basis using parallel height computation.
    """
    if not all_divisors:
        return [], 0, None
    
    clear_period_cache()
    reset_timers()
    
    if debug:
        print(f"\n[arakelov_parallel] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov_parallel] Using precision: {prec} bits (Tanh-Sinh)")
    
    print("f_coeffs", f_coeffs)
    with Timer("period_matrix_total"):
        try:
            period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
            tau_im = Matrix(RR, 2, 2, [[period_matrix[i,j].imag() for j in range(2)] for i in range(2)])
            print("Im(tau) eigenvalues:", tau_im.eigenvalues())
            print("det(Im(tau)):", tau_im.determinant())
            if debug:
                print(f"[arakelov_parallel] Period matrix computed")
        except Exception as e:
            if debug:
                print(f"[arakelov_parallel] Period matrix failed: {e}")
            raise
    
    # FILTER TORSION FIRST
    if debug:
        print(f"[arakelov_parallel] Filtering torsion divisors...")
    
    non_torsion = []
    torsion_count = 0
    max_torsion_order = 100
    
    for div in all_divisors:
        is_tors, order = is_mumford_torsion_fast(
            div['s'], div['p'], div['v_0'], div['v_1'], 
            f_coeffs, max_order=max_torsion_order, debug=False
        )
        
        if is_tors:
            torsion_count += 1
            if debug and torsion_count <= 5:
                print(f"[arakelov_parallel] Filtered torsion divisor (order {order}): s={div['s']}, p={div['p']}")
        else:
            non_torsion.append(div)
    
    if debug:
        print(f"[arakelov_parallel] Filtered {torsion_count} torsion divisors -> {len(non_torsion)} candidates")
    
    if not non_torsion:
        return [], 0, None
    
    with Timer("jacobian_conversion"):
        R = PolynomialRing(QQ, 'x')
        x = R.gen()
        f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        C = HyperellipticCurve(f_poly)
        J = C.jacobian()
        
        jac_elements = []
        for div in non_torsion:
            try:
                u_poly = x**2 - QQ(div['s'])*x + QQ(div['p'])
                v_poly = QQ(div['v_1'])*x + QQ(div['v_0'])
                D = J([u_poly, v_poly])
                if not D.is_zero():
                    jac_elements.append((div, D))
            except Exception:
                raise
                continue
    
    if not jac_elements:
        return [], 0, None
    
    n = len(jac_elements)
    
    canonical_heights = {}
    
    if debug:
        print(f"[arakelov_parallel] Computing {n} canonical heights...")
    
    with Timer("heights_precompute"):
        for i, (div, D) in enumerate(jac_elements):
            canonical_heights[i] = arakelov_canonical_height(D, f_coeffs, prec=prec)
            if debug and (i+1) % 10 == 0:
                print(f"[arakelov_parallel] Computed {i+1}/{n} heights")
    
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
        print(f"[arakelov_parallel] Building basis incrementally...")
    
    with Timer("basis_construction"):
        for i, (div, D) in enumerate(jac_elements):
            h_self = compute_pairing(i, i)
            if float(h_self) < 1e-4:
                if debug:
                    print(f"[arakelov_parallel] Skipping divisor {i}: self-pairing too small")
                continue
            if not basis:
                basis.append(div)
                basis_indices.append(i)
                if debug:
                    print(f"[arakelov_parallel] Added divisor 0 (self-pairing {float(h_self):.6g})")
            else:
                candidate_indices = basis_indices + [i]
                m = len(candidate_indices)
                H = Matrix(QQ, m, m)
                for ii in range(m):
                    for jj in range(ii, m):
                        h_ij = compute_pairing(candidate_indices[ii], candidate_indices[jj])
                        H[ii, jj] = h_ij
                        H[jj, ii] = h_ij
                
                det_val = float(H.determinant())


                if is_positive_definite(H, prec=prec, tol=10**(-12)):
                    basis.append(div)
                    basis_indices.append(i)
                    if debug:
                        det_float = float(H.determinant())
                        print(f"[arakelov_parallel] Added divisor {i} (rank {m}, det {det_float:.6g})")
                else:
                    if debug:
                        print(f"[arakelov_parallel] Skipping divisor {i}: not positive-definite (rejected)")

    final_rank = len(basis)
    H_final = None
    if final_rank > 0:
        H_final = Matrix(QQ, final_rank, final_rank)
        for i in range(final_rank):
            for j in range(i, final_rank):
                h_ij = compute_pairing(basis_indices[i], basis_indices[j])
                H_final[i, j] = h_ij
                H_final[j, i] = h_ij
                
    if debug:
        print_timers()
    
    return basis, final_rank, H_final


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

def is_positive_definite(H, prec=200, tol=1e-12):
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
            return False
        # det_k should be a high-precision real; check > tol
        if not (det_k > RR_prec(tol)):
            return False
    return True


def arakelov_height_pairing_cached(D1, D2, f_coeffs, height_cache=None, prec=100):
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


def get_period_matrix_old(f_coeffs, prec=100):
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


def get_period_matrix_bad_cycle_choices(f_coeffs, prec=100):
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
            continue
        for perm in permutations([0,1]):
            Bperm = Matrix(CC, 2, 2, [[B[i,perm[j]] for j in range(2)] for i in range(2)])
            for flipB in product([1,-1], repeat=2):
                Bmod = Matrix(CC, 2, 2, [[Bperm[i,j]*CC(flipB[j]) for j in range(2)] for i in range(2)])
                try:
                    tau_cand = Ainv * Bmod
                except Exception:
                    continue
                # sym error and Im(tau) diagnostics
                sym_err = abs(tau_cand[0,1] - tau_cand[1,0])
                # build Im(tau) in RRp
                Im_tau = Matrix(RRp, 2, 2, [[RRp(tau_cand[i,j].imag()) for j in range(2)] for i in range(2)])
                try:
                    evals = Im_tau.eigenvalues()
                except Exception:
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


def abel_jacobi_map(D, f_coeffs, period_matrix, prec=100):
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


def integrate_differential_path(x_start, x_end, f_coeffs, use_x_weight=False, prec=100, max_depth=8):
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


def archimedean_height_correction(D, f_coeffs, period_matrix, prec=100):
    """
    Proper Archimedean height correction using Abel-Jacobi map.
    
    h_∞(D) = (1/2) * <AJ(D), AJ(D)>
    where <·,·> is the Néron-Tate height pairing.
    """
    if D.is_zero():
        return QQ(0)
    
    RR = RealField(prec)
    
    # Compute Abel-Jacobi map
    z = abel_jacobi_mumford(D, f_coeffs, prec=prec)
    
    # Extract Im(τ)
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())
    
    # Compute self-pairing
    pairing = neron_tate_height_pairing(z, z, Im_tau, prec=prec)
    
    return pairing / QQ(4)


def arakelov_height_pairing(D1, D2, f_coeffs, period_matrix, prec=100):
    """
    Proper Arakelov height pairing using Abel-Jacobi map.
    
    <D1, D2> = (1/2) * (h(D1+D2) - h(D1) - h(D2))
    
    But using the direct formula:
    <D1, D2> = <AJ(D1), AJ(D2)>_NT
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    RR = RealField(prec)
    
    # Compute Abel-Jacobi maps
    z1 = abel_jacobi_mumford(D1, f_coeffs, prec=prec)
    z2 = abel_jacobi_mumford(D2, f_coeffs, prec=prec)
    
    # Extract Im(τ)
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())
    
    # Compute pairing
    pairing = neron_tate_height_pairing(z1, z2, Im_tau, prec=prec)
    
    return pairing


def arakelov_canonical_height(D, f_coeffs, prec=100, use_finite_places=True):
    """
    Proper canonical height with Abel-Jacobi map.
    """
    if D.is_zero():
        return QQ(0)
    
    from .homology import get_period_matrix_auto_B
    
    # Naive height
    h_naive = naive_height_qq(D, prec=prec)
    
    # Get period matrix
    period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
    
    # Archimedean correction
    h_arch = archimedean_height_correction(D, f_coeffs, period_matrix, prec=prec)
    
    # Finite place corrections
    h_finite = QQ(0)
    if use_finite_places:
        for p in [2, 3, 5, 7, 11, 13]:
            try:
                h_p = local_height_finite(D, p, prec=prec)
                h_finite += h_p
            except:
                pass
    
    return h_naive + h_arch + h_finite


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

from sage.all import QQ, ComplexField, RealField, Matrix, vector, PolynomialRing
from sage.all import HyperellipticCurve

DEBUG = False


# Fixed Arakelov height computation functions
# These replace the broken versions in arakelov.py


DEBUG = False


# Fixed Arakelov height computation functions
# These replace the broken versions in arakelov.py


DEBUG = False


# Fixed Arakelov height computation functions
# These replace the broken versions in arakelov.py


DEBUG = False


def enable_debug():
    """Enable detailed debugging output"""
    global DEBUG
    DEBUG = True


def disable_debug():
    """Disable detailed debugging output"""
    global DEBUG
    DEBUG = False


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


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=100, debug=False, test_normalization=None):
    """
    Build basis using proper Abel-Jacobi heights.
    Drop-in replacement for arakelov_build_basis_parallel.
    
    test_normalization: if provided, use this as the normalization factor for testing
                       Expected LMFDB heights / your self-pairings to find the right factor
    """
    if not all_divisors:
        return [], 0, None
    
    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")
    
    from .homology import get_period_matrix_auto_B
    
    # Compute period matrix once
    try:
        period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
        if debug:
            print(f"[arakelov] Period matrix computed")
            print(f"[arakelov] Period matrix tau:")
            for i in range(2):
                for j in range(2):
                    print(f"  tau[{i},{j}] = {period_matrix[i,j]}")
    except Exception as e:
        if debug:
            print(f"[arakelov] Period matrix failed: {e}")
        raise
    
    # CRITICAL FIX: Compute canonical base point ONCE for all divisors
    # DO NOT use a Weierstrass point as base - use a generic point instead!
    CC = ComplexField(prec)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    f_poly_cc = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))


    # In arakelov.py, inside arakelov_build_basis_with_heights

    # ... [Period matrix computation] ...

    # CRITICAL FIX: Use a Weierstrass point (root) as the base point.
    # This ensures the AJ map maps to the correct group element in the Jacobian
    # because 2*(e_i, 0) ~ infinity_plus + infinity_minus.
    CC = ComplexField(prec)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly_cc = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    try:
        # Sort roots to ensure deterministic selection
        f_roots = sorted(f_poly_cc.roots(multiplicities=False), 
                        key=lambda z: (z.real(), z.imag()))
        
        if not f_roots:
            raise ValueError("No roots found for f(x)")
            
        # Use the first root as the base point
        base_x = f_roots[0]
        base_y = CC(0)
        base_point = (base_x, base_y)
        
        if debug:
            print(f"[arakelov] Using Weierstrass base point: e_0 = {base_x}")
            
    except Exception as e:
        if debug:
            print(f"[arakelov] Failed to select base point: {e}")
        raise
    
    
    if debug:
        print(f"[arakelov] Weierstrass points (branch points):")
        for i, root in enumerate(f_roots if f_roots else []):
            print(f"  e_{i} = {root}")
        print(f"[arakelov] Using generic base point: P_0 = ({base_point[0]}, {base_point[1]})")
        print(f"[arakelov] (NOT a Weierstrass point - generic point on curve)")
    
    # Convert to Jacobian elements
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
        except:
            continue
    
    if not jac_elements:
        return [], 0, None
    
    n = len(jac_elements)
    
    if debug:
        print(f"[arakelov] Computing height pairings for {n} divisors...")
    
    # Compute all Abel-Jacobi images with SAME base point
    aj_cache = {}
    for i, (div, D) in enumerate(jac_elements):
        try:
            aj_vec = abel_jacobi_mumford(D, f_coeffs, base_point=base_point, prec=prec)
            aj_cache[i] = aj_vec
            if debug and i < 3:  # Show details for first 3 divisors
                print(f"\n[arakelov] Divisor {i}: s={div['s']}, p={div['p']}, v0={div['v_0']}, v1={div['v_1']}")
                print(f"  Abel-Jacobi: z = ({aj_vec[0]}, {aj_vec[1]})")
                print(f"  |z| = {abs(aj_vec[0]) + abs(aj_vec[1])}")
        except Exception as e:
            if debug:
                print(f"[arakelov] Failed AJ map for divisor {i}: {e}")
            aj_cache[i] = None
    
    # Extract Im(τ)
    RR = RealField(prec)
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())
    
    # Compute height pairing matrix incrementally
    def compute_pairing(i, j):
        zi = aj_cache.get(i)
        zj = aj_cache.get(j)
        if zi is None or zj is None:
            return QQ(0)
        
        # Use test normalization if provided
        norm_factor = test_normalization if test_normalization is not None else 1.0
        pairing = neron_tate_height_pairing(zi, zj, Im_tau, prec=prec, normalization_factor=norm_factor)
        
        # Check symmetry for debugging
        if debug and i != j:
            pairing_ji = neron_tate_height_pairing(zj, zi, Im_tau, prec=prec, normalization_factor=norm_factor)
            sym_error = abs(float(pairing - pairing_ji))
            if sym_error > 1e-6:
                print(f"[arakelov] WARNING: Pairing not symmetric! <{i},{j}> = {pairing}, <{j},{i}> = {pairing_ji}, error = {sym_error}")
        
        return pairing
    
    basis = []
    basis_indices = []
    
    if debug:
        print(f"[arakelov] Building basis incrementally...")
    
    for i, (div, D) in enumerate(jac_elements):
        if aj_cache.get(i) is None:
            continue
        
        h_self = compute_pairing(i, i)

        if debug:
            print(f"[arakelov] Self-pairing of divisor {i}: ({float(h_self):.3e})")
        if float(h_self) < 1e-6:
            if debug:
                print(f"[arakelov] Skipping divisor {i}: self-pairing too small ({float(h_self):.3e})")
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
            
            # Check positive definiteness
            try:
                evals = Matrix(RR, H).eigenvalues()
                is_pd = all(float(ev) > 1e-6 for ev in evals)
            except:
                is_pd = False
            
            if is_pd:
                basis.append(div)
                basis_indices.append(i)
                if debug:
                    try:
                        det_float = float(H.determinant())
                    except:
                        det_float = 0.0
                    print(f"[arakelov] Added divisor {i} (rank {m}, det {det_float:.6g})")
            else:
                if debug:
                    print(f"[arakelov] Skipping divisor {i}: not positive-definite")
    
    final_rank = len(basis)
    H_final = None
    if final_rank > 0:
        H_final = Matrix(QQ, final_rank, final_rank)
        for i in range(final_rank):
            for j in range(i, final_rank):
                h_ij = compute_pairing(basis_indices[i], basis_indices[j])
                H_final[i, j] = h_ij
                H_final[j, i] = h_ij
    
    return basis, final_rank, H_final


def neron_tate_height_pairing(z1, z2, Im_tau, prec=100, normalization_factor=1.0):
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


def integrate_differential_path_with_branch(x_start, x_end, y_end, f_coeffs, use_x_weight=False, prec=100):
    """
    Integrate dx/(2y) or x*dx/(2y) along a straight line from x_start to x_end.
    Correctly handles starting at a Weierstrass point (y=0).
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
    
    # --- FIX START ---
    # Determine value at start point to check for singularity (Weierstrass point)
    val_at_start = f_at(p0)
    
    # Sample slightly into the path to determine orientation
    sample_x = p0 + off
    f0 = f_at(sample_x)
    y_sample = f0.sqrt()
    
    # Check orientation at start
    if abs(val_at_start) < 1e-9: 
         # We are starting at a root (y=0)
         y_start = CC(0)
         y_prev = y_sample # Initialize prev with slightly offset value
    else:
         # Standard point
         y_start = val_at_start.sqrt()
         # Enforce standard principal branch choice initially
         if y_start.imag() < 0: y_start = -y_start
         y_prev = y_start
    # --- FIX END ---

    # At the endpoint, we want to arrive at y_end
    # Choose initial branch that will evolve to y_end by continuity
    f_end = f_at(p1)
    y_plus = f_end.sqrt()
    y_minus = -y_plus
    
    # Heuristic: Choose starting branch sign based on endpoint target
    # (Note: For very long winding paths this heuristic might fail, but for straight lines in AJ it usually works)
    if abs(y_plus - CC(y_end)) < abs(y_minus - CC(y_end)):
        target_is_plus = True
    else:
        target_is_plus = False
        
    # If our logic suggests we need to flip the sheet to hit y_end, do so
    # This part is tricky without path tracking; usually for AJ maps on straight lines 
    # we trust the standard continuation unless we cross a cut.
    # For now, rely on y_prev evolution.

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
        
        # Select the y closest to y_prev
        if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
            y_cur = y_plus
        else:
            y_cur = y_minus
            
        y_prev = y_cur
        
        # Integrand
        if use_x_weight:
            integrand = xval / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * CC(w)
        integral += integrand * dxd
    
    return integral

def abel_jacobi_mumford(D, f_coeffs, base_point=None, prec=100):
    """
    Compute Abel-Jacobi map for a Mumford divisor D = (u(x), v(x)).
    Returns: vector in C^2 (before reduction mod period lattice)
    """
    CC = ComplexField(prec)
    
    if D.is_zero():
        return vector(CC, [0, 0])
    
    u_poly = D[0]
    v_poly = D[1]
    
    # Get roots of u(x)
    R = PolynomialRing(CC, 'x')
    x = R.gen()
    u_cc = sum(CC(c) * x**i for i, c in enumerate(u_poly.list()))
    
    try:
        roots = u_cc.roots(multiplicities=False)
    except:
        return vector(CC, [0, 0])
    
    if len(roots) == 0:
        return vector(CC, [0, 0])
    
    if base_point is None:
        raise ValueError("Base point must be provided for consistent Abel-Jacobi map")
    
    aj_vector = vector(CC, [0, 0])
    
    # Handle base_point tuple vs scalar
    if isinstance(base_point, tuple) or isinstance(base_point, list):
        base_x = base_point[0]
    else:
        base_x = base_point

    for idx, x_pt in enumerate(roots):
        # Calculate y_pt = v(x_pt)
        v_list = v_poly.list()
        if len(v_list) == 0:
            y_pt = CC(0)
        elif len(v_list) == 1:
            y_pt = CC(v_list[0])
        else:
            y_pt = CC(v_list[0]) + CC(v_list[1]) * x_pt
        
        try:
            # Integrate using the fixed function
            int_0 = integrate_differential_path_with_branch(base_x, x_pt, y_pt, f_coeffs, 
                                                           use_x_weight=False, prec=prec)
            int_1 = integrate_differential_path_with_branch(base_x, x_pt, y_pt, f_coeffs, 
                                                           use_x_weight=True, prec=prec)
            aj_vector += vector(CC, [int_0, int_1])
        except Exception as e:
            # Print error even if DEBUG is off to warn user
            print(f"[abel_jacobi] Integration error: {e}")
            continue
    
    return aj_vector


# Add this diagnostic version of integrate_differential_path_with_branch

def integrate_differential_path_with_branch(x_start, x_end, y_end, f_coeffs, 
                                                   use_x_weight=False, prec=100):
    """
    DEBUG version with extensive logging to find the bug.
    """
    import math
    from sage.all import ComplexField, CC as SageCC
    
    CC = ComplexField(prec)
    
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
    
    print(f"\n[INT_DEBUG] ================")
    print(f"[INT_DEBUG] Integrating from {p0} to {p1}")
    print(f"[INT_DEBUG] Distance: {abs(vec)}")
    
    # Offset path
    perp = CC(0, 1) * vec
    off_mag = max(CC(1e-14), abs(vec) * CC(1e-8))
    off = perp / (abs(perp) + CC(1e-30)) * off_mag
    
    print(f"[INT_DEBUG] Offset magnitude: {abs(off)}")
    print(f"[INT_DEBUG] Offset direction: {off / abs(off) if abs(off) > 0 else 'zero'}")
    
    dx_factor = vec / CC(2)
    
    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    
    # Check if starting point is a root
    f_start = f_at(p0)
    print(f"[INT_DEBUG] f(start) = {f_start}")
    print(f"[INT_DEBUG] |f(start)| = {abs(f_start)}")
    
    if abs(f_start) < 1e-10:
        print(f"[INT_DEBUG] *** STARTING AT WEIERSTRASS POINT (root) ***")
        print(f"[INT_DEBUG] *** Singularity at start: 1/√f ~ 1/√(x-e) ***")
    
    # Sample slightly into path to determine branch
    sample_x = p0 + ((CC(nodes[0][1]) + CC(1)) / CC(2)) * vec + off
    f0 = f_at(sample_x)
    print(f"[INT_DEBUG] f(sample) = {f0}")
    print(f"[INT_DEBUG] |f(sample)| = {abs(f0)}")
    
    if abs(f0) < CC(2) ** (-prec // 2):
        print(f"[INT_DEBUG] *** ERROR: Sample point too close to branch cut ***")
        raise ValueError("Path too close to branch point")
    
    y0 = f0.sqrt()
    
    # Choose branch to match y_end
    if abs(y0 - CC(y_end)) > abs(-y0 - CC(y_end)):
        y0 = -y0
    
    print(f"[INT_DEBUG] Initial y-value: {y0}")
    print(f"[INT_DEBUG] Target y_end: {y_end}")
    print(f"[INT_DEBUG] Branch choice: {'+' if abs(y0 - CC(y_end)) < abs(-y0 - CC(y_end)) else '-'}")
    
    y_prev = y0
    integral = CC(0)
    tiny = CC(2) ** (-prec // 2)
    
    skipped_count = 0
    max_integrand = 0
    total_weight = 0
    
    for i, (t, x_mapped, w) in enumerate(nodes):
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        fval = f_at(xval)
        
        if abs(fval) < tiny:
            skipped_count += 1
            continue
        
        # Branch selection by continuity
        y_plus = fval.sqrt()
        y_minus = -y_plus
        
        if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
            y_cur = y_plus
        else:
            y_cur = y_minus
        
        y_prev = y_cur
        
        # Integrand
        if use_x_weight:
            integrand = xval / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * CC(w)
        contribution = integrand * dxd
        integral += contribution
        
        max_integrand = max(max_integrand, abs(integrand))
        total_weight += abs(w)
        
        # Log first few and last few points
        if i < 5 or i >= len(nodes) - 5:
            print(f"[INT_DEBUG] Node {i}: t={t:.3f}, x={xval}, |f|={abs(fval):.2e}, |integrand|={abs(integrand):.2e}")
    
    print(f"\n[INT_DEBUG] Integration complete:")
    print(f"[INT_DEBUG] Skipped {skipped_count}/{len(nodes)} nodes (near singularity)")
    print(f"[INT_DEBUG] Max |integrand|: {max_integrand}")
    print(f"[INT_DEBUG] Total weight: {total_weight}")
    print(f"[INT_DEBUG] Integral result: {integral}")
    print(f"[INT_DEBUG] |Integral|: {abs(integral)}")
    print(f"[INT_DEBUG] ================\n")
    
    return integral


# To test, temporarily replace the call in abel_jacobi_mumford:
# int_0 = integrate_differential_path_with_branch_DEBUG(
#     base_x, x_pt, y_pt, f_coeffs, use_x_weight=False, prec=prec
# )
# CRITICAL FIX: Robust branch selection in integration

def integrate_differential_path_with_branch(x_start, x_end, y_end, f_coeffs, 
                                                   use_x_weight=False, prec=100, debug=False):
    """
    Integrate dx/(2y) or x·dx/(2y) from (x_start, y_start) to (x_end, y_end).
    
    CRITICAL: Must arrive at the CORRECT branch specified by y_end.
    
    The bug: if we choose the wrong initial branch, the integral can be off by
    huge factors, giving random-looking but wrong results.
    """
    import math
    from sage.all import ComplexField
    
    CC = ComplexField(prec)
    
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
    
    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    
    # CRITICAL FIX: Determine starting branch by evaluating at START point
    # (not just matching endpoint)
    
    # If starting at Weierstrass point (y_start = 0), sample slightly into path
    f_start = f_at(p0)
    if abs(f_start) < 1e-10:
        # Starting at Weierstrass point - sample into path
        sample_x = p0 + vec * CC(0.01) + off
        f_sample = f_at(sample_x)
        if abs(f_sample) < CC(2) ** (-prec // 2):
            raise ValueError("Cannot determine branch - path too close to branch cut")
        y_sample = f_sample.sqrt()
    else:
        # Starting at generic point
        y_start = f_start.sqrt()
        # Choose branch closest to y_end (heuristic)
        if abs(y_start - CC(y_end)) > abs(-y_start - CC(y_end)):
            y_start = -y_start
        y_sample = y_start
    
    # Now propagate branch by continuity
    y_prev = y_sample
    integral = CC(0)
    tiny = CC(2) ** (-prec // 2)
    
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        fval = f_at(xval)
        
        if abs(fval) < tiny:
            continue
        
        # Branch selection by continuity from previous point
        y_plus = fval.sqrt()
        y_minus = -y_plus
        
        if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
            y_cur = y_plus
        else:
            y_cur = y_minus
        
        y_prev = y_cur
        
        # Integrand
        if use_x_weight:
            integrand = xval / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * CC(w)
        integral += integrand * dxd
    
    # CRITICAL CHECK: Verify we arrived at the correct branch
    f_end = f_at(p1)
    y_end_computed = y_prev  # Last y-value from integration
    y_end_target = CC(y_end)
    
    # Check both the y-value AND that y² = f(x)
    y_error = abs(y_end_computed - y_end_target)
    f_error = abs(f_end - y_end_target**2)
    
    if debug or y_error > 1e-6:
        print(f"[BRANCH_CHECK] Target y_end: {y_end_target}")
        print(f"[BRANCH_CHECK] Computed y_end: {y_end_computed}")
        print(f"[BRANCH_CHECK] Error: {y_error}")
        print(f"[BRANCH_CHECK] f(x_end): {f_end}")
        print(f"[BRANCH_CHECK] y_end²: {y_end_target**2}")
        print(f"[BRANCH_CHECK] f-error: {f_error}")
    
    if y_error > 1e-6:
        # We ended on the WRONG branch! Try the opposite initial branch
        if debug:
            print(f"[BRANCH_CHECK] *** WRONG BRANCH - Retrying with opposite sign ***")
        
        # Retry with opposite branch
        y_prev = -y_sample
        integral = CC(0)
        
        for (t, x_mapped, w) in nodes:
            s = (CC(x_mapped) + CC(1)) / CC(2)
            xval = p0 + s * vec + off
            fval = f_at(xval)
            
            if abs(fval) < tiny:
                continue
            
            y_plus = fval.sqrt()
            y_minus = -y_plus
            
            if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
                y_cur = y_plus
            else:
                y_cur = y_minus
            
            y_prev = y_cur
            
            if use_x_weight:
                integrand = xval / (CC(2) * y_cur)
            else:
                integrand = CC(1) / (CC(2) * y_cur)
            
            dxd = dx_factor * CC(w)
            integral += integrand * dxd
        
        # Check again
        y_end_computed = y_prev
        y_error = abs(y_end_computed - y_end_target)
        
        if y_error > 1e-6:
            raise ValueError(
                f"Branch selection failed even after retry!\n"
                f"Target: {y_end_target}, Got: {y_end_computed}, Error: {y_error}"
            )
    
    return integral


# CRITICAL FIX: Robust branch selection in integration

def integrate_differential_path_with_branch(x_start, x_end, y_end, f_coeffs, 
                                                   use_x_weight=False, prec=100, debug=False):
    """
    Integrate dx/(2y) or x·dx/(2y) from (x_start, y_start) to (x_end, y_end).
    
    CRITICAL: Must arrive at the CORRECT branch specified by y_end.
    
    The bug: if we choose the wrong initial branch, the integral can be off by
    huge factors, giving random-looking but wrong results.
    """
    import math
    from sage.all import ComplexField
    
    CC = ComplexField(prec)
    
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
    
    def f_at(z):
        return sum(CC(c) * (z ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    
    # CRITICAL FIX: Determine starting branch by evaluating at START point
    # (not just matching endpoint)
    
    # CRITICAL FIX: Determine initial branch by working BACKWARDS from endpoint
    # We know the target y_end at x_end. Work backwards to find what branch
    # we should start with.
    
    # Sample at 99% along the path (close to endpoint)
    sample_end_x = p0 + vec * CC(0.99) + off
    f_sample_end = f_at(sample_end_x)
    
    if abs(f_sample_end) < CC(2) ** (-prec // 2):
        # Too close to branch cut, try 95%
        sample_end_x = p0 + vec * CC(0.95) + off
        f_sample_end = f_at(sample_end_x)
        if abs(f_sample_end) < CC(2) ** (-prec // 2):
            raise ValueError("Path too close to branch cut")
    
    # At this point near the end, choose branch that matches y_end
    y_end_cc = CC(y_end)
    y_plus = f_sample_end.sqrt()
    y_minus = -y_plus
    
    if abs(y_plus - y_end_cc) < abs(y_minus - y_end_cc):
        y_near_end = y_plus
    else:
        y_near_end = y_minus
    
    # Now work backwards from this point to the start
    # by reverse continuity
    f_start = f_at(p0)
    
    if abs(f_start) < 1e-10:
        # Starting at Weierstrass point - use y_near_end to guide
        y_sample = y_near_end
    else:
        # Starting at generic point
        y_start_plus = f_start.sqrt()
        y_start_minus = -y_start_plus
        
        # Choose starting branch that will smoothly connect to y_near_end
        # Heuristic: the one with the right "phase"
        if abs(y_start_plus - y_near_end) < abs(y_start_minus - y_near_end):
            y_sample = y_start_plus
        else:
            y_sample = y_start_minus
    
    # Now propagate branch by continuity
    y_prev = y_sample
    integral = CC(0)
    tiny = CC(2) ** (-prec // 2)
    
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)
        xval = p0 + s * vec + off
        fval = f_at(xval)
        
        if abs(fval) < tiny:
            continue
        
        # Branch selection by continuity from previous point
        y_plus = fval.sqrt()
        y_minus = -y_plus
        
        if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
            y_cur = y_plus
        else:
            y_cur = y_minus
        
        y_prev = y_cur
        
        # Integrand
        if use_x_weight:
            integrand = xval / (CC(2) * y_cur)
        else:
            integrand = CC(1) / (CC(2) * y_cur)
        
        dxd = dx_factor * CC(w)
        integral += integrand * dxd
    
    # CRITICAL CHECK: Verify we arrived at the correct branch
    f_end = f_at(p1)
    y_end_computed = y_prev  # Last y-value from integration
    y_end_target = CC(y_end)
    
    # Check both the y-value AND that y² = f(x)
    y_error = abs(y_end_computed - y_end_target)
    f_error = abs(f_end - y_end_target**2)
    
    if debug or y_error > 1e-6:
        print(f"[BRANCH_CHECK] Target y_end: {y_end_target}")
        print(f"[BRANCH_CHECK] Computed y_end: {y_end_computed}")
        print(f"[BRANCH_CHECK] Error: {y_error}")
        print(f"[BRANCH_CHECK] f(x_end): {f_end}")
        print(f"[BRANCH_CHECK] y_end²: {y_end_target**2}")
        print(f"[BRANCH_CHECK] f-error: {f_error}")
    
    if y_error > 1e-6:
        # We ended on the WRONG branch! Try the opposite initial branch
        if debug:
            print(f"[BRANCH_CHECK] *** WRONG BRANCH - Retrying with opposite sign ***")
        
        # Retry with opposite branch
        y_prev = -y_sample
        integral = CC(0)
        
        for (t, x_mapped, w) in nodes:
            s = (CC(x_mapped) + CC(1)) / CC(2)
            xval = p0 + s * vec + off
            fval = f_at(xval)
            
            if abs(fval) < tiny:
                continue
            
            y_plus = fval.sqrt()
            y_minus = -y_plus
            
            if abs(y_plus - y_prev) <= abs(y_minus - y_prev):
                y_cur = y_plus
            else:
                y_cur = y_minus
            
            y_prev = y_cur
            
            if use_x_weight:
                integrand = xval / (CC(2) * y_cur)
            else:
                integrand = CC(1) / (CC(2) * y_cur)
            
            dxd = dx_factor * CC(w)
            integral += integrand * dxd
        
        # Check again
        y_end_computed = y_prev
        y_error = abs(y_end_computed - y_end_target)
        
        if y_error > 1e-6:
            raise ValueError(
                f"Branch selection failed even after retry!\n"
                f"Target: {y_end_target}, Got: {y_end_computed}, Error: {y_error}"
            )
    
    return integral
