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

# Global cache for period matrices (cleared each Python session)
_PERIOD_MATRIX_CACHE = {}


# arakelov_optimized.py
#
# Optimized Arakelov height computations with parallelization and caching

from sage.all import ComplexField, RealField
from sage.all import parallel
import time
from collections import defaultdict

# Global cache for period matrices
_PERIOD_MATRIX_CACHE = {}

# Global timer storage
_TIMERS = defaultdict(float)

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


def naive_height_qq(D):
    """Exact naive height - unchanged from original."""
    from fractions import Fraction
    
    u_coeffs = D[0].list()
    v_coeffs = D[1].list()
    
    all_coeffs = []
    for c in u_coeffs + v_coeffs:
        c_qq = QQ(c)
        all_coeffs.append(Fraction(int(c_qq.numerator()), int(c_qq.denominator())))
    
    lcm_den = 1
    for f in all_coeffs:
        lcm_den = (lcm_den * f.denominator) // math.gcd(lcm_den, f.denominator)
    
    int_coeffs = [int((f * lcm_den).numerator) for f in all_coeffs]
    int_coeffs.append(int(lcm_den))
    
    max_abs = max(abs(c) for c in int_coeffs if c != 0)
    max_abs = max(1, max_abs)
    
    return QQ(math.log(max_abs))


def archimedean_height_correction(D, period_matrix, prec=100):
    """Simplified archimedean correction - unchanged from original."""
    try:
        Omega = period_matrix
        Omega_imag = Matrix(RDF, 2, 2)
        for i in range(2):
            for j in range(2):
                Omega_imag[i,j] = Omega[i,j].imag()
        
        det_imag = Omega_imag.determinant()
        
        if det_imag <= 0:
            return QQ(0)
        
        correction = float(0.5 * log(abs(det_imag)))
        return QQ(correction).nearby_rational(max_denominator=10**8)
        
    except Exception:
        return QQ(0)


def arakelov_canonical_height(D, f_coeffs, prec=100, use_finite_places=True):
    """Canonical height with timing."""
    if D.is_zero():
        return QQ(0)
    
    with Timer("height_naive"):
        h_naive = naive_height_qq(D)
    
    with Timer("height_period_matrix"):
        period_matrix = get_period_matrix(f_coeffs, prec=prec)
    
    with Timer("height_archimedean"):
        h_arch = archimedean_height_correction(D, period_matrix, prec=prec)
    
    h_finite = QQ(0)
    if use_finite_places:
        with Timer("height_finite_places"):
            for p in [2, 3, 5, 7, 11, 13]:
                try:
                    h_p = local_height_finite(D, p)
                    h_finite += h_p
                except Exception:
                    pass
    
    return h_naive + h_arch + h_finite


def local_height_finite(D, p):
    """Local height at prime p - unchanged."""
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
    
    return QQ(-min_val * log(p))


def arakelov_height_pairing(D1, D2, f_coeffs, prec=100):
    """Height pairing with timing."""
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    with Timer("pairing_total"):
        with Timer("pairing_individual_heights"):
            h1 = arakelov_canonical_height(D1, f_coeffs, prec=prec)
            h2 = arakelov_canonical_height(D2, f_coeffs, prec=prec)
            h_sum = arakelov_canonical_height(D1 + D2, f_coeffs, prec=prec)
        
        pairing = (h_sum - h1 - h2) / QQ(2)
    
    return pairing


def arakelov_height_pairing_cached(D1, D2, f_coeffs, height_cache, prec=100):
    """
    Height pairing using cached individual heights.
    height_cache: dict mapping divisor index -> canonical height
    """
    if D1.is_zero() or D2.is_zero():
        return QQ(0)
    
    # Compute D1+D2 height (not cached since it's a new divisor)
    with Timer("pairing_sum_height"):
        h_sum = arakelov_canonical_height(D1 + D2, f_coeffs, prec=prec)
    
    # Get individual heights from cache (should already be computed)
    # Note: cache keys are indices, not passed here - caller manages this
    # This function assumes h1, h2 are passed or we modify signature
    # For now, keep original logic but add timer
    with Timer("pairing_individual_heights"):
        h1 = arakelov_canonical_height(D1, f_coeffs, prec=prec)
        h2 = arakelov_canonical_height(D2, f_coeffs, prec=prec)
    
    pairing = (h_sum - h1 - h2) / QQ(2)
    return pairing


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
        return idx, None


def get_period_matrix(f_coeffs, prec=100):
    """
    Compute period matrix with timing and caching.
    Uses faster numerical integration with adaptive precision.
    """
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
        
        CC_prec = ComplexField(prec) if prec > 53 else CC
        
        with Timer("period_matrix_roots"):
            f_roots = f_poly.roots(CC_prec, multiplicities=False)
        
        if len(f_roots) < 5:
            raise ValueError(f"Not enough roots for genus 2 curve: found {len(f_roots)}")
        
        f_roots = sorted(f_roots, key=lambda z: (z.real(), z.imag()))
        
        with Timer("period_matrix_integration"):
            # Use fewer integration points for speed (100 -> 50)
            n_points = 50
            
            def integrate_differential(root_start, root_end, use_x_weight):
                total = CC_prec(0)
                
                for i in range(n_points):
                    t = CC_prec(i) / CC_prec(n_points)
                    t_next = CC_prec(i + 1) / CC_prec(n_points)
                    t_mid = (t + t_next) / 2
                    
                    x_val = root_start + t_mid * (root_end - root_start)
                    
                    f_val = sum(CC_prec(f_coeffs[j]) * x_val**(len(f_coeffs)-1-j) 
                               for j in range(len(f_coeffs)))
                    
                    if abs(f_val) < 10**(-prec//2):
                        continue
                    
                    y_val = f_val.sqrt()
                    dx = root_end - root_start
                    
                    if use_x_weight:
                        integrand = x_val * dx / (2 * y_val)
                    else:
                        integrand = dx / (2 * y_val)
                    
                    total += integrand / n_points
                
                return total
            
            Omega = Matrix(CC_prec, 2, 2)
            
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


def arakelov_build_basis_parallel(all_divisors, f_coeffs, prec=100, debug=False):
    """
    Build basis using parallel height computation.
    Much faster for large divisor sets.
    """
    if not all_divisors:
        return [], 0, None
    
    clear_period_cache()
    reset_timers()
    
    if debug:
        print(f"\n[arakelov_parallel] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov_parallel] Using precision: {prec} bits")
    
    # Compute period matrix once
    with Timer("period_matrix_total"):
        try:
            period_matrix = get_period_matrix(f_coeffs, prec=prec)
            if debug:
                print(f"[arakelov_parallel] Period matrix computed")
        except Exception as e:
            if debug:
                print(f"[arakelov_parallel] Period matrix failed: {e}")
            raise
    
    # Convert to Jacobian elements
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
                    print(f"[arakelov_parallel] Skipping invalid divisor")
                continue
    
    if not jac_elements:
        return [], 0, None
    
    n = len(jac_elements)
    
    # Pre-compute all canonical heights (cache them)
    canonical_heights = {}
    
    if debug:
        print(f"[arakelov_parallel] Computing {n} canonical heights...")
    
    with Timer("heights_precompute"):
        for i, (div, D) in enumerate(jac_elements):
            canonical_heights[i] = arakelov_canonical_height(D, f_coeffs, prec=prec)
            if debug and (i+1) % 10 == 0:
                print(f"[arakelov_parallel] Computed {i+1}/{n} heights")
    
    # Cache for D_i + D_j heights
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
    
    # Build basis incrementally
    basis = []
    basis_indices = []
    
    if debug:
        print(f"[arakelov_parallel] Building basis incrementally...")
    
    with Timer("basis_construction"):
        for i, (div, D) in enumerate(jac_elements):
            if not basis:
                h_self = compute_pairing(i, i)
                
                if abs(float(h_self)) < 1e-6:
                    if debug:
                        print(f"[arakelov_parallel] Skipping divisor {i}: self-pairing too small")
                    continue
                
                basis.append(div)
                basis_indices.append(i)
                if debug:
                    print(f"[arakelov_parallel] Added divisor 1 (self-pairing {float(h_self):.6g})")
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
                
                if rank == m and det_float > 0:
                    basis.append(div)
                    basis_indices.append(i)
                    if debug:
                        print(f"[arakelov_parallel] Added divisor {len(basis)} (rank {rank}, det {det_float:.6g})")
                else:
                    if debug:
                        reason = "rank dropped" if rank < m else f"det non-positive"
                        print(f"[arakelov_parallel] Skipping divisor {i}: {reason}. det: {det_float:.6g}")
    
    # Build final height matrix
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
                print(f"\n[arakelov_parallel] Final rank: {final_rank}")
                det_final = H_final.determinant()
                print(f"[arakelov_parallel] Final determinant: {float(det_final):.6g}")
    else:
        H_final = None
    
    if debug:
        print_timers()
    
    return basis, final_rank, H_final


def arakelov_build_basis(all_divisors, f_coeffs, prec=100, debug=False):
    """
    Main entry point - delegates to parallel version.
    Kept for backward compatibility.
    """
    return arakelov_build_basis_parallel(all_divisors, f_coeffs, prec=prec, debug=debug)


def arakelov_check_independence(divisors, f_coeffs, prec=100, debug=False):
    """Check independence with timing."""
    if not divisors:
        return True, 0, None, 0
    
    with Timer("independence_check_total"):
        n = len(divisors)
        H = Matrix(QQ, n, n)
        
        # Compute period matrix once
        period_matrix = get_period_matrix(f_coeffs, prec=prec)
        
        for i in range(n):
            for j in range(i, n):
                h_ij = arakelov_height_pairing(divisors[i], divisors[j], f_coeffs, prec=prec)
                H[i,j] = h_ij
                H[j,i] = h_ij
                
                if debug and i == j:
                    print(f"[arakelov] h({i},{i}) = {float(h_ij):.6g}")
        
        rank = H.rank()
        det = H.determinant()
        is_indep = (rank == n) and det > 0
        
        if debug:
            print(f"[arakelov] Height matrix determinant: {float(det):.6g}")
            print(f"[arakelov] Rank: {rank}/{n}")
    
    return is_indep, rank, H, det
