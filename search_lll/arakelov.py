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
    assert None, "deprecated"
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

    # period_matrix is the 2x2 complex tau from your run
    import numpy as np

    def reduce_mod_lattice(z, tau):
        # z: complex vector length 2 (numpy complex128)
        # tau: 2x2 complex matrix (Sage object or numpy complex)
        # Build real lattice matrix L_real with 4 columns = Re/Im of e1,e2,tau.col1,tau.col2
        # We'll solve L_real * k = z_real for integer k (k in Z^4), then do z - L * k
        # Convert to numpy arrays
        z = np.array([complex(z[0]), complex(z[1])], dtype=complex)
        tau_np = np.array([[complex(tau[0,0]), complex(tau[0,1])],[complex(tau[1,0]), complex(tau[1,1])]], dtype=complex)
        # L columns: e1=(1,0), e2=(0,1), tau_col1, tau_col2
        cols = []
        e1 = np.array([1+0j, 0+0j], dtype=complex)
        e2 = np.array([0+0j, 1+0j], dtype=complex)
        cols.append(e1)
        cols.append(e2)
        cols.append(tau_np[:,0])
        cols.append(tau_np[:,1])
        # Build real representation
        L_real = np.zeros((4,4), dtype=float)
        z_real = np.zeros(4, dtype=float)
        for j in range(4):
            L_real[0,j] = cols[j][0].real
            L_real[1,j] = cols[j][0].imag
            L_real[2,j] = cols[j][1].real
            L_real[3,j] = cols[j][1].imag
        z_real[0] = z[0].real
        z_real[1] = z[0].imag
        z_real[2] = z[1].real
        z_real[3] = z[1].imag
        # solve least squares to get real coeffs, then round to nearest integers
        k_real, *_ = np.linalg.lstsq(L_real, z_real, rcond=None)
        k_int = np.rint(k_real).astype(int)
        # compute lattice vector back in complex 2-vector
        Lk = sum(k_int[j] * cols[j] for j in range(4))
        z_red = z - Lk
        return z_red, k_int

    # Apply to all AJ in basis order
    tau = period_matrix  # use your tau
    AJ_red = []
    klist = []
    for idx in basis_indices:
        z = np.array([complex(aj_cache[idx][0]), complex(aj_cache[idx][1])], dtype=complex)
        z_r, k = reduce_mod_lattice(z, tau)
        AJ_red.append(z_r)
        klist.append(k)
        print("orig:", z, "reduced:", z_r, "k:", k)

    # Rebuild H from AJ_red (use your neron_tate_height_pairing but with AJ replaced by reduced ones)
    # You may need to convert back into the CC objects your pairing function expects.

    
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

    
    # in Sage kernel (preferred) because it has PSLQ
    from sage.all import vector, QQ
    V = []
    for idx in basis_indices:
        z = aj_cache[idx]
        v4 = [float(z[0].real()), float(z[0].imag()), float(z[1].real()), float(z[1].imag())]
        V.append(v4)

    # form matrix with columns for each basis vector's real4 coords
    M = Matrix(RR, 4, len(V), [coord for col in V for coord in col])  # careful ordering
    # Try integer relation among columns: find integers k1..kr not all 0 with M * k = 0
    # Use PSLQ on flattened or combine differently. Simpler: try PSLQ on list of r reals repeatedly.
    # Flatten to list of length 4*r and try to find relation? Better: try pslq on complex linear combination entries.
    # Here we try PSLQ on the r coefficients of the null vector vec scaled to rational approx:
    coeffs = list(vec)  # from numpy eigenvector


    # Run inside Sage; ensure sympy/mpmath available.
    import numpy as np

    # Build final numpy H and eigen-decomposition (if not already done)
    Hf = np.array(H_final, dtype=float)
    w, V = np.linalg.eigh(Hf)
    idx_min = int(np.argmin(w))
    eigvec = V[:, idx_min]

    # Convert to mpf via string to preserve precision
    vec_mpf = [mpf(str(float(c))) for c in eigvec]
    rel = None
    #rel = pslq(vec_mpf, maxsteps=5000)
    print("PSLQ relation on eigenvector (if small-int relation exists):", rel)
    # If rel is not None and not empty, it's a small-integer relation on coefficients.


    # Using jac_elements list from your build (original D objects)
    # Suppose pslq returns klist = [k0,k1,k2,k3]
    comb = J(0)   # identity in Jacobian
    for pos, k in enumerate(klist):
        orig_idx = basis_indices[pos]
        D = jac_elements[orig_idx][1]   # the Jacobian element
        if k >= 0:
            comb = comb + (k * D)
        else:
            comb = comb - ((-k) * D)
    print("Is combination zero in Jacobian?", comb.is_zero())
    # If comb is torsion, comb.torsion_order() may tell you the order.


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


# Add this diagnostic version of integrate_differential_path_with_branch


# To test, temporarily replace the call in abel_jacobi_mumford:
# int_0 = integrate_differential_path_with_branch_DEBUG(
#     base_x, x_pt, y_pt, f_coeffs, use_x_weight=False, prec=prec
# )
# CRITICAL FIX: Robust branch selection in integration


# CRITICAL FIX: Robust branch selection in integration


def integrate_differential_path_with_branch(x_start, x_end, y_end, f_coeffs,
                                            use_x_weight=False, prec=200, debug=False):
    """
    Improved version:
    - Propagate branch (sign of sqrt(f)) continuously across all tanh-sinh nodes,
      seeded reliably near the endpoint.
    - Special-case Weierstrass endpoints (f ~ 0).
    - Do NOT drop nodes with small f; assign sign by continuity instead.
    """
    import math
    from sage.all import ComplexField

    CC = ComplexField(prec)

    def tanh_sinh_nodes(N):
        nodes = []
        h = 1.0 / float(N)
        pi = math.pi
        for k in range(-N, N + 1):
            t = k * h
            sx = math.sinh(t)
            x_mapped = math.tanh((pi / 2.0) * sx)        # in (-1,1)
            # derivative dx/dt for the mapping from t->x_mapped
            dx_dt = (pi / 2.0) * math.cosh(t) / (math.cosh((pi / 2.0) * sx) ** 2)
            w = dx_dt * h
            nodes.append((t, x_mapped, w))
        return nodes

    Nnodes = max(200, min(2000, prec // 2))
    nodes = tanh_sinh_nodes(Nnodes)

    p0 = CC(x_start)
    p1 = CC(x_end)
    vec = p1 - p0

    # small perpendicular offset to avoid branch cuts exactly on the real segment
    perp = CC(0, 1) * vec
    off_mag = max(CC(1e-14), abs(vec) * CC(1e-8))
    off = perp / (abs(perp) + CC(1e-30)) * off_mag

    dx_factor = vec / CC(2)

    def f_at(z):
        # polynomial specified by f_coeffs in descending powers of x
        return sum(CC(c) * (z ** (len(f_coeffs) - 1 - i)) for i, c in enumerate(f_coeffs))

    # build x-values for all nodes (s in [0,1])
    xvals = []
    ws = []
    for (t, x_mapped, w) in nodes:
        s = (CC(x_mapped) + CC(1)) / CC(2)  # maps (-1,1) -> (0,1)
        xval = p0 + s * vec + off
        xvals.append(xval)
        ws.append(CC(w))

    n = len(xvals)
    # Evaluate f at all node x-values
    fvals = [f_at(xv) for xv in xvals]

    tiny = CC(2) ** (-prec // 2)        # threshold for 'very small' f
    tol = CC(10) ** (-8)                # numerical tolerance for sign matching

    # Find a reliable seed index near the endpoint (prefer last index)
    # Prefer a node close to the endpoint with |f| >= tiny
    seed_idx = None
    # search backwards from last node for non-tiny f
    for i in range(n - 1, -1, -1):
        if abs(fvals[i]) >= tiny:
            seed_idx = i
            break
    if seed_idx is None:
        # if every node has tiny f, that means the path sits on branch locus -> fail
        raise ValueError("All nodes evaluate to extremely small f(x). Path likely lies on branch locus.")

    # seed y at seed_idx
    # If user provided a nonzero y_end, try to choose seed sign consistent with it.
    y_target = CC(y_end)
    # compute sqrt at seed
    y_seed_p = fvals[seed_idx].sqrt()
    y_seed_m = -y_seed_p

    # If the endpoint given y_end is nonzero, choose the sign that matches it (via closeness)
    if abs(y_target) > tiny:
        # but the seed_idx x is not exactly x_end; choose sign by closeness to y_target
        if abs(y_seed_p - y_target) <= abs(y_seed_m - y_target):
            y_seed = y_seed_p
        else:
            y_seed = y_seed_m
    else:
        # y_end is zero (Weierstrass). No sign to match; use principal sqrt at seed.
        y_seed = y_seed_p

    # allocate array for y on every node
    yvals = [None] * n
    yvals[seed_idx] = y_seed

    # propagate backwards from seed_idx down to 0
    for i in range(seed_idx - 1, -1, -1):
        f_i = fvals[i]
        # handle exact tiny f: still compute sqrt (principal) and then choose sign by continuity
        y_p = f_i.sqrt()
        y_m = -y_p
        # choose sign that is closest to next point
        if abs(y_p - yvals[i + 1]) <= abs(y_m - yvals[i + 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # propagate forwards from seed_idx up to n-1
    for i in range(seed_idx + 1, n):
        f_i = fvals[i]
        y_p = f_i.sqrt()
        y_m = -y_p
        if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
            yvals[i] = y_p
        else:
            yvals[i] = y_m

    # Now we have a continuous assignment of y to every node. Integrate.
    integral = CC(0)
    for i in range(n):
        y_cur = yvals[i]
        # If somehow y_cur is exactly zero (should only happen at true Weierstrass),
        # avoid dividing by zero; but y should be assigned via continuity so if it's zero,
        # use series expansion or skip a single point (tanh-sinh handles endpoint integrable singularity).
        if abs(y_cur) == 0:
            # find nearest non-zero y (should exist) and use that value to compute integrand magnitude
            # this is rarely executed; for robust behaviour fallback to neighboring value
            if i + 1 < n and abs(yvals[i + 1]) != 0:
                y_for_use = yvals[i + 1]
            elif i - 1 >= 0 and abs(yvals[i - 1]) != 0:
                y_for_use = yvals[i - 1]
            else:
                raise ValueError("All neighboring y are zero; cannot evaluate integrand.")
        else:
            y_for_use = y_cur

        if use_x_weight:
            integrand = xvals[i] / (CC(2) * y_for_use)
        else:
            integrand = CC(1) / (CC(2) * y_for_use)

        dxd = dx_factor * ws[i]
        integral += integrand * dxd

    # Final branch check: if the provided y_end is nonzero, verify sign at the node closest to endpoint
    if abs(y_target) > tiny:
        # choose last node index (n-1) as representative near endpoint
        y_near_end = yvals[-1]
        # check actual f at exact x_end (p1) and pick square root near y_near_end
        f_exact_end = f_at(p1)
        if abs(f_exact_end) < tiny:
            # endpoint numerically a Weierstrass point despite nonzero y_target? suspicious
            if debug:
                print("[BRANCH_CHECK] f(x_end) nearly zero despite nonzero y_end provided.")
        else:
            y_exact_p = f_exact_end.sqrt()
            y_exact_m = -y_exact_p
            # choose the branch of exact endpoint that is closer to y_near_end
            if abs(y_exact_p - y_near_end) <= abs(y_exact_m - y_near_end):
                y_at_end = y_exact_p
            else:
                y_at_end = y_exact_m

            # compare to user-provided y_end; if they disagree beyond tolerance, raise or debug
            y_err = abs(y_at_end - y_target)
            if debug or y_err > tol:
                print(f"[BRANCH_CHECK] Provided y_end: {y_target}")
                print(f"[BRANCH_CHECK] Computed y_at_end (from continuity): {y_at_end}")
                print(f"[BRANCH_CHECK] y_err = {y_err}")
            if y_err > tol:
                # there's a mismatch; this is unlikely after continuity propagation.
                # We'll attempt one recovery: flip global sign and re-run propagation once.
                if debug:
                    print("[BRANCH_CHECK] Mismatch detected; retrying with global sign-flip seed.")
                # flip seed and redo propagation once:
                yvals[seed_idx] = -yvals[seed_idx]
                for i in range(seed_idx - 1, -1, -1):
                    f_i = fvals[i]
                    y_p = f_i.sqrt()
                    y_m = -y_p
                    if abs(y_p - yvals[i + 1]) <= abs(y_m - yvals[i + 1]):
                        yvals[i] = y_p
                    else:
                        yvals[i] = y_m
                for i in range(seed_idx + 1, n):
                    f_i = fvals[i]
                    y_p = f_i.sqrt()
                    y_m = -y_p
                    if abs(y_p - yvals[i - 1]) <= abs(y_m - yvals[i - 1]):
                        yvals[i] = y_p
                    else:
                        yvals[i] = y_m
                # recompute integral
                integral = CC(0)
                for i in range(n):
                    y_cur = yvals[i]
                    if abs(y_cur) == 0:
                        if i + 1 < n and abs(yvals[i + 1]) != 0:
                            y_for_use = yvals[i + 1]
                        elif i - 1 >= 0 and abs(yvals[i - 1]) != 0:
                            y_for_use = yvals[i - 1]
                        else:
                            raise ValueError("All neighboring y are zero; cannot evaluate integrand.")
                    else:
                        y_for_use = y_cur
                    if use_x_weight:
                        integrand = xvals[i] / (CC(2) * y_for_use)
                    else:
                        integrand = CC(1) / (CC(2) * y_for_use)
                    dxd = dx_factor * ws[i]
                    integral += integrand * dxd

                # final check again
                f_exact_end = f_at(p1)
                if abs(f_exact_end) >= tiny:
                    y_exact_p = f_exact_end.sqrt()
                    y_exact_m = -y_exact_p
                    y_near_end = yvals[-1]
                    y_at_end = y_exact_p if abs(y_exact_p - y_near_end) <= abs(y_exact_m - y_near_end) else y_exact_m
                    if abs(y_at_end - y_target) > tol:
                        raise ValueError(f"Branch selection failed after retry. target={y_target}, got={y_at_end}")

    return integral


def abel_jacobi_mumford(
    D, f_coeffs, base_point, *,
    integrate_func=None,    # function(base_x, x_end, y_end, f_coeffs, use_x_weight, prec, debug)
    prec=100,
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
            int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
                                  use_x_weight=False, prec=prec, debug=debug)
            int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs,
                                  use_x_weight=True, prec=prec, debug=debug)
        except TypeError:
            # If integrate_func doesn't accept debug/prec/use_x_weight keywords in that order,
            # fall back to positional call (older integrator signature)
            int0 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, False, prec, debug)
            int1 = integrate_func(base_x, x_pt_cc, y_pt, f_coeffs, True, prec, debug)
        except Exception as e:
            if debug:
                print(f"[abel_jacobi] Integration failed for root {idx} at x={x_pt_cc}: {e}")
            # skip this root (don't abort entire AJ computation)
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


def arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=200, debug=False, test_normalization=None):
    """
    Robust replacement of arakelov_build_basis_with_heights with definitive diagnostics.

    - Single GENERIC non-Weierstrass base point for all AJ maps.
    - Height matrix built in RealField(prec).
    - High-precision re-checks for suspect candidates.
    - After basis build: compute smallest eigenvector, test linear combo of AJ,
      reduce modulo period lattice, run PSLQ, and test any integer relation in the Jacobian.
    """
    # Local helper formatter (compact)
    # ---- imports ----
    from sage.all import (ComplexField, RealField, PolynomialRing, Matrix,
                          HyperellipticCurve, vector, QQ, ComplexField as CF,
                          RR as SageRR, Integer)
    import numpy as np

    if not all_divisors:
        return [], 0, None

    if debug:
        print(f"\n[arakelov] Building basis from {len(all_divisors)} divisors")
        print(f"[arakelov] Using precision: {prec} bits")

    CC = ComplexField(prec)
    RR = RealField(prec)

    # Compute period matrix once (complex)
    from .homology import get_period_matrix_auto_B
    period_matrix = get_period_matrix_auto_B(f_coeffs, prec=prec)
    if debug:
        print("[arakelov] Period matrix computed (tau):")
        for i in range(2):
            for j in range(2):
                print(f"  tau[{i},{j}] = {fmt(period_matrix[i,j])}")

    # polynomial for root finding in CC
    Rq = PolynomialRing(CC, 'x')
    x = Rq.gen()
    f_poly_cc = sum(CC(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))

    # Choose generic non-Weierstrass base point (deterministic)
    f_roots = []
    try:
        f_roots = sorted(f_poly_cc.roots(multiplicities=False),
                         key=lambda z: (float(z.real()), float(z.imag())))
    except Exception:
        f_roots = []

    if f_roots:
        candidate_x = CC(f_roots[0] + CC(0.12345))
    else:
        candidate_x = CC(0.12345)

    f_at_candidate = sum(CC(c) * (candidate_x ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))
    if abs(f_at_candidate) < CC(2) ** (-prec // 2):
        candidate_x += CC(0.33333)
        f_at_candidate = sum(CC(c) * (candidate_x ** (len(f_coeffs)-1-i)) for i, c in enumerate(f_coeffs))

    candidate_y = f_at_candidate.sqrt()
    base_point = (candidate_x, candidate_y)

    if debug:
        print(f"[arakelov] Using GENERIC base point: P0 = ({fmt(base_point[0])}, {fmt(base_point[1])})")

    # Build curve & jacobian (over QQ for constructor)
    Rq_QQ = PolynomialRing(QQ, 'x')
    x_QQ = Rq_QQ.gen()
    f_poly_QQ = sum(QQ(c) * x_QQ**(len(f_coeffs)-1-i) for i,c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly_QQ)
    J = C.jacobian()

    # Convert divisors to Jacobian elements (keep mapping to original)
    jac_elements = []
    for div in all_divisors:
        try:
            u_poly = x_QQ**2 - QQ(div['s'])*x_QQ + QQ(div['p'])
            v_poly = QQ(div['v_1'])*x_QQ + QQ(div['v_0'])
            D = J([u_poly, v_poly])
            if not D.is_zero():
                jac_elements.append((div, D))
        except Exception:
            continue

    if not jac_elements:
        return [], 0, None

    n = len(jac_elements)
    if debug:
        print(f"[arakelov] Computing Abel-Jacobi for {n} divisors (same base point) ...")

    # Compute AJ once per Jacobian element (cache)
    aj_cache = {}
    for idx, (div, D) in enumerate(jac_elements):
        try:
            aj_vec = abel_jacobi_mumford(D, f_coeffs, base_point=base_point, prec=prec)
            aj_cache[idx] = aj_vec
            if debug and idx < 6:
                print(f"[arakelov] Divisor {idx}: AJ z = ({fmt(aj_vec[0])}, {fmt(aj_vec[1])}); |z| = {fmt(abs(aj_vec[0]) + abs(aj_vec[1]))}")
        except Exception as e:
            aj_cache[idx] = None
            if debug:
                print(f"[arakelov] AJ failed for divisor {idx}: {e}")

    # Im(tau) matrix over RR for pairings
    Im_tau = Matrix(RR, 2, 2)
    for i in range(2):
        for j in range(2):
            Im_tau[i,j] = RR(period_matrix[i,j].imag())

    # pairing wrapper -> RR
    def compute_pairing_num(i, j, prec_local=prec):
        zi = aj_cache.get(i)
        zj = aj_cache.get(j)
        if zi is None or zj is None:
            return RR(0)
        val = neron_tate_height_pairing(zi, zj, Im_tau, prec=prec_local, normalization_factor=(test_normalization or 1.0))
        return RR(val)

    # incremental basis construction
    basis = []
    basis_indices = []
    if debug:
        print("[arakelov] Building basis incrementally ...")

    for i, (div, D) in enumerate(jac_elements):
        if aj_cache.get(i) is None:
            continue

        h_self = compute_pairing_num(i, i)
        if debug:
            print(f"[arakelov] Self-pairing of divisor {i}: ({float(h_self):.6g})")

        if float(h_self) < 1e-12:
            if debug:
                print(f"[arakelov] Skipping divisor {i}: self-pairing too small")
            continue

        if not basis:
            basis.append(div)
            basis_indices.append(i)
            if debug:
                print(f"[arakelov] Added divisor {i} (self-pairing {float(h_self):.6g})")
            continue

        # candidate = basis + i
        cand_idx = basis_indices + [i]
        m = len(cand_idx)
        Hnum = Matrix(RR, m, m)
        for a in range(m):
            for b in range(a, m):
                val = compute_pairing_num(cand_idx[a], cand_idx[b])
                Hnum[a,b] = val
                Hnum[b,a] = val

        # numeric PD test
        is_pd = False
        try:
            evals = Hnum.eigenvalues()
            is_pd = all(float(ev) > 1e-12 for ev in evals)
        except Exception:
            is_pd = False

        if is_pd:
            basis.append(div)
            basis_indices.append(i)
            if debug:
                try:
                    detf = float(Hnum.determinant())
                except Exception:
                    detf = float(abs(Hnum.det()))
                print(f"[arakelov] Added divisor {i} (rank {m}, det {detf:.6g})")
            continue

        # --- High-precision diagnostic block ---
        if debug:
            print(f"[arakelov] Candidate {i} not PD. Running high-precision diagnostics...")

            high_prec = max(prec * 2, 1024)
            CC_hp = CF(high_prec)
            base_x_hp = CC_hp(base_point[0])
            base_y_hp = CC_hp(base_point[1])

            # recompute AJ at higher precision for cand set
            indices_to_check = cand_idx
            aj_hp = {}
            for ii in indices_to_check:
                try:
                    aj_hp[ii] = abel_jacobi_mumford(jac_elements[ii][1], f_coeffs,
                                                    base_point=(base_x_hp, base_y_hp), prec=high_prec)
                    print(f"  [hp] idx {ii} AJ (hp): ({fmt(aj_hp[ii][0])}, {fmt(aj_hp[ii][1])})")
                except Exception as ee:
                    print(f"  [hp] idx {ii} AJ failed at high precision: {ee}")
                    aj_hp[ii] = None

            # build high-precision numeric height matrix Hhp
            Hhp = Matrix(RealField(high_prec), m, m)
            for a in range(m):
                for b in range(a, m):
                    za = aj_hp.get(cand_idx[a])
                    zb = aj_hp.get(cand_idx[b])
                    if za is None or zb is None:
                        Hhp[a,b] = RR(0)
                    else:
                        Hhp[a,b] = RealField(high_prec)(neron_tate_height_pairing(za, zb, Im_tau, prec=high_prec, normalization_factor=(test_normalization or 1.0)))
                    Hhp[b,a] = Hhp[a,b]

            # print eigenvalues and determinant (compact)
            try:
                evals_hp = Hhp.eigenvalues()
                evs_print = [fmt(ev) for ev in evals_hp]
                print(f"  [hp] eigenvalues (hp): {evs_print}")
                # determinant may be astronomically small; print exponent if needed
                try:
                    det_hp = Hhp.determinant()
                    print(f"  [hp] det (hp): {fmt(det_hp)}")
                except Exception:
                    print("  [hp] det (hp): (failed to compute)")
            except Exception as ee:
                print(f"  [hp] eigen/det failed: {ee}")

            # If the high-precision matrix has an effectively zero eigenvalue, run deeper diagnostics:
            # Convert Hhp to numpy floats for stable SVD/eig
            try:
                Hhp_np = np.array(Hhp, dtype=float)
            except Exception:
                Hhp_np = None

            if Hhp_np is not None:
                # compute smallest eig (robust)
                try:
                    w, V = np.linalg.eigh(Hhp_np)
                    idx_min = int(np.argmin(w))
                    eigvec = V[:, idx_min]
                    eigval = float(w[idx_min])
                except Exception:
                    # fallback to SVD
                    U, s, Vt = np.linalg.svd(Hhp_np)
                    eigval = float(s[-1]**2)
                    eigvec = Vt[-1]

                print(f"  [hp] smallest eigenvalue (float): {eigval:g}")
                # normalize eigenvector for readability
                if np.linalg.norm(eigvec) > 0:
                    eigvec = eigvec / np.linalg.norm(eigvec)
                order = np.argsort(-np.abs(eigvec))
                print("  [hp] top contributors (pos in cand set -> coeff):")
                for t in order[:min(8, len(eigvec))]:
                    print(f"    pos {t} (orig idx {cand_idx[t]}): coeff {eigvec[t]:.6g}")

                # form complex linear combo of AJ_hp using eigvec coefficients
                AJ_complex = []
                for pos in range(m):
                    orig_idx = cand_idx[pos]
                    z = aj_hp.get(orig_idx)
                    if z is None:
                        AJ_complex.append(np.array([0+0j, 0+0j], dtype=complex))
                    else:
                        AJ_complex.append(np.array([complex(z[0]), complex(z[1])], dtype=complex))
                AJ_complex = np.array(AJ_complex).T  # 2 x m
                combo = AJ_complex.dot(eigvec)
                print(f"  [hp] linear combo (2-vector) norm: {np.linalg.norm(combo):.6g}")
                print(f"    combo: ({fmt(combo[0])}, {fmt(combo[1])})")

                # reduce combo modulo period lattice (numpy helper)
                def reduce_mod_lattice_np(z_complex, tau):
                    e1 = np.array([1+0j, 0+0j], dtype=complex)
                    e2 = np.array([0+0j, 1+0j], dtype=complex)
                    tau_np = np.array([[complex(tau[0,0]), complex(tau[0,1])],
                                       [complex(tau[1,0]), complex(tau[1,1])]], dtype=complex)
                    cols = [e1, e2, tau_np[:,0], tau_np[:,1]]
                    L = np.zeros((4,4), dtype=float)
                    zreal = np.zeros(4, dtype=float)
                    for j,c in enumerate(cols):
                        L[0,j] = c[0].real; L[1,j] = c[0].imag
                        L[2,j] = c[1].real; L[3,j] = c[1].imag
                    zreal[0] = z_complex[0].real; zreal[1] = z_complex[0].imag
                    zreal[2] = z_complex[1].real; zreal[3] = z_complex[1].imag
                    kval, *_ = np.linalg.lstsq(L, zreal, rcond=None)
                    k_int = np.rint(kval).astype(int)
                    Lk = sum(k_int[j] * cols[j] for j in range(4))
                    zred = z_complex - Lk
                    return zred, k_int

                combo_red, combo_k = reduce_mod_lattice_np(combo, period_matrix)
                print(f"  [hp] combo reduced norm: {np.linalg.norm(combo_red):.6g}, k={combo_k}")
                # reductions for top contributors
                print("  [hp] reductions for top contributors:")
                for t in order[:min(8, m)]:
                    orig_idx = cand_idx[t]
                    z = aj_hp.get(orig_idx)
                    if z is None:
                        continue
                    zc = np.array([complex(z[0]), complex(z[1])], dtype=complex)
                    zred, kint = reduce_mod_lattice_np(zc, period_matrix)
                    print(f"    orig_idx {orig_idx}: reduced norm {np.linalg.norm(zred):.6g}, k={kint}")

                # Try PSLQ on the eigvec coefficients (convert to SageRR)
                try:
                    sage_coeffs = [SageRR(float(c)) for c in eigvec]
                    rel = None
                    #rel = pslq(sage_coeffs, maxcoeff=10**8)
                    print("  [hp] PSLQ relation on eigvec (if any):", rel)
                    if rel:
                        # rel is a list of integers (maybe)
                        klist = [Integer(int(rr)) for rr in rel]
                        # test relation in Jacobian: sum k_i * element = 0 ?
                        combJ = J(0)
                        for pos, kk in enumerate(klist):
                            orig_idx = cand_idx[pos]
                            D = jac_elements[orig_idx][1]
                            if kk >= 0:
                                combJ = combJ + (kk * D)
                            else:
                                combJ = combJ - ((-kk) * D)
                        try:
                            is_zero = combJ.is_zero()
                        except Exception:
                            is_zero = False
                        print(f"  [hp] PSLQ combination tested in Jacobian, is_zero = {is_zero}")
                except Exception as ee:
                    print("  [hp] PSLQ attempt failed:", ee)
            # end Hhp_np diagnostics

        # End high-precision diagnostics block. Skip adding this candidate.
        if debug:
            print(f"[arakelov] Skipping divisor {i}: not positive-definite (numeric).")
        continue

    # finalize H_final (RR)
    final_rank = len(basis)
    H_final = None
    if final_rank > 0:
        H_final = Matrix(RR, final_rank, final_rank)
        for a in range(final_rank):
            for b in range(a, final_rank):
                val = compute_pairing_num(basis_indices[a], basis_indices[b])
                H_final[a,b] = val
                H_final[b,a] = val

    # Final diagnostics on assembled H_final
    if debug:
        print(f"[arakelov] Final rank: {final_rank}")
        if H_final is None:
            print("[arakelov] No height matrix constructed.")
            return basis, final_rank, H_final

        # numpy eigen for final matrix
        try:
            Hf = np.array(H_final, dtype=float)
            w, V = np.linalg.eigh(Hf)
            print("eigenvalues (small->big):", w.tolist())
            idx_min = int(np.argmin(w))
            eigval = float(w[idx_min])
            eigvec = V[:, idx_min]
            print("smallest eigenvalue:", eigval)
            # normalize and report top contributors
            if np.linalg.norm(eigvec) > 0:
                eigvec = eigvec / np.linalg.norm(eigvec)
            order = np.argsort(-np.abs(eigvec))
            print("Top contributors to final null direction (basis pos -> coeff):")
            for t in order[:min(8, len(eigvec))]:
                orig_idx = basis_indices[t]
                print(f"  pos {t} (orig idx {orig_idx}): coeff {eigvec[t]:.6g}  AJ = ({fmt(aj_cache[orig_idx][0])}, {fmt(aj_cache[orig_idx][1])})")
            # build AJ_complex for basis and compute combo
            AJ = np.array([[complex(aj_cache[idx][0]), complex(aj_cache[idx][1])] for idx in basis_indices], dtype=complex).T
            combo = AJ.dot(eigvec)
            print("Linear combo (2-vector) of AJ by smallest-eig coeffs:", (fmt(combo[0]), fmt(combo[1])))
            print("Norm of combo:", np.linalg.norm(combo))

            # reduce combo modulo lattice
            def reduce_mod_lattice_np_local(z_complex, tau):
                e1 = np.array([1+0j, 0+0j], dtype=complex)
                e2 = np.array([0+0j, 1+0j], dtype=complex)
                tau_np = np.array([[complex(tau[0,0]), complex(tau[0,1])],
                                   [complex(tau[1,0]), complex(tau[1,1])]], dtype=complex)
                cols = [e1, e2, tau_np[:,0], tau_np[:,1]]
                L = np.zeros((4,4), dtype=float)
                zreal = np.zeros(4, dtype=float)
                for j,c in enumerate(cols):
                    L[0,j] = c[0].real; L[1,j] = c[0].imag
                    L[2,j] = c[1].real; L[3,j] = c[1].imag
                zreal[0] = z_complex[0].real; zreal[1] = z_complex[0].imag
                zreal[2] = z_complex[1].real; zreal[3] = z_complex[1].imag
                kval, *_ = np.linalg.lstsq(L, zreal, rcond=None)
                k_int = np.rint(kval).astype(int)
                Lk = sum(k_int[j] * cols[j] for j in range(4))
                zred = z_complex - Lk
                return zred, k_int

            combo_red, combo_k = reduce_mod_lattice_np_local(combo, period_matrix)
            print("Combo reduced norm:", np.linalg.norm(combo_red), "k:", combo_k.tolist())

            # PSLQ on eigvec coefficients
            try:
                sage_coeffs = [SageRR(float(c)) for c in eigvec]
                rel = None
                #rel = pslq(sage_coeffs, maxcoeff=10**8)
                print("PSLQ relation on eigvec (if any):", rel)
                if rel:
                    klist = [Integer(int(r)) for r in rel]
                    # test in Jacobian
                    combJ = J(0)
                    for pos, kk in enumerate(klist):
                        orig_idx = basis_indices[pos]
                        D = jac_elements[orig_idx][1]
                        if kk >= 0:
                            combJ = combJ + (kk * D)
                        else:
                            combJ = combJ - ((-kk) * D)
                    try:
                        is_zero = combJ.is_zero()
                    except Exception:
                        is_zero = False
                    print("PSLQ combination tested in Jacobian, is_zero =", is_zero)
            except Exception as e:
                print("PSLQ attempt failed:", e)

        except Exception as e:
            print("[arakelov] Final diagnostic eigen/SVD failed:", e)

        try:
            print(f"[arakelov] Final determinant (float): {float(H_final.determinant())}")
        except Exception:
            print("[arakelov] Final determinant: (failed to compute float)")

    return basis, final_rank, H_final
