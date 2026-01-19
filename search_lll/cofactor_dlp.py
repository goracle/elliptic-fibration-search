from sage.all import Integer, factor, gcd, crt
import math
import sys

def solve_cofactor_dlp_bsgs(target_residue, generator_residue, G, Q, ell, h, 
                            full_order, verbose=True):
    """
    Solve the cofactor DLP to lift from mod ell to the full discrete log.
    
    Given:
        - d_ell ≡ dlog (mod ell), where d_ell is target_residue
        - Full order = ell * h
        - Q = d_full * G where d_full = d_ell + k*ell for some k ∈ [0, h)
    
    Find k using baby-step giant-step on the cofactor subgroup.
    
    Args:
        target_residue: discrete log modulo ell (the 800 from your solve)
        generator_residue: unused (kept for API compatibility)
        G: generator (Jacobian element)
        Q: target (Jacobian element)
        ell: large prime factor
        h: cofactor
        full_order: ell * h
        verbose: print progress
    
    Returns:
        full_dlog: the complete discrete logarithm
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"COFACTOR DLP LIFTING (Baby-Step Giant-Step)")
        print(f"{'='*70}")
        print(f"Prime ℓ: {ell}")
        print(f"Cofactor h: {h}")
        print(f"Known: d ≡ {target_residue} (mod {ell})")
        print(f"Goal: Find k ∈ [0, {h}) such that d_full = {target_residue} + k*{ell}")
        sys.stdout.flush()
    
    # Compute the residue: R = Q - d_ell * G
    d_ell = Integer(target_residue)
    R = Q - d_ell * G
    
    if R.is_zero():
        if verbose:
            print(f"  [Lift] R is zero - d_ell is already the full discrete log!")
            sys.stdout.flush()
        return int(d_ell)
    
    # The cofactor generator: H = ell * G
    H = Integer(ell) * G
    
    if H.is_zero():
        raise RuntimeError("ell * G is zero - cannot lift (degenerate case)")
    
    if verbose:
        print(f"  [Lift] Residue R = Q - {d_ell}*G is non-zero")
        print(f"  [Lift] Computing H = {ell}*G (cofactor generator)")
        print(f"  [Lift] Solving: R = k*H in subgroup of order {h}")
        sys.stdout.flush()
    
    # Baby-step giant-step
    h_int = int(h)
    m = int(math.ceil(math.sqrt(h_int)))
    
    if verbose:
        print(f"  [BSGS] Using m = {m} (sqrt of cofactor size)")
        print(f"  [BSGS] Building baby-step table...")
        sys.stdout.flush()
    
    # Baby steps: store j*H for j in [0, m)
    baby = {}
    current = G.parent()(0)  # identity
    
    for j in range(m):
        key = str(current)
        if key not in baby:
            baby[key] = j
        current = current + H
        
        if verbose and j > 0 and j % 10000 == 0:
            print(f"    [BSGS] Baby steps: {j}/{m}")
            sys.stdout.flush()
    
    if verbose:
        print(f"  [BSGS] Baby-step table complete: {len(baby)} entries")
        print(f"  [BSGS] Starting giant steps...")
        sys.stdout.flush()
    
    # Giant steps: check R - i*(m*H) for i in [0, m]
    giant_step = Integer(m) * H
    current = R
    
    for i in range(m + 1):
        key = str(current)
        if key in baby:
            j = baby[key]
            k = i * m + j
            
            if k < h_int:
                # Found it! Compute full discrete log
                d_full = int(d_ell + Integer(k) * Integer(ell))
                
                # Verify
                if verbose:
                    print(f"\n  [BSGS] Found k = {k} at giant step i={i}, baby step j={j}")
                    print(f"  [Verify] Testing: d_full = {target_residue} + {k}*{ell} = {d_full}")
                    sys.stdout.flush()
                
                verification = Integer(d_full) * G
                if verification == Q:
                    if verbose:
                        print(f"  [Verify] ✓ EXACT MATCH: {d_full} * G == Q")
                        print(f"\n{'='*70}")
                        print(f"SUCCESS: Full discrete logarithm = {d_full}")
                        print(f"{'='*70}\n")
                        sys.stdout.flush()
                    return d_full
                else:
                    if verbose:
                        print(f"  [Verify] Mismatch at k={k}, continuing search...")
                        sys.stdout.flush()
        
        current = current - giant_step
        
        if verbose and i > 0 and i % 100 == 0:
            print(f"    [BSGS] Giant steps: {i}/{m}")
            sys.stdout.flush()
    
    raise RuntimeError(f"BSGS failed to find discrete log in cofactor range [0, {h})")


def solve_cofactor_dlp_crt(target_residue, generator_residue, G, Q, ell, h,
                           full_order, verbose=True):
    """
    Solve cofactor DLP using CRT decomposition.
    
    Factor h and solve smaller DLPs, then combine with Chinese Remainder Theorem.
    This is typically faster when h has small prime factors.
    
    For h = 4 * 289603, we solve:
        - DLP mod 4 (trivial)
        - DLP mod 289603 (using BSGS)
    Then lift using CRT.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"COFACTOR DLP LIFTING (CRT Decomposition)")
        print(f"{'='*70}")
        print(f"Prime ℓ: {ell}")
        print(f"Cofactor h: {h}")
        sys.stdout.flush()
    
    # Factor the cofactor
    h_factors = list(factor(h))
    
    if verbose:
        print(f"  [Factor] h = {h} = {' * '.join(f'{p}^{e}' for p, e in h_factors)}")
        sys.stdout.flush()
    
    # Setup
    d_ell = Integer(target_residue)
    R = Q - d_ell * G
    
    if R.is_zero():
        if verbose:
            print(f"  [Lift] R is zero - d_ell is already the full discrete log!")
            sys.stdout.flush()
        return int(d_ell)
    
    H = Integer(ell) * G
    
    # Solve DLP mod each prime power
    remainders = []
    moduli = []
    
    for p, e in h_factors:
        modulus = int(p ** e)
        moduli.append(modulus)
        
        if verbose:
            print(f"\n  [CRT] Solving DLP mod {p}^{e} = {modulus}...")
            sys.stdout.flush()
        
        # Scale to subgroup: (h/modulus) * R vs (h/modulus) * H
        scale = int(h // modulus)
        R_scaled = Integer(scale) * R
        H_scaled = Integer(scale) * H
        
        # BSGS in this subgroup
        k_mod = _bsgs_small(R_scaled, H_scaled, modulus, verbose=verbose)
        
        if k_mod is None:
            raise RuntimeError(f"Failed to solve DLP mod {modulus}")
        
        remainders.append(k_mod)
        
        if verbose:
            print(f"    [CRT] Found k ≡ {k_mod} (mod {modulus})")
            sys.stdout.flush()
    
    # Combine using CRT
    if verbose:
        print(f"\n  [CRT] Combining remainders using Chinese Remainder Theorem...")
        sys.stdout.flush()
    
    k = int(crt(remainders, moduli))
    
    # Compute full discrete log
    d_full = int(d_ell + Integer(k) * Integer(ell))
    
    # Verify
    if verbose:
        print(f"  [CRT] Combined: k = {k}")
        print(f"  [Verify] Testing: d_full = {target_residue} + {k}*{ell} = {d_full}")
        sys.stdout.flush()
    
    verification = Integer(d_full) * G
    if verification == Q:
        if verbose:
            print(f"  [Verify] ✓ EXACT MATCH: {d_full} * G == Q")
            print(f"\n{'='*70}")
            print(f"SUCCESS: Full discrete logarithm = {d_full}")
            print(f"{'='*70}\n")
            sys.stdout.flush()
        return d_full
    else:
        raise RuntimeError(f"CRT reconstruction failed verification")


def _bsgs_small(R, H, order, verbose=False):
    """
    Baby-step giant-step for small subgroup.
    Solve R = k*H where k ∈ [0, order).
    """
    m = int(math.ceil(math.sqrt(order)))
    
    # Baby steps
    baby = {}
    current = H.parent()(0)
    
    for j in range(m):
        key = str(current)
        if key not in baby:
            baby[key] = j
        current = current + H
    
    # Giant steps
    giant_step = Integer(m) * H
    current = R
    
    for i in range(m + 1):
        key = str(current)
        if key in baby:
            j = baby[key]
            k = i * m + j
            if k < order:
                return k
        current = current - giant_step
    
    return None


# Main entry point - choose best strategy
def solve_cofactor_dlp(target_residue, generator_residue, G, Q, ell, h,
                       full_order, method='auto', verbose=True):
    """
    Master function - automatically choose best cofactor lifting strategy.
    
    Args:
        method: 'auto', 'bsgs', or 'crt'
    """
    if method == 'auto':
        # Choose based on factorization
        h_factors = list(factor(h))
        largest_factor = max(int(p**e) for p, e in h_factors)
        
        # If largest factor < sqrt(h), CRT is faster
        # Otherwise, direct BSGS is simpler
        if largest_factor < math.sqrt(int(h)):
            if verbose:
                print(f"  [Auto] Choosing CRT method (largest factor {largest_factor} < sqrt({h}))")
            method = 'crt'
        else:
            if verbose:
                print(f"  [Auto] Choosing direct BSGS (largest factor {largest_factor})")
            method = 'bsgs'
    
    if method == 'crt':
        return solve_cofactor_dlp_crt(
            target_residue, generator_residue, G, Q, ell, h, full_order, verbose
        )
    else:
        return solve_cofactor_dlp_bsgs(
            target_residue, generator_residue, G, Q, ell, h, full_order, verbose
        )
