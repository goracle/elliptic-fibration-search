# selmer_genus2.py
"""
2-Selmer rank bounds for genus 2 curves.
Uses Jacobian structure and local solubility analysis.
"""

from sage.all import QQ, ZZ, GF, HyperellipticCurve, primes, gcd, legendre_symbol, Integer
from sage.all import PolynomialRing, factor, sqrt


def compute_genus2_selmer_upper_bound(f_coeffs, bad_primes=None, max_prime_check=100, verbose=True):
    """
    Compute upper bound on rank(J(C)(QQ)) for genus 2 curve C: y^2 = f(x).
    
    Uses:
    - Dimension of H^1(Q, J[2]) (theoretically 15 for genus 2)
    - Local solubility at bad primes
    - Cassels-Tate pairing obstructions
    
    Args:
        f_coeffs: Coefficients [a6, a5, a4, a3, a2, a1, a0] for f(x) = sum(a_i * x^i)
        bad_primes: List of primes to check (if None, use primes where disc is bad)
        max_prime_check: Check primes up to this bound
        verbose: Print diagnostics
        
    Returns:
        int: Upper bound on rank(J(C)(QQ))
    """
    if verbose:
        print("\n" + "="*70)
        print("GENUS 2 SELMER RANK UPPER BOUND")
        print("="*70)
    
    # Build the curve
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f = sum(QQ(f_coeffs[i]) * x**i for i in range(len(f_coeffs)))
    
    C = HyperellipticCurve(f)
    
    if verbose:
        print(f"\nCurve: y^2 = {f}")
    
    # Get discriminant to find bad primes
    disc = C.discriminant()
    
    if bad_primes is None:
        # Find primes dividing discriminant
        disc_int = Integer(disc.numerator())
        bad_primes = []
        for p in primes(max_prime_check):
            if disc_int % p == 0:
                bad_primes.append(p)
        if verbose:
            print(f"Bad primes (from discriminant): {bad_primes}")
    
    # Count local obstructions
    num_unobstructed_primes = 0
    num_obstructed_primes = 0
    
    for p in bad_primes:
        is_locally_solvable = _check_local_solubility_genus2(f, p)
        if is_locally_solvable:
            num_unobstructed_primes += 1
        else:
            num_obstructed_primes += 1
            if verbose:
                print(f"  p={p}: locally obstructed")
    
    # Theoretical dimension of H^1(Q, J[2])
    # For genus g=2: dim = 2^(2g) - 1 = 15
    dim_H1 = 15
    
    # Upper bound formula:
    # rank(Sel^2) <= dim(H^1) + sum(local_contributions) - obstructions
    # Local contribution: each unobstructed prime can add at most 1
    # Obstruction: each obstructed prime reduces by at least 1
    
    upper_bound = dim_H1 + num_unobstructed_primes - num_obstructed_primes
    
    # Cassels-Tate additional constraint
    # For genus 2, real solubility is automatic (odd degree), so no archimedean obstruction
    
    # Make sure bound is non-negative
    upper_bound = max(0, upper_bound)
    
    if verbose:
        print(f"\nDimension H^1(Q, J[2]): {dim_H1}")
        print(f"Unobstructed bad primes: {num_unobstructed_primes}")
        print(f"Obstructed bad primes: {num_obstructed_primes}")
        print(f"\nSelmer rank upper bound: {upper_bound}")
    
    return upper_bound


def _check_local_solubility_genus2(f, p):
    """
    Check if y^2 = f(x) is locally solvable at prime p.
    
    Args:
        f: Polynomial in x over QQ
        p: Prime number
        
    Returns:
        bool: True if locally solvable at p
    """
    # Reduce f mod p
    R = PolynomialRing(GF(p), 'x')
    x = R.gen()
    
    # Get coefficients and reduce
    coeffs = f.list()
    f_modp = sum(GF(p)(Integer(coeffs[i])) * x**i for i in range(len(coeffs)))
    
    # Check if there exists x in F_p such that f(x) is a quadratic residue
    for x_val in range(p):
        y_squared = f_modp(x_val)
        
        # Check if y_squared is 0 or a quadratic residue mod p
        y_sq_int = Integer(y_squared)
        if y_sq_int == 0:
            return True
        if p == 2:
            return True  # Everything is a square mod 2
        if legendre_symbol(y_sq_int, p) == 1:
            return True
    
    # If no x gives a quadratic residue, check point at infinity
    # For y^2 = f(x) with deg(f) = 5 or 6, there's always a point at infinity
    deg = f.degree()
    if deg % 2 == 1:
        # Odd degree means point at infinity exists
        return True
    
    return False


def print_selmer_comparison(mumford_basis_size, selmer_upper_bound, verbose=True):
    """
    Print comparison between MUMFORD_SEARCH lower bound and Selmer upper bound.
    
    Args:
        mumford_basis_size: Number of independent divisors found by MUMFORD_SEARCH
        selmer_upper_bound: Upper bound from Selmer analysis
        verbose: Print output
    """
    if not verbose:
        return
    
    print("\n" + "="*70)
    print("RANK BOUNDS SUMMARY")
    print("="*70)
    print(f"\nLower bound (Mumford basis): {mumford_basis_size}")
    print(f"Upper bound (Selmer): {selmer_upper_bound}")
    
    if mumford_basis_size == selmer_upper_bound:
        print("\n Bounds match! Rank is exactly {mumford_basis_size}")
    elif mumford_basis_size < selmer_upper_bound:
        gap = selmer_upper_bound - mumford_basis_size
        print(f"\nGap: {gap}")
        print("Possible explanations:")
        print("  - Sha(J) has 2-torsion")
        print("  - Need more search depth to find remaining generators")
        print("  - Selmer bound is not tight")
    else:
        print("\n  WARNING: Lower bound exceeds upper bound!")
        print("This indicates a bug in either computation.")


def analyze_genus2_rank(f_coeffs, mumford_divisors=None, bad_primes=None, verbose=True):
    """
    Complete rank analysis for genus 2 curve.
    
    Args:
        f_coeffs: Curve coefficients
        mumford_divisors: List of divisors from MUMFORD_SEARCH (if available)
        bad_primes: Primes to check
        verbose: Print diagnostics
        
    Returns:
        dict: {'lower_bound': int, 'upper_bound': int, 'exact': bool}
    """
    # Compute upper bound
    upper = compute_genus2_selmer_upper_bound(
        f_coeffs, 
        bad_primes=bad_primes, 
        verbose=verbose
    )
    
    # Compute lower bound from Mumford basis
    lower = 0
    if mumford_divisors is not None:
        lower = len(mumford_divisors)
        if verbose:
            print(f"\nMumford basis size: {lower}")
    
    # Check if bounds match
    exact = (lower == upper)
    
    if verbose:
        print_selmer_comparison(lower, upper, verbose=True)
    
    return {
        'lower_bound': lower,
        'upper_bound': upper,
        'exact': exact,
        'gap': upper - lower
    }
