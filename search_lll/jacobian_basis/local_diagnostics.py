import math
from sage.all import QQ, Qp, PolynomialRing, HyperellipticCurve

"""Diagnostic tools for debugging local height computations."""

def diagnose_local_height(div, p, f_coeffs, num_doublings=10, padic_prec=1024):
    """
    Detailed diagnostic of local height computation at prime p.

    Returns a dict with:
    - valuation_sequence: list of (k, h_k, s_k) for each doubling
    - convergence_info: statistics about convergence
    - potential_issues: list of warnings
    """
    issues = []

    # Setup
    K = Qp(p, prec=padic_prec)
    R = PolynomialRing(K, 'x')
    x = R.gen()
    f_poly = sum(K(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C_p = HyperellipticCurve(f_poly)
    J_p = C_p.jacobian()

    # Reconstruct divisor
    try:
        if hasattr(div, 'mumford_representation'):
            u_in, v_in = div.mumford_representation()
        else:
            u_in, v_in = div[0], div[1]

        u_p = R([K(c) for c in u_in.list()])
        v_p = R([K(c) for c in v_in.list()])
    except Exception as e:
        raise RuntimeError(f"Cannot reconstruct divisor at p={p}: {e}")

    P = J_p([u_p, v_p])

    # Compute naive height
    def naive_height_at_p(div_p, p_val):
        """Compute -min(v_p(u coeffs)) * log(p)"""
        u_poly = div_p[0]
        coeffs = u_poly.list()

        vals = []
        for c in coeffs:
            if c == 0:
                continue
            try:
                # For p-adic elements, valuation() has no args
                val = c.valuation()
            except TypeError:
                # For rationals, need to pass p
                val = QQ(c).valuation(p_val)
            vals.append(val)

        if not vals:
            return 0.0

        min_val = min(vals)
        return -float(min_val) * math.log(p_val)

    h0 = naive_height_at_p(P, p)

    # Doubling sequence
    sequence = []
    current_P = P

    for k in range(num_doublings + 1):
        if current_P.is_zero():
            issues.append(f"TORSION DETECTED at step {k}")
            break

        h_k = naive_height_at_p(current_P, p)
        s_k = (4.0 ** (-k)) * float(h_k)

        # Extract u-polynomial for inspection
        u_k = current_P[0]
        u_coeffs = [QQ(c) for c in u_k.list()]
        u_vals = [QQ(c).valuation(p) if c != 0 else float('inf')
                  for c in u_coeffs]

        sequence.append({
            'k': k,
            'h_k': h_k,
            's_k': s_k,
            'u_coeffs': u_coeffs,
            'u_valuations': u_vals,
            'min_valuation': min(v for v in u_vals if not math.isinf(v))
                            if any(not math.isinf(v) for v in u_vals) else None
        })

        if k < num_doublings:
            current_P = 2 * current_P

    # Analyze convergence
    s_values = [step['s_k'] for step in sequence]

    if len(s_values) >= 5:
        tail = s_values[-5:]
        mean = sum(tail) / len(tail)
        variance = sum((x - mean)**2 for x in tail) / len(tail)
        std = math.sqrt(variance)

        # Check for monotonicity issues
        diffs = [s_values[i+1] - s_values[i] for i in range(len(s_values)-1)]

        convergence = {
            'tail_mean': mean,
            'tail_std': std,
            'relative_std': std / abs(mean) if abs(mean) > 1e-9 else float('inf'),
            'correction': mean - h0,
            'differences': diffs,
            'is_monotonic': all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)
        }

        # Check for warning signs
        if convergence['relative_std'] > 0.01:
            issues.append(f"Poor convergence: rel_std = {convergence['relative_std']:.2e}")

        if not convergence['is_monotonic']:
            issues.append("Non-monotonic sequence - may indicate precision issues")

        if abs(convergence['correction']) < 1e-6:
            issues.append("Correction ≈ 0 - divisor may be in good reduction")
    else:
        convergence = {
            'tail_mean': s_values[-1] if s_values else 0,
            'correction': s_values[-1] - h0 if s_values else 0
        }

    return {
        'p': p,
        'h_naive': h0,
        'sequence': sequence,
        'convergence': convergence,
        'issues': issues
    }

def compare_divisor_local_heights(divs, f_coeffs, primes=None):
    """
    Compare local heights across multiple divisors to detect relations.

    Args:
        divs: list of Sage Jacobian elements
        f_coeffs: polynomial coefficients
        primes: list of primes to check (default: bad primes)
    """
    if primes is None:
        from local import get_bad_primes
        primes = get_bad_primes(f_coeffs)

    print(f"\n{'='*60}")
    print(f"LOCAL HEIGHT COMPARISON")
    print(f"{'='*60}\n")

    results = {}

    for i, div in enumerate(divs):
        print(f"Divisor {i}:")
        results[i] = {}

        for p in primes:
            diag = diagnose_local_height(div, p, f_coeffs, num_doublings=15)
            results[i][p] = diag

            correction = diag['convergence']['correction']
            rel_std = diag['convergence'].get('relative_std', 0)

            status = "✓" if not diag['issues'] else "⚠"
            print(f"  p={p:3d}: correction = {correction:+.6f}  "
                  f"(rel_std={rel_std:.2e}) {status}")

            if diag['issues']:
                for issue in diag['issues']:
                    print(f"         WARNING: {issue}")
        print()

    # Look for linear relations in corrections
    print(f"\n{'='*60}")
    print(f"CHECKING FOR LINEAR RELATIONS")
    print(f"{'='*60}\n")

    for p in primes:
        corrections = [results[i][p]['convergence']['correction'] for i in range(len(divs))]
        print(f"p={p}: corrections = {corrections}")

        # Check if any divisor's correction is a linear combination
        for target_idx in range(len(divs)):
            target = corrections[target_idx]
            others = [corrections[j] for j in range(len(divs)) if j != target_idx]

            # Simple check: is target ≈ sum of others?
            if len(others) >= 2 and abs(target - sum(others)) < 1e-5:
                print(f"  ⚠ Divisor {target_idx} ≈ sum of others at p={p}")

    return results

def debug_valuation_computation(div, p, padic_prec=1024):
    """
    Step-by-step debugging of valuation computation for a divisor.
    """
    print(f"\n{'='*60}")
    print(f"VALUATION DEBUG: p={p}")
    print(f"{'='*60}\n")

    # Get u polynomial
    if hasattr(div, 'mumford_representation'):
        u_poly, v_poly = div.mumford_representation()
    else:
        u_poly, v_poly = div[0], div[1]

    print(f"u(x) = {u_poly}")
    print(f"v(x) = {v_poly}\n")

    # Check each coefficient
    coeffs = u_poly.list()
    print(f"Coefficients (low to high degree):")

    for i, c in enumerate(coeffs):
        if c == 0:
            print(f"  x^{i}: 0 (valuation = ∞)")
            continue

        c_qq = QQ(c)
        val = c_qq.valuation(p)

        print(f"  x^{i}: {c_qq} = {c_qq.numerator()}/{c_qq.denominator()}")
        print(f"        v_{p}(num) = {QQ(c_qq.numerator()).valuation(p)}")
        print(f"        v_{p}(den) = {QQ(c_qq.denominator()).valuation(p)}")
        print(f"        v_{p}(coeff) = {val}")

    min_val = min(QQ(c).valuation(p) for c in coeffs if c != 0)
    naive_ht = -min_val * math.log(p)

    print(f"\nMinimum valuation: {min_val}")
    print(f"Naive height: {naive_ht:.6f}")

# Example usage template
def run_full_diagnostics(divs_list, f_coeffs):
    """
    Run complete diagnostic suite on a list of divisors.

    Args:
        divs_list: list of canonicalized dict divisors
        f_coeffs: polynomial coefficients
    """
    from heights import mumford_dict_to_jacobian_element

    # Convert to Sage elements
    sage_divs = [mumford_dict_to_jacobian_element(d, f_coeffs) for d in divs_list]

    # Run comparison
    results = compare_divisor_local_heights(sage_divs, f_coeffs)

    # Detailed debug for suspicious primes
    bad_primes = get_bad_primes(f_coeffs)

    print(f"\n{'='*60}")
    print(f"DETAILED VALUATION DEBUG")
    print(f"{'='*60}")

    for i, div in enumerate(sage_divs):
        print(f"\n### Divisor {i} ###")
        for p in bad_primes[:3]:  # First few primes
            debug_valuation_computation(div, p)

    return results
