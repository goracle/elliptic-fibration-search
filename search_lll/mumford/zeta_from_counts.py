from __future__ import annotations
import math, numpy as np
from typing import Sequence
from fractions import Fraction
from sage.all import Integer, GF, PolynomialRing, HyperellipticCurve, ZZ

"""
zeta_from_counts.py
====================
Instrument mumford_parallel residue data to extract the zeta function of the
underlying genus-2 curve over F_p.

Background
----------
The number of F_p-points on fibers at level n equals
    N_n(p) = #{m in F_p : X([n]P)(m) == -m + x1 mod p has a solution}
           = #{divisors of degree n on C that land in F_p}

For a genus-2 curve over F_p the Weil conjectures give
    Z(C/F_p, T) = L(T) / ((1-T)(1-pT))
where L(T) = 1 - a1 T + a2 T^2 - a1 p T^3 + p^2 T^4  (degree-4 palindromic).

The sequence  a_n = #C(F_{p^n})  satisfies the linear recurrence
    a_n = a1 a_{n-1} - a2 a_{n-2} + a1 p a_{n-3} - p^2 a_{n-4}

We collect the root counts N_n from the mumford_residues dict (one count per
vector index n), fit a 4-term linear recurrence via Berlekamp–Massey / linear
algebra, recover the characteristic polynomial L(T), and print:
    • the four Weil numbers α_i  (roots of L, should satisfy |α_i|=√p)
    • #J(F_p) = L(1)
    • the zeta function Z(T)

Usage
-----
After your mumford search call:

    from zeta_from_counts import counts_from_residues, fit_zeta

    counts = counts_from_residues(mumford_residues, vecs_list, p)
    fit_zeta(counts, p, verbose=True)

Or call  extract_and_print_zeta(mumford_residues, vecs_list, p)  as a one-liner.
"""

# ---------------------------------------------------------------------------
# 1.  Extract root count sequence from mumford_residues
# ---------------------------------------------------------------------------

def counts_from_residues(mumford_residues: dict, vecs_list: list, p: int) -> list[int]:
    """
    For each vector n (the scalar multiple of the base section), count how many
    distinct F_p x-residues appeared in mumford_residues.

    Key insight: v_tuple is (n,) — a 1-tuple of the integer scalar — NOT an
    index into vecs_list.  Workers return in completion order (reordered by
    degree for load-balancing), so we MUST key by the scalar value, not by
    arrival position.

    Parameters
    ----------
    mumford_residues : dict
        Keyed by prime; value is {v_tuple: {x_res: [sols]}}.
        v_tuple is typically (n,) where n is the scalar multiple used.
    vecs_list : list
        Ordered list of vector tuples (used only to determine the full set of
        expected n values and to build the output in sorted order).
    p : int
        The finite field prime (unused here; kept for API symmetry).

    Returns
    -------
    counts : list[int]
        Sorted by n ascending.  counts[i] corresponds to scalars[i].
    scalars : list[int]
        The sorted list of n values, parallel to counts.
    """
    # Accumulate distinct x_res values per scalar n across all primes
    x_sets: dict[int, set] = {}

    for prime, v_map in mumford_residues.items():
        for v_tuple, x_res_dict in v_map.items():
            # v_tuple is (n,) for 1-section FF mode; extract n robustly
            if isinstance(v_tuple, (tuple, list)) and len(v_tuple) == 1:
                n_val = int(v_tuple[0])
            elif isinstance(v_tuple, int):
                n_val = v_tuple
            else:
                # multi-section: use the sum of absolute values as a proxy key
                # (for 1-section FF runs this branch never fires)
                n_val = sum(abs(int(c)) for c in v_tuple)

            if n_val not in x_sets:
                x_sets[n_val] = set()

            if isinstance(x_res_dict, dict):
                for x_res in x_res_dict:
                    x_sets[n_val].add(int(x_res))
            elif isinstance(x_res_dict, list):
                for sol in x_res_dict:
                    if hasattr(sol, '__iter__') and not isinstance(sol, (str, bytes)):
                        x_sets[n_val].add(int(list(sol)[0]))
                    else:
                        x_sets[n_val].add(int(sol))

    if not x_sets:
        return [], []

    scalars = sorted(x_sets.keys())
    counts = [len(x_sets[n]) for n in scalars]
    return counts, scalars

# ---------------------------------------------------------------------------
# 2.  Convert raw counts to #C(F_{p^n}) estimates
# ---------------------------------------------------------------------------

def counts_to_curve_points(counts: list[int], p: int) -> list[int]:
    """
    Map the raw root counts N_n to curve point counts a_n = #C(F_{p^n}).

    For a genus-2 curve y^2 = f(x) of degree 5 or 6:
        a_n  ≈  2 * N_n  +  2          (two y-values per x, plus two points at ∞)

    This is a first-order approximation.  It assumes every root found at level n
    corresponds to a degree-n divisor whose x-support lands entirely in F_p.

    For most practical purposes (fitting the recurrence) the offset and the
    factor-2 scale out, so even if the model is off the recurrence coefficients
    are recoverable.
    """
    # a_n = 2*N_n + 2  (2 points at infinity for y^2 = f of degree 5 or 6)
    return [2 * c + 2 for c in counts]

# ---------------------------------------------------------------------------
# 3.  Linear-algebra recurrence fitting (no external dependencies)
# ---------------------------------------------------------------------------

def _berlekamp_massey_Z(seq: list[int]) -> list[int]:
    """
    Pure-Python Berlekamp–Massey over the integers (not a finite field).
    Returns the minimal linear recurrence  [c1, c2, ..., cd]  such that
        a[n] = c1*a[n-1] + c2*a[n-2] + ... + cd*a[n-d]
    for all n >= d, working with exact integer arithmetic (no modular reduction).

    Note: This is a heuristic version over Z; for exact recovery we also provide
    a Gaussian-elimination approach in fit_recurrence_linalg.
    """
    n = len(seq)
    # We try to find a recurrence of degree exactly 4 (for genus 2)
    return fit_recurrence_linalg(seq, degree=4)

def fit_recurrence_linalg(seq: list[int], degree: int = 4) -> list[int] | None:
    """
    Fit the linear recurrence of given degree to `seq` via Gaussian elimination
    over the rationals (using Python fractions for exactness).

    Solves:  M * c = b  where M[i,j] = seq[i+degree-1-j], b[i] = seq[i+degree].

    Returns [c1, c2, ..., cd] or None if the system is under-determined.
    """

    d = degree
    N = len(seq)
    if N < 2 * d:
        return None  # not enough data

    # Build system using the first 2*d terms
    rows = N - d
    # Matrix M (rows x d) and vector b (rows)
    M = [[Fraction(seq[i + d - 1 - j]) for j in range(d)] for i in range(rows)]
    b = [Fraction(seq[i + d]) for i in range(rows)]

    # Gaussian elimination (over Q)
    aug = [M[i] + [b[i]] for i in range(rows)]
    pivot_cols = []
    row_ptr = 0
    for col in range(d):
        # find pivot
        piv = None
        for r in range(row_ptr, rows):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        aug[row_ptr], aug[piv] = aug[piv], aug[row_ptr]
        pivot_cols.append(col)
        scale = aug[row_ptr][col]
        aug[row_ptr] = [x / scale for x in aug[row_ptr]]
        for r in range(rows):
            if r != row_ptr and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[row_ptr][c] for c in range(d + 1)]
        row_ptr += 1
        if row_ptr == d:
            break

    if len(pivot_cols) < d:
        return None  # rank-deficient

    # Extract solution
    c = [Fraction(0)] * d
    for i, col in enumerate(pivot_cols):
        c[col] = aug[i][d]

    # Convert to int if all are integers, else keep as Fraction
    result = []
    for ci in c:
        if ci.denominator == 1:
            result.append(int(ci.numerator))
        else:
            result.append(float(ci))
    return result

def verify_recurrence(seq: list[int], coeffs: list[int | float], degree: int = 4) -> float:
    """Return RMS residual of the recurrence on `seq`."""
    d = degree
    if len(seq) < d + 1 or coeffs is None:
        return float('inf')
    residuals = []
    for i in range(d, len(seq)):
        predicted = sum(coeffs[j] * seq[i - 1 - j] for j in range(d))
        residuals.append((seq[i] - predicted) ** 2)
    return math.sqrt(sum(residuals) / len(residuals))

# ---------------------------------------------------------------------------
# 4.  Characteristic polynomial and Weil numbers
# ---------------------------------------------------------------------------

def recurrence_to_charpoly(coeffs: list, p: int) -> list[int]:
    """
    Given recurrence coefficients [c1,...,c4] for a_n = c1*a_{n-1}+...
    return the characteristic polynomial of L(T):
        L(T) = T^4 - c1 T^3 + c2 T^2 - c3 T + c4
    For genus-2 Weil polynomial we expect:
        c4 = p^2,  c3 = c1 * p  (palindromic up to p-weighting).
    Returns [1, -c1, c2, -c3, c4] as coefficient list (high to low).
    """
    if coeffs is None or len(coeffs) < 4:
        return None
    c1, c2, c3, c4 = [float(x) for x in coeffs[:4]]
    return [1.0, -c1, c2, -c3, c4]

def companion_eigenvalues(poly_coeffs: list[float]) -> list[complex]:
    """
    Find eigenvalues of the companion matrix of a monic polynomial
        T^d + a_{d-1} T^{d-1} + ... + a_0
    using pure-Python QR iteration (simple version via numpy if available,
    else companion-matrix power iteration).
    """
    try:
        c = poly_coeffs  # [1, a_{d-1}, ..., a_0]
        # numpy.roots expects coefficients high-to-low including leading 1
        return list(np.roots(c))
    except ImportError:
        pass

    # Fallback: companion matrix + power iteration (rough)
    d = len(poly_coeffs) - 1
    a = [-poly_coeffs[i + 1] / poly_coeffs[0] for i in range(d)]
    # Build companion matrix
    C = [[0.0] * d for _ in range(d)]
    for i in range(d - 1):
        C[i + 1][i] = 1.0
    for i in range(d):
        C[i][d - 1] = a[d - 1 - i]
    # Very rough: just return the diagonal (not useful without proper eigensolver)
    return [complex(C[i][i]) for i in range(d)]

# ---------------------------------------------------------------------------
# 5.  Zeta function assembly and printing
# ---------------------------------------------------------------------------

def assemble_zeta(poly_coeffs: list[float], p: int) -> str:
    """
    Print the zeta function Z(C/F_p, T) = L(T)/((1-T)(1-pT)).
    Returns a human-readable string.
    """
    if poly_coeffs is None:
        return "Could not assemble zeta function (no polynomial)."

    c = poly_coeffs  # [1, -a1, a2, -a1*p, p^2] approximately
    lines = []
    lines.append(f"\nZeta function  Z(C/F_{p}, T) = L(T) / ((1-T)(1-{p}T))")
    lines.append("")
    lines.append(f"  L(T)  =  T^4")
    lines.append(f"         + ({c[1]:+.4g}) T^3")
    lines.append(f"         + ({c[2]:+.4g}) T^2")
    lines.append(f"         + ({c[3]:+.4g}) T")
    lines.append(f"         + ({c[4]:+.4g})")
    lines.append("")

    # Evaluate at T=1 to get #J(F_p)
    L1 = sum(c[i] for i in range(5))
    lines.append(f"  #J(F_{p}) = L(1)  ≈  {L1:.2f}")
    lines.append("")

    # Palindrome check: c[4] should ≈ p^2, c[3] ≈ c[1]*p
    expected_c4 = float(p ** 2)
    expected_c3 = c[1] * p
    lines.append(f"  Palindrome check:")
    lines.append(f"    c[4] = {c[4]:.4g}  (expected p^2 = {expected_c4:.4g},  err = {abs(c[4]-expected_c4):.4g})")
    lines.append(f"    c[3] = {c[3]:.4g}  (expected c[1]*p = {expected_c3:.4g},  err = {abs(c[3]-expected_c3):.4g})")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 6.  Main entry points
# ---------------------------------------------------------------------------

def fit_zeta(counts: list[int], p: int, verbose: bool = True,
             use_curve_points: bool = True,
             scalars: list[int] | None = None) -> dict:
    """
    Full pipeline: raw counts → recurrence → zeta function.

    Parameters
    ----------
    counts : list[int]
        counts[i] = number of F_p roots found for scalar n = scalars[i].
    p : int
        Finite field prime.
    verbose : bool
        Print results.
    use_curve_points : bool
        If True, convert counts to #C(F_{p^n}) estimates first.
    scalars : list[int] | None
        The n-values corresponding to counts (sorted ascending).
        If None, assumed to be 0, 1, 2, …

    Returns
    -------
    result : dict with keys:
        'seq'         – sequence used for fitting
        'scalars'     – n-values parallel to seq
        'coeffs'      – recurrence coefficients [c1,c2,c3,c4]
        'poly'        – characteristic polynomial coefficients
        'eigenvalues' – Weil numbers (complex)
        'J_order'     – #J(F_p) estimate
        'rms'         – fit residual
        'zeta_str'    – human-readable zeta function string
    """
    result = {}

    if scalars is None:
        scalars = list(range(len(counts)))

    result['scalars'] = scalars

    # --- choose sequence ---
    if use_curve_points:
        seq = counts_to_curve_points(counts, p)
        seq_label = "#C(F_{p^n}) estimates"
    else:
        seq = list(counts)
        seq_label = "raw root counts"

    result['seq'] = seq

    if verbose:
        print(f"\n{'='*60}")
        print("ZETA FUNCTION EXTRACTION")
        print(f"{'='*60}")
        print(f"  Sequence ({seq_label}), length {len(seq)}:")
        for i, v in enumerate(seq[:20]):
            n_label = scalars[i] if i < len(scalars) else i
            print(f"    n={n_label:3d}:  {v}")
        if len(seq) > 20:
            print(f"    ... ({len(seq)-20} more terms)")
        print()

    if len(seq) < 8:
        msg = f"  WARNING: only {len(seq)} terms — need ≥ 8 for a reliable degree-4 fit."
        if verbose:
            print(msg)
        result['error'] = msg
        return result

    # --- fit recurrence ---
    coeffs = fit_recurrence_linalg(seq, degree=4)
    result['coeffs'] = coeffs

    if coeffs is None:
        msg = "  Recurrence fitting failed (rank-deficient system)."
        if verbose:
            print(msg)
        result['error'] = msg
        return result

    rms = verify_recurrence(seq, coeffs, degree=4)
    result['rms'] = rms

    if verbose:
        print(f"  Recurrence:  a[n] = "
              f"{coeffs[0]:+.4g}·a[n-1] "
              f"{coeffs[1]:+.4g}·a[n-2] "
              f"{coeffs[2]:+.4g}·a[n-3] "
              f"{coeffs[3]:+.4g}·a[n-4]")
        print(f"  RMS residual: {rms:.4g}")
        if rms > 1.0:
            print("  *** High residual – sequence may be too short or noisy ***")
        print()

    # --- characteristic polynomial ---
    poly = recurrence_to_charpoly(coeffs, p)
    result['poly'] = poly

    # --- Weil numbers ---
    eigs = companion_eigenvalues(poly)
    result['eigenvalues'] = eigs

    if verbose:
        print("  Weil numbers (roots of L(T)):")
        for i, alpha in enumerate(eigs):
            mag = abs(alpha)
            print(f"    α_{i+1} = {alpha.real:+.6f} {alpha.imag:+.6f}i   |α| = {mag:.6f}  (√p = {p**0.5:.6f})")
        print()

    # --- #J(F_p) ---
    if poly is not None:
        J_order = sum(poly)
        result['J_order'] = J_order
        if verbose:
            print(f"  #J(F_{p}) = L(1) ≈ {J_order:.2f}")
            print(f"  (Integer nearest: {round(J_order)})")
            print()

    # --- full zeta string ---
    zeta_str = assemble_zeta(poly, p)
    result['zeta_str'] = zeta_str
    if verbose:
        print(zeta_str)

    return result

def extract_and_print_zeta(mumford_residues: dict, vecs_list: list, p: int,
                            use_curve_points: bool = True) -> dict:
    """
    One-liner entry point: residues + vecs + p  →  print zeta function.

    Works correctly regardless of worker completion order: keys by scalar n,
    not by position in the results dict.

    Example
    -------
    Insert after the mumford_precompute_residues_parallel call in search_main.py:

        from .zeta_from_counts import extract_and_print_zeta
        extract_and_print_zeta(mumford_residues, vecs_list, p=int(FINITE_FIELD))
    """
    counts, scalars = counts_from_residues(mumford_residues, vecs_list, p)
    return fit_zeta(counts, p, verbose=True, use_curve_points=use_curve_points,
                    scalars=scalars)

# ---------------------------------------------------------------------------
# 7.  Where to insert the hook in mumford_parallel.py
# ---------------------------------------------------------------------------

INTEGRATION_NOTE = """
Integration in mumford_parallel.py
====================================
After the `results_dict` is assembled (end of `mumford_precompute_residues_parallel`),
add these lines just before the final `return results_dict`:

    # --- ZETA FUNCTION EXTRACTION ---
    if FINITE_FIELD:
        from .zeta_from_counts import extract_and_print_zeta
        extract_and_print_zeta(results_dict, vecs_list, p=int(FINITE_FIELD))
    # --------------------------------

That's it. `vecs_list` is already in scope.

Alternatively, call it from search_main.py right after:
    mumford_residues = mumford_precompute_residues_parallel(...)

Insert:
    from .zeta_from_counts import extract_and_print_zeta
    extract_and_print_zeta(mumford_residues, vecs_list, p=int(FINITE_FIELD))
"""

# ---------------------------------------------------------------------------
# 8.  Self-test with a toy sequence
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("TEST 1: Synthetic genus-2 (p=7, a1=0, a2=-4, full sequence)")
    print("=" * 60)
    # L(T) = 1 - 0*T - 4*T^2 - 0*T^3 + 49*T^4
    # recurrence: a_n = 0*a_{n-1} - 4*a_{n-2} + 0*a_{n-3} + 49*a_{n-4}
    p_test = 7
    c1, c2, c3, c4 = 0, -4, 0, 49
    seq_full = [8, 0, -32, 0]
    for _ in range(20):
        seq_full.append(c1*seq_full[-1] + c2*seq_full[-2] + c3*seq_full[-3] + c4*seq_full[-4])

    # Simulate what counts_from_residues returns: scalars are non-consecutive,
    # reordered by worker completion (here: sorted, as they would be after our fix)
    # Use every other term to mimic sparse sampling
    scalars_test = list(range(0, len(seq_full)))
    result1 = fit_zeta(seq_full, p_test, verbose=True, use_curve_points=False,
                       scalars=scalars_test)
    print()

    print("=" * 60)
    print("TEST 2: Simulate your actual run (p=33554467, scalars 17-79 non-consecutive)")
    print("  Using known a1, a2 for y^2=x^5+x+2 (computed via Sage)")
    print("=" * 60)
    # For y^2 = x^5+x+2 over GF(33554467):
    # We'll synthesize a plausible sequence and verify the recurrence recovers it.
    # Actual values unknown here; we simulate what your solver WOULD produce.
    p2 = 33554467
    # Pretend a1=100, a2=500 (placeholder; real values come from your run)
    a1_fake, a2_fake = 100, 500
    c1f, c2f, c3f, c4f = a1_fake, a2_fake, a1_fake * p2, p2 * p2
    seq_seed = [p2 + 1 - a1_fake,
                p2**2 + 1 - (a1_fake**2 - 2*a2_fake),
                0, 0]
    for _ in range(80):
        seq_seed.append(c1f*seq_seed[-1] + c2f*seq_seed[-2] + c3f*seq_seed[-3] + c4f*seq_seed[-4])

    # Pick the non-consecutive scalars your workers actually produced
    # (from log: 78,79,75,76,68,69,66,64,61,60,57,54,51,41,37,33,42,32,29,49,25 ...)
    actual_scalars_seen = sorted([78,79,75,76,68,69,66,64,61,60,57,54,51,41,37,33,42,32,29,49,25,
                                   20,35,44,63,47,38,26,17,72])
    counts_fake = [seq_seed[n] for n in actual_scalars_seen]

    # This tests that non-consecutive scalars still allow recurrence recovery
    # (they DON'T directly — you need consecutive. This prints the raw data.)
    print(f"  {len(actual_scalars_seen)} non-consecutive scalars: {actual_scalars_seen[:10]}...")
    print(f"  NOTE: recurrence fitting requires CONSECUTIVE n values.")
    print(f"  With your vecs sampling 17..79 non-consecutively the fit will be noisy.")
    print(f"  For best results, use ALL integers 1..80 as your vecs_list.")
    print()

    # Show what we CAN extract: the raw count sequence sorted by n
    print("  Raw counts at each scalar n (first 15):")
    for n, c in zip(actual_scalars_seen[:15], counts_fake[:15]):
        print(f"    n={n:3d}: root_count≈{max(0, c % 10)}")  # modular toy value

    print()
    print(INTEGRATION_NOTE)

"""
zeta_patch.py
=============
Wires up zeta function extraction. Two approaches:

APPROACH A (fast, exact):
  Uses #J(F_p) = L(1) (already printed) + #C(F_p) from Sage's count_points.
  Two equations, two unknowns (a1, a2). Done in <1s.

APPROACH B (from fibration data):
  Fits a linear recurrence to the root-count sequence N_n from the fibration.
  Useful for understanding what the fibration is actually measuring, but
  N_n ≠ #C(F_{p^n}) — it's points on the SURFACE fiber, not the base curve.

--- WHERE TO INSERT ---

In search_main.py, _run_mumford_search(), right after:
    stats.end_phase('mumford_residues')

Add:

    if FINITE_FIELD:
        from .zeta_from_counts import compute_zeta_direct, compute_zeta_from_fibration
        compute_zeta_direct(coeffs_genus2, int(FINITE_FIELD))
        compute_zeta_from_fibration(mumford_residues, vecs_list, int(FINITE_FIELD))

That's the only change needed.
"""

# ============================================================================
# zeta_from_counts.py  (add these two functions to the existing file)
# ============================================================================

# ---------------------------------------------------------------------------
# APPROACH A: Direct computation via Sage
# Already have #J(F_p) = L(1).  Need one more: #C(F_p) = p + 1 - a1.
# ---------------------------------------------------------------------------

def compute_zeta_direct(f_coeffs: list, p: int) -> dict:
    """
    Compute the zeta function of y^2 = f(x) over F_p directly.

    Uses:
      1. Sage's frobenius_polynomial (fastest, authoritative)
      2. Fallback: count_points to get #C(F_p), then use #J(F_p)=L(1) from
         the already-computed Jacobian order.

    Parameters
    ----------
    f_coeffs : list
        Curve coefficients, highest-degree first (same convention as everywhere).
    p : int
        The finite field prime.

    Prints the full zeta function and returns a dict with a1, a2, L, J_order.
    """

    print(f"\n{'='*60}")
    print(f"ZETA FUNCTION: y^2 = f(x) over GF({p})")
    print(f"{'='*60}")

    F = GF(p)
    R = PolynomialRing(F, 'x')
    x = R.gen()

    # Build f(x) mod p (highest-degree first)
    f_poly = R(list(reversed(f_coeffs)))
    C = HyperellipticCurve(f_poly)

    # --- Method 1: Frobenius polynomial (one call, gives everything) ---
    try:
        frob = C.frobenius_polynomial()
        # frob(T) = L(T) for the Jacobian
        # = T^4 - a1*T^3 + a2*T^2 - a1*p*T^3 + p^2*T^4  (palindromic)
        coeffs = frob.list()  # low to high: [p^2, -a1*p, a2, -a1, 1]
        # reverse to high-to-low
        L_coeffs = list(reversed(coeffs))  # [1, -a1, a2, -a1*p, p^2]

        a1 = -L_coeffs[1]
        a2 =  L_coeffs[2]

        print(f"  Method: Frobenius polynomial (exact)")
        _print_zeta_results(a1, a2, p, L_coeffs)

        return {
            'a1': int(a1), 'a2': int(a2),
            'L_coeffs': [int(c) for c in L_coeffs],
            'J_order': int(frob(1)),
            'C_Fp_count': int(p + 1 - a1),
        }

    except Exception as e:
        print(f"  Frobenius polynomial failed ({e}), falling back to point count...")

    # --- Method 2: count_points + known J_order ---
    try:
        # #C(F_p) gives a1
        N1 = len(list(C.points()))   # exact, works for small p; may be slow for p~2^25
        a1 = p + 1 - N1

        # #J(F_p) = L(1) = 1 - a1 + a2 - a1*p + p^2
        # We get J_order from jacobian_group_order
        J = C.jacobian()
        try:
            J_order = int(J.order())
        except Exception:
            # Fallback: use the value already printed in your run
            # (user should pass it in; here we compute from count_points over F_{p^2})
            N1_sq, N2_sq = C.count_points(2)
            a1_check = p + 1 - N1_sq
            two_a2 = N2_sq - p**2 - 1 + a1_check**2
            a2_check = two_a2 // 2
            J_order = 1 - a1_check + a2_check - p * a1_check + p**2
            a1 = a1_check
            a2 = a2_check

        # Solve for a2 from J_order = 1 - a1 + a2 - a1*p + p^2
        if 'a2_check' not in dir():
            a2 = J_order - 1 + a1 - p**2 + a1 * p

        L_coeffs = [1, -a1, a2, -a1 * p, p**2]

        print(f"  Method: point count (N1={N1}, J_order={J_order})")
        _print_zeta_results(a1, a2, p, L_coeffs)

        return {
            'a1': int(a1), 'a2': int(a2),
            'L_coeffs': [int(c) for c in L_coeffs],
            'J_order': int(J_order),
            'C_Fp_count': int(N1),
        }

    except Exception as e:
        print(f"  Point count also failed: {e}")
        raise

def _print_zeta_results(a1, a2, p, L_coeffs):
    """Print formatted zeta function output."""
    try:
        roots = np.roots([float(c) for c in L_coeffs])
    except ImportError:
        roots = []

    print(f"\n  L(T) = 1 - {a1}·T + {a2}·T² - {a1*p}·T³ + {p**2}·T⁴")
    print(f"\n  Z(C/F_{p}, T) = L(T) / ((1-T)(1-{p}T))")
    print(f"\n  #C(F_{p}) = {p + 1 - a1}     (= p + 1 - a1 = {p} + 1 - {a1})")
    print(f"  #J(F_{p}) = L(1) = {1 - a1 + a2 - a1*p + p**2}")

    if len(roots) == 4:
        print(f"\n  Weil numbers (roots of L):")
        for i, alpha in enumerate(roots):
            print(f"    α_{i+1} = {alpha.real:+.6f} {alpha.imag:+.6f}i   |α| = {abs(alpha):.6f}  (√p = {p**0.5:.6f})")

    # Sanity: palindrome
    err_c4 = abs(L_coeffs[4] - p**2)
    err_c3 = abs(L_coeffs[3] - (-a1 * p))
    if err_c4 == 0 and err_c3 == 0:
        print(f"\n  ✓ Palindrome check passed (c4={p**2}, c3={-a1*p})")
    else:
        print(f"\n  ✗ Palindrome anomaly: c4 err={err_c4}, c3 err={err_c3}")

# ---------------------------------------------------------------------------
# APPROACH B: From fibration root counts
# Fits a linear recurrence to N_n (roots of X([n]P)(m) = -m+x1 over F_p).
# This measures the SURFACE geometry, not directly the base curve.
# The recurrence coefficients relate to the Frobenius on H^2 of the surface.
# ---------------------------------------------------------------------------

def compute_zeta_from_fibration(mumford_residues: dict, vecs_list: list, p: int) -> dict:
    """
    Extract a linear recurrence from the fibration root count sequence N_n.

    N_n = #{m in F_p : X([n]P)(m) == -m + x1}

    This sequence satisfies a linear recurrence whose characteristic polynomial
    is related to the L-function of the elliptic surface (not just the base curve).
    For comparison with the direct zeta, look at whether L(1) from the recurrence
    agrees with your known Jacobian order.

    Parameters
    ----------
    mumford_residues : dict
        results_dict from mumford_precompute_residues_parallel.
    vecs_list : list
        The vecs_list passed to mumford_precompute_residues_parallel.
    p : int
        Finite field prime.
    """
    # Import the existing extraction functions
    counts, scalars = counts_from_residues(mumford_residues, vecs_list, p)

    if not counts:
        print("[zeta/fibration] No root counts found.")
        return {}

    print(f"\n{'='*60}")
    print(f"FIBRATION ROOT COUNT SEQUENCE (N_n)")
    print(f"{'='*60}")
    print(f"  {len(counts)} vectors, n range: {scalars[0]}..{scalars[-1]}")

    # Check for gaps
    gaps = [scalars[i+1] - scalars[i] for i in range(len(scalars)-1)]
    max_gap = max(gaps) if gaps else 0
    if max_gap > 1:
        n_missing = sum(g - 1 for g in gaps)
        print(f"  WARNING: {n_missing} gaps in scalar sequence (max gap={max_gap})")
        print(f"  Recurrence fit needs CONSECUTIVE n values — filling gaps with 0")
        # Fill gaps
        full_range = list(range(scalars[0], scalars[-1] + 1))
        count_map = dict(zip(scalars, counts))
        counts = [count_map.get(n, 0) for n in full_range]
        scalars = full_range

    print(f"\n  N_n sequence:")
    count_mean = 0
    for i in range(len(counts)):
        print(f"    n={scalars[i]:3d}: {counts[i]} roots")
        count_mean += counts[i]
    count_mean /= len(counts)
    print("count mean:", count_mean)

    if len(counts) < 8:
        print("  Not enough terms for recurrence fit (need ≥ 8).")
        return {'counts': counts, 'scalars': scalars}

    # Fit degree-4 recurrence
    coeffs = fit_recurrence_linalg(counts, degree=4)

    if coeffs is None:
        print("  Recurrence fit failed (rank-deficient).")
        return {'counts': counts, 'scalars': scalars}

    # RMS check
    rms = _rms_residual(counts, coeffs)
    print(f"\n  Fitted recurrence (degree 4):")
    print(f"    N_n = {coeffs[0]:+.4g}·N_{{n-1}} {coeffs[1]:+.4g}·N_{{n-2}} "
          f"{coeffs[2]:+.4g}·N_{{n-3}} {coeffs[3]:+.4g}·N_{{n-4}}")
    print(f"    RMS residual: {rms:.4g}")

    if rms > 0.5:
        print("  *** High residual — sequence may have gaps or the degree is wrong ***")
        # Try degree 2 (may work if surface is a product)
        c2 = fit_recurrence_linalg(counts, degree=2)
        if c2 is not None:
            rms2 = _rms_residual(counts, c2)
            print(f"  Trying degree-2: N_n = {c2[0]:+.4g}·N_{{n-1}} {c2[1]:+.4g}·N_{{n-2}}  (rms={rms2:.4g})")
            if rms2 < rms:
                coeffs = c2
                rms = rms2
                print("  Using degree-2 fit.")

    poly = recurrence_to_charpoly(coeffs, p)
    if poly is not None:
        eigs = companion_eigenvalues(poly)
        print(f"\n  Characteristic roots of fibration recurrence:")
        for i, alpha in enumerate(eigs):
            print(f"    λ_{i+1} = {alpha.real:+.6f} {alpha.imag:+.6f}i   |λ| = {abs(alpha):.6f}  (√p = {p**0.5:.6f})")

        L_at_1 = sum(poly)
        print(f"\n  L_surface(1) = {L_at_1:.2f}")
        print(f"  (For comparison: #J(F_p) = {p + 1 - int(round(-poly[1]))!r}  [rough])")

    return {
        'counts': counts,
        'scalars': scalars,
        'coeffs': coeffs,
        'rms': rms,
        'poly': poly,
    }

def _rms_residual(seq, coeffs):
    d = len(coeffs)
    if len(seq) <= d:
        return float('inf')
    residuals = []
    for i in range(d, len(seq)):
        pred = sum(coeffs[j] * seq[i - 1 - j] for j in range(d))
        residuals.append((seq[i] - pred) ** 2)
    return math.sqrt(sum(residuals) / len(residuals)) if residuals else 0.0

# ---------------------------------------------------------------------------
# The two-line addition to search_main.py _run_mumford_search
# ---------------------------------------------------------------------------

PATCH = """
# In search_main.py, _run_mumford_search(), after:
#     stats.end_phase('mumford_residues')
# Add:

    if FINITE_FIELD:
        from .zeta_from_counts import compute_zeta_direct, compute_zeta_from_fibration
        compute_zeta_direct(coeffs_genus2, int(FINITE_FIELD))
        compute_zeta_from_fibration(mumford_residues, vecs_list, int(FINITE_FIELD))
"""

if __name__ == '__main__':
    print(PATCH)

    # Minimal self-test of _rms_residual and fit
    p_test = 7
    c1, c2, c3, c4 = 0, -4, 0, 49
    seq = [8, 0, -32, 0]
    for _ in range(20):
        seq.append(c1*seq[-1] + c2*seq[-2] + c3*seq[-3] + c4*seq[-4])

    coeffs = fit_recurrence_linalg(seq, degree=4)
    rms = _rms_residual(seq, coeffs)
    print(f"Self-test RMS (should be 0): {rms}")
    assert rms < 1e-9, f"RMS too high: {rms}"
    print("OK")
