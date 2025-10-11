# Elliptic Fibration Rational Point Search

A SageMath-based toolkit for finding rational points on genus 2+ curves via elliptic fibrations and modular methods.

## Overview

This project implements a sophisticated (*cough* overly complicated *cough*) algorithm to find rational points on higher genus curves (genus ≥ 2) by:

1. **Constructing elliptic fibrations** from known rational points
2. **Searching via lattice reduction** (LLL) combined with Chinese Remainder Theorem (CRT)
3. **Leveraging modular arithmetic** across multiple prime subsets for comprehensive coverage
4. **Computing geometric invariants** including Picard number, Mordell-Weil rank, and rational curve counts (Yau-Zaslow)

### Key Features

- **Automated fibration tower construction** from genus 2,3,4,... curves down to elliptic curves
- **Adaptive parameter tuning** based on canonical height estimates
- **Parallel prime subset search** for efficient candidate generation
- **Comprehensive diagnostics**: Tate's algorithm, Shioda-Tate formula, Picard-van Luijk bounds
- **Rational curve enumeration** via Néron-Severi lattice (for Yau-Zaslow analysis)

## Quick Start

### Prerequisites

- SageMath 9.5+ (tested on 10.x)
- Python 3.8+
- Required packages: `tqdm`, `colorama` (for progress bars)

```bash
# Install dependencies
sage -pip install tqdm colorama
```

### Basic Usage (configure stuff in search_common.py)

**Example 1: Genus 2 Curve**

```python
# Define your curve: y^2 = a6*x^6 + a5*x^5 + ... + a0
COEFFS_GENUS2 = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)]

# Provide one known rational point (x-coordinate)
DATA_PTS_GENUS2 = [QQ(1)]  

# Set termination condition (stop after finding N points)
TERMINATE_WHEN_6 = 12

# Run the search
sage search7_genus2.sage
```

**Example 2: Genus 3 Curve**

```python
# y^2 = x^8 + ... (deg 8 polynomial)
COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(0), QQ(0), QQ(2), QQ(0), QQ(-4), QQ(0), QQ(1)]
DATA_PTS_GENUS2 = [QQ(0)]
```

## How It Works

### 1. Fibration Construction ("Tower Method")

Given a genus g curve C and known rational points P₁, ..., Pₙ:

```
Genus g curve C
    ↓ (parameterize by m)
Genus g-1 curve C₁(m) (actually, one degree lower in x per step)
    ↓ (iterate)
Genus 1 curve E(m)  ← Elliptic fibration over ℚ(m)
```

Each fibration is an elliptic curve E_m whose rational points correspond to rational points on the original curve.

### 2. Mordell-Weil Lattice Search

The rational points on E(ℚ(m)) form a finitely-generated abelian group (Mordell-Weil group). We:

1. Compute the **canonical height pairing** matrix H
2. Use **LLL reduction** to find short vectors in the lattice
3. Search for **integer linear combinations** of base sections that yield new rational points

### 3. Modular Prime Subset Strategy

Instead of using all primes, we:

- Generate **diverse random subsets** of primes (size 3-9)
- For each subset, solve **mod p** for each prime p
- Use **CRT** to lift solutions to ℤ/Mℤ (M = product of primes)
- Apply **rational reconstruction** to find candidates m ∈ ℚ

**Why this works:**
- Small subsets (3-5 primes) find different solutions than large subsets (7-9 primes)
- Diversity beats enforcing a fixed modulus threshold
- Critical small primes (3, but not 2) enable reconstruction of small denominators

### 4. Automatic Parameter Configuration

The system automatically determines:

- **HEIGHT_BOUND**: Controls lattice vector enumeration depth (canonical height limit) (based on known point heights)
- **TMAX**: LLL enumeration limit per m-value (density-based heuristic)
- **Prime pool filtering**: Excludes primes where reduction fails
- **Subset generation**: Adaptive based on residue density

## Architecture

### Core Modules

- **`search7_genus2.sage`**: Main entry point for genus 2+ curves
- **`search_lll.py`**: LLL reduction and modular search engine
- **`bounds.py`**: Automatic parameter configuration and prime pool management
- **`tower.sage`**: Fibration tower construction logic
- **`tate.py`**: Tate's algorithm for singular fiber analysis
- **`picard.py`**: Picard number computation via van Luijk's method
- **`yau.py`**: Rational curve counting (Yau-Zaslow formula)
- **`nslattice.py`**: Néron-Severi lattice operations
- **`search_common.py`**: Shared configuration and utilities; contains test curves sourced from LMFDB.org
- **`diagnostics2.py`**: Singular fiber analysis
- **`automorph.py`**: looks for section automorphisms
- **`sat.py`**: saturation analysis
- **`torsion.py`**: torsion analysis

### Key Data Structure: `CurveDataExt`

```python
CurveDataExt(
    E_curve,         # Original genus 1 quartic
    E_weier,         # Weierstrass model E: y² = x³ + a4*x + a6
    a4, a6,          # Coefficients as functions of m
    phi_x,           # Rational map x ↦ X(m)/Z(m)  (x coord on the Weierstrass model side)
    morphs,          # Coordinate transformations (quartic to Weierstrass model transformations)
    singfibs,        # Singular fiber data (Kodaira types)
    bad_primes,      # Primes where reduction fails
    ...
)
```

## Example Output

```
--- Search Iteration 0 with 1 sections ---
Height Pairing Matrix H: [2]
Searching 12 vectors up to height 311...
Searching Prime Subsets: 100%|████████| 995/995 [00:29<00:00]

Found 1315556 potential (m, vector) pairs. Testing for rationality...
Found 12 new point(s) and 0 new section(s).
New x-coordinates: {0, 1, 2, -1/8, 3, 1/2, -2/3, 4/15, 1/3, 13/20, -15/14, -1}

*** Concluded Picard number: ρ = 3 ***
Shioda-Tate Rank Estimate: 1

Final list of known points:
[(-15/14, 16729/2744), (-1, 5), (-2/3, 1/27), ..., (3, 13)]
```

## Configuration

### Static Configuration (`search_common.py`)

```python
# Prime pool (auto-filtered per fibration)
PRIME_POOL = list(primes(90))

# Search termination
TERMINATE_WHEN_6 = 12  # Stop after finding N points

# Search modes
SYMBOLIC_SEARCH = False  # True: solve over ℚ(m), False: mod p search
USE_MINIMAL_MODEL = True  # True: apply Tate's algorithm
DEBUG = False
```

### Dynamic Configuration (Auto-Tuned)

These are computed automatically by `bounds.auto_configure_search()`:

- `HEIGHT_BOUND`: 100 * exp(h_can / 4) + 100
- `MAX_MODULUS`: 10¹⁵ (safety cap)
- `TMAX`: 200-400 based on residue density
- `NUM_PRIME_SUBSETS`: 250-2000 based on fibration degeneracy

## Advanced Features

### Picard Number Computation

Uses van Luijk's reduction method:

```
Lower bound: ρ ≥ 2 + MW_rank + Σ(m_v - 1)
Upper bound: Reduce mod ℓ and count independent classes
```

When lower = upper, the Picard number is **exact**.

### Rational Curve Counting

For surfaces with ρ ≥ 3, enumerate classes of rational curves:

```python
# Find all (-2)-curves (rational curves with self-intersection -2)
counts, reps = staged_rational_curve_search(cd, sections, rho, mw_rank, chi,
                                           height_bounds=(15, 25, 35, 45),
                                           require_S_coeff='positive')
```

Outputs generating functions for curve counts per degree.

### Saturation Diagnostics

Verify if the Mordell-Weil group is p-saturated:

```
✅ p=2 appears to be saturated (witness at m=-3, ℓ=11)
✅ p=3 appears to be saturated (witness at m=-3, ℓ=17)
✅ p=5 appears to be saturated (witness at m=-2, ℓ=17)
```

## Known Limitations

1. **Symbolic search** (solving over ℚ(m)) is often slower and finds almost nothing
2. **Non-minimal models** use sampled height matrices (less precise but sometimes faster)
3. **Very high degree fibrations** (deg > 15) may require increased `MAX_MODULUS`
4. **Ramified primes** (3) are essential for reconstruction—always kept in pool

## Troubleshooting

**Problem**: Search finds 0 points despite known solutions

**Solutions**:
1. Ensure p=3 is in `PRIME_POOL`
2. Increase `HEIGHT_BOUND` (try 2x current value)
3. Increase `NUM_PRIME_SUBSETS` (try 2000+)
4. Check if `y=0` points exist (auto-detected but may need manual verification)

**Problem**: "Cannot clear rational denominators" errors

**Cause**: Prime p divides coefficient denominators

**Solution**: This is expected—prime is auto-skipped. If too many primes fail, check your curve coefficients.

## Performance Tips

- **Start with 1-point fibrations** (fastest) before trying 2- or 3-point fibrations (those are currently broken anyway; it just hangs)
- **Enable multiprocessing** by ensuring `PARALLEL_PRIME_WORKERS > 1`
- **Use non-minimal models** (`USE_MINIMAL_MODEL=False`) for initial exploration
- **Profile hot paths** using `@PROFILE` decorator (requires `line_profiler`)

## Testing

Included test curves (in `search_common.py`):

```python
# Test 1: y² = x⁶ - 12x⁵ + 30x⁴ + 2x³ - 15x² + 2x + 1
# Known to have 12 rational points

# Test 2: y² = x⁶ + 4x⁵ - 2x⁴ - 18x³ + x² + 38x + 25
# Classic example, 11+ points
```

Run test suite:
```bash
sage search7_genus2.sage  # Uses default test curve
```

## References

### Theoretical Background:

i taught myself all of this using ai, so, uhhhhh....

uhh

Read Silverman's textbooks I guess.

TRANS RIGHTS!

### Implementation Notes

- **LLL reduction**: Uses Sage's built-in implementation (δ=0.98 for strong reduction)
- **CRT**: Cached with `functools.lru_cache` for repeated queries
- **Rational reconstruction**: Extended Euclidean algorithm with bounded denominators

## Contributing

Issues and pull requests welcome! Areas for improvement:

- [ ] Support for hyperelliptic curves of arbitrary genus
- [ ] GPU acceleration for parallel prime subsets
- [ ] Better heuristics for `HEIGHT_BOUND` estimation
- [ ] Integration with LMFDB for curve verification

## License

GNU General Public License v3 (or later)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{elliptic_fibration_search,
  title={Elliptic Fibration Rational Point Search},
  author={Claire Hoying},
  year={2025},
  url={https://github.com/goracle/elliptic-fibration-search}
}
```

## Acknowledgments

Built with SageMath and data from LMFDB.org. Special thanks to the computational number theory community for foundational algorithms.
Built with LLM models including:
Chatgpt-4o,5,3.5,...; Gemini 2.5 pro, 2.5 flash; Claude 3.5,4,4.5,
and to a much lesser extent:
Perplexity, Qwen, and Grok

---

**Status**: Active development/Testing Needed | **Last updated**: October 2025

For questions or collaboration: [daniel DOT hoying AT uconn.edu]
