# Genus 2 Curve Arithmetic System

A SageMath-based research toolkit for studying the arithmetic of hyperelliptic curves over **ℚ**. Its central innovation is an iterated **fibration tower** construction that reduces difficult questions about high-genus curves to tractable problems on families of elliptic curves.

## Overview

The system provides three tightly integrated capabilities:

1. **Rational point enumeration** — finding rational points on curves of arbitrary genus (genus 2 fully supported; genus 3 tested; higher genus possible but hard to find test curves for) starting from a single seed, via CRT, LLL lattice reduction, and Hensel lifting.

2. **Jacobian rank determination** (`MUMFORD_SEARCH` mode) — computing a provably independent basis for *J(ℚ)*, the Mordell–Weil group of the Jacobian. Typically completes in under 2 minutes regardless of conductor or rank.

3. **Finite-field DLP / index calculus** (`FINITE_FIELD` mode) — attacking the genus 2 hyperelliptic-curve discrete logarithm problem (HECC) via Mumford divisor smoothness relations and sparse linear algebra.

The system has been validated against the LMFDB on dozens of genus 2 curves with conductors up to ~10⁷ and Mordell–Weil ranks 0 through 4+.

---

## Quick Start

### Prerequisites

- SageMath ≥ 10.x
- Python 3.10+
- Required packages: `numpy`, `scipy`, `tqdm`, `colorama`, `cysignals`, `cypari2`

```bash
sage -pip install numpy scipy tqdm colorama cysignals cypari2
```

Optional (for DLP Baby-step Giant-step acceleration): build the C extension that `walker.py` requires:
```bash
gcc -O2 -shared -fPIC -o libwalk.so collision_walk.c
```
Note: the compiled binary is not included in the repository. Without `libwalk.so`, DLP mode will fall back to pure-Python walks (slower) or raise a `FileNotFoundError` on import of `walker.py`.

### Configuration

All configuration lives in `search_common.py`. Set your curve, seed points, and mode flags there, then run:

```bash
sage search7_genus2.sage
```

---

## Usage Examples

### Example 1: Rational Point Search (Genus 2)

```python
# In search_common.py:

# y² = x⁶ - 12x⁵ + 30x⁴ + 2x³ - 15x² + 2x + 1
COEFFS_GENUS2 = [QQ(1), QQ(-12), QQ(30), QQ(2), QQ(-15), QQ(2), QQ(1)]

# One known rational point (x-coordinate only)
DATA_PTS_GENUS2 = [QQ(1)]

# Stop after finding this many distinct rational x-coordinates
TERMINATE_WHEN_6 = 12

MUMFORD_SEARCH = False
FINITE_FIELD = None
```

```bash
sage search7_genus2.sage
```

### Example 2: Jacobian Rank Finding (MUMFORD_SEARCH)

```python
# y² = x⁶ + 8x⁵ + 10x⁴ - 10x³ - 11x² + 2x + 1
COEFFS_GENUS2 = [QQ(1), QQ(8), QQ(10), QQ(-10), QQ(-11), QQ(2), QQ(1)]
DATA_PTS_GENUS2 = [QQ(-1)]

MUMFORD_SEARCH = True   # find Mumford basis for J(ℚ)
FINITE_FIELD = None
```

Result: certified rank-4 Mumford basis in ~33.5 seconds.

### Example 3: Finite-Field DLP

```python
# y² = x⁵ + 3x³ + 2x² + 5x + 4 over GF(33554467)
COEFFS_GENUS2 = [QQ(1), QQ(0), QQ(3), QQ(2), QQ(5), QQ(4)]
DATA_PTS_GENUS2 = [QQ(0)]

FINITE_FIELD = 33554467   # run index calculus over GF(p)
SECRET_KEY = 800
BLOCK_WIEDEMANN = True
```

### Example Logs

See `example.txt` through `example7.txt` for logs from real runs.

---

## How It Works

### 1. Fibration Tower Construction

Given a curve *C: y² = f(x)* and a seed rational point *P = (x₀, y₀)*, the system constructs an iterated fibration tower that steps down one degree at a time until reaching a genus-1 quartic fiber:

```
Genus g curve C  (deg f = 2g+1 or 2g+2)
    ↓  (parameterize by slope m through P — one step per degree reduction)
    ↓  ...
Genus-1 quartic fiber E(m)           ← Weierstrass model over ℚ(m)
```

This works for **arbitrary genus**: genus 2 (one or two tower steps) is fully supported and well-tested; genus 3 has been tested successfully; higher genera are supported in principle but rarely tested due to difficulty finding suitable curves. The configuration flag `COEFFS_GENUS2` accepts degree-5 or degree-6 polynomials for genus 2, and degree-7 or degree-8 for genus 3.

At each step the algorithm selects the best-scoring auxiliary polynomial *Q(x)* from 10 candidates (trading off coefficient height, degree, and bad-prime content), computes the intersection locus as a rational function *r(m)*, and passes to the residual quartic. A rational value of *m* corresponds exactly to a rational point on *C*.

### 2. Rational Point Search

**The key observation:** a rational point *(x₀, y₀) ∈ C(ℚ)* corresponds to a rational value of *m* satisfying *x₀ = r(m)*. The search finds such *m* by:

1. **LLL/BKZ lattice reduction** on the height pairing matrix to find short vectors (low-multiplier section combinations most likely to produce small-height *m* values).
2. **Modular residue harvesting**: for each prime *p*, solve *Sᵢ(m) ≡ r(m) (mod p)* to find candidate residues.
3. **CRT reconstruction**: combine residues across diverse random prime subsets (size 3–9) to form rational candidates via rational reconstruction.
4. **Hensel lifting** to improve precision on simple roots.
5. **Verification** by substituting candidates back into the curve equation.

**Why diverse subsets?** Small subsets (3–5 primes) find different solutions than large subsets (7–9). The prime 3 is kept in the pool—it is essential for reconstructing small denominators.

### 3. Jacobian Rank Finding (MUMFORD_SEARCH Mode)

Instead of rational points on *C*, this mode directly finds rational Mumford divisors — elements *D = [(u(x), v(x))]* of *J(ℚ)* with *u* monic of degree 2. The rank of *J(ℚ)* equals the size of the largest independent such set.

Each divisor is specified by four rationals *(s, p, v₁, v₀)* where *u(x) = x² − sx + p* and *v(x) = v₁x + v₀*. The pipeline:

1. Solves the Mumford constraint system over 𝔽ₗ for each prime ℓ in the pool.
2. Combines residues via CRT with algebraic pre-filtering.
3. Performs **torsion filtering** (discard elements of finite order).
4. **Incrementally certifies** independence via Smith normal form over ℤ/nℤ at multiple primes.
5. Confirms independence via the **Néron–Tate Gram matrix** (must be positive definite).

The result is a certified (not heuristic) lower bound on the rank. Combining with the 2-Selmer upper bound often pins the rank exactly.

### 4. Index Calculus DLP (FINITE_FIELD Mode)

Given *G, Q ∈ J(𝔽ₚ)* with *Q = [d]G*, find *d*:

1. **Factor base construction**: collect smooth Mumford divisors (those whose *u(x)* splits completely over 𝔽ₚ) using the fibration residue system.
2. **Relation generation**: a structured walk (seeded by fibration residues) and a Baby-step Giant-step collision walk cooperate to collect enough relations.
3. **Linear algebra**: solve the sparse relation matrix over ℤ/ℓℤ (where ℓ is the group order) using Block Wiedemann's algorithm.

### 5. Supporting Analyses

- **Picard number ρ(S)**: Shioda–Tate lower bound (from sections + fiber types) plus Lefschetz trace upper bound (reduction mod ℓ). When bounds agree, ρ is exact and the Mordell–Weil rank is pinned.
- **2-Selmer bounds**: local 2-descent at all bad primes, giving an upper bound on rank *J(ℚ)*.
- **Torsion analysis**: identifies *J(ℚ)_tors* by GCD of group-order specializations at good primes.
- **Automorphism analysis**: Néron–Severi lattice symmetries that can reduce search space.
- **CM fiber detection**: fibers with complex multiplication (special *j*-invariants) often yield high-rank specializations and are tried as candidate rational points.
- **Archimedean height pairings**: Néron–Tate pairings via Arakelov intersection theory (finite places) and numerical integration over the period lattice (archimedean place).

---

## Architecture

### Key Data Structure: `CurveDataExt`

```python
CurveDataExt(
    E_curve,      # Original genus-1 quartic fiber
    E_weier,      # Weierstrass model E: y² = x³ + a₄(m)x + a₆(m)
    a4, a6,       # Coefficients as polynomials in m
    phi_x,        # Rational map x ↦ X(m)/Z(m) (x-coord on Weierstrass side)
    morphs,       # Quartic-to-Weierstrass coordinate transformations
    singfibs,     # Singular fiber data (Kodaira types)
    bad_primes,   # Primes where reduction fails
    ...
)
```

### Module Reference

| Module | Role |
|--------|------|
| `search7_genus2.sage` | **Main entry point**: orchestrates fibration tower, search, Mumford basis, analyses |
| `search_common.py` | **Global configuration** and shared utilities; contains test curves sourced from LMFDB |
| `tower.sage` | **Fibration tower construction**: geometry scoring, polynomial interpolation, tower iteration |
| `search_main.py` | **Rational point search**: LLL vectors, modular pipeline, CRT, Hensel, rational reconstruction |
| `mumford_basis.py` | **Jacobian basis construction**: Mumford CRT reconstruction, torsion filter, mod-p certification |
| `mumford_parallel.py` | Parallel residue computation for the Mumford system across the prime pool |
| `mumford_reconstruction.py` | Algebraic pre-filtering and CRT combination of Mumford coordinate residues |
| `mumford_core.py` | Core Mumford divisor arithmetic and group law |
| `mumford_solver.py` | Mumford constraint solving over finite fields |
| `mumford_equations.py` | Mumford condition polynomials |
| `mumford_height.py` | Height computations for Mumford divisors |
| `mumford_verification.py` | Verification routines for reconstructed divisors |
| `index_calculus.py` | **Finite-field DLP**: factor base generation, relation walk, matrix preparation |
| `sparse_linalg_modp.py` | Block Wiedemann sparse linear algebra solver over ℤ/ℓℤ |
| `arakelov.py` | Arakelov height pairings (finite places) for Néron–Tate Gram matrix |
| `homology.py` | Period lattice and archimedean height integrals |
| `periods.py` | Period computation for the Jacobian |
| `integration.py` | Numerical integration utilities |
| `analytic_pairings.py` | Analytic/archimedean pairing computations |
| `pairings.py` | Height pairing dispatch |
| `picard.py` | Picard number bounds, Lefschetz trace, Shioda–Tate analysis |
| `selmer.py` / `selmer_genus2.py` | 2-Selmer rank computation for genus 2 curves |
| `tate.py` | Tate's algorithm for Kodaira fiber classification and minimal model reduction |
| `bounds.py` | Splitting field degree estimation, height bound auto-tuning, prime pool management |
| `ll_utilities.py` | LLL/BKZ lattice reduction, vector generation, rational reconstruction |
| `heights.py` | Canonical height computations |
| `torsion.py` | Torsion subgroup identification |
| `automorph.py` | Néron–Severi lattice automorphisms |
| `sat.py` | Saturation analysis for the Mordell–Weil group |
| `nslattice.py` | Néron–Severi lattice operations |
| `yau.py` | Rational curve counting (Yau–Zaslow formula) |
| `brauer.py` | Brauer group computations |
| `local.py` / `local_diagnostics.py` | Local arithmetic and diagnostics |
| `diagnostics2.py` | Singular fiber analysis |
| `stats.py` | Phase timing, counter tracking, and run statistics |
| `consensus.py` | Multi-fibration consensus filter |
| `parallel.py` | Parallel execution utilities |
| `core.py` | Core shared arithmetic routines |
| `utilities.py` | Miscellaneous utilities |
| `dedup.py` | Deduplication of candidate points |
| `walker.py` | ⚠️ **Active development** — Baby-step Giant-step collision walk for DLP relation generation. Barely runs; treat as experimental. Requires `libwalk.so` (compiled from `collision_walk.c`, not included as a binary). |
| `collision_walk.c` | C source for the collision walk; must be compiled to `libwalk.so` manually (see below). |

---

## Configuration

### Key Flags (`search_common.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `COEFFS_GENUS2` | — | Coefficient list *[aₙ, ..., a₀]* for *y² = f(x)*. Degree 5–6 for genus 2; degree 7–8 for genus 3; higher degrees accepted (experimental). |
| `DATA_PTS_GENUS2` | — | Seed x-coordinates (rational) on the curve. |
| `TERMINATE_WHEN_6` | — | Stop after finding this many distinct rational x-coordinates (rational point mode). |
| `MUMFORD_SEARCH` | `False` | `True`: find Mumford basis for *J(ℚ)* (rank). `False`: find rational points on *C*. |
| `FINITE_FIELD` | `None` | `None` for over-ℚ search; set to prime *p* for finite-field DLP mode. |
| `SECRET_KEY` | — | DLP mode: secret exponent *d* (for benchmarking). |
| `BLOCK_WIEDEMANN` | `False` | `True`: always use Block Wiedemann for the final DLP solve. |
| `PRIME_POOL` | `primes(590)` | Pool of primes for modular residue computation (auto-filtered per fibration). |
| `NUM_PRIME_SUBSETS` | `500` | Number of independent prime subsets; ≥ 250 for stability. |
| `USE_MINIMAL_MODEL` | `True` | `True`: Tate-minimized Weierstrass model (slower, more correct). `False`: generic fiber model. |
| `SYMBOLIC_SEARCH` | `False` | `True`: solve over ℚ(m) (often slower, rarely finds anything extra). |
| `HEIGHT_BOUND` | auto | Canonical height cutoff for rational reconstruction candidates. |
| `DEBUG` | `False` | Enable verbose diagnostic output. |

### Auto-Tuned Parameters

Computed automatically by `bounds.auto_configure_search()`:

- `HEIGHT_BOUND` = 100 · exp(h_can / 4) + 100
- `TMAX`: 200–400 based on residue density
- `NUM_PRIME_SUBSETS`: 250–2000 based on fibration degeneracy
- `MAX_MODULUS`: 10¹⁵ (safety cap)

---

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

**MUMFORD_SEARCH example** (rank-4 curve, ~33.5 seconds):
```
Certified independent Mumford basis of size 4:
  D₁ = [x² + 3x + 2, ...]
  D₂ = [x² - x - 1, ...]
  D₃ = [x² + x,     ...]
  D₄ = [x² - 2x,    ...]
mod-p rank = 4 (certified at 4 primes)
Gram matrix positive definite ✓
```

**Saturation diagnostics:**
```
✅ p=2 appears to be saturated (witness at m=-3, ℓ=11)
✅ p=3 appears to be saturated (witness at m=-3, ℓ=17)
✅ p=5 appears to be saturated (witness at m=-2, ℓ=17)
```

---

## Performance Notes

- **Rational point search**: most runs complete in seconds to a few minutes depending on the height of the target points and the prime pool size.
- **MUMFORD_SEARCH**: typically under 2 minutes for genus 2 curves with conductors up to ~10⁷ and ranks 0–4. The bottleneck (CRT reconstruction) is parallelized across up to 20 workers.
- **DLP mode**: for a rank ~10¹² Jacobian over GF(33554467), ~18K smooth divisors collected and the relation matrix built in ~112 seconds.
- Use non-minimal models (`USE_MINIMAL_MODEL=False`) for quick initial exploration; switch to minimal models for rigorous results.
- Start with 1-point fibrations (fastest). Multi-point fibrations are not fully stable and may hang.

---

## Known Limitations

1. **Multi-point fibrations** (2+ seed points) are not fully implemented and may hang.
2. **2-Selmer upper bound** can greatly exceed the true rank when Ш(*J*)[2] is large.
3. **Symbolic search** (`SYMBOLIC_SEARCH=True`) is usually slower and rarely finds anything that modular search misses.
4. **Not every rational point** is visible from a given fibration; the multi-fibration strategy mitigates this but cannot guarantee completeness.
5. **Higher genus** (genus ≥ 3) rational point search is supported in principle by the tower construction and has been tested on genus 3 curves, but higher genera are largely untested due to difficulty sourcing suitable test curves. The MUMFORD_SEARCH and DLP modes currently only support genus 2.
6. **Very high-degree fibrations** (deg > 15) may require increased `MAX_MODULUS`.
7. The **posterior completeness estimator** is unreliable and should not be used to conclude the search has found all points.

---

## Troubleshooting

**Problem**: Search finds 0 points despite known solutions.
- Ensure *p = 3* is in `PRIME_POOL` (essential for small-denominator reconstruction).
- Increase `HEIGHT_BOUND` (try 2× current value).
- Increase `NUM_PRIME_SUBSETS` (try 2000+).
- Check for *y = 0* points (auto-detected, but may need manual verification).

**Problem**: `"Cannot clear rational denominators"` errors.
- Expected behavior: prime *p* divides a coefficient denominator and is auto-skipped. If too many primes fail, check your curve coefficients for large common factors.

**Problem**: MUMFORD_SEARCH returns a basis smaller than the expected rank.
- The Mumford basis gives a lower bound. Confirm the 2-Selmer upper bound matches, or increase `NUM_PRIME_SUBSETS` and `HEIGHT_BOUND`.

---

## Testing

Test curves (sourced from LMFDB) are provided in `search_common.py`. Run the default test:

```bash
sage search7_genus2.sage
```

Notable test cases:
- `example.txt` – genus 2, rank 1, 12 rational points recovered.
- `example3.txt` – rank-4 curve, full point set found.
- `example6.txt` – DLP mode, secret key recovered.

---

## References

### Theory

I taught myself all of this using AI, so, uhhhhh....

uhh

Read Silverman's textbooks I guess.

TRANS RIGHTS! 🏳️‍⚧️

### Implementation Notes

- **LLL reduction**: SageMath built-in (δ = 0.98 for strong reduction), with BKZ postprocessing.
- **CRT**: cached with `functools.lru_cache` for repeated queries.
- **Rational reconstruction**: extended Euclidean algorithm with bounded denominators.
- **Block Wiedemann**: sparse linear algebra over ℤ/ℓℤ via `sparse_linalg_modp.py`.
- **Archimedean height pairings**: numerical integration over the Jacobian period lattice (`homology.py`, `periods.py`).

---

## Contributing

Issues and pull requests welcome! Areas for improvement:

- [ ] Stable multi-point fibrations (currently may hang)
- [ ] Complete and stabilize `walker.py` / DLP collision walk (currently barely functional)
- [ ] Full genus 3+ Mumford basis and DLP support
- [ ] GPU acceleration for parallel prime subsets
- [ ] Better heuristics for `HEIGHT_BOUND` estimation
- [ ] Improved completeness certification (current posterior estimator is unreliable)
- [ ] Integration with LMFDB for automated curve verification

---

## License

GNU General Public License v3 (or later)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genus2_arithmetic,
  title  = {Genus 2 Curve Arithmetic System},
  author = {Claire Hoying},
  year   = {2025},
  url    = {https://github.com/goracle/elliptic-fibration-search}
}
```

---

## Acknowledgments

Built with SageMath and curve data from [LMFDB.org](https://lmfdb.org).

Built with generous assistance from LLMs including:
ChatGPT (4o, 5, 3.5, ...); Gemini 2.5 Pro, 2.5 Flash; Claude 3.5, 4, 4.5 — and to a lesser extent Perplexity, Qwen, and Grok.

Special thanks to the computational number theory community for foundational algorithms.

---

**Status**: Active development / Testing Needed &nbsp;|&nbsp; **Last updated**: March 2026

For questions or collaboration: [daniel DOT hoying AT uconn.edu]
