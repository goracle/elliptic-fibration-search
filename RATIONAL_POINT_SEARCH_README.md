# Rational Point Search — Orientation for AI Assistants

This document is a focused companion to the top-level `README.md`, written to get
an AI assistant from "here's a raw run log" to "I understand this pipeline enough
to debug or extend it." It walks through **exactly what happens** when
`search7_genus2.sage` runs in rational-point mode (`MUMFORD_SEARCH = False`,
`FINITE_FIELD = None`), in the order the log prints it, and points at the real
source files for each stage. It does not re-derive the math from scratch — see
`README.md`'s "How It Works" section and `docs.pdf`/`docs.tex` for that.

If you were just handed a log like `example7.txt` (or the one this doc was
written against) and asked to explain, debug, or extend the run, start here.

## Where the code actually lives

The top-level `README.md`'s module table lists files like `search_main.py`,
`ll_utilities.py`, `mumford_basis.py`, `index_calculus.py`, `arakelov.py`,
`homology.py`, `walker.py` as if they were flat files in the repo root. **They
are not** — they live under the `search_lll/` package:

```
search_lll/
  search_config.py       # constants / config re-exports used across the package
  search_main.py         # THE driver: run_standard_lattice_search, run_mumford_search,
                          # the index-calculus attack entry point, the anomalous-sweep loop
  ll_utilities.py         # LLL/BKZ vector enumeration, rational reconstruction
  rational_arithmetic.py  # low-level rational/CRT helpers
  modularthread.py         # per-prime-subset worker, _batch_check_rationality
  search_analysis.py       # residue-pattern / discriminative-power analysis, adaptive tuning
  diagnostics_univariate.py
  archimedean_optim.py
  fiber_augment.py, fiber_augment_hdf5.py   # "augment_known" step, HDF5-backed variant
  smoothness.py, index_calculus.py, lp_incidence_dlp.py   # DLP/factor-base mode only
  mumford/, jacobian_basis/                # MUMFORD_SEARCH mode
  homology.py, selmer_genus2.py            # analytic heights, 2-Selmer
  collision_walk.c, libwalk.so, walker.py  # DLP collision walk (experimental)
```

`search7_genus2.sage` does `from search_lll import *`, which pulls all of the
above in via `search_lll/__init__.py`. When grepping for a function named in a
log line, search `search_lll/` first, not the repo root.

There is also a **second, separate consumer** of this same package:
`markov/` (and `markov/walker/`) is the genus-2 index-calculus DLP research
project — it imports directly from `search_lll` (e.g.
`search_lll/search_main.py` itself does
`from markov.mumford_oscar_bridge import mumford_precompute_residues_oscar`).
So `search_lll/` is shared infrastructure between "find rational points on a
curve" and "attack a DLP instance on a curve over a finite field" — a change
here can affect both. If you're only working on rational-point search, you
generally won't need to touch `markov/`, but be aware it exists and imports
from here.

## The pipeline, in the order the log shows it

Given a curve `y² = f(x)` (`COEFFS_GENUS2`) and one seed rational point
(`DATA_PTS_GENUS2`), here is the call sequence:

### 1. Seed point handling (`search7_genus2.sage`, `tower.sage`)
- Checks for trivial `y = 0` points first.
- Seeds `known_pts` with the seed and its negation (e.g. `(0, 1)` and `(0, -1)`).

### 2. Fibration tower construction (`tower.sage`)
- Builds a "2 Step Fibration Tower" (or 1 step, depending on `deg f`) reducing
  the genus-2 curve to a quartic fiber `E(m)` over `ℚ(m)`.
- At each step it scores ~10 candidate auxiliary polynomials and picks the
  best ("Selected geometry").
- `verify_y2_consistency` sanity-checks that the layers glue together
  correctly across several `m` values before proceeding — if this fails,
  the tower itself is wrong and nothing downstream should be trusted.

### 3. Minimal Weierstrass model (`tate.py`, called from `tower.sage`/`search7_genus2.sage`)
- "MINIMAL MODEL COMPUTATION": checks for poles/common zeros at `m = 0`,
  computes blow-up/blow-down exponents, and (if `USE_MINIMAL_MODEL=True`)
  produces a Tate-minimized `a4(m), a6(m)`.
- Reports bad primes (primes dividing the discriminant identically — i.e.
  the fibration is singular there for all `m`, not just special `m`).
- Runs Tate's algorithm per finite place to classify Kodaira fiber types and
  computes the total Euler characteristic (should equal 12 for a rational
  elliptic surface fibration from a genus-2 curve via a 1-point fibration —
  this is a real consistency check, not decoration).

### 4. Auto-configuration (`bounds.py`)
- `auto_configure_search()` picks `HEIGHT_BOUND`, `TMAX`, `NUM_PRIME_SUBSETS`,
  `PRIME_POOL` based on the fibration's discriminant degree, an empirical
  Galois-group estimate (via sampled factorization patterns mod many primes,
  not an exact `galois_group()` call except as a first attempt), and a
  splitting-field-degree estimate.
- **This step is heuristic.** The `[galois/empirical]` and `[bounds] Estimated
  splitting field degree` lines are statistical guesses from residue
  factorization patterns, not proven values. Don't treat them as certified.
- Rejects "bad" primes for the search pool (`is_good_prime_for_surface`) —
  distinct from the Kodaira-fiber bad primes above; this is about which
  primes are usable for CRT/lattice work, not about the surface's geometry.

### 5. Fibration data assembly (`tower.sage` → `BUILDCD`)
- Produces the `CurveDataExt` object (`E_curve`, `E_weier`, `a4`, `a6`,
  `phi_x`, `morphs`, `singfibs`, `bad_primes`, ...) — the central data
  structure everything downstream operates on. See `README.md`'s
  Architecture section for the full field list.
- `verify_morphism` spot-checks that the quartic ↔ Weierstrass coordinate
  transform actually round-trips on sample points.

### 6. LLL vector generation (`search_lll/ll_utilities.py`)
- Builds the height-pairing matrix `H` from the known Mordell–Weil sections
  on the Weierstrass model, then enumerates short lattice vectors up to a
  height bound (`compute_search_vectors`).
- `[lll_reduce] WARNING: invalid height matrix, skipping LLL` is expected and
  harmless when there's only 1 section (rank-1 `H` has nothing to reduce).

### 7. The main search loop (`search_lll/search_main.py: run_standard_lattice_search`)
This is the heart of the pipeline and where most of the log volume comes from:

- **`prepare_modular_data_lll`**: reduces the fibration mod each prime in the
  pool, discarding primes where the reduced curve is singular or a
  denominator vanishes mod `p` (you'll see `Skipping prime 2: ... singular`
  and `skip p=3: a4 numerator has coeff with denom divisible by p` — this is
  routine, not an error).
- **Residue precomputation**: for each surviving prime and each search
  vector, solves `Sᵢ(m) ≡ r(m) (mod p)` and records the roots
  (`compute_residues_for_prime_worker`, parallelized across
  `ProcessPoolExecutor`).
- **Brauer / density estimates** (`brauer.py`): estimates what fraction of
  `m`-candidates will survive local obstructions, used only to set search
  parameters — not a correctness gate.
- **Adaptive subset count** (`search_analysis.py`): estimates empirical
  residue density and recommends `NUM_SUBSETS`; the actual run always
  respects the user's configured value (`num_subsets_to_use = num_subsets`,
  the adaptive value is advisory only — see the comment in
  `run_standard_lattice_search`).
- **The anomalous-residue sweep** (`run_standard_lattice_search`, look for
  `MAX_ANOMALOUS_SWEEP_ROUNDS`): this is the outer loop that produces the
  repeated "Round 0", "Round 1", ... blocks in the log. Each round:
  1. Generates random prime subsets (sizes 3–12) and, per subset, CRT-combines
     residues into `(m, vector)` candidates (`search_prime_subsets_unified`
     → `modularthread.py` workers).
  2. Batch-checks which candidates are actually rational
     (`_batch_check_rationality` — imported explicitly by name in
     `search_main.py` due to a `from module import *` underscore-name
     gotcha documented right there in the code; worth knowing if this
     function ever seems "missing").
  3. Calls `analyze_unused_residue_orders` (`search_analysis.py`) to check
     whether every observed residue across the prime pool is explained by a
     known rational point.
  4. If not, picks the single most-anomalous unexplained residue per prime
     and **forces those primes into every subset** for the next round, to
     bias the CRT search toward finding whatever's producing them.
  5. Stops when either everything is explained, or a forced-prime round
     produces no new points (as in the reference log — primes `[7, 11, 31]`
     were forced and found nothing new, so the sweep stopped), or
     `MAX_ANOMALOUS_SWEEP_ROUNDS` is hit.
- **`augment_known`** (`fiber_augment.py`): takes the newly found x-coordinates
  and derives the corresponding `(x, ±y)` pairs on the original genus-2
  curve, folding them into the cumulative point list.

### 8. Post-search diagnostics (all optional, all in `search7_genus2.sage`
   after the search loop returns)
These run regardless of whether new points were found and are informational,
not required for correctness:
- **CM fiber search**: checks special `j`-invariants (0, 1728, -1728, ...)
  for complex-multiplication fibers.
- **Automorphism search** (`automorph.py`): NS-lattice automorphisms.
- **Torsion analysis** (`torsion.py`): GCD-of-specializations heuristic for
  `J(ℚ)_tors`.
- **Saturation diagnostics** (`sat.py`): per-prime witnesses that the found
  Mordell–Weil sections are `p`-saturated.
- **Picard number / Shioda–Tate** (`picard.py`): Van Luijk's method — a char-0
  lower bound from sections plus Lefschetz-trace upper bounds from several
  reductions mod ℓ. When they match, ρ (and hence the MW rank via
  Shioda–Tate) is *proven*, not estimated. In the reference log this pins
  ρ = 3 exactly.
- **2-Selmer bounds** (`selmer.py`): an independent rank upper bound from
  local descent at bad primes.
- **Completeness posterior** (`stats.py`): a Bayesian "did we find everything"
  estimate from the observed hit rate and an arithmetic prior. **Treat this
  as a rough heuristic only** — `README.md`'s Known Limitations section
  explicitly flags this estimator as unreliable. A 98.5% posterior does not
  mean the search is provably complete; the LLL-guarantee completeness proof
  earlier in the log (`FORMAL COMPLETENESS PROOF`, based on `M_min` vs.
  `H_can`) is the actual rigorous statement, and it only certifies
  completeness up to the height bound actually searched.

## Reading a log quickly

When handed a new log from this pipeline, the fastest orientation path is:
1. Read the curve equation and seed point at the top.
2. Skip to `FIBRATION SUMMARY` for the Weierstrass model actually being
   searched.
3. Skip to `FORMAL COMPLETENESS PROOF` and `Final list of known points` —
   these are the two trustworthy bottom-line results.
4. Only dig into the `anomalous-sweep` / `residue pattern analysis` /
   `posterior` sections if something looks wrong (0 points found, a round
   that doesn't terminate, etc.) — they're diagnostic detail, not the answer.

## Known rough edges specific to this subsystem

- The repo-root `README.md`'s module table gives flat filenames for things
  that actually live in `search_lll/`; don't be misled when grepping.
- `_batch_check_rationality` has a documented dual-definition trap between
  `modularthread.py` and `search_analysis.py` — always trace calls to the
  explicit import in `search_main.py`, not the wildcard one.
- The adaptive `NUM_SUBSETS` recommendation is computed but currently never
  applied (commented out) — if a run seems under- or over-searched, check
  the user's configured `NUM_PRIME_SUBSETS` in `search_common.py` directly
  rather than trusting the `[Adaptive] Recommended` log line.
- Multi-point fibrations (2+ seed points) are unstable / may hang — this is
  a documented limitation, not a bug to chase if you hit it.
