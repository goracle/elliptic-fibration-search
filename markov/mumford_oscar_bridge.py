"""
mumford_oscar_bridge.py

Drop-in replacement for mumford_precompute_residues_parallel that offloads
the hot polynomial root-finding loop to Julia/Oscar while keeping everything
else (task generation, solve_mumford_mod_p_optimized, verify_mumford_pair,
sign computation, result assembly) in Python unchanged.

Usage
-----
In mumford_parallel.py, replace the call to mumford_precompute_residues_parallel
with a call to mumford_precompute_residues_oscar.  The signature and return type
are identical.

    # was:
    mumford_residues = mumford_precompute_residues_parallel(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list, num_workers=num_workers, ...
    )

    # now:
    from mumford_oscar_bridge import mumford_precompute_residues_oscar
    mumford_residues = mumford_precompute_residues_oscar(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list, ...
    )

Julia startup
-------------
On first call, Julia is started once and mumford_oscar.jl is loaded.
Subsequent calls reuse the same Julia session (juliacall keeps it alive for
the lifetime of the Python process).

Install juliacall:
    pip install juliacall
Then ensure Oscar is installed in the Julia environment Julia will find
(set JULIA_PROJECT or use the default environment with Oscar added).
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any

from sage.all import GF, QQ

# These stay in Python — we do NOT port them.
from search_lll.mumford.mumford_solver import solve_mumford_mod_p_optimized
from search_lll.mumford.mumford_verification import verify_mumford_pair
from search_common import DEBUG, FINITE_FIELD

# ---------------------------------------------------------------------------
# Julia session (lazy init)
# ---------------------------------------------------------------------------

_jl = None
_jl_ready = False

def _ensure_julia(oscar_jl_path: str = "/home/claire/elliptic-fibration-search/markov/mumford_oscar.jl") -> Any:
    """Start Julia and load mumford_oscar.jl exactly once."""
    global _jl, _jl_ready
    if _jl_ready:
        return _jl
    try:
        from juliacall import Main as jl  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "juliacall not installed. Run: pip install juliacall\n"
            "Then make sure Oscar.jl is available in your Julia environment."
        )
    jl.include(oscar_jl_path)
    _jl = jl
    _jl_ready = True
    return _jl


# ---------------------------------------------------------------------------
# Task building (reused from the existing loop in mumford_precompute_residues_parallel)
# ---------------------------------------------------------------------------

def _build_tasks_and_rhs(
    eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
    rhs_modp_list, vecs_list, const_val_int, chunk_size, debug,
):
    """
    Reproduce the task-generation loop from mumford_precompute_residues_parallel
    verbatim, but instead of packing tasks into multiprocessing batches we
    return two dicts:

      tasks_by_prime[p]  : list of (v_tuple, diff_coeffs_ints, rhs_idx)
      rhs_by_prime[p]    : list of (num_coeffs, den_coeffs)   [rhs_reconstruction]
    """
    tasks_by_prime = {}
    rhs_by_prime = {}

    for p in prime_list:
        assert p in Ep_dict, f"Prime {p} missing from Ep_dict"

        Ep = Ep_dict[p]
        p_vecs = vecs_lll.get(p)
        assert p_vecs is not None, f"Prime {p} missing from vecs_lll"
        assert len(p_vecs) >= len(vecs_list), (
            f"Prime {p}: vecs_lll shorter than vecs_list "
            f"({len(p_vecs)} < {len(vecs_list)})"
        )

        Fp = GF(p)
        R_m = Fp["m"]

        # Build rhs_reconstruction for this prime (identical to original)
        rhs_polys_for_p = []
        rhs_reconstruction = []

        for rhs_dict in rhs_modp_list:
            rhs_val = rhs_dict.get(p)
            if rhs_val is not None:
                try:
                    num_poly = R_m(rhs_val.numerator())
                    den_poly = R_m(rhs_val.denominator())
                    rhs_polys_for_p.append(num_poly / den_poly)
                    num_coeffs = [int(c) % p for c in num_poly.list()]
                    den_coeffs = [int(c) % p for c in den_poly.list()]
                    rhs_reconstruction.append((num_coeffs or [0], den_coeffs or [0]))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to build rhs_reconstruction: p={p}, error={e}"
                    )

        if not rhs_polys_for_p:
            m_var = R_m.gen()
            rhs_polys_for_p = [-m_var + Fp(const_val_int)]
            rhs_reconstruction = [([const_val_int % p, p - 1], [1])]

        assert len(rhs_polys_for_p) == len(rhs_reconstruction)
        rhs_by_prime[p] = rhs_reconstruction

        # Build chunk items (identical polynomial-extraction logic)
        p_mults = mult_lll.get(p, {})
        items = []

        for v_idx, v_tuple in enumerate(vecs_list):
            if not v_tuple:
                continue

            v_coeffs = p_vecs[v_idx]
            Pm = Ep(0)
            valid_vec = True

            for i, c in enumerate(v_coeffs):
                k = int(c)
                if k == 0:
                    continue
                try:
                    mults_for_sec = p_mults[i]
                    if k in mults_for_sec:
                        Pm += mults_for_sec[k]
                    else:
                        valid_vec = False
                        break
                except (IndexError, KeyError, TypeError) as e:
                    raise RuntimeError(
                        f"Failed to build section multiple: p={p}, v_idx={v_idx}, "
                        f"i={i}, k={k}, error={e}"
                    )

            if not valid_vec:
                continue
            if Pm[2] == 0:
                continue
            if hasattr(Pm, "is_zero") and Pm.is_zero():
                continue

            for rhs_idx, rhs_poly in enumerate(rhs_polys_for_p):
                try:
                    diff = Pm[0] - Pm[2] * rhs_poly
                    diff_num = diff.numerator()
                    if diff_num.is_zero():
                        continue
                    coeffs_ints = [int(c) for c in diff_num.list()]
                    items.append((v_tuple, coeffs_ints, rhs_idx))
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to extract polynomial: p={p}, v_idx={v_idx}, "
                        f"v_tuple={v_tuple}, rhs_idx={rhs_idx}, error={e}"
                    )

        assert items, f"No tasks generated for p={p} — configuration error"
        tasks_by_prime[p] = items

    return tasks_by_prime, rhs_by_prime


# ---------------------------------------------------------------------------
# Post-processing: call solve_mumford_mod_p_optimized on Julia's output
# ---------------------------------------------------------------------------

def _assemble_results(
    julia_raw,
    f_coeffs_ints,
    const_val_int,
    prime_list,
    debug,
):
    """
    Given julia_raw = {p: {v_tuple: {(x_val, rhs_idx): [(m_root, x_val, rhs_idx), ...]}}}
    call solve_mumford_mod_p_optimized + verify_mumford_pair + sign computation
    and return the final results_dict in the format mumford_precompute_residues_parallel
    would have returned:
      {p: {v_tuple: {(x_val, rhs_idx): [verified_sol_6tuple, ...]}}}
    """
    results_dict = {}
    max_sols = 10000 if FINITE_FIELD else 500

    for p in prime_list:
        p_raw = julia_raw.get(p)
        if not p_raw:
            continue

        p_results = {}

        for v_tuple, xmap in p_raw.items():
            for (x_val, rhs_idx), triples in xmap.items():
                verified_sols = []

                for (m_root, x_val_inner, _rhs_idx) in triples:
                    # x_val_inner == x_val (redundant, but keep for clarity)
                    try:
                        sols = solve_mumford_mod_p_optimized(
                            f_coeffs_ints, p, x_val, const_val_int,
                            max_solutions=max_sols,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Mumford solver failed: p={p}, x_val={x_val}, "
                            f"m_root={m_root}, v_tuple={v_tuple}, rhs_idx={rhs_idx}, error={e}"
                        )

                    for sol in sols:
                        assert len(sol) == 4, f"Invalid solution length: {len(sol)}"
                        s, p_val, v0, v1 = sol

                        if not verify_mumford_pair(
                            f_coeffs_ints, s, p_val, v0, v1, modulus=p
                        ):
                            raise RuntimeError(
                                f"Mumford pair failed verification: "
                                f"p={p}, sol={sol}, v_tuple={v_tuple}, rhs_idx={rhs_idx}"
                            )

                        # Sign computation (identical to _solve_worker_wrapper)
                        xv_v = (v0 + v1 * x_val) % p
                        rhs_val = 0
                        for i, c in enumerate(f_coeffs_ints):
                            rhs_val = (rhs_val + c * pow(x_val, i, p)) % p

                        if rhs_val == 0:
                            canonical_xv = 0
                        elif (p % 4) == 3:
                            canonical_xv = pow(rhs_val, (p + 1) // 4, p)
                            canonical_xv = min(canonical_xv, p - canonical_xv)
                        else:
                            sq = pow(rhs_val, (p + 1) // 4, p)
                            if (sq * sq) % p == rhs_val:
                                canonical_xv = min(sq, p - sq)
                            else:
                                canonical_xv = min(xv_v, p - xv_v) if xv_v != 0 else 0

                        xv_canonical = min(xv_v, p - xv_v) if xv_v != 0 else 0
                        x_val_sign = 1 if xv_canonical == canonical_xv else -1

                        verified_sols.append(
                            (sol, x_val_sign, int(v0), int(v1), int(m_root), int(rhs_idx))
                        )

                if verified_sols:
                    if v_tuple not in p_results:
                        p_results[v_tuple] = {}
                    p_results[v_tuple][(x_val, rhs_idx)] = verified_sols

        if p_results:
            results_dict[p] = p_results

    return results_dict


# ---------------------------------------------------------------------------
# Public API: drop-in replacement
# ---------------------------------------------------------------------------

def mumford_precompute_residues_oscar(
    eqs_dict,
    prime_list,
    Ep_dict,
    mult_lll,
    vecs_lll,
    rhs_modp_list,
    vecs_list,
    num_workers=None,   # ignored — Julia handles parallelism via @threads
    debug=DEBUG,
    chunk_size=4,
    pool=None,          # ignored
    oscar_jl_path="/home/claire/elliptic-fibration-search/markov/mumford_oscar.jl",
):
    """
    Drop-in replacement for mumford_precompute_residues_parallel.

    Identical call signature.  Offloads polynomial root-finding to Julia/Oscar
    (multithreaded via @threads over primes) and keeps solve_mumford_mod_p_optimized,
    verify_mumford_pair, and sign computation in Python.

    Returns the same nested dict as the original.
    """
    assert isinstance(eqs_dict, dict) and "f_coeffs" in eqs_dict and "const" in eqs_dict, \
        "Invalid eqs_dict: must contain 'f_coeffs' and 'const'"
    assert prime_list, "Empty prime_list"
    assert Ep_dict, "Empty Ep_dict"
    assert vecs_list, "Empty vecs_list"

    jl = _ensure_julia(oscar_jl_path)

    f_coeffs = eqs_dict["f_coeffs"]
    f_coeffs_ints = [int(c) for c in f_coeffs]
    const_val_int = int(QQ(eqs_dict["const"]))

    t0 = time.time()

    # Phase 1: build tasks (Python, same logic as before)
    tasks_by_prime, rhs_by_prime = _build_tasks_and_rhs(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list, const_val_int, chunk_size, debug,
    )

    if debug:
        print(f"[oscar_bridge] task generation: {time.time()-t0:.2f}s", flush=True)

    # Phase 2: convert to Julia types and call Oscar
    t1 = time.time()

    # Convert tasks_by_prime: {p: [(v_tuple, diff_coeffs, rhs_idx), ...]}
    # → Julia Dict{Int, Vector{ChunkItem}} via build_chunk_items
    jl_tasks = {}
    for p, items in tasks_by_prime.items():
        jl_tasks[p] = jl.build_chunk_items(items)

    # Convert rhs_by_prime: {p: [(num_coeffs, den_coeffs), ...]}
    # juliacall passes Python tuples/lists as Julia Vectors/Tuples automatically
    jl_rhs = rhs_by_prime   # juliacall handles the conversion

    julia_raw = jl.mumford_residues_oscar(
        list(prime_list),
        jl_tasks,
        jl_rhs,
    )

    if debug:
        total = sum(
            len(vmap) for pmap in julia_raw.values() for vmap in pmap.values()
        )
        print(
            f"[oscar_bridge] Julia root-finding: {time.time()-t1:.2f}s  "
            f"({total} (x_val, rhs_idx) hits across all primes)",
            flush=True,
        )

    # Phase 3: Mumford solve + verification (Python, unchanged)
    t2 = time.time()
    results_dict = _assemble_results(
        julia_raw, f_coeffs_ints, const_val_int, prime_list, debug,
    )

    if debug:
        print(f"[oscar_bridge] Mumford solve + verification: {time.time()-t2:.2f}s", flush=True)
        print(f"[oscar_bridge] total: {time.time()-t0:.2f}s", flush=True)

    assert results_dict, "Oscar bridge returned empty results — check Julia output"
    return results_dict
