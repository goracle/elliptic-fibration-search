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

import json
import os
import pathlib
import subprocess
import sys
import time
import ast
from collections import defaultdict
from typing import Any

from sage.all import GF, QQ

from search_common import DEBUG, FINITE_FIELD

# ---------------------------------------------------------------------------
# Julia subprocess call (avoids juliacall in-process GAP/libgap conflict
# with SageMath's own libgap)
# ---------------------------------------------------------------------------

# Path to mumford_oscar_server.jl — sits next to this file by default.
_SERVER_JL = pathlib.Path(__file__).resolve().parent / "mumford_oscar_server.jl"

# ---------------------------------------------------------------------------
# Persistent Julia server process — started once, reused for every call.
# Pays the Oscar/FLINT JIT cost exactly once per Python process lifetime.
# ---------------------------------------------------------------------------

import threading

_SERVER_PROC: subprocess.Popen | None = None
_SERVER_LOCK = threading.Lock()


def _drain_stderr(proc: subprocess.Popen) -> None:
    """Forward Julia stderr to Python stderr so errors are never silently swallowed."""
    for line in proc.stderr:
        print(f"[julia] {line}", end="", flush=True)


def _get_server() -> subprocess.Popen:
    """Return the live Julia server process, starting it if necessary."""
    global _SERVER_PROC
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        return _SERVER_PROC

    server_path = os.environ.get("OSCAR_SERVER_JL", str(_SERVER_JL))
    julia_bin   = os.environ.get("JULIA_BIN", "julia")
    nthreads    = os.environ.get("JULIA_NUM_THREADS", "auto")

    cmd = [julia_bin, f"--threads={nthreads}", "--startup-file=no", server_path]
    _SERVER_PROC = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered — required so readline() doesn't block forever
    )

    # Block until Julia finishes loading Oscar and prints "[julia] server ready".
    # We read stderr directly here (no drain thread yet) so we can't miss it.
    print("[oscar_bridge] waiting for Julia server to load Oscar...", flush=True)
    for line in _SERVER_PROC.stderr:
        sys.stderr.write(f"[julia] {line}")
        sys.stderr.flush()
        if "[julia] server ready" in line:
            break
        if _SERVER_PROC.poll() is not None:
            raise RuntimeError(
                f"[oscar_bridge] Julia server exited during startup "
                f"(code {_SERVER_PROC.returncode})"
            )
    print("[oscar_bridge] Julia server ready.", flush=True)

    # Hand off ongoing stderr to a drain thread.
    threading.Thread(target=_drain_stderr, args=(_SERVER_PROC,), daemon=True).start()
    return _SERVER_PROC


class _SageEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return int(obj)
        except (TypeError, ValueError):
            pass
        return super().default(obj)


def _call_julia_server(prime_list, tasks_by_prime, rhs_by_prime,
                       section_poly_dict=None, debug=False):
    """
    Serialize tasks to JSON, send to the persistent Julia server, and return
    the raw JSON result string (one newline-delimited line).

    The server loops forever reading one JSON line from stdin and writing one
    JSON line to stdout — Oscar JIT happens once at startup, not per call.

    Returns: str  (raw JSON, parsed by the caller)
    """
    # v_tuples are Python tuples — must become JSON arrays.
    tasks_serial = {
        str(p): [
            [list(v_tuple), diff_coeffs, rhs_idx]
            for v_tuple, diff_coeffs, rhs_idx in items
        ]
        for p, items in tasks_by_prime.items()
    }
    rhs_serial = {
        str(p): [[list(num), list(den)] for num, den in entries]
        for p, entries in rhs_by_prime.items()
    }

    # Include per-prime section poly payloads when provided.
    # section_poly_dict: {p: [{"p":…,"D":…,"X":…,"Y":…,"Z":…,"a4":…,"a6":…}, …]}
    # Julia reads this under "section_polys" and runs run_section_ladder per section.
    section_serial = {}
    if section_poly_dict:
        for p, payloads in section_poly_dict.items():
            non_null = [pl for pl in payloads if pl is not None]
            if non_null:
                section_serial[str(p)] = non_null

    payload = json.dumps({
        "prime_list":    list(prime_list),
        "tasks":         tasks_serial,
        "rhs":           rhs_serial,
        "section_polys": section_serial,   # may be {}
    }, cls=_SageEncoder)

    with _SERVER_LOCK:
        proc = _get_server()

        if debug:
            print(f"[oscar_bridge] sending task to persistent Julia server (pid={proc.pid})", flush=True)

        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            # Skip blank lines emitted to stdout during Oscar/JSON3 module
            # loading — they sit buffered until the first flush() and arrive
            # ahead of the first real JSON response line.
            result_line = ""
            while True:
                result_line = proc.stdout.readline()
                if not result_line or result_line.strip():
                    break
        except BrokenPipeError:
            global _SERVER_PROC
            _SERVER_PROC = None
            raise RuntimeError("[oscar_bridge] Julia server pipe broken — process may have crashed")

    if not result_line:
        rc = proc.poll()
        raise RuntimeError(
            f"[oscar_bridge] Julia server returned empty response "
            f"(process {'still running' if rc is None else f'exited with code {rc}'}) "
            f"— check [julia] lines above for the error"
        )

    result_line = result_line.strip()
    if result_line == "[julia] ERROR":
        raise RuntimeError(
            "[oscar_bridge] Julia server reported an error — check [julia] ERROR lines above"
        )

    return result_line


# ---------------------------------------------------------------------------
# Task building (reused from the existing loop in mumford_precompute_residues_parallel)
# ---------------------------------------------------------------------------

def _build_tasks_and_rhs(
    eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
    rhs_modp_list, vecs_list, const_val_int, chunk_size, debug,
    julia_ladder_caches=None,
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
        # Build a LargePrimeMockCurve once per prime so the per-vector
        # accumulator Pm is type-compatible with LargePrimeMockPoint mults.
        from search_lll.ll_utilities import LargePrimeMockCurve, LargePrimeMockPoint
        _a4_p = Ep.a4() if hasattr(Ep, 'a4') else Ep._a4
        _a6_p = Ep.a6() if hasattr(Ep, 'a6') else Ep._a6
        _base_p = Ep.base_ring() if hasattr(Ep, 'base_ring') else Ep.base_field()
        _mock_curve_p = LargePrimeMockCurve(_base_p, _a4_p, _a6_p)
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
            Pm = _mock_curve_p(0)
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
# Post-processing: convert Julia's string-keyed output to typed Python keys.
# Julia returns: {str(p): {str(v_tuple): {str([rhs_idx]): [m_root, ...]}}}
# We return:     {p(int): {v_tuple(tuple): {rhs_idx(int): [m_root(int), ...]}}}
#
# That's it.  No Mumford solve, no verify, no sign computation — enrich_candidates
# in walkerclass.py does all of that from the fiber geometry using only m_root.
# ---------------------------------------------------------------------------

def _assemble_results(julia_raw, prime_list):
    results_dict = {}
    for p in prime_list:
        p_raw = julia_raw.get(p)
        if not p_raw:
            continue
        p_results = {}
        for v_str, xmap in p_raw.items():
            v_tuple = tuple(ast.literal_eval(v_str))
            rhs_dict = {}
            for rhs_key_str, m_roots in xmap.items():
                rhs_idx = int(ast.literal_eval(rhs_key_str)[0])
                rhs_dict[rhs_idx] = [int(m) for m in m_roots]
            if rhs_dict:
                p_results[v_tuple] = rhs_dict
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
    section_poly_dict=None,   # {p: [payload_per_section]} from prepare_modular_data_lll
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

    const_val_int = int(QQ(eqs_dict["const"]))

    t0 = time.time()

    # Phase 1: build tasks (Python)
    tasks_by_prime, rhs_by_prime = _build_tasks_and_rhs(
        eqs_dict, prime_list, Ep_dict, mult_lll, vecs_lll,
        rhs_modp_list, vecs_list, const_val_int, chunk_size, debug,
    )

    if debug:
        print(f"[oscar_bridge] task generation: {time.time()-t0:.2f}s", flush=True)

    # Phase 2: Julia root-finding
    t1 = time.time()
    julia_raw_str = _call_julia_server(
        prime_list, tasks_by_prime, rhs_by_prime,
        section_poly_dict=section_poly_dict,
        debug=debug,
    )

    # Parse Julia response.
    # New format (when section_polys sent):
    #   {str(p): {"roots": {str(v_tuple): {str([rhs_idx]): [m]}},
    #             "ladder_caches": [{str(k): {"X":[…],"Y":[…],"Z":[…]}}, …]}}
    # Old format (no section_polys): flat {str(p): {str(v_tuple): {…}}}
    _parsed = json.loads(julia_raw_str)
    julia_raw = {}
    julia_ladder_caches = {}   # p(int) -> [cache_per_section]
    for p_str, pval in _parsed.items():
        p_int = int(p_str)
        if isinstance(pval, dict) and "roots" in pval:
            julia_raw[p_int] = pval["roots"]
            lc = pval.get("ladder_caches")
            if lc:
                julia_ladder_caches[p_int] = lc
        else:
            # Old format fallback
            julia_raw[p_int] = pval

    if debug:
        total = sum(
            len(m_list)
            for pmap in julia_raw.values()
            for vmap in pmap.values()
            for m_list in vmap.values()
        )
        lc_count = sum(len(lcs) for lcs in julia_ladder_caches.values())
        print(
            f"[oscar_bridge] Julia root-finding: {time.time()-t1:.2f}s  "
            f"({total} m-roots across all primes, "
            f"{lc_count} ladder cache sections)",
            flush=True,
        )

    # Phase 3: key conversion only — no Mumford solve, no sign computation
    results_dict = _assemble_results(julia_raw, prime_list)

    if debug:
        print(f"[oscar_bridge] total: {time.time()-t0:.2f}s", flush=True)

    assert results_dict, "Oscar bridge returned empty results — check Julia output"
    return results_dict
