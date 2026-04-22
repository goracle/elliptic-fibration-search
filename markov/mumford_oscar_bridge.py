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
from collections import defaultdict
from typing import Any

from sage.all import GF, QQ

# These stay in Python — we do NOT port them.
from search_lll.mumford.mumford_solver import solve_mumford_mod_p_optimized
from search_lll.mumford.mumford_verification import verify_mumford_pair
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


def _call_julia_server(prime_list, tasks_by_prime, rhs_by_prime, debug=False):
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

    payload = json.dumps({
        "prime_list": list(prime_list),
        "tasks":      tasks_serial,
        "rhs":        rhs_serial,
    }, cls=_SageEncoder)

    with _SERVER_LOCK:
        proc = _get_server()

        if debug:
            print(f"[oscar_bridge] sending task to persistent Julia server (pid={proc.pid})", flush=True)

        try:
            print(f"[oscar_bridge] writing {len(payload)} bytes to Julia stdin", flush=True)
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            print(f"[oscar_bridge] waiting for Julia stdout readline...", flush=True)
            # Loop past any blank lines — Oscar/JSON3 may emit a stray newline
            # to stdout during module loading that gets flushed before the first
            # real response line.
            result_line = ""
            while True:
                result_line = proc.stdout.readline()
                if not result_line or result_line.strip():
                    break
                print(f"[oscar_bridge] skipping blank line from Julia stdout", flush=True)
            print(f"[oscar_bridge] readline returned {len(result_line)} bytes: {repr(result_line[:200])}", flush=True)
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
    print(f"[oscar_bridge] result_line stripped: {repr(result_line[:200])}", flush=True)
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

    # Phase 2: persistent Julia server call
    t1 = time.time()

    julia_raw_str = _call_julia_server(prime_list, tasks_by_prime, rhs_by_prime, debug=debug)

    # The server returns string keys: {str(p): {str(v_tuple): {str([x,r]): [[m,x,r],...]}}}
    # Convert back to the typed keys _assemble_results expects:
    #   {p(int): {v_tuple(tuple): {(x_val,rhs_idx)(tuple): [(m,x,r),...]}}}
    import ast
    julia_raw = {}
    for p_str, vmap in json.loads(julia_raw_str).items():
        p = int(p_str)
        p_result = {}
        for v_str, xmap in vmap.items():
            v_tuple = tuple(ast.literal_eval(v_str))
            xr_dict = {}
            for xr_str, triples in xmap.items():
                xr = ast.literal_eval(xr_str)
                xr_dict[(int(xr[0]), int(xr[1]))] = [tuple(t) for t in triples]
            p_result[v_tuple] = xr_dict
        julia_raw[p] = p_result

    if debug:
        total = sum(
            len(xmap) for pmap in julia_raw.values() for xmap in pmap.values()
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
