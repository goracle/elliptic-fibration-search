import random
from sage.all import Integer, Zmod
from sage.matrix.berlekamp_massey import berlekamp_massey
from multiprocessing import Pool, cpu_count
from math import gcd
from multiprocessing import cpu_count
import sys
import time
from sage.all import Integer, Zmod, vector, GF, PolynomialRing
from sage.all import factor, vector
from sage.all import Integer, Zmod, factor, vector


class SparseRelationMatrix:
    def __init__(self, rows, rhs, modulus):
        """
        rows: list of dicts {col: coeff}
        rhs:  list of ints
        modulus: prime ℓ
        """
        self.mod = int(modulus)
        self.n_rows = len(rows)
        self.n_cols = max(
            max(r.keys()) if r else 0 for r in rows
        ) + 1

        # Pack rows as (indices, values) tuples for faster access
        self.packed_rows = []
        for r in rows:
            if r:
                idxs = list(r.keys())
                vals = [int(v) % self.mod for v in r.values()]
                self.packed_rows.append((idxs, vals))
            else:
                self.packed_rows.append(([], []))

        # Build column-wise view for transpose matvec
        self.packed_cols = [[] for _ in range(self.n_cols)]
        for i, (idxs, vals) in enumerate(self.packed_rows):
            for j, v in zip(idxs, vals):
                self.packed_cols[j].append((i, v))


def parallel_matvec(packed_rows, vec, mod, pool):
    nprocs = pool._processes
    chunks = [[] for _ in range(nprocs)]
    for i, r in enumerate(packed_rows):
        chunks[i % nprocs].append((i, r))

    parts = pool.map(
        _matvec_chunk,
        [(chunk, vec, mod) for chunk in chunks]
    )

    out = [0] * len(packed_rows)
    for chunk_idx, part in enumerate(parts):
        chunk = chunks[chunk_idx]
        for local_idx, v in enumerate(part):
            if v:
                actual_i = chunk[local_idx][0]
                out[actual_i] = v
    return out


def _matvec_chunk(args):
    rows, vec, mod = args
    out = [0] * len(rows)
    for local_idx, (i, (idxs, vals)) in enumerate(rows):
        s = 0
        for j, v in zip(idxs, vals):
            s += v * vec[j]
        out[local_idx] = s % mod
    return out


def parallel_transpose_matvec(packed_cols, vec, mod, n, pool):
    nprocs = pool._processes
    chunks = [[] for _ in range(nprocs)]
    for j, col in enumerate(packed_cols):
        chunks[j % nprocs].append((j, col))
    
    parts = pool.map(
        _transpose_matvec_chunk,
        [(chunk, vec, mod) for chunk in chunks]
    )
    
    out = [0] * n
    for chunk_idx, part in enumerate(parts):
        chunk = chunks[chunk_idx]
        for local_idx, v in enumerate(part):
            if v:
                actual_j = chunk[local_idx][0]
                out[actual_j] = v
    return out


def _transpose_matvec_chunk(args):
    cols, vec, mod = args
    out = [0] * len(cols)
    for local_idx, (j, col) in enumerate(cols):
        s = 0
        for i, c in col:
            s += c * vec[i]
        out[local_idx] = s % mod
    return out


# Replace the old parallel matvecs and block solver with these faster single-process versions.


# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints


def matvec_rows(packed_rows, vec, mod, lazy_limit=_LAZY_LIMIT):
    """
    Compute y = A * vec, where packed_rows is list of (idxs, vals).
    Returns list length = number of rows.
    Single-process, minimal overhead, lazy reduction.
    """
    m = len(packed_rows)
    out = [0] * m
    for i, (idxs, vals) in enumerate(packed_rows):
        s = 0
        for j, a in zip(idxs, vals):
            s += a * vec[j]
            if s > lazy_limit:
                s %= mod
        out[i] = s % mod
    return out


def at_a_v_from_packed(packed_rows, vec, n_cols, mod, lazy_limit=_LAZY_LIMIT):
    """
    Convenience: compute A^T(A v) without projection (if you need it).
    """
    # Reuse compute_proj_and_atav with a zero left_vec to avoid double code
    zero_left = [0] * len(packed_rows)
    _, atav = compute_proj_and_atav(packed_rows, vec, zero_left, n_cols, mod, lazy_limit)
    return atav


# Example wrapper that keeps your solve_dlp_mod_l_block_wiedemann outer flow, but uses the new fast core:
def block_wiedemann_solve_wrapper(A, b, block_size=32, iters=None, verbose=True):
    """
    Thin wrapper preserving previous call signature.
    """
    return block_wiedemann_solve(A, b, block_size=block_size, iters=iters, verbose=verbose)


from sage.all import Integer, Zmod, vector

# [Keep your existing SparseRelationMatrix class and helper functions: 
# parallel_matvec, _matvec_chunk, etc. They are fine.]

# [Keep compute_proj_and_atav and at_a_v_from_packed]


def compute_proj_and_atav(packed_rows, vec, left_vec_b, n_cols, mod, lazy_limit=_LAZY_LIMIT):
    """
    Fused computation of:
      s = A * vec   (row-length vector)
      proj = left_vec_b^T * s   (single scalar)
      atav = A^T * s           (length n_cols vector)

    Optimized with local variable binding for the hot loop.
    """
    # initialize A^T(A v) output
    atav = [0] * n_cols
    proj_acc = 0
    
    # Local bindings for speed in the tight loop
    atav_loc = atav
    mod_loc = mod
    lazy_loc = lazy_limit
    
    for (idxs, vals), b_i in zip(packed_rows, left_vec_b):
        # compute row-dot
        s = 0
        for j, a in zip(idxs, vals):
            s += a * vec[j]
            if s > lazy_loc:
                s %= mod_loc
        s %= mod_loc

        # accumulate projection
        if b_i:
            proj_acc += b_i * s
            if proj_acc > lazy_loc:
                proj_acc %= mod_loc

        # scatter into atav: out[j] += a * s
        # Optimization: manual inline loop with local vars
        for j, a in zip(idxs, vals):
            val = atav_loc[j] + a * s
            if val > lazy_loc:
                val %= mod_loc
            atav_loc[j] = val

    proj = proj_acc % mod
    # final modular reduction on atav
    for j in range(n_cols):
        if atav[j]:
            atav[j] %= mod
    return proj, atav


def solve_dlp_mod_l_block_wiedemann(
    valid_rows,
    rhs_values,
    row_q_dict,
    beta_q,
    full_order,
    G, Q,
    *,
    verbose=True,
    block_size=32,
    nprocs=None,
    max_iters=None
):
    """
    Sparse Block-Wiedemann solver for DLP modulo the largest prime ℓ.
    Wrapper updated to include timing and flush calls.
    """
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    if verbose:
        print(f"  [BW] block_size={block_size}, nprocs={nprocs}")
        sys.stdout.flush()

    # Project to ℓ-subgroup
    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(p for p, _ in factors))

    h = int(J_order // ell)

    if verbose:
        print(f"  [BW] Working mod ℓ={ell}, cofactor h={h}")
        sys.stdout.flush()

    # Project RHS (convert to plain ints)
    beta_q_l = int((beta_q * h) % ell)
    row_q_l = {k: int((v * h) % ell) for k, v in row_q_dict.items()}

    # Project relations
    projected_rows = []
    projected_rhs = []

    for row, rhs in zip(valid_rows, rhs_values):
        new_row = {}
        for k, v in row.items():
            vk = int((v * h) % ell)
            if vk:
                new_row[k] = vk
        if new_row:
            projected_rows.append(new_row)
            projected_rhs.append(int((rhs * h) % ell))

    assert projected_rows, "Block-Wiedemann: no nonzero projected relations"

    n_cols = max(k for row in projected_rows for k in row) + 1

    if verbose:
        nnz = sum(len(r) for r in projected_rows)
        print(f"  [BW] Sparse system: {len(projected_rows)} x {n_cols}, nnz={nnz}")
        sys.stdout.flush()

    # Build SparseRelationMatrix
    A = SparseRelationMatrix(projected_rows, projected_rhs, ell)

    # Block-Wiedemann solve
    if verbose:
        print("  [BW] Starting Krylov iterations...")
        sys.stdout.flush()

    t0 = time.time()
    solution = block_wiedemann_solve(
        A=A,
        b=projected_rhs,
        block_size=block_size,
        iters=max_iters,
        verbose=verbose,
    )
    dt = time.time() - t0

    assert solution is not None, "Block-Wiedemann failed to converge"
    
    if verbose:
        print(f"  [BW] Solved in {dt:.2f}s")
        sys.stdout.flush()

    # Recover discrete log
    dlog = beta_q_l
    for k, v in row_q_l.items():
        dlog = (dlog - v * int(solution[k])) % ell

    if verbose:
        print("  [BW] Candidate dlog mod ℓ =", dlog)
        sys.stdout.flush()

    # Verify
    #assert (dlog * G - Q).order() % ell == 0, "Block-Wiedemann solution failed verification"
    D = dlog * G - Q
    assert (ell * D).is_zero()

    return int(dlog)


# [Keep your existing helper functions: compute_proj_and_atav, at_a_v_from_packed, etc.]


def block_wiedemann_solve(A, b, block_size=32, iters=None, verbose=True, ntrials=1):
    mod = int(A.mod)
    m = len(A.packed_rows)
    n = A.n_cols
    if iters is None:
        iters = int(2.2 * n // max(1, block_size)) + 50

    if verbose:
        print(f"[BW-fast] block={block_size}, target_iters={iters}, nrows={m}, ncols={n}")
        sys.stdout.flush()

    left_vec_b = [int(x) % mod for x in b]
    seed_val = random.randrange(1, mod)
    
    # --- PASS 1 ---
    if verbose:
        print("[BW-fast] Pass 1: Generating Sequence (Trace)")
        sys.stdout.flush()
    
    rng_state = random.getstate()
    random.seed(seed_val)
    V = [[random.randrange(mod) for _ in range(n)] for _ in range(block_size)]
    random.setstate(rng_state)

    seq = []
    t_start = time.time()
    last_print = t_start

    for t in range(iters):
        now = time.time()
        if verbose and (now - last_print > 5):
            elapsed = now - t_start
            rate = (t + 1) / max(1e-9, elapsed)
            remaining = (iters - t) / rate if rate > 0 else 0
            print(f"  [BW Pass 1] iter {t}/{iters} ({100.0*t/iters:.1f}%) | elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m")
            sys.stdout.flush()
            last_print = now

        AV_projs = []
        AVs_atav = []
        for v in V:
            proj, atav = compute_proj_and_atav(A.packed_rows, v, left_vec_b, n, mod)
            AV_projs.append(proj)
            AVs_atav.append(atav)
        seq.extend(AV_projs)
        V = AVs_atav

    # --- POLYNOMIAL STEP ---
    if verbose:
        print(f"[BW-fast] Computing Minimal Polynomial from {len(seq)} scalars...")
        sys.stdout.flush()
    
    K = GF(mod)
    seq_mod = [K(s) for s in seq]
    poly = berlekamp_massey(seq_mod)
    coeffs = [int(c) for c in poly.list()]
    
    if verbose:
        print(f"[BW-fast] Minimal polynomial degree: {poly.degree()}")
        sys.stdout.flush()
    
    # --- PASS 2 ---
    if verbose:
        print("[BW-fast] Pass 2: Reconstructing Solution Vector")
        sys.stdout.flush()

    random.seed(seed_val)
    V = [[random.randrange(mod) for _ in range(n)] for _ in range(block_size)]
    random.setstate(rng_state)

    x_accum = [0] * n
    reconstruct_iters = (len(coeffs) + block_size - 1) // block_size
    t_start = time.time()
    last_print = t_start
    
    for t in range(reconstruct_iters):
        now = time.time()
        if verbose and (now - last_print > 5):
             elapsed = now - t_start
             rate = (t + 1) / max(1e-9, elapsed)
             remaining = (reconstruct_iters - t) / rate if rate > 0 else 0
             print(f"  [BW Pass 2] iter {t}/{reconstruct_iters} ({100.0*t/reconstruct_iters:.1f}%) | elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m")
             sys.stdout.flush()
             last_print = now

        base_idx = t * block_size
        for i in range(block_size):
            c_idx = base_idx + i
            if c_idx < len(coeffs):
                c = coeffs[c_idx]
                if c != 0:
                    vec = V[i]
                    for j in range(n):
                        if vec[j]:
                            x_accum[j] = (x_accum[j] + c * vec[j]) % mod
        
        if t < reconstruct_iters - 1:
            V = [at_a_v_from_packed(A.packed_rows, v, n, mod) for v in V]

    return vector(Zmod(mod), x_accum)
