import random
from sage.all import Integer, Zmod
from sage.matrix.berlekamp_massey import berlekamp_massey
from multiprocessing import Pool, cpu_count
from math import ceil, sqrt, gcd
from multiprocessing import cpu_count
import sys
import time
from sage.all import Integer, Zmod, vector, GF, PolynomialRing
from sage.all import factor, vector
from sage.all import Integer, Zmod, factor, vector


# Replace the old parallel matvecs and block solver with these faster single-process versions.


# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints


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


# [Keep your existing helper functions: compute_proj_and_atav, at_a_v_from_packed, etc.]


# Diagnostic helpers for BW verification failures


from sage.all import Integer, Zmod, vector, GF
from math import ceil

# [Keep SparseRelationMatrix, parallel_matvec, _matvec_chunk, parallel_transpose_matvec, _transpose_matvec_chunk]
# [Keep matvec_rows, at_a_v_from_packed, compute_proj_and_atav]


from sage.all import Integer, Zmod, vector, GF, PolynomialRing, matrix
from sage.all import factor

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


def block_wiedemann_solve(A, b, block_size=1, iters=None, verbose=True, ntrials=1):
    mod = int(A.mod)
    m = len(A.packed_rows)
    n = A.n_cols
    
    # Wiedemann complexity: ~2N iterations for scalar, 2N/B for blocked
    # We add a small safety buffer (+50)
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
    
    deg = poly.degree()
    if verbose:
        print(f"[BW-fast] Minimal polynomial degree: {deg}")
        sys.stdout.flush()

    # Safety check: if degree is suspiciously close to block size or very small
    if deg <= block_size * 2 or deg < 100:
         print(f"  [BW-fast] WARNING: Poly degree {deg} is suspiciously low (system size {n}).")
         print(f"            This suggests the Krylov sequence degenerated. Result will likely be wrong.")
         # We do not raise here, to allow 'verify' step to catch it downstream if needed

    # --- PASS 2 ---
    if verbose:
        print("[BW-fast] Pass 2: Reconstructing Solution Vector")
        sys.stdout.flush()

    random.seed(seed_val)
    V = [[random.randrange(mod) for _ in range(n)] for _ in range(block_size)]
    random.setstate(rng_state)

    x_accum = [0] * n
    # For standard Wiedemann (block=1), this reconstructs sum(c_i * A^i * v)
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


def lift_discrete_log_via_bsgs(d_mod_ell, ell, h, G, Q, verbose=False):
    """
    Solve for full discrete log d = d_mod_ell + t*ell with 0 <= t < h,
    given G, Q in the Jacobian (Sage group elements).
    Solves t*(ell*G) = R where R = Q - d_mod_ell*G using baby-step giant-step.

    Returns full_d (int) if found, or None if not found.
    """
    # Compute the correction target R = Q - d_mod_ell * G
    R = Q - Integer(d_mod_ell) * G
    if R.is_zero():
        if verbose:
            print("[lift] Already exact: d_mod_ell is the full discrete log.")
        return int(d_mod_ell)

    H = Integer(ell) * G  # generator for the cofactor subgroup

    # If H is zero then ell*G == 0, so no nontrivial cofactor subgroup; failure.
    if H.is_zero():
        if verbose:
            print("[lift] ell * G is zero: cannot lift via BSGS (degenerate).")
        return None

    # Bound on t
    bound = int(h)
    m = int(ceil(sqrt(bound)))

    if verbose:
        print(f"[lift] Attempting BSGS: bound={bound}, m={m}")

    # Baby steps: store j*H for j in [0, m-1]
    baby = {}
    cur = H.zero() if hasattr(H, 'zero') else H * 0  # identity element
    # Build baby steps incrementally to avoid repeated scalar multiplications from scratch
    cur = Integer(0) * H  # identity
    for j in range(m):
        key = str(cur)
        # keep the smallest j for a given group element
        if key not in baby:
            baby[key] = j
        cur = cur + H

    # Giant steps: R - i*(m*H)
    factor = Integer(m) * H
    giant = R
    for i in range(0, m + 1):
        key = str(giant)
        if key in baby:
            j = baby[key]
            t = i * m + j
            if t < bound:
                full_d = int((Integer(d_mod_ell) + Integer(t) * Integer(ell)))
                # verify
                if full_d * G == Q:
                    if verbose:
                        print(f"[lift] Found lift: t={t}, full_d={full_d}")
                    return full_d
                # else continue searching (rare)
        giant = giant - factor

    if verbose:
        print("[lift] BSGS failed to find a lift in [0, h).")
    return None


def verify_matrix_solution(packed_rows, projected_rhs, solution, mod, verbose=True):
    """
    Check A * solution == b (mod mod) using packed_rows (list of (idxs, vals)).
    Returns True if satisfied, else prints a failing row and returns False.
    """
    n = len(packed_rows)
    # convert solution to plain ints
    sol_ints = [int(solution[i]) for i in range(len(solution))]
    for i, (idxs, vals) in enumerate(packed_rows):
        s = 0
        for j, a in zip(idxs, vals):
            # defensive: guard out-of-range column indices
            if j >= len(sol_ints):
                print(f"[verify_matrix_solution] ERROR: solution length {len(sol_ints)} <= column index {j}")
                return False
            s += int(a) * sol_ints[j]
        if (s - int(projected_rhs[i])) % mod != 0:
            if verbose:
                print(f"[verify_matrix_solution] Row {i} FAILED: sum={s % mod}, rhs={projected_rhs[i] % mod}, mod={mod}")
                print(f"  row idxs sample: {idxs[:10]}, vals sample: {vals[:10]}")
            return False
    if verbose:
        print(f"[verify_matrix_solution] OK: A * solution == b (mod {mod})")
    return True


def dump_group_torsion_info(G, Q, full_order, verbose=True):
    """
    Print ell/h and compute ell*G, ell*Q, h*G, h*Q zero tests.
    """
    J_order = Integer(full_order)
    facs = factor(J_order)
    ell = int(max(int(p) for p, _ in facs))
    h = int(J_order // ell)
    info = {}
    info['ell'] = ell
    info['h'] = h
    try:
        info['ellG_zero'] = bool((Integer(ell) * G).is_zero())
        info['ellQ_zero'] = bool((Integer(ell) * Q).is_zero())
        info['hG_zero'] = bool((Integer(h) * G).is_zero())
        info['hQ_zero'] = bool((Integer(h) * Q).is_zero())
    except Exception as e:
        info['error'] = str(e)
    if verbose:
        print("[dump_group_torsion_info]", info)
    return info


def reconstruct_d_from_solution(beta_q, row_q_dict, solution, mod):
    """
    Recompute d = beta - sum_{k} v_k * sol[k] (mod mod),
    using the same ordering as row_q_dict keys and solution indices.
    """
    d = int(beta_q) % mod
    for k, v in row_q_dict.items():
        if k >= len(solution):
            raise IndexError(f"reconstruct_d_from_solution: solution length {len(solution)} <= index {k}")
        coeff = int(solution[k])  # convert Zmod to int
        d = int((d - int(v) * coeff) % mod)
    return d


def diagnose_bw_failure(
    A_packed_rows, projected_rhs, solution, mod, G, Q, full_order,
    row_q_dict, beta_q, verbose=True
):
    """
    Run a suite of checks to find mismatch causes.
    """
    print("=== BW DIAGNOSTIC START ===")
    # 1) Sanity: solution length vs n_cols
    n_cols = max((j for (idxs, _) in A_packed_rows for j in idxs), default=-1) + 1
    print("n_cols (derived) =", n_cols, "solution length =", len(solution))
    if len(solution) != n_cols:
        print("WARNING: solution length != n_cols. This is a prime suspect for ordering mismatch.")

    # 2) verify linear system
    ok_mat = verify_matrix_solution(A_packed_rows, projected_rhs, solution, mod, verbose=verbose)

    # 3) reconstruct d purely algebraically (mod)
    try:
        d_recon = reconstruct_d_from_solution(beta_q, row_q_dict, solution, mod)
        print("Reconstructed d (mod):", d_recon)
    except Exception as e:
        print("Failed to reconstruct d from solution:", e)
        d_recon = None

    # 4) group torsion checks
    tors = dump_group_torsion_info(G, Q, full_order, verbose=verbose)

    # 5) compute D = d*G - Q and inspect non-zero D's invariants
    if d_recon is not None:
        D = Integer(d_recon) * G - Q
        try:
            D_is_zero = bool(D.is_zero())
        except Exception as e:
            D_is_zero = None
        print("Group check: D.is_zero() =>", D_is_zero)
        # compute order of D if not zero (try ell, try h)
        if D_is_zero is False:
            try:
                # Try orders dividing ell and h: show ell*D and h*D
                print("ell*D is zero? ", bool((Integer(tors['ell']) * D).is_zero()))
                print("h*D is zero?   ", bool((Integer(tors['h']) * D).is_zero()))
            except Exception as e:
                print("Failed to inspect D torsion properties:", e)
    print("=== BW DIAGNOSTIC END ===")
    return {
        'matrix_ok': ok_mat,
        'd_recon': d_recon,
        'torsion_info': tors,
        'D_is_zero': D_is_zero if 'D_is_zero' in locals() else None
    }


def solve_dlp_mod_l_block_wiedemann(
    valid_rows,
    rhs_values,
    row_q_dict,
    beta_q,
    full_order,
    G, Q,
    *,
    verbose=True,
    block_size=1, # Default to 1 (Standard Wiedemann) for safety
    nprocs=None,
    max_iters=None,
    force_direct=False
):
    """
    Sparse Block-Wiedemann wrapper for DLP modulo largest prime ell.
    Returns the discrete log modulo ℓ (prime subgroup).
    Automatically switches to Direct Sparse Solver for n_cols < 10,000.
    """
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    # Compute ell (largest prime factor) and cofactor h
    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(int(p) for p, _ in factors))
    h = int(J_order // ell)

    if verbose:
        print(f"  [BW] Working mod ℓ={ell}, cofactor h={h}")
        sys.stdout.flush()
    
    beta_q_l = int(Integer(beta_q) % ell)
    row_q_l = {k: int(Integer(v) % ell) for k, v in row_q_dict.items()}

    projected_rows = []
    projected_rhs = []

    for row, rhs in zip(valid_rows, rhs_values):
        new_row = {}
        for k, v in row.items():
            vk = int(Integer(v) % ell) 
            if vk:
                new_row[k] = vk
        
        rhs_proj = int(Integer(rhs) % ell)
        
        if not new_row:
            if rhs_proj != 0:
                raise ValueError(f"Inconsistent zero-row: RHS={rhs_proj} ≠ 0 mod ℓ")
            continue 
            
        projected_rows.append(new_row)
        projected_rhs.append(rhs_proj)

    if not projected_rows:
        raise ValueError("No nonzero relations after mod ℓ reduction")

    n_cols = max(k for row in projected_rows for k in row) + 1
    
    if verbose:
        nnz = sum(len(r) for r in projected_rows)
        print(f"  [BW] Sparse system: {len(projected_rows)} x {n_cols}, nnz={nnz}")
        sys.stdout.flush()

    # Build SparseRelationMatrix
    A = SparseRelationMatrix(projected_rows, projected_rhs, ell)
    
    # Decide which solver to use
    # Threshold for direct solve: 10,000 columns
    if force_direct or n_cols < 10000:
        if verbose:
            print(f"  [Solver] Using direct sparse solve (n={n_cols} < 10k)")
            sys.stdout.flush()
        
        solution = solve_sparse_direct_mod_ell(A, projected_rhs, ell, verbose=verbose)
    else:
        if verbose:
            print(f"  [Solver] Using Block-Wiedemann (n={n_cols} >= 10k)")
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
        
        if solution is None:
            raise RuntimeError("Block-Wiedemann failed to converge")
        
        if verbose:
            print(f"  [BW] Solved in {dt:.2f}s")
            sys.stdout.flush()

    dlog = Integer(beta_q_l)
    for k, v in row_q_l.items():
        if k < len(solution):
            coeff = int(solution[k])
            dlog = (dlog - Integer(v) * Integer(coeff)) % Integer(ell)
    dlog = int(dlog)

    if verbose:
        print("  [BW] Candidate dlog mod ℓ =", dlog)
        sys.stdout.flush()

    # Verify
    D = Integer(dlog) * G - Q
    if not D.is_zero():
        diagnose_bw_failure(
            A.packed_rows,
            projected_rhs,
            solution,
            ell,
            G, Q, full_order,
            row_q_l, beta_q_l,
            verbose=True
        )
        raise AssertionError(
            "[BW-Verify] Verification failed: dlog * G != Q in J(F_p)[ℓ]\n"
            f"  dlog = {dlog}, ell = {ell}"
        )

    if verbose:
        print("  [BW-Verify] ✓ Exact equality dlog * G == Q in prime subgroup")

    return int(dlog)


def solve_sparse_direct_mod_ell(A_obj, b_vec_ints, mod, verbose=True):
    """
    Direct sparse solve using Sage's built-in solver over Zmod(mod).
    Recommended for systems with < 10k unknowns.
    """
    if verbose:
        print(f"  [Direct] Building sparse matrix over Zmod({mod})...")
        sys.stdout.flush()
    
    t0 = time.time()
    K = Zmod(mod)
    n_rows = len(A_obj.packed_rows)
    n_cols = A_obj.n_cols
    
    # NEW: Sample rows if severely overdetermined (>2x overdetermined)
    if n_rows > 2 * n_cols:
        target_rows = int(1.2 * n_cols)  # 20% buffer for safety
        indices = random.sample(range(n_rows), target_rows)
        indices.sort()
        if verbose:
            print(f"  [Direct] Sampling {target_rows} of {n_rows} rows (system is {n_rows/n_cols:.1f}x overdetermined)")
            sys.stdout.flush()
    else:
        indices = range(n_rows)
    
    # Build dictionary using sampled rows
    entries = {}
    sampled_b = []
    for idx in indices:
        i = len(sampled_b)  # new row index in sampled system
        idxs, vals = A_obj.packed_rows[idx]
        for j, v in zip(idxs, vals):
            entries[(i, j)] = K(int(v))
        sampled_b.append(K(int(b_vec_ints[idx])))
    
    # Construct Sage sparse matrix with sampled rows
    M = matrix(K, len(sampled_b), n_cols, entries, sparse=True)
    b_sage = vector(K, sampled_b)
    
    if verbose:
        nnz = len(entries)
        size = len(sampled_b) * n_cols
        density = 100.0 * nnz / size if size > 0 else 0
        print(f"  [Direct] Matrix: {len(sampled_b)} x {n_cols}, nnz={nnz}, density={density:.4f}%")
        print(f"  [Direct] Solving system (backend=Sage/generic)...")
        sys.stdout.flush()
    
    solution = M.solve_right(b_sage)
    dt = time.time() - t0
    
    if verbose:
        print(f"  [Direct] ✓ Solved in {dt:.2f}s")
        sys.stdout.flush()
        
    return solution
