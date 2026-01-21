# Keep at top only:
from sage.all import Integer, Zmod, vector, GF, PolynomialRing, matrix, factor
from sage.matrix.berlekamp_massey import berlekamp_massey
from multiprocessing import Pool, cpu_count
from math import ceil, sqrt, gcd
import sys
import time
import random
from search_common import BLOCK_WIEDEMANN


# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints


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

    NOTE: Computes v_next = A^T (A v), effectively iterating M = A^T A.
    This is standard for rectangular matrices in Wiedemann algorithm.

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
        raise
    if verbose:
        print("[dump_group_torsion_info]", info)
    return info


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
        raise

    # 4) group torsion checks
    tors = dump_group_torsion_info(G, Q, full_order, verbose=verbose)

    # 5) compute D = d*G - Q and inspect non-zero D's invariants
    if d_recon is not None:
        D = Integer(d_recon) * G - Q
        try:
            D_is_zero = bool(D.is_zero())
        except Exception as e:
            D_is_zero = None
            raise
        print("Group check: D.is_zero() =>", D_is_zero)
        # compute order of D if not zero (try ell, try h)
        if D_is_zero is False:
            try:
                # Try orders dividing ell and h: show ell*D and h*D
                print("ell*D is zero? ", bool((Integer(tors['ell']) * D).is_zero()))
                print("h*D is zero?   ", bool((Integer(tors['h']) * D).is_zero()))
            except Exception as e:
                print("Failed to inspect D torsion properties:", e)
                raise
    print("=== BW DIAGNOSTIC END ===")
    return {
        'matrix_ok': ok_mat,
        'd_recon': d_recon,
        'torsion_info': tors,
        'D_is_zero': D_is_zero if 'D_is_zero' in locals() else None
    }


# put this near your other solvers in sparse_linalg_modp.py

def _matvec_mod(A, v, p):
    """Matrix-vector product with reduction to GF(p) as sage vector."""
    res = A * v
    # ensure reduction (Sage should handle it but be explicit)
    return vector([int(x) % p for x in list(res)])


def solve_sparse_direct_mod_ell(A_sparse_matrix, b_list, mod, verbose=True):
    """
    Direct sparse solver - uses full matrix when already full rank from pruning.
    
    CRITICAL: After pruning guarantees full rank, we can use the entire system.
    Greedy row selection can fail for columns that only appear as pivots in
    linear combinations, not as leading entries in any single row.
    """
    # Extract rows in dict format
    A_rows = []
    for (idxs, vals) in A_sparse_matrix.packed_rows:
        row_dict = {int(idx): int(val) for idx, val in zip(idxs, vals)}
        A_rows.append(row_dict)
    
    n_cols = A_sparse_matrix.n_cols
    n_rows = len(A_rows)
    
    if verbose:
        print(f"  [Direct] Building full system: {n_rows} rows x {n_cols} cols")
        sys.stdout.flush()
    
    K = GF(mod)
    
    # Check if system is already square or overdetermined
    if n_rows < n_cols:
        raise RuntimeError(
            f"Underdetermined system after pruning:\n"
            f"  {n_rows} rows < {n_cols} columns\n"
            f"  Pruning should have reduced columns to match available rank."
        )
    
    if verbose:
        print(f"  [Direct] System is {'square' if n_rows == n_cols else 'overdetermined'}")
        print(f"  [Direct] Using full matrix (pruning already guaranteed full rank)")
        sys.stdout.flush()
    
    # Build full matrix - use ALL rows
    M_sage = matrix(GF(mod), n_rows, n_cols, sparse=True)
    b_sage = vector(GF(mod), b_list)
    
    for i, row_dict in enumerate(A_rows):
        for col, val in row_dict.items():
            M_sage[i, col] = val
    
    # Verify rank
    if verbose:
        print(f"  [Direct] Verifying rank...")
        sys.stdout.flush()
    
    actual_rank = M_sage.rank()
    
    if actual_rank < n_cols:
        raise RuntimeError(
            f"RANK DEFICIT in full matrix:\n"
            f"  Matrix size: {n_rows} x {n_cols}\n"
            f"  Actual rank: {actual_rank}\n"
            f"  Missing {n_cols - actual_rank} dimensions.\n"
            f"  This should not happen after pruning claimed full rank!"
        )
    
    if verbose:
        print(f"  [Direct] ✓ Rank verified: {actual_rank}/{n_cols}")
        print(f"  [Direct] Solving system...")
        sys.stdout.flush()
    
    try:
        solution = M_sage.solve_right(b_sage)
    except ValueError as e:
        # Inconsistent system
        M_aug = M_sage.augment(b_sage.column(), subdivide=False)
        rank_aug = M_aug.rank()
        raise RuntimeError(
            f"System INCONSISTENT:\n"
            f"  Rank[A] = {actual_rank}\n"
            f"  Rank[A|b] = {rank_aug}\n"
            f"  The RHS is not in the column space.\n"
            f"  Original error: {e}"
        )
    
    if verbose:
        print("  [Direct] ✓ Solve successful")
        sys.stdout.flush()
    
    return solution


def find_exact_pivot_columns_sparse(A_rows, mod, verbose=True):
    """
    Exact pivot column identification via sparse incremental Gaussian elimination.
    Much faster than full RREF for sparse matrices.
    
    Returns: sorted list of pivot column indices
    """
    from sage.all import GF
    
    K = GF(mod)
    
    pivot_cols = []
    row_echelon = []  # Store reduced rows for incremental reduction
    
    n_rows = len(A_rows)
    
    if verbose:
        print(f"  [Pivot] Incremental elimination on {n_rows} rows...")
        sys.stdout.flush()
    
    for i, row in enumerate(A_rows):
        if not row:
            continue
        
        # Reduce current row by previous pivot rows
        current = dict(row)
        
        for pivot_col, pivot_row in zip(pivot_cols, row_echelon):
            if pivot_col in current:
                # Eliminate this column using the pivot row
                multiplier = K(current[pivot_col]) / K(pivot_row[pivot_col])
                for col, val in pivot_row.items():
                    current[col] = K(current.get(col, 0) - multiplier * val)
                    if current[col] == 0:
                        del current[col]
        
        # Find leading column in reduced row
        if current:
            leading_col = min(current.keys())
            
            # Normalize so leading coefficient is 1
            lead_inv = K(current[leading_col])**(-1)
            current = {col: K(val * lead_inv) for col, val in current.items()}
            
            pivot_cols.append(leading_col)
            row_echelon.append(current)
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"    [Pivot] Processed {i+1}/{n_rows} rows, found {len(pivot_cols)} pivots")
            sys.stdout.flush()
    
    if verbose:
        print(f"  [Pivot] Found {len(pivot_cols)} exact pivot columns")
        sys.stdout.flush()
    
    return sorted(pivot_cols)


def randomize_rows_for_bw(A_rows, b_list, mod, compression_factor=2, mix_count=3, verbose=True):
    """
    Apply random row mixing to break local structure and improve Krylov mixing.
    
    Args:
        A_rows: list of row dicts
        b_list: corresponding RHS values
        compression_factor: target reduction (2 = half the rows)
        mix_count: how many random rows to combine (3-4 recommended)
    """
    import random
    
    n_original = len(A_rows)
    n_target = int(n_original // compression_factor)
    
    if verbose:
        print(f"  [RowMix] Mixing {n_original} rows down to {n_target}")
        print(f"  [RowMix] Each new row combines {mix_count} random originals")
    
    mixed_rows = []
    mixed_rhs = []
    
    for _ in range(n_target):
        # Pick random rows to combine
        indices = random.sample(range(n_original), mix_count)
        
        # Random nonzero coefficients
        coeffs = [random.randint(1, mod-1) for _ in range(mix_count)]
        
        # Combine rows
        new_row = {}
        new_rhs = 0
        
        for idx, coeff in zip(indices, coeffs):
            for col, val in A_rows[idx].items():
                new_row[col] = (new_row.get(col, 0) + coeff * val) % mod
            new_rhs = (new_rhs + coeff * b_list[idx]) % mod
        
        # Remove zero entries
        new_row = {k: v for k, v in new_row.items() if v != 0}
        
        if new_row:  # Only keep non-empty rows
            mixed_rows.append(new_row)
            mixed_rhs.append(new_rhs)
    
    if verbose:
        avg_density = sum(len(r) for r in mixed_rows) / len(mixed_rows)
        print(f"  [RowMix] Result: {len(mixed_rows)} rows, avg density {avg_density:.1f}")
    
    return mixed_rows, mixed_rhs


def mix_rows_to_target_count(A_rows, mod, target_count, mix_count=4, verbose=True):
    """
    Mix rows down to a precise target count to control system rank.
    Crucial for creating underdetermined systems (rank = cols - 1) from overdetermined data.
    """
    import random
    
    n_original = len(A_rows)
    if n_original < target_count:
        if verbose:
             print(f"  [RowMix] WARNING: Original rows {n_original} < Target {target_count}. No mixing performed.")
        return A_rows

    if verbose:
        print(f"  [RowMix] Mixing {n_original} rows down to EXACTLY {target_count}")
        print(f"  [RowMix] Strategy: 1-dim kernel targeting (cols - 1)")
    
    mixed_rows = []
    
    for _ in range(target_count):
        # Pick random rows to combine
        indices = random.sample(range(n_original), mix_count)
        coeffs = [random.randint(1, mod-1) for _ in range(mix_count)]
        
        # Combine rows
        new_row = {}
        for idx, coeff in zip(indices, coeffs):
            for col, val in A_rows[idx].items():
                new_row[col] = (new_row.get(col, 0) + coeff * val) % mod
        
        # Remove zero entries
        new_row = {k: v for k, v in new_row.items() if v != 0}
        if new_row:
            mixed_rows.append(new_row)
    
    if verbose:
        print(f"  [RowMix] Result: {len(mixed_rows)} rows")
    
    return mixed_rows


def solve_with_retry(A, b, max_attempts=3, **kwargs):
    """
    Retry Block-Wiedemann with different random seeds if BM degree is suspicious.
    """
    for attempt in range(max_attempts):
        if kwargs.get('verbose'):
            print(f"\n  [Retry] Attempt {attempt + 1}/{max_attempts}")
        
        # Randomize seed for this attempt
        seed = random.randint(0, 2**30) + attempt * 12345
        random.seed(seed)
        
        solution = block_wiedemann_solve(A, b, **kwargs)
        
        # Check if BM degree is reasonable (heuristic: > n/100)
        # This would require block_wiedemann_solve to return (solution, bm_degree)
        # For now, just return and let verification catch failures
        
        return solution
    
    raise RuntimeError("All retry attempts failed")

# ============================================================================
# FIX 3: Atom validation (unchanged, still valid)
# ============================================================================

def block_wiedemann_solve(A, iters=None, verbose=True, 
                          ntrials=1, left_seed=None, right_seed=None,
                          force_cols=None):  # <-- ADD THIS PARAMETER
    """
    CORRECTED: True scalar Wiedemann with single Krylov chain.
    Uses Sage's berlekamp_massey on a valid linear recurrence sequence.
    
    CRITICAL: Explicit seed control for kernel diversity.
    SOLVES: Kernel of A (homogeneous system only).

    CRITICAL: This operates on M = A^T A (not A directly).
    - The Krylov sequence is for the squared operator
    - BM degree can be up to 2*rank(A)
    - Sequence length must be >= 3*n to be safe

    Use this for finding kernel of A (kernel(A) = kernel(A^T A)).
    
    Args:
        A: SparseRelationMatrix
        iters: number of Krylov iterations (default 3*n + 200)
        left_seed: seed for left projection vector u (if None, use random)
        right_seed: seed for right start vector v (if None, use random)
        force_cols: list of column indices to force nonzero in initial v
        
    Returns:
        (solution_vector, bm_degree) or (None, 0) if trivial kernel
    """
    from sage.matrix.berlekamp_massey import berlekamp_massey as sage_bm
    
    mod = int(A.mod)
    m = len(A.packed_rows)
    n = A.n_cols
    
    if iters is None:
        iters = 3 * n + 200
    
    # CRITICAL: Sage BM requires even sequence length
    if iters % 2 != 0:
        iters += 1

    if verbose:
        print(f"[BW] Scalar Wiedemann: iters={iters}, nrows={m}, ncols={n}")
        sys.stdout.flush()

    # Generate seeds if not provided
    if left_seed is None:
        left_seed = random.randrange(1, mod)
    if right_seed is None:
        right_seed = random.randrange(1, mod)
    
    if verbose:
        print(f"[BW] Seeds: left={left_seed}, right={right_seed}")
    
    # CRITICAL: Use local RNGs to avoid polluting global random state
    rng_left = random.Random(left_seed)
    rng_right = random.Random(right_seed)
    
    # Generate left projection vector from left_seed
    left_vec_b = [rng_left.randrange(mod) for _ in range(m)]
    if all(x == 0 for x in left_vec_b):
        left_vec_b[0] = 1
    
    # Generate right start vector from right_seed  
    v = [rng_right.randrange(mod) for _ in range(n)]
    
    # CRITICAL FIX: Force specific columns to be nonzero
    if force_cols:
        for col_idx in force_cols:
            if col_idx < len(v):
                v[col_idx] = rng_right.randrange(1, mod)  # nonzero
    
    # --- PASS 1: Generate Single Krylov Sequence ---
    if verbose:
        print("[BW] Pass 1: Generating Krylov Sequence s_t = <u, A^t v>")
        sys.stdout.flush()
    
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

        proj, v_next = compute_proj_and_atav(A.packed_rows, v, left_vec_b, n, mod)
        seq.append(proj)
        v = v_next

    # --- POLYNOMIAL STEP: Use Sage Berlekamp-Massey ---
    if verbose:
        print(f"[BW] Computing Minimal Polynomial from {len(seq)} scalars (Sage BM)...")
        sys.stdout.flush()
    
    # CRITICAL: Sage BM requires even-length sequences
    if len(seq) % 2 != 0:
        seq = seq[:-1]
        if verbose:
            print(f"[BW] Truncated sequence to even length: {len(seq)}")
    
    assert len(seq) % 2 == 0, "BM sequence must be even length for Sage"
    
    K = GF(mod)
    seq_gf = [K(s) for s in seq]
    
    min_poly = sage_bm(seq_gf)
    deg = min_poly.degree()
    
    if verbose:
        print(f"[BW] Minimal polynomial degree: {deg}")
        sys.stdout.flush()

    if deg == 0:
        if verbose:
            print(f"  [BW] Degree 0: trivial kernel")
        return None, 0

    if deg < 100:
        print(f"  [BW] WARNING: Degree {deg} is low for system size {n}")

    coeffs = [int(min_poly[i]) for i in range(deg + 1)]

    # --- PASS 2: Reconstruct Solution Vector ---
    if verbose:
        print(f"[BW] Pass 2: Reconstructing Solution (applying polynomial)")
        sys.stdout.flush()

    # Reinitialize with same right_seed to get same v
    # CRITICAL: Use local RNG to avoid polluting global state
    rng_right = random.Random(right_seed)
    v = [rng_right.randrange(mod) for _ in range(n)]

    x_accum = [0] * n
    
    t_start = time.time()
    last_print = t_start
    
    for i, c in enumerate(coeffs):
        now = time.time()
        if verbose and (now - last_print > 5):
            elapsed = now - t_start
            rate = (i + 1) / max(1e-9, elapsed)
            remaining = (len(coeffs) - i) / rate if rate > 0 else 0
            print(f"  [BW Pass 2] coeff {i}/{len(coeffs)} ({100.0*i/len(coeffs):.1f}%) | elapsed {elapsed/60:.1f}m | ETA {remaining/60:.1f}m")
            sys.stdout.flush()
            last_print = now
        
        if c != 0:
            for j in range(n):
                if v[j]:
                    x_accum[j] = (x_accum[j] + c * v[j]) % mod
        
        if i < len(coeffs) - 1:
            v = at_a_v_from_packed(A.packed_rows, v, n, mod)

    return vector(Zmod(mod), x_accum), deg


def prune_factor_base_to_pivot_columns(A_rows, b_list, mod, verbose=True):
    """
    Prune factor base to pivot columns via sparse incremental Gaussian elimination.
    
    CRITICAL FIX: Preserves RHS alignment with pruned rows.
    
    Returns:
        (pruned_rows, pruned_rhs, col_map, pivot_cols)
    where:
        - pruned_rows: rows with only pivot columns
        - pruned_rhs: corresponding RHS values (properly aligned)
        - col_map: dict mapping old_col_idx -> new_col_idx (or None if pruned)
        - pivot_cols: list of original pivot column indices
    """
    from sage.all import GF
    
    K = GF(mod)
    
    # Find all columns that appear
    all_cols = set()
    for row in A_rows:
        all_cols.update(row.keys())
    n_cols_orig = max(all_cols) + 1 if all_cols else 0
    
    if verbose:
        print(f"  [Prune] Input: {len(A_rows)} rows x {n_cols_orig} cols")
        sys.stdout.flush()
    
    # Use sparse incremental elimination to find exact pivot columns
    pivot_cols = find_exact_pivot_columns_sparse(A_rows, mod, verbose=verbose)
    
    if not pivot_cols:
        raise RuntimeError("Pruning found zero pivot columns - system is trivial!")
    
    if verbose:
        print(f"  [Prune] Pivot columns: {len(pivot_cols)}/{n_cols_orig}")
        print(f"  [Prune] Rank: {len(pivot_cols)}")
        print(f"  [Prune] Pruned {n_cols_orig - len(pivot_cols)} redundant columns")
        sys.stdout.flush()
    
    # Build column mapping
    col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(pivot_cols)}
    
    # CRITICAL FIX: Preserve RHS alignment with rows
    pruned_rows = []
    pruned_rhs = []
    for i, row in enumerate(A_rows):
        pruned_row = {}
        for old_idx, val in row.items():
            if old_idx in col_map:
                new_idx = col_map[old_idx]
                pruned_row[new_idx] = int(val)
        if pruned_row:
            pruned_rows.append(pruned_row)
            pruned_rhs.append(b_list[i])  # <-- Preserve corresponding RHS

    assert len(pruned_rows) == len(pruned_rhs)
    
    if verbose:
        print(f"  [Prune] Output: {len(pruned_rows)} rows x {len(pivot_cols)} cols")
        assert len(pruned_rows) == len(pruned_rhs), "RHS count must match row count!"
        sys.stdout.flush()
    
    return pruned_rows, pruned_rhs, col_map, pivot_cols


def expand_solution_to_original(solution_vec, col_map):
    """
    Expand pruned solution vector back to original atom indexing.
    
    After pruning removes redundant columns, the solution vector uses
    the new (compact) indices. This function maps it back to original
    indices so it can be used with row_g_dict, row_q_dict.
    
    Args:
        solution_vec: solution in pruned column space
        col_map: dict {old_idx: new_idx} from pruning
        
    Returns:
        solution in original column space (list of ints)
    """
    if not col_map:
        # No pruning was done, return as-is
        return [int(x) for x in solution_vec]
    
    # Find max original index
    n_orig = max(col_map.keys()) + 1
    
    # Build solution in original indexing
    sol_orig = [0] * n_orig
    for old_idx, new_idx in col_map.items():
        if new_idx < len(solution_vec):
            sol_orig[old_idx] = int(solution_vec[new_idx])
    
    return sol_orig


def reconstruct_d_from_solution(beta_q, row_q_dict, solution, mod):
    """
    Recompute d = beta - sum_{k} v_k * sol[k] (mod mod).
    
    CRITICAL: solution must be in the SAME indexing as row_q_dict keys.
    If pruning was used, call expand_solution_to_original first.
    
    Args:
        beta_q: scalar offset from Q smoothing
        row_q_dict: Q's factor base encoding (original indices)
        solution: solution vector (MUST be in original indexing)
        mod: modulus
        
    Returns:
        d (mod mod)
    """
    d = int(beta_q) % mod
    for k, v in row_q_dict.items():
        if k >= len(solution):
            raise IndexError(
                f"reconstruct_d_from_solution: solution length {len(solution)} <= index {k}\n"
                f"This indicates solution indexing doesn't match row_q_dict!\n"
                f"Did you forget to call expand_solution_to_original?"
            )
        coeff = int(solution[k])
        d = int((d - int(v) * coeff) % mod)
    return d


def block_wiedemann_inhomogeneous_solve(A, rhs, verbose=True, max_attempts=5):
    """
    Solve inhomogeneous system A*x = b using Block-Wiedemann.
    
    Uses the augmented system approach:
        [A | b] * [x; -1] = 0
    
    Then solves the homogeneous kernel problem and extracts x.
    
    Args:
        A: SparseRelationMatrix
        rhs: list of RHS values
        verbose: print diagnostics
        max_attempts: retry attempts with different seeds
        
    Returns:
        (solution_vector, success_bool)
    """
    from sage.all import Zmod, vector, Integer
    
    mod = int(A.mod)
    m = len(A.packed_rows)
    n = A.n_cols
    
    if len(rhs) != m:
        raise ValueError(f"RHS length {len(rhs)} != matrix rows {m}")
    
    if verbose:
        print(f"  [BW Inhomogeneous] Augmenting system: {m} rows x {n} cols -> {m} rows x {n+1} cols")
        sys.stdout.flush()
    
    # Build augmented system: [A | b]
    # Each row i becomes: [A_i | b_i]
    # We seek kernel vector [x; -1] such that A*x = b
    augmented_rows = []
    for i, (row, b_val) in enumerate(zip(A.packed_rows, rhs)):
        # row = (idxs, vals)
        idxs_list = list(row[0]) + [n]  # Add column n for RHS
        vals_list = list(row[1]) + [int(b_val) % mod]
        augmented_rows.append((idxs_list, vals_list))
    
    # Convert to dict format for SparseRelationMatrix
    aug_rows_dict = []
    for (idxs, vals) in augmented_rows:
        row_dict = {int(idx): int(val) % mod for idx, val in zip(idxs, vals) if int(val) % mod != 0}
        aug_rows_dict.append(row_dict)
    
    # Build augmented sparse matrix
    A_aug = SparseRelationMatrix(aug_rows_dict, [0]*m, mod)
    
    if verbose:
        print(f"  [BW Inhomogeneous] Searching for kernel vector [x; -1]...")
        sys.stdout.flush()
    
    # Try multiple random seeds to find kernel
    for attempt in range(max_attempts):
        if verbose and attempt > 0:
            print(f"  [BW Inhomogeneous] Retry attempt {attempt + 1}/{max_attempts}")
            sys.stdout.flush()
        
        # Generate unique seeds
        base_seed = random.randint(0, 2**30)
        left_seed = base_seed + attempt * 999983
        right_seed = base_seed + attempt * 777767 + 123456
        
        # Force last column (RHS column) to be nonzero in initial vector
        force_cols = [n]  # Force column n (the RHS column)
        
        kernel_vec, bm_degree = block_wiedemann_solve(
            A_aug,
            iters=None,
            verbose=(verbose and attempt == 0),
            left_seed=left_seed,
            right_seed=right_seed,
            force_cols=force_cols
        )
        
        if kernel_vec is None:
            if verbose:
                print(f"  [BW Inhomogeneous] Attempt {attempt+1}: trivial kernel")
            continue
        
        # Extract solution: kernel_vec = [x; lambda]
        # We want lambda = -1 (mod ell)
        lambda_val = int(kernel_vec[n]) % mod
        
        if lambda_val == 0:
            if verbose:
                print(f"  [BW Inhomogeneous] Attempt {attempt+1}: lambda=0 (degenerate)")
            continue
        
        # Normalize so lambda = -1
        # If lambda != 0, scale: x_solution = -kernel_vec[0:n] / lambda
        K = Zmod(mod)
        lambda_inv = K(lambda_val)**(-1)
        
        x_solution = []
        for i in range(n):
            x_i = K(-int(kernel_vec[i])) * lambda_inv
            x_solution.append(int(x_i))
        
        # Verify: A * x_solution == rhs
        valid = True
        for i, (idxs, vals) in enumerate(A.packed_rows):
            row_sum = 0
            for j, a in zip(idxs, vals):
                if j < len(x_solution):
                    row_sum += int(a) * x_solution[j]
            if (row_sum - int(rhs[i])) % mod != 0:
                valid = False
                break
        
        if valid:
            if verbose:
                print(f"  [BW Inhomogeneous] ✓ Found valid solution (attempt {attempt+1})")
                sys.stdout.flush()
            return vector(Zmod(mod), x_solution), True
        else:
            if verbose:
                print(f"  [BW Inhomogeneous] Attempt {attempt+1}: verification failed")
    
    if verbose:
        print(f"  [BW Inhomogeneous] ✗ All {max_attempts} attempts failed")
        sys.stdout.flush()
    
    return None, False


def solve_dlp_mod_l_block_wiedemann(
    homogeneous_rows,
    row_g_dict,
    alpha_g,
    row_q_dict,
    beta_q,
    full_order,
    G, Q,
    atom_to_idx,
    J,
    *,
    verbose=True,
    nprocs=None,
    use_direct_solver=True,
):
    """
    CORRECTED: Traditional Index Calculus - Inhomogeneous Linear System WITH PRUNING.
    
    Solves the DLP Q = d*G in J[ℓ] by solving:
        R * x = g  (mod ℓ)
    
    where:
        - R: relation matrix (homogeneous FB relations + G-row)
        - g: RHS vector (zeros + 1 for G-row)
        - x: solution vector (FB element exponents)
    
    Then recovers d from:
        d ≡ (q·x) / (g·x)  (mod ℓ)
    
    CRITICAL FIX: Prunes to pivot columns BEFORE building the system.
    This ensures the system is full-rank and consistent.
    
    Args:
        homogeneous_rows: list of relation dicts (homogeneous, RHS=0)
        row_g_dict: G's factor base encoding
        alpha_g: G smoothing offset
        row_q_dict: Q's factor base encoding  
        beta_q: Q smoothing offset
        full_order: |J|
        G, Q: Jacobian elements (for verification)
        atom_to_idx: factor base atom map
        J: Jacobian
        verbose: print diagnostics
        nprocs: number of processes for BW
        use_direct_solver: if True, use Sage's direct sparse solver instead of BW
    
    Returns:
        Integer: discrete log d (mod ℓ)
    """
    from sage.all import Integer, factor, Zmod, matrix, vector
    
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(int(p) for p, _ in factors))

    if verbose:
        print(f"  [Inhomogeneous Solver] Working mod ℓ={ell}")
        print(f"  [Inhomogeneous Solver] Solver: {'Direct' if use_direct_solver else 'Block-Wiedemann'}")
        sys.stdout.flush()
    
    # Project homogeneous rows to mod ℓ
    projected_rows = []
    for row in homogeneous_rows:
        new_row = {int(k): int(v) % ell for k, v in row.items() if int(v) % ell != 0}
        if new_row:
            projected_rows.append(new_row)

    if not projected_rows:
        raise ValueError("No nonzero homogeneous relations after mod ℓ reduction")

    if verbose:
        print(f"  [System] Input: {len(projected_rows)} homogeneous relations")
        sys.stdout.flush()
    
    # ========================================================================
    # CRITICAL FIX: Prune to pivot columns BEFORE building the system
    # ========================================================================
    
    if verbose:
        print(f"  [System] Pruning to pivot columns...")
        sys.stdout.flush()
    
    # Build RHS for homogeneous rows (all zeros)
    homo_rhs = [0] * len(projected_rows)
    
    # Prune to pivot columns
    pruned_rows, pruned_rhs, col_map, pivot_cols = prune_factor_base_to_pivot_columns(
        projected_rows, 
        homo_rhs, 
        ell, 
        verbose=verbose
    )
    
    n_cols_fb = len(pivot_cols)
    
    if verbose:
        print(f"  [System] After pruning: {len(pruned_rows)} rows × {n_cols_fb} cols")
        sys.stdout.flush()
    
    # ========================================================================
    # Map G and Q rows to pruned column indexing
    # ========================================================================
    
    g_row_pruned = {}
    for old_idx, val in row_g_dict.items():
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            val_mod = int(val) % ell
            if val_mod != 0:
                g_row_pruned[new_idx] = val_mod
    
    q_row_pruned = {}
    for old_idx, val in row_q_dict.items():
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            val_mod = int(val) % ell
            if val_mod != 0:
                q_row_pruned[new_idx] = val_mod
    
    # Sanity check: G and Q must still be expressible after pruning
    if not g_row_pruned:
        raise RuntimeError(
            "G row disappeared after pruning!\n"
            "This means G is not in the span of pivot columns.\n"
            "Possible causes:\n"
            "  - G was not actually smooth over the original factor base\n"
            "  - Numerical issues in pruning\n"
            "  - Inconsistent atom indexing"
        )
    
    if not q_row_pruned:
        raise RuntimeError(
            "Q row disappeared after pruning!\n"
            "This means Q is not in the span of pivot columns.\n"
            "Possible causes:\n"
            "  - Q was not actually smooth over the original factor base\n"
            "  - Numerical issues in pruning\n"
            "  - Inconsistent atom indexing"
        )
    
    if verbose:
        print(f"  [System] G-row: {len(g_row_pruned)}/{len(row_g_dict)} coefficients survived pruning")
        print(f"  [System] Q-row: {len(q_row_pruned)}/{len(row_q_dict)} coefficients survived pruning")
        sys.stdout.flush()
    
    # ========================================================================
    # Build inhomogeneous system with PRUNED columns
    # ========================================================================
    
    if verbose:
        print(f"  [System] Building inhomogeneous system R*x = g")
        sys.stdout.flush()
    
    # System: homogeneous rows (RHS=0) + G-row (RHS=1)
    all_rows = list(pruned_rows)
    rhs_values = list(pruned_rhs)
    
    # Add G-row: sum(g_i * x_i) = 1  (enforces G = sum g_i * P_i)
    all_rows.append(g_row_pruned)
    rhs_values.append(1)  # INHOMOGENEOUS
    
    n_rows = len(all_rows)
    
    if verbose:
        print(f"  [System] Matrix: {n_rows} rows x {n_cols_fb} cols")
        print(f"    - {len(pruned_rows)} homogeneous relations (RHS=0)")
        print(f"    - 1 G-row (RHS=1)")
        print(f"  [System] NO Q-ROW - Q handled in post-solve projection")
        sys.stdout.flush()

    # ========================================================================
    # Solve using chosen method
    # ========================================================================
    
    if use_direct_solver:
        # Use Sage's direct sparse solver
        if verbose:
            print(f"  [Direct] Building sparse matrix...")
            sys.stdout.flush()
        
        K = Zmod(ell)
        entries = {}
        for i, row in enumerate(all_rows):
            for j, v in row.items():
                val = K(int(v))
                if val != 0:
                    entries[(i, j)] = val
        
        R = matrix(K, n_rows, n_cols_fb, entries, sparse=True)
        g_vec = vector(K, [K(int(v)) for v in rhs_values])
        
        if verbose:
            print(f"  [Direct] Matrix built: {R.nrows()} × {R.ncols()}, {len(entries)} nonzero entries")
            print(f"  [Direct] Verifying rank...")
            sys.stdout.flush()
        
        # Verify rank before solving
        rank = R.rank()
        if verbose:
            print(f"  [Direct] Rank: {rank}/{n_cols_fb}")
            sys.stdout.flush()
        
        if rank < n_cols_fb:
            raise RuntimeError(
                f"Matrix is rank-deficient after pruning:\n"
                f"  Rank: {rank}\n"
                f"  Columns: {n_cols_fb}\n"
                f"  This should not happen - pruning should guarantee full rank!"
            )
        
        if verbose:
            print(f"  [Direct] Solving R*x = g (mod {ell})...")
            sys.stdout.flush()
        
        try:
            x_sol = R.solve_right(g_vec)
        except ValueError as e:
            # Check if system is inconsistent
            R_aug = R.augment(g_vec.column(), subdivide=False)
            rank_aug = R_aug.rank()
            raise RuntimeError(
                f"Inhomogeneous system R*x = g is inconsistent:\n"
                f"  Rank[R] = {rank}\n"
                f"  Rank[R|g] = {rank_aug}\n"
                f"  Difference: {rank_aug - rank}\n"
                f"  This means G is not in the span of homogeneous relations.\n"
                f"  Original error: {e}"
            )
        
        if verbose:
            print(f"  [Direct] ✓ Solve successful")
            sys.stdout.flush()
    
    else:
        # Use Block-Wiedemann for inhomogeneous solve
        # Build SparseRelationMatrix from pruned rows
        A = SparseRelationMatrix(all_rows, rhs_values, ell)
        
        if verbose:
            print(f"  [BW] Solving inhomogeneous system via Block-Wiedemann...")
            sys.stdout.flush()
        
        # Call BW inhomogeneous solver
        x_sol, success = block_wiedemann_inhomogeneous_solve(
            A, 
            rhs_values,
            verbose=verbose,
            max_attempts=5
        )
        
        if not success or x_sol is None:
            raise RuntimeError(
                "Block-Wiedemann inhomogeneous solve failed.\n"
                "System may be inconsistent or underdetermined."
            )
        
        if verbose:
            print(f"  [BW] ✓ Solve successful")
            sys.stdout.flush()
    
    # ========================================================================
    # Post-solve: recover discrete log
    # ========================================================================
    
    # d ≡ (q·x) / (g·x)  (mod ℓ)
    
    K = Zmod(ell)
    
    if verbose:
        print(f"  [DLog Recovery] Computing g·x and q·x using PRUNED indices...")
        sys.stdout.flush()
    
    # Compute g·x (should equal 1 from the G-row constraint)
    g_dot_x = K(0)
    for idx, g_coeff in g_row_pruned.items():  # Use pruned version!
        if idx < len(x_sol):
            g_dot_x += K(int(g_coeff)) * x_sol[idx]
        else:
            raise IndexError(
                f"g_row_pruned index {idx} >= solution length {len(x_sol)}\n"
                f"This should not happen!"
            )
    
    # Compute q·x
    q_dot_x = K(0)
    for idx, q_coeff in q_row_pruned.items():  # Use pruned version!
        if idx < len(x_sol):
            q_dot_x += K(int(q_coeff)) * x_sol[idx]
        else:
            raise IndexError(
                f"q_row_pruned index {idx} >= solution length {len(x_sol)}\n"
                f"This should not happen!"
            )
    
    if verbose:
        print(f"  [DLog Recovery] g·x = {g_dot_x} (expected 1 from G-row)")
        print(f"  [DLog Recovery] q·x = {q_dot_x}")
        sys.stdout.flush()
    
    # Sanity check: g·x should be 1
    if g_dot_x != K(1):
        raise RuntimeError(
            f"G-row constraint violated: g·x = {g_dot_x} ≠ 1\n"
            f"The solve returned an invalid solution.\n"
            f"This indicates a serious bug in the solver."
        )
    
    # Recover d
    # Since g·x = 1, we have d ≡ q·x (mod ℓ)
    dlog = Integer(int(q_dot_x))
    
    if verbose:
        print(f"  [DLog] d = q·x / g·x = {q_dot_x} / {g_dot_x} = {dlog} (mod ℓ)")
        sys.stdout.flush()

    # ========================================================================
    # Verification
    # ========================================================================
    
    if verbose:
        print(f"  [Verify] Checking d*G == Q in Jacobian...")
        sys.stdout.flush()
    
    D = Integer(dlog) * G - Q
    
    if not D.is_zero():
        raise AssertionError(
            f"Verification failed: d*G ≠ Q\n"
            f"  d = {dlog}\n"
            f"  ℓ = {ell}\n"
            f"  D = d*G - Q is nonzero\n"
            f"  This means the discrete log is incorrect."
        )

    if verbose:
        print("  [Verify] ✓ d*G == Q")
        sys.stdout.flush()

    return Integer(dlog)
