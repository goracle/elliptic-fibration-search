import sys, time, random
from sage.all import Integer, Zmod, vector, GF, PolynomialRing, matrix, factor
from sage.matrix.berlekamp_massey import berlekamp_massey
from multiprocessing import Pool, cpu_count
from math import ceil, sqrt, gcd
from search_common import BLOCK_WIEDEMANN

# Keep at top only:

# sparse_linalg_modp.py

# Tunable threshold for lazy reduction
# Safe headroom for Python 64-bit ints before needing to apply % mod
_LAZY_LIMIT = (1 << 61) - 1

class SparseRelationMatrix:

    def __init__(self, rows, rhs, modulus):
        """
        rows: list of dicts {col: coeff}
        rhs:  list of ints
        modulus: prime ℓ
        """
        self.mod = int(modulus)
        self.n_rows = len(rows)
        # Find the max column index to determine matrix width
        self.n_cols = max(
            (max(r.keys()) if r else 0 for r in rows), default=0
        ) + 1

        # Pack rows as (indices, values) tuples for faster access during matvecs
        self.packed_rows = []
        for r in rows:
            if r:
                idxs = list(r.keys())
                vals = [int(v) % self.mod for v in r.values()]
                self.packed_rows.append((idxs, vals))
            else:
                self.packed_rows.append(([], []))

        # Build column-wise view for transpose matvec (A^T * v)
        self.packed_cols = [[] for _ in range(self.n_cols)]
        for i, (idxs, vals) in enumerate(self.packed_rows):
            for j, v in zip(idxs, vals):
                self.packed_cols[j].append((i, v))

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

def randomize_rows_for_bw(A_rows, b_list, mod, compression_factor=2, mix_count=3, verbose=True):
    """
    Apply random row mixing to break local structure and improve Krylov mixing.

    Args:
        A_rows: list of row dicts
        b_list: corresponding RHS values
        compression_factor: target reduction (2 = half the rows)
        mix_count: how many random rows to combine (3-4 recommended)
    """

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

def block_wiedemann_inhomogeneous_solve(A, rhs, verbose=True, max_attempts=5, iters=None):
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

        dim = min(len(A.packed_rows), A.n_cols)
        iters = iters if iters is not None else 2 * dim + 200
        kernel_vec, bm_degree = block_wiedemann_solve(
            A_aug,
            iters=iters,
            verbose=verbose,
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
        projected_rows + [row_g_dict, row_q_dict],
        homo_rhs + [0, 0],
        ell,
        verbose=verbose
    )

    # strip the appended G and Q rows back off — only keep homogeneous rows
    pruned_rows = pruned_rows[:len(projected_rows)]
    pruned_rhs  = pruned_rhs[:len(projected_rows)]

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

# ============================================================================
# PATCH 1: Add new solver to sparse_linalg_modp.py
# ============================================================================

# CORRECTED: sparse_linalg_modp.py - solve_dlp_mod_l_cofactor_projection
# This replaces the existing function in your sparse_linalg_modp.py

def solve_dlp_mod_l_cofactor_projection(
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
    use_direct_solver=True,
):
    """
    CORRECTED: Index Calculus with cofactor projection.

    KEY FIXES:
    1. ALL relations (homogeneous + G-row) are h-projected TOGETHER
    2. G-row RHS is h (mod ℓ), not 1, after h-projection
    3. Both matrix AND RHS are multiplied by h consistently
    4. Final dlog extraction uses ORIGINAL (non-projected) encodings

    Solves: (h·R) * x = h·[0, ..., 0, 1]ᵀ  (mod ℓ)
    Then recovers: d ≡ (q·x) / (g·x)  (mod ℓ)
    """

    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(int(p) for p, _ in factors))
    h = int(J_order // ell)

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEX CALCULUS - COFACTOR PROJECTION METHOD")
        print(f"{'='*70}")
        print(f"Full order |J|: {J_order}")
        print(f"Largest prime ℓ: {ell}")
        print(f"Cofactor h: {h}")
        sys.stdout.flush()

    # Verify G and Q are ℓ-torsion
    if verbose:
        print(f"  [Check] Verifying G and Q are ℓ-torsion...")
        sys.stdout.flush()

    ell_G = Integer(ell) * G
    ell_Q = Integer(ell) * Q

    if not ell_G.is_zero():
        raise RuntimeError(f"G is NOT ℓ-torsion: ℓ·G = {ell_G} ≠ 0")

    if not ell_Q.is_zero():
        raise RuntimeError(f"Q is NOT ℓ-torsion: ℓ·Q = {ell_Q} ≠ 0")

    if verbose:
        print(f"  [Check] ✓ Both G and Q are in J[ℓ]")
        sys.stdout.flush()

    # ========================================================================
    # STEP 1: Build combined system INCLUDING G-row from the start
    # ========================================================================

    if verbose:
        print(f"\n  [Step 1] Building combined system (homogeneous + G)")
        sys.stdout.flush()

    # Combine homogeneous rows with G-row BEFORE projection
    all_rows_full = list(homogeneous_rows)
    rhs_full = [0] * len(homogeneous_rows)

    # Add G-row with RHS = 1 (before h-projection)
    all_rows_full.append(dict(row_g_dict))
    rhs_full.append(1)

    # Get dimensions
    max_idx = -1
    for r in all_rows_full:
        if r:
            max_idx = max(max_idx, max(r.keys()))
    if row_q_dict:
        max_idx = max(max_idx, max(row_q_dict.keys()))

    n_cols = max_idx + 1
    n_rows = len(all_rows_full)

    if verbose:
        print(f"  [Step 1] Combined system: {n_rows} rows × {n_cols} cols")
        print(f"           - {len(homogeneous_rows)} homogeneous (RHS=0)")
        print(f"           - 1 G-row (RHS=1)")
        sys.stdout.flush()

    # ========================================================================
    # STEP 2: Apply cofactor h to BOTH matrix AND RHS
    # ========================================================================

    if verbose:
        print(f"\n  [Step 2] Applying h-projection: multiplying ALL rows by h={h}")
        sys.stdout.flush()

    projected_rows = []
    projected_rhs = []

    for row, rhs in zip(all_rows_full, rhs_full):
        # Project row: multiply all coefficients by h (mod ℓ)
        new_row = {}
        for k, v in row.items():
            new_val = int((Integer(v) * Integer(h)) % ell)
            if new_val != 0:
                new_row[k] = new_val

        # Project RHS: multiply by h (mod ℓ)
        new_rhs = int((Integer(rhs) * Integer(h)) % ell)

        if new_row or new_rhs != 0:
            projected_rows.append(new_row)
            projected_rhs.append(new_rhs)

    if verbose:
        print(f"  [Step 2] After h-projection: {len(projected_rows)} rows")
        nonzero_rhs_count = sum(1 for r in projected_rhs if r != 0)
        print(f"    RHS values: {len(projected_rhs) - nonzero_rhs_count} zeros, {nonzero_rhs_count} nonzero")
        print(f"    G-row RHS: {projected_rhs[-1]} (should be h mod ℓ = {h % ell})")
        sys.stdout.flush()

    # ========================================================================
    # CRITICAL CHECK: Verify G-row survived h-projection
    # ========================================================================

    # The G-row should be the LAST row after projection
    g_row_rhs = projected_rhs[-1] if projected_rhs else 0

    if g_row_rhs == 0:
        raise RuntimeError(
            f"CRITICAL: G-row RHS became zero after h-projection!\n"
            f"  h = {h}, ℓ = {ell}\n"
            f"  h mod ℓ = {h % ell}\n"
            f"  This means h ≡ 0 (mod ℓ), which violates gcd(h, ℓ) = 1.\n"
            f"  The factorization of |J| is incorrect or ℓ is not the largest prime."
        )

    if g_row_rhs != (h % ell):
        print(f"  [Warning] G-row RHS = {g_row_rhs}, expected {h % ell}")
        print(f"           This can happen if G-row was modified during projection")

    if verbose:
        print(f"  [Check] ✓ G-row survived: RHS = {g_row_rhs}")
        sys.stdout.flush()

    # ========================================================================
    # STEP 3: Prune to pivot columns
    # ========================================================================

    if verbose:
        print(f"\n  [Step 3] Pruning to pivot columns")
        print(f"  CRITICAL: Pruning ALL rows (including G-row) together")
        sys.stdout.flush()

    # CRITICAL: Pass ALL projected rows (including G) to pruning
    pruned_rows, pruned_rhs, col_map, pivot_cols = prune_factor_base_to_pivot_columns(
        projected_rows,
        projected_rhs,
        ell,
        verbose=verbose
    )

    n_cols_pruned = len(pivot_cols)

    if verbose:
        print(f"  [Step 3] After pruning: {len(pruned_rows)} rows × {n_cols_pruned} cols")
        sys.stdout.flush()

    # ========================================================================
    # CRITICAL CHECK: Verify system is still inhomogeneous after pruning
    # ========================================================================

    assert len(pruned_rows) == len(pruned_rhs), \
        f"RHS count mismatch: {len(pruned_rows)} rows but {len(pruned_rhs)} RHS values"

    nonzero_rhs_indices = [i for i, rhs in enumerate(pruned_rhs) if rhs != 0]

    if len(nonzero_rhs_indices) == 0:
        raise RuntimeError(
            "CRITICAL: All RHS values are zero after pruning!\n"
            "The G-row was eliminated during pivot selection.\n"
            "This means G is a linear combination of homogeneous relations,\n"
            "which violates the rank structure we need.\n"
            "\n"
            "Possible causes:\n"
            "  - Homogeneous relations already span the full space\n"
            "  - G-row became zero during h-projection (check gcd(h, ℓ) = 1)\n"
            "  - Row operations reduced G-row RHS to zero mod ℓ"
        )

    if verbose:
        print(f"\n  [Critical Check] ✓ System is still INHOMOGENEOUS")
        print(f"    Found {len(nonzero_rhs_indices)} row(s) with nonzero RHS")
        for idx in nonzero_rhs_indices[:5]:  # Show first 5
            print(f"      Row {idx}: RHS = {pruned_rhs[idx]}")
        sys.stdout.flush()

    # ========================================================================
    # STEP 4: Map Q to pruned coordinates
    # ========================================================================

    if verbose:
        print(f"\n  [Step 4] Mapping Q to pruned coordinates...")
        sys.stdout.flush()

    # Apply h-projection to Q row (same as we did to all other rows)
    q_row_projected = {}
    for old_idx, val in row_q_dict.items():
        val_proj = int((Integer(val) * Integer(h)) % ell)
        if val_proj != 0:
            q_row_projected[old_idx] = val_proj

    # Map to pruned column indices
    q_row_pruned = {}
    for old_idx, val in q_row_projected.items():
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            q_row_pruned[new_idx] = val

    if not q_row_pruned:
        raise RuntimeError("Q encoding vanished after pruning and h-projection!")

    if verbose:
        print(f"  [Step 4] Q-row: {len(q_row_pruned)}/{len(row_q_dict)} coefficients survived")
        sys.stdout.flush()

    # ========================================================================
    # STEP 5: Solve the linear system
    # ========================================================================

    if verbose:
        print(f"\n  [Step 5] Solving system mod ℓ={ell}")
        print(f"  System: R * x = g (INHOMOGENEOUS)")
        sys.stdout.flush()

    if not use_direct_solver:
        raise NotImplementedError("Block-Wiedemann not yet adapted for cofactor projection")

    K = Zmod(ell)
    entries = {}
    for i, row in enumerate(pruned_rows):
        for j, v in row.items():
            val = K(int(v))
            if val != 0:
                entries[(i, j)] = val

    R_matrix = matrix(K, len(pruned_rows), n_cols_pruned, entries, sparse=True)
    g_vec = vector(K, [K(int(v)) for v in pruned_rhs])

    if verbose:
        print(f"  [Matrix] Built: {R_matrix.nrows()} × {R_matrix.ncols()}")
        print(f"  [Matrix] Sparse entries: {len(entries)}")
        print(f"  [Solve] Computing R * x = g...")
        sys.stdout.flush()

    try:
        x_solution = R_matrix.solve_right(g_vec)
    except ValueError as e:
        # System is inconsistent - diagnose
        rank_R = R_matrix.rank()
        R_aug = R_matrix.augment(g_vec.column(), subdivide=False)
        rank_aug = R_aug.rank()

        raise RuntimeError(
            f"System inconsistent after pruning!\n"
            f"  Rank[R]   = {rank_R}\n"
            f"  Rank[R|g] = {rank_aug}\n"
            f"  Rows: {R_matrix.nrows()}, Cols: {R_matrix.ncols()}\n"
            f"  This should not happen if pruning preserved row operations.\n"
            f"  Original error: {e}"
        )

    if verbose:
        print(f"  [Solve] ✓ Solution found")
        sys.stdout.flush()

    # ========================================================================
    # STEP 6: Extract discrete log (accounting for h-projection)
    # ========================================================================

    if verbose:
        print(f"\n  [Step 6] Extracting discrete log")
        sys.stdout.flush()

    # We solved: (h·R) * x = h·g  where R is the homogeneous matrix, g is the G-row
    # The solution x gives us: h·Q = (h·q)·x  and  h = (h·g)·x
    # Therefore: d ≡ (h·q)·x / (h·g)·x ≡ (q·x) / (g·x) (mod ℓ)

    K = Zmod(ell)

    # We need ORIGINAL G encoding (not h-projected) to compute g·x
    # Map original G-row to pruned coordinates
    g_row_pruned_original = {}
    for old_idx, val in row_g_dict.items():
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            val_mod = int(val) % ell
            if val_mod != 0:
                g_row_pruned_original[new_idx] = val_mod

    # Compute g·x (using ORIGINAL G encoding)
    g_dot_x = K(0)
    for idx, coeff in g_row_pruned_original.items():
        if idx < len(x_solution):
            g_dot_x += K(int(coeff)) * x_solution[idx]

    # Compute q·x (using ORIGINAL Q encoding, which we h-projected then pruned)
    # But we want the ORIGINAL q vector, so map from row_q_dict
    q_row_pruned_original = {}
    for old_idx, val in row_q_dict.items():
        if old_idx in col_map:
            new_idx = col_map[old_idx]
            val_mod = int(val) % ell
            if val_mod != 0:
                q_row_pruned_original[new_idx] = val_mod

    q_dot_x = K(0)
    for idx, q_coeff in q_row_pruned_original.items():
        if idx < len(x_solution):
            q_dot_x += K(int(q_coeff)) * x_solution[idx]

    if verbose:
        print(f"  [Extraction] g·x = {g_dot_x}")
        print(f"  [Extraction] q·x = {q_dot_x}")
        sys.stdout.flush()

    if g_dot_x == K(0):
        raise RuntimeError(
            "Degenerate solution: g·x = 0\n"
            "This means the solution doesn't satisfy the G constraint.\n"
            "The system may have become degenerate during pruning."
        )

    # d ≡ (q·x) / (g·x) (mod ℓ)
    dlog = Integer(int(q_dot_x / g_dot_x))

    if verbose:
        print(f"  [DLog] d = (q·x) / (g·x) = {dlog} (mod ℓ)")
        sys.stdout.flush()

    # ========================================================================
    # STEP 7: Verification
    # ========================================================================

    if verbose:
        print(f"\n  [Verify] Checking d·G == Q...")
        sys.stdout.flush()

    D = Integer(dlog) * G - Q

    if not D.is_zero():
        raise RuntimeError(f"Verification FAILED: {dlog}·G ≠ Q")

    if verbose:
        print(f"  [Verify] ✓ Success: {dlog}·G = Q")
        print(f"\n{'='*70}\n")
        sys.stdout.flush()

    return Integer(dlog)

def _matvec_chunk(args):
    """Helper for parallel row-wise matrix-vector product."""
    rows, vec, mod = args
    out = [0] * len(rows)
    for local_idx, (i, (idxs, vals)) in enumerate(rows):
        s = 0
        for j, v in zip(idxs, vals):
            s += v * vec[j]
        out[local_idx] = s % mod
    return out

def parallel_matvec(packed_rows, vec, mod, pool):
    """Parallel implementation of A * vec."""
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

def _transpose_matvec_chunk(args):
    """Helper for parallel column-wise (transpose) matrix-vector product."""
    cols, vec, mod = args
    out = [0] * len(cols)
    for local_idx, (j, col) in enumerate(cols):
        s = 0
        for i, c in col:
            s += c * vec[i]
        out[local_idx] = s % mod
    return out

def parallel_transpose_matvec(packed_cols, vec, mod, n, pool):
    """Parallel implementation of A^T * vec."""
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

def matvec_rows(packed_rows, vec, mod, lazy_limit=_LAZY_LIMIT):
    """Single-process, minimal overhead, lazy reduction matvec."""
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

def compute_proj_and_atav(packed_rows, vec, left_vec_b, n_cols, mod, lazy_limit=_LAZY_LIMIT):
    """
    Fused computation of:
      s = A * vec
      proj = left_vec_b^T * s
      atav = A^T * s
    Iterates M = A^T * A. Essential for Wiedemann on rectangular matrices.
    """
    atav = [0] * n_cols
    proj_acc = 0

    atav_loc = atav
    mod_loc = mod
    lazy_loc = lazy_limit

    for (idxs, vals), b_i in zip(packed_rows, left_vec_b):
        s = 0
        for j, a in zip(idxs, vals):
            s += a * vec[j]
            if s > lazy_loc:
                s %= mod_loc
        s %= mod_loc

        if b_i:
            proj_acc += b_i * s
            if proj_acc > lazy_loc:
                proj_acc %= mod_loc

        for j, a in zip(idxs, vals):
            val = atav_loc[j] + a * s
            if val > lazy_loc:
                val %= mod_loc
            atav_loc[j] = val

    proj = proj_acc % mod
    for j in range(n_cols):
        if atav[j]:
            atav[j] %= mod
    return proj, atav

def at_a_v_from_packed(packed_rows, vec, n_cols, mod, lazy_limit=_LAZY_LIMIT):
    zero_left = [0] * len(packed_rows)
    _, atav = compute_proj_and_atav(packed_rows, vec, zero_left, n_cols, mod, lazy_limit)
    return atav

def lift_discrete_log_via_bsgs(d_mod_ell, ell, h, G, Q, verbose=False):
    """Baby-step Giant-step to lift d (mod ell) to the full discrete log."""
    R = Q - Integer(d_mod_ell) * G
    if R.is_zero():
        return int(d_mod_ell)

    H = Integer(ell) * G
    if H.is_zero():
        return None

    bound = int(h)
    m = int(ceil(sqrt(bound)))

    baby = {}
    cur = Integer(0) * H
    for j in range(m):
        key = str(cur)
        if key not in baby:
            baby[key] = j
        cur = cur + H

    factor_gs = Integer(m) * H
    giant = R
    for i in range(0, m + 1):
        key = str(giant)
        if key in baby:
            j = baby[key]
            t = i * m + j
            if t < bound:
                full_d = int((Integer(d_mod_ell) + Integer(t) * Integer(ell)))
                if full_d * G == Q:
                    return full_d
        giant = giant - factor_gs
    return None

def verify_matrix_solution(packed_rows, projected_rhs, solution, mod, verbose=True):
    sol_ints = [int(solution[i]) for i in range(len(solution))]
    for i, (idxs, vals) in enumerate(packed_rows):
        s = 0
        for j, a in zip(idxs, vals):
            s += int(a) * sol_ints[j]
        if (s - int(projected_rhs[i])) % mod != 0:
            return False
    return True

def solve_sparse_direct_mod_ell(A_sparse_matrix, b_list, mod, verbose=True):
    """Direct solver using Sage's internal solve_right for pruned/small systems."""
    A_rows = []
    for (idxs, vals) in A_sparse_matrix.packed_rows:
        row_dict = {int(idx): int(val) for idx, val in zip(idxs, vals)}
        A_rows.append(row_dict)

    n_cols = A_sparse_matrix.n_cols
    n_rows = len(A_rows)

    if n_rows < n_cols:
        raise RuntimeError("Underdetermined system after pruning.")

    M_sage = matrix(GF(mod), n_rows, n_cols, sparse=True)
    b_sage = vector(GF(mod), b_list)

    for i, row_dict in enumerate(A_rows):
        for col, val in row_dict.items():
            M_sage[i, col] = val

    actual_rank = M_sage.rank()
    if actual_rank < n_cols:
        raise RuntimeError(f"Rank deficit: {actual_rank} < {n_cols}")

    return M_sage.solve_right(b_sage)

def find_exact_pivot_columns_sparse(A_rows, mod, verbose=True):
    """Incremental Gaussian elimination to find pivot columns."""
    K = GF(mod)
    pivot_cols = []
    row_echelon = []

    for i, row in enumerate(A_rows):
        if not row: continue
        current = dict(row)
        for pivot_col, pivot_row in zip(pivot_cols, row_echelon):
            if pivot_col in current:
                multiplier = K(current[pivot_col]) / K(pivot_row[pivot_col])
                for col, val in pivot_row.items():
                    current[col] = K(current.get(col, 0) - multiplier * val)
                    if current[col] == 0: del current[col]

        if current:
            leading_col = min(current.keys())
            lead_inv = K(current[leading_col])**(-1)
            current = {col: K(val * lead_inv) for col, val in current.items()}
            pivot_cols.append(leading_col)
            row_echelon.append(current)

    return sorted(pivot_cols)

def block_wiedemann_solve(A, iters=None, verbose=True, left_seed=None, right_seed=None, force_cols=None):
    """
    Standard scalar Wiedemann algorithm using A^T * A.
    Returns (solution_vector, bm_degree).
    """

    mod = int(A.mod)
    n = A.n_cols
    m = len(A.packed_rows)

    if iters is None:
        iters = 2 * min(m,n) + 200
    if iters % 2 != 0: iters += 1

    left_seed = left_seed or random.randrange(1, mod)
    right_seed = right_seed or random.randrange(1, mod)

    rng_left = random.Random(left_seed)
    rng_right = random.Random(right_seed)

    left_vec_b = [rng_left.randrange(mod) for _ in range(m)]
    v_start = [rng_right.randrange(mod) for _ in range(n)]

    if force_cols:
        for col_idx in force_cols:
            if col_idx < len(v_start):
                v_start[col_idx] = rng_right.randrange(1, mod)

    # Pass 1: Krylov
    seq = []
    v = list(v_start)
    for t in range(iters):
        proj, v_next = compute_proj_and_atav(A.packed_rows, v, left_vec_b, n, mod)
        seq.append(proj)
        v = v_next

    # Pass 2: Berlekamp-Massey
    K = GF(mod)
    min_poly = sage_bm([K(s) for s in seq])
    deg = min_poly.degree()

    if deg == 0: return None, 0
    coeffs = [int(min_poly[i]) for i in range(deg + 1)]

    # Pass 3: Reconstruct
    v = list(v_start)
    x_accum = [0] * n
    for i, c in enumerate(coeffs):
        if c != 0:
            for j in range(n):
                x_accum[j] = (x_accum[j] + c * v[j]) % mod
        if i < len(coeffs) - 1:
            v = at_a_v_from_packed(A.packed_rows, v, n, mod)

    return vector(Zmod(mod), x_accum), deg

def prune_factor_base_to_pivot_columns(A_rows, b_list, mod, verbose=True):
    pivot_cols = find_exact_pivot_columns_sparse(A_rows, mod, verbose=verbose)
    col_map = {old_idx: new_idx for new_idx, old_idx in enumerate(pivot_cols)}

    pruned_rows = []
    pruned_rhs = []
    for i, row in enumerate(A_rows):
        pruned_row = {col_map[k]: v for k, v in row.items() if k in col_map}
        if pruned_row:
            pruned_rows.append(pruned_row)
            pruned_rhs.append(b_list[i])

    return pruned_rows, pruned_rhs, col_map, pivot_cols

def expand_solution_to_original(solution_vec, col_map):
    if not col_map: return [int(x) for x in solution_vec]
    n_orig = max(col_map.keys()) + 1
    sol_orig = [0] * n_orig
    for old_idx, new_idx in col_map.items():
        if new_idx < len(solution_vec):
            sol_orig[old_idx] = int(solution_vec[new_idx])
    return sol_orig

def reconstruct_d_from_solution(beta_q, row_q_dict, solution, mod):
    d = int(beta_q) % mod
    for k, v in row_q_dict.items():
        d = (d - int(v) * int(solution[k])) % mod
    return d

