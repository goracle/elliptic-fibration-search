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
from math import ceil
from sage.all import Integer, Zmod, vector, GF, PolynomialRing, matrix
from sage.all import factor
from search_common import BLOCK_WIEDEMANN


# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints


# put this near your other solvers in sparse_linalg_modp.py
from sage.all import GF, PolynomialRing, vector, randint


def wiedemann_solve(A, b, p, max_trials=5, verbosity=1):
    """
    Solve A x = b over GF(p) using Wiedemann + Berlekamp-Massey (scalar version).

    A: sage matrix (m x n)
    b: sage vector (length m)
    p: modulus (int or sage integer)
    Returns: x as sage vector (length n) if successful.
    Raises RuntimeError on unrecoverable failure.
    """
    p = int(p)
    m, n = A.nrows(), A.ncols()
    if len(b) != m:
        raise RuntimeError("wiedemann_solve: dimension mismatch: b length != A.nrows()")

    # Precompute Krylov length - standard choice 2*n
    seq_len = 2 * n

    # Define function to compute sequence s_k = u^T * (A^k * b)
    def compute_sequence(u_vec):
        # compute w0 = b, then w_{k+1} = A * w_k
        w = [vector([int(x) % p for x in list(b)])]
        # we need up to seq_len - 1 A-multiplications
        for k in range(1, seq_len):
            w.append(_matvec_mod(A, w[-1], p))
        # compute sequence s_k = u^T w_k
        s = [int(sum((int(u_vec[i]) * int(wk[i])) for i, wk in enumerate([wk]) )) % p]  # placeholder safe init
        # compute properly
        s = []
        for wk in w:
            # dot product
            s.append(int(sum((int(u_vec[i]) * int(wk[i])) % p for i in range(len(u_vec)))) % p)
        return s, w

    # Slight optimization: instead of recomputing all w for each trial, compute Krylov with respect to b once
    # and reuse across trials. But because we use random u only, Krylov depends only on b and A -> compute once.
    # compute w_k = A^k b
    if verbosity:
        print(f"Wiedemann: preparing Krylov sequence length {seq_len} (this will do {seq_len-1} matvecs)")
    w = [vector([int(x) % p for x in list(b)])]
    for k in range(1, seq_len):
        w.append(_matvec_mod(A, w[-1], p))

    # Try random u vectors
    for trial in range(1, max_trials + 1):
        # random projection u in GF(p)^m (non-zero)
        u = vector([randint(0, p - 1) for _ in range(m)])
        if all(int(x) % p == 0 for x in u):
            u[0] = 1

        # compute scalar sequence s_k = u^T w_k
        s = [int(sum((int(u[i]) * int(wk[i])) % p for i in range(m))) % p for wk in w]

        # BM to get minimal polynomial m(t)
        C = berlekamp_massey(s, p)
        deg = len(C) - 1
        if verbosity:
            print(f"trial {trial}: BM degree = {deg}")

        if deg == 0:
            # deg 0 means sequence all zeros -> useless
            if verbosity:
                print("  BM produced degree 0 polynomial; retrying")
            continue
        if deg > n:
            # unlikely but possible; BM degree > n => try new u
            if verbosity:
                print("  BM degree > n; retrying with new projection")
            continue

        # Create polynomial ring and construct m_poly(t) = C[0] + C[1] t + ... + C[d] t^d
        R = PolynomialRing(GF(p), 't')
        t = R.gen()
        m_poly = sum( (int(C[i]) % p) * t**i for i in range(len(C)) )
        # require m_poly(0) != 0 so gcd(m,t)=1
        if int(m_poly(0)) % p == 0:
            if verbosity:
                print("  minimal polynomial m(0)==0 (not invertible mod t). retrying.")
            continue

        # compute inverse of t modulo m_poly via xgcd: find u_poly, v_poly with u_poly * t + v_poly * m_poly = gcd = 1
        try:
            g, u_poly, v_poly = R(x=t).xgcd(m_poly)  # xgcd returns (g, s, t) with s*x + t*m = g
            # Note: depending on Sage version, xgcd signature may differ; use polynomial xgcd
        except Exception:
            # fallback: use m_poly.xgcd with t
            try:
                g, u_poly, v_poly = m_poly.xgcd(t)
                # m_poly.xgcd(t) returns g, s, t s.t. s*m_poly + t*t = g
                # rearrange to get u_poly * t + v_poly * m_poly = g
                # but s*m_poly + t*t = g  => t*t + s*m_poly = g  => u_poly = t, v_poly = s
                # We want u_poly * t + v_poly * m_poly = g -> u_poly = t, v_poly = s
                # Therefore swap accordingly:
                tmp_u = u_poly
                tmp_v = v_poly
                u_poly = tmp_v  # polynomial multiplying t
                v_poly = tmp_u
            except Exception as e:
                raise RuntimeError(f"wiedemann_solve: polynomial xgcd failed: {e}")

        # ensure gcd is 1
        if int(g(0)) % p != 1 and int(g) != 1:
            # try normalization if g is constant invertible
            if g.degree() == 0 and int(g.constant_coefficient()) % p != 0:
                inv_g = pow(int(g.constant_coefficient()), -1, p)
                u_poly = (u_poly * inv_g)
            else:
                if verbosity:
                    print("  gcd != 1; retrying with new projection")
                continue

        # u_poly is the polynomial such that u_poly * t + v_poly * m_poly = g == 1
        # thus u_poly is the inverse of t modulo m_poly (up to normalization)
        # reduce u_poly modulo m_poly to degree < deg
        q_poly = u_poly % m_poly
        # get coefficients q_0..q_{deg-1}
        q_coeffs = [int(q_poly[i]) % p if i <= q_poly.degree() else 0 for i in range(deg)]
        if verbosity:
            print(f"  got q polynomial degree {len(q_coeffs)-1}; reconstructing x as sum q_i A^i b")

        # Now compute x = sum_{i=0..deg-1} q_i * w_i  (w_i = A^i b precomputed)
        x_vec = vector([0] * n)
        # But note: w_i are length m vectors (A^i b). To compute x in domain of columns,
        # we need to produce x of length n. The polynomial method actually produces x in the domain
        # of A such that A x = b. That requires computing the Krylov with respect to A^T as well, or using standard derivation.
        # The classical Wiedemann technique: q(A) * b yields a vector y in F^n satisfying A*y = b.
        # However, careful: w_i we computed are length m (they are in RHS space). We need the Krylov of A with respect to column-space.
        # For standard Ax=b with A (m x n), the algorithm works with operator on n-space: define M = A (if square),
        # but with rectangular A we treat the normal equations A^T A possibly. To keep it robust, require A to be square here.
        if m != n:
            raise RuntimeError("wiedemann_solve: current scalar Wiedemann implementation requires a square matrix (m == n).")

        # Now n == m, w_i are length n
        for i, qi in enumerate(q_coeffs):
            if qi % p == 0:
                continue
            # add qi * w_i (w_i length n)
            x_vec = vector([(int(x_vec[j]) + qi * int(w[i][j])) % p for j in range(n)])

        # Verify A * x_vec == b
        lhs = _matvec_mod(A, x_vec, p)
        if list(lhs) == [int(x) % p for x in list(b)]:
            if verbosity:
                print(f"Wiedemann: solution verified on trial {trial}")
            return x_vec
        else:
            if verbosity:
                print(f"  verification failed on trial {trial}; retrying")
            # try next random u

    raise RuntimeError(f"wiedemann_solve: failed to find solution after {max_trials} trials")


from sage.all import Integer, Zmod, vector, matrix


# ============================================================================
# FILE: sparse_linalg_modp.py
# FUNCTION: solve_sparse_direct_mod_ell (COMPLETE REWRITE)
# ============================================================================


from sage.all import vector, GF, PolynomialRing, matrix, factor


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

def berlekamp_massey(seq, mod):
    """
    Return connection polynomial C as list C[0..L] (C[0]=1),
    such that for n>=L: sum_{i=0..L} C[i]*s[n-i] = 0 (mod).
    """
    seq = [int(x) % mod for x in seq]
    nmax = len(seq)
    C = [1]
    B = [1]
    L = 0
    m = 1
    b = 1
    for n in range(nmax):
        # discrepancy
        d = seq[n]
        for i in range(1, L+1):
            d = (d + C[i] * seq[n - i]) % mod
        if d == 0:
            m += 1
        else:
            T = C[:]
            coef = (d * pow(int(b), -1, mod)) % mod
            # C = C - coef * x^m * B
            need = len(B) + m
            if len(C) < need:
                C += [0] * (need - len(C))
            for i in range(len(B)):
                C[i + m] = (C[i + m] - coef * B[i]) % mod
            if 2 * L <= n:
                L_new = n + 1 - L
                B = T
                b = d
                m = 1
                L = L_new
            else:
                m += 1
    # trim trailing zeros
    while len(C) > 1 and C[-1] == 0:
        C.pop()
    return [int(x) % mod for x in C]

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


def prune_factor_base_to_pivot_columns(A_rows, b_list, mod, verbose=True):
    """
    Prune factor base to pivot columns via sparse incremental Gaussian elimination.
    
    Returns:
        (pruned_rows, pruned_rhs, col_map, pivot_cols)
    where:
        - pruned_rows: rows with only pivot columns
        - pruned_rhs: corresponding RHS values
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
    col_map = {}
    for new_idx, old_idx in enumerate(pivot_cols):
        col_map[old_idx] = new_idx
    
    # Prune rows to only pivot columns
    pruned_rows = []
    for row in A_rows:
        pruned_row = {}
        for old_idx, val in row.items():
            if old_idx in col_map:
                new_idx = col_map[old_idx]
                pruned_row[new_idx] = int(val)
        if pruned_row:
            pruned_rows.append(pruned_row)
    
    if verbose:
        print(f"  [Prune] Output: {len(pruned_rows)} rows x {len(pivot_cols)} cols")
        sys.stdout.flush()
    
    return pruned_rows, b_list[:len(pruned_rows)], col_map, pivot_cols


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
# FIX 3: Ensure consistent atom-based indexing throughout
# ============================================================================


# ============================================================================
# TRULY MINIMAL FIX: Use Sage BM + force block_size=1 for correctness
# ============================================================================


# ============================================================================
# FIX 3: Atom-based indexing validation
# (Still valid)
# ============================================================================


# ============================================================================
# CORRECT SCALAR WIEDEMANN: Single Krylov chain, no block structure
# ============================================================================


# ============================================================================
# FIX 2: Worker initialization (unchanged, still valid)
# ============================================================================


# ============================================================================
# FIX 3: Atom validation (unchanged, still valid)
# ============================================================================


# ============================================================================
# CORRECT SCALAR WIEDEMANN: Single Krylov chain, no block structure
# ============================================================================


# ============================================================================
# FIX 2: Worker initialization (unchanged, still valid)
# ============================================================================

def _worker_init(gen_mumford, target_mumford, atom_to_idx, sample_roots_int, 
                 fb_y_cache, p_int, order_int, window_size, offset_coeffs, f_coeffs):
    """
    Worker initialization with proper error handling.
    """
    global _GLOBAL_GENERATOR, _GLOBAL_TARGET_POINT
    global _GLOBAL_SAMPLE_ROOTS_INT, _GLOBAL_BABY, _GLOBAL_P, _GLOBAL_ORDER
    global _GLOBAL_WINDOW_SIZE, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY, _GLOBAL_OFFSET_CACHE
    global _GLOBAL_ATOM_TO_IDX
    
    _GLOBAL_ATOM_TO_IDX = atom_to_idx
    _GLOBAL_SAMPLE_ROOTS_INT = sample_roots_int
    _GLOBAL_FB_Y_CACHE = fb_y_cache
    _GLOBAL_P = int(p_int)
    _GLOBAL_ORDER = int(order_int)
    _GLOBAL_WINDOW_SIZE = int(window_size)
    
    K = GF(int(p_int))
    R = PolynomialRing(K, 'x')
    _GLOBAL_F_POLY = sage_poly_from_coeffs(f_coeffs, R)

    C = HyperellipticCurve(_GLOBAL_F_POLY)
    J = C.jacobian()
    
    if gen_mumford is not None:
        gen_u_coeffs, gen_v_coeffs = gen_mumford
        u_poly = R(gen_u_coeffs)
        v_poly = R(gen_v_coeffs)
        _GLOBAL_GENERATOR = J([u_poly, v_poly])
    else:
        _GLOBAL_GENERATOR = None
    
    if target_mumford is not None:
        target_u_coeffs, target_v_coeffs = target_mumford
        u_poly = R(target_u_coeffs)
        v_poly = R(target_v_coeffs)
        _GLOBAL_TARGET_POINT = J([u_poly, v_poly])
    else:
        _GLOBAL_TARGET_POINT = None
    
    zero = J.zero()
    _GLOBAL_BABY = [zero]
    curr = zero
    for _ in range(1, window_size):
        curr = curr + _GLOBAL_GENERATOR
        _GLOBAL_BABY.append(curr)
    
    _GLOBAL_OFFSET_CACHE = []
    if offset_coeffs:
        x = R.gen()
        failed_offsets = []
        for idx, (s, p_val, v0, v1) in enumerate(offset_coeffs):
            try:
                u_poly = x**2 - K(int(s))*x + K(int(p_val))
                v_poly = K(int(v1))*x + K(int(v0))
                _GLOBAL_OFFSET_CACHE.append(J([u_poly, v_poly]))
            except Exception as e:
                failed_offsets.append((idx, s, p_val, e))
        
        if failed_offsets and len(failed_offsets) < len(offset_coeffs):
            print(f"  [Worker] Warning: {len(failed_offsets)}/{len(offset_coeffs)} offset divisors failed")
        elif failed_offsets:
            raise RuntimeError(f"_worker_init: ALL offsets failed: {failed_offsets[0]}")


# ============================================================================
# FIX 3: Atom validation (unchanged, still valid)
# ============================================================================


def block_wiedemann_solve(A, iters=None, verbose=True, 
                          ntrials=1, left_seed=None, right_seed=None):
    """
    CORRECTED: True scalar Wiedemann with single Krylov chain.
    Uses Sage's berlekamp_massey on a valid linear recurrence sequence.
    
    CRITICAL: Explicit seed control for kernel diversity.
    SOLVES: Kernel of A (homogeneous system only).
    
    Args:
        A: SparseRelationMatrix
        iters: number of Krylov iterations (default 3*n + 200)
        left_seed: seed for left projection vector u (if None, use random)
        right_seed: seed for right start vector v (if None, use random)
        
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
    block_size=1,
    nprocs=None,
    max_iters=None,
    max_retry_attempts=20,
):
    """
    Augmented kernel solver for Index Calculus DLP.
    NOW WITH PROPER KERNEL DEGENERACY HANDLING.
    
    ASSUMES: homogeneous_rows already multiplied by cofactor h upstream.
    """
    if atom_to_idx:
        sample_key = next(iter(atom_to_idx.keys()))
        if not isinstance(sample_key, tuple):
            raise RuntimeError(
                f"solve_dlp_mod_l_block_wiedemann: atom_to_idx must be tuple-keyed!\n"
                f"  Got: {type(sample_key)}, expected tuple like ('d1', x_int, y_can)"
            )
    
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(int(p) for p, _ in factors))

    if verbose:
        print(f"  [Kernel Solver] Working mod ℓ={ell}")
        print(f"  [Kernel Solver] Assuming homogeneous rows already h-multiplied")
        sys.stdout.flush()
    
    # Project homogeneous rows to mod ℓ
    projected_rows = []
    for row in homogeneous_rows:
        new_row = {int(k): int(v) % ell for k, v in row.items() if int(v) % ell != 0}
        if new_row:
            projected_rows.append(new_row)

    if not projected_rows:
        raise ValueError("No nonzero homogeneous relations after mod ℓ reduction")

    # Determine Factor Base width
    n_cols_fb = 0
    for r in projected_rows:
        if r:
            n_cols_fb = max(n_cols_fb, max(r.keys()) + 1)
    
    for r in [row_g_dict, row_q_dict]:
        if r:
            n_cols_fb = max(n_cols_fb, max(r.keys()) + 1)

    # Augmented column indices
    col_G = n_cols_fb
    col_Q = n_cols_fb + 1
    n_total_cols = col_Q + 1
    
    if verbose:
        print(f"  [Augmented System] FB columns: {n_cols_fb}")
        print(f"  [Augmented System] G column: {col_G}, Q column: {col_Q}")
        print(f"  [Augmented System] Total columns: {n_total_cols}")
        sys.stdout.flush()

    # Build augmented system
    augmented_rows = list(projected_rows)
    
    row_for_G = {k: -int(v) % ell for k, v in row_g_dict.items()}
    row_for_G[col_G] = 1
    augmented_rows.append(row_for_G)
    
    row_for_Q = {k: -int(v) % ell for k, v in row_q_dict.items()}
    row_for_Q[col_Q] = 1
    augmented_rows.append(row_for_Q)
    
    # Optional row compression
    target_rows = n_total_cols + 200
    if len(augmented_rows) > target_rows:
        if verbose:
            print(f"  [Kernel] Compressing {len(augmented_rows)} rows to ~{target_rows}")
            sys.stdout.flush()
        
        augmented_rows, _ = randomize_rows_for_bw(
            augmented_rows, 
            [0] * len(augmented_rows),
            ell,
            compression_factor=len(augmented_rows) / target_rows,
            mix_count=3,
            verbose=verbose
        )
    
    projected_rhs = [0] * len(augmented_rows)
    A = SparseRelationMatrix(augmented_rows, projected_rhs, ell)
    
    if verbose:
        print(f"  [Solver] Matrix: {len(augmented_rows)} rows x {A.n_cols} cols")
        print(f"  [Solver] Will retry up to {max_retry_attempts} times for good kernel")
        sys.stdout.flush()

    solution = None
    bm_degree = None
    
    # KERNEL DEGENERACY RETRY LOOP WITH EXPLICIT SEED CONTROL
    for attempt in range(max_retry_attempts):
        if verbose:
            print(f"\n  [Attempt {attempt + 1}/{max_retry_attempts}]")
            sys.stdout.flush()
        
        # Generate unique seeds for this attempt
        base_seed = random.randint(0, 2**30)
        left_seed = base_seed + attempt * 999983
        right_seed = base_seed + attempt * 777767 + 123456
        
        t0 = time.time()
        try:
            solution, bm_degree = block_wiedemann_solve(
                A=A,
                iters=max_iters,
                verbose=verbose,
                left_seed=left_seed,
                right_seed=right_seed,
            )
        except Exception as e:
            if verbose:
                print(f"  [BW] Attempt {attempt + 1} raised: {e}")
            if attempt == max_retry_attempts - 1:
                raise
            continue
        
        dt = time.time() - t0
        
        if solution is None:
            if verbose:
                print(f"  [BW] Degree 0 kernel (trivial)")
            if attempt == max_retry_attempts - 1:
                raise RuntimeError("BW failed: trivial kernel after all retries")
            continue
        
        if verbose:
            print(f"  [BW] Solved in {dt:.2f}s, BM degree: {bm_degree}")
            sys.stdout.flush()
        
        if bm_degree < 100:
            if verbose:
                print(f"  [BW] Note: BM degree {bm_degree} is relatively low")
        
        # CHECK FOR USABLE KERNEL: both d_g and d_q must be nonzero
        if len(solution) < n_total_cols:
            solution = list(solution) + [0]*(n_total_cols - len(solution))
        
        d_g_candidate = int(solution[col_G]) % ell
        d_q_candidate = int(solution[col_Q]) % ell
        
        if d_g_candidate == 0:
            if verbose:
                print(f"  [Kernel] Degeneracy: d_g=0 (kernel doesn't involve G)")
            continue
        
        if d_q_candidate == 0:
            if verbose:
                print(f"  [Kernel] Degeneracy: d_q=0 (kernel doesn't involve Q)")
            continue
        
        # Found usable kernel with both d_g != 0 and d_q != 0
        if verbose:
            print(f"  [Kernel] ✓ Found usable kernel: d_g={d_g_candidate}, d_q={d_q_candidate}")
        break
    else:
        raise RuntimeError(
            f"Failed to find kernel with both d_g ≠ 0 and d_q ≠ 0 after {max_retry_attempts} attempts.\n"
            f"This suggests nullspace is dominated by FB-only relations.\n"
            f"Consider: more relations, different FB, or block Wiedemann."
        )

    # Extract discrete log from kernel vector
    d_g = int(solution[col_G]) % ell
    d_q = int(solution[col_Q]) % ell
    
    assert d_g != 0, "d_g must be nonzero (validated above)"
    assert d_q != 0, "d_q must be nonzero (validated above)"
    
    d_g_inv = pow(int(d_g), -1, ell)
    dlog = (d_q * d_g_inv) % ell

    if verbose:
        print(f"  [Kernel] d_g={d_g}, d_q={d_q}, dlog mod ℓ = {dlog}")
        sys.stdout.flush()

    # Verification
    D = Integer(dlog) * G - Q
    
    if not D.is_zero():
        if verbose:
            print(f"  [Verify] ✗ FAILED: dlog * G ≠ Q")
        raise AssertionError(f"Verification failed: dlog={dlog}, ℓ={ell}")

    if verbose:
        print("  [Verify] ✓ dlog * G == Q")

    return Integer(dlog)
