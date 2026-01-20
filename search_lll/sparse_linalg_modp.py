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


# Example wrapper that keeps your solve_dlp_mod_l_block_wiedemann outer flow, but uses the new fast core:
def block_wiedemann_solve_wrapper(A, b, block_size=32, iters=None, verbose=True):
    """
    Thin wrapper preserving previous call signature.
    """
    return block_wiedemann_solve(A, b, block_size=block_size, iters=iters, verbose=verbose)

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
from sage.all import GF, PolynomialRing, vector, randint

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


def block_wiedemann_solve(A, b, block_size=1, iters=None, verbose=True, ntrials=1):
    mod = int(A.mod)
    m = len(A.packed_rows)
    n = A.n_cols
    
    # Wiedemann complexity: ~2N iterations for scalar, 2N/B for blocked
    # We add a small safety buffer (+50)
    # In block_wiedemann_solve:
    if iters is None:
        base_iters = int(3.0 * n // max(1, block_size))  # Was 2.2, now 3.0
        iters = base_iters + 200  # Was +50, now +200

    if verbose:
        print(f"[BW-fast] block={block_size}, target_iters={iters}, nrows={m}, ncols={n}")
        sys.stdout.flush()

    left_vec_b = [int(x) % mod for x in b]
    seed_val = random.randrange(1, mod)
    
    # --- PASS 1: Generate Krylov Sequence ---
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

    # --- POLYNOMIAL STEP: Berlekamp-Massey ---
    if verbose:
        print(f"[BW-fast] Computing Minimal Polynomial from {len(seq)} scalars...")
        sys.stdout.flush()
    
    # Convert sequence to GF(mod) for Berlekamp-Massey
    K = GF(mod)
    seq_mod = [K(s) for s in seq]
    
    # Call custom Berlekamp-Massey (returns list of int coefficients)
    coeffs = berlekamp_massey(seq_mod, mod)
    deg = len(coeffs) - 1
    
    if verbose:
        print(f"[BW-fast] Minimal polynomial degree: {deg}")
        sys.stdout.flush()

    # Safety check: if degree is suspiciously low
    if deg <= block_size * 2 or deg < 100:
        print(f"  [BW-fast] WARNING: Poly degree {deg} is suspiciously low (system size {n}).")
        print(f"            This suggests the Krylov sequence degenerated. Result will likely be wrong.")

    # --- PASS 2: Reconstruct Solution Vector ---
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

    return vector(Zmod(mod), x_accum), deg


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
    n_target = n_original // compression_factor
    
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


def solve_dlp_mod_l_block_wiedemann(
    homogeneous_rows,
    row_g_dict,
    alpha_g,
    row_q_dict,
    beta_q,
    full_order,
    G, Q,
    atom_to_idx,  # ← ADD THIS PARAMETER
    J,            # ← ADD THIS PARAMETER
    *,
    verbose=True,
    block_size=1,
    nprocs=None,
    max_iters=None,
    max_retry_attempts=3,
):
    """
    CORRECTED: Traditional Index Calculus kernel solver.
    
    System structure:
        A x = 0  (homogeneous relations in ℓ-torsion)
        
    Extract dlog from:
        d = (β_q - row_q · x) / (1 + α_g - row_g · x)  (mod ℓ)
    
    CRITICAL: homogeneous_rows are ALREADY in J[ℓ] (projected by h before calling).
              row_g and row_q are ALREADY in J[ℓ] (G,Q are ℓ-torsion by construction).
    
    Args:
        homogeneous_rows: list of relation dicts (already h-projected to J[ℓ])
        row_g_dict: G encoding as factor base row
        alpha_g: G smoothing offset
        row_q_dict: Q encoding as factor base row
        beta_q: Q smoothing offset
        full_order: full Jacobian order
        G, Q: Jacobian elements for verification
        atom_to_idx: factor base atom map (needed for relation verification)
        J: Jacobian (needed for relation reconstruction)
    """
    if nprocs is None:
        nprocs = max(1, cpu_count() - 1)

    # Compute ℓ (largest prime factor) and cofactor h
    J_order = Integer(full_order)
    factors = factor(J_order)
    ell = int(max(int(p) for p, _ in factors))
    h = int(J_order // ell)

    if verbose:
        print(f"  [Kernel Solver] Working mod ℓ={ell}, cofactor h={h}")
        print(f"  [Kernel Solver] Homogeneous relations already in J[ℓ]")
        sys.stdout.flush()
    
    # === SANITY CHECK: Verify homogeneous relations are in ℓ-torsion ===
    # Sample check a few relations to ensure h-projection worked
    if verbose and atom_to_idx is not None and J is not None:
        print(f"  [Sanity] Checking that homogeneous relations are ℓ-torsion...")
        sys.stdout.flush()
        
        try:
            verify_all_relations_are_ell_torsion(
                homogeneous_rows, atom_to_idx, ell, J,
                sample_size=min(100, len(homogeneous_rows)),
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  [Sanity] WARNING: Relation verification failed: {e}")
                print(f"  [Sanity] Proceeding anyway (verification may be disabled)")
            # Don't crash - just warn
    
    # === BUILD HOMOGENEOUS SYSTEM (kernel) ===
    projected_rows = []
    for row in homogeneous_rows:
        new_row = {int(k): int(v) % ell for k, v in row.items() if int(v) % ell != 0}
        if new_row:
            projected_rows.append(new_row)

    if not projected_rows:
        raise ValueError("No nonzero homogeneous relations after mod ℓ reduction")

    n_cols = max(k for row in projected_rows for k in row) + 1
    
    if verbose:
        nnz = sum(len(r) for r in projected_rows)
        print(f"  [System] Homogeneous system: {len(projected_rows)} rows x {n_cols} cols, nnz={nnz}")
        sys.stdout.flush()

    # === PRUNING STEP ===
    if verbose:
        print(f"  [Kernel] Pruning to pivot columns...")
        sys.stdout.flush()
    
    pruned_rows, _, col_map, pivot_cols = prune_factor_base_to_pivot_columns(
        projected_rows, [0] * len(projected_rows), ell, verbose=verbose
    )
    
    if not pruned_rows:
        raise RuntimeError("All rows vanished during pruning!")
    
    n_cols_pruned = len(pivot_cols)
    
    # Remap G and Q rows to pruned columns
    row_g_pruned = {}
    for old_idx, val in row_g_dict.items():
        new_idx = col_map.get(old_idx)
        if new_idx is not None:
            row_g_pruned[new_idx] = int(val) % ell
    
    row_q_pruned = {}
    for old_idx, val in row_q_dict.items():
        new_idx = col_map.get(old_idx)
        if new_idx is not None:
            row_q_pruned[new_idx] = int(val) % ell
    
    if not row_g_pruned:
        raise RuntimeError("G row vanished after pruning!")
    if not row_q_pruned:
        raise RuntimeError("Q row vanished after pruning!")
    
    if verbose:
        print(f"  [Pruned System] {len(pruned_rows)} rows x {n_cols_pruned} cols")
        print(f"  [Verify] ✓ G row survived: {len(row_g_pruned)} nonzero entries")
        print(f"  [Verify] ✓ Q row survived: {len(row_q_pruned)} nonzero entries")
        sys.stdout.flush()
    
    # Update to pruned versions
    projected_rows = pruned_rows
    n_cols = n_cols_pruned
    row_g_dict = row_g_pruned
    row_q_dict = row_q_pruned

    # === ROW MIXING (if overdetermined) - ONLY on homogeneous rows ===
    row_mixing_applied = False
    if len(projected_rows) > n_cols * 2:
        if verbose:
            print(f"  [Kernel] System is {len(projected_rows)/n_cols:.1f}x overdetermined")
            print(f"  [Kernel] Applying random row mixing to improve Krylov diversity...")
            sys.stdout.flush()
        
        projected_rows, _ = randomize_rows_for_bw(
            projected_rows, 
            [0] * len(projected_rows),  # All zeros (homogeneous)
            ell,
            compression_factor=2,
            mix_count=4,
            verbose=verbose
        )
        row_mixing_applied = True

    # Build RHS (all zeros for kernel)
    projected_rhs = [0] * len(projected_rows)

    # Build SparseRelationMatrix
    A = SparseRelationMatrix(projected_rows, projected_rhs, ell)
    
    # === KERNEL SOLVER ===
    if not BLOCK_WIEDEMANN and n_cols < 10000:
        if verbose:
            print(f"  [Solver] Using direct sparse kernel solve (n={n_cols} < 10k)")
            sys.stdout.flush()
        
        # Direct kernel solve
        K = GF(ell)
        entries = {}
        for i, row in enumerate(projected_rows):
            for j, v in row.items():
                entries[(i, j)] = K(int(v))
        
        M = matrix(K, len(projected_rows), n_cols, entries, sparse=True)
        
        if verbose:
            print(f"  [Solver] Computing kernel...")
            sys.stdout.flush()
        
        kernel = M.right_kernel()
        
        if kernel.dimension() == 0:
            raise RuntimeError("Kernel is trivial - no solution exists!")
        
        # Pick any kernel vector
        solution = kernel.basis()[0]
        
    else:
        if verbose:
            print(f"  [Solver] Using Block-Wiedemann kernel solver (n={n_cols} >= 10k)")
            sys.stdout.flush()

        solution = None
        bm_degree = None
        adaptive_block = max(64, min(128, n_cols // 50))
        
        for attempt in range(max_retry_attempts):
            if attempt > 0 and verbose:
                print(f"\n  [Retry] Attempt {attempt + 1}/{max_retry_attempts} (previous BM degree was {bm_degree})")
                sys.stdout.flush()
            
            base_seed = random.randint(0, 2**30)
            attempt_seed = base_seed + attempt * 999983
            random.seed(attempt_seed)
            
            if verbose and attempt == 0:
                print(f"  [BW] Initial random seed: {base_seed}")
                sys.stdout.flush()
            
            t0 = time.time()
            try:
                solution, bm_degree = block_wiedemann_solve(
                    A=A,
                    b=projected_rhs,
                    block_size=adaptive_block,
                    iters=max_iters,
                    verbose=verbose,
                )
            except Exception as e:
                if verbose:
                    print(f"  [BW] Attempt {attempt + 1} raised exception: {e}")
                if attempt == max_retry_attempts - 1:
                    raise
                continue
            
            dt = time.time() - t0
            
            if solution is None:
                if verbose:
                    print(f"  [BW] Attempt {attempt + 1} returned None")
                if attempt == max_retry_attempts - 1:
                    raise RuntimeError("Block-Wiedemann failed to converge after all retries")
                continue
            
            if verbose:
                print(f"  [BW] Solved in {dt:.2f}s, BM degree: {bm_degree}")
                sys.stdout.flush()
            
            min_expected_degree = n_cols // 100
            
            if bm_degree < min_expected_degree:
                if verbose:
                    print(f"  [BW] WARNING: BM degree {bm_degree} < {min_expected_degree} (n/100)")
                    print(f"  [BW] This suggests Krylov degeneration. Retrying with different seed...")
                
                if attempt < max_retry_attempts - 1:
                    continue
                else:
                    if verbose:
                        print(f"  [BW] Final attempt has low degree. Proceeding to verification anyway...")
            else:
                if verbose:
                    print(f"  [BW] BM degree {bm_degree} looks reasonable (>= n/100)")
                break

    # === EXTRACT DISCRETE LOG FROM KERNEL VECTOR ===
    # Traditional Index Calculus:
    #   G smoothed to: (1 + α_g) * G = Σ row_g[i] * FB[i]
    #   Q smoothed to: Q + β_q * G = Σ row_q[i] * FB[i]
    #   Kernel gives: FB[i] = Σ solution[i] * (some basis)
    #
    # From kernel x:
    #   row_g · x = (1 + α_g) * (scalar for G)
    #   row_q · x = (scalar for Q + β_q * G)
    #
    # Therefore:
    #   d = (row_q · x - β_q * row_g · x) / row_g · x  (mod ℓ)
    
    # Compute row_g · solution
    g_sum = Integer(0)
    for k, v in row_g_dict.items():
        if k < len(solution):
            g_sum += Integer(v) * Integer(solution[k])
    g_sum = int(g_sum % ell)
    
    # Compute row_q · solution
    q_sum = Integer(0)
    for k, v in row_q_dict.items():
        if k < len(solution):
            q_sum += Integer(v) * Integer(solution[k])
    q_sum = int(q_sum % ell)
    
    if g_sum == 0:
        raise RuntimeError("Kernel solution gives zero for G encoding - degeneracy!")
    
    # d = (q_sum - β_q * g_sum) / g_sum  (mod ℓ)
    numerator = (q_sum - beta_q * g_sum) % ell
    g_sum_inv = pow(int(g_sum), -1, ell)
    dlog = (numerator * g_sum_inv) % ell

    if verbose:
        print(f"  [Kernel] row_g · x = {g_sum} (mod ℓ)")
        print(f"  [Kernel] row_q · x = {q_sum} (mod ℓ)")
        print(f"  [Kernel] Candidate dlog mod ℓ = {dlog}")
        sys.stdout.flush()

    # === VERIFICATION ===
    # CORRECTED: Since G and Q are ALREADY ℓ-torsion, just check dlog * G == Q directly
    D = Integer(dlog) * G - Q
    
    if not D.is_zero():
        if verbose:
            print(f"  [Verify] ✗ FAILED: dlog * G ≠ Q")
            print(f"            dlog = {dlog}, ℓ = {ell}")
            # Additional diagnostics
            print(f"            Testing ℓ * D:")
            ellD = Integer(ell) * D
            print(f"              ℓ * D is zero? {ellD.is_zero()}")
            print(f"            Testing h * D:")
            hD = Integer(h) * D
            print(f"              h * D is zero? {hD.is_zero()}")
        raise AssertionError(
            f"[Verify] ✗ Verification failed:\n"
            f"  dlog * G ≠ Q\n"
            f"  dlog = {dlog}, ℓ = {ell}"
        )

    if verbose:
        print("  [Verify] ✓ Exact equality: dlog * G == Q")

    return Integer(dlog)
