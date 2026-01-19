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
from sage.all import Integer, Zmod, vector
from sage.all import Integer, Zmod, vector, GF
from math import ceil
from sage.all import Integer, Zmod, vector, GF, PolynomialRing, matrix
from sage.all import factor
from search_common import BLOCK_WIEDEMANN


# Tunable threshold for lazy reduction
_LAZY_LIMIT = (1 << 61) - 1  # safe headroom for Python ints


# Example wrapper that keeps your solve_dlp_mod_l_block_wiedemann outer flow, but uses the new fast core:


def perform_dlp_attack(G, Q, smooth_divs_or_rels, p, f_coeffs, order,
                       verbose=True, force_index_calculus=False):
    """
    Robust wrapper for Index-Calculus DLP.
    FIXED: Properly constructs inhomogeneous system for G and Q.
    
    CRITICAL: G and Q are assumed to already be in J(F_p)[ℓ] (ℓ-torsion).
    The system we solve is:
      homogeneous_rows · x = 0  (smooth divisor relations)
      row_g · x = 1 + alpha_g   (G smoothing equation)
    Then extract d from: row_q · x = beta_q (Q smoothing equation)
    """
    # Basic validation
    if G is None or Q is None:
        raise ValueError("Generator G and target Q must be provided")
    if order is None or int(order) <= 0:
        raise ValueError("Invalid Jacobian order provided")

    full_order = Integer(order)

    # Check if precomputed relations
    precomputed = False
    if (isinstance(smooth_divs_or_rels, (list, tuple)) and len(smooth_divs_or_rels) == 1
            and isinstance(smooth_divs_or_rels[0], dict)
            and smooth_divs_or_rels[0].get('type') == 'relations'):
        precomputed = True

    # Prepare polynomial ring and curve
    K = GF(p)
    R = PolynomialRing(K, 'x')
    f_p = sage_poly_from_coeffs(f_coeffs, R)
    C = HyperellipticCurve(f_p)
    J = C.jacobian()

    if verbose:
        print(f"\n{'='*70}")
        print(f"INDEX CALCULUS DLP ATTACK (Full Jacobian Group)")
        print(f"{'='*70}")
        print(f"Full Jacobian order |J|: {full_order}")

    # Build or extract factor base and homogeneous relations
    if precomputed:
        data = smooth_divs_or_rels[0]
        homogeneous_rows = data['relations']
        fb_roots = data['fb_roots']
        r_to_idx = data['fb_map']
        
        fb_y_cache = {}
        for x_int in fb_roots:
            y2 = int(f_p(K(x_int)))
            if y2 == 0:
                fb_y_cache[x_int] = 0
            else:
                y_can = tonelli_shanks(y2, p)
                fb_y_cache[x_int] = int(min(y_can, p - y_can))
    else:
        # Legacy path: build from Mumford divisors
        if verbose:
            print("  [Legacy] Building factor base and relations from Mumford divisors...")
        
        # CRITICAL: _legacy_build_relations_from_mumford returns (rows, rhs, fb_roots, r_to_idx, fb_y_cache)
        # The rhs should be all zeros (homogeneous relations only)
        homogeneous_rows, homogeneous_rhs, fb_roots, r_to_idx, fb_y_cache = \
            _legacy_build_relations_from_mumford(smooth_divs_or_rels, G, Q, p, f_coeffs, verbose=verbose)
        
        # Verify all RHS are zero (homogeneous)
        if any(r != 0 for r in homogeneous_rhs):
            raise RuntimeError(
                f"_legacy_build_relations_from_mumford returned non-homogeneous relations:\n"
                f"  Found {sum(1 for r in homogeneous_rhs if r != 0)} nonzero RHS values"
            )

    if not homogeneous_rows:
        raise RuntimeError("No valid homogeneous relations available")

    if verbose:
        print(f"  [Relations] Loaded {len(homogeneous_rows)} homogeneous relations")
        print(f"  [Factor Base] Size: {len(r_to_idx)}")
        sys.stdout.flush()

    # --- CRITICAL FIX: Build inhomogeneous rows for G and Q separately ---
    
    # Smooth G
    row_g = None
    alpha_g = 0
    
    if is_divisor_fb_smooth(G, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
        row_g = canonicalize_divisor_to_factor_base(G, r_to_idx, f_p, p) or \
                _build_signed_row_from_divisor(G, r_to_idx, f_p, p)
        if verbose:
            print("  [Smoothing] Generator G is already smooth.")
    else:
        if verbose:
            print("  [Smoothing] Generator not smooth. Attempting random smoothing...")
        for i in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_G = (1 + r) * G
            if is_divisor_fb_smooth(cand_G, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
                row_g = canonicalize_divisor_to_factor_base(cand_G, r_to_idx, f_p, p) or \
                        _build_signed_row_from_divisor(cand_G, r_to_idx, f_p, p)
                if row_g:
                    alpha_g = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth generator at iter {i}")
                    break
    
    if row_g is None:
        raise RuntimeError("Failed to smooth Generator G")

    # Smooth Q
    row_q = None
    beta_q = 0
    
    if is_divisor_fb_smooth(Q, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
        row_q = canonicalize_divisor_to_factor_base(Q, r_to_idx, f_p, p) or \
                _build_signed_row_from_divisor(Q, r_to_idx, f_p, p)
        if verbose:
            print("  [Smoothing] Target Q is already smooth.")
    else:
        if verbose:
            print("  [Smoothing] Target not smooth. Attempting random smoothing...")
        for i in range(1, 2001):
            r = ZZ.random_element(1, int(full_order))
            cand_Q = Q + r * G
            if is_divisor_fb_smooth(cand_Q, r_to_idx, f_p, p, fb_y_cache=fb_y_cache):
                row_q = canonicalize_divisor_to_factor_base(cand_Q, r_to_idx, f_p, p) or \
                        _build_signed_row_from_divisor(cand_Q, r_to_idx, f_p, p)
                if row_q:
                    beta_q = r
                    if verbose:
                        print(f"  [Smoothing] Found smooth target at iter {i}")
                    break
    
    if row_q is None:
        raise RuntimeError("Failed to smooth Target Q")

    # --- CRITICAL: Construct the FULL system with proper RHS values ---
    # The system is now:
    #   homogeneous_rows[i] · x = 0  (for i in smooth divisors)
    #   row_g · x = 1 + alpha_g      (inhomogeneous for G)
    
    # Append G row to the system
    full_rows = list(homogeneous_rows)  # Copy
    full_rhs = [0] * len(homogeneous_rows)  # All zeros for homogeneous
    
    full_rows.append({int(k): int(v) for k, v in row_g.items()})
    full_rhs.append(int(1 + alpha_g))  # CRITICAL: Non-zero RHS for G
    
    if verbose:
        print(f"\n  [System] Built system: {len(full_rows)} rows ({len(homogeneous_rows)} homogeneous + 1 for G)")
        print(f"  [System] G row RHS: {full_rhs[-1]} (should be 1 + {alpha_g})")
        sys.stdout.flush()

    # Solve the system
    beta_q_int = int(beta_q)
    row_q_dict = {int(k): int(v) for k, v in row_q.items()}

    d_log_val = None
    try:
        if verbose:
            print("  [Solver] Starting Block-Wiedemann...")
        
        d_log_val = solve_dlp_mod_l_block_wiedemann(
            full_rows,  # ← Now includes G with proper RHS
            full_rhs,   # ← Now has [0, 0, ..., 0, 1+alpha_g]
            row_q_dict,
            beta_q_int,
            full_order,
            G, Q,
            verbose=verbose,
            block_size=32,
        )
        
        if verbose:
            print("  [Solver] Block-Wiedemann returned a candidate.")
    except Exception as e:
        raise RuntimeError(f"Block-Wiedemann solver failed: {e}")

    if d_log_val is None:
        raise RuntimeError("Solver produced no result")

    if verbose:
        print(f"  [Result] Discrete log (mod ℓ) candidate: {d_log_val}")

    # Verify
    ell = int(max(int(p) for p, _ in factor(full_order)))
    D = Integer(d_log_val) * G - Q

    if not (Integer(ell) * D).is_zero():
        raise RuntimeError(
            "[Verify] ✗ Block-Wiedemann result FAILED group verification:\n"
            f"        ℓ * (d_log_val * G − Q) ≠ 0\n"
            f"        dlog={d_log_val}, ℓ={ell}"
        )

    if verbose:
        print("  [Verify] ✓ ℓ-torsion verification passed")
        if D.is_zero():
            print("  [Verify] ✓ Exact equality d*G == Q")
        else:
            print("  [Verify] ℹ d*G ≠ Q exactly (cofactor component)")

    return Integer(d_log_val)


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
):
    """
    Sparse Block-Wiedemann wrapper for DLP modulo largest prime ell.
    Returns the discrete log modulo ℓ (prime subgroup).
    Automatically switches to Direct Sparse Solver for n_cols < 10,000.
    
    CRITICAL FIX: Relations from Mumford search hold in J(F_p), not J[ℓ].
    We must multiply by cofactor h before reducing mod ℓ to get valid J[ℓ] relations.
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
        print(f"  [BW] CRITICAL: Multiplying relations by h={h} before mod ℓ reduction")
        sys.stdout.flush()
    
    # CRITICAL FIX: Multiply Q-relation by h as well
    beta_q_l = int((Integer(beta_q) * Integer(h)) % ell)
    row_q_l = {k: int((Integer(v) * Integer(h)) % ell) for k, v in row_q_dict.items()}

    projected_rows = []
    projected_rhs = []

    for row, rhs in zip(valid_rows, rhs_values):
        # CRITICAL FIX: Multiply by h BEFORE reducing mod ℓ
        # This ensures relations hold in J[ℓ], not just J(F_p)
        new_row = {}
        for k, v in row.items():
            vk = int((Integer(v) * Integer(h)) % ell)
            if vk:
                new_row[k] = vk
        
        rhs_proj = int((Integer(rhs) * Integer(h)) % ell)
        
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
        print(f"  [BW] After h-multiplication, all relations now hold in J[ℓ]")
        sys.stdout.flush()

    # Build SparseRelationMatrix
    A = SparseRelationMatrix(projected_rows, projected_rhs, ell)
    
    # Decide which solver to use
    # Threshold for direct solve: 10,000 columns
    if not BLOCK_WIEDEMANN and n_cols < 10000:
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

    # CRITICAL: Extract dlog using h-multiplied row_q
    dlog = Integer(beta_q_l)
    for k, v in row_q_l.items():
        if k < len(solution):
            coeff = int(solution[k])
            dlog = (dlog - Integer(v) * Integer(coeff)) % Integer(ell)
    dlog = int(dlog)

    if verbose:
        print("  [BW] Candidate dlog mod ℓ =", dlog)
        sys.stdout.flush()

    # Verify in J[ℓ]
    D = Integer(dlog) * G - Q
    if not D.is_zero():
        # Before failing, try the diagnostic
        if verbose:
            print("  [BW] Direct verification failed, running diagnostics...")
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
            f"  dlog = {dlog}, ℓ = {ell}"
        )

    if verbose:
        print("  [BW-Verify] ✓ Exact equality dlog * G == Q in prime subgroup")

    return int(dlog)


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

def solve_sparse_direct_mod_ell(A_obj, b_vec_ints, mod, verbose=True):
    """
    Direct sparse solve using Sage's built-in solver over Zmod(mod).
    Recommended for systems with < 10k unknowns.
    
    Fixed: Forces inclusion of non-zero RHS rows during sampling.
    """
    if verbose:
        print(f"  [Direct] Building sparse matrix over Zmod({mod})...")
        sys.stdout.flush()
    
    t0 = time.time()
    K = Zmod(mod)
    n_rows = len(A_obj.packed_rows)
    n_cols = A_obj.n_cols
    
    # Identify critical rows (non-zero RHS)
    critical_indices = [i for i, b in enumerate(b_vec_ints) if int(b) % int(mod) != 0]
    
    # Sampling logic
    if n_rows > 2 * n_cols:
        target_rows = int(1.2 * n_cols)
        
        # Start with all critical rows
        indices = set(critical_indices)
        
        # Fill the rest with random rows
        remaining_slots = target_rows - len(indices)
        if remaining_slots > 0:
            candidate_pool = [i for i in range(n_rows) if i not in indices]
            # Safety check if pool is smaller than needed (unlikely given check above)
            sample_size = min(len(candidate_pool), remaining_slots)
            indices.update(random.sample(candidate_pool, sample_size))
            
        indices = sorted(list(indices))
        
        if verbose:
            print(f"  [Direct] Sampling {len(indices)} rows (forced {len(critical_indices)} non-zero RHS).")
            sys.stdout.flush()
    else:
        indices = range(n_rows)
    
    # Build dictionary using sampled rows
    entries = {}
    sampled_b = []
    
    # Map old row index to new row index (0, 1, 2...)
    for new_row_idx, old_row_idx in enumerate(indices):
        idxs, vals = A_obj.packed_rows[old_row_idx]
        for j, v in zip(idxs, vals):
            entries[(new_row_idx, j)] = K(int(v))
        sampled_b.append(K(int(b_vec_ints[old_row_idx])))
    
    # Construct Sage sparse matrix
    M = matrix(K, len(sampled_b), n_cols, entries, sparse=True)
    b_sage = vector(K, sampled_b)
    
    if verbose:
        nnz = len(entries)
        size = len(sampled_b) * n_cols
        density = 100.0 * nnz / size if size > 0 else 0
        print(f"  [Direct] Matrix: {len(sampled_b)} x {n_cols}, nnz={nnz}, density={density:.4f}%")
        print(f"  [Direct] Solving system (backend=Sage/generic)...")
        sys.stdout.flush()
    
    try:
        solution = M.solve_right(b_sage)
    except ValueError as e:
        # Re-raise with context
        raise RuntimeError(f"Direct sparse solve failed (inconsistent or underdetermined): {e}")

    dt = time.time() - t0
    
    if verbose:
        print(f"  [Direct] ✓ Solved in {dt:.2f}s")
        sys.stdout.flush()
        
    return solution
