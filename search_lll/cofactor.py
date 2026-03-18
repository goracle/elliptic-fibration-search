import sys, time, random
from math import ceil, sqrt, gcd
from multiprocessing import Pool, cpu_count
from collections import Counter, deque
from sage.all import Integer, Zmod, GF, ZZ, matrix, vector, PolynomialRing, factor, crt, prime_factors, set_random_seed
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from sage.matrix.berlekamp_massey import berlekamp_massey

# Add to index_calculus.py
# Standard library

# Sage imports (consolidated)

# === Minimal detection function: compute right-kernel of the projected homogeneous matrix ===

# === Sanity-check routine: verify basis vectors annihilate each relation ===

# ===== helpers to add near top of file =====

def _safe_get_ell_and_h(full_order):
    primes = prime_factors(Integer(full_order))
    if not primes:
        raise RuntimeError("No prime factors found for full_order")
    ell = int(max(primes))
    h = int(Integer(full_order) // Integer(ell))
    return ell, h

def verify_character_vectors(filtered_rows, alive_idx_list, basis_vectors, ell, verbose=True):
    col_map = {old_idx: col for col, old_idx in enumerate(alive_idx_list)}
    Zell = Zmod(int(ell))
    ok_list = []
    for v in basis_vectors:
        # pick a few random FB atoms
        print("vector:", v)
        for idx in random.sample(alive_idx_list, 10):
            print(idx, v[col_map[idx]])
        ok = True
        for r in filtered_rows:
            s = 0
            for old_idx, mult in r.items():
                j = col_map.get(old_idx, None)
                if j is None:
                    continue
                s += int(mult) * int(v[j])
            if int(Zell(s % int(ell))) != 0:
                ok = False
                break
        ok_list.append(ok)
    if verbose:
        for i, ok in enumerate(ok_list):
            print(f"  char basis #{i} verifies relations: {ok}")

    return ok_list

# Put these into your module in place of the originals.

# ---- apply_cofactor_filter (fixed remap + defensive G/Q) ----

# ---- detect_nontrivial_character_from_projection (non-fatal, GF speed) ----

# ---- is_vector_in_rowspace (clearer) ----

# ---- selection helpers: safe fallback + incremental rank test ----

# ---- choose_prune_row_guided_by_g (robust sampling) ----

# ===== Replacements: robust implementations =====
# Assumes imports:
# from collections import deque
# import random
# from sage.all import GF, Zmod, matrix, vector, Integer, prime_factors

def apply_cofactor_filter(precheck_result, atom_to_idx, homogeneous_rows,
                          row_g, row_q, verbose=True):
    """
    Safe remap of master factor base -> filtered factor base using
    precheck_result['alive_fb_indices'].

    Returns (filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q)
    """
    if not precheck_result.get('safe_to_project', False):
        raise RuntimeError(
            f"Cannot apply filter: projection is unsafe. Reason: {precheck_result.get('reason')}"
        )

    alive_indices = set(precheck_result['alive_fb_indices'])

    # Build filtered_atom_to_idx and direct old->new map
    filtered_atom_to_idx = {}
    old_to_new = {}
    for atom, old_idx in atom_to_idx.items():
        if old_idx in alive_indices:
            new_idx = len(filtered_atom_to_idx)
            filtered_atom_to_idx[atom] = new_idx
            old_to_new[old_idx] = new_idx

    # Remap filtered_rows defensively
    filtered_rows = []
    for row in precheck_result.get('filtered_rows', []):
        new_row = {}
        for old_idx, mult in row.items():
            if old_idx in old_to_new:
                new_row[old_to_new[old_idx]] = int(mult)
        if new_row:
            filtered_rows.append(new_row)

    # Remap G, Q defensively (skip dead indices)
    filtered_row_g = {}
    for old_idx, mult in precheck_result.get('filtered_row_g', {}).items():
        if old_idx in old_to_new:
            filtered_row_g[old_to_new[old_idx]] = int(mult)

    filtered_row_q = {}
    for old_idx, mult in precheck_result.get('filtered_row_q', {}).items():
        if old_idx in old_to_new:
            filtered_row_q[old_to_new[old_idx]] = int(mult)

    if verbose:
        print(f"\n[Filter Applied]")
        print(f"  Original FB size: {len(atom_to_idx)}")
        print(f"  Filtered FB size: {len(filtered_atom_to_idx)}")
        print(f"  Original relations: {len(homogeneous_rows)}")
        print(f"  Filtered relations: {len(filtered_rows)}")
        print(f"  Removed {len(atom_to_idx) - len(filtered_atom_to_idx)} dead FB elements")
        print(f"  Removed {len(homogeneous_rows) - len(filtered_rows)} dead relations")

    return filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q

def detect_nontrivial_character_from_projection(filtered_rows, alive_fb_indices, ell, verbose=True):
    """
    Return dict giving the nullspace (right-kernel) of the projected homogeneous matrix
    over GF(ell). Non-fatal: returns found=False when dim != 1 so caller decides.
    """
    if int(ell) <= 2:
        return {'found': False, 'dim': 0, 'basis': [], 'alive_idx_list': sorted(alive_fb_indices)}

    F = GF(int(ell))
    alive_idx_list = sorted(alive_fb_indices)
    col_map = {old_idx: c for c, old_idx in enumerate(alive_idx_list)}
    n_cols = len(alive_idx_list)
    n_rows = len(filtered_rows)

    # build dense rows (list of lists over F)
    rows = []
    for row in filtered_rows:
        vec = [F(0)] * n_cols
        for old_idx, mult in row.items():
            j = col_map.get(old_idx)
            if j is not None:
                vec[j] = F(int(mult) % int(ell))
        rows.append(vec)

    if n_rows == 0 or n_cols == 0:
        if verbose:
            print("[detect] empty matrix")
        return {'found': False, 'dim': 0, 'basis': [], 'alive_idx_list': alive_idx_list}

    A = matrix(F, rows, sparse=False)
    K = A.right_kernel()
    dim = K.dimension()
    basis = []
    for v in K.basis():
        basis.append([int(x) % int(ell) for x in v])

    if verbose:
        print(f"[detect] A_hom: {n_rows}x{n_cols}, kernel dim = {dim}")

    return {'found': (dim == 1), 'dim': dim, 'basis': basis, 'alive_idx_list': alive_idx_list, 'A_hom': A, 'n_cols': n_cols}

def is_vector_in_rowspace(A, vec, verbose=False):
    """
    Check whether row-vector vec (length == A.ncols()) is in the row-space of A.
    Returns (True, coeffs) or (False, None). Coeffs express vec as linear combination
    of A's rows (over A.base_ring()).
    """
    if A.ncols() != len(vec):
        if verbose:
            print("[is_vector_in_rowspace] dimension mismatch")
        return False, None

    F = A.base_ring()
    # make column vector for vec
    vec_col = matrix(F, len(vec), 1, [F(int(x) % int(F.characteristic())) for x in vec])

    try:
        AT = A.transpose()
        alpha = AT.solve_right(vec_col)   # solves AT * alpha = vec_col
        coeffs = [int(x) % int(F.characteristic()) for x in alpha.list()]
        if verbose:
            print("[is_vector_in_rowspace] vector is in row-space; returning coefficients")
        return True, coeffs
    except Exception as e:
        if verbose:
            print("[is_vector_in_rowspace] vector NOT in row-space:", e)
        return False, None

def try_add_row_to_basis(basis_rows, row_vec, ell):
    """
    Rank-based independence test: append row_vec to basis_rows if independent.
    basis_rows: list of Sage vectors or lists (same length); row_vec: list-like.
    Returns True if appended, False otherwise.
    """
    F = GF(int(ell))

    def _to_list(v):
        if hasattr(v, 'list'):
            raw = list(v.list())
        else:
            raw = list(v)
        return [F(int(x) % int(ell)) for x in raw]

    v_list = _to_list(row_vec)

    if not basis_rows:
        basis_rows.append(vector(F, v_list))
        return True

    M_before = matrix(F, [ _to_list(b) for b in basis_rows ])
    rank_before = M_before.rank()
    M_after = matrix(F, [ _to_list(b) for b in basis_rows ] + [v_list])
    rank_after = M_after.rank()

    if rank_after > rank_before:
        basis_rows.append(vector(F, v_list))
        return True
    return False

def select_independent_rows(A_hom, ell, target_count=None):
    """
    Incremental independent-row selector (safe fallback).
    Returns list of selected row indices.
    """
    F = GF(int(ell))
    n_rows = A_hom.nrows()
    n_cols = A_hom.ncols()
    if target_count is None:
        target_count = min(n_rows, n_cols)

    chosen = []
    basis_rows = []

    def _try_append(row_list):
        nonlocal basis_rows
        if not basis_rows:
            basis_rows = [row_list[:]]
            return True
        M_before = matrix(F, basis_rows, sparse=False)
        rank_before = M_before.rank()
        M_after = matrix(F, basis_rows + [row_list], sparse=False)
        rank_after = M_after.rank()
        if rank_after > rank_before:
            basis_rows.append(row_list[:])
            return True
        return False

    for i in range(n_rows):
        if len(chosen) >= target_count:
            break
        row = A_hom.row(i)
        row_list = [F(int(x) % int(ell)) for x in row]
        if all(x == 0 for x in row_list):
            continue
        if _try_append(row_list):
            chosen.append(i)

    return chosen[:target_count]

def select_independent_rows_fast(A_hom, ell, target_count=None):
    """
    Heuristic fast selection (every 4th). Validates the candidate; falls back to select_independent_rows.
    """
    n_rows = A_hom.nrows()
    if target_count is None:
        target_count = min(n_rows, A_hom.ncols())

    candidate = list(range(1, min(1 + 4 * target_count, n_rows), 4))[:target_count]
    F = GF(int(ell))
    chosen_rows = []
    for i in candidate:
        r = A_hom.row(i)
        chosen_rows.append([F(int(x) % int(ell)) for x in r])

    if chosen_rows:
        M = matrix(F, chosen_rows, sparse=False)
        if M.rank() == min(target_count, A_hom.ncols()):
            return candidate[:target_count]

    # fallback
    return select_independent_rows(A_hom, ell, target_count=target_count)

# ===== Efficient constrained-kernel + bounded-prune replacements =====
# Requires: from sage.all import GF, matrix, vector
#           import random

def find_transverse_kernel(A_hom, g_vec_dense, ell, verbose=False):
    """
    Try a single linear solve that finds chi with:
        A_hom * chi = 0
        <g_vec_dense, chi> = 1   (mod ell)

    Returns:
        chi as a list of ints mod ell if found, else None.

    This performs ONE linear solve on the (m+1) x n system instead of many kernel recomputations.
    """
    F = GF(int(ell))
    m, n = A_hom.nrows(), A_hom.ncols()

    # Ensure g_vec_dense length matches columns
    if len(g_vec_dense) != n:
        raise ValueError("g_vec_dense length != ncols")

    # Build M whose rows = rows(A_hom) followed by g_vec
    # Build RHS b = [0,...,0, 1]
    rows = [list(r) for r in A_hom.rows()]
    rows.append([F(int(x) % int(ell)) for x in g_vec_dense])
    M = matrix(F, rows, sparse=False)

    b = vector(F, [F(0)] * m + [F(1)])

    try:
        # Solve M * chi = b (rectangular). solve_right works for consistent systems.
        chi_vec = M.solve_right(b)
    except Exception:
        if verbose:
            print("find_transverse_kernel: direct solve failed")
        return None

    # Verify exactness: A_hom * chi == 0 and <g,chi> == 1
    chi_list = [int(x) % int(ell) for x in chi_vec]
    # verify A_hom * chi == 0
    prod = A_hom * vector(F, [F(x) for x in chi_list])
    if any(int(x) % int(ell) != 0 for x in prod):
        if verbose:
            print("find_transverse_kernel: solve produced A*chi != 0")
        return None
    # verify dot with g
    dot = sum((int(g_vec_dense[j]) % int(ell)) * chi_list[j] for j in range(n)) % int(ell)
    if dot != 1 % int(ell):
        if verbose:
            print(f"find_transverse_kernel: dot != 1 (dot={dot})")
        return None

    return chi_list

def pick_sparse_row_to_remove(dense_rows, max_candidates=100):
    """
    Choose a single row index to remove based on sparsity: prefer rows with few nonzeros.
    `dense_rows` is list-of-lists (field elements or ints) representing A_hom rows.
    Returns index (int).
    """
    # compute nonzero counts (work with ints for speed)
    nz_counts = [(i, sum(1 for x in row if int(x) % 1 != 0 or x != 0)) for i, row in enumerate(dense_rows)]
    # sort by count ascending
    nz_counts.sort(key=lambda t: (t[1], t[0]))
    # choose from top-k sparsest randomly to avoid pathological picks
    k = min(max_candidates, len(nz_counts))
    candidates = [t[0] for t in nz_counts[:k]]
    return random.choice(candidates)

# === Integrate into precheck_cofactor_projection (replace the prune portion) ===

def precheck_cofactor_projection(atom_to_idx, homogeneous_rows, row_g, row_q,
                                 full_order, J, f_coeffs, p, verbose=True):
    """
    Sparse-safe precheck that:
      - builds A_hom as a sparse GF(ell) matrix from homogeneous_rows
      - checks rank structure
      - if full-rank, attempts a bounded prune via choose_prune_row_guided_by_g
      - returns the full contract expected by callers:
          filtered_rows (list of sparse dicts using original atom indices),
          filtered_row_g (projected G row, using original atom indices),
          filtered_row_q (projected Q row),
          alive_fb_indices, removed_row_indices, etc.
    """

    if verbose:
        print("\n" + "="*68)
        print("COFACTOR PROJECTION PRE-CHECK (Rank Defective Check)")
        print("="*68)

    ell, h = _safe_get_ell_and_h(full_order)
    if verbose:
        print(f"  |J| = {full_order}, ell = {ell}, h = {h}")

    # alive indices (no earlier projection performed here)
    alive_fb_indices = set(atom_to_idx.values())
    alive_idx_list = sorted(alive_fb_indices)
    col_map = {old_idx: c for c, old_idx in enumerate(alive_idx_list)}
    n_cols = len(alive_idx_list)

    if verbose:
        print(f"  Alive FB: {n_cols}  Dead FB: 0")

    # Build sparse projected rows (keep original old-index keys)
    sparse_rows = []
    for row in homogeneous_rows:
        row_proj = {}
        for old_idx, mult in row.items():
            if old_idx in alive_fb_indices:
                row_proj[old_idx] = int(mult) % int(ell)
        if row_proj:
            sparse_rows.append(row_proj)

    if not sparse_rows:
        return {
            'safe_to_project': False,
            'reason': 'ALL homogeneous relations vanished under h-projection',
            'ell': ell, 'h': h
        }

    # Build sparse matrix A_hom (rows indexed by sparse_rows, columns by alive_idx_list)
    F = GF(int(ell))
    n_rows = len(sparse_rows)
    A_hom = matrix(F, n_rows, n_cols, sparse=True)
    for i, row in enumerate(sparse_rows):
        for old_idx, mult in row.items():
            j = col_map[old_idx]
            A_hom[i, j] = F(int(mult) % int(ell))

    # compute rank once
    try:
        rank_hom = A_hom.rank()
    except Exception as e:
        # defensive fallback: if rank computation fails, abort gracefully
        return {
            'safe_to_project': False,
            'reason': f'rank computation failed: {e}',
            'ell': ell, 'h': h
        }

    if verbose:
        print(f"  [Rank Check] N_Cols (Alive Atoms): {n_cols}")
        print(f"  [Rank Check] Hom. Rank (before prune): {rank_hom}")

    if rank_hom < n_cols - 1:
        return {
            'safe_to_project': False,
            'reason': f'Homogeneous rank {rank_hom} < n_cols-1 {n_cols-1}; need more relations',
            'ell': ell, 'h': h,
            'rank_hom': rank_hom,
            'alive_fb_indices': alive_fb_indices,
            'filtered_rows': sparse_rows,
            'filtered_row_g': {old_idx: mult for old_idx, mult in (row_g or {}).items() if old_idx in alive_fb_indices},
            'filtered_row_q': {old_idx: mult for old_idx, mult in (row_q or {}).items() if old_idx in alive_fb_indices},
            'fb_projected': {},
        }

    # Build projected G/Q rows (sparse dicts keyed by original old_idx)
    row_g_proj = {old_idx: int(mult) % int(ell) for old_idx, mult in (row_g or {}).items() if old_idx in alive_fb_indices}
    row_q_proj = {old_idx: int(mult) % int(ell) for old_idx, mult in (row_q or {}).items() if old_idx in alive_fb_indices}

    removed_row_indices = []

    # If the homogeneous matrix is full rank, try bounded prune to create the 1-dim kernel
    if rank_hom == n_cols:
        if verbose:
            print("  [Auto-Prune] Homogeneous is full rank; attempting bounded prune")

        # Prepare a dense integer g vector aligned to A_hom columns for the prune function
        g_vec_dense = [0] * n_cols
        for old_idx, mult in row_g_proj.items():
            g_vec_dense[col_map[old_idx]] = int(mult) % int(ell)

        # First: check for singleton columns (exactly 1 nonzero row in that column).
        # Removing that row makes the column all-zero, guaranteeing rank drops by 1.
        col_nonzero_rows = {}
        for i, row in enumerate(sparse_rows):
            for old_idx in row:
                j = col_map[old_idx]
                col_nonzero_rows.setdefault(j, []).append(i)

        singleton_row_idx = None
        for j, rows_with_nonzero in col_nonzero_rows.items():
            if len(rows_with_nonzero) == 1:
                singleton_row_idx = rows_with_nonzero[0]
                if verbose:
                    print("  [Auto-Prune] Found singleton column %d; removing row %d" % (j, singleton_row_idx))
                break

        if singleton_row_idx is not None:
            removed_row_indices = [singleton_row_idx]
        else:
            try:
                removed_row_indices = choose_prune_row_guided_by_g(A_hom, g_vec_dense, ell, verbose=verbose)
            except Exception as e:
                return {
                    'safe_to_project': False,
                    'reason': 'prune attempt failed: ' + str(e),
                    'ell': ell, 'h': h,
                    'rank_hom': rank_hom,
                    'alive_fb_indices': alive_fb_indices,
                    'filtered_rows': sparse_rows,
                    'filtered_row_g': row_g_proj,
                    'filtered_row_q': row_q_proj,
                    'fb_projected': {},
                }

        # If some rows were removed, apply removals to sparse_rows and rebuild A_hom and rank
        if removed_row_indices:
            rem_set = set(removed_row_indices)
            # removed_row_indices are indices in the current sparse_rows list
            keep_rows = [i for i in range(n_rows) if i not in rem_set]
            sparse_rows = [sparse_rows[i] for i in keep_rows]
            # rebuild A_hom from remaining sparse_rows
            A_hom = matrix(F, len(sparse_rows), n_cols, sparse=True)
            for i, row in enumerate(sparse_rows):
                for old_idx, mult in row.items():
                    j = col_map[old_idx]
                    A_hom[i, j] = F(int(mult) % int(ell))
            try:
                rank_hom = A_hom.rank()
            except Exception as e:
                return {
                    'safe_to_project': False,
                    'reason': f'rank computation after prune failed: {e}',
                    'ell': ell, 'h': h,
                    'alive_fb_indices': alive_fb_indices,
                    'alive_idx_list': alive_idx_list,
                    'filtered_rows': sparse_rows,
                    'filtered_row_g': row_g_proj,
                    'filtered_row_q': row_q_proj,
                }

            if verbose:
                print(f"  [Auto-Prune] After pruning: Hom. Rank = {rank_hom}")

            # Ensure prune actually produced the expected rank defect
            if rank_hom != n_cols - 1:
                return {
                    'safe_to_project': False,
                    'reason': 'Prune did not produce desired rank defect',
                    'ell': ell, 'h': h,
                    'rank_hom': rank_hom,
                    'removed_row_indices': removed_row_indices,
                    'filtered_rows': sparse_rows,
                    'filtered_row_g': row_g_proj,
                    'filtered_row_q': row_q_proj,
                    'alive_fb_indices': alive_fb_indices,
                    'alive_idx_list': alive_idx_list,
                }

    # Final check: does adding G row increase rank to n_cols?
    try:
        g_row_F = [F(int(x) % int(ell)) for x in ([row_g_proj.get(old_idx, 0) for old_idx in alive_idx_list])]
        M_aug = A_hom.stack(matrix(F, [g_row_F]))
        rank_aug = M_aug.rank()
    except Exception as e:
        return {
            'safe_to_project': False,
            'reason': f'augmented rank computation failed: {e}',
            'ell': ell, 'h': h,
            'rank_hom': rank_hom
        }

    if verbose:
        print(f"  [Rank Check] Hom Rank: {rank_hom}, Rank with G-row: {rank_aug}")

    if rank_aug == n_cols:
        return {
            'safe_to_project': True,
            'alive_fb_indices': alive_fb_indices,
            'dead_fb_indices': set(),
            'filtered_rows': sparse_rows,           # list of dicts keyed by original atom indices
            'filtered_row_g': row_g_proj,
            'filtered_row_q': row_q_proj,
            'ell': ell, 'h': h,
            'rank_hom': rank_hom, 'rank_aug': rank_aug,
            'fb_projected': {},
            'reason': 'Rank structure verified (Defective-by-1 + G-Fix)',
            'alive_idx_list': alive_idx_list,
            'removed_row_indices': removed_row_indices,
        }

    # fallback: try G+Q augmentation as last resort
    if row_q_proj:
        try:
            q_row_F = [F(int(x) % int(ell)) for x in ([row_q_proj.get(old_idx, 0) for old_idx in alive_idx_list])]
            M_gq = A_hom.stack(matrix(F, [g_row_F, q_row_F]))
            rank_gq = M_gq.rank()
        except Exception as e:
            return {
                'safe_to_project': False,
                'reason': f'G+Q augmented rank computation failed: {e}',
                'ell': ell, 'h': h,
                'rank_hom': rank_hom
            }

        if verbose:
            print(f"  [Rank Check] Rank with G+Q rows: {rank_gq}")

        if rank_gq == n_cols:
            return {
                'safe_to_project': True,
                'alive_fb_indices': alive_fb_indices,
                'dead_fb_indices': set(),
                'filtered_rows': sparse_rows,
                'filtered_row_g': row_g_proj,
                'filtered_row_q': row_q_proj,
                'ell': ell, 'h': h,
                'rank_hom': rank_hom, 'rank_aug': rank_gq,
                'fb_projected': {},
                'reason': 'G alone failed, but G+Q fixes kernel',
                'alive_idx_list': alive_idx_list,
                'removed_row_indices': removed_row_indices,
            }

    # final failure
    return {
        'safe_to_project': False,
        'reason': 'G-row (and G+Q) did not fix kernel',
        'ell': ell, 'h': h,
        'rank_hom': rank_hom, 'rank_aug': rank_aug,
        'alive_fb_indices': alive_fb_indices,
        'filtered_rows': sparse_rows,
        'filtered_row_g': row_g_proj,
        'filtered_row_q': row_q_proj,
        'fb_projected': {},
        'removed_row_indices': removed_row_indices,
    }

def choose_prune_row_guided_by_g(A_hom, g_vec_dense, ell, verbose=True,
                                max_candidates=96, max_pair_tries=128):
    """
    Fiber-aware rank-based prune.

    Heuristics:
      - Prefer dense rows (fiber rows)
      - Prefer rows overlapping G support
      - Avoid trivial rows (<=2 nonzeros)
    """
    from sage.all import GF, matrix

    F = GF(int(ell))
    m, n = A_hom.nrows(), A_hom.ncols()

    # G support
    g_support = set(i for i, x in enumerate(g_vec_dense) if x % ell != 0)

    # Build scoring list
    scored = []
    for i in range(m):
        row = A_hom.row(i)

        try:
            weight = row.hamming_weight()
        except Exception:
            weight = sum(1 for x in row if x != 0)

        # skip trivial rows aggressively
        if weight <= 2:
            continue

        # overlap with G support
        overlap = 0
        for j, val in enumerate(row):
            if val != 0 and j in g_support:
                overlap += 1

        # scoring: prioritize overlap, then density
        score = (10 * overlap) + weight

        scored.append((score, i))

    # fallback: if everything was trivial (unlikely but safe)
    if not scored:
        if verbose:
            print("  [warn] all rows look trivial; falling back to raw sparsity")
        for i in range(m):
            row = A_hom.row(i)
            weight = sum(1 for x in row if x != 0)
            scored.append((weight, i))

    # sort descending (best first)
    scored.sort(reverse=True)

    candidates = [i for (_, i) in scored[:max_candidates]]

    if verbose:
        print(f"  considering {len(candidates)} high-quality candidates (fiber-biased)")

    # Prepare G row
    g_row_F = [F(int(x) % int(ell)) for x in g_vec_dense]
    g_row_mat = matrix(F, [g_row_F])

    # --- SINGLE ROW REMOVAL ---
    for attempt, r_idx in enumerate(candidates):
        keep_rows = [i for i in range(m) if i != r_idx]
        M_reduced = A_hom.matrix_from_rows(keep_rows)

        M_aug = M_reduced.stack(g_row_mat)
        try:
            rank_aug = M_aug.rank()
        except Exception:
            continue

        if rank_aug == n:
            if verbose:
                print(f"  success removing row {r_idx} (fiber-biased, attempt {attempt})")
            return [r_idx]

    # --- PAIR REMOVAL (fiber-only pool) ---
    if verbose:
        print("  trying fiber-biased pair removals...")

    tried = set()
    attempts = 0

    while attempts < max_pair_tries:
        i = random.choice(candidates)
        j = random.choice(candidates)
        if i == j:
            continue

        key = (min(i, j), max(i, j))
        if key in tried:
            continue

        tried.add(key)
        attempts += 1

        keep_rows = [k for k in range(m) if k not in key]
        M_reduced = A_hom.matrix_from_rows(keep_rows)
        M_aug = M_reduced.stack(g_row_mat)

        try:
            rank_aug = M_aug.rank()
        except Exception:
            continue

        if rank_aug == n:
            if verbose:
                print(f"  success removing pair {key} (fiber-biased)")
            return [key[0], key[1]]

    raise RuntimeError("fiber-aware prune failed (no transverse removal found)")
