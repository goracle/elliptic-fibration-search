# Add to index_calculus.py
import sys
# Standard library
import time
import random
from math import ceil, sqrt, gcd
from multiprocessing import Pool, cpu_count
from collections import Counter

# Sage imports (consolidated)
from sage.all import (
    Integer, Zmod, GF, ZZ,
    matrix, vector,
    PolynomialRing,
    factor, crt, prime_factors,
    set_random_seed
)
from sage.schemes.hyperelliptic_curves.constructor import HyperellipticCurve
from sage.matrix.berlekamp_massey import berlekamp_massey


def apply_cofactor_filter(precheck_result, atom_to_idx, homogeneous_rows, 
                          row_g, row_q, verbose=True):
    """
    Apply the filtering from precheck_cofactor_projection results.
    
    Returns filtered data structures with dead FB elements removed.
    
    Args:
        precheck_result: output from precheck_cofactor_projection
        atom_to_idx: original factor base
        homogeneous_rows: original relations
        row_g, row_q: original G and Q rows
        verbose: print stats
    
    Returns:
        (filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q)
    """
    if not precheck_result['safe_to_project']:
        raise RuntimeError(
            f"Cannot apply filter: projection is unsafe. Reason: {precheck_result['reason']}"
        )
    
    alive_indices = precheck_result['alive_fb_indices']
    
    # Rebuild atom_to_idx with only alive elements
    # Remap indices to be contiguous starting from 0
    idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items() if idx in alive_indices}
    
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(alive_indices))}
    
    filtered_atom_to_idx = {
        atom: old_to_new[old_idx] 
        for old_idx, atom in idx_to_atom.items()
    }
    
    # Remap relation rows
    filtered_rows = []
    for row in precheck_result['filtered_rows']:
        new_row = {old_to_new[idx]: mult for idx, mult in row.items() if idx in alive_indices}
        if new_row:
            filtered_rows.append(new_row)
    
    # Remap G and Q rows
    filtered_row_g = {old_to_new[idx]: mult for idx, mult in precheck_result['filtered_row_g'].items()}
    filtered_row_q = {old_to_new[idx]: mult for idx, mult in precheck_result['filtered_row_q'].items()}
    
    if verbose:
        print(f"\n[Filter Applied]")
        print(f"  Original FB size: {len(atom_to_idx)}")
        print(f"  Filtered FB size: {len(filtered_atom_to_idx)}")
        print(f"  Original relations: {len(homogeneous_rows)}")
        print(f"  Filtered relations: {len(filtered_rows)}")
        print(f"  Removed {len(atom_to_idx) - len(filtered_atom_to_idx)} dead FB elements")
        print(f"  Removed {len(homogeneous_rows) - len(filtered_rows)} dead relations")
    
    return filtered_atom_to_idx, filtered_rows, filtered_row_g, filtered_row_q


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

def detect_nontrivial_character_from_projection(filtered_rows, alive_fb_indices, ell, verbose=True):
    Zell = Zmod(int(ell))
    alive_idx_list = sorted(alive_fb_indices)
    col_map = {old_idx: c for c, old_idx in enumerate(alive_idx_list)}
    n_cols = len(alive_idx_list)
    n_rows = len(filtered_rows)

    # build sparse entries
    entries = {}
    for i, row in enumerate(filtered_rows):
        for old_idx, mult in row.items():
            if old_idx not in col_map:
                continue
            j = col_map[old_idx]
            entries[(i, j)] = Zell(int(mult) % int(ell))

    A_hom = matrix(Zell, n_rows, n_cols, entries, sparse=True)

    K = A_hom.right_kernel()   # space of characters that vanish on rows
    dim = K.dimension()
    if dim > 1:
        raise RuntimeError(f"kernel has dim > 1, factor base structure incomplete. dim={dim}")
    basis = [[int(x) % int(ell) for x in v] for v in K.basis()]

    if verbose:
        print(f"[detect] A_hom: {n_rows}x{n_cols}, kernel dim = {dim}")

    return {'found': (dim == 1), 'dim': dim, 'basis': basis, 'alive_idx_list': alive_idx_list, 'A_hom': A_hom, 'n_cols': n_cols}

def is_vector_in_rowspace(A, vec, verbose=False):
    """
    Test whether row-vector 'vec' (length == n_cols) lies in the row-space of A.
    Return (True, coeffs) if yes where coeffs are coefficients expressing vec as linear comb
    of A's rows; otherwise (False, None).
    """
    Zell = A.base_ring()
    # Solve linear system R^T * alpha = vec^T where R are rows of A
    # Build matrix whose rows are A.rows(); then use solve_right for combination
    # We want alpha such that alpha^T * A = vec  -> (A^T) * alpha = vec^T  => solve
    try:
        AT = A.transpose()
        vec_col = matrix(Zell, len(vec), 1, [Zell(int(x) % int(Zell.characteristic())) for x in vec])
        # Solve AT * alpha = vec_col for alpha
        alpha = AT.solve_right(vec_col)   # returns column vector of coefficients if solvable
        coeffs = [int(x) % int(Zell.characteristic()) for x in alpha.list()]
        if verbose:
            print("[is_vector_in_rowspace] vector is in row-space; returning coefficients")
        return True, coeffs
    except Exception:
        if verbose:
            print("[is_vector_in_rowspace] vector NOT in row-space")
        return False, None

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


def precheck_cofactor_projection(atom_to_idx, homogeneous_rows, row_g, row_q,
                                  full_order, J, f_coeffs, p, verbose=True):
    """
    Checks if the system is solvable after projecting to J[ell] via cofactor h.
    
    STRICT RANK REQUIREMENTS:
    1. Homogeneous relations must have rank = n_cols - 1 (Defective by 1).
    2. Augmented system (with G-row) must have rank = n_cols (Full Rank).
    
    AUTO-PRUNING:
    If the homogeneous system is full rank (rank == n_cols), this function will 
    automatically select a subset of linearly independent rows of size n_cols-1 
    to enforce the 1-dimensional kernel requirement.
    """
    if verbose:
        print("\n" + "="*68)
        print("COFACTOR PROJECTION PRE-CHECK (Rank Defective Check)")
        print("="*68)

    ell, h = _safe_get_ell_and_h(full_order)

    if verbose:
        print(f"  |J| = {full_order}, ℓ = {ell}, h = {h}")

    idx_to_atom = {idx: atom for atom, idx in atom_to_idx.items()}

    Zell = Zmod(ell)

    alive_fb_indices = set()
    dead_fb_indices = set()
    fb_projected = {}

    # Setup polynomial ring for reconstructing Mumford polynomials
    K = J.base_ring()
    try:
        R = J.curve().hyperelliptic_polynomials()[0].parent()
        x = R.gen()
    except Exception:
        R = PolynomialRing(K, 'x')
        x = R.gen()

    # --- STEP 1: Project Factor Base Atoms by h ---
    for idx, atom in idx_to_atom.items():
        F_i = None
        
        if atom[0] == 'd1':
            _, x_val, y_val = atom
            try:
                u = x - K(x_val)
                v = R(K(y_val)) 
                F_i = J([u, v])
            except Exception as e:
                if verbose and len(dead_fb_indices) < 5:
                    print(f"  [Warning] Failed to construct d1 atom {atom}: {e}")
                F_i = None
                raise
        else:
            try:
                _, u_coeffs, v_coeffs = atom
                u = R(list(u_coeffs))
                v = R(list(v_coeffs))
                F_i = J([u, v])
            except Exception as e:
                if verbose and len(dead_fb_indices) < 5:
                    print(f"  [Warning] Failed to construct d2 atom {atom}: {e}")
                F_i = None
                raise

        if F_i is None:
            dead_fb_indices.add(idx)
            fb_projected[idx] = None
            continue

        try:
            F_i_proj = Integer(h) * F_i
            if F_i_proj.is_zero():
                dead_fb_indices.add(idx)
                fb_projected[idx] = None
            else:
                alive_fb_indices.add(idx)
                fb_projected[idx] = F_i_proj
        except Exception:
            dead_fb_indices.add(idx)
            fb_projected[idx] = None
            raise

    if verbose:
        print(f"  Alive FB: {len(alive_fb_indices)}  Dead FB: {len(dead_fb_indices)}")

    if not alive_fb_indices:
        return {
            'safe_to_project': False,
            'reason': 'ALL factor base elements died under h-projection'
        }

    # --- STEP 2: Project Relations ---
    alive_rows = []
    for row in homogeneous_rows:
        row_proj = {idx: mult for idx, mult in row.items() if idx in alive_fb_indices}
        if not row_proj:
            continue
        alive_rows.append(row_proj)

    if not alive_rows:
        return {
            'safe_to_project': False,
            'reason': 'ALL homogeneous relations vanished under h-projection'
        }

    row_g_proj = {idx: mult for idx, mult in row_g.items() if idx in alive_fb_indices}
    row_q_proj = {idx: mult for idx, mult in (row_q or {}).items() if idx in alive_fb_indices}

    if not row_g_proj:
        return {
            'safe_to_project': False,
            'reason': 'G uses only dead FB elements after projection'
        }

    # Build col_map
    alive_idx_list = sorted(alive_fb_indices)
    col_map = {old_idx: c for c, old_idx in enumerate(alive_idx_list)}
    n_cols = len(alive_idx_list)

    # Build g_row vector EARLY (before rank checks)
    g_row_vec = [0] * n_cols
    for old_idx, mult in row_g_proj.items():
        g_row_vec[col_map[old_idx]] = int(mult) % int(ell)


    # Build A_hom for initial rank check
    entries_hom = {}
    for i, row in enumerate(alive_rows):
        for old_idx, mult in row.items():
            entries_hom[(i, col_map[old_idx])] = Zmod(int(ell))(int(mult) % int(ell))
    
    A_hom = matrix(Zmod(int(ell)), len(alive_rows), n_cols, entries_hom, sparse=True)
    rank_hom = A_hom.rank()

    if verbose:
        print(f"  [Rank Check] N_Cols (Alive Atoms): {n_cols}")
        print(f"  [Rank Check] Hom. Rank (before prune): {rank_hom}")

    # If full rank, auto-prune using echelon form to extract independent rows
    if rank_hom == n_cols:
        if verbose:
            print("  [Auto-Prune] Homogeneous is full rank; using G-guided row selection")

        # Use G-guided pruning instead of blind selection
        kill_idx = choose_prune_row_guided_by_g(A_hom, g_row_vec, ell, verbose=verbose)

        # Remove the chosen row
        alive_rows.pop(kill_idx)

        # Rebuild matrix
        entries_pruned = {}
        for i, row in enumerate(alive_rows):
            for old_idx, mult in row.items():
                entries_pruned[(i, col_map[old_idx])] = Zmod(ell)(int(mult) % ell)

        A_hom = matrix(Zmod(ell), len(alive_rows), n_cols, entries_pruned, sparse=True)
        rank_hom = A_hom.rank()

        if rank_hom != n_cols - 1:
            raise RuntimeError(f"G-guided prune failed to induce rank defect (got rank {rank_hom})")

        K = A_hom.right_kernel()
        assert K.dimension() == 1, "Kernel should be 1-dimensional after G-guided prune"

        if verbose:
            # Verify G is transverse
            chi = K.basis()[0]
            g_vec = vector(Zmod(ell), g_row_vec)
            dot = chi.dot_product(g_vec)
            print(f"  [Verification] <kernel, G> = {dot} mod {ell} (should be nonzero)")
            assert int(dot) % int(ell) != 0, "G should be transverse to kernel"

            if rank_hom != n_cols - 1:
                raise RuntimeError("auto-prune failed to induce rank defect")

    # Build g_row vector
    g_row_vec = [0] * n_cols
    for old_idx, mult in row_g_proj.items():
        g_row_vec[col_map[old_idx]] = int(mult) % int(ell)

    # Build augmented matrix with G-row
    entries_aug = dict(A_hom.dict())
    for j, val in enumerate(g_row_vec):
        if int(val) != 0:
            entries_aug[(A_hom.nrows(), j)] = Zmod(int(ell))(int(val))
    A_aug = matrix(Zmod(int(ell)), A_hom.nrows() + 1, n_cols, entries_aug, sparse=True)
    rank_aug = A_aug.rank()

    if verbose:
        print(f"  [Rank Check] Hom Rank: {rank_hom}, Rank with G-row: {rank_aug}")

    # Build Q row vector if needed
    q_row_vec = None
    if row_q is not None:
        q_row_vec = [0] * n_cols
        for old_idx, mult in row_q_proj.items():
            q_row_vec[col_map[old_idx]] = int(mult) % int(ell)

    if rank_aug != n_cols:
        # G failed. Try G+Q
        if q_row_vec is not None:
            entries_gq = dict(entries_aug)
            for j, val in enumerate(q_row_vec):
                if int(val) != 0:
                    entries_gq[(A_hom.nrows() + 1, j)] = Zmod(int(ell))(int(val))
            A_gq = matrix(Zmod(int(ell)), A_hom.nrows() + 2, n_cols, entries_gq, sparse=True)
            rank_gq = A_gq.rank()
            if verbose:
                print(f"  [Rank Check] Rank with G+Q rows: {rank_gq}")
            if rank_gq == n_cols:
                return {
                    'safe_to_project': True,
                    'alive_fb_indices': alive_fb_indices,
                    'dead_fb_indices': dead_fb_indices,
                    'filtered_rows': alive_rows,
                    'filtered_row_g': row_g_proj,
                    'filtered_row_q': row_q_proj,
                    'ell': ell, 'h': h,
                    'rank_hom': rank_hom, 'rank_aug': rank_gq,
                    'reason': 'G alone failed, but G+Q fixes kernel (use both to solve).',
                    'fb_projected': fb_projected,
                }

        # Neither G nor G+Q fixed kernel
        kernel_info = detect_nontrivial_character_from_projection(alive_rows, alive_fb_indices, ell, verbose=verbose)
        return {
            'safe_to_project': False,
            'reason': 'G-row (and G+Q) did not fix kernel; nontrivial character(s) exist. Projection unsafe.',
            'ell': ell, 'h': h,
            'rank_hom': rank_hom, 'rank_aug': rank_aug,
            'kernel_dim': kernel_info['dim'],
            'kernel_basis': kernel_info['basis'],
            'alive_fb_indices': alive_fb_indices,
            'filtered_rows': alive_rows,
            'filtered_row_g': row_g_proj,
            'filtered_row_q': row_q_proj,
            'fb_projected': fb_projected,
        }

    # Augmented system fixed kernel
    return {
        'safe_to_project': True,
        'alive_fb_indices': alive_fb_indices,
        'dead_fb_indices': dead_fb_indices,
        'filtered_rows': alive_rows,
        'filtered_row_g': row_g_proj,
        'filtered_row_q': row_q_proj,
        'ell': ell, 'h': h,
        'rank_hom': rank_hom, 'rank_aug': rank_aug,
        'fb_projected': fb_projected,
        'reason': 'Rank structure verified (Defective-by-1 + G-Fix)'
    }


def try_add_row_to_basis(basis_rows, row_vec, ell):
    """
    basis_rows: list of dense row vectors (already independent)
    row_vec: dense row vector to test
    Returns True if row_vec is independent and added, False otherwise.
    """
    Zell = Zmod(ell)
    v = vector(Zell, row_vec)

    for b in basis_rows:
        # eliminate leading pivot
        i = next((j for j,x in enumerate(b) if x != 0), None)
        if i is None:
            continue
        if v[i] != 0:
            v -= (v[i] / b[i]) * b

    if v.is_zero():
        return False

    basis_rows.append(v)
    return True


def choose_prune_row_guided_by_g(A_hom, g_row_vec, ell, verbose=True):
    """
    Choose which row to prune from a full-rank homogeneous matrix such that:
    1. After removal, rank becomes n_cols - 1 (defective by 1)
    2. The resulting kernel is transverse to G (i.e., <kernel_vec, G> != 0)
    
    This implements the "G-guided pruning" strategy instead of blindly
    dropping the first echelon row.
    
    Args:
        A_hom: full-rank homogeneous matrix (rank == n_cols)
        g_row_vec: G-row as list/vector
        ell: the prime modulus
        verbose: print diagnostics
    
    Returns:
        row_index: which row to remove (0-indexed)
    """
    Zell = Zmod(int(ell))
    n_rows, n_cols = A_hom.nrows(), A_hom.ncols()
    
    assert A_hom.rank() == n_cols, "A_hom must be full rank"
    
    # Convert g_row_vec to Zell vector for dot product
    g_vec = vector(Zell, [Zell(int(x) % int(ell)) for x in g_row_vec])
    
    # Get echelon form to identify pivot rows (these are the "important" rows)
    E = A_hom.echelon_form()
    pivot_rows_in_echelon = [i for i, _ in E.nonzero_positions()]
    
    if verbose:
        print(f"[G-Guided Prune] Testing {len(pivot_rows_in_echelon)} pivot rows...")
    
    # Test each pivot row for removal
    for candidate_idx in pivot_rows_in_echelon:
        # Build matrix without this row
        rows_without = [A_hom.row(i) for i in range(n_rows) if i != candidate_idx]
        A_test = matrix(Zell, rows_without)
        
        # Check rank drops to n_cols - 1
        rank_test = A_test.rank()
        if rank_test != n_cols - 1:
            if verbose and candidate_idx < 3:
                print(f"  Row {candidate_idx}: rank={rank_test} (need {n_cols-1}) - skip")
            continue
        
        # Compute kernel (should be 1-dimensional)
        K = A_test.right_kernel()
        if K.dimension() != 1:
            if verbose:
                print(f"  Row {candidate_idx}: kernel dim={K.dimension()} (need 1) - skip")
            continue
        
        # Get the kernel vector
        chi = K.basis()[0]
        
        # Check transversality: <chi, G> != 0
        dot_prod = chi.dot_product(g_vec)
        
        if int(dot_prod) % int(ell) != 0:
            if verbose:
                print(f"  Row {candidate_idx}: GOOD! <chi, G> = {dot_prod} (mod {ell})")
                print(f"    Removing this row gives rank defect with G transverse to kernel")
            return candidate_idx
        else:
            if verbose and candidate_idx < 3:
                print(f"  Row {candidate_idx}: <chi, G> = 0 - G not transverse - skip")
    
    # Fallback: if no good row found, return first pivot
    # (this shouldn't happen if the system is well-formed)
    if verbose:
        print(f"  [Warning] No provably good row found, using first pivot as fallback")
    return pivot_rows_in_echelon[0]
