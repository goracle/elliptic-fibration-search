import random
from typing import Tuple, Dict, List, Optional, Any, Callable

# recursive_smoothing.py

def pairify(a, b):
    if a == b:
        return (a, a)
    return tuple(sorted((a, b)))

class RecursiveSmoother:
    def __init__(self, roots, divisors=None, relations=None, modulus=None):
        self.roots = list(roots)
        self.modulus = modulus

        if divisors is None:
            self.divisors = []
            n = len(self.roots)
            for i in range(n):
                for j in range(i, n):
                    self.divisors.append(pairify(self.roots[i], self.roots[j]))
        else:
            self.divisors = [pairify(*d) for d in divisors]

        self.div_index = {d: i for i, d in enumerate(self.divisors)}
        self.relations = relations[:] if relations else []

    def _ensure_divisor(self, d):
        if d not in self.div_index:
            self.div_index[d] = len(self.divisors)
            self.divisors.append(d)
        return self.div_index[d]

    def random_move(self):
        i, j = random.sample(range(len(self.divisors)), 2)

        A, P1 = self.divisors[i]
        B, P2 = self.divisors[j]

        new1 = pairify(P1, P2)
        new2 = pairify(A, B)

        idx1 = self._ensure_divisor(new1)
        idx2 = self._ensure_divisor(new2)

        rel = {}

        rel[i] = rel.get(i, 0) + 1
        rel[j] = rel.get(j, 0) + 1
        rel[idx1] = rel.get(idx1, 0) - 1
        rel[idx2] = rel.get(idx2, 0) - 1

        if self.modulus:
            rel = {k: v % self.modulus for k, v in rel.items() if v % self.modulus != 0}

        return rel

    def collect_relations(self, num_moves):
        for _ in range(num_moves):
            r = self.random_move()
            self.relations.append(r)

    def to_dense_matrix(self):
        m = len(self.relations)
        n = len(self.divisors)
        M = [[0]*n for _ in range(m)]
        for i, rel in enumerate(self.relations):
            for j, c in rel.items():
                M[i][j] = c
        return M

    def rank_mod_p(self, p):
        M = self.to_dense_matrix()
        if not M:
            return 0

        M = [row[:] for row in M]
        rows, cols = len(M), len(M[0])
        rank = 0

        for col in range(cols):
            pivot = None
            for r in range(rank, rows):
                if M[r][col] % p != 0:
                    pivot = r
                    break
            if pivot is None:
                continue

            M[rank], M[pivot] = M[pivot], M[rank]

            inv = pow(M[rank][col], p-2, p)
            M[rank] = [(x * inv) % p for x in M[rank]]

            for r in range(rows):
                if r != rank and M[r][col] != 0:
                    factor = M[r][col]
                    M[r] = [(M[r][c] - factor * M[rank][c]) % p for c in range(cols)]

            rank += 1

        return rank

def convert_divisor_relations_to_atom_relations(rs):
    """
    Convert RecursiveSmoother output into atom-level relations
    compatible with perform_dlp_attack.
    """
    atom_to_idx = {}
    next_idx = 0

    def get_atom_idx(x):
        nonlocal next_idx
        if x not in atom_to_idx:
            atom_to_idx[x] = next_idx
            next_idx += 1
        return atom_to_idx[x]

    homogeneous_rows = []

    for rel in rs.relations:
        atom_row = {}

        for div_idx, coeff in rel.items():
            a, b = rs.divisors[div_idx]

            ia = get_atom_idx(a)
            ib = get_atom_idx(b)

            atom_row[ia] = atom_row.get(ia, 0) + coeff
            atom_row[ib] = atom_row.get(ib, 0) + coeff

        # prune zeros
        atom_row = {k: v for k, v in atom_row.items() if v != 0}

        if atom_row:
            homogeneous_rows.append(atom_row)

    fb_roots = list(atom_to_idx.keys())

    return homogeneous_rows, fb_roots, atom_to_idx

# helper_smoothing.py (put in same module or importable)

# Placeholder: you already have a function to extract ('d1', x, y) atoms from a Jacobian element.
# Implement or import your actual extractor. This must return a list of atom tuples used by your FB.
# Example signature:
# def jacobian_to_atoms(J_elem, p) -> List[('d1', x_int, y_can), ...]
# If your jacobian_to_dict() already produces such data, adapt as needed.
# We'll assume such a function exists in your codebase.

def _augment_atom_map_with_temp_atoms(atom_to_idx: Dict, temp_atoms: List):
    """Return (new_atom_to_idx, temp_idx_map) where new atom_to_idx includes temp_atoms and returns their indices.
       temp_idx_map maps temp_atom -> index_in_new_map.
    """
    new_map = dict(atom_to_idx)
    temp_map = {}
    next_idx = max(new_map.values()) + 1 if new_map else 0
    for a in temp_atoms:
        if a in new_map:
            temp_map[a] = new_map[a]
        else:
            new_map[a] = next_idx
            temp_map[a] = next_idx
            next_idx += 1
    return new_map, temp_map

def _matrix_rows_to_dense(hrows: List[Dict[int,int]], ncols: int, mod: Optional[int]=None):
    M = []
    for r in hrows:
        row = [0]*ncols
        for c,v in r.items():
            row[c] = v % mod if mod else v
        M.append(row)
    return M

def _nullspace_mod_p(matrix_rows: List[Dict[int,int]], ncols: int, p: int):
    """
    Compute nullspace basis mod p for row-sparse rows.
    Returns list of basis vectors (each length ncols).
    Simple gaussian elimination on dense array; good enough for FB sizes < few thousands.
    """
    M = _matrix_rows_to_dense(matrix_rows, ncols, mod=p)
    m = len(M)
    if m == 0:
        return []
    # convert to augmented copy for elimination
    A = [row[:] for row in M]
    row = 0
    pivots = []
    for col in range(ncols):
        if row >= m:
            break
        sel = None
        for r in range(row, m):
            if A[r][col] % p != 0:
                sel = r; break
        if sel is None:
            continue
        A[row], A[sel] = A[sel], A[row]
        inv = pow(A[row][col], p-2, p)
        A[row] = [(val * inv) % p for val in A[row]]
        for r in range(m):
            if r != row and A[r][col] != 0:
                factor = A[r][col]
                A[r] = [(A[r][c] - factor * A[row][c]) % p for c in range(ncols)]
        pivots.append(col)
        row += 1
    pivot_set = set(pivots)
    free_vars = [j for j in range(ncols) if j not in pivot_set]
    basis = []
    for fv in free_vars:
        sol = [0]*ncols
        sol[fv] = 1
        for r_idx, pc in enumerate(pivots):
            # row r_idx corresponds to pivot column pc; because A is reduced, pivot entry is 1 and other pivot cols are 0
            sol[pc] = (-A[r_idx][fv]) % p
        basis.append(sol)
    return basis

def integrate_smoother_result_into_system(
    smooth_res: Dict[str, Any],
    atom_to_idx: Dict[Any, int],
    homogeneous_rows: List[Dict[int, int]],
    homogeneous_rhs: Optional[List[int]],
    fb_roots: Optional[List[int]],
    ell: int,
    label: str,
    verbose: bool = False
) -> Dict[int, int]:
    """
    Integrate the recursive-smoother result (smooth_res) into the global factor base and
    homogeneous_rows list. Returns the remapped relation row (final indices -> coeff mod ell).
    Mutates atom_to_idx, homogeneous_rows, homogeneous_rhs, fb_roots in place.
    """
    if smooth_res is None:
        raise RuntimeError(f"Recursive smoother produced no result for {label}")

    new_atom_map = smooth_res['new_atom_map']   # atom -> temporary index
    atomrow_newidx = smooth_res['atom_row']     # temporary_index -> coeff

    # invert map
    newidx_to_atom = {idx: atom for atom, idx in new_atom_map.items()}

    # find next index to assign for new atoms
    if atom_to_idx:
        next_idx = max(atom_to_idx.values()) + 1
    else:
        next_idx = 0

    newidx_to_final = {}

    for newidx in sorted(newidx_to_atom.keys()):
        atom = newidx_to_atom[newidx]
        if atom in atom_to_idx:
            newidx_to_final[newidx] = atom_to_idx[atom]
        else:
            atom_to_idx[atom] = next_idx
            newidx_to_final[newidx] = next_idx
            # keep fb_roots diagnostic list up-to-date if it's present and atom is ('d1', x, y)
            if fb_roots is not None and isinstance(atom, tuple) and atom and atom[0] == 'd1':
                xval = atom[1]
                if xval not in fb_roots:
                    fb_roots.append(xval)
            next_idx += 1

    # remap relation to final indices and reduce mod ell
    remapped: Dict[int, int] = {}
    for newidx, coeff in atomrow_newidx.items():
        fin = newidx_to_final[newidx]
        c = int(coeff) % int(ell)
        if c != 0:
            remapped[fin] = (remapped.get(fin, 0) + c) % int(ell)

    # append to homogeneous_rows and homogeneous_rhs
    homogeneous_rows.append({k: v for k, v in remapped.items() if v != 0})
    if homogeneous_rhs is None:
        homogeneous_rhs = [0] * (len(homogeneous_rows) - 1)
    homogeneous_rhs.append(0)

    if verbose:
        print(f"  [Recursive] injected relation for {label} ({len(remapped)} atoms)")

    return remapped

def smooth_element_via_recursive_debug(
    element,
    atom_to_idx,
    f_p, p,
    ell,
    RecursiveSmoother,
    jacobian_to_atoms,
    moves_multiplier=40,
    max_moves=20000,
    bias_target=True,
    verbose=True
):
    """
    Debug wrapper around smooth_element_via_recursive.
    Prints detailed diagnostics at every stage.
    """

    if verbose:
        print("\n" + "-"*60)
        print(f"[SMOOTHER] Starting smoothing")
        print(f"  element: {element}")
        print(f"  FB size: {len(atom_to_idx)}")
        print(f"  ell: {ell}")
        print("-"*60)

    # -----------------------
    # Step 1: try direct decomposition
    # -----------------------
    try:
        atoms = jacobian_to_atoms(element, atom_to_idx)
        if verbose:
            print(f"[SMOOTHER] Direct atoms extracted: {len(atoms)}")
            print(f"  sample: {atoms[:5]}")
    except Exception as e:
        print(f"[SMOOTHER] ERROR in jacobian_to_atoms: {e}")
        return None

    # Check how many are already in FB
    in_fb = sum(1 for a in atoms if a in atom_to_idx)
    if verbose:
        print(f"[SMOOTHER] Atoms in FB: {in_fb}/{len(atoms)}")

    if in_fb == len(atoms) and atoms:
        if verbose:
            print("[SMOOTHER] Element already smooth. No recursion needed.")
        return {
            'atom_row': {atom_to_idx[a]: 1 for a in atoms},
            'new_atom_map': {},
            'status': 'already_smooth'
        }

    # -----------------------
    # Step 2: recursive smoothing
    # -----------------------
    if verbose:
        print("[SMOOTHER] Launching RecursiveSmoother...")
        print(f"  max_moves={max_moves}, moves_multiplier={moves_multiplier}")

    smoother = RecursiveSmoother(
        atom_to_idx=atom_to_idx,
        f_p=f_p,
        p=p,
        ell=ell,
        jacobian_to_atoms=jacobian_to_atoms,
        bias_target=bias_target
    )

    try:
        result = smoother.smooth(
            element,
            max_moves=max_moves,
            moves_multiplier=moves_multiplier
        )
    except Exception as e:
        print(f"[SMOOTHER] ERROR during recursive smoothing: {e}")
        return None

    if result is None:
        print("[SMOOTHER] FAILED: smoother returned None")
        return None

    # -----------------------
    # Step 3: inspect result
    # -----------------------
    atom_row = result.get('atom_row', {})
    new_map = result.get('new_atom_map', {})

    if verbose:
        print("[SMOOTHER] SUCCESS")
        print(f"  atom_row size: {len(atom_row)}")
        print(f"  new atoms introduced: {len(new_map)}")

        if len(atom_row) == 0:
            print("[SMOOTHER] WARNING: empty atom_row")

        # check FB overlap
        overlap = sum(1 for idx in atom_row if idx in atom_to_idx.values())
        print(f"  overlap with FB indices: {overlap}")

    # sanity check
    if not atom_row:
        print("[SMOOTHER] ERROR: atom_row is empty")
        return None

    return result

# Replace the previous smooth_element_via_recursive and debug wrapper with these:

def _call_jacobian_to_atoms(
    jacobian_to_atoms: Callable,
    element: Any,
    p: int,
    f_p=None,
    atom_to_idx: Optional[Dict] = None,
    verbose: bool = False
) -> Optional[List[Tuple]]:
    """
    Try multiple reasonable call signatures for jacobian_to_atoms and return
    the atom list. This avoids silent TypeError failures when the adapter has
    a different signature than expected.
    """
    tries = [
        ("(element, p)", lambda: jacobian_to_atoms(element, p)),
        ("(element,)",    lambda: jacobian_to_atoms(element)),
        ("(element, atom_to_idx)", lambda: jacobian_to_atoms(element, atom_to_idx)),
        ("(element, p, f_p)", lambda: jacobian_to_atoms(element, p, f_p)),
    ]
    last_exc = None
    for desc, fn in tries:
        try:
            atoms = fn()
            if verbose:
                print(f"[jacobian_to_atoms] succeeded with signature {desc}; returned {len(atoms) if atoms is not None else 0} atoms")
            return list(atoms) if atoms is not None else []
        except TypeError as e:
            last_exc = e
            if verbose:
                print(f"[jacobian_to_atoms] signature {desc} raised TypeError: {e}")
            continue
        except Exception as e:
            # other exceptions are real errors we should surface
            if verbose:
                print(f"[jacobian_to_atoms] signature {desc} raised exception: {e}")
            raise
    # if we get here, none matched
    if verbose:
        print("[jacobian_to_atoms] failed: no compatible signature found")
    raise TypeError(f"jacobian_to_atoms callable did not accept any expected signatures; last error: {last_exc}")

def smooth_element_via_recursive(
    element: Any,                 # Jacobian element G or Q
    atom_to_idx: Dict[Any, int],  # existing factor base map {atom -> col_idx}
    f_p,                          # curve polynomial
    p: int,                       # prime
    ell: int,                     # modulus (ell)
    RecursiveSmoother: Any,       # class
    jacobian_to_atoms: Callable,  # function to extract atoms from element (flexible signature)
    moves_multiplier: int = 30,
    max_moves: int = 20000,
    bias_target: bool = True,
    batch: int = 200,
    verbose: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Try to produce an atom-level relation row that involves `element` using the
    RecursiveSmoother. Verbose mode prints diagnostics.

    Returns a dict (see top comment) or None on failure/timeout.
    """
    # Step 0: extract target atoms (flexible call)
    try:
        target_atoms = _call_jacobian_to_atoms(jacobian_to_atoms, element, p, f_p=f_p, atom_to_idx=atom_to_idx, verbose=verbose)
    except Exception as e:
        if verbose:
            print("[smooth_recursive] ERROR extracting atoms from Jacobian element:", e)
        return None

    if not target_atoms:
        if verbose:
            print("[smooth_recursive] No atoms extracted for element; aborting")
        return None

    if verbose:
        print(f"[smooth_recursive] target_atoms ({len(target_atoms)}): {target_atoms[:6]}{'...' if len(target_atoms)>6 else ''}")

    # Quick check: are all target atoms already in FB?
    in_fb = sum(1 for a in target_atoms if a in atom_to_idx)
    if verbose:
        print(f"[smooth_recursive] atoms in FB: {in_fb}/{len(target_atoms)}")

    if in_fb == len(target_atoms) and len(target_atoms) > 0:
        # element is already smooth - return trivial row (col -> coeff 1 for each atom)
        atom_row = {}
        for a in target_atoms:
            atom_row[atom_to_idx[a]] = atom_row.get(atom_to_idx[a], 0) + 1
        if verbose:
            print("[smooth_recursive] element already smooth; returning direct atom_row")
        return {
            'atom_row': atom_row,
            'new_atom_map': {},
            'temp_map': {},
            'status': 'already_smooth'
        }

    # Step 1: extend the atom map with temporary atoms for the target
    new_map, temp_map = _augment_atom_map_with_temp_atoms(atom_to_idx, target_atoms)
    roots_list = list(new_map.keys())   # these are the 'roots' for RecursiveSmoother

    if verbose:
        print(f"[smooth_recursive] built new_map with {len(new_map)} roots (FB {len(atom_to_idx)} + temps {len(temp_map)})")
        print(f"[smooth_recursive] will create RecursiveSmoother over {len(roots_list)} roots")

    # instantiate RS
    rs = RecursiveSmoother(roots=roots_list, modulus=ell, rng=random.Random())

    n_roots = len(roots_list)
    moves_budget = min(max_moves, max(2000, moves_multiplier * n_roots))
    if verbose:
        print(f"[smooth_recursive] moves_budget={moves_budget}, batch={batch}")

    total_scanned = 0
    accepted = 0
    scanned_relations = 0

    temp_indices = set(temp_map[a] for a in target_atoms if a in temp_map)

    # Main loop: collect in batches and scan for relations that include any temp columns
    while total_scanned < moves_budget:
        to_take = min(batch, moves_budget - total_scanned)
        rs.collect_relations(to_take)
        total_scanned += to_take

        if verbose:
            print(f"[smooth_recursive] collected {to_take} moves (total_scanned={total_scanned}) - divisors now {len(rs.divisors)}, relations {len(rs.relations)}")

        # iterate only over newly added relations for efficiency
        start_idx = max(0, len(rs.relations) - to_take)
        for rel_idx, rel in enumerate(rs.relations[start_idx:], start=start_idx):
            scanned_relations += 1

            # expand divisor-level relation to atom-level using new_map
            atom_row = {}
            for div_idx, coeff in rel.items():
                # guard: if div_idx references divisor added earlier, OK
                try:
                    a, b = rs.divisors[div_idx]
                except Exception as e:
                    # defensive guard: skip malformed
                    if verbose:
                        print(f"[smooth_recursive] skipping malformed divisor index {div_idx}: {e}")
                    atom_row = {}
                    break
                ia = new_map[a]
                ib = new_map[b]
                atom_row[ia] = atom_row.get(ia, 0) + coeff
                atom_row[ib] = atom_row.get(ib, 0) + coeff

            # reduce mod ell if given
            if ell:
                for k in list(atom_row.keys()):
                    atom_row[k] %= ell
                    if atom_row[k] == 0:
                        del atom_row[k]

            if not atom_row:
                continue

            # Check if this relation mentions any temporary (target) indices
            mentions_temp = any(idx in temp_indices for idx in atom_row.keys())
            if not mentions_temp:
                continue

            # Quick diagnostic print about this candidate relation
            if verbose:
                sample = list(atom_row.items())[:8]
                print(f"[smooth_recursive] candidate relation found (rel_idx={rel_idx}, scanned={scanned_relations})")
                print(f"  sample atoms (idx,coeff): {sample}{'...' if len(atom_row)>8 else ''}")
                temp_keys = [k for k in atom_row.keys() if k in temp_indices]
                non_temp_keys = [k for k in atom_row.keys() if k not in temp_indices]
                print(f"  mentions temp indices: {temp_keys}")
                print(f"  non-temp count: {len(non_temp_keys)} (max_orig_idx={max(atom_to_idx.values()) if atom_to_idx else -1})")

            # Ensure non-temp part maps back into original FB only (simple conservative check)
            if atom_to_idx:
                max_orig_idx = max(atom_to_idx.values())
            else:
                max_orig_idx = -1
            non_temp_cols = [k for k in atom_row.keys() if k not in temp_indices]
            if any(k > max_orig_idx for k in non_temp_cols):
                if verbose:
                    print("[smooth_recursive] Skipping candidate: uses extra temporary atoms outside original FB")
                continue

            # success! return the raw atom_row and mapping information
            accepted += 1
            if verbose:
                print(f"[smooth_recursive] ACCEPTED relation after scanning {scanned_relations} relations (total moves {total_scanned})")

            return {
                'atom_row': atom_row,
                'new_atom_map': new_map,
                'temp_map': temp_map,
                'status': 'found',
                'stats': {
                    'total_moves': total_scanned,
                    'relations_scanned': scanned_relations,
                    'accepted': accepted
                }
            }

        # end of scanning batch - print progress
        if verbose:
            print(f"[smooth_recursive] batch complete: total_scanned={total_scanned}, relations_scanned={scanned_relations}, accepted={accepted}")

    # timeout/no relation found
    if verbose:
        print(f"[smooth_recursive] TIMEOUT: scanned {scanned_relations} relations over {total_scanned} moves; no usable relation found")
    return {
        'atom_row': {},
        'new_atom_map': new_map,
        'temp_map': temp_map,
        'status': 'timeout',
        'stats': {
            'total_moves': total_scanned,
            'relations_scanned': scanned_relations,
            'accepted': accepted
        }
    }
