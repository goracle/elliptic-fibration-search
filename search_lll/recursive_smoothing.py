import random, time
from typing import Tuple, Dict, List, Optional, Any, Callable

# recursive_smoothing.py

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

def pairify(atoms: tuple) -> tuple:
    """Standardizes atom sets of any weight (0, 1, 2) for indexing."""
    return tuple(sorted(atoms))

class RecursiveSmoother:

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

    def random_move(self):
            """
            Performs a root-preserving swap to generate a new relation.
            D1 + D2 = D_new1 + D_new2
            """
            # Pick two divisors (could be FB or existing P_junk)
            i, j = random.sample(range(len(self.divisors)), 2)

            # A, P1, B, P2 are all points (roots) on the curve
            A, P1 = self.divisors[i]
            B, P2 = self.divisors[j]

            # The 'Shuffle': Create new pairings from the same 4 points
            # This is the "move" that stays within the span of the roots
            new1 = pairify(P1, P2)
            new2 = pairify(A, B)

            idx_new1 = self._ensure_divisor(new1)
            idx_new2 = self._ensure_divisor(new2)

            # Record the relation: Div[i] + Div[j] - Div[new1] - Div[new2] = 0
            # In terms of indices: e_i + e_j - e_new1 - e_new2 = 0
            rel = {i: 1, j: 1, idx_new1: -1, idx_new2: -1}

            # Clean up zeros (if i or j happen to be the same as new1 or new2)
            rel = {k: v % self.modulus for k, v in rel.items() if v % self.modulus != 0}

            return rel

    def update_substitution_map(self, relation):
        """
        Try to use a new relation to express one P_junk in terms of
        FB elements or 'older' P_junks.
        """
        # Find a P_junk in the relation to 'solve' for
        target_idx = -1
        for idx in relation.keys():
            if self.is_temporary.get(idx, False):
                target_idx = idx
                break

        if target_idx != -1:
            # Re-arrange relation: P_junk = sum(coeffs * other_elements)
            # This is your 'formal sum' logic.
            new_expression = self.solve_for_index(relation, target_idx)
            self.substitutions[target_idx] = new_expression

    def collect_relations(self, target_idx, max_moves=10000, verbose=False):
        """
        Attempts to express target_idx (G or Q) as a formal sum of FB elements.
        Uses a substitution map to keep P_junk transient.
        """
        # expressions[idx] = {fb_index: coefficient}
        # FB elements represent themselves: Div_i = 1 * Div_i
        expressions = {i: {i: 1} for i in range(len(self.divisors))
                       if not self.is_temporary.get(i, False)}

        total_moves = 0
        while total_moves < max_moves:
            total_moves += 1

            # 1. Generate a move: D1 + D2 - D3 - D4 = 0
            rel = self.random_move()

            # 2. Substitute known expressions into the relation
            # This reduces the relation to: sum(c_k * FB_k) + sum(c_j * P_junk_j) = 0
            resolved_part = {}  # FB indices -> coeff
            unknowns = {}       # P_junk indices -> coeff

            for idx, coeff in rel.items():
                if idx in expressions:
                    # Substitute the known formal sum
                    for fb_idx, fb_coeff in expressions[idx].items():
                        new_c = (resolved_part.get(fb_idx, 0) + coeff * fb_coeff) % self.modulus
                        if new_c == 0:
                            resolved_part.pop(fb_idx, None)
                        else:
                            resolved_part[fb_idx] = new_c
                else:
                    # Still a P_junk we don't know yet
                    unknowns[idx] = coeff

            # 3. Check if we can "solve" for a new P_junk
            if len(unknowns) == 1:
                # Equation: ResolvedPart + c_u * P_unknown = 0  => P_unknown = -ResolvedPart * (c_u^-1)
                u_idx, u_coeff = list(unknowns.items())[0]
                inv_u = pow(int(u_coeff), -1, self.modulus)

                new_expr = {}
                for fb_idx, fb_coeff in resolved_part.items():
                    # P_unknown = -fb_coeff * inv_u
                    val = (-fb_coeff * inv_u) % self.modulus
                    if val != 0:
                        new_expr[fb_idx] = val

                expressions[u_idx] = new_expr

                # Check if we just resolved our target!
                if u_idx == target_idx:
                    if verbose:
                        print(f"[Smoother] Target resolved in {total_moves} moves.")
                    return expressions[target_idx]

            # 4. Implicit Equation / Loop Detection
            elif len(unknowns) == 0 and len(resolved_part) > 0:
                # We found a relation purely among FB elements!
                # We can store this in self.relations for the global linear system.
                self.relations.append(resolved_part)
                if verbose and len(self.relations) % 100 == 0:
                    print(f"[Smoother] Found {len(self.relations)} implicit FB relations...")

        if verbose:
            print(f"[Smoother] Timeout after {total_moves} moves.")
        return None

    def _expand_expression(self, expr):
        """Helper to ensure an expression is fully reduced mod L."""
        return {idx: (val % self.modulus) for idx, val in expr.items() if (val % self.modulus) != 0}

    def __init__(self, roots, divisors=None, relations=None, modulus=None):
        self.roots = list(roots)
        self.modulus = modulus
        self.is_temporary = {}  # Added: Initialize the state map

        self.roots = list(roots)
        self.modulus = modulus
        # Track divisors as tuples of atoms: () weight 0, (A,) weight 1, (A, B) weight 2
        self.divisors = []
        self.div_index = {}

        # Initialize with atoms and pairs (the initial Factor Base)
        for i in range(len(self.roots)):
            self._ensure_divisor((self.roots[i],))
            for j in range(i, len(self.roots)):
                self._ensure_divisor((self.roots[i], self.roots[j]))

        self.relations = []

        if divisors is None:
            self.divisors = []
            n = len(self.roots)
            for i in range(n):
                for j in range(i, n):
                    # Initial FB is NOT temporary
                    d = pairify(self.roots[i], self.roots[j])
                    idx = len(self.divisors)
                    self.div_index[d] = idx
                    self.divisors.append(d)
                    self.is_temporary[idx] = False
        else:
            self.divisors = []
            self.div_index = {}
            for d in divisors:
                d_pair = pairify(*d)
                idx = len(self.divisors)
                self.div_index[d_pair] = idx
                self.divisors.append(d_pair)
                self.is_temporary[idx] = False # FB elements are permanent

        self.relations = relations[:] if relations else []

    def _ensure_divisor(self, d):
        if d not in self.div_index:
            idx = len(self.divisors)
            self.div_index[d] = idx
            self.divisors.append(d)
            # If it's not in the initial FB, it's a P_junk
            self.is_temporary[idx] = True
            return idx
        return self.div_index[d]

def smooth_element_via_recursive(
    element,
    roots,
    modulus,
    max_moves=20000,
    max_divisors=200000,
    verbose=True,
):
    """
    Memory-safe recursive smoother.

    Divisors are stored as integer pairs referencing the `roots` list.
    Prevents RAM blowups and prints useful progress diagnostics.
    """

    rng = random.Random()

    # atom -> id
    atom_to_id = {a: i for i, a in enumerate(roots)}

    # canonical pair helper
    def canon(a, b):
        return (a, b) if a <= b else (b, a)

    # divisor storage
    divisors = []
    div_index = {}

    # relation storage (compact tuples)
    relations = []

    # initialize with random pair
    a = rng.randrange(len(roots))
    b = rng.randrange(len(roots))
    current = canon(a, b)

    divisors.append(current)
    div_index[current] = 0

    start = time.time()
    last_print = start

    for step in range(max_moves):

        # progress output
        if verbose and time.time() - last_print > 0.5:
            print(
                f"\rstep={step}  divisors={len(divisors)}  relations={len(relations)}",
                end="",
                flush=True,
            )
            last_print = time.time()

        # state explosion protection
        if len(divisors) >= max_divisors:
            raise RuntimeError("Recursive smoother exceeded divisor limit")

        # choose shuffle root
        r = rng.randrange(len(roots))

        A, B = current

        # shuffle rule (simple mixing)
        new1 = canon(A, r)
        new2 = canon(r, B)

        # insert / lookup divisor 1
        if new1 not in div_index:
            div_index[new1] = len(divisors)
            divisors.append(new1)
        i1 = div_index[new1]

        # insert / lookup divisor 2
        if new2 not in div_index:
            div_index[new2] = len(divisors)
            divisors.append(new2)
        i2 = div_index[new2]

        # record relation
        relations.append((A, B, new1[0], new1[1]))

        # check factor-base condition
        if new1[0] < len(roots) and new1[1] < len(roots):
            if verbose:
                print("\n[Smooth] factor-base divisor reached")
            return new1

        # random walk step
        current = new2 if rng.random() < 0.5 else new1

    raise RuntimeError("Recursive smoothing failed to find smooth divisor")
