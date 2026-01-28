import numpy as np, ctypes, os, random, psutil, time, queue
from ctypes import c_uint32, c_uint64, POINTER, byref, c_int
from multiprocessing import Process, Queue, Event, cpu_count, set_start_method, Manager
from queue import Full
from search_common import FINITE_FIELD, sage_poly_from_coeffs
from sage.all import GF, PolynomialRing, ZZ, HyperellipticCurve

# CRITICAL: Force spawn mode to prevent fork-based RAM explosion
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    raise # always raise your exceptions, never pass

# Load C library
_lib_path = os.path.join(os.path.dirname(__file__), "libwalk.so")
if not os.path.exists(_lib_path):
    raise FileNotFoundError(f"libwalk.so not found at {_lib_path}")

lib = ctypes.CDLL(_lib_path)

lib.collision_walk.argtypes = [
    POINTER(c_uint32),  # atom_indices
    c_uint32,           # n_atoms
    POINTER(c_uint64),  # rand_table
    c_uint64,           # target_mask
    c_uint32,           # max_terms
    c_uint64,           # seed
    POINTER(c_uint32),  # touched (out)
    POINTER(c_uint32),  # counts (out)
    POINTER(c_uint32),  # exps (scratch)
    POINTER(c_uint32),  # out_len (out)
    POINTER(c_uint64),  # out_state (out)
    c_uint64,           # max_steps
]
lib.collision_walk.restype = c_int

# ----- module globals for worker/Jacobian hashing -----
_GLOBAL_IDX_TO_ATOM = None   # idx -> atom (filled before spawning workers)
_GLOBAL_ATOM_TO_IDX = None   # atom -> idx  (may already exist elsewhere)
_GLOBAL_J = None             # Sage Jacobian class/instance
_GLOBAL_F_POLY = None        # global curve poly, optional
_GLOBAL_FB_Y_CACHE = None    # optional canonical y cache
# ----------------------------------------------------

def homomorphism_test(J, atom_to_idx, f_p, p,
                             trials=200,
                             max_atoms_in_sum=3,
                             max_weight=1,
                             check_divisors=None,
                             require_all_encodable=False,
                             verbose=True):
    """
    Homomorphism test that is meaningful for a Jacobian-level factor base.

    Philosophy:
      - We test the map Phi: Z^{atoms} -> J(F_p) given by atoms -> Jacobian elements.
      - Pick random, small linear combinations of atoms (in the domain), reconstruct them
        in J via _atom_to_jac_helper, add in J, and confirm that adding the encodings
        equals encoding the sums (i.e., reconstruct(enc1+enc2) == reconstruct(enc1)+reconstruct(enc2)).

    Args:
      J: Sage Jacobian class/instance (used for J.zero()).
      atom_to_idx: { atom_tuple -> col_index } mapping (indices may be ints).
      f_p: curve polynomial (for polynomial ring / R), used by helpers.
      p: prime modulus (int).
      trials: how many random pair-tests to run.
      max_atoms_in_sum: max number of atoms included in a random combination.
      max_weight: maximum absolute coefficient per chosen atom (int >=1).
      check_divisors: optional list of explicit stored divisors (Mumford dicts or [u,v]) to test encoding correctness.
      require_all_encodable: if True, treat any non-encodable divisor in check_divisors as failure.
      verbose: if True, print diagnostics on first failure and final summary.

    Returns:
      True if the test passed, False otherwise.
    """

    if not atom_to_idx:
        if verbose:
            print("homomorphism_test_atomic: empty atom_to_idx -> trivially passes")
        return True

    # Build index -> atom mapping (ensure integer indices)
    idx_to_atom = {}
    for atom, idx in atom_to_idx.items():
        try:
            idxi = int(idx)
        except Exception:
            idxi = idx
            raise
        idx_to_atom[idxi] = atom

    atom_indices = list(idx_to_atom.keys())
    R = None
    try:
        R = f_p.parent()
    except Exception:
        # some helper functions may accept f_p or R; we only pass R where needed
        R = None
        raise

    # helper: reconstruct Jacobian element from an encoding dict {idx: coeff}
    def reconstruct(enc):
        D = J.zero()
        for idx, coeff in enc.items():
            if coeff == 0:
                continue
            atom = idx_to_atom[int(idx)]
            # _atom_to_jac_helper is expected to return a Jacobian element for the atom
            D += int(coeff) * _atom_to_jac_helper(atom, J, R)
        return D

    # helper: add two sparse row-dicts
    def add_rows(a, b):
        res = dict(a)
        for k, v in b.items():
            res[k] = res.get(k, 0) + v
            if res[k] == 0:
                del res[k]
        return res

    # 1) Domain-based random tests: random combinations of FB atoms
    tested = 0
    for t in range(trials):
        # build enc1
        k1 = random.randint(1, max_atoms_in_sum)
        k2 = random.randint(1, max_atoms_in_sum)
        support1 = random.sample(atom_indices, k1)
        support2 = random.sample(atom_indices, k2)
        enc1 = {}
        enc2 = {}
        for idx in support1:
            coeff = random.randint(1, max_weight) * random.choice([-1, 1])
            enc1[idx] = enc1.get(idx, 0) + coeff
            if enc1[idx] == 0:
                del enc1[idx]
        for idx in support2:
            coeff = random.randint(1, max_weight) * random.choice([-1, 1])
            enc2[idx] = enc2.get(idx, 0) + coeff
            if enc2[idx] == 0:
                del enc2[idx]

        # reconstruct and test
        try:
            D1 = reconstruct(enc1)
            D2 = reconstruct(enc2)
        except Exception as e:
            if verbose:
                print("ERROR reconstructing atom combinations:", e)
            raise
            return False

        combined = add_rows(enc1, enc2)
        D_recon_combined = reconstruct(combined)
        D_sum = D1 + D2

        if D_recon_combined != D_sum:
            if verbose:
                print("HOMOMORPHISM FAILURE on atom-generated sums")
                print("enc1:", enc1)
                print("enc2:", enc2)
                print("combined:", combined)
                print("reconstruct(enc1)+reconstruct(enc2) =", D_sum)
                print("reconstruct(combined)               =", D_recon_combined)
            return False

        tested += 1

    if verbose:
        print(f"homomorphism_test: {tested}/{trials} atom-based encodable pairs tested, all passed")

    # 2) Optional: check explicit divisors (if provided)
    if check_divisors:
        encable_count = 0
        for D in check_divisors:
            try:
                enc = get_relation_row(D, atom_to_idx, f_p, p)
            except Exception as e:
                if verbose:
                    print("get_relation_row raised:", e)
                enc = None
                raise

            if enc is None:
                if require_all_encodable:
                    if verbose:
                        print("HOMOMORPHISM FAILURE: provided divisor not encodable:", D)
                    return False
                # skip non-encodable
                continue

            # compare reconstructed atom-sum to explicit Jacobian
            try:
                D_from_atoms = reconstruct(enc)
                D_true = _dict_to_jac_helper(D, J, R) if isinstance(D, dict) else _divisor_to_jac_helper(D, J, R)
            except Exception as e:
                if verbose:
                    print("ERROR during divisor reconstruction check:", e)
                raise
                return False

            if D_from_atoms != D_true:
                if verbose:
                    print("HOMOMORPHISM FAILURE on explicit divisor encoding")
                    print("divisor:", D)
                    print("enc:", enc)
                    print("reconstructed from atoms:", D_from_atoms)
                    print("actual Jacobian element   :", D_true)
                return False
            encable_count += 1

        if verbose:
            print(f"homomorphism_test: checked {encable_count} explicit divisors (encodable ones)")

    return True

def _atom_to_jac_helper(atom, J, R):
    """
    Convert factor base atom -> Jacobian element.
    Used by homomorphism_test and debug_homomorphism_failure.
    """
    if atom is None:
        raise RuntimeError("missing atom for index")

    kind = atom[0]
    x = R.gen()

    if kind == 'd1':
        _, x0, y0 = atom
        u = x - R(x0)
        v = R(y0)
        return J([u, v])

    elif kind == 'd2':
        _, u_coeffs, v_coeffs = atom
        return J([R(list(u_coeffs)), R(list(v_coeffs))])

    else:
        raise RuntimeError(f"unknown atom kind: {kind}")

def debug_homomorphism_failure(J, atom_to_idx, fb_y_cache, p, f_p, D1, D2, enc1, enc2):
    """
    UPDATED: Uses extracted helper functions.

    Detailed diagnostics for a homomorphism failure.
    """
    F = GF(int(p))
    R = PolynomialRing(F, 'x')
    x = R.gen()

    idx_to_atom = {int(idx): atom for atom, idx in atom_to_idx.items()}

    # ---- NEW: export worker globals so children can reconstruct Jacobians ----
    global _GLOBAL_IDX_TO_ATOM, _GLOBAL_ATOM_TO_IDX, _GLOBAL_J, _GLOBAL_F_POLY, _GLOBAL_FB_Y_CACHE
    _GLOBAL_IDX_TO_ATOM = idx_to_atom             # idx -> atom mapping (int -> atom tuple)
    _GLOBAL_ATOM_TO_IDX = {k: int(v) for k, v in atom_to_idx.items()}  # keep old shape if needed
    _GLOBAL_F_POLY = f_p
    _GLOBAL_FB_Y_CACHE = fb_y_cache

    # J: the Jacobian class instance. You likely have it as `J` in caller's scope.
    # If you only have the class, set _GLOBAL_J = J; if you have an instance, set to its class or
    # an object with .zero() etc. Example:
    #    _GLOBAL_J = J  # J must be a Sage Jacobian object/class already imported/constructed
    _GLOBAL_J = J
    # ---- end new globals export ----

    print("\n=== HOMOMORPHISM DEBUG ===")
    print("p =", p)
    print("enc(D1) =", enc1)
    print("enc(D2) =", enc2)
    print("D1 keys:", sorted(D1.keys()))
    print("D2 keys:", sorted(D2.keys()))
    v1 = D1.get('vector')
    v2 = D2.get('vector')
    print("vector D1 (len):", (len(v1) if hasattr(v1, '__len__') else None))
    print("vector D2 (len):", (len(v2) if hasattr(v2, '__len__') else None))
    print("vector D1 sample:", v1[:20] if hasattr(v1, '__len__') else v1)
    print("vector D2 sample:", v2[:20] if hasattr(v2, '__len__') else v2)

    # Show atoms referenced by enc1/enc2
    idxs = sorted(set(list(enc1.keys()) + list(enc2.keys())))
    print("\nAtoms referenced (idx -> atom):")
    for idx in idxs:
        atom = idx_to_atom.get(int(idx), None)
        print(f"  {idx:5d} -> {atom}")

    # Reconstruct combined enc
    combined = {}
    for k, v in enc1.items():
        combined[k] = combined.get(k, 0) + int(v)
    for k, v in enc2.items():
        combined[k] = combined.get(k, 0) + int(v)
    combined = {int(k): int(v) for k, v in combined.items() if int(v) != 0}

    # Reconstruct Jacobians using helpers
    J_recon = J.zero()
    for idx, mult in combined.items():
        atom = idx_to_atom.get(int(idx))
        if atom is None:
            print("WARNING: no atom for idx", idx)
            continue
        atomJ = _atom_to_jac_helper(atom, J, R)
        J_recon += int(mult) * atomJ

    J_true = _dict_to_jac_helper(D1, J, R) + _dict_to_jac_helper(D2, J, R)

    print("\nReconstructed (from atoms) Mumford:", J_recon)
    print("Actual sum (from dicts) Mumford      :", J_true)

    if J_recon == J_true:
        print("=> They are equal (unexpected here).")
        return

    # Print explicit u,v polys for comparison
    u_recon, v_recon = J_recon[0], J_recon[1]
    u_true, v_true = J_true[0], J_true[1]
    print("\nReconstructed u:", u_recon)
    print("Actual        u:", u_true)
    print("\nReconstructed v:", v_recon)
    print("Actual        v:", v_true)

    # Attempt to express difference as factor base combination
    diff = J_recon - J_true
    print("\nDifference (J_recon - J_true):", diff)
    try:
        enc_diff = get_relation_row(diff, atom_to_idx, f_p, p)
        print("Difference is FB-smooth; encoding:", enc_diff)
    except Exception as e:
        print("Could not encode difference (get_relation_row raised):", repr(e))
        raise

    # For each d1 atom referenced, check canonical y vs evaluated v(x)
    print("\nCanonical vs evaluated y checks for d1 atoms in enc:")
    for idx in idxs:
        atom = idx_to_atom.get(int(idx))
        if atom is None:
            continue
        if atom[0] != 'd1':
            continue
        x0 = int(atom[1])
        y_can = int(atom[2])

        v_eval_D1 = None
        v_eval_D2 = None
        try:
            v_eval_D1 = int((R(D1['v_1'])*R(x0) + R(D1['v_0']))) % int(p)
        except Exception:
            raise
        try:
            v_eval_D2 = int((R(D2['v_1'])*R(x0) + R(D2['v_0']))) % int(p)
        except Exception:
            raise

        print(f" idx {idx}: atom x={x0}, y_can={y_can}, v_eval_D1={v_eval_D1}, v_eval_D2={v_eval_D2}")
        if fb_y_cache is not None:
            print("   fb_y_cache[x]:", fb_y_cache.get(int(x0)))

    print("\n=== END DEBUG ===\n")

def _dict_to_jac_helper(div, J, R):
    """
    Convert Mumford dict -> Jacobian element.
    Used by homomorphism_test and debug_homomorphism_failure.
    """
    s = div['s']
    pval = div['p']
    v0 = div['v_0']
    v1c = div['v_1']
    x = R.gen()
    u = x**2 - R(s)*x + R(pval)
    v = R(v1c)*x + R(v0)
    return J([u, v])

def collision_walk_c(atom_indices_np, rand_table_np, target_mask, max_terms, seed, exps_np, max_steps):
    """
    Call C collision walk kernel.
    Returns: (retcode, state, touched_indices, counts)
    retcode: 1 => DP found, 0 => abandoned, -1 => error
    """
    n_atoms = atom_indices_np.shape[0]
    touched = np.empty(max_terms, dtype=np.uint32)
    counts = np.empty(max_terms, dtype=np.uint32)
    out_len   = c_uint32()
    out_state = c_uint64()

    ret = lib.collision_walk(
        atom_indices_np.ctypes.data_as(POINTER(c_uint32)),
        c_uint32(n_atoms),
        rand_table_np.ctypes.data_as(POINTER(c_uint64)),
        c_uint64(int(target_mask)),
        c_uint32(max_terms),
        c_uint64(seed),
        touched.ctypes.data_as(POINTER(c_uint32)),
        counts.ctypes.data_as(POINTER(c_uint32)),
        exps_np.ctypes.data_as(POINTER(c_uint32)),
        byref(out_len),
        byref(out_state),
        c_uint64(max_steps)
    )

    if ret == 1:
        L = int(out_len.value)
        return 1, int(out_state.value), touched[:L].copy(), counts[:L].copy()
    else:
        return int(ret), None, None, None

def get_relation_row_cached(divisor, require_signed_d2=True):
    """
    Worker-safe version: use Mumford v(x) directly and atom-based lookup.

    Returns {col_idx: multiplicity_signed} or None if not FB-smooth / not in FB.
    Raises on unexpected errors (per your loud-failure policy).

    IMPORTANT: prefer signed ('d2', u_coeffs, v_can) atoms when available.
    If require_signed_d2==True and no matching signed d2 atom exists, return None.
    """
    global _GLOBAL_ATOM_TO_IDX, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY
    # REMOVE any disabling assert so this runs in workers
    # assert None, "does this even run anymore?"

    if _GLOBAL_ATOM_TO_IDX is None:
        raise RuntimeError("_GLOBAL_ATOM_TO_IDX not initialized in worker")

    if FINITE_FIELD is None:
        raise RuntimeError("FINITE_FIELD not initialized in worker")

    # Local field objects for evaluation
    p = int(FINITE_FIELD)
    K = GF(p)
    R = PolynomialRing(K, 'x')

    try:
        u_poly, v_poly = divisor[0], divisor[1]
    except Exception as e:
        raise RuntimeError(f"get_relation_row_cached: malformed divisor input: {e}")

    deg = int(u_poly.degree())
    if deg not in (1, 2):
        return None

    # Ensure u splits completely into linear factors over K when we need roots
    try:
        roots_data = u_poly.roots(K)
    except Exception as e:
        raise RuntimeError(f"get_relation_row_cached: u_poly.roots() failed: {e}")

    if sum(m for _, m in roots_data) != deg:
        return None

    row = {}

    # Helper: convert polynomial to canonical tuple (lowest->highest), pad to deg (monic expected)

    # If degree == 2, first try to match a signed d2 atom
    if deg == 2:
        u_key = poly_to_tuple(u_poly, 2)  # (u0, u1, 1) expected for monic
        # Build v_tuple from v_poly coefficients (pad to length 2)
        v_list = [int(K(c)) for c in v_poly.list()]  # could be length 0..1
        # Normalize length to 2
        if len(v_list) < 2:
            v_list = v_list + [0] * (2 - len(v_list))
        v_tuple = tuple(v_list)

        # Search for matching d2 atoms in the global map
        matched = False
        for atom, idx in _GLOBAL_ATOM_TO_IDX.items():
            try:
                if atom[0] != 'd2':
                    continue
                # atom layout expected: ('d2', u_coeffs_tuple) or ('d2', u_coeffs_tuple, v_can_tuple)
                atom_u = tuple(int(x) % p for x in atom[1])
                if atom_u != u_key:
                    continue
                # candidate matches u; check v_can if available
                if len(atom) >= 3:
                    v_can_tuple = tuple(int(x) % p for x in atom[2])
                    # normalize lengths
                    L = max(len(v_tuple), len(v_can_tuple))
                    vt = list(v_tuple) + [0]*(L - len(v_tuple))
                    vc = list(v_can_tuple) + [0]*(L - len(v_can_tuple))
                    vt = [int(K(c)) % p for c in vt]
                    vc = [int(K(c)) % p for c in vc]
                    # check equality or negation
                    if vt == vc:
                        row[int(idx)] = row.get(int(idx), 0) + 1
                        matched = True
                        break
                    neg_vc = [(-c) % p for c in vc]
                    if vt == neg_vc:
                        row[int(idx)] = row.get(int(idx), 0) - 1
                        matched = True
                        break
                    # else, not matching v_can, continue searching
                else:
                    # candidate has u only; ambiguous sign — skip
                    continue
            except Exception:
                # be robust to unexpected atom formatting; skip
                raise
                continue

        if matched:
            # Found a single signed d2 atom representation; done
            return row

        # No matching signed d2 atom found
        if require_signed_d2:
            return None
        # else fall through to split into degree-1 atoms (legacy fallback)

    # Legacy / deg==1 handling (or deg==2 fallback)
    for x_elem, mult in roots_data:
        x_int = int(x_elem)

        # get canonical y (preferred from worker cache)
        y_can = None
        if _GLOBAL_FB_Y_CACHE is not None:
            y_can = _GLOBAL_FB_Y_CACHE.get(int(x_int), None)

        if y_can is None:
            # fallback: try to compute canonical y from stored global f(x) if available
            if _GLOBAL_F_POLY is not None:
                try:
                    y2 = int(_GLOBAL_F_POLY(K(x_int)))
                except Exception:
                    raise
                    return None
                if y2 == 0:
                    y_can = 0
                elif pow(y2, (p - 1) // 2, p) != 1:
                    return None
                else:
                    # minimal tonelli_shanks call (assumes function exists)
                    from .smoothness import tonelli_shanks
                    y_can_tmp = tonelli_shanks(y2, p)
                    y_can = int(min(y_can_tmp, p - y_can_tmp))
            else:
                # no canonical y available -> cannot reliably sign degree-1 atom
                return None

        # evaluate Mumford v at x
        try:
            y_val = int(v_poly(x_elem)) % p
        except Exception:
            raise
            return None

        # determine sign relative to canonical y
        if y_can == 0:
            sign = +1
        elif y_val == int(y_can):
            sign = +1
        elif (p - y_val) % p == int(y_can):
            sign = -1
        else:
            # ambiguous / not matching canonical ±sqrt
            return None

        atom = ('d1', int(x_int), int(y_can))
        idx = _GLOBAL_ATOM_TO_IDX.get(atom)
        if idx is None:
            # atom not present in factor base
            return None

        row[int(idx)] = row.get(int(idx), 0) + int(sign) * int(mult)
        if row[int(idx)] == 0:
            del row[int(idx)]

    return row

def compute_jacobian_hash(exp_dict, atom_indices_list):
    """
    Compute a deterministic hash for the Jacobian element represented by exp_dict.

    Strategy:
      - Use the global _GLOBAL_IDX_TO_ATOM mapping to reconstruct the Jacobian element
        via _atom_to_jac_helper and the globally-provided Jacobian class _GLOBAL_J.
      - Canonicalize the Mumford (u,v) pair to integer coefficient tuples mod p.
      - Mix them into a 64-bit integer via a stable FNV-like mix.
      - If anything goes wrong (missing atom mapping), fall back to a stable hash of the
        exponent vector — but that fallback won't produce meaningful collisions.
    """
    global _GLOBAL_IDX_TO_ATOM, _GLOBAL_J
    try:
        # fast fallback: if no globals set, fall back to exponent-vector hash
        if _GLOBAL_IDX_TO_ATOM is None or FINITE_FIELD is None or _GLOBAL_J is None:
            # deterministic fallback: sort items and FNV-mix them
            items = tuple(sorted((int(k), int(v)) for k, v in exp_dict.items()))
            h = 1469598103934665603
            for k, v in items:
                h ^= (k & 0xFFFFFFFFFFFFFFFF)
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                h ^= (v & 0xFFFFFFFFFFFFFFFF)
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            return h

        # sagemath imports and field setup (lazy inside function to keep module import light)
        from sage.all import GF, PolynomialRing
        p = int(FINITE_FIELD)
        K = GF(p)
        R = PolynomialRing(K, 'x')

        # Reconstruct Jacobian element
        Jclass = _GLOBAL_J  # should be the Jacobian class or an instance with zero()
        Jzero = Jclass.zero()
        J_recon = Jzero

        # idx->atom mapping: integer indices expected
        idx_to_atom = _GLOBAL_IDX_TO_ATOM

        for idx, mult in exp_dict.items():
            if int(mult) == 0:
                continue
            atom = idx_to_atom.get(int(idx))
            if atom is None:
                # missing atom means we cannot canonicalize -> fallback
                raise KeyError(f"missing atom for idx {idx}")
            atomJ = _atom_to_jac_helper(atom, Jclass, R)
            J_recon += int(mult) * atomJ

        # Extract Mumford u, v polys
        u_poly = J_recon[0]
        v_poly = J_recon[1]

        # Canonical coefficient tuples: lowest->highest (pad to consistent lengths)
        # u degree may be 1 or 2 for reduced divisors; pad to deg 2 for stable representation
        def poly_to_int_tuple(poly, deg_pad):
            coeffs = [0] * (deg_pad + 1)
            for i, c in enumerate(poly.list()):
                if i <= deg_pad:
                    coeffs[i] = int(K(c))  # reduce mod p into python int
            return tuple(coeffs)

        u_tuple = poly_to_int_tuple(u_poly, 2)   # (u0, u1, u2) for monic u; u2 likely 1
        # v deg < deg(u), pad to length 2 for deterministic length
        v_tuple = poly_to_int_tuple(v_poly, 1)   # (v0, v1)

        # build stable key and mix to 64-bit hash (FNV-1a-ish)
        key_parts = (u_tuple, v_tuple)
        h = 1469598103934665603
        for part in key_parts:
            for num in part:
                # fold each integer (can be large) into 64-bit lanes
                x = int(num) & 0xFFFFFFFFFFFFFFFF
                h ^= x
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF

        return h

    except Exception:
        raise
        # why tf does a hash function need a fallback?  this is so dumb
        items = tuple(sorted((int(k), int(v)) for k, v in exp_dict.items()))
        h = 1469598103934665603
        for k, v in items:
            h ^= (k & 0xFFFFFFFFFFFFFFFF)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            h ^= (v & 0xFFFFFFFFFFFFFFFF)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

def get_relation_row(divisor, atom_to_idx, f_p, p,
                     fb_y_cache=None, require_signed_d2=True):
    """
    Build factor-base row for `divisor` with support for SIGNED degree-2 atoms.

    OPTIMIZED: Uses cached lookup dictionaries.
    """
    p = int(p)
    K = GF(p)

    try:
        R = f_p.parent()
        x = R.gen()
    except Exception:
        R = PolynomialRing(K, 'x')
        x = R.gen()
        raise # why raise here?  why not raise here?  don't tell me to delete this raise again

    def poly_to_tuple(poly, deg_expected):
        coeffs = [0] * (deg_expected + 1)
        for i, c in enumerate(poly.list()):
            if i <= deg_expected:
                coeffs[i] = int(K(c))
        return tuple(coeffs)

    # Build cache key from atom_to_idx identity
    cache_key = id(atom_to_idx)

    if cache_key not in get_relation_row.cache:
        d1_by_x = {}
        d2_by_u = {}

        for atom, idx in atom_to_idx.items():
            if atom[0] == 'd1':
                x_val = int(atom[1])
                d1_by_x[x_val] = (atom, int(idx))
            elif atom[0] == 'd2':
                u_key = tuple(int(x) % p for x in atom[1])
                entries = d2_by_u.setdefault(u_key, [])
                entries.append((atom, int(idx)))

        get_relation_row.cache[cache_key] = (d1_by_x, d2_by_u)

    d1_by_x, d2_by_u = get_relation_row.cache[cache_key]
    row = {}

    # Extract u_poly, v_poly
    if isinstance(divisor, dict):
        assert 's' in divisor and 'p' in divisor and 'v_0' in divisor and 'v_1' in divisor
        s_val = int(divisor['s'])
        p_val = int(divisor['p'])
        v0 = K(int(divisor['v_0']))
        v1 = K(int(divisor['v_1']))
        u_poly = x**2 - K(s_val)*x + K(p_val)
        v_poly = v1 * x + v0
    else:
        try:
            u_poly = divisor[0]
            v_poly = divisor[1]
        except Exception:
            raise RuntimeError("get_relation_row: unsupported divisor type")

    deg = int(u_poly.degree())
    if deg not in (1, 2):
        return None

    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != deg:
        return None

    # CASE: degree 1
    if deg == 1:
        for x_elem, mult in roots_data:
            x_int = int(x_elem)
            entry = d1_by_x.get(x_int)
            if entry is None:
                return None
            atom, idx = entry
            y_can = int(atom[2])
            v_at_x = int(v_poly(x_elem)) % p
            if v_at_x == y_can:
                sign = 1
            elif (p - v_at_x) % p == y_can:
                sign = -1
            else:
                return None
            row[idx] = row.get(idx, 0) + sign * int(mult)
            if row[idx] == 0:
                del row[idx]
        return row

    # CASE: degree 2 - try signed d2 match first
    u_coeffs = poly_to_tuple(u_poly, 2)
    v_tuple = tuple(int(K(c)) for c in v_poly.list())

    candidates = d2_by_u.get(u_coeffs, [])

    if candidates:
        matched = False
        for atom, idx in candidates:
            if len(atom) >= 3:
                v_can_tuple = tuple(int(x) % p for x in atom[2])
                L = max(len(v_tuple), len(v_can_tuple))
                vt = list(v_tuple) + [0]*(L - len(v_tuple))
                vc = list(v_can_tuple) + [0]*(L - len(v_can_tuple))
                vt = [int(K(c)) % p for c in vt]
                vc = [int(K(c)) % p for c in vc]

                if vt == vc:
                    row[idx] = row.get(idx, 0) + 1
                    matched = True
                    break
                elif all((vt[i] - (-vc[i])) % p == 0 for i in range(L)):
                    row[idx] = row.get(idx, 0) - 1
                    matched = True
                    break

        if matched:
            return row

    if require_signed_d2:
        return None

    # Fallback: split into d1 atoms
    for x_elem, mult in roots_data:
        x_int = int(x_elem)
        entry = d1_by_x.get(x_int)
        if entry is None:
            return None
        atom, idx = entry
        y_can = int(atom[2])
        v_at_x = int(v_poly(x_elem)) % p
        if v_at_x == y_can:
            sign = 1
        elif (p - v_at_x) % p == y_can:
            sign = -1
        else:
            return None
        row[idx] = row.get(idx, 0) + sign * int(mult)
        if row[idx] == 0:
            del row[idx]

    return row
get_relation_row.cache = {}

def build_homogeneous_relations_no_rebase(smooth_divs, atom_to_idx, f_p, p, fb_y_cache, f_coeffs,
                                         verbose=True,
                                         use_collision_walks=True,
                                         target_new_relations=500,
                                         max_walk_steps=200000,
                                         avg_walk_len=300,
                                         distinguished_bits=11,
                                         num_walk_workers=None,
                                         max_dp_table_size=200000):
    """
    Build HOMOGENEOUS relation rows (RHS = 0) from smooth divisors.
    Uses C-based collision walks for speed with proper collision detection.
    """

    K = GF(p)
    R = PolynomialRing(K, 'x')

    idx_to_atom = {int(v): k for k, v in atom_to_idx.items()}

    valid_rows = []
    rhs_values = []
    skipped_no_row = 0

    for d in smooth_divs:
        if 'u_coeffs' in d:
            u_poly = R(d['u_coeffs'])
            v_poly = R(d['v_coeffs'])
        elif 's' in d and 'p' in d:
            x = R.gen()
            u_poly = x**2 - K(int(d['s']))*x + K(int(d['p']))
            v_poly = K(int(d['v_1']))*x + K(int(d['v_0']))
        else:
            continue

        row = get_relation_row(
            [u_poly, v_poly],
            atom_to_idx,
            f_p,
            p,
            require_signed_d2=False
        )

        if not row:
            skipped_no_row += 1
            continue

        valid_rows.append({int(k): int(v) for k, v in row.items()})
        rhs_values.append(0)

    if verbose:
        print(f"  [Relations] Built {len(valid_rows)} homogeneous edge-relations (RHS=0)")
        if skipped_no_row > 0:
            print(f"  [Relations] Skipped {skipped_no_row} divisors (not smooth over FB)")

    if not use_collision_walks:
        return valid_rows, rhs_values

    atom_indices = sorted(int(idx) for idx in idx_to_atom.keys())

    if len(atom_indices) < 10:
        if verbose:
            print(f"  [Walks] Only {len(atom_indices)} atoms - skipping walks")
        return valid_rows, rhs_values

    collision_rows, collision_rhs = _run_collision_walks(
        atom_indices, idx_to_atom, target_new_relations,
        distinguished_bits, num_walk_workers, max_walk_steps,
        max_dp_table_size, f_coeffs, verbose
    )

    valid_rows.extend(collision_rows)
    rhs_values.extend(collision_rhs)

    if verbose:
        print(f"  [Relations] Total: {len(valid_rows)} ({len(valid_rows)-len(collision_rows)} edge + {len(collision_rows)} collision)")

    return valid_rows, rhs_values

def _build_random_hash_table(atom_indices, seed=123456):
    """
    Build random hash table for collision walk state mixing.
    Returns pure Python dict with int keys and int values (pickle-safe).
    """
    rng_hash = random.Random(seed)
    return {int(idx): rng_hash.getrandbits(64) for idx in atom_indices}

def _subsample_atoms_for_walks(atom_indices, idx_to_atom, target_walk_atoms=10000, verbose=True):
    """
    Subsample atoms for collision walks.
    Keep all d1 atoms, subsample d2 atoms to hit target count.
    Returns pure Python list of ints (pickle-safe).
    """
    d1_indices = [int(idx) for idx in atom_indices if idx_to_atom[idx][0] == 'd1']
    d2_indices = [int(idx) for idx in atom_indices if idx_to_atom[idx][0] == 'd2']

    d1_count = len(d1_indices)
    d2_count = len(d2_indices)

    if verbose:
        print(f"  [Walks] Factor base composition: {d1_count} d1 atoms, {d2_count} d2 atoms")

    if d1_count + d2_count <= target_walk_atoms:
        if verbose:
            print(f"  [Walks] Using all {d1_count + d2_count} atoms for walks")
        return sorted(d1_indices + d2_indices)

    d2_keep = max(100, target_walk_atoms - d1_count)
    rng_subsample = random.Random(789)
    d2_sampled = rng_subsample.sample(d2_indices, min(d2_keep, d2_count))
    walk_atom_indices = sorted(d1_indices + d2_sampled)

    if verbose:
        print(f"  [Walks] Subsampled for collisions: {d1_count} d1 + {len(d2_sampled)} d2 = {len(walk_atom_indices)} total")

    return walk_atom_indices

def _pad_to_power_of_two(atom_indices, verbose=True):
    """
    Pad atom index list to next power of two for efficient modulo.
    Returns pure Python list of ints (pickle-safe).
    """
    n_atoms = len(atom_indices)
    if (n_atoms & (n_atoms - 1)) == 0:
        return list(atom_indices)

    next_pow2 = 1 << (n_atoms.bit_length())
    padding_needed = next_pow2 - n_atoms
    padded = list(atom_indices) + [atom_indices[0]] * padding_needed

    if verbose:
        print(f"  [Walks] Padded atom table to power-of-two: {len(padded)}")

    return padded

def _collect_collision_relations(manager_queue, manager_dict, num_workers, distinguished_table,
                                valid_rows, rhs_values, target_new_relations,
                                max_dp_table_size, verbose):
    """
    Collect distinguished points from workers and detect collisions.
    Uses Manager queue (pickle-safe).
    """
    new_relations_found = 0
    workers_done = 0
    last_memory_check = time.time()

    while workers_done < num_workers and new_relations_found < target_new_relations:
        if time.time() - last_memory_check > 5.0:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (2**20)
            if verbose:
                print(f"  [MEM] Current RSS: {rss_mb:.1f} MB, DPs in table: {len(distinguished_table)}")
            last_memory_check = time.time()

        try:
            msg = manager_queue.get(timeout=0.5)
        except Exception:
            raise
            if manager_dict.get('stop', False):
                break
            continue

        if msg is None:
            workers_done += 1
            continue

        if len(distinguished_table) >= max_dp_table_size:
            if verbose:
                print(f"  [Walks] DP table full ({max_dp_table_size}), stopping")
            manager_dict['stop'] = True
            workers_done = num_workers
            break

        new_relations_found, workers_done = _process_collision_batch(
            msg, distinguished_table, valid_rows, rhs_values,
            new_relations_found, target_new_relations,
            manager_dict, num_workers, verbose
        )

        if new_relations_found >= target_new_relations or len(distinguished_table) >= max_dp_table_size:
            break

    return new_relations_found

def _cleanup_workers(processes, manager_queue, manager_dict):
    """
    Stop workers and clean up.
    """
    manager_dict['stop'] = True

    while not manager_queue.empty():
        try:
            manager_queue.get_nowait()
        except Exception:
            raise
            break

    for p_proc in processes:
        p_proc.join(timeout=5)
        if p_proc.is_alive():
            p_proc.terminate()
            p_proc.join()

def _process_collision_batch(batch, distinguished_table, valid_rows, rhs_values,
                             new_relations_found, target_new_relations,
                             manager_dict, num_workers, verbose):
    """
    Process a batch of distinguished points, detect collisions, build relations.

    CRITICAL: When two walks reach the same Jacobian element (same jacobian_key),
    they will almost always have DIFFERENT exponent vectors (different paths).
    This is EXACTLY what we want - subtracting the exponents gives us a relation!

    Returns updated new_relations_found and workers_done count.
    """
    workers_done = 0

    for jacobian_key, exp_dict, a_scalar, b_scalar in batch:
        # Check collision
        if jacobian_key in distinguished_table:
            prev_exp, prev_a, prev_b = distinguished_table[jacobian_key]

            # Check for identical exponent vectors (trivial collision 0=0)
            if prev_exp == exp_dict:
                # This catches both "identical walk" (same scalars)
                # AND "path collision" (diff scalars but same counts).
                # In either case, relation is 0=0, so skip.
                # Do NOT remove from table, so we can collide with others later.
                continue

            # Valid relation found!
            # (prev - current)
            rel = dict(prev_exp)
            for idx, e in exp_dict.items():
                rel[idx] = rel.get(idx, 0) - e

            # Remove zeros
            row = {int(k): int(v) for k, v in rel.items() if v != 0}

            if not row:
                # Should be covered by prev_exp == exp_dict, but safety net
                continue

            # Remove consumed entry from table
            del distinguished_table[jacobian_key]

            valid_rows.append(row)
            rhs_values.append(0)
            new_relations_found += 1

            if verbose:
                print(f"  [Walks] 🎉 COLLISION! Found relation #{new_relations_found}")

            if new_relations_found >= target_new_relations:
                manager_dict['stop'] = True
                workers_done = num_workers
                break
        else:
            # Store new DP
            # Use exp_dict directly (batch owns it, created from C arrays)
            distinguished_table[jacobian_key] = (exp_dict, a_scalar, b_scalar)

    return new_relations_found, workers_done

# --- new helper: reconstruct Mumford (u,v) from exp vector -------------------
# ---------------------------------------------------------------------------

# --- modified worker: filter DPs by FB-smoothness before enqueueing -------
# ---------------------------------------------------------------------------

# --- initialize_global_factor_base -----------------------------------------
def initialize_global_factor_base(mapping):
    """
    Initialize module-global FB maps from either:
      - atom_to_idx: { atom_tuple -> idx_int }
      - idx_to_atom: { idx_int -> atom_tuple }

    This is deliberately simple and loud: missing or malformed input raises.
    """
    global _GLOBAL_ATOM_TO_IDX, _GLOBAL_IDX_TO_ATOM

    # basic sanity checks and conversion
    if mapping is None:
        raise AssertionError("initialize_global_factor_base called with None")

    # detect mapping shape
    first_key = next(iter(mapping.keys()))
    # atom keys are typically tuples like ('d1', x, y) or ('d2', u_coeffs, ...)
    if not isinstance(first_key, int):
        # mapping is atom -> idx
        atom_to_idx = dict(mapping)
        idx_to_atom = {int(v): k for k, v in atom_to_idx.items()}
    else:
        # mapping is idx -> atom
        idx_to_atom = {int(k): v for k, v in mapping.items()}
        atom_to_idx = {v: int(k) for k, v in idx_to_atom.items()}

    # final sanity
    assert atom_to_idx, "atom_to_idx is empty"
    assert idx_to_atom, "idx_to_atom is empty"

    _GLOBAL_ATOM_TO_IDX = atom_to_idx
    _GLOBAL_IDX_TO_ATOM = idx_to_atom
# ---------------------------------------------------------------------------

# --- spawn workers: pass idx_to_atom into each worker -----------------------
# ---------------------------------------------------------------------------

# --- worker: lazy-initialize globals from provided idx_to_atom ------------
# ---------------------------------------------------------------------------

def _run_collision_walks(atom_indices, idx_to_atom, target_new_relations,
                        distinguished_bits, num_workers, max_walk_steps,
                         max_dp_table_size, f_coeffs, verbose):
    """
    Run C-based collision walks to generate additional relations.
    Returns (new_relations, new_rhs_values).

    NOTE: idx_to_atom (int -> atom tuple) is required so workers can initialize
    their local globals. This function forwards idx_to_atom into _spawn_walk_workers.
    """
    # Convert to pure Python types
    atom_indices_py = [int(idx) for idx in atom_indices]

    walk_atom_indices = _subsample_atoms_for_walks(atom_indices_py, idx_to_atom,
                                                   target_walk_atoms=10000, verbose=verbose)

    walk_atom_indices = _pad_to_power_of_two(walk_atom_indices, verbose=verbose)
    n_atoms = len(walk_atom_indices)

    if verbose:
        print(f"  [Walks] Starting C-based collision walks with {n_atoms} atoms...")

    random_hash_table = _build_random_hash_table(walk_atom_indices)

    if num_workers is None:
        num_workers = min(4, max(1, cpu_count() - 1))
        if verbose:
            print(f"  [Walks] Auto-selected {num_workers} workers (conservative)")

    target_mask = int((1 << int(distinguished_bits)) - 1)
    expected_steps_per_dp = 1 << int(distinguished_bits)

    if verbose:
        print(f"  [Walks] Using {num_workers} parallel workers")
        print(f"  [Walks] Distinguished bits: {distinguished_bits}")
        print(f"  [Walks] Expected steps per DP: {expected_steps_per_dp}")
        print(f"  [Walks] Max steps per walk: {max_walk_steps}")

    # Use Manager for pickle-safe queue and dict
    manager = Manager()
    manager_queue = manager.Queue(maxsize=num_workers * 2)
    manager_dict = manager.dict()
    manager_dict['stop'] = False

    # Spawn worker processes and pass idx_to_atom so each worker initializes its globals
    processes = _spawn_walk_workers(int(num_workers), walk_atom_indices, random_hash_table,
                                   target_mask, manager_queue, manager_dict, int(max_walk_steps),
                                    idx_to_atom, f_coeffs)

    valid_rows = []
    rhs_values = []
    distinguished_table = {}

    new_relations_found = _collect_collision_relations(
        manager_queue, manager_dict, int(num_workers), distinguished_table,
        valid_rows, rhs_values, int(target_new_relations),
        int(max_dp_table_size), verbose
    )

    _cleanup_workers(processes, manager_queue, manager_dict)
    manager.shutdown()

    if verbose:
        print(f"  [Walks] Completed: {new_relations_found} collision relations")
        if new_relations_found > 0:
            sample_rel = valid_rows[-1]
            exps = sorted(abs(v) for v in sample_rel.values())
            print(f"  [Sample] Last collision: support={len(sample_rel)}, max_exp={max(exps)}, median_exp={exps[len(exps)//2]}")

    return valid_rows, rhs_values

def compute_jacobian_mumford_from_exp(exp_dict, atom_indices_list):
    """
    Construct Mumford (u,v) from an exponent dictionary over factor-base atoms.

    exp_dict: { idx -> exponent }
    atom_indices_list: list of atom indices involved
    p: base field prime (for F_p)
    """
    assert _GLOBAL_IDX_TO_ATOM is not None, "_GLOBAL_IDX_TO_ATOM must be initialized"

    K = GF(FINITE_FIELD)

    # Start with identity divisor
    D = None

    for idx, e in exp_dict.items():
        if e == 0:
            continue

        atom = _GLOBAL_IDX_TO_ATOM[int(idx)]

        # atom is already canonical (d1 or d2)
        A = mumford_from_atom(atom, K)

        if D is None:
            D = e * A
        else:
            D += e * A

    if D is None:
        raise RuntimeError("Empty divisor reconstructed from exponent dictionary")

    return D[0], D[1]   # u(x), v(x)

# --- helper: reconstruct full Jacobian element from exponent vector -------
def reconstruct_jacobian_from_exp(exp_dict, Jclass, R):
    """
    Reconstruct a Jacobian element by summing atom Jacobians.

    exp_dict: { idx -> multiplicity }
    Jclass: Jacobian class/instance with .zero() and addition
    R: polynomial ring over GF(p) (for _atom_to_jac_helper)
    """
    assert _GLOBAL_IDX_TO_ATOM is not None, "_GLOBAL_IDX_TO_ATOM must be initialized"
    Jzero = Jclass.zero()
    J_recon = Jzero
    for idx, mult in exp_dict.items():
        if int(mult) == 0:
            continue
        atom = _GLOBAL_IDX_TO_ATOM.get(int(idx))
        if atom is None:
            raise KeyError(f"missing atom for idx {idx} while reconstructing Jacobian")
        atomJ = _atom_to_jac_helper(atom, Jclass, R)
        J_recon += int(mult) * atomJ
    return J_recon
# -------------------------------------------------------------------------

# --- spawn workers: pass idx_to_atom, p, f_coeffs into each worker --------
def _spawn_walk_workers(num_workers, atom_indices, random_hash_table, target_mask,
                       manager_queue, manager_dict, max_walk_steps, idx_to_atom, f_coeffs):
    """
    Spawn worker processes and pass canonical idx->atom map plus field data
    so each worker can initialize its globals and local curve/jacobian.
    """
    p = FINITE_FIELD
    processes = []
    for worker_id in range(num_workers):
        p_proc = Process(
            target=_c_collision_walk_worker,
            args=(
                int(worker_id),
                atom_indices,
                random_hash_table,
                int(target_mask),
                manager_queue,
                manager_dict,
                int(max_walk_steps),
                idx_to_atom,
                f_coeffs,
            )
        )
        p_proc.daemon = True
        p_proc.start()
        processes.append(p_proc)
    return processes

# --- worker: initialize local globals, local curve/jacobian, enforce smooth DPs
def _c_collision_walk_worker(worker_id, atom_indices_list, random_hash_table,
                             target_mask, result_queue, stop_flag_dict, max_walk_steps,
                             idx_to_atom, f_coeffs, max_terms=256, batch_size=16):
    """
    Worker process for collision walks.

    Args:
      idx_to_atom: { int_idx -> atom_tuple }
      p: prime for GF(p)
      f_coeffs: coefficient list (picklable) used to reconstruct f in GF(p)[x]
    """
    p = FINITE_FIELD

    pid = os.getpid()

    # Initialize canonical FB mappings in this worker (must be first)
    initialize_global_factor_base(idx_to_atom)

    # Set worker-local globals so other helpers that expect them will work
    global _GLOBAL_P, _GLOBAL_F_POLY, _GLOBAL_J
    _GLOBAL_P = int(p)

    # build field, ring, curve, jacobian in this worker
    K = GF(int(p))
    R = PolynomialRing(K, 'x')
    # Reconstruct f_p from f_coeffs using existing helper if present, else simple build
    try:
        # prefer project helper if available
        f_p = sage_poly_from_coeffs(f_coeffs, R)
    except Exception:
        # fallback: build from list of coefficients (assume highest->lowest)
        raise

    _GLOBAL_F_POLY = f_p
    C = HyperellipticCurve(f_p)
    Jclass = C.jacobian()
    _GLOBAL_J = Jclass

    # robust random seed for independent workers
    try:
        seed_bytes = os.urandom(8)
        seed_int = int.from_bytes(seed_bytes, 'big')
    except Exception:
        seed_int = int(time.time() * 1e9) ^ (pid << 16)
    seed_base = seed_int ^ (worker_id * 0x9e3779b97f4a7c15)
    rng = random.Random(seed_base)

    atom_indices = np.ascontiguousarray(atom_indices_list, dtype=np.uint32)
    n_atoms = atom_indices.shape[0]

    rand_table = np.empty(n_atoms, dtype=np.uint64)
    for i, orig_idx in enumerate(atom_indices_list):
        rand_table[i] = random_hash_table[int(orig_idx)]
    rand_table = np.ascontiguousarray(rand_table, dtype=np.uint64)

    exps = np.zeros(n_atoms, dtype=np.uint32)
    batch = []
    walk_counter = 0

    def safe_put(q, item):
        """Put item in queue with retries if full"""
        while True:
            try:
                q.put(item, timeout=0.5)
                break
            except _local_queue.Full:
                time.sleep(0.01)

    while not stop_flag_dict.get('stop', False):
        walk_seed = rng.getrandbits(64) ^ (walk_counter * 0x517cc1b727220a95)
        walk_counter += 1

        a_scalar = rng.getrandbits(32)
        b_scalar = rng.getrandbits(32)

        ret, state, touched, counts = collision_walk_c(
            atom_indices, rand_table, target_mask, max_terms, walk_seed, exps, max_walk_steps
        )

        if ret == 1:
            exp_dict = {int(atom_indices[i]): int(counts[i]) for i in range(len(touched))}

            # Reconstruct Jacobian element from exponents (uses _atom_to_jac_helper)
            J_recon = reconstruct_jacobian_from_exp(exp_dict, Jclass, R)

            # Extract Mumford polys for smoothness test
            u_poly = J_recon[0]
            v_poly = J_recon[1]

            # require signed d2 atoms (matching earlier behavior); if None, discard DP
            row = get_relation_row_cached([u_poly, v_poly], require_signed_d2=True)
            if row is None:
                # DP not FB-smooth under signed d2 policy; discard
                continue

            # compute canonical jacobian key and enqueue
            jacobian_key = compute_jacobian_hash(exp_dict, atom_indices_list)

            batch.append((jacobian_key, exp_dict, a_scalar, b_scalar))

            if len(batch) >= batch_size:
                safe_put(result_queue, batch)
                batch = []

        elif ret == 0:
            continue
        else:
            break

    # Flush remaining batch before exiting
    if batch:
        safe_put(result_queue, batch)

    # Signal completion
    safe_put(result_queue, None)
