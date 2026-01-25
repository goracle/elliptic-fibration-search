from ctypes import c_uint32, c_uint64, POINTER, byref
import numpy as np
import ctypes
import os
from ctypes import c_uint32, c_uint64, c_int, POINTER, byref
from multiprocessing import Process, Queue, Event, cpu_count
from queue import Full
import random
from multiprocessing import set_start_method
import psutil
import time

# CRITICAL: Force spawn mode to prevent fork-based RAM explosion
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

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


def get_relation_row(divisor, atom_to_idx, f_p, p,
                                     fb_y_cache=None, require_signed_d2=True):
    """
    Build factor-base row for `divisor` with support for SIGNED degree-2 atoms.

    New expectations for atom_to_idx keys:
      - degree-1 atoms: ('d1', x_int, y_can)  (unchanged)
      - degree-2 atoms (signed Jacobian atoms):
            ('d2', (u_coeff_deg0, u_coeff_deg1, u_coeff_deg2), (v_coeffs...))
        where the u_coeffs tuple gives the monic quadratic coefficients (lowest->highest)
        and v_coeffs is the canonical v polynomial coefficient tuple (lowest->highest).
        v_coeffs must represent the canonical choice of v for that Jacobian element.
        The stored v_coeffs are compared to the reconstructed v_poly to decide sign.

    Behavior:
      - For deg 1 divisors: exactly same as previous function (respecting y_can).
      - For deg 2 divisors:
          * If a matching ('d2', u_coeffs, v_coeffs) entry exists, the row is { idx: +1 } or { idx: -1 } 
            depending on whether v_poly == v_can or v_poly == -v_can (mod p).
          * If no matching signed d2 atom exists:
              - If require_signed_d2==True -> return None (not FB-smooth)
              - If require_signed_d2==False -> FALLBACK to splitting into d1 atoms *like before*.
    Returns: dict {col_idx: multiplicity_signed} or None if not FB-smooth.
    """
    from sage.all import GF, PolynomialRing, ZZ

    p = int(p)
    K = GF(p)

    # get polynomial ring consistent with f_p if available; otherwise make one
    try:
        R = f_p.parent()
        x = R.gen()
    except Exception:
        R = PolynomialRing(K, 'x')
        x = R.gen()

    # helpers to convert polynomials to canonical tuples (lowest->highest), pad to degree
    def poly_to_tuple(poly, deg_expected):
        # poly may be in K[x] or Sage polynomial; coerce coefficients to ints modulo p
        coeffs = [0] * (deg_expected + 1)
        for i, c in enumerate(poly.list()):
            if i <= deg_expected:
                coeffs[i] = int(K(c))
        return tuple(coeffs)

    # Build lookup maps
    d1_by_x = {}      # x_int -> (atom, idx)
    d2_by_u = {}      # u_coeffs_tuple -> list of (atom, idx)  (may hold multiple signed atoms if user provided)
    for atom, idx in atom_to_idx.items():
        if atom[0] == 'd1':
            x_val = int(atom[1])
            if x_val in d1_by_x:
                raise RuntimeError(f"Ambiguous factor base: multiple d1 atoms for x={x_val}")
            d1_by_x[x_val] = (atom, int(idx))
        elif atom[0] == 'd2':
            # expected: ('d2', u_coeffs_tuple, v_coeffs_tuple) or ('d2', u_coeffs_tuple)
            if len(atom) < 2:
                raise RuntimeError("Malformed d2 atom key (need at least u_coeffs tuple).")
            u_key = tuple(int(x) % p for x in atom[1])
            entries = d2_by_u.setdefault(u_key, [])
            entries.append((atom, int(idx)))
        else:
            # ignore other atom types
            continue

    row = {}

    # unify handling for dict or [u,v] style
    # extract u_poly, v_poly, multiplicities if present
    if isinstance(divisor, dict):
        assert 's' in divisor and 'p' in divisor and 'v_0' in divisor and 'v_1' in divisor, \
            "get_relation_row_signed_divisors: divisor dict missing keys"
        s_val = int(divisor['s'])
        p_val = int(divisor['p'])
        v0 = K(int(divisor['v_0']))
        v1 = K(int(divisor['v_1']))
        u_poly = x**2 - K(s_val)*x + K(p_val)
        # build v_poly = v1*x + v0  (consistent with how dict supplies v_0,v_1)
        v_poly = v1 * x + v0
    else:
        try:
            u_poly = divisor[0]
            v_poly = divisor[1]
        except Exception:
            raise RuntimeError("get_relation_row_signed_divisors: unsupported divisor type")

    deg = int(u_poly.degree())
    if deg not in (1, 2):
        return None

    # collect linear factors
    roots_data = u_poly.roots(K)
    if sum(m for _, m in roots_data) != deg:
        return None

    # CASE: degree 1 -> same as previous behavior (require d1 atom)
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

    # CASE: degree 2 -> STRICT signed-d2 handling
    # compute canonical u coefficients tuple (lowest->highest), pad to deg2
    # Ensure monic expectation: if u is monic, u_coeffs will be (u0, u1, 1)
    u_coeffs = poly_to_tuple(u_poly, 2)
    # try to find a matching signed d2 atom
    candidates = d2_by_u.get(u_coeffs, [])

    if candidates:
        # compute v_coeffs tuple for comparison to stored v_can
        # degree of v is < deg(u) so at most 1 -> pad to length 2 for storage consistency
        v_tuple = tuple(int(K(c)) for c in v_poly.list())  # variable length allowed
        # try exact match or negation match against candidate atoms' stored v
        matched = False
        for atom, idx in candidates:
            if len(atom) >= 3:
                v_can_tuple = tuple(int(x) % p for x in atom[2])
                # normalize lengths by padding/truncating
                # compare v_tuple to v_can_tuple or its negation
                # bring both to same length
                L = max(len(v_tuple), len(v_can_tuple))
                vt = list(v_tuple) + [0]*(L - len(v_tuple))
                vc = list(v_can_tuple) + [0]*(L - len(v_can_tuple))
                # reduce mod p
                vt = [int(K(c)) % p for c in vt]
                vc = [int(K(c)) % p for c in vc]
                # check equality or negation
                if vt == vc:
                    row[idx] = row.get(idx, 0) + 1
                    matched = True
                    break
                elif all((vt[i] - (-vc[i])) % p == 0 for i in range(L)):  # vt == -vc
                    row[idx] = row.get(idx, 0) - 1
                    matched = True
                    break
                else:
                    continue
            else:
                # candidate has no v_can stored; skip here (user didn't supply signed v)
                continue
        if matched:
            return row
        # no candidate matched both u and v_can: fall through
    # if we reach here: either no d2 atom for this u, or no signed v available / matched
    if require_signed_d2:
        # be strict: user must supply ('d2', u_coeffs, v_can_coeffs) in atom_to_idx
        return None

    # FALLBACK (unsafe): split into degree-1 atoms (old behavior)
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


def get_relation_row_cached(divisor):
    """
    Worker-safe version: use Mumford v(x) directly and atom-based lookup.

    Returns {col_idx: multiplicity_signed} or None if not FB-smooth / not in FB.
    Raises on unexpected errors (per your loud-failure policy).
    """
    global _GLOBAL_ATOM_TO_IDX, _GLOBAL_P, _GLOBAL_FB_Y_CACHE, _GLOBAL_F_POLY
    assert None, "does this even run anymore?"

    if _GLOBAL_ATOM_TO_IDX is None:
        raise RuntimeError("_GLOBAL_ATOM_TO_IDX not initialized in worker")

    if _GLOBAL_P is None:
        raise RuntimeError("_GLOBAL_P not initialized in worker")

    # Local field objects for evaluation
    K = GF(int(_GLOBAL_P))
    R = PolynomialRing(K, 'x')

    try:
        u_poly, v_poly = divisor[0], divisor[1]
    except Exception as e:
        raise RuntimeError(f"get_relation_row_cached: malformed divisor input: {e}")

    deg = u_poly.degree()
    if deg not in (1, 2):
        return None

    # Ensure u splits completely into linear factors over K
    try:
        roots_data = u_poly.roots(K)
    except Exception as e:
        raise RuntimeError(f"get_relation_row_cached: u_poly.roots() failed: {e}")

    if sum(m for _, m in roots_data) != deg:
        return None

    row = {}

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
                elif pow(y2, (int(_GLOBAL_P) - 1) // 2, int(_GLOBAL_P)) != 1:
                    return None
                else:
                    from .smoothness import tonelli_shanks
                    y_can_tmp = tonelli_shanks(y2, int(_GLOBAL_P))
                    y_can = int(min(y_can_tmp, int(_GLOBAL_P) - y_can_tmp))
            else:
                # no canonical y available -> cannot reliably sign degree-1 atom
                return None

        # evaluate Mumford v at x
        try:
            # v_poly is a Sage polynomial defined over the same base field as u_poly
            y_val = int(v_poly(x_elem)) % int(_GLOBAL_P)
        except Exception:
            raise

        # determine sign relative to canonical y
        if y_can == 0:
            sign = +1
        elif y_val == int(y_can):
            sign = +1
        elif (int(_GLOBAL_P) - y_val) % int(_GLOBAL_P) == int(y_can):
            sign = -1
        else:
            # ambiguous / not matching canonical ±sqrt
            return None

        # construct the atom tuple and lookup index in atom map
        atom = ('d1', int(x_int), int(y_can))
        idx = _GLOBAL_ATOM_TO_IDX.get(atom)
        if idx is None:
            # atom not present in factor base
            return None

        row[idx] = row.get(idx, 0) + int(sign) * int(mult)
        if row[idx] == 0:
            del row[idx]

    return row


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
    from sage.all import GF, PolynomialRing
    F = GF(int(p))
    R = PolynomialRing(F, 'x')
    x = R.gen()

    idx_to_atom = {int(idx): atom for atom, idx in atom_to_idx.items()}

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


def build_homogeneous_relations_no_rebase(smooth_divs, atom_to_idx, f_p, p, fb_y_cache, 
                                         verbose=True,
                                         use_collision_walks=True,
                                         target_new_relations=500,
                                         max_walk_steps=200000,
                                         avg_walk_len=300,
                                         distinguished_bits=15,
                                         num_walk_workers=None,
                                         max_dp_table_size=100000):
    """
    Build HOMOGENEOUS relation rows (RHS = 0) from smooth divisors.
    Now uses C-based collision walks for speed.
    """
    from sage.all import GF, PolynomialRing
    
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
    
    atom_indices = [idx for idx, atom in idx_to_atom.items() if atom[0] == 'd1']
    atom_indices = sorted(int(i) for i in atom_indices)
    
    if len(atom_indices) < 10:
        if verbose:
            print(f"  [Walks] Only {len(atom_indices)} atoms - skipping walks")
        return valid_rows, rhs_values
    
    n_atoms = len(atom_indices)
    if (n_atoms & (n_atoms - 1)) != 0:
        next_pow2 = 1 << (n_atoms.bit_length())
        padding_needed = next_pow2 - n_atoms
        atom_indices = atom_indices + [atom_indices[0]] * padding_needed
        if verbose:
            print(f"  [Walks] Padded atom table to power-of-two: {len(atom_indices)}")
    n_atoms = len(atom_indices)
    
    if verbose:
        print(f"  [Walks] Starting C-based collision walks with {n_atoms} atoms...")
    
    rng_hash = random.Random(123456)
    random_hash_table = {idx: rng_hash.getrandbits(64) for idx in atom_indices}
    
    if num_walk_workers is None:
        num_walk_workers = min(4, max(1, cpu_count() - 1))
        if verbose:
            print(f"  [Walks] Auto-selected {num_walk_workers} workers (conservative)")
    
    target_mask = (1 << distinguished_bits) - 1
    expected_steps_per_dp = 1 << distinguished_bits
    
    if verbose:
        print(f"  [Walks] Using {num_walk_workers} parallel workers")
        print(f"  [Walks] Distinguished bits: {distinguished_bits}")
        print(f"  [Walks] Expected steps per DP: {expected_steps_per_dp}")
        print(f"  [Walks] Max steps per walk: {max_walk_steps}")
    
    out_queue = Queue(maxsize=num_walk_workers)
    stop_event = Event()
    
    processes = []
    for worker_id in range(num_walk_workers):
        p_proc = Process(target=_c_collision_walk_worker, args=(
            worker_id,
            atom_indices,
            random_hash_table,
            target_mask,
            out_queue,
            stop_event,
            max_walk_steps
        ))
        p_proc.daemon = True
        p_proc.start()
        processes.append(p_proc)
    
    new_relations_found = 0
    workers_done = 0
    distinguished_table = {}
    last_memory_check = time.time()
    
    while workers_done < num_walk_workers and new_relations_found < target_new_relations:
        if time.time() - last_memory_check > 5.0:
            rss_mb = psutil.Process(os.getpid()).memory_info().rss / (2**20)
            if verbose:
                print(f"  [MEM] Current RSS: {rss_mb:.1f} MB, DPs in table: {len(distinguished_table)}")
            last_memory_check = time.time()
        
        msg = out_queue.get()
        if msg is None:
            workers_done += 1
            continue
        
        for key, exp_dict in msg:
            exp_copy = dict(exp_dict)
            
            if len(distinguished_table) >= max_dp_table_size:
                if verbose:
                    print(f"  [Walks] DP table full ({max_dp_table_size}), stopping")
                stop_event.set()
                break
            
            if key in distinguished_table:
                prev_exp = distinguished_table.pop(key)
                
                if prev_exp != exp_copy:
                    rel = dict(prev_exp)
                    for idx, e in exp_copy.items():
                        rel[idx] = rel.get(idx, 0) - e
                    
                    row = {int(k): int(v) for k, v in rel.items() if v != 0}
                    if row:
                        valid_rows.append(row)
                        rhs_values.append(0)
                        new_relations_found += 1
                        
                        if verbose and (new_relations_found % 50 == 0):
                            print(f"  [Walks] Found {new_relations_found} collision relations")
                        
                        if new_relations_found >= target_new_relations:
                            stop_event.set()
                            break
            else:
                distinguished_table[key] = exp_copy
        
        if new_relations_found >= target_new_relations:
            break
    
    stop_event.set()
    for p_proc in processes:
        p_proc.join(timeout=5)
        if p_proc.is_alive():
            p_proc.terminate()
            p_proc.join()
    
    if verbose:
        print(f"  [Walks] Completed: {new_relations_found} collision relations")
        print(f"  [Relations] Total: {len(valid_rows)} ({len(valid_rows)-new_relations_found} edge + {new_relations_found} collision)")
    
    return valid_rows, rhs_values


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


def _c_collision_walk_worker(worker_id, atom_indices_list, random_hash_table,
                             target_mask, out_queue, stop_event, max_walk_steps,
                             max_terms=256, batch_size=16):
    rng = random.Random(worker_id * 1337 + 42)

    atom_indices = np.ascontiguousarray(atom_indices_list, dtype=np.uint32)
    n_atoms = atom_indices.shape[0]

    rand_table = np.empty(n_atoms, dtype=np.uint64)
    for i, orig_idx in enumerate(atom_indices_list):
        rand_table[i] = random_hash_table[int(orig_idx)]
    rand_table = np.ascontiguousarray(rand_table, dtype=np.uint64)

    exps = np.zeros(n_atoms, dtype=np.uint32)

    batch = []
    while not stop_event.is_set():
        seed = rng.getrandbits(64)
        ret, state, touched, counts = collision_walk_c(
            atom_indices, rand_table, target_mask, max_terms, seed, exps, max_walk_steps
        )

        if ret == 1:
            exp_dict = {}
            for i in range(len(touched)):
                pos = int(touched[i])
                cnt = int(counts[i])
                actual_idx = int(atom_indices[pos])
                exp_dict[actual_idx] = cnt

            batch.append((state, exp_dict))

            if len(batch) >= batch_size:
                try:
                    out_queue.put_nowait(batch)
                except Full:
                    pass
                batch = []
        elif ret == 0:
            continue
        else:
            break

    try:
        if batch:
            out_queue.put_nowait(batch)
        out_queue.put_nowait(None)
    except Full:
        pass
