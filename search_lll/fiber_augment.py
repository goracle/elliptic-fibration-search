import os, sys, math, logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from sage.all import GF, PolynomialRing, Integer
from typing import List, Tuple, Dict, Optional, Any

# fiber_augment.py
#
# Augment index calculus factor base and relations using fiber intersection divisors.
#
# For each F_p point x_s on C (y^2 = f_shifted(x)), the intersection locus
# x = -m + x_b gives m_val = x_b - x_s.  Plugging m_val into the fibration RHS
# g(x) = E_rhs_m(x, m=m_val) yields a univariate polynomial over GF(p).  The roots of
# h(x) = f_shifted(x) - g(x) are x-coords where y^2 agrees on C and the fiber.
# div(h) is principal on C, so the canonical-y branch D+ is 2-torsion in J(C).
# Since ell is always a large odd prime, 2-torsion dies in the cofactor and every
# fiber where ALL roots land on C yields a valid homogeneous relation in J[ell].

# Standard library

# Third-party

# Optional typing hints

# ---------------------------------------------------------------------------
# Serialization helpers: convert Sage objects to plain Python for pickling
# ---------------------------------------------------------------------------

def _serialize_e_rhs_m(E_rhs_m):
    """
    Serialize E_rhs_m (element of PolynomialRing(Fm, 'x')) to a list of
    (num_coeffs, den_coeffs) pairs, each a list of ints.
    """
    result = []
    for c in E_rhs_m.list():
        num_coeffs = [int(x) for x in c.numerator().list()]
        den_coeffs = [int(x) for x in c.denominator().list()]
        result.append((num_coeffs, den_coeffs))
    return result

def _serialize_poly(poly):
    """Serialize a GF(p)[x] polynomial as a list of ints."""
    return [int(c) for c in poly.list()]

def _reconstruct_e_rhs_m(serialized, p):
    """Reconstruct E_rhs_m from serialized form in a worker process."""
    K = GF(p)
    Pm = PolynomialRing(K, 'm')
    Fm = Pm.fraction_field()
    Rx = PolynomialRing(Fm, 'x')
    coeffs_fm = []
    for num_c, den_c in serialized:
        num_poly = Pm([K(v) for v in num_c])
        den_poly = Pm([K(v) for v in den_c])
        coeffs_fm.append(Fm(num_poly) / Fm(den_poly))
    return Rx(coeffs_fm)

def _reconstruct_fpoly(serialized, p):
    """Reconstruct a GF(p)[x] polynomial from serialized form in a worker process."""
    K = GF(p)
    Rx = PolynomialRing(K, 'x')
    return Rx([K(v) for v in serialized])

# ---------------------------------------------------------------------------
# Core per-fiber helpers (used by both serial and parallel paths)
# ---------------------------------------------------------------------------

def _tonelli_shanks(n, p):
    if pow(n, (p - 1) // 2, p) != 1:
        raise ValueError(str(n) + " is not a QR mod " + str(p))
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m_ts = s
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    while t != 1:
        i = 1
        tmp = (t * t) % p
        while tmp != 1:
            tmp = (tmp * tmp) % p
            i += 1
        b = pow(c, 1 << (m_ts - i - 1), p)
        m_ts = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    return r

def _eval_fiber_at_m(e_rhs_m_obj, m_val, K, Rx):
    """
    Evaluate e_rhs_m_obj at m=m_val -> polynomial in x over K.
    Returns None if m_val is a pole of any coefficient.
    """
    coeffs = []
    for c in e_rhs_m_obj.list():
        den_val = K(c.denominator()(m_val))
        if den_val == 0:
            return None
        coeffs.append(K(c.numerator()(m_val)) / den_val)
    return Rx(coeffs)

# ---------------------------------------------------------------------------
# FB mutation helpers (main process only)
# ---------------------------------------------------------------------------

def _encode_row(pts, atom_to_idx):
    row = {}
    for x_int, y_can, mult in pts:
        idx = atom_to_idx[('d1', x_int, y_can)]
        row[idx] = row.get(idx, 0) + mult
    return {k: v for k, v in row.items() if v != 0}

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def filter_fiber_relation(pts, atom_to_idx, ell=None):
    """
    Check whether a fiber intersection relation factors completely
    over the existing factor base.

    Args:
        pts: list of (x_int, y_can, mult)
        atom_to_idx: existing FB dictionary
        ell: optional modulus for coefficients

    Returns:
        row dict {idx: coeff} if smooth
        None if the relation introduces new atoms
    """

    row = {}

    for x_int, y_can, mult in pts:
        atom = ('d1', x_int, y_can)

        if atom not in atom_to_idx:
            return None  # reject relation

        idx = atom_to_idx[atom]
        row[idx] = row.get(idx, 0) + mult

    if ell is not None:
        for k in list(row.keys()):
            row[k] %= ell
            if row[k] == 0:
                del row[k]

    return row if row else None

def build_fiber_augmented_relations(
    E_rhs_m,
    f_shifted_fp,
    x_b,
    p,
    atom_to_idx,
    fb_y_cache,
    full_order,
    ell,
    x_coords=None,
    num_workers=None,
    verbose=True
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    if x_coords is None:
        x_coords_list = list(range(p))
    else:
        x_coords_list = [int(x) for x in x_coords]

    e_rhs_m_ser = _serialize_e_rhs_m(E_rhs_m)
    f_shifted_ser = _serialize_poly(f_shifted_fp)
    x_b_int = int(x_b)

    chunk_size = max(1, len(x_coords_list) // num_workers)
    chunks = [x_coords_list[i:i + chunk_size]
              for i in range(0, len(x_coords_list), chunk_size)]

    args_list = [
        (chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx)
        for chunk in chunks
    ]

    if verbose:
        print(f"[fiber_augment] launching {len(chunks)} chunks across {num_workers} workers ({len(x_coords_list)} x_s values total)")

    global_stats = defaultdict(int)
    global_stats['roots_multiplicities'] = Counter()
    global_stats['roots_per_fiber'] = []

    new_rows = []

    with Pool(processes=num_workers) as pool:
        for chunk_result in tqdm(pool.imap(_process_chunk, args_list),
                                 total=len(args_list),
                                 desc="Fiber augment",
                                 unit="chunk"):
            chunk_fibers, chunk_stats = chunk_result

            # merge chunk stats
            for key in ['fibers_total', 'fibers_accepted', 'fibers_all_roots_on_curve', 'fibers_poles_hit',
                        'roots_total', 'roots_on_curve', 'roots_y0', 'roots_in_fb', 'roots_not_in_fb']:
                global_stats[key] += chunk_stats[key]

            global_stats['roots_multiplicities'].update(chunk_stats['roots_multiplicities'])
            global_stats['roots_per_fiber'].extend(chunk_stats['roots_per_fiber'])

            # assign indices and filter rows
            for pts in chunk_fibers:
                row = filter_fiber_relation(pts, atom_to_idx, ell)
                if row is not None:
                    new_rows.append(row)

    if verbose:
        # print summary
        print("\n[fiber_augment] STAT SUMMARY")
        print(f"fibers total               : {global_stats['fibers_total']}")
        print(f"fibers accepted            : {global_stats['fibers_accepted']}")
        print(f"fibers all roots on-curve  : {global_stats['fibers_all_roots_on_curve']}")
        print(f"fibers poles hit           : {global_stats['fibers_poles_hit']}")
        print(f"roots total                : {global_stats['roots_total']}")
        print(f"roots on-curve             : {global_stats['roots_on_curve']}")
        print(f"roots y=0                  : {global_stats['roots_y0']}")
        print(f"roots in FB (partial smooth): {global_stats['roots_in_fb']}")
        print(f"roots not in FB             : {global_stats['roots_not_in_fb']}")
        mult_counter = global_stats['roots_multiplicities']
        if mult_counter:
            print(f"root multiplicities        : min={min(mult_counter)}, max={max(mult_counter)}, counts={dict(mult_counter)}")
        if global_stats['roots_per_fiber']:
            print(f"roots per fiber            : min={min(global_stats['roots_per_fiber'])}, max={
            max(global_stats['roots_per_fiber'])}, avg={sum(global_stats['roots_per_fiber'])/len(global_stats['roots_per_fiber']):.2f}")

        print(f"relations collected        : {len(new_rows)}")

    return new_rows, global_stats

# Defensive canonical-y: returns integer canonical y, 0 for y2==0, or None for off-curve (non-residue).
def canonical_y(y2_int, p):
    """
    Return canonical y for y^2 = y2_int mod p:
      - if y2_int == 0 -> return 0
      - if y2_int is not a quadratic residue -> return None
      - otherwise -> return the smaller of the two square roots
    This avoids calling Tonelli-Shanks on non-residues.
    """
    y2_int = int(y2_int)  # ensure plain int
    if y2_int == 0:
        return 0
    if pow(y2_int, (p - 1) // 2, p) != 1:
        # not a quadratic residue -> off-curve
        return None
    y = _tonelli_shanks(y2_int, p)
    # canonical representative (smallest of the pair)
    return min(y, p - y)

def _process_chunk(args):
    """
    Worker function: process a chunk of x_s values.
    Returns:
        valid_fibers: list of pts_lists — one per valid fiber (only roots that are on-curve).
        chunk_stats: dict of counters / lists for stats collection
    NOTE: args = (x_s_chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx)
    """
    x_s_chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx = args

    K = GF(p)
    Rx = PolynomialRing(K, 'x')
    e_rhs_m_obj = _reconstruct_e_rhs_m(e_rhs_m_ser, p)
    f_shifted = _reconstruct_fpoly(f_shifted_ser, p)
    x_b = K(x_b_int)

    valid_fibers = []

    # initialize chunk-level stats
    chunk_stats = {
        'fibers_total': 0,
        'fibers_accepted': 0,
        'fibers_all_roots_on_curve': 0,
        'fibers_poles_hit': 0,
        'roots_total': 0,            # count of polynomial roots found
        'roots_on_curve': 0,         # number of roots with QR y2 (nonzero)
        'roots_y0': 0,               # number of roots where y2 == 0
        'roots_off_curve': 0,        # number of roots that are non-residues
        'roots_multiplicities': Counter(),
        'roots_per_fiber': [],
        'roots_in_fb': 0,            # partial smoothness: roots that map to existing FB atoms
        'roots_not_in_fb': 0
    }

    for x_s_int in x_s_chunk:
        x_s = K(x_s_int)
        chunk_stats['fibers_total'] += 1

        # quick pre-check for x_s being on curve (as before)
        y2_s = int(f_shifted(x_s))
        if y2_s != 0 and pow(y2_s, (p - 1) // 2, p) != 1:
            # x_s not a valid curve point -> skip
            continue

        m_val = x_b - x_s
        g_x = _eval_fiber_at_m(e_rhs_m_obj, m_val, K, Rx)
        if g_x is None:
            chunk_stats['fibers_poles_hit'] += 1
            continue

        h = f_shifted - g_x
        if h.is_zero():
            continue

        roots_with_mults = h.roots()
        if not roots_with_mults:
            continue

        # First pass: inspect residues to determine whether ALL roots are on-curve.
        all_on_curve = True
        for x_r, mult in roots_with_mults:
            y2 = int(f_shifted(x_r))
            if y2 == 0:
                continue
            if pow(y2, (p - 1) // 2, p) != 1:
                all_on_curve = False
                break

        if all_on_curve:
            chunk_stats['fibers_all_roots_on_curve'] += 1

        # Second pass: build pts list only for roots that are on-curve (y2==0 or QR).
        # Also collect partial smoothness counts: only on-curve roots can be atoms.
        pts = []
        roots_in_fb = 0
        roots_not_in_fb = 0

        for x_r, mult in roots_with_mults:
            x_int = int(x_r)
            y2 = int(f_shifted(x_r))
            chunk_stats['roots_total'] += 1
            chunk_stats['roots_multiplicities'][int(mult)] += 1

            if y2 == 0:
                # point with y=0 (single y)
                y_can = 0
                pts.append((x_int, y_can, int(mult)))
                chunk_stats['roots_y0'] += 1

                atom = ('d1', x_int, 0)
                if atom in atom_to_idx:
                    roots_in_fb += 1
                else:
                    roots_not_in_fb += 1

            else:
                # check whether y2 is QR and only then compute canonical y
                if pow(y2, (p - 1) // 2, p) != 1:
                    # off-curve root: record and skip constructing a (x,y) atom
                    chunk_stats['roots_off_curve'] += 1
                    continue

                # now safe to compute canonical y
                y_can = _canonical_y(y2, p)
                # _canonical_y is defensive, but we already checked QR above; still guard
                if y_can is None:
                    chunk_stats['roots_off_curve'] += 1
                    continue

                pts.append((x_int, y_can, int(mult)))
                chunk_stats['roots_on_curve'] += 1

                atom = ('d1', x_int, y_can)
                if atom in atom_to_idx:
                    roots_in_fb += 1
                else:
                    roots_not_in_fb += 1

        chunk_stats['roots_in_fb'] += roots_in_fb
        chunk_stats['roots_not_in_fb'] += roots_not_in_fb
        chunk_stats['roots_per_fiber'].append(len(roots_with_mults))

        # If every polynomial root yields a curve point, record the fiber as valid
        # (consistent with previous semantics). Otherwise the fiber is rejected as a relation,
        # but its per-root stats are still tracked (partial smoothness, off-curve count, etc.).
        if all_on_curve:
            valid_fibers.append(pts)
            chunk_stats['fibers_accepted'] += 1

    return valid_fibers, chunk_stats
