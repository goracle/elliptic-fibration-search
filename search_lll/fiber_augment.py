import os, sys, math, logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from sage.all import GF, PolynomialRing, Integer
from typing import List, Tuple, Dict, Optional, Any

logging.getLogger("fiber_augment").setLevel(logging.DEBUG)

# ---------- instrumented fiber_augment helpers ----------
FIBER_AUGMENT_DEBUG = True
logger = logging.getLogger("fiber_augment")
if FIBER_AUGMENT_DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)



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

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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
        x_coords_list = [(i, None) for i in range(p)]
    else:
        x_coords_list = [(int(x), int(y)) for x, y in x_coords]

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

    # ------------------------------------------------------------------
    # Global stats (same as before, plus partial bookkeeping)
    # ------------------------------------------------------------------
    global_stats = defaultdict(int)
    global_stats['roots_multiplicities'] = Counter()
    global_stats['roots_per_fiber'] = []

    # Large-prime tables (in-memory, main process)
    # single-LP: key -> list of FB-rows waiting on this LP
    large_prime_table_single = defaultdict(list)

    # pair-LP: frozenset{lp1,lp2} -> list of FB-rows waiting on this exact unordered pair
    large_prime_table_pair = defaultdict(list)

    # diagnostic counters
    global_stats['partials_seen'] = 0            # number of partial relations encountered
    global_stats['partials_stored_single'] = 0
    global_stats['partials_stored_pair'] = 0
    global_stats['collisions_single'] = 0        # number of single-LP collisions resolved
    global_stats['collisions_pair'] = 0          # number of pair-pair direct resolutions
    global_stats['chain_resolutions'] = 0        # when pair + single resolve to FB
    global_stats['pure_rows_emitted_from_partials'] = 0

    # For large-prime summary (approx same as previous)
    large_prime_counter = Counter()
    large_primes_total = 0

    new_rows = []

    # ------------------------------------------------------------------
    # Main chunk processing loop (workers produce candidate fibers)
    # ------------------------------------------------------------------
    with Pool(processes=num_workers) as pool:
        for chunk_result in tqdm(pool.imap(_process_chunk, args_list),
                                 total=len(args_list),
                                 desc="Fiber augment",
                                 unit="chunk"):
            chunk_fibers, chunk_stats = chunk_result

            # Merge chunk-level stats
            for key in [
                'fibers_total', 'fibers_accepted',
                'fibers_all_roots_on_curve', 'fibers_poles_hit',
                'roots_total', 'roots_on_curve', 'roots_y0',
                'roots_in_fb', 'roots_not_in_fb'
            ]:
                global_stats[key] += chunk_stats[key]

            global_stats['roots_multiplicities'].update(chunk_stats['roots_multiplicities'])
            global_stats['roots_per_fiber'].extend(chunk_stats['roots_per_fiber'])

            # Process each accepted fiber (pts is list of (x_int,y_can,mult))
            for pts in chunk_fibers:
                # classify FB vs large-primes
                row, large_primes = filter_fiber_relation(pts, atom_to_idx, ell)

                # record large-prime frequency summary (for global histogram)
                for (x_int, y_can, mult) in large_primes:
                    key = (x_int, y_can)
                    large_prime_counter[key] += mult
                    large_primes_total += mult

                # If fully smooth right away -> emit
                if row is not None and not large_primes:
                    new_rows.append(row)
                    continue

                # normalize row to empty dict if None
                if row is None:
                    row = {}

                # we only handle partials with up to 2 large primes here
                lp_keys = [ (int(x), int(y)) for (x, y, _) in large_primes ]
                lp_count = len(lp_keys)
                global_stats['partials_seen'] += 1

                if lp_count == 0:
                    # no LP but row was None? skip
                    continue

                # ---------- single-LP case ----------
                if lp_count == 1:
                    A = lp_keys[0]
                    lp_mult_A = int(large_primes[0][2])

                    if large_prime_table_single.get(A):
                        other_row, _ = large_prime_table_single[A].pop()
                        global_stats['collisions_single'] += 1
                        combined = _combine_rows(other_row, row, modulus=int(ell))
                        if combined:
                            new_rows.append(combined)
                            global_stats['pure_rows_emitted_from_partials'] += 1
                    else:
                        large_prime_table_single[A].append((row, lp_mult_A))
                        global_stats['partials_stored_single'] += 1

                # ---------- double-LP case ----------
                elif lp_count == 2:
                    A, B = lp_keys[0], lp_keys[1]
                    lp_mult_A = int(large_primes[0][2])
                    lp_mult_B = int(large_primes[1][2])
                    pair_key = frozenset([A, B])

                    if large_prime_table_pair.get(pair_key):
                        other_row = large_prime_table_pair[pair_key].pop()
                        global_stats['collisions_pair'] += 1
                        combined = _combine_rows(other_row, row, modulus=int(ell))
                        if combined:
                            new_rows.append(combined)
                            global_stats['pure_rows_emitted_from_partials'] += 1
                        continue

                    if large_prime_table_single.get(A):
                        other_row, _ = large_prime_table_single[A].pop()
                        new_partial_row = _combine_rows(other_row, row, modulus=int(ell))
                        if large_prime_table_single.get(B):
                            other2, _ = large_prime_table_single[B].pop()
                            global_stats['chain_resolutions'] += 1
                            combined = _combine_rows(new_partial_row, other2, modulus=int(ell))
                            if combined:
                                new_rows.append(combined)
                                global_stats['pure_rows_emitted_from_partials'] += 1
                        else:
                            large_prime_table_single[B].append((new_partial_row, lp_mult_B))
                            global_stats['partials_stored_single'] += 1
                        continue

                    if large_prime_table_single.get(B):
                        other_row, _ = large_prime_table_single[B].pop()
                        new_partial_row = _combine_rows(other_row, row, modulus=int(ell))
                        if large_prime_table_single.get(A):
                            other2, _ = large_prime_table_single[A].pop()
                            global_stats['chain_resolutions'] += 1
                            combined = _combine_rows(new_partial_row, other2, modulus=int(ell))
                            if combined:
                                new_rows.append(combined)
                                global_stats['pure_rows_emitted_from_partials'] += 1
                        else:
                            large_prime_table_single[A].append((new_partial_row, lp_mult_A))
                            global_stats['partials_stored_single'] += 1
                        continue

                    large_prime_table_pair[pair_key].append(row)
                    global_stats['partials_stored_pair'] += 1

                # ---------- ignore >2 LPs for now ----------
                else:
                    # optionally track them
                    global_stats['partials_too_many_lp'] = global_stats.get('partials_too_many_lp', 0) + 1
                    # don't store them for now

    # ------------------------------------------------------------------
    # Final reporting (verbose + diagnostics)
    # ------------------------------------------------------------------
    if verbose:
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
            avg_roots = sum(global_stats['roots_per_fiber']) / len(global_stats['roots_per_fiber'])
            print(f"roots per fiber            : min={min(global_stats['roots_per_fiber'])}, max={max(global_stats['roots_per_fiber'])}, avg={avg_roots:.2f}")

        print(f"relations collected (pure FB)        : {len(new_rows)}")

        # New diagnostics: weight histogram of new_rows
        weight_hist = Counter()
        sample_rows_by_weight = {}
        for r in new_rows:
            w = len(r)
            weight_hist[w] += 1
            if w not in sample_rows_by_weight and w <= 12:
                sample_rows_by_weight[w] = dict(list(r.items())[:12])

        print("\n[fiber_augment] NEW_ROWS weight histogram (nonzero count -> how many rows):")
        for w in sorted(weight_hist):
            print(f"  {w:3d} -> {weight_hist[w]}")
        print("\n[fiber_augment] NEW_ROWS sample (small weights):")
        for w in sorted(sample_rows_by_weight):
            print(f"  weight {w}: sample row columns (first <=12 entries): {sample_rows_by_weight[w]}")

        # Large prime summary (unchanged)
        print("\n[fiber_augment] LARGE PRIME STATS (sampled during run)")
        distinct_lp = len(large_prime_counter)
        print(f"large primes total occurrences : {large_primes_total}")
        print(f"distinct large primes          : {distinct_lp}")

        if distinct_lp > 0:
            freqs = list(large_prime_counter.values())
            print(f"max frequency                 : {max(freqs)}")
            print(f"avg frequency                 : {sum(freqs)/len(freqs):.4f}")
            num_collisions = sum(1 for v in freqs if v > 1)
            print(f"large primes with collisions  : {num_collisions}")
            hist = Counter(freqs)
            print("frequency histogram (count -> how many primes):")
            for k in sorted(hist)[:10]:
                print(f"  {k} -> {hist[k]}")

        # Partial-storage & collision diagnostics
        print("\n[fiber_augment] PARTIALS / COLLISIONS")
        print(f"partials seen                 : {global_stats['partials_seen']}")
        print(f"partials stored (single)      : {global_stats['partials_stored_single']}")
        print(f"partials stored (pair)        : {global_stats['partials_stored_pair']}")
        print(f"partials with >2 LPs          : {global_stats.get('partials_too_many_lp', 0)}")
        print(f"single-LP collisions resolved : {global_stats['collisions_single']}")
        print(f"pair-pair collisions resolved : {global_stats['collisions_pair']}")
        print(f"chain (pair+single) resolved  : {global_stats['chain_resolutions']}")
        print(f"pure FB rows emitted from partials: {global_stats['pure_rows_emitted_from_partials']}")

        # remaining waiting partial counts
        rem_single = sum(len(v) for v in large_prime_table_single.values())
        rem_pair = sum(len(v) for v in large_prime_table_pair.values())
        print(f"remaining unmatched partials (single) : {rem_single}")
        print(f"remaining unmatched partials (pair)   : {rem_pair}")

    # Return pure rows and stats
    global_stats['large_prime_counter'] = dict(large_prime_counter)
    global_stats['large_prime_table_single'] = dict(large_prime_table_single)
    return new_rows, global_stats

def _encode_row(pts, atom_to_idx):
    """
    Same as before but instrumented: return the d1-indexed sparse row mapping.
    pts: list of (x_int, y_can, mult)
    """
    row = {}
    for x_int, y_can, mult in pts:
        atom = ('d1', int(x_int), int(y_can))
        if atom not in atom_to_idx:
            # Should not raise here; caller expects large-prime behavior.
            # Still record for debugging.
            logger.debug("[_encode_row] atom not in atom_to_idx: %s", atom)
            continue
        idx = atom_to_idx[atom]
        row[idx] = row.get(idx, 0) + int(mult)
    # strip zeros
    row = {k: v for k, v in row.items() if v != 0}
    return row

def filter_fiber_relation(pts, atom_to_idx, ell=None):
    """
    Instrumented version of filter_fiber_relation.

    - Prints diagnostics when a produced row has weight <= 2 (trivial).
    - Prints when modular reduction removes entries.
    - Returns (row_or_None, large_primes_list) same as original.
    """
    if pts is None:
        return None, []

    row = {}
    large_primes = []
    # keep short mapping for debug
    atom_missing = []
    for x_int, y_can, mult in pts:
        x_key = int(x_int)
        y_key = int(y_can)
        multiplicity = int(mult)
        atom = ('d1', x_key, y_key)
        if atom not in atom_to_idx:
            large_primes.append((x_key, y_key, multiplicity))
            atom_missing.append(atom)
            continue
        idx = atom_to_idx[atom]
        row[idx] = row.get(idx, 0) + multiplicity

    # Debug snapshot BEFORE modular reduction
    if logger.isEnabledFor(logging.DEBUG):
        weight_before = len(row)
        logger.debug("[filter_fiber_relation] pts=%s", pts)
        logger.debug("[filter_fiber_relation] mapped_atoms_missing=%d examples=%s",
                     len(atom_missing), atom_missing[:3])
        logger.debug("[filter_fiber_relation] row_before_mod (len=%d) sample=%s",
                     weight_before, dict(list(row.items())[:8]))

    # apply modulus reduction if requested
    if ell is not None:
        mod = int(ell)
        removed_keys = []
        for k in list(row.keys()):
            val = row[k] % mod
            if val == 0:
                removed_keys.append(k)
                del row[k]
            else:
                row[k] = val
        if logger.isEnabledFor(logging.DEBUG) and removed_keys:
            logger.debug("[filter_fiber_relation] modular reduction removed %d entries: %s",
                         len(removed_keys), removed_keys[:8])

    # Instrument when row collapses to trivial size
    if logger.isEnabledFor(logging.DEBUG):
        weight_after = len(row)
        if weight_after <= 2:
            logger.debug("[filter_fiber_relation] TRIVIAL ROW: weight_before=%d weight_after=%d pts=%s",
                         weight_before, weight_after, pts)
            logger.debug("[filter_fiber_relation] resulting row dict: %s", row)

    # normalize empty dict -> None
    row_out = None if not row else row
    return row_out, large_primes

def _combine_rows(r1: Dict[int,int], r2: Dict[int,int], modulus: Optional[int]=None) -> Dict[int,int]:
    """
    Add two sparse row dicts modulo modulus; remove zeros.
    Instrument collisions where combining collapses to trivial or empty.
    """
    if r1 is None:
        r1 = {}
    if r2 is None:
        r2 = {}
    out = dict(r1)  # shallow copy
    for k, v in r2.items():
        if modulus:
            out[k] = (out.get(k, 0) + v) % modulus
        else:
            out[k] = out.get(k, 0) + v
        if modulus and out[k] == 0:
            del out[k]
    if modulus:
        out = {k: (v % modulus) for k, v in out.items() if (v % modulus) != 0}
    else:
        out = {k: v for k, v in out.items() if v != 0}

    if logger.isEnabledFor(logging.DEBUG):
        # If combination produced fewer nonzeros than either input, log
        len1 = len(r1)
        len2 = len(r2)
        lensum = len1 + len2
        if len(out) < min(len1, len2):
            logger.debug("[_combine_rows] combination reduced sparsity: len1=%d len2=%d -> out=%d", len1, len2, len(out))
            logger.debug("[_combine_rows] r1 sample=%s", dict(list(r1.items())[:8]))
            logger.debug("[_combine_rows] r2 sample=%s", dict(list(r2.items())[:8]))
            logger.debug("[_combine_rows] out sample=%s", dict(list(out.items())[:8]))
    return out

def _process_chunk(args):
    """
    Instrumented worker chunk processor. Adds some per-chunk debug prints
    about fiber roots mapping and row weights.
    """
    x_s_chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx = args

    K = GF(p)
    Rx = PolynomialRing(K, 'x')
    e_rhs_m_obj = _reconstruct_e_rhs_m(e_rhs_m_ser, p)
    f_shifted = _reconstruct_fpoly(f_shifted_ser, p)
    x_b = K(x_b_int)

    valid_fibers = []

    chunk_stats = {
        'fibers_total': 0,
        'fibers_accepted': 0,
        'fibers_all_roots_on_curve': 0,
        'fibers_poles_hit': 0,
        'roots_total': 0,
        'roots_on_curve': 0,
        'roots_y0': 0,
        'roots_off_curve': 0,
        'roots_multiplicities': Counter(),
        'roots_per_fiber': [],
        'roots_in_fb': 0,
        'roots_not_in_fb': 0
    }

    for x_s_int, y_s_known in x_s_chunk:
        x_s = K(x_s_int)
        chunk_stats['fibers_total'] += 1

        # compute y_can_s
        if y_s_known is not None:
            y_can_s = int(y_s_known)
        else:
            y2_s = int(f_shifted(x_s))
            if y2_s != 0 and pow(y2_s, (p - 1) // 2, p) != 1:
                continue
            y_can_s = canonical_y(y2_s, p)
            if y_can_s is None:
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

        seed_atom = ('d1', x_s_int, y_can_s)
        pts = [(x_s_int, y_can_s, 1)]
        roots_in_fb = 1 if seed_atom in atom_to_idx else 0
        roots_not_in_fb = 0 if seed_atom in atom_to_idx else 1

        # collect pts
        for x_r, mult in roots_with_mults:
            x_int = int(x_r)
            y2 = int(f_shifted(x_r))
            chunk_stats['roots_total'] += 1
            chunk_stats['roots_multiplicities'][int(mult)] += 1

            if y2 == 0:
                y_can = 0
                pts.append((x_int, y_can, int(mult)))
                chunk_stats['roots_y0'] += 1
                atom = ('d1', x_int, 0)
                if atom in atom_to_idx:
                    roots_in_fb += 1
                else:
                    roots_not_in_fb += 1
            else:
                if pow(y2, (p - 1) // 2, p) != 1:
                    chunk_stats['roots_off_curve'] += 1
                    continue
                y_can = canonical_y(y2, p)
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

        # DEBUG: sample a few fibers for inspection
        if logger.isEnabledFor(logging.DEBUG) and (chunk_stats['fibers_total'] % 256 == 0):
            logger.debug("[_process_chunk] sample fiber pts: %s", pts[:12])

        if all_on_curve:
            valid_fibers.append(pts)
            chunk_stats['fibers_accepted'] += 1

    # small per-chunk debug summary
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[_process_chunk] chunk_stats sample: fibers_total=%d accepted=%d roots_total=%d",
                     chunk_stats['fibers_total'], chunk_stats['fibers_accepted'], chunk_stats['roots_total'])

    return valid_fibers, chunk_stats
