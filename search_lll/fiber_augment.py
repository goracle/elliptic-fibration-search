from __future__ import annotations
import os, sys, math, logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from sage.all import GF, PolynomialRing, Integer
from typing import List, Tuple, Dict, Optional, Any, Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field

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

        #m_val = x_b - x_s
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
        #pts = []
        #if len({(x, y) for x, y, _ in pts}) <= 2:
        #    continue
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

def _combine_rows(r1, r2, modulus=None):
    if not r1:
        r1 = {}
    if not r2:
        r2 = {}

    out = dict(r1)
    if modulus is None:
        for k, v in r2.items():
            out[k] = out.get(k, 0) + v
        return {k: v for k, v in out.items() if v != 0}

    m = int(modulus)
    for k, v in r2.items():
        out[k] = (out.get(k, 0) + v) % m
        if out[k] == 0:
            out.pop(k, None)

    return {k: v % m for k, v in out.items() if (v % m) != 0}

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
    verbose=True,
    promote_atom=None,
    lp_state=None,   # NEW: persistent LP cache across calls
):
    """Build fiber-augmented relations and resolve large-prime partials.

    This version does the simple thing:
      1. collect every valid fiber row
      2. send every partial relation to the large-prime resolver
      3. append the resolver's emitted pure FB rows

    It keeps lp_state for persistence across calls, but the old single/pair
    tables are no longer the main path.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Work on a local copy of the FB map.
    atom_to_idx = dict(atom_to_idx)

    promote_atom = _normalize_atom_key(promote_atom)
    if promote_atom is not None:
        _ensure_atom_in_fb(atom_to_idx, promote_atom)
        if verbose:
            print(f"[fiber_augment] promoting atom into FB: {promote_atom}")

    if lp_state is None:
        lp_state = _init_lp_state()

    # Keep a persistent resolver across calls if possible.
    resolver = lp_state.get("resolver")
    if resolver is None:
        resolver = LargePrimeRelationResolver(promote_threshold=50)
        lp_state["resolver"] = resolver

    if x_coords is None:
        x_coords_list = [(i, None) for i in range(len(atom_to_idx))]
    else:
        x_coords_list = [(int(x), None if y is None else int(y)) for x, y in x_coords]

    e_rhs_m_ser = _serialize_e_rhs_m(E_rhs_m)
    f_shifted_ser = _serialize_poly(f_shifted_fp)
    x_b_int = int(x_b)

    if len(x_coords_list) == 0:
        return [], defaultdict(int), lp_state

    chunk_size = max(1, len(x_coords_list) // num_workers)
    chunks = [x_coords_list[i:i + chunk_size] for i in range(0, len(x_coords_list), chunk_size)]

    args_list = [
        (chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx)
        for chunk in chunks
    ]

    if verbose:
        print(f"[fiber_augment] launching {len(chunks)} chunks across {num_workers} workers ({len(x_coords_list)} x_s values total)")

    global_stats = defaultdict(int)
    global_stats['roots_multiplicities'] = Counter()
    global_stats['roots_per_fiber'] = []

    global_stats['partials_seen'] = 0
    global_stats['partials_stored_single'] = 0
    global_stats['partials_stored_pair'] = 0
    global_stats['collisions_single'] = 0
    global_stats['collisions_pair'] = 0
    global_stats['chain_resolutions'] = 0
    global_stats['pure_rows_emitted_from_partials'] = 0
    global_stats['partials_too_many_lp'] = 0

    large_prime_counter = Counter()
    large_primes_total = 0
    new_rows = []

    with Pool(processes=num_workers) as pool:
        for chunk_result in tqdm(pool.imap(_process_chunk, args_list),
                                 total=len(args_list),
                                 desc="Fiber augment",
                                 unit="chunk"):
            chunk_fibers, chunk_stats = chunk_result

            for key in [
                'fibers_total', 'fibers_accepted',
                'fibers_all_roots_on_curve', 'fibers_poles_hit',
                'roots_total', 'roots_on_curve', 'roots_y0',
                'roots_off_curve', 'roots_in_fb', 'roots_not_in_fb'
            ]:
                global_stats[key] += chunk_stats.get(key, 0)

            global_stats['roots_multiplicities'].update(chunk_stats.get('roots_multiplicities', Counter()))
            global_stats['roots_per_fiber'].extend(chunk_stats.get('roots_per_fiber', []))

            for pts in chunk_fibers:
                row, large_primes = filter_fiber_relation(pts, atom_to_idx, ell)

                for (x_int, y_can, mult) in large_primes:
                    large_prime_counter[(x_int, y_can)] += int(mult)
                    large_primes_total += int(mult)

                # Pure FB relation: keep it immediately.
                if row is not None and not large_primes:
                    new_rows.append(row)
                    continue

                # Nothing usable here.
                if row is None and not large_primes:
                    continue

                if row is None:
                    row = {}

                lp_keys = [(int(x), int(y)) for (x, y, _) in large_primes]
                if not lp_keys:
                    continue

                global_stats['partials_seen'] += 1

                # Feed every partial directly into the resolver.
                resolver.add_relation({
                    'fb_vec': row,
                    'lps': tuple(lp_keys),
                    'meta': {
                        'source': 'fiber_augment',
                        'lp_count': len(lp_keys),
                    },
                })

    # Resolve all partials we collected in this call.
    resolved_rows = resolver.resolve()
    if resolved_rows:
        new_rows.extend(resolved_rows)

    # Pull resolver stats back into the caller-visible stats dict.
    res_summary = resolver.summary()
    global_stats['resolver_emitted_rows'] = len(resolved_rows)
    global_stats['resolver_promoted_lps'] = res_summary.get('promoted_count', 0)
    global_stats['resolver_remaining_partials'] = res_summary.get('remaining_partials', 0)
    for k, v in res_summary.get('stats', {}).items():
        global_stats[f'resolver_{k}'] = v

    if verbose:
        print("\n[fiber_augment] STAT SUMMARY")
        print(f"fibers total               : {global_stats['fibers_total']}")
        print(f"fibers accepted            : {global_stats['fibers_accepted']}")
        print(f"fibers all roots on-curve  : {global_stats['fibers_all_roots_on_curve']}")
        print(f"fibers poles hit           : {global_stats['fibers_poles_hit']}")
        print(f"roots total                : {global_stats['roots_total']}")
        print(f"roots on-curve             : {global_stats['roots_on_curve']}")
        print(f"roots off-curve            : {global_stats['roots_off_curve']}")
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

        print("\n[fiber_augment] PARTIALS / RESOLVER")
        print(f"partials seen                 : {global_stats['partials_seen']}")
        print(f"resolver emitted rows         : {global_stats.get('resolver_emitted_rows', 0)}")
        print(f"resolver promoted LPs         : {global_stats.get('resolver_promoted_lps', 0)}")
        print(f"resolver remaining partials    : {global_stats.get('resolver_remaining_partials', 0)}")
        print(f"pure FB rows emitted total    : {len(new_rows)}")

    global_stats['large_prime_counter'] = dict(large_prime_counter)
    global_stats['lp_state'] = lp_state
    global_stats['resolver_summary'] = res_summary
    return new_rows, global_stats, lp_state

# ---------------------------------------------------------------------------
# Hyper-LP relation helpers
# ---------------------------------------------------------------------------

def _init_lp_state():
    """
    Persistent state across calls.

    single: A -> [(row, lp_terms), ...]
    pair:   frozenset({A,B}) -> [(row, lp_terms), ...]
    hyper_records: list of dicts {row, lp_terms}
    hyper_index: lp_key -> set(record_id)
    """
    return {
        "single": defaultdict(list),
        "pair": defaultdict(list),
        "hyper_records": [],
        "hyper_index": defaultdict(set),
    }

def _normalize_atom_key(atom):
    if atom is None:
        return None
    if len(atom) == 2:
        x, y = atom
        return ('d1', int(x), int(y))
    if len(atom) == 3:
        tag, x, y = atom
        return (str(tag), int(x), int(y))
    raise ValueError(f"Bad atom key: {atom!r}")

def _ensure_atom_in_fb(atom_to_idx, atom):
    atom = _normalize_atom_key(atom)
    if atom not in atom_to_idx:
        atom_to_idx[atom] = len(atom_to_idx)
    return atom_to_idx[atom]

def _lp_terms_from_large_primes(large_primes, modulus):
    """
    Convert [(x, y, mult), ...] into { (x,y): coeff mod modulus }.
    Repeated LPs are aggregated and zero coefficients are removed.
    """
    m = int(modulus)
    terms = {}
    for x, y, mult in large_primes:
        key = (int(x), int(y))
        terms[key] = (terms.get(key, 0) + int(mult)) % m
        if terms[key] == 0:
            del terms[key]
    return terms

def _add_lp_terms(a, b, modulus):
    """
    Add two LP-term dicts modulo modulus.
    """
    m = int(modulus)
    out = dict(a) if a else {}
    for k, v in (b or {}).items():
        out[k] = (out.get(k, 0) + int(v)) % m
        if out[k] == 0:
            out.pop(k, None)
    return out

def _candidate_hyper_ids(lp_terms, hyper_index, max_candidates=256):
    """
    Return record IDs that share at least 2 LPs with lp_terms,
    prioritizing the largest overlaps first.
    """
    counts = Counter()
    for lp in lp_terms:
        for rid in hyper_index.get(lp, ()):
            counts[rid] += 1

    cand = [rid for rid, c in counts.items() if c >= 2]
    cand.sort(key=lambda rid: (-counts[rid], rid))
    return cand[:max_candidates]

def _store_hyper_relation(lp_state, row, lp_terms):
    """
    Store an unreduced hyper relation in the persistent pool and index it.
    """
    rid = len(lp_state["hyper_records"])
    rec = {
        "row": dict(row) if row else {},
        "lp_terms": dict(lp_terms),
    }
    lp_state["hyper_records"].append(rec)
    for lp in lp_terms:
        lp_state["hyper_index"][lp].add(rid)
    return rid

def _try_reduce_hyper_relation(row, lp_terms, lp_state, ell, global_stats, max_passes=4):
    """
    Greedily try to reduce a k-LP relation by merging it with existing
    hyper relations sharing at least 2 LPs.

    Returns (row, lp_terms, merged_flag).
    """
    row = dict(row) if row else {}
    lp_terms = dict(lp_terms) if lp_terms else {}
    m = int(ell)

    for _ in range(max_passes):
        if len(lp_terms) <= 2:
            break

        candidates = _candidate_hyper_ids(lp_terms, lp_state["hyper_index"], max_candidates=256)
        if not candidates:
            break

        merged = False
        for rid in candidates:
            rec = lp_state["hyper_records"][rid]
            if rec is None:
                continue

            overlap = set(lp_terms).intersection(rec["lp_terms"])
            if len(overlap) < 2:
                continue

            combined_lp = _add_lp_terms(lp_terms, rec["lp_terms"], m)

            # Only accept if it actually reduces the LP count.
            if len(combined_lp) >= len(lp_terms):
                continue

            combined_row = _combine_rows(row, rec["row"], modulus=m)
            global_stats["hyper_merges"] += 1

            row, lp_terms = combined_row, combined_lp
            merged = True
            break

        if not merged:
            break

    return row, lp_terms, (len(lp_terms) > 2)

def _route_small_lp_relation(row, lp_terms, single_table, pair_table, ell, global_stats, new_rows):
    """
    Handle 0/1/2-LP relations with exact-key tables.
    This is recursive: after a collision, the combined relation is re-routed.
    """
    row = dict(row) if row else {}
    lp_terms = dict(lp_terms) if lp_terms else {}
    m = int(ell)

    # Pure FB row
    if not lp_terms:
        if row:
            new_rows.append(row)
            global_stats["pure_rows_emitted_from_partials"] += 1
        return

    # One-LP relation
    if len(lp_terms) == 1:
        A = next(iter(lp_terms.keys()))

        if single_table.get(A):
            other_row, other_terms = single_table[A].pop()
            global_stats["collisions_single"] += 1

            combined_row = _combine_rows(other_row, row, modulus=m)
            combined_lp = _add_lp_terms(other_terms, lp_terms, m)

            return _route_partial_relation(
                combined_row, combined_lp, single_table, pair_table,
                ell, global_stats, new_rows
            )
        else:
            single_table[A].append((row, lp_terms))
            global_stats["partials_stored_single"] += 1
            return

    # Two-LP relation
    if len(lp_terms) == 2:
        A, B = sorted(lp_terms.keys())
        pair_key = frozenset([A, B])

        if pair_table.get(pair_key):
            other_row, other_terms = pair_table[pair_key].pop()
            global_stats["collisions_pair"] += 1

            combined_row = _combine_rows(other_row, row, modulus=m)
            combined_lp = _add_lp_terms(other_terms, lp_terms, m)

            return _route_partial_relation(
                combined_row, combined_lp, single_table, pair_table,
                ell, global_stats, new_rows
            )

        # Optional chaining through exact single tables
        if single_table.get(A):
            other_row, other_terms = single_table[A].pop()
            global_stats["chain_resolutions"] += 1

            combined_row = _combine_rows(other_row, row, modulus=m)
            combined_lp = _add_lp_terms(other_terms, lp_terms, m)

            return _route_partial_relation(
                combined_row, combined_lp, single_table, pair_table,
                ell, global_stats, new_rows
            )

        if single_table.get(B):
            other_row, other_terms = single_table[B].pop()
            global_stats["chain_resolutions"] += 1

            combined_row = _combine_rows(other_row, row, modulus=m)
            combined_lp = _add_lp_terms(other_terms, lp_terms, m)

            return _route_partial_relation(
                combined_row, combined_lp, single_table, pair_table,
                ell, global_stats, new_rows
            )

        pair_table[pair_key].append((row, lp_terms))
        global_stats["partials_stored_pair"] += 1
        return

    # Anything larger: let the hypergraph reducer try to shrink it.
    return _route_hyper_relation(
        row, lp_terms, single_table, pair_table,
        ell, global_stats, new_rows
    )

def _route_hyper_relation(row, lp_terms, single_table, pair_table, ell, global_stats, new_rows):
    """
    Route k-LP relations (k >= 3) through the hypergraph reducer.
    If reduced to <=2 LPs, send back to the small-LP logic.
    Otherwise store them in the persistent hyper pool.
    """
    if lp_terms is None:
        lp_terms = {}

    # Use a persistent hyper state attached to global_stats if present.
    lp_state = global_stats.setdefault("lp_state", _init_lp_state())

    reduced_row, reduced_lp_terms, still_hyper = _try_reduce_hyper_relation(
        row=row,
        lp_terms=lp_terms,
        lp_state=lp_state,
        ell=ell,
        global_stats=global_stats,
        max_passes=4,
    )

    if not reduced_lp_terms:
        if reduced_row:
            new_rows.append(reduced_row)
            global_stats["pure_rows_emitted_from_partials"] += 1
        return

    if len(reduced_lp_terms) <= 2 and not still_hyper:
        return _route_small_lp_relation(
            reduced_row, reduced_lp_terms,
            single_table, pair_table, ell, global_stats, new_rows
        )

    # Still hyper: store for later overlaps.
    _store_hyper_relation(lp_state, reduced_row, reduced_lp_terms)
    global_stats["partials_stored_hyper"] += 1

def _encode_row(pts, atom_to_idx):
    row = {}
    for x_int, y_can, mult in pts:
        atom = ('d1', int(x_int), int(y_can))
        if atom not in atom_to_idx:
            logger.debug("[_encode_row] atom not in atom_to_idx: %s", atom)
            continue
        idx = atom_to_idx[atom]
        row[idx] = row.get(idx, 0) + int(mult)
    return {k: v for k, v in row.items() if v != 0}

def filter_fiber_relation(pts, atom_to_idx, ell=None):
    if pts is None:
        return None, []

    row = {}
    large_primes = []

    for x_int, y_can, mult in pts:
        atom = ('d1', int(x_int), int(y_can))
        if atom not in atom_to_idx:
            large_primes.append((int(x_int), int(y_can), int(mult)))
            continue
        idx = atom_to_idx[atom]
        row[idx] = row.get(idx, 0) + int(mult)

    if ell is not None and row:
        m = int(ell)
        row = {k: (v % m) for k, v in row.items() if (v % m) != 0}

    return (None if not row else row), large_primes

"""Large-prime relation resolver for index calculus / HECC pipelines.

This module is designed to sit between partial-relation collection and the
final linear algebra stage.

It handles:
  * pure FB relations
  * 1-LP partials by pairing equal LP buckets
  * 2-LP partials by building a graph and extracting cycles
  * hub LP promotion to the factor base
  * reclassification after promotion

The implementation is intentionally generic:
  * factor-base rows are sparse dicts: {col_index: integer_exponent}
  * large primes are hashable labels (usually ints)
  * a relation is any object exposing:
        - fb_vec: dict[int, int]
        - lps: iterable[hashable]
    or a dict with keys 'fb_vec' and 'lps'

You can adapt the extractor functions near the bottom if your in-memory
relation shape differs.
"""

SparseVec = Dict[int, int]
LP = Hashable

# -----------------------------
# Sparse vector helpers
# -----------------------------

def _copy_vec(v: SparseVec) -> SparseVec:
    return dict(v) if v else {}

def vec_add(a: SparseVec, b: SparseVec, scale_b: int = 1) -> SparseVec:
    """Return a + scale_b*b for sparse integer vectors."""
    out = dict(a) if a else {}
    for k, x in (b or {}).items():
        out[k] = out.get(k, 0) + scale_b * x
        if out[k] == 0:
            del out[k]
    return out

def vec_sub(a: SparseVec, b: SparseVec) -> SparseVec:
    return vec_add(a, b, scale_b=-1)

def vec_is_zero(v: SparseVec) -> bool:
    return not v

def vec_norm_l1(v: SparseVec) -> int:
    return sum(abs(x) for x in v.values())

def pairwise(seq: Sequence[Any]) -> Iterable[Tuple[Any, Any]]:
    it = iter(seq)
    while True:
        try:
            a = next(it)
            b = next(it)
        except StopIteration:
            return
        yield a, b

# -----------------------------
# Relation container
# -----------------------------

@dataclass
class PartialRelation:
    fb_vec: SparseVec
    lps: Tuple[LP, ...]
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_any(obj: Any, fb_key: str = "fb_vec", lps_key: str = "lps") -> "PartialRelation":
        if isinstance(obj, PartialRelation):
            return obj
        if isinstance(obj, dict):
            fb_vec = obj.get(fb_key, {})
            lps = obj.get(lps_key, ())
            meta = {k: v for k, v in obj.items() if k not in {fb_key, lps_key}}
            return PartialRelation(dict(fb_vec), tuple(lps), meta)
        fb_vec = getattr(obj, fb_key)
        lps = getattr(obj, lps_key)
        meta = getattr(obj, "meta", {}) or {}
        return PartialRelation(dict(fb_vec), tuple(lps), dict(meta))

    def with_lps(self, lps: Iterable[LP]) -> "PartialRelation":
        return PartialRelation(self.fb_vec, tuple(sorted(lps, key=repr)), dict(self.meta))

# -----------------------------
# DSU with potentials over sparse FB rows
# -----------------------------

class PotentialDSU:
    """Union-find that stores a sparse FB potential from node -> root.

    For each node x, pot[x] means:
        label(x) + pot[x] = label(root(x))

    The exact convention is flexible as long as union/cycle extraction is
    used consistently. We use it only to construct cycle relations.
    """

    def __init__(self) -> None:
        self.parent: Dict[LP, LP] = {}
        self.rank: Dict[LP, int] = {}
        self.pot: Dict[LP, SparseVec] = {}

    def _ensure(self, x: LP) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.pot[x] = {}

    def find(self, x: LP) -> Tuple[LP, SparseVec]:
        self._ensure(x)
        p = self.parent[x]
        if p == x:
            return x, self.pot[x]
        root, root_pot = self.find(p)
        # compress path: pot[x] := pot[x] + pot[parent]
        self.pot[x] = vec_add(self.pot[x], root_pot)
        self.parent[x] = root
        return root, self.pot[x]

    def union(
        self,
        a: LP,
        b: LP,
        edge_label: SparseVec,
    ) -> Optional[SparseVec]:
        """Join a and b using an edge label.

        Returns:
            None if the edge merged two components.
            A pure FB cycle relation (sparse vector) if the edge closed a cycle.
        """
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            # Cycle: (label(a) + pa) - (label(b) + pb) + edge_label = 0
            # Rearranged into a pure FB row.
            return vec_add(vec_sub(pa, pb), edge_label)

        # We need to attach one root below the other and set a potential.
        # Convention: label(ra) + x = label(rb) + y + edge_label.
        # We choose a formula that keeps the invariant consistent:
        #   pot[child_root] = pa - pb + edge_label  (up to orientation)
        # The exact sign does not matter as long as it is used consistently.
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            self.pot[ra] = vec_add(vec_sub(pb, pa), edge_label)
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            self.pot[rb] = vec_add(vec_sub(pa, pb), edge_label)
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
            self.pot[rb] = vec_add(vec_sub(pa, pb), edge_label)
        return None

# -----------------------------
# Main resolver
# -----------------------------

class LargePrimeRelationResolver:
    """Collects and resolves partial relations into pure FB rows."""

    def __init__(
        self,
        promote_threshold: int = 64,
        max_lps_to_keep: int = 2,
        allow_hub_promotion: bool = True,
        fb_column_of_lp: Optional[Callable[[LP], Optional[int]]] = None,
        relation_extractor: Optional[Callable[[Any], PartialRelation]] = None,
    ) -> None:
        self.promote_threshold = promote_threshold
        self.max_lps_to_keep = max_lps_to_keep
        self.allow_hub_promotion = allow_hub_promotion
        self.fb_column_of_lp = fb_column_of_lp
        self.relation_extractor = relation_extractor or (lambda obj: PartialRelation.from_any(obj))

        self.promoted_lps: List[LP] = []
        self.promoted_set = set()

        self._all_partials: List[PartialRelation] = []
        self._emitted_rows: List[SparseVec] = []
        self._stats: Counter = Counter()

    @property
    def emitted_rows(self) -> List[SparseVec]:
        return self._emitted_rows

    @property
    def stats(self) -> Counter:
        return self._stats

    def add_relation(self, rel: Any) -> None:
        pr = self.relation_extractor(rel)
        pr = PartialRelation(_copy_vec(pr.fb_vec), tuple(pr.lps), dict(pr.meta))
        self._all_partials.append(pr)
        self._stats["seen"] += 1
        if len(pr.lps) == 0:
            self._emit_row(pr.fb_vec)
            self._stats["pure_fb_in"] += 1
        else:
            self._stats[f"partial_{len(pr.lps)}lp_in"] += 1

    def _emit_row(self, row: SparseVec) -> None:
        if not row:
            self._stats["zero_rows_dropped"] += 1
            return
        self._emitted_rows.append(row)
        self._stats["rows_emitted"] += 1
        self._stats["row_l1_total"] += vec_norm_l1(row)

    def _promote_hubs(self, partials: List[PartialRelation]) -> List[PartialRelation]:
        if not self.allow_hub_promotion:
            return partials
        freq = Counter()
        for pr in partials:
            for lp in pr.lps:
                freq[lp] += 1

        promoted = [lp for lp, c in freq.items() if c >= self.promote_threshold]
        if not promoted:
            return partials

        new_promotions = [lp for lp in promoted if lp not in self.promoted_set]
        if not new_promotions:
            return partials

        for lp in new_promotions:
            self.promoted_set.add(lp)
            self.promoted_lps.append(lp)
        self._stats["lp_promoted"] += len(new_promotions)

        # Reclassify all partials with these LPs removed.
        reclassified: List[PartialRelation] = []
        for pr in partials:
            lps = tuple(lp for lp in pr.lps if lp not in self.promoted_set)
            reclassified.append(PartialRelation(pr.fb_vec, tuple(sorted(lps, key=repr)), dict(pr.meta)))
        return reclassified

    def _resolve_one_lp(self, partials: List[PartialRelation]) -> List[PartialRelation]:
        buckets: Dict[LP, List[PartialRelation]] = defaultdict(list)
        leftovers: List[PartialRelation] = []

        for pr in partials:
            if len(pr.lps) == 1:
                buckets[pr.lps[0]].append(pr)
            else:
                leftovers.append(pr)

        for lp, rels in buckets.items():
            self._stats["one_lp_buckets"] += 1
            for a, b in pairwise(rels):
                self._emit_row(vec_sub(a.fb_vec, b.fb_vec))
                self._stats["one_lp_pairs_resolved"] += 1
            if len(rels) % 2 == 1:
                leftovers.append(rels[-1])
                self._stats["one_lp_leftover"] += 1

        return leftovers

    def _resolve_two_lp_graph(self, partials: List[PartialRelation]) -> List[PartialRelation]:
        dsu = PotentialDSU()
        leftovers: List[PartialRelation] = []

        for pr in partials:
            if len(pr.lps) != 2:
                leftovers.append(pr)
                continue

            a, b = pr.lps
            cycle = dsu.union(a, b, pr.fb_vec)
            if cycle is None:
                self._stats["two_lp_edges_added"] += 1
            else:
                self._emit_row(cycle)
                self._stats["two_lp_cycles_resolved"] += 1

        return leftovers

    def _promoted_to_fb(self, pr: PartialRelation) -> PartialRelation:
        if not self.promoted_set:
            return pr
        lps = tuple(lp for lp in pr.lps if lp not in self.promoted_set)
        return PartialRelation(pr.fb_vec, tuple(sorted(lps, key=repr)), dict(pr.meta))

    def resolve(self, partials: Optional[Iterable[Any]] = None, *, reprocess_all: bool = True) -> List[SparseVec]:
        """Run the full resolution pipeline.

        Args:
            partials: optional iterable of relations to add before resolving.
            reprocess_all: if True, repeatedly promote hubs and reclassify
                all stored partials until no new promotion happens.

        Returns:
            list of emitted pure FB rows.
        """
        if partials is not None:
            for rel in partials:
                self.add_relation(rel)

        # Work on a mutable copy of all collected partials.
        working = list(self._all_partials)

        # Repeatedly promote hubs and reclassify, because a promotion can turn
        # many 3-LP rows into 2-LP rows and unlock cycles.
        changed = True
        rounds = 0
        while changed:
            rounds += 1
            before_promotions = len(self.promoted_set)

            working = [self._promoted_to_fb(pr) for pr in working]
            working = self._promote_hubs(working)
            working = [self._promoted_to_fb(pr) for pr in working]

            # One-LP pairing first, then two-LP graph resolution.
            working = self._resolve_one_lp(working)
            working = self._resolve_two_lp_graph(working)

            changed = reprocess_all and (len(self.promoted_set) > before_promotions)
            if not changed:
                break
            if rounds > 8:
                # Safety stop: do not loop forever on unstable promotion rules.
                self._stats["promotion_rounds_capped"] += 1
                break

        # Keep only unresolved partials for later inspection.
        self._remaining = working
        self._stats["remaining_partials"] = len(working)
        self._stats["promoted_total"] = len(self.promoted_set)
        return self._emitted_rows

    @property
    def remaining_partials(self) -> List[PartialRelation]:
        return getattr(self, "_remaining", [])

    def summary(self) -> Dict[str, Any]:
        return {
            "promoted_lps": list(self.promoted_lps),
            "promoted_count": len(self.promoted_set),
            "emitted_rows": len(self._emitted_rows),
            "remaining_partials": len(self.remaining_partials),
            "stats": dict(self._stats),
        }

# -----------------------------
# Optional helpers for integration
# -----------------------------

def default_fb_column_of_lp(lp: LP) -> Optional[int]:
    """Fallback mapping from LP label to FB column.

    In many pipelines the LP itself is the x-coordinate or prime label and you
    maintain an external lookup. Override this with your actual mapping.
    """
    return None

def promote_by_frequency(partials: Iterable[Any], threshold: int) -> List[LP]:
    freq = Counter()
    for rel in partials:
        pr = PartialRelation.from_any(rel)
        for lp in pr.lps:
            freq[lp] += 1
    return [lp for lp, c in freq.items() if c >= threshold]

def extract_rows_and_stats(partials: Iterable[Any], promote_threshold: int = 64) -> Tuple[List[SparseVec], Dict[str, Any]]:
    resolver = LargePrimeRelationResolver(promote_threshold=promote_threshold)
    resolver.resolve(partials)
    return resolver.emitted_rows, resolver.summary()

if __name__ == "__main__":
    # Tiny smoke test.
    rels = [
        {"fb_vec": {0: 1, 2: -1}, "lps": (101,)},
        {"fb_vec": {3: 1, 5: -1}, "lps": (101,)},
        {"fb_vec": {1: 1}, "lps": (7, 11)},
        {"fb_vec": {4: -1}, "lps": (7, 13)},
        {"fb_vec": {6: 1}, "lps": (11, 13)},
    ]
    rows, summary = extract_rows_and_stats(rels, promote_threshold=10)
    print("rows:", rows)
    print("summary:", summary)
