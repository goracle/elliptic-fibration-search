import os, sys, numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sage.all import GF, PolynomialRing
from search_lll.fiber_augment import *

"""
fiber_augment_hdf5.py
---------------------
Drop-in instrumented _process_chunk and build_fiber_augmented_relations.
Workers return a 3rd element (hdf5_rows) that the main-process loop collects
and writes to HDF5 after the pool closes -- no cross-fork globals needed.

INSTALL
-------
Add at the bottom of search_lll/fiber_augment.py:

    from search_lll.fiber_augment_hdf5 import _process_chunk                  # noqa: F811
    from search_lll.fiber_augment_hdf5 import build_fiber_augmented_relations  # noqa: F811

ENVIRONMENT
-----------
FIBER_HDF5_PATH   output file path  (default: fiber_run_xb{x_b}.h5)
FIBER_HDF5_OFF    set "1" to skip writing entirely

HDF5 LAYOUT
-----------
/meta/        x_b, p, fb_size
/factor_base/ x_coords, y_coords, atom_index  (d1 atoms only)
/fibers/      x_s, n_roots, n_in_fb, n_lp, accepted
/large_primes/ x_s, lp_x, lp_y, multiplicity

OFFLINE ANALYSIS
----------------
    python fiber_augment_hdf5.py run1.h5 run2.h5 ...
"""

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Instrumented _process_chunk -- returns 3-tuple: (valid_fibers, stats, hdf5_rows)
# hdf5_rows is a list of plain-Python dicts; no Sage objects, safe to pickle.
# ---------------------------------------------------------------------------

def process_chunk(args):
    x_s_chunk, e_rhs_m_ser, f_shifted_ser, x_b_int, p, atom_to_idx = args

    K   = GF(p)
    Rx  = PolynomialRing(K, 'x')
    e_rhs_m_obj = reconstruct_e_rhs_m(e_rhs_m_ser, p)
    f_shifted   = reconstruct_fpoly(f_shifted_ser, p)
    x_b         = K(x_b_int)

    valid_fibers = []
    hdf5_rows    = []

    def canon(x_int, y_val):
        y_int = int(y_val)
        if y_int > p // 2:
            y_int = p - y_int
        return ('d1', int(x_int), y_int)

    chunk_stats = {
        'fibers_total': 0, 'fibers_accepted': 0, 'fibers_all_roots_on_curve': 0,
        'fibers_poles_hit': 0, 'roots_total': 0, 'roots_on_curve': 0,
        'roots_y0': 0, 'roots_off_curve': 0, 'roots_multiplicities': Counter(),
        'roots_per_fiber': [], 'roots_in_fb': 0, 'roots_not_in_fb': 0,
    }

    for x_s_int, _ in x_s_chunk:
        chunk_stats['fibers_total'] += 1
        x_s   = K(x_s_int)
        m_val = x_b - x_s

        g_x = eval_fiber_at_m(e_rhs_m_obj, m_val, K, Rx)
        if g_x is None:
            chunk_stats['fibers_poles_hit'] += 1
            continue

        h = f_shifted - g_x
        if h.is_zero():
            continue

        roots_with_mults = h.roots()
        if not roots_with_mults:
            continue

        pts          = []
        lps_this     = []
        n_in_fb      = 0
        all_on_curve = True

        for x_r, mult in roots_with_mults:
            x_int = int(x_r)
            y2    = int(f_shifted(x_r))
            mult  = int(mult)
            chunk_stats['roots_total'] += 1
            chunk_stats['roots_multiplicities'][mult] += 1

            if y2 == 0:
                y_val = 0
            elif pow(y2, (p - 1) // 2, p) == 1:
                y_val = tonelli_shanks(y2, p)
            else:
                chunk_stats['roots_off_curve'] += 1
                all_on_curve = False
                continue

            atom = canon(x_int, y_val)
            pts.append((x_int, atom[2], mult))

            if y_val == 0:
                chunk_stats['roots_y0'] += 1
            else:
                chunk_stats['roots_on_curve'] += 1

            if atom in atom_to_idx:
                chunk_stats['roots_in_fb'] += 1
                n_in_fb += 1
            else:
                chunk_stats['roots_not_in_fb'] += 1
                lps_this.append((atom[1], atom[2], mult))

        if not pts:
            continue

        chunk_stats['roots_per_fiber'].append(len(pts))

        hdf5_rows.append({
            'x_s':      int(x_s_int),
            'n_roots':  len(pts),
            'n_in_fb':  n_in_fb,
            'n_lp':     len(lps_this),
            'accepted': bool(all_on_curve),
            'lps':      lps_this,
        })

        if all_on_curve:
            chunk_stats['fibers_all_roots_on_curve'] += 1
            valid_fibers.append(pts)
            chunk_stats['fibers_accepted'] += 1

    return valid_fibers, chunk_stats, hdf5_rows

# ---------------------------------------------------------------------------
# Instrumented build_fiber_augmented_relations
# ---------------------------------------------------------------------------

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
    lp_state=None,
):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    atom_to_idx = dict(atom_to_idx)

    promote_atom = normalize_atom_key(promote_atom)
    if promote_atom is not None:
        ensure_atom_in_fb(atom_to_idx, promote_atom)
        if verbose:
            print(f"[fiber_augment] promoting atom into FB: {promote_atom}")

    if lp_state is None:
        lp_state = init_lp_state()

    resolver = lp_state.get("resolver")
    if resolver is None:
        resolver = LargePrimeRelationResolver(promote_threshold=50)
        lp_state["resolver"] = resolver

    if x_coords is None:
        x_coords_list = [(int(atom[1]), None) for atom in atom_to_idx if isinstance(atom, tuple) and atom[0] == 'd1']
    else:
        x_coords_list = [(int(x), None if y is None else int(y)) for x, y in x_coords]

    e_rhs_m_ser   = serialize_e_rhs_m(E_rhs_m)
    f_shifted_ser = serialize_poly(f_shifted_fp)
    x_b_int       = int(x_b)
    p_int         = int(p)

    if len(x_coords_list) == 0:
        return [], defaultdict(int), lp_state

    chunk_size = max(1, len(x_coords_list) // num_workers)
    chunks     = [x_coords_list[i:i + chunk_size] for i in range(0, len(x_coords_list), chunk_size)]
    args_list  = [(ch, e_rhs_m_ser, f_shifted_ser, x_b_int, p_int, atom_to_idx) for ch in chunks]

    if verbose:
        print(f"[fiber_augment] launching {len(chunks)} chunks across {num_workers} workers ({len(x_coords_list)} x_s values total)")

    global_stats = defaultdict(int)
    global_stats['roots_multiplicities']            = Counter()
    global_stats['roots_per_fiber']                 = []
    global_stats['partials_seen']                   = 0
    global_stats['partials_stored_single']          = 0
    global_stats['partials_stored_pair']            = 0
    global_stats['collisions_single']               = 0
    global_stats['collisions_pair']                 = 0
    global_stats['chain_resolutions']               = 0
    global_stats['pure_rows_emitted_from_partials'] = 0
    global_stats['partials_too_many_lp']            = 0

    large_prime_counter = Counter()
    large_primes_total  = 0
    new_rows            = []

    hdf5_enabled    = H5PY_AVAILABLE and os.environ.get('FIBER_HDF5_OFF', '0') != '1'
    hdf5_fiber_rows = []

    with Pool(processes=num_workers) as pool:
        for chunk_result in tqdm(pool.imap(process_chunk, args_list),
                                 total=len(args_list), desc="Fiber augment", unit="chunk"):
            chunk_fibers, chunk_stats, hdf5_rows = chunk_result

            for key in ('fibers_total', 'fibers_accepted', 'fibers_all_roots_on_curve',
                        'fibers_poles_hit', 'roots_total', 'roots_on_curve', 'roots_y0',
                        'roots_off_curve', 'roots_in_fb', 'roots_not_in_fb'):
                global_stats[key] += chunk_stats.get(key, 0)
            global_stats['roots_multiplicities'].update(chunk_stats.get('roots_multiplicities', Counter()))
            global_stats['roots_per_fiber'].extend(chunk_stats.get('roots_per_fiber', []))

            if hdf5_enabled:
                hdf5_fiber_rows.extend(hdf5_rows)

            for pts in chunk_fibers:
                row, large_primes = filter_fiber_relation(pts, atom_to_idx, ell)

                for (x_int, y_can, mult) in large_primes:
                    large_prime_counter[(x_int, y_can)] += int(mult)
                    large_primes_total += int(mult)

                if row is not None and not large_primes:
                    new_rows.append(row)
                    continue

                if row is None and not large_primes:
                    continue

                if row is None:
                    row = {}

                lp_keys = [(int(x), int(y)) for (x, y, _) in large_primes]
                if not lp_keys:
                    continue

                global_stats['partials_seen'] += 1
                resolver.add_relation({
                    'fb_vec': row,
                    'lps': tuple(lp_keys),
                    'meta': {'source': 'fiber_augment', 'lp_count': len(lp_keys)},
                })

    resolved_rows = resolver.resolve()
    if resolved_rows:
        new_rows.extend(resolved_rows)

    res_summary = resolver.summary()
    global_stats['resolver_emitted_rows']       = len(resolved_rows)
    global_stats['resolver_promoted_lps']       = res_summary.get('promoted_count', 0)
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

        mc = global_stats['roots_multiplicities']
        if mc:
            print(f"root multiplicities        : min={min(mc)}, max={max(mc)}, counts={dict(mc)}")
        if global_stats['roots_per_fiber']:
            rpf = global_stats['roots_per_fiber']
            print(f"roots per fiber            : min={min(rpf)}, max={max(rpf)}, avg={sum(rpf)/len(rpf):.2f}")

        print(f"relations collected (pure FB)        : {len(new_rows)}")

        wh = Counter(len(r) for r in new_rows)
        print("\n[fiber_augment] NEW_ROWS weight histogram:")
        for w in sorted(wh):
            print(f"  {w:3d} -> {wh[w]}")

        print("\n[fiber_augment] LARGE PRIME STATS")
        dlp = len(large_prime_counter)
        print(f"large primes total occurrences : {large_primes_total}")
        print(f"distinct large primes          : {dlp}")
        if dlp > 0:
            freqs = list(large_prime_counter.values())
            print(f"max frequency                 : {max(freqs)}")
            print(f"avg frequency                 : {sum(freqs)/len(freqs):.4f}")
            print(f"large primes with collisions  : {sum(1 for v in freqs if v > 1)}")
            hist = Counter(freqs)
            print("frequency histogram:")
            for k in sorted(hist)[:10]:
                print(f"  {k} -> {hist[k]}")

        print("\n[fiber_augment] PARTIALS / RESOLVER")
        print(f"partials seen                 : {global_stats['partials_seen']}")
        print(f"resolver emitted rows         : {global_stats.get('resolver_emitted_rows', 0)}")
        print(f"resolver promoted LPs         : {global_stats.get('resolver_promoted_lps', 0)}")
        print(f"resolver remaining partials    : {global_stats.get('resolver_remaining_partials', 0)}")
        print(f"pure FB rows emitted total    : {len(new_rows)}")

    global_stats['large_prime_counter'] = dict(large_prime_counter)
    global_stats['lp_state']            = lp_state
    global_stats['resolver_summary']    = res_summary

    if hdf5_enabled and hdf5_fiber_rows:
        write_hdf5(hdf5_fiber_rows, atom_to_idx, x_b_int, p_int, verbose=verbose)

    return new_rows, global_stats, lp_state

def write_hdf5(hdf5_fiber_rows, atom_to_idx, x_b_int, p_int, verbose=True):

    path = os.environ.get('FIBER_HDF5_PATH', f'fiber_run_xb{x_b_int}.h5')

    fb_atoms = [
        (int(a[1]), int(a[2]), int(idx))
        for a, idx in atom_to_idx.items()
        if isinstance(a, tuple) and len(a) == 3 and a[0] == 'd1'
    ]

    f_xs = []; f_nr = []; f_nf = []; f_nl = []; f_ac = []
    l_xs = []; l_lx = []; l_ly = []; l_ml = []

    for row in hdf5_fiber_rows:
        xs = row['x_s']
        f_xs.append(xs); f_nr.append(row['n_roots']); f_nf.append(row['n_in_fb'])
        f_nl.append(row['n_lp']); f_ac.append(1 if row['accepted'] else 0)
        for lx, ly, lm in row['lps']:
            l_xs.append(xs); l_lx.append(lx); l_ly.append(ly); l_ml.append(lm)

    if verbose:
        print(f"\n[hdf5] Writing {len(fb_atoms)} FB atoms, {len(f_xs)} fibers, {len(l_xs)} LP entries -> {path}")

    with h5py.File(path, 'w') as hf:
        mg = hf.create_group('meta')
        mg.create_dataset('x_b', data=np.int64(x_b_int))
        mg.create_dataset('p',   data=np.int64(p_int))
        mg.create_dataset('fb_size', data=np.int64(len(fb_atoms)))

        fbg = hf.create_group('factor_base')
        if fb_atoms:
            arr = np.array(fb_atoms, dtype=np.int32)
            fbg.create_dataset('x_coords',   data=arr[:,0], compression='gzip')
            fbg.create_dataset('y_coords',   data=arr[:,1], compression='gzip')
            fbg.create_dataset('atom_index', data=arr[:,2], compression='gzip')
        else:
            for nm in ('x_coords', 'y_coords', 'atom_index'):
                fbg.create_dataset(nm, data=np.array([], dtype=np.int32))

        fg = hf.create_group('fibers')
        if f_xs:
            fg.create_dataset('x_s',      data=np.array(f_xs, dtype=np.int32), compression='gzip')
            fg.create_dataset('n_roots',  data=np.array(f_nr, dtype=np.uint8))
            fg.create_dataset('n_in_fb',  data=np.array(f_nf, dtype=np.uint8))
            fg.create_dataset('n_lp',     data=np.array(f_nl, dtype=np.uint8))
            fg.create_dataset('accepted', data=np.array(f_ac, dtype=np.uint8))
        else:
            for nm in ('x_s', 'n_roots', 'n_in_fb', 'n_lp', 'accepted'):
                fg.create_dataset(nm, data=np.array([], dtype=np.int32))

        lg = hf.create_group('large_primes')
        if l_xs:
            lg.create_dataset('x_s',          data=np.array(l_xs, dtype=np.int32), compression='gzip')
            lg.create_dataset('lp_x',         data=np.array(l_lx, dtype=np.int32), compression='gzip')
            lg.create_dataset('lp_y',         data=np.array(l_ly, dtype=np.int32), compression='gzip')
            lg.create_dataset('multiplicity', data=np.array(l_ml, dtype=np.uint8))
        else:
            for nm in ('x_s', 'lp_x', 'lp_y', 'multiplicity'):
                lg.create_dataset(nm, data=np.array([], dtype=np.int32))

    if verbose:
        print(f"[hdf5] done: {path}")
        if l_xs:
            top = Counter(zip(l_lx, l_ly)).most_common(10)
            print("[hdf5] top-10 LP atoms by fiber count:")
            for (x, y), cnt in top:
                print(f"       ({x:>10d}, {y:>10d})  {cnt}")

# ---------------------------------------------------------------------------
# Offline analysis
# ---------------------------------------------------------------------------

def lp_correlation_report(hdf5_paths):
    print("\n" + "="*70)
    print("LARGE-PRIME CORRELATION ANALYSIS")
    print("="*70)

    run_data = {}
    for path in hdf5_paths:
        if not os.path.exists(path):
            print(f"  [WARN] missing: {path}"); continue
        with h5py.File(path, 'r') as hf:
            x_b   = int(hf['meta/x_b'][()])
            p     = int(hf['meta/p'][()])
            lp_x  = hf['large_primes/lp_x'][:]
            lp_y  = hf['large_primes/lp_y'][:]
            xs_lp = hf['large_primes/x_s'][:]
            mult  = hf['large_primes/multiplicity'][:]

        counter   = Counter()
        fiber_lps = defaultdict(list)
        for i in range(len(lp_x)):
            key = (int(lp_x[i]), int(lp_y[i]))
            counter[key] += int(mult[i])
            fiber_lps[int(xs_lp[i])].append(key)

        run_data[path] = {'x_b': x_b, 'p': p, 'lp_counter': counter, 'fiber_lps': dict(fiber_lps)}

        print(f"\nRun: {os.path.basename(path)}  x_b={x_b}  p={p}")
        print(f"  distinct LPs={len(counter)}  total occurrences={sum(counter.values())}")
        print("  Top-10 LPs:")
        for (x, y), cnt in counter.most_common(10):
            print(f"    ({x:>10d}, {y:>10d})  count={cnt}")
        co = Counter()
        for lps in fiber_lps.values():
            for i in range(len(lps)):
                for j in range(i+1, len(lps)):
                    co[tuple(sorted([lps[i], lps[j]]))] += 1
        if co:
            print("  Top-5 LP co-occurrence pairs in same fiber:")
            for pair, cnt in co.most_common(5):
                print(f"    {pair[0]} x {pair[1]}  count={cnt}")

    if len(run_data) < 2:
        return

    print("\n" + "-"*70)
    print("CROSS-RUN CORRELATION")
    all_lps = set()
    for rd in run_data.values():
        all_lps.update(rd['lp_counter'])
    print(f"Union of distinct LPs: {len(all_lps)}")

    rd_list = list(run_data.values())
    shared  = {lp for lp in all_lps if all(rd['lp_counter'].get(lp,0)>0 for rd in rd_list)}
    print(f"LPs in ALL {len(rd_list)} runs: {len(shared)}")
    if shared:
        by_tot = sorted(shared, key=lambda lp: -sum(rd['lp_counter'].get(lp,0) for rd in rd_list))
        print("Top-10 universally-present LPs:")
        for lp in by_tot[:10]:
            cnts = [rd['lp_counter'].get(lp,0) for rd in rd_list]
            print(f"  {lp}  counts={cnts}")

    try:
        import scipy.stats as st
        if len(rd_list) == 2:
            lps_sorted = sorted(all_lps)
            v0 = np.array([rd_list[0]['lp_counter'].get(lp,0) for lp in lps_sorted], dtype=float)
            v1 = np.array([rd_list[1]['lp_counter'].get(lp,0) for lp in lps_sorted], dtype=float)
            r, pval = st.pearsonr(v0, v1)
            print(f"\nPearson r: {r:.4f}  (p={pval:.3e})")
    except ImportError:
        print("(install scipy for Pearson r)")



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fiber_augment_hdf5.py <file1.h5> [file2.h5 ...]")
        sys.exit(0)
    lp_correlation_report(sys.argv[1:])
