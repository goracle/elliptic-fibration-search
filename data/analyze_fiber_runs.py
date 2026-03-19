import sys, h5py, numpy as np
from collections import Counter, defaultdict

def load_run(path):
    with h5py.File(path, 'r') as f:
        x_b = int(f['meta/x_b'][()])
        p   = int(f['meta/p'][()])

        fb_x   = f['factor_base/x_coords'][:]
        fb_y   = f['factor_base/y_coords'][:]
        fb_idx = f['factor_base/atom_index'][:]

        lp_xs  = f['large_primes/x_s'][:]
        lp_x   = f['large_primes/lp_x'][:]
        lp_y   = f['large_primes/lp_y'][:]
        lp_mult= f['large_primes/multiplicity'][:]

    fb_atoms = set(zip(fb_x.tolist(), fb_y.tolist()))

    # LP frequency: how many times does each (lp_x, lp_y) appear across all fibers
    lp_freq = Counter()
    # which x_s values produced each LP
    lp_to_xs = defaultdict(list)
    for i in range(len(lp_x)):
        key = (int(lp_x[i]), int(lp_y[i]))
        lp_freq[key] += int(lp_mult[i])
        lp_to_xs[key].append(int(lp_xs[i]))

    # which LPs each x_s produced
    xs_to_lps = defaultdict(list)
    for i in range(len(lp_xs)):
        key = (int(lp_x[i]), int(lp_y[i]))
        xs_to_lps[int(lp_xs[i])].append(key)

    return {
        'path': path,
        'x_b': x_b,
        'p': p,
        'fb_atoms': fb_atoms,
        'lp_freq': lp_freq,
        'lp_to_xs': dict(lp_to_xs),
        'xs_to_lps': dict(xs_to_lps),
    }

def print_run_summary(r):
    print(f"  path : {r['path']}")
    print(f"  x_b  : {r['x_b']}")
    print(f"  p    : {r['p']}")
    print(f"  FB atoms (d1)     : {len(r['fb_atoms'])}")
    print(f"  distinct LPs      : {len(r['lp_freq'])}")
    print(f"  total LP occ.     : {sum(r['lp_freq'].values())}")
    freqs = list(r['lp_freq'].values())
    print(f"  LP freq range     : {min(freqs)} .. {max(freqs)}")
    hist = Counter(freqs)
    print(f"  LP freq histogram : {dict(sorted(hist.items()))}")
    print(f"  Top 10 LPs by frequency:")
    for (x, y), cnt in r['lp_freq'].most_common(10):
        n_fibers = len(r['lp_to_xs'][(x, y)])
        print(f"    ({x}, {y})  freq={cnt}  fibers={n_fibers}")

def analyze(path1, path2):
    r0 = load_run(path1)
    r1 = load_run(path2)

    print("=" * 60)
    print("RUN 0")
    print("=" * 60)
    print_run_summary(r0)

    print()
    print("=" * 60)
    print("RUN 1")
    print("=" * 60)
    print_run_summary(r1)

    lps0 = set(r0['lp_freq'].keys())
    lps1 = set(r1['lp_freq'].keys())
    shared = lps0 & lps1

    print()
    print("=" * 60)
    print("CROSS-RUN LP OVERLAP")
    print("=" * 60)
    print(f"LPs in run 0 only   : {len(lps0 - lps1)}")
    print(f"LPs in run 1 only   : {len(lps1 - lps0)}")
    print(f"LPs in both runs    : {len(shared)}")
    print(f"Union               : {len(lps0 | lps1)}")
    if lps0 and lps1:
        jaccard = len(shared) / len(lps0 | lps1)
        print(f"Jaccard similarity  : {jaccard:.4f}")

    if shared:
        print(f"\nTop 20 shared LPs (sorted by sum of freqs across both runs):")
        shared_sorted = sorted(shared, key=lambda k: -(r0['lp_freq'][k] + r1['lp_freq'][k]))
        for lp in shared_sorted[:20]:
            f0 = r0['lp_freq'][lp]
            f1 = r1['lp_freq'][lp]
            xs0 = r0['lp_to_xs'][lp]
            xs1 = r1['lp_to_xs'][lp]
            xs_shared = set(xs0) & set(xs1)
            print(f"  {lp}  freq=({f0},{f1})  x_s overlap={len(xs_shared)}")

    # Which x_s values in run0 produced LPs that ALSO appear in run1
    xs0_producing_shared = set()
    for lp in shared:
        xs0_producing_shared.update(r0['lp_to_xs'][lp])

    xs1_producing_shared = set()
    for lp in shared:
        xs1_producing_shared.update(r1['lp_to_xs'][lp])

    print()
    print("=" * 60)
    print("X_S CORRELATION")
    print("=" * 60)
    all_xs0 = set(r0['xs_to_lps'].keys())
    all_xs1 = set(r1['xs_to_lps'].keys())
    xs_both = all_xs0 & all_xs1
    print(f"x_s values in run 0     : {len(all_xs0)}")
    print(f"x_s values in run 1     : {len(all_xs1)}")
    print(f"x_s in both runs        : {len(xs_both)}")
    print(f"x_s in run0 with shared LP : {len(xs0_producing_shared)}")
    print(f"x_s in run1 with shared LP : {len(xs1_producing_shared)}")

    # For x_s values present in both runs, how often do they produce the same LPs?
    if xs_both:
        same_lp_set = 0
        for xs in xs_both:
            s0 = set(r0['xs_to_lps'].get(xs, []))
            s1 = set(r1['xs_to_lps'].get(xs, []))
            if s0 == s1:
                same_lp_set += 1
        print(f"x_s with identical LP set in both runs: {same_lp_set} / {len(xs_both)}")

    # FB overlap
    print()
    print("=" * 60)
    print("FACTOR BASE OVERLAP")
    print("=" * 60)
    fb0 = r0['fb_atoms']
    fb1 = r1['fb_atoms']
    fb_shared = fb0 & fb1
    print(f"FB atoms run 0      : {len(fb0)}")
    print(f"FB atoms run 1      : {len(fb1)}")
    print(f"FB atoms shared     : {len(fb_shared)}")
    print(f"Jaccard (FB)        : {len(fb_shared)/len(fb0|fb1):.4f}")

    # Are any shared LPs also in the FB of the other run?
    lp_in_fb0 = shared & fb0
    lp_in_fb1 = shared & fb1
    print()
    print(f"Shared LPs that are in FB of run 0: {len(lp_in_fb0)}")
    print(f"Shared LPs that are in FB of run 1: {len(lp_in_fb1)}")
    if lp_in_fb0:
        print("  (these are atoms the FB found in one run but treated as LP in another)")
        for lp in list(lp_in_fb0)[:5]:
            print(f"    {lp}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python analyze_fiber_runs.py file1.h5 file2.h5")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
