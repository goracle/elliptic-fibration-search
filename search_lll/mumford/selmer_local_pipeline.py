# selmer_local_pipeline.py
from sage.all import QQ, GF, Integer
from math import log2
import math
import sys

# you must have these in scope or import them from your code:
# - solve_mumford_mod_p_optimized
# - prefilter_solutions_algebraic
# - canonicalize_and_dedup  (the canonicalizer you already have)

def local_fake_selmer_space_full(f_coeffs, prime, x_residue, const_val=0, debug=False):
    """
    Return mapping fingerprint -> list of residue tuples for prime.
    Fingerprint is (shape, s_mod) where shape in {'double','split','irr'}.
    Residue tuple is (s_mod, p_mod, v0_mod, v1_mod) as ints in 0..p-1.
    """
    sols = solve_mumford_mod_p_optimized(f_coeffs, prime, x_residue, const_val)
    sols = prefilter_solutions_algebraic(sols, prime, f_coeffs)

    mapping = {}
    for s_val, p_val, v0_val, v1_val in sols:
        s_m = int(s_val) % prime
        p_m = int(p_val) % prime
        v0_m = int(v0_val) % prime
        v1_m = int(v1_val) % prime

        Delta = (s_m * s_m - 4 * p_m) % prime
        if Delta == 0:
            shape = "double"
        else:
            leg = pow(Delta, (prime - 1)//2, prime)
            shape = "split" if leg == 1 else "irr"

        # fingerprint uses shape + s_mod (you may add more info if you want)
        fp = (shape, s_m)

        mapping.setdefault(fp, []).append((s_m, p_m, v0_m, v1_m))

    # Defensive: ensure deterministic ordering of reps
    for fp in mapping:
        mapping[fp] = sorted(set(mapping[fp]))
    return mapping


def accumulate_local_selmer(prime_list, f_coeffs, x_residue,
                            min_primes=3, stable_rounds=2, show_progress=True,
                            const_val=0):
    """
    Iterate primes, accumulate and intersect fingerprint sets.
    Stop when intersection size stabilizes for `stable_rounds` consecutive primes
    after at least min_primes have been processed.

    Returns:
      - intersection_fps : sorted list of fingerprints (stabilized)
      - per_prime_maps : dict prime -> mapping (fingerprint -> list of residues)
      - primes_used : list of primes that were actually used (in order)
      - history : list of (prime, current_intersection_size)
    """
    per_prime_maps = {}
    primes_used = []
    intersection = None
    history = []
    stable_count = 0
    last_size = None

    for idx, p in enumerate(prime_list):
        # compute local map
        try:
            mapping = local_fake_selmer_space_full(f_coeffs, p, x_residue, const_val=const_val)
        except Exception as e:
            if show_progress:
                print(f"[selmer] skipping prime {p} due to exception: {e}", file=sys.stderr)
            continue

        if len(mapping) == 0:
            # nothing useful from this prime
            if show_progress:
                print(f"[selmer] prime {p}: no local solutions", file=sys.stderr)
            continue

        per_prime_maps[p] = mapping
        primes_used.append(p)

        keys = set(mapping.keys())
        if intersection is None:
            intersection = keys.copy()
        else:
            intersection &= keys

        cur_size = len(intersection)
        history.append((p, cur_size))
        if show_progress:
            print(f"[selmer] after prime {p} -> intersection size = {cur_size}")

        # stabilization logic
        if last_size is None or cur_size != last_size:
            stable_count = 0
            last_size = cur_size
        else:
            stable_count += 1

        if idx + 1 >= min_primes and stable_count >= stable_rounds:
            if show_progress:
                print(f"[selmer] stabilized after prime {p}. intersection size = {cur_size}")
            break

    if intersection is None:
        intersection = set()

    return sorted(intersection), per_prime_maps, primes_used, history


def _crt_list(values, moduli):
    """
    Simple CRT combine: values[i] mod moduli[i] -> returns x in 0..M-1 where M = prod(moduli).
    All moduli assumed pairwise coprime primes.
    """
    assert len(values) == len(moduli)
    M = 1
    for q in moduli:
        M *= q

    x = 0
    for a_i, p_i in zip(values, moduli):
        m_i = M // p_i
        # inverse of m_i modulo p_i
        inv = pow(m_i, -1, p_i)
        x = (x + (a_i % p_i) * m_i * inv) % M
    return Integer(x), Integer(M)


def crt_combine_representatives(intersection_fps, per_prime_maps, primes_used, pick_rep='first', show_progress=True):
    """
    For each fingerprint in intersection_fps, pick one representative tuple per prime (deterministic),
    CRT-lift each coefficient (s,p,v0,v1) across primes to an integer modulo M.

    Returns: list of dicts:
      { 'fp': fp, 'lifted': (S_int, P_int, V0_int, V1_int), 'modulus': M, 'primes': primes_used }
    """
    results = []
    moduli = primes_used
    for fp in intersection_fps:
        # for each prime select a rep
        vals_s = []
        vals_p = []
        vals_v0 = []
        vals_v1 = []
        ok = True
        for pr in primes_used:
            mapping = per_prime_maps.get(pr, {})
            reps = mapping.get(fp)
            if not reps or len(reps) == 0:
                ok = False
                break
            # pick deterministic representative
            if pick_rep == 'first':
                rep = reps[0]
            else:
                rep = reps[0]
            s_m, p_m, v0_m, v1_m = rep
            vals_s.append(s_m)
            vals_p.append(p_m)
            vals_v0.append(v0_m)
            vals_v1.append(v1_m)
        if not ok:
            continue

        S_int, M = _crt_list(vals_s, moduli)
        P_int, M2 = _crt_list(vals_p, moduli)
        V0_int, M3 = _crt_list(vals_v0, moduli)
        V1_int, M4 = _crt_list(vals_v1, moduli)

        assert M == M2 == M3 == M4
        # center representatives to signed interval for nicer canonicalization
        def center(x, M):
            x = int(x)
            M = int(M)
            if x > M//2:
                x = x - M
            return Integer(x)

        results.append({
            'fp': fp,
            'lifted': (center(S_int, M), center(P_int, M), center(V0_int, M), center(V1_int, M)),
            'modulus': M,
            'primes': list(primes_used)
        })
        if show_progress:
            S_c, P_c, V0_c, V1_c = results[-1]['lifted']
            print(f"[crt] fp={fp} -> lifted (s,p,v0,v1) = ({S_c},{P_c},{V0_c},{V1_c}) mod M={M}")
    return results


def lift_candidates_and_canonicalize(f_coeffs, crt_results, canonicalize_fn, show_progress=True):
    """
    Given CRT-lifted integer candidates, run canonicalize_and_dedup to turn them into QQ divisors.
    Each crt_result dict is expected to have 'lifted': (s,p,v0,v1)

    Returns:
      - canonicalized_list : result of canonicalize_fn on candidates
      - raw_candidates : list of dicts prepared as input for canonicalize (s,p,v_0,v_1)
    """
    raw_candidates = []
    for entry in crt_results:
        s_i, p_i, v0_i, v1_i = entry['lifted']
        # pass as python ints (canonicalizer will coerce to QQ and try small scalings)
        raw_candidates.append({'s': int(s_i), 'p': int(p_i), 'v_0': int(v0_i), 'v_1': int(v1_i)})
    if show_progress:
        print(f"[lift] passing {len(raw_candidates)} CRT-candidates to canonicalize_and_dedup")
    canonicalized = canonicalize_fn(raw_candidates, f_coeffs, show_progress=show_progress)
    return canonicalized, raw_candidates


def compute_fake_selmer_and_lift(prime_list, f_coeffs, x_residue,
                                 canonicalize_fn,
                                 min_primes=3, stable_rounds=2,
                                 const_val=0, show_progress=True):
    """
    High-level pipeline:
      - build per-prime local spaces
      - intersect until stabilized
      - CRT-lift one rep per fingerprint
      - canonicalize lifted candidates

    Returns dict with:
      - 'stabilized_fp_list': list of fingerprints (the fake-Selmer set)
      - 'upper_bound_rank': integer k where |S| = 2^k or None if not a power of two
      - 'crt_results': list of CRT-lift dicts
      - 'canonicalized': list returned by canonicalize_fn
      - 'history': accumulation history
    """
    fps, per_prime_maps, primes_used, history = accumulate_local_selmer(
        prime_list, f_coeffs, x_residue, min_primes=min_primes,
        stable_rounds=stable_rounds, show_progress=show_progress, const_val=const_val)

    if show_progress:
        print(f"[selmer] stabilized fingerprints: {len(fps)} entries")

    # quick upper bound estimate: if |S| is power of two, rank <= log2(|S|)
    S_size = len(fps)
    ub_rank = None
    if S_size > 0 and (S_size & (S_size - 1)) == 0:
        ub_rank = int(math.log2(S_size))
        if show_progress:
            print(f"[selmer] fake-Selmer size {S_size} → rank ≤ {ub_rank}")
    else:
        if show_progress:
            print(f"[selmer] fake-Selmer size {S_size} (not a power of two). Upper bound = floor(log2(|S|)) = {int(math.floor(math.log2(S_size))) if S_size>0 else 'undef'}")

    # CRT-lift
    #crt_results = crt_combine_representatives(fps, per_prime_maps, primes_used, show_progress=show_progress)

    # earlier you had:
    # crt_results = crt_lift_fp_representatives(fps, per_prime_fp_map, primes_used, show_progress=True)

    # new:
    crt_results_expl = crt_lift_fp_representatives_explore(
        intersection, per_prime_maps, primes_used,
        f_coeffs, canonicalize_and_dedup,
        rep_limit=4,             # try up to 4 representatives per prime (tune)
        max_combinations=3000,   # cap total CRT mixes (tune)
        show_progress=True,
        debug_fp_failures=5
    )

    # convert into the shape your code expects:
    crt_results = []
    for entry in crt_results_expl:
        if entry['accepted']:
            # include accepted canonicalized results as CRT-lift outputs
            for c in entry['accepted']:
                # canonicalized entries are dicts like canonicalize returns
                crt_results.append({'fp': entry['fp'], 'lifted_canonical': c, 'tried': entry['tried']})
        else:
            # optionally include raw tried candidates for later inspection
            crt_results.append({'fp': entry['fp'], 'lifted_canonical': None, 'tried': entry['tried'], 'diagnostics': entry['diagnostics']})


    # Canonicalize / dedup
    canonicalized, raw_candidates = lift_candidates_and_canonicalize(f_coeffs, crt_results, canonicalize_fn, show_progress=show_progress)

    return {
        'stabilized_fp_list': fps,
        'upper_bound_rank': ub_rank,
        'crt_results': crt_results,
        'canonicalized': canonicalized,
        'raw_candidates': raw_candidates,
        'primes_used': primes_used,
        'history': history
    }


def local_fake_selmer_space(f_coeffs, prime, x_residue):
    """
    Return a finite set of local Selmer fingerprints at prime p.
    """
    sols = solve_mumford_mod_p_optimized(f_coeffs, prime, x_residue, const_val=0)
    sols = prefilter_solutions_algebraic(sols, prime, f_coeffs)

    space = set()

    for s, pval, v0, v1 in sols:
        Delta = (s*s - 4*pval) % prime

        if Delta == 0:
            shape = ("double", s % prime)
        elif pow(Delta, (prime-1)//2, prime) == 1:
            shape = ("split", s % prime)
        else:
            shape = ("irr", s % prime)

        # v parity fingerprint
        # invariant under v -> -v
        v_sig = (v1*v1 % prime, v0*v0 % prime)

        space.add((shape, v_sig))

    return space


def intersect_local_selmer_spaces(spaces):
    """
    Intersect local fake Selmer spaces across primes.
    """
    if not spaces:
        return set()
    S = spaces[0]
    for T in spaces[1:]:
        S = S.intersection(T)
        if not S:
            break
    return S

