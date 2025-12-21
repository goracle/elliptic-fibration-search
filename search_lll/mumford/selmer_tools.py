# === BEGIN: fake-selmer utilities ===
from sage.all import QQ, Integer
import math
import sys

def _fingerprint_from_sol_tuple(sol, p):
    """
    sol is a tuple (s_val, p_val, v0_val, v1_val) with integers mod p.
    Returns conservative fingerprint (shape, s_mod).
    """
    s_m = int(sol[0]) % p
    p_m = int(sol[1]) % p
    Delta = (s_m * s_m - 4 * p_m) % p
    if Delta == 0:
        shape = "double"
    else:
        leg = pow(Delta, (p - 1)//2, p)
        shape = "split" if leg == 1 else "irr"
    return (shape, s_m)

def build_per_prime_fp_map_from_prime_data(prime_data, f_coeffs, solver_fn, prefilter_fn, debug=False):
    """
    Given prime_data: dict prime -> list_of_local_solutions_for_that_prime
    (this matches the structure you already build in residue grouping),
    return a map prime -> (set of fingerprints) and map prime -> fingerprint->list(reps).
    Uses solver_fn/prefilter_fn only if needed; but accepts that prime_data already contains lists.
    """
    per_prime_fpset = {}
    per_prime_fp_map = {}
    for p, sols in prime_data.items():
        # sols expected to be list of (s,p,v0,v1) ints or small ints
        fps = set()
        fp_map = {}
        for sol in sols:
            try:
                fp = _fingerprint_from_sol_tuple(sol, p)
            except Exception:
                # skip malformed entries
                continue
            fps.add(fp)
            fp_map.setdefault(fp, []).append(tuple(int(x) % p for x in sol))
        # canonicalize deterministic order
        for k in list(fp_map.keys()):
            fp_map[k] = sorted(set(fp_map[k]))
        per_prime_fpset[p] = fps
        per_prime_fp_map[p] = fp_map
    return per_prime_fpset, per_prime_fp_map

def accumulate_and_stabilize_intersection(prime_list, per_prime_fpset,
                                          min_primes=3, stable_rounds=2, show_progress=True):
    """
    Walk the primes in prime_list order, intersecting their fingerprint sets.
    Stop after at least min_primes and when intersection size is unchanged for stable_rounds iterations.
    Returns (stabilized_set, history_list) where history_list contains (prime, size) pairs.
    """
    intersection = None
    history = []
    stable_count = 0
    last_size = None
    used_primes = []
    for idx, p in enumerate(prime_list):
        fps = per_prime_fpset.get(p, set())
        if not fps:
            # nothing useful at this prime; skip
            if show_progress:
                print(f"[selmer] prime {p} had no fps; skipping", file=sys.stderr)
            continue
        used_primes.append(p)
        if intersection is None:
            intersection = set(fps)
        else:
            intersection &= fps

        cur_size = len(intersection)
        history.append((p, cur_size))
        if show_progress:
            print(f"[selmer] after prime {p} -> intersection size = {cur_size}")

        if last_size is None or cur_size != last_size:
            stable_count = 0
            last_size = cur_size
        else:
            stable_count += 1

        if idx + 1 >= min_primes and stable_count >= stable_rounds:
            if show_progress:
                print(f"[selmer] intersection stabilized after prime {p} (size={cur_size})")
            break

    if intersection is None:
        intersection = set()
    return intersection, history, used_primes


def crt_lift_fp_representatives(intersection_fps, per_prime_fp_map, primes_used, show_progress=True):
    """
    For each fingerprint in intersection_fps pick deterministic one representative per prime (first),
    CRT-lift each coefficient (s,p,v0,v1) across primes to an integer modulo M.
    Returns list of dicts {'fp': fp, 'lifted': (s,p,v0,v1), 'modulus': M, 'primes': primes_used}
    """
    results = []
    for fp in sorted(intersection_fps):
        vals_s = []; vals_p = []; vals_v0 = []; vals_v1 = []
        ok = True
        for pr in primes_used:
            mapping = per_prime_fp_map.get(pr, {})
            reps = mapping.get(fp)
            if not reps:
                ok = False
                break
            rep = reps[0]  # deterministic pick
            s_m, p_m, v0_m, v1_m = rep
            vals_s.append(int(s_m)); vals_p.append(int(p_m)); vals_v0.append(int(v0_m)); vals_v1.append(int(v1_m))
        if not ok:
            continue
        S_int, M = _crt_list(vals_s, primes_used)
        P_int, M2 = _crt_list(vals_p, primes_used)
        V0_int, M3 = _crt_list(vals_v0, primes_used)
        V1_int, M4 = _crt_list(vals_v1, primes_used)
        assert M == M2 == M3 == M4
        # center representatives
        def center(x, M):
            x = int(x); M = int(M)
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
            print(f"[crt] fp={fp} -> lifted (s,p,v0,v1)=({S_c},{P_c},{V0_c},{V1_c}) mod M={M}")
    return results

def compute_fake_selmer_upper_bound_and_lift(prime_list, prime_data, f_coeffs,
                                             canonicalize_fn,
                                             min_primes=3, stable_rounds=2, show_progress=True):
    """
    High-level helper you can call from the reconstruction driver:
      - prime_list: ordered list of primes to try
      - prime_data: dict prime -> list of local (s,p,v0,v1) solutions (this is what your residue stage builds)
      - f_coeffs: curve polynomial coeffs (HIGH->LOW)
      - canonicalize_fn: your canonicalize_and_dedup function
    Returns:
      dict with stabilized_set, upper_bound_rank, crt_results, canonicalized, primes_used, history
    """
    per_prime_fpset, per_prime_fp_map = build_per_prime_fp_map_from_prime_data(prime_data, f_coeffs, None, None)
    intersection, history, primes_used = accumulate_and_stabilize_intersection(prime_list, per_prime_fpset,
                                                                               min_primes=min_primes,
                                                                               stable_rounds=stable_rounds,
                                                                               show_progress=show_progress)
    S_size = len(intersection)
    if show_progress:
        print(f"[selmer] stabilized fake-Selmer size = {S_size}")

    ub_rank = None
    if S_size > 0 and (S_size & (S_size - 1)) == 0:
        ub_rank = int(math.log2(S_size))
        if show_progress:
            print(f"[selmer] fake-Selmer suggests rank ≤ {ub_rank}")
    else:
        if show_progress:
            if S_size > 0:
                print(f"[selmer] fake-Selmer size {S_size} (not power of two); floor(log2) = {int(math.floor(math.log2(S_size)))}")
            else:
                print(f"[selmer] fake-Selmer empty")

    # CRT-lift representatives and canonicalize the integer candidates
    crt_results = crt_lift_fp_representatives(intersection, per_prime_fp_map, primes_used, show_progress=show_progress)
    raw_candidates = []
    for entry in crt_results:
        s_i, p_i, v0_i, v1_i = entry['lifted']
        raw_candidates.append({'s': int(s_i), 'p': int(p_i), 'v_0': int(v0_i), 'v_1': int(v1_i)})
    if show_progress:
        print(f"[selmer] passing {len(raw_candidates)} lifted candidates to canonicalize_and_dedup")
    canonicalized = canonicalize_fn(raw_candidates, f_coeffs, show_progress=show_progress)
    return {
        'stabilized_set': sorted(list(intersection)),
        'upper_bound_rank': ub_rank,
        'crt_results': crt_results,
        'canonicalized': canonicalized,
        'primes_used': primes_used,
        'history': history
    }
# === END: fake-selmer utilities ===


# === BEGIN improved_crt_lift_and_diagnostics ===
from itertools import product, islice
from sage.all import Integer


def crt_lift_fp_representatives_explore(intersection_fps, per_prime_fp_map,
                                        primes_used, f_coeffs, canonicalize_fn,
                                        rep_limit=3, max_combinations=2000,
                                        show_progress=True, debug_fp_failures=3):
    """
    For each fp in intersection_fps, try up to `rep_limit` representatives per prime
    and test up to `max_combinations` CRT mixes. Use canonicalize_fn to validate.
    Returns list of dicts:
      { 'fp': fp, 'accepted': [canonicalized entries], 'tried': tried_count, 'diagnostics': {...} }
    """
    results = []
    primes = list(primes_used)

    for fp in sorted(intersection_fps):
        # collect reps-per-prime for this fp
        reps_per_prime = []
        missing = False
        for p in primes:
            fp_map = per_prime_fp_map.get(p, {})
            reps = fp_map.get(fp, [])
            if not reps:
                missing = True
                break
            # pick smallest reps by lexicographic absolute size heuristics (deterministic)
            def rep_key(rep):
                return tuple(abs(int(x)) for x in rep)
            reps_sorted = sorted(set(reps), key=rep_key)[:rep_limit]
            reps_per_prime.append(reps_sorted)

        if missing:
            if show_progress:
                print(f"[crt-explore] fp={fp} missing reps at some primes -> skipping")
            results.append({'fp': fp, 'accepted': [], 'tried': 0, 'diagnostics': {'reason': 'missing_reps'}})
            continue

        # estimate total combos and limit
        total_possible = 1
        for L in reps_per_prime:
            total_possible *= len(L)
        to_try = total_possible if total_possible <= max_combinations else max_combinations

        if show_progress:
            print(f"[crt-explore] fp={fp} -> trying up to {to_try} combinations (possible={total_possible})")

        accepted = []
        tried = 0
        diagnostics = {'samples': []}

        # deterministic iteration: lexicographic product but truncated
        combo_iter = islice(product(*reps_per_prime), to_try)
        for combo in combo_iter:
            tried += 1
            # combo is a tuple of reps (s_mod,p_mod,v0_mod,v1_mod) per prime in same order
            vals_s = [int(rep[0]) for rep in combo]
            vals_p = [int(rep[1]) for rep in combo]
            vals_v0 = [int(rep[2]) for rep in combo]
            vals_v1 = [int(rep[3]) for rep in combo]

            S_int, M = _crt_list(vals_s, primes)
            P_int, M2 = _crt_list(vals_p, primes)
            V0_int, M3 = _crt_list(vals_v0, primes)
            V1_int, M4 = _crt_list(vals_v1, primes)
            assert M == M2 == M3 == M4

            s_center = _center_rep(S_int, M)
            p_center = _center_rep(P_int, M)
            v0_center = _center_rep(V0_int, M)
            v1_center = _center_rep(V1_int, M)

            candidate = {'s': int(s_center), 'p': int(p_center), 'v_0': int(v0_center), 'v_1': int(v1_center)}
            # quick algebraic check before canonicalize: this will be cheap and give diagnostic info
            try:
                ok_alg = verify_mumford_pair(f_coeffs, candidate['s'], candidate['p'],
                                             candidate['v_0'], candidate['v_1'], modulus=None, debug_first_failure=False)
            except Exception:
                ok_alg = False

            # call canonicalizer on this single candidate
            try:
                canon_res = canonicalize_fn([candidate], f_coeffs, show_progress=False)
            except Exception as e:
                canon_res = []
                # still collect diagnostics
                if len(diagnostics['samples']) < debug_fp_failures:
                    diagnostics['samples'].append({'candidate': candidate, 'ok_alg': ok_alg, 'canon_exc': str(e)})
            else:
                if canon_res:
                    # success: collect and stop exploring further combos for this fp
                    accepted.extend(canon_res)
                    if show_progress:
                        print(f"[crt-explore] fp={fp} accepted lift after {tried} tries -> {canon_res}")
                    break
                else:
                    # record sample failures for diagnostics (limited)
                    if len(diagnostics['samples']) < debug_fp_failures:
                        diagnostics['samples'].append({'candidate': candidate, 'ok_alg': ok_alg, 'canon_res_len': 0})

        if not accepted:
            if show_progress:
                print(f"[crt-explore] fp={fp} -> no canonical candidate found after {tried} tries (sample diagnostics below):")
                for s in diagnostics['samples']:
                    print("   - sample:", s)
            results.append({'fp': fp, 'accepted': [], 'tried': tried, 'diagnostics': diagnostics})
        else:
            # store accepted canonicalized entries (could be multiple variants); dedupe by u/v poly
            unique = {}
            for c in accepted:
                key = (str(c.get('u_poly', '')), str(c.get('v_poly','')), str(c.get('s')), str(c.get('p')))
                if key not in unique:
                    unique[key] = c
            results.append({'fp': fp, 'accepted': list(unique.values()), 'tried': tried, 'diagnostics': diagnostics})

    return results
# === END improved_crt_lift_and_diagnostics ===


# ---- paste/replace: improved CRT explorer ----
from sage.all import Integer, QQ

def _crt_list(values, moduli):
    assert len(values) == len(moduli)
    M = 1
    for q in moduli:
        M *= int(q)
    x = 0
    for a_i, p_i in zip(values, moduli):
        m_i = M // int(p_i)
        inv = pow(int(m_i), -1, int(p_i))
        x = (x + (int(a_i) % int(p_i)) * m_i * inv) % M
    return Integer(x), Integer(M)

def _center_rep(x, M):
    x = int(x); M = int(M)
    if x > M//2:
        x = x - M
    return Integer(x)

def crt_lift_fp_representatives_explore_with_recon(intersection_fps, per_prime_fp_map,
                                        primes_used, f_coeffs, canonicalize_fn,
                                        rep_limit=4, max_combinations=4000,
                                        show_progress=True, debug_fp_failures=5,
                                        rational_recon_factor=10):
    """
    For each fp in intersection_fps, try multiple representatives per prime up to rep_limit,
    mix them (capped by max_combinations), CRT-lift, *then try rational reconstruction*
    of each CRT value to small QQ (bounded by rational_recon_factor*M). If rationals
    reconstruct, pass them to canonicalize_and_dedup (as QQ) instead of big ints.

    Returns per-fingerprint result dicts with diagnostics and accepted canonicalized entries.
    """
    results = []
    primes = list(primes_used)

    for fp in sorted(intersection_fps):
        # collect reps-per-prime for this fp
        reps_per_prime = []
        missing = False
        provenance_counts = {}
        for p in primes:
            fp_map = per_prime_fp_map.get(p, {})
            reps = fp_map.get(fp, [])
            if not reps:
                missing = True
                break
            # deterministic, small-first ordering
            def rep_key(rep):
                return tuple(abs(int(x)) for x in rep)
            reps_sorted = sorted(set(reps), key=rep_key)[:rep_limit]
            reps_per_prime.append(reps_sorted)
            provenance_counts[p] = len(reps_sorted)

        if missing:
            if show_progress:
                print(f"[crt-explore] fp={fp} missing reps for some primes -> skipping")
            results.append({'fp': fp, 'accepted': [], 'tried': 0, 'diagnostics': {'reason': 'missing_reps', 'prov': provenance_counts}})
            continue

        total_possible = 1
        for L in reps_per_prime:
            total_possible *= len(L)
        to_try = total_possible if total_possible <= max_combinations else max_combinations

        if show_progress:
            print(f"[crt-explore] fp={fp} -> trying up to {to_try} combos (possible={total_possible}); per-prime counts: {provenance_counts}")

        accepted = []
        tried = 0
        diagnostics = {'samples': [], 'prov': provenance_counts}

        combo_iter = islice(product(*reps_per_prime), to_try)
        for combo in combo_iter:
            tried += 1
            vals_s = [int(rep[0]) for rep in combo]
            vals_p = [int(rep[1]) for rep in combo]
            vals_v0 = [int(rep[2]) for rep in combo]
            vals_v1 = [int(rep[3]) for rep in combo]

            S_int, M = _crt_list(vals_s, primes)
            P_int, M2 = _crt_list(vals_p, primes)
            V0_int, M3 = _crt_list(vals_v0, primes)
            V1_int, M4 = _crt_list(vals_v1, primes)
            assert M == M2 == M3 == M4

            # center integers (still may be huge)
            s_center = _center_rep(S_int, M)
            p_center = _center_rep(P_int, M)
            v0_center = _center_rep(V0_int, M)
            v1_center = _center_rep(V1_int, M)

            # Attempt rational reconstruction for each coefficient separately
            # Use your rational_reconstruct() helper if available; otherwise skip recon.
            recon_ok = False
            rationals = None
            try:
                # rational_reconstruct(crt_val, M) -> (num,den) or raise
                # Use a permissive bound: abs(num),abs(den) <= rational_recon_factor * M
                num_s, den_s = rational_reconstruct(int(S_int), int(M))
                num_p, den_p = rational_reconstruct(int(P_int), int(M))
                num_v0, den_v0 = rational_reconstruct(int(V0_int), int(M))
                num_v1, den_v1 = rational_reconstruct(int(V1_int), int(M))

                # apply smallness heuristic
                thresh = int(rational_recon_factor * int(M))
                if all(abs(x) <= thresh for x in (num_s, den_s, num_p, den_p, num_v0, den_v0, num_v1, den_v1)):
                    s_q = QQ(int(num_s)) / QQ(int(den_s))
                    p_q = QQ(int(num_p)) / QQ(int(den_p))
                    v0_q = QQ(int(num_v0)) / QQ(int(den_v0))
                    v1_q = QQ(int(num_v1)) / QQ(int(den_v1))
                    recon_ok = True
                    rationals = (s_q, p_q, v0_q, v1_q)
                else:
                    # reconstructed rationals are too large, treat as failure
                    recon_ok = False
            except Exception:
                recon_ok = False

            # If we have reconstructed rationals, run the algebraic verification first
            if recon_ok and rationals is not None:
                s_q, p_q, v0_q, v1_q = rationals
                try:
                    ok_alg = verify_mumford_pair(f_coeffs, s_q, p_q, v0_q, v1_q, modulus=None, debug_first_failure=False)
                except Exception:
                    ok_alg = False
                # pass rational candidate to canonicalizer (it expects QQ)
                if ok_alg:
                    try:
                        canon_res = canonicalize_fn([{'s': s_q, 'p': p_q, 'v_0': v0_q, 'v_1': v1_q}], f_coeffs, show_progress=False)
                    except Exception as e:
                        canon_res = []
                        if len(diagnostics['samples']) < debug_fp_failures:
                            diagnostics['samples'].append({'candidate': (s_q, p_q, v0_q, v1_q), 'ok_alg': ok_alg, 'canon_exc': str(e)})
                    else:
                        if canon_res:
                            accepted.extend(canon_res)
                            if show_progress:
                                print(f"[crt-explore] fp={fp} accepted rational-lift after {tried} tries: {canon_res}")
                            break
                        else:
                            # record one sample reason (canonicalizer rejected)
                            if len(diagnostics['samples']) < debug_fp_failures:
                                diagnostics['samples'].append({'candidate': (s_q, p_q, v0_q, v1_q), 'ok_alg': ok_alg, 'canon_res_len': 0})
                else:
                    # algebraic verification failed even for reconstructed rationals
                    if len(diagnostics['samples']) < debug_fp_failures:
                        diagnostics['samples'].append({'candidate': (s_q, p_q, v0_q, v1_q), 'ok_alg': False, 'reason': 'alg_fail'})
                    # continue to next combo

            else:
                # No good rational reconstruction — optionally record a sample integer candidate
                if len(diagnostics['samples']) < debug_fp_failures:
                    diagnostics['samples'].append({
                        'candidate_ints': (int(s_center), int(p_center), int(v0_center), int(v1_center)),
                        'M': int(M),
                        'recon_ok': recon_ok
                    })
                # we DON'T call canonicalize on huge ints (it almost never helps)
                # continue trying combos

        # finalize result for this fp
        if not accepted:
            if show_progress:
                print(f"[crt-explore] fp={fp} -> no canonical candidate after {tried} tries. Diagnostics samples:")
                for s in diagnostics['samples']:
                    print("   -", s)
            results.append({'fp': fp, 'accepted': [], 'tried': tried, 'diagnostics': diagnostics})
        else:
            # dedupe accepted canonicalized results
            unique = {}
            for c in accepted:
                key = (str(c.get('u_poly','')), str(c.get('v_poly','')), str(c.get('s')), str(c.get('p')))
                unique[key] = c
            results.append({'fp': fp, 'accepted': list(unique.values()), 'tried': tried, 'diagnostics': diagnostics})

    return results
# ---- end improved CRT explorer ----


def _fingerprint_from_sol_tuple(sol, p):
    """
    Enhanced fingerprint with more discriminating power.
    """
    s_m = int(sol[0]) % p
    p_m = int(sol[1]) % p
    v0_m = int(sol[2]) % p
    v1_m = int(sol[3]) % p
    
    Delta = (s_m * s_m - 4 * p_m) % p
    if Delta == 0:
        shape = "double"
    else:
        leg = pow(Delta, (p - 1)//2, p)
        shape = "split" if leg == 1 else "irr"
    
    # Add discriminant square-class (not just shape)
    if p > 2 and Delta != 0:
        disc_class = pow(Delta, (p - 1)//2, p)  # 1 or p-1
    else:
        disc_class = 0
    
    # Add v0 square-class for additional separation
    if p > 2 and v0_m != 0:
        v0_class = pow(v0_m, (p - 1)//2, p)
    else:
        v0_class = 0
    
    return (shape, s_m, disc_class, v0_class)


