# bounds.py - Heuristics for prime pool selection and search bounds.
from sage.all import *
import math
import random
# === safe_splitting_field wrapper ===
import subprocess
import tempfile
import os
import shlex
import multiprocessing, time, traceback

import search_common
from search_common import SEED_INT, DEBUG, NUM_PRIME_SUBSETS

# ==============================================================================
# === High-Level Integration Function ==========================================
# ==============================================================================


# ==============================================================================
# === Core Heuristics ==========================================================
# ==============================================================================



def simple_grh_prime_bound(splitting_field_disc=None, splitting_field_deg=None, fudge=10):
    """
    Rough GRH-inspired bound for how large primes might be needed to see all conjugacy classes.
    Returns an integer bound or None if inputs are missing.
    """
    if splitting_field_disc is None or splitting_field_deg is None:
        return None
    try:
        D = abs(ZZ(splitting_field_disc))
        n = ZZ(splitting_field_deg)
        if D <= 1:
            D = 2 # Handle trivial discriminants
        # Use Sage's log for arbitrary precision
        val = (log(D) + n * log(log(D) + 2.0))**2
        return int(fudge * val + 0.5)
    except (TypeError, ValueError) as e:
        print(f"[bounds] Warning: Could not compute GRH bound due to input error: {e}")
        return None

# ==============================================================================
# === Utility and Helper Functions =============================================
# ==============================================================================

def build_split_poly_from_cd(cd, debug=DEBUG):
    """
    Given a CurveDataExt object, returns a QQ[m] polynomial whose roots are the
    singular fiber m-values (i.e., the numerator of the discriminant Delta).
    """
    try:
        a4, a6 = cd.a4, cd.a6
        if not (hasattr(a4, 'parent') and hasattr(a6, 'parent')):
            raise TypeError("cd.a4 and cd.a6 must be Sage rationals or polynomials.")
        # Symbolic discriminant as a rational function in m
        Delta = -16 * (4 * a4**3 + 27 * a6**2)
    except Exception as e:
        raise RuntimeError(f"Failed to form Delta from cd.a4/a6: {e}")

    try:
        Delta_num = Delta.numerator()
        PR = PolynomialRing(QQ, 'm')
        SPLIT_POLY = PR(Delta_num)
        if debug:
            print(f"[bounds] Built SPLIT_POLY of degree {SPLIT_POLY.degree()} from Delta numerator.")
        return SPLIT_POLY
    except Exception as e:
        raise RuntimeError(f"Could not coerce Delta numerator to QQ['m'] polynomial. Error: {e}")


def compute_poly_diagnostics(poly, run_heavy=False, debug=DEBUG):
    """
    Computes diagnostics for a Sage polynomial over QQ.

    Args:
        poly: A Sage polynomial in one variable over QQ.
        run_heavy (bool): If True, attempts slow computations like Galois group.
        debug (bool): If True, prints status messages.

    Returns:
        dict: A dictionary with diagnostic information.
    """
    if not hasattr(poly, 'parent'):
        raise TypeError("Input must be a Sage polynomial.")

    out = {
        'discriminant': None,
        'gal_group_order': None,
        'gal_group_info': "Not computed",
        'splitting_field_degree': None,
        'splitting_field_discriminant': None
    }

    try:
        out['discriminant'] = ZZ(poly.discriminant())
    except Exception as e:
        if debug:
            print(f"[bounds] Failed to compute polynomial discriminant: {e}")

    if run_heavy:
        try:
            if debug: print("[bounds] Computing Galois group (may be slow)...")
            G = poly.galois_group()
            out['gal_group_info'] = str(G)
            try:
                out['gal_group_order'] = int(G.order())
            except Exception:
                pass
        except Exception as e:
            out['gal_group_info'] = f"galois_group() failed: {e}"

        try:
            if debug: print("[bounds] Computing splitting field (may be slow)...")
            K = poly.splitting_field('a')
            out['splitting_field_degree'] = int(K.degree())
            try:
                out['splitting_field_discriminant'] = ZZ(K.discriminant())
            except Exception:
                pass
        except Exception as e:
             if debug: print(f"[bounds] Failed to compute splitting field: {e}")

    return out



def choose_primes_for_modulus(prime_pool, B, ascending=True):
    """
    Greedily chooses primes from the pool whose product exceeds B.
    """
    if B < 2:
        return [], 1
    primes = sorted(prime_pool)
    if not ascending:
        primes = list(reversed(primes))

    M = 1
    chosen = []
    for p in primes:
        p_int = int(p)
        chosen.append(p_int)
        M *= p_int
        if M > B:
            break
    return chosen, M


def expected_survivors_per_subset(residue_counts, primes):
    """
    Estimates the density of solutions surviving CRT filtering for a given set of primes.
    residue_counts: dict of {prime: num_solutions_mod_prime}.
    """
    density = 1.0
    for p in primes:
        # Default to a conservative 1 residue if not specified
        num_residues = residue_counts.get(p, 1)
        density *= float(num_residues) / float(p)
    return density


def gen_random_subsets_meeting_modulus(prime_pool, subset_size, num_subsets, B, seed=SEED_INT):
    """
    Generate random, distinct prime subsets of a given size whose product exceeds B.
    """
    random.seed(seed)
    chosen_subsets = set()
    max_tries = max(10000, 10 * num_subsets) # Prevent infinite loops

    if subset_size > len(prime_pool):
        return []

    for _ in range(max_tries):
        if len(chosen_subsets) >= num_subsets:
            break
        subset = tuple(sorted(random.sample(prime_pool, subset_size)))
        if subset in chosen_subsets:
            continue

        # Check if product exceeds modulus B
        M = 1
        for p in subset:
            M *= p
        if M > B:
            chosen_subsets.add(subset)

    return list(chosen_subsets)



# ---- paste this into bounds.py, replacing the old functions ----
def modulus_needed_from_canonical_height(h_can, scale_const=2.0, max_modulus=None, debug=DEBUG):
    """
    Safe translation from canonical height h_can to modulus bound B.
    - Uses log-space to avoid math.exp overflow.
    - Caps result at max_modulus (default: search_common.MAX_MODULUS if present).
    Returns integer B >= 2.
    """
    if max_modulus is None:
        max_modulus = getattr(search_common, 'MAX_MODULUS', 10**9)

    if h_can is None:
        return 2

    # target natural-log of B
    logB = float(scale_const) * float(h_can)

    # safe threshold for math.exp on typical systems (~1e308 -> log ≈ 709)
    SAFE_EXP_LOG_LIMIT = 700.0

    # If small enough, compute directly
    if logB <= SAFE_EXP_LOG_LIMIT:
        try:
            B = int(math.exp(logB) + 0.5)
        except OverflowError:
            B = max_modulus
    else:
        # Too big to exp safely. compute number of decimal digits:
        log10 = math.log(10.0)
        digits = int(math.ceil(logB / log10))   # minimal power-of-10 with >= exp(logB)
        # If even the smallest 10^digits exceeds max_modulus, return the cap directly.
        max_digits_allowed = int(math.floor(math.log10(max_modulus))) if max_modulus > 0 else 0
        if digits > max_digits_allowed:
            if debug:
                print(f"[bounds] modulus_needed_from_canonical_height: desired ~exp({logB:.1f}) (~{digits} digits) exceeds max_modulus={max_modulus}, capping.")
            return max(2, int(max_modulus))
        # otherwise build a conservative decimal power (10**digits) which is integer-safe
        B = 10 ** digits

    # final cap and safety
    B = max(2, min(int(B), int(max_modulus)))
    return B


def recommend_subset_size_and_count(prime_pool, residue_counts, h_can,
                                     target_expected_survivors=1.0,
                                     max_subsets=2000,
                                     max_modulus=None,
                                     scale_const=2.0,
                                     debug=DEBUG):
    """
    Recommends subset-size / number-of-subsets.
    - max_modulus: upper cap for the modulus B (defaults to search_common.MAX_MODULUS).
    - scale_const: passed to modulus_needed_from_canonical_height for tuning.
    Returns dict with B, chosen_primes, M, density_per_subset, recommended_num_subsets.
    """
    B = modulus_needed_from_canonical_height(h_can, scale_const=scale_const, max_modulus=max_modulus, debug=debug)
    chosen, M = choose_primes_for_modulus(prime_pool, B, ascending=True)
    dens = expected_survivors_per_subset(residue_counts, chosen)

    if dens > 0:
        rec_subsets = int(max(1, round(target_expected_survivors / dens)))
    else:
        rec_subsets = 0

    rec_subsets = min(rec_subsets, max_subsets)

    if debug:
        print("[bounds] recommend_subset_size_and_count: B =", B, "chosen_primes_len =", len(chosen),
              "product_M (approx) =", M, "density =", dens, "rec_subsets =", rec_subsets)

    return {
        'B': B,
        'chosen_primes': chosen,
        'M': M,
        'density_per_subset': dens,
        'recommended_num_subsets': rec_subsets
    }
# ---- end replacement ----


# Add these functions to bounds.py or search_lll.py




# Add these functions to bounds.py or search_lll.py

def compute_residue_counts_for_primes(cd, rhs_list, prime_pool, max_primes=None):
    """
    Compute how many residue classes mod p satisfy the search equations.
    Returns dict {p: count} for density estimation.
    
    This counts solutions to the quartic polynomial mod p.
    """
    from sage.all import PolynomialRing, GF, QQ
    
    if max_primes is not None:
        prime_pool = prime_pool[:max_primes]
    
    PR_m = PolynomialRing(QQ, 'm')
    residue_counts = {}
    
    # Process the first RHS (usually sufficient for density estimation)
    rhs = rhs_list[0] if rhs_list else None
    if rhs is None:
        # Fallback: assume uniform distribution
        return {p: p//4 for p in prime_pool}
    
    rhs_num = PR_m(rhs.numerator())
    rhs_den = PR_m(rhs.denominator())
    
    for p in prime_pool:
        try:
            # Check if coefficients are p-adic integers
            if any(QQ(c).denominator() % p == 0 for c in rhs_num.coefficients(sparse=False)):
                residue_counts[p] = p // 4  # conservative guess
                continue
            if any(QQ(c).denominator() % p == 0 for c in rhs_den.coefficients(sparse=False)):
                residue_counts[p] = p // 4
                continue
            
            Rp = PolynomialRing(GF(p), 'm')
            
            # Check if denominator vanishes
            if rhs_den.change_ring(GF(p)).is_zero():
                residue_counts[p] = p // 4
                continue
            
            # Count roots of the numerator polynomial mod p
            rhs_num_p = rhs_num.change_ring(GF(p))
            roots = rhs_num_p.roots(multiplicities=False)
            residue_counts[p] = len(roots)
            
        except Exception as e:
            # On any error, use conservative estimate
            residue_counts[p] = max(1, p // 4)
    
    return residue_counts


def generate_diverse_prime_subsets(prime_pool, residue_counts, num_subsets, 
                                   min_size, max_size, seed=SEED_INT, 
                                   force_full_pool=False):
    """
    Generate diverse prime subsets with varying sizes.
    This is MUCH better than enforcing a fixed size and modulus bound.
    
    Key insight: You want DIVERSITY in subset composition, not just meeting
    a modulus threshold. Small subsets (3-5 primes) can find different solutions
    than large subsets (8-10 primes).
    """
    import random
    random.seed(seed)
    
    subsets = []
    
    # Always include the full pool
    if force_full_pool:
        subsets.append(tuple(prime_pool))
    
    # Generate random subsets with varying sizes
    remaining = num_subsets - (1 if force_full_pool else 0)
    
    for _ in range(remaining):
        # Random size in the range [min_size, max_size]
        size = random.randint(min_size, min(max_size, len(prime_pool)))
        subset = tuple(sorted(random.sample(prime_pool, size)))
        subsets.append(subset)
    
    # Deduplicate while preserving order
    seen = set()
    unique_subsets = []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            unique_subsets.append(s)
    
    return unique_subsets


def recommend_subset_strategy_empirical(prime_pool, residue_counts, 
                                       num_subsets=250,
                                       min_size=3, max_size=9):
    """
    Use empirically-validated strategy: generate diverse random subsets
    of varying sizes rather than enforcing a modulus bound.
    
    Returns dict with subset generation parameters.
    """
    # Count usable primes (non-zero residues)
    usable_primes = [p for p in prime_pool if residue_counts.get(p, 1) > 0]
    zero_ratio = 1.0 - len(usable_primes) / len(prime_pool) if prime_pool else 0
    
    # Quick density check to see if we need more coverage
    avg_density = sum(residue_counts.get(p, 1) / p for p in prime_pool) / len(prime_pool)
    
    # Adjust strategy based on how degenerate this fibration is
    if zero_ratio > 0.7:
        # Very degenerate: most primes don't work
        size_bias = "degenerate"
        recommended_min = min_size
        recommended_max = min(max_size, len(usable_primes))
        # Increase num_subsets to compensate for limited prime pool
        recommended_num = min(num_subsets * 2, 500)
    elif avg_density < 0.1:
        size_bias = "large"
        recommended_min = max(min_size, 5)
        recommended_max = max_size
        recommended_num = num_subsets
    elif avg_density > 0.3:
        size_bias = "small"
        recommended_min = min_size
        recommended_max = min(max_size, 7)
        recommended_num = num_subsets
    else:
        size_bias = "mixed"
        recommended_min = min_size
        recommended_max = max_size
        recommended_num = num_subsets
    
    recommended_min = 3 # temporary override
    return {
        'num_subsets': recommended_num,
        'min_size': recommended_min,
        'max_size': recommended_max,
        'avg_density': avg_density,
        'size_bias': size_bias,
        'usable_primes': len(usable_primes),
        'zero_ratio': zero_ratio
    }


def _worker_splitting_field(poly, q):
    """
    Worker to run in separate process. Puts a tuple (success_flag, result_dict_or_error) in queue q.
    Only basic serializable info returned: degree and discriminant (int) and optional gal_group_order if available.
    """
    try:
        # compute splitting field (may call PARI internally)
        K = poly.splitting_field('a')
        deg = int(K.degree())
        try:
            Dk = K.discriminant()
            Dk_int = int(Dk)
        except Exception:
            Dk_int = None
        # try to get Galois group order if cheap
        g_order = None
        try:
            G = poly.galois_group()
            try:
                g_order = int(G.order())
            except Exception:
                g_order = None
        except Exception:
            # ignore heavy galois failure
            g_order = None
        q.put((True, {'splitting_field_degree': deg,
                      'splitting_field_discriminant': Dk_int,
                      'gal_group_order': g_order}))
    except Exception as e:
        q.put((False, str(traceback.format_exc())))




# ==== safe subprocess-based splitting-field helper ====
def safe_compute_splitting_field_info_subprocess(poly, timeout=30, debug=DEBUG):
    """
    Compute basic splitting-field info by launching a separate Sage process.
    Returns a dict possibly containing keys:
      - 'splitting_field_degree' (int)
      - 'splitting_field_discriminant' (int)
    On timeout or error returns {} (empty dict).

    This is safer than calling poly.splitting_field() in-process because the
    OS can reliably kill the external process if it hangs.
    """
    # Serialize polynomial to a string Sage can reparse
    try:
        poly_str = str(poly)   # e.g. "m^12 - 4*m^11 + ..."
    except Exception as e:
        if debug:
            print("[bounds][safe_subproc] Failed to stringify poly:", e)
        return {}

    # Build temporary Python script that uses sage (sage -python)
    script = f"""
from sage.all import QQ, PolynomialRing
import sys, traceback
try:
    PR = PolynomialRing(QQ, 'm')
    f = PR({poly_str!r})
    # attempt splitting field; we only print degree and discriminant (small outputs)
    K = f.splitting_field('a')
    deg = int(K.degree())
    try:
        Dk = K.discriminant()
        Dk_int = int(Dk)
    except Exception:
        Dk_int = None
    # Output as two lines that parent will parse
    print(deg)
    print(Dk_int if Dk_int is not None else "None")
except Exception:
    traceback.print_exc()
    sys.exit(2)
"""
    fd, tmpname = tempfile.mkstemp(prefix="sage_split_", suffix=".py", text=True)
    try:
        os.write(fd, script.encode("utf-8"))
        os.close(fd)
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
    # Compose command: use sage -python so environment is consistent
    cmd = ["sage", "-python", tmpname]
    if debug:
        print("[bounds][safe_subproc] Running:", " ".join(shlex.quote(c) for c in cmd), f" timeout={timeout}s")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # timed out: try to kill and return empty
        if debug:
            print(f"[bounds][safe_subproc] timed out after {timeout}s (process killed).")
        try:
            # best-effort kill; subprocess.run will have killed by timeout
            pass
        except Exception:
            pass
        try:
            os.unlink(tmpname)
        except Exception:
            pass
        return {}
    except FileNotFoundError as e:
        # sage not found on PATH
        if debug:
            print("[bounds][safe_subproc] 'sage' binary not found on PATH:", e)
        try:
            os.unlink(tmpname)
        except Exception:
            pass
        return {}
    # parse output
    out = proc.stdout.strip().splitlines()
    err = proc.stderr.strip()
    if debug:
        if out:
            print("[bounds][safe_subproc] stdout lines:", out[:10])
        if err:
            print("[bounds][safe_subproc] stderr (first 500 chars):", err[:500])
    result = {}
    if proc.returncode != 0:
        # error occurred during splitting field; return empty (caller will fall back)
        if debug:
            print(f"[bounds][safe_subproc] child exited with code {proc.returncode}; falling back.")
        try:
            os.unlink(tmpname)
        except Exception:
            pass
        return {}
    # expect two lines: degree and discriminant (or None)
    try:
        if len(out) >= 1:
            deg_line = out[0].strip()
            if deg_line and deg_line != "None":
                result['splitting_field_degree'] = int(deg_line)
        if len(out) >= 2:
            disc_line = out[1].strip()
            if disc_line not in ("", "None"):
                # discriminant can be very large; read as Python int
                result['splitting_field_discriminant'] = int(disc_line)
    except Exception as e:
        if debug:
            print("[bounds][safe_subproc] parse error:", e)
        result = {}

    try:
        os.unlink(tmpname)
    except Exception:
        pass
    return result



# === Replace recommend_and_update_prime_pool ===
def recommend_and_update_prime_pool(cd, prime_pool=None, run_heavy=True,
                                    grh_fudge=10, debug=DEBUG,
                                    update_search_common=False):
    """
    Build SPLIT_POLY diagnostics and return a filtered list of primes.
    Will NOT overwrite search_common.PRIME_POOL unless update_search_common=True.

    Args:
        cd (CurveDataExt): curve/fibration data
        prime_pool (list, optional): initial primes (defaults to search_common.PRIME_POOL)
        run_heavy (bool): whether to run splitting-field computations
        grh_fudge (int): fudge factor for GRH-based cap
        update_search_common (bool): if True, update search_common.PRIME_POOL in-place
    Returns:
        list: filtered primes (sorted, unique)
    """
    src_pool = list(prime_pool) if prime_pool is not None else list(getattr(search_common, 'PRIME_POOL', []))
    if debug:
        print("[bounds] Starting with prime pool size:", len(src_pool))

    try:
        SPLIT_POLY = build_split_poly_from_cd(cd, debug=debug)
    except RuntimeError as e:
        if debug:
            print("[bounds] Could not build SPLIT_POLY; returning original pool. Error:", e)
        return src_pool

    diag = compute_poly_diagnostics(SPLIT_POLY, run_heavy=False, debug=debug)

    if run_heavy:
        # attempt heavier diagnostics (may fail/timeout)
        diag.update(estimate_galois_signature_modp(SPLIT_POLY, primes_to_test=src_pool, debug=debug))
        #heavy = safe_compute_splitting_field_info_subprocess(SPLIT_POLY, timeout=90, debug=debug)
        #if heavy:
        #    diag.update(heavy)

    if debug:
        print("[bounds] Diagnostics:", {k: diag.get(k) for k in ('discriminant','splitting_field_degree','splitting_field_discriminant')})

    # GRH-inspired cap
    grh_bound = simple_grh_prime_bound(
        splitting_field_disc=diag.get('splitting_field_discriminant'),
        splitting_field_deg=diag.get('splitting_field_degree'),
        fudge=grh_fudge
    )
    filtered_pool = list(src_pool)
    if grh_bound is not None:
        if debug:
            print(f"[bounds] Applying GRH cap p <= {grh_bound}")
        filtered_pool = [p for p in filtered_pool if p <= grh_bound]

    # filter primes dividing the polynomial discriminant (ramified)
    D = diag.get('discriminant')
    if D is not None and D != 0 and False:
        try:
            bad = {int(p) for p in prime_divisors(ZZ(D))}
            filtered_pool = [p for p in filtered_pool if p not in bad]
            if debug:
                print(f"[bounds] Removed ramified primes: {sorted(bad)}")
        except Exception:
            # if discriminant factoring fails, just continue with filtered_pool
            if debug:
                print("[bounds] Could not factor discriminant for ramified-prime filtering; skipping.")

    # apply domain-specific prime filter if available
    final_pool = []
    if hasattr(search_common, 'is_good_prime_for_surface'):
        for p in filtered_pool:
            try:
                if search_common.is_good_prime_for_surface(cd, p):
                    final_pool.append(p)
                elif debug:
                    print(f"[bounds] is_good_prime_for_surface rejected {p}")
            except Exception as e:
                # If the check crashes, keep the prime but log if debug
                if debug:
                    print(f"[bounds] is_good_prime_for_surface error for p={p}: {e}; keeping prime.")
                final_pool.append(p)
    else:
        final_pool = filtered_pool

    new_pool = sorted(list(set(final_pool)))
    if update_search_common:
        search_common.PRIME_POOL = new_pool
        if debug:
            print("[bounds] Updated search_common.PRIME_POOL (explicit).")

    return new_pool


# === Better canonical-height estimate from x-height ===
def estimate_canonical_height_from_xheight(h_x, curve_discriminant=None, fudge_const=None, debug=DEBUG):
    """
    Conservative estimate of canonical height hat{h} from naive x-height h_x = log max(|num_x|,|den_x|).
    We use the classical heuristic
        hat{h}(P) ~= 1/2 * h_x(P) + O(1)
    and bound the O(1) term using a crude curve-dependent constant:
        O(1) <= (1/12)*log|Delta| + 2.0
    Returns a nonnegative float.
    """
    if h_x is None:
        return None

    # default fudge: if user supplies curve_discriminant we use it, otherwise conservative small shift
    c_curve = 0.0
    if curve_discriminant is not None:
        try:
            c_curve = max(0.0, float(log(abs(int(curve_discriminant)) + 1.0)) / 12.0)
        except Exception:
            c_curve = 0.0

    if fudge_const is None:
        fudge_const = 2.0

    est = 0.5 * float(h_x) + c_curve + float(fudge_const)
    if debug:
        print(f"[bounds] estimate_canonical_height_from_xheight: h_x={h_x}, c_curve={c_curve:.3f}, fudge={fudge_const} -> hat_h ≈ {est:.4g}")
    return max(0.0, est)


# === Replace modulus_needed_from_canonical_height with safer default scale ===
def modulus_needed_from_canonical_height(h_can, scale_const=1.0, max_modulus=None, debug=DEBUG):
    """
    Translate canonical height to modulus bound B:
      log(B) := scale_const * h_can
    scale_const is 1.0 by default (more conservative than previous 2.0).
    """
    if max_modulus is None:
        max_modulus = getattr(search_common, 'MAX_MODULUS', 10**9)

    if h_can is None:
        return 2

    logB = float(scale_const) * float(h_can)
    SAFE_EXP_LOG_LIMIT = 700.0

    if logB <= SAFE_EXP_LOG_LIMIT:
        try:
            B = int(math.exp(logB) + 0.5)
        except OverflowError:
            B = max_modulus
    else:
        log10 = math.log(10.0)
        digits = int(math.ceil(logB / log10))
        max_digits_allowed = int(math.floor(math.log10(max_modulus))) if max_modulus > 0 else 0
        if digits > max_digits_allowed:
            if debug:
                print(f"[bounds] modulus_needed_from_canonical_height: capping at max_modulus={max_modulus}")
            return max(2, int(max_modulus))
        B = 10 ** digits

    B = max(2, min(int(B), int(max_modulus)))
    if debug:
        print("[bounds] modulus_needed_from_canonical_height: computed B =", B)
    return B


# === Automatic choice of small primes for Galois signature testing ===
def choose_galois_primes(poly, prime_pool=None, max_primes=8, debug=DEBUG):
    """
    Choose small primes from prime_pool (or from small primes list) suitable for mod-p factorization diagnostics.
    Excludes 2,3 and primes dividing leading coefficient / discriminant.
    """
    from sage.all import ZZ
    if prime_pool is None:
        prime_pool = list(primes(100))
    primes_candidates = [p for p in prime_pool if p not in (2,3)]
    # exclude primes dividing discriminant when possible
    disc = None
    try:
        disc = ZZ(poly.discriminant())
    except Exception:
        disc = None

    chosen = []
    for p in primes_candidates:
        if len(chosen) >= max_primes:
            break
        if disc is not None:
            if int(p) in {int(q) for q in prime_divisors(disc)}:
                continue
        chosen.append(int(p))
    if debug:
        print(f"[bounds] choose_galois_primes -> {chosen}")
    return chosen[:max_primes]


# === Dynamic estimate for tmax ===
def estimate_tmax_from_B_and_density(B, density_per_subset, base_max=500, debug=DEBUG):
    """
    Heuristic: tmax should scale mildly with log(B) and inversely with density (smaller density -> larger tmax).
    Formula used:
        tmax = clamp( int( min(base_max, max(50, alpha*log10(B) / max(density, eps)) )), 50, base_max)
    where alpha is tunable; default alpha=20.
    """
    if B is None or B < 2:
        return min(200, base_max)
    alpha = 20.0
    eps = 1e-6
    log10B = math.log10(float(B))
    density = max(float(density_per_subset), eps)
    raw = int(round(alpha * log10B / density))
    t = max(50, min(base_max, raw))
    if debug:
        print(f"[bounds] estimate_tmax_from_B_and_density: B={B}, log10B={log10B:.2f}, density={density:.4g} -> tmax={t}")
    return t


# === Recommend subset strategy but do not pick magic numbers ===
def recommend_subset_strategy_empirical(prime_pool, residue_counts, target_expected_survivors=1.0,
                                        num_subsets_hint=250, min_size_hint=3, max_size_hint=9, debug=DEBUG):
    """
    Returns an adaptive plan for subset generation: number of subsets, size ranges, and picks.
    Uses residue_counts to adaptively alter min/max and number of subsets.
    """
    usable_primes = [p for p in prime_pool if residue_counts.get(p, 1) > 0]
    zero_ratio = 1.0 - (len(usable_primes) / len(prime_pool)) if prime_pool else 0.0
    avg_density = sum(residue_counts.get(p, 1) / p for p in prime_pool) / max(1, len(prime_pool))

    if zero_ratio > 0.7:
        recommended_min = min_size_hint
        recommended_max = max( min(max_size_hint, max(3, len(usable_primes))), recommended_min )
        recommended_num = min(num_subsets_hint * 2, 2000)
        size_bias = "degenerate"
    elif avg_density < 0.08:
        recommended_min = max(min_size_hint, 5)
        recommended_max = max_size_hint
        recommended_num = num_subsets_hint
        size_bias = "large"
    elif avg_density > 0.25:
        recommended_min = min_size_hint
        recommended_max = min(max_size_hint, 7)
        recommended_num = max(50, num_subsets_hint // 2)
        size_bias = "small"
    else:
        recommended_min = min_size_hint
        recommended_max = max_size_hint
        recommended_num = num_subsets_hint
        size_bias = "mixed"

    # final safeguards (no magic)
    recommended_min = max(3, recommended_min)
    recommended_min = 3 # temporary override
    recommended_max = min(max(recommended_min, recommended_max), len(prime_pool))
    recommended_num = max(10, min(recommended_num, 2000))

    if debug:
        print("[bounds] recommend_subset_strategy_empirical:",
              {"num_subsets": recommended_num, "min_size": recommended_min, "max_size": recommended_max,
               "avg_density": avg_density, "size_bias": size_bias, "usable_primes": len(usable_primes),
               "zero_ratio": zero_ratio})

    return {
        'num_subsets': recommended_num,
        'min_size': recommended_min,
        'max_size': recommended_max,
        'avg_density': avg_density,
        'size_bias': size_bias,
        'usable_primes': len(usable_primes),
        'zero_ratio': zero_ratio
    }


def auto_configure_search(cd, known_pts, prime_pool=None,
                          existing_height_bound=None,
                          max_modulus=10**15,
                          update_search_common=False,
                          num_subsets_hint=NUM_PRIME_SUBSETS,
                          debug=DEBUG):
    """
    SIMPLIFIED automatic configuration for search.
    
    Key insight: With random subset generation, we don't need fancy modulus bounds.
    Just set MAX_MODULUS high and focus on HEIGHT_BOUND.
    """
    import math
    
    # 1) Filter prime pool
    src_pool = list(prime_pool) if prime_pool is not None else list(getattr(search_common, 'PRIME_POOL', list(primes(90))))
    if debug:
        print("[auto_cfg] starting prime pool size:", len(src_pool))

    try:
        pool_filtered = recommend_and_update_prime_pool(cd, prime_pool=src_pool,
                                                        run_heavy=True, debug=debug,
                                                        update_search_common=update_search_common)
    except Exception as e:
        if debug:
            print("[auto_cfg] recommend_and_update_prime_pool failed:", e)
        pool_filtered = src_pool

    # Ensure critical small primes remain
    for p in (2, 3, 5):
        if p in src_pool and p not in pool_filtered:
            pool_filtered.insert(0, p)
    pool_filtered = sorted(set(pool_filtered))

    # 2) Estimate canonical height from known points
    def naive_x_height_from_pts(pts):
        vals = []
        for x, y in pts:
            try:
                n = abs(Integer(x).numerator())
                d = abs(Integer(x).denominator())
                vals.append(max(1, max(n, d)))
            except Exception:
                vals.append(1)
        if not vals:
            return 0.0
        return float(math.log(max(vals)))

    h_x = naive_x_height_from_pts(known_pts)
    disc = None
    try:
        disc = cd.discriminant
    except Exception:
        try:
            disc = getattr(cd, 'weierstrass_discriminant', None)
        except Exception:
            disc = None

    # Estimate canonical height with moderate fudge
    h_can = estimate_canonical_height_from_xheight(h_x, curve_discriminant=disc, 
                                                   fudge_const=3.0, debug=debug)
    
    if debug:
        print(f"[auto_cfg] h_x={h_x:.2f}, h_can≈{h_can:.2f}")

    # 3) Compute residue counts for density estimation
    try:
        residue_counts = compute_residue_counts_for_primes(cd, [SR(cd.phi_x)], 
                                                           pool_filtered, 
                                                           max_primes=min(len(pool_filtered), 30))
    except Exception as e:
        if debug:
            print(f"[auto_cfg] residue count computation failed: {e}")
        residue_counts = {p: max(1, p // 4) for p in pool_filtered}
        raise
    
    avg_density = sum((residue_counts.get(p, 1) / float(p)) for p in pool_filtered) / max(1, len(pool_filtered))

    # 4) MAX_MODULUS: Safety cap on CRT subset products
    # This is just a guard rail to prevent accidentally huge products.
    # Since our prime subsets are bounded (3-9 primes, max ~89), the worst case
    # is ~10^17 for 9 large primes. Setting to 10^15 covers everything reasonable.
    
    B = 10**15  # Large enough to never filter real subsets
    
    if debug:
        print(f"[auto_cfg] MAX_MODULUS = {B} (safety cap for CRT)")

    # 5) Subset strategy based on density
    subset_plan = recommend_subset_strategy_empirical(pool_filtered, residue_counts,
                                                     target_expected_survivors=1.0,
                                                     num_subsets_hint=num_subsets_hint,
                                                     min_size_hint=3, max_size_hint=9,
                                                     debug=debug)
    
    # 6) TMAX: How much LLL/enumeration to do per (m, subset) pair
    # Base it on density - lower density needs more enumeration
    if avg_density < 0.05:
        tmax = 400
    elif avg_density < 0.15:
        tmax = 300
    else:
        tmax = 200
    
    tmax = min(tmax, 500)  # Safety cap
    
    if debug:
        print(f"[auto_cfg] density={avg_density:.4f} -> TMAX={tmax}")

    # 7) Generate diverse prime subsets
    try:
        prime_subsets = generate_diverse_prime_subsets(
            prime_pool=pool_filtered,
            residue_counts=residue_counts,
            num_subsets=subset_plan['num_subsets'],
            min_size=subset_plan['min_size'],
            max_size=subset_plan['max_size'],
            seed=SEED_INT,
            force_full_pool=False
        )
        prime_subsets = sorted({tuple(sorted(s)) for s in prime_subsets}, key=lambda t: (len(t), t))
        prime_subsets = [list(t) for t in prime_subsets]
    except Exception as e:
        if debug:
            print("[auto_cfg] generate_diverse_prime_subsets failed:", e)
        import random
        rnd = random.Random(SEED_INT)
        prime_subsets = []
        for _ in range(subset_plan['num_subsets']):
            size = rnd.randint(subset_plan['min_size'], min(len(pool_filtered), subset_plan['max_size']))
            prime_subsets.append(sorted(rnd.sample(pool_filtered, size)))
        prime_subsets = sorted({tuple(s) for s in prime_subsets}, key=lambda t: (len(t), t))
        prime_subsets = [list(t) for t in prime_subsets]

    # 8) HEIGHT_BOUND: The key parameter for LLL search
    if existing_height_bound is not None:
        # User override
        final_height_bound = existing_height_bound
    else:
        # Formula: HEIGHT_BOUND should be large enough to find the next point
        # Heuristic: Use exponential scaling with generous constant
        # For h_can ≈ 2, this gives ~350
        # For h_can ≈ 5, this gives ~850
        base = 100 * math.exp(h_can / 4.0)
        # Add extra for safety
        final_height_bound = int(base + 100)
        # Reasonable cap
        final_height_bound = min(final_height_bound, 2000)
    
    if debug:
        print(f"[auto_cfg] HEIGHT_BOUND = {final_height_bound}")

    # 9) Package everything
    sconf = {
        'HEIGHT_BOUND': final_height_bound,
        'PRIME_POOL': pool_filtered,
        'RESIDUE_COUNTS': residue_counts,
        'SUBSET_PLAN': subset_plan,
        'PRIME_SUBSETS': prime_subsets,
        'NUM_PRIME_SUBSETS': len(prime_subsets),
        'MIN_PRIME_SUBSET_SIZE': subset_plan['min_size'],
        'MIN_MAX_PRIME_SUBSET_SIZE': subset_plan['max_size'],
        'MAX_MODULUS': int(B),
        'TMAX': int(tmax),
        'AVG_DENSITY': avg_density,
        'H_CAN_ESTIMATE': h_can
    }

    if debug:
        print("\n[auto_cfg] === CONFIGURATION SUMMARY ===")
        print(f"  HEIGHT_BOUND: {sconf['HEIGHT_BOUND']} (controls LLL search)")
        print(f"  MAX_MODULUS: {sconf['MAX_MODULUS']} (CRT modulus cap)")
        print(f"  TMAX: {sconf['TMAX']} (LLL enum per m-value)")
        print(f"  Prime subsets: {sconf['NUM_PRIME_SUBSETS']} subsets")
        print(f"  Subset sizes: [{subset_plan['min_size']}, {subset_plan['max_size']}]")
        print(f"  Prime pool: {len(sconf['PRIME_POOL'])} primes")
        print(f"  Avg density: {sconf['AVG_DENSITY']:.4f}")
        print(f"  Sample subsets: {prime_subsets[:3]}")

    return sconf


def estimate_galois_signature_modp(poly, primes_to_test=None, debug=DEBUG):
    """
    Empirical proxy for Galois/splitting-field complexity.
    Factors poly mod primes and deduplicates factorization patterns.
    
    The splitting-field degree is estimated as the LCM of all distinct
    cycle-type patterns observed. This avoids magic numbers and scales
    with whatever primes you provide.

    Args:
        poly: A Sage polynomial over QQ
        primes_to_test: List of primes to factor mod. If None, uses first 20 odd primes.
        debug: If True, prints diagnostic info
    
    Returns:
        dict with keys:
          - splitting_field_degree_est (int): LCM of distinct cycle patterns
          - unique_patterns (list of tuples): sorted unique factorization patterns
          - num_primes_tested (int): how many primes were actually used
    """
    from sage.all import GF, lcm as sage_lcm

    if primes_to_test is None:
        from sage.all import primes
        primes_to_test = list(primes(100))[:20]

    patterns_seen = set()
    primes_used = 0

    for p in primes_to_test:
        try:
            fp = poly.change_ring(GF(p))
            facs = [f[0].degree() for f in fp.factor()]
            pattern = tuple(sorted(facs))
            patterns_seen.add(pattern)
            primes_used += 1

            if debug:
                print(f"[bounds] mod {p} factorization: {facs} -> pattern {pattern}")

        except Exception as e:
            if debug:
                print(f"[bounds] mod {p} factorization failed: {e}")
            continue

    unique_patterns = sorted(patterns_seen)

    if unique_patterns:
        from math import gcd
        from functools import reduce

        def lcm(a, b):
            return abs(a * b) // gcd(a, b)

        deg_est = 1
        for pattern in unique_patterns:
            for cycle_deg in pattern:
                deg_est = lcm(deg_est, cycle_deg)
    else:
        deg_est = poly.degree()

    if debug:
        print(f"[bounds] Unique patterns from {primes_used} primes: {unique_patterns}")
        print(f"[bounds] Estimated splitting field degree: {deg_est}")

    ret = {
        'splitting_field_degree_est': deg_est,
        'unique_patterns': unique_patterns,
        'num_primes_tested': primes_used,
    }

    if DEBUG:
        print("")
        print("DEBUG: RET:", ret)
        print("")

    return ret
