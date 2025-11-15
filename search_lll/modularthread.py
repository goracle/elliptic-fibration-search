"""
modular_workers.py: Parallel worker functions and modular reduction setup.
"""
from .search_config import (
    multiprocessing, ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
    DEBUG, Fore, Style, tqdm, QQ, ZZ, GF, EllipticCurve, SR_m,
    MIN_PRIME_SUBSET_SIZE, MAX_MODULUS, ROOTS_THRESHOLD,
    reduce, mul, itertools, partial, MAX_K_ABS, LLL_DELTA, BKZ_BLOCK,
    TRUNCATE_MAX_DEG, Integer, vector
)
from .rational_arithmetic import crt_cached, rational_reconstruct, RationalReconstructionError
from .ll_utilities import _trim_poly_coeffs
from .archimedean_optim import minimize_archimedean_t_linear_const
from sage.all import PolynomialRing, var

# ==============================================================
# === Modular Reduction Helpers ================================
# ==============================================================

def reduce_point_hom(E_mod_p, P, p, logger=None):
    """
    Reduce a projective point P (with coordinates in QQ or QQ(m)) to the curve E_mod_p
    (which is over GF(p) or GF(p)(m)).
    """
    def log(msg):
        if logger: logger(msg)
        elif DEBUG: print(msg)

    try:
        Fp_target = E_mod_p.base_field()
        coords = tuple(P)

        if len(coords) == 3:
            X, Y, Z = coords
            try:
                Xr = Fp_target(X)
                Yr = Fp_target(Y)
                Zr = Fp_target(Z)
                return E_mod_p([Xr, Yr, Zr])
            except Exception:
                return None
        if len(coords) == 2:
            x, y = coords
            try:
                xr = Fp_target(x)
                yr = Fp_target(y)
                return E_mod_p(xr, yr)
            except Exception:
                return None
        log("[reduce_point_hom] unsupported coordinate shape")
        return None
    except Exception as outer_e:
        log(f"[reduce_point_hom] p={p} unexpected error: {outer_e}")
        return None

def make_E_mod_p(E_coeffs, p, Fp=None, Fp_m=None):
    """Creates the elliptic curve E modulo p."""
    if Fp is None:
        Fp = GF(p)

    a4_num, a4_den = E_coeffs[3].numerator(), E_coeffs[3].denominator()
    a6_num, a6_den = E_coeffs[4].numerator(), E_coeffs[4].denominator()

    if any(int(QQ(c).denominator()) % p == 0 for c in a4_num.coefficients(sparse=False)):
        if DEBUG: print(f"[prepare_modular_data_lll] skip p={p}: a4 numerator has coeff with denom divisible by p")
        return None
    if any(int(QQ(c).denominator()) % p == 0 for c in a6_num.coefficients(sparse=False)):
        if DEBUG: print(f"[prepare_modular_data_lll] skip p={p}: a6 numerator has coeff with denom divisible by p")
        return None

    if Fp_m is None:
        Rp = PolynomialRing(Fp, 'm')
        Fp_m = Rp.fraction_field()

    try:
        if a4_den.change_ring(Fp).is_zero() or a6_den.change_ring(Fp).is_zero():
            if DEBUG: print(f"[prepare_modular_data_lll] skip p={p}: a4/a6 denominator zero mod p")
            return None
    except Exception:
        if DEBUG: print(f"[prepare_modular_data_lll] skip p={p}: denominator coercion error")
        return None

    a4_modp = Fp_m(a4_num) / Fp_m(a4_den)
    a6_modp = Fp_m(a6_num) / Fp_m(a6_den)

    coeffs_modp = [Fp_m(0), Fp_m(0), Fp_m(0), a4_modp, a6_modp]
    try:
        E_mod_p = EllipticCurve(Fp_m, coeffs_modp)
        return E_mod_p
    except Exception as e:
        if DEBUG: print(f"[prepare_modular_data_lll] EllipticCurve constructor failed at p={p}: {e}")
        return None

def compute_multiples_modp(P_modp, p, max_k_abs=MAX_K_ABS):
    """
    Computes [k]P_modp for k in [-max_k_abs, max_k_abs], and stores them.
    Returns: dict {k: [k]P_modp}
    """
    computed = {}
    k_max = max_k_abs
    k_curr = 1
    P_curr = P_modp
    P_neg = -P_modp

    while k_curr <= k_max:
        computed[k_curr] = P_curr
        computed[-k_curr] = P_neg
        P_curr += P_modp
        P_neg -= P_modp
        k_curr += 1
    return computed

# ==============================================================
# === Preparation and LLL Reduction (Modular) ==================
# ==============================================================

def prepare_modular_data_lll(cd, current_sections, prime_pool, rhs_list, vecs, stats, search_primes=None):
    """ Prepare modular data for LLL-based search across multiple primes. """
    Ep_dict = {}
    rhs_modp_list = {}
    mult_lll = {}
    vecs_lll = {}
    rejected_primes = []
    primes_to_process = search_primes if search_primes is not None else prime_pool
    num_rhs = len(rhs_list)

    E_coeffs = (cd.a1, cd.a2, cd.a3, cd.a4, cd.a6)

    for p in primes_to_process:
        Fp = GF(p)
        Rp = PolynomialRing(Fp, 'm')
        Fp_m = Rp.fraction_field()

        # 1. Create E_mod_p
        try:
            E_mod_p = make_E_mod_p(E_coeffs, p, Fp, Fp_m)
            if E_mod_p is None:
                rejected_primes.append((p, "E_mod_missing"))
                continue
        except Exception as e:
            rejected_primes.append((p, f"E_mod_exception:{e}"))
            continue

        # 2. Reduce sections and compute multiples
        reduced_sections = []
        reduction_failed = False
        mults_for_prime = {}

        for i_sec, S in enumerate(current_sections):
            try:
                Pi_red = reduce_point_hom(E_mod_p, S, p)
            except Exception:
                Pi_red = None

            if Pi_red is None:
                reduction_failed = True
                break
            reduced_sections.append(Pi_red)

            mults_for_prime[i_sec] = compute_multiples_modp(Pi_red, p, MAX_K_ABS)

        if reduction_failed:
            rejected_primes.append((p, "section_reduction_failed"))
            continue

        # 3. Reduce RHS functions
        rhs_reduced_for_p = []
        for rhs_expr in rhs_list:
            try:
                rhs_modp = Fp_m(rhs_expr)
            except Exception:
                rhs_modp = None
            rhs_reduced_for_p.append(rhs_modp)
        rhs_modp_list[p] = rhs_reduced_for_p

        # 4. Reduce search vectors
        vecs_reduced = []
        for v in vecs:
            if isinstance(v, (list, tuple)):
                reduced_v = []
                ok_v = True
                for comp in v:
                    try:
                        val = int(comp) % p
                        reduced_v.append(val)
                    except Exception:
                        ok_v = False
                        break
                if ok_v:
                    vecs_reduced.append(tuple(reduced_v))
                else:
                    vecs_reduced.append(None)
            else:
                try:
                    vecs_reduced.append(int(v) % p)
                except Exception:
                    vecs_reduced.append(None)

        Ep_dict[p] = E_mod_p
        mult_lll[p] = mults_for_prime
        vecs_lll[p] = vecs_reduced

    return Ep_dict, rhs_modp_list, mult_lll, vecs_lll

# ==============================================================
# === Main Worker Function (Single Subset) =====================
# ==============================================================

def _process_prime_subset(p_subset, cd, current_sections, prime_pool, r_m, shift, rhs_list, vecs, tmax):
    """
    Worker function to find m-candidates for a single subset of primes.
    Returns a set of (m_candidate, originating_vector) tuples.
    """
    if not p_subset:
        return set()

    Ep_dict, rhs_modp_list_full, mult_lll, vecs_lll = prepare_modular_data_lll(
        cd, current_sections, prime_pool, rhs_list, vecs, stats=None, search_primes=p_subset
    )

    if not Ep_dict:
        return set()

    found_candidates_for_subset = set()
    r = len(current_sections) # for reference

    for idx, v_orig in enumerate(vecs):
        if all(c == 0 for c in v_orig):
            continue
        v_orig_tuple = tuple(v_orig)

        residue_map = {}
        for p in p_subset:
            if p not in Ep_dict:
                continue

            v_p_list = vecs_lll.get(p)
            if v_p_list is None or idx >= len(v_p_list):
                continue
            v_p_transformed = v_p_list[idx]

            mults = mult_lll.get(p)
            if mults is None:
                continue
            Ep = Ep_dict[p]

            # Compute linear combination of basis points
            Pm = Ep(0)
            for j, coeff in enumerate(v_p_transformed):
                if int(coeff) in mults[j]:
                    Pm += mults[j][int(coeff)]

            if Pm.is_zero():
                continue

            # Find roots for each RHS function
            roots_for_p = set()
            for i, rhs_ff in enumerate(rhs_list):
                if p not in rhs_modp_list_full:
                    continue
                rhs_modp_list = rhs_modp_list_full[p]

                if i >= len(rhs_modp_list) or rhs_modp_list[i] is None:
                    continue

                rhs_p = rhs_modp_list[i]
                try:
                    num_modp = (Pm[0]/Pm[2] - rhs_p).numerator()
                    if not num_modp.is_zero():
                        roots = {int(r) for r in num_modp.roots(ring=GF(p), multiplicities=False)}
                        roots_for_p.update(roots)
                except (ZeroDivisionError, ArithmeticError):
                    continue

            if roots_for_p:
                residue_map[p] = roots_for_p

        # Apply CRT to find m-candidates
        primes_for_crt = [p for p in p_subset if p in residue_map]
        if len(primes_for_crt) < MIN_PRIME_SUBSET_SIZE:
             continue

        lists = [residue_map[p] for p in primes_for_crt]
        for combo in itertools.product(*lists):
            M = reduce(mul, primes_for_crt, 1)

            if M > MAX_MODULUS:
                continue

            m0 = crt_cached(combo, tuple(primes_for_crt))

            try:
                # best_ms is list of (t, m_try, int(x), score)
                best_ms = minimize_archimedean_t_linear_const(int(m0), int(M), r_m, shift, tmax)
            except TypeError:
                best_ms = [(t, QQ(m0 + t * M), 0, 0.0) for t in (-1, 0, 1)]
                if DEBUG: print("here, instead;  why?")

            for t_cand, m_cand, _, _ in best_ms:
                found_candidates_for_subset.add((QQ(m_cand), v_orig_tuple))

            try:
                a, b = rational_reconstruct(m0 % M, M)
                found_candidates_for_subset.add((QQ(a) / QQ(b), v_orig_tuple))
            except RationalReconstructionError:
                pass

    return found_candidates_for_subset

# ==============================================================
# === Parallel Execution Helpers ===============================
# ==============================================================

def _make_executor(max_workers=None):
    """
    Try to create a ProcessPoolExecutor with 'fork' on Linux, fall back to threads.
    """
    try:
        ctx = multiprocessing.get_context("fork")
        return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    except Exception as e:
        if DEBUG:
            print(f"Warning: couldn't start process pool with fork: {e}. Falling back to threads.")
        return ThreadPoolExecutor(max_workers=max_workers)

def process_candidate_numeric(m_val, v_tuple, r_m_callable, shift, rationality_test_func, current_sections):
    """Fast candidate processor (pickleable for multiprocessing)."""
    try:
        x_val = r_m_callable(m_val) - shift
        y_val = rationality_test_func(x_val)
        if y_val is not None:
            v = vector(QQ, v_tuple)
            new_sec = sum(v[i] * current_sections[i] for i in range(len(current_sections))) if current_sections else None
            return m_val, x_val, y_val, v, new_sec
    except (TypeError, ZeroDivisionError, ArithmeticError):
        if DEBUG: print("here we are.")
        return None
    return None

def r_m_numeric_top(m_val, r_m_expr):
    """
    Evaluate symbolic r_m_expr at numeric m_val.
    Returns QQ(x)
    """
    val = r_m_expr.subs({SR_m: m_val})
    return QQ(val)
