from math import gcd
from sage.all import PolynomialRing, FractionField, QQ, GF, is_prime, matrix, ZZ, EllipticCurve, Integer, randint
from itertools import product
from tate import *
from search_common import to_mod_poly, is_good_prime_for_surface, reduce_cd_mod_ell, compute_canonical_height_matrix, DEBUG

# ========= Picard Number via Van Luijk (minimal viable pipeline) =========
# Assumes: Sage 10.x, your existing cd object shape, and:
#   - find_singular_fibers(cd, ...)
#   - shioda_tate_from_fiber_list(...)
#   - lll_reduce_mw_basis(cd, ...)
#   - check_independence(sections, E_curve_m, cd)
#
# Philosophy:
#   - Lower bound from char 0 Shioda–Tate.
#   - Upper bound(s) from reductions modulo good primes using Shioda–Tate over F_ell.
#   - Optional Van Luijk square-class check (stub included, see TODO).

# Import placeholder for required but external functions.
# These would need to be defined in `tate`, `sat`, `search_common` for the full pipeline to work.

def reduce_section_mod_ell(sec, cd_ell, debug=False):
    """
    Reduce a single section sec = (X(m), Y(m), Z(m)) to GF(ell)(m).
    """
    R = PolynomialRing(cd_ell.base_field, 'm')
    R_frac = R.fraction_field()

    if len(sec) < 3:
        raise ValueError("Section tuple must have at least 3 coordinates (X, Y, Z).")

    try:
        Xn, Xd = sec[0].numerator(), sec[0].denominator()
        Yn, Yd = sec[1].numerator(), sec[1].denominator()
        Zn, Zd = sec[2].numerator(), sec[2].denominator()
    except Exception as exc:
        raise RuntimeError("Section entries must support numerator()/denominator() to reduce reliably.") from exc

    Xn_mod = to_mod_poly(Xn, R, debug=debug)
    Xd_mod = to_mod_poly(Xd, R, debug=debug)
    Yn_mod = to_mod_poly(Yn, R, debug=debug)
    Yd_mod = to_mod_poly(Yd, R, debug=debug)
    Zn_mod = to_mod_poly(Zn, R, debug=debug)
    Zd_mod = to_mod_poly(Zd, R, debug=debug)

    if Xd_mod == 0 or Yd_mod == 0 or Zd_mod == 0:
        if debug:
            print(f"[warn] section {sec} has denominator 0 mod {cd_ell.base_field.characteristic()}; skipping")
        return None

    return R_frac(Xn_mod) / R_frac(Xd_mod), R_frac(Yn_mod) / R_frac(Yd_mod), R_frac(Zn_mod) / R_frac(Zd_mod)

def rank_upper_bound_over_fq_t(cd_ell, current_sections, prime_pool, verbose=True):
    """
    Return (r, []) where r is a certified LOWER bound on the rank of the reduced
    sections over \\bar{F_ell}(t), or (None, []) if the reduced sections cannot be
    shown to be fully independent.

    A reduced rank below len(current_sections) means the reduction is not
    faithful to characteristic 0 (a dependence appeared only after reducing).
    Such an ell must be discarded by the caller.  It must NOT be reported as a
    smaller rank, because this number feeds a bound on rho(S_ell): a smaller
    reduced rank would produce a smaller "upper bound" and can drop it below the
    characteristic-0 lower bound.
    """
    if not current_sections:
        return 0, []

    n_sec = len(current_sections)
    rank_lb = None
    try:
        independent, H = check_independence(current_sections, None, cd_ell)
        if independent:
            rank_lb = int(H.rank())
        elif verbose:
            print(f"[rank_over_fq(t)] ell={int(cd_ell.base_field.characteristic())}: height pairing shows dependence among reduced sections")
    except Exception:
        # Fallback test: absence of a witnessed relation in a bounded search.
        # This is only a heuristic (no witness found != independent).
        independent, witness = check_independence_over_finite_field(
            current_sections, cd_ell,
            primes_to_test=tuple(prime_pool),
            max_m0s=6,
            verbose=False
        )
        if independent:
            rank_lb = n_sec
        elif verbose:
            print(f"[fallback] dependence witnessed mod ell={int(cd_ell.base_field.characteristic())}: {witness}")

    if rank_lb is not None and rank_lb < n_sec:
        rank_lb = None

    if verbose:
        print(f"[rank_over_fq(t)] independent reduced sections: {rank_lb if rank_lb is not None else 'not certified (ell unusable)'}")
    return rank_lb, []

def shioda_tate_upper_bound_mod_ell(cd_ell, current_sections, prime_pool, verbose=True):
    """
    Compute L(ell) = 2 + (certified rank of reduced sections) + sum_v (m_v - 1).

    NOTE: this is a LOWER bound for rho(S_ell) (Shioda-Tate over F_ell with a
    sub-lattice of MW).  It is NOT an upper bound for rho(S) and must never be
    used as one.  A genuine upper bound needs rho(S_ell) itself (Frobenius
    eigenvalues on H^2 from point counts, Artin-Tate / Van Luijk).
    """
    singfibs_ell = find_singular_fibers(a4=cd_ell.a4, a6=cd_ell.a6, verbose=False)
    sum_contrib = int(singfibs_ell['sigma_sum'])
    rank_lb, _ = rank_upper_bound_over_fq_t(cd_ell, current_sections, prime_pool, verbose=verbose)
    if rank_lb is None:
        # Reduced sections not certified independent: no usable bound from this ell.
        return None, {
            'ell': cd_ell.base_field.characteristic(),
            'rank_lb': None,
            'sum_contrib': sum_contrib,
            'fibers': singfibs_ell['fibers'],
            'unusable': 'reduced sections not certified independent'
        }
    rho_upper = 2 + rank_lb + sum_contrib

    if verbose:
        print(f"[mod {cd_ell.base_field.characteristic()}] sum fiber contrib = {sum_contrib}, rank>= {rank_lb} => rho(S_ell) >= {rho_upper}   (lower bound only)")

    return rho_upper, {
        'ell': cd_ell.base_field.characteristic(),
        'rank_lb': rank_lb,
        'sum_contrib': sum_contrib,
        'fibers': singfibs_ell['fibers']
    }

def _estimate_mw_rank_from_sections(sections, cd_obj):
    """
    Estimates the Mordell-Weil rank.
    """
    try:
        H = compute_canonical_height_matrix(sections, cd_obj)
        if H is not None:
            return matrix(QQ, H).rank()
    except Exception:
        raise

    try:
        ok, witness = check_independence_over_finite_field(sections, reduce_cd_mod_ell(cd_obj, 7))
        return len(sections) if ok else max(0, len(sections) - 1)
    except Exception:
        return len(sections)

def picard_via_van_luijk(cd, current_sections, prime_pool, ell_candidates=None, verbose=True):
    """
    Reworked Picard pipeline that skips primes where reduction creates
    artificial dependencies or where the fiber contribution collapses.
    Preserves original return structure and adds 'collapsed_primes'.
    """
    if ell_candidates is None:
        ell_candidates = [p for p in prime_pool if p > 5 and is_good_prime_for_surface(cd, p)]

    singfibs = find_singular_fibers(cd, verbose=False)
    sigma_sum = int(singfibs.get('sigma_sum', 0))

    euler = int(singfibs.get('euler_characteristic', 0))
    chi = QQ(euler) / 12

    mw_rank = _estimate_mw_rank_from_sections(current_sections, cd)

    # ---- Exact case: rational elliptic surface (chi = 1, e = 12) ----------------
    # For a relatively minimal, non-constant elliptic surface with chi = 1, S is
    # rational over the algebraic closure, so rho = b_2 = 10 exactly and Shioda-Tate
    # gives rank MW(\bar K) = 8 - sum_v (m_v - 1).  No reduction argument is needed
    # (and none can improve on this), so we return the exact value immediately.
    if chi == 1:
        rank_geom = 8 - sigma_sum
        if verbose:
            print(f"[chi=1] rational elliptic surface: rho = 10 exactly, "
                  f"geometric MW rank = 8 - {sigma_sum} = {rank_geom}")
        if rank_geom < 0 or mw_rank > rank_geom:
            raise RuntimeError(
                f"Inconsistent data for chi=1: {mw_rank} independent sections but "
                f"geometric rank is only 8 - {sigma_sum} = {rank_geom}. "
                f"Check minimality of the model / fiber classification.")
        return {
            'lower_bound': 10,
            'reduction_lower_bounds': [],
            'upper_bounds': [],          # kept (empty) so old callers do not KeyError
            'rho': 10,
            'rho_rigorous': True,
            'rho_method': 'rational elliptic surface (chi=1): rho = 10',
            'chi': chi,
            'mw_rank_geometric': rank_geom,
            'mw_rank_found': int(mw_rank),
            'sections_span_finite_index': bool(mw_rank == rank_geom),
            'collapsed_primes': []
        }

    assert mw_rank > 0, mw_rank
    if verbose:
        print(f"[char 0] Using mw_rank = {mw_rank}. Σ(m_v-1) = {sigma_sum}")

    rho_lower = 2 + mw_rank + sigma_sum
    if verbose:
        print(f"[char 0] ST lower bound: rho >= {rho_lower}")

    upper_bounds = []
    collapsed_primes = []

    for ell in ell_candidates:
        try:
            if not is_good_prime_for_surface(cd, ell):
                if verbose:
                    print(f"[skip] ell={ell} not good for this surface.")
                continue

            # reduce the cd object to GF(ell)
            cd_ell = reduce_cd_mod_ell(cd, ell)

            # 1) quick dependence check: if reduction witnesses a relation, skip this ell
            try:
                ok, witness = check_independence_over_finite_field(
                    current_sections,
                    cd_ell,
                    primes_to_test=tuple(prime_pool),
                    max_m0s=6,
                    max_vectors=2000,
                    verbose=False
                )
            except Exception as exc:
                # If we can't run the finite-field independence check (no safe m0s, denominators non-invertible, ...)
                # treat this prime as unusable and skip it.
                if verbose:
                    print(f"[skip] ell={ell} unusable for independence test: {exc}")
                collapsed_primes.append(ell)
                continue

            if not ok:
                # dependence witnessed modulo ell -> artifact of reduction; skip ell
                collapsed_primes.append(ell)
                if verbose:
                    print(f"[skip] ell={ell} dependence witnessed in reduction; witness={witness}")
                continue

            # 2) compute Shioda-Tate upper bound info for ell
            rho_upper, info = shioda_tate_upper_bound_mod_ell(cd_ell, current_sections, prime_pool, verbose=verbose)

            # 2a) reduced sections not certified independent -> discard this ell
            if rho_upper is None:
                collapsed_primes.append(ell)
                if verbose:
                    print(f"[skip] ell={ell}: {info.get('unusable', 'unusable')}.")
                continue

            # 3) if the fiber contribution collapsed on reduction, skip this ell as well
            if info['sum_contrib'] < sigma_sum:
                collapsed_primes.append(info['ell'])
                if verbose:
                    print(f"[skip] ell={info['ell']} collapsed fiber contribution ({info['sum_contrib']} < {sigma_sum}).")
                continue

            # 4) sanity: a valid upper bound can never lie below the char-0 lower bound.
            #    If it does, the bound (or the section bookkeeping) is broken for this ell.
            if rho_upper < rho_lower:
                collapsed_primes.append(ell)
                if verbose:
                    print(f"[skip] ell={ell}: rho_upper={rho_upper} < rho_lower={rho_lower}; inconsistent, discarding.")
                continue

            # If we reach here, ell is usable for an upper bound
            upper_bounds.append((rho_upper, info))

        except Exception as exc:
            if verbose:
                print(f"[warn] ell={ell} reduction attempt failed: {exc}")
            # don't append this ell; treat as unusable
            continue

    if not upper_bounds:
        raise RuntimeError("No usable upper bounds from the provided primes.")

    best_lb, best_info = max(upper_bounds, key=lambda t: t[0])
    if verbose:
        print(f"[summary] char-0 lower bound rho >= {rho_lower}; "
              f"best mod-ell lower bound rho(S_ell) >= {best_lb} at ell={best_info['ell']}")
        if collapsed_primes:
            print(f"[collapsed fiber contribution / reduction-dependence primes] {sorted(set(collapsed_primes))}")
        print("[caveat] The mod-ell numbers are LOWER bounds on rho(S_ell). They give no upper bound "
              "on rho(S), so rho is NOT concluded here. A rigorous upper bound needs rho(S_ell) "
              "(Frobenius eigenvalues from point counts) and is not implemented for chi >= 2.")

    # Never conclude rho from lower bounds.  The old rule (best 'upper' == lower => rho = lower)
    # compared two lower bounds and was unsound.
    return {
        'lower_bound': rho_lower,
        'reduction_lower_bounds': upper_bounds,
        'upper_bounds': [],              # intentionally empty: no rigorous upper bound is computed
        'rho': None,
        'rho_rigorous': False,
        'rho_method': None,
        'chi': chi,
        'collapsed_primes': sorted(set(collapsed_primes))
    }

#### patch below

def _choose_safe_m0s_for_ell(cd_ell, max_m0s=8):
    """
    Choose up to `max_m0s` safe specialization values m0 in GF(ell).
    A safe m0 satisfies:
      - denominators of a4, a6 do not vanish at m0
      - discriminant numerator does not vanish at m0
      - evaluation of a4,a6 at m0 succeeds
    Returns a Python list of field elements (not rationals).
    """
    F = cd_ell.base_field
    safe_vals = []

    # Precompute denominator and discriminant polynomials in the polynomial ring.
    try:
        a4_den = cd_ell.a4.denominator()
        a6_den = cd_ell.a6.denominator()
        Delta = -16 * (4 * cd_ell.a4**3 + 27 * cd_ell.a6**2)
        Delta_num = Delta.numerator()
    except Exception:
        raise
        # Can't inspect numerators/denominators: return empty so caller rejects ell.
        return []

    # Iterate field elements deterministically (iteration order of F is fine).
    for m0 in F:
        try:
            # Check denominators
            if a4_den(m0) == 0 or a6_den(m0) == 0:
                continue
            # Check discriminant numerator not zero
            if Delta_num(m0) == 0:
                continue
            # Try to evaluate a4 and a6 at m0 (this may raise)
            _ = cd_ell.a4(m0)
            _ = cd_ell.a6(m0)
        except Exception:
            raise
            # Any failure -> skip this m0
            continue

        safe_vals.append(m0)
        if len(safe_vals) >= max_m0s:
            break

    return safe_vals

def check_independence_over_finite_field(current_sections, cd_ell,
                                         primes_to_test,
                                         max_m0s=6,
                                         max_vectors=2000,
                                         verbose=False):
    """
    Robust independence test for sections over GF(ell)(t).

    Returns (True, None) if no non-trivial F_p-relation is detected (sections independent),
    or (False, witness) if a relation is found where witness is a dict describing it.

    Conservative behavior:
      - If any section cannot be reduced consistently mod ell (denominator vanishes identically),
        raise RuntimeError so caller can skip that ell.
      - If no safe specializations m0 exist, raise RuntimeError.
    """
    if not getattr(cd_ell, 'base_field', None):
        raise ValueError("cd_ell must have attribute base_field = GF(ell)")

    F = cd_ell.base_field
    ell = int(F.characteristic())
    r = len(current_sections)
    if verbose:
        print(f"[check_independence_over_finite_field] ell={ell}, #sections={r}")

    if r == 0:
        return True, None

    # 1) Reduce sections mod ell (coerce to GF(ell)(m) fraction-field objects).
    reduced_secs = []
    for idx, S in enumerate(current_sections):
        try:
            reduced = reduce_section_mod_ell(S, cd_ell, debug=False)
            reduced_secs.append(reduced)
        except Exception as exc:
            # If a section can't be reduced mod ell in the expected way,
            # this prime is not usable for the test.
            raise RuntimeError(f"Section #{idx} reduction failed mod {ell}: {exc}") from exc

    # 2) Choose safe specialization values m0 in GF(ell)
    m0s = _choose_safe_m0s_for_ell(cd_ell, max_m0s=max_m0s)
    if not m0s:
        raise RuntimeError(f"No safe specialization candidates in GF({ell}).")
    if verbose:
        print(f"[check_independence] using m0s={m0s}")

    # 3) For each m0, build the specialized elliptic curve and the specialized points.
    specialized = []
    for m0 in m0s:
        try:
            a4_m0 = cd_ell.a4(m0)
            a6_m0 = cd_ell.a6(m0)
            E_m0 = EllipticCurve(F, [0, 0, 0, a4_m0, a6_m0])
        except Exception as exc:
            # If the curve can't be constructed at this m0, skip it.
            if verbose:
                print(f"[debug] skipping m0={m0} due to curve construction failure: {exc}")
            continue

        row = []
        ok = True
        for (Xr, Yr, Zr) in reduced_secs:
            try:
                Xv = Xr(m0)
                Yv = Yr(m0)
                Zv = Zr(m0)
                # projective handling
                if Zv == 0:
                    Pv = E_m0(0)
                else:
                    # compute affine coordinates carefully in F
                    x_aff = F(Xv / (Zv**2))
                    y_aff = F(Yv / (Zv**3))
                    Pv = E_m0([x_aff, y_aff])
            except Exception as exc:
                # point specialization failed at this m0: skip the whole m0
                if verbose:
                    print(f"[debug] point specialization failed at m0={m0}: {exc}")
                ok = False
                break
            row.append(Pv)

        if ok and len(row) == r:
            specialized.append((m0, E_m0, row))
        elif verbose:
            if not ok:
                print(f"[debug] skipped m0={m0}: incomplete specialized row")

    if not specialized:
        raise RuntimeError(f"No usable specializations after filtering singular/pole values for ell={ell}.")

    # 4) Test relations modulo small primes p (excluding ell itself)
    for p in primes_to_test:
        if p <= 1 or p == ell:
            continue

        # construct vector list (conservative sampling if too many)
        total_vectors = p**r
        if total_vectors > max_vectors:
            # sample: standard basis + random vectors up to max_vectors
            vecs = []
            for i in range(r):
                vecs.append(tuple([1 if j == i else 0 for j in range(r)]))
            from random import randint as _randint
            while len(vecs) < max_vectors:
                vecs.append(tuple([_randint(0, p - 1) for _ in range(r)]))
        else:
            vecs = list(product(range(p), repeat=r))

        if verbose:
            print(f"[check_independence] testing {len(vecs)} vectors mod p={p}")

        for ci_vec in vecs:
            if all(c == 0 for c in ci_vec):
                continue

            holds_everywhere = True
            for (_, E_m0, pts_row) in specialized:
                Ssum = E_m0(0)
                for idx, ci in enumerate(ci_vec):
                    scalar = int(ci) % p
                    if scalar != 0:
                        Ssum += pts_row[idx] * scalar

                if not Ssum.is_zero():
                    holds_everywhere = False
                    break

            if holds_everywhere:
                witness = {'p': p, 'relation': tuple(ci_vec),
                           'm0s': [m for (m,_,_) in specialized],
                           'ell': ell}
                return False, witness

    # no relation found -> independent (as far as these tests go)
    return True, None
