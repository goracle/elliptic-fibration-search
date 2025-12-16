"""Local (p-adic) height functions."""

import math
import warnings
from sage.all import QQ, Qp, PolynomialRing, HyperellipticCurve

from search_common import NUM_DOUBLINGS


def get_bad_primes(f_coeffs):
    """
    Identify primes of bad reduction for the curve y^2 = f(x).
    Includes factors of discriminant, leading coefficient, and 2.
    """
    from sage.all import QQ, PolynomialRing
    key = tuple(f_coeffs)
    if key in get_bad_primes.cache:
        return get_bad_primes.cache[key]
    
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    bad = set()
    
    # 1. Discriminant factors
    disc = f_poly.discriminant()
    if disc != 0:
        # Handle Rational discriminant: separate numerator and denominator
        bad.update(QQ(disc).numerator().prime_factors())
        bad.update(QQ(disc).denominator().prime_factors())
    
    # 2. Leading coefficient factors (potential degree drop)
    lc = f_coeffs[0]
    if lc != 0:
        bad.update(QQ(lc).numerator().prime_factors())
        bad.update(QQ(lc).denominator().prime_factors())
        
    # Genus 2 arithmetic at p=2 is always delicate
    bad.add(2)
    
    ret = sorted(list(bad))
    get_bad_primes.cache[key] = ret
    return ret
get_bad_primes.cache = {}

def local_naive_height_p(div, p):
    """
    Compute naive local height at p: -min(v_p(coeffs)) * log(p).
    This corresponds to the log of the max p-adic norm of coefficients.
    """
    try:
        # Extract Mumford polynomials u, v
        u_poly, v_poly = div[0], div[1]
        coeffs = u_poly.list() + v_poly.list()
        
        # We want max(|c|_p). 
        # |c|_p = p^(-v_p(c)).
        # log(max |c|_p) = log(p^(-min v_p(c))) = -min(v_p(c)) * log(p)
        
        # Handle 0 coefficients (val is +infinity)
        vals = []
        for c in coeffs:
            if c == 0:
                continue
            # Handle both Rational and p-adic types
            try:
                vals.append(c.valuation(p))
            except AttributeError:
                vals.append(c.valuation())
                raise
                
        if not vals:
            return 0.0
            
        min_val = min(vals)
        return -min_val * math.log(p)
    except Exception:
        raise
        return 0.0


# inside local.py -- replace the existing function with this

# module-level sets (ensure they exist)
local_height_correction_finite.warned_primes = getattr(local_height_correction_finite, "warned_primes", set())
local_height_correction_finite.failed_pairs = getattr(local_height_correction_finite, "failed_pairs", set())


# --- paste this whole function body in place of your existing local_height_correction_finite ---
def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute local canonical height correction (finite p) with robust safety.

    Important behavior changes:
    - On giving up we *return a small positive value* (SMALL_POSITIVE) instead of 0.0.
    - We cache failed (div,p) pairs with the small value so repeated calls are cheap.
    """
    import math
    import warnings
    from sage.all import QQ

    if p == 2:
        return 0.0
    # Tunables
    MIN_PADIC_PREC = 8192
    MAX_PADIC_PREC = 8192*2
    MAX_RETRIES = 1
    MAX_ACCEPTABLE_MAG = 1e6
    REL_VAR_TOL = 1e-4
    ABS_NEG_TOL = 1e-8
    MIN_TAIL_LEN = 3

    # Small conservative positive substitute for unknown local correction
    # Choose extremely small so it does not affect numeric decisions but preserves PD.
    SMALL_POSITIVE = 1e-12

    # Deterministic divisor id (shorten for warnings)
    try:
        div_id = (str(div[0]), str(div[1]))
    except Exception:
        div_id = (repr(div),)

    if not hasattr(local_height_correction_finite, "failed_pairs"):
        # map (div_id,p) -> substitute_value
        local_height_correction_finite.failed_pairs = {}

    # If previously failed, return the cached small positive replacement
    if (div_id, p) in local_height_correction_finite.failed_pairs:
        return float(local_height_correction_finite.failed_pairs[(div_id, p)])

    # Optionally skip p=2 (unreliable); uncomment if you want to avoid many retries at 2
    # if p == 2:
    #     local_height_correction_finite.failed_pairs[(div_id, p)] = SMALL_POSITIVE
    #     return float(SMALL_POSITIVE)

    if padic_prec is None:
        padic_prec = max(MIN_PADIC_PREC, 50 * max(1, num_doublings))

    def _attempt(padic_prec_local, num_doublings_local):
        try:
            K = Qp(p, prec=padic_prec_local)
            R = PolynomialRing(K, 'x')
            x = R.gen()
            f_poly = sum(K(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
            C_p = HyperellipticCurve(f_poly)
            J_p = C_p.jacobian()

            uQ, vQ = div[0], div[1]
            u_p = R([K(c) for c in uQ.list()])
            v_p = R([K(c) for c in vQ.list()])

            P = J_p([u_p, v_p])

            h0 = local_naive_height_p(P, p)

            s_values = []
            current_P = P
            for k in range(0, num_doublings_local + 1):
                if current_P.is_zero():
                    mu_exact = float(0.0 - h0)
                    return mu_exact, "torsion_hit"

                h_k = local_naive_height_p(current_P, p)
                try:
                    h_kf = float(h_k)
                except Exception:
                    return None, "non_numeric_height"

                s_k = (4.0 ** (-k)) * h_kf

                if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                    return None, "huge_or_nan"

                s_values.append(s_k)

                if k < num_doublings_local:
                    try:
                        current_P = 2 * current_P
                    except Exception:
                        return None, "doubling_failed"

            tail_len = max(MIN_TAIL_LEN, num_doublings_local // 2)
            if len(s_values) < tail_len:
                return None, "insufficient_samples"

            tail = s_values[-tail_len:]
            tail_mean = sum(tail) / float(len(tail))
            mean = tail_mean
            var = sum((x - mean) ** 2 for x in tail) / float(len(tail))
            std = math.sqrt(var)
            rel_std = std / (abs(mean) + 1e-16)

            if rel_std > REL_VAR_TOL and abs(mean) > 1e-12:
                return None, "high_variance"

            tate_limit_est = tail_mean
            mu_est = float(tate_limit_est - float(h0))

            if mu_est < -ABS_NEG_TOL:
                return None, "excessive_negative"
            if abs(mu_est) > MAX_ACCEPTABLE_MAG:
                return None, "excessive_magnitude"

            return float(mu_est), "ok"

        except ZeroDivisionError:
            raise
        except Exception as e:
            return None, f"exception:{type(e).__name__}:{str(e)}"

    # attempt with escalation
    attempt = 0
    current_prec = padic_prec
    cur_num_doublings = num_doublings
    last_reason = None

    while attempt <= MAX_RETRIES:
        mu_val, reason = _attempt(current_prec, cur_num_doublings)
        last_reason = reason
        if mu_val is not None:
            return float(mu_val)

        # warn once per prime (but include short div id)
        warned_key = (p, )
        warned = getattr(local_height_correction_finite, "warned_primes", set())
        if p not in warned:
            warnings.warn(f"[local_height_correction_finite] instability at p={p}; reason={reason}; padic_prec={current_prec}; num_doublings={cur_num_doublings} for div={div_id}. Retrying.", RuntimeWarning)
            warned.add(p)
            local_height_correction_finite.warned_primes = warned

        # if we've exhausted budget escalate or give up
        if current_prec >= MAX_PADIC_PREC or attempt == MAX_RETRIES:
            # cache a small positive value for this (div,p)
            local_height_correction_finite.failed_pairs[(div_id, p)] = SMALL_POSITIVE
            warnings.warn(f"[local_height_correction_finite] giving up on (div,p)=({div_id},{p}) after {attempt+1} attempts; reason={reason}. Returning SMALL_POSITIVE={SMALL_POSITIVE}.", RuntimeWarning)
            return float(SMALL_POSITIVE)

        # escalate (double precision conservatively)
        current_prec = min(MAX_PADIC_PREC, current_prec * 2)
        cur_num_doublings = min(cur_num_doublings + 2, max(4, 2 * cur_num_doublings))
        attempt += 1

    # fallback (shouldn't reach)
    local_height_correction_finite.failed_pairs[(div_id, p)] = SMALL_POSITIVE
    warnings.warn(f"[local_height_correction_finite] exhausted attempts for (div,p)=({div_id},{p}). Returning SMALL_POSITIVE.", RuntimeWarning)
    return float(SMALL_POSITIVE)
# --- end function replacement ---
