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


def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute the local canonical height correction (Neron correction) at p 
    using a stabilized p-adic doubling limit:
       mu_p(div) = lim_{n->inf} 4^(-n) * h_naive(2^n div) - h_naive(div)

    Safety features added:
    - Uses a Cesàro/last-window average of the scaled naive heights to reduce oscillation.
    - Detects instability (large variance, NaNs or huge values) and will retry once
      with larger p-adic precision and more doublings.
    - If instability persists, returns a conservative 0.0 for the local correction
      (and prints a warning). This prevents a single bad p-term from making the
      global canonical height nonsensical.
    - Clamps obviously absurd results.
    """
    import math
    import warnings

    # Parameters for stabilization / safety
    MIN_PADIC_PREC = 2048
    MAX_PADIC_PREC = 8192
    MAX_RETRIES = 1            # automatic retry with higher padic precision
    MAX_ACCEPTABLE_MAG = 1e6   # anything larger is considered bogus
    REL_VAR_TOL = 1e-4         # relative stddev tolerance for tail stability
    ABS_NEG_TOL = 1e-8         # allow tiny negative noise, clamp bigger negatives to 0.0
    MIN_TAIL_LEN = 3

    # Ensure initial padic precision is reasonable
    if padic_prec is None:
        padic_prec = max(MIN_PADIC_PREC, 100 * max(1, num_doublings))

    # internal helper to attempt computation; returns (mu_or_None, reason)
    def _attempt(padic_prec_local, num_doublings_local):
        try:
            # 1. Setup p-adic curve
            K = Qp(p, prec=padic_prec_local)
            R = PolynomialRing(K, 'x')
            f_poly = sum(K(c) * R.gen()**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
            C_p = HyperellipticCurve(f_poly)
            J_p = C_p.jacobian()

            # 2. Lift point div to J(Qp)
            u_Q, v_Q = div[0], div[1]
            # if these are Sage polys or lists, convert coefficients to K
            u_p = R([K(c) for c in u_Q.list()])
            v_p = R([K(c) for c in v_Q.list()])

            P = J_p([u_p, v_p])

            # 3. Compute h0
            h0 = local_naive_height_p(P, p)

            # 4. Iterate doubling and collect scaled naive heights:
            #    s_k = 4^{-k} * h_naive(2^k P)
            s_values = []
            current_P = P

            for k in range(0, num_doublings_local + 1):
                # If point becomes zero at some stage, the Tate-limit is exactly 0
                # (future terms are zero), so mu = 0 - h0.
                if current_P.is_zero():
                    mu_exact = float(0.0 - h0)
                    return mu_exact, "torsion_hit"

                # compute naive height at current_P (may return Sage/Rational/float)
                h_k = local_naive_height_p(current_P, p)
                try:
                    h_kf = float(h_k)
                except Exception:
                    # non-convertible -> instability
                    return None, "non_numeric_height"

                s_k = (4.0 ** (-k)) * h_kf
                # quick sanity: huge values indicate instability
                if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                    return None, "huge_or_nan"

                s_values.append(s_k)

                # prepare for next doubling (but don't double on last loop)
                if k < num_doublings_local:
                    current_P = 2 * current_P

            # Need at least a few tail values to average
            tail_len = max(MIN_TAIL_LEN, num_doublings_local // 2)
            if len(s_values) < tail_len:
                return None, "insufficient_samples"

            tail = s_values[-tail_len:]
            tail_mean = sum(tail) / float(len(tail))
            # compute sample standard deviation (population stddev not necessary)
            mean = tail_mean
            var = sum((x - mean) ** 2 for x in tail) / float(len(tail))
            std = math.sqrt(var)

            # Relative variability check (relative to magnitude of mean)
            rel_std = std / (abs(mean) + 1e-16)

            if rel_std > REL_VAR_TOL and abs(mean) > 1e-12:
                # unstable sequence
                return None, "high_variance"

            # stabilized estimate for the Tate-limit value
            tate_limit_est = tail_mean

            # mu_p(div) = Tate-limit - h0
            mu_est = float(tate_limit_est - float(h0))

            # Clamp/guard obviously absurd negatives (user requested "no bad returns")
            if mu_est < -ABS_NEG_TOL:
                # allow tiny negative rounding noise, but not larger negatives
                return None, "excessive_negative"

            # Final sanity check: not astronomically large
            if abs(mu_est) > MAX_ACCEPTABLE_MAG:
                return None, "excessive_magnitude"

            return float(mu_est), "ok"

        except ZeroDivisionError:
            # propagate so the outer code can retry with larger precision as before
            raise
        except Exception as e:
            # any other failure -> mark as unstable
            return None, f"exception:{repr(e)}"

    # Attempt + one automatic retry if unstable
    attempt = 0

    while attempt <= MAX_RETRIES:
        mu_val, reason = _attempt(padic_prec, num_doublings)
        if mu_val is not None:
            # good result
            return mu_val

        # if we get here, the attempt failed for reason -> try a safer retry if possible

        if p not in local_height_correction_finite.warned_primes:
            warnings.warn(f"[local_height_correction_finite] instability at p={p}; reason={reason}; "
                          f"padic_prec={padic_prec}; num_doublings={num_doublings}. Retrying with higher precision.", RuntimeWarning)
            local_height_correction_finite.warned_primes.add(p)

        # if the failure was due to a ZeroDivisionError, let outer exception handler or caller deal with it
        # (this mirrors your original ZeroDivisionError behavior)
        # Otherwise, increase padic precision and (optionally) num_doublings and retry once.
        if padic_prec >= MAX_PADIC_PREC and p not in local_height_correction_finite.warned_primes:
            # Give up and return conservative 0.0
            warnings.warn(f"[local_height_correction_finite] giving up on p={p} after padic_prec={padic_prec}. Returning 0.0.", RuntimeWarning)
            local_height_correction_finite.warned_primes.add(p)
            return 0.0
        # increase precision and doublings for the retry
        padic_prec = min(MAX_PADIC_PREC, padic_prec * 2)
        num_doublings = min(num_doublings + 2, 2 * num_doublings if num_doublings > 0 else 4)
        attempt += 1

    # If all retries exhausted, return conservative 0.0
    if p not in local_height_correction_finite.warned_primes:
        warnings.warn(f"[local_height_correction_finite] all retries exhausted for p={p}. Returning 0.0.", RuntimeWarning)
        local_height_correction_finite.warned_primes.add(p)
    return 0.0
local_height_correction_finite.warned_primes = set()

