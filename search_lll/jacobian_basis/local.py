"""Local (p-adic) height functions."""

import math
import warnings
from sage.all import QQ, Qp, PolynomialRing, HyperellipticCurve

from search_common import NUM_DOUBLINGS


"""Local (p-adic) height functions."""


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
    # Ensure f_coeffs are treated consistently (High->Low is standard in this codebase)
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    
    bad = set()
    
    # 1. Discriminant factors
    disc = f_poly.discriminant()
    if disc != 0:
        bad.update(QQ(disc).numerator().prime_factors())
        bad.update(QQ(disc).denominator().prime_factors())
    
    # 2. Leading coefficient factors
    lc = f_coeffs[0]
    if lc != 0:
        bad.update(QQ(lc).numerator().prime_factors())
        bad.update(QQ(lc).denominator().prime_factors())
        
    bad.add(2)
    
    ret = sorted(list(bad))
    get_bad_primes.cache[key] = ret
    return ret
get_bad_primes.cache = {}

def local_naive_height_p(div, p):
    """
    Compute naive local height at p: -min(v_p(coeffs)) * log(p).
    Robustly handles Sage Integer/Rational vs Qp element syntax.
    """
    # Extract Mumford polynomials u, v
    # Note: div might be a tuple/list of polynomials or a Jacobian element
    try:
        if hasattr(div, 'mumford_representation'):
            u_poly, v_poly = div.mumford_representation()
        else:
            u_poly, v_poly = div[0], div[1]
    except Exception:
        # Fallback for raw list/tuple input
        u_poly, v_poly = div[0], div[1]

    coeffs = u_poly.list() + v_poly.list()
    
    vals = []
    for c in coeffs:
        if c == 0:
            continue
        
        # Robust valuation check
        try:
            # Try Qp syntax first (no arguments)
            vals.append(c.valuation())
        except (TypeError, ValueError, AttributeError):
            try:
                # Try Integer/Rational syntax (requires p)
                vals.append(c.valuation(p))
            except Exception:
                # Last resort: try casting to QQ (if possible)
                try:
                    vals.append(QQ(c).valuation(p))
                except Exception:
                    raise RuntimeError(f"Cannot compute valuation for coeff {c} type {type(c)}")

    if not vals:
        return 0.0
        
    min_val = min(vals)
    # Check for infinity (0 in Qp to precision limits)
    if math.isinf(min_val):
        return 0.0

    ret = -float(min_val) * math.log(p)
    return ret

def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute local canonical height correction (finite p).
    Returns float mu_p.
    """
    import math
    from sage.all import QQ
    
    key = (str(div), p, tuple(f_coeffs), num_doublings, padic_prec)
    if key in local_height_correction_finite.cache:
        return local_height_correction_finite.cache[key]

    # Tunables
    if p == 2:
        # Massive precision for p=2
        MIN_PADIC_PREC = 4096 
        MAX_PADIC_PREC = 65536
        MAX_RETRIES = 5
    else:
        MIN_PADIC_PREC = 1024
        MAX_PADIC_PREC = 8192
        MAX_RETRIES = 3

    MAX_ACCEPTABLE_MAG = 1e8
    REL_VAR_TOL = 1e-4 # Relaxed slightly
    MIN_TAIL_LEN = 5

    if padic_prec is None:
        mult = 4 if p == 2 else 1
        padic_prec = max(MIN_PADIC_PREC, 100 * max(1, num_doublings)) * mult

    def _attempt(padic_prec_local, num_doublings_local):
        K = Qp(p, prec=padic_prec_local)
        R = PolynomialRing(K, 'x')
        x = R.gen()
        f_poly = sum(K(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
        C_p = HyperellipticCurve(f_poly)
        J_p = C_p.jacobian()

        # Reconstruct div in this ring
        # Expecting div to be [u_poly, v_poly] with coefficients coercible to K
        try:
            u_in = div[0].list()
            v_in = div[1].list()
            u_p = R([K(c) for c in u_in])
            v_p = R([K(c) for c in v_in])
        except Exception as e:
            return None, f"reconstruction_fail: {e}"

        P = J_p([u_p, v_p])

        # Initial naive height
        try:
            h0 = local_naive_height_p(P, p)
        except Exception as e:
             return None, f"h0_fail: {e}"

        s_values = []
        current_P = P
        
        for k in range(0, num_doublings_local + 1):
            # Check for torsion/zero
            # In Qp, is_zero() checks if coeffs are zero to precision.
            if current_P.is_zero():
                # Torsion hit: mu = -h0
                return float(0.0 - h0), "torsion_hit"

            try:
                h_k = local_naive_height_p(current_P, p)
            except Exception as e:
                return None, f"hk_fail_k={k}: {e}"

            s_k = (4.0 ** (-k)) * float(h_k)

            if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                return None, "huge_or_nan"

            s_values.append(s_k)

            if k < num_doublings_local:
                try:
                    current_P = 2 * current_P
                except Exception as e:
                    return None, f"doubling_fail_k={k}: {e}"

        # Check convergence
        tail_len = max(MIN_TAIL_LEN, num_doublings_local // 3)
        tail = s_values[-tail_len:]
        
        mean = sum(tail) / float(len(tail))
        var = sum((x - mean) ** 2 for x in tail) / float(len(tail))
        std = math.sqrt(var)
        
        if abs(mean) < 1e-9:
            is_stable = std < 1e-7
        else:
            rel_std = std / abs(mean)
            is_stable = rel_std < REL_VAR_TOL

        if not is_stable:
            return None, f"unstable_tail_std={std:.2e}_val={mean:.4f}"

        return float(mean - float(h0)), "ok"

    # Escalation loop
    attempt = 0
    current_prec = padic_prec
    cur_num_doublings = num_doublings
    last_reason = "unknown"

    while attempt <= MAX_RETRIES:
        res, reason = _attempt(current_prec, cur_num_doublings)
        if res is not None:
            local_height_correction_finite.cache[key] = res
            return res
        
        last_reason = reason
        # Escalate
        current_prec *= 2
        cur_num_doublings += 5
        attempt += 1

    # If we fail, we MUST raise to prevent silent 0.0 results in heights.py
    raise RuntimeError(f"local_height_correction_finite failed at p={p} after {attempt} attempts. Last reason: {last_reason}")

local_height_correction_finite.cache = {}

def _pairs_to_sage_poly(pairs, p, prec):
    """
    Convert coefficient pairs [(num, den), ...] to a Sage polynomial over Qp(p, prec).
    """
    from sage.all import QQ, Qp, PolynomialRing
    K = Qp(p, prec=prec)
    R = PolynomialRing(K, 'x')
    coeffs = []
    for (num, den) in pairs:
        if den == 0:
            coeffs.append(K(0))
        else:
            coeffs.append(K(QQ(num) / QQ(den)))
    return R(coeffs)

def local_correction_worker(args):
    """
    Worker function.
    """
    idx, div_repr, p, f_coeffs = args
    try:
        # Reconstruct as Sage polynomials
        padic_prec_worker = 4096 if p == 2 else 2048
        u_pairs, v_pairs = div_repr
        u_p = _pairs_to_sage_poly(u_pairs, p, padic_prec_worker)
        v_p = _pairs_to_sage_poly(v_pairs, p, padic_prec_worker)

        div_for_call = [u_p, v_p]
        val = local_height_correction_finite(div_for_call, p, list(f_coeffs))
        return (idx, val)
    except Exception as e:
        # Return exception to main process to handle/log
        return (idx, e)

# module-level sets (ensure they exist)
local_height_correction_finite.warned_primes = getattr(local_height_correction_finite, "warned_primes", set())
# BUG FIX: failed_pairs must be a dictionary {}, not a set()
local_height_correction_finite.failed_pairs = getattr(local_height_correction_finite, "failed_pairs", {})

