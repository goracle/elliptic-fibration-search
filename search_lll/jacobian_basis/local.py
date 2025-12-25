"""Local (p-adic) height functions."""

import math
import warnings
from sage.all import QQ, Qp, PolynomialRing, HyperellipticCurve

from search_common import NUM_DOUBLINGS


"""Local (p-adic) height functions."""


def _pairs_to_sage_poly(pairs, p, prec):
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



def get_bad_primes(f_coeffs):
    """
    Identify primes of bad reduction for the curve y^2 = f(x).
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
    assert ret, ret
    return ret
get_bad_primes.cache = {}

def local_naive_height_p(div, p):
    """
    Compute naive local height at p: -min(v_p(coeffs)) * log(p).
    Strictly uses ONLY u_poly coefficients.
    """
    # Extract Mumford polynomials u, v
    u_poly = None
    try:
        if hasattr(div, 'mumford_representation'):
            u_poly, _ = div.mumford_representation()
        else:
            # Fallback for raw list/tuple input
            u_poly, _ = div[0], div[1]
    except Exception:
        # If both fail, try direct indexing as last resort
        try:
            u_poly = div[0]
        except Exception as e:
            raise RuntimeError(f"Could not extract u_poly from div: {e}")
        raise

    coeffs = u_poly.list() 
    
    vals = []
    for c in coeffs:
        if c == 0:
            continue
        
        # Robust valuation check
        val = None
        # 1. Try p-adic / generic valuation (no args)
        try:
            val = c.valuation()
        except (TypeError, ValueError, AttributeError):
            raise
            
        # 2. Try Integer/Rational syntax (requires p)
        if val is None:
            try:
                val = c.valuation(p)
            except (TypeError, ValueError, AttributeError):
                raise
        
        # 3. Last resort: cast to QQ
        if val is None:
            try:
                val = QQ(c).valuation(p)
            except Exception:
                raise RuntimeError(f"Cannot compute valuation for coeff {c} type {type(c)}")
        
        vals.append(val)

    if not vals:
        return 0.0
        
    min_val = min(vals)
    if math.isinf(min_val):
        return 0.0

    ret = -float(min_val) * math.log(p)
    return ret

def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute local canonical height correction (finite p).
    """
    import math
    from sage.all import QQ

    key = (str(div), p, tuple(f_coeffs), num_doublings, padic_prec)
    if key in local_height_correction_finite.cache:
        return local_height_correction_finite.cache[key]

    if p == 2:
        MIN_PADIC_PREC = 4096*8
        MAX_RETRIES = 5
    else:
        MIN_PADIC_PREC = 1024
        MAX_RETRIES = 3

    MAX_ACCEPTABLE_MAG = 1e8
    REL_VAR_TOL = 1e-4
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

        try:
            u_in = div[0].list()
            v_in = div[1].list()
            # Coerce the input coefficients (which should be QQ) to K (Qp)
            u_p = R([K(c) for c in u_in])
            v_p = R([K(c) for c in v_in])
        except Exception as e:
            raise RuntimeError(f"reconstruction_fail at p={p}: {e}")

        P = J_p([u_p, v_p])

        h0 = local_naive_height_p(P, p)

        s_values = []
        current_P = P
        
        for k in range(0, num_doublings_local + 1):
            if current_P.is_zero():
                # Torsion / Zero detected explicitly
                assert None, "TORSION"
                return float(0.0 - h0), "torsion_hit"

            h_k = local_naive_height_p(current_P, p)
            s_k = (4.0 ** (-k)) * float(h_k)
            
            # EFFECTIVE TORSION CHECK:
            # If h_k is exactly 0.0 (good reduction) after doubling, we might be in the identity disk.
            # If we started with bad reduction (h0 != 0) and jumped to 0, that's effectively torsion.
            if k > 0 and abs(h_k) < 1e-12 and abs(h0) > 1e-9:
                return float(0.0 - h0), "effective_torsion"

            if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                raise ValueError("huge_or_nan")

            s_values.append(s_k)

            if k < num_doublings_local:
                current_P = 2 * current_P

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
            raise ValueError(f"unstable_tail_std={std:.2e}_val={mean:.4f}")

        return float(mean - float(h0)), "ok"

    # Escalation loop
    attempt = 0
    current_prec = padic_prec
    cur_num_doublings = num_doublings
    last_reason = "unknown"

    while attempt <= MAX_RETRIES:
        try:
            res, reason = _attempt(current_prec, cur_num_doublings)
            local_height_correction_finite.cache[key] = res
            return res
        except (ValueError, ArithmeticError, RuntimeError) as e:
            # We only catch math/precision errors to retry. 
            last_reason = str(e)
            current_prec *= 2
            cur_num_doublings += 5
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError(f"local_height_correction_finite failed at p={p} after {attempt} attempts. Last reason: {last_reason}")
            # Loop continues to next attempt

    raise RuntimeError("Unreachable code")

local_height_correction_finite.cache = {}

def _pairs_to_qq_poly(pairs):
    """
    Reconstruct a Sage Polynomial over QQ from (num, den) pairs.
    """
    from sage.all import QQ, PolynomialRing
    R = PolynomialRing(QQ, 'x')
    coeffs = []
    for (num, den) in pairs:
        if den == 0:
            coeffs.append(QQ(0))
        else:
            coeffs.append(QQ(num) / QQ(den))
    return R(coeffs)

def local_correction_worker(args):
    """
    Worker function. 
    CRITICAL CHANGE: Do NOT convert to Qp here.
    Reconstruct as QQ polynomials and pass to local_height_correction_finite.
    This ensures that local_height_correction_finite can escalate precision
    starting from exact data, rather than starting from truncated Qp data.
    """
    idx, div_repr, p, f_coeffs = args
    # Note: we do not set padic_prec_worker here anymore because we are not coercing to Qp yet.
    
    u_pairs, v_pairs = div_repr
    u_qq = _pairs_to_qq_poly(u_pairs)
    v_qq = _pairs_to_qq_poly(v_pairs)

    div_for_call = [u_qq, v_qq]
    # Function will handle Qp coercion with correct high precision (e.g. 32k for p=2)
    val = local_height_correction_finite(div_for_call, p, list(f_coeffs))
    return (idx, val)

# module-level sets (ensure they exist)
local_height_correction_finite.warned_primes = getattr(local_height_correction_finite, "warned_primes", set())
# BUG FIX: failed_pairs must be a dictionary {}, not a set()
local_height_correction_finite.failed_pairs = getattr(local_height_correction_finite, "failed_pairs", {})

