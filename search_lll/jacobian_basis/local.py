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
    key = (str(div), p)
    if key in local_naive_height_p.cache:
        return local_naive_height_p.cache[key]

    # Extract Mumford polynomials u, v
    u_poly, v_poly = div[0], div[1]
    # Note: u is typically monic, so '1' is implicitly in the coefficients.
    # We explicitly trust u_poly.list() to include it if degree matches.
    coeffs = u_poly.list() + v_poly.list()
    
    vals = []
    for c in coeffs:
        if c == 0:
            continue
        try:
            vals.append(c.valuation(p))
        except AttributeError:
            vals.append(c.valuation())
            raise
            
    if not vals:
        return 0.0
        
    min_val = min(vals)
    ret = -min_val * math.log(p)
    local_naive_height_p.cache[key] = ret
    return ret
local_naive_height_p.cache = {}


def local_height_correction_finite(div, p, f_coeffs, num_doublings=NUM_DOUBLINGS, padic_prec=None):
    """
    Compute local canonical height correction (finite p) with robust safety.
    
    This computes lim_{n->inf} 4^{-n} h(2^n P) - h(P).
    """
    import math
    from sage.all import QQ
    
    key = (str(div), p, tuple(f_coeffs), num_doublings, padic_prec)
    if key in local_height_correction_finite.cache:
        return local_height_correction_finite.cache[key]

    # Tunables
    # p=2 requires significantly higher precision
    if p == 2:
        MIN_PADIC_PREC = 32768*4
        MAX_PADIC_PREC = 131072*4
        MAX_RETRIES = 5
    else:
        MIN_PADIC_PREC = 8192
        MAX_PADIC_PREC = 32768
        MAX_RETRIES = 3

    MAX_ACCEPTABLE_MAG = 1e8
    REL_VAR_TOL = 1e-5
    MIN_TAIL_LEN = 4

    if padic_prec is None:
        mult = 2 if p == 2 else 1
        padic_prec = max(MIN_PADIC_PREC, 60 * max(1, num_doublings)) * mult

    def _attempt(padic_prec_local, num_doublings_local):
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

        # Initial naive height
        h0 = local_naive_height_p(P, p)

        s_values = []
        current_P = P
        
        # Run Tate's limit: s_k = 4^{-k} * h(2^k P)
        for k in range(0, num_doublings_local + 1):
            if current_P.is_zero():
                # Torsion hit: canonical local height is 0.
                mu_exact = float(0.0 - h0)
                return mu_exact, "torsion_hit"

            h_k = local_naive_height_p(current_P, p)
            s_k = (4.0 ** (-k)) * float(h_k)

            if math.isnan(s_k) or math.isinf(s_k) or abs(s_k) > MAX_ACCEPTABLE_MAG:
                return None, "huge_or_nan"

            s_values.append(s_k)

            if k < num_doublings_local:
                current_P = 2 * current_P

        # Check convergence of the tail
        tail_len = max(MIN_TAIL_LEN, num_doublings_local // 2)
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
            return None, f"unstable_tail_std={std:.2e}"

        return float(mean - float(h0)), "ok"

    # Escalation loop
    attempt = 0
    current_prec = padic_prec
    cur_num_doublings = num_doublings
    last_reason = "unknown"

    while attempt <= MAX_RETRIES:
        try:
            mu_val, reason = _attempt(current_prec, cur_num_doublings)
            if mu_val is not None:
                local_height_correction_finite.cache[key] = mu_val
                return mu_val
            last_reason = reason
        except Exception as e:
            last_reason = f"exception_{type(e).__name__}"
            raise

        # Escalate parameters
        if attempt < MAX_RETRIES:
            current_prec = min(MAX_PADIC_PREC, current_prec * 2)
            cur_num_doublings = min(cur_num_doublings + 8, 50)
        
        attempt += 1

    raise RuntimeError(f"local_height_correction_finite failed at p={p} after {attempt} attempts. Last reason: {last_reason}")

local_height_correction_finite.cache = {}


def _pairs_to_sage_poly(pairs, p, prec):
    """
    Convert coefficient pairs [(num, den), ...] to a Sage polynomial over Qp(p, prec).
    pairs are given in order corresponding to poly.list() (highest to lowest degree).
    """
    from sage.all import QQ, Qp, PolynomialRing
    K = Qp(p, prec=prec)
    R = PolynomialRing(K, 'x')
    coeffs = []
    for (num, den) in pairs:
        # build rational then embed into K
        if den == 0:
            # defensive
            coeffs.append(K(0))
        else:
            coeffs.append(K(QQ(num) / QQ(den)))
    return R(coeffs)

def local_correction_worker(args):
    """
    Worker function for parallelizing local height corrections.
    Accepts args = (idx, div_repr, p, f_coeffs)
    where div_repr is either:
      - ((u_pairs),(v_pairs)) with pairs=(num,den),
      - or a fallback string (rare)
    Returns (index, value) or (index, Exception-like-object).
    """
    idx, div_repr, p, f_coeffs = args
    try:
        # If div_repr is a string fallback, try to fail gracefully
        if isinstance(div_repr, str):
            raise RuntimeError("received string-serialized-div; unexpected path")

        # reconstruct divisor as Sage polynomials inside the worker
        # choose a moderate precision for reconstruction (let local_height handle escalation)
        padic_prec_worker = 2048 if p != 2 else 8192
        u_pairs, v_pairs = div_repr
        u_p = _pairs_to_sage_poly(u_pairs, p, padic_prec_worker)
        v_p = _pairs_to_sage_poly(v_pairs, p, padic_prec_worker)

        # now call the main function using the Sage representation
        # local_height_correction_finite will create its own Qp with its intended precision,
        # but providing u_p,v_p as Sage polynomials is also ok if you refactor local_height_correction_finite.
        # To avoid double-reconstruction, call an alternative helper that accepts coeff pairs;
        # use local_height_correction_finite by reconstructing a simple "div" container:
        div_for_call = [u_p, v_p]
        val = local_height_correction_finite(div_for_call, p, list(f_coeffs))
        return (idx, val)
    except Exception as e:
        # Avoid including heavy Sage objects in the error message/traceback
        msg = f"worker_error_{type(e).__name__}: {str(e)[:200]}"
        raise
        return (idx, RuntimeError(msg))


# module-level sets (ensure they exist)
local_height_correction_finite.warned_primes = getattr(local_height_correction_finite, "warned_primes", set())
# BUG FIX: failed_pairs must be a dictionary {}, not a set()
local_height_correction_finite.failed_pairs = getattr(local_height_correction_finite, "failed_pairs", {})

