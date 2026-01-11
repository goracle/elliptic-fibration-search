import time
import sys
from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
from search_common import DEBUG, FINITE_FIELD, PREFERRED_X_COORDS
from .mumford_core import _poly_mod_quad_fast
from .mumford_verification import verify_mumford_pair

assert PREFERRED_X_COORDS, PREFERRED_X_COORDS


from sage.all import GF, PolynomialRing
from random import randrange


assert PREFERRED_X_COORDS, "PREFERRED_X_COORDS must be nonempty"


from sage.all import GF, Integer


def get_sqrt_data_sage(p):
    """
    Precomputes square root map using Sage's native quadratic residue checking.
    MEMORY FIX: Only cache for small primes (p < 2^20).
    """
    if p is None:
        return {}
    
    if p > 1048576:
        return None

    key = p
    if key in get_sqrt_data_sage.cache:
        return get_sqrt_data_sage.cache[key]
    
    Fp = GF(p)
    sqrt_map = {}
    
    for i in range((p // 2) + 1):
        sq = int(Fp(i)**2)
        if sq not in sqrt_map:
            sqrt_map[sq] = []
        sqrt_map[sq].append(i)
        if i != 0 and (p - i) != i:
            sqrt_map[sq].append(p - i)
    
    get_sqrt_data_sage.cache[key] = sqrt_map
    return sqrt_map

get_sqrt_data_sage.cache = {}


def solve_mumford_mod_p_sage_native_RANDOM(f_coeffs, p, x_residue, const_val=0, max_solutions=500):
    """
    Sage-native Mumford solver using polynomial rings directly.
    Handles both small primes (cached sqrt) and large primes (on-demand sqrt).
    """
    Fp = GF(p)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    
    f_poly = R([Fp(c) for c in reversed(f_coeffs)])
    
    x_res = Fp(x_residue)
    x_sq = x_res * x_res
    
    solutions = []
    
    sqrt_map = get_sqrt_data_sage(p)
    use_cached = (sqrt_map is not None)
    
    if p > 100000:
        from random import randrange
        sample_size = min(10000, p)
        s_range = [randrange(p) for _ in range(sample_size)]
    else:
        s_range = range(p)
    
    for s_int in s_range:
        if len(solutions) >= max_solutions:
            break
        
        s_val = Fp(s_int)
        p_val = x_res * s_val - x_sq
        
        disc = s_val * s_val - 4 * p_val
        disc_int = int(disc)
        
        if use_cached:
            if disc_int not in sqrt_map:
                continue
        else:
            if not Fp(disc_int).is_square():
                continue
        
        u_poly = x**2 - s_val*x + p_val
        remainder = f_poly % u_poly
        
        rem_coeffs = remainder.list()
        B = rem_coeffs[0] if len(rem_coeffs) > 0 else Fp(0)
        A = rem_coeffs[1] if len(rem_coeffs) > 1 else Fp(0)
        
        a_q = disc
        b_q = -2 * (A * s_val + 2 * B)
        c_q = A * A
        
        Z_roots = []
        if a_q == 0:
            if b_q != 0:
                Z_roots.append(-c_q / b_q)
            elif c_q == 0:
                Z_roots.append(Fp(0))
        else:
            disc_q = b_q * b_q - 4 * a_q * c_q
            disc_q_int = int(disc_q)
            
            if use_cached:
                if disc_q_int in sqrt_map:
                    inv_2a = 1 / (2 * a_q)
                    for sq_root_int in sqrt_map[disc_q_int]:
                        sq_root = Fp(sq_root_int)
                        Z_roots.append((-b_q + sq_root) * inv_2a)
            else:
                if Fp(disc_q_int).is_square():
                    sqrt_disc_q = Fp(disc_q_int).sqrt()
                    inv_2a = 1 / (2 * a_q)
                    Z_roots.append((-b_q + sqrt_disc_q) * inv_2a)
                    Z_roots.append((-b_q - sqrt_disc_q) * inv_2a)
        
        for Z in set(Z_roots):
            Z_int = int(Z)
            
            if use_cached:
                if Z_int in sqrt_map:
                    for v1_int in sqrt_map[Z_int]:
                        v1_val = Fp(v1_int)
                        
                        if v1_val != 0:
                            v0_val = (A - s_val * Z) / (2 * v1_val)
                            if v0_val * v0_val - p_val * Z == B:
                                solutions.append((int(s_val), int(p_val), 
                                                int(v0_val), int(v1_val)))
                        else:
                            if A == 0 and Z_int == 0 and int(B) in sqrt_map:
                                for r in sqrt_map[int(B)]:
                                    solutions.append((int(s_val), int(p_val), r, 0))
            else:
                if Fp(Z_int).is_square():
                    sqrt_Z = Fp(Z_int).sqrt()
                    
                    for v1_val in [sqrt_Z, -sqrt_Z]:
                        if v1_val != 0:
                            v0_val = (A - s_val * Z) / (2 * v1_val)
                            if v0_val * v0_val - p_val * Z == B:
                                solutions.append((int(s_val), int(p_val), 
                                                int(v0_val), int(v1_val)))
                else:
                    if A == 0 and Z_int == 0:
                        if Fp(int(B)).is_square():
                            sqrt_B = Fp(int(B)).sqrt()
                            for v0_val in [sqrt_B, -sqrt_B]:
                                solutions.append((int(s_val), int(p_val), int(v0_val), 0))
    
    return list(set(solutions))


def score_and_record_candidate(u_poly, Fp, preferred_x_coords, bias_stats):
    """
    u_poly: monic quadratic in Fp[x]
    Fp: the finite field
    preferred_x_coords: iterable of preferred x values (ints or Fp)
    bias_stats: BiasStatistics instance

    Returns: (is_acceptable, roots)
    """

    if bias_stats is None:
        raise ValueError("bias_stats must not be None")

    if u_poly.degree() != 2:
        raise ValueError("u_poly must be quadratic")

    pref = set(int(x) for x in preferred_x_coords)

    roots = []
    try:
        for r, mult in u_poly.roots(Fp):
            for _ in range(mult):
                roots.append(r)
    except Exception as e:
        raise RuntimeError("Root finding failed") from e

    bias_stats.record(roots, pref)

    if len(roots) != 2:
        return False, roots

    preferred_count = 0
    for r in roots:
        if int(r) in pref:
            preferred_count += 1

    if preferred_count not in (0, 1, 2):
        raise RuntimeError("Invalid preferred_count")

    if preferred_count >= 1:
        return True, roots
    else:
        return False, roots


class BiasStatistics:
    """
    Robust statistics tracker for the biased solver.
    Now handles type conversion safely to prevent '0 hits' due to int vs GF element mismatches.
    """
    def __init__(self):
        self.total_candidates = 0
        self.split_count = 0
        self.preferred_hits = 0
        self.preferred_histogram = {0: 0, 1: 0, 2: 0}

    def record(self, roots, preferred_x_set):
        """
        roots: iterable of roots (Sage GF elements or ints)
        preferred_x_set: set of python ints
        """
        self.total_candidates += 1
        
        # Robust type conversion for checking
        current_roots_ints = []
        for r in roots:
            try:
                # Handle Sage field elements, numpy types, strings, etc.
                current_roots_ints.append(int(r)) 
            except (ValueError, TypeError):
                continue

        if len(current_roots_ints) == 2:
            self.split_count += 1

        hit_count = 0
        if preferred_x_set:
            for r_int in current_roots_ints:
                if r_int in preferred_x_set:
                    hit_count += 1
        
        self.preferred_hits += hit_count
        
        # Clamp to 2 for histogram safety
        safe_hit_count = min(hit_count, 2)
        self.preferred_histogram[safe_hit_count] = self.preferred_histogram.get(safe_hit_count, 0) + 1


    def report(self):
        if self.total_candidates == 0:
            return {"status": "no_candidates"}
        split_rate = self.split_count / self.total_candidates if self.total_candidates else 0.0
        hit_rate = (self.preferred_hits / max(1, self.split_count)) if self.split_count else 0.0
        return {
            "total": self.total_candidates,
            "split_rate": split_rate,
            "preferred_histogram": dict(self.preferred_histogram),
            "preferred_hits": self.preferred_hits,
            "hit_rate": hit_rate  # hits per split
        }


assert PREFERRED_X_COORDS, "PREFERRED_X_COORDS must be nonempty"


# ============================================================
# Entry Point
# ============================================================

def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=DEBUG):
    if not isinstance(eqs_dict, dict):
        raise TypeError("eqs_dict must be a dict")

    if 'f_coeffs' not in eqs_dict:
        raise KeyError("eqs_dict missing 'f_coeffs'")

    if not isinstance(p, int) or p <= 2:
        raise ValueError(f"Invalid prime p={p}")

    f_coeffs = eqs_dict['f_coeffs']
    const_val = eqs_dict.get('const', 0)

    try:
        const_val = int(QQ(const_val))
    except Exception:
        raise ValueError("const must coerce to rational integer")

    return solve_mumford_mod_p_optimized(
        f_coeffs=f_coeffs,
        p=p,
        x_residue=x_residue,
        const_val=const_val
    )


# ============================================================
# Square Root Cache
# ============================================================

def get_sqrt_data(p):
    if not isinstance(p, int) or p <= 2:
        raise ValueError("p must be an odd prime")

    if p in get_sqrt_data.cache:
        return get_sqrt_data.cache[p]

    sqrt_map = {}
    for i in range((p // 2) + 1):
        sq = (i * i) % p
        sqrt_map.setdefault(sq, []).append(i)
        if i != 0:
            j = p - i
            if j != i:
                sqrt_map[sq].append(j)

    get_sqrt_data.cache[p] = sqrt_map
    return sqrt_map


get_sqrt_data.cache = {}


# ============================================================
# Mumford Doubling (STRICT)
# ============================================================

def _mumford_doubling_mod_p_internal(u_coeffs, v_coeffs, f_coeffs, p, debug=False):
    if not isinstance(p, int) or p <= 2:
        raise ValueError("p must be an odd prime")

    if not u_coeffs or not v_coeffs:
        raise ValueError("Empty Mumford coefficients")

    Fp = GF(p)
    R = PolynomialRing(Fp, 'x')

    try:
        f_poly = R(f_coeffs[::-1])
        C = HyperellipticCurve(f_poly, 0)
        J = C.jacobian()
    except Exception as e:
        raise RuntimeError(f"Failed to construct Jacobian over GF({p}): {e}")

    u = R(u_coeffs[::-1]).monic()
    v = R(v_coeffs[::-1]) % u

    if v.degree() >= u.degree():
        raise ValueError("Invalid Mumford representation: deg(v) >= deg(u)")

    try:
        D = J([u, v])
        D2 = 2 * D
    except Exception as e:
        raise RuntimeError(f"Jacobian doubling failed: {e}")

    u2, v2 = D2

    if not u2.is_monic():
        raise AssertionError("Doubled u-polynomial is not monic")

    return (
        [int(c) for c in u2.list()][::-1],
        [int(c) for c in (v2 % u2).list()][::-1]
    )


# ============================================================
# Algebraic Prefilter (NO SILENCE)
# ============================================================

def prefilter_solutions_algebraic(sol_list, prime, f_coeffs):
    if not isinstance(prime, int) or prime <= 2:
        raise ValueError("prime must be an odd prime")

    Fp = GF(prime)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    f_poly = R(f_coeffs[::-1])

    filtered = []

    for tup in sol_list:
        if len(tup) != 4:
            raise ValueError(f"Invalid solution tuple: {tup}")

        s_val, p_val, v0_val, v1_val = map(Fp, tup)

        u = x**2 - s_val * x + p_val
        v = v1_val * x + v0_val

        if (v*v - f_poly) % u != 0:
            continue

        filtered.append(tup)

    return filtered


# ============================================================
# Prime Filter (STRICT DENOM CHECK)
# ============================================================

def filter_primes_avoiding_denoms(primes_list, divisors):
    if not primes_list:
        raise ValueError("Empty prime list")

    bad_primes = set()

    for d in divisors:
        if not isinstance(d, dict):
            raise TypeError("Each divisor must be a dict")

        for k in ('s', 'p', 'v_0', 'v_1'):
            if k not in d:
                raise KeyError(f"Missing key {k} in divisor")

            q = QQ(d[k])
            den = q.denominator()

            if den == 1:
                continue

            bad_primes |= set(ZZ(den).prime_divisors())

    ret = [p for p in primes_list if p not in bad_primes]

    if not ret:
        raise RuntimeError("All primes eliminated by denominator filtering")

    return ret


# ============================================================
# Batch Solver (STRICT)
# ============================================================

def solve_mumford_batch_sage(f_coeffs, p, x_residues_list, const_val=0, max_solutions=500):
    if p <= 2:
        raise ValueError("p must be an odd prime")

    if not x_residues_list:
        raise ValueError("x_residues_list empty")

    Fp = GF(p)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    f_poly = R(f_coeffs[::-1])

    sqrt_map = get_sqrt_data(p)

    all_solutions = {}

    for x_residue in x_residues_list:
        x_res = Fp(x_residue)
        sols = set()

        for s_int in range(p):
            if len(sols) >= max_solutions:
                break

            s = Fp(s_int)
            p_val = x_res * s - x_res**2
            disc = s*s - 4*p_val

            if int(disc) not in sqrt_map:
                continue

            u = x**2 - s*x + p_val
            rem = f_poly % u
            A = rem[1] if rem.degree() >= 1 else Fp(0)
            B = rem[0] if rem.degree() >= 0 else Fp(0)

            a = disc
            b = -2*(A*s + 2*B)
            c = A*A

            if a == 0:
                if b == 0:
                    continue
                Zs = [ -c / b ]
            else:
                Dq = b*b - 4*a*c
                if int(Dq) not in sqrt_map:
                    continue
                inv = 1 / (2*a)
                Zs = [(-b + Fp(r)) * inv for r in sqrt_map[int(Dq)]]

            for Z in Zs:
                if int(Z) not in sqrt_map:
                    continue
                for v1 in sqrt_map[int(Z)]:
                    v1 = Fp(v1)
                    if v1 == 0:
                        continue
                    v0 = (A - s*Z) / (2*v1)
                    sols.add((int(s), int(p_val), int(v0), int(v1)))

        all_solutions[x_residue] = list(sols)

    return all_solutions


# ============================================================
# Optimized Dispatcher (NO FALLBACKS)
# ============================================================

def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val=0, max_solutions=500):
    """
    Dispatcher.
    CRITICAL FIX: When FINITE_FIELD is True, pass PREFERRED_X_COORDS to the biased solver
    instead of 'const_val'. 
    """
    if FINITE_FIELD:
        # [FIX] Pass the actual preferred list, not the shift constant!
        return solve_mumford_mod_p_sage_native_BIASED(
            f_coeffs, p, x_residue, PREFERRED_X_COORDS, max_solutions
        )
    else:
        return solve_mumford_mod_p_sage_native_RANDOM(
            f_coeffs, p, x_residue, const_val, max_solutions
        )


def solve_mumford_mod_p_sage_native_BIASED(f_coeffs_ints, p, x_res_int, pref_ints, max_solutions=500):
    """
    Optimized Mumford solver that handles pref_ints as either a list or a single integer.
    """
    if pref_ints is None:
        pref_ints = []
    
    # Ensure we are working with an iterable list of target x-coordinates
    if isinstance(pref_ints, (int, Integer)):
        targets = [pref_ints]
    elif isinstance(pref_ints, (list, tuple, set)):
        targets = pref_ints
    else:
        raise ValueError(f"pref_ints must be an iterable or integer, got {type(pref_ints)}")

    Fp = GF(p)
    x_res = Fp(x_res_int)
    
    # Precompute y_1 = sqrt(f(x_res)) using Horner's method
    y1_sq = Fp(0)
    for c in f_coeffs_ints:
        y1_sq = y1_sq * x_res + Fp(c)
    
    if not y1_sq.is_square():
        return []
    
    y1_roots = [y1_sq.sqrt(), -y1_sq.sqrt()]
    
    solutions = []
    seen_solutions = set()
    
    def s_candidate_generator():
        # First priority: the specifically requested targets
        # Note: s = x1 + x2. If x1 = x_res, we want x2 in targets.
        # So s = x_res + target
        for x_pref in targets:
            yield int(x_res + Fp(x_pref))
        
        # Second priority: general search (randomized)
        import random
        for _ in range(10000):
            yield random.randint(0, p - 1)

    for s_int in s_candidate_generator():
        s_val = Fp(s_int)
        r2 = s_val - x_res
        
        # Fast evaluation of f(r2)
        y2_sq = Fp(0)
        for c in f_coeffs_ints:
            y2_sq = y2_sq * r2 + Fp(c)
            
        if not y2_sq.is_square():
            continue
            
        y2_val = y2_sq.sqrt()
        y2_roots = [y2_val, -y2_val]
        
        if s_val == 2 * x_res:
            if y1_sq == 0: continue 
            
            # Derivative evaluation for the double-root case
            f_prime_x = Fp(0)
            deg = len(f_coeffs_ints) - 1
            for i, c in enumerate(f_coeffs_ints[:-1]):
                f_prime_x = f_prime_x * x_res + Fp(c * (deg - i))
            
            for y1 in y1_roots:
                v1 = f_prime_x / (2 * y1)
                v0 = y1 - v1 * x_res
                sol = (int(s_val), int(x_res * r2), int(v0), int(v1))
                if sol not in seen_solutions:
                    solutions.append(sol)
                    seen_solutions.add(sol)
        else:
            # Standard interpolation for the Mumford v-polynomial
            inv_denom = (x_res - r2)**-1
            for y1 in y1_roots:
                for y2 in y2_roots:
                    v1 = (y1 - y2) * inv_denom
                    v0 = y1 - v1 * x_res
                    sol = (int(s_val), int(x_res * r2), int(v0), int(v1))
                    if sol not in seen_solutions:
                        solutions.append(sol)
                        seen_solutions.add(sol)
        
        if len(solutions) >= max_solutions:
            break
            
    return solutions
