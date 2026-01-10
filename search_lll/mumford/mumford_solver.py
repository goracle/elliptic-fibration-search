import time
import sys
from sage.all import QQ, ZZ, GF, PolynomialRing, HyperellipticCurve
from search_common import DEBUG, FINITE_FIELD, PREFERRED_X_COORDS
from .mumford_core import _poly_mod_quad_fast
from .mumford_verification import verify_mumford_pair

assert PREFERRED_X_COORDS, PREFERRED_X_COORDS


def solve_mumford_mod_p(eqs_dict, p, x_residue, debug=DEBUG):
    """Entry point for modular Mumford solving."""
    f_coeffs = eqs_dict['f_coeffs']
    const_val = int(QQ(eqs_dict.get('const', 0)))
    return solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val)

def get_sqrt_data(p):
    """
    Precomputes a square root map for F_p.
    Returns a dictionary mapping squares to their list of roots.
    """
    if p is None:
        return {}
    key = p
    if key in get_sqrt_data.cache:
        return get_sqrt_data.cache[key]
    sqrt_map = {}
    for i in range((p // 2) + 1):
        sq = int((i * i) % p)
        if sq not in sqrt_map:
            sqrt_map[sq] = []
        sqrt_map[sq].append(i)
        if i != 0 and (p - i) != i:
            sqrt_map[sq].append(p - i)
    ret = sqrt_map
    get_sqrt_data.cache[key] = ret
    return sqrt_map
get_sqrt_data.cache = {}


def _mumford_doubling_mod_p_internal(u_coeffs, v_coeffs, f_coeffs, p, debug=False):
    """Robust modular doubling for genus-2 Mumford divisors."""
    if p == 2: return None, None
    try:
        Fp = GF(p)
        R_Fp = PolynomialRing(Fp, 'x')
        f_poly_Fp = R_Fp(f_coeffs[::-1])
        C_Fp = HyperellipticCurve(f_poly_Fp, 0)
        J_Fp = C_Fp.jacobian()
    except Exception as e:
        if debug: print(f"[MOD-DBL] Init failed at p={p}: {e}")
        return None, None

    try:
        u_poly = R_Fp(u_coeffs[::-1]).monic()
        v_poly = R_Fp(v_coeffs[::-1]) % u_poly
        D = J_Fp([u_poly, v_poly])
        D2 = 2 * D
        
        u_res = D2[0].monic()
        v_res = D2[1] % u_res
        
        u_out = [int(c) for c in u_res.list()][::-1]
        v_out = [int(c) for c in v_res.list()][::-1]
        return u_out, v_out
    except Exception as e:
        if debug: print(f"[MOD-DBL] Doubling failed at p={p}: {e}")
        return None, None

def prefilter_solutions_algebraic(sol_list, prime, f_coeffs):
    """Early filter for solutions mod p before CRT."""
    R = PolynomialRing(GF(prime), 'x')
    x = R.gen()
    f_poly = R(f_coeffs[::-1])
    
    filtered = []
    for s_val, p_val, v0_val, v1_val in sol_list:
        u_poly = x**2 - int(s_val)*x + int(p_val)
        v_poly = int(v1_val)*x + int(v0_val)
        if (v_poly**2 - f_poly) % u_poly == 0:
            filtered.append((s_val, p_val, v0_val, v1_val))
    return filtered

def filter_primes_avoiding_denoms(primes_list, divisors):
    """Removes primes that divide denominators of the rational coefficients."""
    key = (tuple(primes_list), divisors)
    if key in filter_primes_avoiding_denoms.cache:
        return filter_primes_avoiding_denoms[key]
    bad = set()
    for d in divisors:
        for k in ('s', 'p', 'v_0', 'v_1'):
            val = d.get(k)
            den = int(QQ(val).denominator())
            if den == 1: continue
            
            temp_den, p = den, 2
            while p*p <= temp_den:
                if temp_den % p == 0:
                    bad.add(p)
                    while temp_den % p == 0: temp_den //= p
                p += 1
            if temp_den > 1: bad.add(temp_den)
    ret = [p for p in primes_list if p not in bad]
    filter_primes_avoiding_denoms.cache[key] = ret
    return ret
filter_primes_avoiding_denoms.cache = {}


def solve_mumford_batch_sage(f_coeffs, p, x_residues_list, const_val=0, max_solutions=500):
    """
    Batch version that solves for multiple x_residues at once.
    This amortizes polynomial setup costs.
    """
    Fp = GF(p)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    
    f_poly = R([Fp(c) for c in reversed(f_coeffs)])
    
    sqrt_map = get_sqrt_data_sage(p)
   
    all_solutions = {}
    
    for x_residue in x_residues_list:
        solutions = []
        x_res = Fp(x_residue)
        x_sq = x_res * x_res

        for s_int in range(min(p, max_solutions * 2)):
            if len(solutions) >= max_solutions:
                break
            
            s_val = Fp(s_int)
            p_val = x_res * s_val - x_sq
            
            disc = s_val * s_val - 4 * p_val
            disc_int = int(disc)
            if sqrt_map is None:
                if not Fp(disc_int).is_square():
                    continue
                sqrt_disc = Fp(disc_int).sqrt()
            elif disc_int not in sqrt_map:
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
                if sqrt_map is not None:
                    if disc_q_int in sqrt_map:
                        inv_2a = 1 / (2 * a_q)
                        for sq_root_int in sqrt_map[disc_q_int]:
                            sq_root = Fp(sq_root_int)
                            Z_roots.append((-b_q + sq_root) * inv_2a)
            
            for Z in set(Z_roots):
                Z_int = int(Z)
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
        
        all_solutions[x_residue] = list(set(solutions))
    
    return all_solutions


def solve_mumford_mod_p_optimized(f_coeffs, p, x_residue, const_val=0, max_solutions=500):
    """
    Wrapper that uses Sage-native implementation.
    For very large primes (>1M), considers parallel s-value search.
    """
    if FINITE_FIELD:
        if p > 1000000 and max_solutions < p // 10:
            return solve_mumford_mod_p_sage_native_BIASED(f_coeffs, p, x_residue, const_val, max_solutions)
        else:
            return solve_mumford_mod_p_sage_native_BIASED(f_coeffs, p, x_residue, const_val, max_solutions)
    else:
        if p > 1000000 and max_solutions < p // 10:
            return solve_mumford_mod_p_sage_native_RANDOM(f_coeffs, p, x_residue, const_val, max_solutions)
        else:
            return solve_mumford_mod_p_sage_native_RANDOM(f_coeffs, p, x_residue, const_val, max_solutions)


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


from sage.all import GF, PolynomialRing
from random import randrange
from search_common import PREFERRED_X_COORDS, DEBUG

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
            
        return {
            "total": self.total_candidates,
            "split_rate": self.split_count / self.total_candidates,
            "preferred_histogram": dict(self.preferred_histogram),
            "hit_rate": self.preferred_hits / max(1, self.split_count) # Hits per split
        }


def solve_mumford_mod_p_sage_native_BIASED(f_coeffs, p, x_residue, const_val=0, 
                                            max_solutions=500, preferred_x_coords=PREFERRED_X_COORDS):
    """
    [cite_start]Robust Biased Mumford Solver[cite: 1, 32].
    
    Improvements:
    1. PRIORITIZES preferred_x_coords but DOES NOT DISCARD random smooth divisors.
    2. Uses robust type handling for x-coordinates.
    3. Prevents "steamed hams" (100% split rate with 0 returned solutions).
    """
    Fp = GF(p)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    
    # Ensure coeffs are low-to-high for Sage
    f_poly = R([Fp(c) for c in reversed(f_coeffs)])
    
    x_res = Fp(x_residue)
    x_sq = x_res * x_res
    
    solutions = []
    seen_solutions = set()
    
    # Pre-cache sqrt map for small primes to speed up verification
    sqrt_map = get_sqrt_data_sage(p)
    use_cached = (sqrt_map is not None)
    
    bias_stats = BiasStatistics()
    
    # 1. Setup Priority Search Space
    # ---------------------------------------------------------
    # We explicitly construct 's' values that force u(x) to have a root 
    # at both x_residue AND a preferred coordinate.
    # u(x) = (x - x_res)(x - x_pref) = x^2 - (x_res + x_pref)x + ...
    # So s = x_res + x_pref.
    
    s_search_order = []
    preferred_set_ints = set()
    
    if preferred_x_coords:
        preferred_set_ints = set(int(x) for x in preferred_x_coords)
        for x_pref in preferred_set_ints:
            # Force the second root to be x_pref
            x_pref_fp = Fp(x_pref)
            s_forced = x_res + x_pref_fp
            s_search_order.append(int(s_forced))
            
    # Remove duplicates from priority list
    s_search_order = list(dict.fromkeys(s_search_order))
    priority_count = len(s_search_order)
    
    # 2. Fill the rest with Random Candidates
    # ---------------------------------------------------------
    # If we need more solutions than the priority list provides, add random s-values.
    needed = max_solutions * 2 # Buffer for non-splitting / non-QR results
    
    if p > 100000:
        # Large prime: sample random s
        for _ in range(needed):
            s_search_order.append(randrange(p))
    else:
        # Small prime: just iterate all s (0 to p-1)
        # Add values not already in priority list
        priority_set = set(s_search_order)
        for s in range(p):
            if s not in priority_set:
                s_search_order.append(s)

    print(f"  [Biased Solver] Checking {len(s_search_order)} candidates (Priority: {priority_count})...")

    # 3. Main Solver Loop
    # ---------------------------------------------------------
    for s_int in s_search_order:
        if len(solutions) >= max_solutions:
            break
            
        s_val = Fp(s_int)
        
        # Calculate p_val to force x_res as a root: p = x_res * s - x_res^2
        p_val = x_res * s_val - x_sq
        
        # Check discriminant: s^2 - 4p
        disc = s_val * s_val - 4 * p_val
        disc_int = int(disc)
        
        # Quick smoothness check (does u(x) split?)
        is_split = False
        if use_cached:
            if disc_int in sqrt_map:
                is_split = True
        else:
            if disc.is_square():
                is_split = True
                
        if not is_split:
            # Skip irreducible u(x)
            bias_stats.record([], preferred_set_ints)
            continue
            
        # Recover roots for stats
        # Note: We know x_res is one root. The other is s - x_res.
        root2 = s_val - x_res
        current_roots = [x_res, root2]
        bias_stats.record(current_roots, preferred_set_ints)
        
        # 4. Mumford v(x) Reconstruction
        # ---------------------------------------------------------
        # We need v(x) such that v(x)^2 = f(x) mod u(x).
        u_poly = x**2 - s_val*x + p_val
        remainder = f_poly % u_poly
        
        # Remainder is linear: B + A*x
        rem_coeffs = remainder.list()
        B = rem_coeffs[0] if len(rem_coeffs) > 0 else Fp(0)
        A = rem_coeffs[1] if len(rem_coeffs) > 1 else Fp(0)
        
        # Solve for v(x) = v1*x + v0
        # See Mumford paper/algorithm for the quadratic residue logic on coefficients
        a_q = disc
        b_q = -2 * (A * s_val + 2 * B)
        c_q = A * A
        
        Z_roots = []
        
        # Solve a_q * Z^2 + b_q * Z + c_q = 0
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
                        sq = Fp(sq_root_int)
                        Z_roots.append((-b_q + sq) * inv_2a)
            else:
                if disc_q.is_square():
                    sqrt_disc_q = disc_q.sqrt()
                    inv_2a = 1 / (2 * a_q)
                    Z_roots.append((-b_q + sqrt_disc_q) * inv_2a)
                    Z_roots.append((-b_q - sqrt_disc_q) * inv_2a)
        
        # Reconstruct v0, v1 from Z
        for Z in set(Z_roots):
            Z_int = int(Z)
            
            # We need v1^2 = Z
            valid_v1s = []
            if use_cached:
                if Z_int in sqrt_map:
                    valid_v1s = [Fp(r) for r in sqrt_map[Z_int]]
            else:
                if Z.is_square():
                    rt = Z.sqrt()
                    valid_v1s = [rt, -rt]
            
            for v1_val in valid_v1s:
                if v1_val != 0:
                    v0_val = (A - s_val * Z) / (2 * v1_val)
                    # Verify: v0^2 - p*v1^2 = B ? 
                    # (Actually the check is v0^2 + v1^2*p - s*v1*v0... simplified check below)
                    if v0_val * v0_val - p_val * Z == B:
                        sol_tuple = (int(s_val), int(p_val), int(v0_val), int(v1_val))
                        if sol_tuple not in seen_solutions:
                            solutions.append(sol_tuple)
                            seen_solutions.add(sol_tuple)
                else:
                    # v1 = 0 case
                    if A == 0 and Z == 0:
                        # v0^2 = B
                        # Solve for v0
                        if use_cached:
                            if int(B) in sqrt_map:
                                for r in sqrt_map[int(B)]:
                                    sol_tuple = (int(s_val), int(p_val), int(r), 0)
                                    if sol_tuple not in seen_solutions:
                                        solutions.append(sol_tuple)
                                        seen_solutions.add(sol_tuple)
                        else:
                            if B.is_square():
                                rt = B.sqrt()
                                for v0_cand in [rt, -rt]:
                                    sol_tuple = (int(s_val), int(p_val), int(v0_cand), 0)
                                    if sol_tuple not in seen_solutions:
                                        solutions.append(sol_tuple)
                                        seen_solutions.add(sol_tuple)

    # Report final stats
    report = bias_stats.report()
    print(f"  [Bias Statistics] {report}")
    print(f"  [Biased Solver] Found {len(solutions)} unique solutions.")
    
    return solutions
