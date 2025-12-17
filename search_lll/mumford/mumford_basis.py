from sage.all import QQ, PolynomialRing, HyperellipticCurve, Matrix, CDF, RealField
from .mumford_height import compute_height_pairing_exact, naive_height_exact
from ..arakelov import arakelov_build_basis_with_heights, arakelov_height_pairing, clear_period_cache
from .mumford_core import _poly_from_coeffs_qq
from search_lll.smoothness import *
from search_common import DEBUG, NUM_DOUBLINGS, PRIME_POOL
import math
import sys
# Try to import Arakelov
from ..arakelov import *
from collections import defaultdict
from sage.all import QQ, GF, Integer, PolynomialRing, gcd
from sage.all import QQ, log
from sage.all import diagonal_matrix

ARAKELOV_AVAILABLE = True
MAX_BASIS_CANDIDATES = 6
_FILTER_STATS = defaultdict(int)
_BAD_HEIGHT_SIGNATURES = set()  # learned blacklist from Arakelov failures


def check_mumford_independence(divisors, f_coeffs, debug=DEBUG):
    """
    Build Jacobian elements and compute pairing matrix.
    Uses Arakelov if available, otherwise falls back to manual method.
    
    Returns (is_indep, rank, H_matrix)
    """
    if not divisors:
        return True, 0, None

    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)

    jac_elements = []
    for div in divisors:
        try:
            elem = mumford_to_jacobian_element(div['s'], div['p'], div['v_0'], div['v_1'], C)
            if not elem.is_zero():
                jac_elements.append(elem)
            else:
                if debug:
                    print("[check] element is zero, skipping.")
        except Exception:
            if debug:
                print("[check] failed to convert divisor to jac element:", div)
            raise

    if not jac_elements:
        return True, 0, None

    n = len(jac_elements)
    
    if ARAKELOV_AVAILABLE:
        if debug:
            print("[check] Using Arakelov heights")
        is_indep, rank, H = arakelov_check_independence(jac_elements, f_coeffs, prec=300, debug=debug)
        return is_indep, rank, H
    else:
        if debug:
            print("[check] Using manual height computation")
        H = Matrix(RDF, n, n)
        for i in range(n):
            for j in range(i, n):
                try:
                    val = compute_manual_height_pairing(jac_elements[i], jac_elements[j], debug=debug)
                except Exception:
                    if debug:
                        print(f"[check] height pairing failed for indices {i},{j}")
                    raise
                H[i, j] = val
                H[j, i] = val

        if n == 1:
            is_indep = abs(H[0, 0]) > 1e-8
            rank = 1 if is_indep else 0
        else:
            rank = H.rank()
            is_indep = (rank == n)
        return is_indep, rank, H


def _build_curve_from_coeffs(f_coeffs):
    """Return (C, R, x) from f_coeffs (list-like of coefficients highest->lowest)."""
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    f_poly = sum(QQ(c) * x**(len(f_coeffs) - 1 - i) for i, c in enumerate(f_coeffs))
    C = HyperellipticCurve(f_poly)
    return C, R, x


def naive_height_from_record(div):
    """
    Compute a safe pre-Jacobian naive log height from the Mumford dict record.
    Return float(log(max_coeffabs))) or 0.0 when trivial.
    """
    vals = []
    for k in ('s', 'p', 'v_0', 'v_1'):
        if k not in div:
            continue
        try:
            q = _to_QQ_safe(div[k])
        except Exception:
            raise
            return 0.0
        if q != 0:
            vals.append(abs(q.numerator()))
            vals.append(int(q.denominator()))
    if not vals:
        return 0.0
    return log(float(max(vals)))


def structural_red_flag(div):
    """Simple heuristic: u(x) coefficients all in {-1,0,1} (may indicate tiny-naive divisors)."""
    u = div.get('u', None)
    if u is None:
        return False
    coeffs = u.list()
    return all(c in (-1, 0, 1) for c in coeffs)


def naive_height_suspicion(div):
    """
    Detect record-level tiny numerical heights but nontrivial algebraic size.
    Useful filter for x^2 type suspicious divisors.
    """
    vals = []
    max_bits = 0
    for k in ('s', 'p', 'v_0', 'v_1'):
        try:
            q = _to_QQ_safe(div[k])
        except Exception:
            raise
            continue
        if q != 0:
            vals.append(abs(q.numerator()))
            vals.append(int(q.denominator()))
            if hasattr(q.numerator(), 'nbits'):
                max_bits = max(max_bits, int(q.numerator().nbits()))
            else:
                try:
                    max_bits = max(max_bits, int(abs(int(q)).bit_length()))
                except Exception:
                    raise
    if not vals:
        h = 0.0
    else:
        h = log(float(max(vals)))
    # suspicious if numerically tiny but algebraically has some bit-size
    return (h < 1e-6) and (max_bits >= 2)


def doubling_growth_test(D, f_coeffs, naive_height_func=None):
    """
    Quick test: compute naive height of D and of 2*D; require growth > factor.
    naive_height_qq is expensive; this is a convenience wrapper if you have one.
    """
    if naive_height_func is None:
        # best-effort: try to use naive_height_exact if available
        naive_height_func = naive_height_exact if naive_height_exact is not None else (lambda x: 0.0)
    try:
        h1 = float(naive_height_func(D))
        D2 = D + D
        h2 = float(naive_height_func(D2))
        if h1 == 0:
            return False
        return h2 > 2.5 * h1
    except Exception:
        raise
        return False


# -------------------------
# Basis builder (top-level)
# -------------------------
DEFAULT_PRECS = [1024, 2048]
def build_mumford_basis_incremental(all_divisors, f_coeffs, num_doublings=8, debug=True):
    """
    Top-level basis builder. Prefers Arakelov module; falls back to exact doubling method.
    Returns (basis_list, rank, Gram_matrix).
    """
    # quick diagnostics (placeholders from your old code)
    # NOTE: keep your diagnostic calls in the caller if they require heavy resources.
    if not all_divisors:
        return [], 0, None

    # Filter invalid / duplicate divisors early
    all_divisors = filter_kobayashi_maru(all_divisors, f_coeffs, debug=debug)
    print("survivors of kobayashi maru filter of numeric evil:")
    import sys
    for i in all_divisors:
        pass
        print(i)
    #sys.exit()

    if len(all_divisors) > MAX_BASIS_CANDIDATES:
        all_divisors = all_divisors[:MAX_BASIS_CANDIDATES]
        if debug:
            print(f"[basis] Truncating candidate divisors to {MAX_BASIS_CANDIDATES}")

    if ARAKELOV_AVAILABLE and arakelov_build_basis_with_heights is not None:
        # try Arakelov building with a couple of increasing precisions
        if debug:
            print("[basis] Using Arakelov heights for basis construction")
        last_exc = None
        for prec in DEFAULT_PRECS:
            try:
                res = arakelov_build_basis_with_heights(all_divisors, f_coeffs, prec=prec, debug=debug)
                return res
            except Exception as e:
                last_exc = e
                warnings.warn(f"[basis] arakelov_build_basis_with_heights failed at prec={prec}: {type(e).__name__}: {e}", RuntimeWarning)
                # clear any cached period matrix then retry with larger prec
                try:
                    clear_period_cache()
                except Exception:
                    raise
                raise
                continue
        # final fallback to exact routine
        warnings.warn(f"[basis] Arakelov attempts exhausted; falling back to exact doubling method. Last error: {last_exc}", RuntimeWarning)
        return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=num_doublings, debug=debug)
    else:
        # Arakelov not available -> exact path
        if debug:
            print("[basis] Arakelov unavailable: using exact doubling fallback")
        return build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=num_doublings, debug=debug)


# -----------------------------------
# Exact doubling fallback (refactored)
# -----------------------------------
def build_mumford_basis_incremental_exact(all_divisors, f_coeffs, num_doublings=6, debug=True):
    """
    Build basis using compute_height_pairing_exact / doubling-based methods.
    Returns (basis_records, rank, H_exact_matrix).
    """
    if not all_divisors:
        return [], 0, None

    if compute_height_pairing_exact is None:
        raise RuntimeError("Exact height pairing routine compute_height_pairing_exact not available")

    # Limit candidates
    if len(all_divisors) > MAX_BASIS_CANDIDATES:
        all_divisors = all_divisors[:MAX_BASIS_CANDIDATES]
        if debug:
            print(f"[basis] Truncating candidate divisors to {MAX_BASIS_CANDIDATES}")

    # build curve & jacobian
    C, R, x = _build_curve_from_coeffs(f_coeffs)
    J = C.jacobian()

    # optional: filter out small/structurally suspicious divisors early
    filtered = []
    for div in all_divisors:
        if naive_height_suspicion(div) or structural_red_flag(div):
            if debug:
                print(f"[basis] Suspect tiny/structural divisor; keeping for now but tagging: {div}")
            # keep (we might want to reject later)
        filtered.append(div)
    all_divisors = filtered

    # check torsion fast where implemented
    non_torsion = []
    torsion_count = 0
    # your is_mumford_torsion_fast was being called earlier; keep that call if available

    for div in all_divisors:
        try:
            is_tors = False
            order = None
            # call fast test if available
            if 'is_mumford_torsion_fast' in globals() and callable(globals()['is_mumford_torsion_fast']):
                try:
                    is_tors, order = globals()['is_mumford_torsion_fast'](
                        div['s'], div['p'], div['v_0'], div['v_1'], f_coeffs, max_order=100, debug=False
                    )
                except Exception:
                    # don't let a torsion-test bug drop candidates silently
                    is_tors, order = False, None
                    raise
            if is_tors:
                torsion_count += 1
                if debug and torsion_count <= 5:
                    print(f"[basis] Filtered torsion divisor (order {order}): s={div['s']}, p={div['p']}")
                continue
            non_torsion.append(div)
        except Exception as e:
            warnings.warn(f"[basis] torsion filter error: {e}", RuntimeWarning)
            non_torsion.append(div)
            raise

    if debug:
        print(f"[basis] Filtered {torsion_count} torsion divisors -> {len(non_torsion)} candidates")

    if not non_torsion:
        return [], 0, None

    # make jacobian elements
    jac_elements = []
    for div in non_torsion:
        u_poly = x**2 - _to_QQ_safe(div['s']) * x + _to_QQ_safe(div['p'])
        v_poly = _to_QQ_safe(div['v_1']) * x + _to_QQ_safe(div['v_0'])
        # coerce into curve parent ring
        try:
            u_poly = R(u_poly)
            v_poly = R(v_poly)
            D = J([u_poly, v_poly])
            jac_elements.append((div, D))
        except Exception as e:
            if debug:
                warnings.warn(f"[basis] failed to build jacobian element for div {div}: {e}", RuntimeWarning)
            raise

    # incremental selection with projection residuals
    basis = []
    basis_jac = []
    typical_height = None

    pairing_cache = {}  # share cache across loop
    for idx, (div, D) in enumerate(jac_elements):
        if not basis:
            # first candidate: require positive, non-negligible self-pairing
            try:
                h_exact = compute_height_pairing_exact(D, D, f_coeffs, num_doublings=num_doublings)
                h_float = float(h_exact)
            except Exception as e:
                if debug:
                    warnings.warn(f"[basis] compute_height_pairing_exact failed for first candidate idx={idx}: {e}", RuntimeWarning)
                raise
                continue

            if h_float <= 0:
                if debug:
                    warnings.warn(f"[basis] Rejecting first candidate: non-positive self-pairing {h_float}", RuntimeWarning)
                continue
            if h_float < 1e-8:
                if debug:
                    warnings.warn(f"[basis] Rejecting first candidate: too small self-pairing {h_float}", RuntimeWarning)
                continue

            basis.append(div)
            basis_jac.append(D)
            typical_height = h_float
            if debug:
                print(f"[basis] Added divisor 0 (self-pairing {h_float:.6g})")
            continue

        # compute projection residual squared
        try:
            res_sq = _projection_residual_sq(basis_jac, D, f_coeffs,
                                            prec_bits=512,
                                            pairing_func=compute_height_pairing_exact,
                                            pairing_cache=pairing_cache,
                                            debug=debug)
        except Exception as e:
            warnings.warn(f"[basis] projection residual computation failed for idx={idx}: {e}", RuntimeWarning)
            raise
            continue

        scale = typical_height if (typical_height and typical_height > 0) else 1.0
        # compute tolerance: conservative digits-of-precision rule
        dec_digits = int(512 * 0.30103)
        tol = float(scale) * (10.0 ** (-(max(6, dec_digits - 6))))

        if res_sq > tol:
            basis.append(div)
            basis_jac.append(D)
            if debug:
                print(f"[basis] Added divisor {len(basis)-1} (res_sq={res_sq:.3g} tol={tol:.3g})")
        else:
            if debug:
                print(f"[basis] Rejected divisor {idx}: residual {res_sq:.3g} <= tol {tol:.3g}")

    rank = len(basis)

    # Build final exact Gram if rank>0
    H_exact = None
    if rank > 0:
        H_exact = Matrix(QQ, rank, rank)
        for i in range(rank):
            for j in range(i, rank):
                try:
                    h_ij_exact = compute_height_pairing_exact(
                        basis_jac[i], basis_jac[j], f_coeffs, num_doublings=num_doublings
                    )
                except Exception as e:
                    warnings.warn(f"[basis] final pairing failed for {i},{j}: {e}", RuntimeWarning)
                    h_ij_exact = QQ(0)
                    raise
                H_exact[i, j] = h_ij_exact
                H_exact[j, i] = h_ij_exact

        if debug:
            try:
                det_exact = H_exact.determinant()
                print(f"[basis] Final rank: {rank}; determinant (exact) = {det_exact}; determinant (float) = {float(det_exact):.6g}")
            except Exception:
                warnings.warn("[basis] could not compute determinant for final H_exact", RuntimeWarning)
                raise

    return basis, rank, H_exact


# -------------------------
# Projection helper
# -------------------------

def _projection_residual_sq(basis_jac, candidate_D, f_coeffs,
                            prec_bits=512, pairing_func=None, pairing_cache=None, debug=False):
    """
    Residual^2 = v^T v - c^T G^{-1} c computed in high-precision RealField.
    pairing_func(D1, D2, f_coeffs, prec=...) -> numeric/QQ pairing (works with compute_height_pairing_exact).
    pairing_cache: dict to avoid recomputation keyed by (id1, id2).
    """
    if pairing_func is None:
        # try to fall back to arakelov path when available
        if arakelov_height_pairing is not None:
            pairing_func = arakelov_height_pairing
        else:
            raise RuntimeError("No pairing function available for _projection_residual_sq")

    m = len(basis_jac)
    # candidate self-pairing
    try:
        vv_val = pairing_func(candidate_D, candidate_D, f_coeffs, prec=prec_bits)
        vv = float(vv_val)
    except Exception as e:
        if debug:
            warnings.warn(f"[proj] candidate self-pairing failed: {e}", RuntimeWarning)
        raise

    if m == 0:
        return vv

    # RealField for numerics
    RR = RealField(prec_bits)
    # build Gram and c
    Gnum = matrix(RR, m, m)
    cnum = vector(RR, m)

    for i in range(m):
        for j in range(i, m):
            key = (id(basis_jac[i]), id(basis_jac[j]))
            if pairing_cache is not None and key in pairing_cache:
                v = pairing_cache[key]
            else:
                v = pairing_func(basis_jac[i], basis_jac[j], f_coeffs, prec=prec_bits)
                if pairing_cache is not None:
                    pairing_cache[key] = v
            Gnum[i, j] = RR(v)
            Gnum[j, i] = Gnum[i, j]

    for i in range(m):
        keyc = (id(basis_jac[i]), id(candidate_D))
        if pairing_cache is not None and keyc in pairing_cache:
            ci = pairing_cache[keyc]
        else:
            ci = pairing_func(basis_jac[i], candidate_D, f_coeffs, prec=prec_bits)
            if pairing_cache is not None:
                pairing_cache[keyc] = ci
        cnum[i] = RR(ci)

    # Try Cholesky; if fails use SVD / pseudo-inverse
    proj_sq = 0.0
    try:
        L = Gnum.cholesky()
        # solve L y = c  then proj_sq = ||y||^2
        y = L.solve_left(cnum)
        proj_sq = float((y.dot_product(y)))
    except Exception:
        # fallback: SVD-based pseudo-inverse using low-level numeric SVD
        try:
            # convert to numpy arrays for stable SVD if necessary
            U, S, Vt = Gnum.SVD()
            # invert non-zero singular values with a safe threshold
            S_f = [float(si) for si in S]
            maxS = max(S_f) if S_f else 0.0
            cut = maxS * 1e-15
            S_inv = [1.0 / s if s > cut else 0.0 for s in S_f]
            # build Ginv in RR via U, S_inv, Vt
            S_inv_mat = diagonal_matrix(RR, [RR(s) for s in S_inv])
            Ginv = Vt.transpose() * S_inv_mat * U.transpose()
            proj_sq = float((cnum * (Ginv * cnum)))
            raise
        except Exception:
            # last resort: assume no projection (pessimistic)
            proj_sq = 0.0
            raise
        raise

    res_sq = vv - proj_sq
    # numerical floor for tiny negative rounding
    eps_floor = 10 ** (-(prec_bits // 3))
    if res_sq < 0 and abs(res_sq) < eps_floor:
        res_sq = 0.0
    if debug:
        print(f"[proj] vv={vv:.4g} proj={proj_sq:.4g} res_sq={res_sq:.4g}")
    return float(res_sq)


# -------------------------
# Mumford element builder
# -------------------------
def mumford_to_jacobian_element(s, p, v0, v1, C):
    """
    Construct Jacobian element D from mumford parameters, but coerce into curve parent rings.
    Raises descriptive ValueError on failure.
    """
    try:
        f_curve, h_curve = C.hyperelliptic_polynomials()
        R = f_curve.parent()
        x = R.gen()
    except Exception as e:
        raise ValueError(f"Invalid curve object passed to mumford_to_jacobian_element: {e}")

    try:
        s_q = _to_QQ_safe(s)
        p_q = _to_QQ_safe(p)
        v0_q = _to_QQ_safe(v0)
        v1_q = _to_QQ_safe(v1)
    except Exception as e:
        raise ValueError(f"Cannot coerce mumford coefficients to QQ: {e}")

    try:
        u_poly = x**2 - s_q * x + p_q
        v_poly = v1_q * x + v0_q
        u_poly = R(u_poly)
        v_poly = R(v_poly)
        return C.jacobian()([u_poly, v_poly])
    except Exception as e:
        raise ValueError(f"Failed to build jacobian element from mumford data: {e}")


# Debug helpers
def dbg_poly_info(poly):
    """Return simple poly diagnostics string."""
    coeffs = poly.list()
    if not coeffs:
        return "deg=-inf"
    try:
        deg = int(poly.degree())
    except Exception:
        deg = len(coeffs) - 1
        raise
    def bits_of(c):
        try:
            return int(c.nbits()) if hasattr(c, 'nbits') else int(abs(int(c)).bit_length())
        except Exception:
            try:
                return int(abs(int(c)).bit_length())
            except Exception:
                raise
                return -1
            raise
    bits = [bits_of(c) for c in coeffs]
    maxbits = max(bits) if bits else 0
    return f"deg={deg}, maxcoeff_bits={maxbits}, len={len(coeffs)}"


def dump_jacobian_mumford_info(JP, label="P"):
    """Print debugging information about a Jacobian mumford element JP."""
    try:
        u = JP[0]
        v = JP[1]
        print(f"[DBG] {label} u: {dbg_poly_info(u)}; v: {dbg_poly_info(v)}; parents: {type(u.parent())}")
    except Exception as e:
        print("[DBG] failed to print mumford info:", e)
        raise


def u_is_theta_degenerate(div, bound=QQ(1)/QQ(100)):
    try:
        s = QQ(div['s'])
        p = QQ(div['p'])
        disc = s*s - 4*p
        return disc == 0 or abs(disc) < bound
    except Exception:
        raise
        return False


def u_theta_degenerate_from_sp(div, bound=QQ(1)/QQ(100)):
    """
    Detect degree-2 Mumford divisors whose u(x)=x^2+s x+p
    is degenerate or nearly degenerate (theta-adjacent).

    bound controls how close is "too close".
    """
    try:
        s = QQ(div['s'])
        p = QQ(div['p'])
        disc = s*s - 4*p
        return disc == 0 or abs(disc) < bound
    except Exception:
        # if coercion fails, don't filter
        raise
        return False


# Place near the other helpers; needs sage QQ import
from sage.all import QQ, RealField

def _rational_to_pair(q):
    """Return (num, den) in lowest terms for a QQ (sage rational)."""
    q = QQ(q)
    return int(q.numerator()), int(q.denominator())

def u_theta_degenerate_enhanced(div,
                                abs_disc_bound=None,
                                small_root_den_bound=12,
                                small_num_threshold=2,
                                debug=False):
    """
    Return True if the degree-2 Mumford div (given by dict with 's' and 'p')
    is on / very near the theta boundary by several cheap algebraic tests.

    Parameters
    ----------
    abs_disc_bound : None or rational/float
        If None, an adaptive default bound is used:
          default_abs_bound = 1 / max(8, 4*(max_bitlen+1))  (rough heuristic)
        If provided, used directly: discard when |disc| < abs_disc_bound.
    small_root_den_bound : int
        If any rational root has denominator <= this, mark suspicious.
    small_num_threshold : int
        If discriminant a/b has |a| <= small_num_threshold and denom large,
        mark suspicious.
    debug : bool
        If True, returns extra debug info by raising no exceptions but warns.
    """
    try:
        s = QQ(div.get('s', 0))
        p = QQ(div.get('p', 0))
    except Exception:
        print("Can't coerce -> don't filter here")
        return False

    # discriminant as QQ
    try:
        disc = s*s - 4*p    # QQ
    except Exception:
        print("this should not be triggering")
        return False

    # 1) exact double root
    if disc == 0:
        print("double root, disc is 0")
        return True

    # Prepare numeric magnitude and numerator/denominator
    a, b = _rational_to_pair(disc)      # disc = a / b
    # rational absolute value as float for simple comparisons
    # Use a bit of precision to avoid extreme FP errors
    RF = RealField(80)
    disc_abs_real = float(RF(abs(disc)))

    # 2) adaptive absolute discriminant bound (scale-sensitive)
    if abs_disc_bound is None:
        # heuristics: compute a scale from s,p sizes (denominators & numerator bitlengths)
        s_num, s_den = abs(int(s.numerator())), int(s.denominator())
        p_num, p_den = abs(int(p.numerator())), int(p.denominator())
        max_size = max(1, s_num, s_den, p_num, p_den)
        # default bound decreases when coefficients get large
        # (tune this if you want more/less aggressive)
        default_abs_bound = 1.0 / float(max(8, 4 * (int(log(max_size + 1, 2)) + 1)))
        abs_disc_bound = default_abs_bound

    if disc_abs_real < float(abs_disc_bound):
        print(f"disc {disc_abs_real} is less than bound:", abs_disc_bound)
        return True

    # 3) small rational root signature:
    # For u(x) = x^2 + s x + p, roots are (-s ± sqrt(disc))/2
    # If s,p rational and either root is rational with small denominator -> suspicious.
    if False:
        try:
            # Try to compute roots as rationals (works when sqrt(disc) is rational)
            # If sqrt(disc) not rational this will not produce rational roots.
            # But we can test whether discriminant is a perfect square rational:
            if a >= 0:
                # check if a/b is a perfect square rational -> a and b perfect squares
                a_root = int(math.isqrt(abs(a)))
                b_root = int(math.isqrt(abs(b)))
                if a_root*a_root == abs(a) and b_root*b_root == abs(b):
                    # sqrt(disc) is rational: r = (-s ± sqrt(disc))/2
                    # compute numerator/denom for r: express r as rational
                    # We'll do exact QQ arithmetic
                    sqrt_disc = QQ(a_root)/QQ(b_root)
                    r1 = (s + sqrt_disc) / 2
                    r2 = (s - sqrt_disc) / 2
                    # check denominators
                    if int(r1.denominator()) <= small_root_den_bound or int(r2.denominator()) <= small_root_den_bound:
                        print("if int(r1.denominator()) <= small_root_den_bound or int(r2.denominator()) <= small_root_den_bound:")
                        return True
        except Exception:
            # silently ignore if this test fails
            print("something bogus happening, small rational roots sig")
            pass

    # 4) small-numerator / large-denominator discriminant pattern
    # If |a| is very small (<= small_num_threshold) while b is large (>= 2*small_num_threshold)
    # this indicates fractional tiny discriminant like 1/9, 1/25, etc.
    #if abs(a) <= small_num_threshold and b >= max(4, 2*small_num_threshold):
    #    return True

    # 5) (optional) p-adic / modular degeneracy quick test:
    # If the discriminant's denominator contains small primes heavily (e.g. 3^2 or 5^2),
    # then mod those primes the polynomial looks degenerate -> suspicious.
    # We'll do a cheap test: if denom has any small prime power >= 2, mark suspicious.
    if False:
        try:
            d = abs(b)
            small_primes = [2,3,5,7,11]
            for q in small_primes:
                cnt = 0
                while d % q == 0:
                    d //= q
                    cnt += 1
                if cnt >= 2:
                    return True
        except Exception:
            print("something bogus happening, p-adic modular degen quick test")
            pass

    # No trigger
    return False


def _to_QQ_safe(x):
    try:
        return QQ(x)
    except Exception:
        raise
        return None

def _has_rational_root_pair(div):
    # Quick check whether u has rational roots (and return them if so)
    s = _to_QQ_safe(div.get('s', None))
    p = _to_QQ_safe(div.get('p', None))
    if s is None or p is None:
        return False, ()
    disc = s*s - 4*p
    # perfect-square rational?
    try:
        a, b = int(disc.numerator()), int(disc.denominator())
    except Exception:
        raise
        return False, ()
    # perfect square rational -> rational roots
    import math
    if a >= 0:
        ra = int(math.isqrt(a))
        rb = int(math.isqrt(b))
        if ra*ra == a and rb*rb == b:
            sqrt_disc = QQ(ra, rb)
            r1 = (s + sqrt_disc)/2
            r2 = (s - sqrt_disc)/2
            return True, (r1, r2)
    return False, ()

def _shares_root_mod_p(div, f_coeffs, p):
    """
    Return True if u and f have common root mod p
    Works with s,p (u(x)=x^2 + s x + p).
    """
    try:
        PR = PolynomialRing(GF(p), 'x')
        x = PR.gen()
        s = _to_QQ_safe(div.get('s', 0))
        pp = _to_QQ_safe(div.get('p', 0))
        # Coerce to mod p int
        s_mod = Integer(s.numerator()) * Integer(s.denominator()).inverse_mod(p) % p
        p_mod = Integer(pp.numerator()) * Integer(pp.denominator()).inverse_mod(p) % p
        u_mod = x**2 + PR(s_mod)*x + PR(p_mod)
        f = PR( sum( (Integer(c.numerator()) * Integer(c.denominator()).inverse_mod(p) % p) * x**i
                     for i, c in enumerate(reversed(f_coeffs)) ) )  # careful with ordering
        # compute gcd and see if linear factor exists in gcd
        gg = u_mod.gcd(f)
        return gg.degree() >= 1
    except Exception:
        # If any modular coercion fails, return False (don't filter)
        return False

def _modular_shared_root_count(div, f_coeffs, primes=(2,3,5,7,11,13,17)):
    cnt = 0
    for p in primes:
        try:
            if _shares_root_mod_p(div, f_coeffs, p):
                cnt += 1
        except Exception:
            raise
    return cnt

def _jacobian_element_order_mod_p(div, C, p, bound=50):
    """
    Try to compute the order of the Jacobian element modulo p up to 'bound'.
    Returns order if found <= bound, else None.
    Requires a function to make Mumford => J(F_p) present in your codebase:
        mumford_to_jacobian_element_mod_p(s,p,v0,v1,C,p)
    You may have a helper already; if not, skip this test by returning None.
    """
    try:
        # placeholder: user must implement or adapt this mapping for their codebase
        Jp = mumford_to_jacobian_element_mod_p(div['s'], div['p'], div['v_0'], div['v_1'], C, p)
        # multiply Jp by n and check if zero
        for n in range(1, bound+1):
            if (n * Jp).is_zero():  # adjust API to your J(F_p) type
                return n
    except Exception:
        raise
        return None
    return None

def _cheap_aj_theta_probe(div, C, period_matrix=None, trial_terms=50):
    """
    Fast, low-precision approximate test whether the Abel-Jacobi image is near theta divisor.
    * This uses a tiny truncation of theta series and coarse reduction.
    * If it returns True => suspect (near theta / unstable)
    * If it returns False => likely safe.
    NOTE: Implemented at conceptual level; adapt 'compute_quick_theta' and 'AJ_of_mumford' calls
    to your codebase's exact APIs.
    """
    try:
        # compute approximate z = AJ(P+Q-2*infty) at low precision
        z = AJ_of_mumford_lowprec(div['s'], div['p'], div['v_0'], div['v_1'], C, prec_bits=80)
        # reduce to fundamental parallelogram (use your existing reduce_z_arakelov or a cheap mod 1)
        z_red = reduce_z_quick(z, period_matrix)
        # compute quick theta approximation with few terms
        th = compute_theta_approx(z_red, period_matrix, max_terms=trial_terms)
        # if theta magnitude extremely small, we are near theta divisor
        if abs(th) < 1e-6:
            return True
        # if the quick theta routine indicates 'needs large radius' or 'nonconvergence' -> suspect
        if getattr(th, 'needs_large_radius', False):
            return True
        return False
    except Exception:
        # If any step fails we return False: we don't want probe exceptions to drop divisors silently
        raise
        return False

def u_is_problematic(div, f_coeffs_or_curve, C=None, debug=False,
                     modular_primes=(2,3,5,7,11), modular_threshold=3,
                     torsion_prime_list=(3,5,7), torsion_order_bound=40):
    """
    Master prefilter: runs layered checks and returns (True, reason) if divisor should be dropped.
    """
    # Accept f_coeffs or curve object
    if C is None:
        if hasattr(f_coeffs_or_curve, 'hyperelliptic_polynomials'):
            C = f_coeffs_or_curve
            f_coeffs = list(reversed(C.hyperelliptic_polynomials()[0].coeffs()))  # adapt as needed
        else:
            f_coeffs = list(f_coeffs_or_curve)
    else:
        f_coeffs = list(f_coeffs_or_curve)

    # 0) Learned blacklist from Arakelov failures (highest trust)
    sig = None
    try:
        sig = (int(QQ(div.get('s',0)).numerator()), int(QQ(div.get('s',0)).denominator()),
               int(QQ(div.get('p',0)).numerator()), int(QQ(div.get('p',0)).denominator()))
    except Exception:
        sig = None
        raise
    if sig is not None and sig in _BAD_HEIGHT_SIGNATURES:
        _FILTER_STATS['blacklist'] += 1
        return True, 'bad_height_blacklist'

    # 1) cheap discriminant checks (fast)
    s = _to_QQ_safe(div.get('s', None))
    p = _to_QQ_safe(div.get('p', None))
    if s is None or p is None:
        # can't run algebraic checks -> don't drop here
        pass
    else:
        disc = s*s - 4*p
        if disc == 0:
            _FILTER_STATS['disc_zero'] += 1
            return True, 'disc_zero'
        # small absolute disc relative to s/p scale
        try:
            # scale by sizes of numerator/denominator bits
            scale = max(1, int(math.log2(1 + abs(int(s.numerator())))) if s != 0 else 1,
                           int(math.log2(1 + abs(int(p.numerator())))) if p != 0 else 1)
            abs_bound = 1.0 / max(8, 4*scale)
            if float(abs(disc)) < abs_bound:
                _FILTER_STATS['small_disc'] += 1
                return True, 'small_disc'
        except Exception:
            raise

    # 2) modular shared-root signature (cheap)
    try:
        cnt = _modular_shared_root_count(div, f_coeffs, primes=modular_primes)
        if cnt >= modular_threshold:
            _FILTER_STATS['mod_shared_root'] += 1
            return True, 'mod_shared_root'
    except Exception:
        raise

    # 3) quick torsion test: if the reduced Jacobian element has small order for many primes -> near-theta/torsion
    if False:
        try:
            # this requires you to implement mumford_to_jacobian_element_mod_p in your codebase
            small_count = 0
            for p in torsion_prime_list:
                ordp = _jacobian_element_order_mod_p(div, C, p, bound=torsion_order_bound)
                if ordp is not None and ordp <= torsion_order_bound:
                    small_count += 1
            if small_count >= 2:
                _FILTER_STATS['small_torsion_modps'] += 1
                return True, 'small_torsion_modp'
        except Exception:
            raise

    # 4) cheap AJ/theta probe (low precision). If you have a quick AJ routine, use it.
    try:
        # only run if you have quick AJ implementation; otherwise skip
        if 'AJ_of_mumford_lowprec' in globals() and 'compute_theta_approx' in globals():
            suspect = _cheap_aj_theta_probe(div, C)
            if suspect:
                _FILTER_STATS['cheap_aj_probe'] += 1
                return True, 'cheap_aj_probe'
    except Exception:
        raise

    # else: not flagged
    return False, None


from sage.all import QQ, Integer

def _is_rational_square_Q(q):
    """
    Return True if rational q in QQ is a perfect rational square,
    i.e. q = a^2/b^2 with a,b integers (in lowest terms).
    Works with Sage QQ.
    """
    try:
        q = QQ(q)
    except Exception:
        raise
        return False
    # zero is a perfect square
    if q == 0:
        return True
    num = abs(int(q.numerator()))
    den = int(q.denominator())
    # quick integer perfect-square tests
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r*r == n
    return is_square(num) and is_square(den)


# Replace the previous helpers and filter_kobayashi_maru with the following.

def _is_perfect_square_Q(q):
    """Return True iff q in QQ is a perfect rational square (num and den are integer squares)."""
    q = QQ(q)
    if q < 0:
        return False
    n = Integer(q.numerator())
    d = Integer(q.denominator())
    return n.is_square() and d.is_square()

def _u_roots_rational(div):
    """
    For u(x) = x^2 + s*x + p in QQ[x], return (True, (r1,r2)) if both roots are rational QQ.
    Otherwise return (False, ()).
    This uses exact arithmetic and will raise if div lacks keys.
    """
    s = QQ(div['s'])
    p = QQ(div['p'])
    disc = s*s - 4*p
    if disc < 0:
        return False, ()
    num = Integer(disc.numerator())
    den = Integer(disc.denominator())
    if not (num.is_square() and den.is_square()):
        return False, ()
    sqrt_num = Integer(num.sqrt())
    sqrt_den = Integer(den.sqrt())
    sqrt_disc = QQ(sqrt_num) / QQ(sqrt_den)
    r1 = (s + sqrt_disc) / 2
    r2 = (s - sqrt_disc) / 2
    return True, (r1, r2)

def _drop_if_both_roots_give_rational_points(div, f_coeffs_or_curve):
    """
    True if both roots of u are rational AND f(root) is a rational perfect square for both roots.
    f_coeffs_or_curve accepts the same input you pass to the filter (either curve object or coeff list).
    Raises on unexpected input (no silent swallowing).
    """
    ok, roots = _u_roots_rational(div)
    if not ok:
        return False

    # Coerce f polynomial: accept either Curve object or coefficient list.
    if hasattr(f_coeffs_or_curve, 'hyperelliptic_polynomials'):
        C = f_coeffs_or_curve
    else:
        C, _, _ = _build_curve_from_coeffs(f_coeffs_or_curve)
    f_poly = C.hyperelliptic_polynomials()[0]

    r1, r2 = roots

    v1 = f_poly(r1)
    v2 = f_poly(r2)

    return _is_perfect_square_Q(v1) and _is_perfect_square_Q(v2)


def u_has_rational_root_lifting_to_point(div, f_coeffs_or_curve):
    # get curve
    if hasattr(f_coeffs_or_curve, 'hyperelliptic_polynomials'):
        C = f_coeffs_or_curve
    else:
        C, _, _ = _build_curve_from_coeffs(f_coeffs_or_curve)

    f = C.hyperelliptic_polynomials()[0]

    s = QQ(div['s'])
    p = QQ(div['p'])

    disc = s*s - 4*p
    if disc < 0:
        return False

    a = Integer(disc.numerator())
    b = Integer(disc.denominator())
    if not (a.is_square() and b.is_square()):
        return False

    sqrt_disc = QQ(Integer(a.sqrt())) / QQ(Integer(b.sqrt()))
    r1 = (s + sqrt_disc) / 2
    r2 = (s - sqrt_disc) / 2

    for r in (r1, r2):
        v = f(r)
        if v < 0:
            continue
        n = Integer(v.numerator())
        d = Integer(v.denominator())
        if n.is_square() and d.is_square():
            return True

    return False


# Put near other helpers (top-level). No imports inside functions.


def compute_canonical_height_with_budget(div, f_coeffs, debug=False):
    try:
        J_elem = mumford_divisor_to_jacobian(div, f_coeffs)
    except Exception:
        raise
        return None  # conversion failure is legitimate signal

    for p in [1024]:
        try:
            h = arakelov_canonical_height(J_elem, f_coeffs, prec=p, debug=False)
            if h is not None and h >= 0:
                return h
        except Exception:
            print("badness")
            raise

    return None


def mumford_divisor_to_jacobian(div, f_coeffs):
    if not isinstance(div, dict):
        raise TypeError("Expected Mumford divisor dict")

    try:
        s = QQ(div['s'])
        p = QQ(div['p'])

        R = PolynomialRing(QQ, 'x')
        x = R.gen()

        u = x**2 - s*x + p
        v = QQ(div['v_1'])*x + QQ(div['v_0'])


        return mumford_pair_to_jacobian(u, v, f_coeffs)

    except Exception as e:
        raise ValueError(
            f"Failed to convert Mumford divisor to Jacobian element: {div}"
        ) from e


from sage.all import PolynomialRing, QQ, HyperellipticCurve, Jacobian

def mumford_pair_to_jacobian(u, v, f_coeffs):
    R = PolynomialRing(QQ, 'x')
    x = R.gen()
    J = Jacobian(HyperellipticCurve(R(list(reversed(f_coeffs)))))
    return J(u, v)


# In mumford_basis.py

def filter_kobayashi_maru(divs, f_coeffs_or_curve, debug=True, aggressive=False):
    """
    Filter divisors that are not finite Jacobian elements or duplicates.

    Accepts either:
      - f_coeffs (list-like): list of coefficients for y^2 = f(x)
      - C: a HyperellipticCurve object

    aggressive: when True, also drop other tiny/structural divisors (heuristic).
    """
    f_coeffs = f_coeffs_or_curve
    # accept either f_coeffs or curve
    if hasattr(f_coeffs_or_curve, 'hyperelliptic_polynomials'):
        C = f_coeffs_or_curve
    else:
        C, _, _ = _build_curve_from_coeffs(f_coeffs_or_curve)

    out = []
    seen = set()
    store_count = 0

    def _is_u_x_squared(rec):
        """
        True if u(x) == x^2 (i.e. s == 0 and p == 0).
        Use safe coercion.
        """
        try:
            s_q = _to_QQ_safe(rec.get('s', 0))
            p_q = _to_QQ_safe(rec.get('p', 0))
            return s_q == 0 and p_q == 0
        except Exception:
            raise
            return False

    def _is_structural_small(rec):
        # a tiny heuristic: u-coeffs in {-1,0,1} OR explicit flagged structural
        if rec.get('u', None) is not None:
            try:
                coeffs = rec['u'].list()
                if all(c in (-1, 0, 1) for c in coeffs):
                    return True
            except Exception:
                raise
        return False

    for div in divs:
        # quick structural short-circuit: explicit drop for u(x)=x^2
        if _is_u_x_squared(div):
            if debug:
                import warnings
                warnings.warn(f"[filter] Dropping explicit u(x)=x^2 divisor: {div}", RuntimeWarning)
            # absolutely drop u(x)=x^2 since it repeatedly causes
            # archimedean/theta blowups and negative heights in your logs.
            continue
            
        # Detect numerically suspicious divisors (tiny naive height)
        if naive_height_suspicion(div):
            if debug:
                warnings.warn(f"[filter] Dropping numerically suspicious (tiny height) divisor: {div}", RuntimeWarning)
            continue

        if u_is_theta_degenerate(div):
            warnings.warn(
                f"[filter] Dropping theta-degenerate divisor: {div}",
                RuntimeWarning
            )
            continue

        if u_theta_degenerate_from_sp(div):
            warnings.warn(
                f"[filter] Dropping theta-degenerate divisor (Δ≈0): {div}",
                RuntimeWarning
            )
            continue

        # inside the loop over divs, early, before heavy work:
        if u_theta_degenerate_enhanced(div, debug=debug):
            warnings.warn(f"[filter] Dropping enhanced theta-degenerate divisor: {div}", RuntimeWarning)
            continue

        # compute diagonal heights once:
        # compute_canonical_height_with_budget now benefits from the ValueError raise in heights.py
        h_diag = compute_canonical_height_with_budget(div, f_coeffs, debug=debug)

        if h_diag is None:
            warnings.warn(
                f"[filter] Dropping divisor due to unstable negative/failure canonical height: {div}",
                RuntimeWarning
            )
            continue


        # inside loop over divs, BEFORE mumford_to_jacobian_element call:
        flagged, reason = u_is_problematic(div, f_coeffs_or_curve, C=C, debug=debug)
        if flagged:
            if debug:
                warnings.warn(f"[filter] Dropping divisor by {reason}: {div}", RuntimeWarning)
            continue


        try:
            D = mumford_to_jacobian_element(div['s'], div['p'], div['v_0'], div['v_1'], C)
        except Exception as e:
            if debug:
                warnings.warn(f"[filter] failed to coerce div -> Jacobian element: {e} -- skipping", RuntimeWarning)
            raise
            continue

        # ignore points that end up not finite / invalid in Jacobian
        try:
            if getattr(D, "is_finite", None) and not D.is_finite():
                if debug:
                    warnings.warn(f"[filter] skipping non-finite jacobian element for div {div}", RuntimeWarning)
                continue
        except Exception:
            # Some Sage types might not have is_finite; skip the check then.
            pass

        # optional aggressive structural drop (toggle via aggressive=True)
        if aggressive and _is_structural_small(div):
            if debug:
                warnings.warn(f"[filter] Aggressively dropping structurally-tiny divisor: {div}", RuntimeWarning)
            continue

        # use reduced representation as a canonical dedupe key
        try:
            key = D.reduced_representation()
        except Exception:
            # Some Jacobian elements may not support reduced_representation consistently;
            # fall back to string repr as a last resort
            try:
                key = str(D)
            except Exception:
                # if even that fails, skip to be safe
                if debug:
                    warnings.warn(f"[filter] cannot build dedupe key for divisor {div}; skipping", RuntimeWarning)
                continue 

        if key in seen:
            # dedup
            continue

        seen.add(key)
        out.append(div)
        # store it so pairings don't recompute it
        div['_h_diag'] = h_diag
        store_count += 1
        if store_count >= MAX_BASIS_CANDIDATES:
            break


    from pprint import pprint
    pprint(dict(_FILTER_STATS))
    return out
