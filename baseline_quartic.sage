# baseline_quartic.sage
#
# BASELINE experiment for the "rank = number of seeds" conjecture.
#
# Builds a GENERIC quartic fibration  y^2 = h(x; m)  over QQ(m) with n prescribed
# sections (x_j, Y_j(m)), x_j constant and Y_j(m) polynomial in m, and NO relation
# to a genus-2 curve (no tangency/rail structure).  Then runs your existing
# pipeline (buildcd -> base sections -> LLL -> Picard) and reports
#       n, chi, sum(m_v-1), rho_lower, best "upper", rho, implied rank.
#
# Purpose: compare against the tower-built surfaces.  If the generic family also
# gives rho = 2 + Sigma + n, the tower is not forcing anything special; if the
# tower surfaces (n <= 3) differ from this baseline, that is the finding.
#
# Usage (from the elliptic-fibration-search directory):
#     sage baseline_quartic.sage            # n=4 default
#     BASE_N=5 BASE_SEED=7 sage baseline_quartic.sage
#     BASE_N=6 BASE_SEED=1 sage baseline_quartic.sage   # n=6..8 via gen_sections.py
#     BASE_N=8 BASE_GEN=evenodd sage baseline_quartic.sage   # BASE_GEN: isotropic | evenodd
#
# NOT TESTED by me.  Every stage prints a banner so a failure is easy to locate.

import os, sys, random, traceback
from sage.all import *
from search_common import *
from tate import *
from picard import *
from diagnostics2 import *

N_SECTIONS = int(os.environ.get("BASE_N", "4"))      # 1..8  (n >= 6 uses gen_sections.py)
SEED       = int(os.environ.get("BASE_SEED", "1"))
YDEG       = int(os.environ.get("BASE_YDEG", "1"))   # degree in m of each Y_j(m)
XNODES     = [0, 1, -1, 2, -2][:N_SECTIONS]
GEN_ENTRIES = None   # (a_j, b_j) when the generator supplies the sections

assert 1 <= N_SECTIONS <= 8, "rank of a rational elliptic surface is <= 8"
GEN_COEFFS = None    # [c0, c1, c2] per section when YDEG == 2 (K3, chi = 2)
if N_SECTIONS >= 6 and YDEG == 2:
    assert N_SECTIONS in (6, 7), "BASE_YDEG=2 generator exists for n=6 and n=7 only"
    from gen_sections_deg2 import generate_ydeg2
    _nodes, GEN_COEFFS = generate_ydeg2(N_SECTIONS, seed=SEED)
    XNODES = [QQ(xj) for xj in _nodes]
    GEN_COEFFS = [[QQ(c) for c in cj] for cj in GEN_COEFFS]
elif N_SECTIONS >= 6:
    # Beyond 5 nodes the Y_j^2 cannot be chosen freely (the interpolant would exceed degree 4).
    # gen_sections.py solves the isotropic-plane condition exactly (n=6) or by search (n=7,8).
    assert YDEG == 1, "gen_sections only builds Y_j(m) = a_j*m + b_j (BASE_YDEG=1); use BASE_YDEG=2 with n=6 or 7"
    from gen_sections import generate
    _nodes, _a, _b = generate(N_SECTIONS, seed=SEED, method=os.environ.get("BASE_GEN") or None)
    XNODES = [QQ(xj) for xj in _nodes]
    GEN_ENTRIES = [(QQ(aj), QQ(bj)) for aj, bj in zip(_a, _b)]
random.seed(SEED)

def banner(s):
    print("\n" + "=" * 70); print(s); print("=" * 70); sys.stdout.flush()

# ---------------------------------------------------------------- ring setup
Pm = PolynomialRing(QQ, 'm')
Fm = FractionField(Pm)
m = Fm.gen()
R_xm = PolynomialRing(Fm, 'x')
x = R_xm.gen()

def rand_poly_in_m(deg, nonzero_const=True):
    """Small random integer polynomial in m of degree exactly `deg`."""
    coeffs = [ZZ(random.randint(-3, 3)) for _ in range(deg)]
    lead = ZZ(random.choice([-3, -2, -1, 1, 2, 3]))
    c0 = ZZ(random.choice([-3, -2, -1, 1, 2, 3])) if nonzero_const else ZZ(random.randint(-3, 3))
    if deg == 0:
        return Fm(c0)
    coeffs[0] = c0
    return Fm(sum(c * m**i for i, c in enumerate(coeffs)) + lead * m**deg)

# ---------------------------------------------------------------- build h
banner(f"BASELINE: n={N_SECTIONS} sections, nodes x={XNODES}, deg_m(Y_j)={YDEG}, seed={SEED}")

if GEN_COEFFS is not None:
    Ys = [sum((Fm(c) * m**k for k, c in enumerate(cj)), Fm(0)) for cj in GEN_COEFFS]
elif GEN_ENTRIES is not None:
    Ys = [Fm(aj) * m + Fm(bj) for aj, bj in GEN_ENTRIES]
else:
    Ys = [rand_poly_in_m(YDEG) for _ in XNODES]      # Y_j(m), nonzero for all m
vals = [Y**2 for Y in Ys]                             # required h(x_j) = Y_j^2

# Lagrange interpolant of the values (degree <= n-1 in x)
L = R_xm(0)
for j, xj in enumerate(XNODES):
    term = R_xm(vals[j])
    for k, xk in enumerate(XNODES):
        if k != j:
            term *= (x - Fm(xk)) / Fm(xj - xk)
    L += term

prod = R_xm(1)
for xj in XNODES:
    prod *= (x - Fm(xj))

# fill up to degree exactly 4 with a random multiple of prod (free parameters)
if N_SECTIONS < 5:
    free_deg = 4 - N_SECTIONS
    R_free = R_xm([rand_poly_in_m(YDEG, nonzero_const=True) for _ in range(free_deg)]
                  + [rand_poly_in_m(YDEG, nonzero_const=True)])
    h = L + prod * R_free
else:
    h = L

assert h.degree() == 4, f"h has degree {h.degree()} in x, expected 4 (try another BASE_SEED)"
for xj, Y in zip(XNODES, Ys):
    assert h(Fm(xj)) == Y**2, f"section check failed at x={xj}"
print("h(x;m) =", h)
print("sections: ", [(xj, Y) for xj, Y in zip(XNODES, Ys)])

# ---------------------------------------------------------------- curve data
banner("STAGE 1: morphism to Weierstrass + buildcd (your code)")
try:
    E_curve_m, one, two, three = compute_morphism(h)
    m_sr = SR.var('m')
    rail = SR(XNODES[0]) - m_sr                 # same form as tower rail; only used by buildcd bookkeeping
    lastrhs = h(x=rail)
    last_phi_x = get_phi_x(one, two, three, rail, lastrhs)
    cd = buildcd(E_curve_m, last_phi_x, lastrhs, h, (one, two, three))
except Exception:
    traceback.print_exc()
    print("\nSTAGE 1 FAILED. Most likely culprit: get_phi_x at the artificial rail "
          "(there is no genuine tangency rail in this baseline). Send me the traceback.")
    sys.exit(1)

# stage-boundary invariant: Shioda-Tate / Lefschetz bound  rank <= 10*chi - 2 - Sigma
_chi = QQ(cd.singfibs['euler_characteristic']) / 12
_sig = cd.singfibs['sigma_sum']
print(f"chi = {_chi}, Sigma = {_sig}, rank bound 10*chi-2-Sigma = {10*_chi - 2 - _sig}")
assert N_SECTIONS <= 10*_chi - 2 - _sig, (
    f"this family cannot carry {N_SECTIONS} independent sections: bound is {10*_chi - 2 - _sig} "
    f"(chi={_chi}, Sigma={_sig})")

# ---------------------------------------------------------------- sections
banner("STAGE 2: sections")
base_pts = [(Fm(xj), Y) for xj, Y in zip(XNODES, Ys)]
try:
    sections = compute_base_sections_m(cd, base_pts, tower=None)
    verify_morphism_on_samples(cd, base_pts)
    sections = lll_reduce_mw_basis(cd, sections)
except Exception:
    traceback.print_exc()
    print("\nSTAGE 2 FAILED. If the error is in one_use(x=..., y=...), the morphism wrapper "
          "does not accept m-dependent y; tell me and I will map the points by hand.")
    sys.exit(1)
print("number of sections:", len(sections))

independent, H = check_independence(sections, None, cd)
print("independent in char 0:", independent)
print("height matrix:\n", H)
print("det(H) =", H.det())

# ---------------------------------------------------------------- Picard
banner("STAGE 3: Picard analysis (patched picard.py)")
singfibs = cd.singfibs
euler = singfibs['euler_characteristic']
sigma = singfibs['sigma_sum']
print("Euler characteristic:", euler, " Sigma(m_v-1):", sigma)

prime_pool = [p for p in primes(3, 400)]
ell_candidates = tuple([p for p in prime_pool if p not in cd.bad_primes][:12])
print("ell candidates:", ell_candidates)

try:
    rep = picard_via_van_luijk(cd, sections, prime_pool, ell_candidates=ell_candidates, verbose=True)
except Exception:
    traceback.print_exc()
    print("\nSTAGE 3 FAILED (Picard).")
    sys.exit(1)

# ---------------------------------------------------------------- summary
banner("SUMMARY (generic quartic baseline)")
lo = rep['lower_bound']
lbs = sorted(set(u for u, _ in rep.get('reduction_lower_bounds', [])))
print(f"n sections            : {len(sections)}")
print(f"chi (= e/12)          : {QQ(euler)/12}")
print(f"Sigma(m_v - 1)        : {sigma}")
if rep.get('rho_rigorous'):
    print(f"rho (exact)           : {rep['rho']}   [{rep['rho_method']}]")
    print(f"geometric MW rank     : {rep['mw_rank_geometric']}   (= 8 - Sigma)")
    print(f"independent sections  : {rep['mw_rank_found']}   "
          f"(finite-index sublattice: {rep['sections_span_finite_index']})")
else:
    print(f"rho lower (char 0)    : {lo}   (= 2 + Sigma + n = {2 + sigma + len(sections)})")
    print(f"mod-ell lower bounds  : {lbs}   [lower bounds on rho(S_ell) only; NOT upper bounds on rho(S)]")
    print("rho                   : not determined (no rigorous upper bound implemented for chi >= 2)")
print(f"skipped primes        : {rep['collapsed_primes']}")
