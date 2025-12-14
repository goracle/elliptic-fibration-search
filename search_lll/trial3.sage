# diagnostics for negative canonical height
from search_lll.jacobianbasis import (
    get_period_matrix_auto_B,
    archimedean_height_correction,
    arakelov_quasi_height,
    naive_height_qq,
    get_bad_primes,
    local_height_correction_finite,
)
from sage.all import QQ
from search_common import *
R.<x> = QQ[]

# replace COEFFS_GENUS2 with your coefficients list if not already defined
f_coeffs = COEFFS_GENUS2

# build curve and jacobian (same pattern you used earlier)
C = HyperellipticCurve(sum(QQ(c) * x^(len(f_coeffs)-1-i) for i, c in enumerate(f_coeffs)))
J = C.jacobian()

# failing divisor from your log: (x^2 + x, y - 9*x - 8)
u = x^2 + x
v = 9*x + 8
div = J([u, v])




prec = 1024   # start modest; increase later to check convergence
print("Using precision:", prec)

PM = get_period_matrix_auto_B(f_coeffs, prec=prec)
print("Period matrix computed.")

# print archimedean part (with debug=True to get the internal prints)
h_arch = archimedean_height_correction(div, f_coeffs, PM, prec=prec, debug=True)
print("archimedean contribution (h_arch) =", h_arch, "≈", float(h_arch))

# helper to compute finite-correction sum and per-prime output
def finite_sum(divpoint):
    bad = get_bad_primes(f_coeffs)
    s = 0.0
    print("bad primes:", bad)
    for p in bad:
        try:
            v = local_height_correction_finite(divpoint, p, f_coeffs)
        except Exception as e:
            v = f"EXN: {e}"
        print("  p =", p, " => ", v)
        if isinstance(v, (int, float)):
            s += v
    return s

# compute for div, 2div, 3div
div1, div2, div3 = div, div + div, div + div + div

print("\n=== naive heights ===")
print("naive(div)  =", naive_height_qq(div1, prec=prec))
print("naive(2div) =", naive_height_qq(div2, prec=prec))
print("naive(3div) =", naive_height_qq(div3, prec=prec))

print("\n=== archimedean overrides (testing linear-scaling behavior) ===")
print("h_arch (for div)         =", float(h_arch))
print("h_arch * 2 (for 2div)    =", float(h_arch * 2))
print("h_arch * 3 (for 3div)    =", float(h_arch * 3))

print("\n=== quasi-heights with finite places DISABLED (so we see architecure) ===")
h1_no_f = arakelov_quasi_height(div1, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=False, arch_override=h_arch)
h2_no_f = arakelov_quasi_height(div2, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=False, arch_override=h_arch*2)
h3_no_f = arakelov_quasi_height(div3, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=False, arch_override=h_arch*3)
print("h1_no_f =", float(h1_no_f))
print("h2_no_f =", float(h2_no_f))
print("h3_no_f =", float(h3_no_f))
print("h_can_no_f = (h3+h1-2*h2)/2 =", float((h3_no_f + h1_no_f - 2*h2_no_f)/2))

print("\n=== finite-place contributions (for div, 2div, 3div) ===")
fs1 = finite_sum(div1)
fs2 = finite_sum(div2)
fs3 = finite_sum(div3)
print("finite-sum(div)  ≈", fs1)
print("finite-sum(2div) ≈", fs2)
print("finite-sum(3div) ≈", fs3)

print("\n=== quasi-heights WITH finite places (full h) ===")
h1 = arakelov_quasi_height(div1, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=True, arch_override=h_arch)
h2 = arakelov_quasi_height(div2, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=True, arch_override=h_arch*2)
h3 = arakelov_quasi_height(div3, f_coeffs, period_matrix=PM, prec=prec, use_finite_places=True, arch_override=h_arch*3)
print("h1 (full) =", float(h1))
print("h2 (full) =", float(h2))
print("h3 (full) =", float(h3))

h_can = (h3 + h1 - 2*h2) / QQ(2)
print("\nFINAL computed h_can =", h_can, "≈", float(h_can))
