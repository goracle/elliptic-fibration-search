# file: search_lll/diag_div.sage
from search_lll.jacobianbasis import (
    archimedean_height_correction,
    arakelov_canonical_height,
    get_period_matrix_auto_B,
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
D = J([u, v])

prec = 8192   # high precision
PM = get_period_matrix_auto_B(f_coeffs, prec=prec)

print("Running archimedean_height_correction (debug True)...")
arch = archimedean_height_correction(D, f_coeffs, PM, prec=prec, debug=True)
print("Archimedean contribution:", arch)

print("Computing full arakelov_canonical_height at same precision...")
h_can = arakelov_canonical_height(D, f_coeffs, prec=prec)
print("Canonical height (h_can):", h_can)

# hard assertion: canonical height must be non-negative
if h_can < 0:
    raise RuntimeError("Canonical height negative; see diagnostics above.")
