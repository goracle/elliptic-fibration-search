# IMPORTANT: run as a module, not a script

from search_lll.jacobianbasis import (
    archimedean_height_correction,
    get_period_matrix_auto_B,
)

from search_common import *
from sage.all import *

# ---- paste your curve data here ----
f_coeffs = COEFFS_GENUS2   # or paste explicit list

# Build curve + divisor
R.<x> = QQ[]
C = HyperellipticCurve(sum(
    QQ(c) * x^(len(f_coeffs)-1-i)
    for i, c in enumerate(f_coeffs)
))
J = C.jacobian()

u = x^2 - 4*x
v = 11*x - 8
D = J([u, v])

prec = 4096
PM = get_period_matrix_auto_B(f_coeffs, prec=prec)

print("Computing archimedean height…")
h = archimedean_height_correction(
    D, f_coeffs, PM, prec=prec, debug=True
)
print("Archimedean contribution:", h)
