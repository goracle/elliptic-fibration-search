from sage.all import *

def build_hyperelliptic_poly(coeffs: Sequence[Any], x_sym=None, base_ring=None, descending: bool = True):
    """Build a polynomial from coefficients or pass through an existing polynomial."""
    if hasattr(coeffs, "parent") and hasattr(coeffs, "degree"):
        return coeffs

    base_ring = base_ring or QQ
    if x_sym is None:
        R = PolynomialRing(base_ring, "x")
        x_sym = R.gen()
    else:
        R = x_sym.parent()

    coeffs = list(coeffs)
    if descending:
        deg = len(coeffs) - 1
        poly = sum(base_ring(coeffs[i]) * x_sym ** (deg - i) for i in range(len(coeffs)))
    else:
        poly = sum(base_ring(c) * x_sym ** i for i, c in enumerate(coeffs))
    return poly

def flatten_roots(roots_with_mult: Sequence[Tuple[Any, int]]) -> List[Any]:
    out: List[Any] = []
    for root, mult in roots_with_mult:
        out.extend([root] * int(mult))
    return out

def _coerce_base_ring(p: Optional[int], base_ring: Optional[Any] = None):
    if base_ring is not None:
        return base_ring
    if p is None:
        return QQ
    return GF(int(p))

def _collect_candidate_x_like_values(obj: Any, out: Optional[List[Any]] = None) -> List[Any]:
    """Fallback collector for x-like payloads in legacy return values.

    This is intentionally permissive and is only used when the Mumford result
    does not expose its x-residue map in a recognizable shape.
    """
    if out is None:
        out = []

    if obj is None:
        return out

    scalar_types = (int, float, complex, str)
    try:
        scalar_types = scalar_types + (Integer,)
    except Exception:
        raise

    if isinstance(obj, dict):
        for key in ('pt_step', 'pt', 'candidate_pt', 'pt_value'):
            if key in obj and obj[key] is not None:
                out.append(obj[key])
        for value in obj.values():
            _collect_candidate_x_like_values(value, out)
        return out

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) in (1, 2, 3) and all(not isinstance(v, (dict, list, tuple, set)) for v in seq):
            out.extend(seq)
            return out
        for value in seq:
            _collect_candidate_x_like_values(value, out)
        return out

    try:
        if isinstance(obj, scalar_types):
            out.append(obj)
            return out
    except Exception:
        raise

    try:
        if hasattr(obj, 'parent') or hasattr(obj, 'degree') or hasattr(obj, 'numerator'):
            out.append(obj)
            return out
    except Exception:
        raise

    return out

def xk_is_fp_point(xk_val, G_poly):
    if isinstance(xk_val, tuple):
        xk_val = xk_val[0]
    if G_poly is None or xk_val is None:
        return False

    try:
        rhs = G_poly(xk_val)
        return bool(hasattr(rhs, "is_square") and rhs.is_square())
    except Exception:
        raise
        return False

def poly_roots_with_multiplicity(poly) -> List[Tuple[Any, int]]:
    """Return roots as (root, multiplicity) pairs over the polynomial's base field."""
    roots = poly.roots(multiplicities=True)
    assert roots, roots
    return [(r, int(m)) for r, m in roots]

