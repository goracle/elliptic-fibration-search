import re
from sage.all import PolynomialRing, QQ

def parse_hyperelliptic_db_entry(db_string):
    """
    Parse a hyperelliptic curve entry from the MIT database and extract coefficients.
    https://math.mit.edu/~drew/gce_genus3_hyperelliptic.txt
    Input format: D:N:[f(x),h(x)]
    where the curve is: y^2 + h(x)*y = f(x)

    We transform this to: Y^2 = h(x)^2 + 4*f(x)
    where Y = 2*y + h(x)

    Returns a coefficient vector [c_0, c_1, ..., c_n] where
    the right-hand side polynomial is c_0 + c_1*x + c_2*x^2 + ...

    Args:
        db_string: String like "10000000:2000000:[-5*x^7-4*x^6-3*x^5-2*x^4,x^3+x^2+x+1]"

    Returns:
        list of QQ coefficients (low to high degree)
    """

    # Parse the database format: D:N:[f(x),h(x)]
    # Extract the part inside the brackets
    match = re.search(r'\[(.*?)\]$', db_string)
    if not match:
        raise ValueError(f"Could not parse database string: {db_string}")

    poly_part = match.group(1)

    # Split by comma at the top level (not inside nested parens)
    parts = []
    depth = 0
    current = ""
    for char in poly_part:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())

    if len(parts) != 2:
        raise ValueError(f"Expected 2 polynomials (f and h), got {len(parts)}: {parts}")

    f_str, h_str = parts

    # Create polynomial ring for parsing
    PR = PolynomialRing(QQ, 'x')
    x = PR.gen()

    # Replace ^ with ** for Python exponentiation
    f_str = f_str.replace('^', '**')
    h_str = h_str.replace('^', '**')

    # Create a safe namespace with the polynomial variable
    namespace = {'x': x}

    # Parse polynomials
    try:
        f = eval(f_str, {"__builtins__": {}}, namespace)
        h = eval(h_str, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(f"Could not parse polynomials: f={f_str}, h={h_str}. Error: {e}")

    # Compute the transformed RHS: h(x)^2 + 4*f(x)
    rhs_poly = h**2 + 4*f

    # Polynomials in Sage's FLINT ring are already expanded, no need to call .expand()

    # Extract coefficients (low to high degree)
    coeffs = rhs_poly.coefficients(sparse=False)

    # Convert to QQ-wrapped integers
    coeffs = [QQ(int(c)) for c in coeffs]

    return coeffs

