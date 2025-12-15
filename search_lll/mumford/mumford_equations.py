from sage.all import var

def build_mumford_equations_from_fibration(tower, f_coeffs):
    return {
        'f_coeffs': f_coeffs,
        'const': 0,
        'm_sym': var('m')
    }


