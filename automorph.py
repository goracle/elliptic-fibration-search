# autos_k3.sage
# Compute candidate automorphisms of an elliptic K3 surface (preserving the elliptic fibration)
# Uses exact arithmetic over QQ and rational function field QQ(m).
#
# Assumptions:
#  - cd is available and has attributes cd.a4 and cd.a6 in a polynomial/rational-function ring whose
#    coefficient field is QQ and whose base generator is the "m" parameter (possibly a polynomial ring generator).
#  - find_singular_fibers(cd) exists and returns dict with key 'fibers' list of dicts with 'r' entries.
#  - base_sections is a list of pairs (xS(m), yS(m)) where xS, yS are expressions coercible to SR or F_m.
#
# API:
#  - mobius_from_3points(p1,p2,p3, q1,q2,q3) -> (a,b,c,d) in SR representing (a m + b)/(c m + d)
#  - find_mobius_candidates(centers, m_sym) -> list of (a,b,c,d)
#  - test_constant_scaling(cd, mobius_tuple, m_sym) -> QQ(u) or None
#  - translation_map_section(section, m_sym) -> (x3_SR, y3_SR) symbolic map for P -> P + S
#  - compute_auts_preserving_fibration(cd, base_sections, m_sym) -> dict report with autos list
#
# Writing style: short functions, explicit raising, no nested functions, no imports inside functions.


# automorph_cleaned.py
# Computes candidate automorphisms of an elliptic K3 surface.
# Uses exact arithmetic over QQ and the rational function field QQ(m).

from sage.all import (
    QQ,
    PolynomialRing,
    SR,
    var,
    Infinity,
    solve,
    Integer,
    Matrix,
    QuadraticForm,
    vector,
    ZZ,
)
import itertools
import time
from diagnostics2 import *

# NOTE: The following dependencies are assumed to be available
# from diagnostics2 import find_singular_fibers, build_ns_gram
# These functions are not defined in this file and must be provided externally.

# -------------------------
# Helpers (top-level)
# -------------------------
def _coerce_m_symbol(cd):
    """
    Infers the base generator symbol for the coefficient ring of cd.a4.
    Returns a Sage symbolic var.
    """
    try:
        parent = cd.a4.parent()
        if hasattr(parent, 'gen'):
            return parent.gen()
    except Exception:
        pass
    return var('m')

def _is_QQ_constant_rational_in_QQm(expr, m_sym):
    """
    Decides if an expression in QQ(m) is a constant in QQ.
    Returns (True, QQ(value)) or (False, None).
    """
    try:
        # Coerce to a rational function in m_sym over QQ
        PR = PolynomialRing(QQ, str(m_sym))
        F = PR.fraction_field()
        fr = F(expr)
        
        num = fr.numerator()
        den = fr.denominator()
        
        # Check if degrees of numerator and denominator are zero
        if num.degree() == 0 and den.degree() == 0:
            val = QQ(num[0]) / QQ(den[0])
            return True, val
        
        # Additional check for expressions that are symbolically constant, e.g., '1'
        if not expr.variables():
            return True, QQ(expr)

    except (TypeError, ValueError, AttributeError):
        # Fallback for simple cases that fail coercion
        try:
            val = QQ(expr)
            return True, val
        except (TypeError, ValueError):
            pass
    
    return False, None


def mobius_from_3points(p1, p2, p3, q1, q2, q3):
    """
    Solves for a Möbius map (a*m + b) / (c*m + d) sending p_i -> q_i.
    Points may be Infinity or algebraic/QQ/SR. Returns (a,b,c,d) as SR expressions.
    Raises ValueError if no solution.
    """
    a, b, c, d = var('a b c d')
    Sa, Sb, Sc, Sd = SR(a), SR(b), SR(c), SR(d)

    def _is_inf(v):
        try:
            return v is None or v == Infinity
        except Exception:
            return False

    triples = [(p1, q1), (p2, q2), (p3, q3)]
    eqs = []
    for p, q in triples:
        p_inf = _is_inf(p)
        q_inf = _is_inf(q)
        if p_inf:
            if q_inf:
                # phi(infty) = a/c = infty implies c = 0
                eqs.append(Sc)
            else:
                # phi(infty) = a/c = q implies a - q*c = 0
                eqs.append(Sa - SR(q) * Sc)
        elif q_inf:
            # phi(p) = (a*p + b)/(c*p + d) = infty implies c*p + d = 0
            eqs.append(Sc * SR(p) + Sd)
        else:
            # a*p + b = q*(c*p + d)
            eqs.append(Sa * SR(p) + Sb - SR(q) * (Sc * SR(p) + Sd))

    sols = solve(eqs, a, b, c, d, solution_dict=True)
    if not sols:
        raise ValueError("mobius_from_3points: no Möbius solution for these triples")
    
    sol = sols[0]
    # Normalize by one of the coefficients to get a unique representation
    if sol[d]:
        return sol[a]/sol[d], sol[b]/sol[d], sol[c]/sol[d], SR(1)
    elif sol[c]:
        return sol[a]/sol[c], sol[b]/sol[c], SR(1), sol[d]/sol[c]
    elif sol[a]:
        return SR(1), sol[b]/sol[a], sol[c]/sol[a], sol[d]/sol[a]
    elif sol[b]:
        return sol[a]/sol[b], SR(1), sol[c]/sol[b], sol[d]/sol[b]
    
    raise ValueError("mobius_from_3points: normalization failed")


def _apply_mobius_on_m(m_sym, mobius_tuple):
    """
    Applies a Möbius transformation to the symbolic variable m.
    """
    a, b, c, d = mobius_tuple
    return (a * m_sym + b) / (c * m_sym + d)

def find_mobius_candidates(centers, m_sym, max_triples=500):
    """
    Attempts to find Möbius maps (a,b,c,d) that permute the finite set `centers`.
    `centers`: iterable of algebraic/QQ/SR values (use None for Infinity).
    Returns a list of unique (a,b,c,d) SR tuples.
    """
    norm_centers = []
    for c in centers:
        norm_centers.append(Infinity if c is None else SR(c))

    n = len(norm_centers)
    if n < 3:
        return []

    # Build list of ordered triples with distinct entries
    triples = [
        (norm_centers[i], norm_centers[j], norm_centers[k])
        for i in range(n)
        for j in range(n)
        for k in range(n)
        if i != j and j != k and i != k
    ]

    # Limit combinatorial explosion
    if len(triples) > max_triples:
        triples = triples[:max_triples]

    candidates = []
    seen = set()
    for s_triple in triples:
        for t_triple in triples:
            try:
                a, b, c, d = mobius_from_3points(s_triple[0], s_triple[1], s_triple[2],
                                                 t_triple[0], t_triple[1], t_triple[2])
            except Exception:
                continue

            phi_m = _apply_mobius_on_m(m_sym, (a, b, c, d))

            # Validate phi permutes the entire center set
            ok = True
            image_set = set()
            for center in norm_centers:
                try:
                    image = phi_m.subs({m_sym: center})
                    image_set.add(image)
                except Exception:
                    ok = False
                    break
            
            if not ok:
                continue

            # Check if the set of images is the same as the set of original centers
            if len(image_set) != len(norm_centers):
                 ok = False # image set size differs, not a permutation
            else:
                for img in image_set:
                    is_in_centers = False
                    for orig_center in norm_centers:
                        try:
                            if (SR(img) - SR(orig_center)) == 0:
                                is_in_centers = True
                                break
                        except Exception:
                            if str(img) == str(orig_center):
                                is_in_centers = True
                                break
                    if not is_in_centers:
                        ok = False
                        break
            
            if ok:
                key = (str(a), str(b), str(c), str(d))
                if key not in seen:
                    seen.add(key)
                    candidates.append((a, b, c, d))
    return candidates

def test_constant_scaling(cd, mobius_tuple, m_sym):
    """
    Tests whether there exists a constant u in QQ* such that
    a4(phi(m)) == u^4 * a4(m) and a6(phi(m)) == u^6 * a6(m).
    Returns QQ(u) if found, else None.
    """
    phi_m = _apply_mobius_on_m(m_sym, mobius_tuple)
    
    try:
        r4 = SR(cd.a4).subs({m_sym: phi_m}) / SR(cd.a4)
        r6 = SR(cd.a6).subs({m_sym: phi_m}) / SR(cd.a6)
    except Exception as exc:
        raise RuntimeError("test_constant_scaling: substitution failed") from exc

    ok4, val4 = _is_QQ_constant_rational_in_QQm(r4, m_sym)
    ok6, val6 = _is_QQ_constant_rational_in_QQm(r6, m_sym)

    if not (ok4 and ok6):
        return None

    # Basic consistency check: val6^2 == val4^3
    if QQ(val6)**2 != QQ(val4)**3:
        return None

    # Find rational u by checking perfect-power for numerator and denominator of val4
    num4 = Integer(val4.numerator())
    den4 = Integer(val4.denominator())
    
    def kth_root_integer(n, k):
        if n < 0 and k % 2 == 0:
            return None
        r = Integer(round(abs(n)**(1.0/k)))
        for cand in range(max(0, r - 2), r + 3):
            if Integer(cand)**k == abs(n):
                return -Integer(cand) if n < 0 else Integer(cand)
        return None

    num_root = kth_root_integer(num4, 4)
    den_root = kth_root_integer(den4, 4)
    
    if num_root is not None and den_root is not None:
        u = QQ(num_root) / QQ(den_root)
        if QQ(u)**6 == QQ(val6):
            return u
            
    # Fallback for simple cases, e.g., val4=1, val6=1
    if val4 == QQ(1) and val6 == QQ(1):
        return QQ(1)
        
    return None

def translation_map_section(section, m_sym):
    """
    Given a section (xS(m), yS(m)), returns SR expressions for the
    translation map (x,y) -> (x,y) + S.
    """
    xS, yS = SR(section[0]), SR(section[1])
    x, y = SR.var('x'), SR.var('y')
    
    lam = (y - yS) / (x - xS)
    x3 = lam**2 - x - xS
    y3 = lam * (x - x3) - y
    
    return x3, y3

def compose_map_symbolic(m_map1, x_map1, y_map1, m_map2, x_map2, y_map2, m_sym):
    """
    Composes two symbolic maps.
    The first map (m1, x1, y1) operates on (m_sym, x, y).
    The second map (m2, x2, y2) operates on (m_sym, x, y).
    The composed map is the second map applied to the output of the first.
    Returns the composed expressions (m_comp, x_comp, y_comp).
    """
    x, y = SR.var('x'), SR.var('y')
    
    m1 = SR(m_map1) if m_map1 is not None else m_sym
    x1 = SR(x_map1)
    y1 = SR(y_map1)
    
    m2 = SR(m_map2) if m_map2 is not None else m_sym
    x2 = SR(x_map2)
    y2 = SR(y_map2)
    
    m_comp = SR(m2).subs({m_sym: m1})
    x_comp = SR(x2).subs({m_sym: m1, x: x1, y: y1})
    y_comp = SR(y2).subs({m_sym: m1, x: x1, y: y1})
    
    return m_comp, x_comp, y_comp


# -------------------------
# Main Automorphism Functions
# -------------------------
def compute_auts_preserving_fibration(cd, base_sections, m_sym=None):
    """
    Main entry point to compute automorphisms preserving the elliptic fibration.
    Returns a dictionary with various types of automorphisms found.
    """
    if m_sym is None:
        m_sym = _coerce_m_symbol(cd)
        
    try:
        # Assumes find_singular_fibers is available from diagnostics2
        sing = find_singular_fibers(cd, verbose=False)
    except NameError:
        raise RuntimeError(
            "compute_auts_preserving_fibration: find_singular_fibers(cd) not found"
        )

    centers = [f.get('r') for f in sing.get('fibers', []) if f.get('r') is not None]
    
    mobius_cands = find_mobius_candidates(centers, m_sym)
    
    scaling_autos = []
    for mob in mobius_cands:
        u = test_constant_scaling(cd, mob, m_sym)
        if u is not None:
            scaling_autos.append({'mobius': mob, 'u': u})
    
    translation_autos = []
    if base_sections:
        for i, sec in enumerate(base_sections):
            x3, y3 = translation_map_section(sec, m_sym)
            translation_autos.append({
                'section_index': i,
                'symbolic_map': (x3, y3),
                'section_coords': sec,
            })
            
    return {
        'mobius_candidates': mobius_cands,
        'scaling_autos': scaling_autos,
        'translation_autos': translation_autos,
        'singular_centers': centers,
        'm_sym': m_sym,
    }


def compute_ns_auts(singfibs, sections):
    """
    Computes lattice automorphisms of the NS Gram matrix that preserve the
    effective cone.
    Returns a list of matrices (Matrix(ZZ)) and the basis 'names'.
    """
    # Assumes build_ns_gram is available from diagnostics2
    G, names = build_ns_gram(singfibs, sections)
    
    # Use the robust search function for automorphisms
    auts_raw, names = compute_ns_auts_via_search(
        G, names, bound=1, max_solutions=20, time_limit=30
    )
    
    auts = []
    n = G.nrows()
    E = [vector(ZZ, [1 if i == j else 0 for i in range(n)]) for j in range(n)]

    for A in auts_raw:
        try:
            M = Matrix(ZZ, A)
        except (TypeError, ValueError):
            M = Matrix(ZZ, len(A), len(A[0]), lambda i, j: A[i][j])
            
        # Check that the image of the fiber class has self-intersection 0
        vF_idx = names.index('F')
        vF = E[vF_idx]
        imgF = M * vF
        
        # Fiber class must map to a vector with self-intersection 0
        if (imgF * G * imgF) != 0:
            continue
            
        # Basic integrality check
        if not all(x in ZZ for x in imgF):
            continue
            
        # Heuristic cone preservation check: map of O+sum(sections) should have positive self-pairing
        amp = vector(ZZ, [0] * n)
        amp[names.index('O')] = 1
        amp[names.index('F')] = 1
        for i in range(len(sections)):
            amp[names.index(f"S{i}")] += 1
        img_amp = M * amp
        
        if (img_amp * G * img_amp) <= 0:
            continue
            
        auts.append(M)
        
    return auts, names


def classify_auts(auts, names):
    """
    Classifies each automorphism matrix M based on its action on sections
    and fiber components.
    Returns a list of (M, labels) tuples.
    """
    n = len(names)
    sec_idx = [i for i, nm in enumerate(names) if nm.startswith('S')]
    comp_idx = [i for i, nm in enumerate(names) if nm.startswith('Comp')]
    E = [vector(ZZ, [1 if i == j else 0 for i in range(n)]) for j in range(n)]

    classified = []
    for M in auts:
        labels = []
        
        # Check for section permutation
        perm_map = {}
        is_perm = True
        for i in sec_idx:
            img = M * E[i]
            # Check if the image is a unit vector corresponding to a section
            found = False
            for j in sec_idx:
                if img == E[j]:
                    perm_map[i] = j
                    found = True
                    break
            if not found:
                is_perm = False
                break
        
        if is_perm and perm_map:
            perm_list = [perm_map[i] for i in sorted(perm_map.keys())]
            labels.append(f"permutes_sections:{perm_list}")
            
        # Check for component permutation
        perm_comp_map = {}
        is_perm_c = True
        for i in comp_idx:
            img = M * E[i]
            found = False
            for j in comp_idx:
                if img == E[j]:
                    perm_comp_map[i] = j
                    found = True
                    break
            if not found:
                is_perm_c = False
                break
        
        if is_perm_c and perm_comp_map:
            # Produce a list of (original_idx, new_idx) tuples
            perm_list = sorted(perm_comp_map.items())
            labels.append(f"permutes_components:{perm_list}")

        if not labels:
            labels.append("general_isometry")
            
        classified.append((M, labels))
        
    return classified

# -------------------------
# Indefinite Lattice Automorphisms
# -------------------------
def _perm_sign_candidates(G, names):
    """
    Detects automorphisms that are permutations and sign flips of the basis vectors.
    Returns a list of Matrix(ZZ).
    """
    n = G.nrows()
    E = [vector(ZZ, [1 if i == j else 0 for i in range(n)]) for j in range(n)]
    norms = [int(E[i] * G * E[i]) for i in range(n)]
    
    norm_classes = {}
    for i, norm in enumerate(norms):
        norm_classes.setdefault(norm, []).append(i)

    mats = []
    class_items = list(norm_classes.items())
    
    # Cap permutation combinations to avoid combinatorial explosion
    total_candidates = 1
    for _, idxs in class_items:
        total_candidates *= len(idxs)
        if total_candidates > 2000:
            return []
            
    # Iterate through all combinations of permutations within norm classes
    for combo in itertools.product(*[itertools.permutations(idxs, len(idxs)) for _, idxs in class_items]):
        mapping = {orig: new for p in combo for orig, new in zip(p, p)}
        
        ncols = []
        for j in range(n):
            tgt = mapping.get(j, j)
            col = E[tgt]
            ncols.append(col)
        
        M = Matrix(ZZ, n, ncols)
        if (M.transpose() * G * M) == G:
            mats.append(M)
            
    return mats


def _backtrack_isometries(G, names, bound=1, max_solutions=50, time_limit=30):
    """
    Backtracking search for integer isometries M solving M^T * G * M = G.
    Returns a list of Matrix(ZZ).
    """
    start_time = time.time()
    n = G.nrows()
    G_target = [[int(G[i, j]) for j in range(n)] for i in range(n)]

    candidate_vectors_by_diag = {}
    vals_range = list(range(-bound, bound + 1))
    
    for diag in set(G_target[i][i] for i in range(n)):
        cand_list = set()
        # Enumerate vectors with up to two non-zero entries
        for i1 in range(n):
            for i2 in range(i1, n):
                for a in vals_range:
                    for b in vals_range:
                        if i1 == i2 and a == 0 and b == 0: continue
                        if i1 != i2 and a == 0 and b == 0: continue

                        v = [0] * n
                        v[i1] += a
                        if i1 != i2: v[i2] += b
                        
                        vec = vector(ZZ, v)
                        if int(vec * G * vec) == diag:
                            cand_list.add(tuple(v))

        candidate_vectors_by_diag[diag] = [vector(ZZ, v) for v in sorted(list(cand_list))]

    solutions = []
    cols = [None] * n

    def build_col(k):
        nonlocal solutions, start_time
        if time.time() - start_time > time_limit:
            return
        if len(solutions) >= max_solutions:
            return
        
        if k == n:
            M = Matrix(ZZ, n, cols)
            if (M.transpose() * G * M) == G:
                solutions.append(M)
            return

        diag = G_target[k][k]
        candidates = candidate_vectors_by_diag.get(diag, [])
        
        for c in candidates:
            if time.time() - start_time > time_limit: break
            
            ok = True
            for j in range(k):
                val = int(c * G * cols[j])
                if val != G_target[k][j]:
                    ok = False
                    break
            if not ok: continue
            
            cols[k] = c
            build_col(k + 1)
            if len(solutions) >= max_solutions: break

    build_col(0)
    return solutions

def compute_ns_auts_via_search(G, names, try_perm_sign=True, bound=1, max_solutions=20, time_limit=30):
    """
    Computes candidate automorphisms for an indefinite Gram matrix G.
    Returns (auts_list, names).
    """
    auts = []
    if try_perm_sign:
        perms = _perm_sign_candidates(G, names)
        auts.extend(perms)
        
    sols = _backtrack_isometries(G, names, bound=bound, max_solutions=max_solutions, time_limit=time_limit)
    auts.extend(sols)
    
    # Deduplicate and return
    seen = set()
    uniq_auts = []
    for M in auts:
        key = str(M)
        if key not in seen:
            seen.add(key)
            uniq_auts.append(M)
            
    return uniq_auts, names


def build_ns_gram(singfibs, sections):
    """
    Build NS Gram matrix and basis names from your find_singular_fibers output.
    Basis order: ['F','O','S0',...,'Comp0_0',...]
    Uses fib['symbol'] when available, else fib['m_v'] numeric fallback.
    """
    if 'fibers' not in singfibs:
        raise RuntimeError("build_ns_gram: singfibs must contain key 'fibers'")

    names = []
    names.append('F')
    names.append('O')

    # add section slots
    for i in range(len(sections)):
        names.append(f"S{i}")

    # collect component counts per fiber using robust extraction
    comp_counts = []
    for j, fib in enumerate(singfibs['fibers']):
        # prefer explicit symbol
        rank = 0
        if 'symbol' in fib and fib['symbol'] is not None:
            try:
                rank = fiber_rank_from_symbol(fib['symbol'])
            except Exception:
                rank = 0
        elif 'm_v' in fib and fib['m_v'] is not None:
            # m_v is component count (unmultiplied) per your function doc
            try:
                mv = int(fib['m_v'])
                rank = max(0, mv - 1)
            except Exception:
                rank = 0
        elif 'n' in fib and fib['n'] is not None:
            try:
                nval = int(fib['n'])
                # sometimes n stores vD_min; fallback: treat n as component-like
                rank = max(0, nval - 1)
            except Exception:
                rank = 0
        else:
            # last resort: treat deg>1 fibers as having no extra components here
            rank = 0
        comp_counts.append(rank)
        for k in range(rank):
            names.append(f"Comp{j}_{k}")

    n = len(names)
    if n == 0:
        raise RuntimeError("build_ns_gram: empty basis")

    G = Matrix(ZZ, n)
    # zero-init
    for i in range(n):
        for j in range(n):
            G[i, j] = Integer(0)

    idx = {name: i for i, name in enumerate(names)}

    # canonical intersections
    G[idx['F'], idx['F']] = Integer(0)
    G[idx['O'], idx['O']] = Integer(-2)
    G[idx['O'], idx['F']] = Integer(1)
    G[idx['F'], idx['O']] = Integer(1)

    # sections placeholders
    for i in range(len(sections)):
        sname = f"S{i}"
        si = idx[sname]
        G[si, si] = Integer(-2)
        G[si, idx['F']] = Integer(1)
        G[idx['F'], si] = Integer(1)
        # O·S left as 0; Shioda height can refine if you want

    # fiber components: diagonal -2
    for name in names:
        if name.startswith("Comp"):
            i = idx[name]
            G[i, i] = Integer(-2)
            # adjacency left as zero here

    return G, names

# --- robust mobius and candidate finder ---
def mobius_from_3points(p1, p2, p3, q1, q2, q3):
    """
    Solve for Möbius map (a*m + b)/(c*m + d) sending p_i -> q_i.
    Returns (a,b,c,d) as SR expressions. Raises ValueError if no solution.
    """
    a, b, c, d = var('a b c d')
    Sa, Sb, Sc, Sd = SR(a), SR(b), SR(c), SR(d)

    def _is_inf(v):
        try:
            return v is None or v == Infinity
        except Exception:
            return False

    triples = [(p1, q1), (p2, q2), (p3, q3)]
    eqs = []
    for p, q in triples:
        p_inf = _is_inf(p)
        q_inf = _is_inf(q)
        if p_inf and q_inf:
            # phi(infty)=infty => c == 0
            eqs.append(Sc)
        elif p_inf and not q_inf:
            eqs.append(Sa - SR(q) * Sc)
        elif (not p_inf) and q_inf:
            eqs.append(Sc * SR(p) + Sd)
        else:
            # a*p + b = q*(c*p + d)
            eqs.append(Sa * SR(p) + Sb - SR(q) * (Sc * SR(p) + Sd))

    sols = solve(eqs, a, b, c, d, solution_dict=True)
    if not sols:
        raise ValueError("mobius_from_3points: no Möbius solution for these triples")

    sol = sols[0]
    # pick a nonzero coefficient to normalize by (a,b,c,d) in that order
    for key in (a, b, c, d):
        if key in sol and SR(sol[key]) != 0:
            norm = SR(sol[key])
            A = SR(sol.get(a, 0)) / norm
            B = SR(sol.get(b, 0)) / norm
            C = SR(sol.get(c, 0)) / norm
            D = SR(sol.get(d, 0)) / norm
            return A, B, C, D

    # if all zero (shouldn't happen), raise
    raise ValueError("mobius_from_3points: solution was zero vector")

def _eval_mobius_at(mobius_tuple, center, m_sym):
    """
    Evaluate (a*m + b)/(c*m + d) at center in a projective-safe way.
    Returns SR value or Infinity.
    """
    a, b, c, d = mobius_tuple
    # handle Infinity explicitly
    if center is None or center == Infinity:
        # phi(infty) = a/c (or Infinity if c==0)
        if SR(c) == 0:
            return Infinity
        return SR(a) / SR(c)
    # normal finite evaluation
    denom = SR(c) * SR(center) + SR(d)
    if denom == 0:
        return Infinity
    return (SR(a) * SR(center) + SR(b)) / denom

def find_mobius_candidates(centers, m_sym, max_triples=500):
    """
    Try to find Möbius maps permuting the finite set `centers`.
    centers: iterable (use None or Infinity for infinity).
    """
    norm_centers = [Infinity if c is None else SR(c) for c in centers]
    n = len(norm_centers)
    if n < 3:
        return []

    triples = [(norm_centers[i], norm_centers[j], norm_centers[k])
               for i in range(n) for j in range(n) for k in range(n)
               if i != j and j != k and i != k]

    if len(triples) > max_triples:
        triples = triples[:max_triples]

    candidates = []
    seen = set()
    for s_triple in triples:
        for t_triple in triples:
            try:
                mob = mobius_from_3points(s_triple[0], s_triple[1], s_triple[2],
                                          t_triple[0], t_triple[1], t_triple[2])
            except ValueError:
                continue

            # Validate phi permutes entire center set using projective evaluation
            ok = True
            image_list = []
            for center in norm_centers:
                try:
                    img = _eval_mobius_at(mob, center, m_sym)
                except Exception:
                    ok = False
                    break
                image_list.append(img)

            if not ok:
                continue

            # check that the multiset of images equals the multiset of centers
            # exact comparison: try direct SR equality first, then fallback to string compare
            if match_sets(image_list, norm_centers):
                key = (str(mob[0]), str(mob[1]), str(mob[2]), str(mob[3]))
                if key not in seen:
                    seen.add(key)
                    candidates.append(mob)
    return candidates


# --- safer test_constant_scaling ---
def test_constant_scaling(cd, mobius_tuple, m_sym):
    """
    Test if exists u in QQ* with a4(phi(m)) == u^4 * a4(m) and a6(phi(m)) == u^6 * a6(m).
    Returns QQ(u) or None.
    """
    phi_m = _apply_mobius_on_m(m_sym, mobius_tuple)

    try:
        r4 = SR(cd.a4).subs({m_sym: phi_m}) / SR(cd.a4)
        r6 = SR(cd.a6).subs({m_sym: phi_m}) / SR(cd.a6)
    except Exception as exc:
        raise RuntimeError("test_constant_scaling: substitution failed") from exc

    # try to coerce to a QQ constant rational by checking polynomial degrees
    ok4, val4 = _is_QQ_constant_rational_in_QQm(r4, m_sym)
    ok6, val6 = _is_QQ_constant_rational_in_QQm(r6, m_sym)

    if not (ok4 and ok6):
        return None

    # basic consistency val6^2 == val4^3
    if QQ(val6)**2 != QQ(val4)**3:
        return None

    # find rational u with u^4 == val4
    num4 = Integer(val4.numerator())
    den4 = Integer(val4.denominator())

    def kth_root_integer(n, k):
        if n < 0 and k % 2 == 0:
            return None
        # integer nth root candidate (small search)
        r = int(abs(n) ** (1.0 / k))
        for cand in range(max(0, r - 3), r + 4):
            if Integer(cand) ** k == abs(n):
                return -Integer(cand) if n < 0 else Integer(cand)
        return None

    num_root = kth_root_integer(num4, 4)
    den_root = kth_root_integer(den4, 4)

    if num_root is not None and den_root is not None:
        u = QQ(num_root) / QQ(den_root)
        if QQ(u) ** 6 == QQ(val6):
            return u

    if val4 == QQ(1) and val6 == QQ(1):
        return QQ(1)

    return None


def _is_QQ_constant_rational_in_QQm(expr, m_sym, test_vals=None):
    """
    Fast check whether `expr` in QQ(m) is actually a constant in QQ.
    Uses sampling at a few rational points to avoid full coercion of large SR expressions.
    
    Returns: (True, QQ(value)) or (False, None)
    """
    from sage.rings.rational import QQ

    # Default test values
    if test_vals is None:
        test_vals = [1, 2, 3, 5, -1]

    # Quick symbolic shortcut: no m appears
    if not expr.variables():
        try:
            return True, QQ(expr)
        except (TypeError, ValueError):
            return True, QQ(1)  # fallback for symbolic 1, etc.

    # Sample evaluation at a few points
    vals = []
    for v in test_vals:
        try:
            val = QQ(expr.subs(m_sym=v))
            vals.append(val)
        except (TypeError, ValueError, ZeroDivisionError):
            return False, None

    # Check if all sampled values are equal
    first = vals[0]
    if all(val == first for val in vals):
        return True, first
    return False, None

# --- imports used by these helpers ---
import time
from sage.all import SR, Infinity, QQ, Integer, var

# --- safe projective evaluation of mobius map ---
def _eval_mobius_at(mobius_tuple, center):
    """
    Evaluate mobius (a*m + b)/(c*m + d) at `center` (SR or Infinity).
    Returns SR(Infinity) for pole.
    """
    a, b, c, d = mobius_tuple
    # handle Infinity explicitly
    if center is None or center == Infinity:
        if SR(c) == 0:
            return Infinity
        return SR(a) / SR(c)
    # finite center
    denom = SR(c) * SR(center) + SR(d)
    if denom == 0:
        return Infinity
    return (SR(a) * SR(center) + SR(b)) / denom


# --- robust approximate equality via sampling ----
def _equal_by_sampling(x, y, m_sym, test_vals=None):
    """
    Decide if symbolic expressions x and y (in m_sym) are equal by evaluating
    at several rational sample points. Returns True/False.
    This is fast and avoids heavy symbolic simplify.
    """
    if test_vals is None:
        test_vals = [1, 2, 3, 5, -1, 7]  # expand if needed

    # quick path if neither depends on m_sym
    try:
        if not x.variables() and not y.variables():
            return SR(x) == SR(y)
    except Exception:
        pass

    for v in test_vals:
        try:
            xv = SR(x).subs({m_sym: v})
            yv = SR(y).subs({m_sym: v})
            # Coerce to QQ if possible to avoid symbolic comparators
            # If substitution produced division by zero, treat as not equal
            if xv is None or yv is None:
                return False
            # Try numeric rational comparison first
            try:
                qx = QQ(xv)
                qy = QQ(yv)
                if qx != qy:
                    return False
            except (TypeError, ValueError):
                # fallback to string equality of simplified forms
                if str(SR(xv).simplify()) != str(SR(yv).simplify()):
                    return False
        except (ZeroDivisionError, TypeError, ValueError):
            return False
    return True


# --- match multisets of centers/images robustly via sampling ---
def match_sets(image_list, norm_centers, m_sym, test_vals=None):
    """
    Return True if multiset(image_list) == multiset(norm_centers) by sampling.
    This avoids heavy symbolic simplification.
    """
    if len(image_list) != len(norm_centers):
        return False

    # Make a mutable copy of centers to consume
    remaining = list(norm_centers)

    for img in image_list:
        found_idx = None
        for j, cand in enumerate(remaining):
            if _equal_by_sampling(img, cand, m_sym, test_vals=test_vals):
                found_idx = j
                break
        if found_idx is None:
            return False
        remaining.pop(found_idx)
    return True


# --- improved find_mobius_candidates with caching & timeouts ---
def find_mobius_candidates(centers, m_sym, max_triples=500, time_limit=10.0):
    """
    Attempts to find Möbius maps (a,b,c,d) that permute the finite set `centers`.
    Uses sampling and caches evaluations. Has a time limit (seconds).
    """
    norm_centers = [Infinity if c is None else SR(c) for c in centers]
    n = len(norm_centers)
    if n < 3:
        return []

    # compose triples list but cap combinatorics early
    triples = [
        (norm_centers[i], norm_centers[j], norm_centers[k])
        for i in range(n) for j in range(n) for k in range(n)
        if i != j and j != k and i != k
    ]
    if len(triples) > max_triples:
        triples = triples[:max_triples]

    candidates = []
    seen = set()
    start_time = time.time()
    # cache evaluations: cache[(mobius_key, center_str)] = value
    eval_cache = {}

    for s_triple in triples:
        for t_triple in triples:
            # global timeout guard
            if time.time() - start_time > time_limit:
                return candidates

            try:
                a, b, c, d = mobius_from_3points(
                    s_triple[0], s_triple[1], s_triple[2],
                    t_triple[0], t_triple[1], t_triple[2]
                )
            except Exception:
                continue

            mob = (a, b, c, d)
            # Evaluate images quickly using cache
            image_list = []
            ok = True
            for center in norm_centers:
                key = (str(mob[0]), str(mob[1]), str(mob[2]), str(mob[3]), str(center))
                if key in eval_cache:
                    img = eval_cache[key]
                else:
                    try:
                        img = _eval_mobius_at(mob, center)
                        eval_cache[key] = img
                    except Exception:
                        ok = False
                        break
                image_list.append(img)
            if not ok:
                continue

            # Now test multiset equality by sampling rather than symbolic simplify
            if match_sets(image_list, norm_centers, m_sym):
                key = (str(mob[0]), str(mob[1]), str(mob[2]), str(mob[3]))
                if key not in seen:
                    seen.add(key)
                    candidates.append(mob)
    return candidates

def test_constant_scaling(cd, mobius_tuple, m_sym, sample_vals=None):
    """
    Test whether exists a rational u with a4(phi(m))/a4(m) == u^4 and a6(phi(m))/a6(m) == u^6.
    Uses sampling to avoid full symbolic coercion for large expressions.
    Returns QQ(u) if found, else None.
    """
    from sage.all import QQ, Integer

    if sample_vals is None:
        sample_vals = [2, 3, 5, 7, 11]  # avoid 0 and small denominators of cd

    phi_m = _apply_mobius_on_m(m_sym, mobius_tuple)
    try:
        r4 = SR(cd.a4).subs({m_sym: phi_m}) / SR(cd.a4)
        r6 = SR(cd.a6).subs({m_sym: phi_m}) / SR(cd.a6)
    except Exception:
        # substitution failed symbolically; bail out
        return None

    # Quick sampling check: is r4 and r6 constant rational numbers?
    r4_vals = []
    r6_vals = []
    for v in sample_vals:
        try:
            v4 = QQ(SR(r4).subs({m_sym: v}))
            v6 = QQ(SR(r6).subs({m_sym: v}))
            r4_vals.append(v4)
            r6_vals.append(v6)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    if not all(val == r4_vals[0] for val in r4_vals):
        return None
    if not all(val == r6_vals[0] for val in r6_vals):
        return None

    val4 = r4_vals[0]
    val6 = r6_vals[0]

    # consistency check
    if QQ(val6)**2 != QQ(val4)**3:
        return None

    # Try to extract rational u with u^4 == val4
    num4 = Integer(val4.numerator())
    den4 = Integer(val4.denominator())

    def kth_root_integer(n, k):
        if n < 0 and k % 2 == 0:
            return None
        r = int(abs(n) ** (1.0 / k))
        for cand in range(max(0, r - 3), r + 4):
            if Integer(cand) ** k == abs(n):
                return -Integer(cand) if n < 0 else Integer(cand)
        return None

    num_root = kth_root_integer(num4, 4)
    den_root = kth_root_integer(den4, 4)

    if num_root is not None and den_root is not None:
        u = QQ(num_root) / QQ(den_root)
        if QQ(u) ** 6 == QQ(val6):
            return u

    # final fallback for trivial case
    if val4 == QQ(1) and val6 == QQ(1):
        return QQ(1)

    return None

