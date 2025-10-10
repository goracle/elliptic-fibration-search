from sage.all import (
    QQ, ZZ, SR, var, Matrix, QuadraticForm,
    vector, lcm, sqrt, Integer, EllipticCurve
)
from sage.modules.free_module_integer import IntegerLattice
from math import gcd as _py_gcd
from math import gcd, ceil
from itertools import product as iter_product
from itertools import product as _product
import multiprocessing
import os

from diagnostics2 import *
from search_common import DEBUG, build_ns_basis_and_Q


# Try to import sympy for q-series construction, but don't fail if it's not present.
try:
    import sympy as _sympy
    _HAS_SYM = True
except ImportError:
    _sympy = None
    _HAS_SYM = False


# ==============================================================================
# === Public API: Curve Enumeration and Analysis ===============================
# ==============================================================================



def decode_vector(vec, labels):
    """
    Converts a coefficient vector into a human-readable string representation.
    """
    out = []
    vec_list = vec.list() if hasattr(vec, 'list') else vec
    for i, lab in enumerate(labels):
        coeff = int(vec_list[i])
        if coeff == 0:
            continue
        
        if coeff == 1:
            out.append(lab)
        elif coeff == -1:
            out.append(f"-{lab}")
        else:
            out.append(f"{coeff}*{lab}")
            
    if not out:
        return "0"

    result = out[0]
    for term in out[1:]:
        if term.startswith('-'):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return result


def try_pari_enumeration(Q_mat, target=-2, maxvectors=5_000_000, require_S_positive=False):
    """
    If possible, use PARI to enumerate integer solutions of v^T * Q_mat * v == target.
    This is only applicable if -Q_mat is a positive-definite form.
    Returns a list of solutions or None if not applicable.
    """
    if target >= 0: return None

    Q_neg = -Q_mat
    try:
        QF = QuadraticForm(ZZ, Q_neg)
        if not QF.is_positive_definite():
            return None
    except Exception:
        return None

    target_pos = -target
    B = int(target_pos) + 1
    try:
        rep_lists = QF.representation_vector_list(B, maxvectors=int(maxvectors))
    except RuntimeError:
        print("PARI search exceeded vector limit, falling back to recursive search.")
        return None
        
    if len(rep_lists) <= target_pos:
        return []

    sols = []
    n = Q_mat.nrows()
    for tup in rep_lists[target_pos]:
        v_coords = list(map(int, tup))
        
        g = 0
        for a in v_coords:
            g = gcd(g, abs(a))
        if g != 1:
            continue

        if require_S_positive and v_coords[0] <= 0:
            continue
            
        v = Matrix(ZZ, n, 1, v_coords)
        sols.append(v)
    return sols

# ==============================================================================
# === Restored and Unchanged Helper Functions ==================================
# ==============================================================================

def detect_section_component(cd, section, sample_ms=None):
    """
    Heuristically detect which local component of each reducible fiber a section meets.
    This is an optional runtime checker.
    """
    if sample_ms is None:
        sample_ms = [QQ(2), QQ(3), QQ(5), QQ(-1), QQ(7)]
        
    try:
        m_var = cd.a4.parent().gen()
        def E_weier_eval_func(m0):
            a4 = cd.a4.subs({m_var: m0})
            a6 = cd.a6.subs({m_var: m0})
            return EllipticCurve([0, a4, 0, 0, a6])
    except Exception:
        return {}

    results = {}
    fibers = cd.singfibs.get('fibers', [])
    for f_idx, f in enumerate(fibers):
        mv = int(f.get('m_v', 1))
        if mv <= 1:
            results[f_idx] = 0
            continue
        # This part is highly dependent on how a section is represented and specialized.
        # As we cannot generalize this safely, we return a default of component 0.
        results[f_idx] = 0
    return results


def analyze_NS_Gram(Gram):
    rank_qq = Gram.rank()
    Q_form = QuadraticForm(QQ, Gram)
    try:
        sig_pos, sig_neg = Q_form.signature()
    except TypeError:
        sig_pos = sig_neg = 0
    return {
        'rank': int(rank_qq),
        'snf': Gram.smith_form(),
        'lattice': IntegerLattice(Gram),
        'signature': (sig_pos, sig_neg, int(Gram.nrows() - rank_qq))
    }

def orthogonal_complement_in_original_basis(basis_labels, Q, gen_vectors):
    n = len(basis_labels)
    if not gen_vectors:
        return [f"orth_{i}" for i in range(n)], [Matrix(ZZ, n, 1, [1 if i == j else 0 for i in range(n)]) for j in range(n)]

    Mcols = [(Q * v).change_ring(QQ) for v in gen_vectors]
    M = Matrix(QQ, Mcols).transpose()
    K = M.left_kernel()
    
    comp_vectors = []
    if K.dimension() == 0: return [], []

    for b in K.basis():
        coeffs = b.list()
        den = lcm([c.denominator() for c in coeffs])
        intvec = [int(c * den) for c in coeffs]
        g = 0
        for a in intvec: g = gcd(g, abs(a))
        if g != 0: intvec = [a // g for a in intvec]
        comp_vectors.append(Matrix(ZZ, n, 1, intvec))
    return [f"orth_{i}" for i in range(len(comp_vectors))], comp_vectors

def prepare_GW_qseries(gen_vectors, Q, h_vec, up_to_degree=10):
    class_counts = {}
    q_counts = {}
    for v in gen_vectors:
        d = int((v.transpose() * h_vec)[0, 0])
        t = tuple(v.list())
        class_counts[t] = class_counts.get(t, 0) + 1
        if 0 <= d <= up_to_degree:
            q_counts[d] = q_counts.get(d, 0) + 1
    return class_counts, q_counts

def summarize_NS_GW_res(res, show_gen=8, save_files=False, prefix="yau_res",
                        output_qseries=True, qseries_var='q', max_degree=None):
    if not isinstance(res, dict): raise TypeError("res must be a dict")
    print("===== NS / GW summary =====")
    gen_labels = res.get('gen_labels', [])
    gen_vectors = res.get('gen_vectors', [])
    print("Total generators (basis + reps):", len(gen_vectors))
    ns_analysis = res.get('ns_analysis', {})
    if ns_analysis:
        print("NS analysis rank:", ns_analysis.get('rank'))
        print("  signature (n_pos, n_neg, nullity):", ns_analysis.get('signature'))
    
    orth_vectors = res.get('orth_vectors', [])
    print("Orthogonal complement dim (ambient):", len(orth_vectors))
    
    q_counts = {int(k): int(v) for k,v in res.get('q_counts', {}).items()}
    print("q_counts (degree -> multiplicity):", q_counts)

    if output_qseries and q_counts:
        degrees = sorted(k for k in q_counts.keys() if (max_degree is None or k <= int(max_degree)))
        max_k = max(degrees) if degrees else -1
        coeff_dict = {d: q_counts.get(d, 0) for d in range(max_k + 1)}
        
        if _HAS_SYM:
            q = _sympy.symbols(qseries_var)
            expr = sum(_sympy.Integer(c) * (q ** d) for d, c in coeff_dict.items())
            qseries_obj = expr
        else:
            parts = []
            for d, c in coeff_dict.items():
                if c == 0: continue
                if d == 0: parts.append(str(c))
                elif d == 1: parts.append(f"{c}*{qseries_var}" if c != 1 else qseries_var)
                else: parts.append(f"{c}*{qseries_var}^{d}" if c != 1 else f"{qseries_var}^{d}")
            qseries_obj = " + ".join(parts).replace("+ -", "- ") if parts else "0"
        
        print("\n--- Q-series ---")
        print(str(qseries_obj))
        return (qseries_obj, coeff_dict, [coeff_dict[d] for d in range(max_k + 1)])
    return None

def compute_NS_and_GW(cd, current_sections, rho, mw_rank, chi, max_search_degree=4, q_terms=20,
                      height_bound=20, max_coord=3):
    basis_labels, Q, h_vec, gen_labels, gen_vectors, Gram = construct_NS_from_cd(
        cd, current_sections, rho, mw_rank, chi, max_search_degree=max_search_degree,
        height_bound=height_bound, max_coord=max_coord
    )
    ns_analysis = analyze_NS_Gram(Gram)
    orth_labels, orth_vectors = orthogonal_complement_in_original_basis(basis_labels, Q, gen_vectors)
    class_counts, q_counts = prepare_GW_qseries(gen_vectors, Q, h_vec, up_to_degree=q_terms)

    out = {
        'basis_labels': basis_labels, 'Q': Q, 'h_vec': h_vec,
        'gen_labels': gen_labels, 'gen_vectors': gen_vectors, 'Gram': Gram,
        'ns_analysis': ns_analysis, 'orth_labels': orth_labels, 'orth_vectors': orth_vectors,
        'class_counts': class_counts, 'q_counts': q_counts
    }
    print("NS generators:", len(gen_vectors))
    print("Gram rank:", ns_analysis['rank'], " signature:", ns_analysis['signature'])
    print("Orth complement dim:", len(orth_vectors))
    print("q_counts (up to degree):", q_counts)
    return out




# --- Replace these functions in your module ---



# Run after you have the 'reps' (return_reps=True) from staged_rational_curve_search
def sanity_check_reps(Q, h_vec, reps):
    bad = []
    for d, mats in sorted(reps.items()):
        for M in mats:
            tup = tuple(int(x) for x in M.list())
            # primitive gcd
            from math import gcd
            g = 0
            for a in tup:
                g = gcd(g, abs(a))
            if g != 1:
                bad.append(("not_primitive", d, tup, g))
            # quadratic check
            C2 = int((M.transpose() * Q * M)[0, 0])
            if C2 != -2:
                bad.append(("C2_bad", d, tup, C2))
            # degree check
            dd = int((M.transpose() * h_vec)[0, 0])
            if dd != d or dd < 0:
                bad.append(("degree_mismatch", d, dd, tup))
    if not bad:
        print("Sanity checks: OK (primitive, C2=-2, degrees nonnegative and match).")
    else:
        print("Sanity checks found problems:")
        for item in bad[:20]:
            print(" ", item)
    return bad


def process_stage_worker(args):
    """
    Top-level worker (picklable). 
    """
    max_coord, s_val, Q_flat, n, brute_max_attempts = args
    # Reconstruct matrix H locally
    H = Matrix(ZZ, n, n, Q_flat)
    coord_range = range(-max_coord, max_coord + 1)
    found = []
    attempts = 0
    
    # Process in smaller batches to prevent excessive memory buildup
    batch_count = 0
    for tpl in _product(coord_range, repeat=(n - 1)):
        if brute_max_attempts is not None:
            attempts += 1
            if attempts >= brute_max_attempts:
                break
                
        batch_count += 1
        # Exit worker early if it's processed too many items
        #if batch_count > 50000:  # Adjust this limit
        #    print(f"Worker reached batch limit, returning {len(found)} results")
        #    break
            
        full_vec = [int(s_val)] + [int(x) for x in tpl]
        # primitive check
        g = 0
        for x in full_vec:
            g = _py_gcd(g, abs(x))
        if g != 1:
            continue
        v_col = Matrix(ZZ, n, 1, full_vec)
        quad_val = int((v_col.transpose() * H * v_col)[0, 0])
        if quad_val == -2:
            found.append(tuple(full_vec))
    return found



def staged_rational_curve_search(cd, mw_sections, rho, mw_rank, chi,
                                 height_bounds=(15, 25, 35, 45, 55),
                                 max_coords=(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
                                 node_cap=15_000_000,
                                 return_reps=False,
                                 require_S_coeff='positive',
                                 S_choices=(1, -1, -2, -3, -4, -6),
                                 targeted_fallback=True,
                                 brute_max_attempts=None,
                                 num_workers=None):
    """
    Parallelized staged search. Same API as before, with optional num_workers.
    Worker tasks are independent (each (max_coord, S) pair).
    """
    if not mw_sections:
        return ({}, {}) if return_reps else {}

    basis_labels, Q, h_vec = build_ns_basis_and_Q(cd, rho, mw_rank, chi)
    if DEBUG:
        print("Q", Q)
        print("basis_labels", basis_labels)
        print("h_vec", h_vec)
    n = Q.nrows()

    # Try PARI fast path (unchanged behavior)
    # skip for now
    #pari_sols = try_pari_enumeration(Q, target=-2, maxvectors=node_cap, require_S_positive=(require_S_coeff == 'positive'))
    pari_sols = None
    if pari_sols is not None:
        print("PARI method was applicable and found", len(pari_sols), "solutions.")
        counts = {}
        reps = {}
        for v_mat in pari_sols:
            tup = tuple(int(x) for x in v_mat.list())
            canon = _canonicalize_vector_list(tup, require_S_coeff=require_S_coeff)
            if canon is None:
                continue
            v = Matrix(ZZ, n, 1, list(canon))
            d = int((v.transpose() * h_vec)[0, 0])
            if d < 0:
                continue
            counts[d] = counts.get(d, 0) + 1
            if return_reps:
                reps.setdefault(d, []).append(v)
        return (counts, reps) if return_reps else counts

    # Fallback: staged brute-force in parallel
    print("PARI method not applicable or found no solutions. Starting staged search.")

    # Prepare stage arguments (plain Python types only)
    Q_flat = list(Q.list())  # flattened entries
    stage_args = []
    for mc in max_coords:
        for s_val in S_choices:
            # still produce stage for all S; canonicalization will filter
            stage_args.append((mc, s_val, Q_flat, int(n), brute_max_attempts))

    # decide number of workers
    if num_workers is None:
        try:
            nc = os.cpu_count() or 1
        except Exception:
            nc = 1
        num_workers = min(8, nc)

    # Run pool
    all_raw_tuples = []
    if len(stage_args) == 0:
        print("No stages to run.")
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            # imap_unordered yields results as workers finish; each is a list of tuples
            for idx, stage_result in enumerate(pool.imap_unordered(process_stage_worker, stage_args, chunksize=1)):
                # extend main list
                if stage_result:
                    all_raw_tuples.extend(stage_result)

    # Deduplicate & canonicalize globally
    seen_canonical = {}
    for tup in all_raw_tuples:
        canon = _canonicalize_vector_list(tup, require_S_coeff=require_S_coeff)
        if canon is None:
            continue
        if canon not in seen_canonical:
            seen_canonical[canon] = Matrix(ZZ, n, 1, list(canon))

    # Post-process into counts/reps
    counts = {}
    reps = {}
    for canon_tup, v_mat in sorted(seen_canonical.items()):
        C2 = int((v_mat.transpose() * Q * v_mat)[0, 0])
        if C2 != -2:
            print("WARNING: canonical vector", canon_tup, "has C2=", C2, "skipping.")
            continue
        d = int((v_mat.transpose() * h_vec)[0, 0])
        if d < 0:
            continue
        counts[d] = counts.get(d, 0) + 1
        if return_reps:
            reps.setdefault(d, []).append(v_mat)

    return (counts, reps) if return_reps else counts



def run_convergence_test(cd, sections, rho, mw_rank, chi, max_coords_seq, require_S_coeff='positive'):
    runs = []
    for mc in max_coords_seq:
        print("Running up to max_coord =", mc)
        counts = staged_rational_curve_search(cd, sections, rho, mw_rank, chi,
                                             max_coords=(mc,),
                                             return_reps=False,
                                             require_S_coeff=require_S_coeff)
        runs.append((mc, counts))
    # compare successive runs
    for i in range(1, len(runs)):
        mc_prev, c_prev = runs[i-1]
        mc_now,  c_now  = runs[i]
        degrees = sorted(set(c_prev.keys()) | set(c_now.keys()))
        diffs = {}
        for d in degrees:
            a = int(c_prev.get(d, 0))
            b = int(c_now.get(d, 0))
            if a != b:
                diffs[d] = (a, b, b - a)
        print(f"\nDelta from max_coord={mc_prev} -> {mc_now}: {len(diffs)} changed degrees")
        if diffs:
            for d in sorted(diffs)[:20]:
                a,b,db = diffs[d]
                print(f" d={d}: {a} -> {b}  (Δ={db})")
    return runs


def _canonicalize_vector_list(vec, require_S_coeff='positive'):
    """
    Normalizes a vector of integers to a canonical, primitive form.

    This function is used to ensure each curve class is counted exactly once,
    regardless of how it's found (e.g., v vs. -v).

    Args:
        vec (list or tuple): A list or tuple of integers representing a vector.
        require_S_coeff (str): Controls the sign of the first coordinate.
            'positive': S-coefficient (vec[0]) must be positive.
            'any': Any sign is allowed, but the first non-zero entry
                   of the canonical vector will be positive.

    Returns:
        A canonical tuple representation of the vector, or None if the
        S-coefficient constraint is not met.
    """
    if not vec:
        return None

    # Step 1: Make primitive by dividing by GCD
    g = 0
    for x in vec:
        g = gcd(g, abs(x))
    
    # Handle the zero vector case. It's primitive by convention.
    if g == 0:
        return tuple(vec)

    primitive_vec = tuple(int(x / g) for x in vec)

    # Step 2 & 3: Handle signs based on `require_S_coeff`
    if require_S_coeff == 'positive':
        # S-coefficient is the first one (vec[0])
        if primitive_vec[0] <= 0:
            return None
        return primitive_vec
    
    elif require_S_coeff == 'any':
        # Find the first non-zero coordinate to determine sign
        first_nonzero_idx = -1
        for i, val in enumerate(primitive_vec):
            if val != 0:
                first_nonzero_idx = i
                break

        # If all zeros, it's already canonical
        if first_nonzero_idx == -1:
            return primitive_vec

        # Canonical form has the first non-zero entry be positive
        if primitive_vec[first_nonzero_idx] < 0:
            return tuple(-x for x in primitive_vec)
        else:
            return primitive_vec
    
    return None



from sage.all import QQ, Integer, Matrix
from math import isclose

def _estimate_local_correction_I_n(symbol, m_v, comp_index):
    """Compute local c_v for multiplicative I_n where possible:
       c = k*(n-k)/n with component index k (0 = identity component).
       Returns QQ(0) if not I_n or if data missing.
    """
    if not symbol: 
        return QQ(0)
    s = symbol.strip()
    if s.startswith("I") and not s.endswith("*"):
        # parse n from "I<n>"
        try:
            n = int(s[1:]) if len(s) > 1 else int(m_v)
        except Exception:
            n = int(m_v) if m_v is not None else 1
        if n <= 0:
            return QQ(0)
        k = int(comp_index) % n
        return QQ(k * (n - k)) / QQ(n)
    return QQ(0)



# --- Replacement: q-series builder that does not force divisibility by rho ---
def build_qseries_from_counts(counts, rho=None, max_degree=10):
    """
    Returns a truncated symbolic q-series sum_d n_d q^d from a counts dictionary.
    `rho` is optional and not enforced here.
    """
    q = var('q')
    series = 0
    for d, n in sorted(counts.items()):
        if rho is not None:
            assert not (n % rho), (n, rho, "rho should divide n")
        if 0 <= d <= max_degree:
            series += Integer(n) * q**Integer(d)
    return series


def compute_mw_height_and_pairings(cd, current_sections, chi):
    """
    Compute canonical heights hat_h(P) and the symmetric height-pairing matrix
    H_ij = <P_i, P_j> for the list current_sections (length = mw_rank).

    Returns:
        heights : [int]  -- canonical heights hat_h(P_i) (integers)
        Hmat    : Matrix(QQ) -- symmetric matrix of pairings (QQ entries, integral expected)
    Preconditions / Assertions:
        - current_sections supports addition: P + Q is the section sum.
        - detect_section_component(cd, section) returns per-fiber component indices.
        - Only implements local correction c_v for multiplicative I_n fibers.
        - Asserts integrality of heights (fail fast if not).
    """
    # basic checks
    mw_rank = len(list(current_sections))
    assert mw_rank >= 0

    # gather fiber list (indexable)
    if isinstance(cd, dict) and 'fibers' in cd:
        fibers = cd['fibers']
    else:
        info = find_singular_fibers(cd)
        fibers = info['fibers']

    # helper: local correction for multiplicative I_n
    def _local_correction_In(symbol, m_v, comp_idx):
        if symbol is None:
            return QQ(0)
        s = str(symbol).strip()
        if s.startswith("I") and not s.endswith("*"):
            # parse n
            if len(s) > 1:
                n = int(s[1:])
            else:
                n = int(m_v)
            k = int(comp_idx) % n
            return QQ(k * (n - k)) / QQ(n)
        return None  # signal: unsupported Kodaira for automatic correction

    # compute hat_h for a single section (via Shioda formula)
    def _hat_h_for_section(P):
        # detect which local components P meets
        comp_map = detect_section_component(cd, P)
        # P·O (intersection with zero section) -- assume zero unless caller provides different method
        # If your pipeline can compute P·O exactly, replace this line with that computation.
        P_dot_O = 0

        sum_local = QQ(0)
        for f_idx, f in enumerate(fibers):
            sym = f.get('symbol', None)
            mv = int(f.get('m_v', 1)) if f.get('m_v') is not None else 1
            comp_idx = comp_map.get(f_idx, 0)
            c = _local_correction_In(sym, mv, comp_idx)
            assert c is not None, ("Unsupported Kodaira type for automatic local correction: "
                                   f"{sym} at fiber index {f_idx}; implement c_v(P) manually.")
            sum_local += c
        hat_h = QQ(2) * QQ(chi) + QQ(2) * QQ(P_dot_O) - sum_local
        assert hat_h.denominator() == 1, f"Non-integral hat_h({P}) = {hat_h}; provide exact local data."
        return int(hat_h)

    # compute heights
    heights = []
    for P in current_sections:
        heights.append(_hat_h_for_section(P))

    # build full symmetric pairing matrix via polarization identity:
    # <P,Q> = ( hat_h(P+Q) - hat_h(P) - hat_h(Q) ) / 2
    H = Matrix(QQ, mw_rank, mw_rank)
    for i in range(mw_rank):
        H[i, i] = QQ(heights[i])
    for i in range(mw_rank):
        for j in range(i + 1, mw_rank):
            P = current_sections[i]
            Qsec = current_sections[j]
            S = P + Qsec
            hat_S = _hat_h_for_section(S)
            # pairing
            val = (QQ(hat_S) - QQ(heights[i]) - QQ(heights[j])) / QQ(2)
            assert val.denominator() == 1, ("Non-integral height pairing <P_i,P_j> = {} ;"
                                           " check data or implement exact local corrections.").format(val)
            H[i, j] = H[j, i] = QQ(int(val))
    return heights, H
