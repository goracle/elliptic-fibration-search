# --- analytic_pairings.py (drop-in replacements) ---------------------------
# Replaces compute_pairing_worker, precompute_pairings_parallel and related wiring
# Uses abel_jacobi_mumford(...) and normalize_periods_and_z(...) from your periods module.

from sage.all import (
    QQ, ComplexField, RealField, Matrix, vector
)
import math, traceback
from multiprocessing import Pool

# Import your existing AJ/integration helpers
from .periods import abel_jacobi_mumford, normalize_periods_and_z, choose_numerical_base_point
from .integration import integrate_differential_path_joint  # used implicitly by abel_jacobi_mumford
from .utilities import *  # keep import placeholder if needed elsewhere

# Module-level AJ context and cache. Call set_aj_context(...) before bulk pairing.
_AJ_CONTEXT = {
    "Omega": None,      # raw period matrix (Omega)
    "tau": None,        # normalized tau
    "Im_tau": None,     # imaginary part of tau (Matrix over CC or numeric)
    "base_point": None, # (x,y) in CC
    "f_coeffs": None,
    "prec": 300,
    "aj_cache": {},     # index -> complex vector length g (Sage vector)
}

def set_aj_context(period_matrix=None, base_point=None, f_coeffs=None, prec=300):
    """
    Populate the global Abel-Jacobi context used by pairing workers.
    - period_matrix: Omega (either g×2g or g×g normalized). If provided, tau and Im_tau are set.
    - base_point: (x0, y0) complex-like; if None, will be chosen heuristically from f_coeffs.
    - f_coeffs: polynomial coefficients (highest->lowest) needed by abel_jacobi_mumford.
    - prec: ComplexField/RealField precision to use for normalization & cache.
    """
    global _AJ_CONTEXT
    _AJ_CONTEXT["Omega"] = period_matrix
    _AJ_CONTEXT["prec"] = int(prec)
    if f_coeffs is not None:
        _AJ_CONTEXT["f_coeffs"] = tuple(f_coeffs)
    if base_point is not None:
        _AJ_CONTEXT["base_point"] = base_point
    else:
        # choose a base point if coefficients provided
        if _AJ_CONTEXT["f_coeffs"] is not None:
            _AJ_CONTEXT["base_point"] = choose_numerical_base_point(_AJ_CONTEXT["f_coeffs"], prec=prec)
    # Normalize tau if Omega provided
    if period_matrix is not None:
        try:
            tau, _ = normalize_periods_and_z(period_matrix, None)
            _AJ_CONTEXT["tau"] = tau
            # compute Im_tau as a simple numeric matrix over complex field - keep it compatible with pairing primitive
            # tau entries are Sage complex-ish; convert to a 2x2 matrix of their imag parts
            Im_tau = Matrix([[c.imag() for c in row] for row in tau])
            _AJ_CONTEXT["Im_tau"] = Im_tau
        except Exception as e:
            # leave tau/Im_tau as None; caller must handle
            _AJ_CONTEXT["tau"] = None
            _AJ_CONTEXT["Im_tau"] = None
    # clear aj_cache (indices -> normalized z vectors)
    _AJ_CONTEXT["aj_cache"] = {}

def _adaptive_guard_bits(prec_bits: int) -> int:
    # modest guard bits policy: do not explode precision
    return min(max(32, prec_bits // 8), 512)

def neron_tate_height_pairing(z1, z2, Im_tau, prec=300, normalization_factor=1.0):
    """
    Analytic (Abel-Jacobi) pairing primitive:
      <z1, z2> = normalization_factor * Re(conj(z1)^T * Im_tau^{-1} * z2)
    - z1, z2: length-g vectors of complex numbers (Sage/ComplexField-friendly)
    - Im_tau: matrix-like giving imaginary part of tau (must be invertible)
    - returns: Python float
    """
    CC = ComplexField(int(prec))
    RR = RealField(int(prec))
    # Invert Im_tau defensively
    try:
        # try to use Sage Matrix inversion if Im_tau is a Sage Matrix
        Im_tau_mat = Matrix(Im_tau)
        Im_tau_inv = Im_tau_mat.inverse()
    except Exception:
        # try numeric fallback using floats
        import numpy as np
        Im_np = [[float(x) for x in row] for row in Im_tau]
        Im_inv_np = np.linalg.inv(Im_np)
        Im_tau_inv = Matrix(CC, Im_inv_np.tolist())

    # convert to CC objects
    CC_mat = Matrix(CC, Im_tau_inv.nrows(), Im_tau_inv.ncols())
    for r in range(Im_tau_inv.nrows()):
        for c in range(Im_tau_inv.ncols()):
            CC_mat[r, c] = CC(Im_tau_inv[r, c])

    z1_vec = vector(CC, [CC(z1[i]) for i in range(len(z1))])
    z2_vec = vector(CC, [CC(z2[i]) for i in range(len(z2))])
    z1_conj = vector(CC, [z1_vec[i].conjugate() for i in range(len(z1))])

    temp = CC_mat * z2_vec
    result = sum(z1_conj[i] * temp[i] for i in range(len(z1)))
    real_part = RealField(int(prec))(result.real()) * RealField(int(prec))(normalization_factor)
    return float(real_part)

# Worker now uses AJ images and analytic pairing. Must call set_aj_context before mass runs.
def compute_pairing_worker(args):
    """
    New worker: compute analytic pairing using Abel-Jacobi images.
    Args (tuple): (i, j, div_i, div_j, f_coeffs, prec, h_i, h_j)
    Returns ((i,j), float_value_or_nan, error_msg_or_None)
    """
    try:
        i, j, div_i, div_j, f_coeffs, prec, h_i, h_j = args
    except Exception as e:
        return (None, None, f"bad_args: {e}")

    # short-circuit identical indices -> use self-height if available
    if i == j:
        try:
            return ((i, j), float(h_i), None)
        except Exception:
            return ((i, j), float('nan'), "self_height_not_castable")

    # Ensure AJ context has period matrix and base point; choose defaults if not present
    ctx = _AJ_CONTEXT
    if ctx.get("f_coeffs") is None:
        ctx["f_coeffs"] = tuple(f_coeffs)
    if ctx.get("base_point") is None:
        try:
            ctx["base_point"] = choose_numerical_base_point(ctx["f_coeffs"], prec=prec)
        except Exception:
            return ((i, j), float('nan'), "no_base_point_and_choose_failed")

    if ctx.get("Im_tau") is None:
        return ((i, j), float('nan'), "no_Im_tau_in_AJ_context")

    # compute or fetch AJ for i and j
    def _compute_and_cache_aj(idx, div_obj):
        cache = ctx["aj_cache"]
        if idx in cache:
            return cache[idx], None
        try:
            # compute raw z (no internal period reduction) at requested precision
            z_raw = abel_jacobi_mumford(div_obj, ctx["f_coeffs"], ctx["base_point"],
                                        integrate_func=integrate_differential_path_joint,
                                        prec=prec, period_matrix=None)
            # normalize using the stored Omega/tau if available
            if ctx.get("tau") is not None:
                try:
                    # normalize_periods_and_z expects Omega and z vector -> returns tau, z_norm
                    # we provide Omega = ctx["Omega"]
                    tau, z_norm = normalize_periods_and_z(ctx["Omega"], z_raw)
                    # z_norm is a matrix g×1; extract entries as complex numbers or CC elements
                    z_vec_norm = [z_norm[r, 0] for r in range(z_norm.nrows())]
                except Exception as e_norm:
                    # If normalization fails, fall back to using z_raw directly (but still proceed)
                    z_vec_norm = [z_raw[r] for r in range(len(z_raw))]
                # cache and return
                cache[idx] = z_vec_norm
                return z_vec_norm, None
            else:
                # no tau: just cache the raw vector
                z_vec = [z_raw[r] for r in range(len(z_raw))]
                cache[idx] = z_vec
                return z_vec, None
        except Exception as e:
            tb = traceback.format_exc()
            return None, tb.splitlines()[-1][:200]

    z_i, err_i = _compute_and_cache_aj(i, div_i)
    z_j, err_j = _compute_and_cache_aj(j, div_j)

    if z_i is None or z_j is None:
        # return nan + error info
        msg = "aj_failed"
        if err_i:
            msg += f"; i_err={err_i}"
        if err_j:
            msg += f"; j_err={err_j}"
        return ((i, j), float('nan'), msg)

    # compute analytic pairing using stored Im_tau
    try:
        val = neron_tate_height_pairing(z_i, z_j, ctx["Im_tau"], prec=prec, normalization_factor=1.0)
        if not math.isfinite(val):
            return ((i, j), float('nan'), "nonfinite_pairing")
        return ((i, j), float(val), None)
    except Exception as e:
        tb = traceback.format_exc()
        return ((i, j), float('nan'), f"pairing_eval_failed: {tb.splitlines()[-1][:200]}")

def precompute_pairings_parallel(indices, jac_elements, pairing_cache, f_coeffs, prec, height_cache, n_jobs=1):
    """
    Precompute pairings for the given indices using analytic AJ pairing.
    - indices: iterable of indices
    - jac_elements: sequence where jac_elements[i][0] is the Mumford divisor record
    - pairing_cache: dict (will be updated in place)
    - f_coeffs, prec, height_cache: used for compatibility & debug
    - n_jobs: number of worker processes (uses Pool)
    Returns updated pairing_cache
    """
    # ensure AJ context has f_coeffs and prec
    if _AJ_CONTEXT.get("f_coeffs") is None:
        _AJ_CONTEXT["f_coeffs"] = tuple(f_coeffs)
    if _AJ_CONTEXT.get("prec") is None:
        _AJ_CONTEXT["prec"] = int(prec)

    tasks = []
    idx_list = list(indices)
    for r in range(len(idx_list)):
        for c in range(r, len(idx_list)):
            i = idx_list[r]
            j = idx_list[c]
            if i > j:
                i, j = j, i
            if (i, j) in pairing_cache:
                continue
            div_i = jac_elements[i][0]
            div_j = jac_elements[j][0]
            h_i = height_cache.get(i, float('nan'))
            h_j = height_cache.get(j, float('nan'))
            tasks.append((i, j, div_i, div_j, f_coeffs, prec, h_i, h_j))

    if not tasks:
        return pairing_cache

    # eager precompute of AJ images for unique indices to avoid duplicated integration work
    unique_indices = set()
    for (i, j, _, _, _, _, _, _) in tasks:
        unique_indices.add(i); unique_indices.add(j)

    # compute and cache AJ images serially (safe) before multiprocessing to avoid duplicating integration work across procs
    for idx in unique_indices:
        if idx in _AJ_CONTEXT["aj_cache"]:
            continue
        div_record = jac_elements[idx][0]
        try:
            z_raw = abel_jacobi_mumford(div_record, _AJ_CONTEXT["f_coeffs"], _AJ_CONTEXT["base_point"],
                                        integrate_func=integrate_differential_path_joint,
                                        prec=_AJ_CONTEXT["prec"], period_matrix=None)
            # normalize if tau present
            if _AJ_CONTEXT.get("tau") is not None:
                try:
                    _, z_norm = normalize_periods_and_z(_AJ_CONTEXT["Omega"], z_raw)
                    z_vec_norm = [z_norm[r, 0] for r in range(z_norm.nrows())]
                except Exception:
                    z_vec_norm = [z_raw[r] for r in range(len(z_raw))]
                _AJ_CONTEXT["aj_cache"][idx] = z_vec_norm
            else:
                _AJ_CONTEXT["aj_cache"][idx] = [z_raw[r] for r in range(len(z_raw))]
        except Exception as e:
            # mark failed entries as None so worker will report nan for any pair involving it
            _AJ_CONTEXT["aj_cache"][idx] = None

    # parallel compute pairings
    results = []
    if len(tasks) > 2 and (n_jobs is None or n_jobs > 1):
        nproc = max(1, int(n_jobs))
        with Pool(processes=nproc) as pool:
            for idx, res in enumerate(pool.imap_unordered(compute_pairing_worker, tasks)):
                results.append(res)
    else:
        for args in tasks:
            results.append(compute_pairing_worker(args))

    # merge into pairing_cache
    for item in results:
        try:
            pair, val, err = item
            if pair is None:
                continue
            i, j = pair
            if err is not None:
                pairing_cache[(i, j)] = float('nan')
                pairing_cache[(j, i)] = float('nan')
            else:
                pairing_cache[(i, j)] = float(val)
                pairing_cache[(j, i)] = float(val)
        except Exception:
            continue

    return pairing_cache

# End of analytic_pairings.py

def setup_analytic_pairing_context(
    *,
    f_coeffs,
    tau=None,
    Omega=None,
    prec=300,
    base_point=None,
    verbose=False
):
    """
    One-shot setup for analytic Néron–Tate pairing.

    You MUST call this before computing any pairings.

    Inputs:
      - f_coeffs: defining polynomial coefficients
      - tau: normalized period matrix (preferred)
      - Omega: raw period matrix (if tau not provided)
      - prec: ComplexField precision
      - base_point: optional (x0, y0); chosen automatically if None
    """
    from sage.all import Matrix, ComplexField, RealField

    CC = ComplexField(prec)
    RR = RealField(prec)

    if tau is None:
        if Omega is None:
            raise ValueError("Must provide tau or Omega")
        tau, _ = normalize_periods_and_z(Omega, None)

    # Build Im(tau) robustly (use your helper)
    Im_tau = build_Im_tau_from_tau(tau, RR, CC)

    if base_point is None:
        base_point = choose_numerical_base_point(f_coeffs, prec=prec)

    set_aj_context(
        period_matrix=Omega if Omega is not None else tau,
        base_point=base_point,
        f_coeffs=f_coeffs,
        prec=prec
    )

    # Overwrite Im_tau explicitly (more robust than recomputing inside workers)
    _AJ_CONTEXT["tau"] = tau
    _AJ_CONTEXT["Im_tau"] = Im_tau

    if verbose:
        print("[AJ] Analytic pairing context initialized")
        print("  base point:", base_point)
        print("  Im(tau) eigenvalues:", Im_tau.eigenvalues())

    return tau, Im_tau


def precompute_abel_jacobi_images(
    indices,
    jac_elements,
    *,
    prec=None,
    debug=False
):
    """
    Compute and cache normalized Abel–Jacobi images z(P) for given indices.

    Uses global AJ context (must already be initialized).
    """
    ctx = _AJ_CONTEXT
    if ctx.get("Im_tau") is None:
        raise RuntimeError("AJ context not initialized")

    if prec is None:
        prec = ctx["prec"]

    for i in indices:
        if i in ctx["aj_cache"]:
            continue

        div = jac_elements[i][0]
        try:
            z_raw = abel_jacobi_mumford(
                div,
                ctx["f_coeffs"],
                ctx["base_point"],
                prec=prec,
                period_matrix=None,   # CRITICAL: no reduction here
                debug=debug
            )

            # Normalize using Omega
            if ctx.get("Omega") is not None:
                _, z_norm = normalize_periods_and_z(ctx["Omega"], z_raw)
                z_vec = [z_norm[r, 0] for r in range(z_norm.nrows())]
            else:
                z_vec = [z_raw[r] for r in range(len(z_raw))]

            ctx["aj_cache"][i] = z_vec

        except Exception as e:
            ctx["aj_cache"][i] = None
            if debug:
                print(f"[AJ] Failed for index {i}: {e}")
            raise

    return ctx["aj_cache"]


def build_analytic_gram_matrix(indices):
    """
    Build symmetric Gram matrix using analytic Néron–Tate pairing.
    """
    from sage.all import Matrix, RealField

    ctx = _AJ_CONTEXT
    RR = RealField(ctx["prec"])
    n = len(indices)

    G = Matrix(RR, n, n)

    for a in range(n):
        for b in range(a, n):
            val = get_analytic_pairing(indices[a], indices[b])
            G[a, b] = RR(val)
            G[b, a] = RR(val)

    return make_matrix_numerically_positive_definite(G)


from sage.all import Matrix

def build_Im_tau_from_tau(tau, RR, CC):
    """
    Construct Im(tau) as a real symmetric matrix over RR.

    tau: g×g complex period matrix
    RR: RealField used for numeric work
    CC: ComplexField used to store tau
    """
    tau = Matrix(CC, tau)
    g = tau.nrows()

    Im_tau = Matrix(RR, g, g)
    for i in range(g):
        for j in range(g):
            Im_tau[i, j] = RR(tau[i, j].imag())

    return Im_tau


def get_analytic_pairing(i, j):
    """
    Return <P_i, P_j> using analytic Néron–Tate pairing.
    Applies the 1/2 factor to all pairings, including off-diagonal.
    """
    ctx = _AJ_CONTEXT

    z_i = ctx["aj_cache"].get(i)
    z_j = ctx["aj_cache"].get(j)

    if z_i is None or z_j is None:
        raise RuntimeError(f"Missing AJ image for ({i},{j})")

    return neron_tate_height_pairing(z_i, z_j, ctx["Im_tau"], prec=ctx["prec"]) / 2
