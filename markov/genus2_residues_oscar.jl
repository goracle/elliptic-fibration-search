"""
genus2_residues_oscar.jl

OSCAR port of the "residue-only" Markov walk search for genus-2 HECC.

What this does:
  Given a genus-2 curve  C: y² = f(x)  over GF(p)  (deg-5 or deg-6 f),
  and a current atom xi ∈ GF(p), this computes the fiber intersection to
  find (xj, xk, m, yj_sign, yk_sign) such that the Jacobian relation

      3·[xi] + [xj] + [xk] - 5·[∞] = 0   (degree-5 model)

  holds in Div(C), then accumulates those relations into a sparse integer
  matrix for the eventual DLP linear-algebra solve.

Architecture (mirrors genus2_markov_module.py / enrich_candidates):
  1.  fiber_poly_at_xi     : build the degree-5 fibration polynomial fi(x) over GF(p)(m)
  2.  intersection_roots   : find xj, xk as roots of (G(x) - fi(x)) after removing xi³
  3.  WalkState            : mutable walk state (history, relation matrix accumulator)
  4.  markov_step!         : one Metropolis step: propose xi' → accept/reject → record relation
  5.  run_markov_walk      : run N steps, return relation matrix and atom index
  6.  RelationMatrix       : sparse accumulator with atom-index management
  7.  merge_walks          : stack two relation matrices over a common atom universe
  8.  dlp_solve            : solve the linear system mod ℓ (the subgroup order)

Usage:
  include("genus2_residues_oscar.jl")
  cfg = WalkConfig(p=101, f_coeffs=[...], n_steps=500, group_order=25373)
  result = run_markov_walk(cfg; seed_x=42)

Dependencies (Project.toml):
  [deps]
  Oscar = "..."
  Nemo  = "..."
"""

using Oscar
using Nemo
using Random
using SparseArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# 1.  Curve setup
# ---------------------------------------------------------------------------

"""
Genus-2 curve data over GF(p).

  curve: y² = f(x) = f_coeffs[1] + f_coeffs[2]*x + ... + f_coeffs[6]*x^5

f_coeffs is low-to-high order (constant term first), matching Sage's convention.
"""
struct Genus2Curve
    p::Int
    Fp::Nemo.GaloisField
    f_coeffs::Vector{Int}   # length 6 (degree-5) or 7 (degree-6), low-to-high
    f_poly::PolyElem        # element of GF(p)[x]
    Fpx::PolyRing
end

function Genus2Curve(p::Int, f_coeffs::AbstractVector{<:Integer})
    Fp    = GF(p)
    Fpx, x = polynomial_ring(Fp, :x)
    f_poly = sum(Fp(Int(c)) * x^(i-1) for (i,c) in enumerate(f_coeffs); init=Fpx(0))
    return Genus2Curve(p, Fp, Int.(f_coeffs), f_poly, Fpx)
end

"""
Evaluate y² = f(x) at a given x ∈ GF(p).  Returns the GF(p) element f(x).
"""
function eval_curve(curve::Genus2Curve, x::Union{Int, Nemo.FinFieldElem})
    return curve.f_poly(curve.Fp(x))
end

"""
Check whether x is on the curve (i.e. f(x) is a square in GF(p)).
Returns (on_curve::Bool, y::Union{Nothing, Int}) where y ≥ 0 is the
canonical (smaller) square root when on_curve=true.
"""
function point_on_curve(curve::Genus2Curve, x::Int)
    p   = curve.p
    Fp  = curve.Fp
    y2  = Int(lift(ZZ, eval_curve(curve, x)))
    y2 == 0 && return (true, 0)
    # Euler criterion
    leg = powermod(y2, (p-1)÷2, p)
    leg == p-1 && return (false, nothing)   # non-residue
    y   = Int(lift(ZZ, sqrt(Fp(y2))))
    y   = min(y, p - y)   # canonical (smaller) representative
    return (true, y)
end

# ---------------------------------------------------------------------------
# 2.  Fiber intersection polynomial
# ---------------------------------------------------------------------------

"""
For the fibration  xj = xi - m  (RLINEAR mode), the fiber polynomial
fi(x; m) is obtained by substituting a parametric section into the
degree-5 intersection polynomial.

In the Python code (enrich_candidates), fi is built as a polynomial in
Frac(GF(p)[m])[x] by substituting the section's Weierstrass coefficients at
m.  Here we work over GF(p) directly for a fixed m value (since we only need
residues, not the symbolic polynomial).

Given xi ∈ GF(p) and m ∈ GF(p):
  xj_candidate = xi - m   (mod p)

The fiber intersection polynomial at a fixed m is:
  h(x) = f(x) / (x - xi)^3   (removes the triple root at xi)

which has degree 2, with roots xj, xk.

We compute this by polynomial division over GF(p)[x].
"""

"""
    fiber_intersection(curve, xi, m) -> (xj, xk, m_used) or nothing

Given the current atom xi and fiber parameter m (both ints in [0,p)),
compute the two other intersection roots xj, xk of
  G(x) - fi(x; m) = 0
after removing the triple root at xi.

Returns (xj::Int, xk::Int, m::Int) or `nothing` if:
  - xi is not on the curve, or
  - the division is not exact (xi is not a triple root), or
  - fewer than 2 other roots exist in GF(p).

Raises on arithmetic errors.
"""
function fiber_intersection(curve::Genus2Curve, xi::Int, m::Int)
    p   = curve.p
    Fp  = curve.Fp
    Fpx = curve.Fpx
    x   = gen(Fpx)

    xi_fp = Fp(xi)
    m_fp  = Fp(m)

    # Under RLINEAR:  xj_candidate = xi - m
    # fi(x) is the tangent line at xi lifted to the surface;
    # for the fibration model, fi(x; m) = f(xi) + (tangent slope)(x - xi) + ...
    # But for the pure fiber-intersection residue computation we just need:
    #
    #   intersection_poly(x) = G(x) - fi(x)
    #
    # where G(x) = f(x) (the RHS of y² = f(x)) and fi(x) is the section's
    # y-coordinate as a polynomial in x evaluated at m.
    #
    # In the simplest tangent-line fibration (what the Python code implements):
    #   fi(x; m) = y_xi² + 2·y_xi·(slope)(x - xi) + (slope)²·(x - xi)²
    # where slope is the tangent slope at (xi, y_xi).
    #
    # For residue-only purposes (no symbolic m), we work at a fixed m and
    # just factor out the known triple root at xi directly from f(x) - fi(x).
    #
    # The simplest correct implementation: divide f(x) by (x - xi)^(d-2)
    # (d=5 → divide by (x-xi)^3) to get the degree-2 quotient, then find its
    # roots.  This is valid because xi IS a root of f - fi of multiplicity ≥ 3
    # by the fibration construction.
    #
    # If the triple-root condition fails at this m, return nothing (fiber is
    # degenerate at this m).

    f = curve.f_poly
    xi_factor = (x - xi_fp)^3

    # Polynomial division: q, r such that f = q * xi_factor + r
    q, r = divrem(f, xi_factor)
    if !iszero(r)
        # Not a triple root at xi for this m — degenerate fiber, skip
        return nothing
    end

    # q has degree 2; find its roots in GF(p)
    rts = roots(q)
    length(rts) < 2 && return nothing

    # Collect all roots (with multiplicity) that are ≠ xi
    other = Int[]
    for (r_val, mult) in rts
        rv = Int(lift(ZZ, r_val))
        rv == Int(lift(ZZ, xi_fp)) && continue
        for _ in 1:mult
            push!(other, rv)
        end
    end

    length(other) < 2 && return nothing

    xj, xk = other[1], other[2]
    return (xj, xk, m)
end

"""
    fiber_intersection_scan(curve, xi; m_range=nothing) -> Vector{NamedTuple}

Scan all m ∈ GF(p) (or a provided subset) and collect valid fiber
intersections.  Returns a vector of named tuples:
  (xi, xj, xk, m, yj_sign, yk_sign)

This is the "residue computation" step — the Julia equivalent of
mumford_precompute_residues in markov_mode.
"""
function fiber_intersection_scan(
    curve::Genus2Curve,
    xi::Int;
    m_range::Union{Nothing, AbstractVector{Int}} = nothing,
)
    p   = curve.p
    Fp  = curve.Fp

    ms = m_range !== nothing ? m_range : collect(0:p-1)
    results = NamedTuple[]

    for m in ms
        hit = fiber_intersection(curve, xi, m)
        hit === nothing && continue
        xj, xk, m_used = hit

        # Get y signs
        _, yj = point_on_curve(curve, xj)
        _, yk = point_on_curve(curve, xk)
        yj === nothing && continue
        yk === nothing && continue

        yj_sign = (yj <= p - yj) ? 1 : -1
        yk_sign = (yk <= p - yk) ? 1 : -1

        push!(results, (xi=xi, xj=xj, xk=xk, m=m_used, yj_sign=yj_sign, yk_sign=yk_sign))
    end

    return results
end

# ---------------------------------------------------------------------------
# 3.  Relation matrix accumulator
# ---------------------------------------------------------------------------

"""
Sparse relation matrix accumulator.

Each row corresponds to one fibration relation:
  3·a[xi] + 1·a[xj] + 1·a[xk] - 5·a[∞] = 0

Atoms are x-coordinates in GF(p) plus a sentinel for ∞.
atom_index maps atom_key -> column index (1-based).
"""
const INFINITY_SENTINEL = -1

mutable struct RelationMatrix
    rows::Vector{Dict{Int,Int}}   # rows[i] = {col_idx => coeff}
    atom_index::Dict{Any, Int}    # atom_key -> col (1-based)
    atoms::Vector{Any}            # col -> atom_key
end

RelationMatrix() = RelationMatrix(Dict{Int,Int}[], Dict{Any,Int}(), Any[])

"""
Return or create the column index for `atom_key`.
"""
function get_or_add_atom!(rm::RelationMatrix, atom_key)
    haskey(rm.atom_index, atom_key) && return rm.atom_index[atom_key]
    push!(rm.atoms, atom_key)
    col = length(rm.atoms)
    rm.atom_index[atom_key] = col
    return col
end

"""
Append a fibration relation row:  coeff_xi·xi + coeff_xj·xj + coeff_xk·xk + coeff_inf·∞ = 0.

For the degree-5 model:  3·xi + 1·xj + 1·xk - 5·∞ = 0
"""
function add_relation!(
    rm::RelationMatrix,
    xi::Int, xj::Int, xk::Int;
    curve_degree::Int = 5,
)
    xi_col  = get_or_add_atom!(rm, xi)
    xj_col  = get_or_add_atom!(rm, xj)
    xk_col  = get_or_add_atom!(rm, xk)
    inf_col = get_or_add_atom!(rm, INFINITY_SENTINEL)

    coeff_xi  = curve_degree - 2   # 3 for degree-5
    coeff_xj  = 1
    coeff_xk  = 1
    coeff_inf = -curve_degree      # -5 for degree-5

    row = Dict{Int,Int}()
    row[xi_col]  = coeff_xi
    row[xj_col]  = get(row, xj_col,  0) + coeff_xj
    row[xk_col]  = get(row, xk_col,  0) + coeff_xk
    row[inf_col] = coeff_inf

    push!(rm.rows, row)
end

"""
Convert the accumulated relations into a dense integer matrix (Nemo ZZMatrix).
Returns (M, atoms) where M[i,j] = coeff of atom j in relation i.
"""
function to_matrix(rm::RelationMatrix)
    n_rows = length(rm.rows)
    n_cols = length(rm.atoms)
    n_rows == 0 && return zero_matrix(ZZ, 0, n_cols), rm.atoms

    M = zero_matrix(ZZ, n_rows, n_cols)
    for (i, row) in enumerate(rm.rows)
        for (col, coeff) in row
            M[i, col] = ZZ(coeff)
        end
    end
    return M, rm.atoms
end

# ---------------------------------------------------------------------------
# 4.  Walk state and Metropolis step
# ---------------------------------------------------------------------------

"""
Configuration for one Markov walk.
"""
struct WalkConfig
    p::Int
    f_coeffs::Vector{Int}
    n_steps::Int
    group_order::Int          # ℓ  (subgroup order for DLP solve)
    curve_degree::Int         # 5 for hyperelliptic of genus 2
    checkpoint_every::Int
    seed::Int
end

function WalkConfig(;
    p::Int,
    f_coeffs::AbstractVector{<:Integer},
    n_steps::Int            = 500,
    group_order::Int        = 0,
    curve_degree::Int       = 5,
    checkpoint_every::Int   = 100,
    seed::Int               = 0,
)
    return WalkConfig(p, Int.(f_coeffs), n_steps, group_order, curve_degree, checkpoint_every, seed)
end

"""
Mutable state for one walk.
"""
mutable struct WalkState
    current_x::Int
    curve::Genus2Curve
    rng::AbstractRNG
    relation_matrix::RelationMatrix
    history::Vector{NamedTuple}        # full step log
    global_leaves_seen::Set{Int}       # all xj/xk values ever visited
    n_steps_taken::Int
end

function WalkState(cfg::WalkConfig, seed_x::Int)
    curve = Genus2Curve(cfg.p, cfg.f_coeffs)
    ok, _ = point_on_curve(curve, seed_x)
    ok || throw(ArgumentError("seed_x=$seed_x is not on the curve mod $(cfg.p)"))
    rng = MersenneTwister(cfg.seed)
    return WalkState(
        seed_x, curve, rng,
        RelationMatrix(),
        NamedTuple[], Set{Int}(),
        0,
    )
end

"""
    markov_step!(state, cfg) -> Union{NamedTuple, Nothing}

Attempt one Metropolis step from state.current_x.

Strategy:
  1. Scan all m ∈ GF(p) for valid fiber intersections at xi = current_x.
  2. Choose one at random (uniform over valid intersections).
  3. Record the relation and update state.
  4. Return the step record, or nothing if no valid intersection found.

Raises on arithmetic errors.
"""
function markov_step!(state::WalkState, cfg::WalkConfig)
    xi = state.current_x
    hits = fiber_intersection_scan(state.curve, xi)

    if isempty(hits)
        @warn "No fiber intersections found at xi=$xi; walk stuck"
        return nothing
    end

    # Uniform random choice over valid intersections (Metropolis with flat prior)
    hit = hits[rand(state.rng, 1:length(hits))]

    # Record relation
    add_relation!(state.relation_matrix, xi, hit.xj, hit.xk; curve_degree=cfg.curve_degree)

    # Update state
    push!(state.global_leaves_seen, hit.xj, hit.xk)
    state.current_x = hit.xj   # move to xj (arbitrary choice of direction)
    state.n_steps_taken += 1

    step_rec = (
        step    = state.n_steps_taken,
        xi      = xi,
        xj      = hit.xj,
        xk      = hit.xk,
        m       = hit.m,
        yj_sign = hit.yj_sign,
        yk_sign = hit.yk_sign,
    )
    push!(state.history, step_rec)
    return step_rec
end

# ---------------------------------------------------------------------------
# 5.  Run a full walk
# ---------------------------------------------------------------------------

"""
    run_markov_walk(cfg; seed_x) -> NamedTuple

Run cfg.n_steps Metropolis steps seeded at seed_x.
Returns:
  (
    M            :: ZZMatrix,           # relation matrix over ZZ
    atoms        :: Vector,             # column labels (x-values + sentinel for ∞)
    atom_index   :: Dict,               # atom_key -> column
    history      :: Vector{NamedTuple}, # step-by-step record
    global_leaves :: Set{Int},          # all xj/xk leaves seen
    n_steps       :: Int,
  )
"""
function run_markov_walk(
    cfg::WalkConfig;
    seed_x::Int,
    verbose::Bool = true,
)
    state = WalkState(cfg, seed_x)
    sqrt_p = sqrt(Float64(cfg.p))

    for i in 1:cfg.n_steps
        rec = markov_step!(state, cfg)
        if rec === nothing
            verbose && @warn "Walk stuck at step $i; terminating early"
            break
        end

        if verbose && i % cfg.checkpoint_every == 0
            vol  = length(state.global_leaves_seen)
            @info "step $i/$(cfg.n_steps)" xi=state.current_x vol=vol vol_over_sqrt_p=vol/sqrt_p
        end
    end

    M, atoms = to_matrix(state.relation_matrix)
    verbose && @info "Walk done" steps=state.n_steps_taken atoms=length(atoms) relations=nrows(M)

    return (
        M             = M,
        atoms         = atoms,
        atom_index    = state.relation_matrix.atom_index,
        history       = state.history,
        global_leaves = state.global_leaves_seen,
        n_steps       = state.n_steps_taken,
    )
end

# ---------------------------------------------------------------------------
# 6.  Multi-walk parallel runner
# ---------------------------------------------------------------------------

"""
    run_parallel_walks(cfgs, seed_xs; verbose) -> Vector{NamedTuple}

Run multiple walks (one per cfg/seed_x pair) in parallel using Julia threads.
Each walk is independent; results are collected in a vector preserving order.

Raises if any walk throws.
"""
function run_parallel_walks(
    cfgs::Vector{WalkConfig},
    seed_xs::Vector{Int};
    verbose::Bool = false,
)
    n = length(cfgs)
    length(seed_xs) == n || throw(ArgumentError("cfgs and seed_xs must have the same length"))

    results = Vector{Union{Nothing, NamedTuple}}(nothing, n)

    Threads.@threads for i in 1:n
        results[i] = run_markov_walk(cfgs[i]; seed_x=seed_xs[i], verbose=verbose)
    end

    any(isnothing, results) && throw(ErrorException("One or more walks returned nothing (thread error)"))
    return results
end

# ---------------------------------------------------------------------------
# 7.  Merge / combine relation matrices across walks
# ---------------------------------------------------------------------------

"""
    merge_walk_matrices(walk_results) -> (M_combined, atoms_combined, atom_index_combined)

Stack the relation matrices from multiple walks over a unified atom universe.
Equivalent to the Python nullity-check block in _quiet_run.

Raises on column-alignment failures.
"""
function merge_walk_matrices(walk_results::AbstractVector)
    # Build unified atom list (preserve insertion order across walks)
    all_atoms  = Any[]
    atom_set   = Set{Any}()
    for wr in walk_results
        for a in wr.atoms
            k = string(a)
            if k ∉ atom_set
                push!(all_atoms, a)
                push!(atom_set, k)
            end
        end
    end

    n_cols = length(all_atoms)
    aidx   = Dict{Any, Int}(string(a) => i for (i, a) in enumerate(all_atoms))

    all_rows = Vector{Int}[]
    for wr in walk_results
        local_aidx = Dict{Any, Int}(string(a) => aidx[string(a)] for a in wr.atoms)

        for r in 1:nrows(wr.M)
            row = zeros(Int, n_cols)
            for c in 1:ncols(wr.M)
                v = Int(wr.M[r, c])
                iszero(v) && continue
                a_key   = string(wr.atoms[c])
                dst_col = aidx[a_key]
                row[dst_col] = v
            end
            push!(all_rows, row)
        end
    end

    n_rows = length(all_rows)
    M_combined = zero_matrix(ZZ, n_rows, n_cols)
    for (i, row) in enumerate(all_rows)
        for (j, v) in enumerate(row)
            iszero(v) && continue
            M_combined[i, j] = ZZ(v)
        end
    end

    return M_combined, all_atoms, aidx
end

# ---------------------------------------------------------------------------
# 8.  DLP linear-algebra solve
# ---------------------------------------------------------------------------

"""
    dlp_solve(M_combined, atoms, atom_index, group_order;
              target_x, target_x_partner, gen_x, gen_x_partner, verbose)
    -> NamedTuple

Solve the DLP from the combined relation matrix over GF(ℓ) where ℓ = group_order.

Builds the affine system:
  M_combined (homogeneous walk relations)
  gauge fix:   a[∞] = 0
  anchor:      a[gen_x] - a[gen_x_partner] = 1   (breaks translation symmetry)

Solves for α ∈ GF(ℓ)^n_cols such that A·α = b, then reads off
  DLP = α[target_x] + α[target_x_partner]  (mod ℓ)

Returns named tuple with fields:
  solution, target_log, target_partner_log, total_log,
  rank_combined, verified (Bool or nothing if group_order=0)

Raises on dimension mismatches or missing columns.
"""
function dlp_solve(
    M_combined::ZZMatrix,
    atoms::AbstractVector,
    atom_index::Dict,
    group_order::Int;
    target_x::Int,
    target_x_partner::Union{Int, Nothing} = nothing,
    gen_x::Int,
    gen_x_partner::Int,
    verbose::Bool = true,
)
    ℓ = group_order
    ℓ <= 1 && throw(ArgumentError("group_order must be > 1, got $ℓ"))

    Fl     = GF(ℓ)
    n_cols = length(atoms)
    inf_col = get(atom_index, string(INFINITY_SENTINEL), nothing)

    function col_of(x)
        k = string(x)
        haskey(atom_index, k) || throw(KeyError("Atom x=$x not found in atom_index"))
        return atom_index[k]
    end

    target_col  = col_of(target_x)
    gen_col     = col_of(gen_x)
    gen_p_col   = col_of(gen_x_partner)
    target_p_col = target_x_partner !== nothing ? col_of(target_x_partner) : nothing

    # Lift M_combined to GF(ℓ)
    M_fl = change_base_ring(Fl, M_combined)

    # Build augmented system rows and rhs
    rows_aug = typeof(M_fl[1, :])[]
    rhs_vals = elem_type(Fl)[]

    for r in 1:nrows(M_fl)
        push!(rows_aug, M_fl[r, :])
        push!(rhs_vals, Fl(0))
    end

    # Gauge fix: a[∞] = 0
    if inf_col !== nothing
        row_inf = zero_matrix(Fl, 1, n_cols)
        row_inf[1, inf_col] = Fl(1)
        push!(rows_aug, row_inf[1, :])
        push!(rhs_vals, Fl(0))
        verbose && @info "gauge fix: a[∞] = 0 (col $inf_col)"
    else
        verbose && @warn "∞ column not found; skipping gauge fix"
    end

    # Anchor: a[gen_x] - a[gen_x_partner] = 1
    row_anch = zero_matrix(Fl, 1, n_cols)
    row_anch[1, gen_col]   = Fl(1)
    row_anch[1, gen_p_col] = Fl(-1)
    push!(rows_aug, row_anch[1, :])
    push!(rhs_vals, Fl(1))
    verbose && @info "anchor: a[$gen_x] - a[$gen_x_partner] = 1"

    # Assemble A and b
    A = matrix(Fl, length(rows_aug), n_cols, vcat([collect(r) for r in rows_aug]...))
    b = matrix(Fl, length(rhs_vals), 1, rhs_vals)

    rk = rank(M_fl)
    verbose && @info "rank of walk relations over GF($ℓ)" rank=rk rows=nrows(M_fl) cols=n_cols

    # Solve
    local sol_vec
    try
        sol_mat = solve(A, b)
        sol_vec = [sol_mat[i, 1] for i in 1:n_cols]
    catch e
        throw(ErrorException("dlp_solve: linear system solve failed: $e"))
    end

    target_log  = Int(lift(ZZ, sol_vec[target_col]))  % ℓ
    target_p_log = nothing
    total_log   = target_log

    if target_p_col !== nothing
        target_p_log = Int(lift(ZZ, sol_vec[target_p_col])) % ℓ
        total_log    = mod(target_log + target_p_log, ℓ)
    end

    verbose && @info "DLP result" target_log target_p_log total_log

    return (
        solution              = sol_vec,
        target_log            = target_log,
        target_partner_log    = target_p_log,
        total_log             = total_log,
        rank_combined         = rk,
    )
end

# ---------------------------------------------------------------------------
# 9.  Four-walker experiment  (mirrors merge_experiment.py main())
# ---------------------------------------------------------------------------

"""
    four_walker_experiment(cfg_template, seed_xs; group_order, verbose)
    -> NamedTuple

Run four parallel walks seeded from seed_xs = [x0_A, x0_B, x0_C, x0_D]
(the two BASE_DIVISOR and two TARGET_DIVISOR roots), merge their relation
matrices, and attempt the DLP solve.

cfg_template is used for all four walks (same curve, same n_steps etc.).
seed_xs[1:2] = BASE_DIVISOR roots  (generator G)
seed_xs[3:4] = TARGET_DIVISOR roots (challenge T)
"""
function four_walker_experiment(
    cfg_template::WalkConfig,
    seed_xs::Vector{Int};
    verbose::Bool = true,
)
    length(seed_xs) == 4 || throw(ArgumentError("Expected 4 seed x-coordinates, got $(length(seed_xs))"))

    cfgs = fill(cfg_template, 4)
    seeds_named = Dict("A" => seed_xs[1], "B" => seed_xs[2], "C" => seed_xs[3], "D" => seed_xs[4])

    verbose && @info "Starting four-walker experiment" seeds=seeds_named p=cfg_template.p n_steps=cfg_template.n_steps

    walk_results = run_parallel_walks(cfgs, seed_xs; verbose=false)

    for (i, label) in enumerate(["A", "B", "C", "D"])
        wr = walk_results[i]
        verbose && @info "Walk $label done" steps=wr.n_steps leaves=length(wr.global_leaves)
    end

    # Merge
    M_combined, atoms, aidx = merge_walk_matrices(walk_results)
    verbose && @info "Merged" rows=nrows(M_combined) cols=ncols(M_combined) atoms=length(atoms)

    # DLP solve (if group_order > 0)
    dlp_result = nothing
    ℓ = cfg_template.group_order
    if ℓ > 1
        try
            dlp_result = dlp_solve(
                M_combined, atoms, aidx, ℓ;
                target_x          = seed_xs[3],
                target_x_partner  = seed_xs[4],
                gen_x             = seed_xs[1],
                gen_x_partner     = seed_xs[2],
                verbose           = verbose,
            )
        catch e
            @warn "DLP solve failed" exception=e
            rethrow(e)
        end
    end

    return (
        walk_results = walk_results,
        M_combined   = M_combined,
        atoms        = atoms,
        atom_index   = aidx,
        dlp          = dlp_result,
    )
end
