#!/usr/bin/env julia
# diag_main.jl
# Entry point: argument parsing and main() orchestrator.
# Also contains stray poly/Mumford helpers that ended up at the bottom of the
# original monolith (check_mumford_invariant, poly_{normalize,add,sub,mul,divrem}_mod,
# poly_mul_mod, inv_mod, eval_poly_mod duplicate, recover_row_mumford, mumford_reduce
# duplicate).  TODO: reconcile these with diag_poly_mumford.jl and deduplicate.
#
# Depends on: diag_bootstrap.jl, diag_utils.jl, diag_poly_mumford.jl,
#             diag_io.jl, diag_linalg.jl, diag_checks.jl

include("diag_bootstrap.jl")
include("diag_utils.jl")
include("diag_poly_mumford.jl")
include("diag_io.jl")
include("diag_linalg.jl")
include("diag_checks.jl")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
using Printf

function main(args=ARGS)
    s = ArgParseSettings(description="Analyse a failed DLP solve from an HDF5 relation-matrix dump.")
    @add_arg_table! s begin
        "hdf5_path"
            help = "Path to the HDF5 file from dump_matrix_hdf5()"
            required = true
        "--group-order"
            help = "Group order l (overrides HDF5 metadata if given)"
            arg_type = Int
            default  = nothing
        "--known-key"
            help = "Known DLP answer for the log-G membership test"
            arg_type = Int
            default  = nothing
        "--col-gen0"
            arg_type = Int; default = nothing
        "--col-gen1"
            arg_type = Int; default = nothing
        "--col-tgt0"
            arg_type = Int; default = nothing
        "--col-tgt1"
            arg_type = Int; default = nothing
        "--farkas-delete"
            help   = "After extracting the certificate, delete those walk rows and re-run."
            action = :store_true
        "--exclude-xs"
            help    = "x-coordinates to exclude as torsion/bad atoms (space-separated ints)"
            arg_type = Int
            nargs    = '+'
            default  = nothing
        "--field-prime"
            help = "Field prime p (the F_p over which the curve is defined; used for Mumford reduction). Overrides HDF5 metadata."
            arg_type = Int
            default  = nothing
        "--reduce"
            help   = "Apply Mumford reduction to each relation row after loading; keep only rows whose reduced u(x) splits completely over GF(p).  Expect low yield."
            action = :store_true
    end
    parsed = parse_args(args, s)

    hdf5_path = parsed["hdf5_path"]
    isfile(hdf5_path) || error("ERROR: file not found: $hdf5_path")

    _log("\n$('#'^70)")
    _log("# DLP CONTRADICTION DIAGNOSTICS")
    _log("#  file: $hdf5_path")
    _log("$('#'^70)")

    _log("\n[load] reading HDF5 matrix ...")
    data = load_matrix_hdf5(hdf5_path)

    M           = data.M
    atoms       = data.atoms
    aidx        = data.aidx
    group_order = something(parsed["group-order"], data.group_order)
    field_prime = something(parsed["field-prime"],  data.field_prime, nothing)
    divisor_xs  = data.divisor_xs

    col_inf  = data.col_inf
    col_gen0 = coalesce(parsed["col-gen0"], data.col_gen0)
    col_gen1 = coalesce(parsed["col-gen1"], data.col_gen1)
    col_tgt0 = coalesce(parsed["col-tgt0"], data.col_tgt0)
    col_tgt1 = coalesce(parsed["col-tgt1"], data.col_tgt1)

    group_order === nothing && error("ERROR: group_order not found in HDF5 and not supplied via --group-order")

    _log("[load] matrix shape : $(size(M,1)) × $(size(M,2))")
    matrix_preview(M, atoms; max_rows=4, max_atoms=6)
    _log("[load] group_order  : $group_order")
    _log("[load] divisor_xs   : $divisor_xs")
    _log("[load] col_inf=$col_inf  col_gen0=$col_gen0  col_gen1=$col_gen1  col_tgt0=$col_tgt0  col_tgt1=$col_tgt1")

    # --- Recover special columns from divisor_xs when metadata omits them ---
    curve_coeffs_for_infer = load_curve_coeffs(hdf5_path)  # never nothing; falls back to DEFAULT
    inferred_specials = infer_special_cols_from_divisor_xs(aidx, divisor_xs;
                                                            curve_coeffs=curve_coeffs_for_infer,
                                                            field_prime=field_prime)
    if divisor_xs !== nothing
        updates = String[]
        if col_gen0 === nothing && haskey(inferred_specials, "gen0") && inferred_specials["gen0"] !== nothing
            col_gen0 = inferred_specials["gen0"]
            push!(updates, "gen0=$(col_gen0) from divisor_xs[1]=$(divisor_xs[1])")
        end
        if col_gen1 === nothing && haskey(inferred_specials, "gen1") && inferred_specials["gen1"] !== nothing
            col_gen1 = inferred_specials["gen1"]
            push!(updates, "gen1=$(col_gen1) from divisor_xs[2]=$(divisor_xs[2])")
        end
        if col_tgt0 === nothing && haskey(inferred_specials, "tgt0") && inferred_specials["tgt0"] !== nothing
            col_tgt0 = inferred_specials["tgt0"]
            push!(updates, "tgt0=$(col_tgt0) from divisor_xs[3]=$(divisor_xs[3])")
        end
        if col_tgt1 === nothing && haskey(inferred_specials, "tgt1") && inferred_specials["tgt1"] !== nothing
            col_tgt1 = inferred_specials["tgt1"]
            push!(updates, "tgt1=$(col_tgt1) from divisor_xs[4]=$(divisor_xs[4])")
        end
        if !isempty(updates)
            _log("[load] inferred special cols: " * join(updates, "; "))
        end
    end

    # --- Exclude torsion/bad atoms ---
    if parsed["exclude-xs"] !== nothing
        protected_cols = Set(c for c in [col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1] if c !== nothing)
        excluded_cols  = Set{Int}()
        exclude_x_set  = Set(parsed["exclude-xs"])
        for (j, atm) in enumerate(atoms)
            s = string(atm)
            # Extract x-component from "(x, y)" or legacy bare-x.
            m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", s)
            x_val = m !== nothing ? parse(Int, m.captures[1]) : tryparse(Int, s)
            x_val === nothing && continue
            x_val ∉ exclude_x_set && continue
            c = j
            if c ∈ protected_cols
                _log("[filter] --exclude-xs $x_val: atom $s is a protected col ($c), skipping")
            else
                push!(excluded_cols, c)
            end
        end
        # Also handle any x not matched as an atom (warn user)
        for x in parsed["exclude-xs"]
            found = any(begin
                s = string(atm)
                m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", s)
                xv = m !== nothing ? parse(Int, m.captures[1]) : tryparse(Int, s)
                xv !== nothing && xv == x
            end for atm in atoms)
            found || _log("[filter] --exclude-xs $x: no atom with this x-coordinate found, skipping")
        end
        if !isempty(excluded_cols)
            excluded_atoms = [atoms[c] for c in sort(collect(excluded_cols))]
            _log("[filter] excluding $(length(excluded_cols)) atom(s): $excluded_atoms")
            n_before = size(M, 1)
            keep_rows = [r for r in 1:size(M,1) if !any(M[r,c] != 0 for c in excluded_cols)]
            M = M[keep_rows, :]
            n_dropped_rows = n_before - size(M,1)
            n_cols_before  = size(M, 2)
            keep_cols = [c for c in 1:n_cols_before if c ∉ excluded_cols]
            col_remap = Dict(old => new for (new, old) in enumerate(keep_cols))
            M = M[:, keep_cols]
            atoms = atoms[keep_cols]
            aidx  = Dict(string(atoms[i]) => i for i in 1:length(atoms))
            remap_col_local(c) = c === nothing ? nothing : get(col_remap, c, nothing)
            col_inf  = remap_col_local(col_inf)
            col_gen0 = remap_col_local(col_gen0)
            col_gen1 = remap_col_local(col_gen1)
            col_tgt0 = remap_col_local(col_tgt0)
            col_tgt1 = remap_col_local(col_tgt1)
            _log("[filter] dropped $n_dropped_rows row(s) touching excluded atoms, $(length(excluded_cols)) col(s) removed")
            _log("[filter] matrix after exclusion: $(size(M,1)) × $(size(M,2))")
            _log("[filter] remapped cols — inf=$col_inf  gen0=$col_gen0  gen1=$col_gen1  tgt0=$col_tgt0  tgt1=$col_tgt1")
        end
    end

    known_key = parsed["known-key"]
    known_key === nothing && _log("[load] --known-key not supplied; log-G membership test will be skipped.")

    # --- Degree-balance filter ---
    # Every valid relation satisfies sum(row) == 0 mod group_order.
    # The ∞ coefficient is stored as its positive representative mod group_order
    # (e.g. -5 mod 25373 = 25368), so the sum over plain ZZ will be ~group_order,
    # not 0.  We must check mod group_order, not over ZZ.
    n_cols     = size(M, 2)
    keep_rows  = Int[]
    malformed  = Int[]
    for r in 1:size(M, 1)
        s = mod(sum(Int(M[r, c]) for c in 1:n_cols), group_order)
        s == 0 ? push!(keep_rows, r) : push!(malformed, r)
    end

    if !isempty(malformed)
        preview = malformed[1:min(16,end)]
        _log("[filter] dropping $(length(malformed)) degree-imbalanced row(s) (coeff sum ≠ 0): $preview" *
             (length(malformed) > 16 ? " ..." : ""))
        isempty(keep_rows) && error("All rows are degree-imbalanced — the log is corrupt or from an old encoder.")
        M = reduce(vcat, [reshape(M[r, :], 1, :) for r in keep_rows])
        _log("[filter] matrix after imbalanced-row drop: $(size(M,1)) × $(size(M,2))")
    else
        _log("[filter] all rows degree-balanced (coeff sums all zero).")
    end

    # --- Mumford reduction filter (--reduce) ---
    if parsed["reduce"]
        field_prime === nothing && error("--reduce requires --field-prime <p> (or field_prime stored in HDF5)")
        curve_coeffs = load_curve_coeffs(hdf5_path)
        M, _, _ = apply_mumford_reduce_filter(M, atoms, col_inf, curve_coeffs, field_prime)
        size(M, 1) == 0 && error("--reduce: no rows survived Mumford filter; cannot continue.")
    end

    # --- Check 1 ---
    if known_key !== nothing
        check_homogeneous(M, atoms, aidx, group_order,
                          known_key, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf)
    else
        _section("CHECK 1: HOMOGENEOUS SYSTEM  (skipped — no --known-key)")
        nr_hom, nc_hom = size(M)
        if nr_hom * nc_hom <= 200_000_000
            Fp      = GF(group_order)
            A_hom   = to_nemo_mat(M, Fp)
            rank_hom = rank(A_hom)
            null_hom = nc_hom - rank_hom
        else
            ker_hom  = right_kernel_basis(M, group_order; expected_nullity=max(2, nc_hom - nr_hom + 4))
            null_hom = length(ker_hom)
            rank_hom = nc_hom - null_hom
        end
        _log("  rows=$nr_hom  cols=$nc_hom  rank=$rank_hom  nullity=$null_hom")
    end

    # --- Check 2 ---
    cert_entries, farkas_walk_rows = extract_contradiction_certificate(
        M, atoms, group_order;
        col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1)

    GC.gc()

    # --- Seed suggestion ---
    suggested = suggest_seeds_from_noparity_nullity(
        M, atoms, group_order,
        col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1; n_seeds=4)
    _log("\n$('#'^70)")
    _log("# SUGGESTED SEEDS (non-parity nullity atoms, most frequent first)")
    _log("$('#'^70)")
    if !isempty(suggested)
        _log(join(string.(suggested), " "))
    else
        _log("(no non-parity kernel directions found; no seeds suggested)")
    end

    # --- Check 3 ---
    check_structural_collapse(M, atoms, group_order;
                               col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1,
                               col_tgt0=col_tgt0, col_tgt1=col_tgt1)

    GC.gc()

    # --- Check 4 ---
    incremental_consistency_filter(M, atoms, group_order;
                                   col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1,
                                   col_tgt0=col_tgt0, col_tgt1=col_tgt1)

    GC.gc()

    # --- Farkas-delete re-run ---
    if parsed["farkas-delete"] && !isempty(farkas_walk_rows)
        farkas_delete_rerun(M, atoms, aidx, group_order, farkas_walk_rows,
                            col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1;
                            known_key=known_key)
    elseif parsed["farkas-delete"]
        _log("\n[farkas-delete] No walk rows in certificate — nothing to delete.")
    end

    _log("\n$('#'^70)")
    _log("# DIAGNOSTICS COMPLETE")
    _log("$('#'^70)\n")
end

function check_mumford_invariant(u::Vector{Int}, v::Vector{Int}, f::Vector{Int}, p::Int)
    # Compute v^2 mod p
    v2 = poly_mul_mod(v, v, p)

    # Compute f - v^2 mod p
    diff = poly_sub_mod(f, v2, p)

    # Compute remainder of diff mod u
    _, r = poly_divrem_mod(diff, u, p)

    # Normalize remainder
    r = poly_normalize_mod(r, p)

    return isempty(r) || all(x -> x % p == 0, r)
end


# normalize: remove trailing zeros
function poly_normalize_mod(a::Vector{Int}, p::Int)
    a = [mod(x, p) for x in a]
    while !isempty(a) && a[end] == 0
        pop!(a)
    end
    return a
end

# addition
function poly_add_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    n = max(length(a), length(b))
    c = zeros(Int, n)
    for i in 1:n
        ai = i <= length(a) ? a[i] : 0
        bi = i <= length(b) ? b[i] : 0
        c[i] = mod(ai + bi, p)
    end
    return poly_normalize_mod(c, p)
end

# subtraction
function poly_sub_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    n = max(length(a), length(b))
    c = zeros(Int, n)
    for i in 1:n
        ai = i <= length(a) ? a[i] : 0
        bi = i <= length(b) ? b[i] : 0
        c[i] = mod(ai - bi, p)
    end
    return poly_normalize_mod(c, p)
end

# multiplication
function poly_mul_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    if isempty(a) || isempty(b)
        return Int[]
    end
    c = zeros(Int, length(a) + length(b) - 1)
    for i in 1:length(a)
        for j in 1:length(b)
            c[i+j-1] = mod(c[i+j-1] + a[i]*b[j], p)
        end
    end
    return poly_normalize_mod(c, p)
end

# scalar inverse
inv_mod(a, p) = powermod(a, p-2, p)

# division with remainder
function poly_divrem_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    a = poly_normalize_mod(copy(a), p)
    b = poly_normalize_mod(copy(b), p)

    isempty(b) && error("division by zero polynomial")

    da = length(a) - 1
    db = length(b) - 1

    if da < db
        return Int[], a
    end

    q = zeros(Int, da - db + 1)
    r = copy(a)

    inv_lead = inv_mod(b[end], p)

    while length(r) >= length(b) && !isempty(r)
        d = length(r) - length(b)
        coeff = mod(r[end] * inv_lead, p)
        q[d+1] = coeff

        # subtract coeff * x^d * b
        for i in 1:length(b)
            r[d+i] = mod(r[d+i] - coeff*b[i], p)
        end

        r = poly_normalize_mod(r, p)
    end

    return poly_normalize_mod(q, p), poly_normalize_mod(r, p)
end

function eval_poly_mod(coeffs, x, p)
    acc = 0
    xp = 1
    for c in coeffs
        acc = mod(acc + c * xp, p)
        xp = mod(xp * x, p)
    end
    return acc
end


"""
Recover a valid Mumford divisor from a single relation row using the stored
(x, y) atom coordinates.

Args:
  row        :: Vector{Int}     # row of matrix (coefficients)
  atoms      :: Vector{String}  # column labels — "(x, y)" strings, "∞" allowed
  col_inf    :: Int             # index of ∞ column
  f_coeffs   :: Vector{Int}     # curve f(x) coeffs (ascending)
  p          :: Int             # field prime

Returns:
  (u, v) if a valid Mumford divisor is found, else nothing
"""
function recover_row_mumford(row, atoms, col_inf, f_coeffs, p)
    # --- extract affine (x, y) pairs with multiplicity ---
    xys = Tuple{Int,Int}[]
    for j in eachindex(row)
        c = row[j]
        if j == col_inf || c == 0
            continue
        end
        s = string(atoms[j])
        m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", s)
        if m !== nothing
            x_val = parse(Int, m.captures[1])
            y_val = parse(Int, m.captures[2])
        else
            # Legacy bare-x atom — compute canonical y branch
            x_val = parse(Int, s)
            fx = eval_poly_mod(f_coeffs, x_val, p)
            y_raw = tonelli_shanks(fx, p)
            y_raw === nothing && return nothing
            y_val = min(y_raw, mod(-y_raw, p))
        end
        # Negative coefficient means involution (negate y).
        actual_y = c > 0 ? y_val : mod(-y_val, p)
        for _ in 1:abs(c)
            push!(xys, (x_val, actual_y))
        end
    end

    isempty(xys) && return nothing

    # --- validate each point lies on the curve ---
    for (x_val, y_val) in xys
        fx = eval_poly_mod(f_coeffs, x_val, p)
        mod(y_val * y_val - fx, p) != 0 && return nothing
        y_val == 0 && return nothing
    end

    # --- build divisor using stored y-coordinates directly ---
    try
        u = [1]
        v = [0]
        for (x_val, y_val) in xys
            u, v = mumford_add_point(u, v, x_val, y_val, f_coeffs, p)
        end
        u, v = mumford_reduce(u, v, f_coeffs, p)
        check_mumford_invariant(u, v, f_coeffs, p)
        return (u, v)
    catch
        return nothing
    end
end

function mumford_reduce(u::Vector{Int}, v::Vector{Int},
                        f_coeffs::Vector{Int}, p::Int)

    g = 2  # genus

    # --- helpers ---
    strip(w) = begin
        z = copy(w)
        while length(z) > 1 && z[end] == 0
            pop!(z)
        end
        z
    end

    make_monic(u) = begin
        u = strip(u)
        if isempty(u)
            return u
        end
        lc = u[end]
        if lc != 1
            inv_lc = invmod(lc, p)
            u = mod.(u .* inv_lc, p)
        end
        u
    end

    u = make_monic(u)
    v = strip(v)

    # --- main reduction loop ---
    while length(u) - 1 > g   # deg(u) > g

        # compute f - v^2
        v2   = poly_mul_mod(v, v, p)
        fmv2 = poly_sub_mod(f_coeffs, v2, p)

        # divide (f - v^2) by u
        u2, rem = poly_divrem_mod(fmv2, u, p)

        if any(x != 0 for x in rem)
            throw(ErrorException(
                "mumford_reduce: (f - v^2)/u not exact; invalid divisor"
            ))
        end

        u2 = make_monic(u2)

        # v' = -v mod u2
        negv = mod.(-v, p)
        _, v2r = poly_divrem_mod(negv, u2, p)

        u = u2
        v = strip(v2r)

        # --- invariant check (new, critical) ---
        check_mumford_invariant(u, v, f_coeffs, p)
    end

    # final normalization
    u = make_monic(u)
    v = strip(v)

    # final invariant check
    check_mumford_invariant(u, v, f_coeffs, p)

    return u, v
end

main()
