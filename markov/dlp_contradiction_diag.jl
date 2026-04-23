#!/usr/bin/env julia
"""
dlp_contradiction_diag.jl

Post-mortem diagnostics for a failed DLP solve, operating directly on the
HDF5 relation-matrix dump produced by dlp_diagnostics.dump_matrix_hdf5().

Four core checks:
  1. HOMOGENEOUS CHECK – does the known log-G vector lie in ker(A_hom)?
  2. CONTRADICTION CERTIFICATE – Farkas left-kernel witness for inconsistency.
  3. STRUCTURAL COLLAPSE TRIAGE – column fusion, special-col order, rank stability.
  4. INCREMENTAL CONSISTENCY FILTER – step-order Gaussian elimination.

Usage
-----
    julia dlp_contradiction_diag.jl relation_matrix.h5 \\
          --group-order 25373 --known-key 802

    # or, if group_order / known_key are stored in the HDF5 metadata:
    julia dlp_contradiction_diag.jl relation_matrix.h5
"""

import Pkg

# ---------------------------------------------------------------------------
# Dependency bootstrap — install missing packages silently on first run.
# ---------------------------------------------------------------------------
const REQUIRED_PKGS = ["HDF5", "JSON3", "ArgParse", "Nemo", "SparseArrays",
                       "LinearAlgebra"]

for pkg in REQUIRED_PKGS
    if !haskey(Pkg.project().dependencies, pkg) &&
       !any(p.name == pkg for p in values(Pkg.dependencies()))
        @info "Installing $pkg …"
        Pkg.add(pkg)
    end
end

using HDF5
using JSON3
using ArgParse
using Nemo          # GF, matrix, kernel, rank
using SparseArrays
using LinearAlgebra
using Random: seed!, randperm
using Printf  # Add this line
# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
const SEP  = "=" ^ 70
const THIN = "-" ^ 70

_log(msg) = (println(msg); flush(stdout))
_section(title) = (_log("\n$SEP"); _log("  $title"); _log(SEP))

function brief_atom_list(atom_rows; max_items=6)
    isempty(atom_rows) && return "[]"
    items = atom_rows[1:min(max_items, end)]
    s = join(["$(a)=$(c)" for (a, c) in items], ", ")
    length(atom_rows) > max_items && (s *= ", ...")
    return "[$s]"
end

function matrix_preview(M, atoms; max_rows=6, max_atoms=6)
    nr, nc = size(M)
    _log("[matrix] shape=$(nr)×$(nc)  atoms=$(length(atoms))")
    for i in 1:min(max_rows, nr)
        row_atoms = [(string(atoms[j]), M[i,j]) for j in 1:nc if M[i,j] != 0]
        _log("[matrix] row $(lpad(i,5)): $(brief_atom_list(row_atoms; max_items=max_atoms))")
    end
    nr > max_rows && _log("[matrix] ... $(nr - max_rows) more row(s)")
end

# ---------------------------------------------------------------------------
# Row deduplication over GF(modulus)
# ---------------------------------------------------------------------------
"""
Collapse exact duplicate rows and scalar multiples over GF(modulus).
Returns (M_dedup::Matrix{Int}, row_sources::Vector{Vector{Int}}).
"""
function dedupe_rows_mod(M::Matrix{Int}, modulus::Int; keep_zero_rows=false)
    modulus === nothing && throw(ArgumentError("modulus is required"))
    nr, nc = size(M)
    seen       = Dict{Vector{Tuple{Int,Int}}, Int}()
    dedup_rows = Vector{Vector{Int}}()
    row_sources = Vector{Vector{Int}}()

    for i in 1:nr
        entries = Tuple{Int,Int}[]
        for j in 1:nc
            v = M[i,j] % modulus
            v < 0 && (v += modulus)
            v != 0 && push!(entries, (j, v))
        end

        if isempty(entries)
            keep_zero_rows || continue
            sig = Tuple{Int,Int}[]
            if !haskey(seen, sig)
                seen[sig] = length(dedup_rows) + 1
                push!(dedup_rows, zeros(Int, nc))
                push!(row_sources, [i])
            else
                push!(row_sources[seen[sig]], i)
            end
            continue
        end

        lead = entries[1][2]
        inv_lead = invmod(lead, modulus)
        sig = [(j, (v * inv_lead) % modulus) for (j, v) in entries]

        if !haskey(seen, sig)
            seen[sig] = length(dedup_rows) + 1
            row = zeros(Int, nc)
            for (j, v) in sig
                row[j] = v
            end
            push!(dedup_rows, row)
            push!(row_sources, [i])
        else
            push!(row_sources[seen[sig]], i)
        end
    end

    if isempty(dedup_rows)
        return Matrix{Int}(undef, 0, nc), row_sources
    end
    M_dedup = reduce(vcat, [reshape(r, 1, :) for r in dedup_rows])
    return M_dedup, row_sources
end

# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------
"""
Load the pruned relation matrix and metadata from an HDF5 dump.
Returns a NamedTuple with fields: M, atoms, aidx, group_order, divisor_xs,
col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1.
"""
function load_matrix_hdf5(path::String)
    isfile(path) || throw(ErrorException("file not found: $path"))

    return h5open(path, "r") do f
        # --- 1. Load Atoms and Index ---
        atoms_raw = read(f["atoms"])
        atoms = isa(atoms_raw[1], AbstractString) ? collect(atoms_raw) :
                [String(a) for a in atoms_raw]
        
        aidx_raw = read(f["atom_index"])
        aidx_str = isa(aidx_raw, AbstractString) ? aidx_raw : String(aidx_raw)
        aidx = Dict{String,Int}(string(k) => v for (k,v) in JSON3.read(aidx_str))

        # --- 2. Load Matrix ---
        # Initialize M in the do-block scope so it's guaranteed to be defined
        M = if haskey(f, "matrix_dense")
            Int.(read(f["matrix_dense"]))
        else
            data_vals = Int.(read(f["csr/data"]))
            indices   = Int.(read(f["csr/indices"])) .+ 1
            indptr    = Int.(read(f["csr/indptr"]))
            shape     = Tuple(Int.(read(f["csr/shape"])))
            nr, nc    = shape
            
            # Create the dense matrix directly from CSR data
            temp_M = zeros(Int, nr, nc)
            for r in 1:nr
                for idx in (indptr[r]+1):indptr[r+1]
                    temp_M[r, indices[idx]] = data_vals[idx]
                end
            end
            temp_M # Return this as the value of the 'if' block
        end

        # --- 3. Load Metadata ---
        group_order = haskey(f, "group_order") ? Int(read(f["group_order"])) : nothing
        divisor_xs  = haskey(f, "divisor_xs")  ? Int.(read(f["divisor_xs"])) : nothing

        function _col(key)
            !haskey(f, key) && return nothing
            v = Int(read(f[key]))
            return v >= 0 ? v + 1 : nothing
        end

        # Return the NamedTuple directly
        (
            M = M,
            atoms = atoms,
            aidx = aidx,
            group_order = group_order,
            divisor_xs = divisor_xs,
            col_inf = _col("col_inf"),
            col_gen0 = _col("col_gen0"),
            col_gen1 = _col("col_gen1"),
            col_tgt0 = _col("col_tgt0"),
            col_tgt1 = _col("col_tgt1")
        )
    end
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function drop_rows(M::Matrix{Int}, rows_to_drop::AbstractSet{Int})
    keep = [i for i in 1:size(M,1) if i ∉ rows_to_drop]
    return M[keep, :]
end

function remap_col(col, old_atoms, pruned_aidx)
    col === nothing && return nothing
    if !(1 <= col <= length(old_atoms))
        return nothing
    end
    key = string(old_atoms[col])
    return get(pruned_aidx, key, nothing)
end

function infer_special_cols_from_divisor_xs(aidx, divisor_xs)
    divisor_xs === nothing && return Dict{String,Union{Nothing,Int}}()
    labels = ["gen0", "gen1", "tgt0", "tgt1"]
    inferred = Dict{String,Union{Nothing,Int}}()
    for (lab, x) in zip(labels, divisor_xs)
        inferred[lab] = get(aidx, string(x), nothing)
    end
    return inferred
end

# ---------------------------------------------------------------------------
# Nemo GF matrix helpers
# ---------------------------------------------------------------------------
"""
Convert a plain Int matrix to a Nemo matrix over GF(p).
"""
function to_nemo_mat(M::Matrix{Int}, Fp)
    nr, nc = size(M)
    rows = [[Fp(M[i,j]) for j in 1:nc] for i in 1:nr]
    return matrix(Fp, nr, nc, reduce(vcat, [r for r in rows]))
end

"""
Compute right kernel basis of A over GF(p).
Returns a Vector of coefficient vectors (each a Vector{Int} mod p).
"""
function right_kernel_basis(A_nemo, p::Int)
    ker_mat = kernel(A_nemo; side=:right)

    # Optional sanity check
    @assert ker_mat isa Nemo.MatElem

    nr_ker, nc_ker = nrows(ker_mat), ncols(ker_mat)

    basis = Vector{Vector{Int}}()
    for j in 1:nc_ker
        vec = [Int(lift(ZZ, ker_mat[i, j])) for i in 1:nr_ker]
        push!(basis, vec)
    end

    return basis
end
"""
Compute left kernel basis of A over GF(p) (= right kernel of A^T).
"""
function left_kernel_basis(A_nemo, p::Int)
    AT = transpose(A_nemo)
    return right_kernel_basis(AT, p)
end

# ---------------------------------------------------------------------------
# Check 1: Homogeneous system + log-G membership
# ---------------------------------------------------------------------------
function check_homogeneous(M::Matrix{Int}, atoms, aidx, group_order::Int,
                            known_key::Int, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf)
    _section("CHECK 1: HOMOGENEOUS SYSTEM  (walk relations, no anchor)")

    n  = group_order
    Fp = GF(n)

    M_pruned, row_sources = dedupe_rows_mod(M, group_order)
    pruned_aidx = Dict(string(atoms[i]) => i for i in 1:length(atoms))
    nr, nc = size(M_pruned)

    n_dedup = size(M,1) - nr
    _log("  special-cols   : kept as loaded; only row dedup applied ($(size(M,1)) rows × $(size(M,2)) cols)")
    _log("  row dedup      : removed $n_dedup duplicate/scalar-multiple row(s)")

    p_col_gen0 = remap_col(col_gen0, atoms, pruned_aidx)
    p_col_gen1 = remap_col(col_gen1, atoms, pruned_aidx)
    p_col_tgt0 = remap_col(col_tgt0, atoms, pruned_aidx)
    p_col_tgt1 = remap_col(col_tgt1, atoms, pruned_aidx)
    p_col_inf  = remap_col(col_inf,  atoms, pruned_aidx)

    for (name, col) in [("gen0", p_col_gen0), ("gen1", p_col_gen1),
                        ("tgt0", p_col_tgt0), ("tgt1", p_col_tgt1)]
        col === nothing && _log("  ⚠  $name unresolved — no column mapping available.")
    end

    A_nemo   = to_nemo_mat(M_pruned, Fp)
    ker_bas  = right_kernel_basis(A_nemo, n)
    null_hom = length(ker_bas)
    rank_hom = nc - null_hom
    _log("\n  Pre-normalization nullity: $null_hom on the $(nr)×$(nc) system")
    extract_pin_rows(ker_bas, atoms, nc, n, p_col_inf; pin_isolated=false, max_preview=8)
    _log("  Isolated atoms are reported as free directions, not pinned.")
    _log("  rows=$nr  cols=$nc  rank=$rank_hom  nullity=$null_hom")
    _log("  (ideal: nullity >= 2 — gauge direction + DLP direction)")

    if null_hom == 0
        _log("\n  ✗  Nullity=0 — walk relations alone are already inconsistent over GF(l).")
        _log("     The contradiction is in the relation rows, not the anchor.")
        _log("     → Proceed to Check 2 for root cause.")
    elseif null_hom == 1
        _log("\n  ⚠  Nullity=1 — gauge and DLP directions are fused or one is missing.")
    else
        _log("\n  ✓  Nullity=$null_hom — at least gauge + DLP directions present.")
    end

    # --- inspect kernel basis vectors ---
    _log("\n  Surviving kernel basis ($(length(ker_bas)) vector(s)):")
    special_cols = Dict(nm => c for (nm, c) in [("gen0",p_col_gen0),("gen1",p_col_gen1),
                                                  ("tgt0",p_col_tgt0),("tgt1",p_col_tgt1),
                                                  ("inf",p_col_inf)] if c !== nothing)
    for (bi, bv) in enumerate(ker_bas[1:min(8,end)])
        support = [(j, bv[j]) for j in 1:nc if bv[j] != 0]
        coeffs  = [c for (_,c) in support]
        is_flat = length(unique(coeffs)) == 1
        special_vals = Dict(nm => (c <= length(bv) ? bv[c] : 0) for (nm,c) in special_cols)
        _log("  basis[$(bi-1)]: support_size=$(length(support)) flat=$(is_flat ? "yes" : "no") specials=$special_vals")
        is_flat  && _log("    flat vector: all atoms share the same log")
        !is_flat && length(unique(values(special_vals))) == 1 && _log("    special atoms all equal")
        !is_flat && length(unique(values(special_vals))) > 1  && _log("    special atoms differ; DLP direction still present")
    end
    length(ker_bas) > 8 && _log("  ... $(length(ker_bas)-8) more basis vector(s)")

    # --- log-G membership test ---
    missing_cols = [nm for (nm,c) in [("gen0",p_col_gen0),("gen1",p_col_gen1),
                                       ("tgt0",p_col_tgt0),("tgt1",p_col_tgt1)] if c === nothing]
    if !isempty(missing_cols)
        _log("\n  ⚠  Cannot build log-G vector — columns missing after prune: $missing_cols")
        _log("     Skipping log-G membership test.")
        return null_hom
    end

    _log("\n  Building log-G candidate vector (known_key=$known_key) ...")
    v_logG = zeros(Int, nc)
    v_logG[p_col_gen0] = 1
    v_logG[p_col_gen1] = 0
    p_col_inf !== nothing && (v_logG[p_col_inf] = 0)

    inv2 = invmod(2, n)
    half_key = mod(known_key * inv2, n)
    v_logG[p_col_tgt0] = half_key
    v_logG[p_col_tgt1] = half_key
    _log("  a[gen0]=1, a[gen1]=0, a[tgt0]=a[tgt1]=$half_key (=$known_key/2 mod $n)")

    # residual = A_hom * v_logG mod n
    residual = [mod(sum(M_pruned[i,j] * v_logG[j] for j in 1:nc), n) for i in 1:nr]
    nonzero_rows = [(i, residual[i]) for i in 1:nr if residual[i] != 0]

    assigned_cols = Set(c for c in [p_col_gen0, p_col_gen1, p_col_tgt0, p_col_tgt1, p_col_inf]
                        if c !== nothing)

    true_failures      = Tuple{Int,Int}[]
    fb_partial_bad     = Tuple{Int,Int,Int}[]
    fb_residuals_clean = Tuple{Int,Int}[]

    for (row_i, resid) in nonzero_rows
        row_support = Set(j for j in 1:nc if M_pruned[row_i, j] != 0)
        if row_support ⊆ assigned_cols
            push!(true_failures, (row_i, resid))
        else
            partial = mod(sum(M_pruned[row_i, j] * v_logG[j]
                              for j in intersect(row_support, assigned_cols);
                              init=0), n)
            if partial != 0
                push!(fb_partial_bad, (row_i, resid, partial))
            else
                push!(fb_residuals_clean, (row_i, resid))
            end
        end
    end

    if isempty(nonzero_rows)
        _log("\n  ✓  log-G vector IS in the kernel of A_hom.")
        _log("     Walk relations are consistent with the known solution.")
        _log("     The failure is introduced by normalization rows, not the walk data.")
    else
        if !isempty(fb_residuals_clean)
            _log("\n  ℹ  $(length(fb_residuals_clean)) row(s) have nonzero residual only from unassigned fb atoms")
            _log("     (partial sum on special cols is zero -- genuinely underdetermined, not contradictory).")
        end
        if !isempty(fb_partial_bad)
            _log("\n  ✗  $(length(fb_partial_bad)) row(s) touch unassigned fb atoms BUT partial sum")
            _log("     on {gen0,gen1,tgt0,tgt1,inf} is already nonzero -- contradiction independent of fb logs:")
            for (row_i, resid, partial) in fb_partial_bad[1:min(30,end)]
                assigned_part   = [(atoms[j], M_pruned[row_i,j])
                                   for j in 1:nc if M_pruned[row_i,j]!=0 && j ∈ assigned_cols]
                unassigned_part = [(atoms[j], M_pruned[row_i,j])
                                   for j in 1:nc if M_pruned[row_i,j]!=0 && j ∉ assigned_cols]
                _log("    row $(lpad(row_i,5))  full_resid=$(lpad(resid,5))  partial_resid=$(lpad(partial,5))")
                _log("      assigned  : $(brief_atom_list(assigned_part; max_items=8))")
                _log("      unassigned: $(brief_atom_list(unassigned_part; max_items=8))")
            end
            length(fb_partial_bad) > 30 && _log("    ... and $(length(fb_partial_bad)-30) more rows")
            _log("\n     -> These rows contradict the known key regardless of fb atom values.")
            _log("        Likely cause: wrong xi multiplicity, wrong inf sign, or bad relation.")
        end
        if !isempty(true_failures)
            _log("\n  ✗  $(length(true_failures)) row(s) fail on assigned atoms only -- genuine contradiction:")
            for (row_i, resid) in true_failures[1:min(30,end)]
                row_atoms = [(atoms[j], M_pruned[row_i,j]) for j in 1:nc if M_pruned[row_i,j]!=0]
                _log("    row $(lpad(row_i,5))  residual=$(lpad(resid,5))  atoms=$row_atoms")
            end
            length(true_failures) > 30 && _log("    ... and $(length(true_failures)-30) more rows")
            _log("\n     -> The walk data itself contradicts the known key.")
            _log("        Likely cause: wrong xi multiplicity, wrong inf sign, or a bad")
            _log("        involution-closure row in the relation matrix.")
        end
        if isempty(true_failures) && isempty(fb_partial_bad)
            _log("\n  ✓  No true failures -- all residuals are from genuinely underdetermined rows.")
            _log("     Walk structure is consistent with the known key.")
        end
    end

    return null_hom
end

# ---------------------------------------------------------------------------
# extract_pin_rows — inspect a right-kernel basis
# ---------------------------------------------------------------------------
"""
Inspect a right-kernel basis and report structurally important directions.
Returns (pin_rows, pin_rhs, pin_labels) — always empty unless pin_isolated=true.
Each kernel vector in ker_bas is a Vector{Int} mod p.
"""
function extract_pin_rows(ker_bas::Vector{Vector{Int}}, atoms, n_cols::Int, p::Int,
                          p_col_inf; pin_isolated=false, max_preview=12)
    pin_rows   = Vector{Vector{Int}}()
    pin_rhs    = Int[]
    pin_labels = String[]

    counts = Dict("gauge"=>0, "isolated"=>0, "fusion"=>0, "parity"=>0, "other"=>0)
    isolated_previews   = String[]
    nonisolated_entries = Tuple{String, Vector{String}}[]

    for (vi, vec) in enumerate(ker_bas)
        support = [(j, vec[j]) for j in 1:n_cols if vec[j] != 0]
        isempty(support) && continue

        kind         = ""
        msg          = ""
        detail_lines = String[]

        if length(support) == 1 && p_col_inf !== nothing && support[1][1] == p_col_inf
            kind = "gauge"
            counts["gauge"] += 1
            msg = "kernel[$(vi-1)]: GAUGE (inf) -- keeping free"

        elseif length(support) == 1
            kind = "isolated"
            counts["isolated"] += 1
            j, coeff = support[1]
            atom = atoms[j]
            msg  = "kernel[$(vi-1)]: ISOLATED atom=$atom coeff=$coeff"
            if pin_isolated
                prow = zeros(Int, n_cols); prow[j] = 1
                push!(pin_rows, prow); push!(pin_rhs, 0)
                push!(pin_labels, "pin a[$atom]=0")
                msg *= " -- pinning"
            else
                msg *= " -- leaving free"
            end

        elseif length(support) == 2
            (j0, c0), (j1, c1) = support
            if (c0 == 1 && c1 == p-1) || (c0 == p-1 && c1 == 1)
                kind = "fusion"
                counts["fusion"] += 1
                a0, a1 = atoms[j0], atoms[j1]
                msg = "kernel[$(vi-1)]: FUSION a[$a0] = a[$a1]"
                push!(detail_lines, "    col $j0: atom=$a0  coeff=$c0")
                push!(detail_lines, "    col $j1: atom=$a1  coeff=$c1")
            else
                kind = "other"
                counts["other"] += 1
                msg = "kernel[$(vi-1)]: OTHER support_size=2 distinct_coeffs=$(sort(unique(c for (_,c) in support)))"
                for (j, c) in support
                    push!(detail_lines, "    col $j: atom=$(atoms[j])  coeff=$c")
                end
            end

        else
            coeffs_vals = [c for (_, c) in support]
            is_flat = length(unique(coeffs_vals)) == 1
            if is_flat
                kind = "parity"
                counts["parity"] += 1
                msg = "kernel[$(vi-1)]: PARITY/CONSERVATION support_size=$(length(support)) all_coeffs=$(coeffs_vals[1])"
                for (j, c) in support[1:min(12,end)]
                    push!(detail_lines, "    col $j: atom=$(atoms[j])  coeff=$c")
                end
                length(support) > 12 && push!(detail_lines, "    ... $(length(support)-12) more atoms")
            else
                kind = "other"
                counts["other"] += 1
                msg = "kernel[$(vi-1)]: OTHER support_size=$(length(support)) distinct_coeffs=$(sort(unique(coeffs_vals)))"
                for (j, c) in support[1:min(20,end)]
                    push!(detail_lines, "    col $j: atom=$(atoms[j])  coeff=$c")
                end
                length(support) > 20 && push!(detail_lines, "    ... $(length(support)-20) more atoms")
            end
        end

        if kind ∈ ("gauge", "isolated")
            length(isolated_previews) < max_preview && push!(isolated_previews, msg)
        else
            push!(nonisolated_entries, (msg, detail_lines))
        end
    end

    _log("  kernel summary: " * join(["$k=$v" for (k,v) in counts], ", "))
    for msg in isolated_previews; _log("  $msg"); end
    n_isolated_total = counts["gauge"] + counts["isolated"]
    omitted = n_isolated_total - length(isolated_previews)
    omitted > 0 && _log("  ... $omitted more isolated/gauge direction(s) omitted")

    if !isempty(nonisolated_entries)
        _log("\n  Non-isolated kernel directions ($(length(nonisolated_entries))) -- printed in full:")
        for (msg, dls) in nonisolated_entries
            _log("  $msg")
            for dl in dls; _log(dl); end
        end
    else
        _log("  (no non-isolated kernel directions)")
    end

    return pin_rows, pin_rhs, pin_labels
end

# ---------------------------------------------------------------------------
# build_balanced_anchor_row
# ---------------------------------------------------------------------------
"""
Return (row::Vector{Int}, rhs::Int, label::String) for the balanced anchor
    a[gen0] + a[gen1] - 5*a[inf] = 0  (or a[gen0] - a[gen1] = 0 if inf absent).
Returns (nothing, nothing, "anchor omitted") when cols unavailable.
"""
function build_balanced_anchor_row(p::Int, n_cols::Int, col_gen0, col_gen1, col_inf)
    (col_gen0 === nothing || col_gen1 === nothing) && return nothing, nothing, "anchor omitted"
    row = zeros(Int, n_cols)
    row[col_gen0] = 1
    row[col_gen1] = 1
    if col_inf !== nothing
        row[col_inf] = mod(-5, p)
        return row, 0, "anchor a[gen0]+a[gen1]-5*a[∞]=0"
    else
        row[col_gen1] = mod(-1, p)
        return row, 0, "anchor a[gen0]-a[gen1]=0"
    end
end

# ---------------------------------------------------------------------------
# Check 2: Left-kernel Farkas certificate
# ---------------------------------------------------------------------------
"""
When the full affine system A*x = b has no solution, extract a left-kernel
vector y s.t. y^T*A = 0 but y^T*b ≠ 0.
Returns (nonzero_entries, walk_row_indices).
"""
function extract_contradiction_certificate(M::Matrix{Int}, atoms, group_order::Int;
                                           col_inf=nothing, col_gen0=nothing, col_gen1=nothing,
                                           n_anchor_rows::Int=2)
    _section("CHECK 2: CONTRADICTION CERTIFICATE  (left-kernel Farkas row)")
    n  = group_order
    Fp = GF(n)

    M_pruned, row_sources = dedupe_rows_mod(M, group_order)
    pruned_aidx = Dict(string(atoms[i]) => i for i in 1:length(atoms))

    n_removed = size(M,1) - size(M_pruned,1)
    _log("  special-cols   : kept as loaded; only row dedup applied ($(size(M,1)) rows × $(size(M,2)) cols)")
    _log("  row dedup      : $n_removed duplicate/scalar-multiple row(s) removed")
    matrix_preview(M_pruned, atoms; max_rows=4, max_atoms=6)

    p_col_inf  = remap_col(col_inf,  atoms, pruned_aidx)
    p_col_gen0 = remap_col(col_gen0, atoms, pruned_aidx)
    p_col_gen1 = remap_col(col_gen1, atoms, pruned_aidx)

    n_walk, n_cols = size(M_pruned)

    A_nemo    = to_nemo_mat(M_pruned, Fp)
    ker_bas   = right_kernel_basis(A_nemo, n)
    null_pre  = length(ker_bas)
    _log("  pre-normalization nullity: $null_pre on the $(n_walk)×$(n_cols) homogeneous system")
    extract_pin_rows(ker_bas, atoms, n_cols, n, p_col_inf; pin_isolated=false, max_preview=10)

    # Augment with gauge row only (balanced anchor is solver-side only)
    extra_rows_int = Vector{Vector{Int}}()
    extra_rhs_int  = Int[]
    extra_labels   = String[]

    if p_col_inf !== nothing
        gr = zeros(Int, n_cols); gr[p_col_inf] = 1
        push!(extra_rows_int, gr); push!(extra_rhs_int, 0)
        push!(extra_labels, "gauge a[∞]=0  (col=$p_col_inf)")
    else
        _log("  no ∞ column available; gauge row omitted")
    end

    if p_col_gen0 !== nothing && p_col_gen1 !== nothing
        try
            arhs = invmod(5, group_order)
            _log("  solver normalization only: a[gen0], a[gen1] scaled by inv(5) mod $group_order = $arhs")
        catch
            _log("  solver normalization only: balanced anchor unavailable (5 not invertible mod group_order)")
        end
    else
        _log("  solver normalization only: balanced anchor unavailable")
    end

    if isempty(extra_rows_int)
        _log("  no augmentation rows; cannot find certificate")
        return Tuple{Int,Int}[], Int[]
    end

    # Stack walk rows + extra rows
    n_extra  = length(extra_rows_int)
    n_full   = n_walk + n_extra
    A_full_int = vcat(M_pruned, reduce(vcat, [reshape(r,1,:) for r in extra_rows_int]))
    b_full_int = vcat(zeros(Int, n_walk), extra_rhs_int)

    row_labels = ["walk[$i]" for i in 1:n_walk]
    append!(row_labels, extra_labels)

    _log("  augmented system: $n_full rows × $n_cols cols over GF($n)")
    A_full_nemo = to_nemo_mat(A_full_int, Fp)
    rank_A = rank(A_full_nemo)
    _log("  rank(A)=$rank_A")

    b_full_nemo = matrix(Fp, n_full, 1, [Fp(v) for v in b_full_int])
    consistent, _ = can_solve_with_solution(A_full_nemo, b_full_nemo; side=:right)

    if consistent
        _log("  ✓  system is consistent — no Farkas certificate exists")
        return Tuple{Int,Int}[], Int[]
    end
    _log("  ✗  inconsistent — extracting left-kernel certificate ...")

    left_bas   = left_kernel_basis(A_full_nemo, n)
    left_null  = length(left_bas)
    _log("  left kernel dimension: $left_null")

    if left_null == 0
        _log("  ✗  left kernel is trivial; unexpected")
        return Tuple{Int,Int}[], Int[]
    end

    certificate_y = nothing
    for bvec in left_bas
        if mod(sum(bvec[i] * b_full_int[i] for i in 1:n_full; init=0), n) != 0
            certificate_y = bvec; break
        end
    end

    if certificate_y === nothing
        _log("  no basis vector satisfies y·b!=0; trying small linear combinations ...")
        found = false
        for i in 1:length(left_bas), j in (i+1):length(left_bas)
            for ci in 1:min(n-1,4), cj in 1:min(n-1,4)
                cand = mod.(ci .* left_bas[i] .+ cj .* left_bas[j], n)
                if mod(sum(cand[k]*b_full_int[k] for k in 1:n_full; init=0), n) != 0
                    certificate_y = cand; found = true; break
                end
                found && break
            end
            found && break
        end
    end

    if certificate_y === nothing
        _log("  ✗  could not find certificate")
        return Tuple{Int,Int}[], Int[]
    end

    dot_b = mod(sum(certificate_y[i]*b_full_int[i] for i in 1:n_full; init=0), n)
    _log("  ✓  certificate found; y^T*b=$dot_b")

    nonzero_entries = [(i, certificate_y[i]) for i in 1:n_full if certificate_y[i] != 0]
    walk_entries    = [(i, c) for (i,c) in nonzero_entries if i <= n_walk]
    extra_entries   = [(i, c) for (i,c) in nonzero_entries if i > n_walk]

    _log("  certificate support: $(length(nonzero_entries)) total  |  walk=$(length(walk_entries))  extra=$(length(extra_entries))")

    if !isempty(walk_entries)
        _log("  walk rows in certificate:")
        for (row_i, coeff) in walk_entries[1:min(12,end)]
            row_atoms = [(atoms[j], M_pruned[row_i,j]) for j in 1:n_cols if M_pruned[row_i,j]!=0]
            _log("    row $(lpad(row_i,5))  weight=$(lpad(coeff,8))  $(brief_atom_list(row_atoms; max_items=6))")
        end
        length(walk_entries) > 12 && _log("    ... $(length(walk_entries)-12) more walk rows")
    else
        _log("  (no walk rows in certificate)")
    end

    if !isempty(extra_entries)
        _log("  augmented rows in certificate:")
        for (row_i, coeff) in extra_entries
            label = row_i <= length(row_labels) ? row_labels[row_i] : "row $row_i"
            _log("    row $(lpad(row_i,5))  weight=$(lpad(coeff,8))  [$label]")
        end
    end

    _section("CERTIFICATE DIAGNOSIS")
    extra_map      = Dict(i => c for (i,c) in extra_entries)
    anchor_row_idx = nothing
    gauge_row_idx  = nothing
    for (row_i, _) in extra_entries
        lbl = row_i <= length(row_labels) ? row_labels[row_i] : ""
        occursin("anchor", lbl) && (anchor_row_idx = row_i)
        occursin("gauge",  lbl) && (gauge_row_idx  = row_i)
    end

    _log("  anchor row weight in y  : $(get(extra_map, anchor_row_idx, nothing))")
    _log("  gauge row weight in y   : $(get(extra_map, gauge_row_idx,  nothing))")

    if anchor_row_idx !== nothing && get(extra_map, anchor_row_idx, 0) != 0
        _log("  the anchor row participates in the contradiction; normalization is suspect")
    end

    if isempty(walk_entries)
        _log("  contradiction is entirely in normalization rows, not in walk data")
    else
        atom_freq = Dict{String,Int}()
        for (row_i, _) in walk_entries
            for j in 1:n_cols
                M_pruned[row_i,j] != 0 || continue
                key = string(atoms[j])
                atom_freq[key] = get(atom_freq, key, 0) + 1
            end
        end
        top_atoms = sort(collect(atom_freq), by=kv->-kv[2])[1:min(8,end)]
        _log("  most frequent atoms in certificate rows:")
        for (atom, freq) in top_atoms
            _log("    $(lpad(atom,8))  appears in $freq certificate row(s)")
        end
    end

    walk_row_indices = sort(unique(i for (i,_) in walk_entries))
    return nonzero_entries, walk_row_indices
end

# ---------------------------------------------------------------------------
# Check 3: Structural collapse triage
# ---------------------------------------------------------------------------
function check_structural_collapse(M::Matrix{Int}, atoms, group_order::Int;
                                   col_inf=nothing, col_gen0=nothing, col_gen1=nothing,
                                   col_tgt0=nothing, col_tgt1=nothing)
    _section("CHECK 3: STRUCTURAL COLLAPSE TRIAGE")
    n  = group_order
    Fp = GF(n)

    M_pruned, row_sources = dedupe_rows_mod(M, group_order)
    pruned_aidx = Dict(string(atoms[i]) => i for i in 1:length(atoms))
    n_removed   = sum(length(v) for v in row_sources) - length(row_sources)
    _log("  row dedup    : $n_removed duplicates removed")

    function remap(col)
        col === nothing && return nothing
        if !(1 <= col <= length(atoms))
            return nothing
        end
        return get(pruned_aidx, string(atoms[col]), nothing)
    end

    p_col_inf  = remap(col_inf)
    p_col_gen0 = remap(col_gen0)
    p_col_gen1 = remap(col_gen1)
    p_col_tgt0 = remap(col_tgt0)
    p_col_tgt1 = remap(col_tgt1)

    n_rows, n_cols = size(M_pruned)
    A_nemo   = to_nemo_mat(M_pruned, Fp)
    rnk      = rank(A_nemo)
    full_null = n_cols - rnk
    _log("  pruned matrix: $n_rows rows × $n_cols cols")
    _log("  nullity       : $full_null")

    # A) Special-column order test
    _log("\n  --- A) Special-column order test ---")
    specials = [("inf",p_col_inf),("gen0",p_col_gen0),("gen1",p_col_gen1),
                ("tgt0",p_col_tgt0),("tgt1",p_col_tgt1)]
    for (name, col) in specials
        if col === nothing
            _log("  $(lpad(name,6)): absent/unresolved"); continue
        end
        ej = zeros(Int, n_cols); ej[col] = 1
        found_k = nothing
        for k in 1:min(24, n-1)
            kej = mod.(k .* ej, n)
            ej_nemo = matrix(Fp, 1, n_cols, [Fp(v) for v in kej])
            # Check if k*e_j is in the row space: reduce and see if zero
            # We test: does [A; k*e_j] have rank = rank(A)?
            Aext = vcat(A_nemo, ej_nemo)
            if rank(Aext) == rnk
                found_k = k; break
            end
        end
        _log("  $(lpad(name,6)) (col $(lpad(col,5))): " * (found_k !== nothing ? "k=$found_k" : "k>24"))
    end

    # B) Rank-without-inf test
    _log("\n  --- B) Rank-without-inf test ---")
    if p_col_inf !== nothing
        cols_no_inf = [j for j in 1:n_cols if j != p_col_inf]
        A_no_inf    = to_nemo_mat(M_pruned[:, cols_no_inf], Fp)
        null_no_inf = length(cols_no_inf) - rank(A_no_inf)
        _log("  without-inf nullity: $null_no_inf")
        if null_no_inf > full_null
            _log("  inf column is absorbing degrees of freedom.")
        elseif null_no_inf == full_null
            _log("  inf column is not the source of collapse.")
        else
            _log("  inf column contributed free directions.")
        end
    else
        _log("  no inf column present; skipped.")
    end

    # C) Direct fusion audit
    _log("\n  --- C) Direct fusion audit ---")
    special_by_col = Dict(c => nm for (nm, c) in specials if c !== nothing)
    col_groups = Dict{Vector{Tuple{Int,Int}}, Vector{Int}}()
    zero_cols  = Int[]
    for j in 1:n_cols
        sig = Tuple{Int,Int}[]
        lead_inv = nothing
        for i in 1:n_rows
            v = Int(lift(ZZ, A_nemo[i,j]))
            if v != 0
                if lead_inv === nothing
                    lead_inv = invmod(v, n)
                end
                push!(sig, (i, mod(v * lead_inv, n)))
            end
        end
        if lead_inv === nothing
            push!(zero_cols, j)
        else
            existing = get(col_groups, sig, nothing)
            if existing === nothing
                col_groups[sig] = [j]
            else
                push!(existing, j)
            end
        end
    end

    fusion_groups = [cols for cols in values(col_groups) if length(cols) > 1]
    if !isempty(zero_cols)
        preview = join(["$(atoms[j])(col=$j)" for j in zero_cols[1:min(10,end)]], ", ")
        _log("  zero columns: $(length(zero_cols))")
        _log("    $preview" * (length(zero_cols) > 10 ? " ..." : ""))
    end
    if !isempty(fusion_groups)
        _log("  fusion classes: $(length(fusion_groups))")
        for cols in fusion_groups[1:min(8,end)]
            labels = String[]
            special_hits = String[]
            for j in cols
                atom = string(atoms[j])
                if haskey(special_by_col, j)
                    push!(special_hits, special_by_col[j])
                    push!(labels, "$atom($(special_by_col[j]))")
                else
                    push!(labels, atom)
                end
            end
            flag = !isempty(special_hits) ? "  special=$(sort(unique(special_hits)))" : ""
            _log("    $labels$flag")
        end
        length(fusion_groups) > 8 && _log("    ... $(length(fusion_groups)-8) more fusion class(es)")
    else
        _log("  no proportional column classes found.")
    end

    # C2) Kernel-basis fusion sanity check (right kernel needed; compute here)
    _log("\n  --- C2) Kernel-basis fusion sanity check ---")
    ker_bas = right_kernel_basis(A_nemo, n)
    kernel_fusions = Tuple{Any,Any}[]
    for vec in ker_bas
        support = [(j, vec[j]) for j in 1:n_cols if vec[j] != 0]
        if length(support) == 2
            (j0, c0), (j1, c1) = support
            if (c0 == 1 && c1 == n-1) || (c0 == n-1 && c1 == 1)
                push!(kernel_fusions, (atoms[j0], atoms[j1]))
            end
        end
    end
    if !isempty(kernel_fusions)
        _log("  support-2 kernel fusions: $(length(kernel_fusions))")
        for (a0, a1) in kernel_fusions[1:min(8,end)]
            _log("    a[$a0] = a[$a1]")
        end
        length(kernel_fusions) > 8 && _log("    ... $(length(kernel_fusions)-8) more")
    else
        _log("  no support-2 kernel fusions.")
    end

    # D) Rows hitting special columns
    _log("\n  --- D) Rows hitting special columns ---")
    for (col_name, p_col) in [("gen0",p_col_gen0),("gen1",p_col_gen1),
                               ("tgt0",p_col_tgt0),("tgt1",p_col_tgt1)]
        p_col === nothing && continue
        hitting_rows = [(i, Int(lift(ZZ, A_nemo[i,p_col])))
                        for i in 1:n_rows if A_nemo[i,p_col] != Fp(0)]
        _log("  $col_name: $(length(hitting_rows)) row(s)")
        for (row_i, coeff) in hitting_rows[1:min(5,end)]
            row_atoms = [(atoms[j], M_pruned[row_i,j]) for j in 1:n_cols if M_pruned[row_i,j]!=0]
            _log("    row $(lpad(row_i,5)) coeff=$(lpad(coeff,6)) atoms=$(brief_atom_list(row_atoms; max_items=5))")
        end
        length(hitting_rows) > 5 && _log("    ... $(length(hitting_rows)-5) more rows")
    end

    # E) Row-subsampling stability
    _log("\n  --- E) Row-subsampling stability ---")
    seed!(42)
    n_drop = max(1, n_rows ÷ 10)
    _log("  base nullity=$full_null; dropping $n_drop/$n_rows rows per trial")
    for trial in 1:5
        perm  = randperm(n_rows)
        drop  = Set(perm[1:n_drop])
        keep  = [i for i in 1:n_rows if i ∉ drop]
        null_sub = size(M_pruned,2) - rank(to_nemo_mat(M_pruned[keep,:], Fp))
        delta = null_sub - full_null
        _log("  trial $trial: nullity=$null_sub delta=$(delta >= 0 ? "+$delta" : "$delta")")
    end
end

# ---------------------------------------------------------------------------
# Check 4: Incremental consistency filter
# ---------------------------------------------------------------------------
function incremental_consistency_filter(M::Matrix{Int}, atoms, group_order::Int;
                                        col_inf=nothing, col_gen0=nothing, col_gen1=nothing,
                                        col_tgt0=nothing, col_tgt1=nothing)
    _section("CHECK 4: INCREMENTAL CONSISTENCY FILTER")
    n  = group_order
    p  = n

    M_pruned, row_sources = dedupe_rows_mod(M, group_order)
    pruned_aidx = Dict(string(atoms[i]) => i for i in 1:length(atoms))
    n_removed   = sum(length(v) for v in row_sources) - length(row_sources)
    _log("  row dedup    : $n_removed duplicates removed")

    function remap(col)
        col === nothing && return nothing
        if !(1 <= col <= length(atoms))
            return nothing
        end
        return get(pruned_aidx, string(atoms[col]), nothing)
    end

    p_col_inf  = remap(col_inf)
    p_col_gen0 = remap(col_gen0)
    p_col_gen1 = remap(col_gen1)
    p_col_tgt0 = remap(col_tgt0)
    p_col_tgt1 = remap(col_tgt1)

    n_rows, n_cols = size(M_pruned)
    _log("  Matrix: $n_rows rows × $n_cols cols over GF($p)")

    # Pivots: col_index => (pivot_row::Vector{Int}, pivot_rhs::Int) — all mod p
    pivots = Dict{Int, Tuple{Vector{Int},Int}}()

    function row_from_matrix(i)
        return [mod(M_pruned[i,j], p) for j in 1:n_cols]
    end

    function reduce_row(row, rhs)
        row = copy(row); rhs = rhs
        for (pc, (prow, prhs)) in pivots
            coeff = mod(row[pc], p)
            coeff == 0 && continue
            row = mod.(row .- coeff .* prow, p)
            rhs = mod(rhs - coeff * prhs, p)
        end
        return row, rhs
    end

    function add_pivot!(row, rhs)
        for j in 1:n_cols
            mod(row[j], p) != 0 || continue
            inv_v = invmod(row[j], p)
            pivots[j] = (mod.(row .* inv_v, p), mod(rhs * inv_v, p))
            return
        end
    end

    if p_col_inf !== nothing
        gauge = zeros(Int, n_cols); gauge[p_col_inf] = 1
        add_pivot!(gauge, 0)
        _log("  Seeded gauge a[∞]=0 (col $p_col_inf)")
    end

    anchor_row, anchor_rhs, anchor_label = build_balanced_anchor_row(p, n_cols, p_col_gen0, p_col_gen1, p_col_inf)
    if anchor_row !== nothing
        add_pivot!(anchor_row, anchor_rhs)
        _log("  Seeded $anchor_label")
    else
        _log("  Balanced anchor unavailable.")
    end

    good_rows  = Int[]
    bad_rows   = Int[]
    dep_rows   = Int[]
    first_bad  = nothing

    for i in 1:n_rows
        row_r, rhs_r = reduce_row(row_from_matrix(i), 0)
        if all(v == 0 for v in row_r)
            if mod(rhs_r, p) == 0
                push!(dep_rows, i)
            else
                push!(bad_rows, i)
                first_bad === nothing && (first_bad = i)
            end
        else
            add_pivot!(row_r, rhs_r)
            push!(good_rows, i)
        end
    end

    _log("  good=$(length(good_rows))  bad=$(length(bad_rows))  dependent=$(length(dep_rows))  first_bad=$first_bad")

    if isempty(bad_rows)
        _log("  ✓  No contradictions found in step order.")
        return
    end

    special_names = Dict{Int,String}()
    for (nm, pc) in [("inf",p_col_inf),("gen0",p_col_gen0),("gen1",p_col_gen1),
                     ("tgt0",p_col_tgt0),("tgt1",p_col_tgt1)]
        pc !== nothing && (special_names[pc] = nm)
    end

    function atom_freq(row_list; top=10)
        freq = Dict{String,Int}()
        for i in row_list, j in 1:n_cols
            mod(M_pruned[i,j], p) != 0 || continue
            key = string(atoms[j])
            freq[key] = get(freq, key, 0) + 1
        end
        return sort(collect(freq), by=kv->-kv[2])[1:min(top,end)], freq
    end

    top_bad,  bad_freq  = atom_freq(bad_rows;  top=10)
    top_good, good_freq = atom_freq(good_rows; top=10)

    _log("  top bad-row atoms:")
    for (atom, cnt) in top_bad[1:min(8,end)]
        col = get(pruned_aidx, atom, nothing)
        sp  = (col !== nothing && haskey(special_names, col)) ? " [$(special_names[col])]" : ""
        _log("    $(lpad(atom,8))  $(lpad(cnt,5)) rows$sp")
    end

    _log("  first bad rows:")
    for i in bad_rows[1:min(8,end)]
        row_atoms = [(string(atoms[j]), M_pruned[i,j]) for j in 1:n_cols if mod(M_pruned[i,j],p)!=0]
        _log("    row $(lpad(i,5)): $(brief_atom_list(row_atoms; max_items=6))")
    end
    length(bad_rows) > 8 && _log("    ... $(length(bad_rows)-8) more bad rows")

    _log("  top enrichment (bad/good):")
    n_bad_  = max(length(bad_rows),  1)
    n_good_ = max(length(good_rows), 1)
    all_atms = union(keys(bad_freq), keys(good_freq))
    enrichment = Tuple{String,Float64,Float64,Float64}[]
    for atom in all_atms
        br = get(bad_freq,  atom, 0) / n_bad_
        gr = get(good_freq, atom, 0) / n_good_
        if gr > 0
            push!(enrichment, (atom, br/gr, br, gr))
        elseif br > 0
            push!(enrichment, (atom, Inf, br, gr))
        end
    end
    sort!(enrichment, by=x->-x[2])
    for (atom, ratio, br, gr) in enrichment[1:min(8,end)]
        col = get(pruned_aidx, atom, nothing)
        sp  = (col !== nothing && haskey(special_names, col)) ? " [$(special_names[col])]" : ""
        ratio_str = isinf(ratio) ? "inf" : @sprintf("%.2f", ratio)
        _log("    $(lpad(atom,8))  enrich=$ratio_str  bad=$(@sprintf("%.4f",br)) good=$(@sprintf("%.4f",gr))$sp")
    end
end

# ---------------------------------------------------------------------------
# Seed suggestion from non-parity nullity
# ---------------------------------------------------------------------------
function suggest_seeds_from_noparity_nullity(M::Matrix{Int}, atoms, group_order::Int,
                                              col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1;
                                              n_seeds::Int=4)
    n  = group_order
    Fp = GF(n)
    M_pruned, _ = dedupe_rows_mod(M, group_order)
    n_cols = size(M_pruned, 2)
    pruned_aidx = Dict(string(atoms[i]) => i for i in 1:length(atoms))

    function remap(col)
        col === nothing && return nothing
        if !(1 <= col <= length(atoms))
            return nothing
        end
        return get(pruned_aidx, string(atoms[col]), nothing)
    end

    p_col_inf = remap(col_inf)
    protected_cols = Set(c for c in [remap(col_inf), remap(col_gen0), remap(col_gen1),
                                      remap(col_tgt0), remap(col_tgt1)] if c !== nothing)

    A_nemo  = to_nemo_mat(M_pruned, Fp)
    ker_bas = right_kernel_basis(A_nemo, n)

    atom_freq = Dict{String,Int}()
    for vec in ker_bas
        support = [(j, vec[j]) for j in 1:n_cols if vec[j] != 0]
        isempty(support) && continue
        # Skip gauge direction
        length(support) == 1 && p_col_inf !== nothing && support[1][1] == p_col_inf && continue
        # Skip isolated singletons
        length(support) == 1 && continue
        # Skip flat / conservation directions
        coeffs = [c for (_,c) in support]
        length(unique(coeffs)) == 1 && continue
        # Non-parity direction: accumulate
        for (j, _) in support
            j ∈ protected_cols && continue
            key = string(atoms[j])
            atom_freq[key] = get(atom_freq, key, 0) + 1
        end
    end

    isempty(atom_freq) && return String[]
    ranked = sort(collect(atom_freq), by=kv->-kv[2])
    return [atom for (atom,_) in ranked[1:min(n_seeds,end)]]
end

# ---------------------------------------------------------------------------
# Farkas-delete re-run
# ---------------------------------------------------------------------------
function farkas_delete_rerun(M::Matrix{Int}, atoms, aidx, group_order::Int,
                              farkas_walk_rows::Vector{Int},
                              col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1;
                              known_key=nothing)
    _log("\n$('#'^70)")
    _log("# FARKAS-DELETE RE-RUN")
    _log("#  Deleting $(length(farkas_walk_rows)) certificate walk row(s) from M")
    _log("$('#'^70)")

    n_orig    = size(M, 1)
    farkas_set = Set(farkas_walk_rows)
    keep       = [i for i in 1:n_orig if i ∉ farkas_set]
    M_reduced  = M[keep, :]
    _log("  Original rows: $n_orig  →  Reduced rows: $(size(M_reduced,1))  (deleted $(length(farkas_walk_rows)) rows)")

    # Check 1 on reduced
    if known_key !== nothing
        check_homogeneous(M_reduced, atoms, aidx, group_order,
                          known_key, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf)
    else
        _section("CHECK 1 (reduced): HOMOGENEOUS SYSTEM  (no --known-key)")
        Fp  = GF(group_order)
        A   = to_nemo_mat(M_reduced, Fp)
        rk  = rank(A)
        nul = size(M_reduced,2) - rk
        _log("  rows=$(size(M_reduced,1))  cols=$(size(M_reduced,2))  rank=$rk  nullity=$nul")
    end

    # Check 2 on reduced
    cert_entries2, farkas_walk_rows2 = extract_contradiction_certificate(
        M_reduced, atoms, group_order;
        col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1)

    if isempty(cert_entries2)
        _log("\n  [farkas-delete] Reduced system is CONSISTENT after deletion.")
        _log("  Contradiction was localized to the certificate rows.")
    else
        _log("\n  [farkas-delete] Reduced system still INCONSISTENT.  New certificate uses $(length(farkas_walk_rows2)) walk row(s).")
        overlap = intersect(farkas_set, Set(farkas_walk_rows2))
        _log("  Overlap with deleted rows: $(length(overlap))  (should be 0 — deleted rows are gone)")
    end

    # Check 3 on reduced
    check_structural_collapse(M_reduced, atoms, group_order;
                               col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1,
                               col_tgt0=col_tgt0, col_tgt1=col_tgt1)

    _log("\n$('#'^70)")
    _log("# FARKAS-DELETE RE-RUN COMPLETE")
    _log("$('#'^70)")
end

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
    inferred_specials = infer_special_cols_from_divisor_xs(aidx, divisor_xs)
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
        for x in parsed["exclude-xs"]
            c = get(aidx, string(x), nothing)
            if c === nothing
                _log("[filter] --exclude-xs $x: not found in atom index, skipping")
            elseif c ∈ protected_cols
                _log("[filter] --exclude-xs $x: is a protected col ($c), skipping")
            else
                push!(excluded_cols, c)
            end
        end
        if !isempty(excluded_cols)
            excluded_atoms = [atoms[c] for c in sort(collect(excluded_cols))]
            _log("[filter] excluding $(length(excluded_cols)) atom(s): $excluded_atoms")
            n_before = size(M, 1)
            keep_rows = [r for r in 1:size(M,1) if !any(M[r,c] != 0 for c in excluded_cols)]
            M = M[keep_rows, :]
            n_dropped_rows = n_before - size(M,1)
            n_cols_before  = length(atoms)
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

    # --- Repair tangent rows (coeff sum == -1) ---
    n_cols    = size(M, 2)
    rows_list = [[M[r,c] for c in 1:n_cols] for r in 1:size(M,1)]
    repaired  = Int[]
    malformed = Int[]
    for (r, row) in enumerate(rows_list)
        s = sum(row)
        s == 0 && continue
        if s == -1
            fixed = false
            for c in 1:n_cols
                if row[c] == 3 && (col_inf === nothing || c != col_inf)
                    rows_list[r][c] = 4
                    push!(repaired, r); fixed = true; break
                end
            end
            !fixed && push!(malformed, r)
        else
            push!(malformed, r)
        end
    end
    malformed_set = Set(malformed)
    M = reduce(vcat, [reshape(rows_list[r], 1, :) for r in 1:length(rows_list) if r ∉ malformed_set])

    if !isempty(repaired)
        preview = repaired[1:min(16,end)]
        _log("[filter] repaired $(length(repaired)) tangent row(s) (xk==xi, xi coeff 3→4): $preview" *
             (length(repaired) > 16 ? " ..." : ""))
    end
    if !isempty(malformed)
        preview = malformed[1:min(16,end)]
        _log("[filter] dropping $(length(malformed)) malformed row(s) (coeff sum ≠ 0, not repairable): $preview" *
             (length(malformed) > 16 ? " ..." : ""))
        _log("[filter] matrix after malformed-row drop: $(size(M,1)) × $(size(M,2))")
    end
    if isempty(repaired) && isempty(malformed)
        _log("[filter] all rows well-formed (coeff sums all zero).")
    end

    # --- Check 1 ---
    if known_key !== nothing
        check_homogeneous(M, atoms, aidx, group_order,
                          known_key, col_gen0, col_gen1, col_tgt0, col_tgt1, col_inf)
    else
        _section("CHECK 1: HOMOGENEOUS SYSTEM  (skipped — no --known-key)")
        Fp      = GF(group_order)
        A_hom   = to_nemo_mat(M, Fp)
        rank_hom = rank(A_hom)
        null_hom = size(M,2) - rank_hom
        _log("  rows=$(size(M,1))  cols=$(size(M,2))  rank=$rank_hom  nullity=$null_hom")
    end

    # --- Check 2 ---
    cert_entries, farkas_walk_rows = extract_contradiction_certificate(
        M, atoms, group_order;
        col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1)

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

    # --- Check 4 ---
    incremental_consistency_filter(M, atoms, group_order;
                                   col_inf=col_inf, col_gen0=col_gen0, col_gen1=col_gen1,
                                   col_tgt0=col_tgt0, col_tgt1=col_tgt1)

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

main()
