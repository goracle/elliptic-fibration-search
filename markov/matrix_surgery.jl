#!/usr/bin/env julia
"""
matrix_surgery.jl

Heuristic pruning and diagnostics for large genus-2 index-calculus relation
matrices stored in the same HDF5 layout used by dlp_contradiction_diag.jl.

This is a first-pass surgery engine with four stages:

  1. Load the matrix and metadata from HDF5.
  2. Structural cleanup:
       - drop zero rows
       - deduplicate exact duplicate rows mod p
       - compute column degrees / support graph components
  3. Iterative pruning:
       - propose non-special low-information columns
       - test them by comparing sampled sparse rank estimates before/after
       - keep only candidates that look rank-redundant and do not disturb
         the designated generator / target columns
  4. Diagnostics and optional write-out:
       - report component structure, candidate removals, rank/nullity estimate
       - save the pruned matrix to a new HDF5 file

The script is deliberately conservative. It does not try to prove that a column
is redundant; it tries to find a smaller matrix that still behaves like the
original under the usual kernel tests.

Usage
-----

  julia matrix_surgery.jl relation_matrix.h5
  julia matrix_surgery.jl relation_matrix.h5 --out pruned_matrix.h5
  julia matrix_surgery.jl relation_matrix.h5 --sample-rows 2048 --max-rounds 6

Notes
-----
- The loader supports the same on-disk shapes as the diagnostics script:
  `matrix_dense` or CSR datasets (`csr/data`, `csr/indices`, `csr/indptr`,
  `csr/shape`), plus metadata such as `atom_index`, `col_inf`, `col_gen0`,
  `col_gen1`, `col_tgt0`, `col_tgt1`, `group_order`, `field_prime`, and
  `divisor_xs`.
- For large matrices, sparse rank estimates are only lower bounds. The pruning
  loop uses them as a fast filter, not as a proof.
"""

import Pkg

const REQUIRED_PKGS = ["HDF5", "JSON3", "ArgParse", "SparseArrays",
                       "LinearAlgebra", "Random", "Printf"]

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
using SparseArrays
using LinearAlgebra
using Random
using Printf

const SEP  = "=" ^ 76
const THIN = "-" ^ 76

_log(msg) = (println(msg); flush(stdout))
_section(title) = (_log("\n$SEP"); _log("  $title"); _log(SEP))

# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

"""
Load the matrix and metadata from the HDF5 relation dump.

Returns a NamedTuple with:
  M, atoms, aidx, group_order, field_prime, divisor_xs,
  col_inf, col_gen0, col_gen1, col_tgt0, col_tgt1
"""
function load_matrix_hdf5(path::String)
    isfile(path) || error("file not found: $path")

    return h5open(path, "r") do f
        atoms_raw = read(f["atoms"])
        atoms = isa(atoms_raw[1], AbstractString) ? collect(atoms_raw) :
                [String(a) for a in atoms_raw]

        aidx_raw = read(f["atom_index"])
        aidx_str = isa(aidx_raw, AbstractString) ? aidx_raw : String(aidx_raw)
        aidx = Dict{String,Int}(string(k) => (v + 1) for (k,v) in JSON3.read(aidx_str))

        M = if haskey(f, "matrix_dense")
            Matrix(transpose(Int.(read(f["matrix_dense"]))))
        else
            data_vals = Int.(read(f["csr/data"]))
            indices   = Int.(read(f["csr/indices"])) .+ 1
            indptr    = Int.(read(f["csr/indptr"]))
            shape     = Tuple(Int.(read(f["csr/shape"])))
            nr, nc    = shape

            I = Vector{Int}(undef, length(data_vals))
            J = Vector{Int}(undef, length(data_vals))
            V = copy(data_vals)

            k = 1
            for r in 1:nr
                for idx in (indptr[r] + 1):(indptr[r + 1])
                    I[k] = r
                    J[k] = indices[idx]
                    k += 1
                end
            end
            sparse(I, J, V, nr, nc)
        end

        group_order = haskey(f, "group_order") ? Int(read(f["group_order"])) : nothing
        field_prime = haskey(f, "field_prime")  ? Int(read(f["field_prime"])) : nothing
        divisor_xs  = haskey(f, "divisor_xs")   ? Int.(read(f["divisor_xs"])) : nothing

        function _col(key)
            !haskey(f, key) && return nothing
            v = Int(read(f[key]))
            return v >= 0 ? v + 1 : nothing
        end

        (
            M = M,
            atoms = atoms,
            aidx = aidx,
            group_order = group_order,
            field_prime = field_prime,
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
# Utilities
# ---------------------------------------------------------------------------

modp(x::Integer, p::Int) = begin
    y = mod(Int(x), p)
    y < 0 ? y + p : y
end

function to_sparse_mod(M::SparseMatrixCSC{Int,Int}, p::Int)::SparseMatrixCSC{Int,Int}
    m, n = size(M)
    rv = rowvals(M)
    nz = nonzeros(M)
    I = Int[]; J = Int[]; V = Int[]
    for j in 1:n
        for idx in nzrange(M, j)
            v = modp(nz[idx], p)
            v == 0 && continue
            push!(I, rv[idx])
            push!(J, j)
            push!(V, v)
        end
    end
    return sparse(I, J, V, m, n)
end

function to_sparse_mod(M::AbstractMatrix{Int}, p::Int)::SparseMatrixCSC{Int,Int}
    m, n = size(M)
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, min(m*n, 100_000))
    sizehint!(J, min(m*n, 100_000))
    sizehint!(V, min(m*n, 100_000))
    for j in 1:n, i in 1:m
        v = modp(M[i, j], p)
        v == 0 && continue
        push!(I, i); push!(J, j); push!(V, v)
    end
    return sparse(I, J, V, m, n)
end

function row_dicts(A::SparseMatrixCSC{Int,Int}, p::Int, rows_sel::Vector{Int})
    m, n = size(A)
    rowset = Set(rows_sel)
    rowmap = Dict(rows_sel[i] => i for i in eachindex(rows_sel))
    rows = [Dict{Int,Int}() for _ in 1:length(rows_sel)]
    rv = rowvals(A)
    nz = nonzeros(A)
    for col in 1:n
        for idx in nzrange(A, col)
            r = rv[idx]
            haskey(rowmap, r) || continue
            v = modp(nz[idx], p)
            v == 0 && continue
            rows[rowmap[r]][col] = v
        end
    end
    return rows
end

function sparse_rank_estimate_rows(rows::Vector{Dict{Int,Int}}, p::Int)
    # Sparse left-to-right elimination on a row sample.
    pivot_cols = Int[]
    pivot_table = Dict{Int, Dict{Int,Int}}()
    rank_est = 0

    for row_in in rows
        row = copy(row_in)
        isempty(row) && continue

        for pc in pivot_cols
            c = get(row, pc, 0)
            c == 0 && continue
            prow = pivot_table[pc]
            for (j, pv) in prow
                cur = get(row, j, 0)
                nv = mod(cur - c * pv, p)
                if nv == 0
                    delete!(row, j)
                else
                    row[j] = nv
                end
            end
        end

        isempty(row) && continue

        pc = minimum(keys(row))
        pv = row[pc]
        inv_pv = invmod(pv, p)

        new_row = Dict{Int,Int}()
        for (j, v) in row
            nv = mod(v * inv_pv, p)
            nv != 0 && (new_row[j] = nv)
        end

        pivot_table[pc] = new_row
        insert!(pivot_cols, searchsortedfirst(pivot_cols, pc), pc)
        rank_est += 1
    end

    return rank_est
end

function sparse_rank_estimate(A::SparseMatrixCSC{Int,Int}, p::Int;
                              n_rows::Int = min(size(A,1), 1024),
                              rng = MersenneTwister(99))
    m, n = size(A)
    n_rows = min(n_rows, m)
    rows_sel = randperm(rng, m)[1:n_rows]
    rows = row_dicts(A, p, rows_sel)
    rank_est = sparse_rank_estimate_rows(rows, p)
    return rank_est, n - rank_est, rows_sel
end


function atom_x_value(atom::AbstractString)
    m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", atom)
    if m !== nothing
        return parse(Int, m.captures[1])
    end
    x = tryparse(Int, atom)
    return x
end

function infer_special_cols_from_divisor_xs(atoms, aidx, divisor_xs)
    divisor_xs === nothing && return Dict{String,Union{Nothing,Int}}()

    labels = ["gen0", "gen1", "tgt0", "tgt1"]
    inferred = Dict{String,Union{Nothing,Int}}()

    x_to_col = Dict{Int, Int}()
    for (j, atm) in enumerate(atoms)
        x_val = atom_x_value(string(atm))
        x_val === nothing && continue
        if !haskey(x_to_col, x_val)
            x_to_col[x_val] = j
        end
    end

    if length(divisor_xs) >= 8
        n_pairs = length(divisor_xs) ÷ 2
        for (idx, lab) in enumerate(labels)
            idx > n_pairs && break
            x_val = Int(divisor_xs[2 * idx - 1])
            y_val = Int(divisor_xs[2 * idx])
            col = get(aidx, "($(x_val), $(y_val))", nothing)
            if col === nothing && haskey(x_to_col, x_val)
                col = x_to_col[x_val]
            end
            inferred[lab] = col
        end
    else
        n_xs = min(length(divisor_xs), length(labels))
        for (idx, lab) in enumerate(labels)
            idx > n_xs && break
            x_val = Int(divisor_xs[idx])
            col = get(aidx, string(x_val), nothing)
            col === nothing && (col = get(x_to_col, x_val, nothing))
            inferred[lab] = col
        end
    end

    return inferred
end

function special_cols(nt)
    cols = Int[]
    for key in (:col_inf, :col_gen0, :col_gen1, :col_tgt0, :col_tgt1)
        v = getfield(nt, key)
        v === nothing || push!(cols, v)
    end

    if hasproperty(nt, :atoms) && hasproperty(nt, :aidx) && hasproperty(nt, :divisor_xs)
        inferred = infer_special_cols_from_divisor_xs(getfield(nt, :atoms),
                                                      getfield(nt, :aidx),
                                                      getfield(nt, :divisor_xs))
        for key in ("gen0", "gen1", "tgt0", "tgt1")
            v = get(inferred, key, nothing)
            v === nothing || push!(cols, v)
        end
    end

    return unique(cols)
end

function col_degrees(A::SparseMatrixCSC{Int,Int})
    n = size(A, 2)
    deg = zeros(Int, n)
    for j in 1:n
        deg[j] = length(nzrange(A, j))
    end
    return deg
end

function row_support_sig(A::SparseMatrixCSC{Int,Int}, p::Int, i::Int)
    rv = rowvals(A)
    nz = nonzeros(A)
    sig = Vector{Tuple{Int,Int}}()
    for j in 1:size(A,2)
        for idx in nzrange(A, j)
            rv[idx] == i || continue
            v = modp(nz[idx], p)
            v == 0 && continue
            push!(sig, (j, v))
        end
    end
    return sig
end

function row_entries(A::SparseMatrixCSC{Int,Int})
    m, n = size(A)
    rows = [Tuple{Int,Int}[] for _ in 1:m]
    rv = rowvals(A)
    nz = nonzeros(A)
    for j in 1:n
        for idx in nzrange(A, j)
            push!(rows[rv[idx]], (j, Int(nz[idx])))
        end
    end
    return rows
end

function prune_zero_rows(A::SparseMatrixCSC{Int,Int})
    rows = row_entries(A)
    keep = [i for i in 1:length(rows) if !isempty(rows[i])]
    return A[keep, :], keep
end

function dedupe_rows_mod(A::SparseMatrixCSC{Int,Int}, p::Int)
    rows = row_entries(A)
    seen = Dict{Tuple, Int}()
    keep = Int[]
    row_sources = Vector{Vector{Int}}()

    for i in 1:length(rows)
        sig = rows[i]
        isempty(sig) && continue
        lead = sig[1][2]
        inv_lead = invmod(modp(lead, p), p)
        canon = Tuple([(j, modp(v * inv_lead, p)) for (j, v) in sig])

        if !haskey(seen, canon)
            seen[canon] = length(keep) + 1
            push!(keep, i)
            push!(row_sources, [i])
        else
            push!(row_sources[seen[canon]], i)
        end
    end

    return A[keep, :], keep, row_sources
end

# ---------------------------------------------------------------------------
# Frobenius orbit quotient on columns
# ---------------------------------------------------------------------------

"""
    frobenius_orbit_quotient(A, p, origcols, special_orig) -> (A', origcols', orbit_map)

Quotient the columns of A by the Frobenius orbit equivalence.

Two columns j, j' are Frobenius-equivalent (over GF(p)) iff their mod-p column
vectors are proportional: col_j ≡ λ·col_{j'} (mod p) for some λ ≠ 0.  This is
the projective equivalence of factor-base atoms induced by the Frobenius
endomorphism acting on D1-atoms that share the same x-coordinate support
modulo scalar twist.

For each equivalence class we keep exactly one representative column:
  - If any column in the class is special (generator/target/inf), that one wins.
  - Otherwise we pick the column with the lowest current-matrix index (most
    stable canonical choice under subsequent reindexing).

Returns:
  A'        — submatrix with one column per orbit
  origcols' — updated original-column index vector
  orbit_map — Dict mapping each dropped original column index to the kept
              representative's original column index

Raises if p is not a prime (sanity) or if special-column conflicts arise.
"""
function frobenius_orbit_quotient(A::SparseMatrixCSC{Int,Int}, p::Int,
                                  origcols::Vector{Int}, special_orig::Set{Int})
    p >= 2 || error("frobenius_orbit_quotient: field_prime must be >= 2, got $p")

    m, n = size(A)
    n == 0 && return A, origcols, Dict{Int,Int}()

    # Build canonical signature for each column: normalise by the first nonzero
    # entry so proportional columns get identical signatures.
    col_sigs = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    rv = rowvals(A)
    nzv = nonzeros(A)

    for j in 1:n
        entries = Tuple{Int,Int}[]
        for idx in nzrange(A, j)
            v = modp(nzv[idx], p)
            v == 0 && continue
            push!(entries, (rv[idx], v))
        end
        if isempty(entries)
            col_sigs[j] = entries
            continue
        end
        # Canonical: divide by first-row value so λ-multiples collapse.
        lead_v = entries[1][2]
        inv_lead = invmod(lead_v, p)
        col_sigs[j] = [(r, mod(v * inv_lead, p)) for (r, v) in entries]
    end

    # Group columns by signature.
    sig_to_cols = Dict{Vector{Tuple{Int,Int}}, Vector{Int}}()
    for j in 1:n
        push!(get!(sig_to_cols, col_sigs[j], Int[]), j)
    end

    # Singleton signatures and zero columns — nothing to do.
    # For each multi-member orbit, elect one representative.
    keep_set = Set{Int}()
    orbit_map = Dict{Int,Int}()   # dropped orig → kept orig

    for (sig, members) in sig_to_cols
        if length(members) == 1
            push!(keep_set, members[1])
            continue
        end

        # Elect representative: prefer specials, then lowest index.
        specials_in_orbit = filter(j -> origcols[j] in special_orig, members)
        if length(specials_in_orbit) > 1
            # Multiple specials with identical column vectors is degenerate but
            # not necessarily wrong (e.g. two generator slots that collapsed).
            # Keep the first special, warn.
            @warn "frobenius_orbit_quotient: orbit contains $(length(specials_in_orbit)) special columns — keeping first, dropping rest" specials_in_orbit
        end

        rep = if !isempty(specials_in_orbit)
            minimum(specials_in_orbit)   # lowest-index special
        else
            minimum(members)
        end

        push!(keep_set, rep)
        for j in members
            j == rep && continue
            orbit_map[origcols[j]] = origcols[rep]
        end
    end

    keep = sort(collect(keep_set))
    A2 = A[:, keep]
    origcols2 = origcols[keep]
    return A2, origcols2, orbit_map
end

function build_col_components(A::SparseMatrixCSC{Int,Int})
    # Union-find on columns: every row induces a clique.
    rows = row_entries(A)
    n = size(A,2)
    parent = collect(1:n)
    rank = zeros(Int, n)

    findp(x) = begin
        y = x
        while parent[y] != y
            parent[y] = parent[parent[y]]
            y = parent[y]
        end
        return y
    end

    function unite(a, b)
        pa, pb = findp(a), findp(b)
        pa == pb && return
        if rank[pa] < rank[pb]
            parent[pa] = pb
        elseif rank[pb] < rank[pa]
            parent[pb] = pa
        else
            parent[pb] = pa
            rank[pa] += 1
        end
    end

    for row in rows
        length(row) <= 1 && continue
        c0 = row[1][1]
        for (c, _) in row[2:end]
            unite(c0, c)
        end
    end

    comps = Dict{Int, Vector{Int}}()
    for c in 1:n
        p = findp(c)
        push!(get!(comps, p, Int[]), c)
    end
    return collect(values(comps))
end


function column_component_info(A::SparseMatrixCSC{Int,Int}, origcols::Vector{Int}, special_orig::Set{Int})
    comps = build_col_components(A)
    comp_id = zeros(Int, size(A, 2))
    comp_size = Dict{Int, Int}()
    comp_has_special = Dict{Int, Bool}()

    for (cid, comp) in enumerate(comps)
        comp_size[cid] = length(comp)
        has_special = false
        for c in comp
            comp_id[c] = cid
            if origcols[c] in special_orig
                has_special = true
            end
        end
        comp_has_special[cid] = has_special
    end

    return comps, comp_id, comp_size, comp_has_special
end



function candidate_columns(A::SparseMatrixCSC{Int,Int}, origcols::Vector{Int}, special_orig::Set{Int};
                            max_degree::Int=6, aggressive::Bool=false, rng=MersenneTwister(1),
                            tiny_component_max::Int=256)
    deg = col_degrees(A)
    comps, comp_id, comp_size, comp_has_special = column_component_info(A, origcols, special_orig)

    scored = Vector{Tuple{Tuple{Int,Int,Int,Int},Int}}()

    for c in 1:size(A,2)
        deg[c] == 0 && continue
        (origcols[c] in special_orig) && continue
        if !aggressive && deg[c] > max_degree
            continue
        end

        cid = comp_id[c]
        comp_penalty = comp_has_special[cid] ? 1 : 0
        # Prefer the giant core first: aggressive pruning gets more traction when
        # we stop spending all our budget on tiny satellites.
        score = (comp_penalty, -comp_size[cid], deg[c], c)
        push!(scored, (score, c))
    end

    sort!(scored, by = x -> x[1])

    cands = [c for (_, c) in scored]

    if aggressive
        buckets = Dict{Tuple{Int,Int,Int}, Vector{Int}}()
        for (score, c) in scored
            key = (score[1], score[2], score[3])
            push!(get!(buckets, key, Int[]), c)
        end
        cands = Int[]
        for key in sort(collect(keys(buckets)))
            bucket = buckets[key]
            shuffle!(rng, bucket)
            append!(cands, bucket)
        end
    end

    return cands, deg, comp_id, comp_size, comp_has_special
end

function drop_columns(A::SparseMatrixCSC{Int,Int}, cols_to_drop::Set{Int})
    keep = [j for j in 1:size(A,2) if !(j in cols_to_drop)]
    return A[:, keep], keep
end

function matrix_from_row_entries(rows::Vector{Vector{Tuple{Int,Int}}}, m::Int, n::Int)
    I = Int[]
    J = Int[]
    V = Int[]
    for i in 1:m
        for (j, v) in rows[i]
            v == 0 && continue
            push!(I, i)
            push!(J, j)
            push!(V, v)
        end
    end
    return sparse(I, J, V, m, n)
end

function current_origcol_index(origcols::Vector{Int}, orig_col::Union{Int,Nothing})
    orig_col === nothing && return nothing
    idx = findfirst(==(orig_col), origcols)
    return idx
end

function adaptive_batch_size(base_batch::Int, comp_size::Int, min_deg::Int, tiny_component_max::Int)
    # Big, dense components deserve larger batches; tiny satellites can be handled
    # with smaller, cheaper elimination steps.
    if comp_size >= max(4 * tiny_component_max, 2048) || min_deg >= 18
        return min(128, max(base_batch, 96))
    elseif comp_size >= max(2 * tiny_component_max, 1024) || min_deg >= 14
        return min(96, max(base_batch, 64))
    elseif comp_size >= tiny_component_max
        return min(64, max(base_batch, 32))
    else
        return min(32, max(8, base_batch ÷ 2))
    end
end

function rebalance_infinity_column(A::SparseMatrixCSC{Int,Int}, modulus::Int, inf_col::Int)
    1 <= inf_col <= size(A, 2) || return A

    rows = row_entries(A)
    I = Int[]
    J = Int[]
    V = Int[]
    m, n = size(A)

    for i in 1:m
        s = 0
        for (j, v) in rows[i]
            j == inf_col && continue
            v = modp(v, modulus)
            v == 0 && continue
            s = mod(s + v, modulus)
            push!(I, i)
            push!(J, j)
            push!(V, v)
        end
        new_inf = mod(-s, modulus)
        if new_inf != 0
            push!(I, i)
            push!(J, inf_col)
            push!(V, new_inf)
        end
    end

    return sparse(I, J, V, m, n)
end

function schur_eliminate_column(A::SparseMatrixCSC{Int,Int}, p::Int, col::Int,
                                origcols::Vector{Int}, special_orig::Set{Int})
    1 <= col <= size(A, 2) || return A, origcols, false, 0
    (origcols[col] in special_orig) && return A, origcols, false, 0

    rows = row_entries(A)
    touched = findall(i -> any(jv -> jv[1] == col, rows[i]), 1:length(rows))
    isempty(touched) && return A, origcols, false, 0

    protected_touch_count(i) = count(jv -> (origcols[jv[1]] in special_orig), rows[i])
    pivot_row = touched[1]
    pivot_key = (length(rows[pivot_row]), protected_touch_count(pivot_row), pivot_row)
    for i in touched[2:end]
        key = (length(rows[i]), protected_touch_count(i), i)
        key < pivot_key && (pivot_row = i; pivot_key = key)
    end

    pivot_coeff = nothing
    pivot_entries = Dict{Int,Int}()
    for (j, v) in rows[pivot_row]
        if j == col
            pivot_coeff = modp(v, p)
        end
    end
    pivot_coeff === nothing && return A, origcols, false, 0
    pivot_coeff == 0 && return A, origcols, false, 0
    inv_pivot = invmod(pivot_coeff, p)

    for (j, v) in rows[pivot_row]
        j == col && continue
        pivot_entries[j] = modp(v * inv_pivot, p)
    end

    new_rows = Vector{Vector{Tuple{Int,Int}}}(undef, 0)
    sizehint!(new_rows, length(rows) - 1)
    for i in 1:length(rows)
        i == pivot_row && continue
        rowmap = Dict{Int,Int}()
        for (j, v) in rows[i]
            j == col && continue
            jj = j > col ? j - 1 : j
            rowmap[jj] = mod(get(rowmap, jj, 0) + modp(v, p), p)
        end
        if any(jv -> jv[1] == col, rows[i])
            factor = 0
            for (j, v) in rows[i]
                j == col && (factor = modp(v, p); break)
            end
            if factor != 0
                for (j, pv) in pivot_entries
                    jj = j > col ? j - 1 : j
                    rowmap[jj] = mod(get(rowmap, jj, 0) - factor * pv, p)
                    rowmap[jj] == 0 && delete!(rowmap, jj)
                end
            end
        end
        push!(new_rows, [(Int(k), Int(v)) for (k, v) in sort!(collect(rowmap), by = first)])
    end

    newA = matrix_from_row_entries(new_rows, length(new_rows), size(A,2) - 1)
    new_origcols = [origcols[j] for j in 1:length(origcols) if j != col]
    return newA, new_origcols, true, 1
end

function schur_prune_batch(A::SparseMatrixCSC{Int,Int}, p::Int, batch::Vector{Int},
                           origcols::Vector{Int}, special_orig::Set{Int};
                           balance_mod::Union{Int,Nothing}=nothing,
                           inf_orig::Union{Int,Nothing}=nothing)
    # process from high to low indices so column positions remain valid
    workA = A
    workcols = copy(origcols)
    row_drops = 0
    cols_sorted = sort(unique(batch), rev=true)
    for c in cols_sorted
        if c < 1 || c > size(workA, 2)
            return A, origcols, false, 0
        end
        (workcols[c] in special_orig) && return A, origcols, false, 0
        workA, workcols, ok, rd = schur_eliminate_column(workA, p, c, workcols, special_orig)
        ok || return A, origcols, false, 0
        row_drops += rd
    end

    workA, _ = prune_zero_rows(workA)
    workA, _, _ = dedupe_rows_mod(workA, p)

    if balance_mod !== nothing && inf_orig !== nothing
        inf_idx = current_origcol_index(workcols, inf_orig)
        inf_idx !== nothing && (workA = rebalance_infinity_column(workA, balance_mod, inf_idx))
    end

    return workA, workcols, true, row_drops
end

function rows_touched_by_columns(A::SparseMatrixCSC{Int,Int}, cols_to_drop::Set{Int})
    m, _ = size(A)
    touched = falses(m)
    rv = rowvals(A)
    for c in cols_to_drop
        1 <= c <= size(A, 2) || continue
        for idx in nzrange(A, c)
            touched[rv[idx]] = true
        end
    end
    return findall(touched)
end

function choose_probe_rows(m::Int, n_rows::Int, rng)
    n_rows = min(n_rows, m)
    return randperm(rng, m)[1:n_rows]
end

function rank_probe(A::SparseMatrixCSC{Int,Int}, p::Int, n_rows::Int, rng; rows_sel=nothing)
    if rows_sel === nothing
        r, nullity_ub, rows_sel = sparse_rank_estimate(A, p; n_rows=n_rows, rng=rng)
    else
        rows = row_dicts(A, p, rows_sel)
        r = sparse_rank_estimate_rows(rows, p)
        nullity_ub = size(A, 2) - r
    end
    return (rank_lb=r, nullity_ub=nullity_ub, rows_sel=rows_sel)
end

function try_drop_batch(A::SparseMatrixCSC{Int,Int}, p::Int, batch::Vector{Int}, base, sample_rows::Int, rng;
                        rows_sel=nothing, rank_slop::Int=1, min_nullity_gain::Int=1,
                        protected_cols::Set{Int}=Set{Int}(), inf_col::Union{Int,Nothing}=nothing,
                        balance_mod::Union{Int,Nothing}=nothing, origcols::Vector{Int}=collect(1:size(A,2)))
    any(c -> c in protected_cols, batch) && return (false, A, origcols,
        (rank_lb=base.rank_lb, nullity_ub=base.nullity_ub, rows_sel=rows_sel), 0, 0)

    A2, keepcols, ok, row_drop_count = schur_prune_batch(A, p, batch, origcols, protected_cols;
                                                         balance_mod=balance_mod,
                                                         inf_orig=inf_col)
    ok || return (false, A, origcols,
        (rank_lb=base.rank_lb, nullity_ub=base.nullity_ub, rows_sel=rows_sel), 0, 0)

    r2, n2, _ = rank_probe(A2, p, sample_rows, rng; rows_sel=nothing)
    gain = base.nullity_ub - n2
    ok2 = (r2 + rank_slop >= base.rank_lb) && (gain >= min_nullity_gain || length(batch) == 1)
    return ok2, A2, keepcols, (rank_lb=r2, nullity_ub=n2, rows_sel=nothing), gain, row_drop_count
end


# ---------------------------------------------------------------------------
# Greedy Schur pruning — no rank gate, no inf rebalancing
# ---------------------------------------------------------------------------
"""
Greedily eliminate every non-special column in degree order via Schur
complementation.  No rank probe, no acceptance gate.  The only invariant
enforced is that special columns are never touched.  Suitable when the matrix
has a dominant giant component and you can afford to be sloppy — the DLP
kernel structure is preserved as long as the special columns survive.

Runs until no more candidates exist or max_cols_to_drop is reached.
"""
function greedy_schur_prune(A::SparseMatrixCSC{Int,Int}, p::Int,
                             origcols::Vector{Int}, special_orig::Set{Int};
                             max_degree::Int=999999,
                             max_cols_to_drop::Int=typemax(Int),
                             log_every::Int=500,
                             inf_orig::Union{Int,Nothing}=nothing,
                             balance_mod::Union{Int,Nothing}=nothing)
    dropped_orig = Int[]
    total = 0

    best_nnz = nnz(A)
    while total < max_cols_to_drop
        deg = col_degrees(A)
        n = size(A, 2)

        best_col = 0
        best_deg = max_degree + 1
        for c in 1:n
            origcols[c] in special_orig && continue
            deg[c] == 0 && continue
            deg[c] > max_degree && continue
            if deg[c] < best_deg
                best_deg = deg[c]
                best_col = c
            end
        end

        best_col == 0 && break

        # Stop if the next candidate would cause fill-in (degree > 2 means
        # eliminating it adds more entries than it removes).  This is the
        # natural nnz-minimizing stopping point.
        if best_deg > 2
            cur_nnz = nnz(A)
            if cur_nnz > best_nnz
                _log("    [greedy] nnz growing ($cur_nnz > $best_nnz) at degree $best_deg — stopping")
                break
            end
            best_nnz = min(best_nnz, cur_nnz)
        end

        push!(dropped_orig, origcols[best_col])
        A, origcols, ok, _ = schur_eliminate_column(A, p, best_col, origcols, special_orig)
        if !ok
            pop!(dropped_orig)
            continue
        end
        A, _ = prune_zero_rows(A)
        total += 1

        if total % log_every == 0
            _log("    [greedy] eliminated $total cols — matrix now $(size(A,1))×$(size(A,2))  nnz=$(nnz(A))")
        end
    end

    _log("    [greedy] done — eliminated $total cols — matrix now $(size(A,1))×$(size(A,2))  nnz=$(nnz(A))")

    # Single rebalance pass: fix the infinity column so every row sums to 0 mod ell.
    # Done once here rather than after every elimination — same result, much faster.
    if balance_mod !== nothing && inf_orig !== nothing
        inf_idx = current_origcol_index(origcols, inf_orig)
        if inf_idx !== nothing
            A = rebalance_infinity_column(A, balance_mod, inf_idx)
            _log("    [greedy] rebalanced infinity column (col $inf_idx) mod $balance_mod")
        else
            _log("    [greedy] WARNING: inf_orig not found in remaining cols — skipping rebalance")
        end
    end

    return A, dropped_orig, origcols
end

function prune_round(A::SparseMatrixCSC{Int,Int}, p::Int,
                     origcols::Vector{Int}, special_orig::Set{Int};
                     sample_rows::Int=4096, rng=MersenneTwister(1),
                     base_seed::Int=1,
                     max_degree::Int=6, budget::Int=240,
                     aggressive::Bool=true, batch_size::Int=32,
                     rank_slop::Int=1, min_nullity_gain::Int=1,
                     stop_min_degree::Int=12, tiny_component_max::Int=256,
                     inf_orig::Union{Int,Nothing}=nothing,
                     balance_mod::Union{Int,Nothing}=nothing,
                     giant_component_boost::Int=2)
    rows_sel = choose_probe_rows(size(A,1), sample_rows, rng)
    base = rank_probe(A, p, sample_rows, rng; rows_sel=rows_sel)
    _log("  baseline sampled-rank-lb=$(base.rank_lb)  sampled-nullity-ub=$(base.nullity_ub)")

    if aggressive
        _log("  aggro mode: component-aware batches; batch_size=$batch_size  budget=$budget")
    else
        _log("  candidate columns will be filtered by degree <= $max_degree; budget=$budget")
    end

    dropped_orig = Int[]
    tried = 0

    while tried < budget
        cands, deg, comp_id, comp_size, comp_has_special = candidate_columns(
            A, origcols, special_orig;
            max_degree=max_degree,
            aggressive=aggressive,
            rng=rng,
            tiny_component_max=tiny_component_max
        )

        isempty(cands) && break

        mincanddeg = minimum(deg[cands])
        giant_comp_size = maximum(values(comp_size))
        giant_comp_present = giant_comp_size > tiny_component_max
        effective_stop = giant_comp_present ? max(stop_min_degree, 24) : stop_min_degree
        if mincanddeg > effective_stop
            _log("  stopping: smallest candidate degree=$mincanddeg exceeds stop_min_degree=$effective_stop")
            break
        end

        groups = Dict{Int, Vector{Int}}()
        for c in cands
            push!(get!(groups, comp_id[c], Int[]), c)
        end

        comp_order = sort(collect(keys(groups)), by = cid -> (
            comp_has_special[cid] ? 1 : 0,
            -comp_size[cid],
            minimum(deg[groups[cid]])
        ))

        accepted = false
        for cid in comp_order
            group = groups[cid]
            sort!(group, by = c -> (deg[c], c))
            local_batch = adaptive_batch_size(batch_size, comp_size[cid], minimum(deg[group]), tiny_component_max)
            if comp_size[cid] >= giant_component_boost * tiny_component_max
                local_batch = min(128, max(local_batch, batch_size * giant_component_boost))
            end

            idx = 1
            while idx <= length(group) && tried < budget
                batch = group[idx:min(idx + local_batch - 1, length(group))]
                tried += 1

                ok, A2, keep2, score2, gain, row_drop_count = try_drop_batch(
                    A, p, batch, base, sample_rows, rng;
                    rows_sel=rows_sel,
                    rank_slop=rank_slop,
                    min_nullity_gain=min_nullity_gain,
                    protected_cols=special_orig,
                    inf_col=inf_orig,
                    balance_mod=balance_mod,
                    origcols=origcols
                )

                if ok
                    batch_orig = origcols[batch]
                    append!(dropped_orig, batch_orig)
                    _log("    drop batch=$(length(batch))  rows=$(row_drop_count)  comp=$(cid)/$(comp_size[cid])  batch_size=$local_batch  orig=[$(join(batch_orig, ", "))]  min_deg=$(minimum(deg[batch]))  rank-lb=$(score2.rank_lb)  nullity-ub=$(score2.nullity_ub)  gain=$gain")
                    A = A2
                    origcols = keep2
                    base = score2
                    rows_sel = score2.rows_sel
                    accepted = true
                    break
                elseif length(batch) > 1
                    # The two halves are independent reads on A — evaluate in parallel.
                    half = max(1, length(batch) ÷ 2)
                    subs = filter(!isempty, [batch[1:half], batch[half+1:end]])
                    sub_results = Vector{Any}(undef, length(subs))
                    Threads.@threads for si in eachindex(subs)
                        sub_results[si] = try_drop_batch(
                            A, p, subs[si], base, sample_rows,
                            MersenneTwister(base_seed + tried * 97 + si);
                            rows_sel=rows_sel,
                            rank_slop=rank_slop,
                            min_nullity_gain=min_nullity_gain,
                            protected_cols=special_orig,
                            inf_col=inf_orig,
                            balance_mod=balance_mod,
                            origcols=origcols
                        )
                    end
                    tried += length(subs)
                    for (si, sub) in enumerate(subs)
                        ok2, A3, keep3, score3, gain2, row_drop_count2 = sub_results[si]
                        if ok2
                            sub_orig = origcols[sub]
                            append!(dropped_orig, sub_orig)
                            _log("    drop subbatch=$(length(sub))  rows=$(row_drop_count2)  comp=$(cid)/$(comp_size[cid])  batch_size=$(length(sub))  orig=[$(join(sub_orig, ", "))]  min_deg=$(minimum(deg[sub]))  rank-lb=$(score3.rank_lb)  nullity-ub=$(score3.nullity_ub)  gain=$gain2")
                            A = A3
                            origcols = keep3
                            base = score3
                            rows_sel = score3.rows_sel
                            accepted = true
                            break
                        end
                    end
                    accepted && break
                end

                idx += local_batch
            end
            accepted && break
        end

        accepted || break
    end

    return A, dropped_orig, origcols
end


function prune_iteratively(A::SparseMatrixCSC{Int,Int}, p::Int, special_orig::Set{Int};
                           rounds::Int=4, sample_rows::Int=4096,
                           max_degree::Int=6, budget::Int=120, seed::Int=1,
                           aggressive::Bool=true, batch_size::Int=32,
                           rank_slop::Int=1, min_nullity_gain::Int=1,
                           stop_min_degree::Int=12, tiny_component_max::Int=256,
                           inf_orig::Union{Int,Nothing}=nothing,
                           balance_mod::Union{Int,Nothing}=nothing,
                           giant_component_boost::Int=2,
                           init_origcols::Union{Vector{Int},Nothing}=nothing)
    rng = MersenneTwister(seed)
    dropped_total = Int[]
    origcols = init_origcols !== nothing ? copy(init_origcols) : collect(1:size(A,2))
    length(origcols) == size(A,2) || error("prune_iteratively: init_origcols length $(length(init_origcols)) != ncols $(size(A,2))")
    for round in 1:rounds
        _section("PRUNE ROUND $round")
        comps = build_col_components(A)
        comp_count = length(comps)
        comp_sizes = sort([length(c) for c in comps], rev=true)
        _log("  matrix: $(size(A,1))×$(size(A,2))  components=$comp_count  nnz=$(nnz(A))")
        _log("  component sizes (top few): $(comp_sizes[1:min(end, 10)])")
        A2, dropped_orig, new_origcols = prune_round(A, p, origcols, special_orig;
                                                     sample_rows=sample_rows,
                                                     rng=rng,
                                                     base_seed=seed + round * 1000,
                                                     max_degree=max_degree,
                                                     budget=budget,
                                                     aggressive=aggressive,
                                                     batch_size=batch_size,
                                                     rank_slop=rank_slop,
                                                     min_nullity_gain=min_nullity_gain,
                                                     stop_min_degree=stop_min_degree,
                                                     tiny_component_max=tiny_component_max,
                                                     inf_orig=inf_orig,
                                                     balance_mod=balance_mod)
        append!(dropped_total, dropped_orig)
        origcols = new_origcols
        A = A2
        _log("  round $round removed $(length(dropped_orig)) columns")
        isempty(dropped_orig) && break
    end
    return A, dropped_total, origcols
end

function row_sum_residuals(A::SparseMatrixCSC{Int,Int}, modulus::Int)
    rows = row_entries(A)
    residuals = Vector{Int}(undef, length(rows))
    for i in 1:length(rows)
        s = 0
        for (_, v) in rows[i]
            s = mod(s + modp(v, modulus), modulus)
        end
        residuals[i] = s
    end
    return residuals
end

function ensure_special_kept(special_orig::Vector{Int}, current_origcols::Vector{Int})
    missing = Int[]
    curr = Set(current_origcols)
    for s in special_orig
        (s in curr) || push!(missing, s)
    end
    return missing
end


function write_output_hdf5(outpath::String, A::SparseMatrixCSC{Int,Int}, meta, origcols::Vector{Int}; dropped=nothing)
    atoms_out = meta.atoms[origcols]
    aidx_out = Dict(string(atoms_out[i]) => i - 1 for i in 1:length(atoms_out))

    function remap_original_col(orig_col)
        orig_col === nothing && return nothing
        idx = findfirst(==(orig_col), origcols)
        return idx === nothing ? nothing : idx
    end

    h5open(outpath, "w") do f
        m, n = size(A)
        row_cols = [Tuple{Int,Int}[] for _ in 1:m]
        rv = rowvals(A)
        nz = nonzeros(A)
        for j in 1:n
            for idx in nzrange(A, j)
                push!(row_cols[rv[idx]], (j, Int(nz[idx])))
            end
        end
        data = Int[]
        indices = Int[]
        indptr = Vector{Int}(undef, m + 1)
        indptr[1] = 0
        nnz_seen = 0
        for i in 1:m
            for (j, v) in row_cols[i]
                push!(data, v)
                push!(indices, j - 1)
                nnz_seen += 1
            end
            indptr[i + 1] = nnz_seen
        end
        write(f, "csr/data", data)
        write(f, "csr/indices", indices)
        write(f, "csr/indptr", indptr)
        write(f, "csr/shape", Int[m, n])

        if meta.group_order !== nothing
            write(f, "group_order", meta.group_order)
        end
        if meta.field_prime !== nothing
            write(f, "field_prime", meta.field_prime)
        end
        if meta.divisor_xs !== nothing
            write(f, "divisor_xs", meta.divisor_xs)
        end

        write(f, "atoms", atoms_out)
        write(f, "atom_index", JSON3.write(aidx_out))

        for (key, value) in (
            ("col_inf", meta.col_inf),
            ("col_gen0", meta.col_gen0),
            ("col_gen1", meta.col_gen1),
            ("col_tgt0", meta.col_tgt0),
            ("col_tgt1", meta.col_tgt1),
        )
            new_col = remap_original_col(value)
            new_col === nothing || write(f, key, new_col - 1)
        end

        if dropped !== nothing
            write(f, "dropped_columns", Int.(dropped))
        end
    end
end

function summarize(A::SparseMatrixCSC{Int,Int}, p::Int, meta, sample_rows::Int, seed::Int)
    _section("SUMMARY")
    _log("  matrix: $(size(A,1)) rows × $(size(A,2)) cols")
    _log("  nnz: $(nnz(A))")
    _log("  field prime: $(meta.field_prime === nothing ? "?" : string(meta.field_prime))")
    specs = special_cols(meta)
    _log("  special columns: $(isempty(specs) ? "none" : join(specs, ", "))")

    comps = build_col_components(A)
    comp_sizes = sort([length(c) for c in comps], rev=true)
    _log("  connected components: $(length(comps))  sizes=$(comp_sizes[1:min(length(comp_sizes),10)])")

    rank_est, nullity_est, _ = sparse_rank_estimate(A, p; n_rows=sample_rows, rng=MersenneTwister(seed))
    _log("  sampled rank estimate: $rank_est")
    _log("  sampled nullity estimate: $nullity_est")
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "input"
            help = "Input relation-matrix HDF5 file"
            arg_type = String
            required = true
        "--out"
            help = "Output HDF5 file for pruned matrix"
            arg_type = String
            default = ""
        "--sample-rows"
            help = "Number of rows to sample for rank tests"
            arg_type = Int
            default = 1024
        "--rounds"
            help = "Maximum pruning rounds"
            arg_type = Int
            default = 4
        "--max-degree"
            help = "Only columns with degree at most this value are considered"
            arg_type = Int
            default = 6
        "--budget"
            help = "Maximum candidate batches tested per round"
            arg_type = Int
            default = 240
        "--aggressive"
            help = "Enable aggressive batch pruning"
            action = :store_true
        "--greedy"
            help = "Greedy Schur pruning: eliminate all non-special columns with no rank gate (fast, for giant-component matrices)"
            action = :store_true
        "--greedy-max-degree"
            help = "Max column degree to eliminate in greedy mode (default: no limit)"
            arg_type = Int
            default = 999999
        "--batch-size"
            help = "Initial batch size for aggressive pruning"
            arg_type = Int
            default = 32
        "--rank-slop"
            help = "How much sampled rank is allowed to wobble when accepting a batch"
            arg_type = Int
            default = 1
        "--min-nullity-gain"
            help = "Minimum sampled nullity-ub gain required to accept a batch"
            arg_type = Int
            default = 1
        "--stop-min-degree"
            help = "Stop pruning once the cheapest remaining candidate exceeds this degree"
            arg_type = Int
            default = 12
        "--tiny-component-max"
            help = "Prefer eliminating columns in connected components up to this size"
            arg_type = Int
            default = 256
        "--giant-component-boost"
            help = "Multiplier for batch size once the giant component dominates"
            arg_type = Int
            default = 2
        "--seed"
            help = "Random seed"
            arg_type = Int
            default = 1
        "--group-order"
            help = "Override the group order / ell metadata"
            arg_type = Int
            default = nothing
        "--ell"
            help = "Alias for --group-order"
            arg_type = Int
            default = nothing
        "--field-prime"
            help = "Override the field prime metadata"
            arg_type = Int
            default = nothing
        "--known-key"
            help = "Optional protected x-value or key identifier"
            arg_type = Int
            default = nothing
        "--exclude-xs"
            help = "Optional x-values to protect from pruning"
            arg_type = Int
            nargs = '*'
            default = Int[]
        "--no-frob-quotient"
            help = "Disable Frobenius orbit column quotient (enabled by default; runs before Schur pruning)"
            action = :store_true
    end
    return parse_args(s)
end



function main()
    args = parse_cli()
    infile = args["input"]
    outpath = args["out"]
    sample_rows = args["sample-rows"]
    rounds = args["rounds"]
    max_degree = args["max-degree"]
    budget = args["budget"]
    aggressive = true
    greedy = args["greedy"]
    greedy_max_degree = args["greedy-max-degree"]
    batch_size = args["batch-size"]
    rank_slop = args["rank-slop"]
    min_nullity_gain = args["min-nullity-gain"]
    stop_min_degree = args["stop-min-degree"]
    tiny_component_max = args["tiny-component-max"]
    giant_component_boost = args["giant-component-boost"]
    seed = args["seed"]
    group_order_override = args["group-order"]
    ell_override = args["ell"]
    field_prime_override = args["field-prime"]
    known_key = args["known-key"]
    exclude_xs = args["exclude-xs"]
    no_frob_quotient = args["no-frob-quotient"]

    _section("LOAD")
    meta = load_matrix_hdf5(infile)
    group_order = ell_override !== nothing ? ell_override : (group_order_override === nothing ? meta.group_order : group_order_override)
    field_prime = field_prime_override === nothing ? meta.field_prime : field_prime_override
    p = field_prime === nothing ? error("field_prime is required for surgery") : field_prime
    ell = group_order === nothing ? error("group_order / ell is required for surgery") : group_order

    meta = merge(meta, (; group_order = group_order, field_prime = field_prime))

    A = to_sparse_mod(meta.M, p)
    _log("  loaded matrix: $(size(A,1))×$(size(A,2))  nnz=$(nnz(A))  mod $p")

    _section("STRUCTURAL CLEANUP")
    A, kept_rows = prune_zero_rows(A)
    _log("  after dropping zero rows: $(size(A,1))×$(size(A,2))")
    A, kept_rows2, row_sources = dedupe_rows_mod(A, p)
    _log("  after dedupe rows: $(size(A,1))×$(size(A,2))  unique rows=$(length(kept_rows2))")

    summarize(A, p, meta, sample_rows, seed)

    specs = special_cols(meta)
    special = Set(specs)

    protected_xs = Int[]
    known_key !== nothing && push!(protected_xs, known_key)
    if exclude_xs !== nothing
        append!(protected_xs, exclude_xs)
    end
    if !isempty(protected_xs)
        for (j, atm) in enumerate(meta.atoms)
            x_val = atom_x_value(string(atm))
            x_val === nothing && continue
            x_val in protected_xs && push!(special, j)
        end
    end

    # -----------------------------------------------------------------------
    # FROBENIUS ORBIT QUOTIENT
    # Run before Schur pruning: reduces column count for free (O(n log n),
    # no rank probes) so Schur elimination starts from a smaller matrix.
    # -----------------------------------------------------------------------
    origcols = collect(1:size(A,2))
    if !no_frob_quotient
        _section("FROBENIUS ORBIT QUOTIENT")
        pre_frob_cols = size(A, 2)
        A, origcols, orbit_map = frobenius_orbit_quotient(A, p, origcols, special)
        # Remap special set to new origcols indices (origcols already tracks
        # original column numbers; special set is in original-column space so
        # it stays valid — just log for sanity).
        n_merged = length(orbit_map)
        _log("  columns before quotient: $pre_frob_cols")
        _log("  columns after  quotient: $(size(A,2))")
        _log("  merged (dropped) cols:   $n_merged")
        if n_merged > 0
            sample_pairs = collect(orbit_map)[1:min(6, n_merged)]
            _log("  sample orbit collapses (dropped_orig → kept_orig): " *
                 join(["$k→$v" for (k,v) in sample_pairs], ", ") *
                 (n_merged > 6 ? ", …" : ""))
        end
        # Prune newly-zero rows introduced by column removal.
        A, kept_rows_frob = prune_zero_rows(A)
        _log("  rows after zero-row prune: $(size(A,1))")
    else
        _log("  [Frobenius orbit quotient disabled via --no-frob-quotient]")
        orbit_map = Dict{Int,Int}()
    end

    _section("PRUNE")
    if greedy
        _log("  mode: GREEDY SCHUR (no rank gate; single rebalance pass at end)")
        _log("  max_degree filter: " * (greedy_max_degree >= 999999 ? "none" : string(greedy_max_degree)))
        origcols_init = copy(origcols)
        A2, dropped, keepcols = greedy_schur_prune(A, p, origcols_init, special;
                                                   max_degree=greedy_max_degree,
                                                   log_every=500,
                                                   inf_orig=meta.col_inf,
                                                   balance_mod=ell)
    else
        A2, dropped, keepcols = prune_iteratively(A, p, special;
                                                  rounds=rounds,
                                                  sample_rows=sample_rows,
                                                  max_degree=max_degree,
                                                  budget=budget,
                                                  seed=seed,
                                                  aggressive=aggressive,
                                                  batch_size=batch_size,
                                                  rank_slop=rank_slop,
                                                  min_nullity_gain=min_nullity_gain,
                                                  stop_min_degree=stop_min_degree,
                                                  tiny_component_max=tiny_component_max,
                                                  inf_orig=meta.col_inf,
                                                  balance_mod=ell,
                                                  giant_component_boost=giant_component_boost,
                                                  init_origcols=origcols)
    end
    _log("  total dropped columns: $(length(dropped))")
    _log("  final matrix: $(size(A2,1))×$(size(A2,2))  nnz=$(nnz(A2))")

    missing_special = ensure_special_kept(specs, keepcols)
    if !isempty(missing_special)
        _log("  WARNING: special columns missing after pruning: $(join(missing_special, ", "))")
    else
        _log("  all designated special columns survived")
    end

    summarize(A2, p, meta, sample_rows, seed + 17)

    residuals = row_sum_residuals(A2, ell)
    bad = findall(!=(0), residuals)
    if isempty(bad)
        _log("  row-sum check: all rows balanced mod $ell")
    else
        _log("  row-sum check: $(length(bad)) rows imbalanced; first few=$(bad[1:min(end, 10)])")
        error("row-sum invariant failed after pruning")
    end

    # -----------------------------------------------------------------------
    # ORPHAN COMPONENT DROP
    # After greedy/iterative pruning the support graph often has singleton or
    # small disconnected components: atoms that appeared in too few relations
    # to stay connected once high-degree columns were eliminated.  These
    # contribute one null vector each to the kernel (isolated directions) and
    # carry no information about the DLP.  Drop every component that:
    #   (a) is not the largest (main) component, AND
    #   (b) contains no special columns.
    # Special-bearing small components are left alone and a warning is emitted.
    # -----------------------------------------------------------------------
    _section("ORPHAN COMPONENT DROP")
    comps2 = build_col_components(A2)
    comp_sizes2 = sort([length(c) for c in comps2], rev=true)
    _log("  components before drop: $(length(comps2))  sizes=$(comp_sizes2[1:min(end,12)])")

    if length(comps2) > 1
        special_set_specs = Set(specs)   # original col indices that are special
        main_idx = argmax(length.(comps2))
        main_comp_cols = Set(comps2[main_idx])

        orphan_cols = Set{Int}()   # current-matrix column indices to drop
        orphan_special_comps = 0

        for (ci, comp) in enumerate(comps2)
            ci == main_idx && continue
            # check for specials (keepcols[c] is original col index)
            has_spec = any(c -> keepcols[c] in special_set_specs, comp)
            if has_spec
                orphan_special_comps += 1
                _log("  WARNING: small component of size $(length(comp)) contains a special column — not dropped")
            else
                union!(orphan_cols, comp)
            end
        end

        orphan_special_comps > 0 && _log("  WARNING: $orphan_special_comps special-bearing non-main component(s) retained")

        if !isempty(orphan_cols)
            orphan_orig = [keepcols[c] for c in sort(collect(orphan_cols))]
            A2, keepcols = drop_columns(A2, orphan_cols)
            A2, _ = prune_zero_rows(A2)
            append!(dropped, orphan_orig)
            _log("  dropped $(length(orphan_cols)) orphan column(s) from $(length(comps2)-1) non-main component(s)")
            _log("  matrix after orphan drop: $(size(A2,1))×$(size(A2,2))  nnz=$(nnz(A2))")

            comps3 = build_col_components(A2)
            _log("  components after drop: $(length(comps3))")

            # Re-check specials survived.
            missing2 = ensure_special_kept(specs, keepcols)
            if !isempty(missing2)
                error("orphan drop accidentally removed special columns: $(join(missing2, ", "))")
            end
        else
            _log("  no orphan columns to drop (all non-main components contain specials)")
        end
    else
        _log("  single component — nothing to drop")
    end

    if !isempty(outpath)
        _section("WRITE")
        # Record both Schur-dropped, Frobenius-orbit-merged, and orphan-dropped columns.
        all_dropped = vcat(dropped, collect(keys(orbit_map)))
        write_output_hdf5(outpath, A2, meta, keepcols; dropped=all_dropped)
        _log("  wrote pruned matrix to: $outpath")
    end
end

main()
