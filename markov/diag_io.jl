# diag_io.jl
# HDF5 matrix loader, matrix/row helpers, and special-column inference
# from divisor_xs stored in the HDF5 dump.
#
# Exports (informally):
#   load_matrix_hdf5, drop_rows, remap_col,
#   infer_special_cols_from_divisor_xs
#
# Depends on: diag_bootstrap.jl, diag_poly_mumford.jl
#   (tonelli_shanks and eval_poly_mod used in infer_special_cols_from_divisor_xs)
function load_matrix_hdf5(path::String)
    isfile(path) || throw(ErrorException("file not found: $path"))

    return h5open(path, "r") do f
        # --- 1. Load Atoms and Index ---
        atoms_raw = read(f["atoms"])
        atoms = isa(atoms_raw[1], AbstractString) ? collect(atoms_raw) :
                [String(a) for a in atoms_raw]
        
        aidx_raw = read(f["atom_index"])
        aidx_str = isa(aidx_raw, AbstractString) ? aidx_raw : String(aidx_raw)
        aidx = Dict{String,Int}(string(k) => (v + 1) for (k,v) in JSON3.read(aidx_str))

        # --- 2. Load Matrix ---
        # Initialize M in the do-block scope so it's guaranteed to be defined
        M = if haskey(f, "matrix_dense")
            # Python/numpy writes HDF5 row-major; Julia reads it column-major,
            # so the on-disk (nrows×ncols) array arrives transposed. Correct it here.
            Matrix(transpose(Int.(read(f["matrix_dense"]))))
        else
            data_vals = Int.(read(f["csr/data"]))
            indices   = Int.(read(f["csr/indices"])) .+ 1
            indptr    = Int.(read(f["csr/indptr"]))
            shape     = Tuple(Int.(read(f["csr/shape"])))
            nr, nc    = shape

            # Build a sparse matrix directly from CSR triplets.
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

        # --- 3. Load Metadata ---
        group_order = haskey(f, "group_order") ? Int(read(f["group_order"])) : nothing
        field_prime = haskey(f, "field_prime")  ? Int(read(f["field_prime"])) : nothing
        # divisor_xs may now be stored as a flat int array [x0,y0,x1,y1,...] or a legacy
        # plain x-only array.  We keep it as-is here; infer_special_cols_from_divisor_xs
        # handles both forms via aidx lookup by "(x, y)" key.
        divisor_xs  = haskey(f, "divisor_xs")   ? Int.(read(f["divisor_xs"])) : nothing

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
# Helpers
# ---------------------------------------------------------------------------
function drop_rows(M::AbstractMatrix{Int}, rows_to_drop::AbstractSet{Int})
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

"""
Infer special column indices from divisor_xs stored in the HDF5.

Handles two on-disk formats:
  - Interleaved xy (len=8): [x0,y0,x1,y1,x2,y2,x3,y3]  — preferred, written by
    updated dump_matrix_hdf5 when coeffs/p are available.
  - Bare-x (len=4): [x0,x1,x2,x3]  — legacy format.

For bare-x format with (x,y)-keyed aidx, we attempt to recover y by scanning
aidx for any key of the form "(x_val, *)".  If curve_coeffs and field_prime are
supplied we also compute y directly via Tonelli-Shanks.

Returns Dict{String,Union{Nothing,Int}} mapping "gen0/gen1/tgt0/tgt1" -> col.
"""
function infer_special_cols_from_divisor_xs(aidx, divisor_xs;
                                             curve_coeffs=nothing, field_prime=nothing)
    divisor_xs === nothing && return Dict{String,Union{Nothing,Int}}()
    labels = ["gen0", "gen1", "tgt0", "tgt1"]
    inferred = Dict{String,Union{Nothing,Int}}()

    # Build a map x_val -> column index by scanning all aidx keys that look like
    # "(x, y)" strings.  Used as fallback when we know x but not y.
    x_to_col = Dict{Int, Int}()
    for (key, col) in aidx
        m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", key)
        if m !== nothing
            x_val = parse(Int, m.captures[1])
            # Keep first encountered (canonical y branch is the smaller one,
            # but either column is the right atom for this x).
            if !haskey(x_to_col, x_val)
                x_to_col[x_val] = col
            end
        end
    end

    # Determine whether we have interleaved xy (len==8) or bare-x (len==4).
    if length(divisor_xs) >= 8
        # Interleaved xy format.
        n_pairs = length(divisor_xs) ÷ 2
        for (idx, lab) in enumerate(labels)
            idx > n_pairs && break
            x_val = divisor_xs[2*idx - 1]
            y_val = divisor_xs[2*idx]
            key_xy = "($(x_val), $(y_val))"
            col = get(aidx, key_xy, nothing)
            if col === nothing
                # Try the other branch: -y mod p.
                if field_prime !== nothing
                    y2 = mod(-y_val, field_prime)
                    col = get(aidx, "($(x_val), $(y2))", nothing)
                end
            end
            # Final fallback: bare-x key or x_to_col scan.
            col === nothing && (col = get(aidx, string(x_val), nothing))
            col === nothing && (col = get(x_to_col, x_val, nothing))
            inferred[lab] = col
        end
    else
        # Bare-x format (len == 4).  Try to recover y.
        n_xs = min(length(divisor_xs), 4)
        for (idx, lab) in enumerate(labels)
            idx > n_xs && break
            x_val = Int(divisor_xs[idx])
            col = nothing

            # 1. Direct bare-x key (legacy atoms).
            col = get(aidx, string(x_val), nothing)

            # 2. Scan x_to_col built from "(x, y)" keys.
            col === nothing && (col = get(x_to_col, x_val, nothing))

            # 3. Tonelli-Shanks from stored curve coefficients.
            if col === nothing && curve_coeffs !== nothing && field_prime !== nothing
                p = field_prime
                y = tonelli_shanks(eval_poly_mod(curve_coeffs, x_val, p), p)
                if y !== nothing
                    col = get(aidx, "($(x_val), $(y))", nothing)
                    if col === nothing
                        col = get(aidx, "($(x_val), $(mod(-y, p)))", nothing)
                    end
                end
            end

            inferred[lab] = col
        end
    end
    return inferred
end
