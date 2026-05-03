# diag_utils.jl
# Logging helpers and row-deduplication over GF(p).
# Depends on: diag_bootstrap.jl
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
Returns (keep_rows::Vector{Int}, row_sources::Vector{Vector{Int}}).

This keeps only the first representative row of each equivalence class and
avoids materializing another dense copy of the matrix.
"""
function dedupe_rows_mod(M::AbstractMatrix{Int}, modulus::Int; keep_zero_rows=false)
    modulus === nothing && throw(ArgumentError("modulus is required"))
    nr, nc = size(M)
    seen        = Dict{Vector{Tuple{Int,Int}}, Int}()
    keep_rows   = Int[]
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
                seen[sig] = length(keep_rows) + 1
                push!(keep_rows, i)
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
            seen[sig] = length(keep_rows) + 1
            push!(keep_rows, i)
            push!(row_sources, [i])
        else
            push!(row_sources[seen[sig]], i)
        end
    end

    return keep_rows, row_sources
end

