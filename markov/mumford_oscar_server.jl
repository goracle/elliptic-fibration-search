"""
mumford_oscar_server.jl

Persistent subprocess server: loops reading one newline-delimited JSON task from
stdin per iteration, writes one newline-delimited JSON result to stdout, then
loops again.  Started once by the Python bridge and kept alive for the duration
of the run — Julia/Oscar JIT cost is paid exactly once.

Protocol (newline-delimited JSON, one object per line)
------------------------------------------------------
  stdin:  {"prime_list": [p, ...],
            "tasks":     {str(p): [[v_tuple, diff_coeffs, rhs_idx], ...], ...},
            "rhs":       {str(p): [[num_coeffs, den_coeffs], ...], ...},
            "section_polys": {str(p): [{"p":…,"D":…,"X":…,"Y":…,"Z":…,"a4":…,"a6":…}, …], ...}}
            # section_polys is optional; omitted when a4/a6 have constant denominators

  stdout (no section_polys):
    {str(p): {str(v_tuple): {str([rhs_idx]): [m_root, ...], ...}, ...}, ...}

  stdout (with section_polys):
    {str(p): {"roots":         {str(v_tuple): {str([rhs_idx]): [m_root, ...]}},
              "ladder_caches": [{str(k): {"X":[…],"Y":[…],"Z":[…]}}, …]}, ...}
    # ladder_caches is a Vector, one entry per section (null if serialisation failed).

  stderr: debug/error messages (drained by Python stderr thread)

  The walker only needs m_root values — x_val, v0/v1, Mumford pairs, and sign
  computation have all been removed.  enrich_candidates in walkerclass.py
  reconstructs xj/xk and signs directly from the fiber geometry.

Exit codes: 0 = clean EOF, 1 = unhandled error.
"""

using Oscar
using Nemo
using Base.Threads: @threads, nthreads
using JSON3

include(joinpath(@__DIR__, "section_ladder.jl"))

# Signal Python that Oscar has finished loading and we are ready for tasks.
# Python blocks on stderr waiting for this line before sending anything.
# Flush stdout first to drain any blank lines emitted to stdout during
# Oscar/JSON3 module loading — otherwise they sit in the buffer and get
# flushed ahead of the first real JSON response, causing readline() on the
# Python side to consume a blank line instead of the JSON payload.
flush(stdout)
println(stderr, "[julia] server ready")
flush(stderr)

# ---------------------------------------------------------------------------
# Root-finding and RHS eval
# ---------------------------------------------------------------------------

function roots_over_fp(coeffs_lohi::Vector{<:Integer}, p::Int)::Vector{Int}
    isempty(coeffs_lohi) && throw(ArgumentError("roots_over_fp: empty coefficient list"))
    Fp = GF(p)
    Fpm, _ = polynomial_ring(Fp, :m)
    f = Fpm([Fp(Int(c) % p) for c in coeffs_lohi])
    iszero(f) && return collect(0:p-1)
    rs = roots(f)   # Vector{FqFieldElem} — no keyword, works on all Nemo versions
    return [Int(lift(ZZ, r)) for r in rs]
end



function eval_rhs_at_m(
    num_coeffs::Vector{<:Integer},
    den_coeffs::Vector{<:Integer},
    m_root::Int,
    p::Int,
)::Union{Int, Nothing}
    function horner(coeffs, x, mod)
        acc = 0
        for c in Iterators.reverse(coeffs)
            acc = (acc * x + Int(c)) % mod
        end
        return acc
    end
    num_val = horner(num_coeffs, m_root, p)
    den_val = horner(den_coeffs, m_root, p)
    iszero(den_val) && return nothing
    return (num_val * powermod(den_val, p - 2, p)) % p
end

# ---------------------------------------------------------------------------
# Per-prime computation — threaded over chunk_items ([n]P section multiples)
# ---------------------------------------------------------------------------

function residues_for_prime(p, chunk_items, rhs_reconstruction)
    n_items = length(chunk_items)

    n_slots = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : nthreads()
    thread_results = [Dict() for _ in 1:n_slots]
    thread_caches  = [Dict() for _ in 1:n_slots]

    @threads for idx in 1:n_items
        tid   = Threads.threadid()
        item  = chunk_items[idx]
        local_result = thread_results[tid]
        local_cache  = thread_caches[tid]

        v_tuple     = item[1]
        diff_coeffs = Int.(item[2])
        rhs_idx     = Int(item[3])

        coeff_key = [c % p for c in diff_coeffs]
        all(iszero, coeff_key) && continue

        if !haskey(local_cache, coeff_key)
            local_cache[coeff_key] = roots_over_fp(coeff_key, p)
        end
        m_roots = local_cache[coeff_key]
        isempty(m_roots) && continue

        rhs_entry  = rhs_reconstruction[rhs_idx + 1]  # 1-indexed
        num_coeffs = Int.(rhs_entry[1])
        den_coeffs = Int.(rhs_entry[2])

        v_key  = string(v_tuple)
        xr_key = string([rhs_idx])   # key is just rhs_idx now; x_val is not needed

        for m_root in m_roots
            x_val = eval_rhs_at_m(num_coeffs, den_coeffs, m_root, p)
            x_val === nothing && continue

            if !haskey(local_result, v_key)
                local_result[v_key] = Dict()
            end
            if !haskey(local_result[v_key], xr_key)
                local_result[v_key][xr_key] = []
            end
            push!(local_result[v_key][xr_key], m_root)
        end
    end

    # Merge per-thread results
    result = Dict()
    for tr in thread_results
        for (v_key, xmap) in tr
            if !haskey(result, v_key)
                result[v_key] = Dict()
            end
            for (xr_key, m_roots) in xmap
                if !haskey(result[v_key], xr_key)
                    result[v_key][xr_key] = []
                end
                append!(result[v_key][xr_key], m_roots)
            end
        end
    end

    return result
end

# ---------------------------------------------------------------------------
# Section ladder dispatch
# ---------------------------------------------------------------------------

"""
    run_ladder_for_prime(p_payloads, max_k) -> Vector{Union{Dict, Nothing}}

Run run_section_ladder for each section payload in p_payloads.
Returns a Vector, one entry per section: the ladder Dict, or nothing on error.
max_k: upper bound for the ladder (should match MAXN / the largest |k| needed).
"""
function run_ladder_for_prime(p_payloads, max_k::Int)::Vector{Any}
    out = Vector{Any}(undef, length(p_payloads))
    for (i, pl) in enumerate(p_payloads)
        if pl === nothing || ismissing(pl)
            out[i] = nothing
            continue
        end
        try
            lc = run_section_ladder(
                Int(pl["p"]),
                Int(pl["D"]),
                Vector{Int}(pl["X"]),
                Vector{Int}(pl["Y"]),
                Vector{Int}(pl["Z"]),
                Vector{Int}(pl["a4"]),
                Vector{Int}(pl["a6"]),
                max_k,
            )
            out[i] = lc
        catch e
            msg = sprint(showerror, e)
            println(stderr, "[julia] section_ladder error (section $i): $msg")
            flush(stderr)
            out[i] = nothing
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Main: persistent read-compute-write loop
# ---------------------------------------------------------------------------

# max_k for the ladder — matches Python MAXN (typically 80 or 200).
# Can be overridden by setting JULIA_LADDER_MAXK env var before starting server.
const LADDER_MAXK = let v = get(ENV, "JULIA_LADDER_MAXK", "200")
    parse(Int, v)
end

function main()
    println(stderr, "[julia] main() entered, waiting for tasks")
    println(stderr, "[julia] LADDER_MAXK=$LADDER_MAXK")
    flush(stderr)
    task_n = 0
    while !eof(stdin)
        line = readline(stdin)
        isempty(strip(line)) && continue
        task_n += 1

        println(stderr, "[julia] task $task_n: received $(length(line)) bytes")
        flush(stderr)

        try
            task = JSON3.read(line)
            println(stderr, "[julia] task $task_n: JSON parsed ok")
            flush(stderr)

            prime_list    = Int.(task["prime_list"])
            tasks         = task["tasks"]
            rhs           = task["rhs"]
            # section_polys is optional — absent in non-FF mode or old clients
            section_polys = get(task, "section_polys", nothing)
            has_sections  = section_polys !== nothing && !isempty(section_polys)

            println(stderr, "[julia] task $task_n: prime_list=$(prime_list), nthreads=$(nthreads()), has_sections=$has_sections")
            flush(stderr)

            result = Dict{String, Any}()
            for p in prime_list
                n_items = length(tasks[string(p)])
                n_rhs   = length(rhs[string(p)])
                println(stderr, "[julia] task $task_n: p=$p  items=$n_items  rhs_entries=$n_rhs")
                flush(stderr)

                roots_result = residues_for_prime(p, tasks[string(p)], rhs[string(p)])
                println(stderr, "[julia] task $task_n: p=$p  residues_for_prime done")
                flush(stderr)

                if has_sections
                    p_payloads = get(section_polys, string(p), [])
                    if !isempty(p_payloads)
                        ladder_caches = run_ladder_for_prime(p_payloads, LADDER_MAXK)
                        println(stderr, "[julia] task $task_n: p=$p  ladder done ($(length(ladder_caches)) sections)")
                        flush(stderr)
                        result[string(p)] = Dict(
                            "roots"          => roots_result,
                            "ladder_caches"  => ladder_caches,
                        )
                    else
                        # section_polys present but empty for this prime — flat format
                        result[string(p)] = roots_result
                    end
                else
                    # Old format: flat roots dict (backward-compatible)
                    result[string(p)] = roots_result
                end

                println(stderr, "[julia] task $task_n: p=$p  done")
                flush(stderr)
            end

            println(stderr, "[julia] task $task_n: serialising result")
            flush(stderr)
            out = JSON3.write(result)
            println(stderr, "[julia] task $task_n: writing $(length(out)) bytes to stdout")
            flush(stderr)
            println(stdout, out)
            flush(stdout)
            println(stderr, "[julia] task $task_n: done")
            flush(stderr)
        catch e
            msg = sprint(showerror, e, catch_backtrace())
            println(stderr, "[julia] ERROR in task $task_n: $msg")
            flush(stderr)
            println(stdout, "[julia] ERROR")
            flush(stdout)
            continue
        end
    end
    println(stderr, "[julia] stdin EOF, exiting cleanly after $task_n tasks")
    flush(stderr)
end

main()
