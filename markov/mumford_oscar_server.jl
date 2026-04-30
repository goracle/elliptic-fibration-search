"""
mumford_oscar_server.jl

Persistent subprocess server: loops reading one newline-delimited JSON task from
stdin per iteration, writes one newline-delimited JSON result to stdout, then
loops again.  Started once by the Python bridge and kept alive for the duration
of the run — Julia/Oscar JIT cost is paid exactly once.

Protocol (newline-delimited JSON, one object per line)
------------------------------------------------------
  NEW path (section_polys + vecs present):
    stdin:  {"prime_list": [p, ...],
              "section_polys": {str(p): [{"p":…,"D":…,"X":…,"Y":…,"Z":…,"a4":…,"a6":…}, …], ...},
              "vecs":          {str(p): [[v_tuple, v_coeffs], ...], ...},
              "rhs":           {str(p): [[num_coeffs, den_coeffs], ...], ...}}

    Julia runs the section ladder, accumulates Pm = Σ v[i]*[k_i]Pi entirely in
    polynomial projective arithmetic, computes diff_poly = Pm.X*rhs_den - Pm.Z*rhs_num,
    finds roots.  No Python Sage arithmetic involved at all.

  OLD path (tasks present, no section_polys):
    stdin:  {"prime_list": [p, ...],
              "tasks": {str(p): [[v_tuple, diff_coeffs, rhs_idx], ...], ...},
              "rhs":   {str(p): [[num_coeffs, den_coeffs], ...], ...}}

  stdout (both paths):
    {str(p): {str(v_tuple): {str([rhs_idx]): [m_root, ...], ...}, ...}, ...}

  stderr: debug/error messages (drained by Python stderr thread)

Exit codes: 0 = clean EOF, 1 = unhandled error.
"""

using Oscar
using Nemo
using Base.Threads: @threads, nthreads
using JSON3

include(joinpath(@__DIR__, "section_ladder.jl"))

flush(stdout)
println(stderr, "[julia] server ready")
flush(stderr)

# ---------------------------------------------------------------------------
# Root-finding and RHS eval (unchanged)
# ---------------------------------------------------------------------------

function roots_over_fp(coeffs_lohi::Vector{<:Integer}, p::Int)::Vector{Int}
    isempty(coeffs_lohi) && throw(ArgumentError("roots_over_fp: empty coefficient list"))
    Fp = GF(p)
    Fpm, _ = polynomial_ring(Fp, :m)
    f = Fpm([Fp(Int(c) % p) for c in coeffs_lohi])
    iszero(f) && return collect(0:p-1)
    rs = roots(f)
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
# OLD path: per-prime root-finding from pre-computed diff_coeffs
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

        rhs_entry  = rhs_reconstruction[rhs_idx + 1]
        num_coeffs = Int.(rhs_entry[1])
        den_coeffs = Int.(rhs_entry[2])

        v_key  = string(v_tuple)
        xr_key = string([rhs_idx])

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

    result = Dict()
    for tr in thread_results
        for (v_key, xmap) in tr
            if !haskey(result, v_key); result[v_key] = Dict(); end
            for (xr_key, m_roots) in xmap
                if !haskey(result[v_key], xr_key); result[v_key][xr_key] = []; end
                append!(result[v_key][xr_key], m_roots)
            end
        end
    end
    return result
end

# ---------------------------------------------------------------------------
# NEW path: section ladder + accumulation + diff poly + roots, all in Julia
# ---------------------------------------------------------------------------

const LADDER_MAXK = let v = get(ENV, "JULIA_LADDER_MAXK", "1600")
    parse(Int, v)
end

"""
    residues_for_prime_sections(p, section_payloads, vecs_with_tuples, rhs_reconstruction)

For one prime p:
  1. Run section_ladder for each section to get all [k]P as PolyPt polynomial arrays.
  2. For each (v_tuple, v_coeffs): accumulate Pm = Σ v[i]*[|k_i|]Pi (negated when k<0).
  3. For each RHS: compute diff_poly = Pm.X * rhs_den - Pm.Z * rhs_num.
  4. Find roots of diff_poly over GF(p).
  5. Return {str(v_tuple): {str([rhs_idx]): [m_root, ...]}}.

No Sage, no Python point arithmetic involved.
"""
function residues_for_prime_sections(p, section_payloads, vecs_with_tuples, rhs_reconstruction)
    # Early exits
    if isempty(section_payloads) || isempty(vecs_with_tuples)
        return Dict{String,Any}()
    end

    # --- FULL materialization of JSON inputs (CRITICAL) ---
    vecs_with_tuples_mat = [
        (collect(Int, item[1]), collect(Int, item[2]))
        for item in vecs_with_tuples
    ]

    Base.@assert !(vecs_with_tuples_mat isa JSON3.Array)
    # --- base curve data ---
    D  = Int(section_payloads[1]["D"])
    a4 = _pad_to(collect(Int, section_payloads[1]["a4"]), D)
    a6 = _pad_to(collect(Int, section_payloads[1]["a6"]), D)
    rhs_padded_mat = [
        (_pad_to(collect(Int, rhs_reconstruction[i][1]), D),
        _pad_to(collect(Int, rhs_reconstruction[i][2]), D))
        for i in eachindex(rhs_reconstruction)
    ]


    # --- build ladders (fully materialized, no lazy JSON arrays) ---
    n_sec = length(section_payloads)
    ladders = Vector{Dict{String,Any}}(undef, n_sec)

    for i in 1:n_sec
        pl = section_payloads[i]

        ladders[i] = run_section_ladder(
            p, D,
            collect(Int, pl["X"]),
            collect(Int, pl["Y"]),
            collect(Int, pl["Z"]),
            a4, a6,
            LADDER_MAXK,
        )
    end

    # --- RHS padding (fully materialized) ---
    rhs_padded = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, length(rhs_reconstruction))
    for i in eachindex(rhs_reconstruction)
        num = _pad_to(collect(Int, rhs_reconstruction[i][1]), D)
        den = _pad_to(collect(Int, rhs_reconstruction[i][2]), D)
        rhs_padded[i] = (num, den)
    end

    # --- thread-local outputs ---
    #nT = Threads.nthreads()
    #thread_results = [Dict{String,Any}() for _ in 1:nT]
    nT = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()
    thread_results = [Dict{String,Any}() for _ in 1:nT]

    Threads.@threads :static for idx in eachindex(vecs_with_tuples_mat)
        tid = Threads.threadid()
        local_result = thread_results[tid]

        #item = vecs_with_tuples[idx]
        (v_tuple, v_coeffs) = vecs_with_tuples_mat[idx]
        # --- fully copy inputs (NO views / SubArray) ---
        #v_tuple  = collect(Int, item[1])
        #v_coeffs = collect(Int, item[2])

        # --- accumulate point ---
        Pm = _identity(a4, a6, p, D)
        skip = false

        for i in eachindex(v_coeffs)
            k = v_coeffs[i]
            if k == 0
                continue
            end

            if i > n_sec
                skip = true
                break
            end

            ladder = ladders[i]
            key = string(abs(k))

            if !haskey(ladder, key)
                skip = true
                break
            end

            entry = ladder[key]

            # --- HARD validation: force materialization ---
            try
                X = _pad_to(collect(Int, entry["X"]), D)
                Y = _pad_to(collect(Int, entry["Y"]), D)
                Z = _pad_to(collect(Int, entry["Z"]), D)

                Pk = PolyPt(X, Y, Z, a4, a6, p, D)
                Pm = _add(Pm, k < 0 ? _neg(Pk) : Pk)

            catch err
                @error "ladder entry corrupt" i=i key=key err
                skip = true
                break
            end
        end

        if skip || _is_id(Pm)
            continue
        end

        v_key = string(v_tuple)

        # --- RHS loop ---
        for rhs_idx in eachindex(rhs_padded)
            (rhs_num, rhs_den) = rhs_padded[rhs_idx]

            diff_poly = _psub(
                _pmul(Pm.X, rhs_den, p, D),
                _pmul(Pm.Z, rhs_num, p, D),
                p, D,
            )

            if all(iszero, diff_poly)
                continue
            end

            m_roots = roots_over_fp(diff_poly, p)
            if isempty(m_roots)
                continue
            end

            xr_key = string([rhs_idx - 1])

            # --- safe dict writes ---
            if !haskey(local_result, v_key)
                local_result[v_key] = Dict{String,Any}()
            end
            if !haskey(local_result[v_key], xr_key)
                local_result[v_key][xr_key] = Int[]
            end

            append!(local_result[v_key][xr_key], m_roots)
        end
    end

    # --- merge ---
    result = Dict{String,Any}()
    for tr in thread_results
        for (v_key, xmap) in tr
            if !haskey(result, v_key)
                result[v_key] = Dict{String,Any}()
            end
            for (xr_key, roots) in xmap
                if !haskey(result[v_key], xr_key)
                    result[v_key][xr_key] = Int[]
                end
                append!(result[v_key][xr_key], roots)
            end
        end
    end

    return result
end

# ---------------------------------------------------------------------------
# Main: persistent read-compute-write loop
# ---------------------------------------------------------------------------

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
            rhs           = task["rhs"]
            section_polys = get(task, "section_polys", nothing)
            vecs          = get(task, "vecs", nothing)
            has_sections  = section_polys !== nothing && !isempty(section_polys) &&
                            vecs !== nothing && !isempty(vecs)

            println(stderr, "[julia] task $task_n: prime_list=$(prime_list), nthreads=$(nthreads()), has_sections=$has_sections")
            flush(stderr)

            result = Dict{String,Any}()

            for p in prime_list
                p_str = string(p)
                n_rhs = length(rhs[p_str])

                if has_sections
                    p_payloads = get(section_polys, p_str, nothing)
                    p_vecs     = get(vecs, p_str, nothing)
                    if p_payloads !== nothing && !isempty(p_payloads) &&
                       p_vecs     !== nothing && !isempty(p_vecs)

                        println(stderr, "[julia] task $task_n: p=$p  vecs=$(length(p_vecs))  rhs_entries=$n_rhs  sections=$(length(p_payloads))  [new path]")
                        flush(stderr)

                        result[p_str] = residues_for_prime_sections(
                            p, p_payloads, p_vecs, rhs[p_str]
                        )
                        println(stderr, "[julia] task $task_n: p=$p  done")
                        flush(stderr)
                        continue
                    end
                end

                # Old path: pre-computed diff_coeffs in tasks
                tasks = task["tasks"]
                n_items = length(tasks[p_str])
                println(stderr, "[julia] task $task_n: p=$p  items=$n_items  rhs_entries=$n_rhs  [old path]")
                flush(stderr)

                result[p_str] = residues_for_prime(p, tasks[p_str], rhs[p_str])
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
