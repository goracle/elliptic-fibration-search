"""
mumford_oscar.jl

Oscar.jl port of the residue-computation core of run_mumford_search(markov_mode=True).

What this replaces
------------------
Only the hot inner loop of mumford_precompute_residues_parallel, specifically
the per-prime task loop in _solve_worker_wrapper:

  for each (v_tuple, diff_poly_coeffs, rhs_idx) in chunk:
    1. find roots of diff_poly over GF(p)          <- Oscar/FLINT (this file)
    2. evaluate rhs rational function at each root  <- modular arithmetic (this file)
    3. call solve_mumford_mod_p_optimized(...)       <- stays in Python

Everything else (walker, relation matrix, WalkState, build_mumford_equations_from_fibration,
prepare_modular_data_lll, solve_mumford_mod_p_optimized, verify_mumford_pair) remains
in Python unchanged.

Python integration
------------------
Drop-in replacement:  see mumford_oscar_bridge.py.
Call pattern:

    from juliacall import Main as jl
    jl.include("mumford_oscar.jl")
    # build tasks_by_prime, rhs_by_prime from the existing task-generation loop
    raw = jl.mumford_residues_oscar(prime_list, tasks_by_prime, rhs_by_prime)
    # raw[(p, v_tuple, (x_val, rhs_idx))] = [(m_root, x_val, rhs_idx), ...]
    # Python then calls solve_mumford_mod_p_optimized per (p, x_val) and assembles
    # the verified_sols list as before.

Dependencies (Project.toml):
  [deps]
  Oscar = "..."
  Nemo  = "..."
"""

using Oscar
using Nemo
using Base.Threads: @threads, nthreads

# ---------------------------------------------------------------------------
# 1.  Fiber collision detection  (detect_fiber_collision in ll_utilities.py)
# ---------------------------------------------------------------------------

"""
    fiber_collision(delta_coeffs, p) -> (Bool, poly)

Detect if Δ(m) has repeated roots mod p by computing gcd(Δ, Δ').
delta_coeffs: low-to-high integer coefficient list of Δ(m) over ZZ.
"""
function fiber_collision(delta_coeffs::Vector{<:Integer}, p::Int)
    Fp = GF(p)
    Fpm, _ = polynomial_ring(Fp, :m)

    delta_modp = Fpm([Fp(Int(c) % p) for c in delta_coeffs])
    ddelta = derivative(delta_modp)

    g = gcd(delta_modp, ddelta)
    return degree(g) > 1, g
end

# ---------------------------------------------------------------------------
# 2.  Polynomial roots over GF(p)   (find_poly_roots_fp_python equivalent)
# ---------------------------------------------------------------------------

"""
    roots_over_fp(coeffs_lohi, p) -> Vector{Int}

Find all roots of f(m) = Σ coeffs_lohi[i] * m^(i-1) in GF(p).
Uses Oscar/FLINT's root-finding (same backend as Sage).
coeffs_lohi: low-to-high integer coefficient list, already reduced mod p.
Returns plain Vector{Int} of roots in [0, p).

Raises on bad input.
"""
function roots_over_fp(coeffs_lohi::Vector{<:Integer}, p::Int)::Vector{Int}
    isempty(coeffs_lohi) && throw(ArgumentError("roots_over_fp: empty coefficient list"))
    p < 3 && throw(ArgumentError("roots_over_fp: p must be an odd prime, got $p"))

    Fp = GF(p)
    Fpm, _ = polynomial_ring(Fp, :m)

    f = Fpm([Fp(Int(c) % p) for c in coeffs_lohi])
    iszero(f) && return collect(0:p-1)

    return [Int(lift(ZZ, r)) for (r, _) in roots(f)]
end

# ---------------------------------------------------------------------------
# 3.  RHS rational function evaluation at a root
#     Mirrors the Horner loops in _solve_worker_wrapper lines 550-560.
# ---------------------------------------------------------------------------

"""
    eval_rhs_at_m(num_coeffs, den_coeffs, m_root, p) -> Union{Int, Nothing}

Evaluate (num/den)(m_root) mod p.
Returns x_val ∈ [0, p) or nothing if denominator ≡ 0 mod p.
"""
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
# 4.  ChunkItem — mirrors the (v_tuple, diff_coeffs_list, rhs_idx) triples
#     that mumford_precompute_residues_parallel puts into tasks.
# ---------------------------------------------------------------------------

struct ChunkItem
    v_tuple::Tuple
    diff_coeffs::Vector{Int}    # low-to-high, already reduced mod p by caller
    rhs_idx::Int
end

# ---------------------------------------------------------------------------
# 5.  Core per-prime computation
#     Replaces the inner body of the _solve_worker_wrapper loop (lines 521-624).
#     Does NOT call solve_mumford_mod_p_optimized — returns (m_root, x_val, rhs_idx)
#     triples so Python can do that call and assemble verified_sols as before.
# ---------------------------------------------------------------------------

"""
    residues_for_prime(p, chunk_items, rhs_reconstruction)
    -> Dict{Tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}}

For one prime p, run the root-finding + rhs-evaluation loop over all chunk_items.

Return structure mirrors the p_results dict built in _solve_worker_wrapper:
  v_tuple => { (x_val, rhs_idx) => [(m_root, x_val, rhs_idx), ...] }

The Python bridge then iterates over this, calls solve_mumford_mod_p_optimized
per (x_val, const_val), runs verify_mumford_pair, computes x_val_sign, and
assembles the final verified_sols exactly as the original worker does.

Raises on arithmetic/argument errors (fail-fast, mirrors Python assert behaviour).
"""
function residues_for_prime(
    p::Int,
    chunk_items::Vector{ChunkItem},
    rhs_reconstruction::Vector{Tuple{Vector{Int}, Vector{Int}}},
)::Dict{Tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}}

    p > 2 || throw(ArgumentError("p must be odd prime, got $p"))
    isempty(chunk_items) && throw(ArgumentError("empty chunk_items for p=$p"))
    isempty(rhs_reconstruction) && throw(ArgumentError("empty rhs_reconstruction for p=$p"))

    result = Dict{Tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}}()
    roots_cache = Dict{Vector{Int}, Vector{Int}}()

    for item in chunk_items
        v_tuple     = item.v_tuple
        diff_coeffs = item.diff_coeffs
        rhs_idx     = item.rhs_idx

        0 <= rhs_idx < length(rhs_reconstruction) ||
            throw(ArgumentError("rhs_idx=$rhs_idx out of range (len=$(length(rhs_reconstruction))) for p=$p v=$v_tuple"))

        coeff_key = [c % p for c in diff_coeffs]
        all(iszero, coeff_key) && continue

        # Root-finding with caching (same as roots_cache in _solve_worker_wrapper)
        if !haskey(roots_cache, coeff_key)
            roots_cache[coeff_key] = roots_over_fp(coeff_key, p)
        end
        m_roots = roots_cache[coeff_key]
        isempty(m_roots) && continue

        num_coeffs, den_coeffs = rhs_reconstruction[rhs_idx + 1]  # Julia is 1-indexed

        for m_root in m_roots
            x_val = eval_rhs_at_m(num_coeffs, den_coeffs, m_root, p)
            x_val === nothing && continue   # denominator zero mod p, skip

            key = (x_val, rhs_idx)
            vmap = get!(result, v_tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}())
            push!(get!(vmap, key, Tuple{Int,Int,Int}[]), (m_root, x_val, rhs_idx))
        end
    end

    return result
end

# ---------------------------------------------------------------------------
# 6.  Parallel dispatch over primes  (@threads, one thread per prime)
# ---------------------------------------------------------------------------

"""
    mumford_residues_oscar(prime_list, tasks_by_prime, rhs_by_prime)
    -> Dict{Int, Dict{Tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}}}

Thread-parallel wrapper over residues_for_prime.

Arguments (plain Julia types; juliacall passes Python dicts/lists transparently):
  prime_list     :: Vector{Int}
  tasks_by_prime :: Dict{Int, Vector{ChunkItem}}   (built by Python bridge)
  rhs_by_prime   :: Dict{Int, Vector{Tuple{Vector{Int}, Vector{Int}}}}
                    (the rhs_reconstruction list built per-prime in
                     mumford_precompute_residues_parallel, passed through unchanged)

Raises if any prime is missing from either dict.
"""
function mumford_residues_oscar(
    prime_list::Vector{Int},
    tasks_by_prime::Dict{Int, Vector{ChunkItem}},
    rhs_by_prime::Dict{Int, Vector{Tuple{Vector{Int}, Vector{Int}}}},
)::Dict{Int, Dict{Tuple, Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int,Int}}}}}

    isempty(prime_list) && throw(ArgumentError("empty prime_list"))

    n   = length(prime_list)
    out = Vector{Any}(undef, n)    # preallocate; filled by threads

    @threads for i in 1:n
        p = prime_list[i]
        haskey(tasks_by_prime, p) || throw(KeyError("tasks_by_prime missing p=$p"))
        haskey(rhs_by_prime, p)   || throw(KeyError("rhs_by_prime missing p=$p"))

        out[i] = p => residues_for_prime(p, tasks_by_prime[p], rhs_by_prime[p])
    end

    return Dict{Int, Any}(p => vmap for (p, vmap) in out)
end

# ---------------------------------------------------------------------------
# 7.  Build helper: convert Python task triples into ChunkItems
#     Called from the Python bridge once per prime after the existing
#     task-generation loop runs.
# ---------------------------------------------------------------------------

"""
    build_chunk_items(raw_items) -> Vector{ChunkItem}

Convert a list of (v_tuple, diff_coeffs, rhs_idx) triples
(as produced by the task-generation loop in mumford_precompute_residues_parallel)
into a Vector{ChunkItem} suitable for residues_for_prime.
"""
function build_chunk_items(raw_items)::Vector{ChunkItem}
    isempty(raw_items) && throw(ArgumentError("build_chunk_items: empty raw_items"))
    out = ChunkItem[]
    sizehint!(out, length(raw_items))
    for item in raw_items
        push!(out, ChunkItem(Tuple(item[1]), Int.(item[2]), Int(item[3])))
    end
    return out
end
