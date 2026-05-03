# diag_poly_mumford.jl
# Polynomial arithmetic over GF(p) and Mumford divisor arithmetic for genus-2
# hyperelliptic curves y^2 = f(x).
#
# Exports (informally):
#   eval_poly_mod, polymul_mod, polyadd_mod, polysub_mod, polydivrem_mod,
#   polygcd_mod, poly_extgcd_mod,
#   mumford_compose, mumford_reduce, mumford_add_point,
#   tonelli_shanks, is_fully_split,
#   reduce_row_mumford, load_curve_coeffs, apply_mumford_reduce_filter
#
# Depends on: diag_bootstrap.jl
# ---------------------------------------------------------------------------
# Mumford arithmetic over GF(p) for genus-2 hyperelliptic curves
# y^2 = f(x),  deg(f) = 5
# ---------------------------------------------------------------------------

"""
Default curve coefficients for y^2 = x^5 + 3x^3 + 2x^2 + 5x + 4.
Stored as [c0, c1, c2, c3, c4, c5] where f(x) = Σ cᵢ xⁱ (ascending degree).
"""
const DEFAULT_CURVE_COEFFS = [4, 5, 2, 3, 0, 1]  # constant term first

"""
Evaluate f(x) mod p given coefficients in ascending-degree order.
"""
function eval_poly_mod(coeffs::Vector{Int}, x::Int, p::Int)::Int
    result = 0
    xpow   = 1
    for c in coeffs
        result = mod(result + c * xpow, p)
        xpow   = mod(xpow * x, p)
    end
    return result
end

"""
Polynomial multiplication mod p.  Both inputs are coefficient vectors
(ascending degree).  Returns coefficient vector of length len(a)+len(b)-1.
"""
function polymul_mod(a::Vector{Int}, b::Vector{Int}, p::Int)::Vector{Int}
    (isempty(a) || isempty(b)) && return Int[]
    out = zeros(Int, length(a) + length(b) - 1)
    for (i, ca) in enumerate(a), (j, cb) in enumerate(b)
        out[i+j-1] = mod(out[i+j-1] + ca * cb, p)
    end
    return out
end

"""
Polynomial addition mod p (ascending degree, zero-padded).
"""
function polyadd_mod(a::Vector{Int}, b::Vector{Int}, p::Int)::Vector{Int}
    n = max(length(a), length(b))
    out = zeros(Int, n)
    for (i, c) in enumerate(a); out[i] = mod(out[i] + c, p); end
    for (i, c) in enumerate(b); out[i] = mod(out[i] + c, p); end
    return out
end

"""
Polynomial subtraction mod p (ascending degree, zero-padded).
"""
function polysub_mod(a::Vector{Int}, b::Vector{Int}, p::Int)::Vector{Int}
    n = max(length(a), length(b))
    out = zeros(Int, n)
    for (i, c) in enumerate(a); out[i] = mod(out[i] + c, p); end
    for (i, c) in enumerate(b); out[i] = mod(out[i] - c, p); end
    return out
end

"""
Polynomial division mod p: returns (quotient, remainder), ascending degree.
Throws on division by zero poly.
"""
function polydivrem_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    isempty(b) && throw(ArgumentError("polydivrem_mod: divisor is empty"))
    # strip trailing zeros from b
    b_deg = findlast(!=(0), b)
    b_deg === nothing && throw(ArgumentError("polydivrem_mod: divisor is zero polynomial"))
    b = b[1:b_deg]

    deg_b = b_deg - 1
    inv_lead_b = invmod(b[end], p)

    r = copy(a)
    # strip trailing zeros from r
    while length(r) > 0 && r[end] == 0; pop!(r); end

    q = Int[]
    while length(r) > 0 && length(r) - 1 >= deg_b
        deg_r  = length(r) - 1
        coeff  = mod(r[end] * inv_lead_b, p)
        # prepend coeff to quotient (will reverse at end)
        pushfirst!(q, coeff)
        # subtract coeff * x^(deg_r - deg_b) * b from r
        shift = deg_r - deg_b
        for (i, bc) in enumerate(b)
            idx = i + shift
            r[idx] = mod(r[idx] - coeff * bc, p)
        end
        # strip trailing zeros
        while length(r) > 0 && r[end] == 0; pop!(r); end
    end

    # pad quotient to correct length
    expected_q_len = max(0, length(a) - deg_b)
    while length(q) < expected_q_len; push!(q, 0); end

    isempty(r) && push!(r, 0)
    return q, r
end

"""
Polynomial GCD mod p using Euclidean algorithm.  Returns monic GCD.
"""
function polygcd_mod(a::Vector{Int}, b::Vector{Int}, p::Int)::Vector{Int}
    # strip trailing zeros
    strip(v) = begin
        w = copy(v)
        while length(w) > 1 && w[end] == 0; pop!(w); end
        w
    end
    a, b = strip(a), strip(b)
    while !(length(b) == 1 && b[1] == 0)
        _, r = polydivrem_mod(a, b, p)
        a, b = b, strip(r)
    end
    # make monic
    a = strip(a)
    if !isempty(a) && a[end] != 0 && a[end] != 1
        inv_lc = invmod(a[end], p)
        a = mod.(a .* inv_lc, p)
    end
    return a
end

"""
One step of Cantor's algorithm: compose two semi-reduced Mumford divisors
(u1,v1) and (u2,v2) on y^2 = f(x) over GF(p).
Returns a (possibly non-reduced) Mumford pair (u, v) with deg(u) = deg(u1)+deg(u2).
All polynomials are coefficient vectors (ascending degree).

Algorithm (Cohen–Frey §14.1 / Cantor 1987):
  d1, e1, e2  s.t.  e1*u1 + e2*u2 = d1 = gcd(u1,u2)
  d,  c1, c2  s.t.  c1*d1 + c2*(v1+v2) = d = gcd(d1, v1+v2)
  s1 = c1*e1,  s2 = c1*e2,  s3 = c2
  u  = u1*u2 / d^2
  v  = (s1*u1*v2 + s2*u2*v1 + s3*(v1*v2 + f)) / d   mod u
"""
function mumford_compose(u1::Vector{Int}, v1::Vector{Int},
                         u2::Vector{Int}, v2::Vector{Int},
                         f_coeffs::Vector{Int}, p::Int)
    # Step 1
    d1, e1, e2 = poly_extgcd_mod(u1, u2, p)

    # Step 2
    v_diff     = polysub_mod(v1, v2, p)
    d, c1, c2  = poly_extgcd_mod(d1, v_diff, p)

    # s1 = c1*e1,  s2 = c1*e2
    s1 = polymul_mod(c1, e1, p)
    s2 = polymul_mod(c1, e2, p)
    s3 = c2

    # u = u1*u2 / d^2
    u1u2     = polymul_mod(u1, u2, p)
    d2       = polymul_mod(d, d, p)
    u, rem_u = polydivrem_mod(u1u2, d2, p)
    all(x == 0 for x in rem_u) || throw(ErrorException("mumford_compose: d^2 does not divide u1*u2"))
    # make u monic
    if !isempty(u) && u[end] != 0 && u[end] != 1
        inv_lc = invmod(u[end], p)
        u = mod.(u .* inv_lc, p)
    end

    # v_num = s1*u1*v2 + s2*u2*v1 + s3*(v1*v2 - f)
    t1    = polymul_mod(s1, polymul_mod(u1, v2, p), p)
    t2    = polymul_mod(s2, polymul_mod(u2, v1, p), p)
    v1v2  = polymul_mod(v1, v2, p)
    fv    = polysub_mod(v1v2, f_coeffs, p)
    t3    = polymul_mod(s3, fv, p)
    v_num = polyadd_mod(polyadd_mod(t1, t2, p), t3, p)

    # v = v_num / d   mod u
    v_quot, rem_v = polydivrem_mod(v_num, d, p)
    all(x == 0 for x in rem_v) || throw(ErrorException("mumford_compose: d does not divide v_num"))
    _, v = polydivrem_mod(v_quot, u, p)

    return u, v
end

"""
Extended GCD for polynomials over GF(p).
Returns (g, s, t) such that s*a + t*b = g (g monic).
"""
function poly_extgcd_mod(a::Vector{Int}, b::Vector{Int}, p::Int)
    strip(v) = begin
        w = copy(v)
        while length(w) > 1 && w[end] == 0; pop!(w); end
        w
    end
    make_monic(v) = begin
        v = strip(v)
        (isempty(v) || v[end] == 0 || v[end] == 1) && return v, 1
        lc = v[end]
        inv_lc = invmod(lc, p)
        return mod.(v .* inv_lc, p), inv_lc
    end

    a, b = strip(a), strip(b)
    # trivial cases
    is_zero(v) = all(x == 0 for x in v)
    if is_zero(a)
        g, sc = make_monic(b)
        return g, [0], [mod(sc, p)]
    end
    if is_zero(b)
        g, sc = make_monic(a)
        return g, [mod(sc, p)], [0]
    end

    old_r, r    = copy(a), copy(b)
    old_s, s    = [1], [0]
    old_t, t    = [0], [1]

    while !is_zero(strip(r))
        q, rem = polydivrem_mod(old_r, r, p)
        old_r, r = r, strip(rem)
        new_s = polysub_mod(old_s, polymul_mod(q, s, p), p)
        old_s, s = s, new_s
        new_t = polysub_mod(old_t, polymul_mod(q, t, p), p)
        old_t, t = t, new_t
    end

    g = strip(old_r)
    s_out, t_out = old_s, old_t
    # make g monic
    if !isempty(g) && g[end] != 0 && g[end] != 1
        inv_lc = invmod(g[end], p)
        g     = mod.(g .* inv_lc, p)
        s_out = mod.(s_out .* inv_lc, p)
        t_out = mod.(t_out .* inv_lc, p)
    end
    return g, s_out, t_out
end

"""
Mumford reduction step: given (u, v) with deg(u) > g=2,
reduce to a semi-reduced divisor of degree ≤ 2.
Returns (u_red, v_red).
"""
function mumford_reduce(u::Vector{Int}, v::Vector{Int},
                        f_coeffs::Vector{Int}, p::Int)
    strip(w) = begin z = copy(w); while length(z)>1 && z[end]==0; pop!(z); end; z end
    g = 2
    u, v = strip(u), strip(v)
    while length(u) - 1 > g   # deg(u) > g
        # u' = (f - v^2) / u
        v2    = polymul_mod(v, v, p)
        fmv2  = polysub_mod(f_coeffs, v2, p)
        u2, rem = polydivrem_mod(fmv2, u, p)
        all(x == 0 for x in rem) || throw(ErrorException("mumford_reduce: (f-v^2)/u not exact; divisor not on curve"))
        u2 = strip(u2)
        # make u2 monic
        if !isempty(u2) && u2[end] != 0 && u2[end] != 1
            inv_lc = invmod(u2[end], p)
            u2 = mod.(u2 .* inv_lc, p)
        end
        # v' = (-v) mod u2
        _, v2r = polydivrem_mod(mod.(Int.(p) .- v, p), u2, p)
        u, v = u2, strip(v2r)
    end
    return u, v
end


"""
Add a single degree-1 divisor point (x=a, y=y_a) to a Mumford
divisor (u, v) over GF(p).  The caller supplies the exact y-coordinate.
Returns the new (u, v) after one Cantor composition + reduction step.
"""
function mumford_add_point(u::Vector{Int}, v::Vector{Int},
                           a::Int, y_a::Int, f_coeffs::Vector{Int}, p::Int)
    y_a == 0 && throw(ErrorException("mumford_add_point: point (a=$a) is a Weierstrass point (y=0)"))

    # degree-1 Mumford divisor: u2 = x - a, v2 = y_a (constant)
    u2 = [mod(-a, p), 1]   # x - a  (ascending: [const, x^1])
    v2 = [mod(y_a, p)]

    u_new, v_new = mumford_compose(u, v, u2, v2, f_coeffs, p)
    u_new, v_new = mumford_reduce(u_new, v_new, f_coeffs, p)
    check_mumford_invariant(u_new, v_new, f_coeffs, p)
    return u_new, v_new
end

"""
Tonelli-Shanks modular square root.  Returns r s.t. r^2 ≡ n (mod p), or nothing.
"""
function tonelli_shanks(n::Int, p::Int)::Union{Int,Nothing}
    n = mod(n, p)
    n == 0 && return 0
    # Euler criterion
    powermod(n, (p-1)÷2, p) == 1 || return nothing
    p == 2 && return n
    # factor out 2s from p-1: p-1 = q * 2^s, q odd
    q, s = p - 1, 0
    while q % 2 == 0; q ÷= 2; s += 1; end
    s == 1 && return powermod(n, (p+1)÷4, p)
    # find quadratic non-residue z
    z = 2
    while powermod(z, (p-1)÷2, p) != p - 1; z += 1; end
    m  = s
    c  = powermod(z, q, p)
    t  = powermod(n, q, p)
    r  = powermod(n, (q+1)÷2, p)
    while true
        t == 1 && return r
        # find least i s.t. t^(2^i) = 1
        i, tmp = 1, mod(t * t, p)
        while tmp != 1; tmp = mod(tmp * tmp, p); i += 1; end
        # b = c^(2^(m-i-1)) by repeated squaring
        b = c
        for _ in 1:(m-i-1); b = mod(b * b, p); end
        m  = i
        c  = mod(b * b, p)
        t  = mod(t * c, p)
        r  = mod(r * b, p)
    end
end

"""
Check whether a monic polynomial u(x) of degree ≤ 2 splits completely over GF(p).
- deg 0 or deg 1: trivially split.
- deg 2: splits iff discriminant is a QR (or zero).
Returns true iff fully split.
"""
function is_fully_split(u::Vector{Int}, p::Int)::Bool
    strip(w) = begin z = copy(w); while length(z)>1 && z[end]==0; pop!(z); end; z end
    u = strip(u)
    deg = length(u) - 1
    deg <= 1 && return true
    deg == 2 || throw(ArgumentError("is_fully_split: expected deg ≤ 2, got $deg"))
    # u = x^2 + bx + c  (monic)  =>  disc = b^2 - 4c
    b    = u[2]   # coefficient of x^1
    c    = u[1]   # constant term
    disc = mod(b * b - 4 * c, p)
    disc == 0 && return true
    return powermod(disc, (p-1)÷2, p) == 1
end

"""
Apply Mumford reduction to a single row of the relation matrix.
Returns (is_split::Bool, u_reduced::Vector{Int}) or throws on error.

atom_xys maps col index -> (x, y) pair.  The y-coordinate is used directly,
so no sign brute-force is needed.
f_coeffs is the curve polynomial in ascending degree.
"""

function reduce_row_mumford(atom_xys::Dict{Int,Tuple{Int,Int}}, row_support::Vector{Tuple{Int,Int}},
                             f_coeffs::Vector{Int}, p::Int)
    # Expand the row into (x, y) pairs with multiplicity, ignoring ∞.
    xys = Tuple{Int,Int}[]
    for (col, coeff) in row_support
        coeff == 0 && continue
        xy = get(atom_xys, col, nothing)
        xy === nothing && throw(ErrorException("reduce_row_mumford: col $col has no (x,y) entry"))
        x_val, y_val = xy
        # Negative coefficient means involution (negate y).
        actual_y = coeff > 0 ? y_val : mod(-y_val, p)
        for _ in 1:abs(coeff)
            push!(xys, (x_val, actual_y))
        end
    end

    isempty(xys) && return true, [1]

    # Validate each point lies on the curve.
    for (x_val, y_val) in xys
        fx = eval_poly_mod(f_coeffs, x_val, p)
        mod(y_val * y_val - fx, p) == 0 || return false, Int[]
        y_val == 0 && return false, Int[]  # Weierstrass point — not supported
    end

    # Compose all points into a Mumford divisor using the stored y-coordinates.
    try
        u = [1]
        v = [0]
        for (x_val, y_val) in xys
            u, v = mumford_add_point(u, v, x_val, y_val, f_coeffs, p)
        end
        u, v = mumford_reduce(u, v, f_coeffs, p)
        check_mumford_invariant(u, v, f_coeffs, p)
        split = is_fully_split(u, p)
        return split, u
    catch
        return false, Int[]
    end
end

"""
Load curve coefficients from HDF5 if present (key "curve_coeffs"),
otherwise return the hardcoded default for y^2 = x^5 + 3x^3 + 2x^2 + 5x + 4.
"""
function load_curve_coeffs(h5_path::String)::Vector{Int}
    h5open(h5_path, "r") do f
        haskey(f, "curve_coeffs") || return DEFAULT_CURVE_COEFFS
        raw = read(f["curve_coeffs"])
        return Int.(raw)
    end
end

"""
Mumford-reduction filter: for each row of M, compose all finite atoms
(+y branch, all positive coefficients) into a Mumford divisor, reduce,
and keep the row iff u(x) splits completely over GF(p).

Returns (M_filtered::AbstractMatrix{Int}, n_kept::Int, n_dropped::Int).
Logs progress and yield statistics.
"""
function apply_mumford_reduce_filter(M::AbstractMatrix{Int}, atoms::Vector,
                                     col_inf::Union{Int,Nothing},
                                     curve_coeffs::Vector{Int}, p::Int)
    _section("MUMFORD REDUCTION FILTER  (--reduce)")
    _log("  curve: y^2 = f(x),  f coeffs (asc degree) = $curve_coeffs")
    _log("  field: GF($p)  |  matrix: $(size(M,1)) rows × $(size(M,2)) cols")

    # Build map: col -> (x, y) pair (atom name is "(x, y)" string; legacy bare-x also accepted)
    atom_xys = Dict{Int, Tuple{Int,Int}}()
    for (j, atm) in enumerate(atoms)
        j == col_inf && continue
        s = string(atm)
        # Try "(x, y)" format first
        m = match(r"^\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", s)
        if m !== nothing
            atom_xys[j] = (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
        else
            # Legacy: bare integer x — compute canonical y branch
            x_val = tryparse(Int, s)
            x_val === nothing && throw(ErrorException("apply_mumford_reduce_filter: atom "$s" (col $j) is not a valid (x,y) or bare-x atom"))
            fx = eval_poly_mod(curve_coeffs, x_val, p)
            y_canon = tonelli_shanks(fx, p)
            y_canon === nothing && throw(ErrorException("apply_mumford_reduce_filter: legacy atom x=$x_val has no square root mod $p"))
            atom_xys[j] = (x_val, min(y_canon, mod(-y_canon, p)))
        end
    end

    # Sanity-test: find any atom s.t. f(x) is a QR and verify we can add it twice
    let test_xy = nothing
        for (_, xy) in atom_xys
            x_val, y_val = xy
            fx = eval_poly_mod(curve_coeffs, x_val, p)
            if fx != 0 && powermod(fx, (p-1)÷2, p) == 1
                test_xy = xy; break
            end
        end
        if test_xy !== nothing
            x_t, y_t = test_xy
            try
                u1, v1 = mumford_add_point([1], [0], x_t, y_t, curve_coeffs, p)
                u2, v2 = mumford_add_point(u1, v1, x_t, y_t, curve_coeffs, p)
                _log("  [sanity] Mumford self-test OK: 2*P at ($x_t,$y_t) → u=$(u2)")
            catch e
                throw(ErrorException("Mumford arithmetic self-test failed at ($x_t,$y_t): $e"))
            end
        else
            _log("  [sanity] no QR atom found for self-test (all atoms may be non-QR?)")
        end
    end

    nr, nc = size(M)
    keep_flags   = falses(nr)
    nchunks      = max(1, min(Threads.nthreads(), nr))
    n_error_t    = zeros(Int, nchunks)
    n_split_t    = zeros(Int, nchunks)
    n_nonsplit_t = zeros(Int, nchunks)
    sample_errors = Vector{Tuple{Int,String}}()
    sample_lock   = ReentrantLock()

    if nchunks == 1 || nr < 256
        for i in 1:nr
            support = Tuple{Int,Int}[]
            for j in 1:nc
                (col_inf !== nothing && j == col_inf) && continue
                M[i,j] != 0 && push!(support, (j, M[i,j]))
            end

            if isempty(support)
                n_split_t[1] += 1
                keep_flags[i] = true
                continue
            end

            try
                split, u_red = reduce_row_mumford(atom_xys, support, curve_coeffs, p)
                if split
                    n_split_t[1] += 1
                    keep_flags[i] = true
                else
                    n_nonsplit_t[1] += 1
                end
            catch e
                n_error_t[1] += 1
                if length(sample_errors) < 5
                    push!(sample_errors, (i, sprint(showerror, e)))
                end
            end
        end
    else
        row_chunks = collect(Iterators.partition(1:nr, cld(nr, nchunks)))
        Threads.@threads for chunk_idx in 1:length(row_chunks)
            rows = row_chunks[chunk_idx]
            for i in rows
                support = Tuple{Int,Int}[]
                for j in 1:nc
                    (col_inf !== nothing && j == col_inf) && continue
                    M[i,j] != 0 && push!(support, (j, M[i,j]))
                end

                if isempty(support)
                    n_split_t[chunk_idx] += 1
                    keep_flags[i] = true
                    continue
                end

                try
                    split, u_red = reduce_row_mumford(atom_xys, support, curve_coeffs, p)
                    if split
                        n_split_t[chunk_idx] += 1
                        keep_flags[i] = true
                    else
                        n_nonsplit_t[chunk_idx] += 1
                    end
                catch e
                    n_error_t[chunk_idx] += 1
                    if length(sample_errors) < 5
                        lock(sample_lock) do
                            if length(sample_errors) < 5
                                push!(sample_errors, (i, sprint(showerror, e)))
                            end
                        end
                    end
                end
            end
        end
    end

    n_split    = sum(n_split_t)
    n_nonsplit = sum(n_nonsplit_t)
    n_error    = sum(n_error_t)
    total_processed = n_split + n_nonsplit + n_error
    pct_kept = nr > 0 ? @sprintf("%.1f%%", 100.0 * n_split / nr) : "N/A"
    _log("  processed: $total_processed / $nr rows")
    _log("  split (kept) : $n_split  ($pct_kept)")
    _log("  non-split    : $n_nonsplit")
    _log("  errors (drop): $n_error")

    keep_rows = findall(identity, keep_flags)
    isempty(keep_rows) && _log("  ⚠  no rows survived Mumford reduction filter — proceeding with empty matrix")

    M_out = isempty(keep_rows) ? Matrix{Int}(undef, 0, nc) : M[keep_rows, :]
    _log("  matrix after filter: $(size(M_out,1)) rows × $(size(M_out,2)) cols")
    return M_out, n_split, n_nonsplit
end
