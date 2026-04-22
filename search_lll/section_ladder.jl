# section_ladder.jl
#
# Projective scalar-multiplication ladder for  E: y² = x³ + a4(m)·x + a6(m)
# over GF(p)[m], using plain polynomial arrays (Vector{Int}).
#
# No GCD normalisation ever occurs — coordinates remain projective numerators
# throughout.  This is the Julia-side replacement for the slow Sage GF(p)(m)
# fallback in ll_utilities.compute_all_mults_for_section.
#
# Public function (called from mumford_oscar_server.jl):
#
#   run_section_ladder(p, D, X_in, Y_in, Z_in, a4_in, a6_in, max_k)
#       -> Dict{String, Any}
#
#   Returns  { "0" => {"X"=>[…],"Y"=>[…],"Z"=>[…]},
#               "1" => …, …, string(max_k) => … }
#   Ready to be JSON-serialised and threaded back to Python as "ladder_cache".
#
# Wire convention: int coefficient lists, low-degree first, length D+1.

# ---------------------------------------------------------------------------
# Polynomial helpers — all arithmetic mod p, truncated to degree ≤ D.
# ---------------------------------------------------------------------------

function _pmul(a::Vector{Int}, b::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    for (i, ca) in enumerate(a)
        ca == 0 && continue
        for (j, cb) in enumerate(b)
            idx = i + j - 1
            idx > D + 1 && break
            out[idx] = (out[idx] + ca * cb) % p
        end
    end
    return out
end

function _padd(a::Vector{Int}, b::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    na = min(length(a), D + 1)
    nb = min(length(b), D + 1)
    for i in 1:na; out[i]  = a[i] % p; end
    for i in 1:nb; out[i]  = (out[i] + b[i]) % p; end
    return out
end

function _psub(a::Vector{Int}, b::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    na = min(length(a), D + 1)
    nb = min(length(b), D + 1)
    for i in 1:na; out[i]  = a[i] % p; end
    for i in 1:nb; out[i]  = (out[i] - b[i] + p) % p; end
    return out
end

function _pneg(a::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    for i in 1:min(length(a), D + 1)
        out[i] = (p - a[i] % p) % p
    end
    return out
end

function _pscalar(s::Int, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    out[1] = s % p
    return out
end

_pzero(D::Int) = zeros(Int, D + 1)
function _pone(D::Int)::Vector{Int}
    v = zeros(Int, D + 1); v[1] = 1; return v
end
_is_zero_poly(v::Vector{Int}) = all(==(0), v)

function _pad_to(v::Vector{Int}, D::Int)::Vector{Int}
    n = length(v)
    n == D + 1 && return v
    n > D + 1  && return v[1:D+1]
    return vcat(v, zeros(Int, D + 1 - n))
end

# ---------------------------------------------------------------------------
# Projective point — mirrors Python _PolyPoint exactly
# ---------------------------------------------------------------------------

struct PolyPt
    X::Vector{Int}
    Y::Vector{Int}
    Z::Vector{Int}
    a4::Vector{Int}
    a6::Vector{Int}
    p::Int
    D::Int
end

_identity(a4, a6, p, D) = PolyPt(_pzero(D), _pone(D), _pzero(D), a4, a6, p, D)
_is_id(pt::PolyPt) = _is_zero_poly(pt.Z)

# Negation
_neg(pt::PolyPt) = PolyPt(pt.X, _pneg(pt.Y, pt.p, pt.D), pt.Z, pt.a4, pt.a6, pt.p, pt.D)

# Doubling — mirrors _PolyPoint._double
function _double(pt::PolyPt)::PolyPt
    p, D = pt.p, pt.D
    X1, Y1, Z1, a4 = pt.X, pt.Y, pt.Z, pt.a4
    mul(a,b) = _pmul(a, b, p, D)
    add(a,b) = _padd(a, b, p, D)
    sub(a,b) = _psub(a, b, p, D)
    sc(n)    = _pscalar(n, p, D)

    W  = add(mul(sc(3), mul(X1, X1)), mul(a4, mul(Z1, Z1)))
    S  = mul(Y1, Z1)
    B  = mul(mul(X1, Y1), S)
    H  = sub(mul(W, W), mul(sc(8), B))
    X3 = mul(mul(sc(2), H), S)
    S2 = mul(S, S)
    Y3 = sub(mul(W, sub(mul(sc(4), B), H)),
             mul(sc(8), mul(mul(Y1, Y1), S2)))
    Z3 = mul(sc(8), mul(S, S2))
    return PolyPt(X3, Y3, Z3, pt.a4, pt.a6, p, D)
end

# Addition — mirrors _PolyPoint.__add__
function _add(pt1::PolyPt, pt2::PolyPt)::PolyPt
    _is_id(pt1) && return pt2
    _is_id(pt2) && return pt1

    p, D = pt1.p, pt1.D
    X1, Y1, Z1 = pt1.X, pt1.Y, pt1.Z
    X2, Y2, Z2 = pt2.X, pt2.Y, pt2.Z
    mul(a,b) = _pmul(a, b, p, D)
    add(a,b) = _padd(a, b, p, D)
    sub(a,b) = _psub(a, b, p, D)

    U1 = mul(X1, Z2);  U2 = mul(X2, Z1)
    S1 = mul(Y1, Z2);  S2 = mul(Y2, Z1)

    if U1 == U2
        S1 != S2 && return _identity(pt1.a4, pt1.a6, p, D)   # P + (−P) = O
        return _double(pt1)
    end

    W  = mul(Z1, Z2)
    Pv = sub(U2, U1)
    R  = sub(S2, S1)
    P2 = mul(Pv, Pv)
    P3 = mul(P2, Pv)
    t  = mul(_pscalar(2, p, D), mul(U1, P2))
    X3 = sub(sub(mul(mul(R, R), W), P3), t)
    Y3 = sub(mul(R, sub(mul(U1, P2), X3)), mul(S1, P3))
    Z3 = mul(W, P3)
    return PolyPt(X3, Y3, Z3, pt1.a4, pt1.a6, p, D)
end

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

"""
    run_section_ladder(p, D, X_in, Y_in, Z_in, a4_in, a6_in, max_k)

Iterative addition ladder  k·Pi  for k = 0 … max_k.
All inputs are plain Int vectors (coefficient lists mod p, length ≤ D+1).

Returns Dict{String,Any}:
    "k" => Dict("X" => [Int…], "Y" => [Int…], "Z" => [Int…])
for k = 0, 1, …, max_k.  Identity (k=0) has Z = [0,…].
"""
function run_section_ladder(p::Int, D::Int,
                             X_in, Y_in, Z_in,
                             a4_in, a6_in,
                             max_k::Int)::Dict{String,Any}
    # Normalise inputs to Vector{Int} of length D+1
    cvt(v) = _pad_to(Vector{Int}(v), D)
    X  = cvt(X_in);  Y  = cvt(Y_in);  Z  = cvt(Z_in)
    a4 = cvt(a4_in); a6 = cvt(a6_in)

    Pi   = PolyPt(X, Y, Z, a4, a6, p, D)
    id   = _identity(a4, a6, p, D)
    out  = Dict{String,Any}()

    # k = 0: identity
    out["0"] = Dict("X" => copy(id.X), "Y" => copy(id.Y), "Z" => copy(id.Z))

    max_k < 1 && return out

    # k = 1 … max_k via iterative addition
    prev = Pi
    out["1"] = Dict("X" => copy(Pi.X), "Y" => copy(Pi.Y), "Z" => copy(Pi.Z))

    for k in 2:max_k
        prev = _add(prev, Pi)
        out[string(k)] = Dict("X" => copy(prev.X), "Y" => copy(prev.Y), "Z" => copy(prev.Z))
    end

    return out
end
