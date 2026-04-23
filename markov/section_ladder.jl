# ---------------------------------------------------------------------------
# Polynomial helpers — all arithmetic mod p, truncated to degree ≤ D.
# ---------------------------------------------------------------------------

function _check_p_D(p::Int, D::Int)
    p > 1 || error("p must be > 1, got $p")
    D >= 0 || error("D must be >= 0, got $D")
end

function _materialize_poly(v, p::Int, D::Int, label::AbstractString)::Vector{Int}
    v isa AbstractVector || error("$label is not an AbstractVector: $(typeof(v))")

    out = zeros(Int, D + 1)
    n = min(length(v), D + 1)

    for i in 1:n
        isassigned(v, i) || error("$label has undef slot at index $i (type=$(typeof(v)))")
        c = v[i]
        try
            out[i] = mod(Int(c), p)
        catch err
            error("$label coefficient $i cannot convert to Int (value=$(repr(c)), type=$(typeof(c))): $err")
        end
    end

    return out
end

function _pmul(a::Vector{Int}, b::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    for (i, ca) in enumerate(a)
        ca == 0 && continue
        for (j, cb) in enumerate(b)
            cb == 0 && continue
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
    for i in 1:na
        out[i] = a[i] % p
    end
    for i in 1:nb
        out[i] = (out[i] + b[i]) % p
    end
    return out
end

function _psub(a::Vector{Int}, b::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    na = min(length(a), D + 1)
    nb = min(length(b), D + 1)
    for i in 1:na
        out[i] = a[i] % p
    end
    for i in 1:nb
        out[i] = (out[i] - b[i]) % p
    end
    for i in 1:D+1
        out[i] %= p
        out[i] < 0 && (out[i] += p)
    end
    return out
end

function _pneg(a::Vector{Int}, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    for i in 1:min(length(a), D + 1)
        out[i] = (p - (a[i] % p)) % p
    end
    return out
end

function _pscalar(s::Int, p::Int, D::Int)::Vector{Int}
    out = zeros(Int, D + 1)
    out[1] = mod(s, p)
    return out
end

_pzero(D::Int) = zeros(Int, D + 1)
function _pone(D::Int)::Vector{Int}
    v = zeros(Int, D + 1)
    v[1] = 1
    return v
end

function _is_zero_poly(v)::Bool
    v isa AbstractVector || return false
    for i in eachindex(v)
        isassigned(v, i) || return false
        v[i] == 0 || return false
    end
    return true
end

function _pad_to(v::Vector{Int}, D::Int)::Vector{Int}
    n = length(v)
    n == D + 1 && return v
    n > D + 1  && return v[1:D+1]
    return vcat(v, zeros(Int, D + 1 - n))
end

# ---------------------------------------------------------------------------
# Projective point
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

_neg(pt::PolyPt) = PolyPt(pt.X, _pneg(pt.Y, pt.p, pt.D), pt.Z, pt.a4, pt.a6, pt.p, pt.D)

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

function _add(pt1::PolyPt, pt2::PolyPt)::PolyPt
    _is_id(pt1) && return pt2
    _is_id(pt2) && return pt1

    p, D = pt1.p, pt1.D
    X1, Y1, Z1 = pt1.X, pt1.Y, pt1.Z
    X2, Y2, Z2 = pt2.X, pt2.Y, pt2.Z
    mul(a,b) = _pmul(a, b, p, D)
    sub(a,b) = _psub(a, b, p, D)

    U1 = mul(X1, Z2)
    U2 = mul(X2, Z1)
    S1 = mul(Y1, Z2)
    S2 = mul(Y2, Z1)

    if U1 == U2
        S1 != S2 && return _identity(pt1.a4, pt1.a6, p, D)
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

function run_section_ladder(p::Int, D::Int,
                            X_in, Y_in, Z_in,
                            a4_in, a6_in,
                            max_k::Int)::Dict{String,Any}
    _check_p_D(p, D)
    max_k >= 0 || error("max_k must be >= 0, got $max_k")

    X  = _materialize_poly(X_in,  p, D, "X_in")
    Y  = _materialize_poly(Y_in,  p, D, "Y_in")
    Z  = _materialize_poly(Z_in,  p, D, "Z_in")
    a4 = _materialize_poly(a4_in, p, D, "a4_in")
    a6 = _materialize_poly(a6_in, p, D, "a6_in")

    Pi = PolyPt(X, Y, Z, a4, a6, p, D)
    id = _identity(a4, a6, p, D)

    out = Dict{String,Any}()
    out["0"] = Dict{String,Any}(
        "X" => copy(id.X),
        "Y" => copy(id.Y),
        "Z" => copy(id.Z),
    )

    max_k == 0 && return out

    out["1"] = Dict{String,Any}(
        "X" => copy(Pi.X),
        "Y" => copy(Pi.Y),
        "Z" => copy(Pi.Z),
    )

    prev = Pi
    for k in 2:max_k
        try
            prev = _add(prev, Pi)
        catch err
            error("run_section_ladder failed at k=$k: $err")
        end

        out[string(k)] = Dict{String,Any}(
            "X" => copy(prev.X),
            "Y" => copy(prev.Y),
            "Z" => copy(prev.Z),
        )
    end

    return out
end
