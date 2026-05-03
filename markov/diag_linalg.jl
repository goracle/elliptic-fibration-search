# diag_linalg.jl
# Linear algebra over GF(p): Nemo dense helpers, Block Wiedemann sparse
# null-space computation, sparse rank estimation, and the main kernel
# entry points (right_kernel_basis, left_kernel_basis).
#
# Exports (informally):
#   to_nemo_mat,
#   spmv_mod, spmv_T_mod, block_bm_scalar_lcm, rand_block, block_apply,
#   right_kernel_basis_wiedemann,
#   to_sparse_mod, sparse_rank_estimate, rank_is_cheap,
#   dense_kernel_from_subsample,
#   right_kernel_basis, left_kernel_basis
#
# Depends on: diag_bootstrap.jl, diag_utils.jl

# ---------------------------------------------------------------------------
# Nemo GF matrix helpers  (retained for small matrices / non-kernel uses)
# ---------------------------------------------------------------------------
"""
Convert a plain Int matrix to a Nemo matrix over GF(p).
Avoids building an intermediate Vector{Vector} copy.
"""
function to_nemo_mat(M::AbstractMatrix{Int}, Fp)
    nr, nc = size(M)
    nr == 0 && return matrix(Fp, 0, nc, elem_type(Fp)[])
    data = Vector{elem_type(Fp)}(undef, nr * nc)
    k = 1
    @inbounds for i in 1:nr
        for j in 1:nc
            data[k] = Fp(M[i, j])
            k += 1
        end
    end
    return matrix(Fp, nr, nc, data)
end

# ---------------------------------------------------------------------------
# Block Wiedemann sparse null-space computation over GF(p)
#
# Works entirely with Julia SparseArrays — never builds a dense Nemo matrix.
# Suitable for matrices where nrows × ncols × 4 bytes would OOM Nemo's kernel.
#
# Algorithm (Coppersmith 1994 / Villard 1997):
#   Given A ∈ GF(p)^{m×n}, find a basis for ker(A) (right null space).
#
#   1. Choose random block matrices U ∈ GF(p)^{b×m}, V ∈ GF(p)^{b×n}.
#   2. Compute Krylov sequence:  F_i = U * A^i * V^T  ∈ GF(p)^{b×b}
#      for i = 0 .. L = 2*ceil(n/b) + slack.
#   3. Run block Berlekamp-Massey on {F_i} to find the minimal matrix
#      polynomial Λ(x) = Σ Λ_k x^k  of degree d ≤ ceil(n/b).
#   4. For each column v_j of V, compute  w_j = Σ_k Λ_k * (A^k * v_j).
#      Non-zero w_j are kernel vectors.  We collect enough starting vectors
#      until we have ≥ expected_nullity kernel vectors.
#
# Memory: O(nnz(A) + b*n) per iteration — never O(m*n).
# ---------------------------------------------------------------------------

"""
Sparse mat-vec: A (m×n sparse Int) times v (length-n Int vector), mod p.
"""
function spmv_mod(A::SparseMatrixCSC{Int,Int}, v::Vector{Int}, p::Int)::Vector{Int}
    m    = size(A, 1)
    nc   = size(A, 2)
    nt   = Threads.nthreads()
    rows = rowvals(A)
    vals = nonzeros(A)
    if nt == 1 || nc < 512
        out = zeros(Int, m)
        @inbounds for col in 1:nc
            vj = v[col]
            vj == 0 && continue
            for idx in nzrange(A, col)
                out[rows[idx]] = mod(out[rows[idx]] + vals[idx] * vj, p)
            end
        end
        return out
    end
    # Each thread accumulates into its own output vector (partitioned by column)
    # to avoid row-write races.  Merge serially at the end.
    col_chunks  = collect(Iterators.partition(1:nc, cld(nc, nt)))
    local_outs  = [zeros(Int, m) for _ in 1:length(col_chunks)]
    Threads.@threads for t in 1:length(col_chunks)
        lo = local_outs[t]
        @inbounds for col in col_chunks[t]
            vj = v[col]
            vj == 0 && continue
            for idx in nzrange(A, col)
                r = rows[idx]
                lo[r] = mod(lo[r] + vals[idx] * vj, p)
            end
        end
    end
    out = local_outs[1]
    @inbounds for t in 2:length(local_outs)
        lo = local_outs[t]
        for i in 1:m
            out[i] = mod(out[i] + lo[i], p)
        end
    end
    return out
end

"""
Sparse mat-vec with A^T: returns A^T * v mod p (v length-m, out length-n).
"""
function spmv_T_mod(A::SparseMatrixCSC{Int,Int}, v::Vector{Int}, p::Int)::Vector{Int}
    nc   = size(A, 2)
    out  = zeros(Int, nc)
    rows = rowvals(A)
    vals = nonzeros(A)
    # Each column of A gives one independent output element — embarrassingly parallel.
    Threads.@threads for col in 1:nc
        s = 0
        @inbounds for idx in nzrange(A, col)
            s += vals[idx] * v[rows[idx]]
        end
        out[col] = mod(s, p)
    end
    return out
end

"""
b×b matrix-vector product mod p.  A is stored column-major (Vector{Int} length b*b).
"""
@inline function matvec_bb_mod(A::Matrix{Int}, v::Vector{Int}, p::Int, b::Int)::Vector{Int}
    out = zeros(Int, b)
    @inbounds for j in 1:b
        vj = v[j]
        vj == 0 && continue
        for i in 1:b
            out[i] = mod(out[i] + A[i,j] * vj, p)
        end
    end
    return out
end

"""
Block Berlekamp-Massey over GF(p) for b×b matrix sequences.

Given sequence F[0..L-1] of b×b matrices over GF(p), returns the minimal
matrix polynomial Λ such that Σ_k Λ[k] * F[i+k] = 0 for all valid i.

Returns (Λ, d) where Λ is a Vector of b×b Int matrices (ascending degree)
and d = length(Λ)-1 is the degree.

This is a scalar BM run on each row of the sequence, then combined.
For our purposes (finding kernel vectors) we run scalar BM on each of the
b projection sequences u_i^T * F_k and take the LCM of their minimal polys,
which gives us a scalar polynomial that annihilates the sequence.
"""
function block_bm_scalar_lcm(F_seq::Vector{Matrix{Int}}, p::Int, b::Int)
    L = length(F_seq)
    # For each pair (i,j) with i in 1:b, run scalar BM on the sequence F_seq[k][i,j].
    # Then take LCM of all resulting minimal polynomials.
    # In practice: run on the diagonal (i==j) and first row — usually sufficient.
    
    # Scalar BM over GF(p) — returns minimal poly as coefficient vector (ascending).
    function scalar_bm(s::Vector{Int})
        n = length(s)
        C = [1]; B = [1]
        L_bm = 0; m = 1; b_bm = 1
        for i in 1:n
            d = mod(sum(C[k+1] * s[i-k] for k in 0:L_bm if i-k >= 1; init=0), p)
            if d == 0
                m += 1
            elseif 2*L_bm <= i-1
                T = copy(C)
                inv_b = invmod(b_bm, p)
                coef  = mod(d * inv_b, p)
                # C = C - coef * x^m * B
                new_len = max(length(C), length(B) + m)
                resize!(C, new_len)
                for k in eachindex(B)
                    C[k+m] = mod(C[k+m] - coef * B[k], p)
                end
                L_bm = i - L_bm
                B = T; b_bm = d; m = 1
            else
                inv_b = invmod(b_bm, p)
                coef  = mod(d * inv_b, p)
                new_len = max(length(C), length(B) + m)
                resize!(C, new_len)
                for k in eachindex(B)
                    C[k+m] = mod(C[k+m] - coef * B[k], p)
                end
                m += 1
            end
        end
        return C  # C[1..L_bm+1], C[1]==1
    end

    # Polynomial LCM over GF(p) via GCD.
    function poly_gcd(a::Vector{Int}, b_poly::Vector{Int})
        strip(v) = begin w=copy(v); while length(w)>1 && w[end]==0; pop!(w); end; w end
        is_zero(v) = all(==(0), v)
        a, b_poly = strip(a), strip(b_poly)
        while !is_zero(strip(b_poly))
            _, r = polydivrem_mod(a, b_poly, p)
            a, b_poly = b_poly, strip(r)
        end
        a = strip(a)
        if !isempty(a) && a[end] != 0 && a[end] != 1
            inv_lc = invmod(a[end], p)
            a = mod.(a .* inv_lc, p)
        end
        return a
    end

    function poly_lcm(a::Vector{Int}, b_poly::Vector{Int})
        g = poly_gcd(a, b_poly)
        # lcm = a * b / gcd; but divide first to avoid degree explosion
        b_div, _ = polydivrem_mod(b_poly, g, p)
        polymul_mod(a, b_div, p)
    end

    min_poly = [1]  # start with 1, take LCM with each scalar minimal poly
    n_probes = min(b * b, 20)  # probe up to 20 scalar sequences
    probed = 0
    for i in 1:b
        for j in 1:b
            probed >= n_probes && break
            seq = [mod(F_seq[k][i, j], p) for k in 1:L]
            mp  = scalar_bm(seq)
            min_poly = poly_lcm(min_poly, mp)
            probed += 1
        end
        probed >= n_probes && break
    end
    return min_poly  # ascending-degree coefficients, monic
end

"""
Build a random b×n matrix over GF(p) as a Vector of b row-vectors.
"""
function rand_block(b::Int, n::Int, p::Int, rng)
    [rand(rng, 0:p-1, n) for _ in 1:b]
end

"""
Apply a b×n block (list of b row vectors) to an n-vector: returns b-vector.
"""
function block_apply(rows::Vector{Vector{Int}}, v::Vector{Int}, p::Int)
    [mod(dot(r, v), p) for r in rows]
end

"""
right_kernel_basis_wiedemann(A_sp, p; block_size, expected_nullity, seed, verbose)

Scalar Wiedemann for the right null space of a rectangular sparse A (m×n) over GF(p).

Strategy: work with C = A*A^T  (m×m, symmetric).
  ker(C) = {y : A*A^T*y = 0}.  For any y in ker(C),  w = A^T*y  satisfies
  A*w = A*(A^T*y) = C*y = 0,  so w is in ker(A).

Per kernel vector:
  1. Random u, v ∈ GF(p)^m.
  2. Scalar Krylov sequence s[k] = u · C^k v  for k=0..L,  L = 2m+slack.
     Each C-step = spmv_mod(A, spmv_T_mod(A^T, x)) [A^T first, then A, since C=A*A^T].
  3. BM on s → minimal poly λ of degree d.
  4. y = λ(C)*v = Σ λ[k] * C^k * v  (length m).
  5. w = A^T * y  (length n).  Verify A*w = 0.

Working vectors always stay in GF(p)^m — no dimension confusion.
"""

function right_kernel_basis_wiedemann(
        A_sp::SparseMatrixCSC{Int,Int},
        p::Int;
        block_size::Int = 64,
        expected_nullity::Int = 1,
        seed::Int = 42,
        verbose::Bool = true)

    m, n = size(A_sp)
    rng  = MersenneTwister(seed)

    # Long-sequence block Krylov / block Wiedemann-style solver.
    #
    # We build a large Krylov-generated subspace for B = A^T*A by repeatedly
    # applying B to a block of random starting vectors, then close the span under
    # B until it is invariant.  Once the restricted operator T is exact, we lift
    # ker(T) back to ambient space and certify each candidate with A*x = 0.
    #
    # This is intentionally conservative: if the span is too small or refuses to
    # close, we keep expanding rather than returning speculative vectors.

    max_basis      = min(n, max(8 * block_size, expected_nullity + 256, 1024))
    max_sweeps     = max(12, cld(max_basis, max(1, block_size)) + 8)
    max_closures   = max(4, cld(expected_nullity, max(1, block_size)) + 4)
    max_restarts   = max(3, cld(expected_nullity, max(1, block_size)) + 2)
    seed_block_sz  = min(n, max(block_size, expected_nullity ÷ 2 + 32, 64))

    verbose && _log("  [bw] Block Krylov solver (B=A^T*A, n×n)  m=$m  n=$n  nnz=$(nnz(A_sp))  block=$block_size  target=$(expected_nullity)")

    # Sparse operator B = A^T*A, applied without materializing B.
    B_apply(v::Vector{Int}) = spmv_T_mod(A_sp, spmv_mod(A_sp, v, p), p)

    # Dense kernel over GF(p) for the small basis matrix T.
    function dense_kernel_basis_mod(M::Matrix{Int}, p::Int)
        Fp = GF(p)
        ker_mat = kernel(to_nemo_mat(M, Fp); side=:right)
        nr_ker, nc_ker = nrows(ker_mat), ncols(ker_mat)
        return [Int[lift(ZZ, ker_mat[i, j]) for i in 1:nr_ker] for j in 1:nc_ker]
    end

    # Reduced row-echelon-style basis for vectors over GF(p).
    basis_vecs  = Vector{Vector{Int}}()
    basis_pivot = Int[]

    function reduce_with_basis(v::Vector{Int})
        w = copy(v)
        coeffs = zeros(Int, length(basis_vecs))
        for k in length(basis_vecs):-1:1
            pk = basis_pivot[k]
            c = w[pk]
            if c != 0
                coeffs[k] = c
                bv = basis_vecs[k]
                @inbounds for j in 1:n
                    w[j] = mod(w[j] - c * bv[j], p)
                end
            end
        end
        return coeffs, w
    end

    function insert_basis!(v::Vector{Int})
        coeffs, w = reduce_with_basis(v)
        all(==(0), w) && return false, coeffs, w

        pv = findfirst(!=(0), w)
        pv === nothing && return false, coeffs, w
        inv_pv = invmod(w[pv], p)
        w = mod.(w .* inv_pv, p)

        # Eliminate the new pivot from all existing basis vectors to keep the
        # basis in reduced form.
        for i in eachindex(basis_vecs)
            c = basis_vecs[i][pv]
            if c != 0
                bi = basis_vecs[i]
                @inbounds for j in 1:n
                    bi[j] = mod(bi[j] - c * w[j], p)
                end
            end
        end

        # Insert while preserving increasing pivot order.
        pos = searchsortedfirst(basis_pivot, pv)
        insert!(basis_pivot, pos, pv)
        insert!(basis_vecs, pos, w)
        return true, coeffs, w
    end

    function basis_coordinates(v::Vector{Int})
        return reduce_with_basis(v)
    end

    function random_block_vectors(k::Int)
        [rand(rng, 0:p-1, n) for _ in 1:k]
    end

    function build_seeded_krylov_basis(; wipe::Bool=false)
        # On restarts we keep the existing basis and seed new random directions
        # into it.  Only wipe when explicitly requested (e.g. first call).
        if wipe
            empty!(basis_vecs)
            empty!(basis_pivot)
        end

        frontier = random_block_vectors(seed_block_sz)
        added_total = 0

        for sweep in 1:max_sweeps
            added_this_sweep = 0
            next_frontier = Vector{Vector{Int}}()
            image_frontier = Vector{Vector{Int}}()

            frontier_len = length(frontier)
            step_stride = max(16, cld(frontier_len, 8))
            verbose && _log("  [bw] sweep=$sweep  frontier=$frontier_len  basis=$(length(basis_vecs))")

            # First absorb the current frontier itself.
            for (idx, v) in enumerate(frontier)
                inserted, _, w = insert_basis!(v)
                if inserted
                    added_this_sweep += 1
                    push!(next_frontier, w)
                end
                if verbose && (idx == 1 || idx % step_stride == 0 || idx == frontier_len)
                    _log("  [bw] sweep=$sweep  frontier pass $idx/$frontier_len  basis=$(length(basis_vecs))  added=$added_this_sweep")
                end
                length(basis_vecs) >= max_basis && break
            end

            # Then absorb one B-step from everything we just discovered,
            # computing B-images in parallel and inserting serially.
            absorb_list = vcat(frontier, next_frontier)
            absorb_len  = length(absorb_list)
            image_stride = max(16, cld(absorb_len, 8))
            batch_sz_seed = max(1, Threads.nthreads() * 4)
            ab_idx = 1
            while ab_idx <= absorb_len
                ab_end  = min(ab_idx + batch_sz_seed - 1, absorb_len)
                ab_batch = absorb_list[ab_idx:ab_end]
                ab_images = Vector{Vector{Int}}(undef, length(ab_batch))
                Threads.@threads for k in 1:length(ab_batch)
                    ab_images[k] = B_apply(ab_batch[k])
                end
                for (k, img) in enumerate(ab_images)
                    inserted, _, w = insert_basis!(img)
                    if inserted
                        added_this_sweep += 1
                        push!(image_frontier, w)
                    end
                    glob_idx = ab_idx + k - 1
                    if verbose && (glob_idx == 1 || glob_idx % image_stride == 0 || glob_idx == absorb_len)
                        _log("  [bw] sweep=$sweep  image pass $glob_idx/$absorb_len  basis=$(length(basis_vecs))  added=$added_this_sweep")
                    end
                end
                length(basis_vecs) >= max_basis && break
                ab_idx = ab_end + 1
            end

            added_total += added_this_sweep
            verbose && _log("  [bw] sweep=$sweep  basis=$(length(basis_vecs))  added=$added_this_sweep  next_frontier=$(length(vcat(next_frontier, image_frontier)))")

            # Advance the frontier.  Even if nothing new was inserted this sweep,
            # keep a few more rounds going: a shallow frontier can stall before the
            # actual invariant subspace has been exposed.
            frontier = vcat(next_frontier, image_frontier)
            isempty(frontier) && break
            length(basis_vecs) >= max_basis && break
            added_this_sweep == 0 && sweep >= 3 && break
        end

        return added_total
    end

    function close_under_B!()
        # Sweep the live basis under B, processing newly inserted vectors in the
        # same round rather than waiting for the next one.  We walk by index so
        # that any vector appended during this sweep is picked up before we exit.
        # A round ends when the live index has caught up to the current end of the
        # basis without inserting anything new; only then is the subspace closed.
        #
        # B_apply calls are batched and parallelised; insert_basis! is serial
        # (it mutates the shared basis and cannot be safely concurrent).
        batch_sz = max(1, Threads.nthreads() * 4)
        for closure_round in 1:max_closures
            start_len = length(basis_vecs)
            added = 0
            idx = 1
            verbose && _log("  [bw] closure_round=$closure_round  start_basis=$start_len")
            while idx <= length(basis_vecs)
                # Grab a batch of basis vectors, compute their B-images in parallel.
                batch_end = min(idx + batch_sz - 1, length(basis_vecs))
                batch     = basis_vecs[idx:batch_end]
                images    = Vector{Vector{Int}}(undef, length(batch))
                Threads.@threads for k in 1:length(batch)
                    images[k] = B_apply(batch[k])
                end
                # Insert results serially to keep the basis consistent.
                for img in images
                    inserted, _, _ = insert_basis!(img)
                    inserted && (added += 1)
                end
                if verbose && (idx == 1 || batch_end % max(16, cld(start_len, 8)) == 0 || batch_end == length(basis_vecs))
                    _log("  [bw] closure_round=$closure_round  progress $batch_end/$(length(basis_vecs))  basis=$(length(basis_vecs))  added=$added")
                end
                length(basis_vecs) >= max_basis && break
                idx = batch_end + 1
            end
            verbose && _log("  [bw] closure_round=$closure_round  basis=$(length(basis_vecs))  added=$added")
            added == 0 && return true
            length(basis_vecs) >= max_basis && break
        end
        return false
    end

    basis = Vector{Vector{Int}}()
    seen  = Set{Vector{Int}}()

    for restart in 1:max_restarts
        prev_dim = length(basis_vecs)
        # First restart seeds from scratch; subsequent ones inject fresh random
        # directions into the existing span so accumulated progress is kept.
        wipe_this = (restart == 1)
        verbose && _log("  [bw] restart=$restart  seeding long Krylov basis (seed_block_sz=$seed_block_sz, max_sweeps=$max_sweeps)")
        build_seeded_krylov_basis(; wipe=wipe_this)

        if isempty(basis_vecs)
            verbose && _log("  [bw] restart=$restart produced no basis vectors; retrying")
            continue
        end

        closed = close_under_B!()
        k = length(basis_vecs)
        status = closed ? "yes" : "no"
        verbose && _log("  [bw] restart=$restart  invariant_basis=$status  dim=$k")

        if !closed
            # If the basis grew since we entered this restart, don't discard it —
            # just move on to the next restart which will seed more directions into
            # the same span.  Only treat it as a true stall if nothing was added.
            if k > prev_dim
                verbose && _log("  [bw] restart=$restart: basis grew ($prev_dim→$k) but not yet closed; continuing")
            else
                verbose && _log("  [bw] restart=$restart: basis stagnant at dim=$k; will re-seed")
            end
            # Either way, attempt kernel extraction before looping — the partially
            # closed subspace may already contain good kernel vectors.
        end

        # Build T, the matrix of B restricted to the current basis:
        #   B * q_j = sum_i T[i,j] q_i.
        T = zeros(Int, k, k)
        invariant_ok = true
        for j in 1:k
            coeffs, residual = basis_coordinates(B_apply(basis_vecs[j]))
            if any(!=(0), residual)
                invariant_ok = false
                continue
            end
            @inbounds for i in 1:k
                T[i, j] = mod(coeffs[i], p)
            end
        end

        if !invariant_ok
            # Some basis vectors leaked out of the current span.  Instead of
            # stopping at the first witness, scan the whole basis and add every
            # leaked residual we can see in this pass.
            leak_count = 0
            current_basis = copy(basis_vecs)
            basis_len = length(current_basis)
            stride = max(16, cld(basis_len, 8))
            verbose && _log("  [bw] restart=$restart  leak_scan start_basis=$basis_len")
            for (idx, q) in enumerate(current_basis)
                coeffs, residual = basis_coordinates(B_apply(q))
                if any(!=(0), residual)
                    inserted, _, _ = insert_basis!(residual)
                    inserted && (leak_count += 1)
                else
                    @inbounds for i in 1:length(coeffs)
                        T[i, idx] = mod(coeffs[i], p)
                    end
                end
                if verbose && (idx == 1 || idx % stride == 0 || idx == basis_len)
                    _log("  [bw] restart=$restart  leak_scan progress $idx/$basis_len  basis=$(length(basis_vecs))  leaks=$leak_count")
                end
                length(basis_vecs) >= max_basis && break
            end
            verbose && _log("  [bw] restart=$restart  leak_scan inserted=$leak_count  dim=$(length(basis_vecs))")
            closed = close_under_B!()
            k = length(basis_vecs)
            status = closed ? "yes" : "no"
            verbose && _log("  [bw] restart=$restart  post-leak closure: invariant_basis=$status  dim=$k")
            if !closed
                continue
            end
            T = zeros(Int, k, k)
            invariant_ok = true
            for j in 1:k
                coeffs, residual = basis_coordinates(B_apply(basis_vecs[j]))
                if any(!=(0), residual)
                    invariant_ok = false
                    break
                end
                @inbounds for i in 1:k
                    T[i, j] = mod(coeffs[i], p)
                end
            end
        end

        if !invariant_ok
            verbose && _log("  [bw] restart=$restart: basis still not invariant; continuing")
            continue
        end

        verbose && _log("  [bw] invariant subspace dimension k=$k")
        ker_T = dense_kernel_basis_mod(T, p)
        if isempty(ker_T)
            verbose && _log("  [bw] restart=$restart: ker(T)=0")
            continue
        end

        # Lift kernel vectors: x = Q * y, where Q columns are the basis vectors.
        candidates = Vector{Vector{Int}}()
        empty!(seen)
        for y in ker_T
            x = zeros(Int, n)
            for j in 1:k
                yj = mod(y[j], p)
                if yj != 0
                    qj = basis_vecs[j]
                    @inbounds for i in 1:n
                        x[i] = mod(x[i] + yj * qj[i], p)
                    end
                end
            end
            all(==(0), x) && continue

            # Verify the candidate in the original system, not just in T.
            Ax = spmv_mod(A_sp, x, p)
            if any(!=(0), Ax)
                verbose && _log("  [bw] restart=$restart: lifted candidate failed A*x=0; discarding")
                continue
            end

            fi = findfirst(!=(0), x)
            fi === nothing && continue
            inv_fi = invmod(x[fi], p)
            x = mod.(x .* inv_fi, p)

            if !(x in seen)
                push!(seen, x)
                push!(candidates, x)
            end
            length(candidates) >= expected_nullity && break
        end

        if !isempty(candidates)
            basis = candidates
            verbose && _log("  [bw] recovered $(length(basis)) kernel vector(s) from restart=$restart")
            if length(basis) >= expected_nullity
                break
            end
        else
            verbose && _log("  [bw] restart=$restart: no verified kernel vectors recovered")
        end
    end

    if isempty(basis)
        verbose && _log("  [bw] no kernel vectors recovered")
    else
        verbose && _log("  [bw] recovered $(length(basis)) kernel vector(s) total")
    end
    return basis
end

function to_sparse_mod(A::SparseMatrixCSC{Int,Int}, p::Int)::SparseMatrixCSC{Int,Int}
    m, n = size(A)
    rv = rowvals(A)
    nz = nonzeros(A)
    I_idx = Int[]; J_idx = Int[]; V_vals = Int[]
    for j in 1:n
        @inbounds for idx in nzrange(A, j)
            v = mod(nz[idx], p)
            v == 0 && continue
            push!(I_idx, rv[idx])
            push!(J_idx, j)
            push!(V_vals, v)
        end
    end
    return sparse(I_idx, J_idx, V_vals, m, n)
end

function to_sparse_mod(M::AbstractMatrix{Int}, p::Int)::SparseMatrixCSC{Int,Int}
    m, n = size(M)
    if Threads.nthreads() == 1 || m * n < 250_000
        I_idx = Int[]; J_idx = Int[]; V_vals = Int[]
        for j in 1:n, i in 1:m
            v = mod(M[i,j], p)
            if v != 0
                push!(I_idx, i); push!(J_idx, j); push!(V_vals, v)
            end
        end
        return sparse(I_idx, J_idx, V_vals, m, n)
    end

    nchunks = max(1, min(Threads.nthreads(), n))
    perI = [Int[] for _ in 1:nchunks]
    perJ = [Int[] for _ in 1:nchunks]
    perV = [Int[] for _ in 1:nchunks]
    col_chunks = collect(Iterators.partition(1:n, cld(n, nchunks)))

    Threads.@threads for chunk_idx in 1:length(col_chunks)
        cols = col_chunks[chunk_idx]
        I_loc = perI[chunk_idx]
        J_loc = perJ[chunk_idx]
        V_loc = perV[chunk_idx]
        for j in cols
            @inbounds for i in 1:m
                v = mod(M[i,j], p)
                if v != 0
                    push!(I_loc, i)
                    push!(J_loc, j)
                    push!(V_loc, v)
                end
            end
        end
    end

    I_idx = vcat(perI...)
    J_idx = vcat(perJ...)
    V_vals = vcat(perV...)
    return sparse(I_idx, J_idx, V_vals, m, n)
end

"""
    sparse_rank_estimate(A_sp, p; n_rows, rng) -> (rank_est, nullity_est)

Cheap rank estimate via sparse Gaussian elimination mod p on a random
row-sample of size `n_rows`.  Returns `(rank_est, nullity_est)` where
`nullity_est = n - rank_est`.  The estimate is a lower bound on rank.

Implementation: rows are stored as sparse Dict{col→val} and reduced in-place
against a pivot table that maps each pivot column to its (sparse) pivot row.
This never allocates an n_rows×n dense array, so it is safe even when
n_rows ≈ n ≈ 14000.  Memory is O(nnz of selected rows) rather than O(n_rows×n).
"""
function sparse_rank_estimate(A_sp::SparseMatrixCSC{Int,Int}, p::Int;
                               n_rows::Int = min(size(A_sp,1), 1024),
                               rng = MersenneTwister(99))
    m, n = size(A_sp)
    n_rows = min(n_rows, m)

    # Select rows: random sample without replacement.
    row_perm = randperm(rng, m)[1:n_rows]

    # Load selected rows as sparse dicts: local_idx -> Dict{col->val}
    # (local_idx is position in row_perm, not original row index)
    row_map = Dict(row_perm[i] => i for i in 1:n_rows)
    rows    = [Dict{Int,Int}() for _ in 1:n_rows]
    rv = rowvals(A_sp); nz_vals = nonzeros(A_sp)
    for col in 1:n
        for idx in nzrange(A_sp, col)
            r = rv[idx]
            haskey(row_map, r) || continue
            v = mod(nz_vals[idx], p)
            v == 0 && continue
            rows[row_map[r]][col] = v
        end
    end

    # Sparse column-pivot GE mod p.
    # pivot_cols: sorted list of active pivot column indices.
    # pivot_table: col -> monic sparse pivot row (Dict{col->val}).
    # One pass in column order suffices because pivot rows are monic and
    # elimination of column pc cannot re-introduce entries at columns < pc.
    pivot_cols  = Int[]
    pivot_table = Dict{Int, Dict{Int,Int}}()
    rank_est    = 0

    for local_i in 1:n_rows
        row = rows[local_i]
        isempty(row) && continue

        # Reduce row against existing pivots in column order (single pass).
        for pc in pivot_cols
            c = get(row, pc, 0)
            c == 0 && continue
            prow = pivot_table[pc]
            for (j, pv) in prow
                cur = get(row, j, 0)
                nv  = mod(cur - c * pv, p)
                if nv == 0
                    delete!(row, j)
                else
                    row[j] = nv
                end
            end
        end

        isempty(row) && continue

        # Find the smallest column index as pivot (leftmost non-zero).
        pc  = minimum(keys(row))
        pv  = row[pc]
        inv_pv = invmod(pv, p)

        # Normalise to make pivot entry 1.
        new_prow = Dict{Int,Int}()
        for (j, v) in row
            nv = mod(v * inv_pv, p)
            nv != 0 && (new_prow[j] = nv)
        end

        pivot_table[pc] = new_prow
        insert!(pivot_cols, searchsortedfirst(pivot_cols, pc), pc)
        rank_est += 1

        # Early exit: rank can't exceed n.
        rank_est >= n && break
    end

    return rank_est, n - rank_est
end

"""
    rank_is_cheap(m, n, rank_est, nullity_est, probe_rows) -> (cheap::Bool, reason::String)

Decide whether an exact rank/kernel computation is cheap enough to prefer
Nemo dense over Block Wiedemann, even when m*n > dense_threshold.

Returns (true, reason) when ANY of the following hold:

  1. THIN MATRIX  — min(m,n) ≤ 512: the smaller dimension is tiny, so Nemo's
     O(min(m,n)^2 * max(m,n)) dense kernel is dominated by the I/O cost.

  2. EXHAUSTIVE PROBE — probe_rows == m: the sparse GE covered every row, so
     rank_est is exact (not just a lower bound).  Rank is known; kernel can be
     computed without BW.  We allow up to 4× the normal dense_threshold since
     the probe already paid the row-scan cost.

  3. PROBE SATURATED — rank_est == min(probe_rows, n): the probe filled all
     available pivot slots, meaning the true rank is likely min(m, n) and
     nullity is near zero.  A full-row dense pass will confirm cheaply.
     Applied only when probe_rows >= 0.75*m (probe is representative).

  4. NULLITY TINY — nullity_est <= 4 AND probe_rows >= 0.5*m: the kernel is
     at most 4-dimensional; BW's overhead per kernel vector is hard to amortize
     for such small nullity.  A targeted dense solve on a row-sufficient
     subsample (n + nullity_est + 64 rows) is cheaper.
"""
function rank_is_cheap(m::Int, n::Int, rank_est::Int, nullity_est::Int,
                       probe_rows::Int; dense_threshold::Int=50_000_000)
    # 1. Thin matrix: short dimension makes dense cheap regardless.
    if min(m, n) <= 512
        return true, "thin matrix (min(m,n)=$(min(m,n)) ≤ 512)"
    end

    # 2. Exhaustive probe: rank is exact, allow 4× dense threshold.
    if probe_rows >= m && m * n <= 4 * dense_threshold
        return true, "exhaustive probe (all $m rows sampled) → exact rank=$rank_est"
    end

    # 3. Probe saturated: rank filled all pivot slots and probe is representative.
    if rank_est >= min(probe_rows, n) && probe_rows >= div(3 * m, 4)
        return true, "probe saturated at rank=$rank_est (probe=$probe_rows / $m rows, min(probe,n)=$(min(probe_rows,n)))"
    end

    # 4. Tiny nullity with a representative probe: BW overhead unwarranted.
    if nullity_est <= 4 && probe_rows >= m ÷ 2
        needed = min(m, n + nullity_est + 64)   # rows sufficient for dense solve
        if needed * n <= 4 * dense_threshold
            return true, "tiny nullity=$nullity_est with representative probe ($probe_rows / $m rows); dense subsample ($needed×$n) is cheap"
        end
    end

    return false, ""
end

"""
dense_kernel_from_subsample(A, p, needed_rows, rank_est, nullity_est)

When rank_is_cheap fires for a matrix that is still too large for a full dense
solve, compute the kernel by extracting a row-sufficient subsample.

Strategy: take the first `needed_rows` rows (after the sparse probe has already
established that rank stabilises quickly), run Nemo dense kernel on that block.
If the resulting nullity matches nullity_est, return it; otherwise fall back to
the full matrix (trusting that m*n <= 4*dense_threshold already checked).
"""
function dense_kernel_from_subsample(A::AbstractMatrix{Int}, p::Int,
                                     needed_rows::Int, nullity_est::Int)
    m, n  = size(A)
    nrows_use = min(m, needed_rows)
    Fp        = GF(p)
    A_sub     = to_nemo_mat(A[1:nrows_use, :], Fp)
    ker_mat   = kernel(A_sub; side=:right)
    nr_k, nc_k = nrows(ker_mat), ncols(ker_mat)
    result = [Int[Int(lift(ZZ, ker_mat[i,j])) for i in 1:nr_k] for j in 1:nc_k]
    # Sanity: if we got fewer kernel vectors than expected and there are more rows,
    # retry on the full matrix (it fits by construction from the caller's check).
    if length(result) < nullity_est && nrows_use < m
        A_full  = to_nemo_mat(A, Fp)
        ker_full = kernel(A_full; side=:right)
        nr_f, nc_f = nrows(ker_full), ncols(ker_full)
        result = [Int[Int(lift(ZZ, ker_full[i,j])) for i in 1:nr_f] for j in 1:nc_f]
    end
    return result
end

function right_kernel_basis(A::AbstractMatrix{Int}, p::Int;
                             expected_nullity::Int=1,
                             dense_threshold::Int=100_000_000)
    m, n = size(A)

    # Block Wiedemann is currently disabled (buggy). All calls route through
    # Nemo dense. If the matrix is genuinely too large, we raise rather than
    # silently falling back to broken BW.
    if m * n > dense_threshold
        throw(ErrorException(
            "[kernel] matrix $(m)×$(n) = $(m*n) elements exceeds dense_threshold=$dense_threshold " *
            "and Block Wiedemann is currently disabled. Reduce the matrix or raise dense_threshold."
        ))
    end

    _log("  [kernel] using Nemo dense kernel ($(m)×$(n) = $(m*n) elements)")
    Fp      = GF(p)
    A_nemo  = to_nemo_mat(A, Fp)
    ker_mat = kernel(A_nemo; side=:right)
    nr_ker, nc_ker = nrows(ker_mat), ncols(ker_mat)
    return [Int[Int(lift(ZZ, ker_mat[i,j])) for i in 1:nr_ker] for j in 1:nc_ker]

    # --- Block Wiedemann (disabled — buggy, do not re-enable without fixing) ---
    # To re-enable: remove the size check / throw above, restore the probe +
    # rank_is_cheap routing below, and delete this comment block.
    #
    # A_sp = to_sparse_mod(A, p)
    # probe_rows = min(m, max(4096, 8 * Int(ceil(sqrt(n)))))
    # rank_est, nullity_est = sparse_rank_estimate(A_sp, p; n_rows=probe_rows)
    # cheap, _ = rank_is_cheap(m, n, rank_est, nullity_est, probe_rows;
    #                          dense_threshold=dense_threshold)
    # if cheap
    #     needed = min(m, max(n + nullity_est + 64, probe_rows))
    #     return dense_kernel_from_subsample(A, p, needed, nullity_est)
    # end
    # effective_nullity = min(nullity_est, expected_nullity)
    # return right_kernel_basis_wiedemann(A_sp, p;
    #                                     block_size=max(32, min(64, effective_nullity + 16)),
    #                                     expected_nullity=effective_nullity,
    #                                     seed=42, verbose=true)
end

"""
Compute left kernel basis of A over GF(p) (= right kernel of A^T).
"""
function left_kernel_basis(A::AbstractMatrix{Int}, p::Int; kwargs...)
    right_kernel_basis(permutedims(A), p; kwargs...)
end
