using LinearAlgebra
using Hungarian


"""
Perform the Seeded Graph Matching algorithm to find an approximate matching
between any two graphs with known matching seeds. m is the number of seeded
vertices.
"""
function sgm(A::Array{Int,2}, B::Array{Int,2}, m::Int; initmethod::Symbol=:barycenter, maxiter::Int=20, returniter::Bool=false)

    # 0. Checks
    @assert size(A,1) == size(A,2)
    @assert size(B,1) == size(B,2)
    @assert size(A,1) == size(B,1)
    @assert initmethod in [:barycenter, :random]

    # 0. Get size
    N = size(A,1)
    n = N - m

    # 0. Convert adjacency matricies
    A = Float32.(A)
    B = Float32.(B)

    # 0. Split A and B into seeded and non-seeded parts

    A11 = A[1:m, 1:m]
    A12 = A[1:m, m+1:end]
    A21 = A[m+1:end, 1:m]
    A22 = A[m+1:end, m+1:end]

    B11 = B[1:m, 1:m]
    B12 = B[1:m, m+1:end]
    B21 = B[m+1:end, 1:m]
    B22 = B[m+1:end, m+1:end]

    # 0. Precompute constant values
    s1 = tr(BLAS.gemm('T', 'N', A11, B11))
    s2 = BLAS.gemm('N', 'T', A21, B21)
    s3 = BLAS.gemm('T', 'N', A12, B12)
    ϵ = 1e-3

    # 1. Initialize P
    P = ones(Float32, n, n) / n
    if initmethod == :random
        P = sample_doublystochastic(n)
    end

    # 2. While stopping criteria not met:
    it = 0
    while it < maxiter
        it += 1

        # 3. Compute the gradient
        s4 = BLAS.gemm('T', 'N', A22, BLAS.gemm('N', 'N', P, B22))
        ∇f = s2 + s3 + s4 + BLAS.gemm('N', 'N', A22, BLAS.gemm('N', 'T', P, B22))

        # 4. Find best permutation via Hungarian algorithm
        mm = maximum(∇f) + ϵ
        matching = Hungarian.munkres(-∇f .+ mm)
        Q = Float32.(Matrix(matching .== Hungarian.STAR))

        # 5. Compute the (maybe) optimal step size
        s5 = BLAS.gemm('T', 'N', A22, BLAS.gemm('N', 'N', Q, B22))
        c = tr(BLAS.gemm('N', 'T', s4, P))
        d = tr(BLAS.gemm('N', 'T', s4, Q)) + tr(BLAS.gemm('N', 'T', s5, P))
        e = tr(BLAS.gemm('N', 'T', s5, Q))
        u = tr(BLAS.gemm('T', 'N', P, s2)) + tr(BLAS.gemm('T', 'N', P, s3))
        v = tr(BLAS.gemm('T', 'N', Q, s2)) + tr(BLAS.gemm('T', 'N', Q, s3))

        α̃::Float32 = 0
        if (c - d + e != 0)
            α̃ = - (d - 2e + u - v) / (2 * (c - d + e))
        end

        # 6. Update P

        f0 = 0
        f1 = c - e + u - v
        falpha = (c - d + e) * (α̃^2) + (d - 2e + u - v) * α̃

        if ϵ < α̃ < 1-ϵ && falpha > f0 && falpha > f1
            P = (α̃ * P) + ((1-α̃) * Q)
        elseif f0 > f1
            P = Q
        else
            break
        end

    end

    # 7. Find the nearest permutation matrix with Hungarian algorithm
    mm = maximum(P) + ϵ
    matching = Hungarian.munkres(-P .+ mm)
    P̂ = Matrix(matching .== Hungarian.STAR)
    P̂ = [Matrix(I, m, m) zeros(Bool, m, n); zeros(Bool, n, m) P̂]
    P̂ = Int.(P̂)

    # 8. Compute the matching associated to P̂
    matching = argmax(P̂', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = sortslices(matching, dims=1)

    if returniter
        return P̂, matching, it
    end

    # 9. Return P̂ and the associated matching
    return P̂, matching

end


"""
Perform the Seeded Graph Matching algorithm to find an approximate matching
between any two graphs with known matching seeds. m is the number of seeded
vertices. Naive (slow) implementation.
"""
function sgm_simple(A::Array{Int,2}, B::Array{Int,2}, m::Int)

    # 0. Size checks
    @assert size(A,1) == size(A,2)
    @assert size(B,1) == size(B,2)
    @assert size(A,1) == size(B,1)

    # 0. Get size
    N = size(A,1)
    n = N - m

    # 0. Split A and B into seeded and non-seeded parts

    A11 = A[1:m, 1:m]
    A12 = A[1:m, m+1:end]
    A21 = A[m+1:end, 1:m]
    A22 = A[m+1:end, m+1:end]

    B11 = B[1:m, 1:m]
    B12 = B[1:m, m+1:end]
    B21 = B[m+1:end, 1:m]
    B22 = B[m+1:end, m+1:end]

    # 0. Define f function
    f = P̃ -> tr(A11' * B11) + tr(P̃' * A21 * B21') + tr(P̃' * A12' * B12) + tr(A22' * P̃ * B22 * P̃')

    # 1. Initialize P
    P = ones(Float64, n, n)
    P = P / n

    # 2. While stopping criteria not met:
    for i = 1:20

        # 3. Compute the gradient
        ∇f = (A21 * B21') + (A12' * B12) + (A22 * P * B22') + (A22' * P * B22)

        # 4. Find best permutation via Hungarian algorithm
        matching = Hungarian.munkres(-∇f)
        Q = map(x -> max(0, x-1), Matrix(matching))

        # 5. Compute the (maybe) optimal step size
        c = tr(A22' * P * B22 * P')
        d = tr((A22' * P * B22 * Q') + (A22' * Q * B22 * P'))
        e = tr(A22' * Q * B22 * Q')
        u = tr((P' * A21 * B21') + (P' * A12' * B12))
        v = tr((Q' * A21 * B21') + (Q' * A12' * B12))

        α̃ = 0
        if (c - d + e != 0)
            α̃ = - (d - 2e + u - v) / (2 * (c - d + e))
        end

        # 6. Update P
        P̃ = (α̃ * P) + ((1-α̃) * Q)
        β = argmax([f(P), f(P̃), f(Q)])
        if α̃ < 1 && α̃ > 0 && β == 2
            P = P̃
        elseif β == 3
            P = Q
        else
            break
        end

    end

    # 7. Find the nearest permutation matrix with Hungarian algorithm
    matching = Hungarian.munkres(-P)
    P̂ = map(x -> max(0, x-1), Matrix(matching))

    # 8. Compute the matching associated to P̂
    matching = argmax(P̂', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = vcat(hcat(1:m, 1:m), matching .+ m)
    matching = sortslices(matching, dims=1)

    # 9. Return P̂ and the associated matching
    return P̂, matching

end
