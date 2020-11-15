using LinearAlgebra
using Hungarian


"""
Perform the FAQ Graph Matching algorithm to find an approximate matching
between any two graphs without seeds.
"""
function faq(A::Array{Int,2}, B::Array{Int,2},; m::Int=0, iter::Int=30, initmethod::Symbol=:barycenter)

    @assert initmethod in [:barycenter, :softseed, :random]
    if (m != 0) @assert (initmethod == :softseed) end

    # 0. Get size
    N = size(A,1)
    n = N - m

    # 0. Convert adjacency matricies
    A = Float32.(Matrix(A))
    B = Float32.(Matrix(B))

    # 0. Set constant
    ϵ = 1e-3

    # 1. Initialize P
    P = ones(Float32, n, n) / n
    if initmethod == :softseed
        P = [[Matrix(I, m, m) zeros(Float32, m, n)]; [zeros(Float32, n, m) P]]
    elseif initmethod == :random
        P = sample_doublystochastic(N)
    end

    # 2. While stopping criteria not met:
    for i = 1:iter

        # 3. Compute the gradient
        s1 = BLAS.gemm('T', 'N', A, BLAS.gemm('N', 'N', P, B))
        s2 = BLAS.gemm('N', 'N', A, BLAS.gemm('N', 'T', P, B))
        ∇f = s1 + s2

        # 4. Find best permutation via Hungarian algorithm
        mm = maximum(∇f)
        matching = Hungarian.munkres(-∇f .+ (mm+0.01))
        Q = Float32.(Matrix(matching .== Hungarian.STAR))

        # 5. Compute the (maybe) optimal step size
        s3 = BLAS.gemm('T', 'N', A, BLAS.gemm('N', 'N', Q, B))
        c = tr(BLAS.gemm('N', 'T', s1, P))
        d = tr(BLAS.gemm('N', 'T', s1, Q)) + tr(BLAS.gemm('N', 'T', s3, P))
        e = tr(BLAS.gemm('N', 'T', s3, Q))

        α̃::Float32 = 0
        if (c - d + e != 0)
            α̃ = - (d - 2e) / (2 * (c - d + e))
        end

        # 6. Update P

        f0 = 0
        f1 = c - e
        falpha = ((c - d + e) * (α̃^2)) + ((d - 2e) * α̃)

        if α̃ < 1-ϵ && α̃ > ϵ && falpha > f0 && falpha > f1
            P = (α̃ * P) + ((1-α̃) * Q)
        elseif f0 > f1
            P = Q
        else
            break
        end

    end

    # 7. Find the nearest permutation matrix with Hungarian algorithm
    mm = maximum(P)
    matching = Hungarian.munkres(-P .+ (mm+0.001))
    P̂ = Matrix(matching .== Hungarian.STAR)

    # 8. Compute the matching associated to P̂
    matching = argmax(P̂', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = sortslices(matching, dims=1)

    # 9. Return P̂ and the associated matching
    return P̂, matching

end

"""
Perform the FAQ Graph Matching algorithm to find an approximate matching
between any two graphs without seeds. Naive (slow) implementation.
"""
function faq_naive(A::Array{Int,2}, B::Array{Int,2}, iter::Int=30)

    # 0. Get N
    N = size(A,1)

    # 0. Define f function
    f = X -> -tr(A * X * B' * X')

    # 0. Define the gradient of f
    ∇f = X -> -(A * X * B') - (A' * X * B)

    # 1. Initialize P
    P = ones(Float32, N, N)
    P = P / N

    # 2. While stopping criteria not met:
    for i = 1:iter

        # 3. Compute the gradient
        grad = ∇f(P)

        # 4. Find best permutation via Hungarian algorithm
        matching = Hungarian.munkres(grad)
        Q = map(x -> max(0, x-1), Matrix(matching))

        # 5. Compute the optimal step size
        w = tr(A*Q*B'*Q')
        x = tr(A*Q*B'*P')
        y = tr(A*P*B'*Q')
        z = tr(A*P*B'*P')
        a = x + y + z - w
        b = - x - y - 2z
        α = -b / 2a
        α = clamp(α, 0, 1)

        if α <= 0
            break
        end

        # 6. Update P
        P = ((1-α) * P) + (α * Q)

    end

    # 7. Find the nearest permutation matrix with Hungarian algorithm
    matching = Hungarian.munkres(-P)
    P̂ = map(x -> max(0, x-1), Matrix(matching))

    # 8. Compute the matching associated to P̂
    matching = argmax(P̂', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = sortslices(matching, dims=1)

    # 9. Return P̂ and the associated matching
    return P̂, matching

end
