using Random
using Distributions
using LinearAlgebra


"""
Generate a pair of graphs from a ρ-correlated stochastic block model with all
of the parameters of the model specified.
"""
function generate_ρsbm(N::Int, Λ::Array{Float64,2}, b::Array{Int,1}, ρ::Float64)

    A = zeros(Int, N, N)
    B = zeros(Int, N, N)

    # sample graph A from a SBM
    for i in 1:N
        for j in i+1:N
            λ = Λ[b[i], b[j]]
            A[i,j] = rand(Bernoulli(λ))

            θ = ((1 - ρ) * λ) + (ρ * A[i,j])
            B[i,j] = rand(Bernoulli(θ))
        end
    end

    # make adjacency matricies symmetric
    A = A + A'
    B = B + B'

    # construct matching between graphs
    matching = hcat(1:N, 1:N)

    return A, B, matching
end

"""
Generate a pair of graphs from a ρ-correlated Erdős-Rényi model with
parameter p.
"""
function generate_erdosrenyi(N::Int, p::Float64, ρ::Float64)

    A = zeros(Int, N, N)
    B = zeros(Int, N, N)

    # sample graph A and B from a ρ-correlated Erdős-Rényi model
    for i in 1:N
        for j in i+1:N
            A[i,j] = rand(Bernoulli(p))

            λ = ((1 - ρ) * p) + (ρ * A[i,j])
            B[i,j] = rand(Bernoulli(λ))
        end
    end

    # make adjacency matricies symmetric
    A = A + A'
    B = B + B'

    # construct matching between graphs
    matching = hcat(1:N, 1:N)

    return A, B, matching
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters generated from uniform(0,1) on N vertices.
"""
function generate_bernoulli(N::Int, ρ::Float64)
    @assert 0 <= ρ <= 1

    P = zeros(Float64, N, N)
    dist = Uniform(0,1)
    for i in 1:N
        for j in i+1:N
            P[i,j] = rand(dist)
        end
    end
    P = P + P'

    A, B, matching = generate_bernoulli(P, ρ)
    return A, B, matching, P
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters generated from uniform(p-δ,p+δ) on N vertices.
"""
function generate_bernoulli(N::Int, p::Float64, δ::Float64, ρ::Float64)
    @assert 0 <= p <= 1
    @assert 0 <= δ <= min(p, 1-p)

    if δ == 0
        A, B, matching = generate_erdosrenyi(N, p, ρ)
        P = fill(p, (N, N)) - (Matrix(I, N, N) * p)
        return A, B, matching, P
    end

    P = zeros(Float64, N, N)
    for i in 1:N
        for j in i+1:N
            P[i,j] = rand(Uniform(p-δ,p+δ))
        end
    end
    P = P + P'

    A, B, matching = generate_bernoulli(P, ρ)
    return A, B, matching, P
end

"""
Generate a pair of graphs from a ρ-correlated Bernoulli model with
parameters p.
"""
function generate_bernoulli(p::Array{Float64,2}, ρ::Float64)

    @assert size(p, 1) == size(p, 2)
    N = size(p, 1)

    A = zeros(Int, N, N)
    B = zeros(Int, N, N)

    # sample graph A and B from a ρ-correlated Bernoulli model
    for i in 1:N
        for j in i+1:N
            A[i,j] = rand(Bernoulli(p[i,j]))

            λ = ((1 - ρ) * p[i,j]) + (ρ * A[i,j])
            B[i,j] = rand(Bernoulli(λ))
        end
    end

    # make adjacency matricies symmetric
    A = A + A'
    B = B + B'

    # construct matching between graphs
    matching = hcat(1:N, 1:N)

    return A, B, matching
end

"""
Generate a pair of graphs from a Bernoulli random graph model with
edge likelihood values Λ and edge correlation values R.
"""
function generate_hetero_bernoulli(Λ::Array{Float64,2}, R::Array{Float64,2})

    @assert size(Λ, 1) == size(Λ, 2)
    @assert size(R, 1) == size(R, 2)
    @assert size(Λ, 1) == size(R, 1)
    N = size(Λ, 1)

    A = zeros(Int, N, N)
    B = zeros(Int, N, N)

    # sample graph A and B from a ρ-correlated Bernoulli model
    for i in 1:N
        for j in i+1:N
            A[i,j] = rand(Bernoulli(Λ[i,j]))

            λ = ((1 - R[i,j]) * Λ[i,j]) + (R[i,j] * A[i,j])
            B[i,j] = rand(Bernoulli(λ))
        end
    end

    # make adjacency matricies symmetric
    A = A + A'
    B = B + B'

    # construct matching between graphs
    matching = hcat(1:N, 1:N)

    return A, B, matching
end

"""
Randomly permute graphB with no seeds.
"""
function permute(A::Array{Int,2}, B::Array{Int,2})
    N = size(A,1)

    perm = randperm(N)
    P = Matrix(I, N, N)[perm, :]
    matching = hcat(1:N, perm)

    B = P * B * P'

    return A, B, matching
end

"""
Randomly permute graphB while maintaining m seeds.
"""
function permute_seeded(A::Array{Int,2}, B::Array{Int,2}, m::Int)
    @assert m>=0 "Cannot have negative seeds."

    N = size(A,1)

    # seed
    seeds = sort(randperm(N)[1:m])
    nonseeds = sort(setdiff(1:N, seeds))
    perm = vcat(seeds, nonseeds)
    P = Matrix(I, N, N)[perm, :]
    A = P * A * P'
    B = P * B * P'

    # permute
    perm = vcat(1:m, shuffle(m+1:N))
    P = Matrix(I, N, N)[perm, :]
    B = P * B * P'
    matching = hcat(1:N, perm)

    return A, B, matching
end

"""
Randomly permute graphB while maintaining m seeds, with seed error rate r.
"""
function permute_seeded_with_errors(A::Array{Int,2}, B::Array{Int,2}, m::Int, r::Float64)
    @assert 0 ≦ r ≦ 1

    A, B, matching = permute_seeded(A, B, m)

    # generate errors
    e = round(Int, r * m)
    errs = shuffle(m-r+1:m)
    while any(errs .== collect(m-r+1:m))
        errs = shuffle(m-r+1:m)
    end

    # create errors in graph
    perm_err = vcat(1:m-r, errs, m+1:N)
    P_err = Matrix(I, N, N)[perm_err, :]
    B = P_err * B * P_err'

    errors = hcat(m-r+1:m, errs)
    return A, B, matching, errors
end

"""
Randomly sample a doubly stochastic matrix by iteratively normalizing the
rows and columns of a uniform random matrix.
"""
function sample_doublystochastic(N::Int; maxiter=50, ε::Float64=0.001)
    err = x -> sum(abs.(sum(x, dims=1) .- 1)) + sum(abs.(sum(x, dims=2) .- 1))

    M = rand(Float32, N, N)

    it = 0
    while err(M) > ε
        M = M ./ sum(M, dims=1)
        M = M ./ sum(M, dims=2)

        it += 1
        if (it == maxiter) break end
    end

    return M
end
