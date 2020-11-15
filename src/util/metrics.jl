"""
Compute the accuracy of an estimated graph matching relative to the canonical
matching, excluding seeeds.
"""
function match_ratio(true_match::Array{Int,2}, est_match::Array{Int,2}, m::Int=0)

    true_match = sortslices(true_match, dims=1)
    est_match  = sortslices(est_match, dims=1)
    @assert true_match[:,1] == est_match[:,1]

    n = size(true_match, 1)
    acc = sum(true_match[m+1:end,2] .== est_match[m+1:end,2]) / (n - m)
    return acc

end

"""
Compute the accuracy of an estimated graph matching relative to the canonical
graph matching, excluding seeds and errors.
"""
function match_ratio(true_match::Array{Int,2}, est_match::Array{Int,2}, m::Int, errors::Array{Int,2})

    ignore = union(1:m, errors[:,1], errors[:,2])
    keepind = setdiff(1:size(true_match, 1), ignore)

    true_match = sortslices(true_match, dims=1)
    est_match  = sortslices(est_match, dims=1)
    @assert true_match[:,1] == est_match[:,1]

    true_match = true_match[keepind,2]
    est_match  = est_match[keepind,2]

    acc = sum(true_match .== est_match) / length(true_match)
    return acc

end

"""
Compute the alignment strength of a given graph matching, excluding seeds.
"""
function alignment_strength(A::Array{Int,2}, B::Array{Int,2}, P::Array{Int,2}, m::Int=0, endbehavior::Symbol=:zero)
    A = A[m+1:end,m+1:end]
    B = B[m+1:end,m+1:end]
    P = P[m+1:end,m+1:end]

    N = size(A,1)
    Nc2 = N * (N-1) / 2

    d = sum(abs.(A*P - P*B)) / 2
    ∂A = (sum(A)/2) / Nc2
    ∂B = (sum(B)/2) / Nc2
    denom = Nc2 * ((∂A * (1 - ∂B)) + (∂B * (1 - ∂A)))

    if denom ≈ 0.0
        if endbehavior == :zero
            return 0.0
        elseif endbehavior == :one
            return 1.0
        end
    end

    str = 1 - (d / denom)
    return str
end

"""
Compute the quadratic objectve function value ||AP - PB||₂² of a given graph
matching.
"""
function qap_objective(A::Array{Int,2}, B::Array{Int,2}, P::Array{Int,2})
    return norm(A*P - P*B, 2)^2
end

"""
Compute the linear objectve function value ||AP - PB||₁ of a given graph
matching.
"""
function l1_objective(A::Array{Int,2}, B::Array{Int,2}, P::Array{Int,2})
    return sum(abs.(A*P - P*B))
end

"""
Compute the indefinite relaxation objectve function value -Trace(PᵀAᵀPB) of a
given graph matching.
"""
function indef_objective(A::Array{Int,2}, B::Array{Int,2}, P::Array{Int,2})
    return -tr(P'*A'*P*B)
end

"""
Compute the total correlation between two graphs, given the matrix of edge
likelihoods used to generate them and the correlation between them.
"""
function total_correlation(p::Array{Float64, 2}, ρe::Float64)
    N = size(p, 1)

    μ = sum(triu(p,1)) / (N * (N-1) / 2)
    σ² = sum(triu(p .- μ,1).^2) / (N * (N-1) / 2)
    ρh = σ² / (μ * (1-μ))

    ρtotal = 1 - ((1-ρh) * (1-ρe))
    return ρtotal
end

"""
Compute the total correlation between two graphs generated from a ρe-correlated
Bernoulli random graph model with parameters generated from Uniform[p-δ, p+δ].
"""
function total_correlation(p::Float64, δ::Float64, ρe::Float64)
    ρh = δ^2 / (3 * p * (1-p))
    ρtotal = 1 - ((1-ρh) * (1-ρe))
    return ρtotal
end
