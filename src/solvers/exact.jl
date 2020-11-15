using JuMP
using Convex
using Gurobi
using LinearAlgebra


"""
Finds an exact matching between two isomorphic graphs using integer progamming.
"""
function solve_iso(A::Array{Int,2}, B::Array{Int,2}, m::Int)

    n = size(A,1) - m

    if m != 0
        A = A[m+1:end, m+1:end]
        B = B[m+1:end, m+1:end]
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, P[1:n,1:n], Bin)

    @constraint(model, sum(P, dims=1) .== 1)
    @constraint(model, sum(P, dims=2) .== 1)

    @constraint(model, P*A*P' .== B)

    optimize!(model)
    status = Int(termination_status(model))

    if status != 1
        error("Non-isomorphic graphs")
    end

    P = Int.(value.(P))
    matching = argmax(P, dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = vcat(hcat(1:m, 1:m), matching .+ m)
    matching = sortslices(matching, dims=1)

    return P, matching
end

"""
Finds the best matching between any two graphs using linear integer progamming.
"""
function solve_exact_linear(A::Array{Int,2}, B::Array{Int,2}, m::Int)

    n = size(A,1) - m

    if m != 0
        A = A[m+1:end, m+1:end]
        B = B[m+1:end, m+1:end]
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, P[1:n,1:n], Bin)
    @variable(model, E₁[1:n,1:n], Bin)
    @variable(model, E₂[1:n,1:n], Bin)

    o = ones(Int, n)
    @constraint(model, P  * o .== o)
    @constraint(model, P' * o .== o)

    @constraint(model, (A*P) - (P*B) .== E₁ - E₂)

    @objective(model, Min, sum(E₁ + E₂))

    optimize!(model)
    status = Int(termination_status(model))

    if status != 1
        error("No optimizal solution found")
    end

    P = Int.(value.(P))
    matching = argmax(P', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = vcat(hcat(1:m, 1:m), matching .+ m)
    matching = sortslices(matching, dims=1)

    return P, matching
end

"""
Finds the best matching between any two graphs using quadratic integer progamming.
"""
function solve_exact_quadratic(A::Array{Int,2}, B::Array{Int,2}, m::Int)

    n = size(A,1) - m

    if m != 0
        A = A[m+1:end, m+1:end]
        B = B[m+1:end, m+1:end]
    end

    model = Model(Gurobi.Optimizer)
    set_silent(model)

    @variable(model, P[1:n,1:n], Bin)

    @constraint(model, sum(P, dims=1) .== 1)
    @constraint(model, sum(P, dims=2) .== 1)

    @objective(model, Max, tr(A * P * B' * P'))

    optimize!(model)
    status = Int(termination_status(model))

    if status != 1
        error("No optimal solution found")
    end

    P = Int.(value.(P))
    matching = argmax(P', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = vcat(hcat(1:m, 1:m), matching .+ m)
    matching = sortslices(matching, dims=1)

    return P, matching
end

"""
Finds the best matching between any two graphs using quadratic integer progamming.
"""
function solve_exact_quadratic_alt(A::Array{Int,2}, B::Array{Int,2}, m::Int)
    n = size(A,1) - m

    if m != 0
        A = A[m+1:end, m+1:end]
        B = B[m+1:end, m+1:end]
    end

    o = ones(Int, n)

    P = Variable((n, n), :Bin)
    problem = minimize(sumsquares(A*P - P*B))
    problem.constraints += P * o == o
    problem.constraints += P' * o == o

    solve!(problem, Gurobi.Optimizer(OutputFlag=0))

    P̂ = Int.(P.value)
    matching = argmax(P̂', dims=2)
    matching = hcat(getindex.(matching,1), getindex.(matching,2))
    matching = vcat(hcat(1:m, 1:m), matching .+ m)
    matching = sortslices(matching, dims=1)

    return P̂, matching
end
