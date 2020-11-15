module GraphMatching

include("solvers/sgm.jl")
include("solvers/faq.jl")
include("solvers/exact.jl")

include("util/generators.jl")
include("util/metrics.jl")

export sgm
export faq
export solve_exact_linear, solve_exact_quadratic

export generate_erdosrenyi, generate_bernoulli, generate_ρsbm
export permute, permute_seeded, permute_seeded_with_errors

export match_ratio
export alignment_strength
export total_correlation

end
