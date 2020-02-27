#!/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia
using Printf
using Statistics
using LinearAlgebra, ForwardDiff

include("util.jl")

################################################################################
################################################################################
## DATA STRUCTURES
## Data structures

# Encapsulates information about which vertices exist in the causal graph and
# their respective indices.

# struct VertexLabels
#   u1::Vector{Int}  # Latent variables, parents of colliders Z1 and outcome Y
#   u2::Vector{Int}  # Latent variables, parents of colliders Z1 and treatment X
#   w::Integer       # Instrument
#   x::Integer       # Treatment
#   y::Integer       # Outcome
#   z1::Vector{Int}  # Colliders between X and Y
#   z2::Vector{Int}  # Mediators between W and X
#   z3::Vector{Int}  # Confounders between X and Y
#   z4::Vector{Int}  # Independent causes of Y
#   topo_order::Vector{Int} # Topological order of corresponding DAG
# end

# Encapsulates the components of a linear model.

struct LinearProblem

  B::Matrix{Float64}            # Matrix of coefficients, B[j, i] is the coefficient
                                # corresponding to edge i -> j.

  Omega::Matrix{Float64}        # Covariance matrix of error terms.
  Sigma::Matrix{Float64}        # Implied covariance matrix.

  G::Matrix{Int}                # The graph represented as a binary matrix,
                                # G[j, i] == 1 iif edge i -> j exists.

  dat::Matrix{Float64}          # Simulated data.
  vertex_labels::VertexLabels   # `VertexLabels` data structure for this model.

end

# Encapsulates the components of a discrete model.

struct DiscreteProblem

  B::Matrix{Float64}            # Matrix of coefficients, B[j, i] is the coefficient
                                # corresponding to edge i -> j.

  Omega::Matrix{Float64}        # Covariance matrix of error terms.
  Sigma::Matrix{Float64}        # Implied covariance matrix.

  G::Matrix{Int}                # The graph represented as a binary matrix,
                                # G[j, i] == 1 iif edge i -> j exists.

  dat::Matrix{Float64}          # Simulated data.
  vertex_labels::VertexLabels   # `VertexLabels` data structure for this model.

end

################################################################################
################################################################################
## GAUSSIAN CASE

"""
    simulate_gaussian(n::Integer, p::Vector{Int};
                      min_noise = 0.2, max_noise = 0.8,
                      x_noise = 0.05, y_noise = 0.10,
                      prob_flip = 0.1, w_effect = 0.2, x_effect = 0.2)
Simulate model and data using sample size `n` and number of vertices given by
vector `p`. Error variances for each vertex are sampled from the uniform distribution in
`[min_noise, max_noise]`. The exception is treatment variable, where the
error variance is set to `x_noise`, and the outcome variable, where the error
variance is set to `y_noise`.
Coefficients are sampled independently and normalized so that each variable has
variance `1`. Two coefficients are set manually: the coefficient of instrument
on treatment is set by `w_effect`, while the coefficient of treatment on
outcome is set to `x_effect`. Finally, to avoid that confounding adds to up
to very small values due to averaging effects (which are more likely as the
entries of `p` grow), each vertex has causal effects on its children all drawn
with the same signal (which can be either positive or negative with equal
probability). Some variability is added by flipping the sign of each edge with
probability `prob_flip`. The closer to `0.5` this is, the higher the probability
that confounding and collider effects will be close to zero, so set it to a
small number in order to make problems harder.
"""
function simulate_gaussian(n::Integer, p::Vector{Int};
                           min_noise = 0.2, max_noise = 0.8, pseudo_pcount = 100,
                           x_noise = 0.05, y_noise = 0.10,
                           prob_flip = 0.1, w_effect = 0.2, x_effect = 0.2)
  z1 = collect(1:p[1])
  z2 = collect((p[1] + 1):(p[1] + p[2]))
  z3 = collect((p[1] + p[2] + 1):(p[1] + p[2] + p[3]))
  z4 = collect((p[1] + p[2] + p[3] + 1):(p[1] + p[2] + p[3] + p[4]))
  u1 = collect((p[1] + p[2] + p[3] + p[4] + 1):(p[1] + p[2] + p[3] + p[4] + p[5]))
  u2 = collect((p[1] + p[2] + p[3] + p[4] + p[5] + 1):(p[1] + p[2] + p[3] + p[4] + 2 * p[5]))
  w  = p[1] + p[2] + p[3] + p[4] + 2 * p[5] + 1
  x  = p[1] + p[2] + p[3] + p[4] + 2 * p[5] + 2
  y  = p[1] + p[2] + p[3] + p[4] + 2 * p[5] + 3
  num_vertices = y

  # Graph definition

  G = zeros(Int, num_vertices, num_vertices)
  G[z1, [w; u1; u2]] .= 1
  G[z2, w] .= 1
  for z2_idx = 2:length(z2) G[z2[z2_idx], z2[1:(z2_idx - 1)]] .= 1 end
  G[x, [z2; z3; u1]] .= 1
  G[y, [x; z3; z4; u2]] .= 1
  topo_order = [u1; u2; w; z3; z4; z1; z2; x; y]

  P = parameter_factory_gaussian_signs(G, prob_flip)

  B, Omega = zeros(num_vertices, num_vertices), zeros(num_vertices, num_vertices)
  Sigma = zeros(num_vertices, num_vertices)

  # Model parameters: exogeneous variables have standard Gaussian marginal
  # distributions. Endogenous variables have a noise variance sampled from
  # an uniform marginal (except if they are treatment X), with coefficients
  # normalized so that marginal variance is 1
  for i = topo_order

    parents = findall(G[i, :] .== 1)
    num_parents = length(parents)

    if num_parents == 0 # exogeneous
      Omega[i, i] = 1
      Sigma[i, i] = 1
    else
      coeffs = parameter_factory_gaussian(i, P)
      if i == x
        Omega[i, i] = x_noise
      elseif i == y
        Omega[i, i] = y_noise
      else
        #Omega[i, i] = rand() * (max_noise - min_noise) + min_noise
        Omega[i, i] = min(max_noise, max(min_noise, (pseudo_pcount - num_parents) / pseudo_pcount))
      end
      signal_var = 1 - Omega[i, i]
      if i == x || i == y
        if i == x
          p_pos = sum(G[i, 1:w])
          coeffs[p_pos] = w_effect
        else
          p_pos = sum(G[i, 1:x])
          coeffs[p_pos] = x_effect
        end
        new_coeffs = simulate_gaussian_adjust_effect(p_pos, coeffs,
                     Sigma[parents, parents], signal_var)
        B[i, parents] = new_coeffs

        Omega[i, i] = 1. - (B[i, parents]' * Sigma[parents, parents] * B[i, parents])
      else
        current_var = coeffs' * Sigma[parents, parents] * coeffs
        B[i, parents] = coeffs * sqrt(signal_var / current_var)
      end
      Sigma[:, i] = Sigma * B[i, :]
      Sigma[i, :] = Sigma[:, i]
      Sigma[i, i] = 1
    end

  end

  # At the end of this, it must be the case that
  # Sigma == ((I - B) \ Omega) / (I - B)'

  # Generate synthetic data
  vertex_labels = VertexLabels(u1, u2, w, x, y, z1, z2, z3, z4, topo_order)
  # verify: is that for the purpose of Monte Carlo Simulation, i.e. to make sure the resulting
  # dataset still retains the covariance matrix of Sigma,
  # as per https://en.wikipedia.org/wiki/Cholesky_decomposition#Monte_Carlo_simulation
  dat = randn(n, num_vertices) * cholesky(Sigma).U

  # Return

  return B, Omega, Sigma, G, dat, vertex_labels

end

"""
    simulate_gaussian_adjust_effect(fix_b, b, S, signal_var)
This function calculates a scaling factor `q` so that
`b_new' * Sigma_par * b_new ==  signal_var`, where `b_new[v] = b[v] * q` if `v`
is different from `fix_b`, and `b_new[fix_b] = b[fix_b]`. The idea is to adjust
the variance assigned to the signal of a random variable in a model by rescaling
the coefficients of its structural equations, while keeping one coefficient
(`b_new[fix_b]`) fixed.
"""
function simulate_gaussian_adjust_effect(fix_b, b, S, signal_var)

  not_fix_b = setdiff(1:length(b), fix_b)

  #f_at_input = q -> simulate_gaussian_adjust_effect_score(q, fix_b, not_fix_b, b, S, signal_var)
  #q_result = optimize(f_at_input, [log(1.)], LBFGS(); autodiff = :forward)
  #q = q_result.minimizer[1]

  aq = b[not_fix_b]' * S[not_fix_b, not_fix_b] * b[not_fix_b]
  bq = 2 * b[fix_b] * b[not_fix_b]' * S[not_fix_b, fix_b]
  cq = b[fix_b]^2 * S[fix_b, fix_b] - signal_var
  delta = bq^2 - 4 * aq * cq
  q = (-bq + sqrt(delta)) / (2 * aq)

  b_new = copy(b) * q
  b_new[fix_b] = b[fix_b]
  return b_new

end

"""
    parameter_factory_gaussian_signs(G, prob_flip)
Generates signs for the parameters for a Gaussian simulated model, where signs
are the same for all coefficients corresponding to the direct causal effect of
a variable into its children in graph `G`. However, we allow these signs to be
then individually flipped with probability `prob_flip`.
"""
function parameter_factory_gaussian_signs(G, prob_flip)
  num_vertices = size(G)[1]
  P = zeros(Int, num_vertices, num_vertices)
  for i = 1:num_vertices
    children = findall(G[:, i] .== 1)
    num_children = length(children)
    # Random.seed!(1234);
    s = 2 * (rand() < 0.5) - 1
    P[children, i] .= s
    flips = 2 * (rand(num_children) .> prob_flip) .- 1
    P[children, i] .*= flips
  end
  return P
end

"""
    parameter_factory_gaussian(v, P)
Generates parameters for a Gaussian simulated model for vertex `v`, where signs
must be according to the sign matrix `P`.
"""
function parameter_factory_gaussian(v, P)
  parents = findall(P[v, :] .!= 0)
  num_parents = length(parents)
  coeffs = abs.(randn(num_parents)) .* 10 .* P[v, parents]
  return coeffs
end