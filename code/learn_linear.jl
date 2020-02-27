#!/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia
################################################################################
## learn_linear.jl
##
## Learning algorithms on linear models.
################################################################################

using ForwardDiff
using PenaltyFunctions
using Gadfly
using Random

include("util.jl")

################################################################################
################################################################################
## DATA STRUCTURES
## Data structures

mutable struct Betas # data sructure to contain weights returned from objective_learn_linear

  beta::Vector{Float64} # abs \rho(W,Y|\phi, X)

end

mutable struct Grads # data sructure to contain gradients returned from objective_learn_linear

  beta::Vector{Float64} # abs \rho(W,Y|\phi, X)

  Grads() = new()
end

mutable struct Objectives # data sructure to contain gradients returned from objective_learn_linear

  f::Float64

  Objectives() = new() 
end

struct Z_labels

  Z::Vector{Int}
end

mutable struct Updates

  beta::Vector{Float64}

end

################################################################################
################################################################################

"""
    bd_learn_linear(Sigma::Matrix{Float64},
                    lambda1::Number, lambda2::Number,
                    vertex_labels::VertexLabels;
                    max_iter = 200, rho = 0.01, verbose = false)
Learn a covariate adjustment for a linear model with covariance matrix `Sigma`.
Penalty `lambda1` is to enforce positivity of the association of instrument and
outcome without conditioning on treatment. Penalty `lambda2` enforces sparsity
on the construction of the covariate adjustment. Data structure `vertex_labels`
indicates which variables in `Sigma` are the instrument, treatment, outcome and
covariates. The algorithm runs for a fixed pre-defined number of iterations
`max_iter` using the Adam optimization algorithm with parameter `rho`. Set
`verbose` to `true` to see printed messages at each step.
"""
function bd_learn_linear(Sigma::Matrix{Float64},
                         lambda1::Number, lambda2::Number,
                         vertex_labels::VertexLabels, num_trial::Number, path_to_file::String;
                         max_iter = 200, lr = 0.01, verbose = false)

  Random.seed!(num_trial);

  w, x, y = vertex_labels.w, vertex_labels.x, vertex_labels.y

  Z = [vertex_labels.z1; vertex_labels.z2; vertex_labels.z3; vertex_labels.z4]
  z_labels = Z_labels(Z)
  
  eta = randn(length(Z))
  l2 = L2Penalty()
  beta = eta / l2(eta)
  betas = Betas(beta)

  beta_hat = copy(beta)
  num_params = length(beta)
  
  vs = Updates(zeros(Float64, num_params))

  objs = zeros(Float64, max_iter)
  objs_sparsity = zeros(Float64, max_iter)
  Ds_xz = zeros(Float64, max_iter)
  Ds_xz_sparsity = zeros(Float64, max_iter)
  regs = zeros(Float64, max_iter)
  regs_sparsity = zeros(Float64, max_iter)
  best_f = -Inf
  best_f_sparsity = -Inf
  corr_px_beta_hat = -Inf 
  corr_p_beta_hat = -Inf
  
  if verbose println("\n* START OPTIMIZATION *") end
  γ = 0.9
  α = lr
  for iter = 1:max_iter
    fs, grads, corrs_diagnost, reg_beta = 
      learn_linear_fg(betas, eta, lambda1, lambda2, w, y, x, z_labels, Sigma)

    objs[iter] = fs.f
    Ds_xz[iter] = corrs_diagnost.corr_px
    regs[iter] = reg_beta
    if fs.f > best_f
      beta_hat = copy(betas.beta)
      corr_px_beta_hat = copy(corrs_diagnost.corr_px)
      corr_p_beta_hat = copy(corrs_diagnost.corr_p)
      best_f = fs.f
    end

    eta, vs.beta = sgd_momentum(eta, grads.beta, 
      vs.beta, iter, γ, α)
    betas.beta = eta / l2(eta)
    
    if verbose println("[Iteration $iter out of $max_iter] Objective: $f_minus") end
  end

  # induce sparsity in learned weights by masking out terms close to zero (defined through thresh)
  mask = ones(length(Z))
  thresh = 1e-3
  small_ind = findall(x->abs(x)<thresh, betas.beta)
  mask[small_ind] .= 0
  eta[small_ind] .= 0
  
  # cotinue training after mask applied
  for iter = 1:trunc(Int,(max_iter/10))
    fs, grads, corrs_diagnost, reg_beta = 
      learn_linear_fg(betas, eta, lambda1, lambda2, w, y, x, z_labels, Sigma)

    objs_sparsity[iter] = fs.f
    Ds_xz_sparsity[iter] = corrs_diagnost.corr_px
    regs_sparsity[iter] = reg_beta
    if fs.f > best_f_sparsity
      beta_hat = copy(betas.beta)
      corr_px_beta_hat = copy(corrs_diagnost.corr_px)
      corr_p_beta_hat = copy(corrs_diagnost.corr_p)
      best_f_sparsity = fs.f
    end
    eta, vs.beta = sgd_momentum(eta, grads.beta.*mask, 
      vs.beta, iter, γ, α)
    betas.beta = eta / l2(eta)
    
    if verbose println("[Iteration $iter out of $max_iter] Objective: $f_minus") end
  end

  return beta_hat, corr_px_beta_hat, corr_p_beta_hat

end

"""
    objective_learn_linear(betas, lambda1, lambda2, w, y, x, Zs, Sigma)
Objective function of linear model search to be minimized. Parameter vector
`beta` encodes which function of `Z` to condition on, while `lambda1` is the
penalty for enforcing positivity of the association of instrument and outcome
without conditioning on treatment. Penaly `lambda2` enforces sparsity. Finally,
`Sigma` is the (estimated) covariance matrix of the system.
"""
function objective_learn_linear(betas::Betas, eta, lambda1, lambda2, w, y, x, Zs::Z_labels, Sigma)
  corrs = ReturnFromGradientDiagnost()
  fs = Objectives()
  
  Sigma_wyx_phi = build_phi_covariance(betas.beta, w, y, x, Zs.Z, Sigma)

  corrs.corr_px = abs(partial_corr([1; 2], [3; 4], Sigma_wyx_phi)[1, 2])

  corrs.corr_p = abs(partial_corr([1; 2], [4], Sigma_wyx_phi)[1, 2])

  p1 = L1Penalty()
    
  fs.f = -(corrs.corr_px - lambda1 * corrs.corr_p + lambda2 * p1(betas.beta))

  return fs, corrs, p1(betas.beta)
end

"""
    learn_linear_fg(beta, lambda1, lambda2, w, y, x, Z, Sigma)
Encapsulation of the objective function `objective_learn_linear` along with
its gradient computed by forward automatic differentiation.
"""
function learn_linear_fg(betas, eta, lambda1, lambda2, w, y, x, Zs, Sigma)
  grads = Grads()

  fs, corrs, reg_beta = 
    objective_learn_linear(betas, eta, lambda1, lambda2, w, y, x, Zs, Sigma)

  g_eval = beta_0 -> ForwardDiff.gradient(beta ->
    objective_learn_linear(beta, lambda1, lambda2, w, y, x, Zs.Z, Sigma), eta);
  grads.beta = g_eval(betas.beta)

  return fs, grads, corrs, reg_beta
end

"""
    objective_learn_linear(beta, lambda1, lambda2, w, y, x, Z, Sigma)
Objective function of linear model search to be minimized. Parameter vector
`beta` encodes which function of `Z` to condition on, while `lambda1` is the
penalty for enforcing positivity of the association of instrument and outcome
without conditioning on treatment. Penaly `lambda2` enforces sparsity. Finally,
`Sigma` is the (estimated) covariance matrix of the system.
"""
function objective_learn_linear(beta, lambda1, lambda2, w, y, x, Z, Sigma)
  Sigma_wyx_phi = build_phi_covariance(beta, w, y, x, Z, Sigma)
  # try 
  corr_px = abs(partial_corr([1; 2], [3; 4], Sigma_wyx_phi)[1, 2])
  corr_p = abs(partial_corr([1; 2], [4], Sigma_wyx_phi)[1, 2])

  p1 = L1Penalty()
  f = -(corr_px - lambda1 * corr_p + lambda2 * p1(beta))
  return f
end

"""
    build_phi_covariance(beta, w, y, x, Z, Sigma)
Encapsulation of the objective function `objective_learn_linear` along with
its gradient computed by forward automatic differentiation.
"""
function build_phi_covariance(beta, w, y, x, Z, Sigma)
  wyx = [w; y; x]
  S_wyx_phi = Sigma[wyx, [wyx; Z]] * [0; 0; 0; beta]
  S_phi_phi = beta' * Sigma[Z, Z] * beta
  Sigma_wyx_phi = [Sigma[wyx, wyx] S_wyx_phi; S_wyx_phi' S_phi_phi]
  return Sigma_wyx_phi
end