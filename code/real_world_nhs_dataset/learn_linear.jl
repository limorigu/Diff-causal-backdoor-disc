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
# using Cairo
# using Fontconfig

include("util.jl")

################################################################################
################################################################################
## DATA STRUCTURES
## Data structures

mutable struct RmvDiagnost # data sructure to contain all results of diagnostic runs
  
  initial_corr_px::Float64 # abs \rho(W,Y|\phi,X) pre-optimization
  final_corr_px::Float64 # abs \rho(W,Y|\phi,X) post-optimization

  RmvDiagnost() = new()
end

mutable struct ReturnFromGradientDiagnost # data sructure to contain all results of diagnostic runs

  corr_px::Float64 # abs \rho(W,Y|\phi, X)
  corr_p::Float64 # abs \rho(W,Y|\phi)

  ReturnFromGradientDiagnost() = new()
end

mutable struct Thetas # data sructure to contain weights returned from objective_learn_linear

  theta::Vector{Float64} # abs \rho(W,Y|\phi, X)

end

mutable struct Grads # data sructure to contain gradients returned from objective_learn_linear

  theta::Vector{Float64} # abs \rho(W,Y|\phi, X)

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

  theta::Vector{Float64}

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
                         max_iter = 200, lr = 0.01, verbose = false, corr_boost)

  w, x, y = vertex_labels.w, vertex_labels.x, vertex_labels.y

  Z = [vertex_labels.z1; vertex_labels.z3]
  z_labels = Z_labels(Z)
  
  eta = randn(length(Z))
  l2 = L2Penalty()
  theta = eta / l2(eta)
  thetas = Thetas(theta)

  theta_hat = copy(theta)
  num_params = length(theta)
  
  vs = Updates(zeros(Float64, num_params))

  objs = zeros(Float64, max_iter)
  objs_sparsity = zeros(Float64, max_iter)
  Ds_xz = zeros(Float64, max_iter)
  Ds_xz_sparsity = zeros(Float64, max_iter)
  regs = zeros(Float64, max_iter)
  regs_sparsity = zeros(Float64, max_iter)
  best_f = -Inf
  best_f_sparsity = -Inf
  corr_px_theta_hat = -Inf 
  corr_p_theta_hat = -Inf
  

  if verbose println("\n* START OPTIMIZATION *") end
  γ = 0.9
  α = lr
  for iter = 1:max_iter
    fs, grads, corrs_diagnost, reg_theta = 
      learn_linear_fg(corr_boost, thetas, eta, lambda1, lambda2, w, y, x, z_labels, Sigma)

    objs[iter] = fs.f
    Ds_xz[iter] = corrs_diagnost.corr_px
    regs[iter] = reg_theta
    if fs.f > best_f
      theta_hat = copy(thetas.theta)
      corr_px_theta_hat = copy(corrs_diagnost.corr_px)
      corr_p_theta_hat = copy(corrs_diagnost.corr_p)
      best_f = fs.f
    end
    f_minus = -fs.f

    eta, vs.theta = sgd_momentum(eta, grads.theta, 
      vs.theta, iter, γ, α)
    thetas.theta = eta / l2(eta)
    
    if verbose println("[Iteration $iter out of $max_iter] Objective: $f_minus") end
  end

  # draw(SVG(path_to_file*"outputfiles/thetas_histo_trial=$num_trial lambda1=$lambda1 lambda2=$lambda2 lr=$lr.svg", 6inch, 3inch), 
    # plot(x=thetas.theta, Geom.histogram))
  mask = ones(length(Z))
  norm = L2Penalty()
  # thresh = 0.01*maximum(norm(thetas.theta))
  thresh = 1e-3
  small_ind = findall(x->abs(x)<thresh, thetas.theta)
  mask[small_ind] .= 0
  eta[small_ind] .= 0
    # sparsity inducing
  for iter = 1:trunc(Int,(max_iter/10))
    fs, grads, corrs_diagnost, reg_theta = 
      learn_linear_fg(corr_boost, thetas, eta, lambda1, lambda2, w, y, x, z_labels, Sigma)

    objs_sparsity[iter] = fs.f
    Ds_xz_sparsity[iter] = corrs_diagnost.corr_px
    regs_sparsity[iter] = reg_theta
    if fs.f > best_f_sparsity
      theta_hat = copy(thetas.theta)
      corr_px_theta_hat = copy(corrs_diagnost.corr_px)
      corr_p_theta_hat = copy(corrs_diagnost.corr_p)
      best_f_sparsity = fs.f
    end
    # f_minus = -fs.f
    eta, vs.theta = sgd_momentum(eta, grads.theta.*mask, 
      vs.theta, iter, γ, α)
    thetas.theta = eta / l2(eta)

    
    if verbose println("[Iteration $iter out of $max_iter] Objective: $f_minus") end
  end


  #result = target_learn_linear(theta_hat, w, y, x, Z, Sigma)
  # open(path_to_file*"outputfiles/loss_trial=$num_trial lambda1=$lambda1 lambda2=$lambda2 lr=$lr.txt", "a") do file
    # write(file, "newRun\n")
    # write(file, repr(objs), "\n")
  # end

  # draw(SVG(path_to_file*"outputfiles/loss_trial=$num_trial lambda1=$lambda1 lambda2=$lambda2 lr=$lr.svg", 6inch, 3inch), 
    # plot(x=range(1,stop=length(objs)), y=objs, Geom.line))

  # draw(SVG(path_to_file*"outputfiles/Dependence_trial=$num_trial lambda1=$lambda1 lambda2=$lambda2 lr=$lr.svg", 6inch, 3inch), 
    # plot(x=range(1,stop=length(Ds_xz)), y=Ds_xz, Geom.line))

  # draw(SVG(path_to_file*"outputfiles/regs_trial=$num_trial lambda1=$lambda1 lambda2=$lambda2 lr=$lr.svg", 6inch, 3inch), 
    # plot(x=range(1,stop=length(regs)), y=regs, Geom.line))

  return theta_hat, corr_px_theta_hat, corr_p_theta_hat

end

"""
    objective_learn_linear(corr_boost, thetas, lambda1, lambda2, w, y, x, Zs, Sigma)
Objective function of linear model search to be minimized. Parameter vector
`theta` encodes which function of `Z` to condition on, while `lambda1` is the
penalty for enforcing positivity of the association of instrument and outcome
without conditioning on treatment. Penaly `lambda2` enforces sparsity. Finally,
`Sigma` is the (estimated) covariance matrix of the system.
"""
function objective_learn_linear(corr_boost, thetas::Thetas, eta, lambda1, lambda2, w, y, x, Zs::Z_labels, Sigma)
  corrs = ReturnFromGradientDiagnost()
  fs = Objectives()
  
  Sigma_wyx_phi = build_phi_covariance(thetas.theta, w, y, x, Zs.Z, Sigma)

  corrs.corr_px = abs(partial_corr([1; 2], [3; 4], Sigma_wyx_phi)[1, 2])

  corrs.corr_p = abs(partial_corr([1; 2], [4], Sigma_wyx_phi)[1, 2])

  p1 = L1Penalty()
  # thetas.theta[findall(x->abs(x)<1e-3, thetas.theta)] .= 0
    
  fs.f = -(corr_boost * corrs.corr_px - lambda1 * corrs.corr_p + lambda2 * p1(thetas.theta))

  return fs, corrs, p1(thetas.theta)
end

"""
    learn_linear_fg(theta, lambda1, lambda2, w, y, x, Z, Sigma)
Encapsulation of the objective function `objective_learn_linear` along with
its gradient computed by forward automatic differentiation.
"""
function learn_linear_fg(corr_boost, thetas, eta, lambda1, lambda2, w, y, x, Zs, Sigma)
  grads = Grads()

  fs, corrs, reg_theta = 
    objective_learn_linear(corr_boost, thetas, eta, lambda1, lambda2, w, y, x, Zs, Sigma)

  g_eval = theta_0 -> ForwardDiff.gradient(theta ->
    objective_learn_linear(corr_boost, theta, lambda1, lambda2, w, y, x, Zs.Z, Sigma), eta);
  grads.theta = g_eval(thetas.theta)

  return fs, grads, corrs, reg_theta
end

"""
    objective_learn_linear(theta, lambda1, lambda2, w, y, x, Z, Sigma)
Objective function of linear model search to be minimized. Parameter vector
`theta` encodes which function of `Z` to condition on, while `lambda1` is the
penalty for enforcing positivity of the association of instrument and outcome
without conditioning on treatment. Penaly `lambda2` enforces sparsity. Finally,
`Sigma` is the (estimated) covariance matrix of the system.
"""
function objective_learn_linear(corr_boost, theta, lambda1, lambda2, w, y, x, Z, Sigma)
  Sigma_wyx_phi = build_phi_covariance(theta, w, y, x, Z, Sigma)
  # try 
  corr_px = partial_corr([1; 2], [3; 4], Sigma_wyx_phi)[1, 2] 
  corr_p = partial_corr([1; 2], [4], Sigma_wyx_phi)[1, 2]

  p1 = L1Penalty()
  # theta[findall(x->abs(x)<1e-3, theta)] .= 0
  f = -(corr_boost * abs(corr_px) - lambda1 * abs(corr_p) + lambda2 * p1(theta))
  return f
end

"""
    build_phi_covariance(theta, w, y, x, Z, Sigma)
Encapsulation of the objective function `objective_learn_linear` along with
its gradient computed by forward automatic differentiation.
"""
function build_phi_covariance(theta, w, y, x, Z, Sigma)
  wyx = [w; y; x]
  S_wyx_phi = Sigma[wyx, [wyx; Z]] * [0; 0; 0; theta]
  S_phi_phi = theta' * Sigma[Z, Z] * theta
  Sigma_wyx_phi = [Sigma[wyx, wyx] S_wyx_phi; S_wyx_phi' S_phi_phi]
  return Sigma_wyx_phi
end

function target_learn_linear(theta, w, y, x, Z, Sigma)
  num_Z = length(Z)
  wyx = [w; y; x]
  S_wyx_phi = Sigma[wyx, [wyx; Z]] * [0; 0; 0; theta]
  S_phi_phi = theta' * Sigma[Z, Z] * theta
  Sigma_wyx_phi = [Sigma[wyx, wyx] S_wyx_phi; S_wyx_phi' S_phi_phi]

  corr_px = partial_corr([1; 2], [3; 4], Sigma_wyx_phi)[1, 2]
  corr_p = partial_corr([1; 2], [4], Sigma_wyx_phi)[1, 2]

  f = [corr_px; corr_p]

end