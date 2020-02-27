#!/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia
################################################################################
## util.jl
##
## This file contains general function to aid optimization and modeling.
################################################################################
using Statistics
using Distributions
using Optim
# using DataFrames, GLM
using ForwardDiff

"""
    adam(w, g, t, m, v, rho)
Perform one step of ADAM optimization with respect to parameter
`w`, with gradient `g`, at step count `t`, with statistics
`m` and `v`, and learning rate `rho`.
"""
function adam(w::Vector{Float64}, g::Vector{Float64}, t::Int,
              m::Vector{Float64}, v::Vector{Float64}, rho::Float64)
  beta_1 = 0.9;
  beta_2 = 0.999;
  m = beta_1 * m + (1 - beta_1) * g;
  v = beta_2 * v + (1 - beta_2) * g.^2;
  m_hat = m / (1 - beta_1^t);
  v_hat = v / (1 - beta_2^t);
  w += rho * m_hat ./ (sqrt.(v_hat) .+ 1.e-8);
  return(w, m, v)
end

"""
    sgd_momentum(w, g, v, t, γ, α)
Perform one step of ADAM optimization with respect to parameter
`w`, with gradient `g`, at step count `t`, with previous update 'v',
 with momentum 'γ', and learning rate `α`.
"""
function sgd_momentum(w::Vector{Float64}, g::Vector{Float64}, v::Vector{Float64}, 
  t::Int, γ::Float64, α::Float64)
  v_t = γ*v + α*g;
  w += v_t;
  return(w, v_t)
end

"""
    sigmoid(x)
Compute the sigmoid function at `x`.
"""
sigmoid(x) = 1. / (1. + exp(-x))

"""
    probit(x)
Compute the probit function at `x`.
"""
probit(x) = 1 .- cdf.(Normal(), x)

"""
    get_features(dat, col_sel)
Transforms columns `col_sel` of dataset `dat` of raw inputs into a parametric
feature space. Currently, this consists of main effects and pairwise
interactions.
"""
function get_features(dat::Matrix{T}, col_sel::Vector{Int}) where T <: Number
  n = size(dat)[1]
  num_cols = length(col_sel)
  num_features = Int(num_cols * (num_cols + 1) / 2) + 1
  features = Matrix{T}(undef, n, num_features)
  features[:, 1:num_cols] = dat[:, col_sel]
  pos = num_cols
  for j = 1:num_cols
    for k = (j + 1):num_cols
      pos += 1
      features[:, pos] = dat[:, col_sel[j]] .* dat[:, col_sel[k]]
    end
  end
  features[:, end] .= 1
  return features
end

function get_features_linear(dat::Matrix{T}, col_sel::Vector{Int}) where T <: Number
  n = size(dat)[1]
  num_cols = length(col_sel)
  num_features = num_cols + 1
  features = Matrix{T}(undef, n, num_features)
  features[:, 1:num_cols] = dat[:, col_sel]
  features[:, end] .= 1
  return features
end

function get_features_linear(num_cols::Integer)
  return num_cols + 1
end

"""
    get_features(num_cols)
Returns the number of features that will be generated when an input of size
`num_cols` is provided.
"""
function get_features(num_cols::Integer)
  return Int(num_cols * (num_cols + 1) / 2) + 1
end

"""
    get_features_intercept(num_cols)
Returns the position of the intercept feature when an input of size
`num_cols` is provided.
"""
function get_features_intercept(num_cols::Integer)
  return get_features(num_cols)
end

"""
    get_features_prob(dat, col_sel)
Given a Monte Carlo sample `dat` of the model, returns the marginal probability
of each feature being equal to `1`, as given by `col_sel`.
"""
function get_features_prob(dat::Matrix{T}, col_sel::Vector{Int}) where T <: Number
  features = get_features(dat, col_sel)
  features_prob = mean(features, dims = 1)
  return features_prob
end

"""
    partial_corr(X, Z, Sigma)
Partial correlation matrix of `X` given `Z` according to
covariance matrix `Sigma`.
"""
function partial_corr(X::Vector, Z::Vector, Sigma::Matrix)
  F = Sigma[X, Z] / cholesky(Sigma[Z, Z]).U
  cov_p = Sigma[X, X] - F * F'
  dcov = sqrt.(diag(cov_p))
  corr_p = cov_p ./ (dcov * dcov')
  return corr_p
end


"""
    partial_corr(x1, x2, z::Integer, R::Matrix{Float64})
Partial correlation matrix of variables `x1` and `x2` given variable `z`
and correlation matrix `R`.
"""
function partial_corr(x1::Integer, x2::Integer, z::Integer, R::Matrix{Float64})
  return (R[x1, x2] - R[x1, z] * R[x2, z]) / sqrt((1 - R[x1, z]^2) * (1 - R[x2, z]^2))
end

"""
    cov2corr(Sigma)
Returns the correlation matrix corresponding to covariance matrix `Sigma`.
"""
function cov2corr(Sigma)
  dcov = sqrt.(diag(Sigma))
  corr_p = Sigma ./ (dcov * dcov')
  return corr_p
end

"""
    z_trans(part_corr)
Returns the Fisher's Z transformation for some partial correlation part_corr.
"""
function z_trans(D)
  return 0.5*log((1. + D)/(1. - D))
end


"""
    ind_null_hypo(Sigma)
Checks the null hypothesis that true partial correlation XY.Z is 0. Returns false if hypothesis reject, true otherwise.
"""
function ind_null_hypo(n, num_Z, D_1; significance_level=0.01)
  D_1_trans = z_trans(D_1)
  return sqrt(n-num_Z-3)*abs(D_1_trans) > quantile(Normal(), 1-significance_level/2)
end

"""
    sigmoid(x)
return Sigmoid function applied to some input x. 
"""
function sigmoid(x)
    return 1. ./ (1. .+ exp.(-x))
end 

"""
    log_likelihood(x)
return *negative* log likelihood, which serves as our logsitic model objective function
"""
function log_likelihood(h, y)
  return sum(-y .* log.(h)
                .- ((1 .- y) .* log.(1 .- h)))   
end

# function logistic_cost_gradient(θ, X, y)
#     # m = length(y)
#     return (θ::Array) -> begin 
#         h = sigmoid(X * θ)   
#         J = log_likelihood(h,y)
#     end, (θ::Array, storage::Array) -> begin  
#         h = sigmoid(X * θ) 
#         storage[:] = (X' * (h .- y))       
#     end
# end

function logistic_regression_Optim(w::Vector{Float64}, x::Vector{Float64}, 
  phi::Vector{Float64}, y::Vector{Float64})
  X = [ones((length(w),1)) w x phi] #add x_0=1.0 column; now X size is (m,d+1)
  initialθ = zeros(size(X,2),1) #initialTheta size is (d+1, 1)

  # cost, gradient! = logistic_cost_gradient(initialθ, X, y)
  println("cost is ")
  println(cost)
  println("gradient is ")
  println(gradient!)
  res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);

  # cost = logistic_cost_gradient(initialθ, X, y)
  # res = optimize(cost, initialθ, autodiff=true, method=LBFGS());
  # res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS());
  θ = Optim.minimizer(res);
  return sigmoid(X * θ)
end

function logistic_regression_Optim(x::Vector{Float64}, phi::Matrix{Float64}, w::Vector{Float64})
  X = [ones(length(w)) x phi] #add x_0=1.0 column; now X size is (m,d+1)
  initialθ = zeros(size(X,2),1) 
  # cost, gradient! = logistic_cost_gradient(initialθ, X, w)
  println("cost is ")
  println(cost)
  println("gradient is ")
  println(gradient!)
  # res = optimize(θ -> logistic_objective(θ, X, w), gradient!, initialθ, method = LBFGS());
  res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);
  θ = Optim.minimizer(res);
  return sigmoid(X * θ)
end

function logistic_objective(θ, X, w)
    h = sigmoid(X*θ)
    return log_likelihood(h,w)  # Use SSE, non-standard for log. reg.
end

# function logistic_gradient_w_storgae(θ, X, w, storage)
#   h = sigmoid(X * θ) 
#   storage[:] = (X' * (h .- w)) 
#   return storage
# end

# function g!(θ, X, w, storage)
#   h = sigmoid(X * θ) 
#   storage[:] = (X' * (h .- w)) 
# end

# function logistic_gradient(θ, X, w)
#   h = sigmoid(X * θ) 
#   return (X' * (h .- w)) 
# end

function logistic_regression_Optim_get_coeffs(x::Vector{Float64}, phi::Matrix{Float64}, w::Vector{Float64})
  # println("size(ones(length(w))), size(x), size(phi)")
  # println(size(ones(length(w))), size(x), size(phi))
  X = [ones(length(w)) x phi] #add x_0=1.0 column; now X size is (m,d+1)
  initialθ = zeros(size(X,2),1) 
  res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);
  # cost, gradient! = logistic_cost_gradient(initialθ, X, w)
  # res = optimize(cost, gradient!, initialθ, method = LBFGS());
  θ = Optim.minimizer(res);
  return sigmoid(X * θ), θ
end


# function logistic_regression_Optim_get_coeffs_check(x::Vector{Float64}, phi::Matrix{Float64}, w::Vector{Float64})
#   # println("size(ones(length(w))), size(x), size(phi)")
#   # println(size(ones(length(w))), size(x), size(phi))
#   X = [ones(length(w)) x phi] #add x_0=1.0 column; now X size is (m,d+1)
#   initialθ = zeros(size(X,2),1) 
#   # g = θ -> ForwardDiff.gradient(logistic_objective, θ);
#   res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);
#   # cost, gradient! = logistic_cost_gradient(initialθ, X, w)
#   # res = optimize(cost, gradient!, initialθ, method = LBFGS());
#   θ = Optim.minimizer(res);
#   return sigmoid(X * θ), θ
# end
# function logistic_regression_Optim_get_coeffs(phi::Vector{Float64}, Z_select::Matrix{Float64}, w::Vector{Float64})
#   X = [ones(length(w)) x phi] #add x_0=1.0 column; now X size is (m,d+1)
#   initialθ = zeros(size(X,2),1) 

#   res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS());
#   θ = Optim.minimizer(res);
#   return sigmoid(X * θ), θ
# end

function logistic_regression_Optim_get_coeffs(phi::Vector{Float64}, w::Vector{Float64})
  X = [ones(length(w)) phi] #add x_0=1.0 column; now X size is (m,d+1)
  initialθ = zeros(size(X,2),1) 
  # cost, gradient! = logistic_cost_gradient(initialθ, X, w)
  # res = optimize(cost, gradient!, initialθ, method = ConjugateGradient(), iterations = 1000);
  # res = optimize(cost, gradient!, initialθ, Newton(); autodiff = :forward);
  # res = optimize(cost, gradient!, initialθ, LBFGS());
  res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);
  # cost, gradient! = logistic_cost_gradient(initialθ, X, y)
  # res = optimize(cost, gradient!, initialθ, method = LBFGS());
  θ = Optim.minimizer(res);
  # println("num of iterations from phi w: ", Optim.iterations(res))
  return sigmoid(X * θ), θ
end

# function logistic_regression_Optim_get_coeffs_check(phi::Vector{Float64}, w::Vector{Float64})
#   X = [ones(length(w)) phi] #add x_0=1.0 column; now X size is (m,d+1)
#   initialθ = zeros(size(X,2),1) 
#   # cost, gradient! = logistic_cost_gradient(initialθ, X, w)
#   # res = optimize(cost, gradient!, initialθ, method = ConjugateGradient(), iterations = 1000);
#   # res = optimize(cost, gradient!, initialθ, Newton(); autodiff = :forward);
#   # res = optimize(cost, gradient!, initialθ, LBFGS());
#   # res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS());
#   # cost, gradient! = logistic_cost_gradient(initialθ, X, y)
#   # res = optimize(cost, gradient!, initialθ, method = LBFGS());
#   res = optimize(θ -> logistic_objective(θ, X, w), initialθ, LBFGS(); autodiff = :forward);
#   θ = Optim.minimizer(res);
#   println("num of iterations from check phi w: ", Optim.iterations(res))
#   return sigmoid(X * θ), θ
# end

function logistic_regression_Optim_predict_proba_target_y(phi::Vector{Float64}, y::Vector{Float64})
  py_giv_phi_set_wx = zeros((2, 2, size(phi)[1], 2))

  clf_w_0_x_0_phi = logistic_regression_Optim(zeros((length(y),)), zeros((length(y),)), phi, y)
  clf_w_1_x_0_xphi = logistic_regression_Optim(ones((length(y),)), zeros((length(y),)), phi, y)
  clf_w_0_x_1_phi = logistic_regression_Optim(zeros((length(y),)), ones((length(y),)), phi, y)
  clf_w_1_x_1_xphi = logistic_regression_Optim(ones((length(y),)), ones((length(y),)), phi, y)

  py_giv_phi_set_wx[0, 0, :, 1] = clf_w_0_x_0_phi

  py_giv_phi_set_wx[1, 0, :, 1] = clf_w_1_x_0_xphi

  py_giv_phi_set_wx[0, 1, :, 1] = clf_w_0_x_1_phi
  
  py_giv_phi_set_wx[1, 1, :, 1] = clf_w_1_x_1_xphi

  py_giv_phi_set_wx[0, 0, :, 0] = 1 .- clf_w_0_x_0_phi
  
  py_giv_phi_set_wx[1, 0, :, 0] = 1 .- clf_w_1_x_0_xphi

  py_giv_phi_set_wx[0, 1, :, 0] = 1 .- clf_w_0_x_1_phi
  
  py_giv_phi_set_wx[1, 1, :, 0] = 1 .- clf_w_1_x_1_xphi

  return py_giv_phi_set_wx
end


function logistic_regression_Optim_proba_target_w(phi::Matrix{Float64}, w::Vector{Float64})
  pw_giv_phi_set_x = zeros((2, size(phi)[1], 2))

  clf_x_0_phi = logistic_regression_Optim(zeros((length(w),1)), phi, w)
  clf_x_1_phi = logistic_regression_Optim(ones((length(w),1)), phi, w)

  pw_giv_phi_set_x[0, :, 1] = predict(clf_x_0_phi)
  pw_giv_phi_set_x[1, :, 1] = predict(clf_x_1_phi)
  pw_giv_phi_set_x[0, :, 0] = 1 .- predict(clf_x_0_phi)
  pw_giv_phi_set_x[1, :, 0] = 1 .- predict(clf_x_1_phi)

  return pw_giv_phi_set_x

end

function logistic_regression_Optim_proba_target_w_giv_phi(phi::Matrix{Float64}, w::Vector{Float64})
  pw_giv_phi = zeros((size(phi)[1], 2))

  clf = logistic_regression_Optim(phi, w)

  pw_giv_phi[:, 1] = predict(clf)
  pw_giv_phi[:, 0] = 1 .- predict(clf)

  return pw_giv_phi

end

"""
    log_odds_ratio(py_giv_phi_set_wx::Matrix{Float64}, pw_giv_phi_set_x::Matrix{Float64})
"""
function log_odds_ratio(py_giv_phi_set_wx::Matrix{Float64}, pw_giv_phi_set_x::Matrix{Float64})
  p_yw_giv_xphi = zeros((2, 2))

  p_yw_giv_xphi[0,0] = py_giv_phi_set_wx[0,0,:,0]*pw_giv_phi_set_x[0,:,0] + py_giv_phi_set_wx[0,1,:,0]*pw_giv_phi_set_x[1,:,0]
  p_yw_giv_xphi[1,1] = py_giv_phi_set_wx[1,0,:,1]*pw_giv_phi_set_x[1,:,0] + py_giv_phi_set_wx[1,1,:,1]*pw_giv_phi_set_x[1,:,1]
  p_yw_giv_xphi[1,0] = py_giv_phi_set_wx[0,0,:,1]*pw_giv_phi_set_x[0,:,0] + py_giv_phi_set_wx[0,1,:,1]*pw_giv_phi_set_x[1,:,0]
  p_yw_giv_xphi[0,1] = py_giv_phi_set_wx[0,1,:,0]*pw_giv_phi_set_x[1,:,0] + py_giv_phi_set_wx[1,1,:,0]*pw_giv_phi_set_x[1,:,1]
  log_p_yw_giv_xphi = log.(p_yw_giv_xphi)

  return log_p_yw_giv_xphi[0,0] + log_p_yw_giv_xphi[1,1] - log_p_yw_giv_xphi[1,0] - log_p_yw_giv_xphi[0,1]

  return 
end

"""
    log_odds_ratio(py_giv_phi_set_w::Matrix{Float64}, pw_giv_phi::Matrix{Float64})
"""
function log_odds_ratio_2nd(py_giv_phi_set_w::Matrix{Float64}, pw_giv_phi::Matrix{Float64})
  p_yw_giv_phi = zeros((2, 2))

  p_yw_giv_phi[0,0] = py_giv_phi_set_w[0,:,0]*pw_giv_phi[:,0]
  p_yw_giv_phi[1,1] = py_giv_phi_set_w[1,:,1]*pw_giv_phi[:,1]
  p_yw_giv_phi[1,0] = py_giv_phi_set_w[0,:,1]*pw_giv_phi[:,0]
  p_yw_giv_phi[0,1] = py_giv_phi_set_w[1,:,0]*pw_giv_phi[:,1]
  log_p_yw_giv_phi = log.(p_yw_giv_phi)

  return log_p_yw_giv_phi[0,0] + log_p_yw_giv_phi[1,1] - log_p_yw_giv_phi[1,0] - log_p_yw_giv_phi[0,1]

  return 
end

"""
    ATE(x::Vector{Float64}, y::Vector{Float64})
Compute ATE for marginal baseline (predict y from treatment x)
"""

function ATE(x::Vector{Float64}, y::Vector{Float64})
  data = DataFrame(x=x, y=y)
  _, coeffs = logistic_regression_Optim_get_coeffs(x, y)
  # _, coeffs_check = logistic_regression_Optim_get_coeffs_check(x, y)
  # println("from ATE")
  # println("coeffs from logistic", coeffs)
  # println("coeffs from logistic check ", coeffs_check)
  return coeffs[2] #second item because julia is 1-indexed, and [1] is for the bias term
end

"""
    ATE(x::Vector{Float64}, y::Vector{Float64})
Compute ATE for allZ baseline (predict y from treatment x and all variables Z)
"""

function ATE(x::Vector{Float64}, Z::Matrix{Float64}, y::Vector{Float64})
  _, coeffs = logistic_regression_Optim_get_coeffs(x, Z, y)
  # _, coeffs_check = logistic_regression_Optim_get_coeffs_check(x, Z, y)
  # println("from ATE")
  # println("coeffs from logistic ", coeffs)
  # println("coeffs from logistic check ", coeffs_check)
  return coeffs[2] #second item because julia is 1-indexed, and [1] is for the bias term
end

"""
    ATE(x::Vector{Float64}, y::Vector{Float64})
Compute ATE for learned phi (predict y from learned phi, i.e. theta*Z, and treatment x)
"""
function ATE_learned(x::Vector{Float64}, phi::Vector{Float64}, y::Vector{Float64})
  data = DataFrame(x=x, phi=phi, y=y)

  # data_x_0 = data[data[!,:x].==0,:]
  # data_x_1 = data[data[!,:x].==1,:]

  phi_0 = phi[x.==0]
  phi_1 = phi[x.==1]
  y_0 = y[x.==0]
  y_1 = y[x.==1]

  # TODO: verify, this makes up for a non-singular matrix. Seems like the following 
  # is the right way to go, using X for separation?
  # clf_x0 = glm(@formula(y ~ x + phi), data_x_0, Binomial(), LogitLink())
  # clf_x1 = glm(@formula(y ~ x + phi), data_x_1, Binomial(), LogitLink())

  clf_x0, coeffs_0 = logistic_regression_Optim_get_coeffs(phi_0, y_0)
  clf_x1, coeffs_1 = logistic_regression_Optim_get_coeffs(phi_1, y_1)

  # p0 = clf_x0.predict(data[!, :phi])
  # p1 = clf_x1.predict(data[!, :phi])
  
  X = [ones(length(y)) phi]
  p0 = sigmoid(X * coeffs_0)
  p1 = sigmoid(X * coeffs_1)

  return mean(p1) - mean(p0)
end

"""
    ATE(x::Vector{Float64}, y::Vector{Float64})
Compute ATE for learned phi (predict y from learned phi, i.e. theta*Z, 
  and treatment x and sel_Z, i.e. Zs selected by theta)
"""
function ATE_learned(x::Vector{Float64}, phi::Vector{Float64}, Z_select::Matrix{Float64}, y::Vector{Float64})
  # data = DataFrame(x=x, phi=phi, Z_select=Z_select, y=y)

  # data_x_0 = data[data[!,:x].==0,:]
  # data_x_1 = data[data[!,:x].==1,:]
  Z_select_x_0 = Z_select[x.==0, :]
  Z_select_x_1 = Z_select[x.==1, :]
  phi_0 = phi[x.==0]
  phi_1 = phi[x.==1]
  y_0 = y[x.==0]
  y_1 = y[x.==1]
  clf_x0, coeffs_0 = logistic_regression_Optim_get_coeffs(phi_0, Z_select_x_0, y_0)
  clf_x1, coeffs_1 = logistic_regression_Optim_get_coeffs(phi_1, Z_select_x_1, y_1)

  X = [ones(length(y)) phi Z_select]
  p0 = sigmoid(X * coeffs_0)
  p1 = sigmoid(X * coeffs_1)
  
  return mean(p1) - mean(p0)

end