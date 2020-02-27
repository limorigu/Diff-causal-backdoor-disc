#!/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia
################################################################################
## util.jl
##
## This file contains general function to aid optimization and modeling.
################################################################################
using Statistics
using Distributions
using Optim
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