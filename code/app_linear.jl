#!/Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia
################################################################################
## app_linear.jl
##
## This file contains the basic functions to two primary tasls:
##
## 1. The assessment of how hard are some synthetic problems generated by
##    particular configurations. Start from function
##    `demo_linear_simulate_evaluate()`
## 2. To assess methods for linear modeling. Start from function
##    `demo_linear_simulate_learn_evaluate()`
################################################################################

using ProgressMeter, Printf
using LinearAlgebra
using StatsBase
using Statistics
using TexTables
using DataFrames, Gadfly, Compose
using PyCall
@pyimport pickle

include("simulate.jl")
include("learn_linear.jl")
include("util.jl")

################################################################################
################################################################################
## DATA STRUCTURES
## Data structures

################ 1. EVALUATION OF HARDNESS OF SIMULATIONS

"""
    linear_simulate_evaluate(n, p, use_population;
                             min_noise, max_noise, pseudo_pcount,
                             x_noise, y_noise,
                             prob_flip, w_effect, x_effect)
Run a simulation using sample size `n` and number of vertices given by vector
`p`, followed by an evaluation of baselines applied to this problem. If
`use_population` is `true`, this uses the model covariance matrix directly.
Error variances for each vertex is set according to the formula
`min(max_noise, max(min_noise, (pseudo_pcount - num_parents) / pseudo_pcount))`,
where `num_parents` is the number of total parents of that vertex. That is,
`pseudo_pcount` represents a number of pseudo "parents" of a vertex,
explicitly represented in the causal graph (as a node) or not (added to the
error term contribution). As the number of explicit parents in the graph
increases, the error variance decreases. It is still forced to lie on the
interval `[min_noise, max_noise]`. The exceptions are the treatment variable,
where the error variance is set to `x_noise`, and the outcome variable, where
the error variance is set to `y_noise`. This is in order to better control the
difficulty of the problem: the lower `x_noise` is, for instance, the harder the
problem becomes for the estimator that conditions on all variables.
Coefficients are sampled independently and normalized so that each variable has
variance `1`. Two coefficients are set manually: the coefficient of instrument
on treatment is set by `w_effect`, while the coefficient of treatment on
outcome is set to `x_effect`. Finally, to avoid confounding adding to up
to very small values due to averaging effects (which are more likely as the
entries of `p` grow), each vertex has causal effects on its children all drawn
with the same signal (which can be either positive or negative with equal
probability). Some variability is added by flipping the sign of each edge with
probability `prob_flip`. The closer to `0.5` this is, the higher the probability
that confounding and collider effects will be close to zero, so set it to a
small number in order to make problems harder.
This function returns matrix `B` of coefficients, covariance matrix `Omega` of
error terms, adjacency matrix `G` encoding a DAG, dataset `dat` of sample
size `n`, and a data structure `vertex_labels` indicating which type of
variable (covariates `z1`, `z2`, `z3`, `z4`, latent variables `u1` and `u2`,
instrument `w`, treatment `x` and outcome `y`) corresponds to which entry.
This function also returns four types of baselines: the consistent estimator
`ATE_right` (which is the true causal effect is `use_population == true`) that
uses the minimal adjustment set `z3`; `ATE_all_z`, the inconsistent estimator
that uses all covariates; `ATE_OK_z`, another consistent estimator that uses
an unnecessarily large covariate set; and `ATE_margin`, the inconsistent
estimator that ignores any covariate adjustment.
"""
function linear_simulate_evaluate(n, p, use_population;
                                  min_noise = 0.1, max_noise = 0.9, pseudo_pcount = 50,
                                  x_noise = 0.1, y_noise = 0.1,
                                  prob_flip = 0.1, w_effect = 0.2, x_effect = 0.2)

  B, Omega, Sigma, G, dat, vertex_labels = simulate_gaussian(n, p,
     min_noise = min_noise, max_noise = max_noise,
     x_noise = x_noise, y_noise = y_noise,
     prob_flip = prob_flip, w_effect = w_effect, x_effect = x_effect)

  x, y = vertex_labels.x, vertex_labels.y
  best_z = [x; vertex_labels.z3]
  all_z = [x; vertex_labels.z1; vertex_labels.z2; vertex_labels.z3; vertex_labels.z4]
  OK_z = [x; vertex_labels.z2; vertex_labels.z3; vertex_labels.z4]

  if use_population
    S = Sigma
  else
    S = cov(dat)
  end

  ATE_right  = (S[best_z, best_z] \ S[best_z, y])[1]
  ATE_all_z  = (S[all_z, all_z] \ S[all_z, y])[1]
  ATE_OK_z   = (S[OK_z, OK_z] \ S[OK_z, y])[1]
  ATE_margin = S[x, y] / S[x, x]

  return B, Omega, Sigma, G, dat, vertex_labels,
         ATE_right, ATE_all_z, ATE_OK_z, ATE_margin

end

"""
    batch_linear_simulate_evaluate(num_trials, n, p, use_population,
                                   min_noise, max_noise, pseudo_pcount,
                                   x_noise, y_noise,
                                   prob_flip, w_effect, x_effect, label)
Runs `num_trials` trials of a linear model simulation and evaluation. See
the documentation of function `linear_simulate_evaluate` for details about the
arguments.
"""
function batch_linear_simulate_evaluate(num_trials, n, p, use_population,
                                        min_noise, max_noise, pseudo_pcount,
                                        x_noise, y_noise,
                                        prob_flip, w_effect, x_effect, label)

  println()
  println(label)
  println(repeat("=", length(label)))
  println()

  for i = 1:num_trials
    B, Omega, Sigma, G, dat, vertex_labels, ATE_right, ATE_all_z, ATE_OK_z, ATE_margin =
      linear_simulate_evaluate(n, p, use_population;
                               min_noise = min_noise, max_noise = max_noise, pseudo_pcount = pseudo_pcount,
                               x_noise = x_noise, y_noise = y_noise,
                               prob_flip = prob_flip, w_effect = w_effect, x_effect = x_effect)
    @printf("[%2d] Right ATE = %+.2f | All_Z ATE = %+.2f | OK_Z ATE = %+.2f | marginal ATE = %+.2f | IV coeff = %+.2f | ATE coeff = %+.2f \n",
            i, ATE_right, ATE_all_z, ATE_OK_z, ATE_margin,
            B[vertex_labels.x, vertex_labels.w],
            B[vertex_labels.y, vertex_labels.x])
  end

  println()

end

############## DEMO
#
# This runs a demonstration of how we can evaluate the effect of the simulation
# parameters on the baseline estimators In this demo, we have the following
# free parameters:
#
# - num_trials: number of runs
# - n: sample size
# - use_population: use population covariance matrix when assessing baselines
# - p_factor: number of variables of each type to be generated
# - min_noise, max_noise: error term variance will be generated inversely
#                         proportional to the number of parents in the graph,
#                         but always bounded to lie on the interval
#                         [min_noise, max_noise]
# - x_noise, y_noise: except for treatment X and outcome Y. In this case,
#                     the error term variance is given by these variables,
#                     respectively
# - prob_flip: direct causal effects of each variable on their children
#              are initially sampled to have the same sign. To add some
#              variability, with probability prob_flip each coefficient
#              sign will be flipped. When p_factor is large and prob_flip
#              is close to 0.5, confounding and collider effects have high
#              probability of being around zero due to averaging effects.
#              Keeping prob_flip low will help to make problems more difficult.
# - w_effect: coefficient of instrument W on treatment X. Keeping it fixed
#             helps to control for an otherwise diminishing coefficient as
#             p_factor grows. Keep this between -1 and 1, as all variables are
#             standardized. As a matter of fact, this needs to be smaller than
#             1 - x_noise in absolute value
# - x_effect: coefficient of treatment X on outcome Y, with analogous comments
# - sim_label: message to be displayed when running batch of experiments

function demo_linear_simulate_evaluate()
  num_trials, n, use_population = 10, 1000, true
  p_factor = 30
  p = ones(Int, 5) * p_factor
  min_noise, max_noise, pseudo_pcount, x_noise, y_noise = 0.1, 0.9, 50, 0.1, 0.1
  prob_flip, w_effect, x_effect = 0.1, 0.2, 0.2
  sim_label = "DIMENSION = " * string(p_factor) *
              ", FLIP PROBABILITY " * string(prob_flip) *
              ", USE POPULATION = " * string(use_population)
  if !use_population sim_label *=  ", n = " * string(n) end
  batch_linear_simulate_evaluate(num_trials, n, p, use_population,
                     min_noise, max_noise, pseudo_pcount, x_noise, y_noise,
                     prob_flip, w_effect, x_effect, sim_label)
end

################ 2. EVALUATION OF LEARNING ALGORITHMS

function choose_lambdas(lambda2, lambda1, i, n, p, num_Z, min_noise, max_noise, pseudo_pcount, x_noise, y_noise,
         prob_flip, w_effect, x_effect, use_population, max_iter, lr, path_to_file)
  initial_lambda1 = lambda1
  ATE_results_train = Matrix{Float64}(undef, 1, 6)
  ATE_results_valid = Matrix{Float64}(undef, 1, 6)
  corr_pxs = Dict() 
  hypo_fail_all_lambdas = Dict()
  Random.seed!(i);

  # Generate dataset
  # training set
  B_train, Omega_train, Sigma_train, G_train, dat_train, vertex_labels = simulate_gaussian(n, p,
     min_noise = min_noise, max_noise = max_noise, pseudo_pcount = pseudo_pcount,
     x_noise = x_noise, y_noise = y_noise,
     prob_flip = prob_flip, w_effect = w_effect, x_effect = x_effect)
  # validation set
  B_valid, Omega_valid, Sigma_valid, G_valid, dat_valid, _ = simulate_gaussian(n, p,
  min_noise = min_noise, max_noise = max_noise, pseudo_pcount = pseudo_pcount,
  x_noise = x_noise, y_noise = y_noise,
  prob_flip = prob_flip, w_effect = w_effect, x_effect = x_effect)


  # pickle datasets and real ATE (weight on X->Y in the simulation graph) for Entner baseline comparison, etc.
  f = pybuiltin("open")(path_to_file*"/outputfiles/train_data_$i.pickle","wb")
  p = pickle.Pickler(f)
  p.dump(dat_train)
  f.close()

  f_real_ATE = pybuiltin("open")(path_to_file*"/outputfiles/train_real_ATE_$i.pickle","wb")
  p_real_ATE = pickle.Pickler(f_real_ATE)
  p_real_ATE.dump(B_train[vertex_labels.y, vertex_labels.x])
  f_real_ATE.close()

  f2 = pybuiltin("open")(path_to_file*"/outputfiles/valid_data_$i.pickle","wb")
  p2 = pickle.Pickler(f2)
  p2.dump(dat_valid)
  f2.close()

  f_real_ATE_valid = pybuiltin("open")(path_to_file*"/outputfiles/valid_real_ATE_$i.pickle","wb")
  p_real_ATE_valid = pickle.Pickler(f_real_ATE_valid)
  p_real_ATE_valid.dump(B_valid[vertex_labels.y, vertex_labels.x])
  f_real_ATE_valid.close()

  # graph definitions
  w, x, y = vertex_labels.w, vertex_labels.x, vertex_labels.y
  Z = [vertex_labels.z1; vertex_labels.z2; vertex_labels.z3; vertex_labels.z4]
  best_Z = vertex_labels.z3

  # Sigma train construction
  use_population ? Sigma_hat = Sigma_train : Sigma_hat = cov(dat_train)

  hypo_fail = Dict()
  reject_null_hypothesis = false
  lambda1 = initial_lambda1
  bonferonni_correction = 1.
  while (!reject_null_hypothesis)
    # learn beta on training set
    beta_hat, corr_px_beta_hat, corr_p_beta_hat = 
        bd_learn_linear(Sigma_hat, lambda1, lambda2, vertex_labels, i, path_to_file,
                                max_iter = max_iter, lr = lr)

    # definition of sel_Z
    thresh = 1e-3
    sel_Z = findall(x->abs(x)>thresh, beta_hat)
    # test for the null hypothesis
    reject_null_hypothesis = 
      ind_null_hypo(n, num_Z, corr_p_beta_hat; significance_level=(0.01/bonferonni_correction))

    # if null rejected, i.e. reject_null_hypothesis=true
    if reject_null_hypothesis
      # compute ATEs training set
      Sigma_wyx_phi = build_phi_covariance(beta_hat, w, y, x, Z, Sigma_hat)

      ate_real_train = B_train[vertex_labels.y, vertex_labels.x]
      ate_hat_train = (Sigma_wyx_phi[[3; 4], [3; 4]] \ Sigma_wyx_phi[[3; 4], 2])[1]
      ate_hat_all_Z_train = (Sigma_hat[[x; Z], [x; Z]] \ Sigma_hat[[x; Z], y])[1]
      ate_hat_best_Z_train = (Sigma_hat[[x; best_Z], [x; best_Z]] \ Sigma_hat[[x; best_Z], y])[1]
      ate_hat_sel_Z_train = (Sigma_hat[[x; sel_Z], [x; sel_Z]] \ Sigma_hat[[x; sel_Z], y])[1]
      ate_hat_marg_Z_train = Sigma_hat[x, y] / Sigma_hat[x, x]

      ATE_results_train = [ate_real_train ate_hat_best_Z_train ate_hat_train ate_hat_all_Z_train ate_hat_sel_Z_train ate_hat_marg_Z_train]

      # compute \rho(W,Y|beta*Z, X) validation
      use_population ? Sigma_hat_valid = Sigma_valid : Sigma_hat_valid = cov(dat_valid)
      Sigma_wyx_phi_valid = build_phi_covariance(beta_hat, w, 
        x, y, Z, Sigma_hat_valid)
      try      
        corr_px_valid = abs(partial_corr([1; 2], [3; 4], Sigma_wyx_phi_valid)[1, 2])
        # compute ATEs validation set
        ate_real_valid = B_valid[vertex_labels.y, vertex_labels.x]
        ate_hat_valid = (Sigma_wyx_phi_valid[[3; 4], [3; 4]] \ Sigma_wyx_phi_valid[[3; 4], 2])[1]
        ate_hat_all_Z_valid = (Sigma_hat_valid[[x; Z], [x; Z]] \ Sigma_hat_valid[[x; Z], y])[1]
        ate_hat_best_Z_valid = (Sigma_hat_valid[[x; best_Z], [x; best_Z]] \ Sigma_hat_valid[[x; best_Z], y])[1]
        ate_hat_sel_Z_valid = (Sigma_hat_valid[[x; sel_Z], [x; sel_Z]] \ Sigma_hat_valid[[x; sel_Z], y])[1]
        ate_hat_marg_Z_valid = Sigma_hat_valid[x, y] / Sigma_hat_valid[x, x]

        ATE_results_valid = [ate_real_valid ate_hat_best_Z_valid ate_hat_valid ate_hat_all_Z_valid ate_hat_sel_Z_valid ate_hat_marg_Z_valid]
        # save corrs
        hypo_fail["corr_px_train"] = corr_px_beta_hat
        hypo_fail["corr_p_train"] = corr_p_beta_hat
        hypo_fail["corr_px_valid"] = corr_px_valid
        hypo_fail["lr"] = lr
        # parameters
        hypo_fail["beta_hat"] = beta_hat
        hypo_fail["lambda1"] = lambda1
        hypo_fail["lambda2"] = lambda2
        # ATEs train
        hypo_fail["ATE_results_train"] = ATE_results_train
        # ATEs valid
        hypo_fail["ATE_results_valid"] = ATE_results_valid
        hypo_fail_all_lambdas[hypo_fail["corr_px_valid"]] = hypo_fail
        break
      catch
        lambda1 = lambda1 * 2
        bonferonni_correction += 1.
      end
    else
      lambda1 = lambda1 * 2
      bonferonni_correction += 1.
    end
  end

  best_setup = hypo_fail_all_lambdas[minimum(keys(hypo_fail_all_lambdas))]

  return hypo_fail_all_lambdas[minimum(keys(hypo_fail_all_lambdas))]
end
############## DEMO
#
# This runs a demonstration of the learning algorithm using `num_trials`
# simulations. Refer to `batch_linear_simulate_learn_evaluate` for an
# explanation of parametrs and outputs.

function demo_linear_simulate_learn_evaluate(trial, path_to_file; lambda2=5e-2, lambda1=5e-6, 
  lr=1e-5, x_noise=0.01, x_effect=0.1)
  num_Z = 4*30
  n, p = 10000, ones(Int, 5) * 30
  min_noise, max_noise, pseudo_pcount, y_noise = 0.1, 0.5, 50, 0.1
  prob_flip, w_effect = 0.1, 0.1
  use_population = false
  max_iter = 1000
  reject_null_hypothesis = true

  if !isdir(path_to_file)
    mkdir(path_to_file)
    mkdir(path_to_file*"/outputfiles")
    mkdir(path_to_file*"/results")
    mkdir(path_to_file*"/results/outputfiles")
  end 

  if !isdir(path_to_file*"/trial_$trial")
    mkdir(path_to_file*"/trial_$trial")  
  end

  results_all_trials = Dict()
  
  println("Solving simulation $trial, lambda1 $lambda1, lambda2 $lambda2, lr $lr...")

  # perform parameter search for trial and given hyperparameters
  best_lambdas_for_trial = 
  choose_lambdas(lambda2, lambda1, trial, n, p, num_Z, min_noise, max_noise, pseudo_pcount, x_noise, y_noise,
         prob_flip, w_effect, x_effect, use_population, max_iter, lr, path_to_file)   

  println("chosen lambda1_: ", best_lambdas_for_trial["lambda1"])
  println("chosen lambda2_: ", best_lambdas_for_trial["lambda2"])
  println("chosen lr_: ", best_lambdas_for_trial["lr"])
  println("partial correlation on train set: ", best_lambdas_for_trial["corr_px_train"])
  println("partial correlation on validation set: ", best_lambdas_for_trial["corr_px_valid"])

  lr_pick = best_lambdas_for_trial["lr"]
  l2_pick = best_lambdas_for_trial["lambda2"]

  # pickle results for trial and given hyperparameters to combine with other trials+hyperparams in Process_ATE.jl
  f = pybuiltin("open")(path_to_file*"/trial_$trial/lr_$lr_pick lambda2_$l2_pick .pickle","wb")
  p = pickle.Pickler(f)
  p.dump(best_lambdas_for_trial)
  f.close()
 end