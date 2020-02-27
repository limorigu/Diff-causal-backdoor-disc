using ProgressMeter, Printf
using LinearAlgebra
using StatsBase
using Statistics
using TexTables
using DataFrames, Gadfly, Compose
using PyCall
using Colors
@pyimport pickle

include("simulate.jl")
include("learn_linear.jl")
include("util.jl")

function process_ATEs(hypo_fail_all_lambdas, path_to_file)
  corrpx_valid = []
  num_trials = length(hypo_fail_all_lambdas)

  err_learned_train = zeros(num_trials)
  err_zsel_learned_train = zeros(num_trials)
  err_all_Z_train = zeros(num_trials)
  err_marg_Z_train = zeros(num_trials)
  err_learned_valid = zeros(num_trials)
  err_zsel_learned_valid = zeros(num_trials)
  err_all_Z_valid = zeros(num_trials)
  err_marg_Z_valid = zeros(num_trials)

  err_learned_B_train = zeros(num_trials)
  err_zsel_learned_B_train = zeros(num_trials)
  err_all_Z_B_train = zeros(num_trials)
  err_marg_Z_B_train = zeros(num_trials)
  err_learned_B_valid = zeros(num_trials)
  err_zsel_learned_B_valid = zeros(num_trials)
  err_all_Z_B_valid = zeros(num_trials)
  err_marg_Z_B_valid = zeros(num_trials)

  for i=1:num_trials

    # creation of array of ATE errors (for each trial) 
    # with respect to best Z combination (in thise case, Z3) for training set
    err_learned_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][3] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][2])
    err_zsel_learned_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][5] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][2])
    err_all_Z_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][4] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][2])
    err_marg_Z_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][6] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][2])

    # creation of array of ATE errors (for each trial) 
    # with respect to best Z combination (in thise case, Z3) for validation set
    err_learned_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][3] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][2])
    err_zsel_learned_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][5] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][2])
    err_all_Z_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][4] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][2])
    err_marg_Z_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][6] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][2])

    # creation of array of ATE errors (for each trial) 
    # w.r.t real ATE (weight on X->Y in the simulation graph) for training set
    err_learned_B_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][3] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][1])
    err_zsel_learned_B_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][5] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][1])
    err_all_Z_B_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][4] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][1])
    err_marg_Z_B_train[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_train"][6] - 
      hypo_fail_all_lambdas[i]["ATE_results_train"][1])

    # creation of array of ATE errors (for each trial) 
    # w.r.t real ATE (weight on X->Y in the simulation graph) for validation set
    err_learned_B_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][3] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][1])
    err_zsel_learned_B_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][5] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][1])
    err_all_Z_B_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][4] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][1])
    err_marg_Z_B_valid[i] = abs.(hypo_fail_all_lambdas[i]["ATE_results_valid"][6] - 
      hypo_fail_all_lambdas[i]["ATE_results_valid"][1])

    append!(corrpx_valid, hypo_fail_all_lambdas[i]["corr_px_valid"])
  end

  # plotting ATE error (w.r.t real ATE) scatter plots on validation set
  # ------------------------------------------------------------------------------------ #
  # ------------------------------------------------------------------------------------ #
  abline = Geom.abline(color="red", style=:dash)

  max_value_plot_valid = max(maximum(err_learned_valid), maximum(err_zsel_learned_valid), 
    maximum(err_all_Z_valid), maximum(err_marg_Z_valid))
  max_value_plot_valid = max_value_plot_valid + (max_value_plot_valid/10)

  max_value_plot_B_valid = max(maximum(err_learned_B_valid), maximum(err_zsel_learned_B_valid), 
    maximum(err_all_Z_B_valid), maximum(err_marg_Z_B_valid))
  println(max_value_plot_B_valid)
  max_value_plot_B_valid = max_value_plot_B_valid + (max_value_plot_B_valid/10)
  println(max_value_plot_B_valid)

  theme_shade = Theme(default_color=Colors.RGBA(9/255, 29/255, 83/255, 0.4))
  theme_plot = Theme(default_color=colorant"rgb(182, 196, 233)", point_label_font_size=20pt,
          highlight_width = 2pt, point_size=5pt,discrete_highlight_color=x->"black",major_label_font_size=16pt, 
          minor_label_font_size=16pt,key_title_font_size=18pt, key_label_font_size=18pt)
  
  x_shade = [0. 0. round(max_value_plot_B_valid,digits=1)+0.1 0.]
  y_shade = [0. round(max_value_plot_B_valid,digits=1)+0.1 round(max_value_plot_B_valid,digits=1)+0.1 0.]    

  shading = layer(x=x_shade,
      y=y_shade,
      Geom.polygon(preserve_order = true, fill = true),
      order = 1, theme_shade)

  dots_1 = layer(x = err_zsel_learned_B_valid, y = err_marg_Z_B_valid, Geom.point, order = 2,theme_plot)
  dots_2 = layer(x = err_zsel_learned_B_valid, y = err_all_Z_B_valid, Geom.point, order = 2,theme_plot)

  p1_B_valid = plot(x=x_shade,
      y=y_shade,
      Geom.polygon(preserve_order = true, fill = true), theme_shade,dots_1,
      Scale.x_continuous(minvalue=0, maxvalue=max_value_plot_B_valid), Scale.y_continuous(minvalue=0, maxvalue=max_value_plot_B_valid),
      abline,Guide.xlabel("| true - ours |"), Guide.ylabel("| true - marginal |"))

  p2_B_valid = plot(x=x_shade,
        y=y_shade,
        Geom.polygon(preserve_order = true, fill = true), theme_shade,dots_2,
    abline, Guide.xlabel("| true - ours |"), Guide.ylabel("| true - allZ |"))

  p_B_valid = hstack(p1_B_valid, p2_B_valid)
  draw(SVG(path_to_file*"outputfiles/scatter_B_valid.svg", 8inch, 4inch), p_B_valid)
  # ------------------------------------------------------------------------------------ #
  # ------------------------------------------------------------------------------------ #

  # ATE errors df with respect to best Z combination (in thise case, Z3)
  ATEs = DataFrame(mean_err_learned = [mean(err_learned_train), mean(err_learned_valid)],
    mean_err_learned_w_zsel = [mean(err_zsel_learned_train), mean(err_zsel_learned_valid)],
    mean_err_marg = [mean(err_marg_Z_train), mean(err_marg_Z_valid)],
    mean_err_allZ = [mean(err_all_Z_train), mean(err_all_Z_valid)])

  # ATE errors df with respect to real ATE (weight on X->Y in the simulation graph)
  ATE_Bs = DataFrame(mean_err_learned = [mean(err_learned_B_train), mean(err_learned_B_valid)],
    mean_err_learned_w_zsel = [mean(err_zsel_learned_B_train), mean(err_zsel_learned_B_valid)],
    mean_err_marg = [mean(err_marg_Z_B_train), mean(err_marg_Z_B_valid)],
    mean_err_allZ = [mean(err_all_Z_B_train), mean(err_all_Z_B_valid)])  

  # print previous ATE dfs as latex tables
  open(path_to_file*"outputfiles/ATE_table.txt", "a") do file
    write(file, repr("text/latex", ATEs))
    write(file, "\n")
    write(file, "\n")
    write(file, repr("text/latex", ATE_Bs))
  end

  # print previous ATE dfs to screen for debugging
  println()
  println("---------------------------------")
  println("comparison to fit on best Zset (Z3 in this case)")
  println("first row train, second row valid")
  println("---------------------------------")
  println()
  println(ATEs)
  println()
  println("---------------------------------")
  println("comparison to weight on X->Y in simulation graph")
  println("first row train, second row valid")
  println("---------------------------------")
  println()
  println(ATE_Bs)

  # convert ATE dfs to dicts
  ATEs_full = Dict("err_learned_train"=>err_learned_train,
   "err_learned_valid"=>err_learned_valid, 
   "err_zsel_learned_train"=>err_zsel_learned_train,
   "err_zsel_learned_valid"=>err_zsel_learned_valid, 
   "err_marg_Z_train"=>err_marg_Z_train,
   "err_marg_Z_valid"=>err_marg_Z_valid, 
   "err_all_Z_train"=>err_all_Z_train, 
   "err_all_Z_valid"=>err_all_Z_valid)

  ATE_Bs_full = Dict("err_learned_train"=>err_learned_B_train,
   "err_learned_valid"=>err_learned_B_valid, 
   "err_zsel_learned_train"=>err_zsel_learned_B_train,
   "err_zsel_learned_valid"=>err_zsel_learned_B_valid, 
   "err_marg_Z_train"=>err_marg_Z_B_train,
   "err_marg_Z_valid"=>err_marg_Z_B_valid, 
   "err_all_Z_train"=>err_all_Z_B_train, 
   "err_all_Z_valid"=>err_all_Z_B_valid)

  # save ATE dicts to file for possible further analysis
  open(path_to_file*"ATEs_full.txt", "w") do file
    write(file, repr(ATEs_full))
  end

  open(path_to_file*"ATE_Bs_full.txt", "w") do file
    write(file, repr(ATE_Bs_full))
  end

  # pick 5 random trials for which to save to file and print to screen
  # Z selection by beta weight
  pick_trial_num = rand(collect(1:num_trials),5)

  open(path_to_file*"outputfiles/Z_sel.txt", "w") do file
    for index in pick_trial_num
      thresh = 1e-3
      write(file, "--------\n")
      println("--------\n")
      println("--------\n")
      sel_Z = findall(abs.(hypo_fail_all_lambdas[index]["beta_hat"]) .> thresh)
      println("--------\n")
      println("--------\n")
      num_selected = length(sel_Z)
      write(file, "# zs selected beta_hat: $num_selected\n")
      write(file, "z selected beta_hat: \n")
      write(file, repr(sel_Z))
      println("z_sel\n")
      println(repr(sel_Z))
      write(file, "\n")
      write(file, "beta: \n")
      write(file, repr(hypo_fail_all_lambdas[index]["beta_hat"]))
      write(file, "\n")
      write(file, "--------\n")
    end
  end
end

# combine trial results computed in parallel, and apply Process_ATE to resulting all_trials dict
function main(dir)
  local_path = pwd()
  path = local_path*"/"*dir
  all_trials = Dict()
    for (root, dirs, files) in walkdir(path)
    dirs = [dir for dir in dirs if ((dir!="results") && (dir!="outputfiles") && (dir!=".ipynb_checkpoints"))]
      for (ind_dir,dir) in enumerate(dirs)
        if (dir!="logistic_losses") && (dir!=".ipynb_checkpoints")
            files = readdir(path*"/"*dir)
            trial = Dict()
            for (index,file) in enumerate(files)
              if endswith(file, ".pickle")
                temp = nothing
                @pywith pybuiltin("open")(path*"/"*dir*"/"*file,"rb") as f begin
                    temp = pickle.load(f) end
                trial[temp["corr_px_valid"]] = temp
              end
            end 
            all_trials[ind_dir] = trial[minimum(keys(trial))]
        end
      end
    end
    open(local_path*"/"*dir*"/results/all_trials.txt", "w") do file
      write(file, repr(all_trials))
    end
   process_ATEs(all_trials, local_path*"/"*dir*"/results/") 
end