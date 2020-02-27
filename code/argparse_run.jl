using ArgParse

include("util.jl")
include("simulate.jl")
include("learn_linear.jl")
include("app_linear.jl")

function main(args)
    lambda_twos = [5e-6, 5e-5, 5e-4, 5e-3, 5e-2]
    learnRS = [1e-6, 5e-5, 1e-5, 1e-4, 5e-4]

    s = ArgParseSettings(description = "run Linear.")

    @add_arg_table s begin
        "--lambda2", "-l"
            arg_type = Int
        "--learningRate", "-r"
            arg_type = Int
        "--trial", "-t"
            arg_type = Int
    end

    parsed_args = parse_args(args, s) # the result is a Dict{String,Any}

    lambda2 = lambda_twos[parsed_args["lambda2"]]
    lr = learnRS[parsed_args["learningRate"]]
    trial = parsed_args["trial"]

    demo_linear_simulate_learn_evaluate(trial, "omega_point_1_sigma2_point_01/"; lambda2=lambda2, lambda1=5e-5, lr=lr, x_noise=0.01, x_effect=0.1)
    demo_linear_simulate_learn_evaluate(trial, "omega_point_1_sigma2_point_6/"; lambda2=lambda2, lambda1=5e-5, lr=lr, x_noise=0.6, x_effect=0.1)
    demo_linear_simulate_learn_evaluate(trial, "omega_point_5_sigma2_point_01/"; lambda2=lambda2, lambda1=5e-5, lr=lr, x_noise=0.01, x_effect=0.5)
    demo_linear_simulate_learn_evaluate(trial, "omega_point_5_sigma2_point_6/"; lambda2=lambda2, lambda1=5e-5, lr=lr, x_noise=0.6, x_effect=0.5)
end

main(ARGS)

### run following command for computation of trials in parallel; l,r args taken from lambda_twos and learnRS as defined above
### t will determine trial num, a sequence of 1-specific number, here set to 25 trials.
### parallel is an open-source library shared generously by O. Tange, available at http://dx.doi.org/10.5281/zenodo.16303 

# parallel --compress julia argparse_run.jl -l {1} -r {2} -t {3} ::: $(seq 5) ::: $(seq 5) ::: $(seq 25)