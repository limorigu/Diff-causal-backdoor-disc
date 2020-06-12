
Differentiable Causal Backdoor Discovery
============

The following repository contains all code and results to reproduce results in the AISTATS 2020 paper: [**Differentiable Causal Backdoor Discovery**](https://arxiv.org/pdf/2003.01461.pdf).


## Repo sructure and content
```
|
|_ code
|
|____ *.jl --> contains code to run the full procedure to compute results (all Figures)
|____ real_world_nhs_dataset
|
|______ nhs_data_smaller_var.csv --> contains NHS dataset used in paper
|______ *.jl --> contains code to run full procedure to compute paper results (Table 1)
|
|____ omega_point_* --> contains datasets, and scatter plots generated for simulation code (see results/outputfiles for scatter plot, for each setup of interest). Included for ease of 
|     comparison and reproduction. Will be regenerated running the code above (see notebooks for clarification).
|
|_ notebooks (for ease of use, refer to these for pipeline of running the full code, contained in the above code folder)
|
|____ ATE_scatter_histplots
|
|______ Compute_ATEs_produce_scatterplots_julia.ipynb --> notebook to reproduce scatterplots. Run first (note: need to run    
|       code in parallel as stated on top of notebook, ideally on many-core server first). Resulting figures will also appear 
|       in respective omega_point_* folder for each setting.
|______ Histplots+Entner_comparison_python.ipynb --> notebook to reproduce Entner baseline and histograms in paper. Run second.
|______ histplots --> this folder contains the resuling histogram plots (on top of them showing in above notebook).
```

## Code dependencies

* ProgressMeter
* StatsBase
* Statistics
* TexTables
* DataFrames
* Gadfly
* Compose
* PyCall
* ForwardDiff
* Random
* Printf
* LinearAlgebra
* Distributions
* Optim
* PenaltyFunctions
* GLM
* Colors
* ArgParse

## to install in one go, in Julia REPL 
`import Pkg`

`Pkg.add(["ProgressMeter", "StatsBase", "Statistics", "TexTables", "DataFrames", "Gadfly", "Compose", "PyCall", "ForwardDiff", "Random", "Printf", 
"LinearAlgebra", "Distributions", "Optim", "PenaltyFunctions", "GLM", "Colors", "ArgParse"])`

If the above fails for compatability reasons, try removing "TexTables" from this list, and installing it seperately with the following:

`Pkg.add("TexTables")`
