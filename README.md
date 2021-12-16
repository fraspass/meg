# Mutually exciting point process graphs for modelling dynamic networks

This repository contains a _python_ library supporting the paper *Sanna Passino, F. and Heard, N. A. (2021+) "Mutually exciting point process graphs for modelling dynamic networks"* ([arXiv preprint](https://arxiv.org/abs/2102.06527)). 

The library `meg` can be installed in edit mode as follows:
```
pip install -e lib/
```
The library can then be imported in any _python_ session:
```python3
import lsbm
```

The repository contains multiple directories:
* `lib` contains the _python_ library;
* `notebooks` contains Jupyter notebooks with examples on how to use the library;
* `scripts` contains _python_ scripts to reproduce the results in the paper;
* `results` contains some of the results described in the paper;
* `plots` contains _python_ scripts for reproducing the plots in the paper;
* `tikz_process` contains _.tex_ files for reproducing Figure 1; 
* `fox` contains additional scripts for implementing the methodology of Fox et al. (2016).

## Methodology

The model and datasets are described in *Sanna Passino, F. and Heard, N. A. (2021+) "Mutually exciting point process graphs for modelling dynamic networks"* ([arXiv preprint](https://arxiv.org/abs/2102.06527)). 

## Understanding and running the code

The main part of the code is contained in the file `meg.py`, which contains a *python* class for the MEG model and inference using gradient ascent methods. 

For the simulation in Section 4.1, the file `simulation.py` is used. For fitting the model on the Enron and ICL data, the files `enron.py` and `icl.py` are used. Details about the possible options are given by the *help* function for each file. For example, running `simulation.py --help` returns: 

* `-f`: name of the destination folder for the output files,
* `-m`: Boolean variable for the main effects (default: FALSE),
* `-i`: Boolean variable for the interactions (default: FALSE),
* `-pm`: Boolean variable (used only if `-m` TRUE), if TRUE a Poisson process is fitted for the main effects (default: FALSE),
* `-pi`: Boolean variable (used only if `-i` TRUE), if TRUE a Poisson process is fitted for the interactions (default: FALSE),
* `-hm`: Boolean variable (used only if `-m` TRUE), if TRUE a Hawkes process is fitted for the main effects, otherwise a Markov process is used (default: FALSE),
* `-hi`: Boolean variable (used only if `-i` TRUE), if TRUE a Hawkes process is fitted for the interactions, otherwise a Markov process is used (default: FALSE),
* `-d`: number of latent features for the interaction term (default: 1),
* `-n`: number of nodes of the graph in the simulation (default: 10),
* `-T`: maximum time of observation for each simulated graph (default: 1000000),
* `-M`: number of simulated events for each graph (default: 10000),
* `-p`: probability of a link in the Erdős–Rényi graph (default: 0.5).

For example, the first simulation is obtained running the following command line:
```
./simulation.py -f simulation_1 -M 5000 -p 0.25 -n 10 -d 1 -m -i -hm -hi & 
```

Similar commands are used for the application on the Enron and ICL data. Running `./enron.py --help` gives two additional options:
* `-z`: Boolean variable, if TRUE <img src="svgs/672501aed245701fd96942cbb527a4f8.svg?invert_in_darkmode" align=middle width=48.90022829999999pt height=21.18721440000001pt/> for <img src="svgs/8947e2418bd54e1b12cad3cc94a795ca.svg?invert_in_darkmode" align=middle width=54.04292024999998pt height=22.465723500000017pt/>, and <img src="svgs/a02256ce6cb9e11c763f64297b938d88.svg?invert_in_darkmode" align=middle width=57.11942114999999pt height=14.15524440000002pt/> if <img src="svgs/22d019d180d7ea88f10cc25bd0e969e8.svg?invert_in_darkmode" align=middle width=54.04292024999998pt height=22.465723500000017pt/> (default: <img src="svgs/d4665663c67bdba16383ab9f10e52bb1.svg?invert_in_darkmode" align=middle width=17.94151424999999pt height=14.15524440000002pt/> set to its MLE);
* `-fl`: Boolean variable, if TRUE, <img src="svgs/672501aed245701fd96942cbb527a4f8.svg?invert_in_darkmode" align=middle width=48.90022829999999pt height=21.18721440000001pt/> for *all* links (default: FALSE).

For example, to obtain the best performing model on the Enron data, the following command line should be run:
```
./enron.py -m -hm -i -d 5 -z -f 'enron_results/tau_Aij/mi_hm_wi_5' &
```

## Reproducing the results in the paper

Since many of the simulations are computationally expensive to run, the output has been stored in the repository in the directories `simulation_main`, `simulation_inter`, `simulation_1` and `simulation_2`. Details on how to obtain such outputs are given in the following paragraphs.

The results, tables and figures in the paper could be reproduced using the following files:

### Figures

* *Figure 1* - Source `.tex` files to reproduce the figures are in the directory `tikz_process`.
* *Figure 2* - It can be reproduced running the following three files in succession:
    - `simulation_main_effects.sh` (WARNING: computationally demanding), which uses `simulation_main_effects.py` with argument `-s SEED`, and stores the simulated graphs in `.npy` files in `simulation_main`, with name `meg_simulate_SEED.npy`;
    - after simulating the grahs, the parameter estimation procedure is run using `estimate_simulation_main_effects.py`, which takes as argument `-n NUMBEREVENTS` the number of events to use for estimation. The output is saved in a directory `simulation_main/estimate_NUMBEREVENTS`. For reproducing *Figure 2*, the argument `-n 3000` should be used;
    - plots are obtained from `plots_simulation_main.py`, run with `-M NUMBEREVENTS` corresponding to the number of events used for inference (for *Figure 2*, `-M 3000`). 
* *Figure 3* - The procedure is similar to *Figure 2*: 
	- `simulation_interaction.sh` (WARNING: computationally demanding), which uses `simulation_interaction.py` with argument `-s SEED`, and stores the simulated graphs in `.npy` files in `simulation_inter`, with name `meg_simulate_SEED.npy`;
	- after simulating the grahs, the parameter estimation procedure is run using `estimate_simulation_interactions.py`, which takes as argument `-n NUMBEREVENTS` the number of events to use for estimation. The output is saved in a directory `simulation_inter/estimate_NUMBEREVENTS`. For reproducing *Figure 3*, the argument `-n 3000` should be used;
	- plots are obtained from `plots_simulation_inter.py`, run with `-M NUMBEREVENTS` corresponding to the number of events used for inference (for *Figure 3*, `-M 3000`).
* *Figure 4* - The plots can be obtained running `estimate_simulation_main_effects.py` and `plots_simulation_main.py` multiple times with arguments `-n` and `-M` 250, 500, 	1000, and 2000.
* *Figure 5* - The boxplots can be reproduced running `./simulation_erdos.sh`, followed by `estimate_simulation_erdos.sh` (both computationally expensive). The plot is then obtained by running followed by `boxplots.py`.
 
### Tables 
* *Table 1* - The results can be reproduced running `./enron_calls.sh` (running the entire file is **not** recommended, since the file contains command lines for **all** the 117 combinations of models in Table 1), which uses the file `enron.py`. Comparisons with the model of Fox et al. (2016) can be run using the files `fox_model.py` and `fox_enron.py`.

### Data
* The Enron data can be downloaded running `enron_filter.sh`;
* For security reasons, the ICL network data have **_not_** been made available, but the code to run the model on such networks (`icl.py`) is available.
