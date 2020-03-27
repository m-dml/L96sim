Simulator for the Lorenz 96 model. Uses Julia if available, otherwise falls back to numba, and finally to pure python.

### Installation
Install [delfi](https://github.com/mackelab/delfi).

For Julia support install the Julia language and [diffeqpy](https://github.com/SciML/diffeqpy).

### Compilation
If Julia and pydiffeq are installed, there will be about 1 minute Julia compile time on startup when using a python installation that has statically linked libraries. Unfortunately, this includes Anaconda. Fortunately, there is no compilation required when starting a new simulation, instantiating a new simulation object or changing simulation parameters.

### Notebooks
[solvers.ipynb](./solvers.ipynb) compares 3 different numerical solvers and plots the results. These are similar, but not identical as L96 is a chaotic system.

[inference.ipynb](./inference.ipynb) trains a neural network to perform Bayesian parameter inference, using the delfi package to apply the [APT](https://arxiv.org/abs/1905.07488) algorithm. 