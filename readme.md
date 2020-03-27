Simulator for the Lorenz 96 model. Uses Julia if available, otherwise falls back to numba, and finally to pure python.

### Installation
Install [delfi](https://github.com/mackelab/delfi).

For Julia support install the Julia language and [diffeqpy](https://github.com/SciML/diffeqpy).

### Compilation
There is a longish (about 1 minute) compile time on startup when using a python installation that has statically linked libraries. Unfortunately, this includes Anaconda. Fortunately, there is not compilation when starting a new simulation or changing simulation parameters.

### Notebooks
[solvers.ipynb](./solvers.ipynb) compares 3 different numerical solvers and plots the results. These are similar, but not identical as L96 is a chaotic system.

[inference.ipynb](./inference.ipynb) trains a neural network to perform Bayesian parameter inference, using the delfi package to apply the [APT](https://arxiv.org/abs/1905.07488) algorithm. 