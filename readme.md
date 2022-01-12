Simulator for the Lorenz 96 model. Uses Julia if available, otherwise falls back to numba, and finally to pure python.

### Installation
`pip install .`

With this basic installation, you can use the gradient functions in `L96_base.py`.

For Julia support install the Julia language and [diffeqpy](https://github.com/SciML/diffeqpy).

If you don't have Julia, numba will be used instead. If numba also isn't present, interpreted numpy will be used.

### Compilation
If Julia and pydiffeq are installed, there will be about 1 minute Julia compile time on startup when using a python installation that has [statically linked libraries](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html). Unfortunately, this includes Anaconda. Fortunately, there is no compilation required when starting a new simulation, instantiating a new simulation object or changing simulation parameters.

### Notebooks
[solvers.ipynb](./solvers.ipynb) compares 3 different numerical solvers and plots the results. These are similar, but not identical as L96 is a chaotic system.

[inference.ipynb](./inference.ipynb) trains a neural network to perform Bayesian parameter inference with the [APT](https://arxiv.org/abs/1905.07488) algorithm. To use it, you will have to install [delf](https://github.com/mackelab/delfi).