"""
Uses code from:
Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com



Update with julia/numba implementations and refactored by David Greenberg 02-2020
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
import warnings

from L96sim import L96_base
from L96sim.L96_base import f1, f2, J1, J1_init, f1_juliadef, f2_juliadef

def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules


if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class BaseSimulator(object):
    def __init__(self, dim_param, seed=None):
        """Abstract base class for simulator models
        Simulator models must at least implement abstract methods and properties
        of this class.
        Parameters
        ----------
        dim_param : int
            Dimensionality of parameter vector
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.dim_param = dim_param

        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed

    def gen(self, params_list, n_reps=1, pbar=None):
        """Forward model for simulator for list of parameters
        Parameters
        ----------
        params_list : list of lists or 1-d np.arrays
            List of parameter vectors, each of which will be simulated
        n_reps : int
            If greater than 1, generate multiple samples given param
        pbar : tqdm.tqdm or None
            If None, will do nothing. Otherwise it will call pbar.update(1)
            after each sample.
        Returns
        -------
        data_list : list of lists containing n_reps dicts with data
            Repetitions are runs with the same parameter set, different
            repetitions. Each dictionary must contain a key data that contains
            the results of the forward run. Additional entries can be present.
        """
        data_list = []
        for param in params_list:
            rep_list = []
            for r in range(n_reps):
                rep_list.append(self.gen_single(param))
            data_list.append(rep_list)
            if pbar is not None:
                pbar.update(1)

        return data_list

    def gen_single(self, params):
        """Forward model for simulator for single parameter set
        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector
        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        pass

    def reseed(self, seed):
        """Reseeds the distribution's RNG"""
        self.rng.seed(seed=seed)
        self.seed = seed

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)


class L96OneSim(BaseSimulator):
    def __init__(self, dim=1, noise_obs=0.1, K=36, seed=None, dt=0.001,
                 obs_times=None, obs_X=None):
        if obs_times is None:
            obs_times = [10.0]
        obs_nsteps = np.unique([np.ceil(t / dt) for t in obs_times]).astype(int)

        if obs_X is None:
            obs_X = np.arange(0, K, 4)
        else:
            assert np.all(obs_X >= 0) and np.all(obs_X < K)
        super().__init__(dim_param=dim, seed=seed)
        self.noise_obs, self.K, self.obs_nsteps, self.obs_X, self.dt = noise_obs, K, obs_nsteps, obs_X, dt

        if L96_base.julia_available:
            self.f_juliadef = L96_base.Main.eval(f1_juliadef.format(K=K))

    def gen_single(self, param, use_julia=True, use_juliadef=True, method='RK45'):
        # method can be LSODA, BDF, Radau or RK45. used only when use_julia is False
        assert param.ndim == 1 and param.size == 1
        F = param[0]

        X_init = F * (0.5 + self.rng.randn(self.K) * 1.0)
        tspan = (0, self.obs_nsteps[-1] * self.dt)

        n = X_init.size
        dX_dt = np.empty(n, dtype=X_init.dtype)
        df_dX = J1_init(n).astype(X_init.dtype)  # initialize as a sparse matrix

        def f(t, X):
            return f1(X, F, dX_dt, n)

        def J(t, X):
            return J1(X, F, df_dX, n)

        if use_julia:
            assert L96_base.julia_available

            if use_juliadef:

                f_julia = self.f_juliadef
                p = np.array([F], dtype=X_init.dtype)
                problem = L96_base.de.ODEProblem(f_julia, X_init, tspan, p)

            else:

                def f_julia(X, p, t):
                    return f(t, X)

                problem = L96_base.de.ODEProblem(f_julia, X_init, tspan)

            self.sol = L96_base.de.solve(problem)
            u = np.array(self.sol(self.obs_nsteps * self.dt))
            return {'data': u[self.obs_X, :].T.reshape(-1)}

        else:

            opts = dict()
            if method in ['Radau', 'BDF'] and not L96_base.numba_available:  # numba doesn't work with sparse matrices
                opts['jac'] = J

            self.sol = solve_ivp(f, tspan, X_init, method=method,
                                 max_step=0.1,
                                 t_eval=self.obs_nsteps * self.dt, **opts)
            return {'data': self.sol.y[self.obs_X, :].T.reshape(-1)}


class L96TwoSim(BaseSimulator):
    def __init__(self, dim=4, noise_obs=0.1, K=36, J=10, seed=None, dt=0.001,
                 obs_times=None, obs_X=None):
        if obs_times is None:
            obs_times = [10.0]
        obs_nsteps = np.unique([np.ceil(t / dt) for t in obs_times]).astype(int)

        if obs_X is None:
            obs_X = np.arange(0, K, 4)
        else:
            assert np.all(obs_X >= 0) and np.all(obs_X < K)
        super().__init__(dim_param=dim, seed=seed)
        self.noise_obs, self.K, self.J, self.obs_nsteps, self.obs_X, self.dt = noise_obs, K, J, obs_nsteps, obs_X, dt
        
        obs_Y = np.zeros((K, J))
        obs_Y[self.obs_X, :] = 1
        self.obs_Y = np.where( obs_Y.flatten() )[0]
        self.obs_X_and_Y = np.concatenate((self.obs_X, self.obs_Y + K))

        if L96_base.julia_available:
            self.f_juliadef = L96_base.Main.eval(f2_juliadef.format(K=K, J=J))

    def gen_single(self, param, use_julia=True, use_juliadef=True, method='LSODA'):
        # method can be LSODA, BDF, Radau or RK45. used only when use_julia is False
        assert param.ndim == 1 and param.size == 4
        F, h, b, c = param[0], param[1], param[2], np.exp(param[3])

        X_init = self.rng.randn(self.K)
        Y_init = self.rng.randn(self.K * self.J)
        X_and_Y_init = np.concatenate((X_init, Y_init))
        tspan = (0, self.obs_nsteps[-1] * self.dt)

        dX_and_Y_dt = np.empty_like(X_and_Y_init)

        def f(t, X_and_Y):
            return f2(X_and_Y, F, h, b, c, dX_and_Y_dt, self.K, self.J)

        if use_julia:
            assert L96_base.julia_available

            if use_juliadef:

                f_julia = self.f_juliadef
                p = np.array([F, h, b, c], dtype=X_and_Y_init.dtype)
                problem = L96_base.de.ODEProblem(f_julia, X_and_Y_init, tspan, p)

            else:

                def f_julia(X_and_Y, p, t):
                    return f(t, X_and_Y)

                problem = L96_base.de.ODEProblem(f_julia, X_and_Y_init, tspan)

            self.sol = L96_base.de.solve(problem)
            u = np.array(self.sol(self.obs_nsteps * self.dt))
            obs_data = u[self.obs_X_and_Y, :]

        else:

            opts = dict()
            # Jacobian not implented in python yet
            #if method in ['Radau', 'BDF']:
            #   opts['jac'] = J

            self.sol = solve_ivp(f, tspan, X_and_Y_init, method=method,
                                 max_step=0.1,
                                 t_eval=self.obs_nsteps * self.dt, **opts)
            obs_data = self.sol.y[self.obs_X_and_Y, :]

        obsx = obs_data[:len(self.obs_X), :].T.reshape(-1)
        obsy = obs_data[len(self.obs_X):, :].T.reshape(-1)
        return {'data': (obsx, obsy)}
