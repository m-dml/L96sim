"""
Definition of the Lorenz96 model.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com

Update with julia/numba implementations and refactored by David Greenberg 02-2020
"""
import sys
import numpy as np
from delfi.simulator.BaseSimulator import BaseSimulator
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
import warnings


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


def f1(X, F, dX_dt, K):
    # awkward programming style is to support efficient JIT compilation
    for k in range(K):
        k_next = k + 1
        if k_next == K:
            k_next = 0

        dX_dt[k] = -X[k - 1] * (X[k - 2] - X[k_next]) - X[k] + F

    return dX_dt


f1_juliadef = """
function f(du, u, p, t)
  K = {K}
  F, = p
  for k = 1:K
  
    k_next = k == K ? 1 : k + 1
    k_prev = k == 1 ? K : k - 1
    k_prev2 = k_prev == 1 ? K : k_prev - 1
        
    du[k] = -u[k_prev] * (u[k_prev2] - u[k_next]) - u[k] + F
   
  end 
end
"""


f2_juliadef = """
function f(du, u, p, t)
    K = {K}
    J = {J}
    
    F, h, b, c = p
    
    for k = 1:K
    
        k_next = k == K ? 1 : k + 1
        k_prev = k == 1 ? K : k - 1
        k_prev2 = k_prev == 1 ? K : k_prev - 1
        
        x = u[k]
        x_prev = u[k_prev]
        x_prev2 = u[k_prev2]
        x_next = u[k_next]
        
        Y_mean = 0.0  # for this X
        o = K + (k - 1) * J  # offset, or index to the first Y for this X
        for j = 1:J   # for each Y 'belonging' to X_k
            
            j_prev = j == 1 ? J : j - 1
            j_next = j == J ? 1 : j + 1
            j_next2 = j_next == J ? 1 : j_next + 1 
                    
            y = u[o + j]
            y_prev = u[o + j_prev]
            y_next = u[o + j_next]
            y_next2 = u[o + j_next2]
            
            du[o + j] = (-b * y_next * (y_next2 - y_prev) - y + (h / J) * x) * c
            Y_mean += y
            
        end

        Y_mean /= J

        du[k] = -x_prev * (x_prev2 - x_next) - x + F - h * c * Y_mean
    
    end
        
end
"""


def f2(X_and_Y, F, h, b, c, dX_and_Y_dt, K, J):
    # ordering is all X, then all Y for first X, etc.
    # awkward programming style is to support efficient JIT compilation
    n = K * (J + 1)
    for k in range(K):  # for each X
        k_next = k + 1
        if k_next == K:
            k_next = 0
        k_prev = k - 1
        if k_prev == -1:
            k_prev = K - 1
        k_prev2 = k_prev - 1
        if k_prev2 == -1:
            k_prev2 = K - 1

        x = X_and_Y[k]
        x_prev = X_and_Y[k_prev]
        x_prev2 = X_and_Y[k_prev2]
        x_next = X_and_Y[k_next]

        Y_mean = 0.0  # for this X
        o = K + k * J  # offset, or index to the first Y for this X
        for j in range(J):  # for each Y 'belonging' to X_k
            ii = o + j  # index into X_and_Y for Y_{j,k}
            ii_next = ii + 1
            if ii_next == n:
                ii_next = K  # wrap around to first Y value
            ii_next2 = ii_next + 1
            if ii_next2 == n:
                ii_next2 = K
            ii_prev = ii - 1
            if ii_prev == K - 1:
                ii_prev = n - 1
            y = X_and_Y[ii]
            y_prev = X_and_Y[ii_prev]
            y_next = X_and_Y[ii_next]
            y_next2 = X_and_Y[ii_next2]

            dX_and_Y_dt[ii] = (-b * y_next * (y_next2 - y_prev) - y + (h / J) * x) * c

            Y_mean += y

        Y_mean /= J

        dX_and_Y_dt[k] = -x_prev * (x_prev2 - x_next) - x + F - h * c * Y_mean

    return dX_and_Y_dt


def J1(X, F, df_dX, n):
    for i in range(n):
        i_next = i + 1
        if i_next == n:
            i_next = 0

        df_dX[i, i] = -1.0
        df_dX[i, i - 1] = -(X[i - 2] - X[i_next])
        df_dX[i, i - 2] = -X[i - 1]
        df_dX[i, i_next] = X[i - 1]

    return df_dX


def J1_init(n):
    J = np.zeros((n, n))
    for i in range(n):
        i_next = i + 1
        if i_next == n:
            i_next = 0

        J[i, i]= 1.0
        J[i, i - 1] = 1.0
        J[i, i - 2] = 1.0
        J[i, i_next] = 1.0

    return csc_matrix(J)


try:
    from numba import jit
    f1 = jit(f1)
    f2 = jit(f2)
    numba_available = False
except ImportError:
    warnings.warn("numba is not available, using slower python code")
    numba_available = True

try:
    print('Updating julia wrappers, compilation might take a while....')
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from diffeqpy import de
    from julia import Main
    julia_available = True
    print('...julia wrappers have been updated.')
except ModuleNotFoundError:
    warnings.warn("Julia is not available, using slower python integrator. Per-seed results will change!!")
    julia_available = False


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

        if julia_available:
            self.f_juliadef = Main.eval(f1_juliadef.format(K=K))

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
            assert julia_available

            if use_juliadef:

                f_julia = self.f_juliadef
                p = np.array([F], dtype=X_init.dtype)
                problem = de.ODEProblem(f_julia, X_init, tspan, p)

            else:

                def f_julia(X, p, t):
                    return f(t, X)

                problem = de.ODEProblem(f_julia, X_init, tspan)

            self.sol = de.solve(problem)
            u = np.array(self.sol(self.obs_nsteps * self.dt))
            return {'data': u[self.obs_X, :].T.reshape(-1)}

        else:

            opts = dict()
            if method in ['Radau', 'BDF'] and not numba_available:  # numba doesn't work with sparse matrices
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

        if julia_available:
            self.f_juliadef = Main.eval(f2_juliadef.format(K=K, J=J))

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
            assert julia_available

            if use_juliadef:

                f_julia = self.f_juliadef
                p = np.array([F, h, b, c], dtype=X_and_Y_init.dtype)
                problem = de.ODEProblem(f_julia, X_and_Y_init, tspan, p)

            else:

                def f_julia(X_and_Y, p, t):
                    return f(t, X_and_Y)

                problem = de.ODEProblem(f_julia, X_and_Y_init, tspan)

            self.sol = de.solve(problem)
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
