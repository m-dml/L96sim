"""
Definition of the Lorenz96 model.

Created on 2019-04-16-12-28
Author: Stephan Rasp, raspstephan@gmail.com

Update with julia/numba implementations and refactored by David Greenberg 02-2020
"""
import numpy as np
from scipy.sparse import csc_matrix
import warnings


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


def pf2(X_and_Y, F, h, b, c, dX_and_Y_dt, K, J):
    # parallelized version of two-level L96 model (takes parallelization from input shape)
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

        x = X_and_Y[k,:]
        x_prev = X_and_Y[k_prev,:]
        x_prev2 = X_and_Y[k_prev2,:]
        x_next = X_and_Y[k_next,:]

        o = K + k * J  # offset, or index to the first Y for this X
        Y_mean = np.zeros(X_and_Y[o,:].shape)  # for this X
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
            y = X_and_Y[ii,:]
            y_prev = X_and_Y[ii_prev,:]
            y_next = X_and_Y[ii_next,:]
            y_next2 = X_and_Y[ii_next2,:]

            dX_and_Y_dt[ii,:] = (-b * y_next * (y_next2 - y_prev) - y + (h / J) * x) * c

            Y_mean += y

        Y_mean /= J

        dX_and_Y_dt[k,:] = -x_prev * (x_prev2 - x_next) - x + F - h * c * Y_mean

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
    pf2 = jit(pf2)
    numba_available = False
except ImportError:
    warnings.warn("numba is not available, using slower python code")
    numba_available = True

try:
    print('Updating julia wrappers, compilation might take a while....')
    from julia.api import Julia
    jl = Julia(compiled_modules=False, runtime='/gpfs/home/nonnenma/julia-1.5.0/bin/julia')
    from diffeqpy import de
    from julia import Main
    julia_available = True
    print('...julia wrappers have been updated.')
except ModuleNotFoundError:
    warnings.warn("Julia is not available, using slower python integrator. Per-seed results will change!!")
    julia_available = False

