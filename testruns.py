import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from scipy.integrate import cumulative_simpson as cumsimp
from scipy.linalg import cholesky, solve_triangular

import dickandballs as db


def fit(x1, x2, f, kernel):

    kernel = kernel
    K_11 = kernel.compute(x1)
    K_12 = kernel.compute(x1, x2)
    K_22 = kernel.compute(x2)

    L = cholesky(K_11, lower=True)
    y = solve_triangular(L, f, lower=True)
    alpha = solve_triangular(L.T, y, lower=False)
    mean_star = K_12.T @ alpha

    v = solve_triangular(L, K_12, lower=True)
    cov_star = K_22 - (v.T @ v)

    bruh = v.T@v
    if x1.shape == x2.shape:
        jitter  = 1e-10
        bruh += np.diag(np.full(bruh.shape[0], jitter))
    np.linalg.cholesky(bruh)

    return mean_star, cov_star

x2 = np.linspace(0, 50, 20)
x1 = np.array([1,40])
f1 = np.array([0.033545,1/3])

my_kernel = db.Kernel('SE', sigma=1, l=1)

mean_star, cov_star = fit(x1, x2, f1, my_kernel)