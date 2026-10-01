import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, k_fold_degrees

from sklearn.linear_model import LinearRegression, Ridge, Lasso

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

k = 5
n = 100
degree = 26
n_lambdas = 30
degrees = np.linspace(1, degree, degree)
lambdas = np.logspace(-10, -6, n_lambdas)

x, y = create_dataset(n=n, rescale=False, split=False)
x = x.reshape(-1, 1)
y = y.ravel()

cv_mse_ridge = np.zeros((degree, n_lambdas))
cv_mse_lasso = np.zeros((degree, n_lambdas))
cv_mse_OLS = np.zeros(degree)

OLS = LinearRegression(fit_intercept=False)

cv_mse_OLS = k_fold_degrees(x, y, k, LinearRegression(), degree)

for idx, lam in enumerate(lambdas):
    cv_mse_ridge[:, idx] = k_fold_degrees(x, y, k, Ridge(alpha=lam), degree)
    cv_mse_lasso[:, idx] = k_fold_degrees(x, y, k, Lasso(alpha=lam, max_iter=10_000, tol=1e-3), degree)


i_r, j_r = np.unravel_index(np.argmin(cv_mse_ridge), cv_mse_ridge.shape)
i_l, j_l = np.unravel_index(np.argmin(cv_mse_lasso), cv_mse_lasso.shape)
print(f"Ridge: degree={degrees[i_r]}, lambda={lambdas[j_r]:.2e}, MSE={cv_mse_ridge[i_r, j_r]:.3e}")
print(f"Lasso: degree={degrees[i_l]}, lambda={lambdas[j_l]:.2e}, MSE={cv_mse_lasso[i_l, j_l]:.3e}")

plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_mse_OLS, label='OLS')
plt.plot(degrees, cv_mse_lasso[:, j_l], label=rf"Lasso, $\lambda={lambdas[j_l]:.1e}$")
plt.plot(degrees, cv_mse_ridge[:, j_r], label=rf"Ridge, $\lambda={lambdas[j_r]:.1e}$")
plt.axhline(0.1**2, ls='--', color='gray', label="Data error")
plt.yscale("log")
plt.xlabel(r"Polynomial degree $(N)$")
plt.ylabel("Cross Validation MSE")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'I-OLS_Ridge_Lasso_CV_compare.pdf'))
