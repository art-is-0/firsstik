import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, closed_form, optimise, cost_OLS, cost_Lasso, cost_Ridge, MSE_score

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

import jax
jax.config.update("jax_enable_x64", True)
from jax import grad


degree = 5
gamma = 1e-2
num_iters = int(1e4)

x_train, x_test, y_train, y_test = create_dataset(n=100)
X_train, X_test = create_feature_matrix(x_train, x_test, degree=degree)


RUNS = (
    (lambda th: grad(cost_OLS)(th, X_train, y_train), r"OLS"),
    (lambda th: grad(cost_Ridge)(th, X_train, y_train, 1e-2), r"Ridge, $\lambda=10^{-2}$"),
    (lambda th: grad(cost_Lasso)(th, X_train, y_train, 1e-5), r"$\lambda=10^{-5}$"),
    (lambda th: grad(cost_Lasso)(th, X_train, y_train, 1e-4), r"$\lambda=10^{-4}$"),
    (lambda th: grad(cost_Lasso)(th, X_train, y_train, 1e-3), r"$\lambda=10^{-3}$"),
    (lambda th: grad(cost_Lasso)(th, X_train, y_train, 1e-2), r"$\lambda=10^{-2}$"),
    (lambda th: grad(cost_Lasso)(th, X_train, y_train, 1e-1), r"$\lambda=10^{-1}$"),
)

theta_train = closed_form(X_train, y_train)
Y_test = X_test @ theta_train

fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(MSE_score(y_test, Y_test), linestyle='--', color='k', label="OLS analytic")

for costRUN, label in RUNS:
    path, _ = optimise(costRUN, np.zeros(degree), "plain", gamma, num_iters=num_iters)
    path = np.asarray(path)
    mse = np.mean((path @ X_test.T - y_test)**2, axis=1) # vectorised over iterations
    ax.plot(np.arange(len(mse)), mse, label=label)

ax.set_xscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel("test MSE")
ax.legend()
plt.savefig(os.path.join(FIGURES_PATH, f'G-Ridge_test_MSE_per_iter.pdf'))