import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, cost_OLS, cost_Ridge, gradient_descent, hessian_eigs, MSE_score

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')


x_train, x_test, y_train, y_test = create_dataset()
X_train, X_test = create_feature_matrix(x_train, x_test, degree=2)

rng = np.random.default_rng(67)
theta_test = rng.standard_normal(2)
grad_OLS_ad = grad(cost_OLS)(theta_test, jnp.asarray(X_train), jnp.asarray(y_train))
grad_OLS_analytical = 2*X_train.T @ (X_train @ theta_test - y_train) / len(y_train)
print("max |AD - analytical| =", np.max(np.abs(np.asarray(grad_OLS_ad) - grad_OLS_analytical)))

lam = 1e-7
grad_Ridge_ad = grad(cost_Ridge)(theta_test, jnp.asarray(X_train), jnp.asarray(y_train), lam)
grad_Ridge_analytical = 2*X_train.T @ (X_train @ theta_test - y_train) / len(y_train) + 2*lam*theta_test
print("max |AD - analytical| =", np.max(np.abs(np.asarray(grad_Ridge_ad) - grad_Ridge_analytical)))

# ==================================

# OLS test
degree = 15
x_train, x_test, y_train, y_test = create_dataset(n=100)
diff = np.zeros((2, degree))

for d in range(1, degree + 1):
    X_train, X_test = create_feature_matrix(x_train, x_test, degree=d)

    eigs = hessian_eigs(X_train)
    lmax, lmin = eigs.max(), eigs.min()

    gammas = np.array([0.9 * 2 / lmax, 2/(lmax + lmin)])

    for idx, gamma in enumerate(gammas):
        with np.errstate(all="ignore"):
            hist, its, err = gradient_descent(X_train, y_train, gamma, num_iters=20000)

        diff[idx, d-1] = err

plt.figure(figsize=(10, 6))
plt.plot(range(1, degree+1), diff[0], label=r"$\gamma = 0.9\cdot 2 /\lambda_{\max}$")
plt.plot(range(1, degree+1), diff[1], label=r"$\gamma = 2 / (\lambda_{\max} + \lambda_{\min}) $")
plt.yscale("log")
plt.xlabel("degrees")
plt.ylabel("|theta - closed form|")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, f'E-OLS_diff_between_optimal_descent.pdf'))
plt.show()


# ====================================

# Ridge test
n_lambdas = 50
max_deg = 15
lambdas = np.logspace(-7, 5, n_lambdas)
degrees = np.linspace(1, max_deg, max_deg)

X, Y = np.meshgrid(degrees, lambdas, indexing='ij')
diff = np.zeros((2, max_deg, n_lambdas))
iters = np.zeros((2, max_deg, n_lambdas))
x_train, x_test, y_train, y_test = create_dataset(n=100)

for d in range(1, degree + 1):
    X_train, X_test = create_feature_matrix(x_train, x_test, degree=d)


    # print(f"\tdegree = {degree}, condition number = {lmax/lmin:.2e}, best rate = {2/(lmax + lmin):.3e}, limit = {2/lmax:.3e}")
    for idy, lam in enumerate(lambdas):
        eigs = hessian_eigs(X_train, lam)
        lmax, lmin = eigs.max(), eigs.min()

        gammas = np.array([0.9 * 2 / lmax, 2/(lmax + lmin)])
        for idx, gamma in enumerate(gammas):

            with np.errstate(all="ignore"):
                hist, its, err = gradient_descent(X_train, y_train, gamma, num_iters=2000, lam=lam)

            diff[idx, d-1, idy] = err
            iters[idx, d-1, idy] = its


from matplotlib.colors import LogNorm

index = np.unravel_index(np.nanargmin(diff[0]), (max_deg, n_lambdas))

plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.pcolor(X, Y, diff[0], shading='auto', cmap='jet', norm=LogNorm(vmax=1e3))
plt.colorbar(label="|theta - closed form|")
plt.scatter(index[0] + 1, lambdas[index[1]], s=200, color='yellow', marker='*')
plt.xlabel('degrees')
plt.ylabel(r'$\lambda$')
plt.savefig(os.path.join(FIGURES_PATH, f'E-Ridge_diff_0.9_optimal_descent.pdf'))

print(f"For gamma = 0.9 * 2 / lmax")
print(f"Lowest |theta - closed form| score of {diff[0,index[0], index[1]]:.2e} for degree {index[0] + 1} with lambda {lambdas[index[1]]:.2e} did {iters[0, index[0], index[1]]:.0f} iterations")

X_train, X_test = create_feature_matrix(x_train, x_test, degree=index[0] + 1)
lam = lambdas[index[1]]
eigs = hessian_eigs(X_train, lam)
lmax, lmin = eigs.max(), eigs.min()
hist, its, err = gradient_descent(X_train, y_train, 0.9 * 2 / lmax, num_iters=2000, lam=lam)
Y_test = X_test @ hist[-1]
MSE = MSE_score(y_test, Y_test)
print(f"The given MSE is {MSE:.2e}")





index = np.unravel_index(np.nanargmin(diff[1, 1:]), (max_deg-1, n_lambdas))
plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.pcolor(X, Y, diff[1], shading='auto', cmap='jet', norm=LogNorm(vmax=1e3))
plt.colorbar(label="|theta - closed form|")
plt.scatter(index[0] + 1, lambdas[index[1]], s=200, color='yellow', marker='*')
plt.xlabel('degrees')
plt.ylabel(r'$\lambda$')
plt.savefig(os.path.join(FIGURES_PATH, f'E-Ridge_diff_optimal_descent.pdf'))

print(f"For optimal gamma")
print(f"Lowest |theta - closed form| score of {diff[1,index[0], index[1]]:.2e} for degree {index[0] + 1} with lambda {lambdas[index[1]]:.2e} did {iters[1, index[0], index[1]]:.0f} iterations")

X_train, X_test = create_feature_matrix(x_train, x_test, degree=index[0] + 1)
lam = lambdas[index[1]]
eigs = hessian_eigs(X_train, lam)
lmax, lmin = eigs.max(), eigs.min()
hist, its, err = gradient_descent(X_train, y_train, 2 / (lmax+lmin), num_iters=2000, lam=lam)
Y_test = X_test @ hist[-1]
MSE = MSE_score(y_test, Y_test)
print(f"The given MSE is {MSE:.2e}")