import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, k_fold_degrees

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

k = 5
n = 100
degree = 26
x, y = create_dataset(n=n, rescale=False, split=False)
x = x.reshape(-1, 1); y = y.reshape(-1, 1)

kfold = KFold(n_splits=k, shuffle=True, random_state=67)
cv_mse = np.zeros(degree)

fig, ax = plt.subplots(figsize=(10, 6))
for k in (5, 10):
    cv_mse = k_fold_degrees(x, y, k, LinearRegression(fit_intercept=False))
    best_deg = np.argmin(cv_mse)
    print(f"{k}-fold CV selects degree {best_deg} (CV-MSE {cv_mse[best_deg]:.4f})")
    ax.plot(range(degree), cv_mse, "o-", label=f"{k}-fold CV MSE")
    ax.scatter(best_deg, cv_mse[best_deg], s=200, color='black', marker='*',label="CV minimum")
ax.set_yscale("log")
ax.set_xlabel("polynomial degree")
ax.set_ylabel("cross-validated MSE")
ax.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'D-Kfold_CV_MSE.pdf'))


n_lambdas = 100
degree = 26
lambdas = np.logspace(-10, 3, n_lambdas)
degrees = np.linspace(1, degree, degree)

X, Y = np.meshgrid(degrees, lambdas, indexing='ij')
MSE = np.zeros((degree, n_lambdas))


# ==========================================



for k in (5, 10):
    for idx, lam in enumerate(lambdas):
        cv_mse = k_fold_degrees(x, y, k, Ridge(alpha=lam), maxdegree=degree)

        MSE[:, idx] = cv_mse

    index = np.unravel_index(np.argmin(MSE), (degree, n_lambdas))

    plt.figure(figsize=(8, 6))
    plt.yscale('log')
    plt.pcolor(X, Y, MSE, shading='auto', cmap='jet')
    plt.colorbar(label="MSE")
    plt.scatter(index[0] + 1, lambdas[index[1]], s=200, color='yellow', marker='*')
    plt.xlabel('degrees')
    plt.ylabel(r'$\lambda$')
    plt.savefig(os.path.join(FIGURES_PATH, f'D-K_fold_ridge_k-{k}.pdf'))

    print(f"Lowest MSE test score of {MSE[index[0], index[1]]:.2e} is for degree {index[0]+1} with lambda {lambdas[index[1]]:.2e}")