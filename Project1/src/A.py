import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, closed_form, MSE_score, r2_score

from matplotlib.colors import SymLogNorm

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

degree = 20
MSE_scores = np.zeros((degree, 2))
r2_scores = np.zeros((degree, 2))
thetas = np.zeros((degree, degree))

x_train, x_test, y_train, y_test = create_dataset()
for d in range(1, degree+1):
    X_train, X_test = create_feature_matrix(x_train, x_test, degree=d)

    theta_train = closed_form(X_train, y_train)

    Y_train = X_train @ theta_train
    Y_test = X_test @ theta_train
    MSE_scores[d-1, :] = [MSE_score(y_train, Y_train), MSE_score(y_test, Y_test)]
    r2_scores[d-1, :] = [r2_score(y_train, Y_train), r2_score(y_test, Y_test)]
    thetas[d-1, :d] = theta_train

min_test_d = np.argmin(MSE_scores[:, 1])+ 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, d+1), MSE_scores[:, 0], label="Train")
plt.plot(range(1, d+1), MSE_scores[:, 1], label="Test")
plt.axhline(0.1**2, ls='--', color='gray')
plt.axvline(min_test_d, ls='--', color='gray')
plt.xlabel("degree")
plt.ylabel(r"MSE")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'A-MSE_train_test_split.pdf'))

print(f"Lowest MSE score of {MSE_scores[min_test_d, 1]:.2e} is for degree {min_test_d}")


err = (r2_scores[:, 1] - r2_scores[:, 0])
print(err)

plt.figure(figsize=(10, 6))
plt.plot(range(1, d+1), r2_scores[:, 0], label="Train")
plt.plot(range(1, d+1), r2_scores[:, 1], label="Test")
plt.xlabel("degree")
plt.ylabel(r"R2 score")
plt.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'A-R2_train_test_split.pdf'))



M = np.full((degree, degree + 1), np.nan)
for i, th in enumerate(thetas):
    M[i, :len(th)] = th

plt.figure(figsize=(8, 6))
plt.imshow(M, aspect="auto", cmap="RdBu_r",
           norm=SymLogNorm(linthresh=1e-2, vmin=-np.nanmax(abs(M)), vmax=np.nanmax(abs(M))),
           origin="lower", extent=[-0.5, degree + 0.5, 0.5, degree + 0.5])
plt.colorbar(label=r"$\theta_j$")
plt.xlabel("parameter index $j$")
plt.ylabel("polynomial degree")
plt.grid(False)
plt.savefig(os.path.join(FIGURES_PATH, 'A-Thetas.pdf'))