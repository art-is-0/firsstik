import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, closed_form, MSE_score

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

n = 100
n_lambdas = 100
max_deg = 20
lambdas = np.logspace(-10, 3, n_lambdas)
degrees = np.linspace(1, 20, max_deg)

X, Y = np.meshgrid(degrees, lambdas, indexing='ij')
MSE = np.zeros((2, max_deg, n_lambdas))

x_train, x_test, y_train, y_test = create_dataset(n=100)

for d in range(1, max_deg+1):
    X_train, X_test = create_feature_matrix(x_train, x_test, degree=d)
    for idx, lam in enumerate(lambdas):
        theta_train = closed_form(X_train, y_train, lam=lam)

        Y_train = X_train @ theta_train
        Y_test = X_test @ theta_train
        MSE[:, d-1, idx] = [MSE_score(y_train, Y_train), MSE_score(y_test, Y_test)]


plt.figure(figsize=(8, 6))
plt.yscale('log')
plt.pcolor(X, Y, MSE[0], shading='auto', cmap='jet')
plt.colorbar(label="MSE")
plt.xlabel('degrees')
plt.ylabel(r'$\lambda$')
plt.savefig(os.path.join(FIGURES_PATH, 'B-Ridge_MSE_train.pdf'))

index = np.unravel_index(np.argmin(MSE[1]), (max_deg, n_lambdas))

plt.figure(figsize=(8,6))
plt.yscale('log')
plt.pcolor(X, Y, MSE[1], shading='auto', cmap='jet')
plt.colorbar(label="MSE")
plt.scatter(index[0] + 1, lambdas[index[1]], s=200, color='yellow', marker='*')
plt.xlabel('degrees')
plt.ylabel(r'$\lambda$')
plt.savefig(os.path.join(FIGURES_PATH, 'B-Ridge_MSE_test.pdf'))

print(f"Lowest MSE test score of {MSE[1, index[0], index[1]]:.2e} is for degree {index[0]+1} with lambda {lambdas[index[1]]:.2e}")

plt.figure(figsize=(10,6))
min_test_d = np.argmin(MSE[1], axis=0) + 1
plt.plot(lambdas, min_test_d)
plt.scatter(lambdas[index[1]], index[0] + 1, s=200, color='black', marker='*')
plt.ylabel("Minimum MSE test degree")
plt.xlabel(r"$\lambda$")
plt.xscale("log")
plt.savefig(os.path.join(FIGURES_PATH, 'B-Best_degree_for_each_lambda.pdf'))