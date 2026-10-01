import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, closed_form, MSE_score

BLUE, RED, YELLOW = "#004488", "#BB5566", "#DDAA33" 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

def bias_variance_sweep(n=40, maxdegree=14, n_bootstraps=100, make_model=None):
    """Bootstrap estimate of error/bias^2/variance vs polynomial degree."""
    if make_model is None:
        make_model = lambda deg: make_pipeline(
            PolynomialFeatures(degree=deg), LinearRegression(fit_intercept=False))
        
    x_train, x_test, y_train, y_test = create_dataset(n=n, rescale=False)
    x_train = x_train.reshape(-1, 1); x_test = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1); y_test = y_test.reshape(-1, 1)
    error, bias, variance = (np.zeros(maxdegree) for _ in range(3))
    for degree in range(maxdegree):
        model = make_model(degree)
        y_pred = np.empty((y_test.shape[0], n_bootstraps))
        for i in range(n_bootstraps):
            x_, y_ = resample(x_train, y_train)
            y_pred[:, i] = model.fit(x_, y_).predict(x_test).ravel()
        error[degree] = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias[degree] = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance[degree] = np.mean(np.var(y_pred, axis=1, keepdims=True))
    return error, bias, variance


degree = 26
MSE_scores = np.zeros((degree, 2))

n = 100
x_train, x_test, y_train, y_test = create_dataset(n=n)
for d in range(1, degree+1):
    X_train, X_test = create_feature_matrix(x_train, x_test, degree=d)

    theta_train = closed_form(X_train, y_train)

    Y_train = X_train @ theta_train
    Y_test = X_test @ theta_train
    MSE_scores[d-1, :] = [MSE_score(y_train, Y_train), MSE_score(y_test, Y_test)]

min_test_d = np.argmin(MSE_scores[:, 1])+ 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, d+1), MSE_scores[:, 0], label="Train")
plt.plot(range(1, d+1), MSE_scores[:, 1], label="Test")
plt.axhline(0.1**2, ls='--', color='gray')
plt.axvline(min_test_d, ls='--', color='gray')
plt.yscale("log")
plt.xlabel("degree | Complexity")
plt.ylabel(r"MSE | Prediction error")
plt.legend(loc="lower left")
plt.savefig(os.path.join(FIGURES_PATH, 'C-Train_Test_split.pdf'))


fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
for ax, n_ in zip(axes, (40, 100, 400)):
    err, b2, var = bias_variance_sweep(n=n_, maxdegree=degree)

    print(f"Minimum MSE for n={n_}, is {err.min():.4f} on degree {np.argmin(err)}")
    ax.plot(range(degree), err, "o-", color=BLUE, label="test error")
    ax.plot(range(degree), b2, "s-", color=RED, label=r"bias$^2$")
    ax.plot(range(degree), var, "d-", color=YELLOW, label="variance")
    ax.set_yscale("log")
    ax.set_title(f"$n = {n_}$")
    ax.set_ylabel("MSE decompositon")
axes[2].set_xlabel("polynomial degree")
axes[0].legend()
plt.savefig(os.path.join(FIGURES_PATH, 'C-Bias_Variance_Sweep.pdf'))
