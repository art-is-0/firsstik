import numpy as np
import matplotlib.pyplot as plt
import matplotlib_params
import os
from utils import create_dataset, create_feature_matrix, closed_form, optimiser_step, gradient, MSE_score

filepath = os.path.dirname(os.path.abspath(__file__))
FIGURES_PATH = os.path.join(filepath, '../figures')

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0 / (t + t1)

def sgd(X, y, method="plain", n_epochs=50, batch_size=5, gamma=0.1, schedule=None, lam=0.0, seed=2026, **kw):
    """Minibatch SGD, Eq. (4.34), with any optimiser. schedule=(t0, t1) replaces gamma by Eq. (4.40).
    Returns the iterate after every epoch."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta, state, t = np.zeros(p), {}, 0
    history = [theta.copy()]
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            g = gradient(theta, X[batch], y[batch])                                  # the gradient of the cost on this minibatch
            gamma_t = gamma if schedule is None else step_length(t, *schedule)                             # constant, or the schedule
            theta, state = optimiser_step(method, theta, g, state, t, gamma_t, **kw)
        history.append(theta.copy())
    return np.array(history)

degree = 5
gamma = 1e-2
num_iters = int(1e4)
stride = 1

n_epochs = 100
n = 100

x_train, x_test, y_train, y_test = create_dataset(n=n)
X_train, X_test = create_feature_matrix(x_train, x_test, degree=degree)


theta_cf = closed_form(X_train, y_train)
m = len(y_test)

evals = np.arange(n_epochs + 1)
RUNS = (
        (dict(batch_size=n, gamma=1e-3), r"full batch, $\gamma=10^{-3}$"),
        (dict(batch_size=15, gamma=1e-3), r"$M=15$, $\gamma=10^{-3}$ constant"),
        (dict(batch_size=5, gamma=1e-3), r"$M=5$, $\gamma=10^{-3}$ constant"),
        (dict(batch_size=5, schedule=(1.0, 1e4)), r"$M=5$, $\gamma_t=1/(t+10^{-4})$")
)

fig, ax = plt.subplots(figsize=(10, 6))
for kwargs, lab in RUNS:
    hist = sgd(X_train, y_train, n_epochs=n_epochs, **kwargs)
    ax.semilogy(evals, MSE_score(hist @ X_test.T, y_test), lw=1.6, label=lab)
ax.set_xlabel("iterations")
ax.set_ylabel(r"SGD MSE")
ax.legend(loc="upper right")
plt.savefig(os.path.join(FIGURES_PATH, f'H-SGD_different_batch_sizes.pdf'))
