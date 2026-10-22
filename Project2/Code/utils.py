import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score



def create_dataset(n=100, sigma=0.1, seed=67, rescale=True, split=True):
    '''
    Returns noisy samples of the Runge function 1/(1+25x²) on [-1, 1].

    Parameters
    --------------
    n : int
        Number of data points
    sigma : float
        Standard deviation of the added Gaussian noise
    seed : int
        Seed for the rng and the train/test split
    rescale : bool
        If True, standardize x and center y
    split : bool
        If True, split the data into training and test sets (67/33)

    Returns
    ------
    x, y : arraylikes

    or

        **x_train, x_test, y_train, y_test**: arrays

    '''
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1.0, 1.0, n)
    y = 1 / (1 + 25* x*x) + sigma * rng.standard_normal((n))

    if rescale:
        x = (x - x.mean()) / x.std()
        y = y - y.mean()

    if split:
        return train_test_split(x, y, test_size=0.33, random_state=seed)

    return x, y

def create_feature_matrix(*args, degree=2):
    """
    Creates Vandermonde matrix for multiple sets of arrays

    Parameters
    ----------
    *args : arraylikes
        Arrays containing x-points
    degree : int
        Degree of the constucted Vandermonde matrix

    Returns
    -------
    X : arraylikes 
        Arrays of size (n x d)
    """
    out_list = []
    for arg in args:
        out = np.column_stack([arg**k for k in range(1, degree + 1)])
        out_list.append(out)

    if len(out_list) != 1:
        return out_list
    
    return out_list[0]

def closed_form(X, y, lam=0.0):
    """(X^T X + n lambda I)^-1 X^T y, the 1/n convention of Eq. (3.95)."""
    n, p = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(p), X.T @ y)

def cost(theta, X, y, lam=0.0):
    """Eqs. (4.12) and (4.16)."""
    return np.sum((X @ theta - y) ** 2) / len(y) + lam * theta @ theta

def cost_OLS(theta, X, y):
    """OLS cost function for JAX"""
    return jnp.sum((X @ theta - y)**2) / len(y)

def cost_Ridge(theta, X, y, lam):
    """Ridge cost function for JAX"""
    return jnp.sum((X @ theta - y)**2) / len(y) + lam * jnp.sum(theta**2)

def cost_Lasso(theta, X, y, lam):
    """Lasso cost function for JAX"""
    return jnp.sum((X @ theta - y)**2) / len(y) + lam * jnp.sum(jnp.abs(theta))

def gradient(theta, X, y, lam=0.0):
    """Gradient of Ridge, and OLS for lam=0"""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def hessian_eigs(X, lam=0.0):
    """Hessian eigenvalues for the feature matrix"""
    n = len(X)
    return np.linalg.eigvalsh((2.0 / n) * X.T @ X + 2.0 * lam * np.eye(X.shape[1]))

def MSE_score(y, y_pred):
    """MSE_score between true and predictions"""
    if len(y.shape) != len(y_pred.shape):
        return np.mean((y - y_pred)**2, axis=1)

    return np.sum((y - y_pred)**2) / len(y)

def r2_score(y, y_pred):
    """R2 score between true and predictions"""
    upper = np.sum((y - y_pred)**2)
    lower = np.sum((y - y.mean())**2)
    return 1 - upper / lower

def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                   beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t = 1, 2, ...; state carries the running quantities."""
    if method == "plain":                                    # Eq. (4.10)
        return theta - gamma * g, state
    if method == "momentum":                                 # Eq. (4.28)
        v = beta * state.get('v', 0) + gamma * g
        state["v"] = v
        return theta - v, state
    if method == "adagrad":                                  # Eqs. (4.42) and (4.45)
        r = state.get('r', 0) + g * g
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state 
    if method == "rmsprop":                                  # Eqs. (4.47) and (4.48)
        r = rho * state.get('r', 0) + (1.0 - rho) * g * g
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state
    if method == "adam":                                     # Eqs. (4.51), (4.52), (4.54), (4.55)
        m = beta1 * state.get('m', 0.0) + (1 - beta1) * g
        r = beta2 * state.get('r', 0.0) + (1 - beta2) * g * g
        state["m"] = m
        state["r"] = r
        m_hat, r_hat = m / (1 - beta1**t), r / (1 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters=1000, tol=1e-8, **kw):
    """Full-gradient loop; returns all iterates and the number of steps taken."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        if np.linalg.norm(g) < tol:
            break
    return np.array(history), t

def gradient_descent(X, y, gamma, lam=0.0, num_iters=1000, tol=1e-8):
    """Plain gradient descent and returns theta, iterations and |theta - theta_closed_form|"""
    p = X.shape[1]
    theta = np.zeros(p)
    history = [theta.copy()]
    theta_cls = closed_form(X, y, lam)
    for t in range(num_iters):
        g = gradient(theta, X, y, lam=lam)
        theta = theta - gamma * g
        history.append(theta.copy())
        err = np.linalg.norm(theta - theta_cls)
        if np.linalg.norm(g) < tol:
            break
    return np.array(history), t + 1, err

def k_fold_degrees(x, y, k, model, maxdegree=26):
    cv_mse = np.zeros(maxdegree)
    kfold = KFold(n_splits=k, shuffle=True, random_state=67)
    for deg in range(1, maxdegree + 1):
        pipe = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), StandardScaler(),
                            model)
        scores = -cross_val_score(pipe, x, y, cv=kfold,
                                scoring="neg_mean_squared_error")
        cv_mse[deg - 1] = scores.mean()

    return cv_mse