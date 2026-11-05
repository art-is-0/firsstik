import numpy as np

def optimizer_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
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


def make_batches(n, batch_size, rng):
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def init_state(layers):
    return [({}, {}) for _ in layers]   # (state for W, state for b) per layer


def update_layers(layers, layer_grads, method, gamma, states, t, **kw):
    new_layers = []
    for (W, b), (dW, db), (sW, sb) in zip(layers, layer_grads, states):
        W, _ = optimizer_step(method, W, dW, sW, t, gamma, **kw)
        b, _ = optimizer_step(method, b, db, sb, t, gamma, **kw)
        new_layers.append((W, b))
    return new_layers, states


def train(net, X, Y, method="adam", gamma=1e-3, epochs=100, 
          batch_size=None, X_val=None, Y_val=None, metric=None, seed=67, 
          patience=None, verbose=False, **kw):
    '''
    Returns the history with the training cost and metric (i.e., accuracy) after 
    every epoch and, when a validation set is given, the validation cost and 
    accuracy.
    
    LLM-assisted
    ------------
    Tool: Claude Sonnet (November 2026)
    Role: Suggested adding an optional "patience" so the train function can
          stop early if the validation cost hasn't improved for this many epochs.
    '''
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    M = n if batch_size is None else batch_size
    states = init_state(net.layers)
    t = 0
    history = {"train_cost": [], "val_cost": [], "train_metric": [], "val_metric": []}
    best_val, best_layers, since_best = np.inf, None, 0     # LLM-assisted: set up for "patience"
    for epoch in range(epochs):
        for batch in make_batches(n, M, rng):
            t += 1
            grads = net.compute_gradient(X[batch], Y[batch])
            new_layers, states = update_layers(net.layers, grads, method, gamma, states, t, **kw)
            net.update_weights(new_layers)
        # Save costs and metrics
        history["train_cost"].append(net.cost(X, Y))
        if not np.isfinite(history["train_cost"][-1]):  # Diverged: stop instead of burning epochs
            history["diverged"] = True
            break
        if metric is not None:
            history["train_metric"].append(metric(net, X, Y))
        if X_val is not None:
            history["val_cost"].append(net.cost(X_val, Y_val))
            if metric is not None:
                history["val_metric"].append(metric(net, X_val, Y_val))
            if patience is not None:        # Early stopping, keeps the best weights
                if history["val_cost"][-1] < best_val:
                    best_val, since_best = history["val_cost"][-1], 0
                    best_layers = [(W.copy(), b.copy()) for W, b in net.layers]
                else:
                    since_best += 1
                    if since_best >= patience:
                        break
        if verbose:
            msg = f"epoch {epoch + 1:.3d}: cost {history['train_cost'][-1]:.4f}"
            if X_val is not None:
                msg += f"  validation accuracy {history['val_metric'][-1]:.4f}"
            print(msg)
    if best_layers is not None:
        net.update_weights(best_layers)
    history["epochs_run"] = len(history["train_cost"])
    return history