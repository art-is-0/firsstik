import numpy as np
EPS = 1e-12     # Keeps log() away from 0

def mse(predict, target):
    return np.mean((predict - target) ** 2)


def mse_der(predict, target):
    return 2.0 * (predict - target) / predict.size


def binary_cross_entropy(predict, target):
    p = np.clip(predict, EPS, 1 - EPS)
    return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))


def binary_cross_entropy_der(predict, target):
    p = np.clip(predict, EPS, 1 - EPS)
    return (p - target) / (p * (1 - p)) / predict.size


def cross_entropy(predict, target):
    '''Softmax cross entropy, target one-hot (n, K)'''
    p = np.clip(predict, EPS, 1.0)
    return np.mean(np.sum(-target * np.log(p), axis=1))


def cross_entropy_der(predict, target):
    return -target / np.clip(predict, EPS, 1.0) / predict.shape[0]


COSTS = {
    "mse": (mse, mse_der),
    "binary_cross_entropy": (binary_cross_entropy, binary_cross_entropy_der),
    "cross_entropy": (cross_entropy, cross_entropy_der)
}


# ---- OUTPUT ERROR dC/dz for matched (output activation, cost) pairs ----------


def softmax_ce_output_error(predict, target):
    return (predict - target) / predict.shape[0]


def sigmoid_bce_output_error(predict, target):
    return (predict - target) / predict.size


MATCHED_OUTPUT_ERROR = {
    ("softmax", "cross_entropy"): softmax_ce_output_error,
    ("sigmoid", "binary_cross_entropy"): sigmoid_bce_output_error
}


# ---- L1 / L2 ON WEIGHTS ONLY (not the biases) --------------------------------


def l2_penalty(layers, lmbd2):
    return lmbd2 * sum(np.sum(W ** 2) for W, _ in layers)


def l2_penalty_der(W, lmbd2):
    return 2.0 * lmbd2 * W


def l1_penalty(layers, lmbd1):
    return lmbd1 * sum(np.sum(np.abs(W)) for W, _ in layers)


def l1_penalty_der(W, lmbd1):
    return lmbd1 * np.sign(W)