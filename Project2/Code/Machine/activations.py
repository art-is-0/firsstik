import numpy as np

def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_der(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_der(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def identity(z):
    return z


def identity_der(z):
    return np.ones_like(z)


def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# name -> (f, f'); softmax has no elementwise derivative, so None
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_der),
    "relu": (relu, relu_der),
    "leaky_relu": (leaky_relu, leaky_relu_der),
    "identity": (identity, identity_der),
    "softmax": (softmax, None)
}