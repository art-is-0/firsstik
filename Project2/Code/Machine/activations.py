import numpy as np

def sigmoid(z):
    return ...


def sigmoid_der(z):
    return ...


def relu(z):
    return ...


def relu_der(z):
    return ...


def leaky_relu(z, alpha=0.01):
    return ...


def leaky_relu_der(z, alpha=0.01):
    return ...


def identity(z):
    return ...


def identity_der(z):
    return ...


def softmax(z):
    return ...


# name -> (f, f'); softmax has no elementwise derivative, so None
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_der),
    "relu": (relu, relu_der),
    "leaky_relu": (leaky_relu, leaky_relu_der),
    "identity": (identity, identity_der),
    "softmax": (softmax, None)
}