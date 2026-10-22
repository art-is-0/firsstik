import numpy as np


def mse(predict, target):
    return ...


def mse_der(predict, target):
    return ...


def binary_cross_entropy(predict, target):
    return ...


def binary_cross_entropy_der(predict, target):
    return ...


def cross_entropy(predict, target):
    return ...


def cross_entropy_der(predict, target):
    return ...


def l2_penalty(layers, lmbd2):
    return ...


def l1_penalty(layers, lmbd1):
    return ...


COSTS = {
    "mse": (mse, mse_der),
    "binary_cross_entropy": (binary_cross_entropy, binary_cross_entropy_der),
    "cross_entropy": (cross_entropy, cross_entropy_der)
}