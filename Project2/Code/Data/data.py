import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist(path):
    return ...


def train_val_split(X, y, val_size=0.1, seed=67):
    return ...


def one_hot(y, n_classes=10):
    return ...


def to_column(x):
    return ...