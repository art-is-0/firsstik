import numpy as np
from sklearn.model_selection import train_test_split
from utils import create_feature_matrix

def load_mnist(path):
    with np.load(path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0
    return x_train, y_train, x_test, y_test


def train_val_split(X, y, val_size=0.1, seed=67, stratify=True):
    '''Returns X_train, X_val, y_train, y_val (sklearn's order)'''
    return train_test_split(X, y, test_size=val_size, random_state=seed, 
                            stratify=y if stratify else None)


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def to_column(x):
    return np.asarray(x).reshape(-1, 1)