import numpy as np

def mse_score(y, y_pred):
    return np.mean((np.ravel(y) - np.ravel(y_pred)) ** 2)


def accuracy(y_true, y_pred):
    '''Integer labels. Get them with np.argmax(net.predict(X), axis=1)'''
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=10):
    C = np.zeros((n_classes, n_classes), dtype=int) # rows = true, columns = pred
    np.add.at(C, (y_true, y_pred), 1)
    return C


def precision_recall_f1(C):
    tp = np.diag(C).astype(float)
    precision = tp / np.maximum(C.sum(axis=0), 1)
    recall = tp / np.maximum(C.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1, f1.mean()     # macro F1


def se(scores):
    '''Standard error of the mean of the fold scores, s / sqrt(k) with the sample standard deviation s'''
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    return np.std(scores, ddof=1) / np.sqrt(len(scores))


def binomial_se(p, n):
    '''Standard error of an accuracy p measured on n independent samples'''
    return np.sqrt(p * (1 - p) / n)