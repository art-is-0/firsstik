import numpy as np
from activations import ACTIVATIONS
from costs import COSTS, l1_penalty, l2_penalty


def create_layers(sizes, activations, seed):
    return ...


class NeuralNetwork:
    def __init__(self, sizes, activations, cost, lmbd1=0.0, lmbd2=0.0, seed=67):
        ...
        
    def predict(self, X):
        return ...
    
    def cost(self, X, Y):
        return ...
    
    def _feed_forward_saver(self, X):
        return ...
    
    def compute_gradient(self, X, Y):
        return ...
    
    def update_weights(self, new_layers):
        ...