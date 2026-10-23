import numpy as np
from .activations import ACTIVATIONS
from .costs import (COSTS, MATCHED_OUTPUT_ERROR,
                    l1_penalty, l1_penalty_der,
                    l2_penalty, l2_penalty_der)


def create_layers(sizes, activations, seed):
    '''sizes = [n_in, h1, ..., n_out]. W has shape (n_in, n_out)'''
    rng = np.random.default_rng(seed)
    layers = []
    for n_in, n_out, name in zip(sizes[:-1], sizes[1:], activations):
        std = np.sqrt(2.0 / n_in) if name in ("relu", "leaky_relu") else np.sqrt(1.0 / n_in)
        layers.append((rng.normal(0.0, std, (n_in, n_out)), np.zeros(n_out)))
    return layers

class NeuralNetwork:
    def __init__(self, sizes, activations, cost, lmbd1=0.0, lmbd2=0.0, seed=67):
        assert len(activations) == len(sizes) - 1, "one activation per layer"
        self.sizes = sizes
        self.activation_names = activations
        self.activation_funcs = [ACTIVATIONS[a][0] for a in activations]
        self.activation_ders = [ACTIVATIONS[a][1] for a in activations]
        self.cost_name = cost
        self.cost_fun, self.cost_der = COSTS[cost]
        self.output_error = MATCHED_OUTPUT_ERROR.get((activations[-1], cost))
        self.lmbd1, self.lmbd2 = lmbd1, lmbd2
        self.layers = create_layers(sizes, activations, seed)
        
    def predict(self, inputs):
        a = inputs
        for (W, b), f in zip(self.layers, self.activation_funcs):
            a = f(a @ W + b)
        return a

    
    def cost(self, inputs, targets):
        '''data term + penalties'''
        l1 = l1_penalty(self.layers, self.lmbd1)
        l2 = l2_penalty(self.layers, self.lmbd2)
        p = self.predict(inputs)
        return self.cost_fun(p, targets) + l2 + l1
    
    
    def _feed_forward_saver(self, inputs):
        layer_inputs, zs = [], []
        a = inputs
        for (W, b), f in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)
            z = a @ W + b
            a = f(z)
            zs.append(z)
        return layer_inputs, zs, a
    
    
    def compute_gradient(self, inputs, targets):
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        L = len(self.layers)
        layer_grads = [() for _ in range(L)]
        for i in reversed(range(L)):
            if i == L - 1:
                if self.output_error is not None:
                    dC_dz = self.output_error(predict, targets)
                else:
                    dC_dz = self.cost_der(predict, targets) * self.activation_ders[i](zs[i])
            else:
                W_next, _ = self.layers[i + 1]
                dC_dz = (dC_dz @ W_next.T) * self.activation_ders[i](zs[i])
                
            W, _ = self.layers[i]
            dW = (layer_inputs[i].T @ dC_dz
                  + l2_penalty_der(W, self.lmbd2) + l1_penalty_der(W, self.lmbd1))
            db = np.sum(dC_dz, axis=0)
            layer_grads[i] = (dW, db)
        return layer_grads
    
    
    def update_weights(self, new_layers):
        self.layers = new_layers