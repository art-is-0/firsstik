import numpy as np
from .activations import ACTIVATIONS
from .costs import (COSTS, MATCHED_OUTPUT_ERROR,
                    l1_penalty, l1_penalty_der,
                    l2_penalty, l2_penalty_der)
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .optimizers import train

# ==============================================================================

def create_layers(sizes, activations, seed):
    '''sizes = [n_in, h1, ..., n_out]. W has shape (n_in, n_out)'''
    rng = np.random.default_rng(seed)
    layers = []
    for n_in, n_out, name in zip(sizes[:-1], sizes[1:], activations):
        scale = np.sqrt(2.0 / n_in) if name in ("relu", "leaky_relu") else np.sqrt(1.0 / n_in)
        W = rng.normal(0.0, scale, (n_in, n_out))
        b = np.zeros(n_out) + 0.01
        layers.append((W, b))
    return layers

class NeuralNetwork:
    def __init__(self, sizes, activations, cost, lmbd1=0.0, lmbd2=0.0, seed=67, penalize_bias=False):
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
        self.penalize_bias = penalize_bias
        
    def predict(self, inputs):
        a = inputs
        for (W, b), f in zip(self.layers, self.activation_funcs):
            a = f(a @ W + b)
        return a

    
    def cost(self, inputs, targets):
        p = self.predict(inputs)
        return self.cost_fun(p, targets)
    
    @property
    def n_params(self):
        return sum(W.size + b.size for W, b in self.layers)
    
    
    def penalty(self):
        l1 = l1_penalty(self.layers, self.lmbd1, self.penalize_bias)
        l2 = l2_penalty(self.layers, self.lmbd2, self.penalize_bias)
        return l1 + l2
    
    
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
                
            W, b = self.layers[i]
            dW = (layer_inputs[i].T @ dC_dz
                  + l2_penalty_der(W, self.lmbd2) + l1_penalty_der(W, self.lmbd1))
            db = np.sum(dC_dz, axis=0)
            if self.penalize_bias:
                db = db + l2_penalty_der(b, self.lmbd2) + l1_penalty_der(b, self.lmbd1)
            layer_grads[i] = (dW, db)
        return layer_grads
    
    
    def update_weights(self, new_layers):
        self.layers = new_layers
        
        
# ==============================================================================

class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    '''
    Wraps our NeuralNetwork so that sklearn (cross_validate, GridSearchCV, Pipeline)
    can use it. __init__ only stores the arguments, and everything learned in fit
    gets a training underscore.
    '''
    def __init__(self, hidden=(50,), activation="sigmoid", method="rmsprop", gamma=0.03, 
                 lmbd1=0.0, lmbd2=0.0, epochs=500, batch_size=10, seed=67, penalize_bias=False):
        self.hidden = hidden
        self.activation = activation
        self.method = method
        self.gamma = gamma
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.penalize_bias = penalize_bias

    def fit(self, X, y):
        sizes = [X.shape[1], *self.hidden, 1]
        acts = [self.activation] * len(self.hidden) + ["identity"]
        self.net_ = NeuralNetwork(sizes, acts, "mse", lmbd1=self.lmbd1, lmbd2=self.lmbd2,
                                  seed=self.seed, penalize_bias=self.penalize_bias)
        with np.errstate(all="ignore"):
            self.history_ = train(self.net_, X, np.asarray(y).reshape(-1, 1), method=self.method,
                                  gamma=self.gamma, epochs=self.epochs,
                                  batch_size=self.batch_size, seed=self.seed)
        return self

    def predict(self, X):
        return self.net_.predict(X)[:, 0]
    
    
# ==============================================================================
    
class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    '''Softmax + cross entropy network for integer labels 0...K-1'''
    def __init__(self, hidden=(64,), activation="relu", method="adam", gamma=1e-3, 
                 lmbd1=0.0, lmbd2=0.0, epochs=10, batch_size=100, seed=67, penalize_bias=False):
        self.hidden = hidden
        self.activation = activation
        self.method = method
        self.gamma = gamma
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.penalize_bias = penalize_bias

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        Y = np.zeros((len(y), K))
        Y[np.arange(len(y)), np.searchsorted(self.classes_, y)] = 1.0
        sizes = [X.shape[1], *self.hidden, K]
        acts = [self.activation] * len(self.hidden) + ["softmax"]
        self.net_ = NeuralNetwork(sizes, acts, "cross_entropy", lmbd1=self.lmbd1,
                                  lmbd2=self.lmbd2, penalize_bias=self.penalize_bias)
        with np.errstate(all="ignore"):
            self.history_ = train(self.net_, X, Y, method=self.method, gamma=self.gamma,
                                  epochs=self.epochs, batch_size=self.batch_size,
                                  seed=self.seed)
        return self

    def predict_proba(self, X):
        return self.net_.predict(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
    
    
def make_nn(**kw):
    '''Scaler is fitted inside each fold. The target is centered inside each fold too.'''
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), NeuralNetRegressor(**kw)),
        transformer=StandardScaler(with_std=False))