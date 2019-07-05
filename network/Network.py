import numpy as np
from .Layer import Layer

class Network():
    def __init__(self, input, layers):
        self.n = input
        self.layers = []
        for layer in layers:
            self.layers.append(Layer(layer[0], input, layer[1]))
            input = layer[0]
        self.L = len(self.layers)-1
    def get(self):
        return [layer.weights for layer in self.layers]
    def set(self, weights):
        for i, w in enumerate(weights):
            self.layers[i].weights = w
    def compute(self, x, cache=False):
        if cache:
            for layer in self.layers:
                x = layer.learn(x)
        else:
            for layer in self.layers:
                x = layer.compute(x)
        return x
    def cost(self, examples):
        y = None
        error = 0
        for example in examples:
            y = self.compute(example['x'])
            error += np.sum(np.square(example['y'] - y))
        return error/(2*len(examples))
    def learn(self, examples, rate):
        for layer in self.layers:
            layer.cache.clear()
        m = len(examples)
        y = None
        for example in examples:
            y = self.compute(example['x'], True)
            self.layers[self.L].cache.delta = self.layers[self.L].d_fn(self.layers[self.L].cache.z)*(y - example['y'])
            self.layers[self.L].cache.d_w += np.tile(self.layers[self.L].cache.x.T, [self.layers[self.L].n, 1])*self.layers[self.L].cache.delta
            self.layers[self.L].cache.d_bias += self.layers[self.L].cache.delta
            for i in range(self.L-1, -1, -1):
                self.layers[i].cache.delta = (self.layers[i+1].weights.T @ self.layers[i+1].cache.delta)*self.layers[i].d_fn(self.layers[i].cache.z)
                self.layers[i].cache.d_w += np.tile(self.layers[i].cache.x.T, [self.layers[i].n, 1])*self.layers[i].cache.delta
                self.layers[i].cache.d_bias += self.layers[i].cache.delta
        for layer in self.layers:
            layer.bias -= (rate/(2*m))*layer.cache.d_bias
            layer.weights -= (rate/(2*m))*layer.cache.d_w
