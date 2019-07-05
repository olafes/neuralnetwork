import numpy as np
from .Cache import Cache
from .activation_functions import functions


class Layer():
    def __init__(self, n, m, function='relu'):
        if not function in functions:
            raise ValueError('Unknown function type')
        self.n = n
        self.m = m
        self.fn = functions[function]['fn']
        self.d_fn = functions[function]['d_fn']
        self.weights = np.random.randn(n, m) * functions[function]['init'](n, m)
        self.bias = np.zeros([n, 1])
        self.cache = Cache(n, m)
    def compute(self, x):
        return self.fn(self.weights @ x + self.bias)
    def learn(self, x):
        self.cache.x = x
        self.cache.z = self.weights @ x + self.bias
        return self.fn(self.cache.z)
