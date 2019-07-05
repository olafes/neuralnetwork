import numpy as np

class Cache():
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.clear()
    def clear(self):
        self.x = np.zeros([self.m, 1])
        self.z = np.zeros([self.n, 1])
        self.delta = np.zeros([self.n, 1])
        self.d_w = np.zeros([self.n, self.m])
        self.d_bias = np.zeros([self.n, 1])
