import numpy as np

functions = {
    'relu': {
        'fn': np.vectorize(lambda x: x*(x>0)),
        'd_fn': np.vectorize(lambda x: 1*(x>0)),
        'init': lambda n, m: np.sqrt(2/m)
    },
    'sigmoid': {
        'fn': np.vectorize(lambda x: 1/(1+np.exp(-x))),
        'd_fn': np.vectorize(lambda x: np.exp(x)/((1+np.exp(x))**2)),
        'init': lambda n, m: np.sqrt(6/n+m)
    }
}
