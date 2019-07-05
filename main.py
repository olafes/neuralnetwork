import numpy as np
from network import Network

def check(network, examples):
    for example in examples:
        print('x: ', example['x'])
        print('expected :', example['y'])
        print('predicted', network.compute(example['x']))
def main():
    examples = [
        {
            'x': np.array([[0], [0]]),
            'y': np.array([[0]])
        }, {
            'x': np.array([[0], [1]]),
            'y': np.array([[1]])
        }, {
            'x': np.array([[1], [0]]),
            'y': np.array([[1]])
        }, {
            'x': np.array([[1], [1]]),
            'y': np.array([[0]])
        }
    ]
    N = 1000000
    rate = 0.01
    network = Network(2, [
        [2, 'sigmoid'],
        [1, 'sigmoid']
    ])
    print('\n\n\n\n')
    print('cost', network.cost(examples))
    check(network, examples)
    for i in range(N):
        network.learn(examples, rate)
    print('\n\n\n\n')
    print('cost', network.cost(examples))
    check(network, examples)
if __name__ == '__main__':
    main()
