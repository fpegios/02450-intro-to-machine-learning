import numpy as np

def hello(name, n):
    M = np.random.rand(n, n)
    print('\nHello {0}! This is your matrix:\n{1}'.format(name, M))

def test(param):
    print('{0} \n {1}'.format(param, "P"))