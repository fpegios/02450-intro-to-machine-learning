import numpy as np
from matplotlib.pyplot import (plot, show, title)

# x = np.linspace(0,2,100)
# noise = np.random.normal(0, 0.2, 100)
# y = np.sin(x) + noise
# plot(x, y, '.-r')
# title('Sine with gaussian noise')
# show()
# plot

def hello(name, n):
    M = np.random.rand(n, n)
    print('\nHello {0}! This is your matrix:\n{1}'.format(name, M))