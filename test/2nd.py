import numpy as np

if __name__ == '__main__':
    a = np.asarray([['a', 'b'], ['c', 'd']])
    a.reshape(((0, 0), (1, 1)))
