import numpy as np

if __name__ == '__main__':
    labels = np.empty((4,), dtype=np.float32)
    labels.fill(-1)

    matrix = np.reshape(np.asarray([0, 1, 2, 3, 4, 5, 6, 7]), (4, 2))
    argmax_x = matrix.argmax(axis=1)

    print(argmax_x)

    max = matrix[np.arange(4), argmax_x]
    print(matrix)
    print(max)

    labels[max > 3] = 1

    print(type(max))
    print(labels)
