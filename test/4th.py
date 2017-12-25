import numpy as np

if __name__ == '__main__':
    A = np.reshape(np.asarray([i for i in range(48)], dtype=np.int32), (1, 4, 4, 3))

    print(A[:, :, 1:, :])
    print(A[:][:][1:][:])
