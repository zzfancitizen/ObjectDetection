import numpy as np

if __name__ == '__main__':
    # labels = np.empty((4,), dtype=np.float32)
    # labels.fill(-1)
    #
    # matrix = np.reshape(np.asarray([0, 1, 2, 3, 4, 5, 6, 7]), (4, 2))
    # argmax_x = matrix.argmax(axis=1)
    #
    # print(argmax_x)
    #
    # max = matrix[np.arange(4), argmax_x]
    # print(matrix)
    # print(max)
    #
    # labels[max > 3] = 1
    #
    # print(type(max))
    # print(labels)

    boxes = np.asarray([[0, 0, 1, 1], [0, 0, 2, 2]]).reshape((2, 4))
    proposal = np.asarray([[0, 0, 1, 1], [1, 1, 4, 4]]).reshape((2, 4))
    scores_temp = np.asarray([0.3, 0.7]).reshape((2, 1))

    dets = np.hstack((proposal, scores_temp))

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    suppressed = np.zeros((dets.shape[0]), dtype=np.int)
    # print(suppressed)
    print(boxes)
    print(proposal)
    print(scores)
    orders = scores.argsort()[::-1]
    print(orders)
