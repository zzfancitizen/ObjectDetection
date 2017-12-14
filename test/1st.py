# import yaml
import numpy as np

if __name__ == '__main__':
    # document = """
    # applications:
    # - name: vcap_xsa-app
    #   command: npm run start
    # """
    #
    # file = yaml.load(document)
    # scales = file.get('scales', (8, 16, 32))
    # allowed_border = file.get('allowed_border', 0)
    #
    # print(scales)

    # a = [1, 2, 3, 4, 5, 6]
    #
    # print(a[-2:])
    _anchors = np.array([[-83., -39., 100., 56.],
                         [-175., -87., 192., 104.],
                         [-359., -183., 376., 200.],
                         [-55., -55., 72., 72.],
                         [-119., -119., 136., 136.],
                         [-247., -247., 264., 264.],
                         [-35., -79., 52., 96.],
                         [-79., -167., 96., 184.],
                         [-167., -343., 184., 360.]])

    A = _anchors.shape[0]

    x = np.arange(0, 14)
    y = np.arange(0, 14)

    nx, ny = np.meshgrid(x, y)

    matrix = np.vstack((nx.ravel(), ny.ravel(),
                        nx.ravel(), ny.ravel())).transpose()

    print(matrix)

    K = matrix.shape[0]

    all_anchors = (
        _anchors.reshape((1, A, 4)) + matrix.reshape((1, K, 4)).transpose((1, 0, 2))
    )

    all_anchors = all_anchors.reshape((K * A, 4))

    print(all_anchors)

    inds_inside = np.where((all_anchors[:, 0] >= 0) &
                           (all_anchors[:, 1] >= 0))[0]

    anchors = all_anchors[inds_inside, :]

    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    print(np.ascontiguousarray(anchors, dtype=np.float))
