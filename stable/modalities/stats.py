import numpy as np
from numpy import corrcoef


def joint_entropy(hyperspectral_cube, values=256):
    """
    Computes cross-entropy for
    :param hyperspectral_cube: ndarray
        first two dimensions are width and height, in third dimension are modalities
    :return: 2d array
        pairwise computed cross entropy
    """
    # linearize
    bins = [np.arange(values), np.arange(values)]
    data = np.transpose(np.reshape(hyperspectral_cube, (hyperspectral_cube.shape[0] * hyperspectral_cube.shape[1],
                                                        hyperspectral_cube.shape[2])))

    out = np.zeros((data.shape[0], data.shape[0]))

    for x in range(data.shape[0]):
        for y in range(x, data.shape[0]):
            pxy = np.histogram2d(data[x], data[y], bins)[0] / data.shape[1]
            # values with zero will be ignored
            pplus = pxy[pxy != 0]
            # compute joint entropy
            out[x, y] = - np.sum(pplus * np.log(pplus))
            out[y, x] = out[x, y]
    return out


def correlation(data):
    """
    Method computes correlation of each band in hyperspectral cube
    :param data: ndarray
        3 dims hyperspectral cube, where last dimension is the one where corelation is computed
    :return: correlation matrix
    """
    return corrcoef(np.transpose(np.reshape(data, ((data.shape[0] * data.shape[1]), data.shape[2]))))
