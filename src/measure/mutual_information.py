import numpy as np
from sklearn.metrics import mutual_info_score as mis


def mi(imgA, imgB, bins=0):
    """
    Method computes mutual information between two images with the same resolution.
    Number of bins determine density of grayscale sampling.
    TODO: If images are not grayscale, they will be converted to grayscale.
    Expected dimensions are (height, width, 1)
    :param imgA: first input image
    :param imgB: second compared image
    :return: float
    """
    imgA = np.array(imgA)
    imgB = np.array(imgB)
    if imgA.shape != imgB.shape:
        raise IOError("Expected same shape of input images")

    a = imgA.reshape(imgA.shape[0] * imgA.shape[1])
    b = imgB.reshape(imgB.shape[0] * imgB.shape[1])
    if bins == 0:
        # If number of bins in undetermined use sqrt from number of pixels
        bins = np.sqrt(imgA.shape[0] * imgA.shape[1]).astype(int)

    a = ((a - np.mean(a)) / np.std(a) * bins).astype(int)
    b = ((b - np.mean(b)) / np.std(b) * bins).astype(int)

    return mis(a, b)


if __name__ == "__main__":
    inA = np.random.random_sample((100, 100))
    inB = inA * 100
    print(mi(inA, inB, 100))
