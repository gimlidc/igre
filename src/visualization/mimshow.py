import matplotlib.pyplot as plt
import numpy as np
import dataset.preprocessing.rescale_range as rr


def draw(intensity_cube, metadata=None, modality_indexes=None):
    """
    Simplified visualization of intensity cube
    :param intensity_cube: numpy ndarray
        pixels
    :param metadata: filenames/strings
        information about intensity cube dimensions
    :param modality_indexes: list of ints
        subset of dimensions which will be shown
    :return: None just print images
    """
    size = (rr.rescale_range(np.asarray(intensity_cube[0].shape)) * 30).astype(np.int8)
    figure = plt.figure(figsize=(size[0], size[1]))
    if modality_indexes is None:
        modality_indexes = np.arange(intensity_cube.shape[3])
    for index, i in enumerate(modality_indexes):
        figure.add_subplot(5, 2, index+1)
        plt.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)
        if metadata is None:
            plt.title(str("(" + i + ")"))
        else:
            plt.title(str("(" + str(i) + ")" + metadata["filenames"][0][i]))
        plt.imshow(intensity_cube[0][:, :, i], cmap='gray')
    figure.show()


def compare(imgA, imgB, grid=6):
    """
    Draw image composed from two inputs.
    :param imgA: first image
    :param imgB: second image
    :param grid: number of rectangles in a row/column
    :return: None just print images
    """
    gridx = np.linspace(0, imgA.shape[0], grid*2, dtype=np.int32)
    gridy = np.linspace(0, imgA.shape[1], grid*2, dtype=np.int32)
    out = imgA.copy()
    for i in range(gridx[0::2].shape[0]):
        for j in range(gridy[0::2].shape[0]):
            out[gridx[0::2][i]: gridx[1::2][i], gridy[0::2][j]: gridy[1::2][j], :] = \
                imgB[gridx[0::2][i]: gridx[1::2][i], gridy[0::2][j]: gridy[1::2][j], :]
    for i in range(gridx[0::2].shape[0] - 1):
        for j in range(gridy[0::2].shape[0] - 1):
            out[gridx[1::2][i]:gridx[0::2][i + 1], gridy[1::2][j]:gridy[0::2][j + 1], :] = \
                imgB[gridx[1::2][i]:gridx[0::2][i + 1], gridy[1::2][j]:gridy[0::2][j + 1], :]
    plt.imshow(out)
    plt.show()
