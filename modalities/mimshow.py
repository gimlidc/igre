import matplotlib.pyplot as plt
import numpy as np
import rescale_range as rr


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
