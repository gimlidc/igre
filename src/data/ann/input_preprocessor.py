import cv2
import numpy as np
from src.config.tools import get_config


# TODO: wavelet decomposition
# TODO: maybe we want to blur even the "visible" image
def blur_preprocessing(image, img_size, blur_size):

    img = image.reshape(img_size[0], img_size[1], -1)

    blurred = cv2.GaussianBlur(img[:, :, 0], (blur_size, blur_size), 0)
    blurred = blurred.reshape(blurred.shape[0], blurred.shape[1], 1)
    blurred = blurred.reshape(blurred.shape[0] * blurred.shape[1], -1)

    return blurred


def training_batch_selection(train_set_size, input_dims):
    """
    Selects data with respect to the maximal expected distortion so that we have coords
        effectively in range [0, image_size-max_distortion] (after transform is applied)

    :param train_set_size: batch size
    :param input_dims: image dimensions
    :return: indices to randomly selected pixels within "safety zone"
    """
    all_data_indices = np.arange(input_dims[0]*input_dims[1])
    all_data_indices = all_data_indices.reshape(input_dims[:-1])
    max_misplacement = int(np.floor(get_config()["expected_max_px_misplacement"]))
    selection = all_data_indices[max_misplacement:-max_misplacement,
                                 max_misplacement:-max_misplacement]
    selection = selection.reshape(-1)
    selection = np.random.permutation(selection)
    selection = selection[:train_set_size]

    return selection
