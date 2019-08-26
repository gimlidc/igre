import utils
import cv2
import numpy as np


def blur_preprocessing(image, img_size):
    outputs = np.ones(image.shape)
    img = image.reshape(img_size[0], img_size[1], -1)

    blurs = utils.config["blur_sizes"]

    for b_size in blurs:
        blurred = cv2.GaussianBlur(img[:, :, 0], (b_size, b_size), 0)
        blurred = blurred.reshape(blurred.shape[0], blurred.shape[1], 1)
        blurred = blurred.reshape(blurred.shape[0] * blurred.shape[1], -1)
        outputs = np.append(outputs, blurred, axis=1)

    outputs = np.append(outputs, image, axis=1)
    return outputs[:, 1:]


def training_batch_selection(train_set_size, input_dims):
    """
    Selects data with respect to the maximal expected distortion so that we have coords
        effectively in range [0, image_size-max_distortion] (after transform is applied)

    :param train_set_size: batch size
    :param input_dims: image dimensions
    :return: indices to randomly selected pixels within "safety zone"
    """
    tmp = np.arange(input_dims[0]*input_dims[1])
    tmp = tmp.reshape(input_dims[:-1])
    expected_max_distortion = int(np.floor(utils.config["expected_max_shift_px"] *
                                           utils.config["expected_max_scale"]))
    tmp = tmp[expected_max_distortion:-expected_max_distortion,
              expected_max_distortion:-expected_max_distortion]
    tmp = tmp.reshape(-1)
    selection = np.random.permutation(tmp)
    selection = selection[:train_set_size]

    return selection
