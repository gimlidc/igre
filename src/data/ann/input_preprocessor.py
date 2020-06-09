import cv2
import numpy as np
from src.config.tools import get_config
import src.config.image_info as ii
from src.logging.verbose import Verbose


# TODO: wavelet decomposition
# TODO: maybe we want to blur even the "visible" image
def blur_preprocessing(image, img_size, blur_size):

    img = image.reshape(img_size[0], img_size[1], -1)

    blurred = cv2.GaussianBlur(img[:, :, 0], (blur_size, blur_size), 0)
    blurred = blurred.reshape(blurred.shape[0], blurred.shape[1], 1)
    blurred = blurred.reshape(blurred.shape[0] * blurred.shape[1], -1)

    return blurred


def training_batch_selection(train_set_size, input_img):
    """
    Selects data with respect to the maximal expected distortion so that we have coords
        effectively in range [0, image_size-max_distortion] (after transform is applied)

    :param train_set_size: batch size
    :param input_img: image
    :return: indices to randomly selected pixels within "safety zone"
    """

    input_dims = input_img.shape
    all_data_indices = np.arange(input_dims[0]*input_dims[1])
    all_data_indices = all_data_indices.reshape(input_dims[:-1])

    conf = get_config()
    inside = int(np.floor(conf["inside_part"]))
    outside = int(np.floor(conf["outside_part"]))
    cx = ii.image.c_x
    cy = ii.image.c_y

    # Find the position of the crop in the image and determine part least affected by radial distortion
    center_x = (conf["crop"]["left_top"]['x'] + conf["crop"]["size"]["height"]/2)*2/ii.image.height-1.
    center_y = (conf["crop"]["left_top"]['y'] + conf["crop"]["size"]["width"]/2)*2/ii.image.width-1.

    left_right_center = max(min((center_y-cy)*2, 1), -1)
    top_bot_center = max(min((center_x - cx) * 2, 1), -1)

    # DEBUG: set your own center
    # left_right_center = 0
    # top_bot_center = 0

    from_x = int(round(inside*(1.-top_bot_center)))
    to_x = min(-int(round(inside*(1.+top_bot_center))), -1)
    from_y = int(round(inside*(1.-left_right_center)))
    to_y = min(-int(round(inside*(1.+left_right_center))), -1)

    # Exclude part that is minimally affected by radial distortion
    selection_exclude = all_data_indices[from_x:to_x, from_y:to_y]
    selection_exclude = selection_exclude.reshape(-1)

    # Exclude outer border in order to avoid index out of bounds
    selection_include = all_data_indices[outside:-outside,
                                         outside:-outside]
    selection_include = selection_include.reshape(-1)

    selection = [x for x in selection_include if x not in selection_exclude]
    np.random.seed()
    selection = np.random.permutation(selection)

    # DEBUG: forcing larger training set
    train_set_size = int(train_set_size/2)

    selection = selection[:train_set_size]

    # DEBUG: display image region for selection
    image = input_img.reshape(-1)
    image[:] = 0
    image[selection_include] = input_img.reshape(-1)[selection_include]
    image[selection_exclude] = 0
    image = image.reshape(input_dims[0], input_dims[1])
    Verbose.imshow(image, Verbose.debug)

    return selection
