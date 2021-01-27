from psd_tools import PSDImage
import numpy as np


def parse_psd(psd_image_file):
    """
    Parse PSD file and returns list of layer names and list of layer pixel intensities
    :param psd_image_file: path to PSD file
    :return: list(str), list(ndarray)
        list of layer names and list of images (layers)
    """
    psd = PSDImage.open(psd_image_file)
    layer_names = []
    arrays = []
    for layer in psd:
        layer_names.append(layer.name)
        arrays.append(np.array(layer.topil()))
    return layer_names, arrays
