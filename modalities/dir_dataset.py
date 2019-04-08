import os
import cv2
import numpy as np


def load_all_images(path, convert_to_grayscale=True):
    """
    Method reads all images in specified directory. Those with the same resolution put into one ndarray.
    Directory should contain various modalities of the same artwork
    :param path: string
        Full or relative path to directory
    :return: list(ndarray), list(list(string)
        list of ndarrays with loaded modalities. According to resolution data are grouped together.
        Second output contains filenames of directories
    """
    resolutions = []
    out = []
    filenames = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            print("Directory: ", file, " not loaded.")
            continue

        img = cv2.imread(file_path)
        if convert_to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))

        if img.shape[:2] in resolutions:
            id = resolutions.index(img.shape[:2])
            out[id] = np.concatenate((out[id], img), axis=2)
            filenames[id].append(file)
        else:
            resolutions.append(img.shape[:2])
            out.append(img)
            filenames.append([file])
    metadata = {
        "filenames": filenames,
        "resolutions": resolutions
    }
    return out, metadata
