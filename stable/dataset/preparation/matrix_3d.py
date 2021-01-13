import scipy.io
import numpy as np
import os
from termcolor import colored


def crop(data, left_top_x=None, left_top_y=None, width=None, height=None, rectangle=None, data_key=None, log=None):
    """
    Load 3D matrix from the file, crop it according to coordinates and save it into a new file with suffix -cropped.
    Preserves path and suffix.
    :param data: string
        path to file
    :param left_top_x: int
    :param left_top_y: int
    :param width: int
    :param height: int
    :param rectangle: tuple(int, int, int, int)
        Rectangle has higher priority than other coordinates
    :param data_key: string
        data_key must be used when datafile structure is not flat (multiple variables)
    :param log: function
        if set, log function print out some user relevant info
    :return: None
    """
    if log is None:
        log = print
    suffix = os.path.splitext(data)[1][1:]
    matrix = None
    if suffix == "npy":  # numpy array
        matrix = np.load(data)
    elif suffix == "mat":
        matfile = scipy.io.loadmat(data)
        datakeys = [key for key in matfile.keys() if not key.startswith("__")]
        if len(datakeys) == 1:
            data_key = datakeys[0]
        if data_key in datakeys:
            matrix = matfile[data_key]
        else:
            log(colored(f"Multiple variables stored in specified matfile: {datakeys}. "
                               f"Please use --data-key option to choose one.", "red"))
    if matrix is None:
        log(colored(f"Unsupported file format: {suffix}.", "red"))
        log("Supported formats - *.mat, *.npy")
        return 1

    if rectangle is None:
        x0 = left_top_x
        x1 = left_top_x + width
        y0 = left_top_y
        y1 = left_top_y + height
    else:
        x0 = rectangle[0]
        y0 = rectangle[1]
        x1 = rectangle[2]
        y1 = rectangle[3]
    log(f"Rectangle coordinates [{x0},{y0},{x1},{y1}]")

    base = os.path.basename(data)
    output_file = f"{data[:-len(base)]}{base[:-len(suffix)-1]}-crop.{suffix}"
    if suffix == "mat":
        out = {}
        log(matrix[x0:x1, y0:y1, :].shape)
        out[data_key] = matrix[x0:x1, y0:y1, :]
        scipy.io.savemat(output_file, out)
    elif suffix == "npy":
        np.save(output_file, matrix[x0:x1, y0:y1, :])
    log(colored(f"Output file: {output_file} successfully written.", "green"))
