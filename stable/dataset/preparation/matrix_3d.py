import scipy.io
import numpy as np
import os
from termcolor import colored
import imageio
from stable.filepath import parse


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
    folders, filename, suffix = parse(data)
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

    output_file = f"{folders}{filename}-crop.{suffix}"
    if suffix == "mat":
        out = {}
        log(matrix[x0:x1, y0:y1, :].shape)
        out[data_key] = matrix[x0:x1, y0:y1, :]
        scipy.io.savemat(output_file, out)
    elif suffix == "npy":
        np.save(output_file, matrix[x0:x1, y0:y1, :])
    log(colored(f"Output file: {output_file} successfully written.", "green"))


def stack_badly_sized_arrays(image_names, arrays, crop, log=print):
    shapes = np.array([np.array(img.shape) for img in arrays])
    size_matches = True
    for dim_name, dim in zip(["width", "height"], [0, 1]):
        if np.min(shapes[:, dim]) != np.max(shapes[:, dim]):
            log(colored(
                f"Warning: Image {dim_name}s mismatch: {np.min(shapes[:, dim])} "
                f"in {image_names[np.argmin(shapes[:, dim])]} "
                f"vs. {np.max(shapes[:, dim])} in {image_names[np.argmax(shapes[:, dim])]}",
                color="yellow"))
            if crop:
                log(colored(f"Data will be cropped to min {dim_name} size."), color="green")
            else:
                log(colored(f"Data will be padded with zeros to match max size.", color="green"))
            size_matches = False

    if crop or size_matches:
        width = np.min(shapes[:, 0])
        height = np.min(shapes[:, 1])
        out = np.zeros((width, height, np.sum(shapes[:, 2])))
        start_dim = 0
        for img in arrays:
            out[:, :, start_dim:start_dim + img.shape[2]] = img[:width, :height, :]
            start_dim += img.shape[2]
    else:  # pad option
        out = np.zeros((np.max(shapes[:, 0]), np.max(shapes[:, 1]), np.sum(shapes[:, 2])))
        start_dim = 0
        for img in arrays:
            out[:img.shape[0], :img.shape[1], start_dim:start_dim + img.shape[2]] = img
            start_dim += img.shape[2]
    return out


def merge_image_files(dir_name, suffix, output, crop2fit, log=print):
    if not os.path.isdir(dir_name):
        log("ERROR: directory not found")
    imgs = []
    data_files = sorted(os.listdir(dir_name))
    log("Loading files ...")
    for file in data_files:
        if file[-len(suffix):] == suffix:
            img = imageio.imread(os.path.join(dir_name, file))
            if len(img.shape) == 2:
                img = img.reshape(img.shape + (1,))
            imgs.append(img)
            log(f"\t{file} OK, shape: {imgs[-1].shape}")

    out = stack_badly_sized_arrays(data_files, imgs, crop2fit, log)
    np.save(output, out)
    log(colored(f"Data successfully saved into {output} with shape {out.shape}.", color="green"))
