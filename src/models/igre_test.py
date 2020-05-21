import yaml
from src.models import igre
from termcolor import colored
from src.logging.verbose import Verbose
import os
import numpy as np
import scipy.io
from src.config.tools import init_config
from src.registration.transformation import Transformation
import src.config.image_info as ii

ROOT_DIR = os.path.abspath(os.curdir)


def data_crop(config, dataset):
    if "crop" in config:
        print("Data crop ... " + colored("YES", "green") + ":", Verbose.debug)
        print("\t["
              + str(config["crop"]["left_top"]["x"]) + ":"
              + str(config["crop"]["left_top"]["x"] + config["crop"]["size"]["height"]) + ", "
              + str(config["crop"]["left_top"]["y"]) + ":"
              + str(config["crop"]["left_top"]["y"] + config["crop"]["size"]["width"]) + ", :]", Verbose.debug)
        dataset = dataset[
                  config["crop"]["left_top"]["x"]: (config["crop"]["left_top"]["x"] + config["crop"]["size"]["height"]),
                  config["crop"]["left_top"]["y"]: (config["crop"]["left_top"]["y"] + config["crop"]["size"]["width"]),
                  :]
    else:
        print("Data crop: " + colored("NO", "red"), Verbose.debug)

    return dataset


def igre_test(conf, shift, output):
    """
    Information gain and registration test works with registered inputs. For testing registration layer, input pixels
    are shifted by shift[0] in x axis and by shift[1] in y axis.
    :param conf: configuration of IGRE run, accepted is dict of parsed values as well as filename with stored params
    :param shift: tuple containing shift in x and in y axis of input dimensions
    :param output: output file for measured data
    :return: registration layer weights (i.e. computed shift)
    """
    # if os.path.isfile(output):
    #     print(output, "already exist. Skipping.")
    #     return

    # Config load and integrity check
    if type(output) == str:
        with open(conf, "rt", encoding='utf-8') as config_file:
            config = yaml.load(config_file)
    else:
        config = conf.copy()
    init_config(config)

    Verbose.print(
        "\nWelcome to " + colored("IGRE-test", "green") + " run with file: " + colored(config["matfile"], "green") +
        " expected shift: " + colored(shift, "green") + "\n")

    if "matfile" in config:
        dataset = np.float64(scipy.io.loadmat(os.path.join(ROOT_DIR, "data", "raw", config["matfile"]))['data'])
    else:
        with open(config["numpyfile"]["data"]) as infile:
            dataset = np.load(infile)
    Verbose.print("Data stats (before normalization): min = " + str(np.min(dataset)) +
                  " max = " + str(np.max(dataset)), Verbose.debug)
    # data normalization - ranged
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

    ii.init(dataset.shape[0], dataset.shape[1])

    Verbose.print("Dataset shape: " + str(dataset.shape), Verbose.debug)
    dataset = data_crop(config, dataset)

    Verbose.print(colored("Input", "green") + " dimensions: " +
                  colored(("[" +
                           str(config["input_dimensions"]["min"]) + "-" +
                           str(config["input_dimensions"]["max"]) + "]"),
                          "green"), Verbose.debug)
    visible = dataset[:, :, config["input_dimensions"]["min"]: config["input_dimensions"]["max"] + 1]
    Verbose.imshow(visible[:, :, 0])

    x = visible.shape[:-1]
    x_size = x[0]
    y_size = x[1]
    indexes = np.indices(x)
    indexes = indexes.reshape((len(visible.shape[:-1]), -1)).transpose().astype(np.float32)

    Verbose.print("\tInputs shape: " + str(visible.shape), Verbose.debug)
    Verbose.print(colored("Output", "green") + " dimensions: " +
                  colored(("[" +
                           str(config["output_dimensions"][
                                   "min"]) + "-" +
                           str(config["output_dimensions"][
                                   "max"]) + "]"),
                          "green"), Verbose.debug)
    outputs = dataset[:, :, config["output_dimensions"]["min"]: config["output_dimensions"]["max"] + 1]

    Verbose.print("\tOutput shape: " + str(outputs.shape), Verbose.debug)

    Verbose.print("\nCalling " + colored("IGRE\n", "green") + "...")

    # coordinate transform up to perspective transform
    shift = (0., 0.)
    tform = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    #tform.set_rotation(0.)  # 0.05236 rad
    #tform.set_shift(shift)

    k1 = -0.021
    k2 = 0.006
    k3 = 0.001
    exp_k1 = -k1
    exp_k2 = 3 * k1 * k1 - k2
    exp_k3 = -12 * k1 * k1 * k1 + 8 * k1 * k2 - k3
    tform.set_distortion(0., 0., k1, k2, k3)
    tform_test = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    tform_test.set_distortion(0., 0., exp_k1, exp_k2, exp_k3)

    inputs = tform.apply_distortion(indexes)
    sanitycheck = tform_test.apply_distortion(inputs)
    diff = abs(indexes - sanitycheck)
    displacement = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
    displacement = displacement.reshape(x_size, y_size)
    mean = np.mean(displacement)
    Verbose.imshow(displacement)

    bias, bias_history = igre.run(inputs,
                                  outputs,
                                  visible=visible)

    tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    tform_inv.set_distortion(0., 0., bias[0], 0, 0)
    inputs_recreated = tform_inv.apply_distortion(inputs)

    tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    tform_inv.set_distortion(0., 0., 0, bias[1], 0)
    inputs_recreated = tform_inv.apply_distortion(inputs_recreated)

    tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    tform_inv.set_distortion(0., 0., 0, 0, bias[2])
    inputs_recreated = tform_inv.apply_distortion(inputs_recreated)

    diff_r = abs(indexes - inputs_recreated)
    displacement_r = np.sqrt(np.power(diff_r[:, 0], 2) + np.power(diff_r[:, 1], 2))
    displacement_r = displacement_r.reshape(x_size, y_size)
    mean_r = np.mean(displacement_r)
    Verbose.imshow(displacement_r)
    print("coefs gt: " + str([exp_k1, exp_k2, exp_k3]))
    print("mean: " + str(float(mean)))
    print("mean_r: " + str(float(mean_r)))
    print("max displacement:" + str(float(np.max(displacement_r))))


    output = None
    if output is not None:
        with open(output, 'w') as ofile:
            config["bias"] = {
                "x": float(bias[0][0]),
                "y": float(bias[0][1])
            }
            config["bias_history"] = {
                "x": [float(hist[0][0]) for hist in bias_history],
                "y": [float(hist[0][1]) for hist in bias_history]
            }
            ofile.write(yaml.dump(config))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="input/config.yaml",
        help="Config file for IGRE. For more see example file.",
    )
    parser.add_argument(
        "-x",
        "--x-shift",
        type=float,
        default=0,
        help="x-Shift of the input data",
    )
    parser.add_argument(
        "-y",
        "--y-shift",
        type=float,
        default=0,
        help="y-Shift of the input data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="yaml output file, where collected data will be placed",
    )
    args = parser.parse_args()
    igre_test(args.config, (args.x_shift, args.y_shift), args.output)
