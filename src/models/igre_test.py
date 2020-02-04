import yaml
from src.models import igre
from termcolor import colored
from src.logging.verbose import Verbose
import os
import numpy as np
import scipy.io
from src.config.tools import init_config
from src.registration.transformation import Transformation

ROOT_DIR = os.path.abspath(os.curdir)


def data_crop(config, dataset):
    if "crop" in config:
        print("Data crop ... " + colored("YES", "green") + ":", Verbose.debug)
        print("\t["
              + str(config["crop"]["left_top"]["x"]) + ":"
              + str(config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]) + ", "
              + str(config["crop"]["left_top"]["y"]) + ":"
              + str(config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]) + ", :]", Verbose.debug)
        dataset = dataset[
                  config["crop"]["left_top"]["x"]: (config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]),
                  config["crop"]["left_top"]["y"]: (config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]),
                  :]
    else:
        print("Data crop: " + colored("NO", "red"), Verbose.debug)

    return dataset


def igre_test(conf, transformation, output):
    """
    Information gain and registration test works with registered inputs. For testing registration layer, input pixels
    are shifted by shift[0] in x axis and by shift[1] in y axis.
    :param conf: configuration of IGRE run, accepted is dict of parsed values as well as filename with stored params
    :param transformation: tuple containing shift in x, in y, rotation and scale [x,y]
    :param output: output file for measured data
    :return: registration layer weights (i.e. computed shift)
    """
    if os.path.isfile(output):
        print(output, "already exist. Skipping.")
        return

    # Config load and integrity check
    if type(conf) == str:
        with open(conf, "rt", encoding='utf-8') as config_file:
            config = yaml.load(config_file)
    else:
        config = conf.copy()
    init_config(config)

    Verbose.print(
        "\nWelcome to " + colored("IGRE-test", "green") + " run with file: " + colored(config["matfile"], "green") +
        " expected transformation: " + colored(transformation, "green") + "\n")

    # Load dataset
    if "matfile" in config:
        dataset = np.float64(scipy.io.loadmat(os.path.join(ROOT_DIR, "data", "raw", config["matfile"]))['data'])
    else:
        with open(config["numpyfile"]["data"]) as infile:
            dataset = np.load(infile)
    Verbose.print("Data stats (before normalization): min = " + str(np.min(dataset)) +
                  " max = " + str(np.max(dataset)), Verbose.debug)
    # data normalization - ranged
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

    # Crop image according to config
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
    # Create [x,y] pairs as the input for ANN
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

    # Use just specified modalities
    outputs = dataset[:, :, config["output_dimensions"]["min"]: config["output_dimensions"]["max"] + 1]

    Verbose.print("\tOutput shape: " + str(outputs.shape), Verbose.debug)

    Verbose.print("\nCalling " + colored("IGRE\n", "green") + "...")

    # Setup transformation:
    shift = (-transformation[0], -transformation[1])
    tform = Transformation(a=(transformation[3], 0.0), b=(0.0, transformation[4],), c=shift)
    tform.set_rotation(transformation[2])  # 0.05236 rad

    Verbose.imshow(tform.tform_img(visible))

    inputs = tform.transform(indexes)
    bias, bias_history = igre.run(inputs,
                                  outputs,
                                  visible=visible)

    if output is not None:
        with open(output, 'w') as ofile:
            config["bias"] = {
                "x": float(bias[0][0][0] * config["layer_normalization"]["shift"]),
                "y": float(bias[0][0][1] * config["layer_normalization"]["shift"]),
                "rotation": float(bias[1][0][0] * config["layer_normalization"]["rotation"] * 180 / np.pi),
                "scale_x": float(bias[2][0][0] * config["layer_normalization"]["scale"] + 1),
                "scale_y": float(bias[2][0][1] * config["layer_normalization"]["scale"] + 1)
            }
            if "print_bias_history" in config["train"]:
                bias_history["shift_x"] = bias_history["shift_x"] * config["layer_normalization"]["shift"]
                bias_history["shift_y"] = bias_history["shift_y"] * config["layer_normalization"]["shift"]
                bias_history["rotation"] = [rot * float(config["layer_normalization"]["rotation"] * 180 / np.pi)
                                            for rot in bias_history["rotation"]]
                bias_history["scale_x"] = [(scale * config["layer_normalization"]["scale"]) + 1
                                           for scale in bias_history["scale_x"]]
                bias_history["scale_y"] = [(scale * config["layer_normalization"]["scale"]) + 1
                                           for scale in bias_history["scale_y"]]
                config["bias_history"] = bias_history

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
        "-r",
        "--rotation",
        type=float,
        default=0,
        help="Rotation of the input data",
    )
    parser.add_argument(
        "-t",
        "--x-scale",
        type=float,
        default=1,
        help="x-scale of the input data",
    )
    parser.add_argument(
        "-u",
        "--y-scale",
        type=float,
        default=1,
        help="y-scale of the input data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="yaml output file, where collected data will be placed",
    )
    args = parser.parse_args()
    igre_test(args.config, (args.x_shift, args.y_shift, args.rotation, args.x_scale, args.y_scale), args.output)
