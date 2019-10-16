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


def igre_test(conf, shift, output):
    """
    Information gain and registration test works with registered inputs. For testing registration layer, input pixels
    are shifted by shift[0] in x axis and by shift[1] in y axis.
    :param conf: configuration of IGRE run
    :param shift: tuple containing shift in x and in y axis of input dimensions
    :param output: output file for measured data
    :return: registration layer weights (i.e. computed shift)
    """

    # Config load and integrity check
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
    tform = Transformation(c=shift)  # a=(1.1, -0.1), b=(0.1, 0.9),
    #tform = Transformation(a=(1.05, 0.0), b=(0.05, 1.0,))
    # TODO: nejdriv at to konverguje subpixel pro shift, pak az zkouset scale, rotaci atd.
    inputs = tform.transform(indexes)
    bias, bias_history = igre.run(inputs,
                                  outputs,
                                  visible=visible)

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
