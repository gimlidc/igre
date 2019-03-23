import yaml
import scipy.io
import numpy as np
import igre
from termcolor import colored


def check_config(config):
    if "layers" in config:
        print("Config integrity: " + colored("OK", "green"))
    else:
        print("Config integrity: " + colored("FAILED", "red"))
        exit(1)


def data_crop(config, dataset):
    if "crop" in config:
        print("Data crop ... " + colored("YES", "green") + ":")
        print("\t["
              + str(config["crop"]["left_top"]["x"]) + ":"
              + str(config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]) + ", "
              + str(config["crop"]["left_top"]["y"]) + ":"
              + str(config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]) + ", :]")
        dataset = dataset[
                  config["crop"]["left_top"]["x"]: (config["crop"]["left_top"]["x"] + config["crop"]["size"]["width"]),
                  config["crop"]["left_top"]["y"]: (config["crop"]["left_top"]["y"] + config["crop"]["size"]["height"]),
                  :]
    else:
        print("Data crop: " + colored("NO", "red"))

    return dataset


def igre_test(config, shift):
    """
    Information gain and registration test works with registered inputs. For testing registration layer, input pixels
    are shifted by shift[0] in x axis and by shift[1] in y axis.
    :param config: configuration of IGRE run
    :param shift: tuple containing shift in x and in y axis of input dimensions
    :return: registration layer weights (i.e. computed shift)
    """

    # Config load and integrity check
    config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    check_config(config)

    print("\nWelcome to " + colored("IGRE-test", "green") + " run with file: " + colored(config["matfile"],"green") +
          " expected shift: " + colored(shift, "green") + "\n")


    dataset = np.float64(scipy.io.loadmat(config["matfile"])['data'])
    print("Data stats (before normalization): min = " + str(np.min(dataset)) + " max = " + str(np.max(dataset)))
    # data normalization - ranged
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

    print("Dataset shape: " + str(dataset.shape))
    dataset = data_crop(config, dataset)

    print(colored("Input", "green") + " dimensions: " + colored(("[" +
                       str(config["input_dimensions"]["min"]) + "-" +
                       str(config["input_dimensions"]["max"]) + "]"), "green"))
    visible = dataset[:, :, config["input_dimensions"]["min"]: config["input_dimensions"]["max"] + 1]
    indexes = np.indices(visible.shape[:-1]).reshape((len(visible.shape[:-1]), -1)).transpose().astype(np.float32)

    print("\tInputs shape: " + str(visible.shape))
    print(colored("Output", "green") + " dimensions: " + colored(("[" +
                                                                              str(config["output_dimensions"][
                                                                                      "min"]) + "-" +
                                                                              str(config["output_dimensions"][
                                                                                      "max"]) + "]"), "green"))
    outputs = dataset[:, :, config["output_dimensions"]["min"]: config["output_dimensions"]["max"] + 1]
    print("\tOutput shape: " + str(outputs.shape))

    print("\nCalling " + colored("IGRE\n", "green") + "...")

    igre.run(indexes + shift,
             outputs,
             visible=visible,
             layers=config["layers"],
             batch_size=config["train"]["batch_size"],
             epochs=config["train"]["epochs"],
             use_gpu=config["train"]["use_gpu"],
             optimizer=config["train"]["optimizer"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./input/config.yaml",
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
    args = parser.parse_args()
    igre_test(args.config, (args.x_shift, args.y_shift))
