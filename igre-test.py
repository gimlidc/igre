import yaml
import igre
from tftools.optimizer_builder import build_optimizer
import utils
from utils import *


def check_config(conf):
    if "layers" in conf:
        print("Config integrity: " + colored("OK", "green"), Verbose.always)
    else:
        print("Config integrity: " + colored("FAILED", "red"), Verbose.always)
        exit(1)


def data_crop(conf, dataset):
    if "crop" in config:
        print("Data crop ... " + colored("YES", "green") + ":", Verbose.debug)
        print("\t["
              + str(conf["crop"]["left_top"]["x"]) + ":"
              + str(conf["crop"]["left_top"]["x"] + conf["crop"]["size"]["width"]) + ", "
              + str(conf["crop"]["left_top"]["y"]) + ":"
              + str(conf["crop"]["left_top"]["y"] + conf["crop"]["size"]["height"]) + ", :]", Verbose.debug)
        dataset = dataset[
                  conf["crop"]["left_top"]["x"]: (conf["crop"]["left_top"]["x"] + conf["crop"]["size"]["width"]),
                  conf["crop"]["left_top"]["y"]: (conf["crop"]["left_top"]["y"] + conf["crop"]["size"]["height"]),
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
    utils.config = yaml.load(open(conf, 'r'), Loader=yaml.FullLoader)
    config = utils.config
    utils.shift_multi = config["train"]["shift_learning_multi"]
    utils.verbose_level = read_from_config(config, "verbose_level", Verbose.normal)

    check_config(config)

    Verbose.print("\nWelcome to " + colored("IGRE-test", "green") + " run with file: " + colored(config["matfile"], "green") +
                  " expected shift: " + colored(shift, "green") + "\n")

    dataset = np.float64(scipy.io.loadmat(config["matfile"])['data'])
    Verbose.print("Data stats (before normalization): min = " + str(np.min(dataset)) +
                  " max = " + str(np.max(dataset)), Verbose.debug)
    # data normalization - ranged
    dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

    Verbose.print("Dataset shape: " + str(dataset.shape), Verbose.debug)
    dataset = data_crop(config, dataset)

    Verbose.print(colored("Input", "green") + " dimensions: " + colored(("[" +
                                                                 str(config["input_dimensions"]["min"]) + "-" +
                                                                 str(config["input_dimensions"]["max"]) + "]"),
                                                                "green"), Verbose.debug)
    visible = dataset[:, :, config["input_dimensions"]["min"]: config["input_dimensions"]["max"] + 1]
    Verbose.imshow(visible[:, :, 0])

    x = visible.shape[:-1]
    indexes = np.indices(x)
    indexes = indexes.reshape((len(visible.shape[:-1]), -1)).transpose().astype(np.float32)

    Verbose.print("\tInputs shape: " + str(visible.shape), Verbose.debug)
    Verbose.print(colored("Output", "green") + " dimensions: " + colored(("[" +
                                                                  str(config["output_dimensions"]["min"]) + "-" +
                                                                  str(config["output_dimensions"]["max"]) + "]"),
                                                                 "green"), Verbose.debug)
    outputs = dataset[:, :, config["output_dimensions"]["min"]: config["output_dimensions"]["max"] + 1]

    # Adding gaussian blur to data
    # for b_size in [21, 51]:  # , 21, 31, 41, 51]:
    #    blurred = cv2.GaussianBlur(outputs[:, :, 0], (b_size, b_size), 0)
    #    blurred = blurred.reshape(blurred.shape[0], blurred.shape[1], 1)
    #    plt.imshow(blurred[:, :, 0], cmap='gray')
    #    plt.show()
    #    outputs = np.append(outputs, blurred, axis=2)

    Verbose.print("\tOutput shape: " + str(outputs.shape), Verbose.debug)

    Verbose.print("\nCalling " + colored("IGRE\n", "green") + "...")

    # coordinate transform up to perspective transform
    T = Transformation(c=shift)  # a=(1.1, -0.1), b=(0.1, 0.9),
    inputs = T.transform(indexes)
    bias, bias_history = igre.run(inputs,
                                  outputs,
                                  visible=visible,
                                  optimizer=build_optimizer(config["train"]["optimizer"]),
                                  layers=config["layers"],
                                  batch_size=config["train"]["batch_size"],
                                  epochs=config["train"]["epochs"]
                                  )

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
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="yaml output file, where collected data will be placed",
    )
    args = parser.parse_args()
    igre_test(args.config, (args.x_shift, args.y_shift), args.output)
