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


def igre_test(conf, transformation, output):
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
        " expected transformation: " + colored(transformation, "green") + "\n")

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
    Verbose.imshow(visible[5:-5, 5:-5, 0])

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

    # Set radial distortion parameters
    k1 = transformation[0]
    k2 = transformation[1]
    k3 = transformation[2]
    tform.set_distortion(0., 0., k1, k2, k3)

    # calculate the best inverse radial distortion
    tform_inverse_gt = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    exp_k1 = -k1
    exp_k2 = 3 * k1 * k1 - k2
    exp_k3 = -12 * k1 * k1 * k1 + 8 * k1 * k2 - k3
    tform_inverse_gt.set_distortion(0., 0., exp_k1, exp_k2, exp_k3)

    inputs = tform.apply_distortion(indexes)
    a=1
    # Calculate error of "ground truth" inverse

    # tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    # tform_inv.set_distortion(0., 0., exp_k1, 0, 0)
    # sanitycheck = tform_inv.apply_distortion(inputs)
    #
    # tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    # tform_inv.set_distortion(0., 0., 0, exp_k2, 0)
    # sanitycheck = tform_inv.apply_distortion(sanitycheck)
    #
    # tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    # tform_inv.set_distortion(0., 0., 0, 0, exp_k3)
    # sanitycheck = tform_inv.apply_distortion(sanitycheck)

    # sanitycheck = tform_inverse_gt.apply_distortion(inputs)
    # diff = abs(indexes - sanitycheck)
    # displacement = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
    # displacement = displacement.reshape(x_size, y_size)
    # mean = np.mean(displacement)
    # Verbose.imshow(displacement, Verbose.debug)

    sanitycheck = tform_inverse_gt.apply_distortion(inputs)
    diff = abs(indexes - sanitycheck)
    displacement = np.sqrt(np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2))
    displacement = displacement.reshape(x_size, y_size)
    displacement = displacement[5:-5, 5:-5]
    mean = np.mean(displacement)
    Verbose.imshow(displacement, Verbose.debug)

    model, bias, bias_history = igre.run(inputs,
                                  outputs,
                                  visible=visible)

    # Given the architecture, the computed inverse is composed of 3 radial distortions
    tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    tform_inv.set_distortion(0., 0., bias[0], bias[1], bias[2])
    inputs_recreated = tform_inv.apply_distortion(inputs)

    # tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    # tform_inv.set_distortion(0., 0., 0, bias[1], 0)
    # inputs_recreated = tform_inv.apply_distortion(inputs_recreated)
    #
    # tform_inv = Transformation(a=(1.0, 0.0), b=(0.0, 1.,), c=shift)
    # tform_inv.set_distortion(0., 0., 0, 0, bias[2])
    # inputs_recreated = tform_inv.apply_distortion(inputs_recreated)

    diff_start = abs(indexes - inputs)
    displacement_start = np.sqrt(np.power(diff_start[:, 0], 2) + np.power(diff_start[:, 1], 2))
    displacement_start = displacement_start.reshape(x_size, y_size)
    displacement_start = displacement_start[5:-5, 5:-5]

    diff_recreated = abs(indexes - inputs_recreated)
    displacement_recreated = np.sqrt(np.power(diff_recreated[:, 0], 2) + np.power(diff_recreated[:, 1], 2))
    displacement_recreated = displacement_recreated.reshape(x_size, y_size)
    displacement_recreated = displacement_recreated[5:-5, 5:-5]
    mean_recreated = np.mean(displacement_recreated)
    Verbose.imshow(displacement_recreated)
    print("coefs gt: " + str([exp_k1, exp_k2, exp_k3]))
    print("mean_gt: " + str(float(mean)))
    print("max_gt: " + str(float(np.max(displacement))))
    print("mean: " + str(float(mean_recreated)))
    print("max: " + str(float(np.max(displacement_recreated))))
    print("mean_start: " + str(float(np.mean(displacement_start))))
    print("max_start: " + str(float(np.max(displacement_start))))


    tformed_image = model.predict(inputs)
    Verbose.imshow(tformed_image.reshape(x_size, y_size)[5:-5, 5:-5])


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
        "-k",
        "--radial-1",
        type=float,
        default=0,
        help="1st radial distortion with the power 2 of x",
    )
    parser.add_argument(
        "-l",
        "--radial-2",
        type=float,
        default=0,
        help="2nd radial distortion coefficient with x with 4th power",
    )
    parser.add_argument(
        "-m",
        "--radial-3",
        type=float,
        default=0,
        help="3rd radial distortion coefficient with x with 6th power",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="yaml output file, where collected data will be placed",
    )
    args = parser.parse_args()
    igre_test(args.config, (args.radial_1, args.radial_2, args.radial_3), args.output)
