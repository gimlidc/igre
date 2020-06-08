from src.models.igre_test import igre_test
import yaml
import numpy as np
from copy import deepcopy
import os
import multiprocessing

OUT_FILENAME_FORMAT = "t_{C_X}_{C_Y}_{K1}_{K2}_{K3}" \
                      + "_modstep{MODALITY_DIFF}_sample{SAMPLE}_{CUSTOM}.result"


def __create_batch(config, transformation, custom):
    """ In principle we expect valid configuration for igre in batch config. But there is possible batch
    configuration in "batch" property. Each variable requires special care, therefore there will be an if
    block in this method for each batch configuration.
    """
    template = deepcopy(config)
    del template["batch"]
    template["output"] = OUT_FILENAME_FORMAT.replace("{C_X}", str(transformation[0])) \
        .replace("{C_Y}", str(transformation[1])) \
        .replace("{K1}", str(transformation[2])) \
        .replace("{K2}", str(transformation[3])) \
        .replace("{K3}", str(transformation[4]))
    batch01 = []
    param = "output_dimension"
    for value in np.arange(config["batch"][param]["min"],
                           config["batch"][param]["max"],
                           config["batch"][param]["step"]):
        new_cfg = deepcopy(template)
        new_cfg["output_dimensions"]["min"] = value
        new_cfg["output_dimensions"]["max"] = value
        new_cfg["output"] = new_cfg["output"] \
            .replace("{MODALITY_DIFF}", str(value - new_cfg["input_dimensions"]["min"]))
        batch01.append(new_cfg)
    batch02 = []
    for cfg in batch01:
        param = "matfile"
        for i in range(config["batch"][param]["min"], config["batch"][param]["max"]):
            new_cfg = deepcopy(cfg)
            new_cfg["matfile"] = config["batch"][param]["template"] \
                .replace(config["batch"][param]["replace"], str(i))
            new_cfg["output"] = new_cfg["output"].replace("{SAMPLE}", str(i)).replace("{CUSTOM}", str(custom))
            batch02.append(new_cfg)
    return batch02


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./data/interim/examples/config-igre-batch-test.yaml",
        help="Config file for IGRE batch. For more see example file.",
    )
    parser.add_argument(
        "-d",
        "--batch_dir",
        type=str,
        default="data/processed/metacentrum/01-registration-experiment",
        help="yaml output file, where collected data will be placed",
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
        "-s",
        "--repeats",
        type=int,
        default=20,
        help="Number of repeated computations with different random seed",
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
        default=0,
        help="x-scale of the input data",
    )
    parser.add_argument(
        "-u",
        "--y-scale",
        type=float,
        default=0,
        help="y-scale of the input data",
    )
    parser.add_argument(
        "-k",
        "--k1",
        type=float,
        default=0,
        help="radial distortion applied on input image (1 + K1 * r^2 + k2 * r^4 + k3 * r^6"
    )
    parser.add_argument(
        "-l",
        "--k2",
        type=float,
        default=0,
        help="radial distortion applied on input image (1 + k1 * r^2 + K2 * r^4 + k3 * r^6"
    )
    parser.add_argument(
        "-m",
        "--k3",
        type=float,
        default=0,
        help="radial distortion applied on input image (1 + k1 * r^2 + k2 * r^4 + K3 * r^6"
    )
    parser.add_argument(
        "-c",
        "--cx",
        type=float,
        default=0,
        help="Center of radial distortion in coordinates of the input image",
    )
    parser.add_argument(
        "-y",
        "--cy",
        type=float,
        default=0,
        help="Center of radial distortion in coordinates of the input image",
    )

    args = parser.parse_args()
    if not os.path.exists(args.batch_dir):
        os.makedirs(args.batch_dir)
    with open(args.config) as config_file:
        batch_config = yaml.load(config_file, Loader=yaml.FullLoader)
    batch = __create_batch(batch_config, (args.cx,
                                          args.cy,
                                          args.k1,
                                          args.k2,
                                          args.k3), args.repeats)
    print("done")

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(1)

    for run_conf in batch:
        pool.apply_async(igre_test, [run_conf, (args.cx,
                                                args.cy,
                                                args.k1,
                                                args.k2,
                                                args.k3),
                                     os.path.join(args.batch_dir, run_conf["output"])])

    pool.close()
    pool.join()
