from src.models.igre_test import igre_test
import yaml
import numpy as np
import scipy
from copy import deepcopy
import os
import multiprocessing
from src.config.radial_distortion_test_generator import RadialDistortionGenerator
from src.config.random_crop_generator import RandomCropGenerator

OUT_FILENAME_FORMAT = "t_{K1}_{K2}_{K3}" \
                      + "_modstep{MODALITY_DIFF}_sample{SAMPLE}_{CUSTOM}.result"

ROOT_DIR = os.path.abspath(os.curdir)


def __create_batch(config, crop_index, transformation, custom):
    """ In principle we expect valid configuration for igre in batch config. But there is possible batch
    configuration in "batch" property. Each variable requires special care, therefore there will be an if
    block in this method for each batch configuration.
    """
    template = deepcopy(config)
    del template["batch"]
    batch = []
    # param = "output_dimension"
    # for value in np.arange(config["batch"][param]["min"],
    #                        config["batch"][param]["max"],
    #                        config["batch"][param]["step"]):
    #     new_cfg = deepcopy(template)
    #     new_cfg["output_dimensions"]["min"] = value
    #     new_cfg["output_dimensions"]["max"] = value
    #     new_cfg["output"] = new_cfg["output"] \
    #         .replace("{MODALITY_DIFF}", str(value - new_cfg["input_dimensions"]["min"]))
    #     batch01.append(new_cfg)

    param = "matfile"
    for input_data in config["batch"][param]["array"]:
        new_cfg = deepcopy(template)
        new_cfg["matfile"] = input_data[:-4]  # remove .mat suffix

        image = scipy.io.loadmat(os.path.join(ROOT_DIR, "data", "raw", input_data))['data']

        crop_generator = RandomCropGenerator(image.shape)
        crop = crop_generator.get_crop(crop_index)

        filename_out = f"{input_data[:-4]}-x{crop[0]}-y{crop[1]}-w{crop[2]}-h{crop[3]}"
        new_cfg["output"] = f"t_{transformation[2]:.2f}_{transformation[3]:.2f}_{transformation[4]:.2f}" \
                            f"_sample_{filename_out}_{custom}.result"

        new_cfg["crop"] = {
            "left_top": {
                "x": crop[0],
                "y": crop[1]
            },
            "size": {
                "width": crop[2],
                "height": crop[3]
            }
        }
        batch.append(new_cfg)
    return batch


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
        "-r",
        "--radial_distortion_index",
        type=int,
        default=0,
        help="One of possible radial distortions (from the generator) is selected",
    )
    # parser.add_argument(
    #     "-m",
    #     "--mat_file",
    #     type=str,
    #     default="leonardo.mat",
    #     help="matfile with input nd-array",
    # )
    parser.add_argument(
        "-t",
        "--crop_index",
        type=int,
        default=0,
        help="from the list of possible crops one is selected (according to this index)",
    )
    parser.add_argument(
        "-s",
        "--repeats",
        type=int,
        default=1,
        help="index of this run - there are multiple with the same configuration",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=1,
        help="path to output directory",
    )

    args = parser.parse_args()
    with open(args.config) as config_file:
        batch_config = yaml.load(config_file, Loader=yaml.FullLoader)

    distortion_generator = RadialDistortionGenerator(max_displacement=5)
    batch = __create_batch(batch_config,
                           args.crop_index,
                           distortion_generator.get_radial_distortion_params(args.radial_distortion_index),
                           args.repeats)
    print("done")

    #cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(1)

    for run_conf in batch:
        # igre_test(run_conf,
        #           distortion_generator.get_radial_distortion_params(args.radial_distortion_index),
        #           os.path.join(args.outdir, run_conf["output"]))
        pool.apply_async(igre_test, [run_conf,
                                     distortion_generator.get_radial_distortion_params(args.radial_distortion_index),
                                     os.path.join(args.outdir, run_conf["output"])])

    pool.close()
    pool.join()
