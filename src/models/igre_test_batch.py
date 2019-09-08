from src.models.igre_test import igre_test
import yaml
import numpy as np
from copy import deepcopy
import os


OUT_FILENAME_FORMAT = "x{SHIFT_X}_y{SHIFT_Y}_modality_step{MODALITY_DIFF}_lr{LEARNING_RATE}.result"


def __create_batch(config, shift):
    """ In principle we expect valid configuration for igre in batch config. But there is possible batch
    configuration in "batch" property. Each variable requires special care, therefore there will be an if
    block in this method for each batch configuration.
    """
    template = deepcopy(config)
    del template["batch"]
    template["output"] = OUT_FILENAME_FORMAT.replace("{SHIFT_X}", str(shift[0])).replace("{SHIFT_Y}", str(shift[1]))
    batch = [template]
    for param in config["batch"]["params"]:
        if param == "output_dimension":
            new_batch = []
            for value in np.arange(config["batch"][param]["min"],
                                   config["batch"][param]["max"],
                                   config["batch"][param]["step"]):
                for cfg in batch:
                    new_cfg = deepcopy(cfg)
                    new_cfg["output_dimensions"]["min"] = value
                    new_cfg["output_dimensions"]["max"] = value
                    new_cfg["output"] = new_cfg["output"]\
                        .replace("{MODALITY_DIFF}", str(np.abs(new_cfg["input_dimensions"]["min"] - value)))
                    new_batch.append(new_cfg)
            batch = new_batch
        elif param == "learning_rate":
            new_batch = []
            for value in config["batch"][param]:
                for cfg in batch:
                    new_cfg = deepcopy(cfg)
                    new_cfg["train"]["optimizer"]["learning_rate"] = value
                    new_cfg["output"] = new_cfg["output"].replace("{LEARNING_RATE}", str(value))
                    new_batch.append(new_cfg)
            batch = new_batch
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
    args = parser.parse_args()
    if not os.path.exists(args.batch_dir):
        os.makedirs(args.batch_dir)
    with open(args.config) as config_file:
        batch_config = yaml.load(config_file)
    batch = __create_batch(batch_config, (args.x_shift, args.y_shift))
    for run_conf in batch:
        igre_test(run_conf, (args.x_shift, args.y_shift), os.path.join(args.batch_dir, run_conf["output"]))
