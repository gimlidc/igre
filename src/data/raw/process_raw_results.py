import os
import re
import pickle
from src.data.yaml import parse_reached_trasformation_from_raw

def pickle_data(directory: str, output: str):
    databunch = list()
    for file in os.listdir(directory):
        pattern = re.compile(
            r"x([\-]{0,1}[0-9\.]*)_y([\-]{0,1}[0-9\.]*)_modality_step([\-]{0,1}[0-9]{1,2})_([0-9]{1,2})\.result")
        params = pattern.match(file)
        data_record = [float(params.group(1)),
                       float(params.group(2)),
                       int(params.group(3)),
                       float(params.group(4))]
        data_record.extend(parse_reached_trasformation_from_raw(os.path.join(directory, file)))
        databunch.append(data_record)

    print(databunch)
    pickle_out = open(output, "wb")
    pickle.dump(databunch, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--batch_dir",
        type=str,
        default="data/processed/metacentrum/06_registration_experiment",
        help="Directory with result files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/processed/metacentrum/06_pickled.pkl",
        help="output file",
    )

    args = parser.parse_args()

    pickle_data(args.batch_dir, args.output)
