import os
import re
import pickle


def pickle_data(directory: str, output: str):
    databunch = list()
    for file in os.listdir(directory):
        pattern = re.compile(
            r"x([\-]{0,1}[0-9\.]*)_y([\-]{0,1}[0-9\.]*)_modality_step([\-]{0,1}[0-9]{1,2})_([0-9]{1,2})\.result")
        params = pattern.match(file)
        with open(os.path.join(directory, file)) as data:
            data.readline()  # bias line
            x = float(data.readline()[5:])
            y = float(data.readline()[5:])
            databunch.append([float(params.group(1)),
                              float(params.group(2)),
                              int(params.group(3)),
                              float(params.group(4)),
                              x,
                              y])
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
