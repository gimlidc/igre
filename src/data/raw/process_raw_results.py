import os
import re
import pickle
from src.data.yaml import parse_reached_trasformation_from_raw

def pickle_data(directory: str, output: str):
    databunch = list()
    for subfolder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, subfolder)):
            #"t_0.0_0.0_5.0_1.1_1.1_modstep0_sample13_1.result"
            pattern = re.compile(
                r"t([\-]{0,1}[0-9\.]*)_" +
                r"([\-]{0,1}[0-9\.]*)_" +
                r"([\-]{0,1}[0-9\.]*)_" +
                r"([\-]{0,1}[0-9\.]*)_" +
                r"([\-]{0,1}[0-9\.]*)_" +
                r"modstep([\-]{0,1}[0-9]{1,2})_" +
                r"sample([0-9]{1,2})_" +
                r"([0-9]{1,2})\.result")
            params = pattern.match(file)
            if params:
                data_record = [float(params.group(1)), # shift x
                               float(params.group(2)), # shift y
                               float(params.group(3)), # rotation
                               float(params.group(4)), # scale x
                               float(params.group(5)), # scale y
                               int(params.group(6)), # modstep
                               int(params.group(7)), # sample no
                               int(params.group(8))] # repeat
                data_record.extend(parse_reached_trasformation_from_raw(os.path.join(directory, subfolder, file)))
                databunch.append(data_record)
            else:
                print(file, "filename parsing failed.")

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
        default="data/processed/metacentrum/40-scale",
        help="Directory with result files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/processed/metacentrum/40_pickled.pkl",
        help="output file",
    )

    args = parser.parse_args()

    pickle_data(args.batch_dir, args.output)
