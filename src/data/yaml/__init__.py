import yaml


def parse_reached_trasformation_from_raw(filename):
    with open(filename, "rt") as file:
        results = yaml.load(file, Loader=yaml.Loader)
    return [float(results["bias"]["x"]),
            float(results["bias"]["y"]),
            float(results["bias"]["rotation"]),
            float(results["bias"]["scale_x"]),
            float(results["bias"]["scale_y"])]


if __name__ == "__main__":
    print(parse_reached_trasformation_from_raw("./data/bias_ouput.yaml"))