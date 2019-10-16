from termcolor import colored
from jsonschema import validate

config_schema = {
    "additionalProperties": True,
    "image": {
        "type": "string"
    },
    "matfile": {
        "required": False
    },
    "crop": {
        "type": "object",
        "properties": {
            "left_top": {
                "type": "object",
                "properties": {
                    "x": "nonNegativeInteger",
                    "y": "nonNegativeInteger"
                },
            },
            "size": {
                "type": "object",
                "properties": {
                    "width": "nonNegativeInteger",
                    "height": "nonNegativeInteger",
                }
            }
        }
    },
    "layers": {
        "type": "array",
        "items": {
            "type": "nonNegativeInteger"
        }
    },
    "input_dimensions": {
        "type": "object",
        "properties": {
            "min": "nonNegativeInteger",
            "max": "nonNegativeInteger"
        }
    },
    "output_dimensions": {
        "type": "object",
        "properties": {
            "min": "nonNegativeInteger",
            "max": "nonNegativeInteger"
        }
    },
    "expected_max_shift_px": "nonNegativeInteger",
    "train": {
        "type": "object",
        "required": True,
        "properties": {
            "batch_size": "nonNegativeInteger",
            "epochs": "nonNegativeInteger",
            "use_gpu": "boolean",
            "optimizer": {
                "required": True,
                "type": "object",
                "properties": {
                    "family": {
                        "required": True,
                        "type": "string"
                    },
                    "learning_rate": {
                        "required": True,
                        "type": "number"
                    },
                    "beta1": "number",
                    "beta2": "number"
                }
            }
        }
    }
}

__instance = None
__setup = False


def init_config(config: dict):
    global __setup
    global __instance
    # if not __setup:
    __check_config(config)
    __instance = config.copy()
    __setup = True
    # else:
    #     raise ValueError("Config was already set")


def get_config():
    global __setup
    global __instance
    """Method can be used after config initialization"""
    return __instance.copy()


def get_or_default(name, default):
    global __setup
    global __instance
    if __instance is None:
        return default
    parts = name.split(".")
    root = __instance.copy()
    for part in parts:
        if part in root:
            root = root[part]
        else:
            return default
    return root


def __check_config(conf):
    try:
        validate(conf, config_schema)
        print("Config integrity: " + colored("OK", "green"))
    except Exception as e:
        print("Config integrity: " + colored("FAILED", "red"))
        print(e)
        exit(1)
