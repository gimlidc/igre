import cv2
import scipy.io
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

# globals

verbose_level = 1  # display flag, tied to Verbose "enum"
#     always = -1       -  displays only things with always tag
#     metacentrum = 0   -  metacentrum setting, display almost nothing possibly dump into file
#     normal = 1        -
#     debug = 4         - display everything but things with never tag
#     never = 666       -

config = {}
shift_multi = 350


class Verbose:
    # enum
    always = -1  # always, prints this
    metacentrum = 0
    normal = 1
    debug = 4
    never = 666

    # TODO: if level = metacentrum:
    #              dump_to_file()
    @staticmethod
    def print(a, level=1):
        if level <= verbose_level:
            print(a)

    @staticmethod
    def imshow(a, level=1):
        if level <= verbose_level:
            plt.imshow(a, cmap='gray')
            plt.show()

    @staticmethod
    def plot(a, level=1):
        if level <= verbose_level:
            plt.plot(a)
            plt.show()


def read_from_config(config, property_name, default_value = None):
    """
    Reads top level property from config, wrapped in try/except for safety
    :param config: config to read form, allows to supply config subsection to get to lover levels properties
    :param property_name: top level property name
    :param default_value: default if property is not in config
    :return: property value
    """
    try:
        property_value = config[property_name]
    except:
        property_value = default_value
    return property_value
