# TODO: Add description and names
import numpy as np
import os
from enum import Enum, auto
ALMA_NAME = 'alma'
FIRENZE_NAME = 'firenze'
# TODO: Solve more elegant way
MATERIALS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), './materials.npz')


class Material(Enum):
    ALMA = auto()
    FIRENZE = auto()


def load_alma():
    return np.load(MATERIALS_FILE)[ALMA_NAME]


def load_firenze():
    return np.load(MATERIALS_FILE)[FIRENZE_NAME]