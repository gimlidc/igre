import cv2
import scipy.io
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

# globals
verbose = 0  # print flag, can be extended to levels e.g. 0 = nothing ... 3 = without images ... 5 = everything
config = {}
shift_multi = 350


def v_print(a):
    # global verbose
    if verbose:
        print(a)


def v_imshow(a):
    # global verbose
    if verbose:
        plt.imshow(a, cmap='gray')
        plt.show()

def v_plot(a):
    if verbose:
        plt.plot(a)
        plt.show()


class Transformation:
    def __init__(self, a=(1, 0), b=(0, 1), c=(0, 0), d=(0, 0), e=(0, 0), f=(1, 1)):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def set_shift(self, shift):
        self.c += shift

    def set_rotation(self, angle, center=(0, 0)):
        M = cv2.getRotationMatrix2D(center, angle, 1)
        self.a = M[:, 0]
        self.b = M[:, 1]
        self.c += M[:, 2]

    def transform(self, coordinates):
        r = coordinates*self.a
        r += coordinates*self.b
        r += self.c
        return r
