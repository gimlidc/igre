import cv2
import numpy as np
from src.config.tools import get_config
import src.config.image_info as ii

class Transformation:
    """ Internal representation of 2D planar transformation.

    Perspective transformation defined as:
    (ax + by + c) / (dx + ey + 1)

    Can be defined directly by matrix setup or just rigid transformation defined by shift and rotation
    """

    def __init__(self, a=np.asarray((1., 0.)),
                 b=np.asarray((0., 1.)),
                 c=np.asarray((0., 0.)),
                 d=np.asarray((0., 0.)),
                 e=np.asarray((0., 0.))):
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        self.d = np.asarray(d)
        self.e = np.asarray(e)

        self.cx = np.asarray(0.)
        self.cy = np.asarray(0.)
        self.k1 = np.asarray(0.)
        self.k2 = np.asarray(0.)
        self.k3 = np.asarray(0.)

    def set_shift(self, shift):
        self.c = shift

    def set_rotation(self, angle_degrees, center=(0, 0)):
        transformation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)
        self.a = transformation_matrix[:, 0]
        self.b = transformation_matrix[:, 1]
        self.c += transformation_matrix[:, 2]

    def transform(self, coordinates):
        """
        Method transform input coordinates.
        :param coordinates: list
            2D planar coordinates
        :return: list
            new 2D planar coordinates
        """
        if len(coordinates)//2 == 1:
            coords = np.reshape(np.asarray(coordinates), (1, 2))
        else:
            coords = np.asarray(coordinates)
        transformed_coordinates = np.einsum("i,j->ij", coords[:, 0], self.a) + \
                                  np.einsum("i,j->ij", coords[:, 1], self.b) + \
                                  self.c
        denominator = np.einsum("i,j->ij", coords[:, 0], self.d) + \
                      np.einsum("i,j->ij", coords[:, 1], self.e) + 1
        transformed_coordinates = transformed_coordinates / denominator

        return transformed_coordinates

    def set_distortion(self, cx, cy, k1, k2, k3=0):
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def apply_distortion(self, coordinates):
        if len(coordinates)//2 == 1:
            coords = np.reshape(np.asarray(coordinates), (1, 2))
        else:
            coords = np.asarray(coordinates)

        config = get_config()
        w = ii.image.width - 1
        h = ii.image.height - 1
        crop_w = 0
        crop_h = 0
        if "crop" in config:
            crop_w = config["crop"]["left_top"]["y"]
            crop_h = config["crop"]["left_top"]["x"]

        image_size = np.amax(coords, axis=0)
        coords_norm = np.divide(np.multiply(coords + [crop_h, crop_w], 2), [h, w]) - [1., 1.]
        radii = np.sqrt(np.power(coords_norm[:, 0] - ii.image.c_x, 2) + np.power(coords_norm[:, 1] - ii.image.c_y, 2))
        L = np.multiply(np.power(radii, 2), self.k1) + \
            np.multiply(np.power(radii, 4), self.k2) + \
            np.multiply(np.power(radii, 6), self.k3) + 1.
        transformed_coordinates_x = ii.image.c_x + np.multiply(coords_norm[:, 0] - ii.image.c_x, L)
        transformed_coordinates_y = ii.image.c_y + np.multiply(coords_norm[:, 1] - ii.image.c_y, L)
        transformed_coordinates = np.vstack((transformed_coordinates_x, transformed_coordinates_y))
        transformed_coordinates = np.transpose(transformed_coordinates)
        transformed_coordinates = np.divide(np.multiply(transformed_coordinates + [1., 1.],
                                                        [h, w]), 2) - [crop_h, crop_w]

        return transformed_coordinates

    def __str__(self):
        return str(self.a[0]) + "x + " + str(self.b[0]) + "y + " + str(self.c[0]) + "\t" + \
               str(self.a[1]) + "x + " + str(self.b[1]) + "y + " + str(self.c[1]) + "\n" + \
               "-----------------\t-----------------\n" + \
               str(self.e[0]) + "x + " + str(self.e[0]) + "y + " + " 1\t" + \
               str(self.e[1]) + "x + " + str(self.e[1]) + "y + " + " 1"
