import cv2
import numpy as np


class Transformation:

    # Default transformation is an identity
    a = np.asarray((1., 0.))
    b = np.asarray((0., 1.))
    c = np.asarray((0., 0.))
    d = np.asarray((0., 0.))
    e = np.asarray((0., 0.))

    """ Internal representation of 2D planar transformation.

    Perspective transformation defined as:
    (ax + by + c) / (dx + ey + 1)

    Can be defined directly by matrix setup or just rigid transformation defined by shift and rotation
    """
    def __init__(self):
        pass

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
        transformed_coordinates = coordinates[0] * self.a + coordinates[1] * self.b + self.c
        denominator = coordinates[0] * self.d + coordinates[1] * self.e + 1
        transformed_coordinates = transformed_coordinates / denominator

        return transformed_coordinates

    def __str__(self):
        return  str(self.a[0]) + "x + " + str(self.b[0]) + "y + " + str(self.c[0]) + "\t" + \
                str(self.a[1]) + "x + " + str(self.b[1]) + "y + " + str(self.c[1]) + "\n" + \
                "-----------------\t-----------------\n" + \
                str(self.e[0]) + "x + " + str(self.e[0]) + "y + " + " 1\t" + \
                str(self.e[1]) + "x + " + str(self.e[1]) + "y + " + " 1"
