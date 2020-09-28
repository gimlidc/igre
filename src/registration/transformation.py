import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline


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
        mtx = np.matmul(transformation_matrix[:, :2], np.stack([self.a, self.b], axis=0))
        self.a = mtx[:, 0]
        self.b = mtx[:, 1]
        self.c += transformation_matrix[:, 2]

    def transform(self, coordinates):
        """
        Method transform input coordinates.
        :param coordinates: list
            2D planar coordinates
        :return: list
            new 2D planar coordinates
        """
        if len(coordinates) // 2 == 1:
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

    @staticmethod
    def normalize_coordinates(coordinates, width=None, height=None):
        if len(coordinates) // 2 == 1:
            coords = np.reshape(np.asarray(coordinates), (1, 2))
        else:
            coords = np.asarray(coordinates)

        out = np.zeros(coords.shape)
        if width is None:
            width = np.max(coords[:, 0])
        if height is None:
            height = np.max(coords[:, 1])
        out[:, 0] = 2 * (coords[:, 0] - width/2) / width
        out[:, 1] = 2 * (coords[:, 1] - height / 2) / height
        return out, np.array(width + 1, height + 1)

    @staticmethod
    def denormalize_coordinates(coordinates, shape):
        return coordinates * (shape-1)/2 + (shape-1)/2

    def apply_distortion(self, coordinates):
        coords, shape = self.normalize_coordinates(coordinates)

        # Compute standard radial distortion, on normalized coordinates [-1, 1],
        # function of even powers of radii (distance pixel - center of distortion) and coefficients of the distortion
        radii = np.sqrt(np.power(coords[:, 0] - self.cx, 2) + np.power(coords[:, 1] - self.cy, 2))
        L = np.multiply(np.power(radii, 2), self.k1) + \
            np.multiply(np.power(radii, 4), self.k2) + \
            np.multiply(np.power(radii, 6), self.k3) + 1.

        # Perform the transformation
        out = np.zeros(coords.shape)
        out[:, 0] = self.cx + np.multiply(coords[:, 0] - self.cx, L)
        out[:, 1] = self.cy + np.multiply(coords[:, 1] - self.cy, L)

        return self.denormalize_coordinates(out, shape)

    def apply_tform(self, img):
        xx, yy = np.meshgrid(range(img.shape[0]), range(img.shape[1]))
        coords = np.stack([xx, yy], axis=2).reshape(-1, 2)
        coords = self.transform(coords)
        coords = self.apply_distortion(coords)
        spline = RectBivariateSpline(range(img.shape[0]), range(img.shape[1]), img)
        return spline(coords[:, 0], coords[:, 1], grid=False).reshape(img.shape[1], img.shape[0]).T

    @staticmethod
    def affine(img, scale, rotation, shift):
        tform = Transformation.build_affine(scale, rotation, shift)
        return tform.apply_tform(img)

    @staticmethod
    def radial(img, cx, cy, k1, k2, k3):
        tform = Transformation.build_radial(cx, cy, k1, k2, k3)
        return tform.apply_tform(img)

    @staticmethod
    def build_affine(scale, rotation, shift):
        a = np.array([1 / scale, 0.])
        b = np.array([0., 1 / scale])
        c = np.array([-shift[0], -shift[1]])
        d = np.array([0., 0.])
        e = np.array([0., 0.])
        tform = Transformation(a, b, c, d, e)
        tform.set_rotation(rotation)
        return tform

    @staticmethod
    def build_radial(cx, cy, k1, k2, k3):
        tform = Transformation()
        tform.set_distortion(cx, cy, k1, k2, k3)
        return tform

    def __str__(self):
        return str(
            f"{self.a[0]:0.2f}x + {self.b[0]:0.2f}y + {self.c[0]:0.2f}\t"
            f"{self.a[1]:0.2f}x + {self.b[1]:0.2f}y + {self.c[1]:0.2f}\n"
            f"--------------------\t--------------------\n"
            f"{self.d[0]:0.2f}x + {self.e[0]:0.2f}y + 1\t"
            f"{self.d[0]:0.2f}x + {self.e[0]:0.2f}y + 1\n"
            f"\n\n"
            f"x + (x - {self.cx}) * ({self.k1}r^2 + {self.k2}r^4 + {self.k3}r^6)\n"
            f"y + (y - {self.cy}) * ({self.k1}r^2 + {self.k2}r^4 + {self.k3}r^6)"
        )
