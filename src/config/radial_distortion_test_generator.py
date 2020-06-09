import numpy as np


class RadialDistortionGenerator:
    """
    For faster production of testing radial distortion we recompute the constraint for pixel displacement to a list of
    possible radial configurations. This class is able to serve requests for radial distortions
    """

    def __init__(self, max_displacement, k1=0.04, k2=0.04, k3=0.04, steps=11, image_size=600):
        self.sampling = steps
        self.a_range = k1
        self.b_range = k2
        self.c_range = k3
        self.image_size = image_size
        self.max_displacement = max_displacement
        self.__generate()

    def __generate(self):
        self.result = [(0,0,0)]
        for i in np.linspace(-self.a_range, self.a_range, self.sampling):
            for j in np.linspace(-self.b_range, self.b_range, self.sampling):
                for k in np.linspace(-self.c_range, self.c_range, self.sampling):
                    L = 2 * i + 4 * j + 8 * k
                    diff = abs(L * self.image_size)
                    if diff < self.max_displacement:
                        self.result.append((0, 0, i, j, k))

    def get_radial_distortion_params(self, index):
        return self.result[index]

    def avail(self):
        return len(self.result)