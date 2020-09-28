from src.registration.transformation import Transformation as T
import numpy as np


class TestTransformation:
    @staticmethod
    def test_normalize_denormalize_coords():
        coords = np.meshgrid(range(5), range(5))
        coord_list = np.stack(coords, axis=2).reshape(25, 2)
        tcoords, shape = T.normalize_coordinates(coord_list)
        ucoords = T.denormalize_coordinates(tcoords, shape)

        np.testing.assert_array_almost_equal(coords, ucoords)
