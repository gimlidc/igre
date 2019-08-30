from unittest import TestCase
from src.registration.transformation import Transformation
import numpy as np
from numpy.testing import assert_array_almost_equal


class TestTransformation(TestCase):

    tform = None

    def setUp(self):
        self.tform = Transformation()

    def tearDown(self) -> None:
        del self.tform

    def test_identity(self):
        src_coords = (1, 1)
        target_coords = np.asarray([[1., 1.]])
        assert_array_almost_equal(self.tform.transform(src_coords), target_coords)

    def test_rotation_90_deg(self):
        self.tform.set_rotation(90)
        src_coords = (1, 1)
        target_coords = np.asarray([[1., -1.]])
        assert_array_almost_equal(self.tform.transform(src_coords), target_coords)

    def test_shift(self):
        self.tform.set_shift((2, 3))
        src_coords = (1, 1)
        target_coords = np.asarray([[3., 4.]])
        assert_array_almost_equal(self.tform.transform(src_coords), target_coords)

    def test_rigid(self):
        self.tform.set_rotation(90)
        self.tform.set_shift((2, 3))
        src_coords = (1, 1)
        target_coords = np.asarray([[3., 2.]])
        assert_array_almost_equal(self.tform.transform(src_coords), target_coords)
