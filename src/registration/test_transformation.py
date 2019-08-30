from unittest import TestCase
from src.registration.transformation import Transformation


class TestTransformation(TestCase):

    tform = None

    def setUp(self):
        self.tform = Transformation()

    def tearDown(self) -> None:
        del self.tform

    def test_identity(self):
        src_coords = (1, 1)
        target_coords = (1., 1.)
        self.assertAlmostEqual(self.tform.transform(src_coords)[0], target_coords[0], delta=0.00001)
        self.assertAlmostEqual(self.tform.transform(src_coords)[1], target_coords[1], delta=0.00001)

    def test_rotation_90_deg(self):
        self.tform.set_rotation(90)
        src_coords = (1, 1)
        target_coords = (1., -1.)
        self.assertAlmostEqual(self.tform.transform(src_coords)[0], target_coords[0], delta=0.00001)
        self.assertAlmostEqual(self.tform.transform(src_coords)[1], target_coords[1], delta=0.00001)

    def test_shift(self):
        self.tform.set_shift((2, 3))
        src_coords = (1, 1)
        target_coords = (3., 4.)
        self.assertAlmostEqual(self.tform.transform(src_coords)[0], target_coords[0], delta=0.00001)
        self.assertAlmostEqual(self.tform.transform(src_coords)[1], target_coords[1], delta=0.00001)

    def test_rigid(self):
        self.tform.set_rotation(90)
        self.tform.set_shift((2, 3))
        src_coords = (1, 1)
        target_coords = (3., 2.)
        self.assertAlmostEqual(self.tform.transform(src_coords)[0], target_coords[0], delta=0.00001)
        self.assertAlmostEqual(self.tform.transform(src_coords)[1], target_coords[1], delta=0.00001)
