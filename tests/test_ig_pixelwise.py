import numpy as np
import pytest
from stable.information_gain.pixelwise import information_gain
import tensorflow as tf
import os
import imageio

LEONARDO_MADONNA_WITH_YARNWINDER = "tests/assets/leonardo.npy"
MAGDALENA_VIS = "tests/assets/mari_magdalena-detail.png"
MAGDALENA_NIR = "tests/assets/mari_magdalenaIR-detail.png"


@pytest.mark.skipif(not os.path.isfile(LEONARDO_MADONNA_WITH_YARNWINDER),
                    reason="Input file is missing")
def test_information_gain_on_leonardo():
    tf.random.set_random_seed(12345)
    data = np.load(LEONARDO_MADONNA_WITH_YARNWINDER)
    [diff, target, _] = information_gain(data[:, :, :16], data[:, :, 20:25])

    np.testing.assert_almost_equal(target, data[:, :, 20:25], 0)
    diff.shape = (data.shape[0], data.shape[1], 5)


@pytest.mark.skipif(not os.path.isfile(MAGDALENA_VIS),
                    reason="Input file is missing")
@pytest.mark.skipif(not os.path.isfile(MAGDALENA_NIR),
                    reason="Input file is missing")
def test_information_gain_on_steborice():
    tf.random.set_random_seed(12345)
    vis = np.array(imageio.imread(MAGDALENA_VIS), dtype=float)/255
    nir = np.array(imageio.imread(MAGDALENA_NIR), dtype=float)/255
    [diff, target, _] = information_gain(vis, nir)

    np.testing.assert_almost_equal(target, nir, 0)
    diff.shape = nir.shape