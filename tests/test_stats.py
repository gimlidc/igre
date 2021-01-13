from stable.modalities.stats import correlation
import numpy as np


def test_correlation():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    tested = np.stack((x.reshape(4, 2), y.reshape(4, 2)))

    np.testing.assert_almost_equal(correlation(tested), np.corrcoef(x, y))
