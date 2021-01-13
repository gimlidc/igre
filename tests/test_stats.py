from modalities.stats import correlation
from numpy import corrcoef


def test_correlation():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 2, 3, 4, 5, 6, 7]

    assert correlation(x, y) == corrcoef(x, y)[0, 1]


if __name__ == "__main__":
    test_correlation()
    print("Stats passed")
