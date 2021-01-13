import numpy as np


def rescale_range(values, mn=None, mx=None):
    """A data normalization function.

     out = rescaleRange(values) shifts values in range [0,1] linearily.

     out = rescaleRange(values, fraction) takes values in range
     [quantile(fraction/2), quantile(1-faction/2)] and shifts them into [0,1]
     values out of this range are set to 0 or 1 respectively. Fraction is a
     double in range [0,1].

     out = rescaleRange(values, min, max) shifts values in range
     [0,1]. Values >= max are transformed to 1 and values <= min are
     transformed to 0. Values between min and max are transformed
     linearily.

     Output values are in range [0,1].

     max and min cannot be the same number (otherwise NaN is in the
     output).
     """
    if mx is None:
        if mn is None:
            mn = np.min(values)
            mx = np.max(values)
        else:
            perc = mn
            mn = np.quantile(values, perc / 2)
            mx = np.quantile(values, 1 - perc / 2)

    out = values.copy()
    out[out < mn] = mn
    out[out > mx] = mx
    out = (out - mn) / (mx - mn)

    return out
