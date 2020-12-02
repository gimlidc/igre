import numpy as np

xyz_sensitivity = np.array([
    [0.0076, 0.0002, 0.0362,],
    [0.0776, 0.0022, 0.3713,],
    [0.3187, 0.0480, 1.7441,],
    [0.0580, 0.1693, 0.6162,],
    [0.0093, 0.5030, 0.1582,],
    [0.1655, 0.8620, 0.0422,],
    [0.4335, 0.9950, 0.0088,],
    [0.7621, 0.9520, 0.0021,],
    [1.0263, 0.7570, 0.0011,],
    [1.0026, 0.5030, 0.0003,],
    [0.6424, 0.2650, 0.0001,],
    [0.2835, 0.1070     , 0,],
    [0.0636, 0.0232     , 0,],
    [0.0081, 0.0029     , 0,],
    [0.0010, 0.0004     , 0,],
    [0.0001, 0.0000     , 0,],
    [0.0003, 0.0001     , 0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,],
    [0,         0,         0,]])

def minMaxScale(arr, max_v=1, min_v=0):
    arr_std = (arr - np.min(arr)) / ( np.max(arr) - np.min(arr))
    return arr_std * (max_v - min_v) + min_v
from colormath import spectral_constants
# Spectrum to CIE
# from 340 nm to 830 by 10 nm steps
class SpectrumColor:
    MIN_NM = 340
    MAX_NM = 830
    REF_ILLUM = spectral_constants.REF_ILLUM_TABLE['d50']
    DENOM =  np.sum(REF_ILLUM
                    * spectral_constants.STDOBSERV_Y10)
    SPECTRUM2XYZ = np.array([REF_ILLUM * spectral_constants.STDOBSERV_X10/DENOM,
                             REF_ILLUM * spectral_constants.STDOBSERV_Y10/DENOM,
                             REF_ILLUM * spectral_constants.STDOBSERV_Z10/DENOM]
                            ).transpose()
    STEP_NM = 10
    @classmethod
    def get_coef(cls, from_nm, to_nm):
        assert cls.MIN_NM <= from_nm <= from_nm
        assert from_nm <= to_nm <= cls.MAX_NM
        to_nm = (to_nm - cls.MIN_NM ) // cls.STEP_NM
        from_nm = (from_nm - cls.MIN_NM) // cls.STEP_NM
        return np.mean(cls.SPECTRUM2XYZ[from_nm:to_nm+1], axis=0)

    @classmethod
    def create_filter(cls, min_wl=380, max_wl=780, step_wl=25):
        assert (max_wl - min_wl) % step_wl == 0
        assert step_wl >= 5, "The approximation function from spectrum to cie has sensitivity 5 nm"
        filter = np.array([SpectrumColor.get_coef(i, i+25) for i in range(min_wl, max_wl, step_wl)])
        return filter

def wavelength2rgb(in_image, min_wl=380, max_wl=780, step_wl=25):
    assert np.max(in_image) <= 1.0 and np.min(in_image) >= 0.0
    from skimage.color import xyz2rgb
    f = SpectrumColor.create_filter(min_wl=min_wl,
                                    max_wl=max_wl,
                                    step_wl=step_wl)
    o = np.matmul(in_image, f)
    return xyz2rgb(minMaxScale(o))
