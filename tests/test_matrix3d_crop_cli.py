import numpy as np
import scipy.io
import tempfile
import os
from stable.dataset.preparation.matrix_3d import crop


def test_matrix3d_mat_crop():
    matfile = tempfile.NamedTemporaryFile(suffix=".mat")
    data = np.random.randn(100, 100, 10)
    scipy.io.savemat(f"{matfile.name}", {"data": data})
    crop(f"{matfile.name}", rectangle=(10, 12, 40, 50))
    # this should create a file
    assert os.path.isfile(f"{matfile.name[:-4]}-crop.mat")
    cropped = scipy.io.loadmat(f"{matfile.name[:-4]}-crop.mat")["data"]
    assert cropped.shape == (30, 38, 10)


def test_matrix3d_npy_flat_crop():
    matfile = tempfile.NamedTemporaryFile(suffix=".npy")
    data = np.random.randn(100, 100, 10)
    np.save(matfile.name, data)
    crop(matfile.name, rectangle=(10, 12, 40, 50))
    assert os.path.isfile(f"{matfile.name[:-4]}-crop.npy")
    cropped = np.load(f"{matfile.name[:-4]}-crop.npy")
    assert cropped.shape == (30, 38, 10)
