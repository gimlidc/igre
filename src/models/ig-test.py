import scipy.io
import numpy as np
from src.models.ig import information_gain

if __name__ == "__main__":
    inputs = scipy.io.loadmat('/Users/gimli/Qsync/MATLAB/vlasic-dataset.mat')
    input_data = inputs['data']
    visible = np.asarray(input_data[:, :, 0:15]).astype(np.float64) / 255
    target = np.asarray(input_data[:, :, 26]).astype(np.float64) / 255
    ig, approx, net = information_gain(visible, target)

    print("Approx done.")
