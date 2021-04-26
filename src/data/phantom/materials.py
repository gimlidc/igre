# TODO: Add description and names
import numpy as np
import os
import glob
import logging
from enum import Enum, auto
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import scipy.io as sio

ALMA_NAME = 'alma'
FIRENZE_NAME = 'firenze'
# TODO: Solve more elegant way
MATERIALS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), './materials.npz')


class Material(Enum):
    ALMA = auto()
    FIRENZE = auto()


def load_alma():
    return np.load(MATERIALS_FILE)[ALMA_NAME]


def load_firenze():
    return np.load(MATERIALS_FILE)[FIRENZE_NAME]


def extract_pigments(vis_wildcard,
                     nir_wildcard,
                     mask_path,
                     mask_name,
                     padding=6,
                     max_value=255.0):
    vis = glob.glob(vis_wildcard)
    nir = glob.glob(nir_wildcard)
    mask = sio.loadmat(mask_path)[mask_name]

    layers = [*vis, *nir]
    x, y = plt.imread(layers[0]).shape
    z = len(layers)

    raw_img = np.zeros((x, y, z))
    for i, l in enumerate(layers):
        raw_img[:, :, i] = plt.imread(l)

    raw_img /= max_value

    z_size = raw_img.shape[-1]
    pigments = []

    for mask_id in range(1, np.max(mask) + 1):
        mask_multid = np.repeat(
            np.expand_dims(
                (mask == mask_id),
                axis=-1),
            repeats=z_size,
            axis=-1
        )

        x_size = np.max(np.sum(mask_multid[:, :, 1], axis=0))
        y_size = np.max(np.sum(mask_multid[:, :, 1], axis=1))
        square = np.reshape(raw_img[mask_multid], (x_size, y_size, z_size))

        no_draw = square[:, :int(square.shape[1] / 2) - padding, :]
        draw = square[:, int(square.shape[1] / 2) + padding:, :]

        pigments.append({'no_draw': no_draw,
                         'draw': draw})

    return pigments


def get_pigment_dist(pigment, pad=1):
    # MARK: returns distribution (pad * 2 + 1, pad * 2 + 1, z_size)
    assert len(pigment.shape) == 3, f"Expected 3 dimensional image, but get {len(pigment.shape)}"
    assert pad >= 1
    z_size = pigment.shape[-1]
    target_shape = (pad * 2 + 1, pad * 2 + 1, z_size)
    # TODO: Do not waste memory, size known
    samples = []
    for i in range(pad, pigment.shape[0] - pad):
        for j in range(pad, pigment.shape[1] - pad):
            sample = pigment[i - pad:i + pad + 1, j - pad:j + pad + 1, :]
            # Normalize from 0 to 1
            assert sample.shape == target_shape
            # Flatten as feature vector
            samples.append(np.ravel(sample))
    samples = np.array(samples)
    covariance_matrix = np.cov(samples, rowvar=False)
    mean_vector = np.mean(samples, axis=0)
    assert mean_vector.shape == np.product(target_shape)
    assert covariance_matrix.shape == (np.product(target_shape), np.product(target_shape))
    # TODO: Is it okey to use 0 - 255
    # TODO: Normality test ??
    # TODO: pigment distribution class ???
    logging.warning(f"Pigments samples: {samples.shape} ")
    return multivariate_normal(mean=mean_vector,
                               cov=covariance_matrix,
                               allow_singular=False)


def create_cnn_phantom(pigments_distribution,
                       used_materials,
                       number_samples,
                       underdrawing_coverage,
                       sample_shape,
                       seed=42,
                       correct_outliers=True):
    no_draws = []
    draws = []
    # If used_materials is number generate pigments randomly
    if not isinstance(used_materials, list):
        used_materials = [int(i) for i in np.random.choice(len(pigments_distribution), used_materials, replace=False)]

    samples_per_pigment = number_samples // len(used_materials)
    underdrawings_per_pigment = int(np.floor(samples_per_pigment * underdrawing_coverage))
    clear_per_pigment = samples_per_pigment - underdrawings_per_pigment

    for i in used_materials:
        clear_dist = pigments_distribution[i]['no_draw']
        underdrawing_dist = pigments_distribution[i]['draw']
        clear_sample = clear_dist.rvs(clear_per_pigment, random_state=seed)
        clear_sample = np.reshape(clear_sample, (clear_per_pigment, *sample_shape))
        underdrawing_sample = underdrawing_dist.rvs(underdrawings_per_pigment, random_state=seed)
        underdrawing_sample = np.reshape(underdrawing_sample, (underdrawings_per_pigment, *sample_shape))
        if correct_outliers:
            logging.warning(f"Some of pigments {i} samples are out of bounds.")
            clear_sample[clear_sample < 0] = 0
            clear_sample[clear_sample > 1] = 1
            underdrawing_sample[underdrawing_sample < 0] = 0
            underdrawing_sample[underdrawing_sample > 1] = 1
        draws.append(underdrawing_sample)
        no_draws.append(clear_sample)
    return {'draw': np.array(draws), 'no_draw': np.array(no_draws)}
