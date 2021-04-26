import matplotlib.pyplot as plt
import numpy as np
from src.visualization.color_transformation import wavelength2rgb


def _draw(pigments, image2color, cmap='viridis'):
    fig, axs = plt.subplots(6,
                            int(len(pigments) / 6),
                            figsize=(20, 20))
    for i, pigment in enumerate(pigments):
        no_draw, draw = pigment['no_draw'], pigment['draw']
        merged_pigment = np.concatenate((no_draw,
                                         np.ones((draw.shape[0], 20, draw.shape[-1])),
                                         draw),
                                        axis=1)
        axs[i % axs.shape[0], i // axs.shape[0]].imshow(
            image2color(merged_pigment), cmap=cmap
        )
    return fig


def draw_pigments_vis(pigments):
    def _vis(img):
        return wavelength2rgb(img[:, :, :16])

    return _draw(pigments, _vis)


def draw_pigments_nir(pigments, channel=25):
    def _nir(img):
        return img[:, :, channel]

    return _draw(pigments, _nir, cmap='gray')


def _compare_distribution_origin(pigments, pigments_dist, image2color, cmap='viridis'):
    fig, axs = plt.subplots(30, 2, figsize=(10, 60))
    for id, pig in enumerate(pigments):
        (ax1, ax2) = axs[id]
        im = []
        for i in range(50):
            l = []
            for s in pigments_dist[id]['no_draw'].rvs(20):
                l.append(np.reshape(s, (3, 3, 32)))
            im.append(l)
        im = np.reshape(np.array(im), (150, 60, 32))
        im[im < 0] = 0
        im[im > 1] = 1.0
        generated = image2color(im)[:120, :50]
        origin = image2color(pig['no_draw'])[:120, :50]
        mean = image2color(np.ones((120, 50, 32)) * np.mean(pig['no_draw'], axis=(0, 1)))
        comp1 = np.concatenate((generated, origin, mean), axis=1)
        im = []
        for i in range(50):
            l = []
            for s in pigments_dist[id]['draw'].rvs(20):
                l.append(np.reshape(s, (3, 3, 32)))
            im.append(l)
        im = np.reshape(np.array(im), (150, 60, 32))
        im[im < 0] = 0
        im[im > 1] = 1.0
        generated = image2color(im)[:120, :50]
        origin = image2color(pig['draw'])[:120, :50]
        mean = image2color(np.ones((120, 50, 32)) * np.mean(pig['draw'], axis=(0, 1)))
        comp2 = np.concatenate((generated, origin, mean), axis=1)

        ax1.imshow(comp1, cmap=cmap)
        ax2.imshow(comp2, cmap=cmap)
    return fig


def compare_distribution_origin_vis(pigments, pigments_dist):
    def _vis(img):
        return wavelength2rgb(img[:, :, :16])

    return _compare_distribution_origin(pigments, pigments_dist, _vis)


def compare_distribution_origin_nir(pigments, pigments_dist, channel=25, cmap='gray'):
    def _nir(img):
        return img[:, :, channel]

    return _compare_distribution_origin(pigments, pigments_dist, _nir, cmap)


def show_cnn_phantom(cnn_phantoms, width2height_ration=20):
    rows = []
    no_draws, draws = (cnn_phantoms['no_draw'], cnn_phantoms['draw'])
    assert no_draws.shape[0] == draws.shape[0], f"Number of used pigments differs"
    assert no_draws.shape[-2] == draws.shape[-2], f"Surroundings of pixel differs"
    assert no_draws.shape[-3] == draws.shape[-3], f"Surroundings of pixel differs"
    height = int(np.floor(np.sqrt(no_draws.shape[1] / width2height_ration)))
    width_no_draw = int(np.floor(no_draws.shape[1] / height))
    width_draw = int(np.floor(draws.shape[1] / height))
    no_draws = no_draws[:, :width_no_draw * height, ...]
    draws = draws[:, :width_draw * height, ...]
    # Multiply by padding
    width_no_draw *= no_draws.shape[-2]
    width_draw *= draws.shape[-2]
    height *= draws.shape[-3]
    for pigment_id in range(no_draws.shape[0]):
        rows.append(np.concatenate((
            np.reshape(no_draws[pigment_id], (height, width_no_draw, no_draws.shape[-1])),
            np.reshape(draws[pigment_id], (height, width_draw, draws.shape[-1]))
        ), axis=1))
    return wavelength2rgb(np.concatenate(rows, axis=0)[:, :, :16])

