import numpy as np

from .materials import load_alma, load_firenze, Material


def create_phantom(pattern, material: Material):
    if material is Material.ALMA:
        mat = load_alma()
    elif material is Material.FIRENZE:
        mat = load_firenze()
    else:
        raise Exception(f'Unknown material: {material}')

    phantom_means = mat[pattern, :, 1]
    phantom_stds = np.square(mat[pattern, :, 1])

    phantom_dirt = (
            np.random.normal(size=phantom_stds.shape)
            *
            phantom_stds
    )
    return phantom_means + phantom_dirt
