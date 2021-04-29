import click
import json
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.data.phantom.materials import extract_pigments, create_cnn_phantom, get_pigment_dist
from src.data.phantom.tools import show_cnn_phantom


def phantom_validation_train_split(phantom, validation_size=0.2, dtype=np.float32):
    def _split(sample):
        val_size = int(np.ceil(sample.shape[1] * validation_size))
        perm = np.random.permutation(sample.shape[1])
        return sample[:, perm[:val_size]], sample[:, perm[val_size:]]
    valid_draw, train_draw = _split(phantom['draw'])
    valid_no_draw, train_no_draw = _split(phantom['no_draw'])
    valid = np.concatenate((valid_draw, valid_no_draw), axis=1).astype(dtype)
    train = np.concatenate((train_draw, train_no_draw), axis=1).astype(dtype)
    # Merge pigments
    valid = np.reshape(valid, (np.prod(valid.shape[:2]), *valid.shape[2:]))
    train = np.reshape(train, (np.prod(train.shape[:2]), *train.shape[2:]))
    # shuffle
    np.random.shuffle(valid)
    np.random.shuffle(train)
    return train, valid


@click.command()
@click.option('-c', '--config', required=True, help='Config, for further details', type=click.File('rb'))
@click.option('-o', '--output_path', default='.', help='Name of output path', type=str)
def generate_cnn_phantom(config, output_path):
    args = json.load(config)
    assert 'pad' in args.keys()
    assert 'extractor_args' in args.keys()
    assert 'phantom_args' in args.keys()
    assert os.path.isdir(output_path)
    output_name = args['name']
    preview_dir = 'previews'
    training_data_dir = 'training'
    raw_data_dir = 'raw'
    # Prepare file structure
    os.makedirs(os.path.join(output_path, output_name), exist_ok=True)
    output_path = os.path.join(output_path, output_name)
    os.makedirs(os.path.join(output_path, preview_dir), exist_ok=True)
    os.makedirs(os.path.join(output_path, training_data_dir), exist_ok=True)
    os.makedirs(os.path.join(output_path, raw_data_dir), exist_ok=True)
    pad = args['pad']
    validation_split = args['validation_split']
    # Separate pigments
    pigments = extract_pigments(**args['extractor_args'])
    # Calc distribution
    pigments_dist = []
    for pigment in pigments:
        no_draw, draw = pigment['no_draw'], pigment['draw']
        pigments_dist.append({"draw": get_pigment_dist(draw, pad),
                              "no_draw": get_pigment_dist(no_draw, pad)})
    # Create phantoms
    for idx in range(args['phantom_counts']):
        # Create phantom
        cnn_phantoms = create_cnn_phantom(pigments_dist,
                                          sample_shape=(2*pad + 1, 2*pad + 1, 32),
                                          **args['phantom_args'])
        # Include config
        cnn_phantoms['config'] = args
        logging.warning(f"Saving {idx}")
        # Original pixels
        np.savez_compressed(os.path.join(output_path, raw_data_dir, f"{idx}_{output_name}_raw"),
                            draw=cnn_phantoms['draw'],
                            no_draw=cnn_phantoms['no_draw'])
        # Phantom sample
        img = show_cnn_phantom(cnn_phantoms=cnn_phantoms)
        plt.imsave(os.path.join(output_path, preview_dir, f"{idx}_{output_name}.png"), img)
        train, valid = phantom_validation_train_split(cnn_phantoms, validation_split)
        np.savez_compressed(os.path.join(output_path, training_data_dir,  f"{idx}_{output_name}"),
                            train=train,
                            valid=valid)


if __name__ == "__main__":
    generate_cnn_phantom()
