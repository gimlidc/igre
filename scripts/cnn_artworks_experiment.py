import click
import os
import json
import numpy as np
import tensorflow as tf
from src.models.ig_cnn import get_cnn_dataset, ig_cnn_model
from glob import glob
from tensorboard.plugins.hparams import api as hp
from src.tftools.callbacks.t2_callbacks import InformationGainCallback, ImagesCallback


@click.command()
@click.option('-c', '--config_file', required=True, help='Config, for further details', type=click.File('r'))
def cnn_artworks(config_file):
    config = json.load(config_file)
    samples = glob(os.path.join(config['artwork_root'], config['artwork_file_wildcard']))
    pad = (config['phantom_size'] - 1) // 2

    for file in samples:
        directory = os.path.basename(os.path.dirname(file))
        sample_name = os.path.basename(file)[:-4]
        logdir = os.path.join(config['log_root'], f"{directory}_{sample_name}")
        # Input phantoms
        inputs, outputs, pad_output, nopad_input = get_cnn_dataset(file,
                                                                   config['phantom_size'],
                                                                   samples_overlay=False)
        model_cnn = ig_cnn_model(config['model_def'])
        model_cnn.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['mean_squared_error'])
        model_cnn.summary()
        # phantoms split
        training_set_size = 2500
        perm = np.random.permutation(inputs.shape[0])
        ins = inputs[perm[:training_set_size], :]
        outs = outputs[perm[:training_set_size], :]

        # TODO: Check log dir existence

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                     InformationGainCallback(input_image=nopad_input[np.newaxis, :],
                                             target_image=pad_output,
                                             logdir=logdir),
                     ImagesCallback(input_image=nopad_input[np.newaxis, :],
                                    target_image=pad_output,
                                    pad=pad,
                                    logdir=logdir),
                     hp.KerasCallback(logdir, {**config,
                                               "sample_name": file}),
                     # NOTE: This callback has to be last
                     tf.keras.callbacks.EarlyStopping(patience=6, min_delta=10 ** -5),
                     ]

        history = model_cnn.fit(ins,
                                outs,
                                epochs=100,
                                validation_split=0.2,
                                verbose=1,
                                callbacks=callbacks
                                )


if __name__ == '__main__':
    cnn_artworks()
