import click
import os
import numpy as np
import tensorflow as tf
from .ig_cnn import get_cnn_dataset, ig_cnn_model
from tensorboard.plugins.hparams import api as hp
from src.tftools.callbacks.t2_callbacks import InformationGainCallback, ImagesCallback


@click.command()
@click.option('--image_path',
              default='/mnt/c/Users/Tomas Karella/Desktop/igre/data/raw/still_life/sample_22.mat',
              show_default=True,
              help='Path to matlab multichannel image')
@click.option('--model_def',
              default='C-25-3-sigmoid-valid,C-25-1-sigmoid-same',
              show_default=True,
              help='Model defined by string, each layer separated by commas')
@click.option('--input_size',
              default='3',
              show_default=True,
              help='The visible spectrum is separated into squares with this size',
              type=int)
@click.option('--log_name',
              default='C3',
              show_default=True,
              help='Name of the log dir')
@click.option('--log_root',
              default='/mnt/c/Users/Tomas Karella/Desktop/igre/logs',
              show_default=True,
              help='Root dir of logs files')
def ig_cnn(image_path, model_def, input_size, log_name, log_root):

    hparams = {
        'image_path': image_path,
        'model_def': model_def,
        'input_size': input_size,
        'log_name': log_name
    }

    # Parameters
    logdir = os.path.join(log_root, hparams['log_name'])
    assert not(os.path.exists(logdir)), f"The log dir of given name exists: {logdir}"
    # Input phantoms
    inputs, outputs, pad_output, nopad_input = get_cnn_dataset(hparams['image_path'],
                                                               input_size=hparams['input_size'])
    # Model Definition
    model = ig_cnn_model(hparams['model_def'])
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # phantoms split
    training_set_size = 2500
    perm = np.random.permutation(inputs.shape[0])
    ins = inputs[perm[:training_set_size], :]
    outs = outputs[perm[:training_set_size], :]

    # TODO: Check log dir existence

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 InformationGainCallback(input_image=nopad_input,
                                         target_image=pad_output,
                                         logdir=logdir),
                 ImagesCallback(input_image=nopad_input,
                                target_image=pad_output,
                                logdir=logdir),
                 hp.KerasCallback(logdir, hparams),
                 # NOTE: This callback has to be last
                 tf.keras.callbacks.EarlyStopping(patience=6, min_delta=10 ** -5),
                 ]

    history = model.fit(ins,
                        outs,
                        epochs=100,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks
                        )


if __name__ == '__main__':
    ig_cnn()

