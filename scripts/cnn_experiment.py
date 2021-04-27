import os
import numpy as np
import tensorflow as tf
from src.models.ig_cnn import ig_cnn_model
from src.tftools.callbacks.t2_callbacks import InformationGainCallback
from tensorboard.plugins.hparams import api as hp
from glob import glob

PATH = "../data/phantoms/phantom_5_5/training/*npz"
files = glob(PATH)

for file in files:
    pad = 1
    predicted_channel = 26
    phantom = np.load(file)
    train_X, train_y = phantom['train'][..., :16], phantom['train'][..., pad, pad, predicted_channel]
    valid_X, valid_y = phantom['valid'][..., :16], phantom['valid'][..., pad, pad, predicted_channel]
    train_y = train_y[..., np.newaxis, np.newaxis, np.newaxis]
    valid_y = valid_y[..., np.newaxis, np.newaxis, np.newaxis]

    hparams = {
        'model_def': 'C-25-5-sigmoid-valid,C-25-1-sigmoid-same',
        'log_name': f'{os.path.split(file)[-1][:-4]}',
        'log_root': '../logs/'
    }
    dataset_params = {
        'input_file': os.path.split(file)[-1],
        'kernel_size': 5
    }

    model_cnn = ig_cnn_model(hparams['model_def'])
    model_cnn.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])

    _log_dir = os.path.join(hparams['log_root'],
                            hparams['log_name'],
                            'cnn')

    model_cnn_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=_log_dir),
                           hp.KerasCallback(_log_dir, {**hparams, **dataset_params}),
                           InformationGainCallback(valid_X, valid_y, _log_dir, 'valid_gain'),
                           InformationGainCallback(train_X, train_y, _log_dir, 'train_gain'),
                           # NOTE: This callback has to be last
                           tf.keras.callbacks.EarlyStopping(patience=6, min_delta=10 ** -5), ]
    history_cnn = model_cnn.fit(x=train_X,
                                y=train_y,
                                validation_data=(valid_X, valid_y),
                                epochs=100,
                                verbose=1,
                                callbacks=model_cnn_callbacks)
