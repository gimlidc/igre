import os
import json
import numpy as np
import tensorflow as tf
from glob import glob
from src.models.ig_cnn import ig_cnn_model
from src.tftools.callbacks.t2_callbacks import InformationGainCallback
from tensorboard.plugins.hparams import api as hp

with open('0.json', 'r') as file:
    config = json.load(file)
    
phantom_wildcard = os.path.join(config['phantom_root'], config['phantom_file_wildcard'])
samples = glob(phantom_wildcard)


for file in samples:
    pad = 1
    predicted_channel = 26
    phantom = np.load(file)
    train_X, train_y = phantom['train'][..., :16], phantom['train'][..., pad, pad, predicted_channel]
    valid_X, valid_y = phantom['valid'][..., :16], phantom['valid'][..., pad, pad, predicted_channel]

    model_cnn = ig_cnn_model(config['model_def'], input_size=(config['phantom_size'], config['phantom_size']))
    model_cnn.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])

    # TODO: Remove the sample number
    _log_dir = os.path.join(config['log_root'],
                            config['log_name'],
                            file[-17])

    model_cnn_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=_log_dir),
                           hp.KerasCallback(_log_dir, {**config, "phantom_sample": int(file[-17])}),
                           InformationGainCallback(valid_X, valid_y, _log_dir, 'valid'),
                           InformationGainCallback(train_X, train_y, _log_dir, 'train'),
                           # NOTE: This callback has to be last
                           tf.keras.callbacks.EarlyStopping(patience=6, min_delta=10 ** -5), ]

    history_cnn = model_cnn.fit(x=train_X,
                                y=train_y,
                                validation_data=(valid_X, valid_y),
                                epochs=100,
                                verbose=1,
                                callbacks=model_cnn_callbacks)
    model_cnn.summary()
