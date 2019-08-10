import sys
import numpy as np
import tensorflow as tf
from time import time
from termcolor import colored
from tftools.idx2pixel_layer import Idx2PixelLayer
from tftools.shift_metric import ShiftMetrics
import matplotlib.pyplot as plt


def __train_networks(inputs,
                     outputs,
                     reg_layer_data,
                     optimizer,
                     layers=None,
                     train_set_size=50000,
                     batch_size=256,
                     epochs=100,
                     ):
    """
    This method builds ANN  with all layers - some layers for registration other layers for information gain computation
    and processes the training.

    :param inputs:
        input coordinates (will be randomized)
    :param outputs:
        output image pixel intensity values
    :param reg_layer_data:
        first layer data - transformation of coords to pixel intensity value
    :param layers:
        not used now - this will define IG layers in the future
    :param train_set_size:
        number of pixels used for training, default is 50k
    :param batch_size:
        number of pixels computed in one batch
    :param epochs:
        number of learning epochs
    :return:
        trained model and training history
    """
    print('Selecting', train_set_size, 'samples randomly for use by algorithm.')
    perm = np.random.permutation(inputs.shape[0])
    indexes = inputs[perm[:train_set_size], :]
    outputs = outputs[perm[:train_set_size], :]

    # define model
    print('Adding input layer, width =', indexes.shape[1])
    input_layer = tf.keras.layers.Input(shape=(indexes.shape[1],),
                                        dtype=tf.float32, name='InputLayer')
    layer = Idx2PixelLayer(visible=reg_layer_data, name='Idx2PixelLayer')(input_layer)

    # TODO: Add InformationGain layers here when necessary
    # for layer_idx in range(len(layers)):
    #     print('Adding dense layer, width =', layers[layer_idx])
    #     layer = tf.keras.layers.Dense(layers[layer_idx],
    #                                   activation='sigmoid', name='Dense' + str(layer_idx))(layer)
    print('Adding ReLU output layer, width =', outputs.shape[1])
    output_layer = tf.keras.layers.ReLU(max_value=1, name='Output', trainable=False)(layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # compile model
    start_time = time()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error']
                  )
    elapsed_time = time() - start_time
    print("Compiling model took {:.4f}'s.".format(elapsed_time))

    # train model

    start_time = time()
    # doing thing iteratively, hardcoded version
    aa = [50, 50, 50]
    for blur_stages in range(outputs.shape[1]):
        shift_metric = ShiftMetrics()
        callbacks = [shift_metric]
        history = model.fit(indexes,
                            outputs[:, -(blur_stages+1)],
                            epochs=aa[blur_stages],
                            validation_split=0.2,
                            verbose=1,
                            callbacks=callbacks,
                            batch_size=batch_size
                            )

        plt.plot(np.array(shift_metric.bias_history)[:, 0])
        plt.show()

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    gain = abs(outputs - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                           outputs.shape[1])
    information_gain_max = gain.flatten().max()
    print('Gain: {:1.4e}'.format(information_gain_max))

    return model, history, shift_metric.bias_history


def __information_gain(coords,
                       target,
                       visible,
                       optimizer,
                       layers=None,
                       train_set_size: int = 25000,
                       batch_size=256,
                       epochs=100,
                       ):

    if coords.shape[0] != target.shape[0] * visible.shape[1]:
        sys.exit("Error: dimension mismatch between 'target' and 'visible'")

    if len(target.shape) == 2:
        target = target.reshape((target.shape[0], target.shape[1], 1))

    if train_set_size is None:
        # use the whole image for training, evaluation and test
        train_set_size = visible.shape[0] * visible.shape[1]

    outputs = target.reshape((target.shape[0] * target.shape[1], target.shape[2]))

    # train ANN
    model, history, bias_history = __train_networks(coords,
                                                    outputs,
                                                    reg_layer_data=visible,
                                                    optimizer=optimizer,
                                                    layers=layers,
                                                    train_set_size=train_set_size,
                                                    batch_size=batch_size,
                                                    epochs=epochs
                                                    )

    # show output of the first two layers
    extrapolation = model.predict(coords, batch_size=batch_size)
    extrapolation = extrapolation.reshape(target.shape[0], target.shape[1], 1)

    ig = target - extrapolation

    return ig, extrapolation, model,bias_history


def run(inputs,
        outputs,
        visible,
        optimizer,
        layers=None,
        batch_size=256,
        epochs=100):
    ig, extrapolation, model, bias_history = __information_gain(inputs,
                                                                outputs,
                                                                visible=visible,
                                                                optimizer=optimizer,
                                                                layers=layers,
                                                                batch_size=batch_size,
                                                                epochs=epochs,
                                                                )
    # print model summary to stdout
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    bias = layer_dict['Idx2PixelLayer'].get_weights()
    print("Shift detected: " + colored(str(bias[0]), "green"))
    return bias, bias_history
