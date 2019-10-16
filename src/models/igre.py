import sys
import tensorflow as tf
from time import time
from src.tftools.idx2pixel_layer import Idx2PixelLayer, reset_visible
from src.tftools.shift_metric import ShiftMetrics
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from src.logging.verbose import Verbose
from src.tftools.optimizer_builder import build_optimizer
from src.config.tools import get_config, get_or_default
import numpy as np
from termcolor import colored
from src.data.ann.input_preprocessor import training_batch_selection, blur_preprocessing
from src.tftools.optimizer_builder import build_refining_optimizer

MODEL_FILE = "best_model.tmp.h5"


def __train_networks(inputs,
                     outputs,
                     reg_layer_data,
                     optimizer,
                     refiner,
                     layers=None,
                     train_set_size=50000,
                     batch_size=256,
                     stages={
                         "type": "polish",
                         "epochs": 50
                     }):
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
    Verbose.print('Selecting ' + str(train_set_size) + ' samples randomly for use by algorithm.')

    selection = training_batch_selection(train_set_size, reg_layer_data.shape)
    indexes = inputs[selection, :]

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
    # TODO: better names for stages
    tf.keras.backend.get_session().run(tf.global_variables_initializer())

    for stage in stages:
        if stage['type'] == 'blur':
            output = blur_preprocessing(outputs, reg_layer_data.shape, stage['params'])
        elif stage['type'] == 'refine':
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])
            output = outputs
        elif stage['type'] == 'polish':
            output = outputs

        reset_visible(output)
        output = output[selection, :]
        shift_metric = ShiftMetrics()
        # mcp_save = ModelCheckpoint(MODEL_FILE,
        #                            save_best_only=True, monitor='val_loss', mode='min')
        # lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=100, verbose=0, mode='auto',
        #                                  min_delta=0.0001, cooldown=0, min_lr=0)
        callbacks = [shift_metric] #mcp_save]# lr_reduction]
        history = model.fit(indexes,
                            output,
                            epochs=stage['epochs'],
                            validation_split=0.2,
                            verbose=1,
                            callbacks=callbacks,
                            batch_size=batch_size
                            )

        bias_history = [x[0] for x in shift_metric.bias_history]  # extract the shift
        bias_history = np.array(bias_history)
        Verbose.plot(bias_history)  # plot the shift (c coeff)

        # bias_history = [x[2][0:2] for x in shift_metric.bias_history]  # extract the a
        # bias_history = np.array(bias_history) / utils.shift_multi
        # Verbose.plot(bias_history)  # plot the  (d coeff)
        #
        # bias_history = [x[2][2:] for x in shift_metric.bias_history]  # extract the b
        # bias_history = np.array(bias_history) / utils.shift_multi
        # Verbose.plot(bias_history)  # plot the  (e coeff)

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    gain = abs(outputs[selection, :] - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                                         outputs.shape[1])
    information_gain_max = gain.flatten().max()
    print('Gain: {:1.4e}'.format(information_gain_max))

    return model, history, shift_metric.bias_history


def __information_gain(coords,
                       target,
                       visible,
                       optimizer,
                       refiner,
                       layers=None,
                       train_set_size: int = 25000,
                       batch_size=256,
                       stages={
                           "type": "polish",
                           "epochs": 50
                       }):
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
                                                    refiner=refiner,
                                                    layers=layers,
                                                    train_set_size=train_set_size,
                                                    batch_size=batch_size,
                                                    stages=stages
                                                    )
    # print(model.get_weights())
    # model.load_weights(MODEL_FILE)
    # print(model.get_weights())

    # show output of the first two layers
    extrapolation = model.predict(coords, batch_size=batch_size)
    extrapolation = extrapolation.reshape(target.shape[0], target.shape[1], 1)

    ig = target - extrapolation

    return ig, extrapolation, model, bias_history


def run(inputs,
        outputs,
        visible):
    config = get_config()
    ig, extrapolation, model, bias_history = \
        __information_gain(inputs,
                           outputs,
                           visible=visible,
                           optimizer=build_optimizer(config["train"]["optimizer"], config["train"]["batch_size"]),
                           refiner=build_refining_optimizer(config["train"]["refiner"]),
                           layers=get_or_default("layers", 1),
                           batch_size=config["train"]["batch_size"],
                           stages=config["train"]["stages"]
                           )
    # print model summary to stdout
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    bias = layer_dict['Idx2PixelLayer'].get_weights()
    # Verbose.print("linear coeffs (a): " + colored(str(bias[1][0:2]), "green"), Verbose.always)
    # Verbose.print("linear coeffs (b): " + colored(str(bias[1][2:]), "green"), Verbose.always)
    Verbose.print("Shift detected (c): " + colored(str(bias[0]), "green"), Verbose.always)
    # Verbose.print("linear coeffs (d): " + colored(str(bias[2][0:2] / utils.shift_multi), "green"), Verbose.always)
    # Verbose.print("linear coeffs (e): " + colored(str(bias[2][2:] / utils.shift_multi), "green"), Verbose.always)

    return bias, bias_history
