import sys
import tensorflow as tf
from time import time
from src.tftools.shift_layer import ShiftLayer
from src.tftools.scale_layer import ScaleLayer
from src.tftools.rotation_layer import RotationLayer
from src.tftools.shear_layer import ShearLayer
from src.tftools.idx2pixel_layer import Idx2PixelLayer, reset_visible
from src.tftools.transform_metric import ShiftMetrics, ScaleMetrics, RotationMetrics
from tensorflow.keras.callbacks import ModelCheckpoint
from src.logging.verbose import Verbose
from src.tftools.optimizer_builder import build_optimizer
from src.config.tools import get_config, get_or_default
import numpy as np
from termcolor import colored
from src.data.ann.input_preprocessor import training_batch_selection, blur_preprocessing
from src.tftools.optimizer_builder import build_refining_optimizer

MODEL_FILE = "best_model.tmp.h5"
tf.keras.utils.Progbar(target=None, width=100)

def __train_networks(inputs,
                     outputs,
                     reg_layer_data,
                     optimizer,
                     refiner,
                     config,
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
    shift_layer = ShiftLayer(name='ShiftLayer')(input_layer)
    scale_layer = ScaleLayer(name='ScaleLayer')(shift_layer)
    rotation_layer = RotationLayer(name='RotationLayer')(scale_layer)
    # shear_layer = ShearLayer(name='ShearLayer')(rotation_layer)
    layer = Idx2PixelLayer(visible=reg_layer_data, name='Idx2PixelLayer')(rotation_layer)

    for layer_idx in range(len(layers)):
        print('Adding dense layer, width =', layers[layer_idx])
        layer = tf.keras.layers.Dense(layers[layer_idx],
                                      activation='sigmoid', name='Dense' + str(layer_idx))(layer)
    print('Adding ReLU output layer, width =', outputs.shape[1])
    output_layer = tf.keras.layers.Dense(outputs.shape[1], name='Output', activation='sigmoid')(layer)
    # output_layer = tf.keras.layers.ReLU(max_value=1, name='Output', trainable=False)(layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # compile model
    start_time = time()
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error']
                  )
    # Set initial transformation as identity
    elapsed_time = time() - start_time
    print("Compiling model took {:.4f}'s.".format(elapsed_time))

    # train model
    start_time = time()
    # TODO: better names for stages
    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
    model.layers[1].set_weights([np.array([0, 0])])  # shift
    model.layers[2].set_weights([np.array([0, 0])])  # scale
    model.layers[3].set_weights([np.array([0])])     # rotation
    # model.layers[4].set_weights([np.array([0])])  # shear_x

    bias_history = {
        "shift_x": [],
        "shift_y": [],
        "rotation": [],
        "scale_x": [],
        "scale_y": []
    }

    for stage in stages:
        if stage['type'] == 'blur':
            output = blur_preprocessing(outputs, reg_layer_data.shape, stage['params'])
            __set_train_registration(model, True)
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'refine':
            output = outputs
            __set_train_registration(model, True, target="shift")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])
        elif stage['type'] == 'polish':
            output = outputs
            __set_train_registration(model, True, target="all")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage["type"] == "mutual_init":
            __set_train_registration(model, False)
            output = blur_preprocessing(outputs, reg_layer_data.shape, stage['blur'])
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

        reset_visible(output)
        output = output[selection, :]
        shift_metric = ShiftMetrics()
        scale_metric = ScaleMetrics()
        rotation_metric = RotationMetrics()

        history = model.fit(indexes,
                            output,
                            epochs=stage['epochs'],
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[shift_metric, scale_metric, rotation_metric],
                            batch_size=batch_size
                            )
        if "print_bash_history" in config:
            bias_history["shift_x"].extend([float(step[0][0]) for step in shift_metric.bias_history])
            bias_history["shift_y"].extend([float(step[0][1]) for step in shift_metric.bias_history])
            bias_history["rotation"].extend([float(step[0][0]) for step in rotation_metric.bias_history])
            bias_history["scale_x"].extend([float(step[0][0]) for step in scale_metric.bias_history])
            bias_history["scale_y"].extend([float(step[0][1]) for step in scale_metric.bias_history])

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    gain = abs(outputs[selection, :] - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                                         outputs.shape[1])
    information_gain_max = gain.flatten().max()
    print('Gain: {:1.4e}'.format(information_gain_max))

    Verbose.plot([float(step[0][0]) for step in shift_metric.bias_history])
    Verbose.plot([float(step[0][1]) for step in shift_metric.bias_history])

    return model, history, bias_history


def __set_train_registration(model, value, target="registration"):
    """
    For various stages of training registration should not be trainable (we are looking for base mutual setup).
    This method allows enabling/disabling of trainability of first three layers of ANN.
    :param model: ANN
    :param value: boolean
    """
    if target == "registration":
        model.layers[1].trainable = value
        model.layers[2].trainable = value
        model.layers[3].trainable = value
        model.layers[5].trainable = (not value)
        model.layers[6].trainable = (not value)
        model.layers[7].trainable = (not value)
    elif target == "all":
        model.layers[1].trainable = value
        model.layers[2].trainable = value
        model.layers[3].trainable = value
        model.layers[5].trainable = value
        model.layers[6].trainable = value
        model.layers[7].trainable = value
    elif target == "shift":
        model.layers[1].trainable = value
        model.layers[2].trainable = not value
        model.layers[3].trainable = not value
        model.layers[5].trainable = not value
        model.layers[6].trainable = not value
        model.layers[7].trainable = not value


def __information_gain(coords,
                       target,
                       visible,
                       optimizer,
                       refiner,
                       train_config,
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
                                                    config=train_config,
                                                    layers=layers,
                                                    train_set_size=train_set_size,
                                                    batch_size=batch_size,
                                                    stages=stages
                                                    )
    # show output of the first two layers
    extrapolation = model.predict(coords, batch_size=batch_size)
    extrapolation = extrapolation.reshape(target.shape[0], target.shape[1], outputs.shape[1])

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
                           train_config=config["train"],
                           layers=get_or_default("layers", 1),
                           batch_size=config["train"]["batch_size"],
                           stages=config["train"]["stages"]
                           )
    # print model summary to stdout
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    bias = []
    bias.append(layer_dict['ShiftLayer'].get_weights())
    Verbose.print("Shift: " + colored(str((bias[-1][0])*config["layer_normalization"]["shift"]), "green"), Verbose.always)

    bias.append(layer_dict['RotationLayer'].get_weights())
    Verbose.print("Rotation: " + colored(str(bias[-1][0]*config["layer_normalization"]["rotation"]*180/np.pi), "green"), Verbose.always)

    bias.append(layer_dict['ScaleLayer'].get_weights())
    Verbose.print("Scale: " + colored(str(bias[-1][0]*config["layer_normalization"]["scale"] + 1), "green"), Verbose.always)

    # bias = layer_dict['ShearLayer'].get_weights()
    # Verbose.print("Shear: " + colored(str(bias[0]*0.1), "green"), Verbose.always)

    Verbose.imshow(extrapolation)
    Verbose.imshow(outputs)
    Verbose.imshow(ig)

    return bias, bias_history
