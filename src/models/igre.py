import sys
import tensorflow as tf
from time import time
from src.tftools.shift_layer import ShiftLayer
from src.tftools.scale_layer import ScaleLayer
from src.tftools.rotation_layer import RotationLayer
# from src.tftools.radial_distortion_layer import RDistortionLayer
# from src.tftools.radial_distortion_layer_2 import RDistortionLayer2
# from src.tftools.radial_distortion_layer_3 import RDistortionLayer3
from src.tftools.radial_distortion_complete import RDCompleteLayer
# from src.tftools.shear_layer import ShearLayer
from src.tftools.idx2pixel_layer import Idx2PixelLayer, reset_visible
from src.tftools.idx2pixel_layer_bc import Idx2PixelBCLayer, reset_visible
from src.tftools.transform_metric import ShiftMetrics, ScaleMetrics, RotationMetrics, DistortionMetrics, RDMetrics
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from src.logging.verbose import Verbose
from src.tftools.optimizer_builder import build_optimizer
from src.config.tools import get_config, get_or_default
import numpy as np
from termcolor import colored
from src.data.ann.input_preprocessor import training_batch_selection, blur_preprocessing
from src.tftools.optimizer_builder import build_refining_optimizer
import datetime
import matplotlib.pyplot as plt

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

    selection = training_batch_selection(train_set_size, reg_layer_data)
    indexes = inputs[selection, :]

    # define model
    print('Adding input layer, width =', indexes.shape[1])

    # TODO: revisit shear layer
    input_layer = tf.keras.layers.Input(shape=(indexes.shape[1],),
                                        dtype=tf.float32, name='InputLayer')
    shift_layer = ShiftLayer(name='ShiftLayer')(input_layer)
    scale_layer = ScaleLayer(name='ScaleLayer')(shift_layer)
    rotation_layer = RotationLayer(name='RotationLayer')(scale_layer)
    radial_distortion_layer = RDCompleteLayer(name='RDistortionLayer')(rotation_layer)
    # radial_distortion_layer = RDistortionLayer(name='RDistortionLayer')(rotation_layer)
    # radial_distortion_layer_2 = RDistortionLayer2(name='RDistortionLayer2')(radial_distortion_layer)
    # radial_distortion_layer_3 = RDistortionLayer3(name='RDistortionLayer3')(radial_distortion_layer_2)
    layer = Idx2PixelBCLayer(visible=reg_layer_data, name='Idx2PixelLayer')(radial_distortion_layer)

    # TODO: Add InformationGain layers here when necessary
    print('Adding ReLU output layer, width =', outputs.shape[1])
    output_layer = tf.keras.layers.ReLU(max_value=1, name='Output', trainable=False)(layer)
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
    tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())

    model.layers[1].set_weights([np.array([0, 0])])  # shift
    model.layers[2].set_weights([np.array([0, 0])])  # scale
    model.layers[3].set_weights([np.array([0])])     # rotation

    # TODO: better names for stages
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
            __set_train_registration(model, True)
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_1':
            output = outputs
            __set_train_registration(model, 1, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_2':
            output = outputs
            __set_train_registration(model, 2, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_1_2':
            output = outputs
            __set_train_registration(model, 1, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_2_2':
            output = outputs
            __set_train_registration(model, 2, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_3':
            output = outputs
            __set_train_registration(model, 4, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_3_2':
            output = outputs
            __set_train_registration(model, 4, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])

        elif stage['type'] == 'refine_dist':
            output = outputs
            __set_train_registration(model, 7, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'refine_dist_2':
            output = outputs
            __set_train_registration(model, 7, target="distortion")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_complete':
            output = outputs
            __set_train_registration(model, True, target="rd")
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
        elif stage['type'] == 'dist_complete_2':
            output = outputs
            __set_train_registration(model, True, target="rd")
            model.compile(loss='mean_squared_error', optimizer=refiner, metrics=['mean_squared_error'])

        reset_visible(output)
        output = output[selection, :]
        shift_metric = ShiftMetrics()
        scale_metric = ScaleMetrics()
        rotation_metric = RotationMetrics()
        # distortion_metric = DistortionMetrics()
        distortion_metric = RDMetrics()

        # mcp_save = ModelCheckpoint(MODEL_FILE,
        #                            save_best_only=True, monitor='val_loss', mode='min')
        # lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=100, verbose=0, mode='auto',
        #                                  min_delta=0.0001, cooldown=0, min_lr=0)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [shift_metric, scale_metric, rotation_metric, distortion_metric, tensorboard_callback]

        history = model.fit(indexes,
                            output,
                            epochs=stage['epochs'],
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[shift_metric, scale_metric, rotation_metric, distortion_metric],
                            batch_size=batch_size
                            )

        config = get_config()
        coef_1 = [x[0][0]*config["layer_normalization"]["radial_distortion"] for x in distortion_metric.bias_history]
        coef_2 = [x[0][1]*config["layer_normalization"]["radial_distortion_2"] for x in distortion_metric.bias_history]
        coef_3 = [x[0][2]*config["layer_normalization"]["radial_distortion_3"] for x in distortion_metric.bias_history]
        plt.plot(coef_1, label="1")
        plt.plot(coef_2, label="2")
        plt.plot(coef_3, label="3")

        plt.title("Transformation")
        plt.legend()
        plt.show()

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    # File "C:\Anaconda\envs\igre\lib\site-packages\tensorflow\python\keras\engine\training.py", line 1078, in predict
    #     callbacks=callbacks)
    #   File "C:\Anaconda\envs\igre\lib\site-packages\tensorflow\python\keras\engine\training_arrays.py", line 370, in model_iteration
    #     aggregator.aggregate(batch_outs, batch_start, batch_end)
    #   File "C:\Anaconda\envs\igre\lib\site-packages\tensorflow\python\keras\engine\training_utils.py", line 169, in aggregate
    #     self.results[i][batch_start:batch_end] = batch_out
    # ValueError: could not broadcast input array from shape (212,212) into shape (212,2048)
    gain = abs(outputs[selection, :] - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                                         outputs.shape[1])
    # information_gain_max = gain.flatten().max()
    # print('Gain: {:1.4e}'.format(information_gain_max))

    return model, history, shift_metric.bias_history


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
        model.layers[4].trainable = (not value)
        model.layers[5].trainable = (not value)
        model.layers[6].trainable = (not value)
        model.layers[7].trainable = (not value)
        model.layers[8].trainable = (not value)
    elif target == "all":
        model.layers[1].trainable = value
        model.layers[2].trainable = value
        model.layers[3].trainable = value
        model.layers[4].trainable = value
        model.layers[5].trainable = value
        model.layers[6].trainable = value
        model.layers[7].trainable = value
        model.layers[8].trainable = value
    elif target == "shift":
        model.layers[1].trainable = value
        model.layers[2].trainable = not value
        model.layers[3].trainable = not value
        model.layers[4].trainable = not value
        model.layers[5].trainable = not value
        model.layers[6].trainable = not value
        model.layers[7].trainable = not value
        model.layers[8].trainable = not value
    elif target == "rd":
        model.layers[1].trainable = not value
        model.layers[2].trainable = not value
        model.layers[3].trainable = not value
        model.layers[4].trainable = value
        model.layers[5].trainable = not value
        model.layers[6].trainable = not value
    elif target == "distortion":
        model.layers[1].trainable = not value
        model.layers[2].trainable = not value
        model.layers[3].trainable = not value
        if value == 1:
            model.layers[4].trainable = value
            model.layers[5].trainable = not value
            model.layers[6].trainable = not value
        if value == 2:
            model.layers[4].trainable = not value
            model.layers[5].trainable = value
            model.layers[6].trainable = not value
        if value == 4:
            model.layers[4].trainable = not value
            model.layers[5].trainable = not value
            model.layers[6].trainable = value
        if value == 7:
            model.layers[4].trainable = value
            model.layers[5].trainable = value
            model.layers[6].trainable = value
        model.layers[7].trainable = not value
        model.layers[8].trainable = not value


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
                           train_config=config["train"],
                           layers=get_or_default("layers", 1),
                           batch_size=config["train"]["batch_size"],
                           stages=config["train"]["stages"]
                           )
    # print model summary to stdout
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    bias = layer_dict['ShiftLayer'].get_weights()
    Verbose.print("Shift: " + colored(str((bias[0])*config["layer_normalization"]["shift"]), "green"), Verbose.always)

    bias = layer_dict['RotationLayer'].get_weights()
    Verbose.print("Rotation: " + colored(str(bias[0]*config["layer_normalization"]["rotation"]*180/np.pi), "green"), Verbose.always)

    bias = layer_dict['ScaleLayer'].get_weights()
    Verbose.print("Scale: " + colored(str(bias[0]*config["layer_normalization"]["scale"] + 1), "green"), Verbose.always)

    k1 = layer_dict['RDistortionLayer'].get_weights()[0][0] * config["layer_normalization"]["radial_distortion"]
    k2 = layer_dict['RDistortionLayer'].get_weights()[0][1] * config["layer_normalization"]["radial_distortion_2"]
    k3 = layer_dict['RDistortionLayer'].get_weights()[0][2] * config["layer_normalization"]["radial_distortion_3"]
    exp_k1 = -k1
    exp_k2 = 3*k1*k1 - k2
    exp_k3 = -12*k1*k1*k1 + 8*k1*k2 - k3
    Verbose.print("coefs computed: " + colored(str([k1, k2, k3]), "green"), Verbose.always)
    # Verbose.print("coefs inverse: " + colored(str([exp_k1, exp_k2, exp_k3]), "green"), Verbose.always)
    bias = [k1, k2, k3]
    # bias = layer_dict['ShearLayer'].get_weights()
    # Verbose.print("Shear: " + colored(str(bias[0]*0.1), "green"), Verbose.always)

    return bias, bias_history
