from tensorflow.keras.callbacks import TensorBoard
from src.tftools.radial_distortion_complete import RDCompleteLayer
from src.logging.verbose import Verbose
from src.data.ann.input_preprocessor import pixels_for_training_radial, training_batch_selection, blur_preprocessing
import tensorflow as tf
from time import time
import numpy as np
import logging
from src.config.tools import get_config, get_or_default, init_config
from src.tftools.optimizer_builder import build_optimizer, build_refining_optimizer
from termcolor import colored
import yaml
import matplotlib.pyplot as plt
import datetime
from src.tftools.callbacks.inter_predict import InterPredictCallback
from src.tftools.idx2pixel_layer import Idx2PixelLayer, reset_visible
from src.tftools.transform_metric import RDMetrics
from src.tftools.shift_layer import ShiftLayer
from src.tftools.scale_layer import ScaleLayer
from src.tftools.rotation_layer import RotationLayer


MODEL_FILE = "best_model.tmp.h5"
logger = logging.getLogger()


def __normalize_img(img):
    logger.info("Image min/max normalization")
    out = np.array(img, dtype=float)
    if len(out.shape) != 3:
        logger.info("Adding third dimension to input data")
        out = out.reshape(out.shape[0], out.shape[1], 1)
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    return out


def find_tform(target, moving):
    """
    Expected input are two grayscale images with limited radial distortion.
    Output is estimated k1, k2, k3 according to Brown Conrady model.
    @see https://www.asprs.org/wp-content/uploads/pers/1966journal/may/1966_may_444-462.pdf
    :param target: ndarray
        image with the object position to fit
    :param moving:
        image which we would like to transform
    :return: [float, float, float]
        k1, k2 and k3 for radial distortion
    """
    coords = np.indices((moving.shape[0], moving.shape[1])).reshape(2, -1).transpose().astype(np.float32)
    __run(coords, __normalize_img(target), __normalize_img(moving))


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
    layer = Idx2PixelLayer(visible=reg_layer_data, name='Idx2PixelLayer')(radial_distortion_layer)

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
    model.layers[3].set_weights([np.array([0])])  # rotation
    # model.load_weights(MODEL_FILE)
    # TODO: better names for stages
    config = get_config()
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
        else:
            stage_params = stage['type'].split('_')

            if stage_params[0] == 'adam':
                opt_type = 'optimizer'
                opt = build_optimizer(config["train"][opt_type], config["train"]["batch_size"])
            elif stage_params[0] == 'sgd':
                opt_type = 'refiner'
                opt = build_refining_optimizer(config["train"][opt_type])

            if stage_params[1] == 'rd':
                targ = "rd"
                output = outputs

            __set_train_registration(model, True, target=targ)
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

        reset_visible(output)
        output = output[selection, :]
        # shift_metric = ShiftMetrics()
        # scale_metric = ScaleMetrics()
        # rotation_metric = RotationMetrics()
        # distortion_metric = DistortionMetrics()
        distortion_metric = RDMetrics()

        # mcp_save = ModelCheckpoint(MODEL_FILE,
        #                            save_best_only=True, monitor='val_loss', mode='min')
        checkpoint_filepath = MODEL_FILE
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        # lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=100, verbose=0, mode='auto',
        #                                  min_delta=0.0001, cooldown=0, min_lr=0)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [distortion_metric, tensorboard_callback,
                     model_checkpoint_callback]

        history = model.fit(indexes,
                            output,
                            epochs=stage['epochs'],
                            validation_split=0.2,
                            verbose=1,
                            callbacks=[distortion_metric,
                                       model_checkpoint_callback
                                       ],
                            batch_size=batch_size
                            )

        coef_1 = [x[0][0] * config["layer_normalization"]["radial_distortion"] for x in distortion_metric.bias_history]
        coef_2 = [x[1][0] * config["layer_normalization"]["radial_distortion_2"] for x in
                  distortion_metric.bias_history]
        coef_3 = [x[2][0] * config["layer_normalization"]["radial_distortion_3"] for x in
                  distortion_metric.bias_history]
        plt.plot(coef_1, label="1")
        plt.plot(coef_2, label="2")
        plt.plot(coef_3, label="3")

        model.load_weights(checkpoint_filepath)

        plt.title("Transformation %f %f %f " % (coef_1[-1], coef_2[-1], coef_3[-1]))
        plt.legend()
        plt.show()

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    model.load_weights(checkpoint_filepath)

    gain = abs(outputs[selection, :] - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                                         outputs.shape[1])
    information_gain_max = gain.flatten().max()
    print('Gain: {:1.4e}'.format(information_gain_max))

    return model, history, None


def __set_train_registration(model, value, target="registration"):
    """
    For various stages of training registration should not be trainable (we are looking for base mutual setup).
    This method allows enabling/disabling of trainability of first three layers of ANN.
    :param model: ANN
    :param value: boolean
    """
    if target == "all":
        for layer in model.layers:
            layer.trainable = value
    elif target == "rd":
        model.layers[1].trainable = value
        model.layers[2].trainable = value
        model.layers[3].trainable = value
    elif target == "mutual_init":
        model.layers[1].trainable = not value
        model.layers[2].trainable = not value
        model.layers[3].trainable = not value
        for i in range(4, len(model.layers)):
            model.layers[i].trainable = value


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
        logger.warning("Error: dimension mismatch between 'target' and 'visible'")

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


def __run(inputs, outputs, moving):
    with open("input/radial-config.yaml", "rt", encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        init_config(config)

    config = get_config()
    ig, extrapolation, model, bias_history = \
        __information_gain(inputs,
                           outputs,
                           visible=moving,
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

    k1 = layer_dict['RDistortionLayer'].get_weights()[0][0] * config["layer_normalization"]["radial_distortion"]
    k2 = layer_dict['RDistortionLayer'].get_weights()[1][0] * config["layer_normalization"]["radial_distortion_2"]
    k3 = layer_dict['RDistortionLayer'].get_weights()[2][0] * config["layer_normalization"]["radial_distortion_3"]

    exp_k1 = -k1
    exp_k2 = 3 * k1 * k1 - k2
    exp_k3 = -12 * k1 * k1 * k1 + 8 * k1 * k2 - k3

    Verbose.print("coefs computed: " + colored(str([k1, k2, k3]), "green"), Verbose.always)
    Verbose.print("coefs inverse: " + colored(str([exp_k1, exp_k2, exp_k3]), "green"), Verbose.always)

    bias = [k1, k2, k3]

    plt.figure(figsize=(15, 8))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(extrapolation[:, :, 0], cmap="gray")
    ax.set_title("Predicted")
    ax = plt.subplot(1, 3, 2)
    ax.imshow(outputs[:, :, 0], cmap="gray")
    ax.set_title("Expected output")
    ax = plt.subplot(1, 3, 3)
    ax.imshow(ig[:, :, 0], cmap="Reds")
    ax.set_title("Difference between predicted and ground truth")
    plt.show()

    return bias, bias_history
