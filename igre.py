import sys
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from time import time
from termcolor import colored
from idx2pixel_layer import Idx2PixelLayer
from shift_metric import ShiftMetrics
import matplotlib.pyplot as plt


def __print_shift_convergence():
    global g_bias_history, g_loss_history
    # g_bias_history = np.array(g_bias_history).squeeze()
    pyplot.subplot(1, 2, 1)
    pyplot.plot(range(len(g_bias_history[:, 0])), g_bias_history[:, 0],
                color='red')
    pyplot.plot(range(len(g_bias_history[:, 1])), g_bias_history[:, 1],
                color='blue')
    pyplot.title("'bias' training history, 0 in red, 1 in blue.")
    pyplot.xlabel("batch number")
    yticks = np.ceil(max(abs(g_bias_history.flatten())) * 10) / 10
    ystep = np.ceil(yticks) / 10
    yticks = np.arange(-yticks, yticks + ystep, step=ystep)
    pyplot.yticks(yticks)
    pyplot.ylabel("'bias' value in pixels")
    pyplot.grid()
    pyplot.subplot(1, 2, 2)
    pyplot.plot(range(len(g_loss_history)), g_loss_history)
    pyplot.grid()
    pyplot.show()


def __trainNetworks(inputs,
                    outputs,
                    reg_layer_data,
                    layers=[25, 25],
                    train_set_size=50000,
                    use_gpu=False,
                    optimizer="adam",
                    batch_size=256,
                    epochs=100
                    ):
    """
        Train several networks and return the best one. Usage:

            bestNetwork, history = trainNetworks(indexes, inputs, outputs, annCount,
                    layers, trainingSetSize, useGPU)

        Mandatory arguments:
            indexes = array of indexes to the input image
            inputs = input image (must be 3D, even if the third dimension is trivial)
            outputs = output vectors
            annCount = number of neural networks wich will be trained
            layers = numbers of neurons in inner layers of ANN
            trainingSetSize = number of inputs/outputs used for training (randomly
                selected)
            useGPU = use GPU for training?

        Function uses trainscg for training ANN and number of epochs is set to
        10k.
    """

    # TODO: split data into training, validation, test by myself
    # TODO: parametrize split of data??? (pass up tuple of 2 or 3 floats)

    print('Selecting', train_set_size,
          'samples randomly for use by algorithm.')
    perm = np.random.permutation(inputs.shape[0])
    indexes = inputs[perm[:train_set_size], :]
    outputs = outputs[perm[:train_set_size], :]

    # define model
    print('Adding input layer, width =', indexes.shape[1])
    input_layer = tf.keras.layers.Input(shape=(indexes.shape[1],),
                                        dtype=tf.float32, name='InputLayer')
    # print(input_layer.shape)
    # layer = input_layer
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Identity": "MyOpGrad"}):
    layer = Idx2PixelLayer(visible=reg_layer_data, name='Idx2PixelLayer')(input_layer)
    # for layer_idx in range(len(layers)):
    #     print('Adding dense layer, width =', layers[layer_idx])
    #     layer = tf.keras.layers.Dense(layers[layer_idx],
    #                                   activation='sigmoid', name='Dense' + str(layer_idx))(layer)
    print('Adding dense layer, width =', outputs.shape[1])
    output_layer = tf.keras.layers.ReLU(max_value=1, name='Output', trainable=False)(layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # compile model
    start_time = time()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.35,
                        beta1=0.01,
                        beta2=0.85)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error']
                  )
    elapsed_time = time() - start_time
    print("Compiling model took {:.4f}'s.".format(elapsed_time))

    # train model
    start_time = time()
    shift_metric = ShiftMetrics()
    callbacks = [shift_metric]
    history = model.fit(indexes,
                        outputs,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=0,
                        callbacks=callbacks,
                        batch_size=batch_size
                        )

    plt.plot(np.array(shift_metric.bias_history)[:, 0])
    plt.show()

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time /
                                                    num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and save best model so far
    gain = abs(outputs - model.predict(indexes, batch_size=batch_size)) / (outputs.shape[0] *
                                                                           outputs.shape[1])
    information_gain_max = gain.flatten().max()
    print('Gain: {:1.4e}'.format(information_gain_max))

    return model, history


def __informationGain(coords,
                      target,
                      visible,
                      layers,
                      train_set_size: int = 25000,
                      use_gpu: bool = False,
                      batch_size=256,
                      epochs=100,
                      optimizer="adam"):

    if coords.shape[0] != target.shape[0] * visible.shape[1]:
        sys.exit("Error: dimension mismatch between 'target' and 'visible'")

    if len(target.shape) == 2:
        target = target.reshape((target.shape[0], target.shape[1], 1))

    if train_set_size is None:
        # use the whole image for training, evaluation and test
        train_set_size = visible.shape[0] * visible.shape[1]

    outputs = target.reshape((target.shape[0] * target.shape[1], target.shape[2]))

    # train ANN
    model, history = __trainNetworks(coords,
                                     outputs,
                                     reg_layer_data=visible,
                                     layers=layers,
                                     train_set_size=train_set_size,
                                     use_gpu=use_gpu,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     optimizer=optimizer)

    # show output of the first two layers
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                     outputs=model.get_layer('Idx2PixelLayer').output)
    output_of_2nd_layer = intermediate_layer_model.predict(coords,
                                                           batch_size=batch_size)
    for i in range(output_of_2nd_layer.shape[1]):
        if i != 5 and i != 10:
            continue
        pyplot.subplot(1, 2, 1)
        pyplot.imshow(visible[:, :, i].squeeze(), vmin=0, vmax=1, cmap='gray')
        pyplot.title("Visible {}, range [0, 1].".format(i))
        pyplot.subplot(1, 2, 2)
        pyplot.imshow(output_of_2nd_layer[:, i].reshape(target.shape).squeeze(),
                      vmin=0, vmax=1, cmap='gray')
        pyplot.title("Output {} of interpolation layer, range [0, 1].".format(i))
        pyplot.show()

    extrapolation = model.predict(coords, batch_size=batch_size)
    extrapolation = extrapolation.reshape(target.shape)

    infGain = target - extrapolation

    return infGain, extrapolation, model


def run(inputs,
        outputs,
        visible,
        layers=[25, 25], batch_size=256, epochs=100, use_gpu=False,
        optimizer="adam"):
    infGain, extrapolation, model = __informationGain(inputs,
                                                      outputs,
                                                      visible=visible,
                                                      use_gpu=use_gpu,
                                                      layers=layers,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      optimizer=optimizer)
    # print model summary to stdout
    model.summary()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    bias = layer_dict['Idx2PixelLayer'].get_weights()
    print("Shift detected: " + colored(str(bias[0]), "green"))
