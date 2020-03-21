import sys
import numpy as np
import tensorflow as tf
from time import time


def __train_networks(inputs, outputs, layers, training_set_size):
    """
    Train several networks and return the best one. Usage:

        bestNetwork, history = trainNetworks(indexes, inputs, outputs, annCount,
                layers, trainingSetSize, useGPU)

    Mandatory arguments:
        indexes = array of indexes to the input image
        inputs = input image (must be 3D, even if the third dimension is trivial)
        outputs = output vectors
        layers = numbers of neurons in inner layers of ANN
        trainingSetSize = number of inputs/outputs used for training (randomly
            selected)

    Function uses trainscg for training ANN and number of epochs is set to 10k.
    """

    # randomly select 'trainingSetSize' data samples
    print('Selecting', training_set_size, 'samples randomly for use by algorithm.')
    perm = np.random.permutation(inputs.shape[0])
    ins = inputs[perm[:training_set_size], :]
    outs = outputs[perm[:training_set_size], :]

    # define model
    print('Adding input layer, width =', inputs.shape[1])
    input_layer = tf.keras.layers.Input(shape=(inputs.shape[1],),
                                        dtype=tf.float32, name='InputLayer')
    layer = input_layer
    if layers is None:
        layers = [25, 25]
    for layer_idx in range(len(layers)):
        print('Adding dense layer, width =', layers[layer_idx])
        layer = tf.keras.layers.Dense(layers[layer_idx],
                                      activation='sigmoid', name='Dense' + str(layer_idx))(layer)
    print('Adding dense layer, width =', outputs.shape[1])
    output_layer = tf.keras.layers.Dense(outputs.shape[1], activation='linear', name='Output')(layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    start_time = time()
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error']
                  )
    elapsed_time = time() - start_time
    print("Compiling model took {:.4f}'s.".format(elapsed_time))

    # train model
    start_time = time()
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=6,
                                                  min_delta=10 ** -5)]
    history = model.fit(ins,
                        outs,
                        epochs=40,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks
                        )

    elapsed_time = time() - start_time
    num_epochs = len(history.history['loss'])

    print("{:.4f}'s time per epoch. Epochs:".format(elapsed_time / num_epochs), num_epochs, sep="")
    print("Total time {:.4f}'s".format(elapsed_time))

    # calculate gain and print it out
    gain = abs(outputs - model.predict(inputs)) / (outputs.shape[0] * outputs.shape[1])
    print('Gain: {:1.4e}'.format(gain.flatten().max()))

    return model, history


def information_gain(visible,
                     target,
                     layers=None,
                     training_set_size: int = 25000):
    """
    Compute information gain of the target for visible. Usage:

        infGain, extrapolation, model = informationGain(visible, target, ...)

    Transform function for transformation of visible to target will be trained by
    FF-ANN. By this neural network extrapolation is computed.  Finally, information
    gain is computed as difference between target and extrapolation.  'visible' and
    'target' must have same dimensions (width, height).

    Mandatory arguments:
        visible = 3-dimensional array of VIS pixel intensities in form (height,
            width, modalities)
        target  = 3-dimensional array of NIR pixel intensities in form (height,
            width, modalities)

    Optional parameters:
        layers = array of layer widths
        training_set_size = number of samples used for training, if 'None' is passed,
            all pixels will be used (for training, evaluation, and testing)

    Example:
        informationGain(visible, target layers=[10, 10],
                trainingSetSize=3000)
    """

    if visible.shape[0] != target.shape[0] or visible.shape[1] != target.shape[1]:
        sys.exit("Error: dimension mismatch between 'target' and 'visible'")

    if len(target.shape) == 2:
        target = target.reshape((target.shape[0], target.shape[1], 1))

    if training_set_size is None:
        # use the whole image for training, evaluation and test
        training_set_size = visible.shape[0] * visible.shape[1]

    inputs = visible.reshape((visible.shape[0] * visible.shape[1], visible.shape[2]))
    outputs = target.reshape((target.shape[0] * target.shape[1], target.shape[2]))

    # train ANN
    model, history = __train_networks(inputs, outputs, layers, training_set_size)

    extrapolation = model.predict(inputs)
    extrapolation = extrapolation.reshape(target.shape)

    inf_gain = target - extrapolation

    return inf_gain, extrapolation, model
