import scipy.io as sio
from tensorflow import keras
import numpy as np


def prepare_data(image_path,
                 visible_channels,
                 predicted_channel):
    """
    Divides data into input visible dataset and target nir dataset
    Parameters
    ----------
    image_path : path to file
    visible_channels : last channel of input
    predicted_channel : target layer

    Returns
    -------
    (visible, target)
    """
    input_data = sio.loadmat(image_path)['data']

    visible = input_data[:, :, 0:visible_channels] / 255.0
    target = np.expand_dims(input_data[:, :, predicted_channel] / 255.0, axis=-1)
    return visible, target


def get_cnn_dataset(image_path,
                    input_size=3,
                    visible_channels=16,
                    predicted_channel=26,
                    ):
    """
    Prepares dataset for CNN training,
     divides the input image into multiple squares of size (input_size) with the middle of target channel as output

    Parameters
    ----------
    image_path : path to multichannel image
    input_size : square size
    visible_channels : number of input channels
    predicted_channel : target channel

    Returns
    -------
    Input channels divided into squares, expected outputs, the target channel with padded width
    """
    visible, target = prepare_data(image_path,
                                   visible_channels=visible_channels,
                                   predicted_channel=predicted_channel)
    assert (input_size - 1) % 2 == 0
    px_padding = (input_size - 1) // 2
    target = target[px_padding:visible.shape[0] - px_padding,
                    px_padding:visible.shape[1] - px_padding]
    arr = visible
    l = []
    for x in range(px_padding, visible.shape[0] - px_padding):
        for y in range(px_padding, visible.shape[1] - px_padding):
            l.append(arr[x - px_padding:x + px_padding + 1, y - px_padding:y + px_padding + 1])
    inputs = np.array(l)
    outputs = target.reshape([target.shape[0] * target.shape[1],
                              1,
                              1,
                              target.shape[2]])
    return inputs, outputs, target, visible


def str2layer(text, option_separator='-'):
    """
    Convert string to layer
    Supported types
      * Convolution2D - C-[filters]-[kernel_size]-[activation]-[padding]
      * Dense - D-[units]-[activation]

    Parameters
    ----------
    text :  string represents layer
    option_separator : character dividing the layer options

    Returns
    -------
    keras.Layer

    """
    text = text.lower()
    options = text.split(option_separator)
    if text[0] == 'c':
        assert 3 <= len(options) <= 5, f"Every conv layer has 3-5 parameters, {text} has {len(options)}"
        kwargs = {'filters': int(options[1]),
                  'kernel_size': int(options[2]),
                  'activation': options[3] if len(options) > 3 else None,
                  'padding': options[4] if len(options) > 4 else 'valid'
                  }
        return keras.layers.Conv2D(**kwargs)
    elif text[0] == 'd':
        assert len(options) == 3, f"Every dense layer has 2-3 parameters, {text} has {len(options)}"
        kwargs = {
            'units': int(options[1]),
            'activation': options[2] if len(options) > 1 else None,
        }
        return keras.layers.Dense(**kwargs)


def ig_cnn_model(def_text,
                 input_channels=16,
                 output_units=1,
                 name='IG-CNN',
                 layer_separator=','):
    """
    Create IG CNN model

    Parameters
    ----------
    def_text : String represents the model
    input_channels :
    output_units :
    name :
    layer_separator :

    Returns
    -------
    keras.Model
    """
    input_shape = [None, None, input_channels]
    layers = [keras.layers.Input(shape=input_shape)]

    layers_text = def_text.split(layer_separator)
    for layer_text in layers_text:
        layers.append(str2layer(layer_text))

    layers.append(keras.layers.Dense(units=output_units,
                                     activation='sigmoid'))

    return keras.Sequential(layers=layers, name=f'{name}')
