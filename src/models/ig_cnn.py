import os
import click
import scipy.io as sio
from colormath import spectral_constants
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorboard.plugins.hparams import api as hp

xyz_sensitivity = np.array([
    [0.0076, 0.0002, 0.0362, ],
    [0.0776, 0.0022, 0.3713, ],
    [0.3187, 0.0480, 1.7441, ],
    [0.0580, 0.1693, 0.6162, ],
    [0.0093, 0.5030, 0.1582, ],
    [0.1655, 0.8620, 0.0422, ],
    [0.4335, 0.9950, 0.0088, ],
    [0.7621, 0.9520, 0.0021, ],
    [1.0263, 0.7570, 0.0011, ],
    [1.0026, 0.5030, 0.0003, ],
    [0.6424, 0.2650, 0.0001, ],
    [0.2835, 0.1070, 0, ],
    [0.0636, 0.0232, 0, ],
    [0.0081, 0.0029, 0, ],
    [0.0010, 0.0004, 0, ],
    [0.0001, 0.0000, 0, ],
    [0.0003, 0.0001, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ],
    [0, 0, 0, ]])


def minMaxScale(arr, max_v=1, min_v=0):
    arr_std = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return arr_std * (max_v - min_v) + min_v


# Spectrum to CIE
# from 340 nm to 830 by 10 nm steps
class SpectrumColor:
    MIN_NM = 340
    MAX_NM = 830
    REF_ILLUM = spectral_constants.REF_ILLUM_TABLE['d50']
    DENOM = np.sum(REF_ILLUM
                   * spectral_constants.STDOBSERV_Y10)
    SPECTRUM2XYZ = np.array([REF_ILLUM * spectral_constants.STDOBSERV_X10 / DENOM,
                             REF_ILLUM * spectral_constants.STDOBSERV_Y10 / DENOM,
                             REF_ILLUM * spectral_constants.STDOBSERV_Z10 / DENOM]
                            ).transpose()
    STEP_NM = 10

    @classmethod
    def get_coef(cls, from_nm, to_nm):
        assert cls.MIN_NM <= from_nm <= from_nm
        assert from_nm <= to_nm <= cls.MAX_NM
        to_nm = (to_nm - cls.MIN_NM) // cls.STEP_NM
        from_nm = (from_nm - cls.MIN_NM) // cls.STEP_NM
        return np.mean(cls.SPECTRUM2XYZ[from_nm:to_nm + 1], axis=0)

    @classmethod
    def create_filter(cls, min_wl=380, max_wl=780, step_wl=25):
        assert (max_wl - min_wl) % step_wl == 0
        assert step_wl >= 5, "The approximation function from spectrum to cie has sensitivity 5 nm"
        filter = np.array([SpectrumColor.get_coef(i, i + 25) for i in range(min_wl, max_wl, step_wl)])
        return filter


def wavelength2rgb(in_image, min_wl=380, max_wl=780, step_wl=25):
    assert np.max(in_image) <= 1.0 and np.min(in_image) >= 0.0
    from skimage.color import xyz2rgb
    f = SpectrumColor.create_filter(min_wl=min_wl,
                                    max_wl=max_wl,
                                    step_wl=step_wl)
    o = np.matmul(in_image, f)
    return xyz2rgb(minMaxScale(o))


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


class InformationGainCallback(keras.callbacks.Callback):
    def __init__(self, input_image, target_image, logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(InformationGainCallback, self).__init__()
        # Create batch dimension expects [width, height, channels]
        self.inputs = np.expand_dims(input_image, 0)
        self.outputs = target_image
        self.max_file_writer = tf.summary.create_file_writer(logdir + "/gain/max")
        self.mean_file_writer = tf.summary.create_file_writer(logdir + "/gain/mean")

    def gain(self):
        # calculate gain and print it out
        gain = abs(self.outputs - self.model.predict(self.inputs)) / (self.outputs.shape[0] * self.outputs.shape[1])
        return gain

    def gain_max(self):
        return self.gain().flatten().max()

    def gain_mean(self):
        return np.mean(self.gain().flatten())

    def on_epoch_end(self, epoch, logs=None):
        with self.max_file_writer.as_default():
            tf.summary.scalar('Information Gain', data=self.gain_max(), step=epoch)
        with self.mean_file_writer.as_default():
            tf.summary.scalar('Information Gain', data=self.gain_mean(), step=epoch)


class ImagesCallback(keras.callbacks.Callback):
    def __init__(self, input_image, target_image, logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(ImagesCallback, self).__init__()
        # Create batch dimension expects [width, height, channels]
        self.inputs = tf.convert_to_tensor(np.expand_dims(input_image, 0), tf.double)
        pad = ((input_image.shape[:2][0] - target_image.shape[:2][0]) // 2)
        pad_input = input_image[pad:input_image.shape[0] - pad,
                    pad:input_image.shape[1] - pad]
        self.visible = tf.convert_to_tensor(np.expand_dims(wavelength2rgb(pad_input), 0), tf.double)
        self.target = tf.convert_to_tensor(np.expand_dims(target_image, 0), tf.double)
        self.image_file_writer = tf.summary.create_file_writer(logdir + '/img')

    def _image(self):
        # calculate gain and print it out
        approx = self.model.predict(self.inputs)
        diff = (self.target - approx) - np.min(self.target - approx)
        diff = tf.image.grayscale_to_rgb(tf.convert_to_tensor(diff / np.max(diff)))
        approx = tf.image.grayscale_to_rgb(tf.convert_to_tensor(approx, tf.double))
        target = tf.image.grayscale_to_rgb(self.target, tf.double)
        imgs = tf.concat(
            [tf.concat([target, diff], axis=2),
             tf.concat([approx, self.visible], axis=2)],
            axis=1)
        return imgs

    def on_epoch_end(self, epoch, logs=None):
        with self.image_file_writer.as_default():
            tf.summary.image('Images', data=self._image(), step=epoch)


def file_check(ctx, param, value):
    if os.path.exists(value):
        return value
    else:
        raise click.BadParameter(f'Invalid path: {value}')


@click.command()
@click.option('--image_path',
              default='/home/karellat/Desktop/igre/data/raw/still_life/sample_22.mat',
              help='Path to matlab multichannel image')
@click.option('--model_def',
              default='C-25-3-sigmoid-valid,C-25-1-sigmoid-same',
              help='Model defined by string, each layer separated by commas')
@click.option('--input_size',
              default='3',
              help='The visible spectrum is separated into squares with this size',
              type=int)
@click.option('--log_name',
              default='C3',
              help='Name of the log dir')
@click.option('--log_root',
              default='/home/karellat/Desktop/igre/logs',
              help='Root dir of logs files')
def ig_cnn(image_path, model_def, input_size, log_name, log_root):

    hparams = {
        'image_path': image_path,
        'model_def': model_def,
        'input_size': input_size,
        'log_name': log_name
    }

    # Parameters
    logdir = os.path.join(log_root, hparams['log_name'])
    assert not(os.path.exists(logdir)), f"The log dir of given name exists: {logdir}"
    # Input Data
    inputs, outputs, pad_output, nopad_input = get_cnn_dataset(hparams['image_path'],
                                                               input_size=hparams['input_size'])
    # Model Definition
    model = ig_cnn_model(hparams['model_def'])
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # Data split
    training_set_size = 2500
    perm = np.random.permutation(inputs.shape[0])
    ins = inputs[perm[:training_set_size], :]
    outs = outputs[perm[:training_set_size], :]

    # TODO: Check log dir existence

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                 InformationGainCallback(input_image=nopad_input,
                                         target_image=pad_output,
                                         logdir=logdir),
                 ImagesCallback(input_image=nopad_input,
                                target_image=pad_output,
                                logdir=logdir),
                 hp.KerasCallback(logdir, hparams),
                 # NOTE: This callback has to be last
                 tf.keras.callbacks.EarlyStopping(patience=6, min_delta=10 ** -5),
                 ]

    history = model.fit(ins,
                        outs,
                        epochs=100,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=callbacks
                        )


if __name__ == '__main__':
    ig_cnn()

