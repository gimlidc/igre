import os
import tensorflow as tf
import numpy as np
from src.visualization.color_transformation import wavelength2rgb


class IgreInputCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, input_image, target_image):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(IgreInputCallbacks, self).__init__()
        # Create batch dimension expects [width, height, channels]
        self.inputs = np.expand_dims(input_image, 0)
        self.target = tf.convert_to_tensor(np.expand_dims(target_image, 0), tf.double)
        self.flatten_input = None

    def on_train_begin(self, logs=None):
        # NOTE: Workaround for only dense network
        if len(self.model.input_shape) == 2:
            self.flatten_input = self.inputs.reshape(
                (self.inputs.shape[1] * self.inputs.shape[2], self.inputs.shape[-1]))

    def _predict(self):
        # TODO: This workaround is slow
        assert self.model is not None
        if self.flatten_input is None:
            return self.model.predict(self.inputs)
        else:
            return self.model.predict(self.flatten_input).reshape(self.target.shape)


class InformationGainCallback(IgreInputCallbacks):
    def __init__(self, input_image, target_image, logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(InformationGainCallback, self).__init__(input_image, target_image)
        self.outputs = tf.reshape(self.target, self.target.shape[1:])
        self.max_file_writer = tf.summary.create_file_writer(logdir + "/gain/max")
        self.mean_file_writer = tf.summary.create_file_writer(logdir + "/gain/mean")

    def gain(self):
        # calculate gain and print it out
        gain = abs(self.outputs - self._predict()) / (np.prod(self.outputs.shape[0]))
        return gain

    def gain_max(self):
        return np.amax(self.gain())

    def gain_mean(self):
        return np.mean(self.gain())

    def on_epoch_end(self, epoch, logs=None):
        with self.max_file_writer.as_default():
            tf.summary.scalar('Information Gain', data=self.gain_max(), step=epoch)
        with self.mean_file_writer.as_default():
            tf.summary.scalar('Information Gain', data=self.gain_mean(), step=epoch)


class ImageNirCallback(IgreInputCallbacks):
    def __init__(self, input_image, target_image, logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(ImageNirCallback, self).__init__(input_image, target_image)
        self.image_file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'img'))

    def _image(self):
        # calculate gain and print it out
        approx = self._predict()
        diff = (self.target - approx) - np.min(self.target - approx)
        diff = tf.image.grayscale_to_rgb(tf.convert_to_tensor(diff / np.max(diff)))
        return diff

    def on_epoch_end(self, epoch, logs=None):
        with self.image_file_writer.as_default():
            tf.summary.image('ImageNir', data=self._image(), step=epoch)


class ImagesCallback(IgreInputCallbacks):
    def __init__(self, input_image, target_image, logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(ImagesCallback, self).__init__(input_image, target_image)
        # Create batch dimension expects [width, height, channels]
        pad = ((input_image.shape[:2][0] - target_image.shape[:2][0]) // 2)
        pad_input = input_image[pad:input_image.shape[0] - pad,
                    pad:input_image.shape[1] - pad]
        self.visible = tf.convert_to_tensor(np.expand_dims(wavelength2rgb(pad_input), 0), tf.double)
        self.image_file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'imgs'))

    def _image(self):
        # calculate gain and print it out
        approx = self._predict()
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
