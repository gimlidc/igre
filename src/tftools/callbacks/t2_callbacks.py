import os
import tensorflow as tf
import numpy as np
from src.visualization.color_transformation import wavelength2rgb


class IgreInputCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, input_image, target_image):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(IgreInputCallbacks, self).__init__()
        # Create batch dimension expects [width, height, channels]

        self.inputs = tf.convert_to_tensor(input_image, tf.float32)
        self.target = tf.convert_to_tensor(target_image, tf.float32)

    def _predict(self):
        assert self.model is not None
        output = self.model.predict(self.inputs)
        return output


class InformationGainCallback(IgreInputCallbacks):
    def __init__(self, input_image, target_image, logdir, name='gain'):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(InformationGainCallback, self).__init__(input_image, target_image)
        self.max_file_writer = tf.summary.create_file_writer(f"{logdir}/{name}/max")
        self.mean_file_writer = tf.summary.create_file_writer(f"{logdir}/{name}/mean")
        self._name = name

    def gain(self):
        # calculate gain and print it out
        prediction = tf.reshape(self._predict(), self.target.shape)
        gain = tf.abs(self.target - prediction)
        return gain

    def gain_max(self):
        return tf.math.reduce_max(self.gain())

    def gain_mean(self):
        return tf.reduce_mean(self.gain())

    def on_epoch_end(self, epoch, logs=None):
        with self.max_file_writer.as_default():
            tf.summary.scalar(f'Information Gain {self._name}', data=self.gain_max(), step=epoch)
        with self.mean_file_writer.as_default():
            tf.summary.scalar(f'Information Gain {self._name}', data=self.gain_mean(), step=epoch)


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
    def __init__(self, input_image, target_image, pad,  logdir):
        assert int(tf.__version__[0]) == 2, "Tensorboard logging only for tf 2"
        super(ImagesCallback, self).__init__(input_image, target_image)
        # Create batch dimension expects [width, height, channels]
        pad_input = input_image[:,
                                pad:input_image.shape[1] - pad,
                                pad:input_image.shape[2] - pad]
        self.visible = tf.convert_to_tensor(np.squeeze(wavelength2rgb(pad_input)), tf.float32)
        self.image_file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'imgs'))

    def _image(self):
        # calculate gain and print it out
        approx = tf.reshape(self._predict(), self.target.shape)
        diff = (self.target - approx) - np.min(self.target - approx)
        diff = tf.image.grayscale_to_rgb(tf.convert_to_tensor(diff / np.max(diff)))
        approx = tf.image.grayscale_to_rgb(tf.convert_to_tensor(approx, tf.float32))
        target = tf.image.grayscale_to_rgb(self.target, tf.float32)
        imgs = tf.concat(
            [tf.concat([target, diff], axis=0),
             tf.concat([approx, self.visible], axis=0)],
            axis=1)

        return tf.expand_dims(imgs, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        with self.image_file_writer.as_default():
            tf.summary.image('Images', data=self._image(), step=epoch)
