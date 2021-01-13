from tensorflow.keras.callbacks import Callback
import os
import imageio
import numpy as np


class InterPredictCallback(Callback):
    """
    Visualization of ANN training steps by prediction.
    Creation of predictions in the middle of a training can be useful not only for documentation of an algorithm.
    InterPredictCallback is able to do this with predefined frequency. Use it as tensorflow Callback
    (@see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/callbacks/Callback)

    Expected formats of the data going thru ANN:
    - input is a vector representing single pixel
    - output is pixel intensity [0-255] (corresponding with input pixel)
    When above conditions are fulfilled, callback creates PNG monochrome image in defined folder with defined frequency.
    """

    def __init__(self, freq, out_dir, inputs, img_shape):
        """
        Configuration of InterPredictCallback is
        :param freq: int
            Number of epochs between each prediction. Warn: prediction is very time consuming operation it is
            recommended to start with higher frequencies or split training according to freq and build images manually
        :param out_dir: string
            Path to the directory where produced predictions will be stored (as PNG images)
        :param inputs: 2D ndarray
            Input vectors (one vector per pixel) i.e. n x D (where n is number of pixels and D dimensionality).
            Typically one can use here data formatted as the training dataset
        :param img_shape: (int, int)
            Original shape of the image (width, height)
        """
        super(InterPredictCallback, self)
        self.freq = freq
        self.out_dir = out_dir
        self.inputs = inputs
        self.image_shape = img_shape
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 0:
            return

        imageio.imwrite(
            os.path.join(self.out_dir, f"{epoch}.png"),
            (self.model.predict(self.inputs).reshape(self.image_shape) * 255).astype(np.uint8)
        )
