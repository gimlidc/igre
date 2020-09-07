from tensorflow.keras.callbacks import Callback
import os
import imageio
import numpy as np


class InterPredictCallback(Callback):

    def __init__(self, freq, out_dir, inputs, img_shape):
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
