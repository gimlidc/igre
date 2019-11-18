import tensorflow as tf
from tensorflow.python.keras.constraints import Constraint
import numpy as np


class MinMaxConstraint(Constraint):
    def __init__(self, mn=-np.inf, mx=np.inf):
        self.minimum = mn
        self.maximum = mx

    def __call__(self, weight):
        return tf.clip_by_value(weight, self.minimum, self.maximum)

    def get_config(self):
        return {'minimum': self.minimum, 'maximum': self.maximum}


class ShiftLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ShiftLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.shift = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True,
                                     constraint=MinMaxConstraint(-25., 25.))

    def call(self, coords, **kwargs):
        # [x',y'] = [x + c_x, y + c_y]
        idx = tf.cast(coords, tf.float32)
        idx = tf.add(idx, self.shift)
        return idx
