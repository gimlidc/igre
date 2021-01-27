import tensorflow as tf
import numpy as np
from src.tftools.custom_constraint import MinMaxConstraint

shear_multi = 0.01


class ShearLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ShearLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # rotation by an angle
        self.shear = self.add_weight(name='shear', shape=(1,), dtype=tf.float32, initializer='zeros',
                                     trainable=True,
                                     constraint=MinMaxConstraint(-1., 1.)
                                     )

    def call(self, coords, **kwargs):
        shear_x = tf.reshape([[1., tf.multiply(self.shear[0], shear_multi)],
                             [0., 1.]], (2, 2))

        idx = tf.cast(coords, tf.float32)

        idx = tf.einsum("ij,kj->ik", idx, shear_x)

        return idx
