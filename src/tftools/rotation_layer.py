import tensorflow as tf
import numpy as np
from src.tftools.custom_constraint import TanhConstraint
from src.config.tools import get_config


# handles +-5 deg


class RotationLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RotationLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # rotation by an angle in radians
        self.rotation = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                        trainable=True,
                                        # constraint=TanhConstraint()
                                        )

    def call(self, coords, **kwargs):
        config = get_config()
        rotation = tf.multiply(self.rotation, config["layer_normalization"]["rotation"])
        affine = tf.reshape([[tf.math.cos(rotation), -tf.math.sin(rotation)],
                             [tf.math.sin(rotation), tf.math.cos(rotation)]], (2, 2))

        # cos(a)  -sin(a)
        # sin(a)  cos(a)
        # rotation matrix,  rotating by angle a

        idx = tf.cast(coords, tf.float32)

        idx = tf.einsum("ij,kj->ik", idx, affine)

        return idx

    def set_trainable(self, value):
        self.rotation._trainable = value