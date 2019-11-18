import tensorflow as tf
import numpy as np


class RotationLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RotationLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # rotation by an angle
        self.rotation = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                        trainable=False)

    def call(self, coords, **kwargs):
        affine = tf.concat(
            [tf.concat([tf.reshape([[tf.math.cos(self.rotation), -tf.math.sin(self.rotation)],
                                   [tf.math.sin(self.rotation), tf.math.cos(self.rotation)]], (2, 2)),
                        tf.reshape(tf.constant([0, 0], dtype=tf.float32), (2, 1))], axis=1),
             tf.constant([[0, 0, 1]], dtype=tf.float32)], axis=0)

        # cos(a)  -sin(a)    0
        # sin(a)  cos(a)     0
        # 0       0          1
        # rotation matrix,  rotating by angle a

        batch_size = tf.shape(coords)[0]
        idx = tf.concat([tf.cast(coords, tf.float32), tf.ones((batch_size, 1))], axis=1)

        # [x',y'] = [x,y].[cos(a),-sin(a)]) , ([x,y].[sin(a),cos(a)]
        rotated = tf.einsum("ij,kj->ik", idx, affine)

        idx = tf.concat([tf.reshape(rotated[:, 0], (batch_size, 1)),
                         tf.reshape(rotated[:, 1], (batch_size, 1))], axis=1)

        return idx
