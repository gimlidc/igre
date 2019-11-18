import tensorflow as tf
import numpy as np


class ScaleLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ScaleLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()

        # scale in x and in y
        self.scale = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=False)

    def call(self, coords, **kwargs):
        affine = tf.concat(
            [tf.concat([tf.reshape([[self.scale[0], 0], [0, self.scale[1]]], (2, 2)),
                       tf.reshape(tf.constant([0, 0], dtype=tf.float32), (2, 1))], axis=1),
             tf.constant([[0, 0, 1]], dtype=tf.float32)], axis=0)

        # s_x  0    0
        # 0    s_y  0
        # 0    0    1
        # scale matrix,  scaling by s_x, s_y

        batch_size = tf.shape(coords)[0]
        idx = tf.concat([tf.cast(coords, tf.float32), tf.ones((batch_size, 1))], axis=1)

        # [x',y'] = [x*s_x, y*s_y]
        scaled = tf.einsum("ij,kj->ik", idx, affine)

        idx = tf.concat([tf.reshape(scaled[:, 0], (batch_size, 1)),
                         tf.reshape(scaled[:, 1], (batch_size, 1))], axis=1)

        return idx
