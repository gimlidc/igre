import tensorflow as tf
import numpy as np


class RegistrationLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RegistrationLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()

        # perspective sheer
        self.perspective = self.add_weight(name='multi', shape=(3,), dtype=tf.float32, initializer='zeros',
                                           trainable=False)

    def call(self, coords, **kwargs):

        batch_size = tf.shape(coords)[0]
        idx = tf.concat([tf.cast(coords, tf.float32), tf.ones((batch_size, 1))], axis=1)

        # dx + ey + 1
        denominator = tf.einsum("ij,j->i", idx, self.perspective)

        # together we have projective transform (a.*x + b.*y + c)/(d.*x + e.*y + 1)
        idx = tf.concat([tf.reshape(tf.divide(idx[:, 0], denominator), (batch_size, 1)),
                         tf.reshape(tf.divide(idx[:, 1], denominator), (batch_size, 1))], axis=1)

        return idx
