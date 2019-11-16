import tensorflow as tf
import numpy as np


class RegistrationLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RegistrationLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.shift = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True)
        # rotation by an angle
        self.rotation = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                        trainable=True)
        # scale in x and in y
        self.scale = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True)
        # perspective sheer
        self.perspective = self.add_weight(name='multi', shape=(3,), dtype=tf.float32, initializer='zeros',
                                           trainable=False)

    def call(self, coords, **kwargs):
        affine = tf.concat(
            [tf.concat([tf.multiply(tf.reshape([[tf.math.cos(self.rotation), -tf.math.sin(self.rotation)],
                                                [tf.math.sin(self.rotation), tf.math.cos(self.rotation)]],
                                               (2, 2)), [[self.scale[0], 0], [0, self.scale[1]]]),
                        tf.reshape(self.shift, (2, 1))], axis=1),
             tf.constant([[0, 0, 1]], dtype=tf.float32)], axis=0)

        # idx = a.*x + b.*y + c where x, y is expected in range [width, height]
        # its very difficult to get the originally used a, b and c so we are estimating
        # the a, b and c of the reverse transform for now
        batch_size = tf.shape(coords)[0]
        idx = tf.concat([tf.cast(coords, tf.float32), tf.ones((batch_size, 1))], axis=1)

        # [x',y'] = ([x,y].[1+a_x,0+b_x]) , ([x,y].[0+a_y,1+b_y])

        # ax + by +c
        affined = tf.einsum("ij,kj->ik", idx, affine)

        # dx + ey + 1
        denominator = tf.einsum("ij,j->i", idx, self.perspective)

        # together we have projective transform (a.*x + b.*y + c)/(d.*x + e.*y + 1)
        idx = tf.concat([tf.reshape(tf.divide(affined[:, 0], denominator), (batch_size, 1)),
                         tf.reshape(tf.divide(affined[:, 1], denominator), (batch_size, 1))], axis=1)

        return idx
