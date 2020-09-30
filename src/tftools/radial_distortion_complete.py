import tensorflow as tf
from src.tftools.custom_constraint import TanhConstraint
from tensorflow.keras.constraints import MinMaxNorm


class RDCompleteLayer(tf.keras.layers.Layer):

    def __init__(self, cx, cy, width, height, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RDCompleteLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.k1 = self.add_weight(name='multi', shape=(1,), dtype=tf.float64, initializer='zeros',
                                  trainable=trainable,
                                  constraint=TanhConstraint()
                                  )
        self.k2 = self.add_weight(name='multi', shape=(1,), dtype=tf.float64, initializer='zeros',
                                  trainable=trainable,
                                  constraint=TanhConstraint()
                                  )
        self.k3 = self.add_weight(name='multi', shape=(1,), dtype=tf.float64, initializer='zeros',
                                  trainable=trainable,
                                  constraint=TanhConstraint()
                                  )
        self.c = tf.constant([cx, cy], dtype=tf.float64)
        self.size = tf.constant([height-1, width-1], dtype=tf.float64)

    def __normalize_coordinates(self, coords):
        # 2 * (coord - size/2) / size
        return tf.divide(
            tf.multiply(
                tf.subtract(coords,
                            tf.divide(self.size, tf.constant(2, dtype=tf.float64))
                            ),
                tf.constant(2, dtype=tf.float64)),
            self.size
        )

    def __denormalize_coordinates(self, coords):
        return tf.add(tf.multiply(coords, self.size / 2), self.size / 2)

    def call(self, coords, **kwargs):
        # x' = x_c + (1 + c_1*r^2 + c_2*r^4 + c_3*r^6)*(x - x_c)
        coords_norm = self.__normalize_coordinates(coords)

        radius = tf.sqrt(
            tf.reduce_sum(
                tf.pow(
                    tf.subtract(coords_norm, self.c),
                    tf.constant(2, dtype=tf.float64)
                ),
                1
            )
        )

        distortion = tf.add(
            tf.add(
                tf.add(
                    tf.multiply(self.k1, tf.pow(radius, tf.constant(2, dtype=tf.float64))),
                    tf.multiply(self.k2, tf.pow(radius, tf.constant(4, dtype=tf.float64)))
                ),
                tf.multiply(self.k3, tf.pow(radius, tf.constant(6, dtype=tf.float64)))),
            tf.constant(1, dtype=tf.float64))

        # distortion = tf.reshape(tf.tile(distortion, [2]), [tf.shape(distortion)[0], 2])
        idx = tf.add(
            self.c,
            tf.multiply(
                distortion[:, tf.newaxis],
                tf.subtract(
                    coords_norm,
                    self.c
                )
            )
        )

        return self.__denormalize_coordinates(idx)
