import tensorflow as tf
from src.tftools.custom_constraint import TanhConstraint
from src.config.tools import get_config




class RDistortionLayer2(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RDistortionLayer2, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.k2 = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                      trainable=trainable,
                                      #constraint=TanhConstraint()
                                      )

    def call(self, coords, **kwargs):
        # x' = x_c + (1 + c_1*r^2 + c_2*r^4 + c_3*r^6)*(x - x_c)
        config = get_config()
        w = config['crop']['size']['width'] - 1
        h = config['crop']['size']['height'] - 1
        coords_norm = tf.subtract(tf.divide(tf.multiply(coords, 2.), tf.constant([h, w], dtype=tf.float32)), [1., 1.])
        radius = tf.sqrt(tf.add(tf.pow(coords_norm[:, 0], 2),
                         tf.pow(coords_norm[:, 1], 2)))
        k2 = tf.multiply(self.k2, config["layer_normalization"]["radial_distortion_2"])
        distortion = tf.add(tf.multiply(k2, tf.pow(radius, 4)), 1)
        distortion = tf.reshape(tf.tile(distortion, [2]), [tf.shape(distortion)[0], 2])
        idx = tf.multiply(distortion, coords_norm)
        coords_transformed = tf.divide(tf.multiply(tf.add(idx, [1., 1.]), tf.constant([h, w], dtype=tf.float32)), 2.)

        return coords_transformed
