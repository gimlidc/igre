import tensorflow as tf
from src.tftools.custom_constraint import TanhConstraint
from src.config.tools import get_config




class RDistortionLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=3, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RDistortionLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.k1 = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                      trainable=(1 & trainable == 1),
                                      #constraint=TanhConstraint()
                                      )
        self.k2 = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                      trainable=(2 & trainable == 2),
                                      #constraint=TanhConstraint()
                                      )
        self.k3 = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                      trainable=(4 & trainable == 4),
                                      #constraint=TanhConstraint()
                                      )
        self.center = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                      trainable=trainable == 7,
                                      #constraint=TanhConstraint()
                                      )

    def call(self, coords, **kwargs):
        # x' = x_c + (1 + c_1*r^2 + c_2*r^4 + c_3*r^6)*(x - x_c)
        config = get_config()
        w = config['crop']['size']['width'] - 1
        h = config['crop']['size']['height'] - 1
        idx = tf.cast(coords, tf.float32)
        coords_norm = tf.subtract(tf.divide(tf.multiply(coords, 2.), tf.constant([h, w], dtype=tf.float32)), [1., 1.])
        dist_from_center = tf.subtract(coords_norm, self.center)
        radius = tf.sqrt(tf.add(tf.pow(dist_from_center[:, 0], 2),
                         tf.pow(dist_from_center[:, 1], 2)))
        k1 = tf.multiply(self.k1, config["layer_normalization"]["radial_distortion"])
        k2 = tf.multiply(self.k2, config["layer_normalization"]["radial_distortion"])
        k3 = tf.multiply(self.k3, config["layer_normalization"]["radial_distortion"])
        distortion = tf.add(tf.add(tf.multiply(k1, tf.pow(radius, 2)),
                                   tf.add(tf.multiply(k2, tf.pow(radius, 4)),
                                          tf.multiply(k3, tf.pow(radius, 6))
                                          )
                                   ),
                            1.)
        distortion = tf.reshape(tf.tile(distortion, [2]), [tf.shape(distortion)[0], 2])
        idx = tf.add(self.center,
                     tf.multiply(distortion, dist_from_center))
        coords_transformed = tf.divide(tf.multiply(tf.add(idx, [1., 1.]), tf.constant([h, w], dtype=tf.float32)), 2.)

        return coords_transformed
