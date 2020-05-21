import tensorflow as tf
from src.tftools.custom_constraint import TanhConstraint
from src.config.tools import get_config
import src.config.image_info as ii



class RDistortionLayer3(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(RDistortionLayer3, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.k3 = self.add_weight(name='multi', shape=(1,), dtype=tf.float32, initializer='zeros',
                                      trainable=trainable,
                                      #constraint=TanhConstraint()
                                      )

    def call(self, coords, **kwargs):
        # x' = x_c + (1 + c_1*r^2 + c_2*r^4 + c_3*r^6)*(x - x_c)
        config = get_config()
        w = ii.image.width - 1
        h = ii.image.height - 1

        crop_w = 0
        crop_h = 0
        if "crop" in config:
            crop_w = config["crop"]["left_top"]["y"]
            crop_h = config["crop"]["left_top"]["x"]

        coords_norm = tf.subtract(tf.divide(tf.multiply(tf.add(coords, tf.constant([crop_h, crop_w], dtype=tf.float32)),
                                                        2.), tf.constant([h, w], dtype=tf.float32)), [1., 1.])
        radius = tf.sqrt(tf.add(tf.pow(tf.subtract(coords_norm[:, 0], ii.image.c_x), 2),
                                tf.pow(tf.subtract(coords_norm[:, 1], ii.image.c_y), 2)))
        k3 = tf.multiply(self.k3, config["layer_normalization"]["radial_distortion_3"])
        distortion = tf.add(tf.multiply(k3, tf.pow(radius, 6)), 1)
        distortion = tf.reshape(tf.tile(distortion, [2]), [tf.shape(distortion)[0], 2])
        idx = tf.add(tf.constant([ii.image.c_x, ii.image.c_y], dtype=tf.float32),
                     tf.multiply(distortion,
                                 tf.subtract(coords_norm, tf.constant([ii.image.c_x, ii.image.c_y], dtype=tf.float32))))
        coords_transformed = tf.subtract(tf.divide(tf.multiply(tf.add(idx, [1., 1.]), tf.constant([h, w], dtype=tf.float32)), 2.),
                                         tf.constant([crop_h, crop_w], dtype=tf.float32))

        return coords_transformed
