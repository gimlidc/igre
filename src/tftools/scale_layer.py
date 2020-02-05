import tensorflow as tf
from src.tftools.custom_constraint import MinMaxConstraint, DiminishLearningRate
from src.config.tools import get_config

scale_base = 1.
scale_multi = 0.1
# 0.1 is doable, its not uniform though


class ScaleLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ScaleLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()

        # scale in x and in y
        self.scale = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True,
                                     # constraint=MinMaxConstraint(-1., 1.)  # DiminishLearningRate(1000.))
                                     )

    def call(self, coords, **kwargs):
        config = get_config()
        idx = tf.cast(coords, tf.float32)
        idx = tf.multiply(idx, tf.add(tf.multiply(self.scale, config["layer_normalization"]["scale"]), scale_base))

        return idx
