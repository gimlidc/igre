import tensorflow as tf
from src.tftools.custom_constraint import TanhConstraint
from src.config.tools import get_config

shift_multi = 2000.
# hadles shift +- 20


class ShiftLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ShiftLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.shift = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True,
                                     constraint=TanhConstraint()
                                     )

    def call(self, coords, **kwargs):
        # [x',y'] = [x + c_x, y + c_y]
        config = get_config()
        idx = tf.cast(coords, tf.float32)
        idx = tf.add(idx, tf.multiply((self.shift), config["layer_normalization"]["shift"]))
        return idx

    def set_trainable(self, value):
        self.shift._trainable = value