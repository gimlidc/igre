import tensorflow as tf


class ShiftLayer(tf.keras.layers.Layer):

    def __init__(self, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(ShiftLayer, self).__init__(**kwargs)
        tf.compat.v1.constant_initializer()
        # shift in pixels
        self.shift = self.add_weight(name='multi', shape=(2,), dtype=tf.float32, initializer='zeros',
                                     trainable=True)

    def call(self, coords, **kwargs):
        # [x',y'] = [x + c_x, y + c_y]
        idx = tf.cast(coords, tf.float32)
        idx = tf.add(idx, self.shift)
        return idx
