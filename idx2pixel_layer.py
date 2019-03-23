import tensorflow as tf

global_visible = None


class Idx2PixelLayer(tf.keras.layers.Layer):

    def __init__(self, visible, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(Idx2PixelLayer, self).__init__(**kwargs)
        self.bias = self.add_weight(name='bias', shape=(2,), dtype=tf.float32,
                                    initializer='zeros',
                                    trainable=trainable)
        self.visible = tf.constant(visible, dtype=tf.float32)
        global global_visible
        global_visible = self.visible

    def call(self, coords, **kwargs):
        # idx = [x - bx, y - by], where x, y is expected in range [width, height]
        idx = tf.subtract(tf.cast(coords, tf.float32), self.bias)
        return linear_interpolation(idx)


# @tf.custom_gradient
def bicubic_interpolation(coords):
    visible = global_visible
    coords = tf.maximum(coords, tf.ones(tf.shape(visible.shape[:-1]), dtype=tf.float32))
    coords = tf.minimum(coords, tf.subtract(tf.cast(visible.shape[:-1], dtype=tf.float32), 3))
    idx_low = tf.floor(coords)
    delta = tf.cast(tf.subtract(coords, idx_low), dtype=tf.float32)

    f00 = tf.gather_nd(visible, tf.cast(idx_low, tf.int32))
    f01 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 1]), tf.int32))
    f10 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 0]), tf.int32))
    f11 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 1]), tf.int32))

    fx00 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 0]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 0]), tf.int32)))
    fx10 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 0]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, 0]), tf.int32)))
    fx01 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 1]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 1]), tf.int32)))
    fx11 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 1]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, 1]), tf.int32)))

    fy00 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, -1]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, +1]), tf.int32)))
    fy10 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, -1]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, +1]), tf.int32)))
    fy01 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 0]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 2]), tf.int32)))
    fy11 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 0]), tf.int32)),
                       tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 2]), tf.int32)))

    fxy00 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, -1]), tf.int32)),
                        tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, +1]), tf.int32)))
    fxy10 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, -1]), tf.int32)),
                        tf.gather_nd(visible, tf.cast(tf.add(idx_low, [2, +1]), tf.int32)))
    fxy01 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 0]), tf.int32)),
                        tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 2]), tf.int32)))
    fxy11 = tf.subtract(tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 0]), tf.int32)),
                        tf.gather_nd(visible, tf.cast(tf.add(idx_low, [2, 2]), tf.int32)))

    f_matrix = tf.stack([tf.stack([f00, f01, fy00, fy01], axis=0),
                         tf.stack([f10, f11, fy10, fy11], axis=0),
                         tf.stack([fx00, fx01, fxy00, fxy01], axis=0),
                         tf.stack([fx10, fx11, fxy10, fxy11], axis=0)], axis=1)

    a_inverse = tf.constant([[ 1,  0,  0,  0],
                             [ 0,  0,  1,  0],
                             [-3,  3, -2, -1],
                             [ 2, -2,  1,  1]], dtype=tf.float32)

    a_matrix = tf.einsum("ijk,ij->ijk", tf.einsum("ij,ijk->ijk", a_inverse, f_matrix), tf.transpose(a_inverse))

    x_vec = tf.stack([tf.ones(tf.shape(delta)[0]),
                      delta[:, 0],
                      tf.pow(delta[:, 0], 2),
                      tf.pow(delta[:, 0], 3)], axis=0)
    y_vec = tf.stack([tf.ones(tf.shape(delta)[0]),
                      coords[:, 1],
                      tf.pow(delta[:, 1], 2),
                      tf.pow(delta[:, 1], 3)], axis=0)

    interpolation = tf.einsum("jk,jk->k", tf.einsum("ik,ijk->jk", x_vec, a_matrix), y_vec)

    # def grad(dy):
    #     x = tf.einsum("jk,jk->k", a_matrix[:, 1, :], y_vec) + \
    #         tf.multiply(2 * dy[0], tf.einsum("jk,jk->k", a_matrix[:, 2, :], y_vec)) + \
    #         tf.multiply(3 * tf.pow(dy[0], 2), tf.einsum("jk,jk->k", a_matrix[:, 3, :], y_vec))
    #     y = tf.einsum("jk,jk->k", a_matrix[1, :, :], x_vec) + \
    #         tf.multiply(2 * dy[1], tf.einsum("jk,jk->k", a_matrix[2, :, :], x_vec)) + \
    #         tf.multiply(3 * tf.pow(dy[1], 2), tf.einsum("jk,jk->k", a_matrix[3, :, :], x_vec))
    #     return tf.stack([x, y], axis=1)

    return interpolation #, grad


@tf.custom_gradient
def linear_interpolation(coords):
    """
        Calculate image pixel intensities from input coordinates by means of bilinear
        interpolation. Also calculate corresponding gradients for ANN training.

        'Bottom-left', 'bottom-right', 'top-left', 'top-right' mean the four
        neighboring pixels closest to input coordinates. top/bottom corresponds to the
        first axis coordinate, right/left to the second. Coordinate values increase
        from left to right, top to bottom.


                top_left                               top_right
                          mid_top
                     X-----------------------------------X
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       . x                         |
           mid_left  X.......*...........................| mid_right
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     |       .                           |
                     X-------X---------------------------X
                          mid_bottom
            bottom_left                                bottom_right
    """
    # multiply gradient by factor to slow down learning of 'bias'
    grad_multiplier = tf.constant(1, dtype=tf.float32)
    visible = global_visible

    # calculate index of top-left point
    coords = tf.maximum(coords, tf.zeros(tf.shape(visible.shape[:-1]), dtype=tf.float32))
    idx_low = tf.floor(coords)

    # idx_low at most second-to-last element
    # TODO: here probably we want to extrapolate last/first column/row to infinity (if there is necessity)
    idx_low = tf.minimum(idx_low, tf.subtract(tf.cast(visible.shape[:-1], dtype=tf.float32), 2))

    # offset of input coordinates from top-left point
    delta = tf.cast(tf.subtract(coords, idx_low), dtype=tf.float32)
    # coords are the size of (batch, 2), delta as well

    top_left = tf.gather_nd(visible, tf.cast(idx_low, tf.int32))
    top_right = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 0]), tf.int32))
    bottom_left = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 1]), tf.int32))
    bottom_right = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 1]), tf.int32))
    # these values are of size of [batch_size, input_dimensions]

    # calculate interpolations on the sides of the square
    # tensor sizes: einsum = [batch, 1] * [batch, input_dims] -> [batch, input_dims]
    mid_bottom = tf.add(bottom_right, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(bottom_left, bottom_right)))
    # tensor sizes: einsum = [batch, 1] * [batch, input_dims] -> [batch, input_dims]
    mid_top = tf.add(top_right, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(top_left, top_right)))

    # NOTE: reason for mid left/right?
    mid_left = tf.add(bottom_left, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(top_left, bottom_left)))
    mid_right = tf.add(bottom_right, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(top_right, bottom_right)))

    interpolation = tf.add(mid_bottom, tf.einsum("i,ij->ij", delta[:, 1],
                                                 tf.subtract(mid_top, mid_bottom)))

    # This will produce Jacobian of size [batch_size, 2, input_dims]
    jacob = tf.stack([tf.subtract(mid_right, mid_left), tf.subtract(mid_bottom, mid_top)],
                     axis=1)

    def grad(dy):
        """ This method should return tensor of gradients [batch_size, 2]"""
        return tf.multiply(tf.einsum("ijk,ik->ij", jacob, dy), grad_multiplier)

    return interpolation, grad
