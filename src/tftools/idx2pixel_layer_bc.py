import tensorflow as tf

global_visible = None


class Idx2PixelBCLayer(tf.keras.layers.Layer):

    def __init__(self, visible, trainable=False, shift_multi=1, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(Idx2PixelBCLayer, self).__init__(**kwargs)
        self.visible = tf.constant(visible, dtype=tf.float32)
        global global_visible
        global_visible = self.visible

    def call(self, coords, **kwargs):
        return bicubic_interpolation(coords)


def reset_visible(stage_data):
    global global_visible
    global_visible = tf.constant(stage_data.copy(), dtype=tf.float32)


# @tf.custom_gradient
def bicubic_interpolation(coords):
    """
        Calculate image pixel intensities from input coordinates by means of bicubic
        interpolation.
        Hermite 1D formula is used first for columns (x axis) and then for row (y axis)
        in order to obtain 2D interpolation.
        This should be automatically differentiable by TF

                    f_00         f_01       f_02        f_03
                     +-----------+-----------+-----------+
                     |                                   |
                     |                                   |
                     |                                   |
                f_10 |          f_11        f_12         | f_13
                     +           +    |x_t   +           +
                     |                |                  |
             col_0   *   col_1   *____x      *col_2      *col_3
                     |             y_t                   |
                f_20 |          f_21        f_22         | f_23
                     +           +           +           +
                     |                                   |
                     |                                   |
                     |                                   |
                     |                                   |
                     +-----------+-----------+-----------+
                    f_30         f_31       f_32        f_33


    """

    def _hermite(A, B, C, D, t):
        """
        Compute the Cubic Hermite spline using the Catmull-Rom formula
        :param A: f(n-1)
        :param B: f(n)
        :param C: f(n+1)
        :param D: f(n+2)
        :param t: x - floor(x)
        :return: f(x) = f(n+t)
        """
        a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
        b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
        c = A * (-0.5) + C * 0.5
        d = B

        return tf.einsum("i,ij->ij", t, (tf.einsum("i,ij->ij", t, (tf.einsum("i,ij->ij", t, a))))) + \
            tf.einsum("i,ij->ij", t, (tf.einsum("i,ij->ij", t, b))) + \
            tf.einsum("i,ij->ij", t, c) + d

    visible = global_visible
    coords = tf.maximum(coords, tf.ones(tf.shape(visible.shape[:-1]), dtype=tf.float32))
    coords = tf.minimum(coords, tf.subtract(tf.cast(visible.shape[:-1], dtype=tf.float32), 3))
    idx_low = tf.floor(coords)
    delta = tf.cast(tf.subtract(coords, idx_low), dtype=tf.float32)
    x_t = delta[:, 0]
    y_t = delta[:, 1]

    f_00 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, -1]), dtype=tf.int32))
    f_10 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [ 0, -1]), dtype=tf.int32))
    f_20 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, -1]), dtype=tf.int32))
    f_30 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, -1]), dtype=tf.int32))

    f_01 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 0]), dtype=tf.int32))
    f_11 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [ 0, 0]), dtype=tf.int32))
    f_21 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 0]), dtype=tf.int32))
    f_31 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, 0]), dtype=tf.int32))

    f_02 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 1]), dtype=tf.int32))
    f_12 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [ 0, 1]), dtype=tf.int32))
    f_22 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 1]), dtype=tf.int32))
    f_32 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, 1]), dtype=tf.int32))

    f_03 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 2]), dtype=tf.int32))
    f_13 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [ 0, 2]), dtype=tf.int32))
    f_23 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+1, 2]), dtype=tf.int32))
    f_33 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [+2, 2]), dtype=tf.int32))

    col_0 = _hermite(f_00, f_10, f_20, f_30, x_t)
    col_1 = _hermite(f_01, f_11, f_21, f_31, x_t)
    col_2 = _hermite(f_02, f_12, f_22, f_32, x_t)
    col_3 = _hermite(f_03, f_13, f_23, f_33, x_t)

    interpolation = _hermite(col_0, col_1, col_2, col_3, y_t)

    # def grad(dy):
    #     x = tf.einsum("jk,jk->k", a_matrix[:, 1, :], y_vec) + \
    #         tf.multiply(2 * dy[0], tf.einsum("jk,jk->k", a_matrix[:, 2, :], y_vec)) + \
    #         tf.multiply(3 * tf.pow(dy[0], 2), tf.einsum("jk,jk->k", a_matrix[:, 3, :], y_vec))
    #     y = tf.einsum("jk,jk->k", a_matrix[1, :, :], x_vec) + \
    #         tf.multiply(2 * dy[1], tf.einsum("jk,jk->k", a_matrix[2, :, :], x_vec)) + \
    #         tf.multiply(3 * tf.pow(dy[1], 2), tf.einsum("jk,jk->k", a_matrix[3, :, :], x_vec))
    #     return tf.stack([x, y], axis=1)

    # coords_off_boundary = tf.greater(tf.cast(coords, dtype=tf.float32), tf.cast(visible.shape[:-1], dtype=tf.float32))
    # boundary_condition = tf.logical_or(coords_off_boundary[:, 0], coords_off_boundary[:, 0])
    # masked = tf.where(boundary_condition, tf.zeros(tf.shape(interpolation)), interpolation)

    return interpolation  #, grad



