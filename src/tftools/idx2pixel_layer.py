import tensorflow as tf
from utils import shift_multi, shift_multi_2

global_visible = None


class Idx2PixelLayer(tf.keras.layers.Layer):

    def __init__(self, visible, trainable=True, **kwargs):
        """
        :param visible: one dimension of visible image (for this dimension [x,y] will be computed)
        """
        super(Idx2PixelLayer, self).__init__(**kwargs)
        self.shift = self.add_weight(name='shift', shape=(2,), dtype=tf.float32,
                                     initializer='zeros',
                                     trainable=True)
        self.a_x = self.add_weight(name='multi', shape=(1,), dtype=tf.float32,
                                          initializer='zeros',
                                          trainable=True)
        self.a_y = self.add_weight(name='multi', shape=(1,), dtype=tf.float32,
                                     initializer='zeros',
                                     trainable=True)
        self.b_x = self.add_weight(name='multi', shape=(1,), dtype=tf.float32,
                                     initializer='zeros',
                                     trainable=True)
        self.b_y = self.add_weight(name='multi', shape=(1,), dtype=tf.float32,
                                     initializer='zeros',
                                     trainable=True)
        self.multi_denom = self.add_weight(name='multi', shape=(4,), dtype=tf.float32,
                                           initializer='zeros',
                                           trainable=False)
        self.visible = tf.constant(visible, dtype=tf.float32)
        global global_visible
        global_visible = self.visible

    def call(self, coords, **kwargs):
        # idx = a.*x + b.*y + c where x, y is expected in range [width, height]
        # its very difficult to get the originally used a, b and c so we are estimating
        # the a, b and c of the reverse transform for now
        idx = tf.cast(coords, tf.float32)

        # [x',y'] = ([x,y].[1+a_x,0+b_x]) , ([x,y].[0+a_y,1+b_y])

        tl = tf.constant([[1.0, 0.0], [0.0, 0.0]])
        tr = tf.constant([[0.0, 1.0], [0.0, 0.0]])
        bl = tf.constant([[0.0, 0.0], [1.0, 0.0]])
        br = tf.constant([[0.0, 0.0], [0.0, 1.0]])

        x_mult = tf.add(tf.add(tl, tf.multiply(tl, tf.divide(self.a_x, shift_multi))), tf.multiply(tr, tf.divide(self.b_x, shift_multi)))
        y_mult = tf.add(tf.add(br, tf.multiply(br, tf.divide(self.b_y, shift_multi))), tf.multiply(bl, tf.divide(self.a_y, shift_multi)))
        mult = tf.add(x_mult, y_mult)

        idx = tf.einsum('ij,kj->ki', mult, idx)
        idx = tf.add(idx, self.shift)

        # denominator is d.*x + e.*y + 1
        denominator = tf.add(tf.multiply(idx, tf.divide(self.multi_denom[0: 2], shift_multi_2)),
                             tf.multiply(idx, tf.divide(self.multi_denom[2:], shift_multi_2)))
        denominator = tf.add(denominator, [1., 1.])

        # together we have projective transform (a.*x + b.*y + c)/(d.*x + e.*y + 1)
        idx = tf.divide(idx, denominator)

        return linear_interpolation(idx)


# @tf.custom_gradient
def bicubic_interpolation(coords):
    visible = global_visible
    coords = tf.maximum(coords, tf.ones(tf.shape(visible.shape[:-1]), dtype=tf.float32))
    coords = tf.minimum(coords, tf.subtract(tf.cast(visible.shape[:-1], dtype=tf.float32), 3))
    idx_low = tf.floor(coords)
    delta = tf.cast(tf.subtract(coords, idx_low), dtype=tf.float32)

    f00 = tf.gather_nd(visible, tf.cast(idx_low, dtype=tf.int32))
    f01 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 1]), dtype=tf.int32))
    f10 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 0]), dtype=tf.int32))
    f11 = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 1]), dtype=tf.int32))

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

    # ensure that the coordinates are in range [1, max-2] so we can take 2x2 neighbourhood of the coord in the Jacobian
    # TODO: We might do precheck outside this function
    # 0 - 400
    coords = tf.subtract(coords, 1)
    # -1 - 399
    coords = tf.mod(coords, tf.subtract(tf.cast(visible.shape.as_list()[:-1], dtype=tf.float32), 4))
    # 0 - (401-4) 397
    coords = tf.add(coords, 1)
    # 1 - 398
    # we can do coords -1 and +2 now

    # calculate index of top-left point
    idx_low = tf.floor(coords)

    # offset of input coordinates from top-left point
    delta = tf.cast(tf.subtract(coords, idx_low), dtype=tf.float32)
    # coords are the size of (batch, 2), delta as well

    top_left = tf.gather_nd(visible, tf.cast(idx_low, tf.int32))
    top_right = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 0]), tf.int32))
    bottom_left = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 1]), tf.int32))
    bottom_right = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 1]), tf.int32))
    # these values are of size of [batch_size, input_dimensions]

    # NOTE: why r + d(l-r) and not vice versa:
    # IF we do that and then flip x and y axis (which we did) it cancels out
    # calculate interpolations on the sides of the square
    # tensor sizes: einsum = [batch, 1] * [batch, input_dims] -> [batch, input_dims] (input_dims = 1)
    mid_bottom = tf.add(bottom_right, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(bottom_left, bottom_right)))
    mid_top = tf.add(top_right, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(top_left, top_right)))

    # NOTE: reason for mid left/right?
    mid_left = tf.add(bottom_left, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(top_left, bottom_left)))
    mid_right = tf.add(bottom_right, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(top_right, bottom_right)))

    interpolation = tf.add(mid_bottom, tf.einsum("i,ij->ij", delta[:, 1],
                                                 tf.subtract(mid_top, mid_bottom)))

    def compute_2x2_jacobian():
        # This will produce Jacobian of size [batch_size, 2, input_dims]
        # Take bigger neighbourhood around the coord
        ttl = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, -1]), tf.int32))
        ttr = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, -1]), tf.int32))
        bbl = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [0, 2]), tf.int32))
        bbr = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [1, 2]), tf.int32))
        tll = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 0]), tf.int32))
        trr = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [2, 0]), tf.int32))
        bll = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [-1, 1]), tf.int32))
        brr = tf.gather_nd(visible, tf.cast(tf.add(idx_low, [2, 1]), tf.int32))

        mid_bb = tf.add(bbr, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(bbl, bbr)))
        mid_tt = tf.add(ttr, tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(ttl, ttr)))
        mid_ll = tf.add(bll, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(tll, bll)))
        mid_rr = tf.add(brr, tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(trr, brr)))

        d_x_r = tf.subtract(mid_rr, mid_right)
        d_x_c = tf.subtract(mid_right, mid_left)
        d_x_l = tf.subtract(mid_left, mid_ll)
        d_y_t = tf.subtract(mid_top, mid_tt)
        d_y_c = tf.subtract(mid_bottom, mid_top)
        d_y_b = tf.subtract(mid_bb, mid_bottom)

        # Weighted average of the derivatives
        d_x = tf.multiply(tf.add(d_x_r, d_x_l), 0.5)
        d_x = tf.multiply(tf.add(d_x, d_x_c), 0.5)
        d_y = tf.multiply(tf.add(d_y_t, d_y_b), 0.5)
        d_y = tf.multiply(tf.add(d_y, d_y_c), 0.5)
        return d_x, d_y

    d_c_x, d_c_y = compute_2x2_jacobian()
    # d_a_x = tf.einsum("ij,i->ij", d_c_x, coords[:,0])
    # d_a_y = tf.einsum("ij,i->ij", d_c_y, coords[:,0])
    # d_b_x = tf.einsum("ij,i->ij", d_c_x, coords[:,1])
    # d_b_y = tf.einsum("ij,i->ij", d_c_y, coords[:,1])
    jacob = tf.stack([d_c_x, d_c_y], axis=1)

    # jacob = get_jacobian(visible, idx_low, delta)

    def grad(dy):
        """ This method should return tensor of gradients [batch_size, 6]"""
        return tf.multiply(tf.einsum("ijk,ik->ij", jacob, dy), grad_multiplier)

    return interpolation, grad


def get_jacobian(image, index, delta):
    """
    Computes Jacobian over 4 x 4 neighbourhood.
    Requires index >=1 and <= image.size - 2 -- ensured up in the modulo shenanigans.
    Simple difference between two neighbouring pixels in purely x or y axis.

    :param image: image... "duh"
    :param index: closest top left pixel to the point in which we are computing Jacobian
    :param delta: offset of point from index
    :return: Jacobian... "duh"
    """

    # TODO: complete rework
    # not possible the elegant way because of !@#$ tf mechanics

    # neighbourhood_indices = tf.constant(np.array([[[-1, -1], [-1, 0], [-1, 1], [-1, 2]],
    #                                   [[ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2]],
    #                                   [[ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2]],
    #                                   [[ 2, -1], [ 2, 0], [ 2, 1], [ 2, 2]]]), dtype=tf.float32)
    #
    # # repeating the neigh. indices to have same length as index (=batch_size)
    # ni = tf.reshape(tf.tile(neighbourhood_indices, [2048, 1, 1]), (2048, 4, 4, 2))
    # print(ni.shape)
    # print(index.shape)
    # print(tf.add(index, ni).shape)
    # # nbh size is [batch_size, 4, 4, 1]
    # nbh = tf.gather_nd(image, tf.cast(tf.add(index, ni), tf.int32))
    # mid_bottom = tf.add(nbh[:, 2, 2], tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(nbh[:, 1, 2], nbh[:, 2, 2])))
    # mid_top = tf.add(nbh[:, 2, 1], tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(nbh[:, 1, 1], nbh[:, 2, 1])))
    # mid_left = tf.add(nbh[:, 1, 2], tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(nbh[:, 1, 1], nbh[:, 1, 2])))
    # mid_right = tf.add(nbh[:, 2, 2], tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(nbh[:, 2, 1], nbh[:, 2, 2])))
    #
    # mid_bottom_2 = tf.add(nbh[:, 2, 3], tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(nbh[:, 1, 3], nbh[:, 2, 3])))
    # mid_top_2 = tf.add(nbh[:, 2, 0], tf.einsum("i,ij->ij", delta[:, 0], tf.subtract(nbh[:, 1, 0], nbh[:, 2, 0])))
    # mid_left_2 = tf.add(nbh[:, 0, 2], tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(nbh[:, 0, 1], nbh[:, 0, 2])))
    # mid_right_2 = tf.add(nbh[:, 3, 2], tf.einsum("i,ij->ij", delta[:, 1], tf.subtract(nbh[:, 3, 1], nbh[:, 3, 2])))
    #
    # dx_right = tf.subtract(mid_right_2, mid_right)
    # dx_center = tf.subtract(mid_right, mid_left)
    # dx_left = tf.subtract(mid_left, mid_left_2)
    # dy_top = tf.subtract(mid_top, mid_top_2)
    # dy_center = tf.subtract(mid_bottom, mid_top)
    # dy_bottom = tf.subtract(mid_bottom_2, mid_bottom)
    #
    # # Weighted average of the derivatives
    # dx = tf.multiply(tf.add(dx_right, dx_left), 0.5)
    # dx = tf.multiply(tf.add(dx, dx_center), 0.5)
    # dy = tf.multiply(tf.add(dy_top, dy_bottom), 0.5)
    # dy = tf.multiply(tf.add(dy, dy_center), 0.5)
    #
    # jacobian = tf.stack([dx, dy], axis=1)

    # return jacobian


def linear_interpolation_1D(x1, x2, delta):
    return
