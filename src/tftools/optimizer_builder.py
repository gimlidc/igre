import tensorflow as tf
from tensorflow.keras import backend as K

# todo: model.predict na vstup -> dostaneme obrazek a koukat se na to


class FLAGS():
    learning_rate = 0.001
    train_batch_size = 2048
    use_learning_rate_warmup = True
    learning_rate_decay = 0.1
    cold_epochs = 10
    warmup_epochs = 10
    learning_rate_decay_epochs = 1000
    NUM_TRAIN_IMAGES = 50000


def build_optimizer(config, batch_size):
    """
    Factory for building optimizer from configuration file. Now supported only adam optimizer with defined lr, beta1,
    beta2.

    :param config:
        dict with defined parameters of optimizer
    :return:
        generated optimizer
    """
    print("Building optimizer: " + str(config))
    learning_rate = tf.compat.v1.train.exponential_decay(
        config["learning_rate"],  # Base learning rate.
        K.variable(1) * batch_size,  # Current index into the dataset.
        5000,  # Decay step.
        0.6,  # Decay rate.
        staircase=False)

    initial_learning_rate = FLAGS.learning_rate * FLAGS.train_batch_size / 256
    adj_initial_learning_rate = initial_learning_rate
    if FLAGS.use_learning_rate_warmup:
        warmup_decay = FLAGS.learning_rate_decay ** (
                (FLAGS.warmup_epochs + FLAGS.cold_epochs) /
                FLAGS.learning_rate_decay_epochs)
        adj_initial_learning_rate = initial_learning_rate * warmup_decay

    final_learning_rate = 0.0000001 * initial_learning_rate

    train_op = None
    training_active = True
    if training_active:
        batches_per_epoch = FLAGS.NUM_TRAIN_IMAGES / FLAGS.train_batch_size
        global_step = tf.compat.v1.train.get_or_create_global_step()
        current_epoch = tf.cast(
            (tf.cast(global_step, tf.float32) / batches_per_epoch), tf.int32)

        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=int(FLAGS.learning_rate_decay_epochs * batches_per_epoch),
            decay_rate=FLAGS.learning_rate_decay,
            staircase=True)

        if FLAGS.use_learning_rate_warmup:
            wlr = 0.1 * adj_initial_learning_rate
            wlr_height = tf.cast(
                0.9 * adj_initial_learning_rate /
                (FLAGS.warmup_epochs + FLAGS.learning_rate_decay_epochs - 1),
                tf.float32)
            epoch_offset = tf.cast(FLAGS.cold_epochs - 1, tf.int32)
            exp_decay_start = (FLAGS.warmup_epochs + FLAGS.cold_epochs +
                               FLAGS.learning_rate_decay_epochs)
            lin_inc_lr = tf.add(
                wlr, tf.multiply(
                    tf.cast(tf.subtract(current_epoch, epoch_offset), tf.float32),
                    wlr_height))
            learning_rate = tf.where(
                tf.greater_equal(current_epoch, FLAGS.cold_epochs),
                (tf.where(tf.greater_equal(current_epoch, exp_decay_start),
                          learning_rate, lin_inc_lr)),
                wlr)

        # Set a minimum boundary for the learning rate.
        learning_rate = tf.maximum(
            learning_rate, final_learning_rate, name='learning_rate')

    return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                            beta1=config["beta1"],
                                            beta2=config["beta2"])


def build_refining_optimizer(config):
    return tf.keras.optimizers.SGD(learning_rate=config["learning_rate"],
                                   decay=config["decay"],
                                   momentum=config["momentum"],
                                   nesterov=True)
