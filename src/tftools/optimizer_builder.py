import tensorflow as tf
from tensorflow.keras import backend as K

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
    return tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                            beta1=config["beta1"],
                                            beta2=config["beta2"])


def build_refining_optimizer(config):
    return tf.keras.optimizers.SGD(learning_rate=config["learning_rate"],
                                   decay=config["decay"],
                                   momentum=config["momentum"],
                                   nesterov=True)
