import tensorflow as tf
from keras import backend as K


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

    learning_rate = tf.train.exponential_decay(
      config["learning_rate"],                # Base learning rate.
      K.variable(1) * batch_size,      # Current index into the dataset.
      2048,          # Decay step.
      0.8,                # Decay rate.
      staircase=False)
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  beta1=config["beta1"],
                                  beta2=config["beta2"])
