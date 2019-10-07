import tensorflow as tf


def build_optimizer(config):
    """
    Factory for building optimizer from configuration file. Now supported only adam optimizer with defined lr, beta1,
    beta2.

    :param config:
        dict with defined parameters of optimizer
    :return:
        generated optimizer
    """
    print("Building optimizer: " + str(config))
    return tf.train.AdamOptimizer(learning_rate=config["learning_rate"],
                                  beta1=config["beta1"],
                                  beta2=config["beta2"])


def build_refining_optimizer(config):
    return tf.keras.optimizers.SGD(learning_rate=config["learning_rate"],
                                   decay=config["decay"],
                                   momentum=config["momentum"],
                                   nesterov=True)
