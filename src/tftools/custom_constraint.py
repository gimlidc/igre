import tensorflow as tf
from tensorflow.python.keras.constraints import Constraint
import numpy as np


class MinMaxConstraint(Constraint):
    def __init__(self, mn=-np.inf, mx=np.inf):
        self.minimum = mn
        self.maximum = mx

    def __call__(self, weight):
        return tf.clip_by_value(weight, self.minimum, self.maximum)

    def get_config(self):
        return {'minimum': self.minimum, 'maximum': self.maximum}


class DiminishLearningRate(Constraint):
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, weight):
        return tf.add(1., tf.divide(tf.subtract(weight, 1.), self.factor))

    def get_config(self):
        return {'factor': self.factor}


class TanhConstraint(Constraint):

    def __call__(self, weight):
        return tf.tanh(weight)

    def get_config(self):
        return {'tanh constraint'}
