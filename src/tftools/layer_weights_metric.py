import tensorflow.keras.callbacks as cbks
import numpy as np


class LayerWeightsMetric(cbks.Callback):

    def __init__(self, layer_indexes):
        super().__init__()
        self.layer_indexes = layer_indexes
        self.weights_history = dict()
        for idx in layer_indexes:
            self.weights_history[idx] = []


    # def on_epoch_begin(self, epoch, logs=None):
        # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        for idx in self.layer_indexes:
            self.weights_history[idx].append(self.model.layers[idx].get_weights())

