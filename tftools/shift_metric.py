import tensorflow.python.keras.callbacks as cbks


class ShiftMetrics(cbks.Callback):

    def __init__(self):
        super().__init__()
        self.bias_history = []
        self.loss_history = []

    # def on_epoch_begin(self, epoch, logs=None):
        # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        self.bias_history.append(self.model.layers[1].get_weights())
        self.loss_history.append(logs['loss'])
