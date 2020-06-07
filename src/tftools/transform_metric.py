import tensorflow.keras.callbacks as cbks


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


class ScaleMetrics(cbks.Callback):

    def __init__(self):
        super().__init__()
        self.bias_history = []
        self.loss_history = []

    # def on_epoch_begin(self, epoch, logs=None):
        # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        self.bias_history.append(self.model.layers[2].get_weights())
        self.loss_history.append(logs['loss'])


class RotationMetrics(cbks.Callback):

    def __init__(self):
        super().__init__()
        self.bias_history = []
        self.loss_history = []

    # def on_epoch_begin(self, epoch, logs=None):
        # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        self.bias_history.append(self.model.layers[3].get_weights())
        self.loss_history.append(logs['loss'])


class DistortionMetrics(cbks.Callback):

    def __init__(self):
        super().__init__()
        self.bias_history1 = []
        self.bias_history2 = []
        self.bias_history3 = []
        self.loss_history = []

    # def on_epoch_begin(self, epoch, logs=None):
    # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        self.bias_history1.append(list(self.model.layers[4].get_weights()))
        self.bias_history2.append(list(self.model.layers[5].get_weights()))
        self.bias_history3.append(list(self.model.layers[6].get_weights()))
        self.loss_history.append(logs['loss'])


class RDMetrics(cbks.Callback):

    def __init__(self):
        super().__init__()
        self.bias_history = []
        self.loss_history = []

    # def on_epoch_begin(self, epoch, logs=None):
    # print("\n" + str(self.model.layers[1].get_weights()))

    def on_batch_end(self, epoch, logs=None):
        self.bias_history.append(list(self.model.layers[4].get_weights()))
        self.loss_history.append(logs['loss'])
