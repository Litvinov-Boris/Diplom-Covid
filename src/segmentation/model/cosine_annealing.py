import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class CosineAnnealingLearningRateSchedule(Callback):
    def __init__(self, n_epochs, n_cycles, lrate_max):
        super(CosineAnnealingLearningRateSchedule, self).__init__()
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max

    def on_epoch_begin(self, epoch, logs=None):
        epochs_per_cycle = np.floor(self.epochs / self.cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        lr = self.lr_max / 2 * (np.cos(cos_inner) + 1)

        K.set_value(self.model.optimizer.lr, lr)
