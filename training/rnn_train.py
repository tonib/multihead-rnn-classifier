from training.base_train import BaseTrain
from dataset.rnn_dataset import RnnDataset
from model.rnn_model import create_rnn_model

import tensorflow as tf

class RnnTrain(BaseTrain):
    def __init__(self):
        super().__init__(RnnDataset)

    def create_model(self) -> tf.keras.Model:
        return create_rnn_model(self.data_definition)

    def create_losses(self):
        # Losses for each output (sum of all will be minimized)
        self.losses = {}
        for output_column_name in self.data_definition.output_columns:
            self.losses[output_column_name] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)