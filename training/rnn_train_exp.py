from training.base_train import BaseTrain
from dataset.rnn_dataset_exp import RnnDatasetExp
from model.rnn_model_exp import create_rnn_model_exp

import tensorflow as tf

class RnnTrainExp(BaseTrain):
    def __init__(self):
        super().__init__(RnnDatasetExp)

    def create_model(self) -> tf.keras.Model:
        return create_rnn_model_exp(self.data_definition)

    def create_losses(self):
        # Losses for each output (sum of all will be minimized)
        self.losses = {}
        for output_column_name in self.data_definition.output_columns:
            self.losses[output_column_name] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def create_metrics(self):
        self.metrics = ['accuracy']
