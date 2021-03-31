from training.base_train import BaseTrain
from dataset.rnn_dataset import RnnDataset
from model.rnn_model import generate_model

import tensorflow as tf

class RnnTrain(BaseTrain):
    def __init__(self):
        super().__init__(RnnDataset)

    def create_model(self) -> tf.keras.Model:
        return generate_model(self.data_definition)
