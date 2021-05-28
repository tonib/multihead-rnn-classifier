from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
import tensorflow as tf
import json

class BasePredictor(tf.Module):

    def __init__(self, data_definition: ModelDataDefinition, model: tf.keras.Model):
        self.data_definition = data_definition
        self.model = model

    @staticmethod
    def remove_tflite_padding(input):
        """ Removes TF lite padding elements (seems TF lite requires fixed size inputs). Padding element is -1 """
        mask = tf.not_equal(input, -1)
        return tf.boolean_mask(input, mask)
