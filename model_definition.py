
from model_data_definition import ModelDataDefinition

from training.gpt_train import GptTrain
from dataset.transformer_dataset import TransformerDataset

from training.rnn_train import RnnTrain
from dataset.rnn_dataset import RnnDataset

class ModelDefinition:
    """
    Definitions for available models
    """

    def __init__(self):

        self.data_definition = ModelDataDefinition()

        if self.data_definition.model_type == "gpt":
            self.trainer_class = GptTrain
            self.dataset_class = TransformerDataset

        elif self.data_definition.model_type == "rnn":
            self.trainer_class = RnnTrain
            self.dataset_class = RnnDataset

        else:
            raise Exception("Unknown model type " + self.data_definition.model_type)
