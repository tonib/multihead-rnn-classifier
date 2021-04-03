
from model_data_definition import ModelDataDefinition

from training.gpt_train import GptTrain
from dataset.transformer_dataset import TransformerDataset
from model.mingpt.model_adapted import GPT
from predict.gpt_predictor import GptPredictor

from training.rnn_train import RnnTrain
from dataset.rnn_dataset import RnnDataset
from model.rnn_model import create_rnn_model
from predict.rnn_predictor import RnnPredictor

class ModelDefinition:
    """
    Defines classes and functions used by each model type
    """

    def __init__(self):

        self.data_definition = ModelDataDefinition.from_file()

        if self.data_definition.model_type == "gpt":
            self.trainer_class = GptTrain
            self.dataset_class = TransformerDataset
            self.create_model_function = GPT.create_model
            self.predictor_class = GptPredictor

        elif self.data_definition.model_type == "rnn":
            self.trainer_class = RnnTrain
            self.dataset_class = RnnDataset
            self.create_model_function = create_rnn_model
            self.predictor_class = RnnPredictor

        else:
            raise Exception("Unknown model type " + self.data_definition.model_type)
