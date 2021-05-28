
from model_data_definition import ModelDataDefinition

from training.gpt_train import GptTrain
from dataset.transformer_dataset import TransformerDataset
from model.mingpt.model_adapted import GPT
from predict.gpt_predictor import GptPredictor, GptPredictorLite

from training.rnn_train import RnnTrain
from dataset.rnn_dataset import RnnDataset
from model.rnn_model import create_rnn_model
from predict.rnn_predictor import RnnPredictor, RnnPredictorLite

from training.rnn_train_exp import RnnTrainExp
from dataset.rnn_dataset_exp import RnnDatasetExp
from model.rnn_model_exp import create_rnn_model_exp
from predict.rnn_predictor_exp import RnnPredictorExp

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
            self.tflite_predictor_class = GptPredictorLite

        elif self.data_definition.model_type == "rnn":
            self.trainer_class = RnnTrain
            self.dataset_class = RnnDataset
            self.create_model_function = create_rnn_model
            self.predictor_class = RnnPredictor
            self.tflite_predictor_class = RnnPredictorLite

        elif self.data_definition.model_type == "exp":
            self.trainer_class = RnnTrainExp
            self.dataset_class = RnnDatasetExp
            self.create_model_function = create_rnn_model_exp
            self.predictor_class = RnnPredictorExp
            self.tflite_predictor_class = None # Not implemented yet

        else:
            raise Exception("Unknown model type " + self.data_definition.model_type)
