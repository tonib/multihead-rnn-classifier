from training.base_train import BaseTrain
from dataset.transformer_dataset import TransformerDataset
from model.mingpt.model_adapted import GPT, GPT1Config

import tensorflow as tf

# TODO: Loss reduction?

class GptTrain(BaseTrain):

    def __init__(self):
        super().__init__(TransformerDataset)

    def create_model(self) -> tf.keras.Model:
        return GPT(GPT1Config(), self.data_definition)

    def print_summary(self):
        # Does not work for keras.Model subclassing. model.build() neither
        # model.summary()
        # The only way I have found to get a summary is to feed a real sample, this seems to compile the model. After that,
        # summary can be printed
        build_model_ds = self.train_dataset.dataset.take(1)
        for input, output in build_model_ds:
            self.model(input)
            self.model.summary()

