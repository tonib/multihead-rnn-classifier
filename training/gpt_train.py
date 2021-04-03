from training.base_train import BaseTrain
from dataset.transformer_dataset import TransformerDataset
from model.mingpt.model_adapted import GPT

import tensorflow as tf

# TODO: Loss reduction?

class GptTrain(BaseTrain):

    def __init__(self):
        super().__init__(TransformerDataset)

    def create_model(self) -> tf.keras.Model:
        return GPT.create_model(self.data_definition)

    def create_losses(self):
        # Losses for each output (sum of all will be minimized)
        self.losses = {}
        for output_column_name in self.data_definition.output_columns:
            self.losses[output_column_name] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)

    def print_summary(self):
        # Does not work for keras.Model subclassing. model.build() neither
        # model.summary()
        # The only way I have found to get a summary is to feed a real sample, this seems to compile the model. After that,
        # summary can be printed
        build_model_ds = self.train_dataset.dataset.take(1)
        for input, output in build_model_ds:
            self.model(input)
            self.model.summary()

