from tensorflow.contrib.estimator import RNNEstimator
import tensorflow as tf
from model_data_definition import ModelDataDefinition
from data_directory import DataDirectory

class Model:

    def __init__(self, data_definition: ModelDataDefinition):        
        # The estimator
        self.estimator = RNNEstimator(
            head = data_definition.get_model_head(),
            sequence_feature_columns = data_definition.get_model_input_columns(),
            num_units=[64, 64], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir='model'
        )

    def export_model(self, export_dir_path : str , data_definition: ModelDataDefinition):
        """ Exports the model to the given directory """
        self.estimator.export_savedmodel( export_dir_path , lambda:data_definition.serving_input_receiver_fn() , strip_default_attrs=True)

    def train_model(self, train_data : DataDirectory , eval_data : DataDirectory , data_definition : ModelDataDefinition ):
        # TODO: When to stop ???
        while True:
            print("Training...")
            self.estimator.train( input_fn=lambda:train_data.get_tf_input_fn( data_definition ) )

            print("Evaluating...")
            result = self.estimator.evaluate( input_fn=lambda:eval_data.get_tf_input_fn( data_definition ) )
            print("Evaluation: ", result)