from tensorflow.contrib.estimator import RNNEstimator
from tensorflow.contrib.estimator import multi_head
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.ops.losses import losses
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.feature_column as feature_column
from model_data_definition import ModelDataDefinition

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