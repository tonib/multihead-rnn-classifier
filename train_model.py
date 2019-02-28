from tensorflow.contrib.estimator import RNNEstimator
import tensorflow as tf
from model_data_definition import ModelDataDefinition
from data_directory import DataDirectory
from prediction_model import PredictionModel
from time import time

class TrainModel:
    """ Model for training """

    def __init__(self, data_definition: ModelDataDefinition):        
        # The estimator
        self.estimator = RNNEstimator(
            head = data_definition.get_model_head(),
            sequence_feature_columns = data_definition.get_model_input_columns(),
            #num_units=[64, 64], # Removed, extra layer reports same results
            num_units=[64], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir='model'
        )

    def export_model(self, data_definition: ModelDataDefinition):
        """ Exports the model to the exports directory """
        self.estimator.export_savedmodel( PredictionModel.EXPORTED_MODELS_DIR_PATH , 
            lambda:data_definition.serving_input_receiver_fn() , strip_default_attrs=True)

    def train_model(self, train_data : DataDirectory , eval_data : DataDirectory , data_definition : ModelDataDefinition ):
        
        epoch = 0
        last_loss = 0
        train_start = time()
        for _ in range(10):
            epoch += 1
            
            epoch_start = time()
            print("Training epoch", epoch, "...")
            self.estimator.train( input_fn=lambda:train_data.get_tf_input_fn( data_definition ) )
            print("Evaluating...")
            result = self.estimator.evaluate( input_fn=lambda:eval_data.get_tf_input_fn( data_definition ) )
            print("Evaluation: ", result)

            new_loss = result['loss']
            if epoch > 1:
                print("Loss decrease:" , ((last_loss-new_loss) / last_loss) * 100 , "%")
            last_loss = new_loss

            print("Epoch time:" , time() - epoch_start , "s")
            print("Total train time:" , time() - train_start , "s")

            print()