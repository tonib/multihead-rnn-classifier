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
            #num_units=[80], 
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
        n_tokens = train_data.get_n_total_tokens()
        for _ in range(data_definition.max_epochs):
            epoch += 1
            
            epoch_start = time()
            print("Training epoch", epoch, "...")
            self.estimator.train( input_fn=lambda:train_data.get_tf_input_fn( data_definition ) )
            train_time = time() - epoch_start
            print("Evaluating...")
            result = self.estimator.evaluate( input_fn=lambda:eval_data.get_tf_input_fn( data_definition ) )
            print("Evaluation: ", result)

            new_loss = result['loss']
            loss_decrease = 0
            if epoch > 1:
                loss_decrease = ((last_loss-new_loss) / last_loss) * 100
                print("Loss decrease:" , loss_decrease , "%")
            last_loss = new_loss

            epoch_time = time() - epoch_start
            print("Epoch time:" , epoch_time , "s")
            total_time = time() - train_start
            print("Train speed: " , n_tokens / train_time , "sequences / s")
            print("Total train time:" , total_time , "s")
            print()

            if data_definition.max_train_seconds > 0 and total_time > data_definition.max_train_seconds:
                print("Max. train time reached, stopping")
                break
            
            if epoch > 1 and data_definition.min_loss_percentage > 0 and loss_decrease < data_definition.min_loss_percentage:
                print("Min. loss decrease reached, stopping")
                break
            