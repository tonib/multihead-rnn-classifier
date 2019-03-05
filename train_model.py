from tensorflow.contrib.estimator import RNNEstimator
import tensorflow as tf
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.feature_column as feature_column
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.contrib.estimator import multi_head
from model_data_definition import ModelDataDefinition
from data_directory import DataDirectory
from prediction_model import PredictionModel
from time import time
from typing import Callable

class TrainModel:
    """ Model for training """

    ###################################
    # MODEL DEFINITION
    ###################################

    def __init__(self, data_definition: ModelDataDefinition):

        # TODO: Store a reference to ModelDataDefinition

        model_dir = data_definition.get_current_model_dir_path()
        print("Current train model dir:" , model_dir)

        # The estimator
        self.estimator = RNNEstimator(
            head = self._get_model_head(data_definition),
            sequence_feature_columns = self._get_model_input_columns(data_definition),
            #num_units=[64, 64], # Removed, extra layer reports same results
            num_units=[64], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir=model_dir
        )


    def _get_model_input_columns(self, data_definition: ModelDataDefinition) -> list:
        """ Returns the model input features list definition """
        result = []
        for def_column in data_definition.columns:
            # Input column
            column = contrib_feature_column.sequence_categorical_column_with_identity( def_column.name , len(def_column.labels) )
            # To indicator column
            column = feature_column.indicator_column( column )
            result.append( column )
        return result


    def _get_model_head(self, data_definition: ModelDataDefinition):
        """ The model head """
        head_parts = []
        for def_column in data_definition.columns:
            head_parts.append( head_lib._multi_class_head_with_softmax_cross_entropy_loss( len(def_column.labels) , name=def_column.name) )
        return multi_head( head_parts )


    ###################################
    # EXPORT MODEL
    ###################################

    def export_model(self, data_definition: ModelDataDefinition):
        """ Exports the model to the exports directory """
        export_path = data_definition.get_exports_dir_path()
        print("Exporting model to " , export_path)
        self.estimator.export_savedmodel( export_path , 
            lambda:data_definition.serving_input_receiver_fn() , strip_default_attrs=True)


    ###################################
    # INPUT MODEL FUNCTION
    ###################################

    def _get_tf_input_fn(self, data_definition : ModelDataDefinition , train_data : DataDirectory ) -> Callable:
        """ Returns the Tensorflow input function for data files """
        # The dataset
        ds = tf.data.Dataset.from_generator( 
            generator=lambda: train_data.traverse_sequences( data_definition ), 
            output_types = self._model_input_output_types(data_definition),
            output_shapes = self._model_input_output_shapes(data_definition)
        )
        ds = ds.shuffle(5000)
        ds = ds.batch(64)
        ds = ds.prefetch(64)
        return ds


    def _model_input_output_types(self, data_definition: ModelDataDefinition ) -> tuple:
        """ Returns data model input and output types definition """
        inputs = {}
        outputs = {}
        for column in data_definition.columns:
            inputs[ column.name ] = tf.int32 # All int numbers: They are indexes to labels (see ColumnInfo)
            outputs[ column.name ] = tf.int32
        return ( inputs , outputs )


    def _model_input_output_shapes(self, data_definition: ModelDataDefinition ) -> tuple:
        """ Returns data model input and output shapes definition """
        inputs = {}
        outputs = {}
        for column in data_definition.columns:
            inputs[ column.name ] = (data_definition.sequence_length,) # Sequence of "self.sequence_length" elements
            outputs[ column.name ] = () # Scalar (one) element
        return ( inputs , outputs )

    ###################################
    # TRAINING
    ###################################

    def train_model(self, train_data : DataDirectory , eval_data : DataDirectory , data_definition : ModelDataDefinition ):
        """ Train dataset """

        epoch = 0
        last_loss = 0
        train_start = time()
        n_tokens = train_data.get_n_total_tokens()
        for _ in range(data_definition.max_epochs):
            epoch += 1
            
            epoch_start = time()
            print("Training epoch", epoch, "...")
            self.estimator.train( input_fn=lambda:self._get_tf_input_fn( data_definition , train_data ) )
            train_time = time() - epoch_start
            
            print("Evaluating...")
            result = self.estimator.evaluate( input_fn=lambda:self._get_tf_input_fn( data_definition , eval_data ) )
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
                return
            
            if epoch > 1 and data_definition.min_loss_percentage > 0 and loss_decrease < data_definition.min_loss_percentage:
                print("Min. loss decrease reached, stopping")
                return

        print("Max. epoch reached, stopping")
