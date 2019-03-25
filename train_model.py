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

        self.data_definition = data_definition

        model_dir = self.data_definition.get_current_model_dir_path()
        print("Current train model dir:" , model_dir)

        # The estimator
        self.estimator = RNNEstimator(
            head = self._get_model_head(),
            sequence_feature_columns = self._get_model_input_columns(),
            #num_units=[64, 64], # Removed, extra layer reports same results
            #num_units=[64], 
            num_units=[ data_definition.n_network_elements ], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir=model_dir
        )


    def _get_model_input_columns(self) -> list:
        """ Returns the model input features list definition """
        result = []
        for def_column in self.data_definition.input_columns:
            # Input column
            column = contrib_feature_column.sequence_categorical_column_with_identity( def_column.name , len(def_column.labels) )
            # To indicator column
            column = feature_column.indicator_column( column )
            result.append( column )
        return result


    def _get_model_head(self):
        """ The model head """
        head_parts = []
        for def_column in self.data_definition.output_columns:
            if len(def_column.labels) > 2:
                head_parts.append( head_lib._multi_class_head_with_softmax_cross_entropy_loss( len(def_column.labels) , name=def_column.name) )
            else:
                # def_column.labels = 2:
                head_parts.append( head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(name=def_column.name) )
        return multi_head( head_parts )


    ###################################
    # EXPORT MODEL
    ###################################

    def export_model(self):
        """ Exports the model to the exports directory """
        export_path = self.data_definition.get_exports_dir_path()
        print("Exporting model to " , export_path)
        self.estimator.export_savedmodel( export_path , 
            lambda:self._serving_input_receiver_fn() , strip_default_attrs=True)


    def _serving_input_receiver_fn(self):
        """ Function to define the model signature """
        inputs_signature = {}
        for def_column in self.data_definition.input_columns:
            # It seems the shape MUST include the batch size (the 1)
            column_placeholder = tf.placeholder(dtype=tf.int32, shape=[1, self.data_definition.sequence_length], name=def_column.name)
            inputs_signature[def_column.name] = column_placeholder

        return tf.estimator.export.ServingInputReceiver(inputs_signature, inputs_signature)


    ###################################
    # INPUT MODEL FUNCTION
    ###################################

    def _get_tf_input_fn(self, train_data : DataDirectory , shuffle: bool = True ) -> Callable:
        """ Returns the Tensorflow input function for data files """
        # The dataset
        ds = tf.data.Dataset.from_generator( 
            generator=lambda: train_data.traverse_sequences( self.data_definition, shuffle ), 
            output_types = self._model_input_output_types(),
            output_shapes = self._model_input_output_shapes()
        )
        if shuffle:
            ds = ds.shuffle(5000)
        ds = ds.batch(64)
        ds = ds.prefetch(64)
        return ds


    def _model_input_output_types(self) -> tuple:
        """ Returns data model input and output types definition """
        # All int numbers: They are indexes to labels (see ColumnInfo)
        inputs = {}
        for column in self.data_definition.input_columns:
            inputs[ column.name ] = tf.int32
        
        outputs = {}
        for column in self.data_definition.output_columns:
            outputs[ column.name ] = tf.int32

        return ( inputs , outputs )


    def _model_input_output_shapes(self) -> tuple:
        """ Returns data model input and output shapes definition """
        inputs = {}
        for column in self.data_definition.input_columns:
            inputs[ column.name ] = (self.data_definition.sequence_length,) # Sequence of "self.data_definition.sequence_length" elements

        outputs = {}
        for column in self.data_definition.output_columns:
            outputs[ column.name ] = () # Scalar (one) element

        return ( inputs , outputs )


    ###################################
    # TRAINING
    ###################################

    def train_model(self, train_data : DataDirectory , eval_data : DataDirectory ):
        """ Train dataset """

        epoch = 0
        last_loss = 0
        train_start = time()
        n_tokens = train_data.get_n_total_tokens()
        
        while True:
            epoch += 1
            
            epoch_start = time()
            print("Training epoch", epoch, "...")
            self.estimator.train( input_fn=lambda:self._get_tf_input_fn( train_data ) )
            train_time = time() - epoch_start
            
            print("Evaluating...")
            result = self.estimator.evaluate( input_fn=lambda:self._get_tf_input_fn( eval_data ) )
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

            if self.data_definition.max_epochs > 0 and epoch >= self.data_definition.max_epochs:
                print("Max. train epoch reached, stopping")
                return

            if self.data_definition.max_train_seconds > 0 and total_time > self.data_definition.max_train_seconds:
                print("Max. train time reached, stopping")
                return
            
            if epoch > 1 and self.data_definition.min_loss_percentage != 0 and loss_decrease < self.data_definition.min_loss_percentage:
                print("Min. loss decrease reached, stopping")
                return

        print("Max. epoch reached, stopping")

    ###################################
    # DEBUG / EVALUATION
    ###################################

    def confusion_matrix(self, eval_data : DataDirectory ,  column_name: str ):

        predictions = list( self.estimator.predict( input_fn=lambda:self._get_tf_input_fn( eval_data , shuffle=False ) ) )
        class_prediction = [ int( p[ (column_name, 'classes') ][0] ) for p in predictions ]
        print( class_prediction[:10] )
        real_label = eval_data.get_column_values(self.data_definition, column_name)
        print( real_label[:10] )

        #confusion_matrix = tf.confusion_matrix(labels, predictions)
