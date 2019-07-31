
from tensorflow.contrib.estimator import RNNEstimator
import tensorflow.contrib.feature_column as contrib_feature_column
from tensorflow.python.estimator.canned import head as head_lib
import tensorflow.feature_column as feature_column
from model_data_definition import ModelDataDefinition
from tensorflow.contrib.estimator import multi_head
import tensorflow as tf

class CannedEstimator:
    """ Canned RNN estimator definition """

    def __init__(self, data_definition: ModelDataDefinition):

        self.data_definition = data_definition

        model_dir = self.data_definition.get_current_model_dir_path()
        print("Current train model dir:" , model_dir)

        # The estimator
        self.estimator = RNNEstimator(
            head = self._get_model_head(),
            sequence_feature_columns = self._get_sequence_columns(),
            context_feature_columns = self._get_context_columns(),
            num_units=[ data_definition.n_network_elements ], 
            cell_type='gru', 
            optimizer=tf.train.AdamOptimizer,
            model_dir=model_dir
        )
    
    def _get_sequence_columns(self) -> list:
        """ Returns the model input features list definition (sequence) """
        result = []
        for col_name in self.data_definition.sequence_columns:
            # Input column
            def_column = self.data_definition.column_definitions[ col_name ]
            column = contrib_feature_column.sequence_categorical_column_with_identity( col_name , len(def_column.labels) )
            # To indicator column
            column = feature_column.indicator_column( column )
            result.append( column )
        return result


    def _get_context_columns(self) -> list:
        """ Returns the input context columns definition, or None if there are no context columns """
        if len( self.data_definition.context_columns ) == 0:
            return None
        result = []
        for col_name in self.data_definition.context_columns:
            # Input column
            def_column = self.data_definition.column_definitions[ col_name ]
            column = feature_column.categorical_column_with_identity( col_name , len(def_column.labels) )
            # To indicator column
            column = feature_column.indicator_column( column )
            result.append( column )
        return result

    def _get_model_head(self):
        """ The model head """
        head_parts = []
        for col_name in self.data_definition.output_columns:
            def_column = self.data_definition.column_definitions[ col_name ]
            if len(def_column.labels) > 2:
                head_parts.append( head_lib._multi_class_head_with_softmax_cross_entropy_loss( len(def_column.labels) , name=def_column.name) )
            else:
                # def_column.labels = 2:
                head_parts.append( head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(name=def_column.name) )
        return multi_head( head_parts )