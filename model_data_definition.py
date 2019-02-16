import os
import json
from column_info import ColumnInfo
from typing import List
import tensorflow as tf
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.feature_column as feature_column
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.contrib.estimator import multi_head

class ModelDataDefinition:

    def __init__(self, data_directory : str):

        self.data_directory = data_directory

        metadata_file_path = os.path.join( data_directory , 'data_info.json' )
        print("Reading data structure info from " , metadata_file_path)

        self.columns = []
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            for json_column in json_metadata['ColumnsInfo']:
                self.columns.append( ColumnInfo( json_column['Name'] , json_column['Labels'] ) )

        # Constant
        self.sequence_length = 50


    def get_padding_element(self) :
        """ The padding element for tokens at object start: ARRAY WITH ALL ZEROS """
        return [0] * len(self.columns)


    def model_input_output_types(self):
        """ The data model input and output types definition """
        inputs = {}
        outputs = {}
        for column in self.columns:
            inputs[ column.name ] = tf.int32 # All int numbers: They are indexes to labels (see ColumnInfo)
            outputs[ column.name ] = tf.int32
        return ( inputs , outputs )


    def model_input_output_shapes(self):
        """ The data model input and output shapes definition """
        inputs = {}
        outputs = {}
        for column in self.columns:
            inputs[ column.name ] = (self.sequence_length,) # Sequence of "self.sequence_length" elements
            outputs[ column.name ] = () # Scalar (one) element
        return ( inputs , outputs )

    def get_model_input_columns(self):
        """ The model input features definition """
        result = []
        for def_column in self.columns:
            # Input column
            column = contrib_feature_column.sequence_categorical_column_with_identity( def_column.name , len(def_column.labels) )
            # To indicator column
            column = feature_column.indicator_column( column )
            result.append( column )
        return result

    def get_model_head(self):
        """ The model head """
        head_parts = []
        for def_column in self.columns:
            head_parts.append( head_lib._multi_class_head_with_softmax_cross_entropy_loss( len(def_column.labels) , name=def_column.name) )
        return multi_head( head_parts )
