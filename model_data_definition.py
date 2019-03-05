import os
import json
from column_info import ColumnInfo
from typing import List
import tensorflow as tf
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.feature_column as feature_column
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.contrib.estimator import multi_head
import argparse

# TODO: Move members here to other classes

class ModelDataDefinition:

    def __init__(self):

        self._read_cmd_line_arguments()

        metadata_file_path = os.path.join( self.data_directory , 'data_info.json' )
        print("Reading data structure info from " , metadata_file_path)

        self.columns = []
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            # Read settings
            self.max_train_seconds = int( ModelDataDefinition._read_setting( json_metadata , 'MaxTrainSeconds' , '0' ) )
            self.min_loss_percentage = int( ModelDataDefinition._read_setting( json_metadata , 'MinLossPercentage' , '0' ) )
            self.max_epochs = ModelDataDefinition._read_setting( json_metadata , 'MaxEpochs' , '10' )
            
            # Read columns
            for json_column in json_metadata['Columns']:
                self.columns.append( ColumnInfo( json_column['Name'] , json_column['Labels'] ) )

        # Constant
        self.sequence_length = 128

    def _read_cmd_line_arguments(self):
        parser = argparse.ArgumentParser(description='Train and predict sequeces')
        parser.add_argument('--datadir', type=str, default='data' , help='Directory path with data. Default="data"')
        args = parser.parse_args()
        self.data_directory = args.datadir
        print("Data directory:" , self.data_directory )

    def get_exports_dir_path(self):
        """ The directory for exported models """
        return os.path.join( self.data_directory , 'exportedmodels' )

    def get_current_model_dir_path(self):
        """ The directory for current train model """
        return os.path.join( self.data_directory , 'model' )

    @staticmethod
    def _read_setting( json_metadata : dict , setting_name : str , default_value : object ) -> str:
        if not setting_name in json_metadata:
            return default_value
        return json_metadata[setting_name]
        

    def get_padding_element(self) :
        """ The padding element for tokens at object start: ARRAY WITH ALL ZEROS """
        return [0] * len(self.columns)


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

    def sequence_to_tf_train_format(self, input_sequence : List[List[int]] , output : List[int] ) -> dict:
        """ Convert a data file sequence input and the theorical output to the Tensorflow expected format """
        input_record = {}
        output_record = {}
        for idx , def_column in enumerate(self.columns):
            input_record[def_column.name] = [item[idx] for item in input_sequence]
            output_record[def_column.name] = output[idx]

        return ( input_record , output_record )


    def input_sequence_to_tf_predict_format( self , input_sequence : List[List[int]] ) -> dict:
        """ Convert an input sequence to the Tensorflow expected format """
        input_record = {}
        for idx , def_column in enumerate(self.columns):
            # TF expects a BATCH of size 1, so that's why the extra []
            input_record[def_column.name] = [ [item[idx] for item in input_sequence] ]
        return input_record

    def serving_input_receiver_fn(self):
        """ Function to define the model signature """

        # TODO: Check if placeholder with variable input lenght  is allowed, for variable input sequences
        # It seems the shape MUST include the batch size (the 1)
        # x = tf.placeholder(dtype=tf.string, shape=[1, self.sequence_length], name='character')
        # #print("Input shape: " , x)
        # inputs =  {'character': x }
        # return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        inputs_signature = {}
        for def_column in self.columns:
            # It seems the shape MUST include the batch size (the 1)
            column_placeholder = tf.placeholder(dtype=tf.int32, shape=[1, self.sequence_length], name=def_column.name)
            inputs_signature[def_column.name] = column_placeholder

        return tf.estimator.export.ServingInputReceiver(inputs_signature, inputs_signature)
