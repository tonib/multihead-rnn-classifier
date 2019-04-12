import os
import json
from column_info import ColumnInfo
from typing import List
import tensorflow as tf
import argparse

class ModelDataDefinition:

    def __init__(self):

        self._read_cmd_line_arguments()

        metadata_file_path = os.path.join( self.data_directory , 'data_info.json' )
        print("Reading data structure info from " , metadata_file_path)
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            # Read settings
            self.max_train_seconds = int( ModelDataDefinition._read_setting( json_metadata , 'MaxTrainSeconds' , '0' ) )
            self.min_loss_percentage = float( ModelDataDefinition._read_setting( json_metadata , 'MinLossPercentage' , '0' ) )
            self.percentage_evaluation = float( ModelDataDefinition._read_setting( json_metadata , 'PercentageEvaluation' , '15' ) ) / 100.0
            self.max_epochs = ModelDataDefinition._read_setting( json_metadata , 'MaxEpochs' , '10' )
            self.sequence_length = int( ModelDataDefinition._read_setting( json_metadata , 'SequenceLength' , '128' ) )
            self.trainable_column = ModelDataDefinition._read_setting( json_metadata , 'TrainableColumn' , None )
            self.n_network_elements = int( ModelDataDefinition._read_setting( json_metadata , 'NNetworkElements' , '64' ) )
            
            # Read columns definitions
            self.column_definitions = {}
            for json_column in json_metadata['ColumnDefinitions']:
                self.column_definitions[ json_column['Name'] ] = ColumnInfo( json_column['Name'] , json_column['Labels'] )

            # Sequence column names
            self.sequence_columns = json_metadata['SequenceColumns']

            # Context column names
            self.context_columns = []
            if 'ContextColumns' in json_metadata:
                self.context_columns = json_metadata['ContextColumns']

            # Output column names
            self.output_columns = json_metadata['OutputColumns']


    @staticmethod
    def _read_setting( json_metadata : dict , setting_name : str , default_value : object ) -> str:
        if not setting_name in json_metadata:
            return default_value
        return json_metadata[setting_name]


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


    def get_empty_element(self) :
        """ Input entry with all zeros """
        element = {}
        for column_name in self.sequence_columns:
            element[column_name] = [0] * self.sequence_length
        for column_name in self.context_columns:
            element[column_name] = 0
        return element


    def input_sequence_to_tf_predict_format( self , input: dict ) -> dict:
        """ Convert an input to the Tensorflow prediction expected format: Batch with size 1 """
        # TF expects a BATCH of size 1, so that's why the extra []
        result = {}
        for key in input:
            result[key] = [ input[key] ]
        return result

    def get_column_names(self) -> set:
        """ Set with all used column names """
        all_columns = self.sequence_columns + self.output_columns + self.context_columns
        if self.trainable_column:
            all_columns.append( self.trainable_column )
        return set( all_columns )
