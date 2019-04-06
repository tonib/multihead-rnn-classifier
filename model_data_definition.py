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
            self.trainable_column_index = int( ModelDataDefinition._read_setting( json_metadata , 'TrainableColumnIndex' , '-1' ) )
            self.n_network_elements = int( ModelDataDefinition._read_setting( json_metadata , 'NNetworkElements' , '64' ) )
            
            # Read columns definitions
            self.input_columns = []
            self._read_columns_definitions( self.input_columns, json_metadata['InputColumns'] )

            self.context_columns = []
            if 'ContextColumns' in json_metadata:
                self._read_columns_definitions( self.context_columns, json_metadata['ContextColumns'] )

            self.output_columns = []
            self._read_columns_definitions( self.output_columns, json_metadata['OutputColumns'] )


    def _read_columns_definitions(self, columns : List[ColumnInfo] , columns_json : List[dict] ):
        """ Read columns definitions """
        for json_column in columns_json:
            columns.append( ColumnInfo( json_column['Name'] , json_column['Labels'] , int( json_column['Index'] ) ) )


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
        return [0] * len(self.input_columns)


    def sequence_to_tf_train_format(self, input_sequence : List[List[int]] , output : List[int] ) -> tuple:
        """ Convert a data file sequence input and the theorical output to the Tensorflow expected format """
        input_record = {}
        for def_column in self.input_columns:
            input_record[def_column.name] = [item[def_column.index] for item in input_sequence]

        for def_column in self.context_columns:
            input_record[def_column.name] = output[def_column.index]

        output_record = {}
        for def_column in self.output_columns:
            output_record[def_column.name] = output[def_column.index]

        return ( input_record , output_record )


    def input_sequence_to_tf_predict_format( self , input_sequence : List[List[int]] ) -> dict:
        """ Convert an input sequence to the Tensorflow prediction expected format """
        input_record = {}
        for def_column in self.input_columns:
            # TF expects a BATCH of size 1, so that's why the extra []
            input_record[def_column.name] = [ [item[def_column.index] for item in input_sequence] ]
        return input_record

    def get_max_column_idx(self) -> int:
        """ Get the maximum column index """
        return max( max(c.index for c in self.input_columns) , max(c.index for c in self.output_columns) , self.trainable_column_index )


    def get_column_index(self, column_name: str) -> int:
        for c in self.input_columns:
            if c.name == column_name:
                return c.index
        for c in self.output_columns:
            if c.name == column_name:
                return c.index
        return -1