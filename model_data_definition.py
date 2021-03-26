import configure_tf_log # Must be FIRST import
import os
import json
from column_info import ColumnInfo
from typing import List, Set, Dict
import tensorflow as tf
import argparse
import logging

class ModelDataDefinition:
    """ Definitions of model data, model settings and training """

    # Directory for model train checkpoints
    CHECKPOINTS_DIR = 'model/checkpoints'

    # Exported model directory
    EXPORTED_MODEL_DIR = 'model/exported_model'

    # Tensorboard logs
    TBOARD_LOGS_DIR = 'model/tensorboard_logs'

    def __init__(self):

        self._read_cmd_line_arguments()

        metadata_file_path = os.path.join( self.data_directory , 'data_info.json' )
        print("Reading data structure info from " , metadata_file_path)
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            # TODO: Remove unused parameters
            # Read settings
            self.max_train_seconds = int( ModelDataDefinition._read_setting( json_metadata , 'MaxTrainSeconds' , '0' ) )
            self.min_loss_percentage = float( ModelDataDefinition._read_setting( json_metadata , 'MinLossPercentage' , '0' ) )
            self.percentage_evaluation = float( ModelDataDefinition._read_setting( json_metadata , 'PercentageEvaluation' , '15' ) ) / 100.0
            self.max_epochs = ModelDataDefinition._read_setting( json_metadata , 'MaxEpochs' , '10' )
            self.sequence_length = int( ModelDataDefinition._read_setting( json_metadata , 'SequenceLength' , '128' ) )
            self.trainable_column = ModelDataDefinition._read_setting( json_metadata , 'TrainableColumn' , None )
            self.n_network_elements = int( ModelDataDefinition._read_setting( json_metadata , 'NNetworkElements' , '64' ) )
            self.learning_rate = float( ModelDataDefinition._read_setting( json_metadata , 'LearningRate' , '0.001' ) )
            self.dropout = float( ModelDataDefinition._read_setting( json_metadata , 'Dropout' , '0' ) )
            self.cell_type = ModelDataDefinition._read_setting( json_metadata , 'CellType' , 'gru' )
            self.log_each_epochs = int( ModelDataDefinition._read_setting( json_metadata , 'LogEachEpochs' , '0' ) )
            self.cache_dataset = bool( ModelDataDefinition._read_setting( json_metadata , 'DatasetCache' , '' ) ) # Yes, bool('') == False
            self.batch_size = int( ModelDataDefinition._read_setting( json_metadata , 'BatchSize' , '64' ) )
            self.max_batches_per_epoch = int( ModelDataDefinition._read_setting( json_metadata , 'MaxBatchesPerEpoch' , '0' ) )

            if self.cache_dataset and self.max_batches_per_epoch > 0:
                raise Exception("DatasetCache = True and MaxBatchesPerEpoch > 0 cannot be set at same time. DatasetCache = True is for small datasets")
            
            # Read columns definitions
            self.column_definitions: Dict[str, ColumnInfo] = {}
            for json_column in json_metadata['ColumnDefinitions']:
                embeddable_dimension = int( ModelDataDefinition._read_setting(json_column, 'EmbeddableDimension', '0' ) )
                self.column_definitions[ json_column['Name'] ] = ColumnInfo( json_column['Name'] , json_column['Labels'] , embeddable_dimension )

            # Sequence column names
            self.sequence_columns = json_metadata['SequenceColumns']

            # Context column names
            self.context_columns = []
            if 'ContextColumns' in json_metadata:
                self.context_columns = json_metadata['ContextColumns']

            # Output column names
            self.output_columns = json_metadata['OutputColumns']

            # Switch to use our custom RNN estimator or the TF RNN canned estimator
            self.use_custom_estimator = bool( ModelDataDefinition._read_setting( json_metadata , 'CustomEstimator' , False ) )


    @staticmethod
    def _read_setting( json_metadata : dict , setting_name : str , default_value : object ) -> object:
        if not setting_name in json_metadata:
            return default_value
        return json_metadata[setting_name]


    def _read_cmd_line_arguments(self):
        parser = argparse.ArgumentParser(description='Train and predict sequences')
        parser.add_argument('--datadir', type=str, default='data' , help='Directory path with data. Default="data"')
        parser.add_argument(configure_tf_log.NOTFWARNINGS_FLAG, action='store_const' , const=True, help='Disable Tensowflow warning messages')
        parser.add_argument('--checkpoint', type=int, default=0, 
            help='Only for export.py script. Number of epoch checkpoint to export (1=First). Default is last completely trained epoch')

        args = parser.parse_args()

        self.data_directory = args.datadir
        print("Data directory:" , self.data_directory )

        self.export_checkpoint = args.checkpoint

        if args.notfwarnings:
            print("TF log info/warning messages disabled")
            tf.get_logger().setLevel(logging.ERROR)

    def get_data_dir_path(self, relative_path: str= None) -> str:
        if relative_path == None:
            return self.data_directory
        return os.path.join( self.data_directory , relative_path )

    def get_export_dir_path(self):
        """ The directory for exported model """
        return self.get_data_dir_path( 'model/exportedmodel' )

    def get_current_model_dir_path(self, relative_path: str= None):
        """ The directory for current train model """
        paths = [ 'model' ]
        if relative_path != None:
            paths.append(relative_path)
        return os.path.join( self.data_directory , *paths )

    def get_empty_element(self) :
        """ Input entry with all zeros """
        element = {}
        for column_name in self.sequence_columns:
            element[column_name] = []
        for column_name in self.context_columns:
            element[column_name] = 0
        return element

    def get_column_names(self) -> Set[str]:
        """ Set with all used column names """
        all_columns = self.sequence_columns + self.output_columns + self.context_columns
        if self.trainable_column:
            all_columns.append( self.trainable_column )
        return set( all_columns )

    def _print_column_summary(self, title: str, column_names: List[str]):
        print(title)
        for col_name in column_names:
            column = self.column_definitions[ col_name ]
            if column.embeddable_dimension > 0:
                txt_embedding = ", Embedable (dim = " + str(column.embeddable_dimension) + ")"
            else:
                txt_embedding = ""
            print("   ", column.name, ":", len(column.labels), "labels", txt_embedding )

    def print_summary(self):
        """ Print definitions summary """
        self._print_column_summary("* SEQUENCE COLUMNS:", self.sequence_columns)
        self._print_column_summary("* CONTEXT COLUMNS:", self.context_columns)
        self._print_column_summary("* OUTPUT COLUMNS:", self.output_columns)
        print("MaxTrainSeconds:", self.max_train_seconds)
        print("MinLossPercentage:", self.min_loss_percentage)
        print("PercentageEvaluation:", self.percentage_evaluation)
        print("MaxEpochs:", self.max_epochs)
        print("SequenceLength:", self.sequence_length)
        print("TrainableColumn:", self.trainable_column)
        print("NNetworkElements:", self.n_network_elements)
        print("CustomEstimator:", self.use_custom_estimator)
        print("LogEachEpochs:", self.log_each_epochs)
        print("DatasetCache:", self.cache_dataset)
        print("BatchSize:", self.batch_size)
        print("MaxBatchesPerEpoch:", self.max_batches_per_epoch)
        
        if self.use_custom_estimator:
            print("LearningRate:", self.learning_rate)
            print("Dropout:", self.dropout)
            print("CellType:", self.cell_type)
        
        
        