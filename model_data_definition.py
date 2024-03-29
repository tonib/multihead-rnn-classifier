from __future__ import annotations

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

    # Tensorboard directory name
    TBOARD_DIR_NAME = 'tensorboard_logs'

    # Tensorboard logs
    TBOARD_LOGS_DIR = 'model/' + TBOARD_DIR_NAME

    # Tensorflow Lite model file path
    TFLITE_PATH = 'model/model.tflite'
    
    def _load(self, read_cmd_line: bool = True, file_path: str = None):
        """ Load data from data_info.json file """

        if read_cmd_line:
            self._read_cmd_line_arguments()

        if file_path == None:
            metadata_file_path = self.get_config_file_path()
        else:
            metadata_file_path = file_path
        print("Reading data structure info from " , metadata_file_path)
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)

            # Read settings
            self.percentage_evaluation = float( ModelDataDefinition._read_setting( json_metadata , 'PercentageEvaluation' , '15' ) ) / 100.0
            self.max_epochs = ModelDataDefinition._read_setting( json_metadata , 'MaxEpochs' , '10' )
            self.sequence_length = int( ModelDataDefinition._read_setting( json_metadata , 'SequenceLength' , '128' ) )
            self.trainable_column = ModelDataDefinition._read_setting( json_metadata , 'TrainableColumn' , None )
            self.learning_rate = float( ModelDataDefinition._read_setting( json_metadata , 'LearningRate' , '0.001' ) )
            self.log_each_batches = int( ModelDataDefinition._read_setting( json_metadata , 'LogEachBatches' , '0' ) )
            self.cache_dataset = bool( ModelDataDefinition._read_setting( json_metadata , 'DatasetCache' , '' ) ) # Yes, bool('') == False
            self.batch_size = int( ModelDataDefinition._read_setting( json_metadata , 'BatchSize' , '64' ) )
            self.max_batches_per_epoch = int( ModelDataDefinition._read_setting( json_metadata , 'MaxBatchesPerEpoch' , '0' ) )
            self.csv_cycle_length = int( ModelDataDefinition._read_setting( json_metadata , 'CsvCycleLength' , '16' ) )
            self.csv_parallel_calls = int( ModelDataDefinition._read_setting( json_metadata , 'CsvParallelCalls' , '16' ) )
            self.shuffle_buffer_size = int( ModelDataDefinition._read_setting( json_metadata , 'ShuffleBufferSize' , '1024' ) )

            self.model_type = ModelDataDefinition._read_setting( json_metadata , 'ModelType' , 'gpt' )

            self.cell_type = ModelDataDefinition._read_setting( json_metadata , 'CellType' , 'gru' )
            self.dropout = float( ModelDataDefinition._read_setting( json_metadata , 'Dropout' , '0' ) )
            self.n_network_elements = int( ModelDataDefinition._read_setting( json_metadata , 'NNetworkElements' , '256' ) )
            self.rnn_embedding_size = int( ModelDataDefinition._read_setting( json_metadata , 'RnnEmbeddingSize' , '0' ) )

            self.gpt_embedding_dropout = float( ModelDataDefinition._read_setting( json_metadata , 'GptEmbeddingDropout' , '0.1' ) )
            self.gpt_residual_dropout = float( ModelDataDefinition._read_setting( json_metadata , 'GptResidualDropout' , '0.1' ) )
            self.gpt_attention_dropout = float( ModelDataDefinition._read_setting( json_metadata , 'GptAttentionDropout' , '0.1' ) )
            self.gpt_n_layers = int( ModelDataDefinition._read_setting( json_metadata , 'GptNLayers' , '2' ) )
            self.gpt_n_heads = int( ModelDataDefinition._read_setting( json_metadata , 'GptNHeads' , '2' ) )
            self.gpt_embedding_size = int( ModelDataDefinition._read_setting( json_metadata , 'GptEmbeddingSize' , '128' ) )
            self.gpt_activation_function = ModelDataDefinition._read_setting( json_metadata , 'GptActivationFunction' , 'gelu' )

            if self.cache_dataset and self.max_batches_per_epoch > 0:
                raise Exception("DatasetCache = True and MaxBatchesPerEpoch > 0 cannot be set at same time. DatasetCache = True is for small datasets")
            
            # Read shared label sets definitions
            self.shared_labels: Dict[str, ColumnInfo] = {}
            if 'SharedLabelsDefinitions' in json_metadata:
                for json_shared_labels in json_metadata['SharedLabelsDefinitions']:
                    column : ColumnInfo = self._create_column_info_from_json(json_shared_labels)
                    self.shared_labels[ column.name ] = column

            # Read columns definitions
            self.column_definitions: Dict[str, ColumnInfo] = {}
            for json_column in json_metadata['ColumnDefinitions']:
                column : ColumnInfo = self._create_column_info_from_json(json_column)
                self.column_definitions[ column.name ] = column

            # Sequence column names
            self.sequence_columns = json_metadata['SequenceColumns']

            # Context column names
            self.context_columns = []
            if 'ContextColumns' in json_metadata:
                self.context_columns = json_metadata['ContextColumns']

            # Output column names
            self.output_columns = json_metadata['OutputColumns']       

    def _create_column_info_from_json(self, json_column: dict) -> ColumnInfo:
        """ Create a ColumnInfo from its json definition """
        # Check if column references a shared labels set
        shared_labels_name : str = ModelDataDefinition._read_setting(json_column, "SharedLabelsId", None)
        if shared_labels_name == None:
            # Column has it's own labels
            embeddable_dimension = int( ModelDataDefinition._read_setting(json_column, 'EmbeddableDimension', '0' ) )
            return ColumnInfo( json_column['Name'] , json_column['Labels'] , embeddable_dimension, None )
        else:
            # Column references a shared labels set
            if shared_labels_name not in self.shared_labels:
                raise f"{shared_labels_name} is not in SharedLabelsDefinitions"
            shared_labels: ColumnInfo = self.shared_labels[shared_labels_name]
            return ColumnInfo( json_column['Name'] , shared_labels.labels , shared_labels.embeddable_dimension, 
                shared_labels_name )


    def to_dict(self) -> dict:
        """ Returns dict with instance values, all serializable. Used to serialize Keras model """
        values_dict = self.__dict__.copy()
        values_dict["shared_labels"] = { col_name : self.shared_labels[col_name].__dict__ for col_name in self.shared_labels }
        values_dict["column_definitions"] = { col_name : self.column_definitions[col_name].__dict__ for col_name in self.column_definitions }
        return values_dict

    @staticmethod
    def from_dict(values_dict: dict) -> ModelDataDefinition:
        """ Creates a ModelDataDefinition from a values dict. Used to deserialize Keras model """
        data_definition = ModelDataDefinition()
        data_definition.__dict__ = values_dict.copy()
        data_definition.shared_labels = { col_name : ColumnInfo(**data_definition.shared_labels[col_name])
                                               for col_name in data_definition.shared_labels }
        data_definition.column_definitions = { col_name : ColumnInfo(**data_definition.column_definitions[col_name])
                                               for col_name in data_definition.column_definitions }
        return data_definition

    def get_config_file_path(self) -> str:
        return os.path.join( self.data_directory , 'data_info.json' )

    @staticmethod
    def from_file(read_cmd_line: bool = True, file_path: str = None) -> ModelDataDefinition:
        """ Create instance and load from file """
        data_definition = ModelDataDefinition()
        data_definition._load(read_cmd_line, file_path)
        return data_definition

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
        parser.add_argument('--productiondir', type=str, default='production', 
            help='Only for production.py script. Directory for production files. Default is "production"')

        args = parser.parse_args()

        self.data_directory = args.datadir
        print("Data directory:" , self.data_directory )

        self.export_checkpoint = args.checkpoint
        self.production_dir = args.productiondir

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

    def get_column_names(self) -> Set[str]:
        """ Set with all used column names """
        all_columns = self.sequence_columns + self.output_columns + self.context_columns
        if self.trainable_column:
            all_columns.append( self.trainable_column )
        return set( all_columns )

    def _print_column_summary(self, title: str, column_names: List[str]) -> int:
        """ Prints a input / output summary and returns the number of dimensions """
        print(title)
        n_total_dimensions = 0
        for col_name in column_names:
            column = self.column_definitions[ col_name ]
            if column.embeddable_dimension > 0:
                txt_embedding = ", Embedable (dim = " + str(column.embeddable_dimension) + ")"
                n_total_dimensions += column.embeddable_dimension
            else:
                txt_embedding = ""
                n_total_dimensions += len(column.labels)
            shared_labels_name = f", Shared labels id: {column.shared_labels_name}" if column.shared_labels_name != None else ""
            print(f"   {column.name}: {len(column.labels)} labels {txt_embedding} {shared_labels_name}" )
        return n_total_dimensions

    def print_summary(self):
        """ Print definitions summary """
        n_sequence_dims: int = self._print_column_summary("* SEQUENCE COLUMNS:", self.sequence_columns)
        n_context_dims: int = self._print_column_summary("* CONTEXT COLUMNS:", self.context_columns)
        print("N. total input dimensions:", n_sequence_dims + n_context_dims)
        self._print_column_summary("* OUTPUT COLUMNS:", self.output_columns)
        print("ModelType:", self.model_type)
        if self.model_type == "gpt":
            print("GptEmbeddingDropout:", self.gpt_embedding_dropout)
            print("GptResidualDropout:", self.gpt_residual_dropout)
            print("GptAttentionDropout:", self.gpt_attention_dropout)
            print("GptNLayers:", self.gpt_n_layers)
            print("GptNHeads:", self.gpt_n_heads)
            print("GptEmbeddingSize:", self.gpt_embedding_size)
            print("GptActivationFunction:", self.gpt_activation_function)
        elif self.model_type == "rnn":
            print("NNetworkElements:", self.n_network_elements)
            print("Dropout:", self.dropout)
            print("CellType:", self.cell_type)
        print("LearningRate:", self.learning_rate)
        print("PercentageEvaluation:", self.percentage_evaluation)
        print("MaxEpochs:", self.max_epochs)
        print("SequenceLength:", self.sequence_length)
        print("TrainableColumn:", self.trainable_column)
        print("LogEachBatches:", self.log_each_batches)
        print("DatasetCache:", self.cache_dataset)
        print("BatchSize:", self.batch_size)
        print("CsvCycleLength:", self.csv_cycle_length)
        print("CsvParallelCalls:", self.csv_parallel_calls)
        print("ShuffleBufferSize:", self.shuffle_buffer_size)
        print("MaxBatchesPerEpoch:", self.max_batches_per_epoch)
        
        