from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition
    from data_directory import DataDirectory

import tensorflow as tf

# 118400
# Batches: 1850 Elements: 118400 Time (s): 43.615002155303955 Elements/s: 2714.662252644221

# 121600
# Batches: 1900 Elements: 121600 Time (s): 44.7081823348999 Elements/s: 2719.8600714544627

# 124800
# Batches: 1950 Elements: 124800 Time (s): 45.92690873146057 Elements/s: 2717.3612038580395

class ClassifierDataset:

    # Padding value. MUST TO BE ZERO (implementation details)
    PADDING_VALUE = 0

    # End of string (EOS) value
    EOS_VALUE = 1

    # Values 0 and 1 are "keywords"
    N_KEYWORD_VALUES = 2

    # CSV files separator character
    CSV_SEPARATOR = ";"

    # Key in dataset dictionary for file path
    FILE_KEY = '_file_path'

    # Key in dataset dictionary for row column
    ROW_KEY = '_file_row'

    def __init__(self, csv_files: DataDirectory, data_definition: ModelDataDefinition, shuffle: bool, debug_columns: bool=False):

        self._csv_files = csv_files
        self._data_definition = data_definition
        self._get_csv_files_structure()
        self.debug_columns = debug_columns
        self.shuffle = shuffle

        self.context_columns = list(data_definition.context_columns)
        if debug_columns:
            self.context_columns.append(ClassifierDataset.FILE_KEY)
            self.context_columns.append(ClassifierDataset.ROW_KEY)
        if data_definition.trainable_column:
            self.context_columns.append(data_definition.trainable_column)

        # Get entire CSV files in pipeline, as a dictionary, key=CSV column name, value=values in that column
        self.dataset = tf.data.Dataset.list_files(csv_files.file_paths, shuffle=shuffle)

        # Get a CSV dataset for each CSV file path
        self.dataset = self.dataset.interleave(self._load_csv,
            cycle_length=1 if not shuffle else None,
            #num_parallel_calls=None if not shuffle else tf.data.experimental.AUTOTUNE,
            num_parallel_calls=None if not shuffle else 16,
            deterministic=not shuffle
        )
        
    def _load_csv(self, file_path):
        """ Load full CSV file content and return it to the pipeline as dict with keys=column names, values=column values """
        csv_ds = tf.data.experimental.CsvDataset(
            file_path, self._default_csv_values,
            header=True,
            field_delim=ClassifierDataset.CSV_SEPARATOR,
            use_quote_delim=False,
            select_cols=self._feature_column_indices
        )

        # Map to dictionary with column names
        if self.debug_columns:
            csv_ds = csv_ds.enumerate()
            csv_ds = csv_ds.map(lambda *row: self._map_csv_row_to_dict_with_debug(file_path, row))
        else:
            csv_ds = csv_ds.map(
                lambda *row: { feature_column_name: csv_column_values for feature_column_name, csv_column_values in zip(self._feature_column_names, row) }
            )

        # Get CSV file sequences
        csv_ds = self._map_csv_file_to_sequences(csv_ds)

        # Process sequences
        csv_ds = self._process_sequences(csv_ds)

        return csv_ds

    def _map_csv_row_to_dict_with_debug(self, file_path, enumerated_row):
        row_dict = { feature_column_name: csv_column_values for feature_column_name, csv_column_values in zip(self._feature_column_names, enumerated_row[1]) }

        row_dict[ClassifierDataset.FILE_KEY] = file_path
        row_dict[ClassifierDataset.ROW_KEY] = enumerated_row[0] + 2
        return row_dict

    def _flat_map_window(self, window_elements_dict):
        """ Get real window values. I don't really understand this step, but it's required """
        result = {}
        for key in window_elements_dict:
            # See https://github.com/tensorflow/tensorflow/issues/23581#issuecomment-529702702
            result[key] = tf.data.experimental.get_single_element( window_elements_dict[key].batch(self._data_definition.sequence_length) )
        #return result
        return tf.data.Dataset.from_tensors(result)

    def _map_csv_file_to_sequences(self, csv_columns_dict) -> tf.data.Dataset:
        """ Map a full csv file to windows of sequence_length elements """
        # We NEED drop_remainder=False, but it's tricky. If the entire csv is smaller than sequence_length, if
        # drop_remainder=True, the entire csv sequence will be dropped, and we don't what that. But, if drop_remainder=False,
        # final sequences with length < sequence_length will be feeded, and they must to be filtered... ¯\_(ツ)_/¯
        windows_ds = csv_columns_dict.window(self._data_definition.sequence_length, shift=1, drop_remainder=False)
        windows_ds = windows_ds.flat_map(self._flat_map_window)
        return windows_ds
    
    def _process_sequences(self, windows_ds):

        # Separate the first window from later windows. First window will generate multiple sequences
        first_window_ds = windows_ds.take(1)
        first_window_ds = first_window_ds.flat_map(self.expand_first_window)
        # Remove non trainable sequences
        if self._data_definition.trainable_column:
            first_window_ds = first_window_ds.filter( lambda input_dict, output_dict: input_dict[self._data_definition.trainable_column] == 1 )

        later_windows_ds = windows_ds.skip(1)
        # Avoid final sequences with length < sequence_length
        later_windows_ds = later_windows_ds.filter( 
            lambda x: tf.shape( x[self._data_definition.sequence_columns[0]] )[0] == self._data_definition.sequence_length )
        # Discard now non trainable sequences, to avoid process them
        if self._data_definition.trainable_column:
            later_windows_ds = later_windows_ds.filter( lambda window_dict: window_dict[self._data_definition.trainable_column][-1] == 1 )
        # TODO: Performance of this could be better applying the operation over a sequences batch, instead of sequence by sequence
        later_windows_ds = later_windows_ds.map(self.process_full_window, num_parallel_calls=tf.data.experimental.AUTOTUNE, 
            deterministic=not self.shuffle)

        return first_window_ds.concatenate( later_windows_ds )

    def expand_first_window(self, window_elements_dict: dict):
        """ Maps the first sequence to a dataset with initial incomplete subsequences. 
            Zero will be used for padding.
            Ex (padding element = 0, eos =-1, sequence_length = 3): 
            [1, 2, 3] -> { "in":[ [-1, 0, 0], [1, -1, 0], [1, 2, -1] ], "out": [ 1, 2 , 3 ] ] } """

        # Inputs
        input_dict = {}
        for key in self._data_definition.sequence_columns:
            inputs = window_elements_dict[key] # [1, 2, 3]

            # Increase the value in 2 because input values 0 and 1 are "keyword" values for padding and EOS
            inputs += ClassifierDataset.N_KEYWORD_VALUES

            elements_length = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, (1, -1)) # [1, 2, 3] -> [[1, 2, 3]]
            inputs = tf.repeat(inputs, repeats=elements_length, axis=0) # [[1, 2, 3]] -> [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
            # Keep lower subdiagonals: [[1, 2, 3], [1, 2, 3], [1, 2, 3]] -> [[1, 0, 0], [1, 2, 0], [1, 2, 3]]
            inputs = tf.linalg.band_part( inputs , elements_length , 0 )
            # Assign EOS: [[1, 0, 0], [1, 2, 0], [1, 2, 3]] -> [[-1, 0, 0], [1, -1, 0], [1, 2, -1]]
            eos_vector = tf.repeat( ClassifierDataset.EOS_VALUE, elements_length)
            inputs = tf.linalg.set_diag(inputs, eos_vector)

            if elements_length < self._data_definition.sequence_length:
                # Pad right up to sequence length. This can happens if the full CSV size is lower than sequence_length
                zeros = tf.zeros( [ elements_length , self._data_definition.sequence_length - elements_length] , dtype=inputs.dtype )
                inputs = tf.concat( [inputs, zeros], axis=1 )
            input_dict[key] = inputs

        for key in self.context_columns:
            input_dict[key] = window_elements_dict[key]
        
        # Outputs
        output_dict = {}
        for key in self._data_definition.output_columns:
            output_dict[key] = window_elements_dict[key]
        
        return tf.data.Dataset.from_tensor_slices( (input_dict, output_dict) )
        
    def process_full_window(self, window_elements_dict) -> Tuple:
        """ Maps full window sequences to (input,output) tuples """
        # Inputs
        input_dict = {}
        for key in self._data_definition.sequence_columns:
            inputs = window_elements_dict[key]

            # Increase the value in 2 because input values 0 and 1 are "keyword" values for padding and EOS
            inputs += ClassifierDataset.N_KEYWORD_VALUES

            # inputs[-1] = eos, hard way # [1, 2, 3] -> [1, 2, EOS]
            inputs = tf.tensor_scatter_nd_update( inputs , [[self._data_definition.sequence_length-1]] , [ClassifierDataset.EOS_VALUE] )
            input_dict[key] = inputs
        for key in self.context_columns:
            input_dict[key] = window_elements_dict[key][-1]

        # Outputs
        output_dict = {}
        for key in self._data_definition.output_columns:
            output_dict[key] = window_elements_dict[key][-1]

        return (input_dict, output_dict)

    def _get_csv_files_structure(self):

        self._feature_column_names = list( self._data_definition.get_column_names() )

        # Tricky things: To get right sequences we must separate CSV contents, and seems not supported by TF CSV high level helpers
        # So, guess the the CSV structure (all files MUST share the same structure):
        with open(self._csv_files.file_paths[0]) as f:
            first_line = f.readline()
            csv_column_names = first_line.split(ClassifierDataset.CSV_SEPARATOR)
            csv_column_names_to_indices = { name:index for index, name in enumerate(csv_column_names) }
            # select_cols parm in tf.data.experimental.CsvDataset must be ordered by column index. So, reorder feature_column_names
            # to follow that order
            self._feature_column_names.sort( key=lambda x: csv_column_names_to_indices[x] )
            self._feature_column_indices = [ csv_column_names_to_indices[feature_column_name] for feature_column_name in self._feature_column_names ]

        # Column types: All int32
        self._default_csv_values = [ tf.int32 ] * len( self._feature_column_names )
