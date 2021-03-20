from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition
    from data_directory import DataDirectory

import tensorflow as tf

# TODO: Parallelize operations (performance)

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

        # Get a CSV dataset for each CSV file path (TODO: Use interleave here)
        self.dataset = self.dataset.map(self._load_csv)
        
        self.dataset = self.dataset.flat_map(lambda x: x)

        #self.dataset = self.dataset.map(self._map_csv_file_to_sequences)

    def _load_csv(self, file_path):
        """ Load full CSV file content and return it to the pipeline as dict with keys=column names, values=column values """
        csv_ds = tf.data.experimental.CsvDataset(
            file_path, self._default_csv_values, 
            header=True,
            field_delim=ClassifierDataset.CSV_SEPARATOR,
            use_quote_delim=False,
            select_cols=self._feature_column_indices
        )
        # Load the entire file
        #csv_ds = tf.data.experimental.get_single_element( csv_ds.batch(1000000) )

        # full_csv_dict = {}
        # for feature_column_name, csv_column_values in zip(self._feature_column_names, csv_ds):
        #     full_csv_dict[feature_column_name] = csv_column_values

        # Map to dictionary with column names
        csv_ds = csv_ds.map(
            lambda *row: { feature_column_name: csv_column_values for feature_column_name, csv_column_values in zip(self._feature_column_names, row) }
        )

        # TODO: Pending
        # if self.debug_columns:
        #     # For debugging and mental health, add file path and row numbers
        #     n_csv_file_elements = tf.shape( full_csv_dict[feature_column_name] )[0]
        #     full_csv_dict[ClassifierDataset.FILE_KEY] = tf.repeat( file_path , n_csv_file_elements )
        #     # +2 to start with 1 based index, and skip titles row
        #     full_csv_dict[ClassifierDataset.ROW_KEY] = tf.range(2, n_csv_file_elements + 2)

        # Get CSV file sequences
        return self._map_csv_file_to_sequences(csv_ds)

        # return tf.data.Dataset.from_tensor_slices(full_csv_dict)

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
        #windows_ds = windows_ds.flat_map(lambda x: x)
        #windows_ds = windows_ds.flat_map(lambda window: window.batch(self._data_definition.sequence_length, drop_remainder=False))
        windows_ds = windows_ds.flat_map(self._flat_map_window)
        return windows_ds

        # windows_ds = windows_ds.map(self._flat_map_window)

        # # Separate the first window from later windows. First window will generate multiple sequences
        # first_window_ds = windows_ds.take(1)
        # first_window_ds = first_window_ds.flat_map(self.expand_first_window)
        # # Remove non trainable sequences
        # if self._data_definition.trainable_column:
        #     first_window_ds = first_window_ds.filter( lambda input_dict, output_dict: input_dict[self._data_definition.trainable_column] == 1 )

        # later_windows_ds = windows_ds.skip(1)
        # # Avoid final sequences with length < sequence_length
        # later_windows_ds = later_windows_ds.filter( 
        #     lambda x: tf.shape( x[self._data_definition.sequence_columns[0]] )[0] == self._data_definition.sequence_length )
        # # Discard now non trainable sequences, to avoid process them
        # if self._data_definition.trainable_column:
        #     later_windows_ds = later_windows_ds.filter( lambda window_dict: window_dict[self._data_definition.trainable_column][-1] == 1 )
        # # TODO: Performance of this could be better applying the operation over a sequences batch, instead of sequence by sequence
        # later_windows_ds = later_windows_ds.map(self.process_full_window, num_parallel_calls=tf.data.experimental.AUTOTUNE, 
        #     deterministic=not self.shuffle)

        # return first_window_ds.concatenate( later_windows_ds )
    
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
