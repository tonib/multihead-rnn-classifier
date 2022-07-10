from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..model_data_definition import ModelDataDefinition
    from ..data_directory import DataDirectory

import tensorflow as tf

class CsvFilesDataset:

    # CSV files separator character
    CSV_SEPARATOR = ";"

    # Key in dataset dictionary for file path
    FILE_KEY = '_file_path'

    # Key in dataset dictionary for row column
    ROW_KEY = '_file_row'

    def __init__(self, csv_files: DataDirectory, data_definition: ModelDataDefinition, sequence_length: int, shuffle: bool, debug_columns: bool=False):

        self._csv_files = csv_files
        self._data_definition = data_definition
        self._get_csv_files_structure()
        self.debug_columns = debug_columns
        self.shuffle = shuffle
        self.sequence_length = sequence_length

        self.context_columns = list(data_definition.context_columns)
        if debug_columns:
            self.context_columns.append(CsvFilesDataset.FILE_KEY)
            self.context_columns.append(CsvFilesDataset.ROW_KEY)
        if data_definition.trainable_column:
            self.context_columns.append(data_definition.trainable_column)

        # Get entire CSV files in pipeline, as a dictionary, key=CSV column name, value=values in that column
        self.dataset = tf.data.Dataset.list_files(csv_files.file_paths, shuffle=shuffle)

        # N. parallel calls to process each CSV. Set a maximum (16 is ok for my computer). TODO: This could be another app setting
        if data_definition.csv_parallel_calls == 0:
            n_parallel_calls = None
        elif data_definition.csv_parallel_calls < 0:
            n_parallel_calls = tf.data.AUTOTUNE
        else:
            n_parallel_calls = data_definition.csv_parallel_calls

        # Get a CSV dataset for each CSV file path
        self.dataset = self.dataset.interleave(self._load_csv,
            # cycle_length: Number of csv files to process at same time. It seems to train faster as larger is this value
            # changing this from 64 to 128 trains slower
            #cycle_length=1 if not shuffle else 16,
            cycle_length=1 if not shuffle else data_definition.csv_cycle_length,
            # If have tested values tf.data.AUTOTUNE, None and from 32 to 2. 2 gives the best GPU use. No idea why
            # Theoretically tf.data.AUTOTUNE should do it fine, but no
            num_parallel_calls=None if not shuffle else n_parallel_calls,
            deterministic=not shuffle
        )
        
    def _load_csv(self, file_path):
        """ Load full CSV file content and return it to the pipeline as dict with keys=column names, values=column values """
        csv_ds = tf.data.experimental.CsvDataset(
            file_path, self._default_csv_values,
            header=True,
            field_delim=CsvFilesDataset.CSV_SEPARATOR,
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
        csv_ds = self._map_csv_file_to_sequences(csv_ds, file_path)

        # Remove train column (avoid keras warning about unused inputs)
        if self._data_definition.trainable_column:
            csv_ds = csv_ds.map(self.remove_trainable_column)

        return csv_ds

    def remove_trainable_column(self, input, output):
        del input[self._data_definition.trainable_column]
        return (input, output)

    def _map_csv_row_to_dict_with_debug(self, file_path, enumerated_row):
        row_dict = { feature_column_name: csv_column_values for feature_column_name, csv_column_values in zip(self._feature_column_names, enumerated_row[1]) }

        row_dict[CsvFilesDataset.FILE_KEY] = file_path
        row_dict[CsvFilesDataset.ROW_KEY] = enumerated_row[0] + 2
        return row_dict

    def _flat_map_window(self, window_elements_dict):
        """ Get real window values. I don't really understand this step, but it's required """
        result = {}
        for key in window_elements_dict:
            # See https://github.com/tensorflow/tensorflow/issues/23581#issuecomment-529702702
            result[key] = tf.data.experimental.get_single_element( window_elements_dict[key].batch(self.sequence_length) )
        #return result
        return tf.data.Dataset.from_tensors(result)

    def _map_csv_file_to_sequences(self, csv_columns_dict, file_path: str) -> tf.data.Dataset:
        """ Map a full csv file to windows of sequence_length elements """
        # We NEED drop_remainder=False, but it's tricky. If the entire csv is smaller than sequence_length, if
        # drop_remainder=True, the entire csv sequence will be dropped, and we don't what that. But, if drop_remainder=False,
        # final sequences with length < sequence_length will be feeded, and they must to be filtered... ¯\_(ツ)_/¯
        windows_ds = csv_columns_dict.window(self.sequence_length, shift=1, drop_remainder=False)
        windows_ds = windows_ds.flat_map(self._flat_map_window)
        return windows_ds
    
    def _get_csv_files_structure(self):

        self._feature_column_names = list( self._data_definition.get_column_names() )

        # Tricky things: To get right sequences we must separate CSV contents, and seems not supported by TF CSV high level helpers
        # So, guess the the CSV structure (all files MUST share the same structure):
        with open(self._csv_files.file_paths[0]) as f:
            first_line = f.readline()
            csv_column_names = first_line.split(CsvFilesDataset.CSV_SEPARATOR)
            csv_column_names_to_indices = { name:index for index, name in enumerate(csv_column_names) }
            # select_cols parm in tf.data.experimental.CsvDataset must be ordered by column index. So, reorder feature_column_names
            # to follow that order
            self._feature_column_names.sort( key=lambda x: csv_column_names_to_indices[x] )
            self._feature_column_indices = [ csv_column_names_to_indices[feature_column_name] for feature_column_name in self._feature_column_names ]

        # Column types: All int32
        self._default_csv_values = [ tf.int32 ] * len( self._feature_column_names )
