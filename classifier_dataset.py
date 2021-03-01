from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition
    from data_directory import DataDirectory

import tensorflow as tf

class ClassifierDataset:

    # CSV files separator character
    CSV_SEPARATOR = ";"

    def __init__(self, csv_files: DataDirectory, data_definition: ModelDataDefinition, shuffle: bool):

        self._csv_files = csv_files
        self._data_definition = data_definition
        self._get_csv_files_structure()

        self.dataset = tf.data.Dataset.list_files(csv_files.file_paths, shuffle)
        self.dataset = self.dataset.map(self._load_csv)

    @tf.function
    def _load_csv(self, file_path: str):
        csv_ds = tf.data.experimental.CsvDataset(
            file_path, self._default_csv_values, 
            header=True,
            field_delim=ClassifierDataset.CSV_SEPARATOR,
            use_quote_delim=False,
            select_cols=self._feature_column_indices
        )
        full_csv_data = tf.data.experimental.get_single_element( csv_ds.batch(1000000 ) )

        full_csv_dict = {}
        for feature_column_name, csv_column_values in zip(self._feature_column_names, full_csv_data):
            full_csv_dict[feature_column_name] = csv_column_values
        return full_csv_dict

    def _get_csv_files_structure(self):

        self._feature_column_names = list( self._data_definition.get_column_names() )

        # Tricky things: To get right sequences we must separate CSV contents, and seems not supporte by TF CSV hight level helpers
        # Soooo, guess the the CSV structure:
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
