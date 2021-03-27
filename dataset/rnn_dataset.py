from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..model_data_definition import ModelDataDefinition
    from ..data_directory import DataDirectory

import tensorflow as tf
from .csv_files_dataset import CsvFilesDataset

# 118400
# Batches: 1850 Elements: 118400 Time (s): 43.615002155303955 Elements/s: 2714.662252644221

# 121600
# Batches: 1900 Elements: 121600 Time (s): 44.7081823348999 Elements/s: 2719.8600714544627

# 124800
# Batches: 1950 Elements: 124800 Time (s): 45.92690873146057 Elements/s: 2717.3612038580395

class RnnDataset(CsvFilesDataset):

    # Padding value. MUST TO BE ZERO (implementation details)
    PADDING_VALUE = 0

    # End of string (EOS) value
    EOS_VALUE = 1

    # Values 0 and 1 are "keywords"
    N_KEYWORD_VALUES = 2

    def __init__(self, csv_files: DataDirectory, data_definition: ModelDataDefinition, shuffle: bool, debug_columns: bool=False):
        super().__init__(csv_files, data_definition, shuffle, debug_columns)
        
    def _map_csv_file_to_sequences(self, csv_columns_dict) -> tf.data.Dataset:
        """ Map a full csv file to windows of sequence_length elements """

        # Get the csv file window sequences
        csv_ds = CsvFilesDataset._map_csv_file_to_sequences(self, csv_columns_dict)

        # Process window sequences
        csv_ds = self._process_sequences(csv_ds)

        return csv_ds
    
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
            inputs += RnnDataset.N_KEYWORD_VALUES

            elements_length = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, (1, -1)) # [1, 2, 3] -> [[1, 2, 3]]
            inputs = tf.repeat(inputs, repeats=elements_length, axis=0) # [[1, 2, 3]] -> [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
            # Keep lower subdiagonals: [[1, 2, 3], [1, 2, 3], [1, 2, 3]] -> [[1, 0, 0], [1, 2, 0], [1, 2, 3]]
            inputs = tf.linalg.band_part( inputs , elements_length , 0 )
            # Assign EOS: [[1, 0, 0], [1, 2, 0], [1, 2, 3]] -> [[-1, 0, 0], [1, -1, 0], [1, 2, -1]]
            eos_vector = tf.repeat( RnnDataset.EOS_VALUE, elements_length)
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
            inputs += RnnDataset.N_KEYWORD_VALUES

            # inputs[-1] = eos, hard way # [1, 2, 3] -> [1, 2, EOS]
            inputs = tf.tensor_scatter_nd_update( inputs , [[self._data_definition.sequence_length-1]] , [RnnDataset.EOS_VALUE] )
            input_dict[key] = inputs
        for key in self.context_columns:
            input_dict[key] = window_elements_dict[key][-1]

        # Outputs
        output_dict = {}
        for key in self._data_definition.output_columns:
            output_dict[key] = window_elements_dict[key][-1]

        return (input_dict, output_dict)
