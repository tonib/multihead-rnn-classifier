from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..model_data_definition import ModelDataDefinition
    from ..data_directory import DataDirectory

import tensorflow as tf
from .csv_files_dataset import CsvFilesDataset

class TransformerDataset(CsvFilesDataset):

    # Padding value (input values)
    PADDING_VALUE = 0

    # Beging of string (beging of file, really) value
    BOS_VALUE = 1

    # Number of "keywords" (PADDING_VALUE,...)
    N_KEYWORD_VALUES = 2

    # Padding value (output values)
    OUTPUT_PADDING_VALUE = -1

    def __init__(self, csv_files: DataDirectory, data_definition: ModelDataDefinition, shuffle: bool, debug_columns: bool=False):
        self.window_length = data_definition.sequence_length + 1
        super().__init__(csv_files, data_definition, self.window_length, shuffle, debug_columns)
        
    def _map_csv_file_to_sequences(self, csv_columns_dict, file_path: str) -> tf.data.Dataset:
        """ Map a full csv file to windows of sequence_length elements """

        # Add a BOS word at the file start
        bos_word = {}
        for key in self._feature_column_names:
            # - TransformerDataset.N_KEYWORD_VALUES because a +TransformerDataset.N_KEYWORD_VALUES will be added in process_full_window()
            bos_word[key] = tf.constant(TransformerDataset.BOS_VALUE - TransformerDataset.N_KEYWORD_VALUES, dtype=tf.int32)
        if self.debug_columns:
            bos_word[CsvFilesDataset.FILE_KEY] = tf.constant("BOS")
            bos_word[CsvFilesDataset.ROW_KEY] = tf.constant(0, dtype=tf.int64)

        bos_word_ds = tf.data.Dataset.from_tensors(bos_word)
        csv_columns_dict = bos_word_ds.concatenate(csv_columns_dict)

        # Get the csv file window sequences
        csv_ds = CsvFilesDataset._map_csv_file_to_sequences(self, csv_columns_dict, file_path)

        # Process window sequences
        csv_ds = self._process_sequences(csv_ds)

        return csv_ds
    
    def _process_sequences(self, windows_ds):

        # Keep always first window. If csv file length is smaller than self._data_definition.sequence_length, there will be a single window 
        # with length equal as the file size
        first_window_ds = windows_ds.take(1)

        # Later windows will have length smaller to self._data_definition.sequence_length at the file length. Ignore these
        later_windows_ds = windows_ds.skip(1)
        later_windows_ds = later_windows_ds.filter( 
            lambda x: tf.shape( x[self._data_definition.sequence_columns[0]] )[0] == self.window_length )

        # Join both
        windows_ds = first_window_ds.concatenate( later_windows_ds )

        # Now do the real processing
        # Here we do NOT filter untrainable sequences (TODO: Mask loss for these untrainable?)
        # Setting num_parallel_calls=tf.data.experimental.AUTOTUNE (or value >= 2) gives me worst GPU utilization. I don't know why
        #return windows_ds.map(self.process_full_window, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=not self.shuffle)
        return windows_ds.map(self.process_full_window)
        
    def pad_sequence(self, inputs, pad_value):
        inputs_length = tf.shape(inputs)[0]
        seq_len_diff = self._data_definition.sequence_length - inputs_length
        tf.debugging.assert_non_negative(seq_len_diff, "seq_len_diff should be >= 0")
        if seq_len_diff > 0:
            # Too short: Pad right up to sequence length
            if pad_value == 0:
                # Use zeros instead repeat, because if debug_columns = True, here can come strings (file name), and "zero" will be an empty string
                padding = tf.zeros( [seq_len_diff] , dtype=inputs.dtype )
            else:
                padding = tf.repeat(pad_value, seq_len_diff)
            inputs = tf.concat( [inputs, padding], axis=0 )
        return inputs

    def process_full_window(self, window):

        # Inputs
        input_dict = {}

        for key in self._data_definition.sequence_columns:
            inputs = window[key]

            # Increase values to reserve keyword values
            inputs += TransformerDataset.N_KEYWORD_VALUES

            # Remove last value
            inputs = inputs[:-1]
            
            # Pad sequence, if needed. It will be if file length is shorter than self._data_definition.sequence_length
            input_dict[key] = self.pad_sequence(inputs, TransformerDataset.PADDING_VALUE)

        for key in self.context_columns:

            # This is tricky. The context what we know about the NEXT token to predict. We feed the context of the i-th output word
            # to the (i-1)-th input word
            input = window[key][1:]

            # Increase values to reserve keyword values (BOS will never be feeded, but yes padding, so here we are wasting a position.
            # Don't make it more complicated...)
            if key != CsvFilesDataset.FILE_KEY and key != CsvFilesDataset.ROW_KEY and key != self._data_definition.trainable_column:
                input += TransformerDataset.N_KEYWORD_VALUES
            
            input_dict[key] = self.pad_sequence(input, TransformerDataset.PADDING_VALUE)

        # Output
        # Mask non trainable outputs, if needed. This is, set a TransformerDataset.OUTPUT_PADDING_VALUE value in non trainable positions 
        if self._data_definition.trainable_column != None:
            # Get indices of non trainable positions
            non_trainable_indices = window[self._data_definition.trainable_column][1:]
            non_trainable_indices = tf.equal( non_trainable_indices , 0 )
            non_trainable_indices = tf.where( non_trainable_indices ) # Ex. [ [0] , [2] ]
            # Prepare update values for tensor_scatter_nd_update call
            number_non_trainable = tf.shape(non_trainable_indices)[0]
            non_trainable_values = tf.repeat( TransformerDataset.OUTPUT_PADDING_VALUE , number_non_trainable )

        output_dict = {}
        for key in self._data_definition.output_columns:
            outputs = window[key][1:]
            if self._data_definition.trainable_column != None:
                # Apply non trainable mask
                outputs = tf.tensor_scatter_nd_update(outputs, non_trainable_indices, non_trainable_values)
            output_dict[key] = self.pad_sequence( outputs, TransformerDataset.OUTPUT_PADDING_VALUE)

        return (input_dict, output_dict)
