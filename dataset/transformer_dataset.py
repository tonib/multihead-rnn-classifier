from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..model_data_definition import ModelDataDefinition
    from ..data_directory import DataDirectory

import tensorflow as tf
from .csv_files_dataset import CsvFilesDataset

class TransformerDataset(CsvFilesDataset):

    # Padding value
    PADDING_VALUE = 0

    # Beging of string (beging of file, really) value
    BOS_VALUE = 1

    # Number of "keywords" (PADDING_VALUE,...)
    N_KEYWORD_VALUES = 2

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
            print("**", file_path)
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
        windows_ds = windows_ds.map(self.process_full_window, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=not self.shuffle)

        # TODO: Currently multioutput is unsupported. Train a single output:
        #windows_ds = windows_ds.map(self.map_single_output)

        return windows_ds
        
    # TODO: Remove this when multiple outputs were supported
    def map_single_output(self, input, output):
        return (input, output['outputTypeIdx'])

    def pad_sequence(self, inputs):
        inputs_length = tf.shape(inputs)[0]
        seq_len_diff = self._data_definition.sequence_length - inputs_length
        tf.debugging.assert_non_negative(seq_len_diff, "seq_len_diff should be >= 0")
        if seq_len_diff > 0:
            # Too short: Pad right up to sequence length
            zeros = tf.zeros( [seq_len_diff] , dtype=inputs.dtype )
            inputs = tf.concat( [inputs, zeros], axis=0 )
        return inputs

        #elif seq_len_diff < 0:
            # Sequence too long, remove tail elements:
            # This should not happen
            # tf.debugging.Assert(False, seq_len_diff)
            #raise Exception("Sequence too long, this should not happen. Seq.len=", str(seq_len_diff))
            #return inputs[:seq_len_diff]
        #else:
        #    return inputs

    def process_full_window(self, window):

        # Inputs
        input_dict = {}

        for key in self._data_definition.sequence_columns:
            inputs = window[key]

            # Increase values to reserve keyword values
            inputs += TransformerDataset.N_KEYWORD_VALUES

            # Remove last value
            inputs = inputs[:-1]
            
            input_dict[key] = self.pad_sequence(inputs)

        for key in self.context_columns:
            # This is tricky. The context what we know about the NEXT token to predict. We feed the context of the i-th output word
            # to the (i-1)-th input word
            input_dict[key] = self.pad_sequence( window[key][1:] )

        # Output
        output_dict = {}
        for key in self._data_definition.output_columns:
            output_dict[key] = self.pad_sequence( window[key][1:] )

        return (input_dict, output_dict)

    # def get_bos_sequence_from_fist_window(self, window):
    #     # For the fist sequence we will generate two samples, one with an initial BOS and other as a normal sequence

    #     # Prepare the BOS sequence:
    #     # Inputs
    #     bos_input = {}
    #     for key in self._data_definition.sequence_columns:
    #         inputs = window[key]
            
    #         # Increase values to reserve keyword values
    #         inputs += RnnDataset.N_KEYWORD_VALUES

    #         # Add BOS
    #         inputs = tf.concat( [[TransformerDataset.BOS_VALUE], inputs], axis=1 )

    #         # Fit to seq. length
    #         bos_input[key] = self.pad_sequence(inputs)

    #     for key in self.context_columns:
    #         # Keep as is: BOS will get context for word with index=0
    #         bos_input[key] = self.pad_sequence( window[key] )

    #     # Output
    #     bos_output = {}
    #     for key in self._data_definition.output_columns:
    #         # Keep as is
    #         bos_output[key] = self.pad_sequence( window[key] )
        
    #     return (bos_input, bos_output)

    # def expand_first_window(self, window):
    #     # Get the first sequences, with an BOS initial symbol
    #     bos_input, bos_output = self.get_bos_sequence_from_fist_window(window)
    #     for key in bos_input:
    #         bos_input[key] = tf.expand_dims(input, axis)

    #     # If the first window is full, process it also as a normal window (without the initial BOS)
    #     if 