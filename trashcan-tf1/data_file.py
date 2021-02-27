import os
from typing import List, Tuple
from model_data_definition import ModelDataDefinition
import pandas as pd
import numpy as np

class DataFile:
    """ CSV train data file content """

    def __init__(self, data_definition : ModelDataDefinition , file_name : str ):
        """
        Read a CSV data file content
            data_definition: Model data definition
            file_name: File name
        """

        self.file_name = file_name
        self.data_definition = data_definition

        csv_file_path = os.path.join( data_definition.data_directory , file_name )
        #print("Reading", csv_file_path)

        # Read raw file with pandas
        raw_file = pd.read_csv( csv_file_path , sep=';')
        # Store only referenced columns, less memory
        self.file_rows = raw_file[ data_definition.get_column_names() ].copy()


    def get_elements(self, padding_element : List , idx_start : int , idx_end : int ) -> List:
        """ Get a range of elements in file. "idx_end" element will not be included """

        if idx_end <= 0:
            # Every element will be padding
            return [ padding_element ] * (idx_end - idx_start)

        result = []
        if idx_start < 0:
            # We will need padding:
            result = [ padding_element ] * (0 - idx_start)
            idx_start = 0

        # Append part inside file_rows
        result = result + self.file_rows[idx_start : idx_end]
        
        return result


    def get_n_trainable_tokens(self, data_definition : ModelDataDefinition) -> int:
        """ Returns the number of trainables tokens in this file """
        if not self.data_definition.trainable_column:
            # All trainable
            return self.file_rows.shape[0]
        else:
            return len(self.file_rows[ self.file_rows['trainable'] == 1])


    def get_input_for_row(self, row_index : int) -> dict:
        # pandas supports negative indices (are ignored). end_index is included in result
        start_idx = row_index - self.data_definition.sequence_length
        end_idx = row_index - 1

        # Get the legth to pad:
        if start_idx < 0:
            pad_length = -start_idx
        else:
            pad_length = 0

        input = {}
        for col_name in self.data_definition.sequence_columns:
            # Get values array for column, unpadded
            col_values = self.file_rows.loc[ start_idx:end_idx , col_name ].to_numpy()
            # Add padding (ZEROS!)
            if pad_length > 0:
                col_values = np.pad( col_values , (pad_length,0) , 'constant')
            input[col_name] = col_values

        # Get context values (single values)
        for col_name in self.data_definition.context_columns:
            input[col_name] = self.file_rows.at[ row_index , col_name ] 

        return input


    def get_train_tuple_for_row(self , row_index : int ) -> tuple:
        input = self.get_input_for_row( row_index )
        output = {}
        for col_name in self.data_definition.output_columns:
            output[col_name] = self.file_rows.at[ row_index , col_name ]
        return( input , output )


    def _is_trainable(self, row_index: int ) -> bool:
        """ Returns true if the given token index is trainable """
        if not self.data_definition.trainable_column:
            return True
        return ( self.file_rows.at[ row_index , self.data_definition.trainable_column ] == 1 )


    def get_train_sequences(self):
        # Traverse rows
        for row_index in range(self.file_rows.shape[0]):
            if self._is_trainable( row_index ):
                yield self.get_train_tuple_for_row( row_index )


