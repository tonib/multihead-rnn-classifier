import os
import csv
from typing import List, Tuple
from model_data_definition import ModelDataDefinition

class DataFile:
    """ CSV train data file content """

    def __init__(self, data_definition : ModelDataDefinition , file_name : str ):
        """
        Read a CSV data file content
            data_definition: Model data definition
            file_name: File name
        """

        self.file_name = file_name

        csv_file_path = os.path.join( data_definition.data_directory , file_name )
        #print("Reading", csv_file_path)

        # Data columns to read are the first ones. There can be others columns after (usually for debug), they will be ignored
        max_column = data_definition.get_max_column_idx() + 1

        self.file_rows = []
        with open( csv_file_path , 'r', encoding='utf-8')  as file:
            csv_reader = csv.reader(file, delimiter=';')
            first = True
            for raw_row in csv_reader:

                # IGNORE HEADER ROW!
                if first:
                    first = False
                    continue

                row = []
                for i in range(max_column):
                    row.append( int(raw_row[i]) )
                #print(row)
                self.file_rows.append(row)


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

    def _is_trainable(self, data_definition : ModelDataDefinition, row: int ) -> bool:
        """ Returns true if the given token index is trainable """
        return data_definition.trainable_column_index < 0 or row[data_definition.trainable_column_index] == 1

    def get_n_trainable_tokens(self, data_definition : ModelDataDefinition) -> int:
        """ Returns the number of trainables tokens in this file """
        return sum( self._is_trainable(data_definition, row) for row in self.file_rows )

    def get_sequences(self, data_definition : ModelDataDefinition ):
        """
        Return sequences in file. There is a sequence for each row in file, of length "data_definition.sequence_length", 
        padded with "padding_element" if needed.
        """
        
        padding_element = data_definition.get_padding_element()
        for i in range(len(self.file_rows)):
            # Check if this row is trainable
            if data_definition.trainable_column_index >= 0 and self.file_rows[i][data_definition.trainable_column_index] == 0:
                # It's not trainable
                continue

            pre_sequence = self.get_elements( padding_element , i - data_definition.sequence_length , i )
            yield data_definition.sequence_to_tf_train_format( pre_sequence , self.file_rows[i] )
