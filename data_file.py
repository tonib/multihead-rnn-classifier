import os
import csv
from typing import List, Tuple
from model_data_definition import ModelDataDefinition

class DataFile:

    def __init__(self, data_definition : ModelDataDefinition , file_name : str ):
        """
        Read a CSV data file content
            data_definition: Model data definition
            file_name: File name
        """

        self.file_name = file_name

        csv_file_path = os.path.join( data_definition.data_directory , file_name )
        #print("Reading", csv_file_path)

        self.file_rows = []
        with open( csv_file_path , 'r', encoding='utf-8')  as file:
            csv_reader = csv.reader(file, delimiter=';')
            for raw_row in csv_reader:
                row = []
                for data in raw_row:
                    row.append( int(data) )
                #print(row)
                self.file_rows.append(row)


    def get_elements(self, padding_element : List , idx_start : int , idx_end : int ) -> List:
        """ Get range of elements in file. "idx_end" element will not be included """

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

    @staticmethod
    def _sequence_to_tensorflow_format(sequence: List) -> object:
        return {
            'column1': [item[0] for item in sequence],
            'column2': [item[1] for item in sequence]
        }

    def get_sequences(self, data_definition : ModelDataDefinition ):
        """
        Return sequences in file. There is a sequence for each row in file, of length "padding_element", 
        padded with "padding_element" if needed.
        """
        
        for i in range(len(self.file_rows)):
            pre_sequence = self.get_elements( data_definition.get_padding_element() , i - data_definition.sequence_length , i )
            yield ( 
                DataFile._sequence_to_tensorflow_format(pre_sequence) ,  # Input sequence
                { 'headcol1' : self.file_rows[i][0] , 'headcol2' : self.file_rows[i][1] } # Output for multihead
            )
