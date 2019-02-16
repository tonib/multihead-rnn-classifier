from typing import List, Callable, Tuple
from data_file import DataFile
import os
import csv
import random
import json
from column_info import ColumnInfo

class DataDirectory:
    """ Train / evaluation data """

    def __init__(self, data_directory : str):
        """
            data_directory: Directory where .csv files are located
        """
       
        self._read_metadata(data_directory)
        self._read_data_files(data_directory)
        # TODO: Print summary


    def _read_metadata(self, data_directory : str):
        print("Reading data structure info")
        metadata_file_path = os.path.join( data_directory , 'data_info.json' )

        self.columns = []
        
        with open( metadata_file_path , 'r' , encoding='utf-8' )  as file:
            json_text = file.read()
            json_metadata = json.loads(json_text)
            #print(json_metadata)

            for json_column in json_metadata['ColumnsInfo']:
                self.columns.append( ColumnInfo( json_column['Name'] , json_column['Labels'] ) )


    def _read_data_files(self, data_directory : str):

        # Array of DataFile
        self._files = []

        print("Reading data files...")
        for file_name in os.listdir(data_directory):
            if not file_name.lower().endswith(".csv"):
                continue
            self._files.append( DataFile( data_directory , file_name ) )


    def traverse_sequences( self, padding_element : List , sequence_length : int ): 

        shuffled_files = self._files.copy()
        random.shuffle(shuffled_files)
        
        for data_file in shuffled_files:
            for row in data_file.get_sequences( padding_element , sequence_length ):
                yield row

