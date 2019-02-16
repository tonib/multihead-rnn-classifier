from typing import List, Callable, Tuple
from data_file import DataFile
import os
import csv
import random
from model_data_definition import ModelDataDefinition

class DataDirectory:
    """ Train / evaluation data """

    def __init__(self, data_definition : ModelDataDefinition):
        """
            data_directory: Directory where .csv files are located
        """
       
        self._read_data_files(data_definition)
        # TODO: Print summary


    def _read_data_files(self, data_definition : ModelDataDefinition):

        # Array of DataFile
        self._files = []

        print("Reading data files from ", data_definition.data_directory)
        for file_name in os.listdir(data_definition.data_directory):
            if not file_name.lower().endswith(".csv"):
                continue
            self._files.append( DataFile( data_definition , file_name ) )


    def traverse_sequences( self, data_definition : ModelDataDefinition ): 

        shuffled_files = self._files.copy()
        random.shuffle(shuffled_files)
        
        for data_file in shuffled_files:
            for row in data_file.get_sequences( data_definition.padding_element , data_definition.sequence_length ):
                yield row

