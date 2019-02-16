#from enum import Enum
from typing import List, Callable, Tuple
from data_file import DataFile
import os
import csv
import random

class DataDirectory:
    """ Train / evaluation data """

    def __init__(self, data_directory : str):
        """
            data_directory: Directory where .csv files are located
        """

        # Array of DataFile
        self._files = []

        print("Traversing data directory")
        for file_name in os.listdir(data_directory):
            if not file_name.lower().endswith(".csv"):
                continue
            self._files.append( DataFile( data_directory , file_name ) )

        # TODO: Print summary

    def traverse_sequences( self, padding_element : List , sequence_length : int ): 
        
        shuffled_files = self._files.copy()
        random.shuffle(shuffled_files)
        
        for data_file in shuffled_files:
            for row in data_file.get_sequences( padding_element , sequence_length ):
                yield row

