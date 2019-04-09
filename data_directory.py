from typing import List, Callable, Tuple
from data_file import DataFile
import os
import csv
import random
from model_data_definition import ModelDataDefinition
import tensorflow as tf

class DataDirectory:
    """ Train / evaluation data """

    def __init__(self):
        # Array of DataFile
        self._files = []

    def read_data_files(self, data_definition : ModelDataDefinition):

        print("Reading data files from", data_definition.data_directory)
        for file_name in os.listdir(data_definition.data_directory):
            if not file_name.lower().endswith(".csv"):
                continue
            #print( file_name )
            self._files.append( DataFile( data_definition , file_name ) )
        
        # Sort by file name to get reproducible results
        self._files.sort(key=lambda f: f.file_name)

    def get_shuffled_files(self) -> List[DataFile]:
        shuffled_files = self._files.copy()
        random.shuffle(shuffled_files)
        return shuffled_files

    def traverse_sequences( self, shuffle: bool = True ): 
        """ Traverse all sequences of all files on this data directory """
        if shuffle:
            shuffled_files = self.get_shuffled_files()
        else:
            shuffled_files = self._files

        # Traverse all sequences. Those sequences are ordered, will be shuffled by the TF dataset in TrainModel class
        for data_file in shuffled_files:
            for row in data_file.get_train_sequences():
                yield row


    def extract_evaluation_files(self, data_definition : ModelDataDefinition) -> object:
        """ Extract randomly a percentage of files to other DataDirectory """
        n_files_to_extract = int( len(self._files) * data_definition.percentage_evaluation )
        if n_files_to_extract <= 0:
            n_files_to_extract = 1

        new_data_dir = DataDirectory()
        for i in range(n_files_to_extract):
            file = random.choice( self._files )
            self._files.remove(file)
            new_data_dir._files.append(file)

        return new_data_dir


    def get_n_total_tokens(self) -> int:
        """ Total number of tokens in all files """
        return sum( len(file.file_rows) for file in self._files )


    def get_n_total_trainable_tokens(self, data_definition : ModelDataDefinition) -> int:
        """ Total number of tokens in all files """
        return sum( file.get_n_trainable_tokens(data_definition) for file in self._files )


    def print_summary(self, data_definition : ModelDataDefinition, name : str):
        """ Print summary with data files info """
        print(name, "summary:")
        print("N. files:" , len(self._files))
        total_tokens = self.get_n_total_tokens()
        print("Total n. tokens:" , total_tokens )
        print("Total n. TRAINABLE tokens:" , self.get_n_total_trainable_tokens( data_definition ) )
        print("Mean tokens / file:" , total_tokens / len(self._files))
        print("Maximum file tokens lenght:" , max( len(file.file_rows) for file in self._files ) )
        print()


    def get_column_values( self, data_definition : ModelDataDefinition, column_name: str) -> List[int]:
        column_index = data_definition.get_column_index(column_name)
        result = []
        for data_file in self._files:
            for row in data_file.file_rows:
                result.append( row[column_index] )
        return result

    def get_n_files(self) -> int:
        """ Get number of files on this instance """
        return len(self._files)

    def get_file(self, file_name : str ) -> DataFile:
        for file in self._files:
            if file.file_name == file_name:
                return file
        return None
        