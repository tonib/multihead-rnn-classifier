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

        print("Reading data files from ", data_definition.data_directory)
        for file_name in os.listdir(data_definition.data_directory):
            if not file_name.lower().endswith(".csv"):
                continue
            self._files.append( DataFile( data_definition , file_name ) )


    def traverse_sequences( self, data_definition : ModelDataDefinition ): 

        shuffled_files = self._files.copy()
        random.shuffle(shuffled_files)
        
        for data_file in shuffled_files:
            for row in data_file.get_sequences( data_definition ):
                yield row

    def extract_evaluation_files(self, percentage : float) -> object:
        """ Extract randomly a percentage of files to other DataDirectory """
        n_files_to_extract = int( len(self._files) * percentage )

        new_data_dir = DataDirectory()
        for i in range(n_files_to_extract):
            file = random.choice( self._files )
            self._files.remove(file)
            new_data_dir._files.append(file)

        return new_data_dir

    def get_n_total_tokens(self) -> int:
        """ Total number of tokens in all files """
        return sum( len(file.file_rows) for file in self._files )

    def print_summary(self, name : str):
        """ Print summary with data files info """
        print(name, "summary:")
        print("N. files:" , len(self._files))
        total_tokens = self.get_n_total_tokens()
        print("Total n. tokens:" , total_tokens )
        print("Mean tokens / file:" , total_tokens / len(self._files))
        print("Maximum file tokens lenght:" , max( len(file.file_rows) for file in self._files ) )
        print()

    def get_tf_input_fn(self, data_definition : ModelDataDefinition ) -> Callable:
        """ Returns the Tensorflow input function for data files """
        # The dataset
        ds = tf.data.Dataset.from_generator( 
            generator=lambda: self.traverse_sequences( data_definition ), 
            output_types = data_definition.model_input_output_types(),
            output_shapes = data_definition.model_input_output_shapes()
        )
        #ds = ds.repeat(1000)
        ds = ds.shuffle(5000)
        ds = ds.batch(64)
        ds = ds.prefetch(64)

        return ds