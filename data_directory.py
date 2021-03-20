from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition

import glob
import os
import random

class DataDirectory:
    """ List of train / evaluation / test data file paths """

    def __init__(self, file_paths:List[str] = None):
        # Array of file paths
        self.file_paths = file_paths if file_paths != None else []

    @property
    def file_paths(self) -> List[str]:
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths: List[str]):
        self._file_paths = file_paths
        # Sort to get reproducible results
        self._file_paths.sort()

    def save(self, file_path: str):
        """ Save the list of paths to the given file """
        print("Saving files list to", file_path)
        with open(file_path, 'w') as f:
            for file_path in self.file_paths:
                print(file_path, file=f)

    def sample(self, n_samples: int) -> DataDirectory:
        """ Get n_samples random files from this set """
        return DataDirectory( random.sample(self.file_paths, n_samples) )

    def substract(self, to_substract_set: DataDirectory) -> DataDirectory:
        """ Return the difference between two datasets """
        return DataDirectory( [file_path for file_path in self.file_paths if file_path not in to_substract_set.file_paths] )

    @staticmethod
    def read_all(data_definition : ModelDataDefinition) -> DataDirectory:
        """ Read all CSV files on data directory """
        # root_dir needs a trailing slash (i.e. /root/dir/)
        print("Searching all csv files")
        return DataDirectory([path for path in glob.iglob(data_definition.data_directory + '/**/*.csv', recursive=True) ])

    @staticmethod
    def load_from_file(file_path: str) -> DataDirectory:
        """ Load set of paths stored in the given file. Returns None if the file don't exist """
        if not os.path.isfile( file_path ):
            return None
        
        # Read file lines, each line is a file name
        print("Loading files list from", file_path)
        with open(file_path) as f:
            return DataDirectory(f.read().splitlines())

    @staticmethod
    def get_train_and_validation_sets(data_definition : ModelDataDefinition) -> Tuple[DataDirectory, DataDirectory]:
        """ Returns (Train dataset, Validation dataset) """

        # Full set of files
        full_set = DataDirectory.read_all(data_definition)

        # Load evaluation dataset
        print("Loading evaluation set")
        eval_set_path = data_definition.get_data_dir_path( 'validationSet.txt' )
        eval_set = DataDirectory.load_from_file( eval_set_path )
        if eval_set == None:
            # If it don't exists, create it
            n_samples = int( round( data_definition.percentage_evaluation * len(full_set.file_paths) ) )
            eval_set = full_set.sample( n_samples )
            eval_set.save( eval_set_path )

        # Substract evaluation dataset from full dataset
        print("Calculate train dataset")
        train_set = full_set.substract( eval_set )
        if len(eval_set.file_paths) == 0:
            # Dataset too small
            print("Warning: Evaluation dataset was empty: Using train dataset as evaluation dataset")
            eval_set = train_set
        
        return train_set, eval_set
