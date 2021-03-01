from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from model_data_definition import ModelDataDefinition

import glob

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

    @staticmethod
    def read_all(data_definition : ModelDataDefinition) -> DataDirectory:
        """ Read all CSV files on data directory """
        # root_dir needs a trailing slash (i.e. /root/dir/)
        return DataDirectory([path for path in glob.iglob(data_definition.data_directory + '/**/*.csv', recursive=True) ])

