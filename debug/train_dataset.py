"""
    Debug train dataset pipeline.
    Use
        python -m debug.train_dataset
"""

from data_directory import DataDirectory
from dataset.csv_files_dataset import CsvFilesDataset
from model_definition import ModelDefinition
from training.base_train import BaseTrain
import numpy as np

model_definition = ModelDefinition()
train_files, eval_files = DataDirectory.get_train_and_validation_sets(model_definition.data_definition)
train_ds = model_definition.dataset_class(train_files, model_definition.data_definition, shuffle=True, debug_columns=True)
BaseTrain.preprocess_train_dataset(model_definition.data_definition, train_ds)

for element in train_ds.dataset.take(32).as_numpy_iterator():
    x = element[0]
    y = element[1]

    file_names = x[CsvFilesDataset.FILE_KEY][:,0]
    seq_row_start = x[CsvFilesDataset.ROW_KEY][:,0]
    
    
    unique_files, seq_file_counts = np.unique(file_names, return_counts=True)

    # print("N. sequences per file in batch")
    # print(np.dstack((unique_files, seq_file_counts)))


    print("N. different files", len(unique_files))
    #print("Batch length", len(file_names))

    #print(element)
    # print(element[CsvFilesDataset.FILE_KEY][:,0])
    # print(element[CsvFilesDataset.ROW_KEY][:,0])
    #print(element[0][CsvFilesDataset.FILE_KEY])
    #print(len(element))
