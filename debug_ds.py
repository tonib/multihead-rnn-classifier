from data_directory import DataDirectory
from model_definition import ModelDefinition
from dataset.csv_files_dataset import CsvFilesDataset
from dataset.rnn_dataset import RnnDataset
from dataset.transformer_dataset import TransformerDataset
from time import time
import tensorflow as tf
import json
import numpy as np

def pretty(title, row):
    if isinstance(row, dict):
        serializable_row = {}
        for key in row:
            serializable_row[key] = row[key].numpy()
            if key == CsvFilesDataset.FILE_KEY:
                if isinstance(serializable_row[key], np.ndarray):
                    # Transformer (array of strings)
                    serializable_row[key] = [ s.decode('UTF-8') for s in serializable_row[key] ]
                else:
                    # rnn (single string)
                    serializable_row[key] = serializable_row[key].decode('UTF-8')
            else:
                serializable_row[key] = serializable_row[key].tolist()
    else:
        serializable_row = row.numpy().tolist()
    
    print(title, json.dumps(serializable_row, sort_keys=True) )

def print_some(print_pretty):
    for row in ds.dataset.take(1000):
        if print_pretty:
            pretty("*** Input:", row[0])
            pretty("*** Output:", row[1])
            print("\n")
        else:
            print(row)

def traverse_all():

    BATCH_SIZE = 64

    def performance(start, n_batches):
        elapsed_time = time() - start
        n_elements = n_batches * BATCH_SIZE
        print("Batches:", n_batches, "Elements:", n_elements, "Time (s):", elapsed_time, "Elements/s:", n_elements / elapsed_time)
        print()

    n_batches = 0
    start = time()
    for _ in ds.dataset.batch(BATCH_SIZE):
        n_batches += 1
        if n_batches % 50 == 0:
            performance(start, n_batches)

    performance(start, n_batches)

if __name__ == '__main__':

    # Read data definition
    model_definition = ModelDefinition()

    # Read all CSV paths
    all_data = DataDirectory.read_all(model_definition.data_definition)

    # True -> Test dataset performance, False -> Print ds values
    TEST_PERFORMANCE = False

    # Create dataset for this model type
    print("Dataset type:", model_definition.dataset_class)
    ds = model_definition.dataset_class(all_data, model_definition.data_definition, shuffle=TEST_PERFORMANCE, debug_columns=not TEST_PERFORMANCE)

    # Test entire eval dataset
    print("Testing data set")

    if TEST_PERFORMANCE:
        traverse_all()
    else:
        print_some(True)

