from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from time import time
import tensorflow as tf
import json

# Read data definition
data_definition = ModelDataDefinition()

# Read all CSV paths
all_data = DataDirectory.read_all(data_definition)

# True -> Test dataset performance, False -> Print ds values
TEST_PERFORMANCE = False

ds = ClassifierDataset(all_data, data_definition, shuffle=TEST_PERFORMANCE, debug_columns=not TEST_PERFORMANCE)

# Test entire eval dataset
print("Testing data set")

def pretty(title, row):
    if isinstance(row, dict):
        serializable_row = {}
        for key in row:
            serializable_row[key] = row[key].numpy()
            if key == ClassifierDataset.FILE_KEY:
                serializable_row[key] = serializable_row[key].decode('UTF-8')
            else:
                serializable_row[key] = serializable_row[key].tolist()
    else:
        serializable_row = row.numpy().tolist()
    
    print(title, json.dumps(serializable_row, sort_keys=True) )

def print_some(print_pretty):
    for row in ds.dataset.take(1000):
        if print_pretty:
            pretty("Input:", row[0])
            pretty("Output:", row[1])
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
            print(n_batches * BATCH_SIZE)
            performance(start, n_batches)

    performance(start, n_batches)

if TEST_PERFORMANCE:
    traverse_all()
else:
    print_some(True)

