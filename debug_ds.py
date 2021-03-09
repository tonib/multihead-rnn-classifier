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

ds = ClassifierDataset(all_data, data_definition, shuffle=False, debug_columns=False)

# Test entire eval dataset
print("Testing data set")

def pretty(title, row):
    pretty_dict = {}
    for key in row:
        pretty_dict[key] = row[key].numpy()
        if key == ClassifierDataset.FILE_KEY:
            pretty_dict[key] = pretty_dict[key].decode('UTF-8')
        else:
            pretty_dict[key] = pretty_dict[key].tolist()
        
    print(title, json.dumps(pretty_dict, sort_keys=True) )

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
    n_batches = 0
    start = time()
    for _ in ds.dataset.batch(BATCH_SIZE):
        n_batches += 1
        if n_batches % 100 == 0:
            print(n_batches * BATCH_SIZE)
    elapsed_time = time() - start
    n_elements = n_batches * BATCH_SIZE
    print("Total:", n_elements, "Time (s):", elapsed_time, "Elements/s:", n_elements / elapsed_time)

#print_some(True)
#print_some(False)
traverse_all()
