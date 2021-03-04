from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from time import time

# Read data definition
data_definition = ModelDataDefinition()

# Read all CSV paths
all_data = DataDirectory.read_all(data_definition)

ds = ClassifierDataset(all_data, data_definition, shuffle=False)

# Test entire eval dataset
print("Testing data set")

def print_some():
    for row in ds.dataset.take(50):
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

#print_some()
traverse_all()
