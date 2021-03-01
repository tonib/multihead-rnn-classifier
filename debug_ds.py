from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset

# Read data definition
data_definition = ModelDataDefinition()

# Read all CSV paths
all_data = DataDirectory.read_all(data_definition)

ds = ClassifierDataset(all_data, data_definition, shuffle=False)

# Test entire eval dataset
print("Testing data set")
for row in ds.dataset.take(3):
    print(row)


