from model_data_definition import ModelDataDefinition
from dataset.transformer_dataset import TransformerDataset
from data_directory import DataDirectory
import json

# Read data definition
data_definition = ModelDataDefinition.from_file()
json.dumps(data_definition.__dict__)

# Read all CSV paths
all_data = DataDirectory.read_all(data_definition)

ds = TransformerDataset(all_data, data_definition, shuffle=False, debug_columns=True)

ds.dataset = ds.dataset.batch(1).take(3)
#ds.dataset = ds.dataset.take(3)

for row in ds.dataset:
    print(row)

# for row in ds.dataset:
#     pass

#print( list(ds.dataset.as_numpy_iterator()) )
