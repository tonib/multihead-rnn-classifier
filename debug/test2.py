from model_data_definition import ModelDataDefinition
from model.mingpt.model_adapted import GPT, GPT1Config
from dataset.transformer_dataset import TransformerDataset
import tensorflow as tf
import numpy as np
from data_directory import DataDirectory

data_definition = ModelDataDefinition.from_file()

model = GPT(GPT1Config(), data_definition)

# Read all CSV paths
all_data = DataDirectory.read_all(data_definition)

ds = TransformerDataset(all_data, data_definition, shuffle=False, debug_columns=False)

#ds.dataset = ds.dataset.batch(1).take(3)
ds.dataset = ds.dataset.batch(1).take(1)

for row in ds.dataset:
    input = row[0]
    print(input)
    output = model(input)
    print( output )
    #print( tf.shape(output) )