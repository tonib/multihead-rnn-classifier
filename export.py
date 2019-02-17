from model_data_definition import ModelDataDefinition
from model import Model
import tensorflow as tf
import os

# Read data definition
data_definition = ModelDataDefinition( 'data' )

# Create model
print("Creating model...")
model = Model( data_definition )

path = 'exportedmodel'
print("Exporting to" , path)
model.export_model( path , data_definition )
