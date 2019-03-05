from model_data_definition import ModelDataDefinition
from train_model import TrainModel
from prediction_model import PredictionModel
import tensorflow as tf
import os

# Read data definition
data_definition = ModelDataDefinition()

# Create model
print("\nCreating train model...")
model = TrainModel( data_definition )

print("\nExporing model...")
model.export_model()
