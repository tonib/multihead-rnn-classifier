from model_data_definition import ModelDataDefinition
from train_model import TrainModel
from prediction_model import PredictionModel
import tensorflow as tf
import os

# Read data definition
data_definition = ModelDataDefinition( 'data' )

# Create model
print("Creating model...")
model = TrainModel( data_definition )

print("Exporting to" , PredictionModel.EXPORTED_MODELS_DIR_PATH)
model.export_model( data_definition )
