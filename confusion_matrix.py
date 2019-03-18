from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from train_model import TrainModel
import tensorflow as tf

# Read data definition
data_definition = ModelDataDefinition()

# Read data
data = DataDirectory()
data.read_data_files( data_definition )
#data = data.extract_evaluation_files(0.01)

# Create model
print("Creating model...")
model = TrainModel( data_definition )
model.confusion_matrix( data , 'outputTypeIdx' )
