from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from train_model import TrainModel
import tensorflow as tf

print("Tensorflow version:" , tf.__version__ )

# Read data definition
data_definition = ModelDataDefinition()
data_definition.print_summary()
print()

# Read data
train_data = DataDirectory()
train_data.read_data_files( data_definition )

# Extract evaluation files
eval_data = train_data.extract_evaluation_files( data_definition )

# Print summary
print()
train_data.print_summary(data_definition, "Train data")
eval_data.print_summary(data_definition, "Evaluation data")

if len( train_data.get_files() ) == 0 or len( eval_data.get_files() ) == 0:
    print("ERROR: No files enough to train")
    exit()

# Create model
print("Creating model...")
model = TrainModel( data_definition )

# Training loop
print()
model.train_model( train_data , eval_data )

