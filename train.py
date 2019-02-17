from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from model import Model
import tensorflow as tf

# Read data definition
data_definition = ModelDataDefinition( 'data' )

# Read data
train_data = DataDirectory()
train_data.read_data_files( data_definition )

# print("Testing data set")
# for row in data_dir.traverse_sequences( data_definition ):
#     print(row)

# Extract 15% of files for evaluation
eval_data = train_data.extract_evaluation_files(0.15)

# Print summary
train_data.print_summary("Train data")
eval_data.print_summary("Evaluation data")

# Create model
print("Creating model...")
model = Model( data_definition )

# Training loop
# TODO: When to stop ???
model.train_model( train_data , eval_data , data_definition )
