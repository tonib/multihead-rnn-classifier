from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from train_model import TrainModel

#################################################
# Evaluates the model performance
#################################################

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
eval_data.print_summary(data_definition, "Evaluation data")

# Create model
print("Creating model...")
model = TrainModel( data_definition )

# Evaluate model
print()
model.evaluate( eval_data )
