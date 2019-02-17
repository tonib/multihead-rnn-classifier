from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from model import Model
import tensorflow as tf

# Read data
data_definition = ModelDataDefinition( 'data' )
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

while True:
    print("Training...")
    model.estimator.train( input_fn=lambda:train_data.get_tf_input_fn( data_definition ) )

    print("Evaluating...")
    result = model.estimator.evaluate( input_fn=lambda:eval_data.get_tf_input_fn( data_definition ) )
    print("Evaluation: ", result)
