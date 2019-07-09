from data_directory import DataDirectory
from model_data_definition import ModelDataDefinition
from train_model import TrainModel

# Read data definition
data_definition = ModelDataDefinition()

# Read data
train_data = DataDirectory()
train_data.read_data_files( data_definition )

# Extract evaluation files
eval_data = train_data.extract_evaluation_files( data_definition )

# Test single file:
# data_file = eval_data.get_file( 'PAlcELin.csv' )
# for row in data_file.get_train_sequences():
#     print("----------------------------------")
#     print("Input:", row[0])
#     print("Output:", row[1])
#     print("----------------------------------")

# Test entire eval dataset
print("Testing data set")
for row in eval_data.traverse_sequences( False ):
    print(row)


