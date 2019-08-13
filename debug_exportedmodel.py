from prediction_model import PredictionModel
from model_data_definition import ModelDataDefinition
from data_directory import DataDirectory
import json
import os

####################################################
# DEBUG PREDICTIONS ON EVALUATION DATA SET
####################################################

# Read data definition
data_definition = ModelDataDefinition()

# Read test data set
test_data = DataDirectory()
test_data.read_test_files( data_definition )

print("Reading latest exported model")
predictor = PredictionModel(data_definition)

# Create debug directory if it does not exists
debug_dir_path = data_definition.get_debug_dir_path()
if not os.path.isdir(debug_dir_path):
    print("Creating directory", debug_dir_path)
    os.mkdir(debug_dir_path)

def input_to_serializable( input_row : dict ):
    """ Dataset rows contain numpy arrays that are not serializable to JSON.
    This converts all numpy arrays to python arrays, and returns the input
    as a JSON object. Same with context columns (int64 scalars) """

    for column_name in data_definition.sequence_columns:
        input_row[ column_name ] = input_row[ column_name ].tolist()

    for column_name in data_definition.context_columns:
        input_row[ column_name ] = int( input_row[ column_name ] )

def output_to_serializable( output_row : dict ):
    """ Same as input_to_serializable, but for outputs """
    for column_name in data_definition.output_columns:
        output_row[ column_name ] = int( output_row[ column_name ] )

# Column name for succeed ratios:
succeed_column_name = 'outputTypeIdx'

# Test eval data set:
total_succeed = 0
total = 0
for data_file in test_data.get_files():
    file_debug = []
    file_succeed = 0
    file_total = 0

    for row in data_file.get_train_sequences():
        sequence = {}

        input_to_serializable(row[0])
        output_to_serializable(row[1])

        sequence['input'] = row[0]
        sequence['output'] = row[1]
        
        real_output_idx = row[1][succeed_column_name]

        prediction = predictor.predict(row[0], data_definition)
        sequence['prediction'] = prediction
        predicted_idx = prediction[succeed_column_name]['class_prediction']
        prediction_ok = ( real_output_idx == predicted_idx )

        file_debug.append( sequence )

        if prediction_ok:
            file_succeed += 1
        file_total += 1

    if file_total > 0:
        print(data_file.file_name, ":" , file_total, ", n. ok predictions:" , file_succeed , ", ratio:" , file_succeed/file_total )

    total_succeed += file_succeed
    total += file_total

    debug_file_path = os.path.join( debug_dir_path , data_file.file_name + ".json" )
    with open( debug_file_path , 'w') as outfile:  
        json.dump(file_debug, outfile)

if total > 0:
    print("N. total:" , total, ", n. ok predictions:" , total_succeed , ", ratio:" , total_succeed/total )
