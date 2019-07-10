from prediction_model import PredictionModel
from model_data_definition import ModelDataDefinition
from data_directory import DataDirectory
import json

####################################################
# DEBUG PREDICTIONS ON EVALUATION DATA SET
####################################################

# Read data definition
data_definition = ModelDataDefinition()

# Read data
train_data = DataDirectory()
train_data.read_data_files( data_definition )

# Extract evaluation files
eval_data = train_data.extract_evaluation_files( data_definition )

print("Reading latest exported model")
predictor = PredictionModel(data_definition)

def input_to_json_string( input_row : dict ) -> str:
    """ Dataset rows contain numpy arrays that are not serializable to JSON.
    This converts all numpy arrays to python arrays, and returns the input
    as a JSON object. Same with context columns (int64 scalars) """

    for column_name in data_definition.sequence_columns:
        input_row[ column_name ] = input_row[ column_name ].tolist()

    for column_name in data_definition.context_columns:
        input_row[ column_name ] = int( input_row[ column_name ] )

    return json.dumps(input_row)

# Test eval data set:
n_succeed = 0
n_total = 0
for data_file in eval_data.get_files():
    for row in data_file.get_train_sequences():
        print("----------------------------------")
        print("Input:", input_to_json_string(row[0]) )
        print("Output:", row[1])

        real_output_idx = row[1]['outputTypeIdx']
        real_label = data_definition.column_definitions['outputTypeIdx'].labels[ real_output_idx ]

        prediction = predictor.predict(row[0], data_definition)
        print( "Prediction:" , prediction )
        predicted_idx = prediction['outputTypeIdx']['class_prediction']
        predicted_label = data_definition.column_definitions['outputTypeIdx'].labels[ predicted_idx ]
        prediction_ok = ( real_output_idx == predicted_idx )

        print()
        print( "Real label:", real_label , "real class:" , real_output_idx )        
        print( "outputTypeIdx predicted label:" , predicted_label , "predicted class:" , predicted_idx , "Ok?:", prediction_ok)

        if prediction_ok:
            n_succeed += 1
        n_total += 1

        print("----------------------------------")

print("N. total:" , n_total, ", n. ok predictions:" , n_succeed , ", ratio:" , n_succeed/n_total )
