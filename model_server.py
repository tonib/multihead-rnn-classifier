from prediction_model import PredictionModel
from model_data_definition import ModelDataDefinition
import sys
import json

# Read data definition
data_definition = ModelDataDefinition( 'data' )

print("# Reading latest exported model")
predictor = PredictionModel()

# This text will be used as flag to start sending requests
print("READY TO SERVE")

print("# Sample:", json.dumps( [data_definition.get_padding_element() ] * data_definition.sequence_length ) )

while True:
    # TODO: Handle exceptions
    txt_line = sys.stdin.readline()
    #print("# input_array: " + " ".join(str(x) for x in input_array))
    result = predictor.predict_json(txt_line, data_definition)
    sys.stdout.write( result + '\n')
