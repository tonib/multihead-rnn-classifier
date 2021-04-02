import configure_tf_log # Must be FIRST import
from model_data_definition import ModelDataDefinition
from predictor import Predictor
import sys
import json
import traceback

# TODO: Define buffer size for std output ???

# Read data definition
data_definition = ModelDataDefinition.from_file()

print("# Reading exported model", flush=True)
predictor = Predictor(data_definition)

print("# Sample:", json.dumps( data_definition.get_empty_element() ) , flush=True)

# This text will be used as flag to start sending requests
print("READY TO SERVE", flush=True)

while True:
    try:
        txt_line = sys.stdin.readline()
        #print("# input_array: " + " ".join(str(x) for x in input_array))
        print( predictor.predict_json(txt_line) , flush=True )

        # sys.stdout.write( result + '\n')
        # sys.stdout.flush()
    except Exception as e:
        # Return exception info
        error_info = {}
        error_info['error'] = str(e) + '\n' + traceback.format_exc()
        print( json.dumps(error_info) , flush=True )
