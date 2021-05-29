from re import I
import configure_tf_log # Must be FIRST import
from model_definition import ModelDefinition
from predict.predictor import Predictor
from tflite.predictor_lite import PredictorLite
import sys
import json
import traceback
import os

# TODO: Define buffer size for std output ???

# Read data definition
model_definition = ModelDefinition()

# Check what model type to use (full TF or TF lite): TF lite is default option
tflite_path = PredictorLite.get_tflite_model_path(model_definition.data_definition)
if os.path.isfile(tflite_path):
    print("# Reading TF lite file from " + tflite_path, flush=True)
    predictor = PredictorLite(model_definition)
else:
    print("# Reading exported model (full TF model)", flush=True)
    predictor = Predictor(model_definition)

print("# Sample:", json.dumps( predictor.get_empty_element() ) , flush=True)

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
