import tensorflow as tf
from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
import numpy as np
from time import time

data_definition = ModelDataDefinition.from_file()
exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )

path = data_definition.get_data_dir_path( 'model/model.tflite' )

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

# # There is only 1 signature defined in the model,
# # so it will return it by default.
# # If there are multiple signatures then we can pass the name.
# my_signature = interpreter.get_signature_runner()
# print(my_signature)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#print(input_details[0])
#exit()
# print(output_details)

# Inputs
element = {}
for column_name in data_definition.sequence_columns:
    py_input = [RnnDataset.EOS_VALUE] + ([0]*63)
    element[column_name] = np.array( [py_input], dtype=np.int32)
    print(column_name, element[column_name])
for column_name in data_definition.context_columns:
    element[column_name] = np.array([0], dtype=np.int32)

def set_values():
    # Set input values
    for input_spec in input_details:
        name = input_spec["name"][ len("serving_default_") : -2 ]
        #print(input_spec)
        #print(i, name, element[name].shape, element[name])
        interpreter.set_tensor(input_spec["index"], element[name])

# for column_name in element:
#     # serving_default_ctxParmCollection:0
#     interpreter.set_tensor(input_details[0]["serving_default_" + column_name + ":0"], element[column_name])

#interpreter.invoke()

# output = {}
# for output_spec in output_details:
#     print(output_spec["name"])
#     name = output_spec["name"][ len("serving_default_") : -2 ]
#     output[name] = interpreter.get_tensor(output_spec['index'])

# print(output)


n_repetitions = 10000
print("Testing performance, n. repetitions:" , n_repetitions)
start = time()
for i in range(n_repetitions):
    set_values()
    interpreter.invoke()
end = time()
print("Total time:" , end - start , "s")
print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")
