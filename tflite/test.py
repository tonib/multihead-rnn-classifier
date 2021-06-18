import tensorflow as tf
from model_definition import ModelDefinition
from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
import numpy as np
from time import time

# OK: In tf 2.3 / 2.4, named input/outputs are pretty broken:
# - If you convert a TF module with TFLiteConverter.from_saved_model, you get wrong output names: Output names are *tensor* name
#   (ex. StatefulPartionedCall:0), not the real *output* name
# - If you convert a TF module with TFLiteConverter.from_saved_model, input names are weird ("serving_default_REALINPUTNAME")
# - If you convert a TF module with TFLiteConverter.from_concrete_functions, input names seems OK
# - If you convert with TFLiteConverter.from_concrete_functions, you get wrong output names: Names are Identity, Identity_1 (beautiful)
# It seems these issues are fixed in 2.5 (not published yet): https://github.com/tensorflow/tensorflow/issues/32180#issuecomment-772140542
# Output names seems can be fixed because they seem to be sorted by name: So, if you sort the real names, you can map them by position

model_definition = ModelDefinition()
data_definition = model_definition.data_definition

path = data_definition.get_data_dir_path( ModelDataDefinition.TFLITE_PATH )

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

# print(len(input_details))
# print(input_details)
# print(output_details)
# exit()

# Inputs
element = model_definition.predictor_class.get_empty_element(data_definition)
for column_name in element:
    element[column_name] = np.array(element[column_name], dtype=np.int32)
    # print(element[column_name])
    element[column_name] = np.pad( element[column_name] , (0, data_definition.sequence_length - element[column_name].shape[0] ), 
        'constant' , constant_values=(-1) )
    # print(element[column_name])
# print(element)
# exit()

# element = {}
# for column_name in data_definition.sequence_columns:
#     py_input = [RnnDataset.EOS_VALUE] + ([0]*63)
#     element[column_name] = np.array( [py_input], dtype=np.int32)
#     print(column_name, element[column_name])
# for column_name in data_definition.context_columns:
#     element[column_name] = np.array([0], dtype=np.int32)

def set_values():
    # Set input values
    for input_spec in input_details:
        #name = input_spec["name"][ len("serving_default_") : -2 ]
        #print(input_spec)
        #print(name, element[name].shape, element[name], input_spec)
        interpreter.set_tensor(input_spec["index"], element[input_spec["name"]])

# for column_name in element:
#     # serving_default_ctxParmCollection:0
#     interpreter.set_tensor(input_details[0]["serving_default_" + column_name + ":0"], element[column_name])

# set_values()
# interpreter.invoke()

# output = {}
# for output_spec in output_details:
#     print(output_spec["name"])
#     #name = output_spec["name"][ len("serving_default_") : -2 ]
#     name = output_spec["name"]
#     output[name] = interpreter.get_tensor(output_spec['index'])

# print(output)


n_repetitions = 1000
print("Testing performance, n. repetitions:" , n_repetitions)
start = time()
for i in range(n_repetitions):
    set_values()
    interpreter.invoke()
end = time()
print("Total time:" , end - start , "s")
print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")
