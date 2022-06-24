"""
SCRIPT TO DEBUG MODEL EXECUTION IN EAGER MODE
"""

from model_definition import ModelDefinition
import tensorflow as tf

# Disable @tf.function decorators
tf.config.run_functions_eagerly(True)

model_definition = ModelDefinition()
model = model_definition.create_model_function(model_definition.data_definition)

predictor = model_definition.predictor_class(model_definition.data_definition, model)

element = model_definition.predictor_class.get_empty_element(model_definition.data_definition)
for key in element:
    element[key] = tf.constant(element[key], dtype=tf.int32)
#print(element)

y = predictor._predict_tf(element)
#print(y)

