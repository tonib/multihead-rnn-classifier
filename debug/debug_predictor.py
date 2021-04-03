"""
    Debug predictor classes
"""

import configure_tf_log # Must be FIRST import
from predict.rnn_predictor import RnnPredictor
from model_definition import ModelDefinition
from debug_ds import pretty
from time import time
import json
import tensorflow as tf


model_definition = ModelDefinition()
data_definition = model_definition.data_definition
predictor = model_definition.predictor_class(data_definition)


def debug_preprocessing(input):
    # Convert input python values to tensors
    for key in input:
        input[key] = tf.constant(input[key], dtype=tf.int32)

    pretty( "Preprocessed:", input )
    pretty( "Postprocessed:", predictor._preprocess_input(input) )
    print()

# debug_preprocessing( predictor.get_empty_element() )

# Debug GPT input preprocessing
for length in range( data_definition.sequence_length + 2 ):
    input = {}
    for idx, col_name in enumerate(data_definition.sequence_columns):
        input[col_name] = range(length)
    for idx, col_name in enumerate(data_definition.context_columns):
        input[col_name] = range(length+1)
    debug_preprocessing( input )
