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

def debug_preprocessing(input):
    # Convert input python values to tensors
    for key in input:
        input[key] = tf.constant(input[key], dtype=tf.int32)

    pretty( "Preprocessed:", input )
    pretty( "Postprocessed:", predictor._preprocess_input(input) )
    print()

def debug_gpt_preprocessing():
    for length in range( data_definition.sequence_length + 2 ):
        input = {}
        for idx, col_name in enumerate(data_definition.sequence_columns):
            input[col_name] = range(length)
        for idx, col_name in enumerate(data_definition.context_columns):
            input[col_name] = range(length+1)
        debug_preprocessing( input )

def pretty_prediction(prediction, data_definition):
    for col_name in prediction:
        # Get top predictions:
        probabilities = prediction[col_name]["probabilities"]
        #print(probabilities)
        top_indices = sorted(range(len(probabilities)), key=lambda k: -probabilities[k])
        #print(top_indices)
        n_top_indices = 5
        if len(top_indices) > n_top_indices:
            top_indices = top_indices[:n_top_indices]
        print(col_name, ":", str( { "(" + str(idx) + ") " + data_definition.column_definitions[col_name].labels[idx] : probabilities[idx] for idx in top_indices } ) )

def debug_gpt_prediction():
    print("Empty events from workpanel")
    input = {
        "controlType": [], "ctxIsVariable": [0], "ctxParmAccess": [0], "ctxParmCollection": [2], "ctxParmDecimals": [10], 
        "ctxParmExtTypeHash": [0], "ctxParmLength": [18], "ctxParmType": [0], "dataTypeExtTypeHash": [], "dataTypeIdx": [], 
        "decimalsBucket": [], "isCollection": [], "kbObjectTypeIdx": [], "keywordIdx": [], "lengthBucket": [], 
        "objectType": [4], "partType": [0], "textHash0": [], "textHash1": [], "textHash2": [], "textHash3": [], "wordType": []
    }
    prediction = predictor.predict(input)
    pretty_prediction(prediction, data_definition)
    print()

    print("Empty procedure part from procedure")
    input = {
        "controlType": [], "ctxIsVariable": [0], "ctxParmAccess": [0], "ctxParmCollection": [2], "ctxParmDecimals": [10], 
        "ctxParmExtTypeHash": [0], "ctxParmLength": [18], "ctxParmType": [0], "dataTypeExtTypeHash": [], "dataTypeIdx": [], 
        "decimalsBucket": [], "isCollection": [], "kbObjectTypeIdx": [], "keywordIdx": [], "lengthBucket": [], 
        "objectType": [2], "partType": [2], "textHash0": [], "textHash1": [], "textHash2": [], "textHash3": [], "wordType": []
    }
    #debug_preprocessing(input)
    prediction = predictor.predict(input)
    pretty_prediction(prediction, data_definition)
    print()

def debug_rnn_prediction():
    print("Empty events from workpanel")
    input = {
        "controlType": [], "ctxIsVariable": 0, "ctxParmAccess": 0, "ctxParmCollection": 2, "ctxParmDecimals": 10, 
        "ctxParmExtTypeHash": 0, "ctxParmLength": 18, "ctxParmType": 0, "dataTypeExtTypeHash": [], "dataTypeIdx": [], 
        "decimalsBucket": [], "isCollection": [], "kbObjectTypeIdx": [], "keywordIdx": [], "lengthBucket": [], 
        "objectType": 4, "partType": 0, "textHash0": [], "textHash1": [], "textHash2": [], "textHash3": [], "wordType": []
    }
    prediction = predictor.predict(input)
    pretty_prediction(prediction, data_definition)
    print()

    print("Empty procedure part from procedure")
    input = {
        "wordType": [], "keywordIdx": [], "kbObjectTypeIdx": [], "dataTypeIdx": [], "dataTypeExtTypeHash": [], "isCollection": [], 
        "lengthBucket": [], "decimalsBucket": [], "textHash0": [], "textHash1": [], "textHash2": [], "textHash3": [], "controlType": [], 
        "ctxParmType": 0, "ctxParmExtTypeHash": 0, "ctxParmLength": 18, "ctxParmDecimals": 10, "ctxParmCollection": 2, 
        "ctxParmAccess": 0, "ctxIsVariable": 0, "objectType": 2, "partType": 2
    }
    #debug_preprocessing(input)
    prediction = predictor.predict(input)
    pretty_prediction(prediction, data_definition)
    print()

if __name__ == '__main__':

    model_definition = ModelDefinition()
    data_definition = model_definition.data_definition
    predictor = model_definition.predictor_class(data_definition)

    # Debug empty element preprocessing
    # debug_preprocessing( predictor.get_empty_element() )

    if data_definition.model_type == "gpt":
        debug_gpt_prediction()
    else:
        debug_rnn_prediction()
