from model_definition import ModelDefinition
from tflite.predictor_lite import PredictorLite
from predict.predictor import Predictor
from debug.debug_predictor import pretty_prediction
from time import time
import random
import numpy as np

model_definition = ModelDefinition()

# TF Lite prediction
predictor_lite = PredictorLite(model_definition)
empty_element = predictor_lite.get_empty_element()

# Full TF prediction
predictor = Predictor(model_definition)

def test_lite_performance():
    # Test performance
    n_repetitions = 1000
    print("Testing performance, n. repetitions:" , n_repetitions)
    start = time()
    for i in range(n_repetitions):
        predictor_lite.predict( empty_element )
    end = time()
    print("Total time:" , end - start , "s")
    print("Prediction performance:" , ((end - start) / n_repetitions) * 1000 , "ms")

def test_empty_prediction():
    prediction = predictor_lite.predict( empty_element )
    pretty_prediction(prediction, model_definition.data_definition)
    print()

    prediction = predictor.predict( empty_element )
    pretty_prediction(prediction, model_definition.data_definition)

def test_random():
    for _ in range(10):
        input = {}
        seq_len = random.randint(0, model_definition.data_definition.sequence_length)
        #seq_len = model_definition.data_definition.sequence_length

        for col_name in model_definition.data_definition.sequence_columns:
            col_definition = model_definition.data_definition.column_definitions[col_name]
            input[col_name] = np.random.randint(0, len(col_definition.labels), seq_len, np.int32)

        for col_name in model_definition.data_definition.context_columns:
            col_definition = model_definition.data_definition.column_definitions[col_name]
            if model_definition.data_definition.model_type == "gpt" or model_definition.data_definition.model_type == "exp":
                # Array of sequence length + 1 (item to predict)
                input[col_name] = np.random.randint(0, len(col_definition.labels), seq_len + 1, np.int32)
            else:
                # Scalar
                input[col_name] = np.random.randint(0, len(col_definition.labels), (), np.int32)

        prediction = predictor_lite.predict( input )
        pretty_prediction(prediction, model_definition.data_definition)
        print()

        prediction = predictor.predict( input )
        pretty_prediction(prediction, model_definition.data_definition)

        print("-----------------------------\n")

# test_random()
test_lite_performance()

