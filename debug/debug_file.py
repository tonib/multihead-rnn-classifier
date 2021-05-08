
import configure_tf_log # Must be FIRST import
from model_definition import ModelDefinition
from predict.predictor import Predictor
from debug_ds import pretty
from time import time
import json
import tensorflow as tf
from data_directory import DataDirectory
from debug.debug_predictor import pretty_prediction

model_definition = ModelDefinition()
data_definition = model_definition.data_definition
predictor = Predictor(model_definition)

data_dir = DataDirectory(["data/PAlcCanFac.csv"])
ds = model_definition.dataset_class(data_dir, data_definition, shuffle=False, debug_columns=False)

ds.dataset = ds.dataset.batch(1).take(1)
#ds.dataset = ds.dataset.take(3)

# TODO: I think this was done to debug GPT only. Make it work with rnn too?

for row in ds.dataset:
    input = row[0]
    expected_output = row[1]

    #print(input)
    batched_logits = predictor.model(input)
    #print(batched_logits)

    for key in expected_output:
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state( expected_output[key][0], batched_logits[key][0])
        print( "Accuracy " + key + ":", m.result().numpy() )

    # m = tf.keras.metrics.SparseCategoricalAccuracy()
    # m.update_state( expected_output["outputTypeIdx"][0], batched_logits["outputTypeIdx"][0])
    # print( "Accuracy :", m.result().numpy() )

    for i in range(data_definition.sequence_length):
        m = tf.keras.metrics.SparseCategoricalAccuracy()
        m.update_state( expected_output["outputTypeIdx"][0][i], batched_logits["outputTypeIdx"][0][i])
        print( "Accuracy " + str(i) + ":", m.result().numpy() )

    # m = tf.keras.metrics.SparseCategoricalAccuracy()
    # m.update_state( expected_output["outputTypeIdx"][0], batched_logits["outputTypeIdx"][0])
    # print( "Accuracy :", m.result().numpy() )

    for idx in range(17):
        output = {}
        labels_probs = {}
        for key in batched_logits:
            # Model returned values are logits. Convert to probabilities and unbatch result
            output[key] = tf.nn.softmax( batched_logits[key][0][idx] )
            output[key] = { "probabilities": output[key].numpy().tolist() }

            # labels_probs[key] = {}
            # for i, label in enumerate(data_definition.column_definitions[key].labels):
            #     labels_probs[key][label] = output[key][i]

        #print(output)
        #print(labels_probs)
        print(idx,")")
        pretty_prediction(output, data_definition)
        
        print()

    print(expected_output)
