from predictor import Predictor
from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from data_directory import DataDirectory
import tensorflow as tf
import numpy as np

data_definition = ModelDataDefinition()
predictor = Predictor(data_definition)

for l in range(0, 3): # range(0, data_definition.sequence_length):
    input = {}
    for seq_column_name in data_definition.sequence_columns:
        input[seq_column_name] = tf.zeros(l, dtype=tf.int32)
    for ctx_column_name in data_definition.context_columns:
        input[ctx_column_name] = tf.constant(0, dtype=tf.int32)

    print( predictor.predict(input) )

print( predictor._predict_tf_function.pretty_printed_concrete_signatures() )
