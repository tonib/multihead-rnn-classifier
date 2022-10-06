from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
from dataset.rnn_dataset_exp import RnnDatasetExp
if TYPE_CHECKING:
    from column_info import ColumnInfo
    from ..model_data_definition import ModelDataDefinition

import tensorflow as tf
from .masked_one_hot_encoding import MaskedOneHotEncoding

def _get_preprocess_input_layer(column_info: ColumnInfo, column_name: str) -> tf.keras.layers.Layer:
    n_labels = len(column_info.labels) + RnnDatasetExp.N_KEYWORD_VALUES
    # Encode input, add masking
    if column_info.embeddable_dimension > 0:
        return tf.keras.layers.Embedding(n_labels, column_info.embeddable_dimension, mask_zero=True, 
            name="embedding_" + column_name)
    else:
        return MaskedOneHotEncoding(n_labels, name='one_hot_' + column_name)


def _get_input(data_definition: ModelDataDefinition, column_name: str, model_inputs: dict, 
    shared_preprocessing_layers : dict[str, tf.keras.layers.Layer]):

    column_info: ColumnInfo = data_definition.column_definitions[column_name]
    input = tf.keras.Input(name=column_name, dtype=tf.int32, shape=[data_definition.sequence_length])

    # This is the dict of Keras model inputs
    model_inputs[column_name] = input

    if column_info.shared_labels_name != None:
        # This column uses a shared labels set. All inputs using this shared set will use the same encoder:
        preprocess_input_layer = shared_preprocessing_layers[column_info.shared_labels_name]
    else:
        # Create a custom encoder for this input
        preprocess_input_layer = _get_preprocess_input_layer(column_info, column_name)

    # return encoded input
    return preprocess_input_layer(input)


def create_rnn_model_exp(data_definition: ModelDataDefinition):

    # Define shared preprocessing layers
    shared_preprocessing_layers : dict[str, tf.keras.layers.Layer] = {} 
    for shared_labels in data_definition.shared_labels.values():
        shared_preprocessing_layers[shared_labels.name] = _get_preprocess_input_layer(shared_labels, shared_labels.name)

    # Define sequence inputs
    model_inputs = {}
    sequences = []
    for column_name in ( data_definition.sequence_columns + data_definition.context_columns ):
        sequences.append( _get_input(data_definition, column_name, model_inputs, shared_preprocessing_layers) )
    
    # Merge all inputs for each timestep
    if len(sequences) > 1:
        input = tf.keras.layers.Concatenate()( sequences )
    else:
        input = sequences[0]
    
    # Optional "embedding" to reduce model size:
    if data_definition.rnn_embedding_size > 0:
        input = tf.keras.layers.Dense(data_definition.rnn_embedding_size, name="input_embedding")(input)

    if data_definition.cell_type == "gru":
        model = tf.keras.layers.GRU(data_definition.n_network_elements, name="rnn")(input)
    else:
        model = tf.keras.layers.LSTM(data_definition.n_network_elements, name="rnn")(input)

    if data_definition.dropout > 0.0:
        model = tf.keras.layers.Dropout(data_definition.dropout, name="dropout")(model)

    # Create a classifier for each output
    outputs = {}
    for output_column_name in data_definition.output_columns:
        output_column: ColumnInfo = data_definition.column_definitions[ output_column_name ]
        classifier = tf.keras.layers.Dense( len(output_column.labels) , name=output_column.name + "_classifier" , activation=None )( model )
        outputs[output_column_name] = classifier

    return tf.keras.Model(inputs=model_inputs, outputs=outputs)
