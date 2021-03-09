from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
from classifier_dataset import ClassifierDataset
if TYPE_CHECKING:
    from column_info import ColumnInfo

import tensorflow as tf
from model_data_definition import ModelDataDefinition

def _get_input(data_definition: ModelDataDefinition, column_name: str, is_sequence: bool) -> Tuple:

    column_info: ColumnInfo  = data_definition.column_definitions[column_name]
    shape = [None] if is_sequence else ()
    input = tf.keras.Input(name=column_name, dtype=tf.int32, shape=shape)

    n_labels = len(column_info.labels) + ClassifierDataset.N_KEYWORD_VALUES

    # Encode input, add masking
    if column_info.embeddable_dimension > 0:
        processed_input = tf.keras.layers.Embedding(n_labels, column_info.embeddable_dimension, mask_zero=True, 
            name="embedding_" + column_name)(input)
    else:
        processed_input = tf.keras.layers.Lambda(
            lambda x: tf.one_hot(x, n_labels), 
            mask=lambda inputs, mask: tf.cast( inputs , tf.bool ), # This expects zero for padding element
            name='one_hot_' + column_name)(input)
    return input, processed_input

def _get_inputs(data_definition: ModelDataDefinition, column_names: List[str], are_sequences: bool) -> Tuple[List, List]:
    inputs = [_get_input(data_definition, column_name, are_sequences) for column_name in column_names]
    # This are tuples
    raw_inputs, processed_inputs = list(zip(*inputs))    
    # Return lists
    return list(raw_inputs), list(processed_inputs)

def generate_model(data_definition: ModelDataDefinition):

    # Define sequence inputs
    raw_sequence_inputs, sequence_inputs = _get_inputs(data_definition, data_definition.sequence_columns, True)
    raw_context_inputs, context_inputs = _get_inputs(data_definition, data_definition.context_columns, False)
    raw_inputs = raw_sequence_inputs + raw_context_inputs

    # Merge context to each sequence timestep
    context_inputs = tf.keras.layers.Concatenate(name="concatenated_context")( context_inputs )
    context_inputs = tf.keras.layers.RepeatVector(data_definition.sequence_length, name="repeated_context")(context_inputs)
    
    # Merge "sequenced" context and sequences on each timestep (sequence_inputs: array and context_inputs: Tensor)
    input = tf.keras.layers.Concatenate()( sequence_inputs + [ context_inputs ] )

    if data_definition.cell_type == "gru":
        model = tf.keras.layers.GRU(data_definition.n_network_elements, name="rnn")(input)
    else:
        model = tf.keras.layers.LSTM(data_definition.n_network_elements, name="rnn")(input)

    if data_definition.dropout > 0.0:
        model = tf.keras.layers.Dropout(data_definition.dropout, name="dropout")(model)

    return tf.keras.Model(inputs=raw_inputs, outputs=model)
