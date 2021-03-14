from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
from classifier_dataset import ClassifierDataset
if TYPE_CHECKING:
    from column_info import ColumnInfo

import tensorflow as tf
from model_data_definition import ModelDataDefinition

class MaskedOneHotEncoding(tf.keras.layers.Layer):
    """ Compute masked one hot encoding from an integer input. Mask value is 0. Input mask is ignored. """
    def __init__(self, input_n_labels: int, name=None):
        """
            Arguments: 
                input_n_labels: Number of labels expected in input, including the padding value (zero). Ex. {0, 1, 2} -> n.labels = 3
        """
        super().__init__(name=name)
        self.input_n_labels = input_n_labels

    def call(self, inputs):
        # -1 is to optimize the output size. As zero is reserved for padding, only 1+ values will be used as real inputs
        return tf.one_hot(inputs - 1, self.input_n_labels - 1)

    def compute_mask(self, inputs, mask=None):
        return tf.cast( inputs , tf.bool )

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
        processed_input = MaskedOneHotEncoding(n_labels, name='one_hot_' + column_name)(input)
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

    # Currently, only one classifier. TODO: Add all
    # output_column_definition: ColumnInfo = data_definition.column_definitions[ data_definition.output_columns[0] ]
    # model = tf.keras.layers.Dense( len(output_column_definition.labels) , name=output_column_definition.name + "_output" , activation=None )( model )

    # Create a classifier for each output
    outputs = {}
    for output_column_name in data_definition.output_columns:
        output_column: ColumnInfo = data_definition.column_definitions[ output_column_name ]
        classifier = tf.keras.layers.Dense( len(output_column.labels) , name=output_column.name + "_out" , activation=None )( model )
        outputs[output_column_name] = classifier

    return tf.keras.Model(inputs=raw_inputs, outputs=outputs)
