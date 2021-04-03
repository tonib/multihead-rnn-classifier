from predict.base_predictory import BasePredictor
from model.mingpt.model_adapted import GPT
from dataset.transformer_dataset import TransformerDataset
from model_data_definition import ModelDataDefinition

import tensorflow as tf

class GptPredictor(BasePredictor):

    def __init__(self, data_definition: ModelDataDefinition):
        super().__init__(data_definition)

        self._load_model({"GPT": GPT})
        self.all_column_names = self.data_definition.sequence_columns + self.data_definition.context_columns

        # self._predict_tf CANNOT be decorated with tf.function, because inputs can have 
        # different shapes (dict with keys specified by self.data_definition, with different sequence lengths). If you decorate it with @tf.function, 
        # a different graph will be generated for each different sequence length feeded to the funcion. 
        # So, declare the AutoGraph here, with the right signature:
        signature = {}
        for column_name in self.all_column_names:
            signature[column_name] = tf.TensorSpec(shape=[None], dtype=tf.int32)
        self._predict_tf_function = tf.function(func=self._predict_tf, input_signature=[signature])

    def pad_sequence(self, inputs):
        inputs_length = tf.shape(inputs)[0]
        seq_len_diff = self.data_definition.sequence_length - inputs_length
        if seq_len_diff > 0:
            # Too short: Pad right up to sequence length
            zeros = tf.zeros( [seq_len_diff] , dtype=inputs.dtype )
            inputs = tf.concat( [inputs, zeros], axis=0 )
        elif seq_len_diff < 0:
            # Sequence too long, remove head elements:
            inputs = inputs[-seq_len_diff:]
        return inputs

    def _add_bos(self, input: dict) -> dict:

        # Check if we must to add BOS
        seq_len = tf.shape( input[self.data_definition.sequence_columns[0]] )[0]
        if seq_len >= self.data_definition.sequence_length:
            # No
            return input

        # Add it to sequence features
        input_with_bos = {}
        bos = [TransformerDataset.BOS_VALUE]
        for column_name in self.data_definition.sequence_columns:
            input_with_bos[column_name] = tf.concat( [ bos , input[column_name] ] , axis=0 )

        # In context features the value is not important, as the first value will be discarded. So, add a zero
        zero = zeros = tf.zeros( [1] , dtype=tf.int32 )
        for column_name in self.data_definition.context_columns:
            input_with_bos[column_name] = tf.concat( [ zero , input[column_name] ] , axis=0 )

        return input_with_bos

    def _preprocess_input(self, input: dict):

        # TODO: Add assertions about sequence / context lengths?

        # Increase input values, to reserve values for keywords
        processed_input = {}
        for column_name in self.all_column_names:
            processed_input[column_name] = input[column_name] + TransformerDataset.N_KEYWORD_VALUES

        # Add BOS if needed
        processed_input = self._add_bos(processed_input)

        for column_name in self.data_definition.sequence_columns:
            # Truncate/pad sequence
            sequence_feature_values = self.pad_sequence( processed_input[column_name] )
            # Create batch of 1
            processed_input[column_name] = tf.expand_dims( sequence_feature_values , axis=0 )

        for column_name in self.data_definition.context_columns:
            # In context columns, ignore the first value
            sequence_feature_values = processed_input[column_name][1:]
            # Truncate/pad sequence
            sequence_feature_values = self.pad_sequence(sequence_feature_values)
            # Create batch of 1
            processed_input[column_name] = tf.expand_dims( sequence_feature_values , axis=0 )

        return processed_input

    # @tf.function < NO (see __init__ comments)
    def _predict_tf(self, input: dict) -> dict:

        # Get original sequence length now: input will be modified by _preprocess_input call
        seq_len = tf.shape( input[self.data_definition.sequence_columns[0]] )[0]

        input = self._preprocess_input(input)
        batched_logits = self.model(input)

        # GPT will return predictions for all timesteps. Get the timestep index for the current prediction
        # Ex, if sequence length = 4:
        # BOS 0 0 0 | input len = 0 | get idx 0
        # BOS 1 0 0 | input len = 1 | get idx 1
        # BOS 1 2 0 | input len = 2 | get idx 2
        # BOS 1 2 3 | input len = 3 | get idx 3
        # 1   2 3 4 | input len = 4 | get idx 3
        # 2   3 4 5 | input len = 4 | get idx 3
        idx = tf.minimum(seq_len, self.data_definition.sequence_length-1)

        output = {}
        for key in batched_logits:
            # Model returned values are logits. Convert to probabilities and unbatch result
            output[key] = tf.nn.softmax( batched_logits[key][0][idx] )
        
        return output

    def get_empty_element(self) :
        """ Input entry with context all zeros """
        element = {}
        for column_name in self.data_definition.sequence_columns:
            element[column_name] = []
        for column_name in self.data_definition.context_columns:
            element[column_name] = [0]
        return element