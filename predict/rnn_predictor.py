from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
from predict.base_predictory import BasePredictor
import tensorflow as tf

class RnnPredictor(BasePredictor):

    def __init__(self, data_definition: ModelDataDefinition, model: tf.keras.Model):
        super().__init__(data_definition, model)

        # self._load_model({'MaskedOneHotEncoding': MaskedOneHotEncoding}, model)

        # self._predict_tf CANNOT be decorated with tf.function, because inputs can have 
        # different shapes (dict with keys specified by self.data_definition, with different sequence lengths). If you decorate it with @tf.function, 
        # a different graph will be generated for each different sequence length feeded to the funcion. 
        # So, declare the AutoGraph here, with the right signature:
        signature = {}
        for seq_column_name in self.data_definition.sequence_columns:
            signature[seq_column_name] = tf.TensorSpec(shape=[None], dtype=tf.int32)
        for cxt_column_name in self.data_definition.context_columns:
            signature[cxt_column_name] = tf.TensorSpec(shape=(), dtype=tf.int32)
        self.predict_tf_function = tf.function(func=self._predict_tf, input_signature=[signature])

    def _preprocess_input(self, input: dict):
        postprocessed = {}
        # Be sure sequence inputs match data_definition.sequence_length - 1
        for seq_column_name in self.data_definition.sequence_columns:

            sequence_feature_values = input[seq_column_name]
            sequence_feature_values += RnnDataset.N_KEYWORD_VALUES

            # Truncate sequence if too long
            expected_length = self.data_definition.sequence_length - 1
            sequence_feature_length = tf.shape( sequence_feature_values )[0]
            if sequence_feature_length > expected_length:
                sequence_feature_values = sequence_feature_values[ sequence_feature_length - expected_length : ]
            
            # Add EOS symbol
            sequence_feature_values = tf.concat( [ sequence_feature_values , [RnnDataset.EOS_VALUE] ] , axis=0 )

            # Add padding if too short
            sequence_feature_length = tf.shape( sequence_feature_values )[0]
            if sequence_feature_length < self.data_definition.sequence_length:
                n_padding_elements = self.data_definition.sequence_length - sequence_feature_length
                sequence_feature_values = tf.concat( [ sequence_feature_values , tf.zeros( n_padding_elements , dtype=tf.int32 ) ] , axis=0 )

            # Create batch of 1
            postprocessed[seq_column_name] = tf.expand_dims( sequence_feature_values , axis=0 )

        # Context inputs
        for ctx_column_name in self.data_definition.context_columns:
            postprocessed[ctx_column_name] = tf.expand_dims( input[ctx_column_name] , axis=0 )

        return postprocessed
    
    # @tf.function < NO (see __init__ comments)
    def _predict_tf(self, input: dict) -> dict:
        input = self._preprocess_input(input)
        batched_logits = self.model(input)

        # Model returned values are logits. Convert to probabilities and unbatch result
        output = {}
        for key in batched_logits:
            output[key] = tf.nn.softmax(batched_logits[key][0])
        return output

    @staticmethod
    def get_empty_element(data_definition: ModelDataDefinition) :
        """ Empty input entry with context all zeros """
        element = {}
        for column_name in data_definition.sequence_columns:
            element[column_name] = []
        for column_name in data_definition.context_columns:
            element[column_name] = 0
        return element
