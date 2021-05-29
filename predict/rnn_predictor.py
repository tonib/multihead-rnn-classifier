from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
from predict.base_predictory import BasePredictor
import tensorflow as tf

class RnnPredictorBase(BasePredictor):
    """ Base class for RNN predictors with context only for token to predict (TF and TF lite) """

    def __init__(self, data_definition: ModelDataDefinition, model: tf.keras.Model):
        super().__init__(data_definition, model)

        # self._load_model({'MaskedOneHotEncoding': MaskedOneHotEncoding}, model)

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


class RnnPredictor(RnnPredictorBase):
    """ Tensorflow RNN predictor with context only for token to predict """

    def __init__(self, data_definition: ModelDataDefinition, model: tf.keras.Model):
        super().__init__(data_definition, model)

        # self._predict_tf CANNOT be decorated with tf.function, because inputs can have 
        # different shapes (dict with keys specified by self.data_definition, with different sequence lengths). If you decorate it with @tf.function, 
        # a different graph will be generated for each different sequence length feeded to the funcion. 
        # So, declare the AutoGraph here, with the right signature:
        signature = {}
        for seq_column_name in self.data_definition.sequence_columns:
            signature[seq_column_name] = tf.TensorSpec(shape=[None], dtype=tf.int32, name=seq_column_name)
        for cxt_column_name in self.data_definition.context_columns:
            signature[cxt_column_name] = tf.TensorSpec(shape=(), dtype=tf.int32, name=cxt_column_name)
        self.predict_tf_function = tf.function(func=self._predict_tf, input_signature=[signature])

class RnnPredictorLite(RnnPredictorBase):
    """ Tensorflow lite RNN predictor with context only for token to predict """

    def __init__(self, data_definition: ModelDataDefinition, model: tf.keras.Model):
        super().__init__(data_definition, model)

        # Signature for TF.lite (it seems not to support very well dynamic lengths)
        # See https://stackoverflow.com/questions/55701663/input-images-with-dynamic-dimensions-in-tensorflow-lite
        # So, have a fixed shape input, with a "padding" value of -1
        signature = {}
        for seq_column_name in self.data_definition.sequence_columns:
            signature[seq_column_name] = tf.TensorSpec(shape=[data_definition.sequence_length], dtype=tf.int32, name=seq_column_name)
        for cxt_column_name in self.data_definition.context_columns:
            signature[cxt_column_name] = tf.TensorSpec(shape=(), dtype=tf.int32, name=cxt_column_name)
        self.predict_tflite_function = tf.function(func=self._predict_tflite, input_signature=[signature])

    def _predict_tflite(self, input:dict) -> dict:
        # Remove padding value (-1)
        processed_input = {}
        for key in self.data_definition.sequence_columns:
            processed_input[key] = BasePredictor.remove_tflite_padding( input[key] )
        for key in self.data_definition.context_columns:
            processed_input[key] = input[key]

        # Do real preprocessing/prediction
        return self._predict_tf(processed_input)
