from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
import tensorflow as tf

class Predictor:

    def __init__(self, data_definition: ModelDataDefinition):
        self.data_definition = data_definition

        # Load exported model
        # TODO: What does the compile parameter in load_model (default=True) ??? If this includes add an optimizer, it should be false!
        # TODO: This print warnings, see why..
        # W tensorflow/core/common_runtime/graph_constructor.cc:808] Node 'cond/while' has 13 outputs but the _output_shapes attribute specifies shapes for 44 outputs. Output shapes may be inaccurate
        print("Loading model from " + ModelDataDefinition.EXPORTED_MODEL_DIR)
        self.model: tf.keras.Model = tf.keras.models.load_model( ModelDataDefinition.EXPORTED_MODEL_DIR, compile=False )

    @tf.function
    def _preprocess_input(self, input: dict):
        postprocessed = {}
        # Be sure sequence inputs match data_definition.sequence_length - 1
        for seq_column_name in self.data_definition.sequence_columns:

            sequence_feature_values = input[seq_column_name]
            sequence_feature_values += ClassifierDataset.N_KEYWORD_VALUES

            # Truncate sequence if too long
            expected_length = self.data_definition.sequence_length - 1
            sequence_feature_length = tf.shape( sequence_feature_values )[0]
            if sequence_feature_length > expected_length:
                sequence_feature_values = sequence_feature_values[ sequence_feature_length - expected_length : ]
            
            # Add EOS symbol
            sequence_feature_values = tf.concat( [ sequence_feature_values , [ClassifierDataset.EOS_VALUE] ] , axis=0 )

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
        