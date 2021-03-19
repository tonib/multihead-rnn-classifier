from model_data_definition import ModelDataDefinition
from classifier_dataset import ClassifierDataset
from model import MaskedOneHotEncoding
import tensorflow as tf
import json

class Predictor:

    def __init__(self, data_definition: ModelDataDefinition):
        self.data_definition = data_definition

        # Load exported model
        # TODO: What does the compile parameter in load_model (default=True) ??? If this includes add an optimizer, it should be false!
        # TODO: This print warnings, see why..
        # W tensorflow/core/common_runtime/graph_constructor.cc:808] Node 'cond/while' has 13 outputs but the _output_shapes attribute specifies shapes for 44 outputs. Output shapes may be inaccurate
        exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
        print("Loading model from " + exported_model_dir)
        self.model: tf.keras.Model = tf.keras.models.load_model( exported_model_dir, 
            custom_objects={'MaskedOneHotEncoding': MaskedOneHotEncoding},
            compile=False )

        # self._predict_tf CANNOT be decorated with tf.function, because inputs can have 
        # different shapes (dict with keys specified by self.data_definition, with different sequence lengths). If you decorate it with @tf.function, 
        # a different graph will be generated for each different sequence length feeded to the funcion. 
        # So, declare the AutoGraph here, with the right signature:
        signature = {}
        for seq_column_name in self.data_definition.sequence_columns:
            signature[seq_column_name] = tf.TensorSpec(shape=[None], dtype=tf.int32)
        for txt_column_name in self.data_definition.context_columns:
            signature[txt_column_name] = tf.TensorSpec(shape=(), dtype=tf.int32)
        self._predict_tf_function = tf.function(func=self._predict_tf, input_signature=[signature])


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
    
    # @tf.function < NO (__init__ comments)
    def _predict_tf(self, input: dict) -> dict:
        input = self._preprocess_input(input)
        batched_logits = self.model(input)

        # Model returned values are logits. Convert to probabilities and unbatch result
        output = {}
        for key in batched_logits:
            output[key] = tf.nn.softmax(batched_logits[key][0])
        return output

    def predict(self, input: dict, debug=False) -> dict:
        for key in input:
            input[key] = tf.constant(input[key], dtype=tf.int32)

        # Call the TF graph prediction function
        output = self._predict_tf_function(input)

        # Convert tensors to python values
        # The "probabilities" property is needed to keep backward compatibility with v1
        for key in output:
            output[key] = { "probabilities": output[key].numpy().tolist() }
            if debug:
                labels_probs = {}
                for i, label in enumerate(self.data_definition.column_definitions[key].labels):
                    labels_probs[label] = output[key]["probabilities"][i]
                output[key]["labels_probabilities"] = labels_probs

        return output

    def predict_json(self, input_json: str, debug=False) -> str:
        input = json.loads(input_json)
        return json.dumps( self.predict(input, debug) )