from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
import tensorflow as tf
import json

class BasePredictor:

    def __init__(self, data_definition: ModelDataDefinition):
        self.data_definition = data_definition

    def _load_model(self, custom_objects: dict):
        # Load exported model
        # TODO: What does the compile parameter in load_model (default=True) ??? If this includes add an optimizer, it should be false!
        # TODO: This print warnings, see why..
        # W tensorflow/core/common_runtime/graph_constructor.cc:808] Node 'cond/while' has 13 outputs but the _output_shapes attribute specifies shapes for 44 outputs. Output shapes may be inaccurate
        exported_model_dir = self.data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
        print("Loading model from " + exported_model_dir)
        self.model: tf.keras.Model = tf.keras.models.load_model( exported_model_dir, 
            custom_objects=custom_objects,
            compile=False )
        
        # According to https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects , this should load the TF graph only
        # and could be used to do predictions. I don't know what it loads, but it gives different predictions...
        #self.model = tf.saved_model.load(exported_model_dir)

    def predict(self, input: dict, debug=False) -> dict:
        # Convert input python values to tensors
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

    def get_empty_element(self) :
        """ Input entry with context all zeros """
        # To be defined in inherited classes
        return None
        
