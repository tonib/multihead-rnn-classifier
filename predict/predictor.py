from model_data_definition import ModelDataDefinition
from model_definition import ModelDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
import tensorflow as tf
import json

class Predictor:
    """ To make predictions in Python """
    
    def __init__(self, model_definition: ModelDefinition, model: tf.keras.Model = None):
        self.model_definition = model_definition

        if model != None:
            # Create the TF prediction module
            self.prediction_module = model_definition.predictor_class(self.model_definition.data_definition, model)
        else:
            # Load it from the export directory
            exported_model_dir = self.model_definition.data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
            print("Loading prediction module from " + exported_model_dir)
            self.prediction_module = tf.saved_model.load(exported_model_dir)

    def predict(self, input: dict, debug=False) -> dict:
        # Convert input python values to tensors
        for key in input:
            input[key] = tf.constant(input[key], dtype=tf.int32)

        # Call the TF graph prediction function
        output = self.prediction_module.predict_tf_function(input)

        # Convert tensors to python values
        # The "probabilities" property is needed to keep backward compatibility with v1
        for key in output:
            output[key] = { "probabilities": output[key].numpy().tolist() }
            if debug:
                labels_probs = {}
                for i, label in enumerate(self.model_definition.data_definition.column_definitions[key].labels):
                    labels_probs[label] = output[key]["probabilities"][i]
                output[key]["labels_probabilities"] = labels_probs

        return output

    def predict_json(self, input_json: str, debug=False) -> str:
        input = json.loads(input_json)
        return json.dumps( self.predict(input, debug) )

    def get_empty_element(self) :
        """ Input entry with context all zeros """
        return self.model_definition.predictor_class.get_empty_element(self.model_definition.data_definition)
        
