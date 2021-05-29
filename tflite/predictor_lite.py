from model_data_definition import ModelDataDefinition
from model_definition import ModelDefinition
from dataset.rnn_dataset import RnnDataset
from model.masked_one_hot_encoding import MaskedOneHotEncoding
import tensorflow as tf
import json
import numpy as np

# OK: In tf 2.3 / 2.4 / 2.5, named input/outputs are pretty broken:
# - If you convert a TF module with TFLiteConverter.from_saved_model, you get wrong output names: Output names are *tensor* name
#   (ex. StatefulPartionedCall:0), not the real *output* name
# - If you convert a TF module with TFLiteConverter.from_saved_model, input names are weird ("serving_default_REALINPUTNAME")
# - If you convert a TF module with TFLiteConverter.from_concrete_functions, input names seems OK
# - If you convert with TFLiteConverter.from_concrete_functions, you get wrong output names: Names are Identity, Identity_1 (beautiful)
# It seems these issues are fixed in 2.5 (not published yet): https://github.com/tensorflow/tensorflow/issues/32180#issuecomment-772140542
# Output names seems can be fixed because they seem to be sorted by name: So, if you sort the real names, you can map them by position

# EDIT tf 2.5: TFLiteConverter.from_concrete_functions output names are still broken, I have not tested TFLiteConverter.from_saved_model

# Here we use a model converted with TFLiteConverter.from_concrete_functions

class PredictorLite:
    """ To make predictions in Python with TF Lite """
    
    def __init__(self, model_definition: ModelDefinition):
        self.model_definition = model_definition

        # Load TF Lite model
        path = PredictorLite.get_tflite_model_path( model_definition.data_definition )
        print("Loading prediction module from " + path)
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Output names in TF lite are wrong, but they can be mapped by position. These are the right names order:
        self.real_output_names = list(self.model_definition.data_definition.output_columns)
        self.real_output_names.sort()

    @staticmethod
    def get_tflite_model_path(data_definition: ModelDataDefinition):
        """ Returns expected path for TF Lite model file """
        return data_definition.get_data_dir_path( 'model/model.tflite' )

    def predict(self, input: dict) -> dict:

        # Convert input python values to numpy, and pad them
        for input_spec in self.input_details:
            input_value = input[input_spec["name"]]
            input_value = np.array(input_value, dtype=np.int32)

            input_shape = input_spec["shape"]
            # Pad only if input is not scalar
            if len(input_shape) > 0:
                pad_size = input_shape[0] - input_value.shape[0]
                if pad_size > 0:
                    input_value = np.pad( input_value , (0, pad_size), 'constant' , constant_values=(-1) )
            
            self.interpreter.set_tensor(input_spec["index"], input_value)

        # Run the prediction
        self.interpreter.invoke()

        # Convert tensors to python values
        # The "probabilities" property is needed to keep backward compatibility with v1
        output = {}
        for idx, output_spec in enumerate(self.output_details):
            col_name = self.real_output_names[idx]
            output_values = self.interpreter.get_tensor(output_spec['index'])
            output[col_name] = { "probabilities": output_values.tolist() }
        return output

    def predict_json(self, input_json: str) -> str:
        input = json.loads(input_json)
        return json.dumps( self.predict(input) )

    def get_empty_element(self) :
        """ Input entry with context all zeros """
        return self.model_definition.predictor_class.get_empty_element(self.model_definition.data_definition)
        
