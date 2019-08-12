import os
import tensorflow as tf
from model_data_definition import ModelDataDefinition
from typing import List
import json

class PredictionModel:
    """ Model for prediction """

    def __init__(self, data_definition : ModelDataDefinition):
        """ Loads the latest exported model """

        # Get latest export. Exports are versioned by a timestamp
        latest_export = ''
        max_timestamp = 0
        exports_dir_path = data_definition.get_exports_dir_path()
        for export_dir in os.listdir( exports_dir_path ):
            try:
                timestamp = int(export_dir)
            except:
                timestamp = 0

            if timestamp > max_timestamp:
                max_timestamp = timestamp
                latest_export = export_dir
        # The full path
        latest_export = os.path.join( exports_dir_path , latest_export )
        print("Using export from" , latest_export)

        # Import model
        # TODO: Ok, the canned RNN estimator exports its prediction function with a 'predict' signature name
        # TODO: The custom RNN estimator exports its prediction as 'serving_default'
        # TODO: I did not found any way to change the exported signature name. So change this if the estimator is changed
        if data_definition.use_custom_estimator:
            # Custom estimator signature name:
            signature_key = 'serving_default'
        else:
            # Canned estimator signature name:
            signature_key = 'predict'
        
        self._predict_fn = tf.contrib.predictor.from_saved_model(latest_export , signature_def_key=signature_key)

    def predict(self, input : dict , data_definition : ModelDataDefinition ) -> object:

        batched_input = data_definition.input_sequence_to_tf_predict_format(input)
        #print(batched_input)
        prediction = self._predict_fn( batched_input )
        # prediction contains numpy arrays, they are not serializable to JSON. Return an "unpacked" prediction version
        result = {}
        for col_name in data_definition.output_columns:
            column_result = {}

            if data_definition.use_custom_estimator:
                # For custom estimator
                column_result['class_prediction'] = int( prediction[ col_name + '/classes' ][0] )
            else:
                # For canned estimator
                column_result['class_prediction'] = int( prediction[ col_name + '/classes' ][0][0] )

            column_result['probabilities'] = prediction[ col_name + '/probabilities' ][0].tolist()
            result[col_name] = column_result

        return result

    def predict_json(self, raw_sequence_json : str , data_definition : ModelDataDefinition ) -> str:
        input = json.loads(raw_sequence_json)
        prediction = self.predict(input , data_definition)
        return json.dumps(prediction)
        