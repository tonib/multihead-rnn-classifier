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
        self._predict_fn = tf.contrib.predictor.from_saved_model(latest_export , signature_def_key='predict')

    def predict(self, raw_sequence : List[List[int]] , data_definition : ModelDataDefinition ) -> object:

        prediction = self._predict_fn( data_definition.input_sequence_to_tf_predict_format(raw_sequence) )
        # prediction contains numpy arrays, they are not serializable to JSON. Return an "unpacked" prediction version
        result = {}
        for column in data_definition.output_columns:
            column_result = {}
            column_result['class_prediction'] = int( prediction[ column.name + '/classes' ][0][0] )
            column_result['probabilities'] = prediction[ column.name + '/probabilities' ][0].tolist()
            result[column.name] = column_result

        return result

    def predict_json(self, raw_sequence_json : str , data_definition : ModelDataDefinition ) -> str:
        input_array = json.loads(raw_sequence_json)
        prediction = self.predict(input_array , data_definition)
        return json.dumps(prediction)
        