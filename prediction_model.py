import os
import tensorflow as tf
from model_data_definition import ModelDataDefinition
from typing import List

class PredictionModel:
    """ Model for prediction """

    EXPORTED_MODELS_DIR_PATH = 'exportedmodels'

    def __init__(self):
        """ Loads the latest exported model """

        # Get latest export. Exports are versioned by a timestamp
        latest_export = ''
        max_timestamp = 0
        for export_dir in os.listdir( PredictionModel.EXPORTED_MODELS_DIR_PATH ):
            try:
                timestamp = int(export_dir)
            except:
                timestamp = 0

            if timestamp > max_timestamp:
                max_timestamp = timestamp
                latest_export = export_dir
        # The full path
        latest_export = os.path.join( PredictionModel.EXPORTED_MODELS_DIR_PATH , latest_export )
        print("Using export from" , latest_export)

        # Import model
        self._predict_fn = tf.contrib.predictor.from_saved_model(latest_export , signature_def_key='predict')

    def predict(self, raw_sequence : List[List[int]] , data_definition : ModelDataDefinition ) -> object:
        return self._predict_fn( data_definition.input_sequence_to_tf_predict_format(raw_sequence) )
