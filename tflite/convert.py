import tensorflow as tf
from model_data_definition import ModelDataDefinition
from dataset.rnn_dataset import RnnDataset
import numpy as np

data_definition = ModelDataDefinition.from_file()
exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(exported_model_dir) # path to the SavedModel directory
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# # Save the model.
path = data_definition.get_data_dir_path( 'model/model.tflite' )
with open( path, 'wb') as f:
    f.write( tflite_model )
