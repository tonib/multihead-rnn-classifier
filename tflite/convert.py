import tensorflow as tf
from model_definition import ModelDefinition
from model_data_definition import ModelDataDefinition
from export import get_model_checkpoint_to_export
from predict.gpt_predictor import GptPredictorLite

model_definition = ModelDefinition()
model = get_model_checkpoint_to_export(model_definition)

data_definition = model_definition.data_definition
exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )

# Module to run predictions with TF lite
predict_module = model_definition.tflite_predictor_class(data_definition, model)

# Convert the predict function, with preprocessing
converter = tf.lite.TFLiteConverter.from_concrete_functions([predict_module.predict_tflite_function.get_concrete_function()])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model.
path = data_definition.get_data_dir_path( ModelDataDefinition.TFLITE_PATH )
with open( path, 'wb') as f:
    f.write( tflite_model )
