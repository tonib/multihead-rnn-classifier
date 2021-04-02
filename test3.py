from model_data_definition import ModelDataDefinition
from dataset.transformer_dataset import TransformerDataset
from data_directory import DataDirectory
import json
import tensorflow as tf
from model.mingpt.model_adapted import GPT

# Read data definition
data_definition = ModelDataDefinition.from_file()

exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
print("Loading model from " + exported_model_dir)
model: tf.keras.Model = tf.keras.models.load_model( exported_model_dir, 
    custom_objects={"GPT": GPT},
    compile=False )
