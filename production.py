"""
    Script to create a model directory for production (or to save a snapshot to compare models)
"""

from model_data_definition import ModelDataDefinition
import os
import shutil

data_definition = ModelDataDefinition.from_file()

if os.path.exists(data_definition.production_dir):
    print("Deleting", data_definition.production_dir)
    shutil.rmtree(data_definition.production_dir)

# Create production dir
if not os.path.exists(data_definition.production_dir):
    print("Creating directory " + data_definition.production_dir)
    os.mkdir(data_definition.production_dir)

# Copy data_info.json
src = data_definition.get_config_file_path()
dst = os.path.join( data_definition.production_dir , os.path.basename(src) )
print("Copying " + src + " to " + dst)
shutil.copyfile( src , dst )

# Copy exported model
src = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
dst = os.path.join( data_definition.production_dir , ModelDataDefinition.EXPORTED_MODEL_DIR )
print("Copying " + src + " to " + dst)
shutil.copytree(src, dst)

# Copy TF Lite model if it exists
tf_lite_model = data_definition.get_data_dir_path( ModelDataDefinition.TFLITE_PATH )
if os.path.isfile(tf_lite_model):
    dst = os.path.join( data_definition.production_dir , ModelDataDefinition.TFLITE_PATH )
    print("Copying " + tf_lite_model + " to " + dst )
    shutil.copyfile(tf_lite_model, dst)

# Copy tensorboard 
src = data_definition.get_data_dir_path( ModelDataDefinition.TBOARD_LOGS_DIR )
dst = os.path.join( data_definition.production_dir , ModelDataDefinition.TBOARD_LOGS_DIR )
print("Copying " + src + " to " + dst + ". This is the Tensorboard logs directory. Not required for production, but useful to compare models")
shutil.copytree(src, dst)
