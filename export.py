from model_data_definition import ModelDataDefinition
from model_definition import ModelDefinition
import tensorflow as tf
import os

# Read model definition
model_definition = ModelDefinition()

# Get latest trained model
checkpoints_dir = model_definition.data_definition.get_data_dir_path( ModelDataDefinition.CHECKPOINTS_DIR )
if model_definition.data_definition.export_checkpoint <= 0:
    print("Exporting latest trained epoch checkpoint")
    export_cp_path = tf.train.latest_checkpoint( checkpoints_dir )
    if export_cp_path == None:
        print("No checkpoint found at " + checkpoints_dir + ": Nothing exported")
        exit()
else:
    # Export from specific checkpoint
    export_cp_path = checkpoints_dir + "/checkpoint-{0:04d}.ckpt".format( model_definition.data_definition.export_checkpoint )
    print("Export specific checkpoint", export_cp_path)

print("Exporting model from checkpoint " + export_cp_path)
model = model_definition.create_model_function(model_definition.data_definition)
model.load_weights(export_cp_path)

# TODO: Fails with GPT model (missing signature?)
# See https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
# From previous, see: https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=4Onp-8rGyeQG

# Save full model
exported_model_dir = model_definition.data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
model.save( exported_model_dir, include_optimizer=False )
print("Model exported to " + exported_model_dir)
