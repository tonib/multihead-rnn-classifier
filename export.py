from model_data_definition import ModelDataDefinition
from model import generate_model
import tensorflow as tf
import os

# Read data definition
data_definition = ModelDataDefinition()

# Get latest trained model
checkpoints_dir = data_definition.get_data_dir_path( ModelDataDefinition.CHECKPOINTS_DIR )
if data_definition.export_checkpoint <= 0:
    print("Exporting latest trained epoch checkpoint")
    export_cp_path = tf.train.latest_checkpoint( checkpoints_dir )
    if export_cp_path == None:
        print("No checkpoint found at " + checkpoints_dir + ": Nothing exported")
        exit()
else:
    # Export from specific checkpoint
    export_cp_path = checkpoints_dir + "/checkpoint-{0:04d}.ckpt".format( data_definition.export_checkpoint )
    print("Export specific checkpoint", export_cp_path)

print("Exporting model from checkpoint " + export_cp_path)
model = generate_model(data_definition)
model.load_weights(export_cp_path)

# Save full model
exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
model.save( exported_model_dir, include_optimizer=False )
print("Model exported to " + exported_model_dir)
