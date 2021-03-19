from model_data_definition import ModelDataDefinition
from model import generate_model
import tensorflow as tf

# Read data definition
data_definition = ModelDataDefinition()

# Get latest trained model
checkpoints_dir = data_definition.get_data_dir_path( ModelDataDefinition.CHECKPOINTS_DIR )
latest_cp = tf.train.latest_checkpoint( checkpoints_dir )
if latest_cp == None:
    print("No checkpoint found at " + checkpoints_dir + ": Nothing exported")
    exit()

print("Exporting model from checkpoint " + latest_cp)
model = generate_model(data_definition)
model.load_weights(latest_cp)

# Save full model
exported_model_dir = data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
model.save( exported_model_dir, include_optimizer=False )
print("Model exported to " + exported_model_dir)
