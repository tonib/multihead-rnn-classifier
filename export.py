from model_data_definition import ModelDataDefinition
from model import generate_model
import tensorflow as tf

# Read data definition
data_definition = ModelDataDefinition()

# Get latest trained model
latest_cp = tf.train.latest_checkpoint( ModelDataDefinition.CHECKPOINTS_DIR )
if latest_cp == None:
    print("No checkpoint found at " + ModelDataDefinition.CHECKPOINTS_DIR + ": Nothing exported")
    exit()

print("Exporting model from checkpoint " + latest_cp)
model = generate_model(data_definition)
model.load_weights(latest_cp)

# Save full model
export_path = "model/exported_model"
model.save( export_path, include_optimizer=False )
print("Model exported to " + export_path)
