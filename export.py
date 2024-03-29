from model_data_definition import ModelDataDefinition
from model_definition import ModelDefinition
from data_directory import DataDirectory
import tensorflow as tf
import os

def get_model_checkpoint_to_export(model_definition: ModelDefinition) -> tf.keras.Model:

    checkpoints_dir = model_definition.data_definition.get_data_dir_path( ModelDataDefinition.CHECKPOINTS_DIR )
    if model_definition.data_definition.export_checkpoint <= 0:
        # Get latest trained model
        print("Exporting latest trained epoch checkpoint")
        export_cp_path = tf.train.latest_checkpoint( checkpoints_dir )
        if export_cp_path == None:
            print("No checkpoint found at " + checkpoints_dir + ": Nothing exported")
            exit()
    else:
        # Export from specific checkpoint
        export_cp_path = checkpoints_dir + "/checkpoint-{0:04d}.ckpt".format( model_definition.data_definition.export_checkpoint )
        print("Export specific checkpoint", export_cp_path)

    print("Loading checkpoint " + export_cp_path)
    model = model_definition.create_model_function(model_definition.data_definition)
    model.load_weights(export_cp_path)

    # TODO: Fails with GPT model (missing signature?)
    # TODO: It seems because input shapes are not specified. Try model.get_concrete_function (https://github.com/tensorflow/tensorflow/issues/40344)
    # See https://stackoverflow.com/questions/51806852/cant-save-custom-subclassed-model
    # From previous, see: https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=4Onp-8rGyeQG
    # The only way to build a subclassed keras model I have found is to run predictions. So, here is:
    print("Building model...")
    all_data = DataDirectory.read_all(model_definition.data_definition)
    ds = model_definition.dataset_class(all_data, model_definition.data_definition, shuffle=False, debug_columns=False)
    build_model_ds = ds.dataset.batch(1).take(1)
    for input, output in build_model_ds:
        model(input)

    return model

if __name__ == "__main__":

    # Read model definition
    model_definition = ModelDefinition()

    model = get_model_checkpoint_to_export(model_definition)
    
    # Save the TF prediction module with input preprocessing
    print("Saving model...")
    exported_model_dir = model_definition.data_definition.get_data_dir_path( ModelDataDefinition.EXPORTED_MODEL_DIR )
    prediction_module = model_definition.predictor_class(model_definition.data_definition, model)
    tf.saved_model.save(prediction_module, exported_model_dir)
    print("Model, with preprocessing, exported to " + exported_model_dir)
