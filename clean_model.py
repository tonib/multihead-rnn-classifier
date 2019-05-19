
from model_data_definition import ModelDataDefinition
import os
import shutil

# Read data definition
data_definition = ModelDataDefinition()

def delete_dir(dir_path: str):
    if os.path.isdir(dir_path):
        print("Deleting", dir_path)
        shutil.rmtree(dir_path)
    else:
        print(dir_path, "does not exists")

delete_dir( data_definition.get_current_model_dir_path() )
delete_dir( data_definition.get_exports_dir_path() )

eval_path = data_definition.get_validation_set_path()
if os.path.isfile(eval_path):
    print("Deleting", eval_path)
    os.remove(eval_path)
else:
    print(eval_path, "does not exists")
