"""
    Search all tensorboard logs in given directories and display them in tensorboard, with names according to their definition
    Use:
        python -m debug.tensorboard -l dir1 dir2..
"""
import argparse
import glob
from pathlib import Path
from model_data_definition import ModelDataDefinition
import subprocess

parser = argparse.ArgumentParser(description="Run tensorboard across a set of directories",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-l','--list', nargs='+', help='<Required> Directories to include recursively', required=True)
args = parser.parse_args()

models_arg = []

for dir in args.list:
    for logs_dir in glob.glob(dir + '/**/' + ModelDataDefinition.TBOARD_DIR_NAME, recursive=True):
        model_name = Path(logs_dir).parent.parent.name
        models_arg.append(model_name + ":" + logs_dir)
        
tb_parameter = "--logdir_spec=" + ",".join(models_arg)
print("tensorboard", tb_parameter)
subprocess.run(['tensorboard', tb_parameter])


