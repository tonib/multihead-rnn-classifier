
# The only thing that can be done to disable TF warnings is to import this file BEFORE ANY import tensorflow...
# https://github.com/tensorflow/tensorflow/issues/31870

import sys
import os

NOTFWARNINGS_FLAG = "--notfwarnings"

if NOTFWARNINGS_FLAG in sys.argv:
    #cur_value = and os.environ['TF_CPP_MIN_LOG_LEVEL'] != '2'
    print("Disabling info/warning messages from Tensorflow C++ library")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
